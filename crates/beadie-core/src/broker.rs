// broker.rs — Background compilation broker.
//
// A dedicated OS thread drains a bounded work queue, calling user-supplied
// compile closures and installing results into beads — entirely off the
// interpreter thread.

use std::{sync::Arc, thread};

use crossbeam_channel::{bounded, Sender, TrySendError};
use log::{debug, info, warn};

use crate::bead::Bead;

// ─────────────────────────────────────────────────────────────────────────────
// Public type alias
// ─────────────────────────────────────────────────────────────────────────────

/// A compile closure.
///
/// Receives the bead (so it can call `bead.core_handle()` at any time during
/// compilation — always getting the GC-updated pointer). Returns a pointer to
/// the compiled native code, or null on failure.
///
/// ## Lifetime
/// The closure must be `'static` because it is sent to the broker thread.
/// Capture resources via `Arc` or move them in.
///
/// ## GC safety
/// The runtime is responsible for ensuring the core remains accessible (pinned,
/// rooted, or handle-stable) for the duration of the compile closure.
/// If the core is moved, call `bead.update_core(new_ptr)` — the closure will
/// see the updated pointer on the next `bead.core_handle()` read.
/// If the core is destroyed, call `bead.invalidate()` — the broker will
/// detect it and discard the result.
pub type CompileFn = Box<dyn FnOnce(&Arc<Bead>) -> *mut () + Send + 'static>;

// ─────────────────────────────────────────────────────────────────────────────
// Internal message
// ─────────────────────────────────────────────────────────────────────────────

enum Message {
    Job { bead: Arc<Bead>, compile: CompileFn },
    Shutdown,
}

// ─────────────────────────────────────────────────────────────────────────────
// Broker
// ─────────────────────────────────────────────────────────────────────────────

/// Background compilation broker.
///
/// Owns a dedicated OS thread (`beadie-broker`) and a bounded channel.
/// The interpreter submits hot beads via [`Broker::submit`]; the worker
/// compiles them and installs results without blocking the interpreter.
///
/// ## Drop behaviour
/// Dropping the broker sends a shutdown signal and joins the worker thread —
/// all in-flight jobs complete before the broker is fully torn down.
pub struct Broker {
    sender: Sender<Message>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Broker {
    /// Spawn a broker with the given job-queue capacity.
    ///
    /// A larger capacity absorbs burst promotions without back-pressure;
    /// a smaller one keeps memory bounded. 256 is a sensible default.
    pub fn with_capacity(capacity: usize) -> Self {
        let (tx, rx) = bounded::<Message>(capacity);

        let worker = thread::Builder::new()
            .name("beadie-broker".to_owned())
            .spawn(move || {
                debug!("broker thread started");
                for msg in &rx {
                    match msg {
                        Message::Shutdown => {
                            debug!("broker thread shutting down");
                            break;
                        }
                        Message::Job { bead, compile } => Self::process(bead, compile),
                    }
                }
            })
            .expect("beadie: failed to spawn broker thread");

        Self {
            sender: tx,
            worker: Some(worker),
        }
    }

    // ── Job processing ────────────────────────────────────────────────────────

    fn process(bead: Arc<Bead>, compile: CompileFn) {
        // Gate 1: bead may have been invalidated while queued.
        if !bead.mark_compiling() {
            debug!(
                "bead {:p}: skipped (invalid or already past Queued)",
                &*bead
            );
            return;
        }

        let t0 = std::time::Instant::now();
        let code = compile(&bead);
        let elapsed = t0.elapsed();
        info!("bead {:p}: compiled in {elapsed:.2?}", &*bead);

        // Gate 2: install — or revert cleanly on failure.
        if !bead.install_compiled(code) {
            if bead.reload_pending() {
                // A reload() was called mid-compile. Discard the result,
                // clear the flag, and revert to Interpreted so the bead
                // re-promotes on the next invocation.
                bead.clear_reload_pending();
                bead.revert_compiling();
                debug!("bead {:p}: compile discarded (reload pending)", &*bead);
            } else {
                // Genuinely invalid — drive to Deopt.
                bead.invalidate();
                warn!("bead {:p}: compile failed, invalidated", &*bead);
            }
        }
    }

    // ── Submit ────────────────────────────────────────────────────────────────

    /// Submit a bead for background compilation.
    ///
    /// Returns [`SubmitResult`] describing the outcome. The CAS on the bead
    /// ensures exactly one compile job is ever created per bead.
    pub fn submit(
        &self,
        bead: Arc<Bead>,
        compile: impl FnOnce(&Arc<Bead>) -> *mut () + Send + 'static,
    ) -> SubmitResult {
        // Win the promotion race — only one caller succeeds.
        if !bead.try_queue() {
            return SubmitResult::AlreadyQueued;
        }

        let msg = Message::Job {
            bead: Arc::clone(&bead),
            compile: Box::new(compile),
        };

        match self.sender.try_send(msg) {
            Ok(()) => SubmitResult::Accepted,
            Err(TrySendError::Full(_)) => {
                // Queue is full — revert so a later tick can retry.
                bead.revert_queued();
                warn!("bead {:p}: broker queue full, reverted", &*bead);
                SubmitResult::QueueFull
            }
            Err(TrySendError::Disconnected(_)) => {
                bead.revert_queued();
                warn!("bead {:p}: broker shut down, reverted", &*bead);
                SubmitResult::BrokerShutDown
            }
        }
    }
}

impl Default for Broker {
    fn default() -> Self {
        Self::with_capacity(256)
    }
}

impl Drop for Broker {
    fn drop(&mut self) {
        // Best-effort shutdown signal; ignore send errors (already shut down).
        let _ = self.sender.send(Message::Shutdown);
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SubmitResult
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a [`Broker::submit`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitResult {
    /// Job accepted; broker will compile in the background.
    Accepted,
    /// Bead was already queued or beyond; no job created.
    AlreadyQueued,
    /// Job queue is full; bead reverted to `Interpreted`. Retry later.
    QueueFull,
    /// Broker has been shut down.
    BrokerShutDown,
}

impl SubmitResult {
    pub fn is_accepted(self) -> bool {
        self == Self::Accepted
    }
}
