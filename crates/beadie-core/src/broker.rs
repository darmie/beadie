// broker.rs — Background compilation broker.
//
// A dedicated OS thread drains a bounded work queue, calling user-supplied
// compile closures and installing results into beads — entirely off the
// interpreter thread.
//
// ## Single-shot vs batched
//
// Backends with `JITModule`-style finalization (Cranelift, LLVM ORC, custom
// MIR pipelines) pay a fixed cost per `finalize_definitions()` call. To
// amortize this across many small functions, the broker can run in *batched*
// mode: drain up to `batch_limit` jobs without blocking, stage each via the
// user closure, then call a single user-provided `flush` once before
// resolving entry pointers.
//
// The single-shot path remains the default and is unchanged. Construct
// with [`Broker::with_batching`] to opt in.

use std::{sync::Arc, thread};

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use log::{debug, info, warn};

use crate::bead::Bead;
use crate::osr::OsrEntry;

// ─────────────────────────────────────────────────────────────────────────────
// Public type aliases
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a compile closure.
///
/// - [`CompileOutcome::Ready`] — code is ready immediately; the broker
///   installs the pointer at once. This is the only variant produced by
///   single-shot backends (Cranelift's one-finalize-per-function path,
///   any backend that synchronously returns a *mut () from `compile()`).
///
/// - [`CompileOutcome::Deferred`] — the backend has *staged* the compile
///   (e.g., called `module.declare_function()` + `module.define_function()`
///   on a shared `JITModule`) but hasn't finalised yet. The carried
///   resolver closure is invoked by the broker *after* its batched flush
///   runs, and returns the now-resolved entry pointer.
///
/// Batched backends combine `Deferred` with [`Broker::with_batching`]'s
/// `flush` callback so one expensive `finalize_definitions()` (or LLVM
/// `ExecutionSession::commit()`, etc.) covers many staged compiles.
pub enum CompileOutcome {
    /// Entry pointer is ready; install immediately.
    Ready(*mut ()),
    /// Compile is staged. The resolver runs after the broker's per-batch
    /// flush hook and returns the entry pointer for this specific bead.
    Deferred(ResolverFn),
}

// SAFETY: CompileOutcome carries raw pointers managed by the runtime's JIT.
// Beadie never dereferences them; the runtime guarantees they remain valid.
unsafe impl Send for CompileOutcome {}

/// Resolver closure that yields the entry pointer for a previously-staged
/// compile after the broker calls the batch flush hook.
pub type ResolverFn = Box<dyn FnOnce() -> *mut () + Send + 'static>;

/// Per-batch flush callback. The broker invokes this once per drain cycle,
/// between staging closures (which may return [`CompileOutcome::Deferred`])
/// and resolver invocation. Typically wraps a backend's
/// `module.finalize_definitions()` or equivalent.
pub type FlushFn = Arc<dyn Fn() + Send + Sync + 'static>;

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

/// Result of an OSR-aware compile: a normal entry point plus zero or more
/// OSR entry points (one per hot loop header emitted by the JIT).
pub struct OsrCompileResult {
    pub entry: *mut (),
    pub osr: Vec<OsrEntry>,
}

// SAFETY: OsrCompileResult carries raw pointers managed by the runtime's JIT.
// Beadie never dereferences them; the runtime guarantees they remain valid.
unsafe impl Send for OsrCompileResult {}

type OsrCompileFn = Box<dyn FnOnce(&Arc<Bead>) -> OsrCompileResult + Send + 'static>;

// ─────────────────────────────────────────────────────────────────────────────
// Internal message
// ─────────────────────────────────────────────────────────────────────────────

/// Closure variant used by batched submissions. Returns a [`CompileOutcome`]
/// so the broker can either install at once or defer until a flush.
pub type CompileOutcomeFn = Box<dyn FnOnce(&Arc<Bead>) -> CompileOutcome + Send + 'static>;

enum Job {
    Simple(CompileFn),
    Outcome(CompileOutcomeFn),
    WithOsr(OsrCompileFn),
}

enum Message {
    Compile { bead: Arc<Bead>, job: Job },
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

/// Per-broker batching config. Moved into the worker thread at spawn time.
struct BatchConfig {
    batch_limit: usize,
    flush: FlushFn,
}

impl Broker {
    /// Spawn a broker with the given job-queue capacity.
    ///
    /// A larger capacity absorbs burst promotions without back-pressure;
    /// a smaller one keeps memory bounded. 256 is a sensible default.
    ///
    /// Each job is compiled and installed independently. To amortise the
    /// fixed cost of `finalize_definitions()`-style finalization across
    /// many small functions, construct via [`Broker::with_batching`].
    pub fn with_capacity(capacity: usize) -> Self {
        let (tx, rx) = bounded::<Message>(capacity);

        let worker = thread::Builder::new()
            .name("beadie-broker".to_owned())
            .spawn(move || Self::run_worker(rx, None))
            .expect("beadie: failed to spawn broker thread");

        Self {
            sender: tx,
            worker: Some(worker),
        }
    }

    /// Spawn a broker that drains up to `batch_limit` jobs without blocking,
    /// then invokes `flush` once before resolving any deferred compilations.
    ///
    /// Use this when the backend's compile path stages work into a shared
    /// module that pays a fixed cost on finalization — e.g. Cranelift's
    /// `JITModule::finalize_definitions()`, LLVM ORC commits, custom MIR
    /// pipelines with one-shot register allocators.
    ///
    /// `batch_limit` caps the worst-case work per flush cycle; pick it so
    /// that finalize-time per batch stays below the latency you can tolerate
    /// for the first invocation of a freshly hot function (a few ms is
    /// typical). 32 is a sensible default for small functions; raise it for
    /// large modules where finalize dominates.
    ///
    /// Compile closures may continue to return [`CompileOutcome::Ready`] in
    /// batched mode — those install at once. Only `Deferred` outcomes wait
    /// for the flush.
    pub fn with_batching(capacity: usize, batch_limit: usize, flush: FlushFn) -> Self {
        assert!(batch_limit >= 1, "batch_limit must be >= 1");
        let (tx, rx) = bounded::<Message>(capacity);

        let cfg = BatchConfig { batch_limit, flush };
        let worker = thread::Builder::new()
            .name("beadie-broker-batched".to_owned())
            .spawn(move || Self::run_worker(rx, Some(cfg)))
            .expect("beadie: failed to spawn broker thread");

        Self {
            sender: tx,
            worker: Some(worker),
        }
    }

    // ── Worker loop ───────────────────────────────────────────────────────────

    fn run_worker(rx: Receiver<Message>, batch: Option<BatchConfig>) {
        debug!("broker thread started (batched={})", batch.is_some());
        let mut pending: Vec<(Arc<Bead>, ResolverFn)> = Vec::new();

        loop {
            // Block on the first message of a cycle; any subsequent messages
            // are absorbed non-blockingly until `batch_limit` is reached.
            let first = match rx.recv() {
                Ok(Message::Shutdown) => {
                    debug!("broker thread shutting down");
                    break;
                }
                Ok(Message::Compile { bead, job }) => (bead, job),
                Err(_) => break, // channel closed
            };

            let mut drained = 0usize;
            let limit = batch.as_ref().map_or(1, |b| b.batch_limit);
            Self::process(first.0, first.1, &mut pending);
            drained += 1;

            // Absorb additional ready messages without blocking.
            while drained < limit {
                match rx.try_recv() {
                    Ok(Message::Shutdown) => {
                        // Drain everything before shutting down — important
                        // for tests that submit then immediately drop.
                        Self::flush_pending(batch.as_ref(), &mut pending);
                        return;
                    }
                    Ok(Message::Compile { bead, job }) => {
                        Self::process(bead, job, &mut pending);
                        drained += 1;
                    }
                    Err(_) => break,
                }
            }

            Self::flush_pending(batch.as_ref(), &mut pending);
        }

        // Drain anything still staged on shutdown.
        Self::flush_pending(batch.as_ref(), &mut pending);
    }

    /// Invoke the flush hook (if batched) and install all deferred compiles.
    fn flush_pending(batch: Option<&BatchConfig>, pending: &mut Vec<(Arc<Bead>, ResolverFn)>) {
        if pending.is_empty() {
            return;
        }
        if let Some(cfg) = batch {
            (cfg.flush)();
        } else {
            // Non-batched broker still got Deferred outcomes — degenerate but
            // legal. Resolvers run independently; no shared finalize step.
        }
        let drained: Vec<_> = std::mem::take(pending);
        for (bead, resolver) in drained {
            let code = resolver();
            Self::install_or_recover(&bead, code, /*osr*/ None);
        }
    }

    // ── Job processing ────────────────────────────────────────────────────────

    fn process(bead: Arc<Bead>, job: Job, pending: &mut Vec<(Arc<Bead>, ResolverFn)>) {
        // Gate 1: bead may have been invalidated while queued.
        if !bead.mark_compiling() {
            debug!(
                "bead {:p}: skipped (invalid or already past Queued)",
                &*bead
            );
            return;
        }

        let t0 = std::time::Instant::now();
        match job {
            Job::Simple(compile) => {
                let code = compile(&bead);
                let elapsed = t0.elapsed();
                info!("bead {:p}: compiled in {elapsed:.2?}", &*bead);
                Self::install_or_recover(&bead, code, None);
            }
            Job::Outcome(compile) => match compile(&bead) {
                CompileOutcome::Ready(code) => {
                    let elapsed = t0.elapsed();
                    info!("bead {:p}: compiled in {elapsed:.2?}", &*bead);
                    Self::install_or_recover(&bead, code, None);
                }
                CompileOutcome::Deferred(resolver) => {
                    let elapsed = t0.elapsed();
                    debug!(
                        "bead {:p}: staged in {elapsed:.2?} (awaiting flush)",
                        &*bead
                    );
                    pending.push((bead, resolver));
                }
            },
            Job::WithOsr(compile) => {
                let result = compile(&bead);
                let elapsed = t0.elapsed();
                info!(
                    "bead {:p}: compiled in {elapsed:.2?} ({} OSR entries)",
                    &*bead,
                    result.osr.len(),
                );
                let installed = bead.install_compiled_with_osr(result.entry, result.osr);
                if !installed {
                    Self::recover_after_failed_install(&bead);
                }
            }
        }
    }

    /// Install a compiled pointer onto `bead`; recover cleanly on failure.
    /// Factored out so both the simple and outcome paths share the policy.
    fn install_or_recover(bead: &Arc<Bead>, code: *mut (), _osr: Option<Vec<OsrEntry>>) {
        let installed = bead.install_compiled(code);
        if !installed {
            Self::recover_after_failed_install(bead);
        }
    }

    fn recover_after_failed_install(bead: &Arc<Bead>) {
        if bead.reload_pending() {
            // A reload() was called mid-compile. Discard the result,
            // clear the flag, and revert to Interpreted so the bead
            // re-promotes on the next invocation.
            bead.clear_reload_pending();
            bead.revert_compiling();
            debug!("bead {:p}: compile discarded (reload pending)", &**bead);
        } else {
            // Genuinely invalid — drive to Deopt.
            bead.invalidate();
            warn!("bead {:p}: compile failed, invalidated", &**bead);
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
        self.submit_job(bead, Job::Simple(Box::new(compile)))
    }

    /// Submit a bead for background compilation, returning a [`CompileOutcome`].
    ///
    /// Use this when the backend may stage the compile (return
    /// [`CompileOutcome::Deferred`]) and the broker was constructed via
    /// [`Broker::with_batching`]. The non-batched broker handles `Deferred`
    /// gracefully too — each deferred resolver runs immediately after its
    /// staging closure, without a shared finalize, so there's no batching
    /// benefit but no correctness loss either.
    ///
    /// `Ready` outcomes from this path are identical to [`Broker::submit`]
    /// in observable behaviour.
    pub fn submit_outcome(
        &self,
        bead: Arc<Bead>,
        compile: impl FnOnce(&Arc<Bead>) -> CompileOutcome + Send + 'static,
    ) -> SubmitResult {
        self.submit_job(bead, Job::Outcome(Box::new(compile)))
    }

    /// Submit a bead for background compilation with OSR entry points.
    ///
    /// Same promotion / broker semantics as [`Broker::submit`], but the
    /// compile closure returns both a normal entry pointer and a list of
    /// OSR entries (one per hot loop header in the compiled code).
    pub fn submit_osr(
        &self,
        bead: Arc<Bead>,
        compile: impl FnOnce(&Arc<Bead>) -> OsrCompileResult + Send + 'static,
    ) -> SubmitResult {
        self.submit_job(bead, Job::WithOsr(Box::new(compile)))
    }

    fn submit_job(&self, bead: Arc<Bead>, job: Job) -> SubmitResult {
        // Win the promotion race — only one caller succeeds.
        if !bead.try_queue() {
            return SubmitResult::AlreadyQueued;
        }

        let msg = Message::Compile {
            bead: Arc::clone(&bead),
            job,
        };

        match self.sender.try_send(msg) {
            Ok(()) => SubmitResult::Accepted,
            Err(TrySendError::Full(_)) => {
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
