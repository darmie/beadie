// lib.rs — beadie: hot-function promotion broker for interpreter-to-JIT tiering.
//
//  ┌──────────────────────────────────────────────────────────────┐
//  │  Mental model                                                │
//  │                                                              │
//  │  Chain (yours):  [bead]──►[bead]──►[bead]──► None           │
//  │                     │        │        │                      │
//  │  Runtime (theirs): core?   core?   core?    (opaque)        │
//  │                                                              │
//  │  You own the thread (links).                                 │
//  │  The runtime owns the cores.                                 │
//  │  Beadie manages the promotion from one world to the other.   │
//  └──────────────────────────────────────────────────────────────┘
//
//  Usage sketch
//  ─────────────
//
//  let beadie = Beadie::new();
//
//  // At function definition time:
//  let bead = beadie.register(my_core_ptr, None);
//
//  // In the interpreter dispatch loop (hot path):
//  if let Some(code) = beadie.on_invoke(&bead, |b| my_jit_compile(b)) {
//      dispatch_native(code, args);
//  } else {
//      interpret(bytecode, args);
//  }
//
//  // When the runtime GC moves a core:
//  bead.update_core(new_ptr);
//
//  // When the runtime destroys a core:
//  bead.invalidate();

#![deny(unsafe_op_in_unsafe_fn)]

use log::debug;

mod bead;
mod broker;
mod chain;
mod deopt;
mod policy;
mod swap;

pub use bead::{Bead, BeadState, CoreHandle};
pub use broker::{Broker, SubmitResult};
pub use chain::Chain;
pub use deopt::{
    AlwaysRecompilePolicy, BailoutInfo, DeoptDecision, DeoptPolicy, ExponentialBackoffPolicy,
    ThresholdDeoptPolicy, TieredDeoptPolicy,
};
pub use policy::{HotnessPolicy, ThresholdPolicy, TieredPolicy};
pub use swap::{ReloadOutcome, SwapResult};

use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Beadie — the orchestrator
// ─────────────────────────────────────────────────────────────────────────────

/// Hot-function promotion service.
///
/// Owns a [`Chain`] of registered beads and a [`Broker`] background thread.
/// Call [`Beadie::register`] once per function, then [`Beadie::on_invoke`]
/// on every interpreter dispatch. Everything else is automatic.
///
/// ## Type parameter
/// `P` is the [`HotnessPolicy`] — defaults to [`ThresholdPolicy`] (1 000
/// invocations). Supply [`TieredPolicy`] or your own for custom strategies.
///
/// ## Example
/// ```ignore
/// use beadie::{Beadie, TieredPolicy};
///
/// let beadie = Beadie::with_policy(TieredPolicy::default());
/// ```
pub struct Beadie<P: HotnessPolicy = ThresholdPolicy> {
    chain: Arc<Chain>,
    broker: Broker,
    policy: P,
}

// ── Constructors ──────────────────────────────────────────────────────────────

impl Beadie {
    /// Create a `Beadie` with the default [`ThresholdPolicy`] (1 000
    /// invocations) and a broker queue capacity of 256.
    pub fn new() -> Self {
        Self::with_policy(ThresholdPolicy::default())
    }
}

impl Default for Beadie {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: HotnessPolicy> Beadie<P> {
    /// Create a `Beadie` with a custom hotness policy.
    pub fn with_policy(policy: P) -> Self {
        Self {
            chain: Arc::new(Chain::new()),
            broker: Broker::default(),
            policy,
        }
    }

    /// Create a `Beadie` with a custom policy and broker queue capacity.
    pub fn with_policy_and_capacity(policy: P, capacity: usize) -> Self {
        Self {
            chain: Arc::new(Chain::new()),
            broker: Broker::with_capacity(capacity),
            policy,
        }
    }

    // ── Core API ──────────────────────────────────────────────────────────────

    /// Register a function. Returns an `Arc<Bead>` to store alongside it.
    ///
    /// - `core` — opaque pointer to the runtime's function representation.
    ///   Beadie stores but never dereferences it.
    /// - `on_invalidate` — optional callback fired when `bead.invalidate()`
    ///   is first called. Use a `Weak<Bead>` capture to avoid reference cycles.
    pub fn register(
        &self,
        core: CoreHandle,
        on_invalidate: Option<Box<dyn Fn() + Send + Sync>>,
    ) -> Arc<Bead> {
        self.chain.push(core, on_invalidate)
    }

    /// Call from the interpreter on **every function invocation**.
    ///
    /// ## Returns
    /// - `Some(ptr)` — compiled code is ready; dispatch to it.
    /// - `None` — keep interpreting.
    ///
    /// ## The compile closure
    /// The `compile` closure is only ever called once per bead, by the broker
    /// thread, when the bead wins promotion. It receives the bead so it can
    /// call `bead.core_handle()` whenever it needs the (always-current) core
    /// pointer. It must return a pointer to the compiled native code, or null
    /// on failure.
    ///
    /// The closure must be `'static` — capture any resources via `Arc`.
    ///
    /// ## Hot-path cost
    /// When compiled: one `Acquire` load (branch predicted after warmup).  
    /// When cold: one `Relaxed` fetch-add + one `Relaxed` load.  
    /// The chain lock is **never taken** on this path.
    #[inline]
    pub fn on_invoke<F>(&self, bead: &Arc<Bead>, compile: F) -> Option<*mut ()>
    where
        F: FnOnce(&Arc<Bead>) -> *mut () + Send + 'static,
    {
        // Fast path: already compiled — no tick, no policy check.
        if let Some(code) = bead.compiled() {
            return Some(code);
        }

        let (count, state) = bead.tick();

        if state == BeadState::Interpreted && self.policy.should_promote(count) {
            debug!("bead {:p}: promoting at invocation {count}", &**bead);
            self.broker.submit(Arc::clone(bead), compile);
        }

        None
    }

    // ── Management ────────────────────────────────────────────────────────────

    /// Prune deopt'd beads from the chain, releasing their memory.
    ///
    /// Call this periodically — on GC cycles, or every N invocations via a
    /// separate housekeeping thread. Does not block the interpreter.
    pub fn prune(&self) {
        self.chain.prune();
    }

    /// Walk every bead in the chain. Holds the chain lock for the duration.
    ///
    /// Use for inspection, bulk invalidation, or metrics collection.
    /// Do **not** call from the interpreter hot path.
    pub fn walk(&self, f: impl FnMut(&Arc<Bead>)) {
        self.chain.walk(f);
    }

    /// Number of beads currently in the chain (approximate).
    pub fn chain_len(&self) -> usize {
        self.chain.len()
    }

    /// Access the underlying chain directly for advanced use cases.
    pub fn chain(&self) -> &Arc<Chain> {
        &self.chain
    }

    // ── Hot reload ────────────────────────────────────────────────────────────

    /// Reload every compiled bead — force full recompilation of all hot functions.
    ///
    /// Returns the number of beads that will recompile.
    /// Useful for development hot-reload or after a global optimization change.
    pub fn reload_all(&self) -> usize {
        let mut count = 0;
        self.chain.walk(|bead| {
            if bead.reload().will_recompile() {
                count += 1;
            }
        });
        count
    }

    /// Reload beads matching a predicate.
    ///
    /// ```ignore
    /// # use beadie::{Beadie, Bead};
    /// # use std::sync::Arc;
    /// # let beadie = Beadie::new();
    /// // Reload only functions that have been compiled more than 10_000 times.
    /// beadie.reload_matching(|b: &Arc<Bead>| b.invocation_count() > 10_000);
    /// ```
    pub fn reload_matching(&self, predicate: impl Fn(&Arc<Bead>) -> bool) -> usize {
        let mut count = 0;
        self.chain.walk(|bead| {
            if predicate(bead) && bead.reload().will_recompile() {
                count += 1;
            }
        });
        count
    }

    /// Atomically swap compiled code across all beads matching a predicate.
    ///
    /// `new_code_for` receives the bead and returns the new code pointer
    /// (or null to skip that bead). Returns a `Vec` of [`SwapResult`]s —
    /// one per successfully swapped bead — for the caller to reclaim old
    /// code at the appropriate quiescent point.
    pub fn swap_matching(&self, new_code_for: impl Fn(&Arc<Bead>) -> *mut ()) -> Vec<SwapResult> {
        let mut results = Vec::new();
        self.chain.walk(|bead| {
            let new_code = new_code_for(bead);
            if !new_code.is_null() {
                if let Some(result) = bead.swap_compiled(new_code) {
                    results.push(result);
                }
            }
        });
        results
    }

    // ── Direct broker submission ──────────────────────────────────────────────

    /// Submit a bead to the broker for background compilation without
    /// going through the invocation-policy tick in [`Self::on_invoke`].
    ///
    /// Runtimes with their own tick machinery (for example, Bead is
    /// ticked elsewhere via [`Bead::tick`] and the runtime decides when
    /// to promote) use this to schedule the actual compile while keeping
    /// the tick path decoupled from submission.
    ///
    /// Same semantics as the internal broker submit: the job is only
    /// created if the bead wins the promotion CAS (Interpreted → Queued).
    /// Returns [`SubmitResult`] so the caller can distinguish Accepted
    /// from AlreadyQueued / QueueFull / BrokerShutDown.
    pub fn submit(
        &self,
        bead: &Arc<Bead>,
        compile: impl FnOnce(&Arc<Bead>) -> *mut () + Send + 'static,
    ) -> SubmitResult {
        self.broker.submit(Arc::clone(bead), compile)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::manual_dangling_ptr)]
mod tests {
    use super::*;
    use std::{
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
        time::Duration,
    };

    fn null_core() -> CoreHandle {
        core::ptr::null_mut()
    }

    #[test]
    fn bead_state_machine() {
        let bead = Bead::new(null_core(), None);
        assert_eq!(bead.state(), BeadState::Interpreted);
        assert!(bead.try_queue());
        assert_eq!(bead.state(), BeadState::Queued);
        // Second try_queue must fail — CAS rejects.
        assert!(!bead.try_queue());
        assert!(bead.mark_compiling());
        assert_eq!(bead.state(), BeadState::Compiling);
        assert!(bead.install_compiled(0x1 as *mut ())); // non-null sentinel
        assert_eq!(bead.state(), BeadState::Compiled);
        assert!(bead.compiled().is_some());
    }

    #[test]
    fn invalidation_is_idempotent() {
        let fired = Arc::new(AtomicBool::new(false));
        let fired2 = Arc::clone(&fired);
        let bead = Bead::new(
            null_core(),
            Some(Box::new(move || {
                fired2.store(true, Ordering::SeqCst);
            })),
        );
        bead.invalidate();
        bead.invalidate(); // second call is a no-op
        assert!(fired.load(Ordering::SeqCst));
        assert_eq!(bead.state(), BeadState::Deopt);
    }

    #[test]
    fn install_rejected_after_invalidation() {
        let bead = Bead::new(null_core(), None);
        assert!(bead.try_queue());
        assert!(bead.mark_compiling());
        bead.invalidate(); // invalidated mid-compile
        assert!(!bead.install_compiled(0x1 as *mut ()));
        assert_eq!(bead.state(), BeadState::Deopt);
    }

    #[test]
    fn chain_push_and_prune() {
        let chain = Chain::new();
        let b1 = chain.push(null_core(), None);
        let b2 = chain.push(null_core(), None);
        assert_eq!(chain.len(), 2);

        b1.invalidate(); // b1 → Deopt
        chain.prune();
        assert_eq!(chain.len(), 1);

        // b2 still reachable via walk
        let mut seen = 0usize;
        chain.walk(|b| {
            assert!(Arc::ptr_eq(b, &b2));
            seen += 1;
        });
        assert_eq!(seen, 1);
    }

    #[test]
    fn promotion_end_to_end() {
        // Use a very low threshold so the test doesn't spin 1 000 times.
        let beadie = Beadie::with_policy(ThresholdPolicy::new(5));
        let bead = beadie.register(null_core(), None);

        let compiled = Arc::new(AtomicBool::new(false));
        let compiled2 = Arc::clone(&compiled);

        // Warm up — 4 invocations, below threshold.
        for _ in 0..4 {
            let c = Arc::clone(&compiled2);
            assert!(beadie
                .on_invoke(&bead, move |_| {
                    c.store(true, Ordering::SeqCst);
                    0x1 as *mut () // sentinel non-null code ptr
                })
                .is_none());
        }

        // 5th invocation crosses threshold — broker is notified.
        {
            let c = Arc::clone(&compiled2);
            beadie.on_invoke(&bead, move |_| {
                c.store(true, Ordering::SeqCst);
                0x1 as *mut ()
            });
        }

        // Give the broker thread time to process.
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            if compiled.load(Ordering::SeqCst) {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "broker timeout");
            std::thread::sleep(Duration::from_millis(1));
        }

        // Next on_invoke should hit the compiled fast path.
        let code = beadie.on_invoke(&bead, |_| unreachable!());
        assert_eq!(code, Some(0x1 as *mut ()));
    }

    #[test]
    fn swap_compiled_replaces_code_and_bumps_generation() {
        let bead = Bead::new(null_core(), None);
        assert!(bead.try_queue());
        assert!(bead.mark_compiling());
        assert!(bead.install_compiled(0x1 as *mut ()));

        let result = bead.swap_compiled(0x2 as *mut ()).expect("should swap");
        assert_eq!(result.old_code, 0x1 as *mut ());
        assert_eq!(result.new_generation, 1);
        assert_eq!(bead.compiled(), Some(0x2 as *mut ()));
        assert_eq!(bead.generation(), 1);

        // Second swap
        let result2 = bead.swap_compiled(0x3 as *mut ()).expect("should swap");
        assert_eq!(result2.old_code, 0x2 as *mut ());
        assert_eq!(result2.new_generation, 2);
    }

    #[test]
    fn swap_returns_none_when_not_compiled() {
        let bead = Bead::new(null_core(), None);
        assert!(bead.swap_compiled(0x1 as *mut ()).is_none());
    }

    #[test]
    fn reload_compiled_reverts_to_interpreted() {
        let bead = Bead::new(null_core(), None);
        assert!(bead.try_queue());
        assert!(bead.mark_compiling());
        assert!(bead.install_compiled(0x1 as *mut ()));
        assert_eq!(bead.state(), BeadState::Compiled);

        let outcome = bead.reload();
        assert_eq!(outcome, ReloadOutcome::WillRecompile);
        assert_eq!(bead.state(), BeadState::Interpreted);
        assert!(bead.compiled().is_none());
    }

    #[test]
    fn reload_mid_compile_sets_pending_and_broker_reverts() {
        let bead = Bead::new(null_core(), None);
        // Simulate: broker picked up job, now compiling
        assert!(bead.try_queue());
        assert!(bead.mark_compiling());

        // reload() called while broker is compiling
        assert_eq!(bead.reload(), ReloadOutcome::PendingCurrentCompile);
        assert!(bead.reload_pending());

        // Broker finishes — install_compiled returns false due to pending flag
        assert!(!bead.install_compiled(0x1 as *mut ()));

        // Broker detects pending, reverts cleanly
        bead.clear_reload_pending();
        bead.revert_compiling();
        assert_eq!(bead.state(), BeadState::Interpreted);
        assert!(!bead.reload_pending());
    }

    #[test]
    fn beadie_reload_all() {
        let beadie = Beadie::with_policy(ThresholdPolicy::new(1));
        let b1 = beadie.register(null_core(), None);
        let b2 = beadie.register(null_core(), None);

        // Manually drive both beads to Compiled.
        for b in [&b1, &b2] {
            assert!(b.try_queue());
            assert!(b.mark_compiling());
            assert!(b.install_compiled(0x1 as *mut ()));
        }

        let reloaded = beadie.reload_all();
        assert_eq!(reloaded, 2);
        assert_eq!(b1.state(), BeadState::Interpreted);
        assert_eq!(b2.state(), BeadState::Interpreted);
    }

    #[test]
    fn beadie_reload_matching_selective() {
        let beadie = Beadie::with_policy(ThresholdPolicy::new(1));
        let b1 = beadie.register(null_core(), None);
        let b2 = beadie.register(null_core(), None);

        for b in [&b1, &b2] {
            assert!(b.try_queue());
            assert!(b.mark_compiling());
            assert!(b.install_compiled(0x1 as *mut ()));
        }

        // Tick b1 extra times so it passes our predicate
        for _ in 0..10 {
            b1.tick();
        }

        let reloaded = beadie.reload_matching(|b| b.invocation_count() > 5);
        assert_eq!(reloaded, 1);
        assert_eq!(b1.state(), BeadState::Interpreted);
        assert_eq!(b2.state(), BeadState::Compiled); // untouched
    }

    #[test]
    fn core_handle_survives_update() {
        let bead = Bead::new(0x10 as *mut (), None);
        assert_eq!(bead.core_handle(), 0x10 as *mut ());
        bead.update_core(0x20 as *mut ());
        assert_eq!(bead.core_handle(), 0x20 as *mut ());
    }
}
