// beadie-backend — JIT backend abstraction layer.
//
// Depends on beadie-core for Bead, Chain, Broker, Policy, Deopt.
// Provides the trait + adapters; does not link any JIT compiler itself.

pub mod tiered;
pub use tiered::{TieredAdapter, TieredBound};

use std::sync::Arc;

use beadie_core::{
    Bead, Beadie, CompileOutcome, CoreHandle, FlushFn, HotnessPolicy, OsrCompileResult, OsrEntry,
    ReloadOutcome, SwapResult, ThresholdPolicy,
};

// ─────────────────────────────────────────────────────────────────────────────
// CompileError
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct CompileError {
    pub message: String,
}
impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for CompileError {}
impl CompileError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
    pub fn from_err<E: std::error::Error>(e: E) -> Self {
        Self {
            message: e.to_string(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JitBackend trait
// ─────────────────────────────────────────────────────────────────────────────

/// An IR-agnostic compilation backend.
///
/// Implement this to plug any JIT compiler into beadie.
///
/// - `FunctionDef` — the vendor IR container; built by the caller using the
///   native builder API, then passed to [`compile`](JitBackend::compile).
/// - `Error` — compilation failure type.
///
/// ## Batched compilation
///
/// Single-shot backends (one compile → one entry pointer) only need
/// [`compile`](JitBackend::compile). Backends that benefit from batched
/// finalization — Cranelift's `JITModule::finalize_definitions()`, LLVM ORC
/// commits, custom MIR pipelines with one-shot register allocators — can
/// additionally override [`compile_outcome`](JitBackend::compile_outcome)
/// to return [`CompileOutcome::Deferred`], paired with
/// [`flush`](JitBackend::flush) for the once-per-batch finalize step. See
/// [`BackendAdapter::with_batching`].
pub trait JitBackend: Send + Sync + 'static {
    type FunctionDef: Send + 'static;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Compile `def` to native code and return the entry-point pointer.
    /// Return `null` or `Err` on failure — the bead will be deopt'd.
    fn compile(&self, bead: &Arc<Bead>, def: Self::FunctionDef) -> Result<*mut (), Self::Error>;

    /// Compile or stage `def`. Default impl wraps [`compile`](JitBackend::compile)
    /// in [`CompileOutcome::Ready`] — backends that benefit from batched
    /// finalization override this to stage work into a shared module and
    /// return [`CompileOutcome::Deferred`] with a resolver closure that
    /// reads the entry pointer after the broker invokes [`flush`].
    ///
    /// The default implementation preserves backward compatibility with
    /// pre-batching `JitBackend` implementors.
    fn compile_outcome(
        &self,
        bead: &Arc<Bead>,
        def: Self::FunctionDef,
    ) -> Result<CompileOutcome, Self::Error> {
        self.compile(bead, def).map(CompileOutcome::Ready)
    }

    /// Finalize any pending staged compilations. The broker calls this
    /// once per batch cycle (after staging closures, before resolver
    /// invocation) — see [`BackendAdapter::with_batching`].
    ///
    /// Default: no-op. Override when [`compile_outcome`](Self::compile_outcome)
    /// returns [`CompileOutcome::Deferred`].
    fn flush(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OsrBuild — output of an OSR-aware factory
// ─────────────────────────────────────────────────────────────────────────────

/// Output of an OSR-aware factory closure passed to [`BackendAdapter::on_invoke_osr`].
///
/// Carries the backend-specific [`JitBackend::FunctionDef`] that compiles to
/// the main entry point, plus a vector of [`OsrEntry`]s — one per hot loop
/// header the JIT plans to emit. The adapter compiles the `def` via the
/// backend and hands both the resulting code pointer and `osr` entries to
/// the bead atomically.
pub struct OsrBuild<D> {
    pub def: D,
    pub osr: Vec<OsrEntry>,
}

// ─────────────────────────────────────────────────────────────────────────────
// BoundBead
// ─────────────────────────────────────────────────────────────────────────────

/// A bead pre-wired to a specific JIT backend.
#[derive(Clone)]
pub struct BoundBead<B: JitBackend> {
    pub(crate) bead: Arc<Bead>,
    pub(crate) backend: Arc<B>,
}

impl<B: JitBackend> BoundBead<B> {
    pub fn bead(&self) -> &Arc<Bead> {
        &self.bead
    }
    pub fn backend(&self) -> &Arc<B> {
        &self.backend
    }

    /// Eagerly compile and install. Skips the broker — use for pre-compilation.
    pub fn compile(&self, def: B::FunctionDef) -> Result<Arc<Bead>, B::Error> {
        let ptr = self.backend.compile(&self.bead, def)?;
        self.bead.eager_install(ptr);
        Ok(Arc::clone(&self.bead))
    }

    pub fn reload(&self) -> ReloadOutcome {
        self.bead.reload()
    }

    pub fn swap_compiled(&self, new_code: *mut ()) -> Option<SwapResult> {
        self.bead.swap_compiled(new_code)
    }

    /// Tier-up swap: replace the code pointer and the OSR table together.
    /// See [`Bead::swap_compiled_with_osr`].
    pub fn swap_compiled_with_osr(
        &self,
        new_code: *mut (),
        osr: Vec<OsrEntry>,
    ) -> Option<SwapResult> {
        self.bead.swap_compiled_with_osr(new_code, osr)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BackendAdapter
// ─────────────────────────────────────────────────────────────────────────────

/// Beadie wired to a single JIT backend.
pub struct BackendAdapter<B: JitBackend, P: HotnessPolicy = ThresholdPolicy> {
    beadie: Beadie<P>,
    backend: Arc<B>,
}

impl<B: JitBackend> BackendAdapter<B> {
    pub fn new(backend: B) -> Self {
        Self::with_policy(backend, ThresholdPolicy::default())
    }

    /// Construct from an existing `Arc<B>` with the default
    /// [`ThresholdPolicy`]. Useful when the embedder already shares the
    /// backend with other subsystems — a runtime that owns the JIT
    /// alongside a GC, a test harness probing internals, or two adapters
    /// (single-shot + batched) over one backend.
    pub fn from_arc(backend: Arc<B>) -> Self {
        Self::from_arc_with_policy(backend, ThresholdPolicy::default())
    }
}

impl<B: JitBackend, P: HotnessPolicy> BackendAdapter<B, P> {
    /// Construct with a custom policy, taking the backend by value. The
    /// adapter wraps it in its own `Arc<B>`.
    pub fn with_policy(backend: B, policy: P) -> Self {
        Self::from_arc_with_policy(Arc::new(backend), policy)
    }

    /// Construct with a custom policy from an existing `Arc<B>`. The
    /// adapter adopts the supplied `Arc` rather than allocating a fresh
    /// one — callers can keep their own clone and observe the backend's
    /// state from outside.
    ///
    /// Equivalent to [`Self::with_policy`] when you happen to be holding
    /// the backend by value; equivalent to nothing else available when
    /// the backend already lives behind an `Arc` (e.g. shared with a
    /// runtime, GC, or test harness).
    pub fn from_arc_with_policy(backend: Arc<B>, policy: P) -> Self {
        Self {
            beadie: Beadie::with_policy(policy),
            backend,
        }
    }

    /// Construct an adapter that runs the broker in batched mode.
    ///
    /// The broker drains up to `batch_limit` compile jobs without blocking,
    /// invokes the backend's [`JitBackend::flush`] once per batch, then
    /// installs deferred compilations via their resolver closures. This
    /// amortises the fixed cost of `module.finalize_definitions()`-style
    /// finalization across many small functions — relevant for Cranelift,
    /// LLVM ORC, and custom MIR pipelines.
    ///
    /// The `flush` callback registered with the broker delegates to the
    /// backend's `flush()` method on the broker thread. Backends that don't
    /// stage (i.e. always return [`CompileOutcome::Ready`]) can still use
    /// this constructor — the no-op default `flush` impl is harmless.
    ///
    /// Pair with [`Self::on_invoke_outcome`] at the dispatch site to feed
    /// the broker compile closures that can return either `Ready` or
    /// `Deferred`.
    pub fn with_policy_batched(backend: B, policy: P, capacity: usize, batch_limit: usize) -> Self {
        Self::from_arc_with_policy_batched(Arc::new(backend), policy, capacity, batch_limit)
    }

    /// Batched-mode constructor that adopts an existing `Arc<B>`.
    ///
    /// See [`Self::from_arc_with_policy`] for the Arc-sharing rationale;
    /// see [`Self::with_policy_batched`] for the batching semantics.
    pub fn from_arc_with_policy_batched(
        backend: Arc<B>,
        policy: P,
        capacity: usize,
        batch_limit: usize,
    ) -> Self {
        let backend_for_flush = Arc::clone(&backend);
        let flush: FlushFn = Arc::new(move || {
            if let Err(e) = backend_for_flush.flush() {
                eprintln!("beadie: flush error: {e}");
            }
        });
        Self {
            beadie: Beadie::with_policy_batched(policy, capacity, batch_limit, flush),
            backend,
        }
    }

    pub fn register(
        &self,
        core: CoreHandle,
        on_invalidate: Option<Box<dyn Fn() + Send + Sync>>,
    ) -> BoundBead<B> {
        let bead = self.beadie.register(core, on_invalidate);
        BoundBead {
            bead,
            backend: Arc::clone(&self.backend),
        }
    }

    #[inline]
    pub fn on_invoke<F>(&self, bound: &BoundBead<B>, factory: F) -> Option<*mut ()>
    where
        F: FnOnce(&Arc<Bead>) -> B::FunctionDef + Send + 'static,
    {
        let backend = Arc::clone(&bound.backend);
        self.beadie.on_invoke(&bound.bead, move |bead| {
            let def = factory(bead);
            match backend.compile(bead, def) {
                Ok(ptr) => ptr,
                Err(e) => {
                    eprintln!("beadie: compile error: {e}");
                    core::ptr::null_mut()
                }
            }
        })
    }

    /// Batched-compile counterpart to [`Self::on_invoke`].
    ///
    /// Routes the staging through [`JitBackend::compile_outcome`] so a
    /// backend that's been configured for batching can return
    /// [`CompileOutcome::Deferred`] and let the broker's flush hook
    /// finalize. Use with [`Self::with_policy_batched`].
    ///
    /// Behaviourally identical to [`Self::on_invoke`] when the backend
    /// returns `Ready`. The hot path (compiled? + tick) is the same atomic
    /// load + relaxed counter check.
    #[inline]
    pub fn on_invoke_outcome<F>(&self, bound: &BoundBead<B>, factory: F) -> Option<*mut ()>
    where
        F: FnOnce(&Arc<Bead>) -> B::FunctionDef + Send + 'static,
    {
        let backend = Arc::clone(&bound.backend);
        self.beadie.on_invoke_outcome(&bound.bead, move |bead| {
            let def = factory(bead);
            match backend.compile_outcome(bead, def) {
                Ok(outcome) => outcome,
                Err(e) => {
                    eprintln!("beadie: compile error: {e}");
                    CompileOutcome::Ready(core::ptr::null_mut())
                }
            }
        })
    }

    /// OSR-aware dispatch.
    ///
    /// The factory returns an [`OsrBuild`] — the backend function def plus
    /// one [`OsrEntry`] per hot loop header the compiled code will expose.
    /// The adapter compiles `def` via the backend and publishes both the
    /// main entry and the OSR table atomically on the bead.
    ///
    /// Back-edge probes in the runtime use [`Bead::osr_entry`] to look up
    /// a resume point and transfer a live interpreter frame into native code.
    #[inline]
    pub fn on_invoke_osr<F>(&self, bound: &BoundBead<B>, factory: F) -> Option<*mut ()>
    where
        F: FnOnce(&Arc<Bead>) -> OsrBuild<B::FunctionDef> + Send + 'static,
    {
        let backend = Arc::clone(&bound.backend);
        self.beadie.on_invoke_osr(&bound.bead, move |bead| {
            let build = factory(bead);
            let entry = match backend.compile(bead, build.def) {
                Ok(ptr) => ptr,
                Err(e) => {
                    eprintln!("beadie: compile error: {e}");
                    core::ptr::null_mut()
                }
            };
            OsrCompileResult {
                entry,
                osr: build.osr,
            }
        })
    }

    pub fn prune(&self) {
        self.beadie.prune();
    }
    pub fn reload_all(&self) -> usize {
        self.beadie.reload_all()
    }
    pub fn reload_matching(&self, pred: impl Fn(&Arc<Bead>) -> bool) -> usize {
        self.beadie.reload_matching(pred)
    }
    pub fn chain_len(&self) -> usize {
        self.beadie.chain_len()
    }
    pub fn beadie(&self) -> &Beadie<P> {
        &self.beadie
    }
    pub fn backend(&self) -> &Arc<B> {
        &self.backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A trivial backend that counts how many times `compile` and `flush`
    /// are called. Used to prove the from_arc constructors actually share
    /// state with the embedder's Arc rather than allocating a fresh one.
    struct CountingBackend {
        compiles: AtomicUsize,
        flushes: AtomicUsize,
    }

    #[derive(Debug)]
    struct NoErr;
    impl std::fmt::Display for NoErr {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("(no error)")
        }
    }
    impl std::error::Error for NoErr {}

    impl JitBackend for CountingBackend {
        type FunctionDef = ();
        type Error = NoErr;

        fn compile(&self, _bead: &Arc<Bead>, _def: ()) -> Result<*mut (), NoErr> {
            self.compiles.fetch_add(1, Ordering::SeqCst);
            // Return a non-null sentinel so install succeeds.
            Ok(0xdeadbeef as *mut ())
        }

        fn flush(&self) -> Result<(), NoErr> {
            self.flushes.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// from_arc + from_arc_with_policy adopt the caller's Arc — proven by
    /// observing the same backend state through the embedder's clone after
    /// the adapter has driven a compile.
    #[test]
    fn from_arc_shares_backend_state() {
        let shared = Arc::new(CountingBackend {
            compiles: AtomicUsize::new(0),
            flushes: AtomicUsize::new(0),
        });
        let adapter: BackendAdapter<CountingBackend> =
            BackendAdapter::from_arc(Arc::clone(&shared));

        // Eager compile via BoundBead — no broker thread.
        let bound = adapter.register(core::ptr::null_mut(), None);
        let _ = bound.compile(());

        // The embedder's clone observes the compile count.
        assert_eq!(shared.compiles.load(Ordering::SeqCst), 1);
    }

    /// from_arc_with_policy_batched wires the shared backend's flush() into
    /// the broker's flush hook — exercising a real Deferred outcome would
    /// require a busy broker round-trip, so the simpler assertion is that
    /// the adapter's `backend()` accessor returns the same Arc pointer the
    /// caller supplied.
    #[test]
    fn from_arc_batched_keeps_arc_identity() {
        let shared = Arc::new(CountingBackend {
            compiles: AtomicUsize::new(0),
            flushes: AtomicUsize::new(0),
        });
        let adapter: BackendAdapter<CountingBackend> = BackendAdapter::from_arc_with_policy_batched(
            Arc::clone(&shared),
            ThresholdPolicy::default(),
            /*capacity=*/ 8,
            /*batch_limit=*/ 4,
        );
        // The adapter's backend Arc points at the same allocation we
        // supplied — strong_count = 3 (caller's `shared`, adapter's
        // `backend` field, and the closure inside the broker's flush hook).
        assert!(Arc::strong_count(adapter.backend()) >= 2);
        assert!(Arc::ptr_eq(adapter.backend(), &shared));
    }
}
