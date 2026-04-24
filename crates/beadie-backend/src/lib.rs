// beadie-backend — JIT backend abstraction layer.
//
// Depends on beadie-core for Bead, Chain, Broker, Policy, Deopt.
// Provides the trait + adapters; does not link any JIT compiler itself.

pub mod tiered;
pub use tiered::{TieredAdapter, TieredBound};

use std::sync::Arc;

use beadie_core::{
    Bead, Beadie, CoreHandle, HotnessPolicy, OsrCompileResult, OsrEntry, ReloadOutcome, SwapResult,
    ThresholdPolicy,
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
pub trait JitBackend: Send + Sync + 'static {
    type FunctionDef: Send + 'static;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Compile `def` to native code and return the entry-point pointer.
    /// Return `null` or `Err` on failure — the bead will be deopt'd.
    fn compile(&self, bead: &Arc<Bead>, def: Self::FunctionDef) -> Result<*mut (), Self::Error>;
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
}

impl<B: JitBackend, P: HotnessPolicy> BackendAdapter<B, P> {
    pub fn with_policy(backend: B, policy: P) -> Self {
        Self {
            beadie: Beadie::with_policy(policy),
            backend: Arc::new(backend),
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
