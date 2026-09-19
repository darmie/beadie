// beadie-cranelift — Cranelift JIT backend for beadie.

use std::sync::{Arc, Mutex};

use cranelift_codegen::{
    control::ControlPlane,
    ir::{FuncRef, Function, Signature, UserFuncName},
    isa::{TargetFrontendConfig, TargetIsa},
    settings::{self, Configurable, Flags},
    Context,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, ModuleReloc};

use beadie_backend::JitBackend;
use beadie_core::{Bead, CompileOutcome};

// ─────────────────────────────────────────────────────────────────────────────
// CraneliftFunctionDef
// ─────────────────────────────────────────────────────────────────────────────

/// IR container for one Cranelift function.
///
/// Obtain via [`CraneliftBackend::new_def`], fill in the body with
/// [`builder`](CraneliftFunctionDef::builder), then pass to
/// [`beadie_backend::BackendAdapter::on_invoke`] or
/// [`beadie_backend::BoundBead::compile`].
pub struct CraneliftFunctionDef {
    pub ctx: Context,
    pub func_ctx: FunctionBuilderContext,
    pub func_id: FuncId,
    /// Target description required by `FunctionBuilder::finalize`.
    ///
    /// Copied from the backend's ISA at [`CraneliftBackend::new_def`] time so
    /// callers can reach it without holding a borrow on the backend while the
    /// builder is alive. See [`frontend_config`](Self::frontend_config).
    pub frontend_config: TargetFrontendConfig,
}

impl CraneliftFunctionDef {
    /// Create a scoped [`FunctionBuilder`] to construct the IR body.
    /// Drop the builder before handing `self` to the adapter.
    ///
    /// As of cranelift 0.136, `FunctionBuilder::finalize` takes a
    /// [`TargetFrontendConfig`]. Capture it *before* creating the builder,
    /// since the builder mutably borrows `self`:
    ///
    /// ```ignore
    /// let fcfg = def.frontend_config();
    /// let mut b = def.builder();
    /// // ... build the body ...
    /// b.seal_all_blocks();
    /// b.finalize(fcfg);
    /// ```
    pub fn builder(&mut self) -> FunctionBuilder<'_> {
        FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx)
    }

    /// Target description to hand to `FunctionBuilder::finalize`.
    ///
    /// [`TargetFrontendConfig`] is `Copy`, so call this before
    /// [`builder`](Self::builder) and keep the value across the build block.
    pub fn frontend_config(&self) -> TargetFrontendConfig {
        self.frontend_config
    }
}

// SAFETY: Context and FunctionBuilderContext are data containers with no
// thread-local state. Safe to move to the broker thread.
unsafe impl Send for CraneliftFunctionDef {}

// ─────────────────────────────────────────────────────────────────────────────
// CraneliftConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a [`CraneliftBackend`].
///
/// Allows setting Cranelift codegen flags, optimization level, and
/// native symbols that compiled code can call.
///
/// ## Example
/// ```ignore
/// use beadie_cranelift::CraneliftConfig;
///
/// let backend = CraneliftConfig::new()
///     .opt_level("speed")
///     .symbol("my_runtime_fn", my_fn as *const u8)
///     .build()?;
/// ```
pub struct CraneliftConfig {
    flags: Vec<(String, String)>,
    symbols: Vec<(String, *const u8)>,
}

impl CraneliftConfig {
    pub fn new() -> Self {
        Self {
            flags: vec![
                ("use_colocated_libcalls".into(), "false".into()),
                ("is_pic".into(), "false".into()),
            ],
            symbols: Vec::new(),
        }
    }

    /// Set the optimization level: `"none"`, `"speed"`, or `"speed_and_size"`.
    pub fn opt_level(self, level: &str) -> Self {
        self.set("opt_level", level)
    }

    /// Set a Cranelift codegen flag.
    ///
    /// Replaces any existing value for the same key.
    /// See Cranelift's `SharedFlags` documentation for available flags.
    pub fn set(mut self, key: &str, value: &str) -> Self {
        if let Some(entry) = self.flags.iter_mut().find(|(k, _)| k == key) {
            entry.1 = value.into();
        } else {
            self.flags.push((key.into(), value.into()));
        }
        self
    }

    /// Register a native symbol that compiled code can call.
    ///
    /// The pointer must be a valid function pointer for the duration
    /// of the backend's lifetime.
    pub fn symbol(mut self, name: &str, ptr: *const u8) -> Self {
        self.symbols.push((name.into(), ptr));
        self
    }

    /// Build a [`CraneliftBackend`] with these settings.
    pub fn build(self) -> anyhow::Result<CraneliftBackend> {
        CraneliftBackend::with_config(self)
    }
}

impl Default for CraneliftConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CraneliftBackend
// ─────────────────────────────────────────────────────────────────────────────

/// Beadie JIT backend backed by Cranelift's `JITModule`.
///
/// One `JITModule` per backend instance. The module is shared across all beads
/// and protected by a `Mutex`. Code generation runs on the calling thread
/// before the lock is taken and only the definition, finalization and
/// pointer lookup run under it, so a broker with several workers compiles
/// several functions at once and serialises on the module alone.
///
/// ## Single-shot vs batched
///
/// The default [`compile`](JitBackend::compile) implementation invokes
/// `module.finalize_definitions()` once per function — fine for cold
/// startup, but expensive when many small functions cross the hotness
/// threshold in a short window.
///
/// When wired through [`beadie_backend::BackendAdapter::with_policy_batched`],
/// the backend's [`compile_outcome`](JitBackend::compile_outcome) returns
/// [`CompileOutcome::Deferred`] — only `declare_function` and
/// `define_function` run during the staging closure. The broker then calls
/// [`flush`](JitBackend::flush) once per batch (a single
/// `finalize_definitions`), and each pending resolver reads its
/// `get_finalized_function(FuncId)` in the resolved-pointer phase.
pub struct CraneliftBackend {
    module: Arc<Mutex<JITModule>>,
    isa: Arc<dyn TargetIsa>,
}

impl CraneliftBackend {
    /// Create a backend targeting the current host CPU with default settings.
    pub fn new() -> anyhow::Result<Self> {
        Self::with_config(CraneliftConfig::default())
    }

    /// Create a backend with custom configuration.
    pub fn with_config(config: CraneliftConfig) -> anyhow::Result<Self> {
        let mut flag_builder = settings::builder();
        for (key, value) in &config.flags {
            flag_builder.set(key, value)?;
        }
        let flags = Flags::new(flag_builder);
        let isa: Arc<dyn TargetIsa> = cranelift_native::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .finish(flags)?;
        let mut jit_builder =
            JITBuilder::with_isa(Arc::clone(&isa), cranelift_module::default_libcall_names());
        for (name, ptr) in &config.symbols {
            jit_builder.symbol(name, *ptr);
        }
        Ok(Self {
            module: Arc::new(Mutex::new(JITModule::new(jit_builder))),
            isa,
        })
    }

    /// Create a new [`Signature`] with the host's default calling convention.
    pub fn make_signature(&self) -> Signature {
        Signature::new(self.isa.default_call_conv())
    }

    /// Declare a new function in the module and return a def ready for
    /// IR construction.
    #[allow(clippy::result_large_err)]
    pub fn new_def(
        &self,
        sig: Signature,
        name: &str,
    ) -> Result<CraneliftFunctionDef, cranelift_module::ModuleError> {
        let func_id = self
            .module
            .lock()
            .unwrap()
            .declare_function(name, Linkage::Local, &sig)?;
        let mut ctx = Context::new();
        ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);
        Ok(CraneliftFunctionDef {
            ctx,
            func_ctx: FunctionBuilderContext::new(),
            func_id,
            frontend_config: self.isa.frontend_config(),
        })
    }

    /// Declare an imported function and register it in a function being built.
    ///
    /// Call this **before** creating a [`FunctionBuilder`] — it needs
    /// `&mut Function` which conflicts with an active builder.
    ///
    /// Returns a [`FuncRef`] for use in `builder.ins().call(func_ref, ..)`.
    #[allow(clippy::result_large_err)]
    pub fn import_function(
        &self,
        name: &str,
        sig: &Signature,
        func: &mut Function,
    ) -> Result<FuncRef, cranelift_module::ModuleError> {
        let mut module = self.module.lock().unwrap();
        let func_id = module.declare_function(name, Linkage::Import, sig)?;
        Ok(module.declare_func_in_func(func_id, func))
    }

    /// Access the target ISA for building signatures and types.
    pub fn isa(&self) -> &dyn TargetIsa {
        &*self.isa
    }

    /// Generate `def`'s machine code on this thread, then define it in the
    /// module. The lock is held only for the definition, which is what lets
    /// two workers compile at once; `def.ctx` still holds the compiled code
    /// afterwards for a caller that reads its tables.
    #[allow(clippy::result_large_err)]
    fn define_compiled(
        &self,
        def: &mut CraneliftFunctionDef,
    ) -> Result<(), cranelift_module::ModuleError> {
        def.ctx.compile(&*self.isa, &mut ControlPlane::default())?;
        // Re-read through the shared accessor: `compile` hands back a borrow
        // that would keep `def.ctx.func` out of reach below.
        let compiled = def
            .ctx
            .compiled_code()
            .expect("compile succeeded, so the context holds the code");
        let alignment = compiled.buffer.alignment as u64;
        let relocs: Vec<ModuleReloc> = compiled
            .buffer
            .relocs()
            .iter()
            .map(|r| ModuleReloc::from_mach_reloc(r, &def.ctx.func, def.func_id))
            .collect();
        let mut module = self.module.lock().unwrap();
        module.define_function_bytes(def.func_id, alignment, compiled.code_buffer(), &relocs)
    }

    /// Compile like [`JitBackend::compile`], also returning the source map
    /// the function's srclocs describe.
    ///
    /// Cranelift carries a `SourceLoc` on every instruction and the emitter
    /// folds them into runs over the finished code, so a caller that set them
    /// while building can map a machine offset back to whatever it encoded.
    /// The runs are sorted by `start` and cover only the code that was given a
    /// location; a function whose builder never set one yields an empty map.
    ///
    /// This cannot be layered on top of `compile` from outside: the map lives
    /// in the context and `clear_context` drops it.
    #[allow(clippy::result_large_err)]
    pub fn compile_with_source_map(
        &self,
        _bead: &Arc<Bead>,
        mut def: CraneliftFunctionDef,
    ) -> Result<(*mut (), Vec<SourceSpan>), cranelift_module::ModuleError> {
        self.define_compiled(&mut def)?;
        let spans = def
            .ctx
            .compiled_code()
            .map(|code| {
                code.buffer
                    .get_srclocs_sorted()
                    .iter()
                    // The all-ones pattern is Cranelift's "no location", which
                    // is what an instruction the caller never labelled gets.
                    .filter(|run| !run.loc.is_default())
                    .map(|run| SourceSpan {
                        start: run.start,
                        end: run.end,
                        loc: run.loc.bits(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut module = self.module.lock().unwrap();
        module.clear_context(&mut def.ctx);
        module.finalize_definitions()?;
        Ok((module.get_finalized_function(def.func_id) as *mut (), spans))
    }
}

/// One run of machine code that came from a single source location.
///
/// `start` and `end` are byte offsets from the start of the function, and
/// `loc` is whatever the builder put in the `SourceLoc`: Cranelift does not
/// interpret it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceSpan {
    pub start: u32,
    pub end: u32,
    pub loc: u32,
}

impl JitBackend for CraneliftBackend {
    type FunctionDef = CraneliftFunctionDef;
    type Error = cranelift_module::ModuleError;

    /// Single-shot compile — declare + define + finalize + get-pointer all
    /// in one call. Use this when batching isn't needed.
    fn compile(
        &self,
        _bead: &Arc<Bead>,
        mut def: CraneliftFunctionDef,
    ) -> Result<*mut (), cranelift_module::ModuleError> {
        self.define_compiled(&mut def)?;
        let mut module = self.module.lock().unwrap();
        module.clear_context(&mut def.ctx);
        module.finalize_definitions()?;
        Ok(module.get_finalized_function(def.func_id) as *mut ())
    }

    /// Batched compile — stage the function (declare already happened in
    /// `new_def`; this adds `define_function`) and return a resolver that
    /// reads the entry pointer after [`flush`](Self::flush) finalizes.
    fn compile_outcome(
        &self,
        _bead: &Arc<Bead>,
        mut def: CraneliftFunctionDef,
    ) -> Result<CompileOutcome, cranelift_module::ModuleError> {
        let func_id = def.func_id;
        self.define_compiled(&mut def)?;
        self.module.lock().unwrap().clear_context(&mut def.ctx);
        let module_for_resolver = Arc::clone(&self.module);
        Ok(CompileOutcome::Deferred(Box::new(move || {
            let module = module_for_resolver.lock().unwrap();
            module.get_finalized_function(func_id) as *mut ()
        })))
    }

    /// Run `JITModule::finalize_definitions()` once. The broker invokes
    /// this between staging closures and resolver invocations in batched
    /// mode (see [`beadie_backend::BackendAdapter::with_policy_batched`]).
    fn flush(&self) -> Result<(), cranelift_module::ModuleError> {
        self.module.lock().unwrap().finalize_definitions()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_codegen::ir::{types, AbiParam, InstBuilder};

    /// Build a trivial `(i64) -> i64` function that returns `arg + arg`.
    /// Used by both single-shot and batched tests to keep the test
    /// surface tiny — focus is on the dispatch plumbing, not the IR.
    fn build_double_def(backend: &CraneliftBackend, name: &str) -> CraneliftFunctionDef {
        let mut sig = backend.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        let mut def = backend.new_def(sig, name).unwrap();
        let fcfg = def.frontend_config();
        {
            let mut b = def.builder();
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            let arg = b.block_params(entry)[0];
            let sum = b.ins().iadd(arg, arg);
            b.ins().return_(&[sum]);
            b.seal_all_blocks();
            b.finalize(fcfg);
        }
        def
    }

    /// A source map comes back covering the code the builder labelled, and
    /// the function still runs. Two srclocs over two adds, so a map that
    /// collapsed to one run or came back empty fails.
    #[test]
    fn compile_with_source_map_reports_labelled_runs() {
        use cranelift_codegen::ir::SourceLoc;

        let backend = CraneliftBackend::new().expect("cranelift backend");
        let mut sig = backend.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        let mut def = backend.new_def(sig, "mapped").unwrap();
        let fcfg = def.frontend_config();
        {
            let mut b = def.builder();
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            let arg = b.block_params(entry)[0];
            b.set_srcloc(SourceLoc::new(0x1111));
            let doubled = b.ins().iadd(arg, arg);
            b.set_srcloc(SourceLoc::new(0x2222));
            let quad = b.ins().iadd(doubled, doubled);
            b.ins().return_(&[quad]);
            b.seal_all_blocks();
            b.finalize(fcfg);
        }

        let bead = Arc::new(Bead::new(core::ptr::null_mut(), None));
        let (code, map) = backend.compile_with_source_map(&bead, def).unwrap();
        assert!(!code.is_null(), "null entry pointer");

        let locs: Vec<u32> = map.iter().map(|run| run.loc).collect();
        assert!(locs.contains(&0x1111), "first srcloc missing from {map:?}");
        assert!(locs.contains(&0x2222), "second srcloc missing from {map:?}");
        // Sorted by start, and every run is non-empty: a consumer binary
        // searches this by offset.
        assert!(
            map.windows(2).all(|w| w[0].start <= w[1].start),
            "unsorted: {map:?}"
        );
        assert!(
            map.iter().all(|run| run.start < run.end),
            "empty run: {map:?}"
        );
        // The default `!0` location is filtered, never reported.
        assert!(
            !locs.contains(&u32::MAX),
            "unlabelled run reported: {map:?}"
        );

        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code) };
        assert_eq!(f(3), 12);
    }

    /// Sanity: `compile_outcome` returns `Deferred`, the resolver yields
    /// the same entry pointer after `flush()` runs. Verifies the batched
    /// JitBackend impl produces correctly executable native code.
    ///
    /// Bead state transitions are skipped here — they're a broker concern.
    /// The backend's `compile_outcome` only reads `_bead`'s identity, not
    /// its state, so a freshly-`new`'d bead is enough.
    #[test]
    fn batched_compile_outcome_runs_after_flush() {
        let backend = CraneliftBackend::new().expect("cranelift backend");

        let def = build_double_def(&backend, "double_batched");
        let bead = Arc::new(Bead::new(core::ptr::null_mut(), None));

        let outcome = backend.compile_outcome(&bead, def).unwrap();
        let resolver = match outcome {
            CompileOutcome::Deferred(r) => r,
            CompileOutcome::Ready(_) => {
                panic!("batched Cranelift backend should return Deferred")
            }
        };

        backend.flush().unwrap();
        let code = resolver();
        assert!(!code.is_null(), "resolver returned null after flush");

        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code) };
        assert_eq!(f(21), 42);
    }

    /// Two functions staged via `compile_outcome` + one `flush` resolves
    /// both — the headline batched amortisation case.
    #[test]
    fn batched_two_functions_share_one_flush() {
        let backend = CraneliftBackend::new().expect("cranelift backend");

        let def_a = build_double_def(&backend, "two_a");
        let def_b = build_double_def(&backend, "two_b");
        let bead = Arc::new(Bead::new(core::ptr::null_mut(), None));

        let outcome_a = backend.compile_outcome(&bead, def_a).unwrap();
        let outcome_b = backend.compile_outcome(&bead, def_b).unwrap();
        let resolver_a = match outcome_a {
            CompileOutcome::Deferred(r) => r,
            CompileOutcome::Ready(_) => unreachable!(),
        };
        let resolver_b = match outcome_b {
            CompileOutcome::Deferred(r) => r,
            CompileOutcome::Ready(_) => unreachable!(),
        };

        // One finalize covers both staged compiles.
        backend.flush().unwrap();

        let code_a = resolver_a();
        let code_b = resolver_b();
        assert!(!code_a.is_null());
        assert!(!code_b.is_null());
        assert_ne!(
            code_a, code_b,
            "distinct functions must get distinct entries"
        );

        let f_a: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_a) };
        let f_b: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code_b) };
        assert_eq!(f_a(3), 6);
        assert_eq!(f_b(5), 10);
    }

    /// The exact flag set the ash Cranelift tier configures must keep being
    /// accepted by `settings::builder().set()` — cranelift occasionally
    /// renames or drops setting names across versions, and an unknown name is
    /// a hard `SetError`, not a warning. Guards the git-pinned bump.
    #[test]
    fn ash_tier_flag_names_still_accepted() {
        let backend = CraneliftConfig::new()
            .opt_level("speed")
            .set("enable_probestack", "false")
            .set("preserve_frame_pointers", "true")
            .set("is_pic", "false")
            .build()
            .expect("ash tier flag set must be accepted");

        // Flags that are accepted but silently ignored would still pass the
        // line above, so round-trip real code through the configured ISA.
        let def = build_double_def(&backend, "ash_flags_double");
        let bead = Arc::new(Bead::new(core::ptr::null_mut(), None));
        let code = backend.compile(&bead, def).unwrap();
        assert!(!code.is_null());
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code) };
        assert_eq!(f(50), 100);
    }

    /// Single-shot `compile` still works (default-impl-unchanged check).
    #[test]
    fn single_shot_compile_still_works() {
        let backend = CraneliftBackend::new().expect("cranelift backend");
        let def = build_double_def(&backend, "double_single");
        let bead = Arc::new(Bead::new(core::ptr::null_mut(), None));
        let code = backend.compile(&bead, def).unwrap();
        assert!(!code.is_null());
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(code) };
        assert_eq!(f(7), 14);
    }
}
