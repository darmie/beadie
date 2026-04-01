// beadie-cranelift — Cranelift JIT backend for beadie.

use std::sync::{Arc, Mutex};

use cranelift_codegen::{
    Context,
    ir::{FuncRef, Function, Signature, UserFuncName},
    isa::TargetIsa,
    settings::{self, Configurable, Flags},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use beadie_backend::JitBackend;
use beadie_core::Bead;

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
    pub ctx:      Context,
    pub func_ctx: FunctionBuilderContext,
    pub func_id:  FuncId,
}

impl CraneliftFunctionDef {
    /// Create a scoped [`FunctionBuilder`] to construct the IR body.
    /// Drop the builder before handing `self` to the adapter.
    pub fn builder(&mut self) -> FunctionBuilder<'_> {
        FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx)
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
    fn default() -> Self { Self::new() }
}

// ─────────────────────────────────────────────────────────────────────────────
// CraneliftBackend
// ─────────────────────────────────────────────────────────────────────────────

/// Beadie JIT backend backed by Cranelift's `JITModule`.
///
/// One `JITModule` per backend instance. The module is shared across all beads
/// and protected by a `Mutex` — compilations are serialised. For parallel
/// compilation, create multiple `CraneliftBackend` instances.
pub struct CraneliftBackend {
    module: Mutex<JITModule>,
    isa:    Arc<dyn TargetIsa>,
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
            .finish(flags)?
            .into();
        let mut jit_builder = JITBuilder::with_isa(
            Arc::clone(&isa),
            cranelift_module::default_libcall_names(),
        );
        for (name, ptr) in &config.symbols {
            jit_builder.symbol(name, *ptr);
        }
        Ok(Self { module: Mutex::new(JITModule::new(jit_builder)), isa })
    }

    /// Create a new [`Signature`] with the host's default calling convention.
    pub fn make_signature(&self) -> Signature {
        Signature::new(self.isa.default_call_conv())
    }

    /// Declare a new function in the module and return a def ready for
    /// IR construction.
    pub fn new_def(
        &self,
        sig: Signature,
        name: &str,
    ) -> Result<CraneliftFunctionDef, cranelift_module::ModuleError> {
        let func_id = self.module.lock().unwrap()
            .declare_function(name, Linkage::Local, &sig)?;
        let mut ctx = Context::new();
        ctx.func = Function::with_name_signature(
            UserFuncName::user(0, func_id.as_u32()), sig,
        );
        Ok(CraneliftFunctionDef { ctx, func_ctx: FunctionBuilderContext::new(), func_id })
    }

    /// Declare an imported function and register it in a function being built.
    ///
    /// Call this **before** creating a [`FunctionBuilder`] — it needs
    /// `&mut Function` which conflicts with an active builder.
    ///
    /// Returns a [`FuncRef`] for use in `builder.ins().call(func_ref, ..)`.
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
    pub fn isa(&self) -> &dyn TargetIsa { &*self.isa }
}

impl JitBackend for CraneliftBackend {
    type FunctionDef = CraneliftFunctionDef;
    type Error       = cranelift_module::ModuleError;

    fn compile(
        &self,
        _bead: &Arc<Bead>,
        mut def: CraneliftFunctionDef,
    ) -> Result<*mut (), cranelift_module::ModuleError> {
        let mut module = self.module.lock().unwrap();
        module.define_function(def.func_id, &mut def.ctx)?;
        module.clear_context(&mut def.ctx);
        module.finalize_definitions()?;
        Ok(module.get_finalized_function(def.func_id) as *mut ())
    }
}
