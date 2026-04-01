// beadie-llvm — LLVM JIT backend for beadie (via inkwell).

// Adjust the feature flag to match your installed LLVM:
//   https://github.com/TheDan64/inkwell#versioning

use std::sync::{Arc, Mutex};

use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::ExecutionEngine,
    module::Module,
    OptimizationLevel,
};

use beadie_backend::{CompileError, JitBackend};
use beadie_core::Bead;

// ─────────────────────────────────────────────────────────────────────────────
// LlvmFunctionDef
// ─────────────────────────────────────────────────────────────────────────────

/// IR container for one LLVM function.
///
/// Each function gets its own fresh `Module` — independently compiled and
/// added to the execution engine. Obtain via [`LlvmBackend::new_def`].
pub struct LlvmFunctionDef<'ctx> {
    pub module:     Module<'ctx>,
    pub builder:    Builder<'ctx>,
    pub function:   inkwell::values::FunctionValue<'ctx>,
    pub(crate) entry_name: String,
}

// SAFETY: inkwell objects hold LLVM C++ pointers but are safe to move between
// threads when the context is not shared across threads concurrently.
unsafe impl<'ctx> Send for LlvmFunctionDef<'ctx> {}

// ─────────────────────────────────────────────────────────────────────────────
// LlvmBackend
// ─────────────────────────────────────────────────────────────────────────────

/// Beadie JIT backend backed by LLVM's MCJIT engine via inkwell.
///
/// One `ExecutionEngine` per backend. Module addition is serialised through
/// a `Mutex`. Compiled code lives for the engine's lifetime — drop the
/// `LlvmBackend` only after all compiled code has stopped executing.
pub struct LlvmBackend {
    context: Arc<Context>,
    engine:  Mutex<ExecutionEngine<'static>>,
    opt:     OptimizationLevel,
}

// SAFETY: All mutable access to the ExecutionEngine is serialised through
// the Mutex. The Context is only accessed through &self (immutable) or via
// modules that are themselves serialised. No concurrent LLVM C API calls
// occur on the same engine or context.
unsafe impl Send for LlvmBackend {}
unsafe impl Sync for LlvmBackend {}

impl LlvmBackend {
    pub fn new(opt: OptimizationLevel) -> Result<Self, String> {
        let context = Arc::new(Context::create());
        // SAFETY: seed module lifetime erased to 'static; context is
        // kept alive in the same struct via Arc.
        let seed: Module<'static> = unsafe {
            std::mem::transmute(context.create_module("__beadie_seed"))
        };
        let engine = seed.create_jit_execution_engine(opt).map_err(|e| e.to_string())?;
        Ok(Self { context, engine: Mutex::new(engine), opt })
    }

    /// Allocate a new module + function scaffold for the caller to populate.
    pub fn new_def<'ctx>(
        &'ctx self,
        name: &str,
        fn_type: inkwell::types::FunctionType<'ctx>,
    ) -> LlvmFunctionDef<'ctx> {
        let module   = self.context.create_module(name);
        let function = module.add_function(name, fn_type, None);
        let builder  = self.context.create_builder();
        LlvmFunctionDef { module, builder, function, entry_name: name.to_owned() }
    }

    pub fn context(&self) -> &Context { &self.context }

    pub fn optimization_level(&self) -> OptimizationLevel { self.opt }
}

impl JitBackend for LlvmBackend {
    type FunctionDef = LlvmFunctionDef<'static>;
    type Error       = CompileError;

    fn compile(&self, _bead: &Arc<Bead>, def: LlvmFunctionDef<'static>) -> Result<*mut (), CompileError> {
        if !def.function.verify(true) {
            return Err(CompileError::new("LLVM: function verification failed"));
        }
        let name   = def.entry_name.clone();
        let engine = self.engine.lock().unwrap();
        engine.add_module(&def.module)
            .map_err(|_| CompileError::new("LLVM: failed to add module"))?;
        let addr = engine.get_function_address(&name)
            .map_err(|e| CompileError::new(e.to_string()))?;
        Ok(addr as *mut ())
    }
}
