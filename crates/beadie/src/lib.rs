//! # beadie
//!
//! Hot-function promotion broker for interpreter-to-JIT tiering.
//!
//! ## Crate structure
//!
//! | Crate | Contents |
//! |---|---|
//! | `beadie-core` | [`Bead`], [`Chain`], [`Broker`], hotness policy, swap, deopt |
//! | `beadie-backend` | [`JitBackend`] trait, [`BackendAdapter`], [`TieredAdapter`] |
//! | `beadie-cranelift` | [`CraneliftBackend`] (feature `cranelift`) |
//! | `beadie-llvm` | [`LlvmBackend`] (feature `llvm`) |
//! | `beadie` | This facade — re-exports everything |
//!
//! ## Quick start
//!
//! ```ignore
//! use beadie::{Beadie, ThresholdPolicy};
//!
//! let b   = Beadie::new();
//! let bead = b.register(core_ptr, None);
//!
//! loop {
//!     if let Some(code) = b.on_invoke(&bead, |b| my_compile(b)) {
//!         dispatch_native(code);
//!     } else {
//!         interpret();
//!     }
//! }
//! ```

// ── Core re-exports ───────────────────────────────────────────────────────────
pub use beadie_core::{
    // Bead
    Bead, BeadState, CoreHandle,
    // Chain + orchestrator
    Chain, Beadie,
    // Broker
    Broker, SubmitResult,
    // Hotness policy
    HotnessPolicy, ThresholdPolicy, TieredPolicy,
    // Swap / reload
    SwapResult, ReloadOutcome,
    // Deopt
    BailoutInfo, DeoptDecision, DeoptPolicy,
    AlwaysRecompilePolicy, ThresholdDeoptPolicy,
    ExponentialBackoffPolicy, TieredDeoptPolicy,
};

// ── Backend re-exports ────────────────────────────────────────────────────────
pub use beadie_backend::{
    JitBackend, CompileError,
    BackendAdapter, BoundBead,
    TieredAdapter, TieredBound,
};

// ── Optional backend re-exports ───────────────────────────────────────────────
#[cfg(feature = "cranelift")]
pub use beadie_cranelift::{CraneliftBackend, CraneliftConfig, CraneliftFunctionDef};

#[cfg(feature = "llvm")]
pub use beadie_llvm::{LlvmBackend, LlvmFunctionDef};

// Convenience re-export of the deopt sub-module for doc discoverability.
pub mod deopt {
    pub use beadie_core::{
        BailoutInfo, DeoptDecision, DeoptPolicy,
        AlwaysRecompilePolicy, ThresholdDeoptPolicy,
        ExponentialBackoffPolicy, TieredDeoptPolicy,
    };
}
