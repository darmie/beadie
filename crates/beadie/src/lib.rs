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
    AlwaysRecompilePolicy,
    // Deopt
    BailoutInfo,
    // Bead
    Bead,
    BeadState,
    Beadie,
    // Broker
    Broker,
    // Chain + orchestrator
    Chain,
    CoreHandle,
    DeoptDecision,
    DeoptPolicy,
    ExponentialBackoffPolicy,
    // Hotness policy
    HotnessPolicy,
    ReloadOutcome,
    SubmitResult,
    // Swap / reload
    SwapResult,
    ThresholdDeoptPolicy,
    ThresholdPolicy,
    TieredDeoptPolicy,
    TieredPolicy,
};

// ── Backend re-exports ────────────────────────────────────────────────────────
pub use beadie_backend::{
    BackendAdapter, BoundBead, CompileError, JitBackend, TieredAdapter, TieredBound,
};

// ── Optional backend re-exports ───────────────────────────────────────────────
#[cfg(feature = "cranelift")]
pub use beadie_cranelift::{CraneliftBackend, CraneliftConfig, CraneliftFunctionDef};

#[cfg(feature = "llvm")]
pub use beadie_llvm::{LlvmBackend, LlvmFunctionDef};

// Convenience re-export of the deopt sub-module for doc discoverability.
pub mod deopt {
    pub use beadie_core::{
        AlwaysRecompilePolicy, BailoutInfo, DeoptDecision, DeoptPolicy, ExponentialBackoffPolicy,
        ThresholdDeoptPolicy, TieredDeoptPolicy,
    };
}
