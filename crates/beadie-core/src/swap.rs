// swap.rs — Types for hot function swapping and reloading.

// ─────────────────────────────────────────────────────────────────────────────
// SwapResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a successful [`Bead::swap_compiled`] call.
///
/// The old code pointer is returned to the caller for reclamation.
/// Beadie does not manage the lifetime of compiled code — that is the
/// runtime's responsibility.
///
/// ## When is it safe to free `old_code`?
///
/// After a swap, threads that entered the old code before the swap are still
/// executing it. `old_code` must not be freed until all such threads have
/// returned. The right mechanism depends on your runtime:
///
/// - **Safepoint / stop-the-world**: free after the next GC pause.
/// - **Epoch-based reclamation** (`crossbeam-epoch`): defer the free into a
///   `Guard`-pinned deferred closure.
/// - **Quiescent state detection**: free once every thread has passed through
///   a known-safe point (e.g. a function return, a yield, a poll).
///
/// Beadie deliberately does not dictate which model you use.
pub struct SwapResult {
    /// The old compiled code pointer. Reclaim after a quiescent state.
    pub old_code: *mut (),
    /// Generation number **after** this swap. Monotonically increasing.
    pub new_generation: u64,
}

// SAFETY: the pointer is opaque — the runtime owns the pointee.
unsafe impl Send for SwapResult {}
unsafe impl Sync for SwapResult {}

// ─────────────────────────────────────────────────────────────────────────────
// ReloadOutcome
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a [`Bead::reload`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReloadOutcome {
    /// Bead transitioned back to `Interpreted`.
    /// The next hot invocations will trigger recompilation immediately
    /// (the invocation count is preserved, so the threshold is already met).
    WillRecompile,

    /// A compilation was already in flight.
    /// The broker will discard its result and revert the bead to `Interpreted`,
    /// allowing a fresh compile to be triggered.
    PendingCurrentCompile,

    /// Bead was `Queued` but not yet picked up.
    /// Reverted to `Interpreted` — will re-queue on the next invocation.
    Reverted,

    /// Bead was already interpreting — nothing to do.
    AlreadyInterpreting,

    /// Bead is `Deopt` — dead, cannot reload.
    Dead,
}

impl ReloadOutcome {
    /// True if the bead will be recompiled (in any form).
    pub fn will_recompile(self) -> bool {
        matches!(
            self,
            Self::WillRecompile | Self::PendingCurrentCompile | Self::Reverted
        )
    }
}
