// policy.rs — Hotness promotion policies.

// ─────────────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────────────

/// Decides when a bead's invocation count warrants JIT promotion.
///
/// Implement this to express any promotion strategy — simple threshold,
/// weighted decay, profile-guided, or tiered.
pub trait HotnessPolicy: Send + Sync + 'static {
    /// Return `true` when `invocations` has crossed the promotion threshold.
    fn should_promote(&self, invocations: u32) -> bool;
}

// ─────────────────────────────────────────────────────────────────────────────
// ThresholdPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// Promote when invocation count reaches a fixed threshold.
///
/// The default threshold is 1 000 invocations — a reasonable starting point
/// that lets the interpreter gather type feedback before committing to JIT.
///
/// ## Queue-ahead
///
/// Compilation happens on a background thread and takes non-zero time.
/// Set `queue_ahead` to submit the compile job early, so compiled code
/// is ready by the time the function is truly hot:
///
/// ```ignore
/// // Target: compiled by invocation 1000, start compiling 200 ticks early.
/// ThresholdPolicy::new(1000).queue_ahead(200)
/// ```
///
/// The compile job is submitted at `threshold - queue_ahead` invocations.
pub struct ThresholdPolicy {
    pub threshold: u32,
    queue_ahead_offset: u32,
}

impl ThresholdPolicy {
    pub fn new(threshold: u32) -> Self {
        Self {
            threshold,
            queue_ahead_offset: 0,
        }
    }

    /// Submit the compile job this many invocations before the threshold.
    pub fn queue_ahead(mut self, offset: u32) -> Self {
        self.queue_ahead_offset = offset;
        self
    }

    /// The effective invocation count at which the compile job is submitted.
    pub fn queue_at(&self) -> u32 {
        self.threshold.saturating_sub(self.queue_ahead_offset)
    }
}

impl Default for ThresholdPolicy {
    fn default() -> Self {
        Self {
            threshold: 1_000,
            queue_ahead_offset: 0,
        }
    }
}

impl HotnessPolicy for ThresholdPolicy {
    #[inline]
    fn should_promote(&self, invocations: u32) -> bool {
        invocations >= self.threshold.saturating_sub(self.queue_ahead_offset)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TieredPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// Two-tier promotion policy.
///
/// - **Tier 1** (`tier1_threshold`): promote for a fast, lightly-optimised
///   compile (e.g. baseline JIT with no inlining).
/// - **Tier 2** (`tier2_threshold`): promote again for a heavy, optimising
///   compile (e.g. full speculative optimisation).
///
/// `should_promote` fires at tier-1. Your compile closure can call
/// [`TieredPolicy::tier`] to decide *how aggressively* to compile.
pub struct TieredPolicy {
    pub tier1_threshold: u32,
    pub tier2_threshold: u32,
}

impl Default for TieredPolicy {
    fn default() -> Self {
        Self {
            tier1_threshold: 500,
            tier2_threshold: 10_000,
        }
    }
}

impl TieredPolicy {
    /// Which compilation tier is appropriate at this invocation count?
    /// Returns `None` if below tier-1 threshold (still cold).
    pub fn tier(&self, invocations: u32) -> Option<u8> {
        if invocations >= self.tier2_threshold {
            Some(2)
        } else if invocations >= self.tier1_threshold {
            Some(1)
        } else {
            None
        }
    }
}

impl HotnessPolicy for TieredPolicy {
    #[inline]
    fn should_promote(&self, invocations: u32) -> bool {
        invocations >= self.tier1_threshold
    }
}
