// deopt.rs — Bailout and deoptimisation policy.
//
// Terminology used here:
//
//   Bailout  — compiled code detects a failed speculation guard at runtime
//              and transfers control back to the interpreter at a known point.
//              The bead is still alive; the compiled code may or may not keep
//              running on future calls depending on the policy decision.
//
//   Deopt    — a policy decision that causes the bead to stop running compiled
//              code. May result in recompilation, permanent blacklisting, or
//              a reversion to a lower tier.
//
//   Blacklist — a bead that will never be compiled again. It stays in the
//               interpreter indefinitely.

use crate::bead::Bead;





// ─────────────────────────────────────────────────────────────────────────────
// BailoutInfo
// ─────────────────────────────────────────────────────────────────────────────

/// Describes a single speculation failure inside compiled code.
///
/// Compiled code emits a runtime guard check — e.g. "this value must be an
/// integer", "this array access must be in bounds". When the check fails, the
/// code bails out and hands this struct back to the runtime.
///
/// All fields are runtime-defined — beadie is entirely agnostic about what
/// the numbers mean. The policy sees them and decides what to do.
#[derive(Debug, Clone)]
pub struct BailoutInfo {
    /// Which guard failed. Defined by the compiler backend.
    /// Examples: type-check guard = 0, bounds guard = 1, overflow guard = 2.
    pub guard_id: u32,

    /// Bytecode offset at which the bailout occurred.
    /// Used by the interpreter to resume execution at the right point (OSR exit).
    pub pc_offset: u32,

    /// Generation of compiled code that bailed.
    /// `0` = tier1 (e.g. Cranelift), `1+` = tier2 (e.g. LLVM post-swap).
    /// Lets the policy distinguish tier1 vs tier2 failures.
    pub generation: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// DeoptDecision
// ─────────────────────────────────────────────────────────────────────────────

/// What the deopt policy decided after a bailout.
///
/// Returned by [`DeoptPolicy::on_bailout`] and acted on by the adapter.
/// Also returned by [`Bead::on_bailout`] so the runtime knows what happened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeoptDecision {
    /// Revert to `Interpreted` and allow re-promotion on future invocations.
    ///
    /// The new compiled code may incorporate the bailout information
    /// (e.g. widened type assumptions) if the factory closure captures it.
    Recompile,

    /// Permanently stop compiling this bead. It stays interpreted forever.
    ///
    /// Use when the function is too polymorphic or too unstable to profit
    /// from JIT compilation.
    Blacklist,

    /// For tier2 bailouts: revert to tier1 code and disable further tier2
    /// promotion. For tier1 bailouts: treated the same as `Blacklist`.
    ///
    /// Useful when LLVM's aggressive speculation fails but Cranelift's
    /// conservative compilation is still profitable.
    RevertToTier1,

    /// Keep the current compiled code running, but suppress recompilation
    /// until the bead's invocation count reaches `until_invocations`.
    ///
    /// Useful for transient mispredictions — give the function time to
    /// accumulate a better profile before recompiling.
    PauseRecompile { until_invocations: u32 },
}

impl DeoptDecision {
    /// True if this decision allows the bead to be compiled again eventually.
    pub fn allows_recompile(&self) -> bool {
        matches!(self, Self::Recompile | Self::PauseRecompile { .. })
    }

    /// True if this decision permanently stops compilation.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Blacklist | Self::RevertToTier1)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DeoptPolicy trait
// ─────────────────────────────────────────────────────────────────────────────

/// Decides what to do when compiled code bails out.
///
/// Implement this to express your speculation recovery strategy.
/// The policy is called infrequently (only on bailouts), so runtime dispatch
/// via `Arc<dyn DeoptPolicy>` is acceptable and preferred over more type params.
///
/// ## Contract
/// - Must be `Send + Sync` — called from the broker or interpreter thread.
/// - Must be `'static` — stored in the adapter.
/// - Should be cheap to call — it's on the bailout path, not the hot path.
pub trait DeoptPolicy: Send + Sync + 'static {
    /// Called each time a bead's compiled code bails out.
    ///
    /// - `bead` — the bead whose compiled code bailed. Read `bead.generation()`
    ///   to determine which tier failed (same as `info.generation`).
    /// - `info` — details of the failed guard.
    /// - `bailout_count` — total number of bailouts this bead has ever had.
    fn on_bailout(
        &self,
        bead: &Bead,
        info: &BailoutInfo,
        bailout_count: u32,
    ) -> DeoptDecision;
}

// ─────────────────────────────────────────────────────────────────────────────
// Standard policy implementations
// ─────────────────────────────────────────────────────────────────────────────

/// Always recompile. Never blacklists.
///
/// Use during development or for functions that are known to benefit from
/// repeated recompilation with updated profiles.
pub struct AlwaysRecompilePolicy;

impl DeoptPolicy for AlwaysRecompilePolicy {
    fn on_bailout(&self, _bead: &Bead, _info: &BailoutInfo, _count: u32) -> DeoptDecision {
        DeoptDecision::Recompile
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Blacklist after a fixed number of bailouts.
///
/// The first `recompile_limit` bailouts trigger recompilation.
/// After that, the bead is blacklisted permanently.
///
/// ## Example
/// ```ignore
/// use beadie::deopt::ThresholdDeoptPolicy;
/// // Recompile up to 3 times, then give up.
/// let policy = ThresholdDeoptPolicy::new(3);
/// ```
pub struct ThresholdDeoptPolicy {
    /// Maximum number of recompilations before blacklisting.
    pub recompile_limit: u32,
}

impl ThresholdDeoptPolicy {
    pub fn new(recompile_limit: u32) -> Self {
        Self { recompile_limit }
    }
}

impl Default for ThresholdDeoptPolicy {
    fn default() -> Self {
        Self { recompile_limit: 3 }
    }
}

impl DeoptPolicy for ThresholdDeoptPolicy {
    fn on_bailout(
        &self,
        _bead: &Bead,
        _info: &BailoutInfo,
        bailout_count: u32,
    ) -> DeoptDecision {
        if bailout_count <= self.recompile_limit {
            DeoptDecision::Recompile
        } else {
            DeoptDecision::Blacklist
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Exponential backoff between recompilations.
///
/// After each bailout, doubles the number of additional invocations required
/// before the next recompile is allowed. After `max_recompiles` total
/// recompilations, blacklists permanently.
///
/// Example progression with `base = 1_000` and `max = 4`:
/// - Bailout 1 → recompile after 2 000 more invocations
/// - Bailout 2 → recompile after 4 000 more invocations
/// - Bailout 3 → recompile after 8 000 more invocations
/// - Bailout 4 → blacklist
pub struct ExponentialBackoffPolicy {
    /// Base pause duration (in invocations) after the first bailout.
    pub base_pause: u32,
    /// Maximum number of recompiles before blacklisting.
    pub max_recompiles: u32,
}

impl ExponentialBackoffPolicy {
    pub fn new(base_pause: u32, max_recompiles: u32) -> Self {
        Self { base_pause, max_recompiles }
    }
}

impl Default for ExponentialBackoffPolicy {
    fn default() -> Self {
        Self { base_pause: 1_000, max_recompiles: 4 }
    }
}

impl DeoptPolicy for ExponentialBackoffPolicy {
    fn on_bailout(
        &self,
        bead: &Bead,
        _info: &BailoutInfo,
        bailout_count: u32,
    ) -> DeoptDecision {
        if bailout_count > self.max_recompiles {
            return DeoptDecision::Blacklist;
        }
        // Pause = base × 2^(bailout_count - 1), resume at current count + pause.
        let pause = self.base_pause.saturating_mul(1u32 << (bailout_count - 1).min(31));
        let resume_at = bead.invocation_count().saturating_add(pause);
        DeoptDecision::PauseRecompile { until_invocations: resume_at }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Tier-aware policy.
///
/// Applies different strategies to tier1 (Cranelift) and tier2 (LLVM) failures:
/// - Tier2 bailouts → `RevertToTier1` (keep running Cranelift code)
/// - Tier1 bailouts → `ThresholdDeoptPolicy` behaviour (recompile up to N times)
///
/// This is the recommended policy for `TieredAdapter`.
pub struct TieredDeoptPolicy {
    /// Max tier1 recompiles before tier1 blacklists.
    pub tier1_recompile_limit: u32,
    /// How many tier2 bailouts before reverting permanently to tier1.
    /// After this, `tier1_only` is set on the `TieredBound`.
    pub tier2_revert_limit: u32,
}

impl TieredDeoptPolicy {
    pub fn new(tier1_recompile_limit: u32, tier2_revert_limit: u32) -> Self {
        Self { tier1_recompile_limit, tier2_revert_limit }
    }
}

impl Default for TieredDeoptPolicy {
    fn default() -> Self {
        Self {
            tier1_recompile_limit: 3,
            tier2_revert_limit: 1,
        }
    }
}

impl DeoptPolicy for TieredDeoptPolicy {
    fn on_bailout(
        &self,
        _bead: &Bead,
        info: &BailoutInfo,
        bailout_count: u32,
    ) -> DeoptDecision {
        if info.generation >= 1 {
            // This was tier2 compiled code that bailed.
            // Revert to tier1 — unless it's bailed too many times.
            if bailout_count <= self.tier2_revert_limit {
                DeoptDecision::RevertToTier1
            } else {
                DeoptDecision::Blacklist
            }
        } else {
            // Tier1 bailed.
            if bailout_count <= self.tier1_recompile_limit {
                DeoptDecision::Recompile
            } else {
                DeoptDecision::Blacklist
            }
        }
    }
}
