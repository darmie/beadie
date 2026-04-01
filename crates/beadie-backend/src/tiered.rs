// tiered.rs — N-tier JIT coordination.
//
// Supports any number of compilation tiers (e.g. interpreter → Cranelift →
// LLVM → aggressive LLVM). Each tier has its own promotion policy and
// background worker thread.

use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    thread,
};

use crossbeam_channel::{bounded, Sender};
use log::{debug, info, warn};

use beadie_core::{
    Bead, BeadState, Broker, Chain, CoreHandle, HotnessPolicy,
    BailoutInfo, DeoptDecision, DeoptPolicy, TieredDeoptPolicy,
};

// ─────────────────────────────────────────────────────────────────────────────
// PromotionBroker — background worker for tier N+1 (uses swap_compiled)
// ─────────────────────────────────────────────────────────────────────────────

type PromoteFn = Box<dyn FnOnce(&Arc<Bead>) -> *mut () + Send + 'static>;

struct PromotionJob {
    bead: Arc<Bead>,
    compile: PromoteFn,
}

enum PromotionMsg {
    Job(PromotionJob),
    Shutdown,
}

struct PromotionBroker {
    sender: Sender<PromotionMsg>,
    worker: Option<thread::JoinHandle<()>>,
}

impl PromotionBroker {
    fn new(capacity: usize, name: String) -> Self {
        let (tx, rx) = bounded::<PromotionMsg>(capacity);
        let worker = thread::Builder::new()
            .name(name)
            .spawn(move || {
                for msg in &rx {
                    match msg {
                        PromotionMsg::Shutdown => break,
                        PromotionMsg::Job(job) => Self::process(job),
                    }
                }
            })
            .expect("beadie: failed to spawn promoter thread");
        Self { sender: tx, worker: Some(worker) }
    }

    fn process(job: PromotionJob) {
        if !job.bead.is_valid() { return; }
        let t0 = std::time::Instant::now();
        let new_code = (job.compile)(&job.bead);
        let elapsed = t0.elapsed();
        if new_code.is_null() {
            warn!("bead {:p}: tier promotion failed in {elapsed:.2?}, keeping current tier", &*job.bead);
            return;
        }
        info!("bead {:p}: tier promotion compiled in {elapsed:.2?}", &*job.bead);
        // Atomic swap: old code pointer returned; runtime reclaims at quiescent point.
        let _ = job.bead.swap_compiled(new_code);
    }

    fn try_submit(
        &self,
        bead: Arc<Bead>,
        compile: impl FnOnce(&Arc<Bead>) -> *mut () + Send + 'static,
    ) -> bool {
        self.sender
            .try_send(PromotionMsg::Job(PromotionJob { bead, compile: Box::new(compile) }))
            .is_ok()
    }
}

impl Drop for PromotionBroker {
    fn drop(&mut self) {
        let _ = self.sender.send(PromotionMsg::Shutdown);
        if let Some(h) = self.worker.take() { let _ = h.join(); }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TierBroker — either the primary broker (tier 0) or a promotion broker
// ─────────────────────────────────────────────────────────────────────────────

enum TierBroker {
    /// Tier 0 — first compilation. Uses `install_compiled` via the core Broker.
    Primary(Broker),
    /// Tier 1+ — promotion. Uses `swap_compiled` via a PromotionBroker.
    Promotion(PromotionBroker),
}

// ─────────────────────────────────────────────────────────────────────────────
// Tier — one tier's configuration
// ─────────────────────────────────────────────────────────────────────────────

struct Tier {
    policy: Box<dyn HotnessPolicy>,
    broker: TierBroker,
}

// ─────────────────────────────────────────────────────────────────────────────
// TieredBound
// ─────────────────────────────────────────────────────────────────────────────

/// A bead registered with a [`TieredAdapter`].
///
/// Tracks per-tier promotion state. Not generic over backend types — backends
/// are captured in the compile closure passed to [`TieredAdapter::on_invoke`].
#[derive(Clone)]
pub struct TieredBound {
    bead: Arc<Bead>,
    /// One flag per tier >= 1: `queued[0]` = tier 1 queued, etc.
    queued: Vec<Arc<AtomicBool>>,
    /// Maximum tier this bead is allowed to reach. Set by deopt policy.
    max_tier: Arc<AtomicUsize>,
    /// Total number of tiers (for bounds checking and reset).
    num_tiers: usize,
}

impl TieredBound {
    pub fn bead(&self) -> &Arc<Bead> { &self.bead }

    /// Current compilation tier (0 = first compiled tier, 1 = second, etc).
    /// Returns `None` if the bead is not yet compiled.
    pub fn current_tier(&self) -> Option<usize> {
        if self.bead.state() == BeadState::Compiled {
            Some(self.bead.generation() as usize)
        } else {
            None
        }
    }

    pub fn generation(&self) -> u64 { self.bead.generation() }

    /// Whether the bead has been promoted beyond the given tier index.
    pub fn is_promoted_beyond(&self, tier: usize) -> bool {
        self.bead.generation() as usize > tier
    }

    /// Whether promotion to the given tier has been queued.
    pub fn is_queued_for(&self, tier: usize) -> bool {
        if tier == 0 { return false; }
        self.queued.get(tier - 1)
            .map_or(false, |q| q.load(Ordering::Relaxed))
    }

    /// Maximum tier this bead is allowed to reach.
    pub fn max_tier(&self) -> usize {
        self.max_tier.load(Ordering::Acquire)
    }

    /// Full reset: back to `Interpreted`, all queued flags and max_tier cleared.
    pub fn reset_to_interpreter(&self) {
        self.bead.reload();
        for q in &self.queued {
            q.store(false, Ordering::Release);
        }
        self.max_tier.store(self.num_tiers - 1, Ordering::Release);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TieredAdapter
// ─────────────────────────────────────────────────────────────────────────────

/// Coordinates N compilation tiers with independent promotion policies.
///
/// State flow per bead:
/// ```text
/// Interpreted ──[P0]──► tier 0 compile ──► Compiled(gen=0)
///                                               │
///                                             [P1]
///                                               │
///                                       tier 1 compile + swap ──► Compiled(gen=1)
///                                                                       │
///                                                                     [P2]
///                                                                       ▼
///                                                                     ...
/// ```
///
/// Each tier runs uninterrupted while the next tier compiles.
/// Higher-tier failure is silent — the bead keeps running its current tier.
///
/// ## Compile closure
///
/// [`on_invoke`](TieredAdapter::on_invoke) takes a single closure
/// `Fn(tier_index, &Arc<Bead>) -> *mut ()` that dispatches to the appropriate
/// backend based on the tier index:
///
/// ```ignore
/// adapter.on_invoke(&bound, |tier, bead| match tier {
///     0 => cranelift.compile(bead, build_baseline(bead)).unwrap_or(null_mut()),
///     1 => llvm.compile(bead, build_optimized(bead)).unwrap_or(null_mut()),
///     _ => null_mut(),
/// });
/// ```
pub struct TieredAdapter {
    chain:        Arc<Chain>,
    tiers:        Vec<Tier>,
    deopt_policy: Arc<dyn DeoptPolicy>,
}

// ── Constructors ──────────────────────────────────────────────────────────────

impl TieredAdapter {
    /// Create a tiered adapter with the given per-tier promotion policies.
    ///
    /// At least one tier is required. The first policy governs initial
    /// compilation; subsequent policies govern promotion to higher tiers.
    ///
    /// # Panics
    /// Panics if `policies` is empty.
    pub fn new(policies: Vec<Box<dyn HotnessPolicy>>) -> Self {
        Self::with_deopt_policy(policies, TieredDeoptPolicy::default())
    }

    /// Create a tiered adapter with custom policies and a deopt policy.
    pub fn with_deopt_policy(
        policies: Vec<Box<dyn HotnessPolicy>>,
        deopt: impl DeoptPolicy,
    ) -> Self {
        assert!(!policies.is_empty(), "TieredAdapter requires at least one tier");
        let mut tiers = Vec::with_capacity(policies.len());
        for (i, policy) in policies.into_iter().enumerate() {
            let broker = if i == 0 {
                TierBroker::Primary(Broker::default())
            } else {
                TierBroker::Promotion(PromotionBroker::new(
                    256,
                    format!("beadie-promoter-{i}"),
                ))
            };
            tiers.push(Tier { policy, broker });
        }
        Self {
            chain: Arc::new(Chain::new()),
            tiers,
            deopt_policy: Arc::new(deopt),
        }
    }

    /// Number of tiers.
    pub fn num_tiers(&self) -> usize { self.tiers.len() }

    // ── Registration ──────────────────────────────────────────────────────────

    pub fn register(
        &self,
        core: CoreHandle,
        on_invalidate: Option<Box<dyn Fn() + Send + Sync>>,
    ) -> TieredBound {
        let bead = self.chain.push(core, on_invalidate);
        let num_tiers = self.tiers.len();
        let queued = (1..num_tiers)
            .map(|_| Arc::new(AtomicBool::new(false)))
            .collect();
        TieredBound {
            bead,
            queued,
            max_tier: Arc::new(AtomicUsize::new(num_tiers - 1)),
            num_tiers,
        }
    }

    // ── Hot path ──────────────────────────────────────────────────────────────

    /// Call on every invocation.
    ///
    /// The `compile` closure receives `(tier_index, &Arc<Bead>)` and must return
    /// a pointer to compiled native code, or null on failure. It is called at most
    /// once per tier per bead, on the appropriate broker thread.
    ///
    /// The closure must be `Fn + Clone` because it may be cloned when submitting
    /// to different tier brokers. Closures capturing `Arc`-wrapped backends
    /// satisfy these bounds naturally.
    #[inline]
    pub fn on_invoke<F>(&self, bound: &TieredBound, compile: F) -> Option<*mut ()>
    where
        F: Fn(usize, &Arc<Bead>) -> *mut () + Send + Sync + Clone + 'static,
    {
        let (count, state) = bound.bead.tick();

        if bound.bead.is_blacklisted() { return None; }

        if let Some(code) = bound.bead.compiled() {
            self.maybe_promote(bound, count, &compile);
            return Some(code);
        }

        // Not compiled — check tier 0 promotion.
        if state == BeadState::Interpreted && self.tiers[0].policy.should_promote(count) {
            debug!("bead {:p}: tier 0 promotion at invocation {count}", &*bound.bead);
            let c = compile.clone();
            match &self.tiers[0].broker {
                TierBroker::Primary(broker) => {
                    broker.submit(Arc::clone(&bound.bead), move |b| c(0, b));
                }
                _ => unreachable!("tier 0 must use primary broker"),
            }
        }

        None
    }

    fn maybe_promote<F>(&self, bound: &TieredBound, count: u32, compile: &F)
    where
        F: Fn(usize, &Arc<Bead>) -> *mut () + Send + Sync + Clone + 'static,
    {
        let current_gen = bound.bead.generation() as usize;
        let next_tier = current_gen + 1;
        let max = bound.max_tier.load(Ordering::Acquire);

        if next_tier >= self.tiers.len() || next_tier > max {
            return;
        }

        if bound.bead.is_blacklisted() { return; }

        // queued index: tier 1 → queued[0], tier 2 → queued[1], etc.
        let qi = next_tier - 1;
        if bound.queued[qi].load(Ordering::Relaxed) {
            return;
        }

        if !self.tiers[next_tier].policy.should_promote(count) {
            return;
        }

        if bound.queued[qi]
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            debug!(
                "bead {:p}: tier {next_tier} promotion queued at invocation {count}",
                &*bound.bead,
            );
            let c = compile.clone();
            let bead = Arc::clone(&bound.bead);
            let tier = next_tier;
            match &self.tiers[tier].broker {
                TierBroker::Promotion(broker) => {
                    broker.try_submit(bead, move |b| c(tier, b));
                }
                _ => unreachable!("tier >= 1 must use promotion broker"),
            }
        }
    }

    // ── Force promotion ───────────────────────────────────────────────────────

    /// Force immediate promotion to a specific tier, bypassing the threshold.
    ///
    /// Returns `false` if the bead is not compiled, the tier is out of range,
    /// or promotion to that tier has already been queued.
    pub fn force_promote(
        &self,
        bound: &TieredBound,
        target_tier: usize,
        compile: impl FnOnce(&Arc<Bead>) -> *mut () + Send + 'static,
    ) -> bool {
        if bound.bead.compiled().is_none() { return false; }
        if target_tier == 0 || target_tier >= self.tiers.len() { return false; }

        let qi = target_tier - 1;
        if bound.queued[qi]
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return false;
        }

        debug!("bead {:p}: force promote to tier {target_tier}", &*bound.bead);
        match &self.tiers[target_tier].broker {
            TierBroker::Promotion(broker) => {
                broker.try_submit(Arc::clone(&bound.bead), compile)
            }
            _ => false,
        }
    }

    // ── Bailout ───────────────────────────────────────────────────────────────

    /// Runtime entry point for failed speculation guards in compiled code.
    ///
    /// Consults the deopt policy and applies the decision. `RevertToTier1`
    /// caps the bead at tier 0 (the baseline compiled tier) and clears all
    /// promotion-queued flags.
    pub fn on_bailout(&self, bound: &TieredBound, info: BailoutInfo) -> DeoptDecision {
        let decision = bound.bead.on_bailout(info, &*self.deopt_policy);
        if decision == DeoptDecision::RevertToTier1 {
            debug!("bead {:p}: capping at tier 0 (RevertToTier1)", &*bound.bead);
            bound.max_tier.store(0, Ordering::Release);
            for q in &bound.queued {
                q.store(false, Ordering::Release);
            }
        }
        decision
    }

    // ── Management ────────────────────────────────────────────────────────────

    pub fn prune(&self) { self.chain.prune(); }
    pub fn chain_len(&self) -> usize { self.chain.len() }
    pub fn walk(&self, f: impl FnMut(&Arc<Bead>)) { self.chain.walk(f); }
    pub fn deopt_policy(&self) -> &Arc<dyn DeoptPolicy> { &self.deopt_policy }

    pub fn reload_all(&self) -> usize {
        let mut n = 0;
        self.chain.walk(|b| { if b.reload().will_recompile() { n += 1; } });
        n
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use beadie_core::{ThresholdPolicy, BeadState, BailoutInfo, ThresholdDeoptPolicy};
    use std::{
        sync::{atomic::{AtomicUsize, Ordering}, Arc},
        time::Duration,
    };

    fn null_core() -> CoreHandle { core::ptr::null_mut() }

    fn wait_for(f: impl Fn() -> bool, timeout: Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        while !f() {
            if std::time::Instant::now() > deadline { return false; }
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    }

    fn make_adapter(thresholds: &[u32]) -> TieredAdapter {
        TieredAdapter::new(
            thresholds.iter()
                .map(|&t| Box::new(ThresholdPolicy::new(t)) as Box<dyn HotnessPolicy>)
                .collect()
        )
    }

    // Compile closure that returns a fixed pointer per tier.
    // Store as usize to satisfy Send + Sync (raw ptrs are not Send/Sync).
    fn fixed_compile(ptrs: &[*mut ()]) -> impl Fn(usize, &Arc<Bead>) -> *mut () + Send + Sync + Clone + 'static {
        let addrs: Vec<usize> = ptrs.iter().map(|p| *p as usize).collect();
        move |tier, _bead| {
            addrs.get(tier).map_or(core::ptr::null_mut(), |&a| a as *mut ())
        }
    }

    // ── Tier promotion tests ──────────────────────────────────────────────────

    #[test]
    fn tier0_promotes_at_threshold() {
        let adapter = make_adapter(&[3, 100]);
        let bound = adapter.register(null_core(), None);
        let compile = fixed_compile(&[0x1 as *mut (), 0x2 as *mut ()]);

        for _ in 0..3 { adapter.on_invoke(&bound, compile.clone()); }

        assert!(wait_for(|| bound.bead().state() == BeadState::Compiled, Duration::from_secs(2)));
        assert_eq!(adapter.on_invoke(&bound, compile.clone()), Some(0x1 as *mut ()));
    }

    #[test]
    fn tier1_swaps_in_after_threshold() {
        let adapter = make_adapter(&[3, 8]);
        let bound = adapter.register(null_core(), None);
        let compile = fixed_compile(&[0x1 as *mut (), 0x2 as *mut ()]);

        for _ in 0..4 { adapter.on_invoke(&bound, compile.clone()); }
        assert!(wait_for(|| bound.bead().state() == BeadState::Compiled, Duration::from_secs(2)));

        for _ in 0..6 { adapter.on_invoke(&bound, compile.clone()); }
        assert!(wait_for(|| bound.is_promoted_beyond(0), Duration::from_secs(2)));
        assert_eq!(adapter.on_invoke(&bound, compile.clone()), Some(0x2 as *mut ()));
        assert_eq!(bound.generation(), 1);
    }

    #[test]
    fn tier1_only_compiled_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = Arc::clone(&calls);
        let compile = move |tier: usize, _b: &Arc<Bead>| -> *mut () {
            match tier {
                0 => 0x1 as *mut (),
                1 => {
                    calls2.fetch_add(1, Ordering::Relaxed);
                    0x2 as *mut ()
                }
                _ => core::ptr::null_mut(),
            }
        };

        let adapter = make_adapter(&[2, 5]);
        let bound = adapter.register(null_core(), None);

        for _ in 0..20 {
            adapter.on_invoke(&bound, compile.clone());
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(wait_for(|| bound.is_promoted_beyond(0), Duration::from_secs(2)));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn tier0_keeps_running_while_tier1_compiles() {
        let compile = |tier: usize, _b: &Arc<Bead>| -> *mut () {
            match tier {
                0 => 0x1 as *mut (),
                1 => {
                    std::thread::sleep(Duration::from_millis(50));
                    0x2 as *mut ()
                }
                _ => core::ptr::null_mut(),
            }
        };

        let adapter = make_adapter(&[2, 5]);
        let bound = adapter.register(null_core(), None);

        for _ in 0..3 { adapter.on_invoke(&bound, compile.clone()); }
        assert!(wait_for(|| bound.bead().state() == BeadState::Compiled, Duration::from_secs(1)));

        for _ in 0..4 { adapter.on_invoke(&bound, compile.clone()); }
        // Tier 1 still compiling — tier 0 pointer expected
        if !bound.is_promoted_beyond(0) {
            assert_eq!(adapter.on_invoke(&bound, compile.clone()), Some(0x1 as *mut ()));
        }
        assert!(wait_for(|| bound.is_promoted_beyond(0), Duration::from_secs(2)));
        assert_eq!(adapter.on_invoke(&bound, compile.clone()), Some(0x2 as *mut ()));
    }

    #[test]
    fn reset_to_interpreter_clears_all_state() {
        let adapter = make_adapter(&[2, 5]);
        let bound = adapter.register(null_core(), None);
        let compile = fixed_compile(&[0x1 as *mut (), 0x2 as *mut ()]);

        for _ in 0..3 { adapter.on_invoke(&bound, compile.clone()); }
        assert!(wait_for(|| bound.bead().state() == BeadState::Compiled, Duration::from_secs(2)));
        for _ in 0..4 { adapter.on_invoke(&bound, compile.clone()); }
        assert!(wait_for(|| bound.is_promoted_beyond(0), Duration::from_secs(2)));

        bound.reset_to_interpreter();
        assert_eq!(bound.bead().state(), BeadState::Interpreted);
        assert!(!bound.is_queued_for(1));
        assert_eq!(bound.max_tier(), 1); // reset to max (num_tiers - 1)
    }

    // ── Three-tier test ───────────────────────────────────────────────────────

    #[test]
    fn three_tiers_promote_sequentially() {
        let adapter = make_adapter(&[3, 8, 15]);
        let bound = adapter.register(null_core(), None);
        let compile = fixed_compile(&[0x1 as *mut (), 0x2 as *mut (), 0x3 as *mut ()]);

        // Tier 0
        for _ in 0..4 { adapter.on_invoke(&bound, compile.clone()); }
        assert!(wait_for(|| bound.current_tier() == Some(0), Duration::from_secs(2)));
        assert_eq!(adapter.on_invoke(&bound, compile.clone()), Some(0x1 as *mut ()));

        // Tier 1
        for _ in 0..5 { adapter.on_invoke(&bound, compile.clone()); }
        assert!(wait_for(|| bound.current_tier() == Some(1), Duration::from_secs(2)));
        assert_eq!(adapter.on_invoke(&bound, compile.clone()), Some(0x2 as *mut ()));

        // Tier 2
        for _ in 0..6 { adapter.on_invoke(&bound, compile.clone()); }
        assert!(wait_for(|| bound.current_tier() == Some(2), Duration::from_secs(2)));
        assert_eq!(adapter.on_invoke(&bound, compile.clone()), Some(0x3 as *mut ()));
    }

    // ── Force promotion ───────────────────────────────────────────────────────

    #[test]
    fn force_promote_bypasses_threshold() {
        let adapter = make_adapter(&[3, 100_000]);
        let bound = adapter.register(null_core(), None);
        let compile = fixed_compile(&[0x1 as *mut (), 0x2 as *mut ()]);

        // First, get to tier 0
        for _ in 0..4 { adapter.on_invoke(&bound, compile.clone()); }
        assert!(wait_for(|| bound.current_tier() == Some(0), Duration::from_secs(2)));

        // Force tier 1 — well below the 100k threshold
        assert!(adapter.force_promote(&bound, 1, |_| 0x2 as *mut ()));
        assert!(wait_for(|| bound.is_promoted_beyond(0), Duration::from_secs(2)));
        assert_eq!(adapter.on_invoke(&bound, compile.clone()), Some(0x2 as *mut ()));
    }

    // ── Deopt / bailout tests ─────────────────────────────────────────────────

    #[test]
    fn blacklist_stops_all_compilation() {
        let adapter = make_adapter(&[2, 5]);
        let bound = adapter.register(null_core(), None);
        let compile = fixed_compile(&[0x1 as *mut (), 0x2 as *mut ()]);

        bound.bead().blacklist();

        for _ in 0..100 {
            assert!(adapter.on_invoke(&bound, compile.clone()).is_none());
        }
        assert!(bound.bead().is_blacklisted());
        assert_eq!(bound.bead().state(), BeadState::Interpreted);
    }

    #[test]
    fn on_bailout_blacklists_after_threshold() {
        let adapter = TieredAdapter::with_deopt_policy(
            vec![
                Box::new(ThresholdPolicy::new(2)) as Box<dyn HotnessPolicy>,
                Box::new(ThresholdPolicy::new(100)),
            ],
            ThresholdDeoptPolicy::new(2),
        );
        let bound = adapter.register(null_core(), None);
        let compile = fixed_compile(&[0x1 as *mut (), 0x2 as *mut ()]);
        let info = || BailoutInfo { guard_id: 1, pc_offset: 0, generation: 0 };

        for recompile in 1..=2 {
            for _ in 0..3 { adapter.on_invoke(&bound, compile.clone()); }
            assert!(
                wait_for(|| bound.bead().state() == BeadState::Compiled, Duration::from_secs(2)),
                "compile #{recompile} timed out"
            );
            let d = adapter.on_bailout(&bound, info());
            assert!(d.allows_recompile(), "expected Recompile on bailout #{recompile}");
            assert!(!bound.bead().is_blacklisted());
        }

        // Third bailout — exceeds limit
        for _ in 0..3 { adapter.on_invoke(&bound, compile.clone()); }
        assert!(wait_for(|| bound.bead().state() == BeadState::Compiled, Duration::from_secs(2)));
        let d = adapter.on_bailout(&bound, info());
        assert_eq!(d, DeoptDecision::Blacklist);
        assert!(bound.bead().is_blacklisted());

        for _ in 0..20 {
            assert!(adapter.on_invoke(&bound, compile.clone()).is_none());
        }
    }

    #[test]
    fn revert_to_tier0_caps_max_tier() {
        let adapter = TieredAdapter::with_deopt_policy(
            vec![
                Box::new(ThresholdPolicy::new(2)) as Box<dyn HotnessPolicy>,
                Box::new(ThresholdPolicy::new(5)),
            ],
            beadie_core::TieredDeoptPolicy::new(3, 1),
        );
        let bound = adapter.register(null_core(), None);
        let compile = fixed_compile(&[0x1 as *mut (), 0x2 as *mut ()]);

        for _ in 0..3 { adapter.on_invoke(&bound, compile.clone()); }
        assert!(wait_for(|| bound.bead().state() == BeadState::Compiled, Duration::from_secs(2)));
        for _ in 0..5 { adapter.on_invoke(&bound, compile.clone()); }
        assert!(wait_for(|| bound.is_promoted_beyond(0), Duration::from_secs(2)));

        let info = BailoutInfo { guard_id: 0, pc_offset: 0, generation: 1 };
        let d = adapter.on_bailout(&bound, info);
        assert_eq!(d, DeoptDecision::RevertToTier1);
        assert_eq!(bound.max_tier(), 0);
        assert!(!bound.is_queued_for(1));

        // Tier 1 must never be re-queued, even with many invocations
        for _ in 0..50 { adapter.on_invoke(&bound, compile.clone()); }
        std::thread::sleep(Duration::from_millis(20));
        assert!(!bound.is_queued_for(1));
    }
}
