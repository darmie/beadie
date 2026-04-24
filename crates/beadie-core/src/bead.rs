// bead.rs — The bead: stable heap node, opaque core, atomic state machine.

use std::sync::{
    atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicU8, Ordering},
    Arc,
};

use crossbeam_epoch::{self as epoch, Atomic, Owned, Shared};
use log::{debug, info, trace, warn};

use crate::deopt::{BailoutInfo, DeoptDecision, DeoptPolicy};
use crate::osr::{OsrEntry, OsrTable};
use crate::swap::{ReloadOutcome, SwapResult};

// ─────────────────────────────────────────────────────────────────────────────
// CoreHandle
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque pointer to a runtime-managed core object.
/// Beadie stores but **never dereferences** this pointer.
pub type CoreHandle = *mut ();

// ─────────────────────────────────────────────────────────────────────────────
// BeadState
// ─────────────────────────────────────────────────────────────────────────────

/// Lifecycle of a bead's compiled representation.
///
/// ```text
///   Interpreted ──try_queue()──► Queued ──mark_compiling()──► Compiling
///        ▲                                                          │
///        │  (queue full / revert / reload / deopt)      install_compiled()
///        │                                                          │
///        └──────────────────── (invalidate anytime) ──► Deopt   Compiled
/// ```
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeadState {
    Interpreted = 0,
    Queued = 1,
    Compiling = 2,
    Compiled = 3,
    Deopt = 4,
}

impl TryFrom<u8> for BeadState {
    type Error = ();
    fn try_from(v: u8) -> Result<Self, ()> {
        Ok(match v {
            0 => Self::Interpreted,
            1 => Self::Queued,
            2 => Self::Compiling,
            3 => Self::Compiled,
            4 => Self::Deopt,
            _ => return Err(()),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bead
// ─────────────────────────────────────────────────────────────────────────────

pub struct Bead {
    /// Opaque core pointer. Runtime updates via `update_core`. Never dereferenced here.
    core: AtomicPtr<()>,
    valid: AtomicBool,
    invocations: AtomicU32,
    state_atom: AtomicU8,
    compiled_code: AtomicPtr<()>,
    on_invalidate: Option<Box<dyn Fn() + Send + Sync>>,
    generation: AtomicU64,
    reload_pending: AtomicBool,

    // ── OSR ────────────────────────────────────────────────────────────────
    /// Optional OSR entry table. Null when the runtime doesn't use OSR,
    /// which is the common case — zero cost for non-OSR runtimes.
    /// Installed alongside compiled code; lookup is lock-free.
    ///
    /// Uses `crossbeam-epoch` so superseded tables can be safely reclaimed
    /// after all threads that might still hold a reference have moved on.
    osr_table: Atomic<OsrTable>,

    // ── Deopt / bailout fields ──────────────────────────────────────────────
    /// Total bailout count across all compiled versions.
    bailout_count: AtomicU32,
    /// Permanently stop compiling. Set by `blacklist()`.
    blacklisted: AtomicBool,
    /// Suppress promotion until invocation count reaches this value.
    /// Set by `PauseRecompile`. Zero means no pause.
    recompile_after: AtomicU32,
}

impl Bead {
    pub fn new(core: CoreHandle, on_invalidate: Option<Box<dyn Fn() + Send + Sync>>) -> Arc<Self> {
        Arc::new(Self {
            core: AtomicPtr::new(core),
            valid: AtomicBool::new(true),
            invocations: AtomicU32::new(0),
            state_atom: AtomicU8::new(BeadState::Interpreted as u8),
            compiled_code: AtomicPtr::new(core::ptr::null_mut()),
            on_invalidate,
            generation: AtomicU64::new(0),
            reload_pending: AtomicBool::new(false),
            osr_table: Atomic::null(),
            bailout_count: AtomicU32::new(0),
            blacklisted: AtomicBool::new(false),
            recompile_after: AtomicU32::new(0),
        })
    }

    // ── Hot path ──────────────────────────────────────────────────────────────

    #[inline]
    pub fn tick(&self) -> (u32, BeadState) {
        let n = self.invocations.fetch_add(1, Ordering::Relaxed) + 1;
        let s = BeadState::try_from(self.state_atom.load(Ordering::Relaxed))
            .unwrap_or(BeadState::Deopt);
        (n, s)
    }

    #[inline]
    pub fn compiled(&self) -> Option<*mut ()> {
        if self.state_atom.load(Ordering::Acquire) == BeadState::Compiled as u8 {
            let p = self.compiled_code.load(Ordering::Acquire);
            if !p.is_null() {
                return Some(p);
            }
        }
        None
    }

    // ── State transitions (crate-internal) ───────────────────────────────────

    pub(crate) fn try_queue(&self) -> bool {
        if self.blacklisted.load(Ordering::Acquire) {
            return false;
        }
        let after = self.recompile_after.load(Ordering::Acquire);
        if after > 0 && self.invocations.load(Ordering::Relaxed) < after {
            return false;
        }
        let ok = self
            .state_atom
            .compare_exchange(
                BeadState::Interpreted as u8,
                BeadState::Queued as u8,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok();
        if ok {
            debug!(
                "bead {:p}: Interpreted -> Queued (invocations={})",
                self,
                self.invocations.load(Ordering::Relaxed),
            );
        }
        ok
    }

    pub(crate) fn revert_queued(&self) {
        let ok = self
            .state_atom
            .compare_exchange(
                BeadState::Queued as u8,
                BeadState::Interpreted as u8,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok();
        if ok {
            debug!("bead {:p}: Queued -> Interpreted (reverted)", self);
        }
    }

    pub(crate) fn mark_compiling(&self) -> bool {
        let ok = self.is_valid()
            && self
                .state_atom
                .compare_exchange(
                    BeadState::Queued as u8,
                    BeadState::Compiling as u8,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_ok();
        if ok {
            debug!("bead {:p}: Queued -> Compiling", self);
        }
        ok
    }

    pub(crate) fn install_compiled(&self, code: *mut ()) -> bool {
        if !self.is_valid() || code.is_null() {
            return false;
        }
        if self.reload_pending.load(Ordering::Acquire) {
            return false;
        }
        self.compiled_code.store(code, Ordering::Release);
        let ok = self
            .state_atom
            .compare_exchange(
                BeadState::Compiling as u8,
                BeadState::Compiled as u8,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok();
        if ok {
            info!(
                "bead {:p}: Compiling -> Compiled (code={code:p}, invocations={})",
                self,
                self.invocations.load(Ordering::Relaxed),
            );
        }
        ok
    }

    /// Install compiled code together with an OSR entry table.
    ///
    /// The OSR table is published before the `Compiling -> Compiled` state
    /// transition; `osr_entry()` gates on that state, so any reader that
    /// sees the transition will also see the matching table.
    ///
    /// Reclamation of any previous OSR table is deferred via
    /// `crossbeam-epoch`: the old table is dropped once every thread that
    /// may hold a reference to it has moved past the current epoch.
    pub(crate) fn install_compiled_with_osr(&self, code: *mut (), mut osr: Vec<OsrEntry>) -> bool {
        if !self.is_valid() || code.is_null() {
            return false;
        }
        if self.reload_pending.load(Ordering::Acquire) {
            return false;
        }

        // Sort by site so lookup can binary-search.
        osr.sort_by_key(|e| e.site);
        let table_generation = self.generation.load(Ordering::Acquire);
        let new_table = Owned::new(OsrTable {
            generation: table_generation,
            entries: osr.into_boxed_slice(),
        });
        let osr_len = new_table.entries.len();

        let guard = &epoch::pin();
        // Publish OSR table before flipping to Compiled.
        let old = self.osr_table.swap(new_table, Ordering::AcqRel, guard);

        self.compiled_code.store(code, Ordering::Release);
        let ok = self
            .state_atom
            .compare_exchange(
                BeadState::Compiling as u8,
                BeadState::Compiled as u8,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok();

        // Defer reclamation of the previous table regardless of install
        // success — it was atomically replaced and is no longer reachable.
        // SAFETY: `old` came from a prior `Owned::new(OsrTable)`; no thread
        // can keep it live past the epoch boundary.
        if !old.is_null() {
            unsafe { guard.defer_destroy(old) };
        }

        if ok {
            info!(
                "bead {:p}: Compiling -> Compiled (code={code:p}, invocations={}, osr={osr_len})",
                self,
                self.invocations.load(Ordering::Relaxed),
            );
        }
        ok
    }

    /// Look up the OSR entry point for a given site.
    ///
    /// Lock-free: one acquire load + binary search on a small sorted slice.
    /// Returns `None` if the bead is not compiled, no OSR table is
    /// installed, or no entry exists for `site`.
    ///
    /// Uses `crossbeam-epoch` to keep the table alive for the duration of
    /// the lookup even while a concurrent install replaces it.
    #[inline]
    pub fn osr_entry(&self, site: u64) -> Option<*mut ()> {
        // Gate on Compiled first — same pattern as `compiled()`.
        if self.state_atom.load(Ordering::Acquire) != BeadState::Compiled as u8 {
            return None;
        }
        let guard = &epoch::pin();
        let shared = self.osr_table.load(Ordering::Acquire, guard);
        // SAFETY: while the `guard` is live, epoch-deferred destruction
        // cannot drop the table we point at.
        let table = unsafe { shared.as_ref() }?;
        // Reject tables from a superseded generation (in-flight swap).
        if table.generation != self.generation.load(Ordering::Acquire) {
            return None;
        }
        table
            .entries
            .binary_search_by_key(&site, |e| e.site)
            .ok()
            .map(|i| table.entries[i].code)
    }

    pub(crate) fn revert_compiling(&self) {
        let ok = self
            .state_atom
            .compare_exchange(
                BeadState::Compiling as u8,
                BeadState::Interpreted as u8,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok();
        self.compiled_code
            .store(core::ptr::null_mut(), Ordering::Release);
        if ok {
            debug!("bead {:p}: Compiling -> Interpreted (reverted)", self);
        }
    }

    pub(crate) fn clear_reload_pending(&self) {
        self.reload_pending.store(false, Ordering::Release);
    }

    // ── Hot swap and reload ───────────────────────────────────────────────────

    pub fn swap_compiled(&self, new_code: *mut ()) -> Option<SwapResult> {
        if self.state_atom.load(Ordering::Acquire) != BeadState::Compiled as u8 {
            return None;
        }
        let old_code = self.compiled_code.swap(new_code, Ordering::AcqRel);
        let new_generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        info!(
            "bead {:p}: code swapped (old={old_code:p}, new={new_code:p}, gen={new_generation})",
            self,
        );
        Some(SwapResult {
            old_code,
            new_generation,
        })
    }

    pub fn reload(&self) -> ReloadOutcome {
        if !self.is_valid() {
            return ReloadOutcome::Dead;
        }

        if self
            .state_atom
            .compare_exchange(
                BeadState::Compiled as u8,
                BeadState::Interpreted as u8,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            self.compiled_code
                .store(core::ptr::null_mut(), Ordering::Release);
            info!(
                "bead {:p}: Compiled -> Interpreted (reload, will recompile)",
                self
            );
            return ReloadOutcome::WillRecompile;
        }

        if self.state_atom.load(Ordering::Acquire) == BeadState::Compiling as u8 {
            self.reload_pending.store(true, Ordering::Release);
            debug!("bead {:p}: reload pending (currently compiling)", self);
            return ReloadOutcome::PendingCurrentCompile;
        }

        if self
            .state_atom
            .compare_exchange(
                BeadState::Queued as u8,
                BeadState::Interpreted as u8,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            debug!("bead {:p}: Queued -> Interpreted (reload reverted)", self);
            return ReloadOutcome::Reverted;
        }

        ReloadOutcome::AlreadyInterpreting
    }

    // ── Runtime-facing API ────────────────────────────────────────────────────

    pub fn update_core(&self, new_ptr: CoreHandle) {
        self.core.store(new_ptr, Ordering::Release);
        trace!("bead {:p}: core updated to {new_ptr:p}", self);
    }

    pub fn invalidate(&self) {
        if self.valid.swap(false, Ordering::AcqRel) {
            self.state_atom
                .store(BeadState::Deopt as u8, Ordering::Release);
            warn!("bead {:p}: invalidated -> Deopt", self);
            if let Some(cb) = &self.on_invalidate {
                cb();
            }
        }
    }

    // ── Bailout and deoptimisation ────────────────────────────────────────────

    /// Called by the runtime when compiled code hits a failed speculation guard.
    ///
    /// Increments the bailout counter, consults the policy, and acts:
    ///
    /// | Decision             | Action                                        |
    /// |----------------------|-----------------------------------------------|
    /// | `Recompile`          | `reload()` — fresh compile on next invocations |
    /// | `Blacklist`          | `blacklist()` — never compile again            |
    /// | `RevertToTier1`      | `reload()` — caller must clear `tier2_queued`  |
    /// | `PauseRecompile`     | sets `recompile_after` threshold               |
    pub fn on_bailout(&self, info: BailoutInfo, policy: &dyn DeoptPolicy) -> DeoptDecision {
        let count = self.bailout_count.fetch_add(1, Ordering::AcqRel) + 1;
        let decision = policy.on_bailout(self, &info, count);
        warn!(
            "bead {:p}: bailout #{count} (guard={}, gen={}) -> {decision:?}",
            self, info.guard_id, info.generation,
        );
        match &decision {
            DeoptDecision::Recompile | DeoptDecision::RevertToTier1 => {
                self.reload();
            }
            DeoptDecision::Blacklist => {
                self.blacklist();
            }
            DeoptDecision::PauseRecompile { until_invocations } => {
                self.recompile_after
                    .store(*until_invocations, Ordering::Release);
            }
        }
        decision
    }

    /// Permanently stop compiling. Reverts to `Interpreted`; `try_queue` always fails.
    pub fn blacklist(&self) {
        self.blacklisted.store(true, Ordering::Release);
        self.state_atom
            .store(BeadState::Interpreted as u8, Ordering::Release);
        self.compiled_code
            .store(core::ptr::null_mut(), Ordering::Release);
        warn!("bead {:p}: blacklisted", self);
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    pub fn core_handle(&self) -> CoreHandle {
        self.core.load(Ordering::Acquire)
    }

    /// Eagerly install compiled code, bypassing the broker state machine.
    ///
    /// Used by [`beadie_backend::BoundBead::compile`] for synchronous
    /// (ahead-of-time) compilation. Drives the bead through
    /// `Interpreted → Queued → Compiling → Compiled` in one call.
    pub fn eager_install(&self, code: *mut ()) -> bool {
        if self.try_queue() {
            self.mark_compiling();
        }
        self.install_compiled(code)
    }
    pub fn state(&self) -> BeadState {
        BeadState::try_from(self.state_atom.load(Ordering::Acquire)).unwrap_or(BeadState::Deopt)
    }
    pub fn is_valid(&self) -> bool {
        self.valid.load(Ordering::Acquire)
    }
    pub fn is_blacklisted(&self) -> bool {
        self.blacklisted.load(Ordering::Acquire)
    }
    pub fn invocation_count(&self) -> u32 {
        self.invocations.load(Ordering::Relaxed)
    }
    pub fn bailout_count(&self) -> u32 {
        self.bailout_count.load(Ordering::Relaxed)
    }
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }
    pub fn reload_pending(&self) -> bool {
        self.reload_pending.load(Ordering::Acquire)
    }
    pub fn recompile_after(&self) -> u32 {
        self.recompile_after.load(Ordering::Acquire)
    }
}

unsafe impl Send for Bead {}
unsafe impl Sync for Bead {}

impl Drop for Bead {
    fn drop(&mut self) {
        // Swap out the final OSR table (if any) and defer its destruction.
        // The bead itself is going away, so no new reader can reach this
        // pointer through it; but an in-flight lookup on another thread may
        // still hold a Shared reference to the table. Epoch reclamation
        // runs the destructor only after all such guards have dropped.
        let guard = &epoch::pin();
        let shared = self.osr_table.swap(Shared::null(), Ordering::AcqRel, guard);
        if !shared.is_null() {
            // SAFETY: `shared` came from a prior `Owned::new(OsrTable)`.
            unsafe { guard.defer_destroy(shared) };
        }
    }
}
