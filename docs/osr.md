# OSR (On-Stack Replacement) — design sketch

This is a proposal for extending beadie so it can model runtimes that
need to transfer a **currently-executing interpreter frame** into
compiled native code mid-function — "on-stack replacement."

Status: **design only, not yet implemented.** Input welcome.

## Why OSR doesn't fit the current model

Today a `Bead` has one code slot (`compiled_code: AtomicPtr<()>`).
`on_invoke` checks that slot first — if set, dispatch native; if not,
tick the counter and maybe submit a compile job. This is the right
shape for the *entry* case: a call arrives at the top of the function,
we have the chance to redirect it.

OSR is the other case: the function is **already running** in the
interpreter, and a hot loop back-edge wants to transfer control into
native code **mid-function** — without returning to the caller and
re-entering from the top. For that, beadie would need to hand the
runtime a pointer not to the function's normal entry, but to a
**specific continuation point** that knows how to resume execution
from that block.

A well-optimized tiered VM emits one such entry per hot loop header.
Bead's single code slot can't represent that.

## Proposed extension

Add an optional OSR-entry table to `Bead`, independent from the
normal code slot:

```rust
pub struct Bead {
    // ... existing fields unchanged ...
    compiled_code: AtomicPtr<()>,

    // NEW — populated only by runtimes that opt in. Empty for the
    // common case of "one code pointer per bead."
    osr_entries: Mutex<SmallVec<[OsrEntry; 4]>>,
}

#[derive(Clone, Copy)]
pub struct OsrEntry {
    /// Opaque key. The runtime picks the encoding. Typical choices:
    /// MIR block id, bytecode offset, source-line hash.
    pub site: u64,
    /// The entry point the runtime jumps to when it transfers a live
    /// frame into compiled code at `site`.
    pub code: *mut (),
}
```

New methods on `Bead`:

- `install_osr_entries(entries: Vec<OsrEntry>)` — populate atomically
  as part of an install, replacing any prior entries. Called by the
  compile closure (it already returns the normal code pointer;
  installation of OSR entries uses a parallel path — see below).

- `osr_entry(site: u64) -> Option<*mut ()>` — O(N) lookup, N small
  by construction. Returns the entry point to jump to, or `None` if
  no OSR entry was emitted for that site.

- `clear_osr_entries()` — called implicitly by `invalidate()` and
  `reload()` so stale entries don't survive a recompile.

### Submission API

The natural extension to `Beadie::submit`:

```rust
/// OSR-aware submit. The compile closure returns both the normal
/// entry pointer and a (possibly empty) vec of OSR entries.
pub fn submit_with_osr<F>(&self, bead: &Arc<Bead>, compile: F) -> SubmitResult
where
    F: FnOnce(&Arc<Bead>) -> OsrCompileResult + Send + 'static,
{ ... }

pub struct OsrCompileResult {
    pub entry: *mut (),          // normal entry
    pub osr: Vec<OsrEntry>,      // one per hot loop header
}
```

The existing `submit`/`on_invoke` stay backwards compatible and keep
returning a single `*mut ()`. Runtimes that don't do OSR see no
change.

### What the runtime still owns

OSR requires live-value reconstruction at the transfer point:
the interpreter's register file and value stack must be copied into
shapes the compiled code expects. Beadie will **not** model this:

- **Safepoint metadata** — mapping from `site` to the live SSA
  values / interpreter registers that need to be forwarded. Each
  runtime's representation differs (MIR block ids vs. bytecode
  offsets vs. source spans). This stays runtime-side.

- **Back-edge counters** — what's hot enough to trigger an OSR
  compile is policy, and the granularity (per-loop, per-function,
  per-trace) varies. Beadie's `invocation_count` is the right
  primitive for one axis; OSR runtimes add their own counters
  keyed by `site` and check them in their own hot-path logic.

- **Transfer trampoline** — the actual jmp that switches execution
  from interpreter to native at a specific block. Beadie hands the
  runtime a pointer; the trampoline code is emitted per-runtime.

## Open questions

1. **Lookup cost.** `SmallVec` + linear scan is fine for the "3 hot
   loops per function" case. For functions with dozens of loops
   (codegen'd state machines, large switch statements), a tiny
   hash or sorted-vec binary-search would win. Measure before
   picking.

2. **Interaction with `swap_compiled`.** When the runtime replaces
   baseline with optimized code, should OSR entries come along?
   Probably yes — the two vectors are installed together as one
   atomic step. This matches how WrenLift does it today with
   `baseline_osr_entries` / `optimized_osr_entries`.

3. **Deopt from an OSR entry.** If a compiled OSR entry bails
   back to the interpreter, is that a whole-bead deopt or just
   site-specific? Realistic runtimes treat it as whole-bead (once
   a function gets a deopt, its other entries are suspect too).
   Start with whole-bead for simplicity.

4. **GC during OSR transfer.** The transfer trampoline executes
   with the interpreter frame partially dismantled — if the GC
   runs there, live values in registers that haven't yet been
   stored to the native stack are at risk. This is a runtime
   problem; beadie's `update_core` mechanism doesn't help here
   because the problem is transient state inside the trampoline,
   not the `core` handle.

## Rollout

This is a purely additive change:

- Phase 1 — add `osr_entries` field, methods, and `submit_with_osr`.
  Existing `submit` / `on_invoke` unchanged.
- Phase 2 — publish a 0.3 release with the feature. Runtimes that
  don't opt in see no change.
- Phase 3 — migrate one reference runtime (WrenLift, which
  currently maintains its own `jit_osr_entries: Vec<Vec<NativeOsrEntry>>`
  on the engine) to the beadie-native shape as a validation.

The reference migration is the forcing function — if WrenLift
can't drop its engine-side OSR vectors cleanly, the API shape is
wrong and needs another pass before publishing.
