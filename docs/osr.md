# OSR (On-Stack Replacement)

Beadie supports **on-stack replacement**: transferring a currently-executing
interpreter frame into compiled native code mid-function, rather than only
at the next call entry.

Status: **shipped in v0.3** (opt-in, zero cost when unused).

## Why OSR matters

Classic "invocation count → promote" tiering only redirects *the next call
to a function*. That's fine for code with many short calls, but it's useless
for workloads where a single invocation does all the work:

- Long-running single invocations (numeric kernels, simulations)
- Top-level scripts (Python `__main__`, JS entrypoints — called once)
- Event-loop workers / async handlers (rare second call)
- Generated state machines with per-block hot rates
- FFI-driven computation (runtime entered once from C)

OSR fixes this by letting **back-edge counters** drive promotion. The JIT
emits one entry per hot loop header; the runtime probes beadie at each
back-edge and, on a hit, transfers the live frame into native code.

## Design

### Data model

A [`Bead`] gets one additional field:

```rust
pub struct Bead {
    // ... existing fields ...
    osr_table: AtomicPtr<OsrTable>,  // null until install
}
```

When a bead is compiled with OSR, the broker installs an immutable
`OsrTable` — a sorted slice of `OsrEntry { site, code }` pairs — and
publishes the pointer via Release store *before* the state flips to
`Compiled`. A reader that observes `state == Compiled` is guaranteed to
see a matching OSR table. Generation numbers catch the tiny mid-swap
window where a superseded table might still be visible.

### Hot-path cost

`Bead::osr_entry(site)` is lock-free:

1. One `Acquire` load of the state byte — fail-fast if not `Compiled`.
2. One `Acquire` load of the OSR table pointer — fail-fast if null.
3. Generation check (one `Acquire` load + compare) — fail-fast if stale.
4. Binary search on `Box<[OsrEntry]>` (small N, typically 1–8).

For non-OSR runtimes, the one extra field on `Bead` is always null; no
allocation, no lookup, no dispatch overhead.

### API

Public types, re-exported from [`beadie`](../crates/beadie/src/lib.rs):

```rust
#[derive(Clone, Copy)]
pub struct OsrEntry {
    pub site: u64,      // opaque key (bytecode offset, MIR block id, ...)
    pub code: *mut (),  // native entry point for that site
}

pub struct OsrCompileResult {
    pub entry: *mut (),
    pub osr: Vec<OsrEntry>,
}

pub struct OsrBuild<D> {
    pub def: D,              // backend-specific IR
    pub osr: Vec<OsrEntry>,
}
```

Submission paths (all three levels mirror the existing non-OSR API):

| Layer | Non-OSR | OSR |
|---|---|---|
| Low-level | `Broker::submit` | `Broker::submit_osr` |
| Orchestrator | `Beadie::on_invoke` | `Beadie::on_invoke_osr` |
| Typed adapter | `BackendAdapter::on_invoke` | `BackendAdapter::on_invoke_osr` |

Lookup:

```rust
// Inside the runtime's back-edge probe:
if let Some(code) = bead.osr_entry(loop_header_id) {
    transfer_to_native(code, /* live values */);
}
```

### What stays runtime-side

Beadie stores and dispatches entry points. Everything else is the
runtime's responsibility:

- **Safepoint metadata** — mapping from `site` to live SSA values /
  interpreter registers. Encoding differs per runtime.
- **Back-edge counters** — what's hot enough to trigger an OSR compile
  is policy. `Bead::invocation_count` covers one axis; OSR runtimes add
  per-site counters keyed by their own `site` encoding.
- **Transfer trampoline** — the actual jmp that reconstructs register
  state and switches to native at a specific block. Per-runtime.
- **GC during transfer** — transient state inside the trampoline is
  outside beadie's scope.

## Example sketch

```rust
use beadie::{BackendAdapter, OsrBuild, OsrEntry, ThresholdPolicy};

let adapter = BackendAdapter::with_policy(backend, ThresholdPolicy::new(100));
let bound = adapter.register(core_ptr, None);

// Interpreter entry — unchanged:
if let Some(code) = adapter.on_invoke_osr(&bound, |bead| {
    let (def, osr_entries) = my_jit.compile_with_osr(bead);
    OsrBuild { def, osr: osr_entries }
}) {
    dispatch_native(code);
}

// Back-edge probe inside an interpreter loop:
if let Some(entry) = bound.bead().osr_entry(current_block_id as u64) {
    // Reconstruct live values, jump to native at `entry`.
    runtime.transfer_to_native(entry);
}
```

## Reclamation

Superseded OSR tables are reclaimed via `crossbeam-epoch`. On every
`install_compiled_with_osr`, the previous table (if any) is handed to
`guard.defer_destroy`, which runs its destructor once every thread that
may still hold a reference has moved past the current epoch. The bead's
`Drop` defers the final table the same way. No leaks, no global locks,
no runtime-side quiescent-point API required.

## Known limitations

1. **No OSR during tier swaps.** `Bead::swap_compiled` (used by tiered
   adapters for tier-N → tier-N+1 promotion) bumps the generation but
   does not install a new OSR table. OSR entries remain valid only for
   the initial compile's generation. A `swap_compiled_with_osr` variant
   is an obvious follow-up.

2. **Deopt from an OSR entry is whole-bead.** If compiled code bailed out
   from inside an OSR region, the entire bead is treated as deopt
   (`reload()` or `blacklist()` depending on policy). Per-site suppression
   is not modeled.

Neither blocks the v0.3 shipping API.
