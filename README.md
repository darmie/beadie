# beadie

Hot-function promotion broker for interpreter-to-JIT tiering.

Beadie sits between your interpreter and JIT compiler, automatically detecting hot functions and promoting them to native code via a background compilation thread. It supports single-backend and multi-tier (e.g. Cranelift baseline + LLVM optimizing) compilation strategies.

## Architecture

```
Interpreter thread              Broker thread
      │                              │
  on_invoke()                        │
      │                              │
  tick + policy ── threshold ──► compile job
      │            crossed           │
      │                         compile(bead, def)
      │                              │
  compiled?  ◄── install ───── *mut () code ptr
      │
  dispatch_native()
```

```
Bead state machine:

  Interpreted ──► Queued ──► Compiling ──► Compiled
       ▲                                      │
       └───────── invalidate / deopt ◄────────┘
```

## Usage

```toml
[dependencies]
beadie = "0.2"

# Optional JIT backends
beadie = { version = "0.2", features = ["cranelift"] }
beadie = { version = "0.2", features = ["llvm"] }
```

### Basic promotion

```rust
use beadie::{Beadie, Bead};
use std::sync::Arc;

struct Function {
    bytecode: Vec<u8>,
    bead: Arc<Bead>,
}

let beadie = Beadie::new(); // default: promote after 1,000 invocations

// When the runtime creates a function, register it with beadie.
// The core handle is an opaque pointer beadie stores but never dereferences —
// use it to map back to your runtime's function object inside the compile closure.
let func = Function {
    bytecode: load_bytecode(),
    bead: beadie.register(core_ptr, None),
};

// Interpreter dispatch: on every call, ask beadie whether compiled code exists.
// beadie tracks invocation count internally — once the threshold is crossed,
// it submits the compile closure to its background broker thread.
if let Some(code) = beadie.on_invoke(&func.bead, |bead| {
    // Called once, on the broker thread, when this function wins promotion.
    // bead.core_handle() returns the (always-current) core pointer.
    jit_compile(bead.core_handle())
}) {
    // Compiled code is ready — call the native entry point.
    unsafe { call_native(code, args) };
} else {
    // Still warming up or compiling — keep interpreting.
    interpret(&func.bytecode, args);
}
```

### Custom policy

```rust
use beadie::{Beadie, ThresholdPolicy, TieredPolicy};

// Fixed threshold
let b = Beadie::with_policy(ThresholdPolicy::new(500));

// Two-tier thresholds (tier1 at 500, tier2 at 10,000)
let b = Beadie::with_policy(TieredPolicy::default());
```

### Backend adapter (typed compilation)

```rust
use beadie::{BackendAdapter, JitBackend};

let adapter = BackendAdapter::new(my_backend);
let bound = adapter.register(core_ptr, None);

if let Some(code) = adapter.on_invoke(&bound, |bead| build_ir(bead)) {
    dispatch_native(code, args);
}
```

### Tiered compilation

`TieredAdapter` supports any number of tiers. Each tier has its own promotion
policy and background worker thread. Backends are dispatched inside a single
compile closure indexed by tier:

```rust
use beadie::{TieredAdapter, HotnessPolicy, ThresholdPolicy};
use std::ptr::null_mut;

let tiered = TieredAdapter::new(vec![
    Box::new(ThresholdPolicy::new(1_000))  as Box<dyn HotnessPolicy>,  // tier 0: baseline
    Box::new(ThresholdPolicy::new(10_000)) as Box<dyn HotnessPolicy>,  // tier 1: optimizing
    Box::new(ThresholdPolicy::new(50_000)) as Box<dyn HotnessPolicy>,  // tier 2: aggressive
]);
let bound = tiered.register(core_ptr, None);

// The compile closure receives (tier_index, &bead) — dispatch to the
// appropriate backend based on tier.
if let Some(code) = tiered.on_invoke(&bound, |tier, bead| match tier {
    0 => cranelift.compile(bead, build_baseline(bead)).unwrap_or(null_mut()),
    1 => llvm.compile(bead, build_optimized(bead)).unwrap_or(null_mut()),
    2 => llvm_aggressive.compile(bead, build_aggressive(bead)).unwrap_or(null_mut()),
    _ => null_mut(),
}) {
    unsafe { call_native(code, args) };
}
```

### Cranelift configuration

```rust
use beadie_cranelift::{CraneliftConfig, CraneliftBackend};

let backend = CraneliftConfig::new()
    .opt_level("speed")
    .set("use_colocated_libcalls", "false")
    .set("is_pic", "false")
    .build()?;
```

### Runtime lifecycle

```rust
// GC moved the function object
bead.update_core(new_ptr);

// Function destroyed
bead.invalidate();

// Force recompilation of all hot functions (e.g. hot reload)
b.reload_all();

// Reclaim dead beads periodically
b.prune();
```

## API

### Core types

| Type | Description |
|---|---|
| `Beadie<P>` | Orchestrator. Owns the bead chain and broker thread |
| `Bead` | Atomic state machine representing a single function |
| `Chain` | Thread-safe linked list of beads |
| `Broker` | Background compilation worker thread |

### Policies

| Type | Description |
|---|---|
| `ThresholdPolicy` | Promote after N invocations (default: 1,000) |
| `TieredPolicy` | Two thresholds for tier1/tier2 promotion |
| `HotnessPolicy` | Trait for custom promotion strategies |

### Backend layer

| Type | Description |
|---|---|
| `JitBackend` | Trait for pluggable JIT compilers |
| `BackendAdapter<B, P>` | Beadie wired to a single backend |
| `BoundBead<B>` | Bead pre-wired to a specific backend |
| `TieredAdapter` | N-tier compilation orchestrator |

### Deoptimization

| Type | Description |
|---|---|
| `DeoptPolicy` | Trait for bailout recovery strategies |
| `AlwaysRecompilePolicy` | Always recompile on bailout |
| `ThresholdDeoptPolicy` | Blacklist after N bailouts |
| `ExponentialBackoffPolicy` | Increasing delay between recompile attempts |
| `TieredDeoptPolicy` | Revert to tier1 before blacklisting |

### Hot-path performance

- **Compiled:** one `Acquire` load (branch-predicted)
- **Cold:** one `Relaxed` fetch-add + one `Relaxed` load
- Chain lock is never taken on the invoke path

## Example

The [fib example](crates/beadie-cranelift/examples/fib.rs) demonstrates the
full pipeline — source AST, frontend compilation to MIR, interpreter, and
beadie-managed JIT promotion via Cranelift:

```
cargo run -p beadie-cranelift --example fib
```

`queue_ahead` lets you absorb backend compile latency by submitting the
compile job early. Tune it to your backend (Cranelift ~5ms, LLVM ~50ms):

```
fib(20) x 500 — recursive fibonacci, ~11M dispatches total

config                       time        throughput
------------------------ --------   ---------------
interpreter (no beadie)      2.1s      5.1M dispatch/s
beadie (qa=0)             522.8ms     20.9M dispatch/s
beadie (qa=500)           521.5ms     21.0M dispatch/s
beadie (qa=990)           523.8ms     20.9M dispatch/s
beadie (pre-compiled)     521.9ms     21.0M dispatch/s
```

Beadie's dispatch overhead is effectively zero — the pre-compiled baseline
(every call hits the fast path) matches the tiered runs. The 4x speedup
over raw interpretation is Cranelift's native codegen.

Enable `RUST_LOG=beadie_core=debug` to trace bead state transitions,
broker compile times, and promotion decisions.

## License

MIT
