// osr.rs — On-Stack Replacement support.
//
// Lets compiled code expose multiple entry points (one per hot loop header),
// so a runtime can transfer a *currently-executing* interpreter frame into
// native code mid-function without returning to the caller.
//
// Use case: long-running single invocations where top-level invocation
// counts never reach the JIT threshold — scripts, event-loop workers,
// numeric kernels called from FFI. OSR lets back-edge counters drive
// promotion instead.

/// A single OSR entry point: a native code address the runtime can jump
/// to in order to transfer a running interpreter frame into compiled code
/// at a specific loop header (or other resume point).
///
/// ## `site` semantics
///
/// The `site` field is an opaque key chosen by the runtime. Typical
/// encodings:
///
/// - Bytecode offset (fits in `u32`, cast to `u64`)
/// - MIR block id
/// - Source-line hash
///
/// Beadie stores the key but does not interpret it. The runtime uses the
/// same encoding at back-edge probe time to look up the entry point.
///
/// ## Safety
///
/// `code` must be a valid native function pointer with the calling
/// convention the runtime expects at its OSR transfer point. Reconstructing
/// the live interpreter state (register file, value stack) into that shape
/// is the runtime's responsibility.
#[derive(Clone, Copy, Debug)]
pub struct OsrEntry {
    pub site: u64,
    pub code: *mut (),
}

// SAFETY: OsrEntry holds a raw pointer to JIT'd code. The runtime is
// responsible for keeping that code alive while any bead holds an OSR
// table referencing it. Beadie never dereferences the pointer.
unsafe impl Send for OsrEntry {}
unsafe impl Sync for OsrEntry {}

/// Immutable OSR entry table installed alongside a bead's compiled code.
///
/// Entries are sorted by `site` at install time, enabling
/// `O(log N)` lookups via binary search on the back-edge path.
/// The `generation` field pins the table to a specific compilation —
/// stale tables from mid-swap windows are rejected at lookup time.
pub(crate) struct OsrTable {
    pub(crate) generation: u64,
    pub(crate) entries: Box<[OsrEntry]>,
}
