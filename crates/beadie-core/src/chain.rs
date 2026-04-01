// chain.rs — The bead chain: a singly-linked list of beads.
//
// You own the links (the thread).
// The runtime owns the cores (inside each bead).

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use log::debug;

use crate::bead::{Bead, BeadState, CoreHandle};

// ─────────────────────────────────────────────────────────────────────────────
// Node — internal linked-list cell
// ─────────────────────────────────────────────────────────────────────────────

struct Node {
    bead: Arc<Bead>,
    next: Option<Box<Node>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Chain
// ─────────────────────────────────────────────────────────────────────────────

/// A singly-linked bead chain.
///
/// ## Ownership split
///
/// ```text
///  Chain owns:  [Node] ──► [Node] ──► [Node] ──► None
///                  │           │           │
///  Runtime owns:  core?       core?       core?   (opaque)
/// ```
///
/// The chain holds the topology. You traverse it. You prune it.
/// You never see inside a core.
///
/// ## Thread safety
/// Chain mutations (`push`, `prune`) are guarded by an internal `Mutex`.
/// These are infrequent management operations, **not** on the interpreter hot
/// path. Hot-path operations work directly on `Arc<Bead>` — no chain lock
/// is taken during normal dispatch.
pub struct Chain {
    head: Mutex<Option<Box<Node>>>,
    len: AtomicUsize,
}

impl Chain {
    pub fn new() -> Self {
        Self {
            head: Mutex::new(None),
            len: AtomicUsize::new(0),
        }
    }

    // ── Mutation ──────────────────────────────────────────────────────────────

    /// Prepend a new bead to the chain.
    ///
    /// Returns a shared `Arc<Bead>` handle. Store this next to the function
    /// in your interpreter and pass it to [`crate::Beadie::on_invoke`] on
    /// every call.
    pub fn push(
        &self,
        core: CoreHandle,
        on_invalidate: Option<Box<dyn Fn() + Send + Sync>>,
    ) -> Arc<Bead> {
        let bead = Bead::new(core, on_invalidate);
        let mut guard = self.head.lock().unwrap();
        let prev = guard.take();
        *guard = Some(Box::new(Node {
            bead: Arc::clone(&bead),
            next: prev,
        }));
        let new_len = self.len.fetch_add(1, Ordering::Relaxed) + 1;
        debug!("bead {:p}: registered (chain len={})", &*bead, new_len);
        bead
    }

    /// Remove all `Deopt` beads from the chain.
    ///
    /// Call this periodically (e.g. on GC cycle or every N invocations) to
    /// reclaim memory from invalidated beads.
    pub fn prune(&self) {
        let mut guard = self.head.lock().unwrap();

        // Drain the chain into a Vec, preserving insertion order.
        let mut nodes: Vec<Box<Node>> = Vec::new();
        let mut cur = guard.take();
        while let Some(mut node) = cur {
            cur = node.next.take();
            nodes.push(node);
        }

        let before = nodes.len();
        nodes.retain(|n| n.bead.state() != BeadState::Deopt);
        let pruned = before - nodes.len();

        // Rebuild the chain in original order.
        let mut head: Option<Box<Node>> = None;
        for mut node in nodes.into_iter().rev() {
            node.next = head;
            head = Some(node);
        }

        *guard = head;
        // Saturating sub — guards against races with concurrent `push`.
        self.len.fetch_sub(pruned, Ordering::Relaxed);

        if pruned > 0 {
            debug!("chain: pruned {} deopt beads (remaining={})", pruned, self.len());
        }
    }

    // ── Traversal ─────────────────────────────────────────────────────────────

    /// Walk every bead in chain order, holding the chain lock.
    ///
    /// Use for management tasks (inspection, bulk invalidation).
    /// Do **not** call from the interpreter hot path.
    pub fn walk(&self, mut f: impl FnMut(&Arc<Bead>)) {
        let guard = self.head.lock().unwrap();
        let mut cur = guard.as_deref();
        while let Some(node) = cur {
            f(&node.bead);
            cur = node.next.as_deref();
        }
    }

    // ── Metrics ───────────────────────────────────────────────────────────────

    /// Approximate chain length. May be transiently stale under concurrency.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for Chain {
    fn default() -> Self {
        Self::new()
    }
}
