#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links, unreachable_pub)]
#![doc = include_str!("../../README.md")]
//!
//! ## Internal Benchmark Hook
//!
//! The hidden `__internal` module is only available with the `benchmarks`
//! feature. It exists solely for benchmark binaries, is unstable, and is
//! excluded from the public beta compatibility promise.

// ---------------------------------------------------------------------------
// Modules
// ---------------------------------------------------------------------------

mod parameters;
mod propagation;
mod topology;

// ---------------------------------------------------------------------------
// Public beta surface — re-exports
// ---------------------------------------------------------------------------
//
// The three modules above are private (`mod`, not `pub mod`).
// Only the names listed here are visible to downstream crates.
//
// `pub use` re-exports each name at the crate root so users write
//     use instnct_core::ConnectionGraph;
// instead of
//     use instnct_core::topology::ConnectionGraph;   // would require `pub mod`
//
// `#[doc(inline)]` tells rustdoc to render the full documentation of each
// item directly on the crate root page rather than showing a bare hyperlink.
// This keeps docs.rs browsable without extra clicks.

#[doc(inline)]
pub use topology::{ConnectionGraph, DirectedEdge};

#[doc(inline)]
pub use propagation::{
    propagate_token, PropagationConfig, PropagationError, PropagationParameters, PropagationState,
    PropagationWorkspace,
};

// ---------------------------------------------------------------------------
// Benchmark internals (feature-gated, unstable)
// ---------------------------------------------------------------------------
//
// Three layers of protection keep this out of normal builds and docs:
//
// 1. `#[cfg(feature = "benchmarks")]` — conditional compilation.
//    The module only exists in the binary when built with
//    `cargo bench --features benchmarks`. Normal builds skip it entirely.
//
// 2. `#[doc(hidden)]` — hidden from rustdoc / docs.rs.
//    Even with the feature on, users won't discover it by browsing docs.
//
// 3. `pub mod __internal` — the `__` prefix is a Rust convention for
//    "hands off". The `pub` is required because `benches/forward_bench.rs`
//    is an external binary that can only reach crate items through `pub`.
//
// The function inside (`propagate_token_unchecked`) is the same forward
// pass as the public `propagate_token`, but skips input validation
// (slice lengths, workspace size). Benchmarks measure raw propagation
// speed without paying for the checked boundary.
//
// `#[inline(always)]` guarantees the compiler eliminates this 1:1
// wrapper — the bench calls the inner function directly.

/// Benchmark-only internal hooks — not part of the public beta surface.
#[cfg(feature = "benchmarks")]
#[doc(hidden)]
pub mod __internal {
    use crate::{
        propagation, ConnectionGraph, PropagationConfig, PropagationParameters, PropagationState,
        PropagationWorkspace,
    };

    /// Fast-path propagation for benchmark binaries.
    ///
    /// Skips public API validation — assumes graph, workspace, and slice
    /// shapes are already verified by the caller.
    #[inline(always)]
    pub fn propagate_token_unchecked(
        input: &[i32],
        graph: &ConnectionGraph,
        params: &PropagationParameters<'_>,
        state: &mut PropagationState<'_>,
        config: &PropagationConfig,
        workspace: &mut PropagationWorkspace,
    ) {
        propagation::propagate_token_unchecked(input, graph, params, state, config, workspace);
    }
}
