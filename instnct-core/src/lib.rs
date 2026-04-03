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
