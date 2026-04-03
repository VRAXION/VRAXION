#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(unreachable_pub)]
#![doc = include_str!("../../README.md")]
#![doc = "\n## Internal Benchmark Hook\n\nThe hidden `__internal` module is only available with the `benchmarks` feature. It exists solely for benchmark binaries, is unstable, and is excluded from the public beta compatibility promise.\n"]

mod parameters;
mod propagation;
mod topology;

#[doc(inline)]
pub use topology::{ConnectionGraph, DirectedEdge};

#[doc(inline)]
pub use propagation::{
    propagate_token, PropagationConfig, PropagationError, PropagationParameters, PropagationState,
    PropagationWorkspace,
};

/// Benchmark-only internal hooks.
///
/// Available only with the `benchmarks` feature. This module is unstable and
/// not part of the public beta compatibility promise.
#[cfg(feature = "benchmarks")]
#[doc(hidden)]
pub mod __internal {
    use crate::{
        propagation, ConnectionGraph, PropagationConfig, PropagationParameters, PropagationState,
        PropagationWorkspace,
    };

    /// Fast-path propagation for internal benchmark binaries.
    ///
    /// This entrypoint skips public API validation and assumes the graph,
    /// workspace, and slice shapes are already valid.
    #[inline]
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
