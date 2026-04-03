#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(unreachable_pub)]

//! # INSTNCT Core
//!
//! `instnct-core` is the low-level recurrent spiking substrate behind VRAXION v5.
//! The public beta surface is intentionally small and rooted at the crate level.
//!
//! ## Quickstart
//!
//! ```
//! use instnct_core::{
//!     propagate_token, ConnectionGraph, PropagationConfig, PropagationParameters,
//!     PropagationState, PropagationWorkspace,
//! };
//!
//! let mut graph = ConnectionGraph::new(2);
//! assert!(graph.add_edge(0, 1));
//!
//! let input = [4, 0];
//! let threshold = [1, 1];
//! let channel = [1, 1];
//! let polarity = [1, 1];
//! let mut activation = [0, 0];
//! let mut charge = [0, 0];
//! let mut workspace = PropagationWorkspace::new(2);
//!
//! propagate_token(
//!     &input,
//!     &graph,
//!     &PropagationParameters {
//!         threshold: &threshold,
//!         channel: &channel,
//!         polarity: &polarity,
//!     },
//!     &mut PropagationState {
//!         activation: &mut activation,
//!         charge: &mut charge,
//!     },
//!     &PropagationConfig {
//!         ticks: 2,
//!         input_duration: 1,
//!         decay_period: 0,
//!     },
//!     &mut workspace,
//! )?;
//!
//! # Ok::<(), instnct_core::PropagationError>(())
//! ```
//!
//! ## Stable Beta Surface
//!
//! - [`ConnectionGraph`] stores sparse directed topology with checked mutation methods.
//! - [`PropagationWorkspace`] owns reusable buffers for repeated forward passes.
//! - [`PropagationParameters`], [`PropagationState`], and [`PropagationConfig`] describe one propagation run.
//! - [`propagate_token`] is the checked public propagation entrypoint.

mod parameters;
mod propagation;
mod topology;

pub use propagation::{
    propagate_token, PropagationConfig, PropagationError, PropagationParameters, PropagationState,
    PropagationWorkspace,
};
pub use topology::{ConnectionGraph, DirectedEdge};

/// Benchmark-only internal hooks.
///
/// This module is not part of the stable public beta compatibility promise.
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
