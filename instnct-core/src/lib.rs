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
//
// Keep implementation modules private and expose only the curated crate-root
// beta surface below. This preserves freedom to refactor internals without
// changing downstream import paths.

mod network;
mod parameters;
mod projection;
mod propagation;
mod sdr;
mod topology;

// ---------------------------------------------------------------------------
// Public beta surface — re-exports
// ---------------------------------------------------------------------------
//
// Only these re-exports are part of the supported public beta API.

#[doc(inline)]
pub use network::{Network, NetworkError, NetworkSnapshot};

#[doc(inline)]
pub use projection::{Int8Projection, WeightBackup};

#[doc(inline)]
pub use sdr::{SdrError, SdrTable};

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
// This hidden module exists only for benchmark binaries. It bridges the
// visibility gap to the unchecked fast path without widening the stable
// public beta contract.

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
    /// Skips checked input validation and assumes the caller already
    /// verified graph, workspace, and slice shapes.
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
