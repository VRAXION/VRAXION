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

mod checkpoint;
mod corpus;
mod eval;
mod eval_bytepair;
mod evolution;
mod fitness;
mod init;
mod network;
mod parameters;
mod projection;
mod propagation;
mod sdr;
mod topology;
mod vcbp_io;

#[doc(hidden)]
pub mod experimental_route_grammar;

#[cfg(feature = "parquet")]
mod parquet_fineweb;

// ---------------------------------------------------------------------------
// Public beta surface — re-exports
// ---------------------------------------------------------------------------
//
// Only these re-exports are part of the supported public beta API.

#[doc(inline)]
pub use checkpoint::{load_checkpoint, save_checkpoint, CheckpointMeta};

#[doc(inline)]
pub use corpus::{build_bigram_table, load_corpus};

#[doc(inline)]
pub use eval::{eval_accuracy, eval_smooth};

#[doc(inline)]
pub use eval_bytepair::{
    build_bytepair_bigram, eval_bytepair_accuracy, eval_bytepair_smooth,
    eval_bytepair_smooth_bigram,
};

#[doc(inline)]
pub use vcbp_io::{VcbpError, VcbpTable};

#[doc(inline)]
pub use fitness::{cosine_similarity, cosine_to_onehot, softmax};

#[doc(inline)]
pub use evolution::{
    evolution_step, evolution_step_cow, evolution_step_jackpot, evolution_step_jackpot_traced,
    evolution_step_jackpot_traced_with_policy,
    evolution_step_jackpot_traced_with_policy_and_operator_weights, mutation_operator_baseline_probability,
    mutation_operator_ids, mutation_operator_index, AcceptancePolicy, CandidateTraceRecord,
    EvolutionConfig, MutationOperatorSpec, StepOutcome, MUTATION_OPERATORS,
};

#[doc(inline)]
pub use init::{build_network, InitConfig};

#[doc(inline)]
pub use network::{MutationUndo, Network, NetworkError, NetworkSnapshot, SpikeData};

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

#[cfg(feature = "parquet")]
#[doc(inline)]
pub use parquet_fineweb::{
    extract_to_bytes, extract_to_file, extract_to_writer, find_parquet_files, ExtractConfig,
    ExtractError, ExtractStats,
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
