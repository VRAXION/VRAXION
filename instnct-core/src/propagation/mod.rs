//! # Signal Propagation — Spiking Forward Pass
//!
//! Simulates one token's passage through the recurrent spiking network.
//! This is the performance-critical inner loop.
//!
//! ## Integer-Only Design
//!
//! The forward pass uses `i32` activations (supports inhibitory `-1`) and
//! `u32` charge (always non-negative). The wave gating lookup table is fixed
//! point (`x1000` scale). No floating-point arithmetic appears in the hot path.

// =========================================================================
// Imports
// =========================================================================
//
// All tunable constants come from `parameters` — no magic numbers here.
// `ConnectionGraph` provides the sparse edge topology iterated by the
// scatter-add loop.  `Error` + `fmt` support the public `PropagationError`.

use crate::parameters::{
    GLOBAL_CHARGE_DECAY_INTERVAL_TICKS, GLOBAL_INPUT_DURATION_TICKS, GLOBAL_TICKS_PER_TOKEN,
    GLOBAL_WAVE_AMPLITUDE_PERMILLE, GLOBAL_WAVE_CHANNEL_COUNT, GLOBAL_WAVE_TICKS_PER_PERIOD,
    LIMIT_MAX_CHARGE,
};
use crate::topology::ConnectionGraph;
use std::error::Error;
use std::fmt;

// =========================================================================
// Wave gating lookup table
// =========================================================================
//
// Fixed-size 2D array: [channel][tick] -> threshold multiplier (x1000).
// Channel 0 is "no gating" (all 1000s). Channels 1..=8 each have a
// cosine curve phase-shifted by their channel index, so different
// channels prefer different ticks within the period.
//
// The x1000 scale is a fixed-point trick: the hot path stays integer-
// only while preserving three decimal digits of precision.
// 1000 = "no change", 700 = "30% lower threshold" (fires easier),
// 1300 = "30% higher threshold" (fires harder).
//
// Built once at workspace creation, then read-only during propagation.

type WaveGatingTable = [[u32; GLOBAL_WAVE_TICKS_PER_PERIOD]; GLOBAL_WAVE_CHANNEL_COUNT + 1];

// =========================================================================
// Workspace — pre-allocated hot-path buffers
// =========================================================================
//
// `propagate_token` is called hundreds of thousands of times (every
// token, every worker, every eval step).  Allocating buffers inside
// the function would mean a heap alloc + drop per call.  The workspace
// moves allocation to construction time: create once, reuse forever.
//
// Two things live here:
//   wave_table  — the cosine LUT above (stack-allocated, [9][8] u32)
//   scratch     — per-neuron incoming-signal accumulator (heap Vec<i32>)

/// Precomputed and reusable buffers for repeated propagation calls.
///
/// This keeps lookup-table construction and scratch allocation out of the
/// hot path while giving callers explicit control over workspace reuse.
#[derive(Clone, Debug)]
pub struct PropagationWorkspace {
    wave_table: WaveGatingTable,
    scratch: Vec<i32>,
}

impl PropagationWorkspace {
    /// Create a workspace sized for `neuron_count` neurons.
    pub fn new(neuron_count: usize) -> Self {
        Self {
            wave_table: build_wave_gating_table(),
            scratch: vec![0; neuron_count],
        }
    }

    /// Ensure the scratch buffer can hold `neuron_count` entries.
    ///
    /// Use this after growing a graph so the checked public API does not
    /// reject the workspace as undersized.
    pub fn ensure_neuron_capacity(&mut self, neuron_count: usize) {
        if self.scratch.len() < neuron_count {
            self.scratch.resize(neuron_count, 0);
        }
    }

    #[cfg(test)]
    fn from_parts(wave_table: WaveGatingTable, scratch: Vec<i32>) -> Self {
        Self {
            wave_table,
            scratch,
        }
    }
}

// Builds the cosine-modulated wave gating LUT.
//
// Formula per entry:
//   multiplier = round((1 - A * cos(2*PI*(tick - (ch-1)) / P)) * 1000)
//
// where A = GLOBAL_WAVE_AMPLITUDE_PERMILLE / 1000 (0.3 at default)
// and   P = GLOBAL_WAVE_TICKS_PER_PERIOD (8 at default).
//
// Channel 0 is skipped (stays all-1000) — it means "no wave gating".
// Float math is acceptable here because this runs once at startup,
// never in the hot path.
fn build_wave_gating_table() -> WaveGatingTable {
    let mut table = [[1000u32; GLOBAL_WAVE_TICKS_PER_PERIOD]; GLOBAL_WAVE_CHANNEL_COUNT + 1];
    for (channel, row) in table.iter_mut().enumerate().skip(1) {
        for (tick, entry) in row.iter_mut().enumerate() {
            let phase_offset = tick as f64 - (channel - 1) as f64;
            let angle =
                2.0 * std::f64::consts::PI * phase_offset / GLOBAL_WAVE_TICKS_PER_PERIOD as f64;
            let amplitude = GLOBAL_WAVE_AMPLITUDE_PERMILLE as f64 / 1000.0;
            *entry = ((1.0 - amplitude * angle.cos()) * 1000.0).round() as u32;
        }
    }
    table
}

/// Per-neuron learned parameters for one propagation run.
pub struct PropagationParameters<'a> {
    /// Firing threshold per neuron. Range: `[1, 15]`.
    pub threshold: &'a [u32],
    /// Wave gating channel per neuron. Range: `[1, 8]`.
    pub channel: &'a [u8],
    /// Polarity per neuron: `+1` (excitatory) or `-1` (inhibitory).
    pub polarity: &'a [i32],
}

/// Mutable neuron state carried across tokens.
pub struct PropagationState<'a> {
    /// Activation per neuron: `+1`, `-1`, or `0`.
    pub activation: &'a mut [i32],
    /// Accumulated charge per neuron. Range: `[0, LIMIT_MAX_CHARGE]`.
    pub charge: &'a mut [u32],
}

/// Timing configuration for one forward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PropagationConfig {
    /// Total number of simulation ticks to execute for one token.
    pub ticks: usize,
    /// Number of initial ticks during which the external input is injected.
    pub input_duration: usize,
    /// Charge decay period. Every `decay_period` ticks, each neuron loses one charge.
    pub decay_period: usize,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            ticks: GLOBAL_TICKS_PER_TOKEN,
            input_duration: GLOBAL_INPUT_DURATION_TICKS,
            decay_period: GLOBAL_CHARGE_DECAY_INTERVAL_TICKS,
        }
    }
}

/// Structured propagation input validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PropagationError {
    /// The activation buffer length does not match the graph neuron count.
    ActivationLengthMismatch {
        /// Expected activation slice length.
        expected: usize,
        /// Actual activation slice length.
        actual: usize,
    },
    /// The input slice length does not match the graph neuron count.
    InputLengthMismatch {
        /// Expected input slice length.
        expected: usize,
        /// Actual input slice length.
        actual: usize,
    },
    /// The charge buffer length does not match the graph neuron count.
    ChargeLengthMismatch {
        /// Expected charge slice length.
        expected: usize,
        /// Actual charge slice length.
        actual: usize,
    },
    /// The threshold slice length does not match the graph neuron count.
    ThresholdLengthMismatch {
        /// Expected threshold slice length.
        expected: usize,
        /// Actual threshold slice length.
        actual: usize,
    },
    /// The wave-channel slice length does not match the graph neuron count.
    ChannelLengthMismatch {
        /// Expected channel slice length.
        expected: usize,
        /// Actual channel slice length.
        actual: usize,
    },
    /// The polarity slice length does not match the graph neuron count.
    PolarityLengthMismatch {
        /// Expected polarity slice length.
        expected: usize,
        /// Actual polarity slice length.
        actual: usize,
    },
    /// The workspace scratch buffer is smaller than the graph neuron count.
    ScratchTooSmall {
        /// Required scratch length.
        required: usize,
        /// Actual scratch length.
        actual: usize,
    },
    /// The graph source/target endpoint caches do not have the same length.
    EdgeLengthMismatch {
        /// Number of source endpoints.
        sources: usize,
        /// Number of target endpoints.
        targets: usize,
    },
    /// A cached graph source endpoint points outside the neuron range.
    EdgeSourceOutOfBounds {
        /// Edge index within the cached endpoint view.
        index: usize,
        /// Out-of-range source endpoint value.
        value: usize,
        /// Graph neuron count.
        neuron_count: usize,
    },
    /// A cached graph target endpoint points outside the neuron range.
    EdgeTargetOutOfBounds {
        /// Edge index within the cached endpoint view.
        index: usize,
        /// Out-of-range target endpoint value.
        value: usize,
        /// Graph neuron count.
        neuron_count: usize,
    },
}

impl fmt::Display for PropagationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActivationLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "activation length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InputLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "input length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ChargeLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "charge length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ThresholdLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "threshold length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ChannelLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "channel length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::PolarityLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "polarity length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ScratchTooSmall { required, actual } => {
                write!(f, "scratch buffer too small: need {required}, got {actual}")
            }
            Self::EdgeLengthMismatch { sources, targets } => {
                write!(
                    f,
                    "edge length mismatch: {sources} sources vs {targets} targets"
                )
            }
            Self::EdgeSourceOutOfBounds {
                index,
                value,
                neuron_count,
            } => write!(
                f,
                "edge source out of bounds at edge {index}: {value} >= {neuron_count}"
            ),
            Self::EdgeTargetOutOfBounds {
                index,
                value,
                neuron_count,
            } => write!(
                f,
                "edge target out of bounds at edge {index}: {value} >= {neuron_count}"
            ),
        }
    }
}

impl Error for PropagationError {}

fn validate_propagation_inputs(
    input: &[i32],
    graph: &ConnectionGraph,
    params: &PropagationParameters<'_>,
    state: &PropagationState<'_>,
    workspace: &PropagationWorkspace,
) -> Result<usize, PropagationError> {
    let neuron_count = graph.neuron_count();
    let (edge_sources, edge_targets) = graph.edge_endpoints();

    if state.activation.len() != neuron_count {
        return Err(PropagationError::ActivationLengthMismatch {
            expected: neuron_count,
            actual: state.activation.len(),
        });
    }
    if state.charge.len() != neuron_count {
        return Err(PropagationError::ChargeLengthMismatch {
            expected: neuron_count,
            actual: state.charge.len(),
        });
    }
    if input.len() != neuron_count {
        return Err(PropagationError::InputLengthMismatch {
            expected: neuron_count,
            actual: input.len(),
        });
    }
    if params.threshold.len() != neuron_count {
        return Err(PropagationError::ThresholdLengthMismatch {
            expected: neuron_count,
            actual: params.threshold.len(),
        });
    }
    if params.channel.len() != neuron_count {
        return Err(PropagationError::ChannelLengthMismatch {
            expected: neuron_count,
            actual: params.channel.len(),
        });
    }
    if params.polarity.len() != neuron_count {
        return Err(PropagationError::PolarityLengthMismatch {
            expected: neuron_count,
            actual: params.polarity.len(),
        });
    }
    if workspace.scratch.len() < neuron_count {
        return Err(PropagationError::ScratchTooSmall {
            required: neuron_count,
            actual: workspace.scratch.len(),
        });
    }
    if edge_sources.len() != edge_targets.len() {
        return Err(PropagationError::EdgeLengthMismatch {
            sources: edge_sources.len(),
            targets: edge_targets.len(),
        });
    }
    for (index, &source) in edge_sources.iter().enumerate() {
        if source >= neuron_count {
            return Err(PropagationError::EdgeSourceOutOfBounds {
                index,
                value: source,
                neuron_count,
            });
        }
    }
    for (index, &target) in edge_targets.iter().enumerate() {
        if target >= neuron_count {
            return Err(PropagationError::EdgeTargetOutOfBounds {
                index,
                value: target,
                neuron_count,
            });
        }
    }

    Ok(neuron_count)
}

/// Propagate one token through the spiking network.
///
/// This is the checked public API boundary. It validates slice shapes and
/// graph endpoint invariants in release builds, then dispatches to the fast path.
pub fn propagate_token(
    input: &[i32],
    graph: &ConnectionGraph,
    params: &PropagationParameters<'_>,
    state: &mut PropagationState<'_>,
    config: &PropagationConfig,
    workspace: &mut PropagationWorkspace,
) -> Result<(), PropagationError> {
    validate_propagation_inputs(input, graph, params, state, workspace)?;
    propagate_token_unchecked(input, graph, params, state, config, workspace);
    Ok(())
}

pub(crate) fn propagate_token_unchecked(
    input: &[i32],
    graph: &ConnectionGraph,
    params: &PropagationParameters<'_>,
    state: &mut PropagationState<'_>,
    config: &PropagationConfig,
    workspace: &mut PropagationWorkspace,
) {
    let neuron_count = graph.neuron_count();
    let (edge_sources, edge_targets) = graph.edge_endpoints();

    debug_assert_eq!(state.activation.len(), neuron_count);
    debug_assert_eq!(state.charge.len(), neuron_count);
    debug_assert_eq!(input.len(), neuron_count);
    debug_assert_eq!(params.threshold.len(), neuron_count);
    debug_assert_eq!(params.channel.len(), neuron_count);
    debug_assert_eq!(params.polarity.len(), neuron_count);
    debug_assert_eq!(edge_sources.len(), edge_targets.len());
    debug_assert!(workspace.scratch.len() >= neuron_count);

    let wave_table = &workspace.wave_table;

    for tick in 0..config.ticks {
        if config.decay_period > 0 && tick % config.decay_period == 0 {
            for charge in state.charge.iter_mut() {
                *charge = charge.saturating_sub(1);
            }
        }

        if tick < config.input_duration {
            for (activation, &input_value) in state.activation.iter_mut().zip(input.iter()) {
                *activation += input_value;
            }
        }

        let incoming = &mut workspace.scratch[..neuron_count];
        incoming.fill(0);
        for (source_chunk, target_chunk) in edge_sources.chunks(4).zip(edge_targets.chunks(4)) {
            for chunk_idx in 0..source_chunk.len() {
                incoming[target_chunk[chunk_idx]] += state.activation[source_chunk[chunk_idx]];
            }
        }

        for (charge, &signal) in state.charge.iter_mut().zip(incoming.iter()) {
            let new_charge = *charge as i32 + signal;
            *charge = new_charge.clamp(0, LIMIT_MAX_CHARGE as i32) as u32;
        }

        let tick_in_period = tick % GLOBAL_WAVE_TICKS_PER_PERIOD;
        for neuron_idx in 0..neuron_count {
            let channel = params.channel[neuron_idx] as usize;
            let wave_multiplier = if channel <= GLOBAL_WAVE_CHANNEL_COUNT {
                wave_table[channel][tick_in_period]
            } else {
                1000
            };
            let charge_scaled = state.charge[neuron_idx] * 1000;
            let threshold_scaled = params.threshold[neuron_idx] * wave_multiplier;

            if charge_scaled >= threshold_scaled {
                state.activation[neuron_idx] = params.polarity[neuron_idx];
                state.charge[neuron_idx] = 0;
            } else {
                state.activation[neuron_idx] = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests;
