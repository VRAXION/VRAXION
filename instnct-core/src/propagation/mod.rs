//! # Signal Propagation — Spiking Forward Pass
//!
//! Simulates one token's passage through the recurrent spiking network.
//! This is the performance-critical inner loop.
//!
//! ## Integer-Only Design
//!
//! The forward pass uses `i32` activations (supports inhibitory `-1`) and
//! `u32` charge (always non-negative). Phase gating uses an 8-byte cosine
//! base pattern with `x10` fixed-point scale. No floating-point arithmetic
//! appears in the hot path.

// =========================================================================
// Imports
// =========================================================================
//
// All tunable constants come from `parameters` — no magic numbers here.
// `ConnectionGraph` provides the sparse edge topology iterated by the
// scatter-add loop.  `Error` + `fmt` support the public `PropagationError`.

use crate::parameters::{
    GLOBAL_CHARGE_DECAY_INTERVAL_TICKS, GLOBAL_INPUT_DURATION_TICKS, GLOBAL_TICKS_PER_TOKEN,
    GLOBAL_PHASE_CHANNEL_COUNT, GLOBAL_PHASE_TICKS_PER_PERIOD, LIMIT_MAX_CHARGE,
};
use crate::topology::ConnectionGraph;
use std::error::Error;
use std::fmt;

// =========================================================================
// Phase gating — 8-byte cosine base pattern
// =========================================================================
//
// Cosine-derived threshold multipliers at x10 fixed-point scale.
// Channel N (1-8) rotates this pattern so the easiest tick (7) aligns
// with tick N-1, giving each channel a distinct temporal preference.
//
//   7  = 0.7x threshold (fires easiest)
//   10 = 1.0x threshold (neutral)
//   13 = 1.3x threshold (fires hardest)
//
// Amplitude 0.3 is baked in. Damping experiments (2026-04-04) confirmed:
// this 8-byte LUT + 3-bit channel is optimal. Per-tick learnable
// multipliers, damping scores, and other extensions all failed to
// improve over this minimal design.

/// Cosine phase gating base pattern (x10 fixed-point).
/// Index with `PHASE_BASE[(tick + 9 - channel) & 7]` for channels 1-8.
const PHASE_BASE: [u8; GLOBAL_PHASE_TICKS_PER_PERIOD] = [7, 8, 10, 12, 13, 12, 10, 8];

// =========================================================================
// Workspace — reusable hot-path state
// =========================================================================
//
// Keeps scratch-buffer allocation out of repeated propagation calls.
// Phase gating uses the static PHASE_BASE constant (no precomputation).
//
//   scratch — per-neuron incoming-signal accumulator, Vec<i32>

/// Reusable buffers for repeated propagation calls.
///
/// Stores the per-neuron scratch buffer. Phase gating reads directly
/// from the static `PHASE_BASE` constant (8 bytes, no allocation).
/// Allocate once, pass to every `propagate_token` call.
#[derive(Debug)]
pub struct PropagationWorkspace {
    scratch: Vec<i32>,
}

impl PropagationWorkspace {
    /// Create a workspace sized for `neuron_count` neurons.
    pub fn new(neuron_count: usize) -> Self {
        Self {
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
    fn with_scratch(scratch: Vec<i32>) -> Self {
        Self { scratch }
    }
}

/// Per-neuron learned parameters for one propagation run.
pub struct PropagationParameters<'a> {
    /// Stored firing threshold per neuron. Range: `[0, 15]` (full int4).
    /// Effective threshold = stored + 1, giving range `[1, 16]`.
    pub threshold: &'a [u32],
    /// Phase gating channel per neuron. Range: `[1, 8]`.
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
    /// The phase-channel slice length does not match the graph neuron count.
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

        // =============================================================
        // Spike stage: phase-gated threshold comparison (x10 scale).
        //
        // Both sides are promoted to u16 for the comparison:
        //   charge_x10     = charge * 10              (max 15*10 = 150)
        //   threshold_x10  = (theta+1) * PHASE_BASE   (max 16*13 = 208)
        //
        // Max value 208 fits in u8, but u16 is used for safety headroom.
        // The x10 scale matches the PHASE_BASE values [7,8,10,12,13].
        // =============================================================
        let tip = tick % GLOBAL_PHASE_TICKS_PER_PERIOD;
        for neuron_idx in 0..neuron_count {
            let ch = params.channel[neuron_idx] as usize;
            let phase_mult: u16 = if ch >= 1 && ch <= GLOBAL_PHASE_CHANNEL_COUNT {
                // Rotate PHASE_BASE by channel: ch=1 peaks at tick 0, ch=2 at tick 1, etc.
                PHASE_BASE[(tip + 9 - ch) & 7] as u16
            } else {
                10 // neutral (no gating)
            };
            let charge_x10: u16 = state.charge[neuron_idx] as u16 * 10;
            // +1 shift: stored threshold 0-15, effective 1-16.
            // Uses full int4 range; stored=15 -> effective=16 (supergate).
            let threshold_x10: u16 =
                (params.threshold[neuron_idx] as u16 + 1) * phase_mult;

            if charge_x10 >= threshold_x10 {
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
