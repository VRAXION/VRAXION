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

use crate::parameters::{
    GLOBAL_CHARGE_DECAY_PERIOD, GLOBAL_INPUT_DURATION, GLOBAL_TICKS_PER_TOKEN,
    GLOBAL_WAVE_AMPLITUDE_PERMILLE, GLOBAL_WAVE_CHANNEL_COUNT, GLOBAL_WAVE_TICKS_PER_PERIOD,
    LIMIT_MAX_CHARGE,
};
use crate::topology::ConnectionGraph;
use std::error::Error;
use std::fmt;

type WaveGatingTable = [[u32; GLOBAL_WAVE_TICKS_PER_PERIOD]; GLOBAL_WAVE_CHANNEL_COUNT + 1];

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
            input_duration: GLOBAL_INPUT_DURATION,
            decay_period: GLOBAL_CHARGE_DECAY_PERIOD,
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
mod tests {
    use super::*;
    use crate::topology::DirectedEdge;

    fn default_config() -> PropagationConfig {
        PropagationConfig {
            ticks: 8,
            input_duration: 2,
            decay_period: 6,
        }
    }

    fn graph_with_edges(neuron_count: usize, pairs: &[(u16, u16)]) -> ConnectionGraph {
        ConnectionGraph::from_pairs(neuron_count, pairs)
    }

    #[test]
    fn isolated_neurons_remain_charge_bounded() {
        let neuron_count = 16;
        let graph = ConnectionGraph::new(neuron_count);
        let mut activation = vec![0i32; neuron_count];
        let mut charge = vec![0u32; neuron_count];
        let input = vec![1i32; neuron_count];
        let threshold = vec![6u32; neuron_count];
        let channel = vec![1u8; neuron_count];
        let polarity = vec![1i32; neuron_count];
        let mut workspace = PropagationWorkspace::new(neuron_count);

        propagate_token(
            &input,
            &graph,
            &PropagationParameters {
                threshold: &threshold,
                channel: &channel,
                polarity: &polarity,
            },
            &mut PropagationState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &default_config(),
            &mut workspace,
        )
        .unwrap();

        assert!(charge.iter().all(|&value| value <= LIMIT_MAX_CHARGE));
    }

    #[test]
    fn excitatory_chain_propagates_signal() {
        let neuron_count = 3;
        let graph = graph_with_edges(neuron_count, &[(0, 1), (1, 2)]);
        let mut activation = vec![0i32; neuron_count];
        let mut charge = vec![0u32; neuron_count];
        let mut input = vec![0i32; neuron_count];
        input[0] = 10;
        let threshold = vec![1u32; neuron_count];
        let channel = vec![1u8; neuron_count];
        let polarity = vec![1i32; neuron_count];
        let mut workspace = PropagationWorkspace::new(neuron_count);

        propagate_token(
            &input,
            &graph,
            &PropagationParameters {
                threshold: &threshold,
                channel: &channel,
                polarity: &polarity,
            },
            &mut PropagationState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 3,
                input_duration: 2,
                decay_period: 100,
            },
            &mut workspace,
        )
        .unwrap();

        let any_downstream_activity =
            charge[1] > 0 || charge[2] > 0 || activation[1] != 0 || activation[2] != 0;
        assert!(
            any_downstream_activity,
            "excitatory chain must propagate: c1={} a1={} c2={} a2={}",
            charge[1], activation[1], charge[2], activation[2]
        );
    }

    #[test]
    fn inhibitory_spike_suppresses_downstream_charge() {
        let neuron_count = 3;
        let graph = graph_with_edges(neuron_count, &[(0, 1)]);
        let mut activation = vec![0i32; neuron_count];
        let mut charge = vec![0u32; neuron_count];
        let mut input = vec![0i32; neuron_count];
        input[0] = 10;
        let threshold = vec![2u32; neuron_count];
        let channel = vec![1u8; neuron_count];
        let polarity = vec![-1i32, 1, 1];
        let mut workspace = PropagationWorkspace::new(neuron_count);

        charge[1] = 5;

        propagate_token(
            &input,
            &graph,
            &PropagationParameters {
                threshold: &threshold,
                channel: &channel,
                polarity: &polarity,
            },
            &mut PropagationState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 4,
                input_duration: 2,
                decay_period: 100,
            },
            &mut workspace,
        )
        .unwrap();

        assert!(
            charge[1] < 5,
            "inhibitory spike should suppress downstream charge, got {}",
            charge[1]
        );
    }

    #[test]
    fn extreme_input_does_not_overflow_charge() {
        let neuron_count = 8;
        let graph = ConnectionGraph::new(neuron_count);
        let mut activation = vec![0i32; neuron_count];
        let mut charge = vec![0u32; neuron_count];
        let input = vec![100i32; neuron_count];
        let threshold = vec![1u32; neuron_count];
        let channel = vec![1u8; neuron_count];
        let polarity = vec![1i32; neuron_count];
        let mut workspace = PropagationWorkspace::new(neuron_count);

        propagate_token(
            &input,
            &graph,
            &PropagationParameters {
                threshold: &threshold,
                channel: &channel,
                polarity: &polarity,
            },
            &mut PropagationState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 100,
                input_duration: 2,
                decay_period: 6,
            },
            &mut workspace,
        )
        .unwrap();

        for &charge_level in &charge {
            assert!(
                charge_level <= LIMIT_MAX_CHARGE,
                "charge out of bounds: {charge_level}"
            );
        }
    }

    #[test]
    fn workspace_reuse_produces_identical_results() {
        let neuron_count = 4;
        let graph = graph_with_edges(neuron_count, &[(0, 1), (1, 2), (2, 3)]);
        let input = vec![8i32, 0, 0, 0];
        let threshold = vec![2u32; neuron_count];
        let channel = vec![1u8; neuron_count];
        let polarity = vec![1i32; neuron_count];
        let config = PropagationConfig {
            ticks: 4,
            input_duration: 2,
            decay_period: 100,
        };
        let mut workspace = PropagationWorkspace::new(neuron_count);

        let mut activation_a = vec![0i32; neuron_count];
        let mut charge_a = vec![0u32; neuron_count];
        propagate_token(
            &input,
            &graph,
            &PropagationParameters {
                threshold: &threshold,
                channel: &channel,
                polarity: &polarity,
            },
            &mut PropagationState {
                activation: &mut activation_a,
                charge: &mut charge_a,
            },
            &config,
            &mut workspace,
        )
        .unwrap();

        let mut activation_b = vec![0i32; neuron_count];
        let mut charge_b = vec![0u32; neuron_count];
        propagate_token(
            &input,
            &graph,
            &PropagationParameters {
                threshold: &threshold,
                channel: &channel,
                polarity: &polarity,
            },
            &mut PropagationState {
                activation: &mut activation_b,
                charge: &mut charge_b,
            },
            &config,
            &mut workspace,
        )
        .unwrap();

        assert_eq!(activation_a, activation_b);
        assert_eq!(charge_a, charge_b);
    }

    #[test]
    fn activation_length_mismatch_returns_error() {
        let graph = ConnectionGraph::new(4);
        let mut activation = vec![0i32; 3];
        let mut charge = vec![0u32; 4];
        let mut workspace = PropagationWorkspace::new(4);

        let err = propagate_token(
            &[1; 4],
            &graph,
            &PropagationParameters {
                threshold: &[1; 4],
                channel: &[1; 4],
                polarity: &[1; 4],
            },
            &mut PropagationState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 1,
                input_duration: 1,
                decay_period: 0,
            },
            &mut workspace,
        )
        .unwrap_err();

        assert_eq!(
            err,
            PropagationError::ActivationLengthMismatch {
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn short_input_returns_error() {
        let graph = ConnectionGraph::new(4);
        let mut activation = vec![0i32; 4];
        let mut charge = vec![0u32; 4];
        let mut workspace = PropagationWorkspace::new(4);

        let err = propagate_token(
            &[1, 1],
            &graph,
            &PropagationParameters {
                threshold: &[1; 4],
                channel: &[1; 4],
                polarity: &[1; 4],
            },
            &mut PropagationState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 1,
                input_duration: 1,
                decay_period: 0,
            },
            &mut workspace,
        )
        .unwrap_err();

        assert_eq!(
            err,
            PropagationError::InputLengthMismatch {
                expected: 4,
                actual: 2,
            }
        );
    }

    #[test]
    fn charge_length_mismatch_returns_error() {
        let graph = ConnectionGraph::new(4);
        let mut activation = vec![0i32; 4];
        let mut charge = vec![0u32; 3];
        let mut workspace = PropagationWorkspace::new(4);

        let err = propagate_token(
            &[1; 4],
            &graph,
            &PropagationParameters {
                threshold: &[1; 4],
                channel: &[1; 4],
                polarity: &[1; 4],
            },
            &mut PropagationState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 1,
                input_duration: 1,
                decay_period: 0,
            },
            &mut workspace,
        )
        .unwrap_err();

        assert_eq!(
            err,
            PropagationError::ChargeLengthMismatch {
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn threshold_length_mismatch_returns_error() {
        let graph = ConnectionGraph::new(4);
        let mut activation = vec![0i32; 4];
        let mut charge = vec![0u32; 4];
        let mut workspace = PropagationWorkspace::new(4);

        let err = propagate_token(
            &[1; 4],
            &graph,
            &PropagationParameters {
                threshold: &[1; 3],
                channel: &[1; 4],
                polarity: &[1; 4],
            },
            &mut PropagationState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 1,
                input_duration: 1,
                decay_period: 0,
            },
            &mut workspace,
        )
        .unwrap_err();

        assert_eq!(
            err,
            PropagationError::ThresholdLengthMismatch {
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn channel_length_mismatch_returns_error() {
        let graph = ConnectionGraph::new(4);
        let mut activation = vec![0i32; 4];
        let mut charge = vec![0u32; 4];
        let mut workspace = PropagationWorkspace::new(4);

        let err = propagate_token(
            &[1; 4],
            &graph,
            &PropagationParameters {
                threshold: &[1; 4],
                channel: &[1; 3],
                polarity: &[1; 4],
            },
            &mut PropagationState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 1,
                input_duration: 1,
                decay_period: 0,
            },
            &mut workspace,
        )
        .unwrap_err();

        assert_eq!(
            err,
            PropagationError::ChannelLengthMismatch {
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn polarity_length_mismatch_returns_error() {
        let graph = ConnectionGraph::new(4);
        let mut activation = vec![0i32; 4];
        let mut charge = vec![0u32; 4];
        let mut workspace = PropagationWorkspace::new(4);

        let err = propagate_token(
            &[1; 4],
            &graph,
            &PropagationParameters {
                threshold: &[1; 4],
                channel: &[1; 4],
                polarity: &[1; 3],
            },
            &mut PropagationState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 1,
                input_duration: 1,
                decay_period: 0,
            },
            &mut workspace,
        )
        .unwrap_err();

        assert_eq!(
            err,
            PropagationError::PolarityLengthMismatch {
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn edge_length_mismatch_returns_error() {
        let graph = ConnectionGraph::from_raw_parts_for_tests(
            4,
            vec![
                DirectedEdge {
                    source: 0,
                    target: 1,
                },
                DirectedEdge {
                    source: 1,
                    target: 2,
                },
            ],
            vec![0, 1],
            vec![1],
        );
        let mut activation = vec![0i32; 4];
        let mut charge = vec![0u32; 4];
        let mut workspace = PropagationWorkspace::new(4);

        let err = propagate_token(
            &[1; 4],
            &graph,
            &PropagationParameters {
                threshold: &[1; 4],
                channel: &[1; 4],
                polarity: &[1; 4],
            },
            &mut PropagationState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 1,
                input_duration: 1,
                decay_period: 0,
            },
            &mut workspace,
        )
        .unwrap_err();

        assert_eq!(
            err,
            PropagationError::EdgeLengthMismatch {
                sources: 2,
                targets: 1,
            }
        );
    }

    #[test]
    fn out_of_range_edge_source_returns_error() {
        let graph = ConnectionGraph::from_raw_parts_for_tests(
            4,
            vec![DirectedEdge {
                source: 0,
                target: 1,
            }],
            vec![4],
            vec![1],
        );
        let mut activation = vec![0i32; 4];
        let mut charge = vec![0u32; 4];
        let mut workspace = PropagationWorkspace::new(4);

        let err = propagate_token(
            &[1; 4],
            &graph,
            &PropagationParameters {
                threshold: &[1; 4],
                channel: &[1; 4],
                polarity: &[1; 4],
            },
            &mut PropagationState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 1,
                input_duration: 1,
                decay_period: 0,
            },
            &mut workspace,
        )
        .unwrap_err();

        assert_eq!(
            err,
            PropagationError::EdgeSourceOutOfBounds {
                index: 0,
                value: 4,
                neuron_count: 4,
            }
        );
    }

    #[test]
    fn out_of_range_edge_target_returns_error() {
        let graph = ConnectionGraph::from_raw_parts_for_tests(
            4,
            vec![DirectedEdge {
                source: 0,
                target: 1,
            }],
            vec![0],
            vec![4],
        );
        let mut activation = vec![0i32; 4];
        let mut charge = vec![0u32; 4];
        let mut workspace = PropagationWorkspace::new(4);

        let err = propagate_token(
            &[1; 4],
            &graph,
            &PropagationParameters {
                threshold: &[1; 4],
                channel: &[1; 4],
                polarity: &[1; 4],
            },
            &mut PropagationState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 1,
                input_duration: 1,
                decay_period: 0,
            },
            &mut workspace,
        )
        .unwrap_err();

        assert_eq!(
            err,
            PropagationError::EdgeTargetOutOfBounds {
                index: 0,
                value: 4,
                neuron_count: 4,
            }
        );
    }

    #[test]
    fn scratch_too_small_returns_error() {
        let graph = ConnectionGraph::new(4);
        let mut activation = vec![0i32; 4];
        let mut charge = vec![0u32; 4];
        let mut workspace = PropagationWorkspace::from_parts(build_wave_gating_table(), vec![0; 3]);

        let err = propagate_token(
            &[1; 4],
            &graph,
            &PropagationParameters {
                threshold: &[1; 4],
                channel: &[1; 4],
                polarity: &[1; 4],
            },
            &mut PropagationState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 1,
                input_duration: 1,
                decay_period: 0,
            },
            &mut workspace,
        )
        .unwrap_err();

        assert_eq!(
            err,
            PropagationError::ScratchTooSmall {
                required: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn wave_table_range_is_valid() {
        let table = build_wave_gating_table();
        for (channel, row) in table
            .iter()
            .enumerate()
            .take(GLOBAL_WAVE_CHANNEL_COUNT + 1)
            .skip(1)
        {
            for (tick, &value) in row.iter().enumerate() {
                assert!(
                    (600..=1400).contains(&value),
                    "wave_table[{channel}][{tick}] = {value}"
                );
            }
        }
        assert!(table[0].iter().all(|&value| value == 1000));
    }
}
