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

use crate::parameters::{
    GLOBAL_CHARGE_DECAY_INTERVAL_TICKS, GLOBAL_INPUT_DURATION_TICKS, GLOBAL_TICKS_PER_TOKEN,
    GLOBAL_PHASE_CHANNEL_COUNT, GLOBAL_PHASE_TICKS_PER_PERIOD, LIMIT_MAX_CHARGE,
};
use crate::topology::ConnectionGraph;
use std::error::Error;
use std::fmt;

// Phase gating — 8-byte cosine base pattern (x10 fixed-point).
// Channel N (1-8) rotates this so the easiest tick (7) aligns with tick N-1.
//   7 = 0.7x threshold (easiest) | 10 = neutral | 13 = 1.3x (hardest)
// Amplitude 0.3 baked in. Damping experiments (2026-04-04) confirmed optimal.
const PHASE_BASE: [u8; GLOBAL_PHASE_TICKS_PER_PERIOD] = [7, 8, 10, 12, 13, 12, 10, 8];

// ---- Workspace ----

/// Reusable scratch buffer. Allocate once, pass to every `propagate_token` call.
#[derive(Debug)]
pub struct PropagationWorkspace {
    incoming_scratch: Vec<i32>, // per-neuron incoming-signal accumulator
}

impl PropagationWorkspace {
    /// Create a workspace sized for `neuron_count` neurons.
    pub fn new(neuron_count: usize) -> Self {
        Self { incoming_scratch: vec![0; neuron_count] }
    }

    /// Grow buffer to fit `neuron_count`. Only grows, never shrinks.
    pub fn ensure_neuron_count(&mut self, neuron_count: usize) {
        if self.incoming_scratch.len() < neuron_count {
            self.incoming_scratch.resize(neuron_count, 0);
        }
    }

    #[cfg(test)]
    fn with_scratch(incoming_scratch: Vec<i32>) -> Self {
        Self { incoming_scratch }
    }
}

// ---- Per-neuron parameters and state ----

/// Per-neuron learned parameters for one propagation run.
pub struct PropagationParameters<'a> {
    /// Stored [0,15], effective = stored+1 -> [1,16].
    pub threshold: &'a [u32],
    /// Phase gating channel [1,8].
    pub channel: &'a [u8],
    /// +1 excitatory, -1 inhibitory.
    pub polarity: &'a [i32],
}

/// Mutable neuron state carried across tokens.
pub struct PropagationState<'a> {
    /// +1, -1, or 0.
    pub activation: &'a mut [i32],
    /// [0, LIMIT_MAX_CHARGE].
    pub charge: &'a mut [u32],
}

/// Timing configuration for one forward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PropagationConfig {
    /// Total simulation ticks per token.
    pub ticks_per_token: usize,
    /// Injection window at start.
    pub input_duration_ticks: usize,
    /// Charge -= 1 every N ticks.
    pub decay_interval_ticks: usize,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            ticks_per_token: GLOBAL_TICKS_PER_TOKEN,
            input_duration_ticks: GLOBAL_INPUT_DURATION_TICKS,
            decay_interval_ticks: GLOBAL_CHARGE_DECAY_INTERVAL_TICKS,
        }
    }
}

// ---- Validation errors ----

/// Propagation input validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(missing_docs)] // variant field names are self-evident (expected/actual, etc.)
pub enum PropagationError {
    /// Activation buffer length != neuron count.
    ActivationLengthMismatch { expected: usize, actual: usize },
    /// Input slice length != neuron count.
    InputLengthMismatch { expected: usize, actual: usize },
    /// Charge buffer length != neuron count.
    ChargeLengthMismatch { expected: usize, actual: usize },
    /// Threshold slice length != neuron count.
    ThresholdLengthMismatch { expected: usize, actual: usize },
    /// Channel slice length != neuron count.
    ChannelLengthMismatch { expected: usize, actual: usize },
    /// Polarity slice length != neuron count.
    PolarityLengthMismatch { expected: usize, actual: usize },
    /// Workspace scratch buffer too small.
    ScratchTooSmall { required: usize, actual: usize },
    /// Edge source/target caches have different lengths.
    EdgeLengthMismatch { sources: usize, targets: usize },
    /// Edge source index >= neuron count.
    EdgeSourceOutOfBounds { index: usize, value: usize, neuron_count: usize },
    /// Edge target index >= neuron count.
    EdgeTargetOutOfBounds { index: usize, value: usize, neuron_count: usize },
}

impl fmt::Display for PropagationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActivationLengthMismatch { expected, actual } =>
                write!(f, "activation length mismatch: expected {expected}, got {actual}"),
            Self::InputLengthMismatch { expected, actual } =>
                write!(f, "input length mismatch: expected {expected}, got {actual}"),
            Self::ChargeLengthMismatch { expected, actual } =>
                write!(f, "charge length mismatch: expected {expected}, got {actual}"),
            Self::ThresholdLengthMismatch { expected, actual } =>
                write!(f, "threshold length mismatch: expected {expected}, got {actual}"),
            Self::ChannelLengthMismatch { expected, actual } =>
                write!(f, "channel length mismatch: expected {expected}, got {actual}"),
            Self::PolarityLengthMismatch { expected, actual } =>
                write!(f, "polarity length mismatch: expected {expected}, got {actual}"),
            Self::ScratchTooSmall { required, actual } =>
                write!(f, "scratch buffer too small: need {required}, got {actual}"),
            Self::EdgeLengthMismatch { sources, targets } =>
                write!(f, "edge length mismatch: {sources} sources vs {targets} targets"),
            Self::EdgeSourceOutOfBounds { index, value, neuron_count } =>
                write!(f, "edge source out of bounds at edge {index}: {value} >= {neuron_count}"),
            Self::EdgeTargetOutOfBounds { index, value, neuron_count } =>
                write!(f, "edge target out of bounds at edge {index}: {value} >= {neuron_count}"),
        }
    }
}

impl Error for PropagationError {}

// ---- Validation ----

fn validate_propagation_inputs(
    input: &[i32],
    graph: &ConnectionGraph,
    params: &PropagationParameters<'_>,
    state: &PropagationState<'_>,
    workspace: &PropagationWorkspace,
) -> Result<usize, PropagationError> {
    let n = graph.neuron_count();
    let (edge_src, edge_tgt) = graph.edge_endpoints();

    if state.activation.len() != n { return Err(PropagationError::ActivationLengthMismatch { expected: n, actual: state.activation.len() }); }
    if state.charge.len() != n     { return Err(PropagationError::ChargeLengthMismatch { expected: n, actual: state.charge.len() }); }
    if input.len() != n            { return Err(PropagationError::InputLengthMismatch { expected: n, actual: input.len() }); }
    if params.threshold.len() != n { return Err(PropagationError::ThresholdLengthMismatch { expected: n, actual: params.threshold.len() }); }
    if params.channel.len() != n   { return Err(PropagationError::ChannelLengthMismatch { expected: n, actual: params.channel.len() }); }
    if params.polarity.len() != n  { return Err(PropagationError::PolarityLengthMismatch { expected: n, actual: params.polarity.len() }); }
    if workspace.incoming_scratch.len() < n { return Err(PropagationError::ScratchTooSmall { required: n, actual: workspace.incoming_scratch.len() }); }
    if edge_src.len() != edge_tgt.len() { return Err(PropagationError::EdgeLengthMismatch { sources: edge_src.len(), targets: edge_tgt.len() }); }

    for (i, &s) in edge_src.iter().enumerate() {
        if s >= n { return Err(PropagationError::EdgeSourceOutOfBounds { index: i, value: s, neuron_count: n }); }
    }
    for (i, &t) in edge_tgt.iter().enumerate() {
        if t >= n { return Err(PropagationError::EdgeTargetOutOfBounds { index: i, value: t, neuron_count: n }); }
    }

    Ok(n)
}

// ---- Public API ----

/// Propagate one token through the spiking network (checked).
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

/// Unchecked fast path — caller must guarantee valid inputs.
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
    debug_assert!(workspace.incoming_scratch.len() >= neuron_count);

    for tick in 0..config.ticks_per_token {
        // Charge decay: subtract 1 every N ticks
        if config.decay_interval_ticks > 0 && tick % config.decay_interval_ticks == 0 {
            for charge in state.charge.iter_mut() { *charge = charge.saturating_sub(1); }
        }

        // Input injection: first N ticks only
        if tick < config.input_duration_ticks {
            for (activation, &input_value) in state.activation.iter_mut().zip(input.iter()) {
                *activation += input_value;
            }
        }

        // Scatter-add: accumulate incoming signals per neuron
        let incoming = &mut workspace.incoming_scratch[..neuron_count];
        incoming.fill(0);
        for (source_chunk, target_chunk) in edge_sources.chunks(4).zip(edge_targets.chunks(4)) {
            for chunk_idx in 0..source_chunk.len() {
                incoming[target_chunk[chunk_idx]] += state.activation[source_chunk[chunk_idx]];
            }
        }

        // Charge accumulation: clamp to [0, MAX_CHARGE]
        for (charge, &signal) in state.charge.iter_mut().zip(incoming.iter()) {
            *charge = (*charge as i32 + signal).clamp(0, LIMIT_MAX_CHARGE as i32) as u32;
        }

        // Spike stage: charge*10 >= (theta+1) * PHASE_BASE  (max 150 vs 208, fits u16)
        let tip = tick % GLOBAL_PHASE_TICKS_PER_PERIOD;
        for neuron_idx in 0..neuron_count {
            let ch = params.channel[neuron_idx] as usize;
            let phase_mult: u16 = if ch >= 1 && ch <= GLOBAL_PHASE_CHANNEL_COUNT {
                PHASE_BASE[(tip + 9 - ch) & 7] as u16 // rotate: ch=1 peaks at tick 0
            } else {
                10 // neutral (no gating)
            };
            let charge_x10: u16 = state.charge[neuron_idx] as u16 * 10;
            let threshold_x10: u16 = (params.threshold[neuron_idx] as u16 + 1) * phase_mult; // +1 shift: stored 0-15, effective 1-16

            if charge_x10 >= threshold_x10 {
                state.activation[neuron_idx] = params.polarity[neuron_idx]; // fire
                state.charge[neuron_idx] = 0; // reset
            } else {
                state.activation[neuron_idx] = 0; // silent
            }
        }
    }
}

#[cfg(test)]
mod tests;
