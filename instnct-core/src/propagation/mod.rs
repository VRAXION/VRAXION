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
    GLOBAL_CHARGE_DECAY_INTERVAL_TICKS, GLOBAL_INPUT_DURATION_TICKS, GLOBAL_PHASE_CHANNEL_COUNT,
    GLOBAL_PHASE_TICKS_PER_PERIOD, GLOBAL_TICKS_PER_TOKEN, LIMIT_MAX_CHARGE,
};
use crate::topology::ConnectionGraph;
use std::error::Error;
use std::fmt;

// Phase gating — 8-byte cosine base pattern (x10 fixed-point).
// Channel N (1-8) rotates this so the easiest tick (7) aligns with tick N-1.
//   7 = 0.7x threshold (easiest) | 10 = neutral | 13 = 1.3x (hardest)
// Amplitude 0.3 baked in. Damping experiments (2026-04-04) confirmed optimal.
const PHASE_BASE: [u8; GLOBAL_PHASE_TICKS_PER_PERIOD] = [7, 8, 10, 12, 13, 12, 10, 8];

// Compile-time safety: spike stage uses u16 arithmetic.
// charge * 10 must fit u16 (max 150), and (threshold+1) * max_phase must fit u16 (max 208).
const _: () = assert!(LIMIT_MAX_CHARGE as u64 * 10 <= u16::MAX as u64);
const _: () = assert!((LIMIT_MAX_CHARGE as u64 + 1) * 13 <= u16::MAX as u64);

// ---- Workspace ----

/// Reusable scratch buffer. Allocate once, pass to every `propagate_token` call.
#[derive(Clone, Debug)]
pub struct PropagationWorkspace {
    incoming_scratch: Vec<i32>, // per-neuron incoming-signal accumulator
}

impl PropagationWorkspace {
    /// Create a workspace sized for `neuron_count` neurons.
    pub fn new(neuron_count: usize) -> Self {
        Self {
            incoming_scratch: vec![0; neuron_count],
        }
    }

    /// Grow buffer to fit `neuron_count`. Only grows, never shrinks.
    pub fn ensure_neuron_count(&mut self, neuron_count: usize) {
        if self.incoming_scratch.len() < neuron_count {
            self.incoming_scratch.resize(neuron_count, 0);
        }
    }

    /// Mutable access to the scratch buffer (used by Network CSR propagation).
    pub(crate) fn incoming_scratch_mut(&mut self) -> &mut [i32] {
        &mut self.incoming_scratch
    }

    #[cfg(test)]
    fn with_scratch(incoming_scratch: Vec<i32>) -> Self {
        Self { incoming_scratch }
    }
}

// ---- Per-neuron parameters and state ----

/// Per-neuron learned parameters for one propagation run.
#[allow(missing_docs)]
pub struct PropagationParameters<'a> {
    pub threshold: &'a [u32], // stored [0,15], effective = stored+1 -> [1,16]
    pub channel: &'a [u8],    // phase gating channel [1,8]
    pub polarity: &'a [i32],  // +1 excitatory, -1 inhibitory
}

/// Mutable neuron state carried across tokens.
#[allow(missing_docs)]
pub struct PropagationState<'a> {
    pub activation: &'a mut [i32], // +1, -1, or 0
    pub charge: &'a mut [u32],     // [0, LIMIT_MAX_CHARGE]
    pub refractory: &'a mut [u8],  // 0 = ready, >0 = ticks remaining in cooldown
}

/// Timing configuration for one forward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct PropagationConfig {
    pub ticks_per_token: usize,      // total simulation ticks per token
    pub input_duration_ticks: usize, // injection window at start
    pub decay_interval_ticks: usize, // charge -= 1 every N ticks
    pub use_refractory: bool,        // 1-tick cooldown after firing (matches Python)
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            ticks_per_token: GLOBAL_TICKS_PER_TOKEN,
            input_duration_ticks: GLOBAL_INPUT_DURATION_TICKS,
            decay_interval_ticks: GLOBAL_CHARGE_DECAY_INTERVAL_TICKS,
            use_refractory: false,
        }
    }
}

// ---- Validation errors ----

/// Propagation input validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(missing_docs)] // variant names + field names are self-evident
pub enum PropagationError {
    ActivationLengthMismatch {
        expected: usize,
        actual: usize,
    }, // activation.len() != neuron_count
    InputLengthMismatch {
        expected: usize,
        actual: usize,
    }, // input.len() != neuron_count
    ChargeLengthMismatch {
        expected: usize,
        actual: usize,
    }, // charge.len() != neuron_count
    /// refractory.len() != neuron_count
    RefractoryLengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    ThresholdLengthMismatch {
        expected: usize,
        actual: usize,
    }, // threshold.len() != neuron_count
    ChannelLengthMismatch {
        expected: usize,
        actual: usize,
    }, // channel.len() != neuron_count
    PolarityLengthMismatch {
        expected: usize,
        actual: usize,
    }, // polarity.len() != neuron_count
    ScratchTooSmall {
        required: usize,
        actual: usize,
    }, // workspace buffer < neuron_count
    EdgeLengthMismatch {
        sources: usize,
        targets: usize,
    }, // sources.len() != targets.len()
    EdgeSourceOutOfBounds {
        index: usize,
        value: usize,
        neuron_count: usize,
    }, // source >= n
    EdgeTargetOutOfBounds {
        index: usize,
        value: usize,
        neuron_count: usize,
    }, // target >= n
    ThresholdOutOfRange {
        index: usize,
        value: u32,
    }, // threshold > 15
    ChannelOutOfRange {
        index: usize,
        value: u8,
    }, // channel not in 1..=8
    PolarityOutOfRange {
        index: usize,
        value: i32,
    }, // polarity not ±1
}

impl fmt::Display for PropagationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActivationLengthMismatch { expected, actual } => write!(
                f,
                "activation length mismatch: expected {expected}, got {actual}"
            ),
            Self::InputLengthMismatch { expected, actual } => write!(
                f,
                "input length mismatch: expected {expected}, got {actual}"
            ),
            Self::ChargeLengthMismatch { expected, actual } => write!(
                f,
                "charge length mismatch: expected {expected}, got {actual}"
            ),
            Self::RefractoryLengthMismatch { expected, actual } => write!(
                f,
                "refractory length mismatch: expected {expected}, got {actual}"
            ),
            Self::ThresholdLengthMismatch { expected, actual } => write!(
                f,
                "threshold length mismatch: expected {expected}, got {actual}"
            ),
            Self::ChannelLengthMismatch { expected, actual } => write!(
                f,
                "channel length mismatch: expected {expected}, got {actual}"
            ),
            Self::PolarityLengthMismatch { expected, actual } => write!(
                f,
                "polarity length mismatch: expected {expected}, got {actual}"
            ),
            Self::ScratchTooSmall { required, actual } => {
                write!(f, "scratch buffer too small: need {required}, got {actual}")
            }
            Self::EdgeLengthMismatch { sources, targets } => write!(
                f,
                "edge length mismatch: {sources} sources vs {targets} targets"
            ),
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
            Self::ThresholdOutOfRange { index, value } => {
                write!(f, "threshold out of range at neuron {index}: {value} > 15")
            }
            Self::ChannelOutOfRange { index, value } => write!(
                f,
                "channel out of range at neuron {index}: {value} not in 1..=8"
            ),
            Self::PolarityOutOfRange { index, value } => write!(
                f,
                "polarity out of range at neuron {index}: {value} not in {{-1, +1}}"
            ),
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
) -> Result<(), PropagationError> {
    let n = graph.neuron_count();
    let (edge_src, edge_tgt) = graph.edge_endpoints();

    // Slice length checks
    if state.activation.len() != n {
        return Err(PropagationError::ActivationLengthMismatch {
            expected: n,
            actual: state.activation.len(),
        });
    }
    if state.charge.len() != n {
        return Err(PropagationError::ChargeLengthMismatch {
            expected: n,
            actual: state.charge.len(),
        });
    }
    if state.refractory.len() != n {
        return Err(PropagationError::RefractoryLengthMismatch {
            expected: n,
            actual: state.refractory.len(),
        });
    }
    if input.len() != n {
        return Err(PropagationError::InputLengthMismatch {
            expected: n,
            actual: input.len(),
        });
    }
    if params.threshold.len() != n {
        return Err(PropagationError::ThresholdLengthMismatch {
            expected: n,
            actual: params.threshold.len(),
        });
    }
    if params.channel.len() != n {
        return Err(PropagationError::ChannelLengthMismatch {
            expected: n,
            actual: params.channel.len(),
        });
    }
    if params.polarity.len() != n {
        return Err(PropagationError::PolarityLengthMismatch {
            expected: n,
            actual: params.polarity.len(),
        });
    }
    if workspace.incoming_scratch.len() < n {
        return Err(PropagationError::ScratchTooSmall {
            required: n,
            actual: workspace.incoming_scratch.len(),
        });
    }
    if edge_src.len() != edge_tgt.len() {
        return Err(PropagationError::EdgeLengthMismatch {
            sources: edge_src.len(),
            targets: edge_tgt.len(),
        });
    }

    // Value range checks (single pass over neurons)
    for i in 0..n {
        if params.threshold[i] > 15 {
            return Err(PropagationError::ThresholdOutOfRange {
                index: i,
                value: params.threshold[i],
            });
        }
        if !(1..=GLOBAL_PHASE_CHANNEL_COUNT as u8).contains(&params.channel[i]) {
            return Err(PropagationError::ChannelOutOfRange {
                index: i,
                value: params.channel[i],
            });
        }
        let p = params.polarity[i];
        if p != 1 && p != -1 {
            return Err(PropagationError::PolarityOutOfRange { index: i, value: p });
        }
    }

    // Edge endpoint bounds
    for (i, (&s, &t)) in edge_src.iter().zip(edge_tgt.iter()).enumerate() {
        if s >= n {
            return Err(PropagationError::EdgeSourceOutOfBounds {
                index: i,
                value: s,
                neuron_count: n,
            });
        }
        if t >= n {
            return Err(PropagationError::EdgeTargetOutOfBounds {
                index: i,
                value: t,
                neuron_count: n,
            });
        }
    }

    Ok(())
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
    debug_assert_eq!(state.refractory.len(), neuron_count);
    debug_assert_eq!(input.len(), neuron_count);
    debug_assert_eq!(params.threshold.len(), neuron_count);
    debug_assert_eq!(params.channel.len(), neuron_count);
    debug_assert_eq!(params.polarity.len(), neuron_count);
    debug_assert_eq!(edge_sources.len(), edge_targets.len());
    debug_assert!(workspace.incoming_scratch.len() >= neuron_count);

    for tick in 0..config.ticks_per_token {
        // Charge decay: subtract 1 every N ticks
        if config.decay_interval_ticks > 0 && tick % config.decay_interval_ticks == 0 {
            for charge in state.charge.iter_mut() {
                *charge = charge.saturating_sub(1);
            }
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
        for (source_chunk, target_chunk) in edge_sources
            .chunks_exact(4)
            .zip(edge_targets.chunks_exact(4))
        {
            incoming[target_chunk[0]] += state.activation[source_chunk[0]]; // unrolled — LLVM drops bounds checks
            incoming[target_chunk[1]] += state.activation[source_chunk[1]]; // because chunks_exact guarantees len==4
            incoming[target_chunk[2]] += state.activation[source_chunk[2]];
            incoming[target_chunk[3]] += state.activation[source_chunk[3]];
        }
        let rem_start = edge_sources.len() / 4 * 4; // remainder (0-3 edges)
        for i in rem_start..edge_sources.len() {
            incoming[edge_targets[i]] += state.activation[edge_sources[i]];
        }

        // Charge accumulation: clamp to [0, MAX_CHARGE]
        for (charge, &signal) in state.charge.iter_mut().zip(incoming.iter()) {
            *charge = charge.saturating_add_signed(signal).min(LIMIT_MAX_CHARGE);
        }

        // Spike stage: charge*10 >= (theta+1) * PHASE_BASE  (max 150 vs 208, fits u16)
        let phase_tick = tick % GLOBAL_PHASE_TICKS_PER_PERIOD;
        for neuron_idx in 0..neuron_count {
            // Refractory gate: neuron in cooldown cannot fire
            if config.use_refractory && state.refractory[neuron_idx] > 0 {
                state.refractory[neuron_idx] -= 1;
                state.activation[neuron_idx] = 0;
                continue;
            }
            let channel_idx = params.channel[neuron_idx] as usize;
            let phase_mult: u16 = if (1..=GLOBAL_PHASE_CHANNEL_COUNT).contains(&channel_idx) {
                PHASE_BASE[(phase_tick + 9 - channel_idx) & 7] as u16 // rotate: channel 1 peaks at tick 0
            } else {
                10 // neutral (no gating)
            };
            let charge_x10: u16 = state.charge[neuron_idx] as u16 * 10;
            let threshold_x10: u16 = (params.threshold[neuron_idx] as u16 + 1) * phase_mult; // +1 shift: stored 0-15, effective 1-16

            if charge_x10 >= threshold_x10 {
                state.activation[neuron_idx] = params.polarity[neuron_idx]; // fire
                state.charge[neuron_idx] = 0; // reset
                if config.use_refractory {
                    state.refractory[neuron_idx] = 1; // 1-tick cooldown
                }
            } else {
                state.activation[neuron_idx] = 0; // silent
            }
        }
    }
}

#[cfg(test)]
mod tests;
