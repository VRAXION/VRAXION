//! # Signal Propagation — Spiking Forward Pass
//!
//! Simulates one token's passage through the recurrent spiking network.
//! This is the **performance-critical inner loop**.
//!
//! ## Integer-Only Design
//!
//! The forward pass uses `i32` activations (supports inhibitory -1) and
//! `u32` charge (always non-negative). The wave gating LUT is fixed-point
//! (×1000 scale). No floating-point arithmetic in the hot path.
//!
//! ## Inhibitory Mechanism
//!
//! Excitatory neurons fire as `+1`, inhibitory neurons fire as `-1`.
//! The scatter-add accumulates into a signed `i32` buffer, so inhibitory
//! spikes genuinely suppress downstream charge. Charge is clamped to
//! `[0, LIMIT_MAX_CHARGE]` after accumulation, preventing negative charge.

use crate::parameters::{
    GLOBAL_CHARGE_DECAY_PERIOD, GLOBAL_INPUT_DURATION, GLOBAL_TICKS_PER_TOKEN,
    GLOBAL_WAVE_AMPLITUDE_PERMILLE, GLOBAL_WAVE_CHANNEL_COUNT, GLOBAL_WAVE_TICKS_PER_PERIOD,
    LIMIT_MAX_CHARGE,
};

// =========================================================================
// 1. DATA TYPES — what the network is made of
// =========================================================================

/// Per-neuron learned parameters — evolved during training, fixed during inference.
///
/// Each neuron has three co-evolved properties that determine its firing behavior.
/// These are stored as flat slices of length `neuron_count`.
pub struct NeuronParameters<'a> {
    /// **[per-neuron, learnable]** Firing threshold. Range: [1, 15].
    /// Higher = harder to fire. Evolved via `theta` mutation.
    pub threshold: &'a [u32],

    /// **[per-neuron, learnable]** Wave gating channel. Range: [1, 8].
    /// Determines which tick in the 8-tick period is the neuron's "preferred" firing time.
    /// Evolved via `channel` mutation. See [Wave Gating](#wave-gating) below.
    pub channel: &'a [u8],

    /// **[per-neuron, learnable]** Polarity: +1 (excitatory) or -1 (inhibitory).
    /// Excitatory spikes add charge downstream; inhibitory spikes subtract.
    /// ~90% excitatory, ~10% inhibitory (fly-realistic ratio).
    /// Evolved via `flip` mutation.
    pub polarity: &'a [i32],
}

/// Persistent internal state — carried across tokens, mutated each tick.
///
/// Unlike feedforward networks, INSTNCT neurons retain state between inputs.
/// The network's response to token N depends on residual charge from tokens 0..N-1.
pub struct NeuronState<'a> {
    /// **[per-neuron, runtime]** Activation: +1 (excitatory fire), -1 (inhibitory fire), 0 (silent).
    /// Changes every tick. Signed to support inhibitory subtraction in scatter-add.
    pub activation: &'a mut [i32],

    /// **[per-neuron, runtime]** Accumulated charge. Range: [0, LIMIT_MAX_CHARGE].
    /// Changes every tick. Incoming signals add; firing resets to zero; decay subtracts.
    pub charge: &'a mut [u32],
}

/// Timing configuration for one forward pass.
pub struct PropagationConfig {
    /// **[global, fixed]** Simulation ticks per token. More ticks = deeper signal propagation.
    /// Typical: 12 (H=256), 16 (H=1024+). A loop of length N needs N ticks.
    pub ticks: usize,

    /// **[global, fixed]** Ticks during which external input is injected. Typical: 2.
    pub input_duration: usize,

    /// **[global, fixed]** Subtract 1 from all charges every N ticks. Typical: 6.
    /// Prevents runaway charge in high-in-degree neurons.
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

// =========================================================================
// 2. FORWARD PASS — the hot path, what happens every token
// =========================================================================

/// Propagate one token through the spiking network.
///
/// This runs the 5-step tick loop that is the core of INSTNCT:
///
/// ```text
/// For each tick:
///   1. DECAY    — leak charge (every decay_period ticks)
///   2. INPUT    — inject external signal (first input_duration ticks)
///   3. SCATTER  — propagate spikes along edges (the hot inner loop)
///   4. ACCUMULATE — add incoming signals to charge (clamp to [0, 15])
///   5. SPIKE    — fire if charge exceeds wave-gated threshold, reset charge
/// ```
///
/// # Arguments
///
/// * `input` — External signal to inject (length: neuron_count)
/// * `edge_sources`, `edge_targets` — Sparse edge list from `ConnectionGraph`
/// * `params` — Per-neuron threshold, channel, polarity
/// * `state` — Mutable activation + charge (persists across tokens)
/// * `config` — Ticks, input duration, decay period
/// * `wave_table` — Precomputed threshold modulation (build once, reuse)
/// * `scratch` — Pre-allocated temp buffer (length >= neuron_count)
///
/// # Panics
///
/// Panics if slice lengths are inconsistent or edge arrays differ in length.
pub fn propagate_token(
    input: &[i32],
    edge_sources: &[usize],
    edge_targets: &[usize],
    params: &NeuronParameters,
    state: &mut NeuronState,
    config: &PropagationConfig,
    wave_table: &[[u32; GLOBAL_WAVE_TICKS_PER_PERIOD]; GLOBAL_WAVE_CHANNEL_COUNT + 1],
    scratch: &mut [i32],
) {
    let neuron_count = state.activation.len();

    // --- Shape guards (active in ALL builds) ---
    assert_eq!(state.charge.len(), neuron_count, "charge length mismatch");
    assert_eq!(input.len(), neuron_count, "input length mismatch");
    assert_eq!(
        params.threshold.len(),
        neuron_count,
        "threshold length mismatch"
    );
    assert_eq!(
        params.channel.len(),
        neuron_count,
        "channel length mismatch"
    );
    assert_eq!(
        params.polarity.len(),
        neuron_count,
        "polarity length mismatch"
    );
    assert!(scratch.len() >= neuron_count, "scratch buffer too small");
    assert_eq!(
        edge_sources.len(),
        edge_targets.len(),
        "edge source/target length mismatch"
    );
    debug_assert!(
        edge_sources.iter().all(|&s| s < neuron_count),
        "edge source index out of bounds"
    );
    debug_assert!(
        edge_targets.iter().all(|&t| t < neuron_count),
        "edge target index out of bounds"
    );

    let edge_count = edge_sources.len();

    // --- Tick loop ---
    for tick in 0..config.ticks {
        // Step 1: DECAY — periodic charge leak
        if config.decay_period > 0 && tick % config.decay_period == 0 {
            for charge in state.charge.iter_mut() {
                *charge = charge.saturating_sub(1);
            }
        }

        // Step 2: INPUT — inject external signal into activation
        if tick < config.input_duration {
            for (activation, &input_val) in state.activation.iter_mut().zip(input.iter()) {
                *activation += input_val;
            }
        }

        // Step 3: SCATTER-ADD — propagate spikes along edges
        // This is THE HOT INNER LOOP. For each edge, the source neuron's
        // activation (+1, -1, or 0) is added to the target's incoming buffer.
        // Inhibitory neurons contribute -1, genuinely suppressing downstream charge.
        let incoming = &mut scratch[..neuron_count];
        incoming.fill(0);
        for edge_idx in 0..edge_count {
            incoming[edge_targets[edge_idx]] += state.activation[edge_sources[edge_idx]];
        }

        // Step 4: ACCUMULATE — add incoming signals to charge
        for (charge, &signal) in state.charge.iter_mut().zip(incoming.iter()) {
            let new_charge = *charge as i32 + signal;
            *charge = new_charge.clamp(0, LIMIT_MAX_CHARGE as i32) as u32;
        }

        // Step 5: SPIKE — wave-gated threshold comparison
        // Each neuron fires if: charge × 1000 >= threshold × wave_multiplier
        // (integer comparison avoids division)
        for neuron_idx in 0..neuron_count {
            let channel = params.channel[neuron_idx] as usize;
            let wave_mult = if channel <= GLOBAL_WAVE_CHANNEL_COUNT {
                wave_table[channel][tick % GLOBAL_WAVE_TICKS_PER_PERIOD]
            } else {
                1000
            };
            let charge_scaled = state.charge[neuron_idx] * 1000;
            let threshold_scaled = params.threshold[neuron_idx] * wave_mult;

            if charge_scaled >= threshold_scaled {
                state.activation[neuron_idx] = params.polarity[neuron_idx];
                state.charge[neuron_idx] = 0;
            } else {
                state.activation[neuron_idx] = 0;
            }
        }
    }
}

// =========================================================================
// 3. WAVE GATING — temporal specialization via cosine-modulated thresholds
// =========================================================================
//
// Each neuron is assigned a "channel" (1-8) that determines when in the
// 8-tick period it fires most easily. Channel N peaks at tick N-1.
//
// The lookup table stores threshold multipliers × 1000 (fixed-point):
//   - 700  = 0.7× threshold (easiest to fire)
//   - 1000 = 1.0× threshold (neutral)
//   - 1300 = 1.3× threshold (hardest to fire)
//
// This creates temporal structure: different neurons respond at different
// times, letting the network process sequential information within the
// tick window of a single token.
//
// Validated: wave gating (23.8%) > sine-wave gating (21.4%) > none (6.7%).

/// Build the wave gating lookup table. Call once at startup.
///
/// Returns a `[9][8]` array: `table[channel][tick]` is the threshold
/// multiplier × 1000. Channel 0 is neutral (all 1000), channels 1-8
/// follow cosine curves offset by their channel number.
pub fn build_wave_gating_table(
) -> [[u32; GLOBAL_WAVE_TICKS_PER_PERIOD]; GLOBAL_WAVE_CHANNEL_COUNT + 1] {
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

// =========================================================================
// 4. TESTS
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_wave_table() -> [[u32; GLOBAL_WAVE_TICKS_PER_PERIOD]; GLOBAL_WAVE_CHANNEL_COUNT + 1] {
        build_wave_gating_table()
    }

    fn default_config() -> PropagationConfig {
        PropagationConfig {
            ticks: 8,
            input_duration: 2,
            decay_period: 6,
        }
    }

    #[test]
    fn isolated_neurons_remain_charge_bounded() {
        let h = 16;
        let mut activation = vec![0i32; h];
        let mut charge = vec![0u32; h];
        let mut scratch = vec![0i32; h];
        let input = vec![1i32; h];
        let threshold = vec![6u32; h];
        let channel = vec![1u8; h];
        let polarity = vec![1i32; h];
        let wt = make_wave_table();

        propagate_token(
            &input,
            &[],
            &[],
            &NeuronParameters {
                threshold: &threshold,
                channel: &channel,
                polarity: &polarity,
            },
            &mut NeuronState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &default_config(),
            &wt,
            &mut scratch,
        );
        assert!(charge.iter().all(|&c| c <= LIMIT_MAX_CHARGE));
    }

    #[test]
    fn excitatory_chain_propagates_signal() {
        let h = 3;
        let mut activation = vec![0i32; h];
        let mut charge = vec![0u32; h];
        let mut scratch = vec![0i32; h];
        let mut input = vec![0i32; h];
        input[0] = 10;

        let sources = vec![0, 1];
        let targets = vec![1, 2];
        let threshold = vec![1u32; h];
        let channel = vec![1u8; h];
        let polarity = vec![1i32; h];

        let wt = make_wave_table();
        propagate_token(
            &input,
            &sources,
            &targets,
            &NeuronParameters {
                threshold: &threshold,
                channel: &channel,
                polarity: &polarity,
            },
            &mut NeuronState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 3,
                input_duration: 2,
                decay_period: 100,
            },
            &wt,
            &mut scratch,
        );
        let any_downstream =
            charge[1] > 0 || charge[2] > 0 || activation[1] != 0 || activation[2] != 0;
        assert!(
            any_downstream,
            "excitatory chain must propagate: c1={} a1={} c2={} a2={}",
            charge[1], activation[1], charge[2], activation[2]
        );
    }

    #[test]
    fn inhibitory_spike_suppresses_downstream_charge() {
        let h = 3;
        let mut activation = vec![0i32; h];
        let mut charge = vec![0u32; h];
        let mut scratch = vec![0i32; h];
        let mut input = vec![0i32; h];
        input[0] = 10;

        let sources = vec![0];
        let targets = vec![1];
        let threshold = vec![2u32; h];
        let channel = vec![1u8; h];
        let polarity = vec![-1i32, 1, 1]; // neuron 0 is INHIBITORY

        charge[1] = 5; // pre-charge so we can see suppression

        let wt = make_wave_table();
        propagate_token(
            &input,
            &sources,
            &targets,
            &NeuronParameters {
                threshold: &threshold,
                channel: &channel,
                polarity: &polarity,
            },
            &mut NeuronState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 4,
                input_duration: 2,
                decay_period: 100,
            },
            &wt,
            &mut scratch,
        );
        assert!(
            charge[1] < 5,
            "inhibitory spike should suppress downstream charge, got {}",
            charge[1]
        );
    }

    #[test]
    fn extreme_input_does_not_overflow_charge() {
        let h = 8;
        let mut activation = vec![0i32; h];
        let mut charge = vec![0u32; h];
        let mut scratch = vec![0i32; h];
        let input = vec![100i32; h];
        let wt = make_wave_table();

        propagate_token(
            &input,
            &[],
            &[],
            &NeuronParameters {
                threshold: &vec![1; h],
                channel: &vec![1; h],
                polarity: &vec![1; h],
            },
            &mut NeuronState {
                activation: &mut activation,
                charge: &mut charge,
            },
            &PropagationConfig {
                ticks: 100,
                input_duration: 2,
                decay_period: 6,
            },
            &wt,
            &mut scratch,
        );
        for &c in charge.iter() {
            assert!(c <= LIMIT_MAX_CHARGE, "charge out of bounds: {c}");
        }
    }

    #[test]
    fn wave_table_range_is_valid() {
        let table = build_wave_gating_table();
        for ch in 1..=GLOBAL_WAVE_CHANNEL_COUNT {
            for tick in 0..GLOBAL_WAVE_TICKS_PER_PERIOD {
                let v = table[ch][tick];
                assert!(v >= 600 && v <= 1400, "wave_table[{ch}][{tick}] = {v}");
            }
        }
        assert!(table[0].iter().all(|&v| v == 1000));
    }
}
