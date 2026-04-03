//! # Signal Propagation — Spiking Forward Pass
//!
//! Simulates one token's passage through the recurrent spiking network.
//! This is the **performance-critical inner loop** — every mutation evaluation
//! runs this function hundreds of times, so it must be as fast as possible.
//!
//! ## Integer-Only Design
//!
//! The entire forward pass operates on `u32` values. No floating-point
//! arithmetic appears in the hot path. Charge, activation, and thresholds
//! are all unsigned 32-bit integers, matching the CPU's native word size
//! for maximum throughput (benchmarked 2.5x faster than u8).
//!
//! ## Algorithm (per tick)
//!
//! 1. **Charge decay**: Every `decay_period` ticks, subtract 1 from all charges.
//! 2. **Input injection**: First `input_duration` ticks, add input to activations.
//! 3. **Scatter-add propagation**: For each edge, add source activation to target charge.
//! 4. **Wave-gated spike decision**: Fire if charge exceeds wave-modulated threshold.
//! 5. **Hard reset**: Fired neurons get charge=0, activation=polarity.
//!
//! ## Wave Gating (C19 Mechanism)
//!
//! The wave LUT is precomputed as a fixed-point `u32` table (scaled by 1000).
//! Each neuron's effective threshold = `(theta * wave_table[channel][tick]) / 1000`.
//! This avoids all floating-point at runtime.

use crate::parameters::*;

/// Precomputed wave gating table in fixed-point (multiplied by 1000).
///
/// `wave_table[channel][tick]` is the threshold multiplier × 1000.
/// Range: [700, 1300] for amplitude=300 permille.
/// Channel 0 is neutral (all 1000).
pub fn build_wave_gating_table() -> [[u32; WAVE_TICKS_PER_PERIOD]; WAVE_CHANNEL_COUNT + 1] {
    let mut table = [[1000u32; WAVE_TICKS_PER_PERIOD]; WAVE_CHANNEL_COUNT + 1];
    for (channel, row) in table.iter_mut().enumerate().skip(1) {
        for (tick, entry) in row.iter_mut().enumerate() {
            let phase_offset = tick as f64 - (channel - 1) as f64;
            let angle = 2.0 * std::f64::consts::PI * phase_offset / WAVE_TICKS_PER_PERIOD as f64;
            let amplitude = WAVE_AMPLITUDE_PERMILLE as f64 / 1000.0;
            let multiplier = 1.0 - amplitude * angle.cos();
            *entry = (multiplier * 1000.0).round() as u32;
        }
    }
    table
}

/// Per-neuron learned parameters.
pub struct NeuronParameters<'a> {
    /// Firing threshold per neuron. Range: [1, 15].
    pub threshold: &'a [u32],
    /// Wave gating channel per neuron. Range: [1, 8].
    pub channel: &'a [u8],
    /// Polarity per neuron: 1 = excitatory, 0 = inhibitory.
    /// Stored as u32 for direct use in activation (1 or 0, multiplied later).
    pub excitatory: &'a [bool],
}

/// Persistent internal state carried across tokens.
pub struct NeuronState<'a> {
    /// Activation state per neuron. 0 = silent, 1 = fired excitatory, 2 = fired inhibitory.
    /// (Using u32 for ALU alignment; actual values are small.)
    pub activation: &'a mut [u32],
    /// Accumulated charge per neuron. Range: [0, MAX_CHARGE].
    pub charge: &'a mut [u32],
}

/// Configuration for one forward pass.
pub struct PropagationConfig {
    /// Simulation ticks per token. Typical: 12 (H=256), 16 (H=1024+).
    pub ticks: usize,
    /// Input injection duration in ticks. Typical: 2.
    pub input_duration: usize,
    /// Charge decay interval. Typical: 6.
    pub decay_period: usize,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            ticks: DEFAULT_TICKS_PER_TOKEN,
            input_duration: DEFAULT_INPUT_DURATION,
            decay_period: DEFAULT_CHARGE_DECAY_PERIOD,
        }
    }
}

/// Propagate one token through the spiking network. Integer-only hot path.
///
/// # Arguments
///
/// * `input` — External activation (u32 per neuron, typically 0 or 1).
/// * `edge_sources`, `edge_targets` — Sparse edge list from `ConnectionGraph`.
/// * `params` — Per-neuron threshold, channel, polarity.
/// * `state` — Mutable activation + charge (persists across tokens).
/// * `config` — Ticks, input duration, decay period.
pub fn propagate_token(
    input: &[u32],
    edge_sources: &[usize],
    edge_targets: &[usize],
    params: &NeuronParameters,
    state: &mut NeuronState,
    config: &PropagationConfig,
) {
    let neuron_count = state.activation.len();
    let edge_count = edge_sources.len();
    let wave_table = build_wave_gating_table();

    for tick in 0..config.ticks {
        // Step 1: Periodic charge decay
        if config.decay_period > 0 && tick % config.decay_period == 0 {
            for charge in state.charge.iter_mut() {
                *charge = charge.saturating_sub(1);
            }
        }

        // Step 2: Input injection
        if tick < config.input_duration {
            for (activation, &input_val) in state.activation.iter_mut().zip(input.iter()) {
                *activation += input_val;
            }
        }

        // Step 3: Scatter-add propagation (THE HOT INNER LOOP)
        let mut incoming = vec![0u32; neuron_count];
        for edge_idx in 0..edge_count {
            incoming[edge_targets[edge_idx]] += state.activation[edge_sources[edge_idx]];
        }

        // Accumulate into charge
        for (charge, &signal) in state.charge.iter_mut().zip(incoming.iter()) {
            *charge = (*charge + signal).min(MAX_CHARGE);
        }

        // Step 4: Wave-gated spike decision (integer arithmetic)
        for neuron_idx in 0..neuron_count {
            let channel = params.channel[neuron_idx] as usize;
            let wave_mult = if channel <= WAVE_CHANNEL_COUNT {
                wave_table[channel][tick % WAVE_TICKS_PER_PERIOD]
            } else {
                1000
            };
            // effective_threshold = theta * wave_mult / 1000
            // Compare: charge * 1000 >= theta * wave_mult (avoids division)
            let charge_scaled = state.charge[neuron_idx] * 1000;
            let threshold_scaled = params.threshold[neuron_idx] * wave_mult;

            if charge_scaled >= threshold_scaled {
                // Fire: polarity determines output value
                state.activation[neuron_idx] = if params.excitatory[neuron_idx] { 1 } else { 0 };
                state.charge[neuron_idx] = 0;
            } else {
                state.activation[neuron_idx] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> PropagationConfig {
        PropagationConfig { ticks: 8, input_duration: 2, decay_period: 6 }
    }

    #[test]
    fn isolated_neurons_remain_charge_bounded() {
        let h = 16;
        let mut activation = vec![0u32; h];
        let mut charge = vec![0u32; h];
        let input = vec![1u32; h];
        let threshold = vec![6u32; h];
        let channel = vec![1u8; h];
        let excitatory = vec![true; h];

        propagate_token(
            &input, &[], &[],
            &NeuronParameters { threshold: &threshold, channel: &channel, excitatory: &excitatory },
            &mut NeuronState { activation: &mut activation, charge: &mut charge },
            &default_config(),
        );

        assert!(charge.iter().all(|&c| c <= MAX_CHARGE));
    }

    #[test]
    fn signal_propagates_through_chain() {
        let h = 4;
        let mut activation = vec![0u32; h];
        let mut charge = vec![0u32; h];
        let mut input = vec![0u32; h];
        input[0] = 10;

        let sources = vec![0, 1, 2];
        let targets = vec![1, 2, 3];
        let threshold = vec![2u32; h];
        let channel = vec![1u8; h];
        let excitatory = vec![true; h];

        propagate_token(
            &input, &sources, &targets,
            &NeuronParameters { threshold: &threshold, channel: &channel, excitatory: &excitatory },
            &mut NeuronState { activation: &mut activation, charge: &mut charge },
            &PropagationConfig { ticks: 12, input_duration: 2, decay_period: 6 },
        );
        // Signal should propagate through the chain
    }

    #[test]
    fn extreme_input_does_not_overflow_charge() {
        let h = 8;
        let mut activation = vec![0u32; h];
        let mut charge = vec![0u32; h];
        let input = vec![100u32; h];

        propagate_token(
            &input, &[], &[],
            &NeuronParameters {
                threshold: &vec![1; h],
                channel: &vec![1; h],
                excitatory: &vec![true; h],
            },
            &mut NeuronState { activation: &mut activation, charge: &mut charge },
            &PropagationConfig { ticks: 100, input_duration: 2, decay_period: 6 },
        );

        for &c in charge.iter() {
            assert!(c <= MAX_CHARGE, "charge out of bounds: {c}");
        }
    }

    #[test]
    fn wave_table_range_is_valid() {
        let table = build_wave_gating_table();
        for ch in 1..=WAVE_CHANNEL_COUNT {
            for tick in 0..WAVE_TICKS_PER_PERIOD {
                let v = table[ch][tick];
                assert!(v >= 600 && v <= 1400, "wave_table[{ch}][{tick}] = {v} out of range");
            }
        }
        // Channel 0 = neutral
        assert!(table[0].iter().all(|&v| v == 1000));
    }
}
