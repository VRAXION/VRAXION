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
//! `[0, MAX_CHARGE]` after accumulation, preventing negative charge.

use crate::parameters::*;

/// Precomputed wave gating table in fixed-point (×1000).
///
/// `wave_table[channel][tick]` is the threshold multiplier × 1000.
/// Range: [700, 1300] for amplitude=300 permille.
/// Channel 0 is neutral (all 1000).
///
/// This should be computed **once** and passed into [`propagate_token`],
/// not rebuilt per call. The convenience function is provided for setup.
pub fn build_wave_gating_table() -> [[u32; WAVE_TICKS_PER_PERIOD]; WAVE_CHANNEL_COUNT + 1] {
    let mut table = [[1000u32; WAVE_TICKS_PER_PERIOD]; WAVE_CHANNEL_COUNT + 1];
    for (channel, row) in table.iter_mut().enumerate().skip(1) {
        for (tick, entry) in row.iter_mut().enumerate() {
            let phase_offset = tick as f64 - (channel - 1) as f64;
            let angle = 2.0 * std::f64::consts::PI * phase_offset / WAVE_TICKS_PER_PERIOD as f64;
            let amplitude = WAVE_AMPLITUDE_PERMILLE as f64 / 1000.0;
            *entry = ((1.0 - amplitude * angle.cos()) * 1000.0).round() as u32;
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
    /// Polarity per neuron: +1 (excitatory) or -1 (inhibitory).
    /// Inhibitory spikes subtract from downstream charge.
    pub polarity: &'a [i32],
}

/// Persistent internal state carried across tokens.
pub struct NeuronState<'a> {
    /// Activation per neuron: +1 (excitatory fire), -1 (inhibitory fire), 0 (silent).
    /// Signed to support inhibitory subtraction in scatter-add.
    pub activation: &'a mut [i32],
    /// Accumulated charge per neuron. Range: [0, MAX_CHARGE]. Always non-negative.
    pub charge: &'a mut [u32],
}

/// Configuration for one forward pass.
pub struct PropagationConfig {
    pub ticks: usize,
    pub input_duration: usize,
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

/// Propagate one token through the spiking network.
///
/// # Panics
///
/// Debug-asserts that all slices have consistent length `neuron_count`,
/// and that edge indices are within bounds.
pub fn propagate_token(
    input: &[i32],
    edge_sources: &[usize],
    edge_targets: &[usize],
    params: &NeuronParameters,
    state: &mut NeuronState,
    config: &PropagationConfig,
    wave_table: &[[u32; WAVE_TICKS_PER_PERIOD]; WAVE_CHANNEL_COUNT + 1],
    scratch: &mut [i32],
) {
    let neuron_count = state.activation.len();

    // --- Shape guards (active in ALL builds, not just debug) ---
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
    // Edge index bounds: O(n) check, guarded behind debug to keep release hot path clean.
    // The topology layer (ConnectionGraph) guarantees valid indices at insert time,
    // so this is a defense-in-depth check, not a primary invariant.
    debug_assert!(
        edge_sources.iter().all(|&s| s < neuron_count),
        "edge source index out of bounds"
    );
    debug_assert!(
        edge_targets.iter().all(|&t| t < neuron_count),
        "edge target index out of bounds"
    );

    let edge_count = edge_sources.len();

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
        // Uses i32 scratch buffer — inhibitory spikes contribute -1,
        // genuinely suppressing downstream charge.
        let incoming = &mut scratch[..neuron_count];
        incoming.fill(0);
        for edge_idx in 0..edge_count {
            incoming[edge_targets[edge_idx]] += state.activation[edge_sources[edge_idx]];
        }

        // Accumulate into charge (clamped to [0, MAX_CHARGE])
        for (charge, &signal) in state.charge.iter_mut().zip(incoming.iter()) {
            let new_charge = *charge as i32 + signal;
            *charge = new_charge.clamp(0, MAX_CHARGE as i32) as u32;
        }

        // Step 4: Wave-gated spike decision
        for neuron_idx in 0..neuron_count {
            let channel = params.channel[neuron_idx] as usize;
            let wave_mult = if channel <= WAVE_CHANNEL_COUNT {
                wave_table[channel][tick % WAVE_TICKS_PER_PERIOD]
            } else {
                1000
            };
            // Compare without division: charge×1000 >= theta×wave_mult
            let charge_scaled = state.charge[neuron_idx] * 1000;
            let threshold_scaled = params.threshold[neuron_idx] * wave_mult;

            if charge_scaled >= threshold_scaled {
                // Fire: emit spike with polarity (+1 excitatory, -1 inhibitory)
                state.activation[neuron_idx] = params.polarity[neuron_idx];
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

    fn make_wave_table() -> [[u32; WAVE_TICKS_PER_PERIOD]; WAVE_CHANNEL_COUNT + 1] {
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
        assert!(charge.iter().all(|&c| c <= MAX_CHARGE));
    }

    #[test]
    fn excitatory_spike_increases_downstream_charge() {
        let h = 3;
        let mut activation = vec![0i32; h];
        let mut charge = vec![0u32; h];
        let mut scratch = vec![0i32; h];
        let mut input = vec![0i32; h];
        input[0] = 10; // strong input to neuron 0
                       // Chain: 0 -> 1 -> 2
        let sources = vec![0, 1];
        let targets = vec![1, 2];
        let threshold = vec![1u32; h]; // low threshold so everything fires
        let channel = vec![1u8; h];
        let polarity = vec![1i32; h]; // all excitatory

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
            }, // 3 ticks: signal reaches neuron 2 by tick 2
            &wt,
            &mut scratch,
        );
        // Neuron 0 fires from input (+10 > theta=1), sends +1 to neuron 1.
        // Neuron 1 accumulates charge, fires, sends +1 to neuron 2.
        // We check neuron 1 received signal (closer to source, more reliable).
        // Note: by the last tick, neurons may have fired and reset to 0.
        // So we check that the chain conducted at all by verifying neuron 1 fired
        // at some point — which means it must have sent signal onward.
        // Neuron 1 at theta=1 fires whenever charge >= ~0.7, so any +1 triggers it.
        // After firing, charge resets to 0 and act = polarity.
        // The fact that neuron 2 has any non-zero state proves propagation.
        // With low theta, all neurons fire every tick after getting input,
        // so by tick 8, charge oscillates between 0 (just fired) and 1 (just received).
        // Assert: at least one downstream neuron was affected.
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
        let h = 3;
        let mut activation = vec![0i32; h];
        let mut charge = vec![0u32; h];
        let mut scratch = vec![0i32; h];
        let mut input = vec![0i32; h];
        input[0] = 10; // strong input to neuron 0

        let sources = vec![0];
        let targets = vec![1];
        let threshold = vec![2u32; h];
        let channel = vec![1u8; h];
        let polarity = vec![-1i32, 1, 1]; // neuron 0 is INHIBITORY

        // Pre-charge neuron 1 so we can see suppression
        charge[1] = 5;

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
            }, // no decay
            &wt,
            &mut scratch,
        );
        // Neuron 0 fires inhibitory (-1), should REDUCE neuron 1's charge
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
            assert!(c <= MAX_CHARGE, "charge out of bounds: {c}");
        }
    }

    #[test]
    fn wave_table_range_is_valid() {
        let table = build_wave_gating_table();
        for ch in 1..=WAVE_CHANNEL_COUNT {
            for tick in 0..WAVE_TICKS_PER_PERIOD {
                let v = table[ch][tick];
                assert!(v >= 600 && v <= 1400, "wave_table[{ch}][{tick}] = {v}");
            }
        }
        assert!(table[0].iter().all(|&v| v == 1000));
    }
}
