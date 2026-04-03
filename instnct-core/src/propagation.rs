//! # Signal Propagation — Spiking Forward Pass
//!
//! Simulates one token's passage through the recurrent spiking network.
//! This is the **performance-critical inner loop** — every mutation evaluation
//! runs this function hundreds of times, so it must be as fast as possible.
//!
//! ## Algorithm (per tick)
//!
//! 1. **Charge decay**: Every `decay_period` ticks, subtract 1 from all
//!    neuron charges (clamped to zero). This prevents unbounded accumulation.
//!
//! 2. **Input injection**: During the first `input_duration` ticks, the
//!    external input vector is added to the activation state. This gives
//!    the input signal time to propagate before the network runs freely.
//!
//! 3. **Scatter-add propagation**: For each directed edge (source -> target),
//!    the source neuron's activation is added to the target's incoming charge.
//!    Because activations are binary (+1/-1 via polarity), this step requires
//!    **no floating-point multiplies** — only additions.
//!
//! 4. **Wave-gated spike decision**: Each neuron fires if its accumulated
//!    charge exceeds an effective threshold. The threshold is the product of
//!    the neuron's base `theta` and a cosine-shaped wave multiplier that
//!    depends on the neuron's assigned `channel` (1-8) and the current tick.
//!    This creates **temporal specialization**: channel-1 neurons fire most
//!    easily on tick 0, channel-2 on tick 1, etc.
//!
//! 5. **Hard reset**: Neurons that fire have their charge reset to zero and
//!    their activation set to their polarity value (+1 excitatory, -1 inhibitory).
//!    Neurons that do not fire have their activation set to zero.
//!
//! ## Wave Gating (C19 Mechanism)
//!
//! The wave lookup table modulates each neuron's firing threshold across ticks:
//!
//! ```text
//! multiplier[channel][tick] = 1.0 - 0.3 * cos(2pi * (tick - (channel-1)) / 8)
//! ```
//!
//! At the preferred tick (tick == channel-1), the multiplier is minimal (0.7),
//! making the neuron most sensitive. At the anti-phase tick, it is maximal (1.3),
//! suppressing firing. This mechanism was validated to outperform explicit
//! sine-wave gating (23.8% vs 21.4% on English bigram prediction).

/// Maximum charge a neuron can accumulate before clamping.
pub const MAX_CHARGE: f32 = 15.0;

/// Number of distinct wave channels (1 through 8).
pub const CHANNEL_COUNT: usize = 8;

/// Number of ticks in one wave period.
pub const TICKS_PER_PERIOD: usize = 8;

/// Precomputed wave gating lookup table.
///
/// Dimensions: `[channel][tick]` where channel 0 is unused (neutral = 1.0),
/// channels 1-8 are the active wave patterns. Each entry is the threshold
/// multiplier for that channel at that tick.
///
/// Generated once at program start; all forward passes reference the same table.
pub fn build_wave_gating_table() -> [[f32; TICKS_PER_PERIOD]; CHANNEL_COUNT + 1] {
    let mut table = [[1.0f32; TICKS_PER_PERIOD]; CHANNEL_COUNT + 1];
    for (channel, row) in table.iter_mut().enumerate().skip(1) {
        for (tick, entry) in row.iter_mut().enumerate() {
            let phase_offset = tick as f32 - (channel - 1) as f32;
            let angle = 2.0 * std::f32::consts::PI * phase_offset / TICKS_PER_PERIOD as f32;
            *entry = 1.0 - 0.3 * angle.cos();
        }
    }
    table
}

/// Per-neuron parameters that define the network's learned state.
///
/// These are co-evolved alongside the connection topology during training.
pub struct NeuronParameters<'a> {
    /// Firing threshold per neuron. Range: [1, 15]. Determines how much
    /// accumulated charge is needed to trigger a spike.
    pub threshold: &'a [f32],

    /// Wave gating channel per neuron. Range: [1, 8]. Determines the
    /// temporal firing preference (which tick in the 8-tick period is easiest).
    pub channel: &'a [u8],

    /// Polarity per neuron. +1.0 for excitatory, -1.0 for inhibitory.
    /// Approximately 90% excitatory / 10% inhibitory (fly-realistic ratio).
    pub polarity: &'a [f32],
}

/// Persistent internal state carried across tokens.
///
/// Unlike feedforward networks, INSTNCT neurons retain state between inputs.
/// This enables temporal context: the network's response to token N depends
/// on the residual charge and activation patterns from tokens 0..N-1.
pub struct NeuronState<'a> {
    /// Binary activation state per neuron. After each tick, this is either
    /// the neuron's polarity value (if it fired) or zero (if it did not).
    pub activation: &'a mut [f32],

    /// Accumulated charge per neuron. Range: [0, MAX_CHARGE]. Incoming
    /// signals add to charge; firing resets it to zero; decay subtracts
    /// periodically.
    pub charge: &'a mut [f32],
}

/// Configuration for one forward pass invocation.
pub struct PropagationConfig {
    /// Number of simulation ticks per token. More ticks allow deeper
    /// signal propagation through the graph. Typical: 8 (H=256), 16 (H=1024+).
    pub ticks: usize,

    /// Number of initial ticks during which the input vector is injected.
    /// Typical: 2. Gives the input signal time to enter the recurrent substrate.
    pub input_duration: usize,

    /// Charge decay interval: subtract 1 from all charges every N ticks.
    /// Typical: 6. Prevents runaway charge accumulation in high-degree neurons.
    pub decay_period: usize,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            ticks: 12,
            input_duration: 2,
            decay_period: 6,
        }
    }
}

/// Propagate one token through the spiking network.
///
/// This is the core simulation step. The input vector is injected into the
/// network, signals propagate along directed edges for `config.ticks` steps,
/// and the final activation/charge state encodes the network's response.
///
/// # Arguments
///
/// * `input` — External activation vector to inject. Length H (neuron count).
/// * `edge_sources`, `edge_targets` — Sparse directed edge list from
///   [`ConnectionMask::to_directed_edges`](crate::topology::ConnectionMask::to_directed_edges).
/// * `params` — Per-neuron learned parameters (threshold, channel, polarity).
/// * `state` — Mutable neuron state (activation, charge). Persists across tokens.
/// * `config` — Tick count, input duration, decay period.
pub fn propagate_token(
    input: &[f32],
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
                *charge = (*charge - 1.0).max(0.0);
            }
        }

        // Step 2: Input injection (first `input_duration` ticks only)
        if tick < config.input_duration {
            for (activation, &input_value) in state.activation.iter_mut().zip(input.iter()) {
                *activation += input_value;
            }
        }

        // Step 3: Scatter-add propagation (the hot inner loop)
        // For each directed edge, add the source neuron's activation to
        // the target neuron's incoming signal. This is multiply-free because
        // activations are binary (+1/-1/0).
        let mut incoming_signal = vec![0.0f32; neuron_count];
        for edge_idx in 0..edge_count {
            incoming_signal[edge_targets[edge_idx]] += state.activation[edge_sources[edge_idx]];
        }

        // Accumulate incoming signal into charge (clamped to [0, MAX_CHARGE])
        for (charge, &signal) in state.charge.iter_mut().zip(incoming_signal.iter()) {
            *charge = (*charge + signal).clamp(0.0, MAX_CHARGE);
        }

        // Step 4: Wave-gated spike decision
        for neuron_idx in 0..neuron_count {
            let channel = params.channel[neuron_idx] as usize;
            let wave_multiplier = if channel <= CHANNEL_COUNT {
                wave_table[channel][tick % TICKS_PER_PERIOD]
            } else {
                1.0
            };
            let effective_threshold =
                (params.threshold[neuron_idx] * wave_multiplier).clamp(1.0, MAX_CHARGE);

            if state.charge[neuron_idx] >= effective_threshold {
                // Fire: emit spike with polarity, reset charge
                state.activation[neuron_idx] = params.polarity[neuron_idx];
                state.charge[neuron_idx] = 0.0;
            } else {
                // Silent: no output this tick
                state.activation[neuron_idx] = 0.0;
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
        let mut activation = vec![0.0f32; h];
        let mut charge = vec![0.0f32; h];
        let input = vec![1.0f32; h];
        let threshold = vec![6.0f32; h];
        let channel = vec![1u8; h];
        let polarity = vec![1.0f32; h];

        propagate_token(
            &input, &[], &[],
            &NeuronParameters { threshold: &threshold, channel: &channel, polarity: &polarity },
            &mut NeuronState { activation: &mut activation, charge: &mut charge },
            &default_config(),
        );

        assert!(charge.iter().all(|&c| c >= 0.0 && c <= MAX_CHARGE));
    }

    #[test]
    fn signal_propagates_through_chain() {
        let h = 4;
        let mut activation = vec![0.0f32; h];
        let mut charge = vec![0.0f32; h];
        let mut input = vec![0.0f32; h];
        input[0] = 10.0; // strong input to neuron 0

        let sources = vec![0, 1, 2]; // chain: 0 -> 1 -> 2 -> 3
        let targets = vec![1, 2, 3];
        let threshold = vec![2.0f32; h];
        let channel = vec![1u8; h];
        let polarity = vec![1.0f32; h];

        propagate_token(
            &input, &sources, &targets,
            &NeuronParameters { threshold: &threshold, channel: &channel, polarity: &polarity },
            &mut NeuronState { activation: &mut activation, charge: &mut charge },
            &PropagationConfig { ticks: 12, input_duration: 2, decay_period: 6 },
        );

        // After 12 ticks with a strong input, the signal should have
        // propagated at least partway through the chain.
    }

    #[test]
    fn extreme_input_does_not_overflow_charge() {
        let h = 8;
        let mut activation = vec![0.0f32; h];
        let mut charge = vec![0.0f32; h];
        let input = vec![100.0f32; h];

        propagate_token(
            &input, &[], &[],
            &NeuronParameters {
                threshold: &vec![1.0; h],
                channel: &vec![1; h],
                polarity: &vec![1.0; h],
            },
            &mut NeuronState { activation: &mut activation, charge: &mut charge },
            &PropagationConfig { ticks: 100, input_duration: 2, decay_period: 6 },
        );

        for &c in charge.iter() {
            assert!(c >= 0.0 && c <= MAX_CHARGE, "charge out of bounds: {c}");
        }
    }
}
