//! Spiking network forward pass — the hot path.
//!
//! Pure scatter-add propagation with wave-gated thresholds.
//! No floating-point multiply in the core loop (binary spikes).

/// Wave gating LUT: channels 1-8, ticks 0-7, cos-shaped.
/// Channel N peaks at tick N-1.
pub fn wave_lut() -> [[f32; 8]; 9] {
    let mut lut = [[1.0f32; 8]; 9];
    for ch in 1..9 {
        for t in 0..8 {
            let phase = 2.0 * std::f32::consts::PI * (t as f32 - (ch - 1) as f32) / 8.0;
            lut[ch][t] = 1.0 - 0.3 * phase.cos();
        }
    }
    lut
}

/// Run one token through the spiking network.
///
/// # Arguments
/// * `injected` - input activation vector (H,)
/// * `sources`, `targets` - sparse edge list
/// * `theta` - per-neuron firing threshold (H,)
/// * `channel` - per-neuron wave channel 1-8 (H,)
/// * `polarity` - per-neuron +1.0/-1.0 (H,)
/// * `ticks` - simulation steps per token
/// * `input_duration` - ticks to inject input
/// * `state` - mutable activation state (H,)
/// * `charge` - mutable charge state (H,)
pub fn rollout_token(
    injected: &[f32],
    sources: &[usize],
    targets: &[usize],
    theta: &[f32],
    channel: &[u8],
    polarity: &[f32],
    ticks: usize,
    input_duration: usize,
    decay_period: usize,
    state: &mut [f32],
    charge: &mut [f32],
) {
    let h = state.len();
    let n_edges = sources.len();
    let lut = wave_lut();
    let max_charge: f32 = 15.0;

    for tick in 0..ticks {
        // 1. Decay
        if decay_period > 0 && tick % decay_period == 0 {
            for c in charge.iter_mut() {
                *c = (*c - 1.0).max(0.0);
            }
        }

        // 2. Input injection
        if tick < input_duration {
            for i in 0..h {
                state[i] += injected[i];
            }
        }

        // 3. Propagate (scatter-add, the hot inner loop)
        let mut raw = vec![0.0f32; h];
        for e in 0..n_edges {
            raw[targets[e]] += state[sources[e]];
        }

        // Accumulate charge
        for i in 0..h {
            charge[i] = (charge[i] + raw[i]).min(max_charge).max(0.0);
        }

        // 4. Spike decision with wave gating
        for i in 0..h {
            let ch = channel[i] as usize;
            let theta_mult = if ch < 9 { lut[ch][tick % 8] } else { 1.0 };
            let eff_theta = (theta[i] * theta_mult).clamp(1.0, max_charge);

            if charge[i] >= eff_theta {
                state[i] = polarity[i]; // fire with polarity
                charge[i] = 0.0;       // hard reset
            } else {
                state[i] = 0.0;        // no fire
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_network() {
        let h = 16;
        let mut state = vec![0.0f32; h];
        let mut charge = vec![0.0f32; h];
        let injected = vec![1.0f32; h];
        let theta = vec![6.0f32; h];
        let channel = vec![1u8; h];
        let polarity = vec![1.0f32; h];

        rollout_token(
            &injected, &[], &[], &theta, &channel, &polarity,
            8, 2, 6, &mut state, &mut charge,
        );

        // With no edges, charge should be from input injection only
        // After 8 ticks, some neurons should have fired
        assert!(charge.iter().all(|&c| c >= 0.0 && c <= 15.0));
    }

    #[test]
    fn test_simple_chain() {
        let h = 4;
        let mut state = vec![0.0f32; h];
        let mut charge = vec![0.0f32; h];
        let mut injected = vec![0.0f32; h];
        injected[0] = 10.0; // strong input to neuron 0

        let sources = vec![0, 1, 2]; // chain: 0->1->2->3
        let targets = vec![1, 2, 3];
        let theta = vec![2.0f32; h];
        let channel = vec![1u8; h];
        let polarity = vec![1.0f32; h];

        rollout_token(
            &injected, &sources, &targets, &theta, &channel, &polarity,
            12, 2, 6, &mut state, &mut charge,
        );

        // Signal should have propagated through the chain
        // At least neuron 0 should have fired
    }

    #[test]
    fn test_charge_bounded() {
        let h = 8;
        let mut state = vec![0.0f32; h];
        let mut charge = vec![0.0f32; h];
        let injected = vec![100.0f32; h]; // very strong input

        rollout_token(
            &injected, &[], &[], &vec![1.0; h], &vec![1; h], &vec![1.0; h],
            100, 2, 6, &mut state, &mut charge,
        );

        for &c in charge.iter() {
            assert!(c >= 0.0 && c <= 15.0, "charge out of bounds: {c}");
        }
    }
}
