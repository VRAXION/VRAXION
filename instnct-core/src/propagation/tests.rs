use super::*;
use crate::topology::DirectedEdge;

fn default_config() -> PropagationConfig {
    PropagationConfig {
        ticks_per_token: 8,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
        use_refractory: false,
    }
}

fn graph_with_edges(neuron_count: usize, pairs: &[(u16, u16)]) -> ConnectionGraph {
    ConnectionGraph::from_pairs(neuron_count, pairs)
}

#[test]
fn isolated_neurons_remain_charge_bounded() {
    let neuron_count = 16;
    let graph = ConnectionGraph::new(neuron_count);
    let mut activation = vec![0i8; neuron_count];
    let mut charge = vec![0u8; neuron_count];
    let mut refractory = vec![0u8; neuron_count];
    let input = vec![1i32; neuron_count];
    let threshold = vec![6u8; neuron_count];
    let channel = vec![1u8; neuron_count];
    let polarity = vec![1i8; neuron_count];
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
            refractory: &mut refractory,
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
    let mut activation = vec![0i8; neuron_count];
    let mut charge = vec![0u8; neuron_count];
    let mut refractory = vec![0u8; neuron_count];
    let mut input = vec![0i32; neuron_count];
    input[0] = 10;
    let threshold = vec![1u8; neuron_count];
    let channel = vec![1u8; neuron_count];
    let polarity = vec![1i8; neuron_count];
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
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 3,
            input_duration_ticks: 2,
            decay_interval_ticks: 100,
            use_refractory: false,
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
    let mut activation = vec![0i8; neuron_count];
    let mut charge = vec![0u8; neuron_count];
    let mut refractory = vec![0u8; neuron_count];
    let mut input = vec![0i32; neuron_count];
    input[0] = 10;
    let threshold = vec![2u8; neuron_count];
    let channel = vec![1u8; neuron_count];
    let polarity = vec![-1i8, 1, 1];
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
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 4,
            input_duration_ticks: 2,
            decay_interval_ticks: 100,
            use_refractory: false,
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
    let mut activation = vec![0i8; neuron_count];
    let mut charge = vec![0u8; neuron_count];
    let mut refractory = vec![0u8; neuron_count];
    let input = vec![100i32; neuron_count];
    let threshold = vec![1u8; neuron_count];
    let channel = vec![1u8; neuron_count];
    let polarity = vec![1i8; neuron_count];
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
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 100,
            input_duration_ticks: 2,
            decay_interval_ticks: 6,
            use_refractory: false,
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
    let threshold = vec![2u8; neuron_count];
    let channel = vec![1u8; neuron_count];
    let polarity = vec![1i8; neuron_count];
    let config = PropagationConfig {
        ticks_per_token: 4,
        input_duration_ticks: 2,
        decay_interval_ticks: 100,
        use_refractory: false,
    };
    let mut workspace = PropagationWorkspace::new(neuron_count);

    let mut activation_a = vec![0i8; neuron_count];
    let mut charge_a = vec![0u8; neuron_count];
    let mut refractory_a = vec![0u8; neuron_count];
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
            refractory: &mut refractory_a,
        },
        &config,
        &mut workspace,
    )
    .unwrap();

    let mut activation_b = vec![0i8; neuron_count];
    let mut charge_b = vec![0u8; neuron_count];
    let mut refractory_b = vec![0u8; neuron_count];
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
            refractory: &mut refractory_b,
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
    let mut activation = vec![0i8; 3];
    let mut charge = vec![0u8; 4];
    let mut refractory = vec![0u8; 4];
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
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 1,
            decay_interval_ticks: 0,
            use_refractory: false,
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
    let mut activation = vec![0i8; 4];
    let mut charge = vec![0u8; 4];
    let mut refractory = vec![0u8; 4];
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
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 1,
            decay_interval_ticks: 0,
            use_refractory: false,
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
    let mut activation = vec![0i8; 4];
    let mut charge = vec![0u8; 3];
    let mut refractory = vec![0u8; 4];
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
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 1,
            decay_interval_ticks: 0,
            use_refractory: false,
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
    let mut activation = vec![0i8; 4];
    let mut charge = vec![0u8; 4];
    let mut refractory = vec![0u8; 4];
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
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 1,
            decay_interval_ticks: 0,
            use_refractory: false,
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
    let mut activation = vec![0i8; 4];
    let mut charge = vec![0u8; 4];
    let mut refractory = vec![0u8; 4];
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
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 1,
            decay_interval_ticks: 0,
            use_refractory: false,
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
    let mut activation = vec![0i8; 4];
    let mut charge = vec![0u8; 4];
    let mut refractory = vec![0u8; 4];
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
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 1,
            decay_interval_ticks: 0,
            use_refractory: false,
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
        vec![0u16, 1],
        vec![1u16],
    );
    let mut activation = vec![0i8; 4];
    let mut charge = vec![0u8; 4];
    let mut refractory = vec![0u8; 4];
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
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 1,
            decay_interval_ticks: 0,
            use_refractory: false,
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
        vec![4u16],
        vec![1u16],
    );
    let mut activation = vec![0i8; 4];
    let mut charge = vec![0u8; 4];
    let mut refractory = vec![0u8; 4];
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
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 1,
            decay_interval_ticks: 0,
            use_refractory: false,
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
        vec![0u16],
        vec![4u16],
    );
    let mut activation = vec![0i8; 4];
    let mut charge = vec![0u8; 4];
    let mut refractory = vec![0u8; 4];
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
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 1,
            decay_interval_ticks: 0,
            use_refractory: false,
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
    let mut activation = vec![0i8; 4];
    let mut charge = vec![0u8; 4];
    let mut refractory = vec![0u8; 4];
    let mut workspace = PropagationWorkspace::with_scratch(vec![0; 3]);

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
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 1,
            decay_interval_ticks: 0,
            use_refractory: false,
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
fn phase_base_values_are_valid() {
    // PHASE_BASE must contain exactly the x10 cosine values
    assert_eq!(PHASE_BASE, [7, 8, 10, 12, 13, 12, 10, 8]);
    // All values in valid range
    for &v in &PHASE_BASE {
        assert!((7..=13).contains(&v), "PHASE_BASE value out of range: {v}");
    }
    // Circular symmetry: PHASE_BASE[i] == PHASE_BASE[(8-i) % 8]
    for i in 0..8 {
        assert_eq!(
            PHASE_BASE[i],
            PHASE_BASE[(8 - i) % 8],
            "PHASE_BASE not circularly symmetric at {i}"
        );
    }
}

#[test]
fn phase_rotation_peaks_at_correct_tick() {
    // Channel N should have easiest firing (lowest multiplier) at tick N-1
    for ch in 1..=GLOBAL_PHASE_CHANNEL_COUNT {
        let mut min_val = u8::MAX;
        let mut min_tick = 0;
        for tick in 0..GLOBAL_PHASE_TICKS_PER_PERIOD {
            let val = PHASE_BASE[(tick + 9 - ch) & 7];
            if val < min_val {
                min_val = val;
                min_tick = tick;
            }
        }
        assert_eq!(
            min_tick,
            ch - 1,
            "channel {ch} peaks at tick {min_tick}, expected {}",
            ch - 1
        );
        assert_eq!(min_val, 7, "peak value should be 7 (0.7x threshold)");
    }
}

#[test]
fn threshold_out_of_range_returns_error() {
    let graph = ConnectionGraph::new(4);
    let mut activation = vec![0i8; 4];
    let mut charge = vec![0u8; 4];
    let mut refractory = vec![0u8; 4];
    let mut workspace = PropagationWorkspace::new(4);

    let err = propagate_token(
        &[1; 4],
        &graph,
        &PropagationParameters {
            threshold: &[1, 1, 16, 1],
            channel: &[1; 4],
            polarity: &[1; 4],
        },
        &mut PropagationState {
            activation: &mut activation,
            charge: &mut charge,
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 1,
            decay_interval_ticks: 0,
            use_refractory: false,
        },
        &mut workspace,
    )
    .unwrap_err();

    assert_eq!(
        err,
        PropagationError::ThresholdOutOfRange {
            index: 2,
            value: 16
        }
    );
}

#[test]
fn channel_out_of_range_returns_error() {
    let graph = ConnectionGraph::new(4);
    let mut activation = vec![0i8; 4];
    let mut charge = vec![0u8; 4];
    let mut refractory = vec![0u8; 4];
    let mut workspace = PropagationWorkspace::new(4);

    let err = propagate_token(
        &[1; 4],
        &graph,
        &PropagationParameters {
            threshold: &[1; 4],
            channel: &[1, 1, 0, 1],
            polarity: &[1; 4],
        },
        &mut PropagationState {
            activation: &mut activation,
            charge: &mut charge,
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 1,
            decay_interval_ticks: 0,
            use_refractory: false,
        },
        &mut workspace,
    )
    .unwrap_err();

    assert_eq!(
        err,
        PropagationError::ChannelOutOfRange { index: 2, value: 0 }
    );
}

#[test]
fn polarity_out_of_range_returns_error() {
    let graph = ConnectionGraph::new(4);
    let mut activation = vec![0i8; 4];
    let mut charge = vec![0u8; 4];
    let mut refractory = vec![0u8; 4];
    let mut workspace = PropagationWorkspace::new(4);

    let err = propagate_token(
        &[1; 4],
        &graph,
        &PropagationParameters {
            threshold: &[1; 4],
            channel: &[1; 4],
            polarity: &[1, 1, 2, 1],
        },
        &mut PropagationState {
            activation: &mut activation,
            charge: &mut charge,
            refractory: &mut refractory,
        },
        &PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 1,
            decay_interval_ticks: 0,
            use_refractory: false,
        },
        &mut workspace,
    )
    .unwrap_err();

    assert_eq!(
        err,
        PropagationError::PolarityOutOfRange { index: 2, value: 2 }
    );
}
