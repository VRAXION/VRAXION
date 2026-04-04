use instnct_core::{
    propagate_token, ConnectionGraph, PropagationConfig, PropagationError, PropagationParameters,
    PropagationState, PropagationWorkspace,
};

#[test]
fn root_api_propagates_through_graph() {
    let mut graph = ConnectionGraph::new(3);
    assert!(graph.add_edge(0, 1));
    assert!(graph.add_edge(1, 2));

    let input = [8, 0, 0];
    let threshold = [1, 1, 1];
    let channel = [1, 1, 1];
    let polarity = [1, 1, 1];
    let mut activation = [0, 0, 0];
    let mut charge = [0, 0, 0];
    let mut workspace = PropagationWorkspace::new(3);

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
            ticks_per_token: 3,
            input_duration_ticks: 2,
            decay_interval_ticks: 100,
        },
        &mut workspace,
    )
    .unwrap();

    assert!(charge[1] > 0 || charge[2] > 0 || activation[1] != 0 || activation[2] != 0);
}

#[test]
fn root_api_rejects_malformed_input() {
    let graph = ConnectionGraph::new(4);
    let threshold = [1, 1, 1, 1];
    let channel = [1, 1, 1, 1];
    let polarity = [1, 1, 1, 1];
    let mut activation = [0, 0, 0, 0];
    let mut charge = [0, 0, 0, 0];
    let mut workspace = PropagationWorkspace::new(4);

    let err = propagate_token(
        &[1, 1],
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
        &PropagationConfig::default(),
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
