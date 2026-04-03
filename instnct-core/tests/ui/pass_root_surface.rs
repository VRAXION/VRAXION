use instnct_core::{
    propagate_token, ConnectionGraph, PropagationConfig, PropagationParameters,
    PropagationState, PropagationWorkspace,
};

fn main() -> Result<(), instnct_core::PropagationError> {
    let mut graph = ConnectionGraph::new(2);
    assert!(graph.add_edge(0, 1));

    let input = [4, 0];
    let threshold = [1, 1];
    let channel = [1, 1];
    let polarity = [1, 1];
    let mut activation = [0, 0];
    let mut charge = [0, 0];
    let mut workspace = PropagationWorkspace::new(2);

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
            ticks: 2,
            input_duration: 1,
            decay_period: 0,
        },
        &mut workspace,
    )?;

    Ok(())
}
