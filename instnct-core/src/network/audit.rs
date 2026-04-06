use super::*;
use crate::propagation::{propagate_token, PropagationParameters, PropagationState};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone, PartialEq, Eq)]
struct SemanticState {
    edges: Vec<(u16, u16)>,
    threshold: Vec<u32>,
    channel: Vec<u8>,
    polarity: Vec<i32>,
    activation: Vec<i32>,
    charge: Vec<u32>,
}

fn capture_semantic(net: &Network) -> SemanticState {
    let mut edges: Vec<_> = net
        .graph
        .iter_edges()
        .map(|edge| (edge.source, edge.target))
        .collect();
    edges.sort_unstable();

    SemanticState {
        edges,
        threshold: net.threshold.clone(),
        channel: net.channel.clone(),
        polarity: net.polarity.clone(),
        activation: net.activation.clone(),
        charge: net.charge.clone(),
    }
}

fn capture_csr_rows(net: &mut Network) -> Vec<Vec<u16>> {
    if net.csr_dirty {
        net.rebuild_csr();
    }

    let mut rows = vec![Vec::new(); net.graph.neuron_count()];
    for (neuron_idx, row) in rows.iter_mut().enumerate() {
        let start = net.csr_offsets[neuron_idx] as usize;
        let end = net.csr_offsets[neuron_idx + 1] as usize;
        row.extend_from_slice(&net.csr_targets[start..end]);
        row.sort_unstable();
    }
    rows
}

fn capture_graph_rows(net: &Network) -> Vec<Vec<u16>> {
    let mut rows = vec![Vec::new(); net.graph.neuron_count()];
    for edge in net.graph.iter_edges() {
        rows[edge.source as usize].push(edge.target);
    }
    for row in &mut rows {
        row.sort_unstable();
    }
    rows
}

fn apply_random_mutation(net: &mut Network, rng: &mut StdRng) -> bool {
    match rng.gen_range(0..9u32) {
        0 => net.mutate_add_edge(rng),
        1 => net.mutate_remove_edge(rng),
        2 => net.mutate_rewire(rng),
        3 => net.mutate_reverse(rng),
        4 => net.mutate_mirror(rng),
        5 => net.mutate_enhance(rng),
        6 => net.mutate_add_affinity(rng),
        7 => net.mutate_theta(rng),
        _ => {
            if rng.gen_bool(0.5) {
                net.mutate_channel(rng)
            } else {
                net.mutate_polarity(rng)
            }
        }
    }
}

fn random_input(rng: &mut StdRng, neuron_count: usize) -> Vec<i32> {
    (0..neuron_count).map(|_| rng.gen_range(-1..=1)).collect()
}

fn random_config(rng: &mut StdRng) -> PropagationConfig {
    let ticks_per_token = rng.gen_range(1..=8usize);
    PropagationConfig {
        ticks_per_token,
        input_duration_ticks: rng.gen_range(0..=ticks_per_token),
        decay_interval_ticks: rng.gen_range(0..=8usize),
        use_refractory: false,
    }
}

fn random_network(rng: &mut StdRng, neuron_count: usize) -> Network {
    let mut net = Network::new(neuron_count);

    for _ in 0..(neuron_count * neuron_count).max(1) {
        let _ = net.mutate_add_edge(rng);
    }

    for idx in 0..neuron_count {
        net.threshold[idx] = rng.gen_range(0..=15);
        net.channel[idx] = rng.gen_range(1..=8);
        net.polarity[idx] = if rng.gen_bool(0.25) { -1 } else { 1 };
        net.activation[idx] = rng.gen_range(-1..=1);
        net.charge[idx] = rng.gen_range(0..=LIMIT_MAX_CHARGE);
    }

    net
}

fn assert_network_matches_shared_path(mut net: Network, input: &[i32], config: &PropagationConfig) {
    let mut reference = net.clone();

    net.propagate(input, config).unwrap();

    propagate_token(
        input,
        &reference.graph,
        &PropagationParameters {
            threshold: &reference.threshold,
            channel: &reference.channel,
            polarity: &reference.polarity,
        },
        &mut PropagationState {
            activation: &mut reference.activation,
            charge: &mut reference.charge,
            refractory: &mut reference.refractory,
        },
        config,
        &mut reference.workspace,
    )
    .unwrap();

    assert_eq!(
        net.activation, reference.activation,
        "CSR path diverged from shared path activation for input={input:?}, config={config:?}"
    );
    assert_eq!(
        net.charge, reference.charge,
        "CSR path diverged from shared path charge for input={input:?}, config={config:?}"
    );
    let graph_rows = capture_graph_rows(&net);
    let csr_rows = capture_csr_rows(&mut net);
    assert_eq!(
        csr_rows, graph_rows,
        "CSR cache no longer matches graph semantics after propagate"
    );
}

#[test]
fn csr_path_matches_shared_kernel_on_controlled_fixtures() {
    let mut chain = Network::new(4);
    chain.graph_mut().add_edge(0, 1);
    chain.graph_mut().add_edge(1, 2);
    chain.threshold_mut().fill(1);
    assert_network_matches_shared_path(
        chain,
        &[1, 0, 0, 0],
        &PropagationConfig {
            ticks_per_token: 3,
            input_duration_ticks: 2,
            decay_interval_ticks: 0,
            use_refractory: false,
        },
    );

    let mut fan_in = Network::new(5);
    fan_in.graph_mut().add_edge(0, 2);
    fan_in.graph_mut().add_edge(1, 2);
    fan_in.graph_mut().add_edge(2, 3);
    fan_in.threshold_mut().fill(1);
    fan_in.polarity_mut()[1] = -1;
    assert_network_matches_shared_path(
        fan_in,
        &[1, 1, 0, 0, 0],
        &PropagationConfig {
            ticks_per_token: 4,
            input_duration_ticks: 2,
            decay_interval_ticks: 0,
            use_refractory: false,
        },
    );

    let mut phase_gate = Network::new(3);
    phase_gate.graph_mut().add_edge(0, 1);
    phase_gate.threshold_mut()[1] = 4;
    phase_gate.channel_mut()[1] = 8;
    assert_network_matches_shared_path(
        phase_gate,
        &[1, 0, 0],
        &PropagationConfig {
            ticks_per_token: 8,
            input_duration_ticks: 2,
            decay_interval_ticks: 6,
            use_refractory: false,
        },
    );
}

#[test]
fn csr_path_matches_shared_kernel_on_random_small_cases() {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);

    for _case in 0..128 {
        let neuron_count = rng.gen_range(0..=8usize);
        let net = random_network(&mut rng, neuron_count);
        let input = random_input(&mut rng, neuron_count);
        let config = random_config(&mut rng);
        assert_network_matches_shared_path(net, &input, &config);
    }
}

#[test]
fn csr_rows_track_graph_after_mixed_mutations_and_restore() {
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let mut net = random_network(&mut rng, 24);
    let input = random_input(&mut rng, net.neuron_count());
    let config = random_config(&mut rng);
    net.propagate(&input, &config).unwrap();

    let snapshot = net.save_state();
    let before_semantic = capture_semantic(&net);
    let before_rows = capture_graph_rows(&net);
    assert_eq!(capture_csr_rows(&mut net), before_rows);

    for _ in 0..64 {
        let _ = apply_random_mutation(&mut net, &mut rng);
    }
    let changed_input = random_input(&mut rng, net.neuron_count());
    net.propagate(&changed_input, &config).unwrap();

    net.restore_state(&snapshot);

    assert_eq!(
        capture_semantic(&net),
        before_semantic,
        "restore_state failed to round-trip mixed topology/param/state changes"
    );
    assert_eq!(
        capture_csr_rows(&mut net),
        before_rows,
        "CSR rows diverged from graph after restore"
    );
}

#[test]
fn repeated_reject_only_mutations_leave_no_semantic_drift() {
    let mut rng = StdRng::seed_from_u64(0xBAD5EED);
    let mut net = random_network(&mut rng, 32);
    let warmup_input = random_input(&mut rng, net.neuron_count());
    let warmup_config = random_config(&mut rng);
    net.propagate(&warmup_input, &warmup_config).unwrap();

    let baseline = capture_semantic(&net);
    let control_input = random_input(&mut rng, net.neuron_count());
    let control_config = random_config(&mut rng);
    let mut control = net.clone();
    control.propagate(&control_input, &control_config).unwrap();
    let expected_after = (control.activation.clone(), control.charge.clone());

    for _ in 0..1000 {
        let snapshot = net.save_state();
        let _ = apply_random_mutation(&mut net, &mut rng);
        net.restore_state(&snapshot);

        assert_eq!(
            capture_semantic(&net),
            baseline,
            "reject-only mutation loop leaked semantic state"
        );
        let graph_rows = capture_graph_rows(&net);
        let csr_rows = capture_csr_rows(&mut net);
        assert_eq!(
            csr_rows, graph_rows,
            "CSR cache drifted during reject-only mutation loop"
        );
    }

    net.propagate(&control_input, &control_config).unwrap();
    assert_eq!(
        (net.activation.clone(), net.charge.clone()),
        expected_after,
        "reject-only mutation loop changed later propagation semantics"
    );
}
