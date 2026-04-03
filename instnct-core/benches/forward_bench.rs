use instnct_core::__internal::propagate_token_unchecked;
use instnct_core::{
    ConnectionGraph, PropagationConfig, PropagationParameters, PropagationState,
    PropagationWorkspace,
};
use std::hint::black_box;
use std::time::Instant;

fn bench_case(
    name: &str,
    neuron_count: usize,
    edge_probability_percent: u64,
    config: PropagationConfig,
) {
    let mut graph = ConnectionGraph::new(neuron_count);
    let mut rng_state: u64 = 42;
    for i in 0..neuron_count {
        for j in 0..neuron_count {
            if i == j {
                continue;
            }
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng_state >> 32) % 100 < edge_probability_percent {
                graph.add_edge(i as u16, j as u16);
            }
        }
    }

    let threshold = vec![6u32; neuron_count];
    let channel = vec![1u8; neuron_count];
    let polarity = vec![1i32; neuron_count];
    let mut input = vec![0i32; neuron_count];
    input[0] = 1;

    let mut activation = vec![0i32; neuron_count];
    let mut charge = vec![0u32; neuron_count];
    let mut workspace = PropagationWorkspace::new(neuron_count);

    let iterations = if neuron_count <= 256 { 2_000 } else { 500 };
    let start = Instant::now();

    for _ in 0..iterations {
        activation.fill(0);
        charge.fill(0);

        let mut state = PropagationState {
            activation: &mut activation,
            charge: &mut charge,
        };
        let params = PropagationParameters {
            threshold: &threshold,
            channel: &channel,
            polarity: &polarity,
        };

        propagate_token_unchecked(
            black_box(&input),
            black_box(&graph),
            black_box(&params),
            black_box(&mut state),
            black_box(&config),
            black_box(&mut workspace),
        );
    }

    let elapsed = start.elapsed();
    let ns_per_iter = elapsed.as_secs_f64() * 1_000_000_000.0 / iterations as f64;
    println!(
        "{name}: {iterations} iterations in {:?} ({ns_per_iter:.0} ns/iter)",
        elapsed
    );
}

fn main() {
    bench_case(
        "propagate_h256_12ticks_i32",
        256,
        5,
        PropagationConfig {
            ticks: 12,
            input_duration: 2,
            decay_period: 6,
        },
    );
    bench_case(
        "propagate_h1024_16ticks_i32",
        1024,
        2,
        PropagationConfig {
            ticks: 16,
            input_duration: 2,
            decay_period: 6,
        },
    );
}
