use criterion::{black_box, criterion_group, criterion_main, Criterion};
use instnct_core::propagation::{
    build_wave_gating_table, propagate_token, NeuronParameters, NeuronState, PropagationConfig,
};
use instnct_core::topology::ConnectionGraph;

fn bench_propagation_h256(c: &mut Criterion) {
    let h = 256;
    let mut graph = ConnectionGraph::new(h);
    let mut rng_state: u64 = 42;
    for i in 0..h {
        for j in 0..h {
            if i == j {
                continue;
            }
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng_state >> 32) % 100 < 5 {
                graph.add_edge(i as u16, j as u16);
            }
        }
    }
    let sources = graph.sources();
    let targets = graph.targets();
    let threshold = vec![6u32; h];
    let channel = vec![1u8; h];
    let polarity = vec![1i32; h];
    let mut input = vec![0i32; h];
    input[0] = 1;
    let wave_table = build_wave_gating_table();

    // Pre-allocate outside the benchmark loop (no allocator cost measured)
    let mut activation = vec![0i32; h];
    let mut charge = vec![0u32; h];
    let mut scratch = vec![0i32; h];

    c.bench_function("propagate_h256_12ticks_i32", |b| {
        b.iter(|| {
            activation.fill(0);
            charge.fill(0);
            propagate_token(
                black_box(&input),
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
                    ticks: 12,
                    input_duration: 2,
                    decay_period: 6,
                },
                &wave_table,
                &mut scratch,
            );
        });
    });
}

fn bench_propagation_h1024(c: &mut Criterion) {
    let h = 1024;
    let mut graph = ConnectionGraph::new(h);
    let mut rng_state: u64 = 42;
    for i in 0..h {
        for j in 0..h {
            if i == j {
                continue;
            }
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng_state >> 32) % 100 < 2 {
                graph.add_edge(i as u16, j as u16);
            }
        }
    }
    let sources = graph.sources();
    let targets = graph.targets();
    let threshold = vec![6u32; h];
    let channel = vec![1u8; h];
    let polarity = vec![1i32; h];
    let mut input = vec![0i32; h];
    input[0] = 1;
    let wave_table = build_wave_gating_table();

    let mut activation = vec![0i32; h];
    let mut charge = vec![0u32; h];
    let mut scratch = vec![0i32; h];

    c.bench_function("propagate_h1024_16ticks_i32", |b| {
        b.iter(|| {
            activation.fill(0);
            charge.fill(0);
            propagate_token(
                black_box(&input),
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
                    ticks: 16,
                    input_duration: 2,
                    decay_period: 6,
                },
                &wave_table,
                &mut scratch,
            );
        });
    });
}

criterion_group!(benches, bench_propagation_h256, bench_propagation_h1024);
criterion_main!(benches);
