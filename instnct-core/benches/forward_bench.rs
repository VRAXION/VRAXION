use criterion::{black_box, criterion_group, criterion_main, Criterion};
use instnct_core::propagation::{propagate_token, NeuronParameters, NeuronState, PropagationConfig};
use instnct_core::topology::ConnectionMask;

fn bench_propagation_h256(c: &mut Criterion) {
    let h = 256;
    let mut mask = ConnectionMask::new(h);
    // Deterministic 5% density initialization
    let mut rng_state: u64 = 42;
    for idx in 0..mask.pair_count() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        if (rng_state >> 32) % 100 < 5 {
            mask.pairs[idx] = 1; // forward edge
        }
    }
    let (sources, targets) = mask.to_directed_edges();
    let threshold = vec![6.0f32; h];
    let channel = vec![1u8; h];
    let polarity = vec![1.0f32; h];
    let mut input = vec![0.0f32; h];
    input[0] = 1.0;

    c.bench_function("propagate_h256_12ticks", |b| {
        b.iter(|| {
            let mut activation = vec![0.0f32; h];
            let mut charge = vec![0.0f32; h];
            propagate_token(
                black_box(&input),
                &sources, &targets,
                &NeuronParameters { threshold: &threshold, channel: &channel, polarity: &polarity },
                &mut NeuronState { activation: &mut activation, charge: &mut charge },
                &PropagationConfig { ticks: 12, input_duration: 2, decay_period: 6 },
            );
        });
    });
}

fn bench_propagation_h1024(c: &mut Criterion) {
    let h = 1024;
    let mut mask = ConnectionMask::new(h);
    let mut rng_state: u64 = 42;
    for idx in 0..mask.pair_count() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        if (rng_state >> 32) % 100 < 2 {
            mask.pairs[idx] = 1;
        }
    }
    let (sources, targets) = mask.to_directed_edges();
    let threshold = vec![6.0f32; h];
    let channel = vec![1u8; h];
    let polarity = vec![1.0f32; h];
    let mut input = vec![0.0f32; h];
    input[0] = 1.0;

    c.bench_function("propagate_h1024_16ticks", |b| {
        b.iter(|| {
            let mut activation = vec![0.0f32; h];
            let mut charge = vec![0.0f32; h];
            propagate_token(
                black_box(&input),
                &sources, &targets,
                &NeuronParameters { threshold: &threshold, channel: &channel, polarity: &polarity },
                &mut NeuronState { activation: &mut activation, charge: &mut charge },
                &PropagationConfig { ticks: 16, input_duration: 2, decay_period: 6 },
            );
        });
    });
}

criterion_group!(benches, bench_propagation_h256, bench_propagation_h1024);
criterion_main!(benches);
