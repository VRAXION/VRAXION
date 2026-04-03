use criterion::{black_box, criterion_group, criterion_main, Criterion};
use instnct_core::forward::rollout_token;
use instnct_core::quaternary_mask::QuaternaryMask;

fn bench_forward_h256(c: &mut Criterion) {
    let h = 256;
    let mut mask = QuaternaryMask::new(h);
    // 5% density
    let mut rng_seed: u64 = 42;
    for idx in 0..mask.n_pairs() {
        rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        if (rng_seed >> 32) % 100 < 5 {
            mask.data[idx] = 1;
        }
    }
    let (sources, targets) = mask.to_directed_edges();
    let theta = vec![6.0f32; h];
    let channel = vec![1u8; h];
    let polarity = vec![1.0f32; h];
    let mut injected = vec![0.0f32; h];
    injected[0] = 1.0;

    c.bench_function("forward_h256_12tick", |b| {
        b.iter(|| {
            let mut state = vec![0.0f32; h];
            let mut charge = vec![0.0f32; h];
            rollout_token(
                black_box(&injected),
                &sources, &targets, &theta, &channel, &polarity,
                12, 2, 6,
                &mut state, &mut charge,
            );
        });
    });
}

fn bench_forward_h1024(c: &mut Criterion) {
    let h = 1024;
    let mut mask = QuaternaryMask::new(h);
    let mut rng_seed: u64 = 42;
    for idx in 0..mask.n_pairs() {
        rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        if (rng_seed >> 32) % 100 < 2 {
            mask.data[idx] = 1;
        }
    }
    let (sources, targets) = mask.to_directed_edges();
    let theta = vec![6.0f32; h];
    let channel = vec![1u8; h];
    let polarity = vec![1.0f32; h];
    let mut injected = vec![0.0f32; h];
    injected[0] = 1.0;

    c.bench_function("forward_h1024_16tick", |b| {
        b.iter(|| {
            let mut state = vec![0.0f32; h];
            let mut charge = vec![0.0f32; h];
            rollout_token(
                black_box(&injected),
                &sources, &targets, &theta, &channel, &polarity,
                16, 2, 6,
                &mut state, &mut charge,
            );
        });
    });
}

criterion_group!(benches, bench_forward_h256, bench_forward_h1024);
criterion_main!(benches);
