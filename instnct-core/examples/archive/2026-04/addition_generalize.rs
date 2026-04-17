//! Does the network GENERALIZE or just MEMORIZE?
//! Train on subset of addition examples, test on held-out pairs.
//! If test accuracy > random → generalization. If 0% → pure memorization.
//!
//! Also test larger range (0-9) where memorization fails.
//!
//! Run: cargo run --example addition_generalize --release

use instnct_core::{
    build_network, evolution_step, EvolutionConfig, InitConfig, Int8Projection,
    SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const H: usize = 256;
const SDR_ACTIVE_PCT: usize = 20;
const WALL_SECS: u64 = 120;

fn eval_examples(
    net: &mut instnct_core::Network, proj: &Int8Projection,
    examples: &[(usize, usize, usize)],
    sdr_a: &SdrTable, sdr_b: &SdrTable,
    prop_cfg: &instnct_core::PropagationConfig, output_start: usize, nc: usize,
) -> f64 {
    if examples.is_empty() { return 0.0; }
    let mut correct = 0;
    for &(a, b, target) in examples {
        net.reset();
        let pa = sdr_a.pattern(a); let pb = sdr_b.pattern(b);
        let mut combined = vec![0i32; nc];
        for i in 0..nc { combined[i] = pa[i] + pb[i]; }
        for _ in 0..6 { let _ = net.propagate(&combined, prop_cfg); }
        if proj.predict(&net.charge_vec(output_start..nc)) == target { correct += 1; }
    }
    correct as f64 / examples.len() as f64
}

fn run_experiment(
    label: &str, digits: usize, train_examples: &[(usize, usize, usize)],
    test_examples: &[(usize, usize, usize)], num_classes: usize,
) {
    println!("--- {} ---", label);
    println!("  train: {} examples, test: {} examples, classes: {}, random: {:.1}%",
        train_examples.len(), test_examples.len(), num_classes, 100.0 / num_classes as f64);

    for &seed in &[42u64, 1042, 2042] {
        let mut rng = StdRng::seed_from_u64(seed);
        let init = InitConfig::empty(H);
        let mut net = build_network(&init, &mut rng);
        let sdr_a = SdrTable::new(digits, H, init.input_end() / 2, SDR_ACTIVE_PCT,
            &mut StdRng::seed_from_u64(seed + 100)).unwrap();
        let sdr_b = SdrTable::new(digits, H, init.input_end(), SDR_ACTIVE_PCT,
            &mut StdRng::seed_from_u64(seed + 200)).unwrap();
        let mut proj = Int8Projection::new(init.phi_dim, num_classes, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
        let evo_config = EvolutionConfig { edge_cap: 300, accept_ties: false };
        let prop_cfg = init.propagation.clone();
        let output_start = init.output_start();
        let nc = init.neuron_count;

        let deadline = Instant::now() + Duration::from_secs(WALL_SECS);
        let mut steps = 0;
        while Instant::now() < deadline {
            evolution_step(
                &mut net, &mut proj, &mut rng, &mut eval_rng,
                |net, proj, _| eval_examples(net, proj, train_examples, &sdr_a, &sdr_b, &prop_cfg, output_start, nc),
                &evo_config,
            );
            steps += 1;
        }

        let train_acc = eval_examples(&mut net, &proj, train_examples, &sdr_a, &sdr_b, &prop_cfg, output_start, nc);
        let test_acc = eval_examples(&mut net, &proj, test_examples, &sdr_a, &sdr_b, &prop_cfg, output_start, nc);

        println!("  seed {}: train={:.0}% test={:.0}% ({} edges, {} steps)",
            seed, train_acc * 100.0, test_acc * 100.0, net.edge_count(), steps);
    }
    println!();
}

fn main() {
    println!("=== GENERALIZATION TEST ===\n");

    // --- Test 1: 0-4 addition, train/test split ---
    // Hold out: (1,3,4), (2,2,4), (3,1,4), (0,4,4), (4,0,4) — all sum=4 examples
    let all_5: Vec<(usize, usize, usize)> = (0..5).flat_map(|a| (0..5).map(move |b| (a, b, a+b))).collect();
    let train_5: Vec<_> = all_5.iter().filter(|&&(_,_,s)| s != 4).cloned().collect();
    let test_5: Vec<_> = all_5.iter().filter(|&&(_,_,s)| s == 4).cloned().collect();
    run_experiment("0-4 addition: train on sum≠4, test on sum=4", 5, &train_5, &test_5, 9);

    // --- Test 2: 0-4 addition, random hold-out ---
    let mut rng = StdRng::seed_from_u64(777);
    let mut shuffled = all_5.clone();
    for i in (1..shuffled.len()).rev() { let j = rng.gen_range(0..=i); shuffled.swap(i, j); }
    let train_rand: Vec<_> = shuffled[..20].to_vec();
    let test_rand: Vec<_> = shuffled[20..].to_vec();
    println!("  held-out: {:?}", test_rand.iter().map(|&(a,b,s)| format!("{}+{}={}", a, b, s)).collect::<Vec<_>>());
    run_experiment("0-4 addition: random 20/5 split", 5, &train_rand, &test_rand, 9);

    // --- Test 3: 0-9 addition (100 examples, 19 classes) ---
    let all_10: Vec<(usize, usize, usize)> = (0..10).flat_map(|a| (0..10).map(move |b| (a, b, a+b))).collect();
    // Train on all, see how far it gets
    run_experiment("0-9 addition: train on ALL 100 examples", 10, &all_10, &all_10, 19);

    // --- Test 4: 0-9 addition, train/test split ---
    let train_10: Vec<_> = all_10.iter().filter(|&&(a,b,_)| (a + b) % 3 != 0).cloned().collect();
    let test_10: Vec<_> = all_10.iter().filter(|&&(a,b,_)| (a + b) % 3 == 0).cloned().collect();
    run_experiment("0-9 addition: train on sum%3≠0, test on sum%3=0", 10, &train_10, &test_10, 19);
}
