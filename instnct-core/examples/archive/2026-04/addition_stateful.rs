//! Stateful addition: NO reset between examples. State carries across the sequence.
//!
//! Run: cargo run --example addition_stateful --release

use instnct_core::{
    build_network, evolution_step, EvolutionConfig, InitConfig, Int8Projection,
    Network, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const DIGITS: usize = 5;
const SUMS: usize = 9;
const H: usize = 256;
const SDR_ACTIVE_PCT: usize = 20;
const WALL_SECS: u64 = 120;

fn make_examples() -> Vec<(usize, usize, usize)> {
    let mut ex = Vec::new();
    for a in 0..DIGITS { for b in 0..DIGITS { ex.push((a, b, a + b)); } }
    ex
}

/// Eval a SEQUENCE of examples WITHOUT resetting between them.
/// The network carries state from one example to the next.
fn eval_sequential(
    net: &mut Network, proj: &Int8Projection,
    examples: &[(usize, usize, usize)],
    sdr_a: &SdrTable, sdr_b: &SdrTable,
    prop_cfg: &instnct_core::PropagationConfig,
    output_start: usize, nc: usize,
    rng: &mut StdRng,
) -> f64 {
    // Shuffle order each eval (so it can't memorize sequence position)
    let mut order: Vec<usize> = (0..examples.len()).collect();
    for i in (1..order.len()).rev() { let j = rng.gen_range(0..=i); order.swap(i, j); }

    net.reset(); // reset only ONCE at start of sequence
    let mut correct = 0;

    for &idx in &order {
        let (a, b, target) = examples[idx];
        let pa = sdr_a.pattern(a); let pb = sdr_b.pattern(b);
        let mut combined = vec![0i32; nc];
        for i in 0..nc { combined[i] = pa[i] + pb[i]; }

        // Process this example (6 ticks) — state carries from previous
        for _ in 0..6 { let _ = net.propagate(&combined, prop_cfg); }

        if proj.predict(&net.charge_vec(output_start..nc)) == target {
            correct += 1;
        }
        // NO RESET here — state carries to next example
    }
    correct as f64 / examples.len() as f64
}

/// Standard eval WITH reset between each example (for comparison)
fn eval_reset(
    net: &mut Network, proj: &Int8Projection,
    examples: &[(usize, usize, usize)],
    sdr_a: &SdrTable, sdr_b: &SdrTable,
    prop_cfg: &instnct_core::PropagationConfig,
    output_start: usize, nc: usize,
) -> f64 {
    let mut correct = 0;
    for &(a, b, target) in examples {
        net.reset(); // reset each time
        let pa = sdr_a.pattern(a); let pb = sdr_b.pattern(b);
        let mut combined = vec![0i32; nc];
        for i in 0..nc { combined[i] = pa[i] + pb[i]; }
        for _ in 0..6 { let _ = net.propagate(&combined, prop_cfg); }
        if proj.predict(&net.charge_vec(output_start..nc)) == target { correct += 1; }
    }
    correct as f64 / examples.len() as f64
}

fn main() {
    let all_examples = make_examples();
    // Train/test split: hold out sum=4 (5 examples)
    let train: Vec<_> = all_examples.iter().filter(|&&(_,_,s)| s != 4).cloned().collect();
    let test: Vec<_> = all_examples.iter().filter(|&&(_,_,s)| s == 4).cloned().collect();

    println!("=== STATEFUL ADDITION ===");
    println!("No reset between examples. State carries across sequence.");
    println!("H={}, {}s/seed, train: sum≠4 (20 ex), test: sum=4 (5 ex)\n", H, WALL_SECS);

    // A: Train stateful, test stateful
    println!("--- A: Stateful train + stateful test ---");
    for &seed in &[42u64, 1042, 2042] {
        let mut rng = StdRng::seed_from_u64(seed);
        let init = InitConfig::empty(H);
        let mut net = build_network(&init, &mut rng);
        let sdr_a = SdrTable::new(DIGITS, H, init.input_end() / 2, SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 100)).unwrap();
        let sdr_b = SdrTable::new(DIGITS, H, init.input_end(), SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 200)).unwrap();
        let mut proj = Int8Projection::new(init.phi_dim, SUMS, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
        let evo_config = EvolutionConfig { edge_cap: 300, accept_ties: false };
        let prop_cfg = init.propagation.clone();
        let os = init.output_start(); let nc = init.neuron_count;

        let deadline = Instant::now() + Duration::from_secs(WALL_SECS);
        while Instant::now() < deadline {
            evolution_step(
                &mut net, &mut proj, &mut rng, &mut eval_rng,
                |net, proj, eval_rng| eval_sequential(net, proj, &train, &sdr_a, &sdr_b, &prop_cfg, os, nc, eval_rng),
                &evo_config,
            );
        }

        let train_acc = eval_sequential(&mut net, &proj, &train, &sdr_a, &sdr_b, &prop_cfg, os, nc, &mut eval_rng.clone());
        let test_acc = eval_sequential(&mut net, &proj, &test, &sdr_a, &sdr_b, &prop_cfg, os, nc, &mut eval_rng.clone());
        let test_reset = eval_reset(&mut net, &proj, &test, &sdr_a, &sdr_b, &prop_cfg, os, nc);
        let all_acc = eval_sequential(&mut net, &proj, &all_examples, &sdr_a, &sdr_b, &prop_cfg, os, nc, &mut eval_rng.clone());
        println!("  seed {}: train={:.0}% test_stateful={:.0}% test_reset={:.0}% all={:.0}% ({} edges)",
            seed, train_acc*100.0, test_acc*100.0, test_reset*100.0, all_acc*100.0, net.edge_count());
    }

    // B: Train with reset (standard), test both ways
    println!("\n--- B: Standard train (reset), test both ways ---");
    for &seed in &[42u64, 1042, 2042] {
        let mut rng = StdRng::seed_from_u64(seed);
        let init = InitConfig::empty(H);
        let mut net = build_network(&init, &mut rng);
        let sdr_a = SdrTable::new(DIGITS, H, init.input_end() / 2, SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 100)).unwrap();
        let sdr_b = SdrTable::new(DIGITS, H, init.input_end(), SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 200)).unwrap();
        let mut proj = Int8Projection::new(init.phi_dim, SUMS, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
        let evo_config = EvolutionConfig { edge_cap: 300, accept_ties: false };
        let prop_cfg = init.propagation.clone();
        let os = init.output_start(); let nc = init.neuron_count;

        let deadline = Instant::now() + Duration::from_secs(WALL_SECS);
        while Instant::now() < deadline {
            evolution_step(
                &mut net, &mut proj, &mut rng, &mut eval_rng,
                |net, proj, _| eval_reset(net, proj, &train, &sdr_a, &sdr_b, &prop_cfg, os, nc),
                &evo_config,
            );
        }

        let train_acc = eval_reset(&mut net, &proj, &train, &sdr_a, &sdr_b, &prop_cfg, os, nc);
        let test_reset = eval_reset(&mut net, &proj, &test, &sdr_a, &sdr_b, &prop_cfg, os, nc);
        let test_stateful = eval_sequential(&mut net, &proj, &test, &sdr_a, &sdr_b, &prop_cfg, os, nc, &mut eval_rng.clone());
        println!("  seed {}: train={:.0}% test_reset={:.0}% test_stateful={:.0}% ({} edges)",
            seed, train_acc*100.0, test_reset*100.0, test_stateful*100.0, net.edge_count());
    }
}
