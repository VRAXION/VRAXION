//! Addition freeze-crystal: evolve edges → crystallize → freeze → repeat
//! Using INSTNCT library (the winner on addition).
//!
//! Run: cargo run --example addition_freeze --release

use instnct_core::{
    build_network, evolution_step, EvolutionConfig, InitConfig, Int8Projection,
    Network, SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const DIGITS: usize = 5;
const SUMS: usize = 9;
const H: usize = 256;
const SDR_ACTIVE_PCT: usize = 20;
const MUTABLE_CAP: usize = 50;  // small cap per cycle
const CYCLES: usize = 8;
const STEPS_PER_CYCLE: usize = 30_000;

fn make_examples() -> Vec<(usize, usize, usize)> {
    let mut ex = Vec::new();
    for a in 0..DIGITS { for b in 0..DIGITS { ex.push((a, b, a + b)); } }
    ex
}

fn eval_addition(
    net: &mut Network, proj: &Int8Projection, examples: &[(usize, usize, usize)],
    sdr_a: &SdrTable, sdr_b: &SdrTable,
    prop_cfg: &instnct_core::PropagationConfig, output_start: usize, neuron_count: usize,
) -> f64 {
    let mut correct = 0;
    for &(a, b, target) in examples {
        net.reset();
        let pa = sdr_a.pattern(a);
        let pb = sdr_b.pattern(b);
        let mut combined = vec![0i32; neuron_count];
        for i in 0..neuron_count { combined[i] = pa[i] + pb[i]; }
        for _ in 0..6 { let _ = net.propagate(&combined, prop_cfg); }
        if proj.predict(&net.charge_vec(output_start..neuron_count)) == target { correct += 1; }
    }
    correct as f64 / examples.len() as f64
}

fn main() {
    let examples = make_examples();

    println!("=== ADDITION FREEZE-CRYSTAL (INSTNCT library) ===");
    println!("H={}, mutable_cap={}, {} steps/cycle, {} cycles", H, MUTABLE_CAP, STEPS_PER_CYCLE, CYCLES);
    println!("Random baseline: {:.1}%\n", 100.0 / SUMS as f64);

    // Also run flat baseline (no freeze, just edge_cap=300, same total steps)
    println!("--- Baseline: flat cap=300, {} steps ---", STEPS_PER_CYCLE * CYCLES);
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
        let output_start = init.output_start();
        let neuron_count = init.neuron_count;

        for _ in 0..(STEPS_PER_CYCLE * CYCLES) {
            evolution_step(
                &mut net, &mut proj, &mut rng, &mut eval_rng,
                |net, proj, _| eval_addition(net, proj, &examples, &sdr_a, &sdr_b, &prop_cfg, output_start, neuron_count),
                &evo_config,
            );
        }
        let acc = eval_addition(&mut net, &proj, &examples, &sdr_a, &sdr_b, &prop_cfg, output_start, neuron_count);
        println!("  seed {}: {:.0}% ({} edges)", seed, acc * 100.0, net.edge_count());
    }

    // Freeze-crystal version
    println!("\n--- Freeze-crystal: cap={}/cycle, {} cycles ---", MUTABLE_CAP, CYCLES);
    for &seed in &[42u64, 1042, 2042] {
        let mut rng = StdRng::seed_from_u64(seed);
        let init = InitConfig::empty(H);
        let mut net = build_network(&init, &mut rng);
        let sdr_a = SdrTable::new(DIGITS, H, init.input_end() / 2, SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 100)).unwrap();
        let sdr_b = SdrTable::new(DIGITS, H, init.input_end(), SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 200)).unwrap();
        let mut proj = Int8Projection::new(init.phi_dim, SUMS, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
        let prop_cfg = init.propagation.clone();
        let output_start = init.output_start();
        let neuron_count = init.neuron_count;

        print!("  seed {}:", seed);

        for cycle in 0..CYCLES {
            // Evolve with small cap
            let evo_config = EvolutionConfig { edge_cap: net.edge_count() + MUTABLE_CAP, accept_ties: false };

            let edges_before = net.edge_count();
            for _ in 0..STEPS_PER_CYCLE {
                evolution_step(
                    &mut net, &mut proj, &mut rng, &mut eval_rng,
                    |net, proj, _| eval_addition(net, proj, &examples, &sdr_a, &sdr_b, &prop_cfg, output_start, neuron_count),
                    &evo_config,
                );
            }

            let acc = eval_addition(&mut net, &proj, &examples, &sdr_a, &sdr_b, &prop_cfg, output_start, neuron_count);
            let edges_now = net.edge_count();
            print!(" c{}={:.0}%/{}e", cycle, acc * 100.0, edges_now);

            // "Freeze" = just raise the floor. Next cycle's cap = current edges + MUTABLE_CAP.
            // The library doesn't let us freeze edges, but we achieve the same effect
            // by setting edge_cap = current + MUTABLE_CAP each cycle.
        }

        let final_acc = eval_addition(&mut net, &proj, &examples, &sdr_a, &sdr_b, &prop_cfg, output_start, neuron_count);
        println!(" → final={:.0}% ({} edges)", final_acc * 100.0, net.edge_count());
    }
}
