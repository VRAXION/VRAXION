//! Push for 100% on addition: jackpot + more steps + higher cap
//!
//! Run: cargo run --example addition_push100 --release

use instnct_core::{
    build_network, evolution_step_jackpot, EvolutionConfig, InitConfig, Int8Projection,
    SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::{Duration, Instant};

const DIGITS: usize = 5;
const SUMS: usize = 9;
const H: usize = 256;
const SDR_ACTIVE_PCT: usize = 20;
const WALL_SECS: u64 = 300; // 5 min per seed

fn make_examples() -> Vec<(usize, usize, usize)> {
    let mut ex = Vec::new();
    for a in 0..DIGITS { for b in 0..DIGITS { ex.push((a, b, a + b)); } }
    ex
}

fn eval_add(
    net: &mut instnct_core::Network, proj: &Int8Projection, examples: &[(usize, usize, usize)],
    sdr_a: &SdrTable, sdr_b: &SdrTable,
    prop_cfg: &instnct_core::PropagationConfig, output_start: usize, neuron_count: usize,
) -> f64 {
    let mut correct = 0;
    for &(a, b, target) in examples {
        net.reset();
        let pa = sdr_a.pattern(a); let pb = sdr_b.pattern(b);
        let mut combined = vec![0i32; neuron_count];
        for i in 0..neuron_count { combined[i] = pa[i] + pb[i]; }
        for _ in 0..6 { let _ = net.propagate(&combined, prop_cfg); }
        if proj.predict(&net.charge_vec(output_start..neuron_count)) == target { correct += 1; }
    }
    correct as f64 / examples.len() as f64
}

fn main() {
    let examples = make_examples();

    println!("=== PUSH FOR 100% ADDITION ===");
    println!("H={}, {}s/seed, jackpot=9, edge_caps=[100, 300, 500]\n", H, WALL_SECS);

    for &edge_cap in &[100usize, 300, 500] {
        println!("--- edge_cap={} ---", edge_cap);

        for &seed in &[42u64, 1042, 2042] {
            let mut rng = StdRng::seed_from_u64(seed);
            let init = InitConfig::empty(H);
            let mut net = build_network(&init, &mut rng);
            let sdr_a = SdrTable::new(DIGITS, H, init.input_end() / 2, SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 100)).unwrap();
            let sdr_b = SdrTable::new(DIGITS, H, init.input_end(), SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 200)).unwrap();
            let mut proj = Int8Projection::new(init.phi_dim, SUMS, &mut rng);
            let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
            let evo_config = EvolutionConfig { edge_cap, accept_ties: false };
            let prop_cfg = init.propagation.clone();
            let output_start = init.output_start();
            let neuron_count = init.neuron_count;

            let mut steps = 0usize;
            let mut peak = 0.0f64;
            let start = Instant::now();
            let deadline = start + Duration::from_secs(WALL_SECS);

            while Instant::now() < deadline {
                evolution_step_jackpot(
                    &mut net, &mut proj, &mut rng, &mut eval_rng,
                    |net, proj, _| eval_add(net, proj, &examples, &sdr_a, &sdr_b, &prop_cfg, output_start, neuron_count),
                    &evo_config, 9,
                );
                steps += 1;

                // Check progress every 10K steps
                if steps % 10000 == 0 {
                    let acc = eval_add(&mut net, &proj, &examples, &sdr_a, &sdr_b, &prop_cfg, output_start, neuron_count);
                    peak = peak.max(acc);
                    if acc >= 1.0 {
                        let elapsed = start.elapsed().as_secs();
                        println!("  seed {}: 100% at step {} ({}s) edges={}", seed, steps, elapsed, net.edge_count());
                        break;
                    }
                }
            }

            let final_acc = eval_add(&mut net, &proj, &examples, &sdr_a, &sdr_b, &prop_cfg, output_start, neuron_count);
            peak = peak.max(final_acc);
            if final_acc < 1.0 {
                println!("  seed {}: {:.0}% (peak={:.0}%) at {} steps, {} edges",
                    seed, final_acc * 100.0, peak * 100.0, steps, net.edge_count());
            }
        }
        println!();
    }
}
