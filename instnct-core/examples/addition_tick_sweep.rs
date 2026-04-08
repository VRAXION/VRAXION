//! Addition tick sweep: does more propagation time help?
//! 6 ticks (default) vs 12 vs 24 — deeper circuits need more ticks.
//!
//! Run: cargo run --example addition_tick_sweep --release

use instnct_core::{
    build_network, evolution_step, EvolutionConfig, InitConfig, Int8Projection,
    Network, PropagationConfig, SdrTable,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::{Duration, Instant};

const DIGITS: usize = 5;
const SUMS: usize = 9;
const H: usize = 256;
const SDR_ACTIVE_PCT: usize = 20;
const WALL_SECS: u64 = 120;
const SEEDS: [u64; 3] = [42, 1042, 2042];

fn make_examples() -> Vec<(usize, usize, usize)> {
    let mut ex = Vec::new();
    for a in 0..DIGITS { for b in 0..DIGITS { ex.push((a, b, a + b)); } }
    ex
}

fn eval_add(
    net: &mut Network, proj: &Int8Projection, examples: &[(usize, usize, usize)],
    sdr_a: &SdrTable, sdr_b: &SdrTable,
    prop_cfg: &PropagationConfig, output_start: usize, neuron_count: usize, ticks: usize,
) -> f64 {
    let mut correct = 0;
    for &(a, b, target) in examples {
        net.reset();
        let pa = sdr_a.pattern(a); let pb = sdr_b.pattern(b);
        let mut combined = vec![0i32; neuron_count];
        for i in 0..neuron_count { combined[i] = pa[i] + pb[i]; }
        for _ in 0..ticks { let _ = net.propagate(&combined, prop_cfg); }
        if proj.predict(&net.charge_vec(output_start..neuron_count)) == target { correct += 1; }
    }
    correct as f64 / examples.len() as f64
}

fn main() {
    let examples = make_examples();

    println!("=== ADDITION TICK SWEEP ===");
    println!("H={}, empty init, {}s/seed, {} seeds\n", H, WALL_SECS, SEEDS.len());

    for &ticks in &[6usize, 12, 24] {
        println!("--- ticks={} ---", ticks);

        let mut all_acc = Vec::new();
        let mut all_steps = Vec::new();
        let mut all_edges = Vec::new();

        for &seed in &SEEDS {
            let mut rng = StdRng::seed_from_u64(seed);
            let init = InitConfig::empty(H);
            let mut net = build_network(&init, &mut rng);
            let sdr_a = SdrTable::new(DIGITS, H, init.input_end() / 2, SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 100)).unwrap();
            let sdr_b = SdrTable::new(DIGITS, H, init.input_end(), SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 200)).unwrap();
            let mut proj = Int8Projection::new(init.phi_dim, SUMS, &mut rng);
            let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
            let evo_config = EvolutionConfig { edge_cap: 300, accept_ties: false };
            let prop_cfg = PropagationConfig {
                ticks_per_token: ticks,
                input_duration_ticks: 2,
                decay_interval_ticks: ticks, // decay at end
                use_refractory: false,
            };
            let output_start = init.output_start();
            let neuron_count = init.neuron_count;

            let mut steps = 0usize;
            let deadline = Instant::now() + Duration::from_secs(WALL_SECS);
            while Instant::now() < deadline {
                evolution_step(
                    &mut net, &mut proj, &mut rng, &mut eval_rng,
                    |net, proj, _| eval_add(net, proj, &examples, &sdr_a, &sdr_b, &prop_cfg, output_start, neuron_count, ticks),
                    &evo_config,
                );
                steps += 1;
            }

            let acc = eval_add(&mut net, &proj, &examples, &sdr_a, &sdr_b, &prop_cfg, output_start, neuron_count, ticks);
            all_acc.push(acc); all_steps.push(steps); all_edges.push(net.edge_count());
        }

        let best = all_acc.iter().cloned().fold(0.0f64, f64::max);
        let mean = all_acc.iter().sum::<f64>() / all_acc.len() as f64;
        let ms = all_steps.iter().sum::<usize>() / all_steps.len();
        let me = all_edges.iter().sum::<usize>() / all_edges.len();
        let seeds_str: Vec<String> = all_acc.iter().map(|a| format!("{:.0}%", a*100.0)).collect();

        println!("  step/s={:.0} edges={} best={:.0}% mean={:.0}% seeds=[{}]",
            ms as f64 / WALL_SECS as f64, me, best*100.0, mean*100.0, seeds_str.join(", "));
    }
}
