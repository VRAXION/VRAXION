//! Evolution loop time breakdown: mutate vs evaluate vs snapshot/restore.
//!
//! Run: cargo run --example loop_breakdown --release

use instnct_core::{Network, PropagationConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const NEURON_COUNT: usize = 256;
const DENSITY_PCT: u64 = 5;
const NUM_TOKENS: usize = 26;
const STEPS: usize = 1000;

fn evaluate(net: &mut Network, config: &PropagationConfig) -> u32 {
    let mut score = 0u32;
    let mut prev_pattern: Vec<i8> = vec![];
    for token in 0..NUM_TOKENS {
        net.reset();
        let mut input = vec![0i32; NEURON_COUNT];
        input[token % NEURON_COUNT] = 1;
        net.propagate(&input, config).unwrap();
        let pattern = net.activation().to_vec();
        if pattern != prev_pattern {
            score += 1;
        }
        prev_pattern = pattern;
    }
    score
}

fn main() {
    let config = PropagationConfig::default();
    let mut net = Network::new(NEURON_COUNT);
    let mut rng = StdRng::seed_from_u64(42);

    // Build initial graph
    let target_edges = (NEURON_COUNT as u64 * NEURON_COUNT as u64 * DENSITY_PCT / 100) as usize;
    for _ in 0..target_edges * 3 {
        net.mutate_add_edge(&mut rng);
        if net.edge_count() >= target_edges {
            break;
        }
    }
    for i in 0..NEURON_COUNT {
        net.threshold_mut()[i] = rng.gen_range(0..=15);
        net.channel_mut()[i] = rng.gen_range(1..=8);
    }

    let mut best_score = evaluate(&mut net, &config);

    // Measure each phase of the evolution loop
    let mut time_save = 0u64;
    let mut time_mutate = 0u64;
    let mut time_eval = 0u64;
    let mut time_restore = 0u64;
    let mut restores = 0u32;

    let loop_start = Instant::now();

    for _ in 0..STEPS {
        // SAVE
        let t = Instant::now();
        let snapshot = net.save_state();
        time_save += t.elapsed().as_nanos() as u64;

        // MUTATE
        let t = Instant::now();
        let mutated = if net.edge_count() == 0 || rng.gen_ratio(7, 10) {
            net.mutate_add_edge(&mut rng)
        } else {
            net.mutate_remove_edge(&mut rng)
        };
        time_mutate += t.elapsed().as_nanos() as u64;

        if !mutated {
            continue;
        }

        // EVALUATE (26 tokens)
        let t = Instant::now();
        let score = evaluate(&mut net, &config);
        time_eval += t.elapsed().as_nanos() as u64;

        // ACCEPT/RESTORE
        if score >= best_score {
            best_score = score;
        } else {
            let t = Instant::now();
            net.restore_state(&snapshot);
            time_restore += t.elapsed().as_nanos() as u64;
            restores += 1;
        }
    }

    let loop_total = loop_start.elapsed().as_nanos() as u64;

    println!("Evolution loop: H={NEURON_COUNT}, {NUM_TOKENS} tokens, {STEPS} steps\n");
    println!("  TOTAL:            {:>10} us  (100%)", loop_total / 1000);
    println!(
        "  save_state:       {:>10} us  ({:.1}%)",
        time_save / 1000,
        time_save as f64 / loop_total as f64 * 100.0
    );
    println!(
        "  mutate:           {:>10} us  ({:.1}%)",
        time_mutate / 1000,
        time_mutate as f64 / loop_total as f64 * 100.0
    );
    println!(
        "  evaluate (26tok): {:>10} us  ({:.1}%)  <-- {:.0} us per eval",
        time_eval / 1000,
        time_eval as f64 / loop_total as f64 * 100.0,
        time_eval as f64 / STEPS as f64 / 1000.0
    );
    println!(
        "  restore_state:    {:>10} us  ({:.1}%)  ({restores} restores)",
        time_restore / 1000,
        time_restore as f64 / loop_total as f64 * 100.0,
    );

    let other = loop_total - time_save - time_mutate - time_eval - time_restore;
    println!(
        "  overhead:         {:>10} us  ({:.1}%)",
        other / 1000,
        other as f64 / loop_total as f64 * 100.0
    );

    println!(
        "\n  Steps/sec: {:.0}",
        STEPS as f64 / loop_total as f64 * 1e9
    );
    println!(
        "  Final: edges={}, score={best_score}/{NUM_TOKENS}",
        net.edge_count()
    );
}
