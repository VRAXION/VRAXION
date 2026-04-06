//! Cooperative evolution v3: specialized mutation workers on one shared network.
//!
//! 7 workers, each specialized in one mutation type, compete per round.
//! The best delta wins → natural edge balance (add competes against remove).
//!
//! Workers:
//!   0: add_edge    (grows topology)
//!   1: remove_edge (prunes topology)
//!   2: rewire      (restructures)
//!   3: reverse     (flips direction)
//!   4: mirror      (adds reciprocal)
//!   5: param       (theta/channel/polarity)
//!   6: projection  (W weight perturbation)
//!
//! Run: cargo run --example clustered_evolution --release -- <corpus-path>

use instnct_core::{load_corpus, 
    build_network, evolution_step, InitConfig, Int8Projection, Network, PropagationConfig,
    SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const N_WORKERS: usize = 7;

const WORKER_NAMES: [&str; N_WORKERS] = [
    "add", "remove", "rewire", "reverse", "mirror", "param", "projection",
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------


fn sample_eval_offset(corpus_len: usize, len: usize, rng: &mut StdRng) -> Option<usize> {
    if corpus_len <= len { return None; }
    Some(rng.gen_range(0..=corpus_len - len - 1))
}

#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network,
    projection: &Int8Projection,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    config: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
) -> f64 {
    let Some(off) = sample_eval_offset(corpus.len(), len, rng) else { return 0.0; };
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        if projection.predict(&net.charge()[output_start..neuron_count]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

fn bar(val: f64, max_val: f64, width: usize) -> String {
    let filled = ((val / max_val) * width as f64).round().min(width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
}

// ---------------------------------------------------------------------------
// Specialized mutations: one type per worker
// ---------------------------------------------------------------------------

fn apply_mutation(
    worker_id: usize,
    net: &mut Network,
    proj: &mut Int8Projection,
    rng: &mut impl Rng,
) -> bool {
    match worker_id {
        0 => net.mutate_add_edge(rng),
        1 => net.mutate_remove_edge(rng),
        2 => net.mutate_rewire(rng),
        3 => net.mutate_reverse(rng),
        4 => net.mutate_mirror(rng),
        5 => {
            // Param: rotate between theta/channel/polarity
            let sub = rng.gen_range(0..3u32);
            match sub {
                0 => net.mutate_theta(rng),
                1 => net.mutate_channel(rng),
                _ => net.mutate_polarity(rng),
            }
        }
        _ => { let _ = proj.mutate_one(rng); true }
    }
}

// ---------------------------------------------------------------------------
// Jackpot cooperative evolution (specialized workers)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_specialized(
    corpus: &[u8],
    init: &InitConfig,
    sdr: &SdrTable,
    rounds: usize,
    seed: u64,
) -> f64 {
    let output_start = init.output_start();
    let neuron_count = init.neuron_count;
    let edge_cap = init.edge_cap();

    let mut init_rng = StdRng::seed_from_u64(seed);
    let mut jackpot_net = build_network(init, &mut init_rng);
    let mut jackpot_proj = Int8Projection::new(
        init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200),
    );
    let mut global_eval_rng = StdRng::seed_from_u64(seed + 1000);

    let mut wins = [0u32; N_WORKERS];
    let mut total_accepted = 0u32;

    for round in 0..rounds {
        let eval_seed = seed.wrapping_add(5000).wrapping_add(round as u64);

        // All 7 workers try their specialized mutation in parallel
        let proposals: Vec<(Network, Int8Projection, f64)> = (0..N_WORKERS)
            .into_par_iter()
            .map(|wid| {
                let mut net = jackpot_net.clone();
                let mut proj = jackpot_proj.clone();
                let mut mrng = StdRng::seed_from_u64(
                    seed.wrapping_add(9000).wrapping_add(round as u64 * 100 + wid as u64),
                );

                // Paired eval: before
                let mut erng = StdRng::seed_from_u64(eval_seed);
                let before = eval_accuracy(
                    &mut net, &proj, corpus, 100, &mut erng, sdr,
                    &init.propagation, output_start, neuron_count,
                );

                // Apply specialized mutation
                apply_mutation(wid, &mut net, &mut proj, &mut mrng);

                // Paired eval: after (same segment)
                let mut erng2 = StdRng::seed_from_u64(eval_seed);
                let after = eval_accuracy(
                    &mut net, &proj, corpus, 100, &mut erng2, sdr,
                    &init.propagation, output_start, neuron_count,
                );

                (net, proj, after - before)
            })
            .collect();

        // Density-capped: pick best positive delta
        let best = proposals.iter().enumerate()
            .filter(|(_, (net, _, delta))| {
                if net.edge_count() < edge_cap { *delta >= 0.0 } else { *delta > 0.0 }
            })
            .max_by(|a, b| a.1 .2.partial_cmp(&b.1 .2).unwrap());

        if let Some((wid, (net, proj, _))) = best {
            jackpot_net = net.clone();
            jackpot_proj = proj.clone();
            wins[wid] += 1;
            total_accepted += 1;
        }

        if (round + 1) % 500 == 0 {
            let full = eval_accuracy(
                &mut jackpot_net, &jackpot_proj, corpus, 2000, &mut global_eval_rng,
                sdr, &init.propagation, output_start, neuron_count,
            );
            let rate = total_accepted as f64 / (round + 1) as f64 * 100.0;
            let win_str: String = wins.iter().enumerate()
                .map(|(i, &w)| format!("{}={}", &WORKER_NAMES[i][..3], w))
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "  round {:>6}: |{}| {:.1}%  edges={}  accept={:.0}%  [{}]",
                round + 1, bar(full, 0.30, 30), full * 100.0,
                jackpot_net.edge_count(), rate, win_str,
            );
        }
    }

    // Final win summary
    println!("\n  Worker wins:");
    for (i, &w) in wins.iter().enumerate() {
        println!("    {:<12} {:>5} wins ({:.1}%)", WORKER_NAMES[i], w, w as f64 / rounds as f64 * 100.0);
    }

    eval_accuracy(
        &mut jackpot_net, &jackpot_proj, corpus, 5000, &mut global_eval_rng,
        sdr, &init.propagation, output_start, neuron_count,
    )
}

// ---------------------------------------------------------------------------
// Baseline: single-thread evolution_step (same total mutation attempts)
// ---------------------------------------------------------------------------

fn run_baseline(
    corpus: &[u8],
    init: &InitConfig,
    sdr: &SdrTable,
    steps: usize,
    seed: u64,
) -> f64 {
    let output_start = init.output_start();
    let neuron_count = init.neuron_count;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(init, &mut rng);
    let mut proj = Int8Projection::new(
        init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200),
    );
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let evo_config = init.evolution_config();

    let mut accepted = 0u32;
    let mut rejected = 0u32;

    for step in 0..steps {
        let outcome = evolution_step(
            &mut net, &mut proj, &mut rng, &mut eval_rng,
            |net, proj, eval_rng| {
                eval_accuracy(net, proj, corpus, 100, eval_rng, sdr,
                    &init.propagation, output_start, neuron_count)
            },
            &evo_config,
        );
        match outcome {
            StepOutcome::Accepted => accepted += 1,
            StepOutcome::Rejected => rejected += 1,
            StepOutcome::Skipped => {}
        }

        if (step + 1) % 5000 == 0 {
            let full = eval_accuracy(
                &mut net, &proj, corpus, 2000, &mut eval_rng, sdr,
                &init.propagation, output_start, neuron_count,
            );
            let tot = accepted + rejected;
            let rate = if tot > 0 { accepted as f64 / tot as f64 * 100.0 } else { 0.0 };
            println!(
                "  [baseline]  step  {:>6}: |{}| {:.1}%  edges={}  accept={:.0}%",
                step + 1, bar(full, 0.30, 30), full * 100.0, net.edge_count(), rate,
            );
        }
    }

    eval_accuracy(
        &mut net, &proj, corpus, 5000, &mut eval_rng, sdr,
        &init.propagation, output_start, neuron_count,
    )
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let init = InitConfig::phi(256);
    let seed = 42u64;
    let rounds = 15000;

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let sdr = SdrTable::new(CHARS, init.neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(seed + 100)).unwrap();

    println!("=== Specialized Cooperative Evolution v3 ===");
    println!("  H={}, workers={} (1 per mutation type)", init.neuron_count, N_WORKERS);
    println!("  Types: {:?}", WORKER_NAMES);
    println!("  Rounds={rounds}, seed={seed}, edge_cap={}\n", init.edge_cap());

    // --- Specialized cooperative ---
    let t0 = Instant::now();
    let spec_acc = run_specialized(&corpus, &init, &sdr, rounds, seed);
    let spec_time = t0.elapsed();
    println!("\n  SPECIALIZED FINAL: {:.1}%  ({:.1}s)\n", spec_acc * 100.0, spec_time.as_secs_f64());

    // --- Baseline (same total mutations: rounds × 7) ---
    let baseline_steps = rounds * N_WORKERS;
    println!("--- Baseline: {} steps (= {} rounds × {} workers) ---\n", baseline_steps, rounds, N_WORKERS);
    let t1 = Instant::now();
    let baseline_acc = run_baseline(&corpus, &init, &sdr, baseline_steps, seed);
    let baseline_time = t1.elapsed();
    println!("\n  BASELINE FINAL: {:.1}%  ({:.1}s)\n", baseline_acc * 100.0, baseline_time.as_secs_f64());

    // --- Summary ---
    println!("=== COMPARISON ===");
    println!("  Specialized: {:.1}%  ({:.1}s)", spec_acc * 100.0, spec_time.as_secs_f64());
    println!("  Baseline:    {:.1}%  ({:.1}s)", baseline_acc * 100.0, baseline_time.as_secs_f64());
    println!("  Delta:       {:+.1}pp", (spec_acc - baseline_acc) * 100.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_mutation_all_types() {
        let init = InitConfig::phi(64);
        let mut rng = StdRng::seed_from_u64(42);
        let mut net = build_network(&init, &mut rng);
        let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut rng);

        // Each worker type should not panic
        for wid in 0..N_WORKERS {
            let _ = apply_mutation(wid, &mut net, &mut proj, &mut rng);
        }
    }

    #[test]
    fn worker_names_match_count() {
        assert_eq!(WORKER_NAMES.len(), N_WORKERS);
    }
}
