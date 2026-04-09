//! Teach the network simple addition: A + B → sum.
//!
//! Two tokens per example: first token = digit A (0-9), second = digit B (0-9).
//! After both tokens propagated, the network's charge must predict A+B (0-18).
//!
//! This tests whether the spiking network can form actual computational circuits:
//! - It must REMEMBER A while processing B (working memory)
//! - The correct answer is deterministic (not probabilistic like bigram)
//! - Success proves real computation, not frequency matching
//!
//! Run: cargo run --example learn_addition --release

use instnct_core::{
    apply_mutation, build_network, cosine_to_onehot, InitConfig, Int8Projection, Network, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const INPUT_SYMBOLS: usize = 10;  // digits 0-9
const OUTPUT_CLASSES: usize = 19;  // sums 0-18
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 50_000;

/// All 100 addition problems: (a, b, a+b)
fn all_problems() -> Vec<(usize, usize, usize)> {
    let mut problems = Vec::with_capacity(100);
    for a in 0..10 {
        for b in 0..10 {
            problems.push((a, b, a + b));
        }
    }
    problems
}

/// Evaluate: feed A then B, predict A+B. Returns (accuracy, mean_cosine).
fn evaluate(
    net: &mut Network,
    proj: &Int8Projection,
    sdr: &SdrTable,
    init: &InitConfig,
    problems: &[(usize, usize, usize)],
) -> (f64, f64) {
    let mut correct = 0u32;
    let mut total_cos = 0.0f64;

    for &(a, b, sum) in problems {
        net.reset();
        // Token 1: digit A
        net.propagate(sdr.pattern(a), &init.propagation).unwrap();
        // Token 2: digit B
        net.propagate(sdr.pattern(b), &init.propagation).unwrap();

        // Read prediction from output zone
        let charge = &net.charge()[init.output_start()..init.neuron_count];
        let predicted = proj.predict(charge);
        if predicted == sum {
            correct += 1;
        }

        // Cosine to one-hot target
        let scores = proj.raw_scores(charge);
        let cos = cosine_to_onehot(&scores, sum);
        total_cos += cos;
    }

    let acc = correct as f64 / problems.len() as f64;
    let cos = total_cos / problems.len() as f64;
    (acc, cos)
}

/// Fitness function for evolution: mean cosine over a random subset of problems.
fn eval_fitness(
    net: &mut Network,
    proj: &Int8Projection,
    sdr: &SdrTable,
    init: &InitConfig,
    problems: &[(usize, usize, usize)],
    rng: &mut StdRng,
    sample_size: usize,
) -> f64 {
    let mut total_cos = 0.0f64;
    for _ in 0..sample_size {
        let idx = rng.gen_range(0..problems.len());
        let (a, b, sum) = problems[idx];
        net.reset();
        net.propagate(sdr.pattern(a), &init.propagation).unwrap();
        net.propagate(sdr.pattern(b), &init.propagation).unwrap();

        let scores = proj.raw_scores(&net.charge()[init.output_start()..init.neuron_count]);
        total_cos += cosine_to_onehot(&scores, sum);
    }
    total_cos / sample_size as f64
}

struct Config {
    seed: u64,
    jackpot: usize,
}

#[allow(dead_code)]
struct RunResult {
    seed: u64,
    jackpot: usize,
    final_acc: f64,
    peak_acc: f64,
    final_edges: usize,
    accepted: u32,
}

fn run_one(cfg: &Config, problems: &[(usize, usize, usize)]) -> RunResult {
    let init = InitConfig::phi(256);
    let edge_cap = init.edge_cap();

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(
        init.phi_dim,
        OUTPUT_CLASSES,
        &mut StdRng::seed_from_u64(cfg.seed + 200),
    );
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(
        INPUT_SYMBOLS,
        init.neuron_count,
        init.input_end(),
        SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100),
    )
    .unwrap();

    let mut peak_acc = 0.0f64;
    let mut accepted = 0u32;

    for step in 0..STEPS {
        // Paired eval
        let snap = eval_rng.clone();
        let before = eval_fitness(&mut net, &proj, &sdr, &init, problems, &mut eval_rng, 20);
        eval_rng = snap;

        let parent_net = net.save_state();
        let parent_proj = proj.clone();
        let edges_before = net.edge_count();

        // Jackpot: try N candidates
        let mut best_delta = f64::NEG_INFINITY;
        let mut best_net = None;
        let mut best_proj = None;
        let mut any_mutated = false;

        for c in 0..cfg.jackpot {
            net.restore_state(&parent_net);
            proj = parent_proj.clone();

            let mut cand_rng = StdRng::seed_from_u64(
                cfg.seed.wrapping_add(300).wrapping_add(step as u64 * 100 + c as u64),
            );
            let mutated = apply_mutation(&mut net, &mut proj, &mut cand_rng);
            if !mutated {
                continue;
            }
            any_mutated = true;

            let cand_snap = eval_rng.clone();
            let after = eval_fitness(&mut net, &proj, &sdr, &init, problems, &mut eval_rng, 20);
            eval_rng = cand_snap;

            let edge_grew = net.edge_count() > edges_before;
            let within_cap = !edge_grew || net.edge_count() <= edge_cap;
            let delta = after - before;

            if delta > best_delta && within_cap {
                best_delta = delta;
                best_net = Some(net.save_state());
                best_proj = Some(proj.clone());
            }
        }

        // Advance eval_rng
        net.restore_state(&parent_net);
        proj = parent_proj.clone();
        let _ = eval_fitness(&mut net, &proj, &sdr, &init, problems, &mut eval_rng, 20);

        if !any_mutated {
            net.restore_state(&parent_net);
            proj = parent_proj;
            continue;
        }

        if best_delta > 0.0 {
            if let (Some(ns), Some(ps)) = (best_net, best_proj) {
                net.restore_state(&ns);
                proj = ps;
                accepted += 1;
            }
        } else {
            net.restore_state(&parent_net);
            proj = parent_proj;
        }

        if (step + 1) % 10_000 == 0 {
            let (acc, cos) = evaluate(&mut net, &proj, &sdr, &init, problems);
            if acc > peak_acc { peak_acc = acc; }

            // Show per-digit breakdown
            let mut digit_correct = [0u32; 10];
            let mut digit_total = [0u32; 10];
            for &(a, b, sum) in problems {
                net.reset();
                net.propagate(sdr.pattern(a), &init.propagation).unwrap();
                net.propagate(sdr.pattern(b), &init.propagation).unwrap();
                let predicted = proj.predict(&net.charge()[init.output_start()..init.neuron_count]);
                digit_total[a] += 1;
                if predicted == sum { digit_correct[a] += 1; }
            }

            print!("  jack={} seed={} step {:>5}: acc={:.0}% cos={:.4} edges={} accepted={} | per-A:",
                cfg.jackpot, cfg.seed, step + 1, acc * 100.0, cos, net.edge_count(), accepted);
            for d in 0..10 {
                print!(" {}={}/{}",
                    d, digit_correct[d], digit_total[d]);
            }
            println!();
        }
    }

    let (final_acc, final_cos) = evaluate(&mut net, &proj, &sdr, &init, problems);
    if final_acc > peak_acc { peak_acc = final_acc; }

    println!("  jack={} seed={} FINAL: acc={:.0}% ({}/{}) cos={:.4} peak={:.0}% edges={} accepted={}",
        cfg.jackpot, cfg.seed, final_acc * 100.0,
        (final_acc * 100.0).round() as u32, problems.len(),
        final_cos, peak_acc * 100.0, net.edge_count(), accepted);

    RunResult {
        seed: cfg.seed,
        jackpot: cfg.jackpot,
        final_acc,
        peak_acc,
        final_edges: net.edge_count(),
        accepted,
    }
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .ok();

    println!("=== Learn Addition: A + B ===");
    println!("  Input: two digits 0-9 (fed as two tokens)");
    println!("  Output: sum 0-18 (19 classes)");
    println!("  100 total problems, deterministic target");
    println!("  Smooth fitness (cosine to one-hot) + jackpot\n");

    let problems = all_problems();

    // Baselines
    let random_acc = 1.0 / OUTPUT_CLASSES as f64;
    let freq_acc = 10.0 / 100.0; // most common sum is 9 (10 ways), 10/100
    println!("  Random baseline: {:.1}%", random_acc * 100.0);
    println!("  Frequency baseline (always predict 9): {:.1}%\n", freq_acc * 100.0);

    let seeds = [42u64, 123, 7];

    let mut configs: Vec<Config> = Vec::new();
    // Test with jackpot=9
    for &s in &seeds {
        configs.push(Config { seed: s, jackpot: 9 });
    }

    println!("  {} configs: jackpot=9, {} seeds\n", configs.len(), seeds.len());

    let start = Instant::now();

    let results: Vec<RunResult> = configs.par_iter()
        .map(|cfg| run_one(cfg, &problems))
        .collect();

    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== SUMMARY ===\n");
    println!("{:>6} {:>8} {:>7} {:>7} {:>7} {:>8}",
        "Seed", "Jackpot", "Final%", "Peak%", "Edges", "Accepted");
    println!("{}", "-".repeat(50));
    for r in &results {
        println!("{:>6} {:>8} {:>6.0}% {:>6.0}% {:>7} {:>8}",
            r.seed, r.jackpot, r.final_acc * 100.0, r.peak_acc * 100.0,
            r.final_edges, r.accepted);
    }

    let mean = results.iter().map(|r| r.final_acc).sum::<f64>() / results.len() as f64;
    let peak = results.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
    println!("\n  Mean: {:.0}%  Best peak: {:.0}%", mean * 100.0, peak * 100.0);
    println!("  Random: {:.1}%  Frequency: {:.1}%", random_acc * 100.0, freq_acc * 100.0);
    println!("  Total time: {:.0}s", elapsed);
}
