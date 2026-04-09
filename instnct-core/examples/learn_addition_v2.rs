//! Addition learning v2: three variants to isolate what works.
//!
//! A: Sequential (A then B, 2 tokens) — baseline, tests memory
//! B: Parallel (A,B encoded as single combined SDR) — tests pure computation
//! C: Small (A,B in 0-4, sum 0-8) — easier version of sequential
//!
//! All use smooth fitness + jackpot=9, 50K steps, 6 seeds each.
//!
//! Run: cargo run --example learn_addition_v2 --release

use instnct_core::{
    apply_mutation, build_network, cosine_to_onehot, InitConfig, Int8Projection, Network, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 50_000;

// ---- Problem definitions ----

struct Problem {
    /// Input tokens to feed sequentially (1 or 2)
    inputs: Vec<usize>,
    /// Target output class
    target: usize,
}

#[derive(Clone, Copy)]
enum Variant {
    /// A then B as 2 tokens, predict A+B (0-18). Tests memory.
    Sequential,
    /// (A,B) as single token from 100 combined patterns, predict A+B (0-18). Tests computation.
    Parallel,
    /// A then B in 0-4, predict A+B (0-8). Easier sequential.
    SmallSeq,
}

impl Variant {
    fn name(&self) -> &str {
        match self {
            Self::Sequential => "seq_10x10",
            Self::Parallel => "par_10x10",
            Self::SmallSeq => "seq_5x5",
        }
    }

    fn input_symbols(&self) -> usize {
        match self {
            Self::Sequential => 10,
            Self::Parallel => 100,  // 10*10 combined pairs
            Self::SmallSeq => 5,
        }
    }

    fn output_classes(&self) -> usize {
        match self {
            Self::Sequential => 19,
            Self::Parallel => 19,
            Self::SmallSeq => 9,  // 0-8
        }
    }

    fn problems(&self) -> Vec<Problem> {
        match self {
            Self::Sequential => {
                let mut p = Vec::new();
                for a in 0..10 {
                    for b in 0..10 {
                        p.push(Problem { inputs: vec![a, b], target: a + b });
                    }
                }
                p
            }
            Self::Parallel => {
                let mut p = Vec::new();
                for a in 0..10 {
                    for b in 0..10 {
                        // Single combined token: index = a*10 + b
                        p.push(Problem { inputs: vec![a * 10 + b], target: a + b });
                    }
                }
                p
            }
            Self::SmallSeq => {
                let mut p = Vec::new();
                for a in 0..5 {
                    for b in 0..5 {
                        p.push(Problem { inputs: vec![a, b], target: a + b });
                    }
                }
                p
            }
        }
    }

    fn random_baseline(&self) -> f64 {
        1.0 / self.output_classes() as f64
    }

    fn freq_baseline(&self) -> f64 {
        // Most common target / total problems
        let problems = self.problems();
        let mut counts = vec![0u32; self.output_classes()];
        for p in &problems { counts[p.target] += 1; }
        let max_count = counts.iter().copied().max().unwrap_or(1);
        max_count as f64 / problems.len() as f64
    }
}

// ---- Eval and fitness ----

fn evaluate(
    net: &mut Network, proj: &Int8Projection, sdr: &SdrTable,
    init: &InitConfig, problems: &[Problem],
) -> (f64, f64) {
    let mut correct = 0u32;
    let mut total_cos = 0.0f64;
    for p in problems {
        net.reset();
        for &inp in &p.inputs {
            net.propagate(sdr.pattern(inp), &init.propagation).unwrap();
        }
        let charge = &net.charge()[init.output_start()..init.neuron_count];
        if proj.predict(charge) == p.target { correct += 1; }
        total_cos += cosine_to_onehot(&proj.raw_scores(charge), p.target);
    }
    (correct as f64 / problems.len() as f64, total_cos / problems.len() as f64)
}

fn eval_fitness(
    net: &mut Network, proj: &Int8Projection, sdr: &SdrTable,
    init: &InitConfig, problems: &[Problem], rng: &mut StdRng, sample: usize,
) -> f64 {
    let mut total = 0.0f64;
    for _ in 0..sample {
        let p = &problems[rng.gen_range(0..problems.len())];
        net.reset();
        for &inp in &p.inputs {
            net.propagate(sdr.pattern(inp), &init.propagation).unwrap();
        }
        let scores = proj.raw_scores(&net.charge()[init.output_start()..init.neuron_count]);
        total += cosine_to_onehot(&scores, p.target);
    }
    total / sample as f64
}

// ---- Run ----

struct Config { variant: Variant, seed: u64 }

#[allow(dead_code)]
struct RunResult {
    variant_name: String, seed: u64,
    final_acc: f64, peak_acc: f64,
    final_edges: usize, accepted: u32,
}

fn run_one(cfg: &Config) -> RunResult {
    let variant = cfg.variant;
    let problems = variant.problems();
    let init = InitConfig::phi(256);
    let edge_cap = init.edge_cap();

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(
        init.phi_dim, variant.output_classes(),
        &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(
        variant.input_symbols(), init.neuron_count, init.input_end(),
        SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    let mut peak_acc = 0.0f64;
    let mut accepted = 0u32;

    for step in 0..STEPS {
        let snap = eval_rng.clone();
        let before = eval_fitness(&mut net, &proj, &sdr, &init, &problems, &mut eval_rng, 20);
        eval_rng = snap;

        let parent_net = net.save_state();
        let parent_proj = proj.clone();
        let edges_before = net.edge_count();

        let mut best_delta = f64::NEG_INFINITY;
        let mut best_net = None;
        let mut best_proj = None;
        let mut any_mutated = false;

        for c in 0..9 {
            net.restore_state(&parent_net);
            proj = parent_proj.clone();
            let mut cand_rng = StdRng::seed_from_u64(
                cfg.seed.wrapping_add(300).wrapping_add(step as u64 * 100 + c as u64));
            if !apply_mutation(&mut net, &mut proj, &mut cand_rng) { continue; }
            any_mutated = true;

            let cs = eval_rng.clone();
            let after = eval_fitness(&mut net, &proj, &sdr, &init, &problems, &mut eval_rng, 20);
            eval_rng = cs;

            let eg = net.edge_count() > edges_before;
            let cap = !eg || net.edge_count() <= edge_cap;
            let d = after - before;
            if d > best_delta && cap {
                best_delta = d;
                best_net = Some(net.save_state());
                best_proj = Some(proj.clone());
            }
        }

        net.restore_state(&parent_net);
        proj = parent_proj.clone();
        let _ = eval_fitness(&mut net, &proj, &sdr, &init, &problems, &mut eval_rng, 20);

        if !any_mutated {
            net.restore_state(&parent_net); proj = parent_proj; continue;
        }
        if best_delta > 0.0 {
            if let (Some(ns), Some(ps)) = (best_net, best_proj) {
                net.restore_state(&ns); proj = ps; accepted += 1;
            }
        } else {
            net.restore_state(&parent_net); proj = parent_proj;
        }

        if (step + 1) % 10_000 == 0 {
            let (acc, cos) = evaluate(&mut net, &proj, &sdr, &init, &problems);
            if acc > peak_acc { peak_acc = acc; }
            println!("  {} seed={} step {:>5}: acc={:.0}% ({}/{}) cos={:.4} edges={} accepted={}",
                variant.name(), cfg.seed, step + 1,
                acc * 100.0, (acc * problems.len() as f64).round() as u32,
                problems.len(), cos, net.edge_count(), accepted);
        }
    }

    let (final_acc, _) = evaluate(&mut net, &proj, &sdr, &init, &problems);
    if final_acc > peak_acc { peak_acc = final_acc; }
    println!("  {} seed={} FINAL: acc={:.0}% peak={:.0}% edges={} accepted={}",
        variant.name(), cfg.seed, final_acc * 100.0, peak_acc * 100.0,
        net.edge_count(), accepted);

    RunResult {
        variant_name: variant.name().to_string(), seed: cfg.seed,
        final_acc, peak_acc, final_edges: net.edge_count(), accepted,
    }
}

fn main() {
    rayon::ThreadPoolBuilder::new().num_threads(6).build_global().ok();

    println!("=== Learn Addition v2: Isolate Memory vs Computation ===");
    println!("  A: seq_10x10 — A then B, predict A+B (tests memory)");
    println!("  B: par_10x10 — (A,B) as one token, predict A+B (tests computation)");
    println!("  C: seq_5x5   — small A,B in 0-4 (easier sequential)");
    println!("  All: smooth fitness + jackpot=9, 50K steps\n");

    let variants = [Variant::Sequential, Variant::Parallel, Variant::SmallSeq];
    for v in &variants {
        println!("  {}: {} inputs, {} outputs, random={:.1}%, freq={:.1}%",
            v.name(), v.input_symbols(), v.output_classes(),
            v.random_baseline() * 100.0, v.freq_baseline() * 100.0);
    }
    println!();

    let seeds = [42u64, 123, 7, 1042, 555, 8042];

    let mut configs: Vec<Config> = Vec::new();
    for &v in &variants {
        for &s in &seeds {
            configs.push(Config { variant: v, seed: s });
        }
    }
    println!("  {} configs: 3 variants x {} seeds\n", configs.len(), seeds.len());

    let start = Instant::now();
    let results: Vec<RunResult> = configs.par_iter().map(run_one).collect();
    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== SUMMARY ===\n");
    println!("{:<12} {:>7} {:>7} {:>7} {:>7} {:>8}",
        "Variant", "Mean%", "Best%", "Peak%", "Edges", "Accepted");
    println!("{}", "-".repeat(55));

    for v in &variants {
        let g: Vec<_> = results.iter().filter(|r| r.variant_name == v.name()).collect();
        let n = g.len() as f64;
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / n;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        let acc_mean = g.iter().map(|r| r.accepted as f64).sum::<f64>() / n;
        println!("{:<12} {:>6.0}% {:>6.0}% {:>6.0}% {:>7} {:>8.0}",
            v.name(), mean * 100.0, best * 100.0, peak * 100.0, edges, acc_mean);
    }

    println!("\nBaselines:");
    for v in &variants {
        println!("  {}: random={:.1}%, freq={:.1}%",
            v.name(), v.random_baseline() * 100.0, v.freq_baseline() * 100.0);
    }

    println!("\nPer-seed:");
    println!("{:<12} {:>6} {:>7} {:>7} {:>7}",
        "Variant", "Seed", "Final%", "Peak%", "Edges");
    println!("{}", "-".repeat(45));
    for r in &results {
        println!("{:<12} {:>6} {:>6.0}% {:>6.0}% {:>7}",
            r.variant_name, r.seed, r.final_acc * 100.0,
            r.peak_acc * 100.0, r.final_edges);
    }

    println!("\n  Total time: {:.0}s", elapsed);
}
