//! Addition from EMPTY network — 0 edges, build everything from scratch.
//!
//! Every edge the evolution adds must EARN its place.
//! With 0 starting edges, each add has maximum impact on charge patterns.
//! If this works → the network builds real computational circuits.
//!
//! Also runs a "prefilled" control (standard 5% density) for comparison.
//!
//! Run: cargo run --example addition_empty --release

use instnct_core::{
    apply_mutation, build_network, cosine_to_onehot, InitConfig, Int8Projection, Network, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const INPUT_SYMBOLS: usize = 5;
const OUTPUT_CLASSES: usize = 9;
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 50_000;

fn all_problems() -> Vec<(usize, usize, usize)> {
    let mut p = Vec::new();
    for a in 0..5 { for b in 0..5 { p.push((a, b, a + b)); } }
    p
}

fn eval_fitness(
    net: &mut Network, proj: &Int8Projection, sdr: &SdrTable,
    init: &InitConfig, problems: &[(usize, usize, usize)], rng: &mut StdRng,
) -> f64 {
    let mut total = 0.0f64;
    for _ in 0..20 {
        let &(a, b, sum) = &problems[rng.gen_range(0..problems.len())];
        net.reset();
        net.propagate(sdr.pattern(a), &init.propagation).unwrap();
        net.propagate(sdr.pattern(b), &init.propagation).unwrap();
        let scores = proj.raw_scores(&net.charge()[init.output_start()..init.neuron_count]);
        total += cosine_to_onehot(&scores, sum);
    }
    total / 20.0
}

fn evaluate_full(
    net: &mut Network, proj: &Int8Projection, sdr: &SdrTable,
    init: &InitConfig, problems: &[(usize, usize, usize)],
) -> (u32, Vec<bool>) {
    let mut correct = 0u32;
    let mut results = Vec::new();
    for &(a, b, sum) in problems {
        net.reset();
        net.propagate(sdr.pattern(a), &init.propagation).unwrap();
        net.propagate(sdr.pattern(b), &init.propagation).unwrap();
        let ok = proj.predict(&net.charge()[init.output_start()..init.neuron_count]) == sum;
        if ok { correct += 1; }
        results.push(ok);
    }
    (correct, results)
}

#[derive(Clone, Copy)]
enum Variant {
    Empty,     // 0 edges, 0 chains — build from scratch
    Prefilled, // standard 5% density + chains
}

impl Variant {
    fn name(&self) -> &str {
        match self {
            Self::Empty => "empty_0",
            Self::Prefilled => "prefill_5%",
        }
    }

    fn init_config(&self) -> InitConfig {
        match self {
            Self::Empty => {
                let mut cfg = InitConfig::phi(256);
                cfg.chain_count = 0;
                cfg.density_pct = 0;
                cfg
            }
            Self::Prefilled => InitConfig::phi(256),
        }
    }
}

struct Config { variant: Variant, seed: u64 }

#[allow(dead_code)]
struct RunResult {
    variant_name: String, seed: u64,
    final_acc: f64, peak_acc: f64,
    final_edges: usize, accepted: u32,
}

fn run_one(cfg: &Config) -> RunResult {
    let problems = all_problems();
    let init = cfg.variant.init_config();
    let edge_cap = init.edge_cap();

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, OUTPUT_CLASSES,
        &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(INPUT_SYMBOLS, init.neuron_count, init.input_end(),
        SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    let start_edges = net.edge_count();
    let mut peak_acc = 0.0f64;
    let mut accepted = 0u32;

    for step in 0..STEPS {
        let snap = eval_rng.clone();
        let before = eval_fitness(&mut net, &proj, &sdr, &init, &problems, &mut eval_rng);
        eval_rng = snap;

        let parent_net = net.save_state();
        let parent_proj = proj.clone();
        let edges_before = net.edge_count();

        let mut best_delta = f64::NEG_INFINITY;
        let mut best_net = None;
        let mut best_proj = None;
        let mut any = false;

        for c in 0..9 {
            net.restore_state(&parent_net);
            proj = parent_proj.clone();
            let mut cr = StdRng::seed_from_u64(
                cfg.seed.wrapping_add(300).wrapping_add(step as u64 * 100 + c as u64));
            if !apply_mutation(&mut net, &mut proj, &mut cr) { continue; }
            any = true;
            let cs = eval_rng.clone();
            let after = eval_fitness(&mut net, &proj, &sdr, &init, &problems, &mut eval_rng);
            eval_rng = cs;
            let eg = net.edge_count() > edges_before;
            let cap = !eg || net.edge_count() <= edge_cap;
            let d = after - before;
            if d > best_delta && cap {
                best_delta = d; best_net = Some(net.save_state()); best_proj = Some(proj.clone());
            }
        }
        net.restore_state(&parent_net); proj = parent_proj.clone();
        let _ = eval_fitness(&mut net, &proj, &sdr, &init, &problems, &mut eval_rng);
        if !any { net.restore_state(&parent_net); proj = parent_proj; continue; }
        if best_delta > 0.0 {
            if let (Some(ns), Some(ps)) = (best_net, best_proj) {
                net.restore_state(&ns); proj = ps; accepted += 1;
            }
        } else { net.restore_state(&parent_net); proj = parent_proj; }

        if (step + 1) % 5_000 == 0 {
            let (correct, _) = evaluate_full(&mut net, &proj, &sdr, &init, &problems);
            let acc = correct as f64 / 25.0;
            if acc > peak_acc { peak_acc = acc; }
            println!("  {} seed={} step {:>5}: acc={}/25 ({:.0}%) edges={} (started={}) accepted={}",
                cfg.variant.name(), cfg.seed, step + 1, correct,
                acc * 100.0, net.edge_count(), start_edges, accepted);
        }
    }

    let (correct, results) = evaluate_full(&mut net, &proj, &sdr, &init, &problems);
    let final_acc = correct as f64 / 25.0;
    if final_acc > peak_acc { peak_acc = final_acc; }

    // Print prediction grid
    println!("\n  {} seed={} PREDICTION GRID:", cfg.variant.name(), cfg.seed);
    print!("       ");
    for b in 0..5 { print!("  B={}", b); }
    println!();
    for a in 0..5 {
        print!("    A={}", a);
        for b in 0..5 {
            let idx = a * 5 + b;
            let mark = if results[idx] { " +" } else { " X" };
            net.reset();
            net.propagate(sdr.pattern(a), &init.propagation).unwrap();
            net.propagate(sdr.pattern(b), &init.propagation).unwrap();
            let pred = proj.predict(&net.charge()[init.output_start()..init.neuron_count]);
            print!("  {:>2}{}", pred, mark);
        }
        println!();
    }

    println!("  {} seed={} FINAL: {}/25 ({:.0}%) peak={:.0}% edges={} (started={}) accepted={}",
        cfg.variant.name(), cfg.seed, correct, final_acc * 100.0,
        peak_acc * 100.0, net.edge_count(), start_edges, accepted);

    RunResult {
        variant_name: cfg.variant.name().to_string(), seed: cfg.seed,
        final_acc, peak_acc, final_edges: net.edge_count(), accepted,
    }
}

fn main() {
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global().ok();

    println!("=== Addition: Empty Start vs Prefilled ===");
    println!("  Empty: 0 edges, 0 chains — every edge must earn its place");
    println!("  Prefilled: standard 5% density + chain-50 highways");
    println!("  Both: seq_5x5, smooth fitness, jackpot=9, 50K steps\n");

    let seeds = [42u64, 123, 7, 1042, 555, 8042];

    let mut configs: Vec<Config> = Vec::new();
    for &v in &[Variant::Empty, Variant::Prefilled] {
        for &s in &seeds {
            configs.push(Config { variant: v, seed: s });
        }
    }
    println!("  {} configs: 2 variants x {} seeds\n", configs.len(), seeds.len());

    let start = Instant::now();
    let results: Vec<RunResult> = configs.par_iter().map(run_one).collect();
    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== SUMMARY ===\n");
    println!("{:<14} {:>7} {:>7} {:>7} {:>7} {:>8}",
        "Variant", "Mean%", "Best%", "Peak%", "Edges", "Accepted");
    println!("{}", "-".repeat(55));

    for v in &[Variant::Empty, Variant::Prefilled] {
        let g: Vec<_> = results.iter().filter(|r| r.variant_name == v.name()).collect();
        let n = g.len() as f64;
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / n;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        let acc_mean = g.iter().map(|r| r.accepted as f64).sum::<f64>() / n;
        println!("{:<14} {:>6.0}% {:>6.0}% {:>6.0}% {:>7} {:>8.0}",
            v.name(), mean * 100.0, best * 100.0, peak * 100.0, edges, acc_mean);
    }

    println!("\nPer-seed:");
    println!("{:<14} {:>6} {:>7} {:>7} {:>7} {:>8}",
        "Variant", "Seed", "Final%", "Peak%", "Edges", "Accepted");
    println!("{}", "-".repeat(55));
    for r in &results {
        println!("{:<14} {:>6} {:>6.0}% {:>6.0}% {:>7} {:>8}",
            r.variant_name, r.seed, r.final_acc * 100.0,
            r.peak_acc * 100.0, r.final_edges, r.accepted);
    }

    println!("\n  Baselines: random={:.1}%, freq={:.1}%",
        1.0 / OUTPUT_CLASSES as f64 * 100.0, 20.0);
    println!("  Total time: {:.0}s", elapsed);
}
