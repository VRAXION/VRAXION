//! Addition scale test: does empty-start addition scale from 5x5 to 10x10?
//!
//! 5x5 (0-4 + 0-4): 80% proven with 83 edges
//! 10x10 (0-9 + 0-9): 100 problems, 19 output classes — harder
//!
//! Both from empty start (InitConfig::empty). Smooth fitness + jackpot=9.
//!
//! Run: cargo run --example addition_scale --release

use instnct_core::{
    apply_mutation, build_network, cosine_to_onehot, InitConfig, Int8Projection, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 50_000;

#[derive(Clone, Copy)]
enum Scale { Small, Large }
impl Scale {
    fn name(&self) -> &str { match self { Self::Small => "5x5", Self::Large => "10x10" } }
    fn digits(&self) -> usize { match self { Self::Small => 5, Self::Large => 10 } }
    fn max_sum(&self) -> usize { match self { Self::Small => 9, Self::Large => 19 } }
    fn problems(&self) -> Vec<(usize, usize, usize)> {
        let d = self.digits();
        let mut p = Vec::new();
        for a in 0..d { for b in 0..d { p.push((a, b, a + b)); } }
        p
    }
}

struct Config { scale: Scale, seed: u64 }

#[allow(dead_code)]
struct RunResult {
    scale_name: String, seed: u64,
    final_acc: f64, peak_acc: f64, final_edges: usize, accepted: u32,
}

fn run_one(cfg: &Config) -> RunResult {
    let problems = cfg.scale.problems();
    let n_inputs = cfg.scale.digits();
    let n_outputs = cfg.scale.max_sum() + 1;
    let init = InitConfig::empty(256);
    let edge_cap = init.edge_cap();

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, n_outputs,
        &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(n_inputs, init.neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    let mut peak_acc = 0.0f64;
    let mut accepted = 0u32;

    for step in 0..STEPS {
        // Fitness: sample 20 random problems
        let snap = eval_rng.clone();
        let before = {
            let mut t = 0.0f64;
            for _ in 0..20 {
                let &(a, b, sum) = &problems[eval_rng.gen_range(0..problems.len())];
                net.reset();
                net.propagate(sdr.pattern(a), &init.propagation).unwrap();
                net.propagate(sdr.pattern(b), &init.propagation).unwrap();
                let scores = proj.raw_scores(&net.charge()[init.output_start()..init.neuron_count]);
                t += cosine_to_onehot(&scores, sum);
            }
            t / 20.0
        };
        eval_rng = snap;

        let parent_net = net.save_state();
        let parent_proj = proj.clone();
        let edges_before = net.edge_count();

        let mut best_delta = f64::NEG_INFINITY;
        let mut best_net = None;
        let mut best_proj = None;
        let mut any = false;

        for c in 0..9 {
            net.restore_state(&parent_net); proj = parent_proj.clone();
            let mut cr = StdRng::seed_from_u64(cfg.seed.wrapping_add(300).wrapping_add(step as u64 * 100 + c as u64));
            if !apply_mutation(&mut net, &mut proj, &mut cr) { continue; }
            any = true;
            let eg = net.edge_count() > edges_before;
            let cap = !eg || net.edge_count() <= edge_cap;
            if !cap { continue; }

            let cs = eval_rng.clone();
            let after = {
                let mut t = 0.0f64;
                for _ in 0..20 {
                    let &(a, b, sum) = &problems[eval_rng.gen_range(0..problems.len())];
                    net.reset();
                    net.propagate(sdr.pattern(a), &init.propagation).unwrap();
                    net.propagate(sdr.pattern(b), &init.propagation).unwrap();
                    let scores = proj.raw_scores(&net.charge()[init.output_start()..init.neuron_count]);
                    t += cosine_to_onehot(&scores, sum);
                }
                t / 20.0
            };
            eval_rng = cs;
            let d = after - before;
            if d > best_delta { best_delta = d; best_net = Some(net.save_state()); best_proj = Some(proj.clone()); }
        }

        net.restore_state(&parent_net); proj = parent_proj.clone();
        // advance eval_rng
        for _ in 0..20 { let _ = eval_rng.gen_range(0..problems.len()); }

        if !any { continue; }
        if best_delta > 0.0 {
            if let (Some(ns), Some(ps)) = (best_net, best_proj) { net.restore_state(&ns); proj = ps; accepted += 1; }
        } else { net.restore_state(&parent_net); proj = parent_proj; }

        if (step + 1) % 10_000 == 0 {
            let mut correct = 0u32;
            for &(a, b, sum) in &problems {
                net.reset();
                net.propagate(sdr.pattern(a), &init.propagation).unwrap();
                net.propagate(sdr.pattern(b), &init.propagation).unwrap();
                if proj.predict(&net.charge()[init.output_start()..init.neuron_count]) == sum { correct += 1; }
            }
            let acc = correct as f64 / problems.len() as f64;
            if acc > peak_acc { peak_acc = acc; }
            println!("  {} seed={} step {:>5}: acc={}/{} ({:.0}%) edges={} accepted={}",
                cfg.scale.name(), cfg.seed, step + 1, correct, problems.len(),
                acc * 100.0, net.edge_count(), accepted);
        }
    }

    let mut correct = 0u32;
    for &(a, b, sum) in &problems {
        net.reset();
        net.propagate(sdr.pattern(a), &init.propagation).unwrap();
        net.propagate(sdr.pattern(b), &init.propagation).unwrap();
        if proj.predict(&net.charge()[init.output_start()..init.neuron_count]) == sum { correct += 1; }
    }
    let final_acc = correct as f64 / problems.len() as f64;
    if final_acc > peak_acc { peak_acc = final_acc; }

    println!("  {} seed={} FINAL: {}/{} ({:.0}%) peak={:.0}% edges={} accepted={}",
        cfg.scale.name(), cfg.seed, correct, problems.len(), final_acc * 100.0,
        peak_acc * 100.0, net.edge_count(), accepted);

    RunResult {
        scale_name: cfg.scale.name().to_string(), seed: cfg.seed,
        final_acc, peak_acc, final_edges: net.edge_count(), accepted,
    }
}

fn main() {
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global().ok();

    println!("=== Addition Scale: 5x5 vs 10x10 (empty start) ===");
    println!("  Both: empty start, smooth fitness, jackpot=9, 50K steps\n");

    let seeds = [42u64, 123, 7, 1042, 555, 8042];
    let mut configs: Vec<Config> = Vec::new();
    for &s in &[Scale::Small, Scale::Large] {
        for &seed in &seeds { configs.push(Config { scale: s, seed }); }
    }
    println!("  {} configs: 2 scales x {} seeds\n", configs.len(), seeds.len());

    let start = Instant::now();
    let results: Vec<RunResult> = configs.par_iter().map(run_one).collect();
    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== SUMMARY ===\n");
    println!("{:<8} {:>7} {:>7} {:>7} {:>7} {:>8}",
        "Scale", "Mean%", "Best%", "Peak%", "Edges", "Accepted");
    println!("{}", "-".repeat(50));
    for s in &[Scale::Small, Scale::Large] {
        let g: Vec<_> = results.iter().filter(|r| r.scale_name == s.name()).collect();
        let n = g.len() as f64;
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / n;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        let acc = g.iter().map(|r| r.accepted as f64).sum::<f64>() / n;
        println!("{:<8} {:>6.0}% {:>6.0}% {:>6.0}% {:>7} {:>8.0}",
            s.name(), mean * 100.0, best * 100.0, peak * 100.0, edges, acc);
    }
    println!("\nPer-seed:");
    for r in &results {
        println!("  {} seed={}: {:.0}% peak={:.0}% edges={}",
            r.scale_name, r.seed, r.final_acc * 100.0, r.peak_acc * 100.0, r.final_edges);
    }
    println!("\n  Total time: {:.0}s", elapsed);
}
