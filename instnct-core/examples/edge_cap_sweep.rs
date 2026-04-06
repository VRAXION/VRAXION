//! Edge cap sweep: does raising the density cap break the ~17% ceiling?
//!
//! The mutation_profile run (100K steps, no cap) peaked at 24.2% with 7694 edges
//! (11.7% density). The standard 7% cap (4587 edges) may be artificially limiting.
//!
//! Tests cap_pct = {7, 10, 12, 15} × 3 seeds × 50K steps.
//!
//! Run: cargo run --example edge_cap_sweep --release -- <corpus-path>

use instnct_core::{
    build_network, evolution_step, InitConfig, Int8Projection, Network, PropagationConfig,
    SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::fs;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = fs::read(path).expect("cannot read corpus");
    raw.iter()
        .filter_map(|&b| {
            if b.is_ascii_lowercase() { Some(b - b'a') }
            else if b.is_ascii_uppercase() { Some(b.to_ascii_lowercase() - b'a') }
            else if b == b' ' || b == b'\n' || b == b'\t' { Some(26) }
            else { None }
        })
        .collect()
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
    if corpus.len() <= len { return 0.0; }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
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

#[allow(dead_code)]
struct RunResult {
    cap_pct: usize,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    peak_step: usize,
    final_edges: usize,
    peak_edges: usize,
    accept_rate: f64,
}

fn run_one(cap_pct: usize, seed: u64, steps: usize, corpus: &[u8]) -> RunResult {
    let mut init = InitConfig::phi(256);
    init.edge_cap_pct = cap_pct;
    let output_start = init.output_start();
    let neuron_count = init.neuron_count;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let sdr = SdrTable::new(CHARS, neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(seed + 100)).unwrap();

    let evo_config = init.evolution_config();
    let mut accepted = 0u32;
    let mut total = 0u32;
    let mut peak_acc = 0.0f64;
    let mut peak_step = 0usize;
    let mut peak_edges = 0usize;

    for step in 0..steps {
        let outcome = evolution_step(
            &mut net, &mut proj, &mut rng, &mut eval_rng,
            |net, proj, eval_rng| {
                eval_accuracy(net, proj, corpus, 100, eval_rng, &sdr,
                    &init.propagation, output_start, neuron_count)
            },
            &evo_config,
        );
        match outcome {
            StepOutcome::Accepted => { accepted += 1; total += 1; }
            StepOutcome::Rejected => { total += 1; }
            StepOutcome::Skipped => {}
        }

        if (step + 1) % 10_000 == 0 {
            let mut check_rng = StdRng::seed_from_u64(seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut check_rng,
                &sdr, &init.propagation, output_start, neuron_count);
            if acc > peak_acc {
                peak_acc = acc;
                peak_step = step + 1;
                peak_edges = net.edge_count();
            }
            let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };
            println!("  cap={:<3}% seed={:<4} step {:>5}: {:.1}%  edges={}  accept={:.0}%",
                cap_pct, seed, step + 1, acc * 100.0, net.edge_count(), rate);
        }
    }

    let mut final_rng = StdRng::seed_from_u64(seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut final_rng,
        &sdr, &init.propagation, output_start, neuron_count);
    let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };

    println!("  cap={:<3}% seed={:<4} FINAL: {:.1}%  peak={:.1}% @{}  edges={}",
        cap_pct, seed, final_acc * 100.0, peak_acc * 100.0, peak_step, net.edge_count());

    RunResult {
        cap_pct, seed, final_acc, peak_acc, peak_step,
        final_edges: net.edge_count(), peak_edges, accept_rate: rate,
    }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path);
    println!("  {} chars\n", corpus.len());

    let caps = [7, 10, 12, 15];
    let seeds = [42u64, 123, 7];
    let steps = 50_000;

    let mut configs: Vec<(usize, u64)> = Vec::new();
    for &cap in &caps {
        for &seed in &seeds {
            configs.push((cap, seed));
        }
    }

    println!("=== Edge Cap Sweep: {} configs, {} steps ===", configs.len(), steps);
    println!("caps: {:?}, seeds: {:?}\n", caps, seeds);

    let results: Vec<RunResult> = configs.par_iter()
        .map(|&(cap, seed)| run_one(cap, seed, steps, &corpus))
        .collect();

    println!("\n=== SUMMARY ===\n");
    println!("{:<8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8}",
        "cap%", "mean%", "best%", "worst%", "peak%", "edges", "accept%");
    println!("{}", "-".repeat(68));

    for &cap in &caps {
        let group: Vec<_> = results.iter().filter(|r| r.cap_pct == cap).collect();
        let mean = group.iter().map(|r| r.final_acc).sum::<f64>() / group.len() as f64;
        let best = group.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let worst = group.iter().map(|r| r.final_acc).fold(1.0f64, f64::min);
        let peak = group.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = group.iter().map(|r| r.final_edges).sum::<usize>() / group.len();
        let accept = group.iter().map(|r| r.accept_rate).sum::<f64>() / group.len() as f64;

        println!("{:<8} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>10} {:>7.0}%",
            format!("{}%", cap), mean * 100.0, best * 100.0, worst * 100.0,
            peak * 100.0, edges, accept);
    }

    println!("\nPer-seed detail:");
    println!("{:<8} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "cap%", "seed", "final%", "peak%", "p_step", "edges");
    println!("{}", "-".repeat(52));
    for r in &results {
        println!("{:<8} {:>6} {:>7.1}% {:>7.1}% {:>8} {:>8}",
            format!("{}%", r.cap_pct), r.seed,
            r.final_acc * 100.0, r.peak_acc * 100.0, r.peak_step, r.final_edges);
    }
}
