//! Accept ties sweep: does strict-only vs permissive matter with the new
//! separated edge cap + quality gate?
//!
//! The old density-capped logic mixed cap and quality. Now they're independent:
//! - edge_cap: hard limit (always enforced)
//! - accept_ties: true (>=) vs false (>)
//!
//! Run: cargo run --example accept_sweep --release -- <corpus-path>

use instnct_core::{load_corpus, 
    build_network, evolution_step, EvolutionConfig, InitConfig, Int8Projection, Network,
    SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;


#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network, proj: &Int8Projection, corpus: &[u8], len: usize,
    rng: &mut StdRng, sdr: &SdrTable, init: &InitConfig,
) -> f64 {
    if corpus.len() <= len { return 0.0; }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), &init.propagation).unwrap();
        if proj.predict(&net.charge()[init.output_start()..init.neuron_count]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

#[allow(dead_code)]
struct RunResult {
    accept_ties: bool,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    peak_step: usize,
    final_edges: usize,
    accept_rate: f64,
}

fn run_one(accept_ties: bool, seed: u64, steps: usize, corpus: &[u8]) -> RunResult {
    let init = InitConfig::phi(256);
    let evo = EvolutionConfig { edge_cap: init.edge_cap(), accept_ties };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let sdr = SdrTable::new(CHARS, init.neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(seed + 100)).unwrap();

    let mut peak_acc = 0.0f64;
    let mut peak_step = 0usize;
    let mut accepted = 0u32;
    let mut total = 0u32;

    for step in 0..steps {
        let outcome = evolution_step(
            &mut net, &mut proj, &mut rng, &mut eval_rng,
            |n, p, e| eval_accuracy(n, p, corpus, 100, e, &sdr, &init),
            &evo,
        );
        match outcome {
            StepOutcome::Accepted => { accepted += 1; total += 1; }
            StepOutcome::Rejected => { total += 1; }
            StepOutcome::Skipped => {}
        }

        if (step + 1) % 10_000 == 0 {
            let mut cr = StdRng::seed_from_u64(seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut cr, &sdr, &init);
            if acc > peak_acc { peak_acc = acc; peak_step = step + 1; }
            let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };
            let label = if accept_ties { "ties=yes" } else { "ties=no " };
            println!("  {} seed={:<4} step {:>5}: {:.1}%  edges={}  accept={:.0}%",
                label, seed, step + 1, acc * 100.0, net.edge_count(), rate);
        }
    }

    let mut fr = StdRng::seed_from_u64(seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr, &sdr, &init);
    let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };
    let label = if accept_ties { "ties=yes" } else { "ties=no " };
    println!("  {} seed={:<4} FINAL: {:.1}%  peak={:.1}% @{}  edges={}  accept={:.0}%",
        label, seed, final_acc * 100.0, peak_acc * 100.0, peak_step, net.edge_count(), rate);

    RunResult { accept_ties, seed, final_acc, peak_acc, peak_step,
        final_edges: net.edge_count(), accept_rate: rate }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let seeds: Vec<u64> = (0..6).map(|i| 42 + i * 1000).collect();
    let steps = 30_000;

    let mut configs: Vec<(bool, u64)> = Vec::new();
    for &ties in &[true, false] {
        for &s in &seeds {
            configs.push((ties, s));
        }
    }

    println!("=== Accept Sweep: {} configs, {} steps, 7% edge cap ===\n", configs.len(), steps);

    let results: Vec<RunResult> = configs.par_iter()
        .map(|&(ties, s)| run_one(ties, s, steps, &corpus))
        .collect();

    println!("\n=== SUMMARY ===\n");
    println!("{:<10} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "mode", "mean%", "best%", "peak%", "edges", "accept%");
    println!("{}", "-".repeat(56));

    for &ties in &[true, false] {
        let g: Vec<_> = results.iter().filter(|r| r.accept_ties == ties).collect();
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / g.len() as f64;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        let acc_rate = g.iter().map(|r| r.accept_rate).sum::<f64>() / g.len() as f64;
        let label = if ties { "ties=yes" } else { "ties=no" };
        println!("{:<10} {:>7.1}% {:>7.1}% {:>7.1}% {:>8} {:>7.0}%",
            label, mean * 100.0, best * 100.0, peak * 100.0, edges, acc_rate);
    }

    println!("\nPer-seed:");
    println!("{:<10} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "mode", "seed", "final%", "peak%", "p_step", "edges", "accept%");
    println!("{}", "-".repeat(60));
    for r in &results {
        let label = if r.accept_ties { "ties=yes" } else { "ties=no" };
        println!("{:<10} {:>6} {:>7.1}% {:>7.1}% {:>8} {:>8} {:>7.0}%",
            label, r.seed, r.final_acc * 100.0, r.peak_acc * 100.0,
            r.peak_step, r.final_edges, r.accept_rate);
    }
}
