//! SDR active percentage sweep: does lower input activity let the network
//! rely more on internal state (context memory)?
//!
//! Hypothesis: 20% SDR overwhelms residual charge from previous chars.
//! Lower SDR = weaker current-char signal = more weight on prior context.
//!
//! Sweep: SDR active = {5, 10, 15, 20, 30}% × 3 seeds × 30K steps
//!
//! Run: cargo run --example sdr_sweep --release -- <corpus-path>

use instnct_core::{load_corpus, 
    build_network, evolution_step, InitConfig, Int8Projection, Network, PropagationConfig,
    SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHARS: usize = 27;


#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network, proj: &Int8Projection, corpus: &[u8], len: usize,
    rng: &mut StdRng, sdr: &SdrTable, config: &PropagationConfig,
    os: usize, nc: usize,
) -> f64 {
    if corpus.len() <= len { return 0.0; }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        if proj.predict(&net.charge()[os..nc]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

#[allow(dead_code)]
struct RunResult {
    sdr_pct: usize,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    peak_step: usize,
    final_edges: usize,
}

fn run_one(sdr_pct: usize, seed: u64, steps: usize, corpus: &[u8]) -> RunResult {
    let init = InitConfig::phi(256);
    let os = init.output_start();
    let nc = init.neuron_count;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);

    let sdr = SdrTable::new(CHARS, nc, init.input_end(), sdr_pct,
        &mut StdRng::seed_from_u64(seed + 100)).unwrap();
    let evo = init.evolution_config();

    let mut peak_acc = 0.0f64;
    let mut peak_step = 0usize;

    for step in 0..steps {
        let _ = evolution_step(&mut net, &mut proj, &mut rng, &mut eval_rng,
            |n, p, e| eval_accuracy(n, p, corpus, 100, e, &sdr, &init.propagation, os, nc),
            &evo);

        if (step + 1) % 10_000 == 0 {
            let mut cr = StdRng::seed_from_u64(seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut cr,
                &sdr, &init.propagation, os, nc);
            if acc > peak_acc { peak_acc = acc; peak_step = step + 1; }
            println!("  sdr={:<3}% seed={:<4} step {:>5}: {:.1}%  edges={}",
                sdr_pct, seed, step + 1, acc * 100.0, net.edge_count());
        }
    }

    let mut fr = StdRng::seed_from_u64(seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr,
        &sdr, &init.propagation, os, nc);
    println!("  sdr={:<3}% seed={:<4} FINAL: {:.1}%  peak={:.1}% @{}",
        sdr_pct, seed, final_acc * 100.0, peak_acc * 100.0, peak_step);

    RunResult { sdr_pct, seed, final_acc, peak_acc, peak_step, final_edges: net.edge_count() }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let sdr_pcts = [5, 10, 15, 20, 30];
    let seeds = [42u64, 123, 7];
    let steps = 30_000;

    let mut configs: Vec<(usize, u64)> = Vec::new();
    for &pct in &sdr_pcts {
        for &s in &seeds {
            configs.push((pct, s));
        }
    }

    println!("=== SDR Sweep: {} configs, {} steps ===\n", configs.len(), steps);

    let results: Vec<RunResult> = configs.par_iter()
        .map(|&(pct, s)| run_one(pct, s, steps, &corpus))
        .collect();

    println!("\n=== SUMMARY ===\n");
    println!("{:<8} {:>8} {:>8} {:>8} {:>8}",
        "sdr%", "mean%", "best%", "peak%", "edges");
    println!("{}", "-".repeat(44));

    for &pct in &sdr_pcts {
        let g: Vec<_> = results.iter().filter(|r| r.sdr_pct == pct).collect();
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / g.len() as f64;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        println!("{:<8} {:>7.1}% {:>7.1}% {:>7.1}% {:>8}",
            format!("{}%", pct), mean * 100.0, best * 100.0, peak * 100.0, edges);
    }
}
