//! SDR input density sweep: how many neurons should fire per input token?
//!
//! Current default: 20% of phi_dim = ~31/158 neurons.
//! Sweep: 5%, 10%, 20%, 30%, 40%, 60%, 80%.
//!
//! Denser input = stronger signal but more pattern overlap.
//! Sparser input = weaker signal but more distinct patterns.
//!
//! Uses smooth fitness + jackpot=9 on language task.
//!
//! Run: cargo run --example sdr_density_sweep --release -- <corpus-path>

use instnct_core::{
    build_bigram_table, build_network, eval_accuracy, eval_smooth, evolution_step_jackpot,
    load_corpus, InitConfig, Int8Projection, SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::time::Instant;

const CHARS: usize = 27;
const STEPS: usize = 30_000;

struct Config {
    sdr_pct: usize,
    seed: u64,
}

#[allow(dead_code)]
struct RunResult {
    sdr_pct: usize,
    active_bits: usize,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    final_edges: usize,
    accepted: u32,
}

fn run_one(cfg: &Config, corpus: &[u8], bigram: &[Vec<f64>]) -> RunResult {
    let init = InitConfig::phi(256);
    let evo = init.evolution_config();
    let active_bits = init.phi_dim * cfg.sdr_pct / 100;

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(
        init.phi_dim, CHARS,
        &mut StdRng::seed_from_u64(cfg.seed + 200),
    );
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);

    let sdr = match SdrTable::new(
        CHARS, init.neuron_count, init.input_end(), cfg.sdr_pct,
        &mut StdRng::seed_from_u64(cfg.seed + 100),
    ) {
        Ok(s) => s,
        Err(e) => {
            println!("  sdr_pct={}% seed={}: SDR error: {}", cfg.sdr_pct, cfg.seed, e);
            return RunResult {
                sdr_pct: cfg.sdr_pct, active_bits, seed: cfg.seed,
                final_acc: 0.0, peak_acc: 0.0, final_edges: 0, accepted: 0,
            };
        }
    };

    let mut peak_acc = 0.0f64;
    let mut accepted = 0u32;
    let mut total = 0u32;

    for step in 0..STEPS {
        let outcome = evolution_step_jackpot(
            &mut net, &mut proj, &mut rng, &mut eval_rng,
            |n, p, e| eval_smooth(n, p, corpus, 100, e, &sdr,
                &init.propagation, init.output_start(), init.neuron_count, bigram),
            &evo, 9,
        );
        match outcome {
            StepOutcome::Accepted => { accepted += 1; total += 1; }
            StepOutcome::Rejected => { total += 1; }
            StepOutcome::Skipped => {}
        }

        if (step + 1) % 10_000 == 0 {
            let mut cr = StdRng::seed_from_u64(cfg.seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut cr, &sdr,
                &init.propagation, init.output_start(), init.neuron_count);
            if acc > peak_acc { peak_acc = acc; }
            let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };
            println!("  sdr={}% ({} bits) seed={} step {:>5}: acc={:.1}% edges={} accept={:.1}%",
                cfg.sdr_pct, active_bits, cfg.seed, step + 1, acc * 100.0,
                net.edge_count(), rate);
        }
    }

    let mut fr = StdRng::seed_from_u64(cfg.seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr, &sdr,
        &init.propagation, init.output_start(), init.neuron_count);
    if final_acc > peak_acc { peak_acc = final_acc; }
    let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };

    println!("  sdr={}% seed={} FINAL: acc={:.1}% peak={:.1}% edges={} accept={:.1}%",
        cfg.sdr_pct, cfg.seed, final_acc * 100.0, peak_acc * 100.0,
        net.edge_count(), rate);

    RunResult {
        sdr_pct: cfg.sdr_pct, active_bits, seed: cfg.seed,
        final_acc, peak_acc, final_edges: net.edge_count(), accepted,
    }
}

fn main() {
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global().ok();

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });

    println!("=== SDR Input Density Sweep ===");
    println!("  How many neurons fire per input token?");
    println!("  phi_dim=158, sweep: 5% 10% 20% 30% 40% 60% 80%");
    println!("  Smooth fitness + jackpot=9, 30K steps\n");

    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars", corpus.len());
    let bigram = build_bigram_table(&corpus, CHARS);

    let pcts = [5usize, 10, 20, 30, 40, 60, 80];
    let seeds = [42u64, 123, 7];

    // Show what each pct means
    let phi_dim = 158;
    for &p in &pcts {
        println!("  sdr={}% → {} active / {} input neurons", p, phi_dim * p / 100, phi_dim);
    }
    println!();

    let mut configs: Vec<Config> = Vec::new();
    for &p in &pcts {
        for &s in &seeds {
            configs.push(Config { sdr_pct: p, seed: s });
        }
    }
    println!("  {} configs: {} densities × {} seeds\n", configs.len(), pcts.len(), seeds.len());

    let start = Instant::now();
    let results: Vec<RunResult> = configs.par_iter()
        .map(|c| run_one(c, &corpus, &bigram))
        .collect();
    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== SUMMARY ===\n");
    println!("{:>6} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "SDR%", "Bits", "Mean%", "Best%", "Peak%", "Edges", "Accept");
    println!("{}", "-".repeat(55));

    for &p in &pcts {
        let g: Vec<_> = results.iter().filter(|r| r.sdr_pct == p).collect();
        let n = g.len() as f64;
        let bits = g.first().map(|r| r.active_bits).unwrap_or(0);
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / n;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        let accept = g.iter().map(|r| r.accepted as f64).sum::<f64>() / n;
        println!("{:>5}% {:>6} {:>6.1}% {:>6.1}% {:>6.1}% {:>7} {:>7.0}",
            p, bits, mean * 100.0, best * 100.0, peak * 100.0, edges, accept);
    }

    println!("\nPer-seed:");
    println!("{:>6} {:>6} {:>7} {:>7} {:>7}",
        "SDR%", "Seed", "Final%", "Peak%", "Edges");
    println!("{}", "-".repeat(40));
    for r in &results {
        println!("{:>5}% {:>6} {:>6.1}% {:>6.1}% {:>7}",
            r.sdr_pct, r.seed, r.final_acc * 100.0, r.peak_acc * 100.0, r.final_edges);
    }

    println!("\n  Total time: {:.0}s", elapsed);
}
