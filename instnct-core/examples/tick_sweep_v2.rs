//! Tick count sweep v2: with smooth fitness + jackpot.
//!
//! The original tick_sweep (stepwise fitness) found no difference across
//! ticks=6/12/18/24/36. Does smooth fitness + jackpot change this?
//!
//! Sweep: ticks = 4, 6, 8, 12, 18 (with proportional input_duration and decay).
//!
//! Run: cargo run --example tick_sweep_v2 --release -- <corpus-path>

use instnct_core::{
    build_bigram_table, build_network, eval_accuracy, eval_smooth, evolution_step_jackpot,
    load_corpus, InitConfig, Int8Projection, PropagationConfig, SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 30_000;

struct Config {
    ticks: usize,
    seed: u64,
}

#[allow(dead_code)]
struct RunResult {
    ticks: usize,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    final_edges: usize,
    accepted: u32,
}

fn run_one(cfg: &Config, corpus: &[u8], bigram: &[Vec<f64>]) -> RunResult {
    let mut init = InitConfig::phi(256);
    // Scale input_duration proportionally, decay stays at 6
    init.propagation = PropagationConfig {
        ticks_per_token: cfg.ticks,
        input_duration_ticks: (cfg.ticks / 3).max(1),
        decay_interval_ticks: 6,
        use_refractory: false,
    };
    let evo = init.evolution_config();

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS,
        &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(CHARS, init.neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

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
            println!("  ticks={} seed={} step {:>5}: acc={:.1}% edges={} accept={:.1}%",
                cfg.ticks, cfg.seed, step + 1, acc * 100.0, net.edge_count(), rate);
        }
    }

    let mut fr = StdRng::seed_from_u64(cfg.seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr, &sdr,
        &init.propagation, init.output_start(), init.neuron_count);
    if final_acc > peak_acc { peak_acc = final_acc; }

    println!("  ticks={} seed={} FINAL: acc={:.1}% peak={:.1}% edges={}",
        cfg.ticks, cfg.seed, final_acc * 100.0, peak_acc * 100.0, net.edge_count());

    RunResult { ticks: cfg.ticks, seed: cfg.seed, final_acc, peak_acc,
        final_edges: net.edge_count(), accepted }
}

fn main() {
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global().ok();

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });

    println!("=== Tick Sweep v2 (smooth + jackpot) ===\n");

    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars", corpus.len());
    let bigram = build_bigram_table(&corpus, CHARS);

    let tick_values = [4usize, 6, 8, 12, 18];
    let seeds = [42u64, 123, 7];

    for &t in &tick_values {
        let input_dur = (t / 3).max(1);
        println!("  ticks={}: input_dur={}, decay=6", t, input_dur);
    }
    println!();

    let mut configs: Vec<Config> = Vec::new();
    for &t in &tick_values {
        for &s in &seeds { configs.push(Config { ticks: t, seed: s }); }
    }
    println!("  {} configs: {} tick values x {} seeds\n", configs.len(), tick_values.len(), seeds.len());

    let start = Instant::now();
    let results: Vec<RunResult> = configs.par_iter()
        .map(|c| run_one(c, &corpus, &bigram))
        .collect();
    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== SUMMARY ===\n");
    println!("{:>6} {:>7} {:>7} {:>7} {:>7} {:>8}",
        "Ticks", "Mean%", "Best%", "Peak%", "Edges", "Accepted");
    println!("{}", "-".repeat(50));

    for &t in &tick_values {
        let g: Vec<_> = results.iter().filter(|r| r.ticks == t).collect();
        let n = g.len() as f64;
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / n;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        let acc = g.iter().map(|r| r.accepted as f64).sum::<f64>() / n;
        println!("{:>6} {:>6.1}% {:>6.1}% {:>6.1}% {:>7} {:>8.0}",
            t, mean * 100.0, best * 100.0, peak * 100.0, edges, acc);
    }

    println!("\n  Total time: {:.0}s", elapsed);
}
