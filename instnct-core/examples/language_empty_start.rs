//! Language prediction: empty start vs prefilled.
//!
//! Does the "sparse evolution builds better circuits" finding from
//! addition (80% with 83 edges) transfer to language prediction?
//!
//! A: Empty start (0 edges, 0 chains) + smooth fitness + jackpot=9
//! B: Prefilled (chain-50 + 5% density) + smooth fitness + jackpot=9
//!
//! Run: cargo run --example language_empty_start --release -- <corpus-path>

use instnct_core::{
    build_bigram_table, build_network, eval_accuracy, eval_smooth, evolution_step_jackpot,
    load_corpus, InitConfig, Int8Projection, SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 30_000;

#[derive(Clone, Copy)]
enum Variant { Empty, Prefilled }
impl Variant {
    fn name(&self) -> &str { match self { Self::Empty => "empty", Self::Prefilled => "prefill" } }
    fn init(&self) -> InitConfig {
        match self {
            Self::Empty => InitConfig::empty(256),
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

fn run_one(cfg: &Config, corpus: &[u8], bigram: &[Vec<f64>]) -> RunResult {
    let init = cfg.variant.init();
    let evo = init.evolution_config();

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS,
        &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(CHARS, init.neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    let init_edges = net.edge_count();
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

        if (step + 1) % 5_000 == 0 {
            let mut cr = StdRng::seed_from_u64(cfg.seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut cr, &sdr,
                &init.propagation, init.output_start(), init.neuron_count);
            if acc > peak_acc { peak_acc = acc; }
            let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };
            println!("  {} seed={} step {:>5}: acc={:.1}% edges={} (init={}) accept={:.1}%",
                cfg.variant.name(), cfg.seed, step + 1, acc * 100.0,
                net.edge_count(), init_edges, rate);
        }
    }

    let mut fr = StdRng::seed_from_u64(cfg.seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr, &sdr,
        &init.propagation, init.output_start(), init.neuron_count);
    if final_acc > peak_acc { peak_acc = final_acc; }
    let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };

    println!("  {} seed={} FINAL: acc={:.1}% peak={:.1}% edges={} (init={}) accept={:.1}%",
        cfg.variant.name(), cfg.seed, final_acc * 100.0, peak_acc * 100.0,
        net.edge_count(), init_edges, rate);

    RunResult {
        variant_name: cfg.variant.name().to_string(), seed: cfg.seed,
        final_acc, peak_acc, final_edges: net.edge_count(), accepted,
    }
}

fn main() {
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global().ok();

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });

    println!("=== Language: Empty Start vs Prefilled ===");
    println!("  Both: smooth cosine-bigram fitness + jackpot=9");
    println!("  Empty: 0 edges (InitConfig::empty), Prefilled: chain-50 + 5%\n");

    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars", corpus.len());
    let bigram = build_bigram_table(&corpus, CHARS);
    println!();

    let seeds = [42u64, 123, 7, 1042, 555, 8042];
    let mut configs: Vec<Config> = Vec::new();
    for &v in &[Variant::Empty, Variant::Prefilled] {
        for &s in &seeds { configs.push(Config { variant: v, seed: s }); }
    }
    println!("  {} configs: 2 variants x {} seeds\n", configs.len(), seeds.len());

    let start = Instant::now();
    let results: Vec<RunResult> = configs.par_iter()
        .map(|cfg| run_one(cfg, &corpus, &bigram))
        .collect();
    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== SUMMARY ===\n");
    println!("{:<10} {:>7} {:>7} {:>7} {:>7} {:>8}",
        "Variant", "Mean%", "Best%", "Peak%", "Edges", "Accepted");
    println!("{}", "-".repeat(52));

    for v in &[Variant::Empty, Variant::Prefilled] {
        let g: Vec<_> = results.iter().filter(|r| r.variant_name == v.name()).collect();
        let n = g.len() as f64;
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / n;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        let acc_mean = g.iter().map(|r| r.accepted as f64).sum::<f64>() / n;
        println!("{:<10} {:>6.1}% {:>6.1}% {:>6.1}% {:>7} {:>8.0}",
            v.name(), mean * 100.0, best * 100.0, peak * 100.0, edges, acc_mean);
    }

    println!("\nPer-seed:");
    println!("{:<10} {:>6} {:>7} {:>7} {:>7} {:>8}",
        "Variant", "Seed", "Final%", "Peak%", "Edges", "Accepted");
    println!("{}", "-".repeat(50));
    for r in &results {
        println!("{:<10} {:>6} {:>6.1}% {:>6.1}% {:>7} {:>8}",
            r.variant_name, r.seed, r.final_acc * 100.0, r.peak_acc * 100.0,
            r.final_edges, r.accepted);
    }

    println!("\n  Total time: {:.0}s", elapsed);
}
