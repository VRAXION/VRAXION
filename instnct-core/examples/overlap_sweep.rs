//! I/O overlap sweep: how much overlap between input and output zones matters?
//!
//! H=256 fixed, input always 0..158 (SDR). Output zone start varies.
//! Tests: separated I/O, partial overlap, full overlap.
//!
//! Run: cargo run --example overlap_sweep --release -- <corpus-path>

use instnct_core::{eval_accuracy, load_corpus,
    build_network, evolution_step, EvolutionConfig, InitConfig, Int8Projection,
    PropagationConfig, SdrTable,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;


struct OverlapConfig {
    output_start: usize,
    label: &'static str,
    seed: u64,
}

#[allow(dead_code)]
struct OverlapResult {
    label: &'static str,
    output_start: usize,
    overlap: usize,
    output_dim: usize,
    seed: u64,
    accuracy: f64,
    edge_count: usize,
}

fn run_one(cfg: &OverlapConfig, corpus: &[u8]) -> OverlapResult {
    let h = 256;
    let input_end: usize = 158;
    let output_dim = h - cfg.output_start; // W matrix input dimension
    let overlap = input_end.saturating_sub(cfg.output_start);
    let edge_cap = h * h * 7 / 100;

    let prop_config = PropagationConfig {
        ticks_per_token: 6,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
        use_refractory: false,
    };

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let init = InitConfig::phi(h);
    let mut net = build_network(&init, &mut rng);

    // W matrix: output_dim × 27 (varies with output_start)
    let mut proj = Int8Projection::new(output_dim, CHARS, &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(CHARS, h, input_end, SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    let evo_config = EvolutionConfig { edge_cap, accept_ties: true };
    let steps = 15000;

    for _ in 0..steps {
        let _ = evolution_step(
            &mut net, &mut proj, &mut rng, &mut eval_rng,
            |net, proj, eval_rng| {
                eval_accuracy(net, proj, corpus, 100, eval_rng, &sdr,
                    &prop_config, cfg.output_start, h)
            },
            &evo_config,
        );
    }

    let final_acc = eval_accuracy(
        &mut net, &proj, corpus, 5000, &mut eval_rng, &sdr,
        &prop_config, cfg.output_start, h,
    );

    println!("  {:<12} out_start={:<4} overlap={:<4} W={}×27  seed={:<5} -> {:.1}%  edges={}",
        cfg.label, cfg.output_start, overlap, output_dim, cfg.seed, final_acc * 100.0, net.edge_count());

    OverlapResult {
        label: cfg.label,
        output_start: cfg.output_start,
        overlap,
        output_dim,
        seed: cfg.seed,
        accuracy: final_acc,
        edge_count: net.edge_count(),
    }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let seeds = [42u64, 123, 7];

    let overlap_configs: Vec<(usize, &str)> = vec![
        (200, "tiny"),       // overlap=0, output=[200..256)=56 neurons
        (158, "separated"),  // overlap=0, output=[158..256)=98 neurons
        (128, "small"),      // overlap=30
        (98,  "phi"),        // overlap=60 (current default)
        (58,  "large"),      // overlap=100
        (0,   "full"),       // overlap=158, output=all 256
    ];

    let mut configs = Vec::new();
    for &(out_start, label) in &overlap_configs {
        for &seed in &seeds {
            configs.push(OverlapConfig { output_start: out_start, label, seed });
        }
    }

    println!("=== I/O Overlap Sweep: H=256, input=0..158 ===");
    println!("  {} configs ({} overlaps × {} seeds), 15K steps each\n",
        configs.len(), overlap_configs.len(), seeds.len());

    let results: Vec<OverlapResult> = configs
        .par_iter()
        .map(|cfg| run_one(cfg, &corpus))
        .collect();

    println!("\n=== SUMMARY (mean across {} seeds) ===\n", seeds.len());
    println!("{:<12} {:>5} {:>6} {:>5} {:>8} {:>8}", "label", "o_start", "overlap", "W_dim", "mean%", "edges");
    println!("{}", "-".repeat(50));

    let mut seen = Vec::new();
    for r in &results {
        if seen.contains(&r.output_start) { continue; }
        seen.push(r.output_start);
        let group: Vec<_> = results.iter()
            .filter(|x| x.output_start == r.output_start)
            .collect();
        let mean_acc = group.iter().map(|x| x.accuracy).sum::<f64>() / group.len() as f64;
        let mean_edges = group.iter().map(|x| x.edge_count).sum::<usize>() / group.len();
        println!("{:<12} {:>5} {:>6} {:>5} {:>7.1}% {:>8}",
            r.label, r.output_start, r.overlap, r.output_dim, mean_acc * 100.0, mean_edges);
    }
}
