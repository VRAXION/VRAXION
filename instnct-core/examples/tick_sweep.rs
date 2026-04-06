//! Tick speed sweep: does more propagation time per token help?
//!
//! Sweep ticks_per_token with decay_interval = ticks (1 decay per token constant).
//! More ticks = signal propagates further per token = potentially more context.
//!
//! Run: cargo run --example tick_sweep --release -- <corpus-path>

use instnct_core::{load_corpus, 
    build_network, evolution_step, InitConfig, Int8Projection, Network, PropagationConfig,
    SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 30_000;


#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network, proj: &Int8Projection, corpus: &[u8], len: usize,
    rng: &mut StdRng, sdr: &SdrTable, prop: &PropagationConfig, init: &InitConfig,
) -> f64 {
    if corpus.len() <= len { return 0.0; }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), prop).unwrap();
        if proj.predict(&net.charge()[init.output_start()..init.neuron_count]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

struct Config {
    ticks: usize,
    input_dur: usize,
    seed: u64,
}

#[allow(dead_code)]
struct RunResult {
    ticks: usize,
    input_dur: usize,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    edges: usize,
}

fn run_one(cfg: &Config, corpus: &[u8]) -> RunResult {
    let init = InitConfig::phi(256);
    let evo = init.evolution_config();

    let prop = PropagationConfig {
        ticks_per_token: cfg.ticks,
        input_duration_ticks: cfg.input_dur,
        decay_interval_ticks: cfg.ticks, // decay = ticks → 1 decay per token
        use_refractory: false,
    };

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS,
        &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(CHARS, init.neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    let mut peak = 0.0f64;

    for step in 0..STEPS {
        let _ = evolution_step(
            &mut net, &mut proj, &mut rng, &mut eval_rng,
            |n, p, e| eval_accuracy(n, p, corpus, 100, e, &sdr, &prop, &init),
            &evo,
        );

        if (step + 1) % 10_000 == 0 {
            let mut cr = StdRng::seed_from_u64(cfg.seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut cr, &sdr, &prop, &init);
            if acc > peak { peak = acc; }
            println!("  ticks={:<2} input={} seed={:<4} step {:>5}: {:.1}%  edges={}",
                cfg.ticks, cfg.input_dur, cfg.seed, step + 1, acc * 100.0, net.edge_count());
        }
    }

    let mut fr = StdRng::seed_from_u64(cfg.seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr, &sdr, &prop, &init);
    if final_acc > peak { peak = final_acc; }
    println!("  ticks={:<2} input={} seed={:<4} FINAL: {:.1}%  peak={:.1}%  edges={}",
        cfg.ticks, cfg.input_dur, cfg.seed, final_acc * 100.0, peak * 100.0, net.edge_count());

    RunResult { ticks: cfg.ticks, input_dur: cfg.input_dur, seed: cfg.seed,
        final_acc, peak_acc: peak, edges: net.edge_count() }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("=== Tick Speed Sweep ===\n");

    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let seeds = [42u64, 123, 7];

    // Sweep: ticks with decay=ticks, input_dur proportional
    // input_dur = ticks/3 (keep the ratio from baseline 2/6)
    let tick_configs: Vec<(usize, usize)> = vec![
        (6, 2),    // baseline
        (12, 4),   // 2x
        (18, 6),   // 3x
        (24, 8),   // 4x
        (36, 12),  // 6x
    ];

    let mut configs: Vec<Config> = Vec::new();
    for &(ticks, input_dur) in &tick_configs {
        for &s in &seeds {
            configs.push(Config { ticks, input_dur, seed: s });
        }
    }

    println!("  {} configs: {} tick settings × {} seeds\n", configs.len(), tick_configs.len(), seeds.len());

    let results: Vec<RunResult> = configs.par_iter()
        .map(|cfg| run_one(cfg, &corpus))
        .collect();

    println!("\n=== SUMMARY ===\n");
    println!("{:<12} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "ticks/decay", "input", "mean%", "best%", "peak%", "edges");
    println!("{}", "-".repeat(56));

    for &(ticks, input_dur) in &tick_configs {
        let g: Vec<_> = results.iter()
            .filter(|r| r.ticks == ticks && r.input_dur == input_dur)
            .collect();
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / g.len() as f64;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.edges).sum::<usize>() / g.len();
        println!("{:<12} {:>6} {:>7.1}% {:>7.1}% {:>7.1}% {:>8}",
            format!("{}/{}", ticks, ticks), input_dur,
            mean * 100.0, best * 100.0, peak * 100.0, edges);
    }

    println!("\nPer-seed:");
    println!("{:<12} {:>6} {:>6} {:>8} {:>8} {:>8}",
        "ticks", "input", "seed", "final%", "peak%", "edges");
    println!("{}", "-".repeat(56));
    for r in &results {
        println!("{:<12} {:>6} {:>6} {:>7.1}% {:>7.1}% {:>8}",
            format!("{}/{}", r.ticks, r.ticks), r.input_dur,
            r.seed, r.final_acc * 100.0, r.peak_acc * 100.0, r.edges);
    }
}
