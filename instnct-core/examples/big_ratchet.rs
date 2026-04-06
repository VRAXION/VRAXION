//! Big network ratchet: H=1024/2048 with BUILD→CRYSTALLIZE cycles.
//!
//! A/B test: does a bigger network with ratchet break the ~17% ceiling?
//! H=1024 fits L2 cache (512KB), H=2048 fits L3.
//!
//! Run: cargo run --example big_ratchet --release -- <corpus-path>

use instnct_core::{load_corpus, 
    build_network, evolution_step, EvolutionConfig, InitConfig, Int8Projection, Network,
    SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const BUILD_STEPS: usize = 15_000;
const CYCLES: usize = 5;


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

/// Adaptive batch crystallize: remove edge batches, keep if accuracy holds.
fn crystallize(
    net: &mut Network, proj: &Int8Projection, corpus: &[u8],
    sdr: &SdrTable, init: &InitConfig, rng: &mut StdRng,
) -> (f64, usize, usize) {
    let off = rng.gen_range(0..=corpus.len() - 2001);
    let seg = &corpus[off..off + 2001];

    let eval = |n: &mut Network| -> f64 {
        let os = init.output_start();
        let nc = init.neuron_count;
        n.reset();
        let mut correct = 0u32;
        for i in 0..2000 {
            n.propagate(sdr.pattern(seg[i] as usize), &init.propagation).unwrap();
            if proj.predict(&n.charge()[os..nc]) == seg[i + 1] as usize {
                correct += 1;
            }
        }
        correct as f64 / 2000.0
    };

    let baseline = eval(net);
    let edges_before = net.edge_count();
    let mut current_acc = baseline;
    let mut removal_pct = 0.50f64;

    for round in 0..15 {
        let edges_now = net.edge_count();
        if edges_now < 100 { break; }
        let batch_size = ((edges_now as f64 * removal_pct) as usize).max(1);

        let all_edges: Vec<_> = net.graph().iter_edges().collect();
        let mut indices: Vec<usize> = (0..all_edges.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        let batch: Vec<(u16, u16)> = indices[..batch_size.min(indices.len())]
            .iter()
            .map(|&i| (all_edges[i].source, all_edges[i].target))
            .collect();
        for &(s, t) in &batch { net.graph_mut().remove_edge(s, t); }

        let after = eval(net);

        if after >= current_acc - 0.02 {
            removal_pct = (removal_pct * 1.5).min(0.80);
            current_acc = after;
            println!("      crystal round {}: ACCEPT -{} edges ({}→{}) acc {:.1}%",
                round, batch.len(), edges_now, net.edge_count(), after * 100.0);
        } else {
            for &(s, t) in &batch { net.graph_mut().add_edge(s, t); }
            removal_pct /= 2.0;
            if removal_pct < 0.02 { break; }
        }
    }

    let removed = edges_before - net.edge_count();
    (current_acc, net.edge_count(), removed)
}

struct Config {
    h: usize,
    seed: u64,
}

#[allow(dead_code)]
struct RunResult {
    h: usize,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    final_edges: usize,
    total_time: f64,
    cycle_peaks: Vec<f64>,
}

fn run_one(cfg: &Config, corpus: &[u8]) -> RunResult {
    let init = InitConfig::phi(cfg.h);
    let evo = EvolutionConfig { edge_cap: init.edge_cap(), accept_ties: false };

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS,
        &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(CHARS, init.neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    let start = Instant::now();
    let mut peak_acc = 0.0f64;
    let mut cycle_peaks: Vec<f64> = Vec::new();

    println!("  H={} seed={}: init edges={}", cfg.h, cfg.seed, net.edge_count());

    for cycle in 0..CYCLES {
        let cycle_start = Instant::now();
        let mut cycle_accepted = 0u32;
        let mut cycle_total = 0u32;

        // BUILD phase
        for step in 0..BUILD_STEPS {
            let outcome = evolution_step(
                &mut net, &mut proj, &mut rng, &mut eval_rng,
                |n, p, e| eval_accuracy(n, p, corpus, 100, e, &sdr, &init),
                &evo,
            );
            match outcome {
                StepOutcome::Accepted => { cycle_accepted += 1; cycle_total += 1; }
                StepOutcome::Rejected => { cycle_total += 1; }
                StepOutcome::Skipped => {}
            }

            if (step + 1) % 5000 == 0 {
                let mut cr = StdRng::seed_from_u64(cfg.seed + 6000 + cycle as u64 * 100000 + step as u64);
                let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut cr, &sdr, &init);
                if acc > peak_acc { peak_acc = acc; }
                let rate = if cycle_total > 0 { cycle_accepted as f64 / cycle_total as f64 * 100.0 } else { 0.0 };
                println!("    H={} seed={} cycle {} step {:>5}: {:.1}% edges={} accept={:.1}%",
                    cfg.h, cfg.seed, cycle, step + 1, acc * 100.0, net.edge_count(), rate);
            }
        }

        // BUILD accuracy
        let mut br = StdRng::seed_from_u64(cfg.seed + 7000 + cycle as u64 * 100000);
        let build_acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut br, &sdr, &init);
        if build_acc > peak_acc { peak_acc = build_acc; }
        cycle_peaks.push(build_acc);

        // CRYSTALLIZE phase
        let mut cryst_rng = StdRng::seed_from_u64(cfg.seed + 8000 + cycle as u64 * 100000);
        let (cryst_acc, cryst_edges, removed) = crystallize(
            &mut net, &proj, corpus, &sdr, &init, &mut cryst_rng,
        );

        let cycle_time = cycle_start.elapsed().as_secs_f64();
        let rate = if cycle_total > 0 { cycle_accepted as f64 / cycle_total as f64 * 100.0 } else { 0.0 };
        println!("    H={} seed={} cycle {} DONE: build={:.1}% crystal={:.1}% edges={} (-{}) accept={:.1}% time={:.0}s",
            cfg.h, cfg.seed, cycle, build_acc * 100.0, cryst_acc * 100.0,
            cryst_edges, removed, rate, cycle_time);
    }

    let total_time = start.elapsed().as_secs_f64();

    // Final eval (5000 chars)
    let mut fr = StdRng::seed_from_u64(cfg.seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr, &sdr, &init);
    if final_acc > peak_acc { peak_acc = final_acc; }

    println!("  H={} seed={} FINAL: {:.1}% peak={:.1}% edges={} time={:.0}s\n",
        cfg.h, cfg.seed, final_acc * 100.0, peak_acc * 100.0,
        net.edge_count(), total_time);

    RunResult {
        h: cfg.h, seed: cfg.seed, final_acc, peak_acc,
        final_edges: net.edge_count(), total_time, cycle_peaks,
    }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("=== Big Network Ratchet A/B ===\n");

    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let seeds = [42u64, 123, 7];

    // H sizes to test
    let sizes = [256usize, 512, 1024, 2048];

    let mut configs: Vec<Config> = Vec::new();
    for &h in &sizes {
        for &s in &seeds {
            configs.push(Config { h, seed: s });
        }
    }

    println!("  {} configs: H={{{}}} × {} seeds", configs.len(),
        sizes.iter().map(|h| h.to_string()).collect::<Vec<_>>().join(","),
        seeds.len());
    println!("  {} cycles × {} BUILD steps per cycle = {} total steps per config",
        CYCLES, BUILD_STEPS, CYCLES * BUILD_STEPS);

    for &h in &sizes {
        let init = InitConfig::phi(h);
        let edge_5pct = h * h * 5 / 100;
        let cap = init.edge_cap();
        println!("  H={}: phi_dim={} edges_init~{} cap={} overlap={}",
            h, init.phi_dim, edge_5pct, cap, init.input_end() as i64 - init.output_start() as i64);
    }
    println!();

    let start = Instant::now();

    let results: Vec<RunResult> = configs.par_iter()
        .map(|cfg| run_one(cfg, &corpus))
        .collect();

    let total = start.elapsed().as_secs_f64();

    println!("\n=== SUMMARY ===\n");
    println!("{:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "H", "mean%", "best%", "peak%", "edges", "time/cfg");
    println!("{}", "-".repeat(54));

    for &h in &sizes {
        let g: Vec<_> = results.iter().filter(|r| r.h == h).collect();
        let n = g.len() as f64;
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / n;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        let time = g.iter().map(|r| r.total_time).sum::<f64>() / n;
        println!("{:>6} {:>7.1}% {:>7.1}% {:>7.1}% {:>8} {:>7.0}s",
            h, mean * 100.0, best * 100.0, peak * 100.0, edges, time);
    }

    // Per-cycle peak progression
    println!("\n  Per-cycle BUILD peaks:");
    for &h in &sizes {
        let g: Vec<_> = results.iter().filter(|r| r.h == h).collect();
        print!("  H={:<5}", h);
        for cycle in 0..CYCLES {
            let mean_peak = g.iter()
                .filter_map(|r| r.cycle_peaks.get(cycle))
                .sum::<f64>() / g.len() as f64;
            print!(" C{}={:.1}%", cycle, mean_peak * 100.0);
        }
        println!();
    }

    println!("\n  Per-seed detail:");
    println!("{:>6} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "H", "seed", "final%", "peak%", "edges", "time");
    println!("{}", "-".repeat(50));
    for r in &results {
        println!("{:>6} {:>6} {:>7.1}% {:>7.1}% {:>8} {:>7.0}s",
            r.h, r.seed, r.final_acc * 100.0, r.peak_acc * 100.0,
            r.final_edges, r.total_time);
    }

    println!("\n  Total wall time: {:.0}s", total);
}
