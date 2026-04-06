//! Simulated annealing sweep: explicit temperature schedule for evolution.
//!
//! The ~17-18% ceiling may be caused by the evolution getting stuck in local optima.
//! Current system: density-capped >= (noisy implicit annealing via 100-char eval).
//! SA: P(accept worse) = exp(delta / T) where T decays from T0 to 0.
//!
//! Sweep: T0 = {0.01, 0.03, 0.05, 0.10} × 3 seeds × 50K steps
//! Control: standard density-capped >= (T0=0 = no SA)
//!
//! Run: cargo run --example annealing_sweep --release -- <corpus-path>

use instnct_core::{
    build_network, InitConfig, Int8Projection, Network, PropagationConfig, SdrTable,
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
    t0: f64,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    peak_step: usize,
    final_edges: usize,
}

/// SA acceptance: always accept improvements, accept worse with P = exp(delta/T).
/// T decays linearly from t0 to 0 over total_steps.
fn sa_accept(delta: f64, t: f64, rng: &mut impl Rng) -> bool {
    if delta > 0.0 { return true; }
    if t <= 0.0 { return false; }
    let p = (delta / t).exp();
    rng.gen::<f64>() < p
}

fn run_one(t0: f64, seed: u64, steps: usize, corpus: &[u8]) -> RunResult {
    let init = InitConfig::phi(256);
    let os = init.output_start();
    let nc = init.neuron_count;
    let edge_cap = init.edge_cap();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let sdr = SdrTable::new(CHARS, nc, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(seed + 100)).unwrap();

    let mut peak_acc = 0.0f64;
    let mut peak_step = 0usize;
    let is_control = t0 == 0.0;

    for step in 0..steps {
        // Temperature: linear decay from t0 to 0
        let t = if is_control { 0.0 } else { t0 * (1.0 - step as f64 / steps as f64) };

        // Paired eval: before
        let snap = eval_rng.clone();
        let before = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng,
            &sdr, &init.propagation, os, nc);
        eval_rng = snap;

        let state = net.save_state();
        let roll = rng.gen_range(0..100u32);
        let mut wb = None;
        let mutated = match roll {
            0..25 => net.mutate_add_edge(&mut rng),
            25..40 => net.mutate_remove_edge(&mut rng),
            40..50 => net.mutate_rewire(&mut rng),
            50..65 => net.mutate_reverse(&mut rng),
            65..72 => net.mutate_mirror(&mut rng),
            72..80 => net.mutate_enhance(&mut rng),
            80..85 => net.mutate_theta(&mut rng),
            85..90 => net.mutate_channel(&mut rng),
            _ => { wb = Some(proj.mutate_one(&mut rng)); true }
        };

        if !mutated {
            let _ = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng,
                &sdr, &init.propagation, os, nc);
            continue;
        }

        let after = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng,
            &sdr, &init.propagation, os, nc);

        let delta = after - before;
        let accepted = if is_control {
            // Standard density-capped >=
            if net.edge_count() < edge_cap { delta >= 0.0 } else { delta > 0.0 }
        } else {
            // SA: always accept better, probabilistic accept worse
            // But still enforce edge cap with strict > when over
            if net.edge_count() >= edge_cap && delta <= 0.0 {
                false
            } else {
                sa_accept(delta, t, &mut rng)
            }
        };

        if !accepted {
            net.restore_state(&state);
            if let Some(b) = wb { proj.rollback(b); }
        }

        if (step + 1) % 10_000 == 0 {
            let mut cr = StdRng::seed_from_u64(seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut cr,
                &sdr, &init.propagation, os, nc);
            if acc > peak_acc {
                peak_acc = acc;
                peak_step = step + 1;
            }
            let label = if is_control { "ctrl".to_string() } else { format!("T0={:.2}", t0) };
            println!("  {:<8} seed={:<4} step {:>5}: {:.1}%  edges={}  T={:.4}",
                label, seed, step + 1, acc * 100.0, net.edge_count(), t);
        }
    }

    let mut fr = StdRng::seed_from_u64(seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr,
        &sdr, &init.propagation, os, nc);
    let label = if is_control { "ctrl".to_string() } else { format!("T0={:.2}", t0) };
    println!("  {:<8} seed={:<4} FINAL: {:.1}%  peak={:.1}% @{}  edges={}",
        label, seed, final_acc * 100.0, peak_acc * 100.0, peak_step, net.edge_count());

    RunResult { t0, seed, final_acc, peak_acc, peak_step, final_edges: net.edge_count() }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path);
    println!("  {} chars\n", corpus.len());

    let temps = [0.0, 0.01, 0.03, 0.05, 0.10]; // 0.0 = control (standard >=)
    let seeds = [42u64, 123, 7];
    let steps = 50_000;

    let mut configs: Vec<(f64, u64)> = Vec::new();
    for &t in &temps {
        for &s in &seeds {
            configs.push((t, s));
        }
    }

    println!("=== SA Sweep: {} configs, {} steps ===", configs.len(), steps);
    println!("T0 values: {:?}, seeds: {:?}\n", temps, seeds);

    let results: Vec<RunResult> = configs.par_iter()
        .map(|&(t, s)| run_one(t, s, steps, &corpus))
        .collect();

    println!("\n=== SUMMARY ===\n");
    println!("{:<10} {:>8} {:>8} {:>8} {:>8}",
        "T0", "mean%", "best%", "peak%", "edges");
    println!("{}", "-".repeat(48));

    for &t in &temps {
        let g: Vec<_> = results.iter().filter(|r| (r.t0 - t).abs() < 0.001).collect();
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / g.len() as f64;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        let label = if t == 0.0 { "control".to_string() } else { format!("T0={:.2}", t) };
        println!("{:<10} {:>7.1}% {:>7.1}% {:>7.1}% {:>8}",
            label, mean * 100.0, best * 100.0, peak * 100.0, edges);
    }

    println!("\nPer-seed:");
    println!("{:<10} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "T0", "seed", "final%", "peak%", "p_step", "edges");
    println!("{}", "-".repeat(52));
    for r in &results {
        let label = if r.t0 == 0.0 { "control".to_string() } else { format!("T0={:.2}", r.t0) };
        println!("{:<10} {:>6} {:>7.1}% {:>7.1}% {:>8} {:>8}",
            label, r.seed, r.final_acc * 100.0, r.peak_acc * 100.0, r.peak_step, r.final_edges);
    }
}
