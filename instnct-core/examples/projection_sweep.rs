//! Projection dimension sweep: output zone only vs full charge readout.
//!
//! Currently: proj reads charge[output_start..neuron_count] = 158 neurons.
//! Test: what if proj reads ALL 256 neurons? The input zone has SDR + propagated
//! charge — more information for the readout.
//!
//! Also tests: charge[0..neuron_count] = all 256, and smaller output-only slices.
//!
//! Run: cargo run --example projection_sweep --release -- <corpus-path>

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
    read_start: usize, read_end: usize,
) -> f64 {
    if corpus.len() <= len { return 0.0; }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        if proj.predict(&net.charge()[read_start..read_end]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

#[allow(dead_code)]
struct RunResult {
    label: String,
    proj_dim: usize,
    read_start: usize,
    read_end: usize,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    peak_step: usize,
    final_edges: usize,
}

fn run_one(
    label: &str, read_start: usize, read_end: usize, seed: u64,
    steps: usize, corpus: &[u8],
) -> RunResult {
    let init = InitConfig::phi(256);
    let nc = init.neuron_count;
    let edge_cap = init.edge_cap();
    let proj_dim = read_end - read_start;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(proj_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let sdr = SdrTable::new(CHARS, nc, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(seed + 100)).unwrap();

    let mut peak_acc = 0.0f64;
    let mut peak_step = 0usize;

    for step in 0..steps {
        // Paired eval
        let snap = eval_rng.clone();
        let before = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng,
            &sdr, &init.propagation, read_start, read_end);
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
                &sdr, &init.propagation, read_start, read_end);
            continue;
        }

        let after = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng,
            &sdr, &init.propagation, read_start, read_end);
        let accepted = if net.edge_count() < edge_cap { after >= before } else { after > before };
        if !accepted {
            net.restore_state(&state);
            if let Some(b) = wb { proj.rollback(b); }
        }

        if (step + 1) % 10_000 == 0 {
            let mut cr = StdRng::seed_from_u64(seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut cr,
                &sdr, &init.propagation, read_start, read_end);
            if acc > peak_acc { peak_acc = acc; peak_step = step + 1; }
            println!("  {:<14} seed={:<4} step {:>5}: {:.1}%  edges={}",
                label, seed, step + 1, acc * 100.0, net.edge_count());
        }
    }

    let mut fr = StdRng::seed_from_u64(seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr,
        &sdr, &init.propagation, read_start, read_end);
    println!("  {:<14} seed={:<4} FINAL: {:.1}%  peak={:.1}% @{}  edges={}  dim={}",
        label, seed, final_acc * 100.0, peak_acc * 100.0, peak_step, net.edge_count(), proj_dim);

    RunResult {
        label: label.to_string(), proj_dim, read_start, read_end, seed,
        final_acc, peak_acc, peak_step, final_edges: net.edge_count(),
    }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path);
    println!("  {} chars\n", corpus.len());

    let seeds = [42u64, 123, 7];
    let steps = 30_000;

    // H=256: input=0..158, output=98..256, overlap=98..158
    // Variants:
    //   "output"     = charge[98..256]  = 158 dims (current standard)
    //   "full"       = charge[0..256]   = 256 dims (all neurons)
    //   "hidden"     = charge[158..256] = 98 dims  (pure output, no overlap)
    //   "overlap"    = charge[98..158]  = 60 dims  (overlap zone only)
    let variants: Vec<(&str, usize, usize)> = vec![
        ("output-158", 98, 256),    // current standard
        ("full-256", 0, 256),       // all neurons
        ("hidden-98", 158, 256),    // pure output zone
        ("overlap-60", 98, 158),    // overlap zone only
    ];

    let mut configs: Vec<(&str, usize, usize, u64)> = Vec::new();
    for &(label, rs, re) in &variants {
        for &seed in &seeds {
            configs.push((label, rs, re, seed));
        }
    }

    println!("=== Projection Sweep: {} configs, {} steps ===\n", configs.len(), steps);

    let results: Vec<RunResult> = configs.par_iter()
        .map(|&(label, rs, re, seed)| run_one(label, rs, re, seed, steps, &corpus))
        .collect();

    println!("\n=== SUMMARY ===\n");
    println!("{:<14} {:>5} {:>8} {:>8} {:>8} {:>8}",
        "variant", "dim", "mean%", "best%", "peak%", "edges");
    println!("{}", "-".repeat(56));

    for &(label, _, _) in &variants {
        let g: Vec<_> = results.iter().filter(|r| r.label == label).collect();
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / g.len() as f64;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        let dim = g[0].proj_dim;
        println!("{:<14} {:>5} {:>7.1}% {:>7.1}% {:>7.1}% {:>8}",
            label, dim, mean * 100.0, best * 100.0, peak * 100.0, edges);
    }
}
