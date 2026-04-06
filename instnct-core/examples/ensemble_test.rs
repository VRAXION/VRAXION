//! Ensemble diagnostic: majority vote from independently trained networks.
//!
//! If ensemble >> single best → the networks have uncorrelated errors (breed potential).
//! If ensemble ≈ single best → the networks make the same errors (ceiling is real).
//!
//! Run: cargo run --example ensemble_test --release -- <corpus-path>

use instnct_core::{
    build_network, evolution_step, InitConfig, Int8Projection, Network, PropagationConfig,
    SdrTable,
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
fn eval_single(
    net: &mut Network, proj: &Int8Projection, corpus: &[u8], len: usize,
    off: usize, sdr: &SdrTable, config: &PropagationConfig, os: usize, nc: usize,
) -> (u32, Vec<usize>) {
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    let mut predictions = Vec::with_capacity(len);
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        let pred = proj.predict(&net.charge()[os..nc]);
        predictions.push(pred);
        if pred == seg[i + 1] as usize { correct += 1; }
    }
    (correct, predictions)
}

fn train_one(
    seed: u64, steps: usize, corpus: &[u8], init: &InitConfig, sdr: &SdrTable,
) -> (Network, Int8Projection, f64) {
    let os = init.output_start();
    let nc = init.neuron_count;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let evo = init.evolution_config();

    for _ in 0..steps {
        let _ = evolution_step(&mut net, &mut proj, &mut rng, &mut eval_rng,
            |n, p, e| {
                if corpus.len() <= 100 { return 0.0; }
                let off = e.gen_range(0..=corpus.len() - 101);
                let seg = &corpus[off..off + 101];
                n.reset();
                let mut c = 0u32;
                for i in 0..100 {
                    n.propagate(sdr.pattern(seg[i] as usize), &init.propagation).unwrap();
                    if p.predict(&n.charge()[os..nc]) == seg[i + 1] as usize { c += 1; }
                }
                c as f64 / 100.0
            },
            &evo);
    }

    let mut fr = StdRng::seed_from_u64(seed + 9999);
    let off = fr.gen_range(0..=corpus.len() - 5001);
    let (correct, _) = eval_single(&mut net, &proj, corpus, 5000, off, sdr, &init.propagation, os, nc);
    let acc = correct as f64 / 5000.0;
    println!("  trained seed={}: {:.1}%", seed, acc * 100.0);
    (net, proj, acc)
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path);
    println!("  {} chars\n", corpus.len());

    let init = InitConfig::phi(256);
    let os = init.output_start();
    let nc = init.neuron_count;
    let parent_seeds = [42u64, 123, 7, 1042, 2042, 4042];
    let steps = 15_000;

    let sdr = SdrTable::new(CHARS, nc, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(999)).unwrap();

    // Train 6 networks in parallel
    println!("=== Training {} networks ({} steps each) ===\n", parent_seeds.len(), steps);
    let nets: Vec<(u64, Network, Int8Projection, f64)> = parent_seeds.par_iter()
        .map(|&seed| {
            let (net, proj, acc) = train_one(seed, steps, &corpus, &init, &sdr);
            (seed, net, proj, acc)
        })
        .collect();

    // Ensemble eval on same 10K chars
    let eval_len = 10_000;
    let mut eval_rng = StdRng::seed_from_u64(77777);
    let off = eval_rng.gen_range(0..=corpus.len() - eval_len - 1);
    let seg = &corpus[off..off + eval_len + 1];

    println!("\n=== Evaluating on {} chars (offset={}) ===\n", eval_len, off);

    // Get per-network predictions
    let mut all_preds: Vec<Vec<usize>> = Vec::new();
    let mut individual_accs: Vec<(u64, f64)> = Vec::new();

    for (seed, net, proj, _) in &nets {
        let mut net_clone = net.clone();
        let (correct, preds) = eval_single(
            &mut net_clone, proj, &corpus, eval_len, off,
            &sdr, &init.propagation, os, nc,
        );
        let acc = correct as f64 / eval_len as f64;
        individual_accs.push((*seed, acc));
        all_preds.push(preds);
    }

    // Majority vote ensemble
    let mut ensemble_correct = 0u32;
    for pos in 0..eval_len {
        let mut votes = [0u32; CHARS];
        for net_preds in &all_preds {
            votes[net_preds[pos]] += 1;
        }
        let ensemble_pred = votes.iter().enumerate()
            .max_by_key(|&(_, &count)| count)
            .map(|(cls, _)| cls)
            .unwrap_or(0);
        if ensemble_pred == seg[pos + 1] as usize {
            ensemble_correct += 1;
        }
    }
    let ensemble_acc = ensemble_correct as f64 / eval_len as f64;

    // Agreement analysis
    let mut agree_count = 0u64;
    let mut total_pairs = 0u64;
    for (i, preds_i) in all_preds.iter().enumerate() {
        for preds_j in &all_preds[(i + 1)..] {
            for pos in 0..eval_len {
                if preds_i[pos] == preds_j[pos] { agree_count += 1; }
                total_pairs += 1;
            }
        }
    }
    let agreement = agree_count as f64 / total_pairs as f64;

    // Oracle: correct if ANY network got it right
    let mut oracle_correct = 0u32;
    for pos in 0..eval_len {
        let target = seg[pos + 1] as usize;
        if all_preds.iter().any(|preds| preds[pos] == target) {
            oracle_correct += 1;
        }
    }
    let oracle_acc = oracle_correct as f64 / eval_len as f64;

    println!("\n=== RESULTS ===\n");
    println!("Individual networks:");
    for (seed, acc) in &individual_accs {
        println!("  seed={:<5} {:.1}%", seed, acc * 100.0);
    }
    let best_single = individual_accs.iter().map(|x| x.1).fold(0.0f64, f64::max);
    let mean_single = individual_accs.iter().map(|x| x.1).sum::<f64>() / individual_accs.len() as f64;

    println!("\n  Mean single:      {:.1}%", mean_single * 100.0);
    println!("  Best single:      {:.1}%", best_single * 100.0);
    println!("  ENSEMBLE (vote):  {:.1}%", ensemble_acc * 100.0);
    println!("  ORACLE (any):     {:.1}%", oracle_acc * 100.0);
    println!("  Agreement:        {:.1}%", agreement * 100.0);
    println!("\n  Ensemble - best:  {:+.1}pp", (ensemble_acc - best_single) * 100.0);
    println!("  Oracle - best:    {:+.1}pp", (oracle_acc - best_single) * 100.0);
}
