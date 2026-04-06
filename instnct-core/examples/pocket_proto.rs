//! Pocket chain prototype: independently trained pockets chained at eval time.
//!
//! Q1: Do pockets with local next-char fitness just become copies?
//! Q2: Does chaining them (A output → B input) add anything?
//!
//! Run: cargo run --example pocket_proto --release -- <corpus-path>

use instnct_core::{load_corpus, 
    build_network, evolution_step, InitConfig, Int8Projection, Network, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const N_POCKETS: usize = 4;


/// Train one pocket independently with local next-char fitness.
fn train_pocket(
    seed: u64, steps: usize, corpus: &[u8], init: &InitConfig, sdr: &SdrTable,
) -> (Network, Int8Projection) {
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
            }, &evo);
    }
    println!("  pocket seed={} trained", seed);
    (net, proj)
}

/// Convert output zone charge to input signal for next pocket.
/// Neurons with charge > threshold get activation +1 in the input signal.
fn charge_to_input(charge: &[u32], output_start: usize, neuron_count: usize,
                   input_end: usize, threshold: u32) -> Vec<i32> {
    let mut input = vec![0i32; neuron_count];
    let output_zone = &charge[output_start..neuron_count];
    // Map output zone → input zone (wrap if sizes differ)
    for (i, &c) in output_zone.iter().enumerate() {
        if c > threshold {
            let input_idx = i % input_end;
            input[input_idx] += 1;
        }
    }
    input
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let init = InitConfig::phi(256);
    let os = init.output_start();
    let nc = init.neuron_count;
    let ie = init.input_end();
    let steps = 15_000;
    let pocket_seeds: [u64; N_POCKETS] = [42, 123, 7, 1042];

    let sdr = SdrTable::new(CHARS, nc, ie, SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(999)).unwrap();

    // === Train pockets in parallel ===
    println!("=== Training {} pockets ({} steps each) ===\n", N_POCKETS, steps);
    let pockets: Vec<(Network, Int8Projection)> = pocket_seeds.par_iter()
        .map(|&seed| train_pocket(seed, steps, &corpus, &init, &sdr))
        .collect();

    // === Eval on same 10K chars ===
    let eval_len = 10_000;
    let mut eval_rng = StdRng::seed_from_u64(77777);
    let off = eval_rng.gen_range(0..=corpus.len() - eval_len - 1);
    let seg = &corpus[off..off + eval_len + 1];

    println!("\n=== Evaluating on {} chars ===\n", eval_len);

    // --- Individual pocket accuracy ---
    let mut individual_accs = Vec::new();
    let mut all_preds: Vec<Vec<usize>> = Vec::new();

    for (idx, (net, proj)) in pockets.iter().enumerate() {
        let mut n = net.clone();
        n.reset();
        let mut correct = 0u32;
        let mut preds = Vec::with_capacity(eval_len);
        for i in 0..eval_len {
            n.propagate(sdr.pattern(seg[i] as usize), &init.propagation).unwrap();
            let pred = proj.predict(&n.charge()[os..nc]);
            preds.push(pred);
            if pred == seg[i + 1] as usize { correct += 1; }
        }
        let acc = correct as f64 / eval_len as f64;
        individual_accs.push(acc);
        all_preds.push(preds);
        println!("  Pocket {} (seed={}): {:.1}%", idx, pocket_seeds[idx], acc * 100.0);
    }

    // --- Pairwise prediction agreement ---
    println!("\n  Pairwise agreement:");
    for i in 0..N_POCKETS {
        for j in (i+1)..N_POCKETS {
            let agree = (0..eval_len).filter(|&pos| all_preds[i][pos] == all_preds[j][pos]).count();
            println!("    {} vs {}: {:.1}%", i, j, agree as f64 / eval_len as f64 * 100.0);
        }
    }

    // --- Chain eval: A → B → C → D ---
    // Token feeds SDR into pocket A. A's output charge → B's input. etc.
    // Only the LAST pocket's W projection is used for prediction.
    println!("\n=== Chain eval (A → B → C → D, readout from D) ===\n");

    for &threshold in &[1u32, 3, 5, 7] {
        let mut nets: Vec<Network> = pockets.iter().map(|(n, _)| n.clone()).collect();
        for n in &mut nets { n.reset(); }

        let mut chain_correct = 0u32;
        for i in 0..eval_len {
            // Pocket 0 gets SDR input
            nets[0].propagate(sdr.pattern(seg[i] as usize), &init.propagation).unwrap();

            // Chain: each subsequent pocket gets previous pocket's output as input
            for p in 1..N_POCKETS {
                let input = charge_to_input(nets[p-1].charge(), os, nc, ie, threshold);
                nets[p].propagate(&input, &init.propagation).unwrap();
            }

            // Readout from last pocket
            let last_proj = &pockets[N_POCKETS - 1].1;
            let pred = last_proj.predict(&nets[N_POCKETS - 1].charge()[os..nc]);
            if pred == seg[i + 1] as usize { chain_correct += 1; }
        }
        let chain_acc = chain_correct as f64 / eval_len as f64;
        println!("  Chain (threshold={}): {:.1}%", threshold, chain_acc * 100.0);
    }

    // --- Chain with EACH pocket's own projection ---
    println!("\n=== Chain eval, per-pocket readout ===\n");
    {
        let mut nets: Vec<Network> = pockets.iter().map(|(n, _)| n.clone()).collect();
        for n in &mut nets { n.reset(); }

        let mut per_pocket_correct = [0u32; N_POCKETS];
        for i in 0..eval_len {
            nets[0].propagate(sdr.pattern(seg[i] as usize), &init.propagation).unwrap();
            for p in 1..N_POCKETS {
                let input = charge_to_input(nets[p-1].charge(), os, nc, ie, 3);
                nets[p].propagate(&input, &init.propagation).unwrap();
            }
            for (p, (_, proj)) in pockets.iter().enumerate() {
                let pred = proj.predict(&nets[p].charge()[os..nc]);
                if pred == seg[i + 1] as usize { per_pocket_correct[p] += 1; }
            }
        }
        for p in 0..N_POCKETS {
            let acc = per_pocket_correct[p] as f64 / eval_len as f64;
            println!("  Pocket {} in chain: {:.1}% (standalone: {:.1}%)",
                p, acc * 100.0, individual_accs[p] * 100.0);
        }
    }

    // --- Oracle in chain ---
    println!("\n=== Oracle (any pocket correct, standalone) ===\n");
    let oracle = (0..eval_len).filter(|&pos| {
        all_preds.iter().any(|preds| preds[pos] == seg[pos + 1] as usize)
    }).count();
    println!("  Oracle: {:.1}%", oracle as f64 / eval_len as f64 * 100.0);

    println!("\n=== SUMMARY ===\n");
    let mean_ind = individual_accs.iter().sum::<f64>() / N_POCKETS as f64;
    let best_ind = individual_accs.iter().fold(0.0f64, |a, &b| a.max(b));
    println!("  Mean individual: {:.1}%", mean_ind * 100.0);
    println!("  Best individual: {:.1}%", best_ind * 100.0);
    println!("  Oracle:          {:.1}%", oracle as f64 / eval_len as f64 * 100.0);
}
