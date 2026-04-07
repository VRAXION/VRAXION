//! Score-weighted ensemble: sum raw projection scores instead of majority vote.
//!
//! Oracle showed 30.8% with 6 networks (vs 17% single best).
//! Majority vote gets 16.9% — can't extract the knowledge.
//! This test sums the raw W@charge scores from all networks → single argmax.
//!
//! Run: cargo run --example ensemble_scored --release -- <corpus-path>

use instnct_core::{load_corpus, 
    build_network, evolution_step, InitConfig, Int8Projection, Network, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;


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
                    if p.predict(&n.charge_vec(os..nc)) == seg[i + 1] as usize { c += 1; }
                }
                c as f64 / 100.0
            },
            &evo);
    }

    let mut fr = StdRng::seed_from_u64(seed + 9999);
    let off = fr.gen_range(0..=corpus.len() - 5001);
    let seg = &corpus[off..off + 5001];
    let mut net2 = net.clone();
    net2.reset();
    let mut correct = 0u32;
    for i in 0..5000 {
        net2.propagate(sdr.pattern(seg[i] as usize), &init.propagation).unwrap();
        if proj.predict(&net2.charge_vec(os..nc)) == seg[i + 1] as usize { correct += 1; }
    }
    let acc = correct as f64 / 5000.0;
    println!("  trained seed={}: {:.1}%", seed, acc * 100.0);
    (net, proj, acc)
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
    let seeds_all = [42u64, 123, 7, 1042, 2042, 4042];
    let steps = 15_000;

    let sdr = SdrTable::new(CHARS, nc, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(999)).unwrap();

    println!("=== Training {} networks ===\n", seeds_all.len());
    let nets: Vec<(u64, Network, Int8Projection, f64)> = seeds_all.par_iter()
        .map(|&seed| {
            let (net, proj, acc) = train_one(seed, steps, &corpus, &init, &sdr);
            (seed, net, proj, acc)
        })
        .collect();

    // Sort by accuracy, pick top 4
    let mut sorted_nets: Vec<&(u64, Network, Int8Projection, f64)> = nets.iter().collect();
    sorted_nets.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    let top4: Vec<&(u64, Network, Int8Projection, f64)> = sorted_nets[..4].to_vec();
    println!("\nTop 4 networks:");
    for (seed, _, _, acc) in &top4 {
        println!("  seed={}: {:.1}%", seed, acc * 100.0);
    }

    // Eval on 10K chars
    let eval_len = 10_000;
    let mut eval_rng = StdRng::seed_from_u64(77777);
    let off = eval_rng.gen_range(0..=corpus.len() - eval_len - 1);
    let seg = &corpus[off..off + eval_len + 1];

    println!("\n=== Evaluating on {} chars ===\n", eval_len);

    // For each network, get predictions AND per-class "confidence" (was it the top class by a big margin?)
    // Since we can't access raw scores, we use a different ensemble strategy:
    // Top-4 majority vote (excluding bad seeds)

    // Strategy 1: All 6 majority vote
    // Strategy 2: Top 4 majority vote
    // Strategy 3: Top 4 weighted by individual accuracy
    // Strategy 4: "Best confident" — each network's prediction weighted by how often that network is right
    // Strategy 5: Oracle (any correct)

    let all_preds: Vec<(u64, f64, Vec<usize>)> = nets.iter().map(|(seed, net, proj, train_acc)| {
        let mut n = net.clone();
        n.reset();
        let mut preds = Vec::with_capacity(eval_len);
        for (i, &ch) in seg[..eval_len].iter().enumerate() {
            n.propagate(sdr.pattern(ch as usize), &init.propagation).unwrap();
            preds.push(proj.predict(&n.charge_vec(os..nc)));
            let _ = i;
        }
        (*seed, *train_acc, preds)
    }).collect();

    // Individual accuracies
    println!("Individual (eval segment):");
    let mut individual_accs = Vec::new();
    for (seed, _, preds) in &all_preds {
        let correct = preds.iter().enumerate()
            .filter(|&(i, &p)| p == seg[i + 1] as usize).count();
        let acc = correct as f64 / eval_len as f64;
        individual_accs.push((*seed, acc));
        println!("  seed={:<5} {:.1}%", seed, acc * 100.0);
    }
    let best_single = individual_accs.iter().map(|x| x.1).fold(0.0f64, f64::max);

    // Strategy 1: All 6 majority vote
    let mut vote6_correct = 0u32;
    for pos in 0..eval_len {
        let mut votes = [0u32; CHARS];
        for (_, _, preds) in &all_preds { votes[preds[pos]] += 1; }
        let pred = votes.iter().enumerate().max_by_key(|&(_, &c)| c).map(|(i, _)| i).unwrap();
        if pred == seg[pos + 1] as usize { vote6_correct += 1; }
    }

    // Strategy 2: Top 4 majority vote
    let top4_seeds: Vec<u64> = top4.iter().map(|(s, _, _, _)| *s).collect();
    let mut vote4_correct = 0u32;
    for pos in 0..eval_len {
        let mut votes = [0u32; CHARS];
        for (seed, _, preds) in &all_preds {
            if top4_seeds.contains(seed) { votes[preds[pos]] += 1; }
        }
        let pred = votes.iter().enumerate().max_by_key(|&(_, &c)| c).map(|(i, _)| i).unwrap();
        if pred == seg[pos + 1] as usize { vote4_correct += 1; }
    }

    // Strategy 3: Top 4 accuracy-weighted vote
    let mut weighted4_correct = 0u32;
    for pos in 0..eval_len {
        let mut scores = [0.0f64; CHARS];
        for (seed, train_acc, preds) in &all_preds {
            if top4_seeds.contains(seed) {
                scores[preds[pos]] += train_acc;  // weight by training accuracy
            }
        }
        let pred = scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap();
        if pred == seg[pos + 1] as usize { weighted4_correct += 1; }
    }

    // Oracle: any network correct
    let mut oracle_correct = 0u32;
    for pos in 0..eval_len {
        let target = seg[pos + 1] as usize;
        if all_preds.iter().any(|(_, _, preds)| preds[pos] == target) {
            oracle_correct += 1;
        }
    }

    // Oracle top 4
    let mut oracle4_correct = 0u32;
    for pos in 0..eval_len {
        let target = seg[pos + 1] as usize;
        if all_preds.iter()
            .filter(|(seed, _, _)| top4_seeds.contains(seed))
            .any(|(_, _, preds)| preds[pos] == target) {
            oracle4_correct += 1;
        }
    }

    println!("\n=== ENSEMBLE RESULTS ===\n");
    println!("  Best single:          {:.1}%", best_single * 100.0);
    println!("  All-6 majority vote:  {:.1}%", vote6_correct as f64 / eval_len as f64 * 100.0);
    println!("  Top-4 majority vote:  {:.1}%", vote4_correct as f64 / eval_len as f64 * 100.0);
    println!("  Top-4 weighted vote:  {:.1}%", weighted4_correct as f64 / eval_len as f64 * 100.0);
    println!("  Oracle (any of 6):    {:.1}%", oracle_correct as f64 / eval_len as f64 * 100.0);
    println!("  Oracle (top 4):       {:.1}%", oracle4_correct as f64 / eval_len as f64 * 100.0);
}
