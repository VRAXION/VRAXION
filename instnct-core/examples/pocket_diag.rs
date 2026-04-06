//! Pocket diagnostic: analyze what changed between generations.
//!
//! Loads checkpoints from Gen 0 (individuals), Gen 1 (merged), Gen 2 (continued)
//! and compares: edge topology, charge patterns at interface, signal throughput,
//! W projection similarity, per-pocket contribution.
//!
//! Run: cargo run --example pocket_diag --release -- <corpus-path>

use instnct_core::{load_corpus, load_checkpoint, Int8Projection, Network, PropagationConfig, SdrTable};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::collections::HashSet;

const MASTER_SEED: u64 = 1337;
const H: usize = 256;
const PHI_DIM: usize = 158;
const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const EVAL_LEN: usize = 1000;

fn output_start() -> usize { H - PHI_DIM } // 98

fn charge_transfer(female: &Network) -> Vec<i32> {
    let os = output_start();
    let mut input = vec![0i32; H];
    for (i, &c) in female.charge()[os..H].iter().enumerate() {
        if i < PHI_DIM { input[i] = c as i32; }
    }
    input
}

fn edge_set(net: &Network) -> HashSet<(u16, u16)> {
    net.graph().iter_edges().map(|e| (e.source, e.target)).collect()
}

fn jaccard(a: &HashSet<(u16, u16)>, b: &HashSet<(u16, u16)>) -> f64 {
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 { 0.0 } else { inter as f64 / union as f64 }
}


/// Run chain, return per-token: (charge at interface, charge at male output, prediction)
#[allow(clippy::type_complexity)]
fn trace_chain(
    female: &mut Network, male: &mut Network, proj: &Int8Projection,
    segment: &[u8], sdr: &SdrTable, prop: &PropagationConfig,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<usize>, u32) {
    let os = output_start();
    let len = segment.len() - 1;
    female.reset();
    male.reset();

    let mut interface_charges: Vec<Vec<u32>> = Vec::new(); // female output zone charge per token
    let mut output_charges: Vec<Vec<u32>> = Vec::new();    // male output zone charge per token
    let mut predictions: Vec<usize> = Vec::new();
    let mut correct = 0u32;

    for i in 0..len {
        female.propagate(sdr.pattern(segment[i] as usize), prop).unwrap();

        // Capture female output zone charge (the interface signal)
        let f_out_charge: Vec<u32> = female.charge()[os..H].to_vec();
        interface_charges.push(f_out_charge);

        let transfer = charge_transfer(female);
        male.propagate(&transfer, prop).unwrap();

        // Capture male output zone charge
        let m_out_charge: Vec<u32> = male.charge()[os..H].to_vec();
        output_charges.push(m_out_charge);

        let pred = proj.predict(&male.charge()[os..H]);
        predictions.push(pred);
        if pred == segment[i + 1] as usize { correct += 1; }
    }

    (interface_charges, output_charges, predictions, correct)
}

/// Stats about charge vectors
fn charge_stats(charges: &[Vec<u32>]) -> (f64, f64, usize, f64) {
    let n = charges.len();
    if n == 0 { return (0.0, 0.0, 0, 0.0); }
    let dim = charges[0].len();

    // Mean total charge per token
    let mean_total: f64 = charges.iter()
        .map(|c| c.iter().sum::<u32>() as f64)
        .sum::<f64>() / n as f64;

    // Mean nonzero neurons per token
    let mean_active: f64 = charges.iter()
        .map(|c| c.iter().filter(|&&v| v > 0).count() as f64)
        .sum::<f64>() / n as f64;

    // How many unique charge patterns? (distinct fingerprints)
    let fingerprints: HashSet<Vec<u32>> = charges.iter().cloned().collect();
    let unique = fingerprints.len();

    // Mean per-neuron charge across all tokens (for neuron utilization)
    let mean_per_neuron: f64 = mean_total / dim as f64;

    (mean_total, mean_active, unique, mean_per_neuron)
}

/// W similarity: cosine similarity of flattened weight vectors
fn w_cosine(a: &Int8Projection, b: &Int8Projection) -> f64 {
    // Predict with unit vectors to extract effective weights
    // Simple approach: compare predictions on random charge patterns
    // Actually, just use a set of test vectors and compare outputs
    let mut agree = 0usize;
    let total = 1000;
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..total {
        let test: Vec<u32> = (0..PHI_DIM).map(|_| (rng.next_u64() % 16) as u32).collect();
        if a.predict(&test) == b.predict(&test) { agree += 1; }
    }
    agree as f64 / total as f64
}

struct PocketState {
    label: String,
    female: Network,
    male: Network,
    proj: Int8Projection,
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });

    let prop = PropagationConfig {
        ticks_per_token: 6, input_duration_ticks: 2,
        decay_interval_ticks: 6, use_refractory: false,
    };

    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");

    let mut seed_gen = StdRng::seed_from_u64(MASTER_SEED);
    let sdr_seed = seed_gen.next_u64();
    let sdr = SdrTable::new(CHARS, H, PHI_DIM, SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(sdr_seed)).unwrap();

    println!("=== POCKET DIAGNOSTIC ===\n");

    // Load all available states
    let mut states: Vec<PocketState> = Vec::new();

    // Gen 0: original best individuals
    let gen0_units = ["BM", "EM", "BF", "AM", "DM"];
    for name in &gen0_units {
        let f_path = format!("checkpoints/pocket_pair/{}_female.ckpt", name);
        let m_path = format!("checkpoints/pocket_pair/{}_male.ckpt", name);
        if let (Ok((f, proj, _)), Ok((m, _, _))) = (load_checkpoint(&f_path), load_checkpoint(&m_path)) {
            states.push(PocketState {
                label: format!("Gen0_{}", name), female: f, male: m, proj,
            });
        }
    }

    // Gen 1: merged
    if let (Ok((f, proj, _)), Ok((m, _, _))) = (
        load_checkpoint("checkpoints/pocket_pair/A_merged_female.ckpt"),
        load_checkpoint("checkpoints/pocket_pair/A_merged_male.ckpt"),
    ) {
        states.push(PocketState {
            label: "Gen1_merged".into(), female: f, male: m, proj,
        });
    }

    // Gen 2: continued
    if let (Ok((f, proj, _)), Ok((m, _, _))) = (
        load_checkpoint("checkpoints/pocket_continue/gen2_female.ckpt"),
        load_checkpoint("checkpoints/pocket_continue/gen2_male.ckpt"),
    ) {
        states.push(PocketState {
            label: "Gen2_cont".into(), female: f, male: m, proj,
        });
    }

    println!("Loaded {} states\n", states.len());

    // Fixed eval segment for consistent comparison
    let seg_off = 50000usize; // fixed offset
    let segment = &corpus[seg_off..seg_off + EVAL_LEN + 1];

    // =========================================================================
    // 1. EDGE TOPOLOGY COMPARISON
    // =========================================================================
    println!("=== 1. EDGE TOPOLOGY ===\n");
    println!("  {:>15} {:>8} {:>8}", "State", "F_edges", "M_edges");
    println!("  {:>15} {:>8} {:>8}", "-----", "-------", "-------");
    for s in &states {
        println!("  {:>15} {:>8} {:>8}", s.label, s.female.edge_count(), s.male.edge_count());
    }

    // Pairwise Jaccard on male pockets (that's where the merge happened)
    println!("\n  Male pocket Jaccard matrix:");
    print!("  {:>15}", "");
    for s in &states { print!(" {:>10}", &s.label[..s.label.len().min(10)]); }
    println!();
    let male_edge_sets: Vec<_> = states.iter().map(|s| edge_set(&s.male)).collect();
    for (i, s) in states.iter().enumerate() {
        print!("  {:>15}", &s.label);
        for (j, ej) in male_edge_sets.iter().enumerate() {
            if i == j { print!(" {:>10}", "--"); continue; }
            print!(" {:>9.1}%", jaccard(&male_edge_sets[i], ej) * 100.0);
        }
        println!();
    }

    // Female pocket Jaccard
    println!("\n  Female pocket Jaccard matrix:");
    print!("  {:>15}", "");
    for s in &states { print!(" {:>10}", &s.label[..s.label.len().min(10)]); }
    println!();
    let female_edge_sets: Vec<_> = states.iter().map(|s| edge_set(&s.female)).collect();
    for (i, s) in states.iter().enumerate() {
        print!("  {:>15}", &s.label);
        for (j, ej) in female_edge_sets.iter().enumerate() {
            if i == j { print!(" {:>10}", "--"); continue; }
            print!(" {:>9.1}%", jaccard(&female_edge_sets[i], ej) * 100.0);
        }
        println!();
    }

    // =========================================================================
    // 2. CHAIN ACCURACY ON SAME SEGMENT
    // =========================================================================
    println!("\n=== 2. CHAIN ACCURACY (same {} chars) ===\n", EVAL_LEN);

    let mut all_preds: Vec<Vec<usize>> = Vec::new();
    let mut all_interface: Vec<Vec<Vec<u32>>> = Vec::new();
    let mut all_output: Vec<Vec<Vec<u32>>> = Vec::new();

    for s in &mut states {
        let (interface, output, preds, correct) =
            trace_chain(&mut s.female, &mut s.male, &s.proj, segment, &sdr, &prop);

        let acc = correct as f64 / EVAL_LEN as f64;
        let (if_total, if_active, if_unique, if_per_neuron) = charge_stats(&interface);
        let (out_total, out_active, out_unique, out_per_neuron) = charge_stats(&output);

        println!("  {} accuracy: {:.1}%", s.label, acc * 100.0);
        println!("    Interface (F→M): mean_charge={:.1} active_neurons={:.1}/{} unique_patterns={}/{} per_neuron={:.2}",
            if_total, if_active, PHI_DIM, if_unique, EVAL_LEN, if_per_neuron);
        println!("    Male output:     mean_charge={:.1} active_neurons={:.1}/{} unique_patterns={}/{} per_neuron={:.2}",
            out_total, out_active, PHI_DIM, out_unique, EVAL_LEN, out_per_neuron);
        println!();

        all_preds.push(preds);
        all_interface.push(interface);
        all_output.push(output);
    }

    // =========================================================================
    // 3. PREDICTION AGREEMENT BETWEEN GENERATIONS
    // =========================================================================
    println!("=== 3. PREDICTION AGREEMENT ===\n");
    print!("  {:>15}", "");
    for s in &states { print!(" {:>10}", &s.label[..s.label.len().min(10)]); }
    println!();
    for (i, si) in states.iter().enumerate() {
        print!("  {:>15}", &si.label);
        for (j, _sj) in states.iter().enumerate() {
            if i == j { print!(" {:>10}", "--"); continue; }
            let agree = (0..EVAL_LEN)
                .filter(|&k| all_preds[i][k] == all_preds[j][k]).count();
            print!(" {:>9.1}%", agree as f64 / EVAL_LEN as f64 * 100.0);
        }
        println!();
    }

    // Oracle
    println!();
    let gen0_count = gen0_units.len().min(states.len());
    let oracle_gen0 = (0..EVAL_LEN).filter(|&k| {
        (0..gen0_count).any(|i| all_preds[i][k] == segment[k + 1] as usize)
    }).count();

    let oracle_all = (0..EVAL_LEN).filter(|&k| {
        (0..states.len()).any(|i| all_preds[i][k] == segment[k + 1] as usize)
    }).count();

    println!("  Oracle (Gen0 top-5): {:.1}%", oracle_gen0 as f64 / EVAL_LEN as f64 * 100.0);
    println!("  Oracle (all states): {:.1}%", oracle_all as f64 / EVAL_LEN as f64 * 100.0);

    // =========================================================================
    // 4. INTERFACE CHARGE SIMILARITY (F→M signal)
    // =========================================================================
    println!("\n=== 4. INTERFACE CHARGE SIMILARITY ===\n");
    println!("  How similar is the Female→Male signal across generations?");
    println!("  (cosine similarity of charge vectors, averaged over {} tokens)\n", EVAL_LEN);

    for i in 0..states.len() {
        for j in (i+1)..states.len() {
            let mut cos_sum = 0.0f64;
            let mut count = 0usize;
            for (a, b) in all_interface[i].iter().zip(&all_interface[j]) {
                let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
                let norm_a: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                let norm_b: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
                if norm_a > 0.0 && norm_b > 0.0 {
                    cos_sum += dot / (norm_a * norm_b);
                    count += 1;
                }
            }
            let mean_cos = if count > 0 { cos_sum / count as f64 } else { 0.0 };
            println!("  {} vs {}: mean cosine = {:.3} ({} valid tokens)",
                states[i].label, states[j].label, mean_cos, count);
        }
    }

    // =========================================================================
    // 5. W PROJECTION SIMILARITY
    // =========================================================================
    println!("\n=== 5. W PROJECTION SIMILARITY ===\n");
    println!("  Prediction agreement on 1000 random charge vectors:\n");
    for i in 0..states.len() {
        for j in (i+1)..states.len() {
            let sim = w_cosine(&states[i].proj, &states[j].proj);
            println!("  {} vs {}: {:.1}% agreement",
                states[i].label, states[j].label, sim * 100.0);
        }
    }

    // =========================================================================
    // 6. FEMALE-ONLY vs FULL CHAIN
    // =========================================================================
    println!("\n=== 6. FEMALE-ONLY PREDICTION (bypass Male) ===\n");
    println!("  What if we predict directly from Female output zone (no Male)?\n");
    let os = output_start();
    for s in &mut states {
        s.female.reset();
        let mut correct = 0u32;
        for i in 0..EVAL_LEN {
            s.female.propagate(sdr.pattern(segment[i] as usize), &prop).unwrap();
            if s.proj.predict(&s.female.charge()[os..H]) == segment[i + 1] as usize {
                correct += 1;
            }
        }
        println!("  {} female-only: {:.1}%", s.label, correct as f64 / EVAL_LEN as f64 * 100.0);
    }

    println!("\n=== DIAGNOSTIC COMPLETE ===");
}
