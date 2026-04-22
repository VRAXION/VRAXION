//! Chain diagnosis: trace signal through each stage, find the bottleneck.
//!
//! Tests:
//! 1. Input differentiation: do different byte-pairs produce different input vectors?
//! 2. Propagation reach: does signal from neurons 0..31 reach output zone?
//! 3. Channel isolation: does channel 2 (32..63) contribute ANYTHING?
//! 4. Output sensitivity: which neurons carry discriminative info?
//! 5. Projection exploitation: what does the projection actually use?

use instnct_core::{load_checkpoint, InitConfig, Int8Projection, VcbpTable, softmax};
use std::collections::HashSet;
use std::env;
use std::path::Path;

const MAX_CHARGE: i32 = 7;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("Usage: chain_diagnosis <checkpoint.bin> <packed.bin>");
        std::process::exit(1);
    }

    let (mut net, proj, meta) = load_checkpoint(&args[0]).expect("load");
    let table = VcbpTable::from_packed(Path::new(&args[1])).unwrap();
    let h = net.neuron_count();
    let init = InitConfig::phi(h);
    let e = table.e;
    let output_start = init.output_start();

    println!("=== CHAIN DIAGNOSIS ===");
    println!("Checkpoint: step={}, acc={:.2}%, H={h}", meta.step, meta.accuracy * 100.0);

    // Test pairs
    let test_pairs: Vec<(u16, &str)> = vec![
        (VcbpTable::pair_id(b't', b'h'), "th"),
        (VcbpTable::pair_id(b'e', b' '), "e_"),
        (VcbpTable::pair_id(b' ', b't'), "_t"),
        (VcbpTable::pair_id(b'a', b'l'), "al"),
        (VcbpTable::pair_id(b'.', b' '), "._"),
        (VcbpTable::pair_id(b'i', b'n'), "in"),
        (VcbpTable::pair_id(b' ', b'a'), "_a"),
        (VcbpTable::pair_id(b'o', b'n'), "on"),
    ];

    // ══════ TEST 1: Input differentiation ══════
    println!("\n══ TEST 1: Input Vector Differentiation ══\n");
    println!("Do different byte-pairs produce different quantized input vectors?");
    let mut input_vecs: Vec<Vec<i32>> = Vec::new();
    for (pid, label) in &test_pairs {
        let emb = table.embed_id(*pid);
        let mut input = vec![0i32; e];
        table.quantize_to_input(emb, &mut input, MAX_CHARGE);
        let nonzero = input.iter().filter(|&&v| v > 0).count();
        println!("  '{}': nonzero={}/{e}, first8={:?}", label, nonzero, &input[..8]);
        input_vecs.push(input);
    }
    // Pairwise diff
    let mut total_diffs = 0;
    let mut pairs_checked = 0;
    for i in 0..input_vecs.len() {
        for j in (i+1)..input_vecs.len() {
            let diff = input_vecs[i].iter().zip(input_vecs[j].iter())
                .filter(|(a, b)| a != b).count();
            total_diffs += diff;
            pairs_checked += 1;
        }
    }
    println!("  Avg pairwise diff: {:.1}/{e} dims ({:.0}%)",
        total_diffs as f64 / pairs_checked as f64,
        total_diffs as f64 / pairs_checked as f64 / e as f64 * 100.0);

    // ══════ TEST 2: Propagation reach per input neuron ══════
    println!("\n══ TEST 2: Per-Neuron Propagation Reach ══\n");
    println!("Which input neurons affect which output neurons?");

    // Baseline: zero input
    net.reset();
    let zero_input = vec![0i32; h];
    net.propagate(&zero_input, &init.propagation).unwrap();
    let baseline_charges: Vec<u8> = net.charge_vec(output_start..h);

    // Test: activate ONE input neuron at a time
    let mut neuron_reach = vec![0u32; h]; // how many output dims change
    for src in 0..e.min(64) {
        net.reset();
        let mut single = vec![0i32; h];
        single[src] = MAX_CHARGE;
        net.propagate(&single, &init.propagation).unwrap();
        let charges: Vec<u8> = net.charge_vec(output_start..h);
        let diffs = charges.iter().zip(baseline_charges.iter())
            .filter(|(a, b)| a != b).count();
        neuron_reach[src] = diffs as u32;
    }

    println!("  Input neurons 0..31 (C embedding zone):");
    let reach_0_31: u32 = neuron_reach[..32].iter().sum();
    let active_0_31 = neuron_reach[..32].iter().filter(|&&r| r > 0).count();
    println!("    {active_0_31}/32 neurons reach output, total impact: {reach_0_31} output-dim changes");
    println!("    Top-5: {:?}", {
        let mut v: Vec<(usize, u32)> = neuron_reach[..32].iter().enumerate().map(|(i,&r)| (i,r)).collect();
        v.sort_by(|a,b| b.1.cmp(&a.1));
        v.truncate(5);
        v
    });

    if e <= 64 {
        println!("  Input neurons 32..63 (channel 2 zone, if dual):");
        let reach_32_63: u32 = neuron_reach[32..64.min(h)].iter().sum();
        let active_32_63 = neuron_reach[32..64.min(h)].iter().filter(|&&r| r > 0).count();
        println!("    {active_32_63}/32 neurons reach output, total impact: {reach_32_63} output-dim changes");
    }

    // ══════ TEST 3: Full input → output charge diversity ══════
    println!("\n══ TEST 3: Output Charge Diversity (full pairs) ══\n");

    let mut all_charges: Vec<Vec<u8>> = Vec::new();
    let mut all_preds: Vec<usize> = Vec::new();
    for (pid, label) in &test_pairs {
        net.reset();
        let emb = table.embed_id(*pid);
        let mut input = vec![0i32; h];
        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
        net.propagate(&input, &init.propagation).unwrap();

        let charges = net.charge_vec(output_start..h);
        let pred = proj.predict(&charges);
        all_preds.push(pred);

        // Count non-zero output dims
        let nz = charges.iter().filter(|&&c| c > 0).count();
        let sum: u32 = charges.iter().map(|&c| c as u32).sum();
        println!("  '{}' → pred={:>3}, alive={:>3}/{}, sum={:>4}, charges[0..8]={:?}",
            label, pred, nz, h - output_start, sum, &charges[..8]);
        all_charges.push(charges);
    }

    // Pairwise output charge diff
    let mut out_diffs = 0usize;
    let mut out_pairs = 0usize;
    let out_dims = h - output_start;
    for i in 0..all_charges.len() {
        for j in (i+1)..all_charges.len() {
            let diff = all_charges[i].iter().zip(all_charges[j].iter())
                .filter(|(a, b)| a != b).count();
            out_diffs += diff;
            out_pairs += 1;
        }
    }
    println!("\n  Avg pairwise output diff: {:.1}/{} dims ({:.1}%)",
        out_diffs as f64 / out_pairs as f64, out_dims,
        out_diffs as f64 / out_pairs as f64 / out_dims as f64 * 100.0);
    let unique_preds: HashSet<usize> = all_preds.iter().cloned().collect();
    println!("  Unique predictions: {}/{}", unique_preds.len(), all_preds.len());

    // ══════ TEST 4: Projection weight analysis ══════
    println!("\n══ TEST 4: Projection Weight Analysis ══\n");

    // Which output dims does the projection USE most?
    // For a typical charge vector, compute which dims contribute most to scores
    let sample_charges = &all_charges[0];
    let scores = proj.raw_scores(sample_charges);
    let max_score = *scores.iter().max().unwrap_or(&0);
    let min_score = *scores.iter().min().unwrap_or(&0);
    let pred = scores.iter().enumerate().max_by_key(|(_, &s)| s).unwrap().0;
    println!("  Score range: [{min_score}, {max_score}], predicted class: {pred}");
    println!("  Score distribution (top-5 classes):");
    let mut score_sorted: Vec<(usize, i32)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    score_sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for &(cls, score) in score_sorted.iter().take(5) {
        println!("    class {:>3}: score={:>6}", cls, score);
    }

    // ══════ TEST 5: Sequential context effect ══════
    println!("\n══ TEST 5: Sequential Context Effect ══\n");
    println!("Does the recurrent state carry information?");

    // Sequence: "th" → "e_" → "_t" → "he"
    let sequence = vec![
        (VcbpTable::pair_id(b't', b'h'), "th"),
        (VcbpTable::pair_id(b'e', b' '), "e_"),
        (VcbpTable::pair_id(b' ', b't'), "_t"),
        (VcbpTable::pair_id(b'h', b'e'), "he"),
    ];

    // Run with full sequence (recurrent state accumulates)
    net.reset();
    let mut seq_preds = Vec::new();
    for (pid, label) in &sequence {
        let emb = table.embed_id(*pid);
        let mut input = vec![0i32; h];
        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
        net.propagate(&input, &init.propagation).unwrap();
        let charges = net.charge_vec(output_start..h);
        let pred = proj.predict(&charges);
        seq_preds.push(pred);
        let nz = charges.iter().filter(|&&c| c > 0).count();
        println!("  seq '{}' → pred={:>3}, alive={}", label, pred, nz);
    }

    // Compare: same tokens but ISOLATED (reset between each)
    let mut iso_preds = Vec::new();
    for (pid, label) in &sequence {
        net.reset();
        let emb = table.embed_id(*pid);
        let mut input = vec![0i32; h];
        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
        net.propagate(&input, &init.propagation).unwrap();
        let charges = net.charge_vec(output_start..h);
        let pred = proj.predict(&charges);
        iso_preds.push(pred);
        println!("  iso '{}' → pred={:>3}", label, pred);
    }

    let ctx_diffs = seq_preds.iter().zip(iso_preds.iter())
        .filter(|(s, i)| s != i).count();
    println!("\n  Context-dependent predictions: {}/{} differ",
        ctx_diffs, sequence.len());
    if ctx_diffs == 0 {
        println!("  ⚠ NO context effect — recurrent state not carrying info!");
    } else {
        println!("  ✓ Context effect present — recurrent state IS active");
    }
}
