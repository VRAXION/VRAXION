//! Adversarial test: does the Brain actually distinguish inputs?
//!
//! Feeds N different byte-pairs, records output charge vector + prediction.
//! Reports: unique outputs, unique predictions, input-dependence.

use instnct_core::{load_checkpoint, InitConfig, VcbpTable};
use std::collections::{HashMap, HashSet};
use std::env;
use std::path::Path;

const MAX_CHARGE: i32 = 7;
const N_TEST: usize = 200;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("Usage: adversarial_test <checkpoint.bin> <packed.bin>");
        std::process::exit(1);
    }

    let (mut net, proj, meta) = load_checkpoint(&args[0]).expect("load");
    let table = VcbpTable::from_packed(Path::new(&args[1])).unwrap();
    let h = net.neuron_count();
    let init = InitConfig::phi(h);
    let e = table.e;

    println!("Checkpoint: step={}, acc={:.2}%", meta.step, meta.accuracy * 100.0);
    println!("Testing {} different inputs...\n", N_TEST);

    // Collect hot pair IDs
    let hot_ids: Vec<u16> = (0..65536u32)
        .filter(|&v| table.is_hot(v as u16))
        .map(|v| v as u16)
        .collect();

    let mut unique_charges: HashSet<Vec<u8>> = HashSet::new();
    let mut unique_predictions: HashSet<usize> = HashSet::new();
    let mut prediction_counts: HashMap<usize, u32> = HashMap::new();
    let mut input_to_pred: Vec<(u16, usize, Vec<u8>)> = Vec::new();

    // Test 1: Different inputs, fresh reset each time
    println!("=== TEST 1: Fresh reset per input ===");
    for i in 0..N_TEST.min(hot_ids.len()) {
        let pair_id = hot_ids[i];
        net.reset();

        let emb = table.embed_id(pair_id);
        let mut input = vec![0i32; h];
        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
        net.propagate(&input, &init.propagation).unwrap();

        let charges = net.charge_vec(init.output_start()..h);
        let pred = proj.predict(&charges);

        unique_charges.insert(charges.clone());
        unique_predictions.insert(pred);
        *prediction_counts.entry(pred).or_insert(0) += 1;
        input_to_pred.push((pair_id, pred, charges));
    }

    println!("  Unique charge patterns: {}/{N_TEST}", unique_charges.len());
    println!("  Unique predictions:     {}/{N_TEST}", unique_predictions.len());
    println!("  Top-5 predicted classes:");
    let mut pred_sorted: Vec<_> = prediction_counts.iter().collect();
    pred_sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (pred, count) in pred_sorted.iter().take(5) {
        let pct = **count as f64 / N_TEST as f64 * 100.0;
        println!("    class {:>4}: {count:>4}× ({pct:.1}%)", pred);
    }

    // Show a few examples
    println!("\n  Sample predictions:");
    println!("  {:>8} {:>6} {:>20}", "Input", "Pred", "Charge[0..8]");
    for (pid, pred, charges) in input_to_pred.iter().take(10) {
        let (hi, lo) = VcbpTable::pair_bytes(*pid);
        let label = format!("{}{}",
            if hi.is_ascii_graphic() || hi == b' ' { hi as char } else { '.' },
            if lo.is_ascii_graphic() || lo == b' ' { lo as char } else { '.' });
        let ch_str: Vec<String> = charges[..8.min(charges.len())].iter().map(|c| format!("{c}")).collect();
        println!("  {:>8} {:>6} {:>20}", label, pred, ch_str.join(","));
    }

    // Test 2: Same input twice — determinism check
    println!("\n=== TEST 2: Determinism (same input twice) ===");
    let test_id = hot_ids[0];
    net.reset();
    let emb = table.embed_id(test_id);
    let mut input = vec![0i32; h];
    table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
    net.propagate(&input, &init.propagation).unwrap();
    let charges_a = net.charge_vec(init.output_start()..h);

    net.reset();
    net.propagate(&input, &init.propagation).unwrap();
    let charges_b = net.charge_vec(init.output_start()..h);
    println!("  Identical: {}", charges_a == charges_b);

    // Test 3: Opposite inputs — maximally different embeddings
    println!("\n=== TEST 3: Maximally different inputs ===");
    // Find two hot pairs with maximum L2 distance
    let mut max_dist = 0.0f32;
    let mut pair_a = 0u16;
    let mut pair_b = 0u16;
    for i in 0..100.min(hot_ids.len()) {
        for j in (i+1)..100.min(hot_ids.len()) {
            let ea = table.embed_id(hot_ids[i]);
            let eb = table.embed_id(hot_ids[j]);
            let dist: f32 = ea.iter().zip(eb.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
            if dist > max_dist {
                max_dist = dist;
                pair_a = hot_ids[i];
                pair_b = hot_ids[j];
            }
        }
    }
    let (ah, al) = VcbpTable::pair_bytes(pair_a);
    let (bh, bl) = VcbpTable::pair_bytes(pair_b);
    println!("  Most distant pair: '{}{}' vs '{}{}'  (L2={:.2})",
        ah as char, al as char, bh as char, bl as char, max_dist.sqrt());

    // Feed pair A
    net.reset();
    let emb_a = table.embed_id(pair_a);
    let mut inp_a = vec![0i32; h];
    table.quantize_to_input(emb_a, &mut inp_a[..e], MAX_CHARGE);
    net.propagate(&inp_a, &init.propagation).unwrap();
    let ch_a = net.charge_vec(init.output_start()..h);
    let pred_a = proj.predict(&ch_a);

    // Feed pair B
    net.reset();
    let emb_b = table.embed_id(pair_b);
    let mut inp_b = vec![0i32; h];
    table.quantize_to_input(emb_b, &mut inp_b[..e], MAX_CHARGE);
    net.propagate(&inp_b, &init.propagation).unwrap();
    let ch_b = net.charge_vec(init.output_start()..h);
    let pred_b = proj.predict(&ch_b);

    // Compare
    let diff_dims = ch_a.iter().zip(ch_b.iter()).filter(|(a, b)| a != b).count();
    let total_dims = ch_a.len();
    println!("  Charge diff: {diff_dims}/{total_dims} dims differ ({:.1}%)",
        diff_dims as f64 / total_dims as f64 * 100.0);
    println!("  Pred A: {pred_a}  Pred B: {pred_b}  Same? {}", pred_a == pred_b);
    println!("  Charge A[0..8]: {:?}", &ch_a[..8.min(ch_a.len())]);
    println!("  Charge B[0..8]: {:?}", &ch_b[..8.min(ch_b.len())]);

    // Test 4: Sequential context — does prior input affect next output?
    println!("\n=== TEST 4: Context sensitivity (sequential tokens) ===");
    net.reset();
    // Feed "th" then read
    let emb1 = table.embed_id(VcbpTable::pair_id(b't', b'h'));
    let mut inp1 = vec![0i32; h];
    table.quantize_to_input(emb1, &mut inp1[..e], MAX_CHARGE);
    net.propagate(&inp1, &init.propagation).unwrap();
    let after_th = net.charge_vec(init.output_start()..h);
    let pred_after_th = proj.predict(&after_th);

    // Feed "e " on top (no reset)
    let emb2 = table.embed_id(VcbpTable::pair_id(b'e', b' '));
    let mut inp2 = vec![0i32; h];
    table.quantize_to_input(emb2, &mut inp2[..e], MAX_CHARGE);
    net.propagate(&inp2, &init.propagation).unwrap();
    let after_th_e = net.charge_vec(init.output_start()..h);
    let pred_after_th_e = proj.predict(&after_th_e);

    // Compare: feed "e " alone (fresh)
    net.reset();
    net.propagate(&inp2, &init.propagation).unwrap();
    let after_e_only = net.charge_vec(init.output_start()..h);
    let pred_after_e_only = proj.predict(&after_e_only);

    let ctx_diff = after_th_e.iter().zip(after_e_only.iter()).filter(|(a, b)| a != b).count();
    println!("  After 'th'→'e_' vs 'e_' alone:");
    println!("    Charge diff: {ctx_diff}/{total_dims} dims ({:.1}%)",
        ctx_diff as f64 / total_dims as f64 * 100.0);
    println!("    Pred 'th'→'e_': {pred_after_th_e}  Pred 'e_' alone: {pred_after_e_only}  Differ? {}",
        pred_after_th_e != pred_after_e_only);
    println!("    Pred after 'th': {pred_after_th}");
}
