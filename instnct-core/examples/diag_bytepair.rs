//! Diagnostic: inspect what the ABC↔Brain actually produces.
//!
//! Runs a few tokens, dumps:
//! - Input charges (quantized C embedding)
//! - Output charges (Brain output neurons)
//! - Dequantized output vector
//! - Nearest neighbor prediction vs actual target
//! - Smooth fitness (cosine similarity) per token
//! - Charge statistics (are outputs stuck at 0? all same value?)

use instnct_core::{build_network, InitConfig, VcbpTable};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::Path;

const MAX_CHARGE: u8 = 7;
const DIAG_TOKENS: usize = 20;

fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn main() {
    let corpus_path = "instnct-core/tests/fixtures/alice_corpus.txt";
    let packed_path = "output/block_c_bytepair_champion/packed.bin";

    let table = VcbpTable::from_packed(Path::new(packed_path)).unwrap();
    let corpus = std::fs::read(corpus_path).unwrap();

    println!("=== ABC↔Brain Diagnostic ===\n");

    // Build network
    let h = 52;
    let mut init = InitConfig::phi(h);
    init.propagation.ticks_per_token = 4;
    init.propagation.decay_interval_ticks = 4;

    let mut rng = StdRng::seed_from_u64(42);
    let mut net = build_network(&init, &mut rng);

    let output_start = init.output_start();
    let e = table.e;

    println!("Network: H={h}, phi_dim={}, edges={}", init.phi_dim, net.edge_count());
    println!("Input zone: 0..{}", init.input_end());
    println!("Output zone: {}..{}", output_start, h);
    println!("Overlap: {}..{}\n", output_start, init.input_end());

    // Track charge statistics
    let mut all_out_charges: Vec<Vec<u8>> = Vec::new();
    let mut all_cosines: Vec<f32> = Vec::new();

    net.reset();

    println!("{:<6} {:<12} {:<12} {:<8} {:<8} {:<8}",
        "Token", "Input pair", "Target pair", "Predict", "Cos sim", "Match?");
    println!("{}", "-".repeat(60));

    for i in 0..DIAG_TOKENS {
        let idx = 1000 + i * 2; // start from offset 1000
        if idx + 3 >= corpus.len() { break; }

        let cur_id = ((corpus[idx] as u16) << 8) | corpus[idx + 1] as u16;
        let tgt_id = ((corpus[idx + 2] as u16) << 8) | corpus[idx + 3] as u16;

        // Input
        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; h];
        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE as i32);

        net.propagate(&input, &init.propagation).unwrap();

        // Output
        let out_charges = net.charge_vec(output_start..output_start + e);
        let query = table.dequantize_output(&out_charges, MAX_CHARGE);
        let pred_id = table.nearest_hot(&query);

        // Cosine similarity
        let tgt_emb = table.embed_id(tgt_id);
        let cos = cosine_f32(&query, tgt_emb);

        let cur_hi = (cur_id >> 8) as u8;
        let cur_lo = (cur_id & 0xFF) as u8;
        let tgt_hi = (tgt_id >> 8) as u8;
        let tgt_lo = (tgt_id & 0xFF) as u8;
        let pred_hi = (pred_id >> 8) as u8;
        let pred_lo = (pred_id & 0xFF) as u8;

        let cur_s = format!("{}{}",
            if cur_hi.is_ascii_graphic() || cur_hi == b' ' { cur_hi as char } else { '.' },
            if cur_lo.is_ascii_graphic() || cur_lo == b' ' { cur_lo as char } else { '.' });
        let tgt_s = format!("{}{}",
            if tgt_hi.is_ascii_graphic() || tgt_hi == b' ' { tgt_hi as char } else { '.' },
            if tgt_lo.is_ascii_graphic() || tgt_lo == b' ' { tgt_lo as char } else { '.' });
        let pred_s = format!("{}{}",
            if pred_hi.is_ascii_graphic() || pred_hi == b' ' { pred_hi as char } else { '.' },
            if pred_lo.is_ascii_graphic() || pred_lo == b' ' { pred_lo as char } else { '.' });

        let matched = if pred_id == tgt_id { "YES" } else { "" };

        println!("{:<6} {:<12} {:<12} {:<8} {:<8.4} {:<8}",
            i, cur_s, tgt_s, pred_s, cos, matched);

        all_out_charges.push(out_charges.to_vec());
        all_cosines.push(cos);
    }

    // Charge statistics
    println!("\n=== Output Charge Statistics ===\n");
    let e = table.e;
    let n = all_out_charges.len();

    // Per-dimension stats
    println!("{:<6} {:<6} {:<6} {:<6} {:<8}", "Dim", "Min", "Max", "Mean", "Unique");
    println!("{}", "-".repeat(36));
    let mut total_nonzero = 0usize;
    let mut total_unique = 0usize;
    for d in 0..e {
        let vals: Vec<u8> = all_out_charges.iter().map(|c| c[d]).collect();
        let min = *vals.iter().min().unwrap();
        let max = *vals.iter().max().unwrap();
        let mean = vals.iter().map(|&v| v as f32).sum::<f32>() / n as f32;
        let mut unique: Vec<u8> = vals.clone();
        unique.sort();
        unique.dedup();
        let n_unique = unique.len();
        if d < 8 || d >= e - 2 {
            println!("{:<6} {:<6} {:<6} {:<6.1} {:<8}", d, min, max, mean, n_unique);
        } else if d == 8 {
            println!("  ...");
        }
        total_nonzero += vals.iter().filter(|&&v| v > 0).count();
        total_unique += n_unique;
    }

    let total_cells = n * e;
    println!("\nNon-zero charges: {}/{} ({:.1}%)",
        total_nonzero, total_cells,
        total_nonzero as f64 / total_cells as f64 * 100.0);
    println!("Avg unique values per dim: {:.1}/{}", total_unique as f64 / e as f64, n);

    // Cosine stats
    println!("\n=== Cosine Similarity Stats ===\n");
    let cos_mean = all_cosines.iter().sum::<f32>() / all_cosines.len() as f32;
    let cos_min = all_cosines.iter().cloned().fold(f32::MAX, f32::min);
    let cos_max = all_cosines.iter().cloned().fold(f32::MIN, f32::max);
    println!("Mean: {:.4}  Min: {:.4}  Max: {:.4}", cos_mean, cos_min, cos_max);
    println!("(random vectors in 32-d: expected cosine ≈ 0.0)");

    // Input charge sample
    println!("\n=== Sample Input Charges (first token) ===\n");
    let first_id = ((corpus[1000] as u16) << 8) | corpus[1001] as u16;
    let first_emb = table.embed_id(first_id);
    let mut first_input = vec![0i32; e];
    table.quantize_to_input(first_emb, &mut first_input, MAX_CHARGE as i32);
    println!("Embedding: {:?}", &first_emb[..8]);
    println!("Quantized: {:?}", &first_input[..8]);
}
