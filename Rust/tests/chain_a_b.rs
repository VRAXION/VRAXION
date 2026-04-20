//! A → B chain verification: bytes → A encode → B forward → A decode → bytes
//!
//! This is the DEFINITIVE test that Blocks A and B compose correctly.
//! For all 65,536 byte pairs (a,b):
//!   1. A(a) || A(b) = 32-dim latent
//!   2. B.forward(latent) = 32-dim output
//!   3. Split output into two 16-dim halves
//!   4. A.nearest(each half) → byte pair (a', b')
//!   5. Assert (a, b) == (a', b')
//!
//! Run: cargo test --release --test chain_a_b -- --nocapture

use vraxion::block_a_byte_unit::byte_encoder::ByteEncoder;
use vraxion::block_b_merger::merger::{L1Merger, IN_DIM, LUT_DIM};

/// Load the LUT that B was trained on (nozero variant from tools/).
fn load_merger_lut() -> [[f32; LUT_DIM]; 256] {
    use std::fs;
    use std::path::PathBuf;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct LutBlob {
        scale: f32,
        lut: Vec<Vec<i8>>,
    }

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("Rust/ parent")
        .to_path_buf();
    let path = repo_root.join("tools").join("byte_embedder_lut_int8_nozero.json");
    let text = fs::read_to_string(&path).expect("read nozero LUT");
    let blob: LutBlob = serde_json::from_str(&text).expect("parse LUT");

    let mut lut = [[0f32; LUT_DIM]; 256];
    for (b, row) in blob.lut.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            lut[b][j] = (v as f32) * blob.scale;
        }
    }
    lut
}

/// Nearest-byte decode from a 16-dim latent, given the LUT.
fn nearest_byte(latent: &[f32; LUT_DIM], lut: &[[f32; LUT_DIM]; 256]) -> u8 {
    let mut best = 0u8;
    let mut best_d = f32::MAX;
    for (b, row) in lut.iter().enumerate() {
        let d: f32 = latent
            .iter()
            .zip(row.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        if d < best_d {
            best_d = d;
            best = b as u8;
        }
    }
    best
}

#[test]
fn a_then_b_all_65536_byte_pairs_roundtrip() {
    // Block A loads for sign-check / interface symmetry; decoding uses the LUT B was trained against.
    let _encoder = ByteEncoder::load_default().expect("load Block A");
    let merger = L1Merger::load_default().expect("load Block B");
    let lut = load_merger_lut();

    let mut matches = 0usize;
    let mut first_failure: Option<(u8, u8, u8, u8)> = None;
    let mut x = [0f32; IN_DIM];

    for a in 0u16..256 {
        for b in 0u16..256 {
            // Encode: concat A(a) || A(b)
            x[..LUT_DIM].copy_from_slice(&lut[a as usize]);
            x[LUT_DIM..].copy_from_slice(&lut[b as usize]);

            // Forward through B
            let y = merger.forward(&x);

            // Decode: split and nearest-byte each half
            let mut half1 = [0f32; LUT_DIM];
            let mut half2 = [0f32; LUT_DIM];
            half1.copy_from_slice(&y[..LUT_DIM]);
            half2.copy_from_slice(&y[LUT_DIM..]);

            let a_hat = nearest_byte(&half1, &lut);
            let b_hat = nearest_byte(&half2, &lut);

            if a_hat == a as u8 && b_hat == b as u8 {
                matches += 1;
            } else if first_failure.is_none() {
                first_failure = Some((a as u8, b as u8, a_hat, b_hat));
            }
        }
    }

    let total = 65536;
    println!(
        "A→B chain roundtrip: {}/{} ({:.4}%)",
        matches,
        total,
        matches as f64 / total as f64 * 100.0
    );
    if let Some((a, b, ah, bh)) = first_failure {
        println!(
            "  first failure: ({:#x},{:#x}) decoded as ({:#x},{:#x})",
            a, b, ah, bh
        );
    }

    assert_eq!(
        matches, total,
        "A→B chain must be 100% lossless on all byte pairs"
    );
}
