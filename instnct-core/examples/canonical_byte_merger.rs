//! Canonical 2-Byte Merger — L1 Byte Merger (FROZEN)
//!
//! The WINNER from activation sweep over backprop STE search.
//! LINEAR activation with int8 weights: simplest, fastest, 100% int8 output.
//!
//! Architecture:
//!   2 neurons, 4 inputs each (2 LUT_2N values per byte × 2 bytes)
//!   All weights int8 (range [-12, +12])
//!   All biases int8
//!   LINEAR activation (no C19, no float) — pure integer dot product
//!   Output: scale to int8 → 729/729 distinct pairs
//!
//! Hardware deployment:
//!   LUT path: (char_a, char_b) → MERGER_LUT[a*27+b] → [int8, int8]
//!   729 entries × 2 bytes = 1458 bytes total.
//!   ZERO compute — one memory read per byte pair.
//!
//! Pipeline:
//!   L0: raw byte → char (0-26)           [zero compute, byte mask]
//!   L1: (char_a, char_b) → merger LUT    [zero compute, 1458 bytes]
//!   Output: 2048 bytes → 1024 pairs × 2 int8 = 2048 int8
//!   Compression: 2× (4096 → 2048 int8)
//!   Conv benefit: k=3 covers 6 original bytes (was 3)
//!
//! Merges 27×27 = 729 possible byte-pairs with 100% round-trip fidelity.
//! Found by backprop STE, verified exhaustively on all 729 combinations.
//!
//! Run: cargo run --example canonical_byte_merger --release

use std::time::Instant;

// ============================================================
// L0 LUT (from canonical_byte_encoder.rs) — needed for input
// ============================================================

const LUT_2N: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

// ============================================================
// CANONICAL MERGER WEIGHTS — do not modify
// Linear activation, int8 weights, backprop STE verified.
// ============================================================

const MERGER_W: [[i8; 4]; 2] = [
    [-8, 12, -7, -12],  // N0
    [-8, -6, -1,   2],  // N1
];
const MERGER_B: [i8; 2] = [-1, 10];

// ============================================================
// FROZEN MERGER LUT — deploy path (zero compute, 1458 bytes)
// merger_lut[char_a * 27 + char_b] → [int8, int8]
// Precomputed from linear dot products, scaled to int8 range.
// ============================================================

fn compute_merger_lut() -> [[i8; 2]; 729] {
    // Step 1: compute raw integer dot products for all 729 pairs
    let mut raw = [[0i32; 2]; 729];
    for a in 0..27u8 {
        for b in 0..27u8 {
            let idx = a as usize * 27 + b as usize;
            let in0 = LUT_2N[a as usize][0] as i32;
            let in1 = LUT_2N[a as usize][1] as i32;
            let in2 = LUT_2N[b as usize][0] as i32;
            let in3 = LUT_2N[b as usize][1] as i32;
            for k in 0..2 {
                raw[idx][k] = MERGER_B[k] as i32
                    + MERGER_W[k][0] as i32 * in0
                    + MERGER_W[k][1] as i32 * in1
                    + MERGER_W[k][2] as i32 * in2
                    + MERGER_W[k][3] as i32 * in3;
            }
        }
    }

    // Step 2: find range per dimension
    let mut min = [i32::MAX; 2];
    let mut max = [i32::MIN; 2];
    for r in &raw {
        for k in 0..2 {
            if r[k] < min[k] { min[k] = r[k]; }
            if r[k] > max[k] { max[k] = r[k]; }
        }
    }

    // Step 3: scale to [-127, 127]
    let mut lut = [[0i8; 2]; 729];
    for i in 0..729 {
        for k in 0..2 {
            let range = max[k] - min[k];
            let scaled = if range > 0 {
                ((raw[i][k] - min[k]) as f32 / range as f32 * 254.0 - 127.0).round() as i8
            } else {
                0i8
            };
            lut[i][k] = scaled;
        }
    }
    lut
}

// ============================================================
// Encode functions
// ============================================================

/// Encode a pair of characters using the merger LUT
fn encode_pair(char_a: u8, char_b: u8, lut: &[[i8; 2]; 729]) -> [i8; 2] {
    lut[char_a as usize * 27 + char_b as usize]
}

/// Decode: find which (char_a, char_b) pair produced this code
fn decode_pair(code: [i8; 2], lut: &[[i8; 2]; 729]) -> (u8, u8) {
    let mut best_idx = 0;
    let mut best_d = i32::MAX;
    for i in 0..729 {
        let d = (code[0] as i32 - lut[i][0] as i32).pow(2)
              + (code[1] as i32 - lut[i][1] as i32).pow(2);
        if d < best_d { best_d = d; best_idx = i; }
    }
    ((best_idx / 27) as u8, (best_idx % 27) as u8)
}

/// Encode a full chunk: 2048 bytes → 1024 pairs × 2 int8
fn encode_chunk(chars: &[u8], lut: &[[i8; 2]; 729]) -> Vec<[i8; 2]> {
    chars.chunks(2).map(|pair| {
        let a = pair[0];
        let b = if pair.len() > 1 { pair[1] } else { 26 }; // pad with space
        encode_pair(a, b, lut)
    }).collect()
}

fn main() {
    let t0 = Instant::now();

    println!("=== CANONICAL 2-BYTE MERGER (L1 FROZEN) ===\n");

    // Compute LUT
    let lut = compute_merger_lut();

    // ── Verify 100% round-trip on all 729 pairs ──
    println!("--- Verification: all 729 byte-pairs ---\n");

    let mut ok = 0;
    let mut collisions = Vec::new();
    for a in 0..27u8 {
        for b in 0..27u8 {
            let code = encode_pair(a, b, &lut);
            let (da, db) = decode_pair(code, &lut);
            if da == a && db == b {
                ok += 1;
            } else {
                collisions.push((a, b, da, db));
            }
        }
    }

    println!("  Round-trip: {}/729", ok);
    if ok == 729 {
        println!("  *** 100% — ALL PAIRS VERIFIED ***\n");
    } else {
        println!("  COLLISIONS:");
        for (a, b, da, db) in &collisions {
            println!("    ({},{}) → decoded ({},{})  WRONG", a, b, da, db);
        }
        println!();
    }

    // ── Uniqueness check ──
    let mut unique = std::collections::HashSet::new();
    for i in 0..729 { unique.insert((lut[i][0], lut[i][1])); }
    println!("  Unique int8 pairs: {}/729", unique.len());

    // ── Min distance between any two codes ──
    let mut min_d = i32::MAX;
    for i in 0..729 {
        for j in (i+1)..729 {
            let d = (lut[i][0] as i32 - lut[j][0] as i32).pow(2)
                  + (lut[i][1] as i32 - lut[j][1] as i32).pow(2);
            if d < min_d { min_d = d; }
        }
    }
    println!("  Min pairwise distance²: {}", min_d);

    // ── Print sample entries ──
    println!("\n--- Sample LUT entries ---\n");
    let chars = "abcdefghijklmnopqrstuvwxyz ";
    let samples = [(0,0),(0,1),(4,18),(19,7),(26,4),(7,4),(19,7),(26,26)];
    for &(a, b) in &samples {
        let ca = chars.as_bytes()[a as usize] as char;
        let cb = chars.as_bytes()[b as usize] as char;
        let code = lut[a as usize * 27 + b as usize];
        println!("    '{}{}' → [{:>4}, {:>4}]", ca, cb, code[0], code[1]);
    }

    // ── Pipeline demo ──
    println!("\n--- Pipeline demo: \"hello world\" ---\n");
    let text = "hello world";
    let text_chars: Vec<u8> = text.bytes().map(|b| match b {
        b'a'..=b'z' => b - b'a',
        b' ' => 26,
        _ => 26,
    }).collect();

    // Pad to even length
    let mut padded = text_chars.clone();
    if padded.len() % 2 != 0 { padded.push(26); }

    let encoded = encode_chunk(&padded, &lut);
    print!("  Input:  \"{}\"  chars=", text);
    for &c in &text_chars { print!("{} ", c); } println!();

    print!("  Merged: ");
    for (i, code) in encoded.iter().enumerate() {
        let pair_chars: String = [padded[i*2], padded[i*2+1]].iter()
            .map(|&c| chars.as_bytes()[c as usize] as char).collect();
        print!("'{}'=[{:>4},{:>4}] ", pair_chars, code[0], code[1]);
    }
    println!();

    // Decode back
    print!("  Decoded: \"");
    for code in &encoded {
        let (a, b) = decode_pair(*code, &lut);
        print!("{}{}", chars.as_bytes()[a as usize] as char, chars.as_bytes()[b as usize] as char);
    }
    println!("\"");

    // ── Stats ──
    println!("\n--- Deploy stats ---\n");
    println!("  LUT size: 729 × 2 = 1458 bytes");
    println!("  Input:  2048 bytes (one chunk)");
    println!("  Output: 1024 pairs × 2 int8 = 2048 int8 values");
    println!("  Compression: 2× sequence length reduction");
    println!("  Compute: zero (one memory read per pair)");
    println!("  Weights: N0=[-8,12,-7,-12] b=-1  N1=[-8,-6,-1,2] b=10");
    println!("  Activation: LINEAR (pure integer dot product)");

    println!("\n  Total time: {:.3}s", t0.elapsed().as_secs_f64());
}
