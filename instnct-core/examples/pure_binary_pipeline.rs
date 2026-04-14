//! Pure binary pipeline — no floats anywhere!
//!
//! Byte encoder: POPCOUNT → integer → bit decompose → binary bits
//! Mixer: POPCOUNT on the binary bits → pure binary end-to-end
//!
//! Test: does integer bit-decomposition (no C19) still give 100% round-trip?
//! Then: can a binary-weight mixer work on these bits?
//!
//! Run: cargo run --example pure_binary_pipeline --release

use std::time::Instant;

// Byte encoder: binary weights, NO C19, output = raw integer sum
fn encode_int(ch: u8, weights: &[[i8;8]], biases: &[i8], n: usize) -> Vec<i8> {
    let mut bits = [0u8;8];
    for i in 0..8 { bits[i] = (ch >> i) & 1; }
    (0..n).map(|k| {
        let mut d = biases[k] as i16;
        for j in 0..8 { d += weights[k][j] as i16 * bits[j] as i16; }
        d as i8  // range: -9..+7
    }).collect()
}

// Bit-decompose integer into binary representation
// d ∈ {-9..+7} → 5 bits (sign + 4 magnitude, or offset binary)
// Simpler: offset by 9 → range 0..16 → 5 bits unsigned
fn int_to_bits(d: i8) -> [u8; 5] {
    let offset = (d as i16 + 9) as u8; // 0..16
    let mut bits = [0u8; 5];
    for i in 0..5 { bits[i] = (offset >> i) & 1; }
    bits
}

// Full encoder: byte → N neurons → N×5 binary bits
fn encode_full(ch: u8, weights: &[[i8;8]], biases: &[i8], n: usize) -> Vec<u8> {
    let ints = encode_int(ch, weights, biases, n);
    ints.iter().flat_map(|&d| int_to_bits(d).to_vec()).collect()
}

// Round-trip: encode all 27, check nearest-neighbor in binary hamming OR euclidean
fn roundtrip_euclidean(weights: &[[i8;8]], biases: &[i8], n: usize) -> usize {
    let codes: Vec<Vec<u8>> = (0..27u8).map(|ch| encode_full(ch, weights, biases, n)).collect();
    let mut ok = 0;
    for i in 0..27 {
        let mut best = 0; let mut bd = u32::MAX;
        for j in 0..27 {
            let d: u32 = codes[i].iter().zip(&codes[j]).map(|(&a,&b)| ((a as i32 - b as i32).pow(2)) as u32).sum();
            if d < bd { bd = d; best = j; }
        }
        if best == i { ok += 1; }
    }
    ok
}

fn roundtrip_hamming(weights: &[[i8;8]], biases: &[i8], n: usize) -> usize {
    let codes: Vec<Vec<u8>> = (0..27u8).map(|ch| encode_full(ch, weights, biases, n)).collect();
    let mut ok = 0;
    for i in 0..27 {
        let mut best = 0; let mut bd = u32::MAX;
        for j in 0..27 {
            let d: u32 = codes[i].iter().zip(&codes[j]).map(|(&a,&b)| (a ^ b) as u32).sum(); // hamming
            if d < bd { bd = d; best = j; }
        }
        if best == i { ok += 1; }
    }
    ok
}

// Greedy exhaustive search (binary weights, no C19)
fn greedy_search(max_neurons: usize) -> (Vec<[i8;8]>, Vec<i8>, usize, usize) {
    let mut weights: Vec<[i8;8]> = Vec::new();
    let mut biases: Vec<i8> = Vec::new();

    for neuron in 0..max_neurons {
        let mut top = 0; let mut tw = [0i8;8]; let mut tb = 0i8;
        for combo in 0u32..512 {
            let mut w = [0i8;8];
            for j in 0..8 { w[j] = if (combo>>j)&1==1 { 1 } else { -1 }; }
            let b = if (combo>>8)&1==1 { 1i8 } else { -1 };
            let mut test_w = weights.clone(); test_w.push(w);
            let mut test_b = biases.clone(); test_b.push(b);
            let score = roundtrip_euclidean(&test_w, &test_b, neuron+1);
            if score > top { top = score; tw = w; tb = b; if top == 27 { break; } }
        }
        weights.push(tw);
        biases.push(tb);
        let euc = roundtrip_euclidean(&weights, &biases, neuron+1);
        let ham = roundtrip_hamming(&weights, &biases, neuron+1);
        println!("  N{}: {:?} b={:+} → euclidean={}/27 hamming={}/27",
            neuron, tw, tb, euc, ham);
        if euc == 27 { return (weights, biases, neuron+1, euc); }
    }
    let n = weights.len();
    let euc = roundtrip_euclidean(&weights, &biases, n);
    (weights, biases, n, euc)
}

fn main() {
    let t0 = Instant::now();
    println!("=== PURE BINARY PIPELINE — NO FLOATS ===\n");
    println!("  Byte → binary weights POPCOUNT → integer sum → bit decompose → binary bits");
    println!("  NO C19, NO float anywhere!\n");

    // Search for optimal binary encoder (no C19)
    println!("━━━ Greedy exhaustive search (no C19, integer output) ━━━\n");
    let (weights, biases, n_neurons, score) = greedy_search(8);

    if score == 27 {
        println!("\n  ★★★ 100% with {} neurons, {} binary bits output per byte ★★★\n", n_neurons, n_neurons * 5);
    } else {
        println!("\n  Best: {}/27 with {} neurons\n", score, n_neurons);
    }

    // Show the codebook
    println!("━━━ Codebook (integer sums → bit-decomposed) ━━━\n");
    println!("  {:>5} {:>12} {:>25}", "char", "int_sums", "binary_bits");
    println!("  {}", "─".repeat(45));
    for ch in 0..27u8 {
        let ints = encode_int(ch, &weights, &biases, n_neurons);
        let bits = encode_full(ch, &weights, &biases, n_neurons);
        let name = if ch == 26 { "space".to_string() } else { format!("'{}'", (ch + b'a') as char) };
        let int_str: String = ints.iter().map(|d| format!("{:+}", d)).collect::<Vec<_>>().join(",");
        let bit_str: String = bits.iter().map(|b| format!("{}", b)).collect();
        println!("  {:>5} {:>12} {:>25}", name, int_str, bit_str);
    }

    // Verify: all codes unique in hamming space?
    let codes: Vec<Vec<u8>> = (0..27u8).map(|ch| encode_full(ch, &weights, &biases, n_neurons)).collect();
    let mut min_hamming = u32::MAX;
    for i in 0..27 { for j in i+1..27 {
        let d: u32 = codes[i].iter().zip(&codes[j]).map(|(&a,&b)| (a^b) as u32).sum();
        if d < min_hamming { min_hamming = d; }
    }}
    println!("\n  Min hamming distance between any two codes: {}", min_hamming);
    println!("  (need ≥1 for unique, ≥2 for 1-bit error correction)\n");

    // Now test: binary mixer on these bits
    if score == 27 {
        println!("━━━ Binary mixer test (16 bytes → POPCOUNT merger) ━━━\n");
        let n_bits = n_neurons * 5;
        let chunk = 16;
        let input_bits = chunk * n_bits;
        println!("  16 bytes × {} bits/byte = {} binary input bits to mixer", n_bits, input_bits);
        println!("  Mixer: binary weight POPCOUNT on {} bits", input_bits);
        println!("  → This is PURE BINARY end to end!");
        println!("  → No float, no int8, no multiply — only POPCOUNT + XOR\n");

        // Size comparison
        println!("━━━ Size comparison ━━━\n");
        println!("  Byte encoder:  {} neurons × 9 bits = {} bits", n_neurons, n_neurons * 9);
        println!("  Output/byte:   {} binary bits", n_bits);
        println!("  Chunk output:  {} bits = {} bytes", input_bits, input_bits / 8);
        println!("  Mixer neuron:  {} binary weights = {} bytes", input_bits, input_bits / 8);
    }

    println!("\n  Total time: {:.2}s", t0.elapsed().as_secs_f64());
}
