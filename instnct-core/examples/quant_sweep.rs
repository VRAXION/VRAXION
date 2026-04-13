//! Quantization sweep: train best MLP autoencoder (H=12, tied weights),
//! then test round-trip accuracy at int8, int6, int5, int4, int3, int2, ternary, binary.
//!
//! Shows: what's the minimum bit-width that preserves perfect encoding?
//!
//! Run: cargo run --example quant_sweep --release

use std::time::Instant;

// ══════════════════════════════════════════════════════
// PRNG
// ══════════════════════════════════════════════════════
struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 {
        self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.s
    }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn normal(&mut self) -> f32 {
        let u1 = self.f32().max(1e-7);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

// ══════════════════════════════════════════════════════
// DATA
// ══════════════════════════════════════════════════════
fn load_unique_bytes(path: &str) -> Vec<u8> {
    let text = std::fs::read(path).expect("failed to read corpus");
    let mut seen = [false; 256];
    for &b in &text { seen[b as usize] = true; }
    (0..=255u8).filter(|&b| seen[b as usize]).collect()
}

fn byte_to_bits(b: u8) -> [f32; 8] {
    let mut bits = [0.0f32; 8];
    for i in 0..8 { bits[i] = ((b >> i) & 1) as f32; }
    bits
}

// ══════════════════════════════════════════════════════
// MLP AUTOENCODER (tied weights)
// ══════════════════════════════════════════════════════
#[derive(Clone)]
struct Mlp {
    w1: Vec<Vec<f32>>,  // H × 8
    b1: Vec<f32>,       // H
    w2: Vec<Vec<f32>>,  // B × H
    b2: Vec<f32>,       // B
    b3: Vec<f32>,       // H
    b4: Vec<f32>,       // 8
    h: usize,
    b: usize,
}

impl Mlp {
    fn new(h: usize, b: usize, rng: &mut Rng) -> Self {
        let s1 = (2.0 / 8.0f32).sqrt();
        let s2 = (2.0 / h as f32).sqrt();
        Mlp {
            w1: (0..h).map(|_| (0..8).map(|_| rng.normal() * s1).collect()).collect(),
            b1: vec![0.0; h],
            w2: (0..b).map(|_| (0..h).map(|_| rng.normal() * s2).collect()).collect(),
            b2: vec![0.0; b],
            b3: vec![0.0; h],
            b4: vec![0.0; 8],
            h, b,
        }
    }

    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

    fn forward(&self, inp: &[f32; 8]) -> Fwd {
        let mut z1 = vec![0.0f32; self.h];
        let mut a1 = vec![0.0f32; self.h];
        for i in 0..self.h {
            z1[i] = self.b1[i];
            for j in 0..8 { z1[i] += self.w1[i][j] * inp[j]; }
            a1[i] = z1[i].max(0.0);
        }
        let mut z2 = vec![0.0f32; self.b];
        let mut a2 = vec![0.0f32; self.b];
        for i in 0..self.b {
            z2[i] = self.b2[i];
            for j in 0..self.h { z2[i] += self.w2[i][j] * a1[j]; }
            a2[i] = Self::sigmoid(z2[i]);
        }
        let mut z3 = vec![0.0f32; self.h];
        let mut a3 = vec![0.0f32; self.h];
        for i in 0..self.h {
            z3[i] = self.b3[i];
            for j in 0..self.b { z3[i] += self.w2[j][i] * a2[j]; }
            a3[i] = z3[i].max(0.0);
        }
        let mut z4 = [0.0f32; 8];
        let mut a4 = [0.0f32; 8];
        for i in 0..8 {
            z4[i] = self.b4[i];
            for j in 0..self.h { z4[i] += self.w1[j][i] * a3[j]; }
            a4[i] = Self::sigmoid(z4[i]);
        }
        Fwd { a1, z1, a2, z2, a3, z3, a4, z4 }
    }

    fn train_step(&mut self, inp: &[f32; 8], lr: f32) -> f32 {
        let f = self.forward(inp);
        let mut loss = 0.0f32;
        let mut d4 = [0.0f32; 8];
        for i in 0..8 {
            let e = f.a4[i] - inp[i];
            loss += e * e;
            d4[i] = 2.0 * e * f.a4[i] * (1.0 - f.a4[i]);
        }
        let mut da3 = vec![0.0f32; self.h];
        for j in 0..self.h {
            for i in 0..8 {
                da3[j] += d4[i] * self.w1[j][i];
                self.w1[j][i] -= lr * d4[i] * f.a3[j];
            }
        }
        for i in 0..8 { self.b4[i] -= lr * d4[i]; }
        let mut d3 = vec![0.0f32; self.h];
        for i in 0..self.h { d3[i] = if f.z3[i] > 0.0 { da3[i] } else { 0.0 }; }
        let mut da2 = vec![0.0f32; self.b];
        for j in 0..self.b {
            for i in 0..self.h {
                da2[j] += d3[i] * self.w2[j][i];
                self.w2[j][i] -= lr * d3[i] * f.a2[j];
            }
        }
        for i in 0..self.h { self.b3[i] -= lr * d3[i]; }
        let mut d2 = vec![0.0f32; self.b];
        for i in 0..self.b { d2[i] = da2[i] * f.a2[i] * (1.0 - f.a2[i]); }
        let mut da1 = vec![0.0f32; self.h];
        for i in 0..self.b {
            for j in 0..self.h {
                da1[j] += d2[i] * self.w2[i][j];
                self.w2[i][j] -= lr * d2[i] * f.a1[j];
            }
            self.b2[i] -= lr * d2[i];
        }
        let mut d1 = vec![0.0f32; self.h];
        for i in 0..self.h { d1[i] = if f.z1[i] > 0.0 { da1[i] } else { 0.0 }; }
        for i in 0..self.h {
            for j in 0..8 { self.w1[i][j] -= lr * d1[i] * inp[j]; }
            self.b1[i] -= lr * d1[i];
        }
        loss
    }
}

struct Fwd {
    a1: Vec<f32>, z1: Vec<f32>,
    a2: Vec<f32>, z2: Vec<f32>,
    a3: Vec<f32>, z3: Vec<f32>,
    a4: [f32; 8], z4: [f32; 8],
}

// ══════════════════════════════════════════════════════
// QUANTIZED EVAL — parameterized bit-width
// ══════════════════════════════════════════════════════
fn quantize_to_bits(val: f32, scale: f32, max_int: i32) -> i32 {
    (val / scale).round().max(-max_int as f32).min(max_int as f32) as i32
}

fn compute_scale(vals: &[f32], max_int: i32) -> f32 {
    let max_abs = vals.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-7);
    max_abs / max_int as f32
}

fn compute_scale_mat(m: &[Vec<f32>], max_int: i32) -> f32 {
    let max_abs = m.iter().flat_map(|r| r.iter()).map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-7);
    max_abs / max_int as f32
}

/// Evaluate round-trip accuracy with weights quantized to `bits` bit-width
/// bits=8 → max_int=127, bits=4 → max_int=7, bits=2 → max_int=1 (ternary), bits=1 → max_int=0..1 (binary)
fn eval_quantized(mlp: &Mlp, unique_bytes: &[u8], bits: u32) -> (usize, usize, Vec<u8>, Vec<u8>) {
    let max_int: i32 = if bits == 1 { 1 } else { (1i32 << (bits - 1)) - 1 };

    // Quantize all weight matrices and biases
    let sw1 = compute_scale_mat(&mlp.w1, max_int);
    let sw2 = compute_scale_mat(&mlp.w2, max_int);
    let sb1 = compute_scale(&mlp.b1, max_int);
    let sb2 = compute_scale(&mlp.b2, max_int);
    let sb3 = compute_scale(&mlp.b3, max_int);
    let sb4 = compute_scale(&mlp.b4, max_int);

    let qw1: Vec<Vec<i32>> = mlp.w1.iter().map(|row|
        row.iter().map(|&v| quantize_to_bits(v, sw1, max_int)).collect()
    ).collect();
    let qw2: Vec<Vec<i32>> = mlp.w2.iter().map(|row|
        row.iter().map(|&v| quantize_to_bits(v, sw2, max_int)).collect()
    ).collect();
    let qb1: Vec<i32> = mlp.b1.iter().map(|&v| quantize_to_bits(v, sb1, max_int)).collect();
    let qb2: Vec<i32> = mlp.b2.iter().map(|&v| quantize_to_bits(v, sb2, max_int)).collect();
    let qb3: Vec<i32> = mlp.b3.iter().map(|&v| quantize_to_bits(v, sb3, max_int)).collect();
    let qb4: Vec<i32> = mlp.b4.iter().map(|&v| quantize_to_bits(v, sb4, max_int)).collect();

    let sigmoid = |x: f32| -> f32 { 1.0 / (1.0 + (-x).exp()) };

    let mut correct = 0usize;
    let mut failed_bytes = Vec::new();
    let mut all_codes = Vec::new();

    for &b in unique_bytes {
        let inp = byte_to_bits(b);

        // Encoder hidden
        let mut a1 = vec![0.0f32; mlp.h];
        for i in 0..mlp.h {
            let mut sum = qb1[i] as f32 * sb1;
            for j in 0..8 { sum += qw1[i][j] as f32 * sw1 * inp[j]; }
            a1[i] = sum.max(0.0);
        }

        // Bottleneck
        let mut a2 = vec![0.0f32; mlp.b];
        let mut code = vec![0u8; mlp.b];
        for i in 0..mlp.b {
            let mut sum = qb2[i] as f32 * sb2;
            for j in 0..mlp.h { sum += qw2[i][j] as f32 * sw2 * a1[j]; }
            a2[i] = sigmoid(sum);
            code[i] = if a2[i] >= 0.5 { 1 } else { 0 };
        }

        // Decoder hidden (W2^T)
        let mut a3 = vec![0.0f32; mlp.h];
        for i in 0..mlp.h {
            let mut sum = qb3[i] as f32 * sb3;
            for j in 0..mlp.b { sum += qw2[j][i] as f32 * sw2 * a2[j]; }
            a3[i] = sum.max(0.0);
        }

        // Decoder output (W1^T)
        let mut a4 = [0.0f32; 8];
        for i in 0..8 {
            let mut sum = qb4[i] as f32 * sb4;
            for j in 0..mlp.h { sum += qw1[j][i] as f32 * sw1 * a3[j]; }
            a4[i] = sigmoid(sum);
        }

        // Check round-trip
        let mut ok = true;
        for i in 0..8 {
            if (a4[i] - inp[i]).abs() > 0.4 { ok = false; break; }
        }
        if ok { correct += 1; } else { failed_bytes.push(b); }
        all_codes.push(code);
    }

    // Count unique codes
    let mut uniq = all_codes.clone();
    uniq.sort(); uniq.dedup();

    (correct, unique_bytes.len(), failed_bytes, unique_bytes.to_vec())
}

/// Count unique values in quantized weight matrix
fn count_unique_qvals(m: &[Vec<f32>], scale: f32, max_int: i32) -> usize {
    let mut vals: Vec<i32> = m.iter().flat_map(|r| r.iter())
        .map(|&v| quantize_to_bits(v, scale, max_int))
        .collect();
    vals.sort(); vals.dedup();
    vals.len()
}

// ══════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════
fn main() {
    let t0 = Instant::now();

    let corpus_path = "instnct-core/tests/fixtures/alice_corpus.txt";
    let unique = load_unique_bytes(corpus_path);
    let train_data: Vec<[f32; 8]> = unique.iter().map(|&b| byte_to_bits(b)).collect();

    println!("=== QUANTIZATION SWEEP ===");
    println!("Corpus: {} unique bytes", unique.len());
    println!("Architecture: 8 → H(ReLU) → 7(sigmoid) → H(ReLU, W2ᵀ) → 8(sigmoid, W1ᵀ)");
    println!();

    // Train best models at different H sizes
    for &h_size in &[12, 16, 24] {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Training H={}, B=7...", h_size);

        // Find best seed
        let mut best_mlp = None;
        let mut best_loss = f32::MAX;
        for seed in 0..10 {
            let mut rng = Rng::new(42 + seed * 1000);
            let mut mlp = Mlp::new(h_size, 7, &mut rng);
            let epochs = 8000;
            for epoch in 0..epochs {
                let lr = 0.05 * (1.0 - epoch as f32 / epochs as f32 * 0.9);
                let mut total_loss = 0.0f32;
                for inp in &train_data { total_loss += mlp.train_step(inp, lr); }
                let avg = total_loss / train_data.len() as f32;
                if epoch == epochs - 1 && avg < best_loss {
                    best_loss = avg;
                    best_mlp = Some(mlp.clone());
                }
            }
        }
        let mlp = best_mlp.unwrap();

        // Float baseline
        let fwd_test = |mlp: &Mlp| -> usize {
            unique.iter().filter(|&&b| {
                let inp = byte_to_bits(b);
                let f = mlp.forward(&inp);
                (0..8).all(|i| (f.a4[i] - inp[i]).abs() < 0.4)
            }).count()
        };
        let float_acc = fwd_test(&mlp);
        println!("  Float32: {}/{} ({:.1}%), loss={:.6}",
            float_acc, unique.len(), float_acc as f64 / unique.len() as f64 * 100.0, best_loss);

        // Quantization sweep
        println!("\n  {:>6} {:>8} {:>10} {:>10} {:>15} {:>15} {:>10}",
            "bits", "max_int", "accuracy", "pct", "unique_W1_vals", "unique_W2_vals", "failed");
        println!("  {}", "─".repeat(85));

        for &bits in &[8, 7, 6, 5, 4, 3, 2, 1] {
            let max_int: i32 = if bits == 1 { 1 } else { (1i32 << (bits - 1)) - 1 };

            let (acc, total, failed, _) = eval_quantized(&mlp, &unique, bits);

            let sw1 = compute_scale_mat(&mlp.w1, max_int);
            let sw2 = compute_scale_mat(&mlp.w2, max_int);
            let uv1 = count_unique_qvals(&mlp.w1, sw1, max_int);
            let uv2 = count_unique_qvals(&mlp.w2, sw2, max_int);

            let failed_str = if failed.is_empty() {
                "—".to_string()
            } else {
                let fs: Vec<String> = failed.iter().take(5).map(|&b| {
                    if b >= 32 && b < 127 { format!("'{}'", b as char) } else { format!("x{:02x}", b) }
                }).collect();
                if failed.len() > 5 { format!("{}+{}", fs.join(","), failed.len()-5) }
                else { fs.join(",") }
            };

            let pct = acc as f64 / total as f64 * 100.0;
            let marker = if acc == total { " ★" } else { "" };

            println!("  {:>4}b  {:>7}  {:>7}/{:<2}  {:>8.1}%  {:>13}  {:>13}  {}{}",
                bits, max_int, acc, total, pct, uv1, uv2, failed_str, marker);
        }

        // Detailed analysis at the boundary
        println!("\n  ─── Detailed boundary analysis ───");
        for &bits in &[8, 6, 5, 4, 3, 2] {
            let max_int: i32 = if bits == 1 { 1 } else { (1i32 << (bits - 1)) - 1 };
            let (acc, _, _, _) = eval_quantized(&mlp, &unique, bits);
            if acc == unique.len() {
                // Show the quantized weight values
                let sw1 = compute_scale_mat(&mlp.w1, max_int);
                let sw2 = compute_scale_mat(&mlp.w2, max_int);

                println!("\n  {}b (max_int={}) — PERFECT:", bits, max_int);

                // W1 matrix
                println!("    W1 quantized ({} × 8):", mlp.h);
                for (i, row) in mlp.w1.iter().enumerate() {
                    let qr: Vec<String> = row.iter()
                        .map(|&v| format!("{:>4}", quantize_to_bits(v, sw1, max_int)))
                        .collect();
                    let qb = quantize_to_bits(mlp.b1[i], compute_scale(&mlp.b1, max_int), max_int);
                    println!("      h{:>2}: [{}]  b={:>4}", i, qr.join(","), qb);
                }

                // W2 matrix
                println!("    W2 quantized (7 × {}):", mlp.h);
                for (i, row) in mlp.w2.iter().enumerate() {
                    let qr: Vec<String> = row.iter()
                        .map(|&v| format!("{:>4}", quantize_to_bits(v, sw2, max_int)))
                        .collect();
                    let qb = quantize_to_bits(mlp.b2[i], compute_scale(&mlp.b2, max_int), max_int);
                    println!("      b{}: [{}]  b={:>4}", i, qr.join(","), qb);
                }

                // Total storage
                let total_params = mlp.h * 8 + mlp.h + 7 * mlp.h + 7 + mlp.h + 8;
                let total_bits = total_params * bits as usize;
                let total_bytes = (total_bits + 7) / 8;
                println!("    Total: {} params × {}b = {} bits = {} bytes",
                    total_params, bits, total_bits, total_bytes);

                break; // Only show the smallest perfect quantization
            }
        }

        println!("\n  Time: {:.1}s\n", t0.elapsed().as_secs_f64());
    }

    println!("Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
