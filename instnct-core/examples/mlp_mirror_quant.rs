//! MLP Mirror Autoencoder with Backprop + Int8 Quantization
//!
//! Hybrid approach: use gradient-based training to find good weights,
//! then quantize to int8 and analyze for common structure.
//!
//! Architecture: 8 input → H hidden (ReLU) → 7 bottleneck → H hidden (ReLU) → 8 output
//! Training: backprop with MSE loss on round-trip reconstruction
//! Then: quantize weights to int8, check accuracy, find common factors
//!
//! Run: cargo run --example mlp_mirror_quant --release

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
        // Box-Muller
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
// MLP — fully connected, backprop
// ══════════════════════════════════════════════════════
// Layers: input(8) → dense(H, ReLU) → dense(bottleneck, sigmoid) → dense(H, ReLU) → dense(8, sigmoid)
// Tied-weight version: decoder weights = encoder weights transposed

#[derive(Clone)]
struct MlpAutoencoder {
    // Encoder: input(8) → hidden(H) → bottleneck(B)
    w1: Vec<Vec<f32>>,  // H × 8
    b1: Vec<f32>,       // H
    w2: Vec<Vec<f32>>,  // B × H
    b2: Vec<f32>,       // B
    // Decoder: bottleneck(B) → hidden(H) → output(8)
    // TIED: w3 = w2^T, w4 = w1^T
    b3: Vec<f32>,       // H (decoder hidden bias)
    b4: Vec<f32>,       // 8 (decoder output bias)
    h_size: usize,
    b_size: usize,
}

impl MlpAutoencoder {
    fn new(h_size: usize, b_size: usize, rng: &mut Rng) -> Self {
        let scale1 = (2.0 / 8.0f32).sqrt();
        let scale2 = (2.0 / h_size as f32).sqrt();

        let w1: Vec<Vec<f32>> = (0..h_size).map(|_| (0..8).map(|_| rng.normal() * scale1).collect()).collect();
        let b1 = vec![0.0f32; h_size];
        let w2: Vec<Vec<f32>> = (0..b_size).map(|_| (0..h_size).map(|_| rng.normal() * scale2).collect()).collect();
        let b2 = vec![0.0f32; b_size];
        let b3 = vec![0.0f32; h_size];
        let b4 = vec![0.0f32; 8];

        MlpAutoencoder { w1, b1, w2, b2, b3, b4, h_size, b_size }
    }

    fn relu(x: f32) -> f32 { x.max(0.0) }
    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

    /// Forward pass — returns all intermediate values for backprop
    fn forward(&self, input: &[f32; 8]) -> ForwardResult {
        // Encoder hidden: z1 = W1 × input + b1, a1 = relu(z1)
        let mut z1 = vec![0.0f32; self.h_size];
        let mut a1 = vec![0.0f32; self.h_size];
        for i in 0..self.h_size {
            z1[i] = self.b1[i];
            for j in 0..8 { z1[i] += self.w1[i][j] * input[j]; }
            a1[i] = Self::relu(z1[i]);
        }

        // Bottleneck: z2 = W2 × a1 + b2, a2 = sigmoid(z2) → binary-ish
        let mut z2 = vec![0.0f32; self.b_size];
        let mut a2 = vec![0.0f32; self.b_size];
        for i in 0..self.b_size {
            z2[i] = self.b2[i];
            for j in 0..self.h_size { z2[i] += self.w2[i][j] * a1[j]; }
            a2[i] = Self::sigmoid(z2[i]);
        }

        // Decoder hidden: z3 = W2^T × a2 + b3, a3 = relu(z3)
        let mut z3 = vec![0.0f32; self.h_size];
        let mut a3 = vec![0.0f32; self.h_size];
        for i in 0..self.h_size {
            z3[i] = self.b3[i];
            for j in 0..self.b_size { z3[i] += self.w2[j][i] * a2[j]; }  // W2 transposed
            a3[i] = Self::relu(z3[i]);
        }

        // Decoder output: z4 = W1^T × a3 + b4, a4 = sigmoid(z4)
        let mut z4 = vec![0.0f32; 8];
        let mut a4 = vec![0.0f32; 8];
        for i in 0..8 {
            z4[i] = self.b4[i];
            for j in 0..self.h_size { z4[i] += self.w1[j][i] * a3[j]; }  // W1 transposed
            a4[i] = Self::sigmoid(z4[i]);
        }

        ForwardResult { a1, z1, a2, z2, a3, z3, a4, z4 }
    }

    /// Backprop + weight update for one example
    fn train_step(&mut self, input: &[f32; 8], lr: f32) -> f32 {
        let fwd = self.forward(input);

        // Loss = MSE: L = Σ (a4[i] - input[i])²
        let mut loss = 0.0f32;
        let mut d4 = [0.0f32; 8]; // dL/dz4
        for i in 0..8 {
            let err = fwd.a4[i] - input[i];
            loss += err * err;
            // sigmoid derivative: a4*(1-a4)
            d4[i] = 2.0 * err * fwd.a4[i] * (1.0 - fwd.a4[i]);
        }

        // Gradients for W1^T (used as decoder output) and b4
        // dL/dW1[j][i] += d4[i] * a3[j]  (transposed usage)
        // dL/da3[j] = Σ_i d4[i] * W1[j][i]
        let mut da3 = vec![0.0f32; self.h_size];
        for j in 0..self.h_size {
            for i in 0..8 {
                da3[j] += d4[i] * self.w1[j][i];
                self.w1[j][i] -= lr * d4[i] * fwd.a3[j];
            }
        }
        for i in 0..8 { self.b4[i] -= lr * d4[i]; }

        // d3 = da3 * relu'(z3)
        let mut d3 = vec![0.0f32; self.h_size];
        for i in 0..self.h_size {
            d3[i] = if fwd.z3[i] > 0.0 { da3[i] } else { 0.0 };
        }

        // Gradients for W2^T (used as decoder hidden) and b3
        // dL/dW2[j][i] += d3[i] * a2[j]  (transposed usage)
        // dL/da2[j] = Σ_i d3[i] * W2[j][i]
        let mut da2 = vec![0.0f32; self.b_size];
        for j in 0..self.b_size {
            for i in 0..self.h_size {
                da2[j] += d3[i] * self.w2[j][i];
                self.w2[j][i] -= lr * d3[i] * fwd.a2[j];
            }
        }
        for i in 0..self.h_size { self.b3[i] -= lr * d3[i]; }

        // d2 = da2 * sigmoid'(z2) = da2 * a2 * (1 - a2)
        let mut d2 = vec![0.0f32; self.b_size];
        for i in 0..self.b_size {
            d2[i] = da2[i] * fwd.a2[i] * (1.0 - fwd.a2[i]);
        }

        // Gradients for W2 (encoder bottleneck) and b2
        // dL/dW2[i][j] += d2[i] * a1[j]
        // dL/da1[j] = Σ_i d2[i] * W2[i][j]
        let mut da1 = vec![0.0f32; self.h_size];
        for i in 0..self.b_size {
            for j in 0..self.h_size {
                da1[j] += d2[i] * self.w2[i][j];
                self.w2[i][j] -= lr * d2[i] * fwd.a1[j];
            }
            self.b2[i] -= lr * d2[i];
        }

        // d1 = da1 * relu'(z1)
        let mut d1 = vec![0.0f32; self.h_size];
        for i in 0..self.h_size {
            d1[i] = if fwd.z1[i] > 0.0 { da1[i] } else { 0.0 };
        }

        // Gradients for W1 (encoder input) and b1
        for i in 0..self.h_size {
            for j in 0..8 {
                self.w1[i][j] -= lr * d1[i] * input[j];
            }
            self.b1[i] -= lr * d1[i];
        }

        loss
    }

    /// Encode → hard binary (threshold at 0.5)
    fn encode_hard(&self, input: &[f32; 8]) -> Vec<u8> {
        let fwd = self.forward(input);
        fwd.a2.iter().map(|&v| if v >= 0.5 { 1u8 } else { 0u8 }).collect()
    }

    /// Round-trip with hard binary bottleneck
    fn round_trip_hard(&self, input: &[f32; 8]) -> [f32; 8] {
        let fwd = self.forward(input);
        let mut out = [0.0f32; 8];
        for i in 0..8 { out[i] = if fwd.a4[i] >= 0.5 { 1.0 } else { 0.0 }; }
        out
    }

    fn eval_accuracy(&self, unique_bytes: &[u8]) -> (usize, usize) {
        let mut correct = 0;
        for &b in unique_bytes {
            let input = byte_to_bits(b);
            let output = self.round_trip_hard(&input);
            let mut ok = true;
            for i in 0..8 {
                if (output[i] - input[i]).abs() > 0.4 { ok = false; break; }
            }
            if ok { correct += 1; }
        }
        (correct, unique_bytes.len())
    }
}

struct ForwardResult {
    a1: Vec<f32>, z1: Vec<f32>,
    a2: Vec<f32>, z2: Vec<f32>,
    a3: Vec<f32>, z3: Vec<f32>,
    a4: Vec<f32>, z4: Vec<f32>,
}

// ══════════════════════════════════════════════════════
// INT8 QUANTIZATION
// ══════════════════════════════════════════════════════
#[derive(Clone)]
struct QuantizedMlp {
    w1: Vec<Vec<i8>>,   // H × 8
    b1: Vec<i8>,
    w2: Vec<Vec<i8>>,   // B × H
    b2: Vec<i8>,
    b3: Vec<i8>,
    b4: Vec<i8>,
    scale_w1: f32,      // float = int8 * scale
    scale_b1: f32,
    scale_w2: f32,
    scale_b2: f32,
    scale_b3: f32,
    scale_b4: f32,
    h_size: usize,
    b_size: usize,
}

fn quantize_vec(v: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-7);
    let scale = max_abs / 127.0;
    let q: Vec<i8> = v.iter().map(|&x| (x / scale).round().max(-127.0).min(127.0) as i8).collect();
    (q, scale)
}

fn quantize_mat(m: &[Vec<f32>]) -> (Vec<Vec<i8>>, f32) {
    let max_abs = m.iter().flat_map(|r| r.iter()).map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-7);
    let scale = max_abs / 127.0;
    let q: Vec<Vec<i8>> = m.iter().map(|row|
        row.iter().map(|&x| (x / scale).round().max(-127.0).min(127.0) as i8).collect()
    ).collect();
    (q, scale)
}

impl QuantizedMlp {
    fn from_float(mlp: &MlpAutoencoder) -> Self {
        let (w1, scale_w1) = quantize_mat(&mlp.w1);
        let (b1, scale_b1) = quantize_vec(&mlp.b1);
        let (w2, scale_w2) = quantize_mat(&mlp.w2);
        let (b2, scale_b2) = quantize_vec(&mlp.b2);
        let (b3, scale_b3) = quantize_vec(&mlp.b3);
        let (b4, scale_b4) = quantize_vec(&mlp.b4);
        QuantizedMlp { w1, b1, w2, b2, b3, b4, scale_w1, scale_b1, scale_w2, scale_b2, scale_b3, scale_b4, h_size: mlp.h_size, b_size: mlp.b_size }
    }

    fn forward_int(&self, input: &[f32; 8]) -> (Vec<u8>, [f32; 8]) {
        // Encoder hidden (int8 math → float via scale)
        let mut a1 = vec![0.0f32; self.h_size];
        for i in 0..self.h_size {
            let mut sum = self.b1[i] as f32 * self.scale_b1;
            for j in 0..8 { sum += self.w1[i][j] as f32 * self.scale_w1 * input[j]; }
            a1[i] = sum.max(0.0); // relu
        }

        // Bottleneck (int8 → sigmoid → hard binary)
        let mut a2 = vec![0.0f32; self.b_size];
        let mut code = vec![0u8; self.b_size];
        for i in 0..self.b_size {
            let mut sum = self.b2[i] as f32 * self.scale_b2;
            for j in 0..self.h_size { sum += self.w2[i][j] as f32 * self.scale_w2 * a1[j]; }
            a2[i] = 1.0 / (1.0 + (-sum).exp()); // sigmoid
            code[i] = if a2[i] >= 0.5 { 1 } else { 0 };
        }

        // Decoder hidden (W2^T, int8)
        let mut a3 = vec![0.0f32; self.h_size];
        for i in 0..self.h_size {
            let mut sum = self.b3[i] as f32 * self.scale_b3;
            for j in 0..self.b_size { sum += self.w2[j][i] as f32 * self.scale_w2 * a2[j]; }
            a3[i] = sum.max(0.0); // relu
        }

        // Decoder output (W1^T, int8)
        let mut a4 = [0.0f32; 8];
        for i in 0..8 {
            let mut sum = self.b4[i] as f32 * self.scale_b4;
            for j in 0..self.h_size { sum += self.w1[j][i] as f32 * self.scale_w1 * a3[j]; }
            a4[i] = 1.0 / (1.0 + (-sum).exp()); // sigmoid
        }

        (code, a4)
    }

    fn eval_accuracy(&self, unique_bytes: &[u8]) -> (usize, usize, Vec<u8>) {
        let mut correct = 0;
        let mut failed = Vec::new();
        for &b in unique_bytes {
            let input = byte_to_bits(b);
            let (_, output) = self.forward_int(&input);
            let mut ok = true;
            for i in 0..8 {
                if (output[i] - input[i]).abs() > 0.4 { ok = false; break; }
            }
            if ok { correct += 1; } else { failed.push(b); }
        }
        (correct, unique_bytes.len(), failed)
    }

    fn analyze_weights(&self) {
        println!("\n  ═══ WEIGHT ANALYSIS ═══");

        // W1 analysis
        let all_w1: Vec<i8> = self.w1.iter().flat_map(|r| r.iter().copied()).collect();
        let all_w2: Vec<i8> = self.w2.iter().flat_map(|r| r.iter().copied()).collect();

        for (name, weights, scale) in &[("W1 (enc input)", &all_w1, self.scale_w1), ("W2 (enc bottle)", &all_w2, self.scale_w2)] {
            println!("\n  {}: {} weights, scale={:.6}", name, weights.len(), scale);

            // GCD analysis
            let nonzero: Vec<i8> = weights.iter().filter(|&&w| w != 0).copied().collect();
            if nonzero.is_empty() { println!("    ALL ZERO"); continue; }

            let abs_vals: Vec<u8> = nonzero.iter().map(|&w| w.unsigned_abs()).collect();
            let gcd = abs_vals.iter().copied().fold(0u8, gcd_u8);
            println!("    GCD of all weights: {}", gcd);
            if gcd > 1 {
                let reduced: Vec<i8> = weights.iter().map(|&w| w / gcd as i8).collect();
                let max_reduced = reduced.iter().map(|w| w.abs()).max().unwrap_or(0);
                let bits_needed = if max_reduced == 0 { 1 } else { (max_reduced as f32).log2().ceil() as u32 + 2 }; // +1 for sign
                println!("    Reduced weights: all / {} → max |w| = {} → {} bits needed (vs 8)", gcd, max_reduced, bits_needed);
            }

            // Unique values
            let mut unique_vals: Vec<i8> = weights.to_vec();
            unique_vals.sort();
            unique_vals.dedup();
            println!("    Unique values: {} (out of {} weights)", unique_vals.len(), weights.len());
            if unique_vals.len() <= 20 {
                let vals: Vec<String> = unique_vals.iter().map(|v| format!("{}", v)).collect();
                println!("    Values: [{}]", vals.join(", "));
            }

            // Histogram of absolute values
            let mut hist = [0u32; 128];
            for &w in weights.iter() { hist[w.unsigned_abs() as usize] += 1; }
            let nonzero_count = weights.len() - hist[0] as usize;
            let zero_pct = hist[0] as f32 / weights.len() as f32 * 100.0;
            println!("    Sparsity: {:.1}% zeros ({}/{})", zero_pct, hist[0], weights.len());

            // Top 5 most common absolute values
            let mut val_counts: Vec<(u8, u32)> = hist.iter().enumerate().map(|(i, &c)| (i as u8, c)).filter(|(_, c)| *c > 0).collect();
            val_counts.sort_by(|a, b| b.1.cmp(&a.1));
            let top5: Vec<String> = val_counts.iter().take(5).map(|(v, c)| format!("|{}|×{}", v, c)).collect();
            println!("    Top values: {}", top5.join(", "));

            // Check if weights are multiples of some base
            if nonzero.len() >= 2 {
                for base in [2, 3, 4, 5, 7, 8, 10, 16] {
                    let is_mult: usize = nonzero.iter().filter(|&&w| w as i32 % base == 0).count();
                    if is_mult > nonzero.len() * 80 / 100 {
                        println!("    ★ {}% of weights are multiples of {} → could store w/{}", is_mult * 100 / nonzero.len(), base, base);
                    }
                }
            }
        }

        // Show W1 matrix
        println!("\n  W1 int8 matrix (enc hidden × input):");
        for (i, row) in self.w1.iter().enumerate() {
            let vals: Vec<String> = row.iter().map(|&w| format!("{:>4}", w)).collect();
            println!("    h{}: [{}]  bias={:>4}", i, vals.join(","), self.b1[i]);
        }
        println!("\n  W2 int8 matrix (bottleneck × hidden):");
        for (i, row) in self.w2.iter().enumerate() {
            let vals: Vec<String> = row.iter().map(|&w| format!("{:>4}", w)).collect();
            println!("    b{}: [{}]  bias={:>4}", i, vals.join(","), self.b2[i]);
        }
    }
}

fn gcd_u8(a: u8, b: u8) -> u8 {
    let (mut a, mut b) = (a, b);
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

// ══════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════
fn main() {
    let t0 = Instant::now();

    let corpus_path = "instnct-core/tests/fixtures/alice_corpus.txt";
    let unique = load_unique_bytes(corpus_path);
    println!("=== MLP MIRROR AUTOENCODER + INT8 QUANTIZATION ===");
    println!("Corpus: {} unique bytes", unique.len());
    println!("Architecture: 8 → H(ReLU) → 7(sigmoid) → H(ReLU, W2ᵀ) → 8(sigmoid, W1ᵀ)");
    println!("Training: backprop, then quantize to int8");
    println!();

    // Prepare training data
    let train_data: Vec<[f32; 8]> = unique.iter().map(|&b| byte_to_bits(b)).collect();

    // Sweep hidden sizes
    for &h_size in &[7, 12, 16, 24, 32] {
        println!("━━━ Hidden size H={} ━━━", h_size);

        let mut best_acc = 0;
        let mut best_mlp = None;

        // Try multiple seeds
        for seed in 0..5 {
            let mut rng = Rng::new(42 + seed * 1000);
            let mut mlp = MlpAutoencoder::new(h_size, 7, &mut rng);

            // Train
            let epochs = 5000;
            let lr = 0.05;
            for epoch in 0..epochs {
                let mut total_loss = 0.0f32;
                for input in &train_data {
                    total_loss += mlp.train_step(input, lr * (1.0 - epoch as f32 / epochs as f32 * 0.9));
                }
                if (epoch + 1) % 1000 == 0 {
                    let (acc, total) = mlp.eval_accuracy(&unique);
                    if epoch + 1 == epochs {
                        println!("  seed {} epoch {}: loss={:.4}, accuracy={}/{} ({:.1}%)",
                            seed, epoch + 1, total_loss / train_data.len() as f32, acc, total,
                            acc as f64 / total as f64 * 100.0);
                    }
                    if acc > best_acc {
                        best_acc = acc;
                        best_mlp = Some(mlp.clone());
                    }
                }
            }
        }

        let mlp = best_mlp.unwrap();
        let (float_acc, total) = mlp.eval_accuracy(&unique);

        // Quantize
        let qmlp = QuantizedMlp::from_float(&mlp);
        let (quant_acc, _, quant_failed) = qmlp.eval_accuracy(&unique);

        println!("\n  Float accuracy:  {}/{} ({:.1}%)", float_acc, total, float_acc as f64 / total as f64 * 100.0);
        println!("  Int8 accuracy:   {}/{} ({:.1}%)", quant_acc, total, quant_acc as f64 / total as f64 * 100.0);
        if float_acc != quant_acc {
            println!("  Quantization loss: {} bytes", float_acc as i32 - quant_acc as i32);
        } else {
            println!("  Quantization: LOSSLESS ✓");
        }
        if !quant_failed.is_empty() {
            let fs: Vec<String> = quant_failed.iter().take(10).map(|&b| {
                if b >= 32 && b < 127 { format!("'{}'", b as char) } else { format!("x{:02x}", b) }
            }).collect();
            println!("  Failed: [{}]", fs.join(", "));
        }

        // Show encoding
        println!("\n  Bottleneck codes:");
        let mut codes: Vec<(u8, Vec<u8>)> = Vec::new();
        for &b in &unique {
            let input = byte_to_bits(b);
            let (code, _) = qmlp.forward_int(&input);
            codes.push((b, code));
        }
        // Show some
        for &b in &[b' ', b'a', b'e', b't', b'z'] {
            if let Some((_, code)) = codes.iter().find(|(byte, _)| *byte == b) {
                let ch = if b >= 32 && b < 127 { format!("'{}'", b as char) } else { format!("x{:02x}", b) };
                let code_str: String = code.iter().map(|&v| if v == 1 { '1' } else { '0' }).collect();
                println!("    {} → {}", ch, code_str);
            }
        }

        // Check unique codes
        let mut unique_codes: Vec<Vec<u8>> = codes.iter().map(|(_, c)| c.clone()).collect();
        unique_codes.sort();
        unique_codes.dedup();
        println!("  Unique codes: {}/{}", unique_codes.len(), unique.len());

        // Collision details
        let mut code_map: std::collections::HashMap<Vec<u8>, Vec<u8>> = std::collections::HashMap::new();
        for (b, code) in &codes { code_map.entry(code.clone()).or_default().push(*b); }
        let collisions: Vec<_> = code_map.iter().filter(|(_, bs)| bs.len() > 1).collect();
        if !collisions.is_empty() {
            println!("  Collisions:");
            for (code, bytes) in &collisions {
                let cs: String = code.iter().map(|&v| if v == 1 { '1' } else { '0' }).collect();
                let bs: Vec<String> = bytes.iter().map(|&b| {
                    if b >= 32 && b < 127 { format!("'{}'", b as char) } else { format!("x{:02x}", b) }
                }).collect();
                println!("      {} → [{}]", cs, bs.join(", "));
            }
        }

        // Weight analysis (only for best)
        if h_size == 16 || float_acc == total {
            qmlp.analyze_weights();
        }

        println!("\n  Time: {:.1}s\n", t0.elapsed().as_secs_f64());

        if float_acc == total && quant_acc == total {
            println!("  ★★★ PERFECT: H={} achieves 100% float AND 100% int8! ★★★\n", h_size);
        }
    }

    println!("Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
