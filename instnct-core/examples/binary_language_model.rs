//! Binary Language Model — 128-char context → predict next char
//!
//! Architecture:
//!   128 previous chars (each 5-bit encoded) = 640 input bits
//!   → Hidden layers (C19, tied where possible)
//!   → 27 output (next char logits)
//!
//! Training: backprop float32 → freeze to binary {-1,+1}
//! Corpus: Alice (27 symbols: a-z + space)
//! Metric: argmax next-char accuracy (comparable to INSTNCT 24.6% baseline)
//!
//! Run: cargo run --example binary_language_model --release

use std::time::Instant;

// ══════════════════════════════════════════════════════
// PRNG
// ══════════════════════════════════════════════════════
struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn normal(&mut self) -> f32 { let u1 = self.f32().max(1e-7); let u2 = self.f32(); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() }
    fn range(&mut self, lo: usize, hi: usize) -> usize { lo + (self.next() as usize % (hi - lo)) }
}

// ══════════════════════════════════════════════════════
// CORPUS — same as INSTNCT: a-z=0-25, space=26, discard rest
// ══════════════════════════════════════════════════════
fn load_corpus(path: &str) -> Vec<u8> {
    let raw = std::fs::read(path).expect("failed to read corpus");
    let mut corpus = Vec::with_capacity(raw.len());
    for &b in &raw {
        match b {
            b'a'..=b'z' => corpus.push(b - b'a'),
            b'A'..=b'Z' => corpus.push(b - b'A'),
            b' ' | b'\n' | b'\t' | b'\r' => corpus.push(26),
            _ => {} // discard
        }
    }
    corpus
}

fn char_to_bits(c: u8) -> [f32; 5] {
    // 0-26 fits in 5 bits
    let mut bits = [0.0f32; 5];
    for i in 0..5 { bits[i] = ((c >> i) & 1) as f32; }
    bits
}

// ══════════════════════════════════════════════════════
// MLP LANGUAGE MODEL
// ══════════════════════════════════════════════════════
// Input: 128 chars × 5 bits = 640 inputs
// Hidden: H neurons (ReLU or C19)
// Output: 27 classes (softmax / argmax)

const CTX: usize = 128;   // context window
const IN_BITS: usize = 5;  // bits per char
const IN_DIM: usize = CTX * IN_BITS; // 640
const N_CLASSES: usize = 27;

#[derive(Clone)]
struct LangModel {
    // Layer 1: IN_DIM → H
    w1: Vec<Vec<f32>>,  // H × IN_DIM
    b1: Vec<f32>,       // H
    // Layer 2: H → H2
    w2: Vec<Vec<f32>>,  // H2 × H
    b2: Vec<f32>,       // H2
    // Output: H2 → N_CLASSES
    wo: Vec<Vec<f32>>,  // N_CLASSES × H2
    bo: Vec<f32>,       // N_CLASSES
    h1: usize,
    h2: usize,
}

impl LangModel {
    fn new(h1: usize, h2: usize, rng: &mut Rng) -> Self {
        let s1 = (2.0 / IN_DIM as f32).sqrt();
        let s2 = (2.0 / h1 as f32).sqrt();
        let so = (2.0 / h2 as f32).sqrt();
        LangModel {
            w1: (0..h1).map(|_| (0..IN_DIM).map(|_| rng.normal() * s1).collect()).collect(),
            b1: vec![0.0; h1],
            w2: (0..h2).map(|_| (0..h1).map(|_| rng.normal() * s2).collect()).collect(),
            b2: vec![0.0; h2],
            wo: (0..N_CLASSES).map(|_| (0..h2).map(|_| rng.normal() * so).collect()).collect(),
            bo: vec![0.0; N_CLASSES],
            h1, h2,
        }
    }

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Layer 1: ReLU
        let mut a1 = vec![0.0f32; self.h1];
        for i in 0..self.h1 {
            let mut s = self.b1[i];
            for j in 0..IN_DIM { s += self.w1[i][j] * input[j]; }
            a1[i] = s.max(0.0);
        }
        // Layer 2: ReLU
        let mut a2 = vec![0.0f32; self.h2];
        for i in 0..self.h2 {
            let mut s = self.b2[i];
            for j in 0..self.h1 { s += self.w2[i][j] * a1[j]; }
            a2[i] = s.max(0.0);
        }
        // Output: raw logits (softmax in loss)
        let mut logits = vec![0.0f32; N_CLASSES];
        for i in 0..N_CLASSES {
            logits[i] = self.bo[i];
            for j in 0..self.h2 { logits[i] += self.wo[i][j] * a2[j]; }
        }
        (a1, a2, logits)
    }

    fn predict(&self, input: &[f32]) -> usize {
        let (_, _, logits) = self.forward(input);
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
    }

    fn train_step(&mut self, input: &[f32], target: usize, lr: f32) -> f32 {
        let (a1, a2, logits) = self.forward(input);

        // Softmax + cross-entropy gradient
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum_exp: f32 = exp.iter().sum();
        let softmax: Vec<f32> = exp.iter().map(|&e| e / sum_exp).collect();

        let loss = -(softmax[target].max(1e-7)).ln();

        // dL/dlogits = softmax - one_hot(target)
        let mut dl = softmax.clone();
        dl[target] -= 1.0;

        // Output layer gradients
        let mut da2 = vec![0.0f32; self.h2];
        for i in 0..N_CLASSES {
            for j in 0..self.h2 {
                da2[j] += dl[i] * self.wo[i][j];
                self.wo[i][j] -= lr * dl[i] * a2[j];
            }
            self.bo[i] -= lr * dl[i];
        }

        // Layer 2 ReLU gradient
        let mut d2 = vec![0.0f32; self.h2];
        let mut da1 = vec![0.0f32; self.h1];
        for i in 0..self.h2 {
            d2[i] = if a2[i] > 0.0 { da2[i] } else { 0.0 };
            for j in 0..self.h1 {
                da1[j] += d2[i] * self.w2[i][j];
                self.w2[i][j] -= lr * d2[i] * a1[j];
            }
            self.b2[i] -= lr * d2[i];
        }

        // Layer 1 ReLU gradient
        for i in 0..self.h1 {
            let d1 = if a1[i] > 0.0 { da1[i] } else { 0.0 };
            for j in 0..IN_DIM { self.w1[i][j] -= lr * d1 * input[j]; }
            self.b1[i] -= lr * d1;
        }

        loss
    }

    fn total_params(&self) -> usize {
        self.h1 * IN_DIM + self.h1 + self.h2 * self.h1 + self.h2 + N_CLASSES * self.h2 + N_CLASSES
    }
}

// ══════════════════════════════════════════════════════
// BINARY QUANTIZED EVAL
// ══════════════════════════════════════════════════════
fn eval_binary(model: &LangModel, corpus: &[u8], n_samples: usize, seed: u64) -> (f64, f64) {
    let mut rng = Rng::new(seed);

    // Quantize all weights to {-1, 0, +1} based on sign (zero stays zero)
    // Actually binary {-1, +1}: sign of each weight
    let sign = |v: f32| -> f32 { if v >= 0.0 { 1.0 } else { -1.0 } };

    let qw1: Vec<Vec<f32>> = model.w1.iter().map(|r| r.iter().map(|&w| sign(w)).collect()).collect();
    let qb1: Vec<f32> = model.b1.iter().map(|&b| sign(b)).collect();
    let qw2: Vec<Vec<f32>> = model.w2.iter().map(|r| r.iter().map(|&w| sign(w)).collect()).collect();
    let qb2: Vec<f32> = model.b2.iter().map(|&b| sign(b)).collect();
    let qwo: Vec<Vec<f32>> = model.wo.iter().map(|r| r.iter().map(|&w| sign(w)).collect()).collect();
    let qbo: Vec<f32> = model.bo.iter().map(|&b| sign(b)).collect();

    let forward_q = |input: &[f32]| -> usize {
        let mut a1 = vec![0.0f32; model.h1];
        for i in 0..model.h1 {
            let mut s = qb1[i];
            for j in 0..IN_DIM { s += qw1[i][j] * input[j]; }
            a1[i] = s.max(0.0);
        }
        let mut a2 = vec![0.0f32; model.h2];
        for i in 0..model.h2 {
            let mut s = qb2[i];
            for j in 0..model.h1 { s += qw2[i][j] * a1[j]; }
            a2[i] = s.max(0.0);
        }
        let mut logits = vec![0.0f32; N_CLASSES];
        for i in 0..N_CLASSES {
            logits[i] = qbo[i];
            for j in 0..model.h2 { logits[i] += qwo[i][j] * a2[j]; }
        }
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
    };

    let mut float_correct = 0usize;
    let mut binary_correct = 0usize;
    let mut total = 0usize;

    for _ in 0..n_samples {
        if corpus.len() < CTX + 1 { break; }
        let off = rng.range(0, corpus.len() - CTX - 1);
        let seg = &corpus[off..off + CTX + 1];

        // Build input: 128 chars × 5 bits
        let mut input = vec![0.0f32; IN_DIM];
        for i in 0..CTX {
            let bits = char_to_bits(seg[i]);
            for b in 0..IN_BITS { input[i * IN_BITS + b] = bits[b]; }
        }
        let target = seg[CTX] as usize;

        // Float prediction
        let float_pred = model.predict(&input);
        if float_pred == target { float_correct += 1; }

        // Binary prediction
        let bin_pred = forward_q(&input);
        if bin_pred == target { binary_correct += 1; }

        total += 1;
    }

    (float_correct as f64 / total as f64 * 100.0,
     binary_correct as f64 / total as f64 * 100.0)
}

// ══════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════
fn main() {
    let t0 = Instant::now();

    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    println!("=== BINARY LANGUAGE MODEL ===");
    println!("Corpus: {} chars (27 symbols: a-z + space)", corpus.len());
    println!("Context: {} chars, Input: {}×{} = {} bits", CTX, CTX, IN_BITS, IN_DIM);
    println!("Output: {} classes (next char argmax)", N_CLASSES);
    println!("Baseline: INSTNCT 24.6% (gradient-free evolution)\n");

    // Frequency baseline
    let mut freq = [0usize; N_CLASSES];
    for &c in &corpus { freq[c as usize] += 1; }
    let most_common = freq.iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap();
    let freq_baseline = freq[most_common] as f64 / corpus.len() as f64 * 100.0;
    let mc = if most_common == 26 { "space".to_string() } else { format!("'{}'", (most_common as u8 + b'a') as char) };
    println!("Frequency baseline: {:.1}% (always predict {})\n", freq_baseline, mc);

    // Sweep model sizes
    for &(h1, h2) in &[(64, 32), (128, 64), (256, 128), (512, 256)] {
        let mut rng = Rng::new(42);
        let mut model = LangModel::new(h1, h2, &mut rng);
        let params = model.total_params();
        let binary_bytes = (params + 7) / 8; // 1 bit per param

        println!("━━━ H1={}, H2={} ({} params, {} KB float, {} KB binary) ━━━",
            h1, h2, params, params * 4 / 1024, binary_bytes / 1024);

        // Training
        let epochs = 20;
        let samples_per_epoch = 5000;
        let lr_init = 0.01;

        for epoch in 0..epochs {
            let lr = lr_init * (1.0 - epoch as f32 / epochs as f32 * 0.7);
            let mut total_loss = 0.0f32;

            for _ in 0..samples_per_epoch {
                if corpus.len() < CTX + 1 { break; }
                let off = rng.range(0, corpus.len() - CTX - 1);
                let seg = &corpus[off..off + CTX + 1];

                let mut input = vec![0.0f32; IN_DIM];
                for i in 0..CTX {
                    let bits = char_to_bits(seg[i]);
                    for b in 0..IN_BITS { input[i * IN_BITS + b] = bits[b]; }
                }
                let target = seg[CTX] as usize;

                total_loss += model.train_step(&input, target, lr);
            }

            if (epoch + 1) % 5 == 0 || epoch == 0 {
                let (float_acc, binary_acc) = eval_binary(&model, &corpus, 2000, 999 + epoch as u64);
                println!("  epoch {:>2}: loss={:.3} float_acc={:.1}% binary_acc={:.1}% lr={:.4}",
                    epoch + 1, total_loss / samples_per_epoch as f32, float_acc, binary_acc, lr);
            }
        }

        // Final eval (more samples)
        let (float_acc, binary_acc) = eval_binary(&model, &corpus, 5000, 777);

        println!("\n  FINAL: float={:.1}% binary={:.1}%", float_acc, binary_acc);
        println!("  vs baseline: freq={:.1}%, INSTNCT=24.6%", freq_baseline);

        let verdict = if binary_acc > 24.6 { "★ BEATS INSTNCT" }
            else if float_acc > 24.6 { "float beats, binary doesn't (yet)" }
            else { "below INSTNCT" };
        println!("  Verdict: {}", verdict);

        println!("  Binary model size: {} KB ({} params × 1 bit)", binary_bytes / 1024, params);
        println!("  Time: {:.1}s\n", t0.elapsed().as_secs_f64());
    }

    println!("Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
