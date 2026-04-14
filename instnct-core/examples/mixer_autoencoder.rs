//! Mixer Autoencoder — Level 2 abstraction
//!
//! Level 0 (done): byte → 7 signals (frozen, 77 bits)
//! Level 1 (this): N×7 signals → bottleneck → N×7 signals (tied weights, backprop)
//!
//! The mixer's job: COMPRESS a sequence of byte codes into a smaller representation
//! that can reconstruct the original. NOT prediction — just faithful encoding.
//!
//! Test: "the cat" → preproc → codes → mixer encode → bottleneck → mixer decode → codes back
//!       Round-trip: reconstructed codes ≈ original codes?
//!
//! Run: cargo run --example mixer_autoencoder --release

use std::time::Instant;

struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn normal(&mut self) -> f32 { let u1 = self.f32().max(1e-7); let u2 = self.f32(); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() }
    fn range(&mut self, lo: usize, hi: usize) -> usize { if hi <= lo { lo } else { lo + (self.next() as usize % (hi - lo)) } }
}

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = std::fs::read(path).expect("read");
    let mut c = Vec::new();
    for &b in &raw { match b { b'a'..=b'z' => c.push(b-b'a'), b'A'..=b'Z' => c.push(b-b'A'), b' '|b'\n'|b'\t'|b'\r' => c.push(26), _ => {} } }
    c
}

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0*c;
    if x >= l { return x-l; } if x <= -l { return x+l; }
    let s = x/c; let n = s.floor(); let t = s-n; let h = t*(1.0-t);
    let sg = if (n as i32)%2==0 { 1.0 } else { -1.0 }; c*(sg*h+rho*h*h)
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ══════════════════════════════════════════════════════
// FROZEN PREPROCESSOR (Level 0)
// ══════════════════════════════════════════════════════
struct Preproc { w: [[i8;8];7], b: [i8;7], c: [f32;7], rho: [f32;7] }
impl Preproc {
    fn new() -> Self { Preproc {
        w: [[-1,1,1,-1,-1,1,1,-1],[1,-1,1,1,-1,-1,-1,-1],[-1,-1,1,-1,1,1,-1,-1],
            [-1,-1,-1,1,-1,-1,1,-1],[-1,1,1,1,-1,-1,-1,-1],[1,1,-1,-1,-1,-1,-1,-1],
            [-1,1,-1,1,1,1,-1,-1]],
        b: [1,1,1,1,1,1,1], c: [10.0;7], rho: [2.0,0.0,0.0,0.0,0.0,0.0,0.0],
    }}
    fn encode(&self, ch: u8) -> [f32;7] {
        let mut bits=[0.0f32;8]; for i in 0..8 { bits[i]=((ch>>i)&1) as f32; }
        let mut o=[0.0f32;7];
        for k in 0..7 { let mut d=self.b[k] as f32; for j in 0..8 { d+=self.w[k][j] as f32*bits[j]; } o[k]=c19(d,self.c[k],self.rho[k]); }
        o
    }
    fn encode_sequence(&self, chars: &[u8]) -> Vec<f32> {
        let mut signals = Vec::with_capacity(chars.len() * 7);
        for &ch in chars { signals.extend_from_slice(&self.encode(ch)); }
        signals
    }
}

// ══════════════════════════════════════════════════════
// MIXER AUTOENCODER (Level 1) — tied weights, bottleneck
// Input: N×7 signals → encoder(W) → bottleneck(B) → decoder(Wᵀ) → N×7 signals
// ══════════════════════════════════════════════════════
#[derive(Clone)]
struct MixerAutoenc {
    // Encoder: input_dim → bottleneck (sigmoid for bottleneck)
    w: Vec<Vec<f32>>,    // bottleneck × input_dim — TIED (decoder uses Wᵀ)
    enc_bias: Vec<f32>,  // bottleneck
    dec_bias: Vec<f32>,  // input_dim (decoder output bias)
    input_dim: usize,
    bottleneck: usize,
}

impl MixerAutoenc {
    fn new(input_dim: usize, bottleneck: usize, rng: &mut Rng) -> Self {
        let s = (2.0 / input_dim as f32).sqrt();
        MixerAutoenc {
            w: (0..bottleneck).map(|_| (0..input_dim).map(|_| rng.normal() * s).collect()).collect(),
            enc_bias: vec![0.0; bottleneck],
            dec_bias: vec![0.0; input_dim],
            input_dim, bottleneck,
        }
    }

    fn encode(&self, input: &[f32]) -> Vec<f32> {
        let mut hidden = vec![0.0f32; self.bottleneck];
        for k in 0..self.bottleneck {
            hidden[k] = self.enc_bias[k];
            for j in 0..self.input_dim { hidden[k] += self.w[k][j] * input[j]; }
            hidden[k] = sigmoid(hidden[k]);
        }
        hidden
    }

    fn decode(&self, hidden: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; self.input_dim];
        for j in 0..self.input_dim {
            output[j] = self.dec_bias[j];
            for k in 0..self.bottleneck {
                output[j] += self.w[k][j] * hidden[k]; // Wᵀ: transposed access
            }
        }
        output
    }

    fn round_trip(&self, input: &[f32]) -> Vec<f32> {
        self.decode(&self.encode(input))
    }

    fn train_step(&mut self, input: &[f32], lr: f32) -> f32 {
        let hidden = self.encode(input);
        let output = self.decode(&hidden);

        // MSE loss
        let mut loss = 0.0f32;
        let mut d_out = vec![0.0f32; self.input_dim];
        for j in 0..self.input_dim {
            let err = output[j] - input[j];
            loss += err * err;
            d_out[j] = 2.0 * err / self.input_dim as f32;
        }

        // Backprop decoder (Wᵀ) → update W and dec_bias
        let mut d_hidden = vec![0.0f32; self.bottleneck];
        for j in 0..self.input_dim {
            for k in 0..self.bottleneck {
                d_hidden[k] += d_out[j] * self.w[k][j]; // Wᵀ gradient
                self.w[k][j] -= lr * d_out[j] * hidden[k]; // update W via decoder path
            }
            self.dec_bias[j] -= lr * d_out[j];
        }

        // Backprop encoder (sigmoid + W)
        for k in 0..self.bottleneck {
            let dh = d_hidden[k] * hidden[k] * (1.0 - hidden[k]); // sigmoid derivative
            for j in 0..self.input_dim {
                self.w[k][j] -= lr * dh * input[j]; // update W via encoder path
            }
            self.enc_bias[k] -= lr * dh;
        }

        loss / self.input_dim as f32
    }

    fn eval_reconstruction(&self, pp: &Preproc, corpus: &[u8], ctx: usize, n_samples: usize, seed: u64) -> (f64, f64) {
        let mut rng = Rng::new(seed);
        let mut total_mse = 0.0f64;
        let mut char_correct = 0usize;
        let mut char_total = 0usize;

        for _ in 0..n_samples {
            if corpus.len() < ctx { break; }
            let off = rng.range(0, corpus.len() - ctx);
            let seg = &corpus[off..off+ctx];

            let signals = pp.encode_sequence(seg);
            let reconstructed = self.round_trip(&signals);

            // MSE
            let mse: f32 = signals.iter().zip(&reconstructed)
                .map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / signals.len() as f32;
            total_mse += mse as f64;

            // Can we recover the original characters from reconstructed signals?
            // For each position, find nearest original character code
            for i in 0..ctx {
                let recon_slice = &reconstructed[i*7..(i+1)*7];
                // Find which char has the closest code
                let mut best_ch = 0u8;
                let mut best_dist = f32::MAX;
                for ch in 0..27u8 {
                    let code = pp.encode(ch);
                    let dist: f32 = code.iter().zip(recon_slice).map(|(a,b)| (a-b)*(a-b)).sum();
                    if dist < best_dist { best_dist = dist; best_ch = ch; }
                }
                if best_ch == seg[i] { char_correct += 1; }
                char_total += 1;
            }
        }

        let avg_mse = total_mse / n_samples as f64;
        let char_acc = char_correct as f64 / char_total as f64 * 100.0;
        (avg_mse, char_acc)
    }
}

// ══════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════
fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let pp = Preproc::new();

    println!("=== MIXER AUTOENCODER — Level 2 Sequence Compression ===");
    println!("Level 0: byte → 7 signals (frozen, 77 bits)");
    println!("Level 1: N×7 signals → bottleneck → N×7 signals (tied Wᵀ)");
    println!("Test: can bottleneck faithfully compress a sequence of byte codes?\n");

    // Demo: show what the preprocessor produces
    println!("━━━ Demo: preprocessor codes ━━━");
    let demo = "the cat";
    let demo_chars: Vec<u8> = demo.bytes().map(|b| match b {
        b'a'..=b'z' => b - b'a', b' ' => 26, _ => 0
    }).collect();
    for (i, &ch) in demo_chars.iter().enumerate() {
        let code = pp.encode(ch);
        let c = if ch < 26 { (ch + b'a') as char } else { ' ' };
        let cs: Vec<String> = code.iter().map(|v| format!("{:.1}", v)).collect();
        println!("  '{}' → [{}]", c, cs.join(", "));
    }

    // Sweep: context sizes × bottleneck sizes
    println!("\n━━━ Bottleneck sweep ━━━\n");
    println!("  {:>4} {:>6} {:>8} {:>10} {:>10} {:>10} {:>8}",
        "ctx", "bottle", "ratio", "params", "mse", "char_acc", "time");
    println!("  {}", "─".repeat(65));

    for &ctx in &[8, 16, 32] {
        let input_dim = ctx * 7;
        for &bneck in &[4, 8, 16, 32, 64] {
            if bneck >= input_dim { continue; }
            let ratio = bneck as f32 / input_dim as f32;

            let tc = Instant::now();
            let mut rng = Rng::new(42);
            let mut ae = MixerAutoenc::new(input_dim, bneck, &mut rng);

            // Train
            let epochs = 50;
            let samples = 5000;
            for epoch in 0..epochs {
                let lr = 0.01 * (1.0 - epoch as f32 / epochs as f32 * 0.7);
                let mut rng_t = Rng::new(epoch as u64 * 1000 + 42);
                for _ in 0..samples {
                    if corpus.len() < ctx { break; }
                    let off = rng_t.range(0, corpus.len() - ctx);
                    let signals = pp.encode_sequence(&corpus[off..off+ctx]);
                    ae.train_step(&signals, lr);
                }
            }

            let (mse, char_acc) = ae.eval_reconstruction(&pp, &corpus, ctx, 2000, 999);
            let params = bneck * input_dim + bneck + input_dim; // W + enc_bias + dec_bias
            let marker = if char_acc > 99.0 { " ★★" } else if char_acc > 90.0 { " ★" } else { "" };

            println!("  {:>4} {:>6} {:>7.0}% {:>10} {:>10.4} {:>9.1}% {:>7.1}s{}",
                ctx, bneck, ratio * 100.0, params, mse, char_acc, tc.elapsed().as_secs_f64(), marker);
        }
        println!();
    }

    // Detailed analysis on best config
    println!("━━━ Detailed reconstruction test ━━━\n");
    let ctx = 16;
    let bneck = 32; // ~28% compression
    let input_dim = ctx * 7;
    let mut rng = Rng::new(42);
    let mut ae = MixerAutoenc::new(input_dim, bneck, &mut rng);

    for epoch in 0..80 {
        let lr = 0.01 * (1.0 - epoch as f32 / 80.0 * 0.7);
        let mut rng_t = Rng::new(epoch as u64 * 1000 + 42);
        for _ in 0..8000 {
            if corpus.len() < ctx { break; }
            let off = rng_t.range(0, corpus.len() - ctx);
            let signals = pp.encode_sequence(&corpus[off..off+ctx]);
            ae.train_step(&signals, lr);
        }
    }

    // Show some reconstructions
    let test_strings = ["the cat sat on t", "alice was beginni", "said the queen of"];
    for &s in &test_strings {
        let chars: Vec<u8> = s.bytes().map(|b| match b {
            b'a'..=b'z' => b - b'a', b' ' => 26, _ => 0
        }).collect();
        if chars.len() != ctx { continue; }

        let signals = pp.encode_sequence(&chars);
        let hidden = ae.encode(&signals);
        let reconstructed = ae.round_trip(&signals);

        // Decode back to characters
        let mut decoded = String::new();
        let mut correct = 0;
        for i in 0..ctx {
            let recon_slice = &reconstructed[i*7..(i+1)*7];
            let mut best_ch = 0u8;
            let mut best_dist = f32::MAX;
            for ch in 0..27u8 {
                let code = pp.encode(ch);
                let dist: f32 = code.iter().zip(recon_slice).map(|(a,b)| (a-b)*(a-b)).sum();
                if dist < best_dist { best_dist = dist; best_ch = ch; }
            }
            decoded.push(if best_ch < 26 { (best_ch + b'a') as char } else { ' ' });
            if best_ch == chars[i] { correct += 1; }
        }

        let bottleneck_str: Vec<String> = hidden.iter().take(8).map(|v| format!("{:.2}", v)).collect();
        println!("  \"{}\" → [{}...] → \"{}\" ({}/{})",
            s, bottleneck_str.join(","), decoded, correct, ctx);
    }

    println!("\n  Bottleneck: {} dims for {} input signals = {:.0}% compression",
        bneck, input_dim, (1.0 - bneck as f64 / input_dim as f64) * 100.0);
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
