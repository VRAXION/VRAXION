//! Quantization margin analysis: how CONFIDENT are the bottleneck neurons
//! at different bit-widths? And how much SEPARATION is there between codes?
//!
//! If int8 gives wider margins than int5, the downstream core has cleaner signals.
//!
//! Run: cargo run --example quant_margin --release

use std::time::Instant;

struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn normal(&mut self) -> f32 { let u1 = self.f32().max(1e-7); let u2 = self.f32(); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() }
}

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
// MLP (same as before, compact)
// ══════════════════════════════════════════════════════
#[derive(Clone)]
struct Mlp {
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    b3: Vec<f32>, b4: Vec<f32>,
    h: usize, bo: usize,
}

impl Mlp {
    fn new(h: usize, bo: usize, rng: &mut Rng) -> Self {
        let s1 = (2.0 / 8.0f32).sqrt();
        let s2 = (2.0 / h as f32).sqrt();
        Mlp {
            w1: (0..h).map(|_| (0..8).map(|_| rng.normal() * s1).collect()).collect(),
            b1: vec![0.0; h], w2: (0..bo).map(|_| (0..h).map(|_| rng.normal() * s2).collect()).collect(),
            b2: vec![0.0; bo], b3: vec![0.0; h], b4: vec![0.0; 8], h, bo,
        }
    }
    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

    fn forward_full(&self, inp: &[f32; 8]) -> (Vec<f32>, Vec<f32>, [f32; 8]) {
        // Returns: (z_bottleneck_pre_sigmoid, a_bottleneck_post_sigmoid, output)
        let mut a1 = vec![0.0f32; self.h];
        let mut z1 = vec![0.0f32; self.h];
        for i in 0..self.h {
            z1[i] = self.b1[i]; for j in 0..8 { z1[i] += self.w1[i][j] * inp[j]; }
            a1[i] = z1[i].max(0.0);
        }
        let mut z2 = vec![0.0f32; self.bo];
        let mut a2 = vec![0.0f32; self.bo];
        for i in 0..self.bo {
            z2[i] = self.b2[i]; for j in 0..self.h { z2[i] += self.w2[i][j] * a1[j]; }
            a2[i] = Self::sigmoid(z2[i]);
        }
        let mut a3 = vec![0.0f32; self.h];
        for i in 0..self.h {
            let mut s = self.b3[i]; for j in 0..self.bo { s += self.w2[j][i] * a2[j]; }
            a3[i] = s.max(0.0);
        }
        let mut a4 = [0.0f32; 8];
        for i in 0..8 {
            let mut s = self.b4[i]; for j in 0..self.h { s += self.w1[j][i] * a3[j]; }
            a4[i] = Self::sigmoid(s);
        }
        (z2, a2, a4)
    }

    fn train_step(&mut self, inp: &[f32; 8], lr: f32) -> f32 {
        let mut a1 = vec![0.0f32; self.h]; let mut z1 = vec![0.0f32; self.h];
        for i in 0..self.h { z1[i] = self.b1[i]; for j in 0..8 { z1[i] += self.w1[i][j] * inp[j]; } a1[i] = z1[i].max(0.0); }
        let mut z2 = vec![0.0f32; self.bo]; let mut a2 = vec![0.0f32; self.bo];
        for i in 0..self.bo { z2[i] = self.b2[i]; for j in 0..self.h { z2[i] += self.w2[i][j] * a1[j]; } a2[i] = Self::sigmoid(z2[i]); }
        let mut z3 = vec![0.0f32; self.h]; let mut a3 = vec![0.0f32; self.h];
        for i in 0..self.h { z3[i] = self.b3[i]; for j in 0..self.bo { z3[i] += self.w2[j][i] * a2[j]; } a3[i] = z3[i].max(0.0); }
        let mut a4 = [0.0f32; 8];
        for i in 0..8 { a4[i] = self.b4[i]; for j in 0..self.h { a4[i] += self.w1[j][i] * a3[j]; } a4[i] = Self::sigmoid(a4[i]); }

        let mut loss = 0.0f32; let mut d4 = [0.0f32; 8];
        for i in 0..8 { let e = a4[i] - inp[i]; loss += e*e; d4[i] = 2.0*e*a4[i]*(1.0-a4[i]); }
        let mut da3 = vec![0.0f32; self.h];
        for j in 0..self.h { for i in 0..8 { da3[j] += d4[i]*self.w1[j][i]; self.w1[j][i] -= lr*d4[i]*a3[j]; } }
        for i in 0..8 { self.b4[i] -= lr*d4[i]; }
        let mut d3 = vec![0.0f32; self.h];
        for i in 0..self.h { d3[i] = if z3[i] > 0.0 { da3[i] } else { 0.0 }; }
        let mut da2 = vec![0.0f32; self.bo];
        for j in 0..self.bo { for i in 0..self.h { da2[j] += d3[i]*self.w2[j][i]; self.w2[j][i] -= lr*d3[i]*a2[j]; } }
        for i in 0..self.h { self.b3[i] -= lr*d3[i]; }
        let mut d2 = vec![0.0f32; self.bo];
        for i in 0..self.bo { d2[i] = da2[i]*a2[i]*(1.0-a2[i]); }
        let mut da1 = vec![0.0f32; self.h];
        for i in 0..self.bo { for j in 0..self.h { da1[j] += d2[i]*self.w2[i][j]; self.w2[i][j] -= lr*d2[i]*a1[j]; } self.b2[i] -= lr*d2[i]; }
        let mut d1 = vec![0.0f32; self.h];
        for i in 0..self.h { d1[i] = if z1[i] > 0.0 { da1[i] } else { 0.0 }; }
        for i in 0..self.h { for j in 0..8 { self.w1[i][j] -= lr*d1[i]*inp[j]; } self.b1[i] -= lr*d1[i]; }
        loss
    }
}

// ══════════════════════════════════════════════════════
// QUANTIZED FORWARD — returns raw bottleneck activations
// ══════════════════════════════════════════════════════
fn quant_forward(mlp: &Mlp, inp: &[f32; 8], bits: u32) -> (Vec<f32>, Vec<f32>, [f32; 8]) {
    let max_int: i32 = if bits == 1 { 1 } else { (1i32 << (bits - 1)) - 1 };
    let q = |v: f32, scale: f32| -> f32 { (v / scale).round().max(-max_int as f32).min(max_int as f32) * scale };
    let scale_mat = |m: &[Vec<f32>]| -> f32 { m.iter().flat_map(|r| r.iter()).map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-7) / max_int as f32 };
    let scale_vec = |v: &[f32]| -> f32 { v.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-7) / max_int as f32 };

    let sw1 = scale_mat(&mlp.w1); let sw2 = scale_mat(&mlp.w2);
    let sb1 = scale_vec(&mlp.b1); let sb2 = scale_vec(&mlp.b2);
    let sb3 = scale_vec(&mlp.b3); let sb4 = scale_vec(&mlp.b4);

    let mut a1 = vec![0.0f32; mlp.h];
    for i in 0..mlp.h {
        let mut s = q(mlp.b1[i], sb1);
        for j in 0..8 { s += q(mlp.w1[i][j], sw1) * inp[j]; }
        a1[i] = s.max(0.0);
    }
    let mut z2 = vec![0.0f32; mlp.bo];
    let mut a2 = vec![0.0f32; mlp.bo];
    for i in 0..mlp.bo {
        z2[i] = q(mlp.b2[i], sb2);
        for j in 0..mlp.h { z2[i] += q(mlp.w2[i][j], sw2) * a1[j]; }
        a2[i] = Mlp::sigmoid(z2[i]);
    }
    let mut a3 = vec![0.0f32; mlp.h];
    for i in 0..mlp.h {
        let mut s = q(mlp.b3[i], sb3);
        for j in 0..mlp.bo { s += q(mlp.w2[j][i], sw2) * a2[j]; }
        a3[i] = s.max(0.0);
    }
    let mut a4 = [0.0f32; 8];
    for i in 0..8 {
        let mut s = q(mlp.b4[i], sb4);
        for j in 0..mlp.h { s += q(mlp.w1[j][i], sw1) * a3[j]; }
        a4[i] = Mlp::sigmoid(s);
    }
    (z2, a2, a4)
}

// ══════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════
fn main() {
    let t0 = Instant::now();
    let unique = load_unique_bytes("instnct-core/tests/fixtures/alice_corpus.txt");
    let train_data: Vec<[f32; 8]> = unique.iter().map(|&b| byte_to_bits(b)).collect();

    println!("=== QUANTIZATION MARGIN ANALYSIS ===");
    println!("Q: Does int8 give better margins than int5?");
    println!("   → Would the downstream core get cleaner signals?\n");

    // Train best H=12 model
    let mut best_mlp = None;
    let mut best_loss = f32::MAX;
    for seed in 0..20 {
        let mut rng = Rng::new(42 + seed * 1000);
        let mut mlp = Mlp::new(12, 7, &mut rng);
        for epoch in 0..10000 {
            let lr = 0.05 * (1.0 - epoch as f32 / 10000.0 * 0.9);
            let mut tl = 0.0f32;
            for inp in &train_data { tl += mlp.train_step(inp, lr); }
            if epoch == 9999 && tl < best_loss { best_loss = tl; best_mlp = Some(mlp.clone()); }
        }
    }
    let mlp = best_mlp.unwrap();
    println!("Trained H=12, B=7, best loss={:.6}\n", best_loss / train_data.len() as f32);

    // Analyze margins at each bit width
    println!("━━━ BOTTLENECK CONFIDENCE (distance from 0.5 threshold) ━━━\n");
    println!("{:>6} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "bits", "acc", "min_conf", "mean_conf", "median", "min_margin", "mean_margin");
    println!("{}", "─".repeat(75));

    for &bits in &[32, 8, 7, 6, 5, 4, 3] {
        let mut all_confidences: Vec<f32> = Vec::new(); // |activation - 0.5|
        let mut all_margins: Vec<f32> = Vec::new();     // min |activation - 0.5| per byte
        let mut correct = 0usize;

        for &b in &unique {
            let inp = byte_to_bits(b);
            let (_, a2, a4) = if bits == 32 {
                mlp.forward_full(&inp)
            } else {
                quant_forward(&mlp, &inp, bits)
            };

            // Check accuracy
            let ok = (0..8).all(|i| (a4[i] - inp[i]).abs() < 0.4);
            if ok { correct += 1; }

            // Confidence: how far is each bottleneck neuron from 0.5?
            let confs: Vec<f32> = a2.iter().map(|&v| (v - 0.5).abs()).collect();
            let min_conf = confs.iter().cloned().fold(f32::MAX, f32::min);
            all_margins.push(min_conf);
            all_confidences.extend(confs);
        }

        all_confidences.sort_by(|a, b| a.partial_cmp(b).unwrap());
        all_margins.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_conf = all_confidences.first().copied().unwrap_or(0.0);
        let mean_conf = all_confidences.iter().sum::<f32>() / all_confidences.len() as f32;
        let median_conf = all_confidences[all_confidences.len() / 2];
        let min_margin = all_margins.first().copied().unwrap_or(0.0);
        let mean_margin = all_margins.iter().sum::<f32>() / all_margins.len() as f32;

        let bits_str = if bits == 32 { "f32".to_string() } else { format!("{}b", bits) };
        let acc_str = format!("{}/{}", correct, unique.len());
        let marker = if correct == unique.len() { " ★" } else { "" };

        println!("{:>6} {:>8} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}{}",
            bits_str, acc_str, min_conf, mean_conf, median_conf, min_margin, mean_margin, marker);
    }

    // Detailed per-byte analysis: float vs int8 vs int5
    println!("\n━━━ PER-BYTE BOTTLENECK ACTIVATIONS ━━━\n");
    println!("{:>6} {:>6}  {:>50}  {:>50}  {:>50}",
        "byte", "char", "float32 activations", "int8 activations", "int5 activations");

    for &b in &unique {
        let inp = byte_to_bits(b);
        let (_, a2_f, _) = mlp.forward_full(&inp);
        let (_, a2_8, _) = quant_forward(&mlp, &inp, 8);
        let (_, a2_5, _) = quant_forward(&mlp, &inp, 5);

        let fmt = |a: &[f32]| -> String {
            a.iter().map(|&v| {
                if v > 0.99 { " 1.00".to_string() }
                else if v < 0.01 { " 0.00".to_string() }
                else { format!(" {:.2}", v) }
            }).collect::<Vec<_>>().join("")
        };

        let ch = if b >= 32 && b < 127 { format!("'{}'", b as char) } else { format!("x{:02x}", b) };
        // Only show a few interesting ones
        if b == b' ' || b == b'a' || b == b'e' || b == b't' || b == b'z' || b == b'b' || b == b'q' {
            println!("{:>6} {:>6}  {:>50}  {:>50}  {:>50}", b, ch, fmt(&a2_f), fmt(&a2_8), fmt(&a2_5));
        }
    }

    // Hamming distance analysis between codes
    println!("\n━━━ CODE SEPARATION (Hamming distance between all pairs) ━━━\n");

    for &bits in &[32, 8, 5] {
        let codes: Vec<Vec<u8>> = unique.iter().map(|&b| {
            let inp = byte_to_bits(b);
            let (_, a2, _) = if bits == 32 { mlp.forward_full(&inp) } else { quant_forward(&mlp, &inp, bits) };
            a2.iter().map(|&v| if v >= 0.5 { 1u8 } else { 0u8 }).collect()
        }).collect();

        let mut min_hd = 8usize;
        let mut total_hd = 0usize;
        let mut n_pairs = 0usize;
        let mut collisions = 0usize;

        for i in 0..codes.len() {
            for j in (i+1)..codes.len() {
                let hd: usize = codes[i].iter().zip(&codes[j]).filter(|(a, b)| a != b).count();
                if hd < min_hd { min_hd = hd; }
                if hd == 0 { collisions += 1; }
                total_hd += hd;
                n_pairs += 1;
            }
        }

        let mut unique_codes = codes.clone();
        unique_codes.sort(); unique_codes.dedup();

        let bits_str = if bits == 32 { "f32" } else if bits == 8 { "int8" } else { "int5" };
        println!("  {}: unique_codes={}/{}, min_hamming={}, mean_hamming={:.2}, collisions={}",
            bits_str, unique_codes.len(), unique.len(), min_hd,
            total_hd as f64 / n_pairs as f64, collisions);
    }

    // Output quality comparison
    println!("\n━━━ OUTPUT RECONSTRUCTION QUALITY ━━━\n");
    println!("{:>6} {:>10} {:>10} {:>10} {:>12}",
        "bits", "mean_err", "max_err", "min_gap", "description");
    println!("{}", "─".repeat(55));

    for &bits in &[32, 8, 7, 6, 5, 4] {
        let mut total_err = 0.0f32;
        let mut max_err = 0.0f32;
        let mut min_gap = f32::MAX; // smallest distance from 0.5 in output
        let mut n = 0usize;

        for &b in &unique {
            let inp = byte_to_bits(b);
            let (_, _, a4) = if bits == 32 { mlp.forward_full(&inp) } else { quant_forward(&mlp, &inp, bits) };
            for i in 0..8 {
                let err = (a4[i] - inp[i]).abs();
                total_err += err;
                if err > max_err { max_err = err; }
                let gap = (a4[i] - 0.5).abs();
                if gap < min_gap { min_gap = gap; }
                n += 1;
            }
        }

        let bits_str = if bits == 32 { "f32".to_string() } else { format!("{}b", bits) };
        let desc = if max_err < 0.1 { "excellent" }
            else if max_err < 0.3 { "good" }
            else if max_err < 0.5 { "borderline" }
            else { "BROKEN" };
        println!("{:>6} {:>10.5} {:>10.5} {:>10.5} {:>12}",
            bits_str, total_err / n as f32, max_err, min_gap, desc);
    }

    println!("\n━━━ VERDICT ━━━");
    println!("  Does int8 give advantages over int5 beyond just accuracy?");
    println!("  → Check the margins, separations, and output quality above.");
    println!("\n  Time: {:.1}s", t0.elapsed().as_secs_f64());
}
