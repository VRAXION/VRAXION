//! Bottleneck sweep: 8 → N → 8 mirrored MLP, how low can N go?
//!
//! Tests tied-weight MLP autoencoder at bottleneck sizes 7,6,5,4,3,2,1
//! with backprop training. Shows where perfect round-trip breaks.
//!
//! Run: cargo run --example bottleneck_sweep --release

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
    let mut seen = [false; 256]; for &b in &text { seen[b as usize] = true; }
    (0..=255u8).filter(|&b| seen[b as usize]).collect()
}

fn byte_to_bits(b: u8) -> [f32; 8] {
    let mut bits = [0.0f32; 8]; for i in 0..8 { bits[i] = ((b >> i) & 1) as f32; } bits
}

#[derive(Clone)]
struct Mlp { w1: Vec<Vec<f32>>, b1: Vec<f32>, w2: Vec<Vec<f32>>, b2: Vec<f32>, b3: Vec<f32>, b4: Vec<f32>, h: usize, bo: usize }

impl Mlp {
    fn new(h: usize, bo: usize, rng: &mut Rng) -> Self {
        let s1 = (2.0 / 8.0f32).sqrt(); let s2 = (2.0 / h as f32).sqrt();
        Mlp {
            w1: (0..h).map(|_| (0..8).map(|_| rng.normal() * s1).collect()).collect(),
            b1: vec![0.0; h],
            w2: (0..bo).map(|_| (0..h).map(|_| rng.normal() * s2).collect()).collect(),
            b2: vec![0.0; bo], b3: vec![0.0; h], b4: vec![0.0; 8], h, bo,
        }
    }
    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

    fn train_step(&mut self, inp: &[f32; 8], lr: f32) -> f32 {
        let mut z1 = vec![0.0f32; self.h]; let mut a1 = vec![0.0f32; self.h];
        for i in 0..self.h { z1[i] = self.b1[i]; for j in 0..8 { z1[i] += self.w1[i][j] * inp[j]; } a1[i] = z1[i].max(0.0); }
        let mut z2 = vec![0.0f32; self.bo]; let mut a2 = vec![0.0f32; self.bo];
        for i in 0..self.bo { z2[i] = self.b2[i]; for j in 0..self.h { z2[i] += self.w2[i][j] * a1[j]; } a2[i] = Self::sigmoid(z2[i]); }
        let mut z3 = vec![0.0f32; self.h]; let mut a3 = vec![0.0f32; self.h];
        for i in 0..self.h { z3[i] = self.b3[i]; for j in 0..self.bo { z3[i] += self.w2[j][i] * a2[j]; } a3[i] = z3[i].max(0.0); }
        let mut a4 = [0.0f32; 8]; let mut z4 = [0.0f32; 8];
        for i in 0..8 { z4[i] = self.b4[i]; for j in 0..self.h { z4[i] += self.w1[j][i] * a3[j]; } a4[i] = Self::sigmoid(z4[i]); }

        let mut loss = 0.0f32; let mut d4 = [0.0f32; 8];
        for i in 0..8 { let e = a4[i] - inp[i]; loss += e*e; d4[i] = 2.0*e*a4[i]*(1.0-a4[i]); }
        let mut da3 = vec![0.0f32; self.h];
        for j in 0..self.h { for i in 0..8 { da3[j] += d4[i]*self.w1[j][i]; self.w1[j][i] -= lr*d4[i]*a3[j]; } }
        for i in 0..8 { self.b4[i] -= lr*d4[i]; }
        let mut d3 = vec![0.0f32; self.h]; for i in 0..self.h { d3[i] = if z3[i] > 0.0 { da3[i] } else { 0.0 }; }
        let mut da2 = vec![0.0f32; self.bo];
        for j in 0..self.bo { for i in 0..self.h { da2[j] += d3[i]*self.w2[j][i]; self.w2[j][i] -= lr*d3[i]*a2[j]; } }
        for i in 0..self.h { self.b3[i] -= lr*d3[i]; }
        let mut d2 = vec![0.0f32; self.bo]; for i in 0..self.bo { d2[i] = da2[i]*a2[i]*(1.0-a2[i]); }
        let mut da1 = vec![0.0f32; self.h];
        for i in 0..self.bo { for j in 0..self.h { da1[j] += d2[i]*self.w2[i][j]; self.w2[i][j] -= lr*d2[i]*a1[j]; } self.b2[i] -= lr*d2[i]; }
        let mut d1 = vec![0.0f32; self.h]; for i in 0..self.h { d1[i] = if z1[i] > 0.0 { da1[i] } else { 0.0 }; }
        for i in 0..self.h { for j in 0..8 { self.w1[i][j] -= lr*d1[i]*inp[j]; } self.b1[i] -= lr*d1[i]; }
        loss
    }

    fn eval(&self, inp: &[f32; 8]) -> (Vec<u8>, [f32; 8]) {
        let mut a1 = vec![0.0f32; self.h];
        for i in 0..self.h { let mut s = self.b1[i]; for j in 0..8 { s += self.w1[i][j] * inp[j]; } a1[i] = s.max(0.0); }
        let mut a2 = vec![0.0f32; self.bo];
        for i in 0..self.bo { let mut s = self.b2[i]; for j in 0..self.h { s += self.w2[i][j] * a1[j]; } a2[i] = Self::sigmoid(s); }
        let code: Vec<u8> = a2.iter().map(|&v| if v >= 0.5 { 1 } else { 0 }).collect();
        let mut a3 = vec![0.0f32; self.h];
        for i in 0..self.h { let mut s = self.b3[i]; for j in 0..self.bo { s += self.w2[j][i] * a2[j]; } a3[i] = s.max(0.0); }
        let mut a4 = [0.0f32; 8];
        for i in 0..8 { let mut s = self.b4[i]; for j in 0..self.h { s += self.w1[j][i] * a3[j]; } a4[i] = Self::sigmoid(s); }
        (code, a4)
    }

    fn accuracy(&self, bytes: &[u8]) -> (usize, Vec<u8>, usize) {
        let mut ok = 0; let mut failed = Vec::new();
        let mut codes: Vec<Vec<u8>> = Vec::new();
        for &b in bytes {
            let inp = byte_to_bits(b);
            let (code, out) = self.eval(&inp);
            let pass = (0..8).all(|i| (out[i] - inp[i]).abs() < 0.4);
            if pass { ok += 1; } else { failed.push(b); }
            codes.push(code);
        }
        codes.sort(); codes.dedup();
        (ok, failed, codes.len())
    }
}

fn main() {
    let t0 = Instant::now();
    let unique = load_unique_bytes("instnct-core/tests/fixtures/alice_corpus.txt");
    let train: Vec<[f32; 8]> = unique.iter().map(|&b| byte_to_bits(b)).collect();

    println!("=== BOTTLENECK SWEEP: How low can N go? ===");
    println!("Architecture: 8 → H(ReLU) → N(sigmoid) → H(W2ᵀ, ReLU) → 8(W1ᵀ, sigmoid)");
    println!("Tied weights. {} unique bytes. Theoretical min = ceil(log2({})) = {} bits.\n",
        unique.len(), unique.len(), (unique.len() as f64).log2().ceil() as usize);

    println!("{:>4} {:>4} {:>8} {:>8} {:>10} {:>8} {:>12} {:>10}",
        "N", "H", "acc", "pct", "unique", "capacity", "theory", "status");
    println!("{}", "─".repeat(80));

    for &bottleneck in &[7, 6, 5, 4, 3, 2, 1] {
        // Try multiple H sizes and seeds
        let mut best_acc = 0usize;
        let mut best_unique = 0usize;
        let mut best_h = 0usize;
        let mut best_failed: Vec<u8> = Vec::new();

        for &h in &[8, 12, 16, 24, 32, 48] {
            for seed in 0..20 {
                let mut rng = Rng::new(42 + seed * 1000 + h as u64 * 7 + bottleneck as u64 * 31);
                let mut mlp = Mlp::new(h, bottleneck, &mut rng);

                let epochs = 10000;
                for epoch in 0..epochs {
                    let lr = 0.05 * (1.0 - epoch as f32 / epochs as f32 * 0.9);
                    for inp in &train { mlp.train_step(inp, lr); }
                }

                let (acc, failed, uniq) = mlp.accuracy(&unique);
                if acc > best_acc || (acc == best_acc && uniq > best_unique) {
                    best_acc = acc;
                    best_unique = uniq;
                    best_h = h;
                    best_failed = failed;
                }
                if best_acc == unique.len() { break; }
            }
            if best_acc == unique.len() { break; }
        }

        let capacity = 1usize << bottleneck;
        let possible = if capacity >= unique.len() { "POSSIBLE" } else { "MUST GROUP" };
        let status = if best_acc == unique.len() { "★ PERFECT" }
            else if best_acc >= unique.len() * 90 / 100 { "good" }
            else if best_acc >= unique.len() * 70 / 100 { "partial" }
            else { "poor" };

        println!("{:>4} {:>4} {:>5}/{:<2} {:>7.1}% {:>7}/{:<2} {:>8} {:>12} {:>10}",
            bottleneck, best_h, best_acc, unique.len(),
            best_acc as f64 / unique.len() as f64 * 100.0,
            best_unique, unique.len(), capacity, possible, status);

        if !best_failed.is_empty() && best_failed.len() <= 10 {
            let fs: Vec<String> = best_failed.iter().map(|&b| {
                if b >= 32 && b < 127 { format!("'{}'", b as char) } else { format!("x{:02x}", b) }
            }).collect();
            println!("     failed: [{}]", fs.join(", "));
        }
    }

    println!("\n━━━ SUMMARY ━━━");
    println!("  29 unique bytes, theoretical minimum = 5 bits (32 codes ≥ 29)");
    println!("  Capacity at each level: 7b=128, 6b=64, 5b=32, 4b=16, 3b=8, 2b=4, 1b=2");
    println!("\n  Time: {:.1}s", t0.elapsed().as_secs_f64());
}
