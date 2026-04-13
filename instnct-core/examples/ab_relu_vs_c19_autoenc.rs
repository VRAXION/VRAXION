//! A/B test: ReLU vs C19 activation in mirrored autoencoder
//!
//! Same tied-weight architecture, same bottleneck sweep, same backprop.
//! A = ReLU (baseline, already proven)
//! B = C19 with learnable c,rho per neuron (backpropped)
//!
//! Question: does C19 need fewer hidden neurons for perfect round-trip?
//!
//! Run: cargo run --example ab_relu_vs_c19_autoenc --release

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

// ══════════════════════════════════════════════════════
// C19 ACTIVATION + DERIVATIVE (learnable c, rho)
// ══════════════════════════════════════════════════════
fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1);
    let rho = rho.max(0.0);
    let l = 6.0 * c;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let scaled = x / c;
    let n = scaled.floor();
    let t = scaled - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

/// Numerical derivative of c19 w.r.t. x (for backprop)
fn c19_dx(x: f32, c: f32, rho: f32) -> f32 {
    let eps = 0.001;
    (c19(x + eps, c, rho) - c19(x - eps, c, rho)) / (2.0 * eps)
}

/// Numerical derivative of c19 w.r.t. c
fn c19_dc(x: f32, c: f32, rho: f32) -> f32 {
    let eps = 0.001;
    (c19(x, c + eps, rho) - c19(x, c - eps, rho)) / (2.0 * eps)
}

/// Numerical derivative of c19 w.r.t. rho
fn c19_drho(x: f32, c: f32, rho: f32) -> f32 {
    let eps = 0.001;
    (c19(x, c, rho + eps) - c19(x, c, rho - eps)) / (2.0 * eps)
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ══════════════════════════════════════════════════════
// MLP with selectable activation: ReLU or C19
// ══════════════════════════════════════════════════════
#[derive(Clone)]
struct Mlp {
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    b3: Vec<f32>, b4: Vec<f32>,
    // C19 params (per hidden neuron, encoder + decoder separate)
    c_enc: Vec<f32>,    // learnable c per encoder hidden neuron
    rho_enc: Vec<f32>,  // learnable rho per encoder hidden neuron
    c_dec: Vec<f32>,    // learnable c per decoder hidden neuron
    rho_dec: Vec<f32>,  // learnable rho per decoder hidden neuron
    h: usize, bo: usize,
    use_c19: bool,
}

impl Mlp {
    fn new(h: usize, bo: usize, use_c19: bool, rng: &mut Rng) -> Self {
        let s1 = (2.0 / 8.0f32).sqrt(); let s2 = (2.0 / h as f32).sqrt();
        Mlp {
            w1: (0..h).map(|_| (0..8).map(|_| rng.normal() * s1).collect()).collect(),
            b1: vec![0.0; h],
            w2: (0..bo).map(|_| (0..h).map(|_| rng.normal() * s2).collect()).collect(),
            b2: vec![0.0; bo], b3: vec![0.0; h], b4: vec![0.0; 8],
            c_enc: vec![3.0; h], rho_enc: vec![1.0; h],
            c_dec: vec![3.0; h], rho_dec: vec![1.0; h],
            h, bo, use_c19,
        }
    }

    fn act_enc(&self, x: f32, i: usize) -> f32 {
        if self.use_c19 { c19(x, self.c_enc[i], self.rho_enc[i]) } else { x.max(0.0) }
    }

    fn act_dec(&self, x: f32, i: usize) -> f32 {
        if self.use_c19 { c19(x, self.c_dec[i], self.rho_dec[i]) } else { x.max(0.0) }
    }

    fn train_step(&mut self, inp: &[f32; 8], lr: f32) -> f32 {
        // Forward
        let mut z1 = vec![0.0f32; self.h]; let mut a1 = vec![0.0f32; self.h];
        for i in 0..self.h {
            z1[i] = self.b1[i]; for j in 0..8 { z1[i] += self.w1[i][j] * inp[j]; }
            a1[i] = self.act_enc(z1[i], i);
        }
        let mut z2 = vec![0.0f32; self.bo]; let mut a2 = vec![0.0f32; self.bo];
        for i in 0..self.bo {
            z2[i] = self.b2[i]; for j in 0..self.h { z2[i] += self.w2[i][j] * a1[j]; }
            a2[i] = sigmoid(z2[i]);
        }
        let mut z3 = vec![0.0f32; self.h]; let mut a3 = vec![0.0f32; self.h];
        for i in 0..self.h {
            z3[i] = self.b3[i]; for j in 0..self.bo { z3[i] += self.w2[j][i] * a2[j]; }
            a3[i] = self.act_dec(z3[i], i);
        }
        let mut a4 = [0.0f32; 8]; let mut z4 = [0.0f32; 8];
        for i in 0..8 {
            z4[i] = self.b4[i]; for j in 0..self.h { z4[i] += self.w1[j][i] * a3[j]; }
            a4[i] = sigmoid(z4[i]);
        }

        // Backward
        let mut loss = 0.0f32; let mut d4 = [0.0f32; 8];
        for i in 0..8 { let e = a4[i] - inp[i]; loss += e*e; d4[i] = 2.0*e*a4[i]*(1.0-a4[i]); }

        // Layer 4 → 3
        let mut da3 = vec![0.0f32; self.h];
        for j in 0..self.h { for i in 0..8 { da3[j] += d4[i]*self.w1[j][i]; self.w1[j][i] -= lr*d4[i]*a3[j]; } }
        for i in 0..8 { self.b4[i] -= lr*d4[i]; }

        // Decoder activation gradient
        let mut d3 = vec![0.0f32; self.h];
        for i in 0..self.h {
            if self.use_c19 {
                let dx = c19_dx(z3[i], self.c_dec[i], self.rho_dec[i]);
                d3[i] = da3[i] * dx;
                // Update c_dec, rho_dec
                let dc = da3[i] * c19_dc(z3[i], self.c_dec[i], self.rho_dec[i]);
                let dr = da3[i] * c19_drho(z3[i], self.c_dec[i], self.rho_dec[i]);
                self.c_dec[i] -= lr * dc;
                self.rho_dec[i] -= lr * dr;
                self.c_dec[i] = self.c_dec[i].max(0.1).min(20.0);
                self.rho_dec[i] = self.rho_dec[i].max(0.0).min(10.0);
            } else {
                d3[i] = if z3[i] > 0.0 { da3[i] } else { 0.0 };
            }
        }

        // Layer 3 → 2
        let mut da2 = vec![0.0f32; self.bo];
        for j in 0..self.bo { for i in 0..self.h { da2[j] += d3[i]*self.w2[j][i]; self.w2[j][i] -= lr*d3[i]*a2[j]; } }
        for i in 0..self.h { self.b3[i] -= lr*d3[i]; }

        let mut d2 = vec![0.0f32; self.bo];
        for i in 0..self.bo { d2[i] = da2[i]*a2[i]*(1.0-a2[i]); }

        // Layer 2 → 1
        let mut da1 = vec![0.0f32; self.h];
        for i in 0..self.bo { for j in 0..self.h { da1[j] += d2[i]*self.w2[i][j]; self.w2[i][j] -= lr*d2[i]*a1[j]; } self.b2[i] -= lr*d2[i]; }

        // Encoder activation gradient
        let mut d1 = vec![0.0f32; self.h];
        for i in 0..self.h {
            if self.use_c19 {
                let dx = c19_dx(z1[i], self.c_enc[i], self.rho_enc[i]);
                d1[i] = da1[i] * dx;
                let dc = da1[i] * c19_dc(z1[i], self.c_enc[i], self.rho_enc[i]);
                let dr = da1[i] * c19_drho(z1[i], self.c_enc[i], self.rho_enc[i]);
                self.c_enc[i] -= lr * dc;
                self.rho_enc[i] -= lr * dr;
                self.c_enc[i] = self.c_enc[i].max(0.1).min(20.0);
                self.rho_enc[i] = self.rho_enc[i].max(0.0).min(10.0);
            } else {
                d1[i] = if z1[i] > 0.0 { da1[i] } else { 0.0 };
            }
        }

        // Layer 1 → input
        for i in 0..self.h { for j in 0..8 { self.w1[i][j] -= lr*d1[i]*inp[j]; } self.b1[i] -= lr*d1[i]; }
        loss
    }

    fn accuracy(&self, bytes: &[u8]) -> (usize, usize) {
        let mut ok = 0;
        for &b in bytes {
            let inp = byte_to_bits(b);
            // Forward
            let mut a1 = vec![0.0f32; self.h];
            for i in 0..self.h { let mut s = self.b1[i]; for j in 0..8 { s += self.w1[i][j]*inp[j]; } a1[i] = self.act_enc(s, i); }
            let mut a2 = vec![0.0f32; self.bo];
            for i in 0..self.bo { let mut s = self.b2[i]; for j in 0..self.h { s += self.w2[i][j]*a1[j]; } a2[i] = sigmoid(s); }
            let mut a3 = vec![0.0f32; self.h];
            for i in 0..self.h { let mut s = self.b3[i]; for j in 0..self.bo { s += self.w2[j][i]*a2[j]; } a3[i] = self.act_dec(s, i); }
            let mut a4 = [0.0f32; 8];
            for i in 0..8 { let mut s = self.b4[i]; for j in 0..self.h { s += self.w1[j][i]*a3[j]; } a4[i] = sigmoid(s); }
            if (0..8).all(|i| (a4[i] - inp[i]).abs() < 0.4) { ok += 1; }
        }
        (ok, bytes.len())
    }
}

// ══════════════════════════════════════════════════════
// MAIN — A/B sweep
// ══════════════════════════════════════════════════════
fn main() {
    let t0 = Instant::now();
    let unique = load_unique_bytes("instnct-core/tests/fixtures/alice_corpus.txt");
    let train: Vec<[f32; 8]> = unique.iter().map(|&b| byte_to_bits(b)).collect();

    println!("=== A/B TEST: ReLU vs C19 Mirrored Autoencoder ===");
    println!("{} unique bytes, tied weights, backprop\n", unique.len());

    println!("{:>4} {:>4} {:>12} {:>12} {:>10}",
        "N", "H", "A: ReLU", "B: C19", "winner");
    println!("{}", "─".repeat(55));

    for &bottleneck in &[7, 6, 5, 4, 3, 2] {
        let mut best_relu = 0usize;
        let mut best_c19 = 0usize;
        let mut best_relu_h = 0usize;
        let mut best_c19_h = 0usize;

        for &h in &[8, 12, 16, 24, 32, 48] {
            // ReLU arm
            for seed in 0..15 {
                let mut rng = Rng::new(42 + seed*1000 + h as u64*7 + bottleneck as u64*31);
                let mut mlp = Mlp::new(h, bottleneck, false, &mut rng);
                for epoch in 0..10000 {
                    let lr = 0.05 * (1.0 - epoch as f32 / 10000.0 * 0.9);
                    for inp in &train { mlp.train_step(inp, lr); }
                }
                let (acc, _) = mlp.accuracy(&unique);
                if acc > best_relu { best_relu = acc; best_relu_h = h; }
                if best_relu == unique.len() { break; }
            }

            // C19 arm
            for seed in 0..15 {
                let mut rng = Rng::new(42 + seed*1000 + h as u64*7 + bottleneck as u64*31 + 999);
                let mut mlp = Mlp::new(h, bottleneck, true, &mut rng);
                for epoch in 0..10000 {
                    let lr = 0.03 * (1.0 - epoch as f32 / 10000.0 * 0.9); // lower lr for c19 stability
                    for inp in &train { mlp.train_step(inp, lr); }
                }
                let (acc, _) = mlp.accuracy(&unique);
                if acc > best_c19 { best_c19 = acc; best_c19_h = h; }
                if best_c19 == unique.len() { break; }
            }

            if best_relu == unique.len() && best_c19 == unique.len() { break; }
        }

        let winner = if best_relu > best_c19 { "ReLU" }
            else if best_c19 > best_relu { "C19" }
            else { "TIE" };

        let relu_str = format!("{}/{}(H={})", best_relu, unique.len(), best_relu_h);
        let c19_str = format!("{}/{}(H={})", best_c19, unique.len(), best_c19_h);

        println!("{:>4} {:>4} {:>12} {:>12} {:>10}",
            bottleneck,
            best_relu_h.max(best_c19_h),
            relu_str, c19_str, winner);
    }

    println!("\n  Key question: does C19 achieve 100% with smaller H?");
    println!("  If C19 wins at lower H → fewer params → smaller preprocessor → faster edge inference");
    println!("\n  Time: {:.1}s", t0.elapsed().as_secs_f64());
}
