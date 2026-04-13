//! Minimum parameter sweep: 8→2 bit C19 autoencoder
//! Find the absolute smallest H that still achieves 100% round-trip.
//! Then try to minimize further with different strategies.
//!
//! Run: cargo run --example min_params_sweep --release

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

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0 * c;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let scaled = x / c; let n = scaled.floor(); let t = scaled - n;
    let h = t * (1.0 - t); let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

fn c19_dx(x: f32, c: f32, rho: f32) -> f32 { let e = 0.001; (c19(x+e,c,rho) - c19(x-e,c,rho)) / (2.0*e) }
fn c19_dc(x: f32, c: f32, rho: f32) -> f32 { let e = 0.001; (c19(x,c+e,rho) - c19(x,c-e,rho)) / (2.0*e) }
fn c19_drho(x: f32, c: f32, rho: f32) -> f32 { let e = 0.001; (c19(x,c,rho+e) - c19(x,c,rho-e)) / (2.0*e) }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

#[derive(Clone)]
struct Mlp {
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    b3: Vec<f32>, b4: Vec<f32>,
    c_enc: Vec<f32>, rho_enc: Vec<f32>,
    c_dec: Vec<f32>, rho_dec: Vec<f32>,
    h: usize, bo: usize,
}

impl Mlp {
    fn new(h: usize, bo: usize, rng: &mut Rng) -> Self {
        let s1 = (2.0 / 8.0f32).sqrt(); let s2 = (2.0 / h as f32).sqrt();
        Mlp {
            w1: (0..h).map(|_| (0..8).map(|_| rng.normal() * s1).collect()).collect(),
            b1: vec![0.0; h],
            w2: (0..bo).map(|_| (0..h).map(|_| rng.normal() * s2).collect()).collect(),
            b2: vec![0.0; bo], b3: vec![0.0; h], b4: vec![0.0; 8],
            c_enc: vec![3.0; h], rho_enc: vec![1.0; h],
            c_dec: vec![3.0; h], rho_dec: vec![1.0; h],
            h, bo,
        }
    }

    fn train_step(&mut self, inp: &[f32; 8], lr: f32) -> f32 {
        // Forward
        let mut z1 = vec![0.0f32; self.h]; let mut a1 = vec![0.0f32; self.h];
        for i in 0..self.h { z1[i] = self.b1[i]; for j in 0..8 { z1[i] += self.w1[i][j]*inp[j]; } a1[i] = c19(z1[i], self.c_enc[i], self.rho_enc[i]); }
        let mut z2 = vec![0.0f32; self.bo]; let mut a2 = vec![0.0f32; self.bo];
        for i in 0..self.bo { z2[i] = self.b2[i]; for j in 0..self.h { z2[i] += self.w2[i][j]*a1[j]; } a2[i] = sigmoid(z2[i]); }
        let mut z3 = vec![0.0f32; self.h]; let mut a3 = vec![0.0f32; self.h];
        for i in 0..self.h { z3[i] = self.b3[i]; for j in 0..self.bo { z3[i] += self.w2[j][i]*a2[j]; } a3[i] = c19(z3[i], self.c_dec[i], self.rho_dec[i]); }
        let mut a4 = [0.0f32; 8]; let mut z4 = [0.0f32; 8];
        for i in 0..8 { z4[i] = self.b4[i]; for j in 0..self.h { z4[i] += self.w1[j][i]*a3[j]; } a4[i] = sigmoid(z4[i]); }

        let mut loss = 0.0f32; let mut d4 = [0.0f32; 8];
        for i in 0..8 { let e = a4[i]-inp[i]; loss += e*e; d4[i] = 2.0*e*a4[i]*(1.0-a4[i]); }

        let mut da3 = vec![0.0f32; self.h];
        for j in 0..self.h { for i in 0..8 { da3[j] += d4[i]*self.w1[j][i]; self.w1[j][i] -= lr*d4[i]*a3[j]; } }
        for i in 0..8 { self.b4[i] -= lr*d4[i]; }

        let mut d3 = vec![0.0f32; self.h];
        for i in 0..self.h {
            let dx = c19_dx(z3[i], self.c_dec[i], self.rho_dec[i]); d3[i] = da3[i]*dx;
            self.c_dec[i] -= lr * da3[i] * c19_dc(z3[i], self.c_dec[i], self.rho_dec[i]);
            self.rho_dec[i] -= lr * da3[i] * c19_drho(z3[i], self.c_dec[i], self.rho_dec[i]);
            self.c_dec[i] = self.c_dec[i].max(0.1).min(20.0); self.rho_dec[i] = self.rho_dec[i].max(0.0).min(10.0);
        }

        let mut da2 = vec![0.0f32; self.bo];
        for j in 0..self.bo { for i in 0..self.h { da2[j] += d3[i]*self.w2[j][i]; self.w2[j][i] -= lr*d3[i]*a2[j]; } }
        for i in 0..self.h { self.b3[i] -= lr*d3[i]; }

        let mut d2 = vec![0.0f32; self.bo]; for i in 0..self.bo { d2[i] = da2[i]*a2[i]*(1.0-a2[i]); }

        let mut da1 = vec![0.0f32; self.h];
        for i in 0..self.bo { for j in 0..self.h { da1[j] += d2[i]*self.w2[i][j]; self.w2[i][j] -= lr*d2[i]*a1[j]; } self.b2[i] -= lr*d2[i]; }

        let mut d1 = vec![0.0f32; self.h];
        for i in 0..self.h {
            let dx = c19_dx(z1[i], self.c_enc[i], self.rho_enc[i]); d1[i] = da1[i]*dx;
            self.c_enc[i] -= lr * da1[i] * c19_dc(z1[i], self.c_enc[i], self.rho_enc[i]);
            self.rho_enc[i] -= lr * da1[i] * c19_drho(z1[i], self.c_enc[i], self.rho_enc[i]);
            self.c_enc[i] = self.c_enc[i].max(0.1).min(20.0); self.rho_enc[i] = self.rho_enc[i].max(0.0).min(10.0);
        }

        for i in 0..self.h { for j in 0..8 { self.w1[i][j] -= lr*d1[i]*inp[j]; } self.b1[i] -= lr*d1[i]; }
        loss
    }

    fn accuracy(&self, bytes: &[u8]) -> usize {
        let mut ok = 0;
        for &b in bytes {
            let inp = byte_to_bits(b);
            let mut a1 = vec![0.0f32; self.h];
            for i in 0..self.h { let mut s = self.b1[i]; for j in 0..8 { s += self.w1[i][j]*inp[j]; } a1[i] = c19(s, self.c_enc[i], self.rho_enc[i]); }
            let mut a2 = vec![0.0f32; self.bo];
            for i in 0..self.bo { let mut s = self.b2[i]; for j in 0..self.h { s += self.w2[i][j]*a1[j]; } a2[i] = sigmoid(s); }
            let mut a3 = vec![0.0f32; self.h];
            for i in 0..self.h { let mut s = self.b3[i]; for j in 0..self.bo { s += self.w2[j][i]*a2[j]; } a3[i] = c19(s, self.c_dec[i], self.rho_dec[i]); }
            let mut a4 = [0.0f32; 8];
            for i in 0..8 { let mut s = self.b4[i]; for j in 0..self.h { s += self.w1[j][i]*a3[j]; } a4[i] = sigmoid(s); }
            if (0..8).all(|i| (a4[i] - inp[i]).abs() < 0.4) { ok += 1; }
        }
        ok
    }

    fn encoder_params(&self) -> usize {
        // Only encoder: W1 + b1 + W2 + b2 + c_enc + rho_enc
        self.h * 8 + self.h + self.bo * self.h + self.bo + self.h + self.h
    }

    fn total_params(&self) -> usize {
        // Full autoencoder (for reference)
        self.encoder_params() + self.h + 8 + self.h + self.h // b3 + b4 + c_dec + rho_dec
    }
}

fn main() {
    let t0 = Instant::now();
    let unique = load_unique_bytes("instnct-core/tests/fixtures/alice_corpus.txt");
    let train: Vec<[f32; 8]> = unique.iter().map(|&b| byte_to_bits(b)).collect();

    println!("=== MINIMUM PARAMETER SWEEP: C19 8→2 bit ===");
    println!("{} unique bytes, C19 activation, tied weights, backprop", unique.len());
    println!("Goal: smallest H that achieves 100% round-trip at B=2\n");

    // Fine-grained H sweep for B=2
    println!("━━━ Phase 1: H sweep for B=2 (C19) ━━━\n");
    println!("{:>4} {:>8} {:>10} {:>12} {:>12} {:>12}",
        "H", "best_acc", "pct", "enc_params", "enc_bytes", "seeds_tried");
    println!("{}", "─".repeat(65));

    for h in 2..=24 {
        let mut best_acc = 0usize;
        let mut best_mlp: Option<Mlp> = None;
        let n_seeds = if h <= 6 { 50 } else if h <= 12 { 30 } else { 15 };

        for seed in 0..n_seeds {
            let mut rng = Rng::new(seed as u64 * 1000 + h as u64 * 31 + 7);
            let mut mlp = Mlp::new(h, 2, &mut rng);
            let epochs = if h <= 6 { 15000 } else { 10000 };
            for epoch in 0..epochs {
                let lr = 0.03 * (1.0 - epoch as f32 / epochs as f32 * 0.9);
                for inp in &train { mlp.train_step(inp, lr); }
            }
            let acc = mlp.accuracy(&unique);
            if acc > best_acc { best_acc = acc; best_mlp = Some(mlp); }
            if best_acc == unique.len() { break; }
        }

        let enc_p = if let Some(ref m) = best_mlp { m.encoder_params() } else { 0 };
        let marker = if best_acc == unique.len() { " ★" } else { "" };

        println!("{:>4} {:>5}/{:<2} {:>9.1}% {:>12} {:>10}B {:>12}{}",
            h, best_acc, unique.len(),
            best_acc as f64 / unique.len() as f64 * 100.0,
            enc_p, enc_p, n_seeds, marker);

        // Show details for perfect solutions
        if best_acc == unique.len() {
            if let Some(ref mlp) = best_mlp {
                // Check int8 quantized accuracy
                let q_acc = eval_int8(mlp, &unique);
                let q5_acc = eval_intN(mlp, &unique, 5);
                let q4_acc = eval_intN(mlp, &unique, 4);
                println!("      → int8: {}/{}  int5: {}/{}  int4: {}/{}",
                    q_acc, unique.len(), q5_acc, unique.len(), q4_acc, unique.len());

                // Show learned c,rho
                let cs: Vec<String> = mlp.c_enc.iter().map(|c| format!("{:.1}", c)).collect();
                let rs: Vec<String> = mlp.rho_enc.iter().map(|r| format!("{:.1}", r)).collect();
                println!("      → c_enc: [{}]", cs.join(", "));
                println!("      → rho_enc: [{}]", rs.join(", "));
            }
        }
    }

    // Also sweep B=3 and B=4 for comparison
    println!("\n━━━ Phase 2: Minimum H for B=2,3,4 (C19) ━━━\n");
    println!("{:>4} {:>6} {:>10} {:>12} {:>12}",
        "B", "min_H", "enc_params", "enc_bytes", "status");
    println!("{}", "─".repeat(50));

    for &bo in &[2, 3, 4] {
        let mut found_h = 0;
        let mut found_params = 0;
        for h in 2..=32 {
            let n_seeds = if h <= 6 { 50 } else if h <= 12 { 30 } else { 15 };
            let mut found = false;
            for seed in 0..n_seeds {
                let mut rng = Rng::new(seed as u64 * 1000 + h as u64 * 31 + bo as u64 * 97);
                let mut mlp = Mlp::new(h, bo, &mut rng);
                let epochs = if h <= 6 { 15000 } else { 10000 };
                for epoch in 0..epochs {
                    let lr = 0.03 * (1.0 - epoch as f32 / epochs as f32 * 0.9);
                    for inp in &train { mlp.train_step(inp, lr); }
                }
                if mlp.accuracy(&unique) == unique.len() {
                    found_h = h; found_params = mlp.encoder_params(); found = true; break;
                }
            }
            if found { break; }
        }
        if found_h > 0 {
            println!("{:>4} {:>6} {:>12} {:>10}B {:>12}",
                bo, found_h, found_params, found_params, "★ PERFECT");
        } else {
            println!("{:>4} {:>6} {:>12} {:>10} {:>12}", bo, ">32", "—", "—", "NOT FOUND");
        }
    }

    println!("\n━━━ SUMMARY ━━━");
    println!("  C19 tied-weight autoencoder, 29 unique bytes");
    println!("  Encoder-only deploy (decoder discarded after validation)");
    println!("  Time: {:.1}s", t0.elapsed().as_secs_f64());
}

// ══════════════════════════════════════════════════════
// INT8 eval
// ══════════════════════════════════════════════════════
fn eval_int8(mlp: &Mlp, bytes: &[u8]) -> usize { eval_intN(mlp, bytes, 8) }

fn eval_intN(mlp: &Mlp, bytes: &[u8], bits: u32) -> usize {
    let max_int = if bits == 1 { 1i32 } else { (1i32 << (bits - 1)) - 1 };
    let scale = |v: &[Vec<f32>]| -> f32 { v.iter().flat_map(|r| r.iter()).map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-7) / max_int as f32 };
    let scalev = |v: &[f32]| -> f32 { v.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-7) / max_int as f32 };
    let q = |v: f32, s: f32| -> f32 { (v/s).round().max(-max_int as f32).min(max_int as f32) * s };

    let sw1 = scale(&mlp.w1); let sw2 = scale(&mlp.w2);
    let sb1 = scalev(&mlp.b1); let sb2 = scalev(&mlp.b2);
    let sb3 = scalev(&mlp.b3); let sb4 = scalev(&mlp.b4);

    let mut ok = 0;
    for &b in bytes {
        let inp = byte_to_bits(b);
        let mut a1 = vec![0.0f32; mlp.h];
        for i in 0..mlp.h { let mut s = q(mlp.b1[i],sb1); for j in 0..8 { s += q(mlp.w1[i][j],sw1)*inp[j]; } a1[i] = c19(s, mlp.c_enc[i], mlp.rho_enc[i]); }
        let mut a2 = vec![0.0f32; mlp.bo];
        for i in 0..mlp.bo { let mut s = q(mlp.b2[i],sb2); for j in 0..mlp.h { s += q(mlp.w2[i][j],sw2)*a1[j]; } a2[i] = sigmoid(s); }
        let mut a3 = vec![0.0f32; mlp.h];
        for i in 0..mlp.h { let mut s = q(mlp.b3[i],sb3); for j in 0..mlp.bo { s += q(mlp.w2[j][i],sw2)*a2[j]; } a3[i] = c19(s, mlp.c_dec[i], mlp.rho_dec[i]); }
        let mut a4 = [0.0f32; 8];
        for i in 0..8 { let mut s = q(mlp.b4[i],sb4); for j in 0..mlp.h { s += q(mlp.w1[j][i],sw1)*a3[j]; } a4[i] = sigmoid(s); }
        if (0..8).all(|i| (a4[i]-inp[i]).abs() < 0.4) { ok += 1; }
    }
    ok
}
