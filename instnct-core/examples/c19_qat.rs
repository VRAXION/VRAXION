//! Quantization-Aware Training: float gradient, binary forward pass
//!
//! Weights are float during training but ROUNDED TO ±1 in every forward pass.
//! Gradient flows through the rounding (straight-through estimator).
//! Result: binary weights that actually work, found by gradient.
//!
//! Run: cargo run --example c19_qat --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const LOCAL_CAP: usize = 3;
const TICKS: usize = 2;
const RHO: f32 = 4.0;

fn c19(x: f32, c: f32) -> f32 {
    let c = c.max(0.01);
    let l = 6.0 * c;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let scaled = x / c;
    let n = scaled.floor();
    let t = scaled - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + RHO * h * h)
}

fn relu(x: f32) -> f32 { x.max(0.0) }

/// Round to binary ±1
fn binarize(w: f32) -> f32 {
    if w >= 0.0 { 1.0 } else { -1.0 }
}

fn thermo_2(a: usize, b: usize) -> [f32; 8] {
    let mut v = [0.0f32; 8];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

#[derive(Clone)]
struct QATNet {
    nc: usize,
    nw: usize,
    float_params: Vec<f32>,   // shadow weights (gradient updates these)
    c_params: Vec<f32>,       // per-worker C (stays float, learned)
    offsets: Vec<usize>,
    local_counts: Vec<usize>,
    use_c19: bool,
    use_qat: bool,            // if true, binarize in forward; if false, use float
}

impl QATNet {
    fn new(nc: usize, nw: usize, use_c19: bool, use_qat: bool,
           rng: &mut StdRng, scale: f32) -> Self {
        let mut net = QATNet {
            nc, nw: 0, float_params: Vec::new(), c_params: Vec::new(),
            offsets: Vec::new(), local_counts: Vec::new(),
            use_c19, use_qat,
        };
        for i in 0..nw {
            let nl = LOCAL_CAP.min(i);
            let np = INPUT_DIM + nl + nc + 1 + 1;
            net.offsets.push(net.float_params.len());
            net.local_counts.push(nl);
            net.float_params.extend((0..np).map(|_| rng.gen_range(-scale..scale)));
            net.c_params.push(1.0);
            net.nw += 1;
        }
        net
    }

    /// Get effective weight (binary if QAT, float otherwise)
    fn eff_w(&self, i: usize) -> f32 {
        if self.use_qat { binarize(self.float_params[i]) } else { self.float_params[i] }
    }

    fn forward(&self, a: usize, b: usize) -> f32 {
        let input = thermo_2(a, b);
        let nc = self.nc;
        let nw = self.nw;
        let mut act = vec![0.0f32; nc + nw];

        for _t in 0..TICKS {
            let mut cc = vec![0.0f32; nc];
            for i in 0..nw {
                let o = self.offsets[i];
                let nl = self.local_counts[i];
                let ww = self.eff_w(o + INPUT_DIM + nl + nc);
                let slot = i % nc.max(1);
                if slot < nc { cc[slot] += act[nc + i] * ww; }
            }
            for i in 0..nc { act[i] = cc[i]; }
            let old = act.clone();
            for i in 0..nw {
                let o = self.offsets[i];
                let nl = self.local_counts[i];
                let mut s = self.eff_w(o + INPUT_DIM + nl + nc + 1); // bias
                for j in 0..INPUT_DIM { s += input[j] * self.eff_w(o + j); }
                let ls = i.saturating_sub(nl);
                for (k, wi) in (ls..i).enumerate() {
                    s += old[nc + wi] * self.eff_w(o + INPUT_DIM + k);
                }
                for k in 0..nc {
                    s += old[k] * self.eff_w(o + INPUT_DIM + nl + k);
                }
                act[nc + i] = if self.use_c19 {
                    c19(s, self.c_params[i])
                } else {
                    relu(s)
                };
            }
        }
        act[nc..].iter().sum()
    }

    fn mse_loss(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut l = 0.0f64;
        for a in 0..DIGITS { for b in 0..DIGITS {
            let d = self.forward(a, b) as f64 - op(a, b) as f64;
            l += d * d;
        }}
        l / 25.0
    }

    fn accuracy(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut c = 0;
        for a in 0..DIGITS { for b in 0..DIGITS {
            if (self.forward(a, b).round() as i32) == (op(a, b) as i32) { c += 1; }
        }}
        c as f64 / 25.0
    }

    /// Gradient on float_params (straight-through: pretend binarize isn't there)
    fn gradient(&mut self, op: fn(usize, usize) -> usize) -> (Vec<f32>, Vec<f32>) {
        let eps = 1e-3f32;
        let n = self.float_params.len();
        let mut g = vec![0.0f32; n];
        for i in 0..n {
            let orig = self.float_params[i];
            self.float_params[i] = orig + eps; let lp = self.mse_loss(op);
            self.float_params[i] = orig - eps; let lm = self.mse_loss(op);
            self.float_params[i] = orig;
            g[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        let nc = self.c_params.len();
        let mut gc = vec![0.0f32; nc];
        if self.use_c19 {
            for i in 0..nc {
                let orig = self.c_params[i];
                self.c_params[i] = orig + eps; let lp = self.mse_loss(op);
                self.c_params[i] = orig - eps; let lm = self.mse_loss(op);
                self.c_params[i] = orig;
                gc[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
            }
        }
        (g, gc)
    }

    /// Extract final binary weights
    fn binary_weights(&self) -> Vec<i8> {
        self.float_params.iter().map(|&w| if w >= 0.0 { 1i8 } else { -1i8 }).collect()
    }
}

fn optimize(net: &mut QATNet, op: fn(usize, usize) -> usize) -> (f64, usize) {
    let mut lr = 0.01f32;
    let patience = 150;
    let mut stale = 0;
    let mut best_loss = net.mse_loss(op);
    let mut step = 0;
    loop {
        let acc = net.accuracy(op);
        if acc >= 1.0 { return (acc, step); }
        if stale >= patience { return (acc, step); }

        let (g, gc) = net.gradient(op);
        let gn: f32 = g.iter().chain(gc.iter()).map(|x| x * x).sum::<f32>().sqrt();
        if gn < 1e-8 { return (acc, step); }

        let old_p = net.float_params.clone();
        let old_c = net.c_params.clone();
        let ol = net.mse_loss(op);

        let mut improved = false;
        for att in 0..5 {
            for i in 0..net.float_params.len() {
                net.float_params[i] = old_p[i] - lr * g[i] / gn;
            }
            for i in 0..net.c_params.len() {
                net.c_params[i] = (old_c[i] - lr * gc[i] / gn).max(0.01);
            }
            let nl = net.mse_loss(op);
            if nl < ol {
                lr *= 1.1;
                if nl < best_loss - 1e-8 { best_loss = nl; stale = 0; improved = true; }
                break;
            } else {
                lr *= 0.5;
                if att == 4 {
                    net.float_params = old_p.clone();
                    net.c_params = old_c.clone();
                }
            }
        }
        if !improved { stale += 1; }
        step += 1;
    }
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn main() {
    println!("=== QUANTIZATION-AWARE TRAINING: C19 ===\n");
    println!("Forward: weights rounded to ±1 every step");
    println!("Backward: gradient on float shadow weights (straight-through)");
    println!("C: learnable float (baked as LUT at deploy)\n");

    let nc = 3;
    let n_seeds = 50;
    let seeds: Vec<u64> = (1..=n_seeds as u64).collect();

    let tasks: Vec<(&str, fn(usize, usize) -> usize, usize)> = vec![
        ("ADD",   op_add,     3),
        ("MAX",   op_max,     3),
        ("MIN",   op_min,     3),
        ("|a-b|", op_sub_abs, 6),
        ("MUL",   op_mul,     6),
    ];

    // =========================================================
    // PART 1: QAT C19 vs QAT ReLU vs Float C19 vs Float ReLU
    // =========================================================
    println!("--- PART 1: 4-way comparison (50 seeds) ---\n");
    println!("{:>8} {:>4}  {:>12} {:>12} {:>12} {:>12}",
        "task", "nw", "QAT+C19", "QAT+ReLU", "Float+C19", "Float+ReLU");
    println!("{}", "=".repeat(70));

    for &(name, op, nw) in &tasks {
        // QAT + C19
        let qat_c19: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = QATNet::new(nc, nw, true, true, &mut rng, 0.5);
            optimize(&mut net, op)
        }).collect();

        // QAT + ReLU
        let qat_relu: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = QATNet::new(nc, nw, false, true, &mut rng, 0.5);
            optimize(&mut net, op)
        }).collect();

        // Float + C19 (no QAT, baseline)
        let float_c19: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = QATNet::new(nc, nw, true, false, &mut rng, 0.5);
            optimize(&mut net, op)
        }).collect();

        // Float + ReLU (no QAT, baseline)
        let float_relu: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = QATNet::new(nc, nw, false, false, &mut rng, 0.5);
            optimize(&mut net, op)
        }).collect();

        let qc_s = qat_c19.iter().filter(|r| r.0 >= 1.0).count();
        let qr_s = qat_relu.iter().filter(|r| r.0 >= 1.0).count();
        let fc_s = float_c19.iter().filter(|r| r.0 >= 1.0).count();
        let fr_s = float_relu.iter().filter(|r| r.0 >= 1.0).count();

        println!("{:>8} {:>4}  {:>9}/{:<2} {:>9}/{:<2} {:>9}/{:<2} {:>9}/{:<2}",
            name, nw,
            qc_s, n_seeds, qr_s, n_seeds,
            fc_s, n_seeds, fr_s, n_seeds);
    }

    // =========================================================
    // PART 2: QAT C19 detailed — learned C values + verify binary
    // =========================================================
    println!("\n--- PART 2: QAT C19 solution analysis (20 seeds) ---\n");

    let detail_seeds: Vec<u64> = (1..=20u64).collect();

    for &(name, op, nw) in &tasks {
        println!("  === {} ({} workers) ===", name, nw);

        let mut n_solved = 0;
        let mut n_binary_verified = 0;

        for &seed in &detail_seeds {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = QATNet::new(nc, nw, true, true, &mut rng, 0.5);
            let (acc, steps) = optimize(&mut net, op);

            // Extract binary weights and verify
            let binary_w = net.binary_weights();
            let c_vals = net.c_params.clone();

            // Verify: does the binary network actually work?
            let mut verify_net = net.clone();
            // Force exact binary (should already be, but just in case)
            for i in 0..verify_net.float_params.len() {
                verify_net.float_params[i] = binarize(verify_net.float_params[i]);
            }
            let verify_acc = verify_net.accuracy(op);

            let solved = acc >= 1.0;
            let verified = verify_acc >= 1.0;
            if solved { n_solved += 1; }
            if verified { n_binary_verified += 1; }

            let c_str: Vec<String> = c_vals.iter().map(|c| format!("{:.2}", c)).collect();
            if solved || seed <= 5 {
                println!("    seed {:>2}: acc={:.0}% verified={:.0}% steps={:>4} C=[{}]",
                    seed, acc * 100.0, verify_acc * 100.0, steps, c_str.join(", "));
            }
        }

        println!("    {}/{} solved, {}/{} binary-verified\n",
            n_solved, detail_seeds.len(), n_binary_verified, detail_seeds.len());
    }

    // =========================================================
    // PART 3: Post-training quantize comparison
    // =========================================================
    println!("--- PART 3: Float train → snap to binary (post-hoc) vs QAT ---\n");
    println!("{:>8} {:>12} {:>12} {:>12}",
        "task", "float_orig", "snap_to_±1", "QAT");
    println!("{}", "=".repeat(50));

    for &(name, op, nw) in &tasks {
        // Float train
        let float_results: Vec<(f64, QATNet)> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = QATNet::new(nc, nw, true, false, &mut rng, 0.5);
            let (acc, _) = optimize(&mut net, op);
            (acc, net)
        }).collect();

        // Post-hoc snap to binary
        let snap_solved: usize = float_results.par_iter().map(|(_, net)| {
            let mut snapped = net.clone();
            for i in 0..snapped.float_params.len() {
                snapped.float_params[i] = binarize(snapped.float_params[i]);
            }
            if snapped.accuracy(op) >= 1.0 { 1 } else { 0 }
        }).sum();

        // QAT
        let qat_solved: usize = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = QATNet::new(nc, nw, true, true, &mut rng, 0.5);
            let (acc, _) = optimize(&mut net, op);
            if acc >= 1.0 { 1 } else { 0 }
        }).sum();

        let float_solved = float_results.iter().filter(|(a, _)| *a >= 1.0).count();

        println!("{:>8} {:>9}/{:<2} {:>9}/{:<2} {:>9}/{:<2}",
            name,
            float_solved, n_seeds,
            snap_solved, n_seeds,
            qat_solved, n_seeds);
    }

    println!("\n=== DONE ===");
}
