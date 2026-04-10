//! Float → Integer extraction
//!
//! Train float, examine converged weights, find integer equivalent.
//! If weights are [0.5, 0.5, -0.3, 1.0] maybe [1, 1, -1, 2] works?
//!
//! Run: cargo run --example c19_float_to_int --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

fn thermo_2(a: usize, b: usize) -> [f32; 8] {
    let mut v = [0.0f32; 8];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

#[derive(Clone)]
struct Net {
    nc: usize,
    nw: usize,
    params: Vec<f32>,
    c_params: Vec<f32>,
    rho_params: Vec<f32>,
    offsets: Vec<usize>,
    local_counts: Vec<usize>,
}

impl Net {
    fn new(nc: usize, nw: usize, rng: &mut StdRng, scale: f32) -> Self {
        let mut net = Net {
            nc, nw: 0, params: Vec::new(), c_params: Vec::new(), rho_params: Vec::new(),
            offsets: Vec::new(), local_counts: Vec::new(),
        };
        for i in 0..nw {
            let nl = LOCAL_CAP.min(i);
            let np = INPUT_DIM + nl + nc + 1 + 1;
            net.offsets.push(net.params.len());
            net.local_counts.push(nl);
            net.params.extend((0..np).map(|_| rng.gen_range(-scale..scale)));
            net.c_params.push(1.0);
            net.rho_params.push(4.0);
            net.nw += 1;
        }
        net
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
                let ww = self.params[o + INPUT_DIM + nl + nc];
                let slot = i % nc.max(1);
                if slot < nc { cc[slot] += act[nc + i] * ww; }
            }
            for i in 0..nc { act[i] = cc[i]; }
            let old = act.clone();
            for i in 0..nw {
                let o = self.offsets[i];
                let nl = self.local_counts[i];
                let mut s = self.params[o + INPUT_DIM + nl + nc + 1];
                for j in 0..INPUT_DIM { s += input[j] * self.params[o + j]; }
                let ls = i.saturating_sub(nl);
                for (k, wi) in (ls..i).enumerate() {
                    s += old[nc + wi] * self.params[o + INPUT_DIM + k];
                }
                for k in 0..nc { s += old[k] * self.params[o + INPUT_DIM + nl + k]; }
                act[nc + i] = c19_with_rho(s, self.c_params[i], self.rho_params[i]);
            }
        }
        act[nc..].iter().sum()
    }

    fn mse(&self, op: fn(usize, usize) -> usize) -> f64 {
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

    fn gradient(&mut self, op: fn(usize, usize) -> usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let eps = 1e-3f32;
        let n = self.params.len();
        let mut g = vec![0.0f32; n];
        for i in 0..n {
            let orig = self.params[i];
            self.params[i] = orig + eps; let lp = self.mse(op);
            self.params[i] = orig - eps; let lm = self.mse(op);
            self.params[i] = orig;
            g[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        let nw = self.c_params.len();
        let mut gc = vec![0.0f32; nw];
        let mut gr = vec![0.0f32; nw];
        for i in 0..nw {
            let orig = self.c_params[i];
            self.c_params[i] = orig + eps; let lp = self.mse(op);
            self.c_params[i] = orig - eps; let lm = self.mse(op);
            self.c_params[i] = orig;
            gc[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
            let orig = self.rho_params[i];
            self.rho_params[i] = orig + eps; let lp = self.mse(op);
            self.rho_params[i] = orig - eps; let lm = self.mse(op);
            self.rho_params[i] = orig;
            gr[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        (g, gc, gr)
    }

    /// Get per-worker weight slices
    fn worker_weights(&self, i: usize) -> &[f32] {
        let o = self.offsets[i];
        let nl = self.local_counts[i];
        let np = INPUT_DIM + nl + self.nc + 1 + 1;
        &self.params[o..o+np]
    }
}

fn c19_with_rho(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01);
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

// Patch forward to use rho
impl Net {
    fn forward_rho(&self, a: usize, b: usize) -> f32 {
        let input = thermo_2(a, b);
        let nc = self.nc;
        let nw = self.nw;
        let mut act = vec![0.0f32; nc + nw];
        for _t in 0..TICKS {
            let mut cc = vec![0.0f32; nc];
            for i in 0..nw {
                let o = self.offsets[i];
                let nl = self.local_counts[i];
                let ww = self.params[o + INPUT_DIM + nl + nc];
                let slot = i % nc.max(1);
                if slot < nc { cc[slot] += act[nc + i] * ww; }
            }
            for i in 0..nc { act[i] = cc[i]; }
            let old = act.clone();
            for i in 0..nw {
                let o = self.offsets[i];
                let nl = self.local_counts[i];
                let mut s = self.params[o + INPUT_DIM + nl + nc + 1];
                for j in 0..INPUT_DIM { s += input[j] * self.params[o + j]; }
                let ls = i.saturating_sub(nl);
                for (k, wi) in (ls..i).enumerate() {
                    s += old[nc + wi] * self.params[o + INPUT_DIM + k];
                }
                for k in 0..nc { s += old[k] * self.params[o + INPUT_DIM + nl + k]; }
                act[nc + i] = c19_with_rho(s, self.c_params[i], self.rho_params[i]);
            }
        }
        act[nc..].iter().sum()
    }
}

fn optimize(net: &mut Net, op: fn(usize, usize) -> usize) -> (f64, usize) {
    let mut lr = 0.01f32;
    let patience = 100;
    let mut stale = 0;
    let mut best_loss = net.mse(op);
    let mut step = 0;
    loop {
        let acc = net.accuracy(op);
        if acc >= 1.0 { return (acc, step); }
        if stale >= patience { return (acc, step); }
        let (g, gc, gr) = net.gradient(op);
        let gn: f32 = g.iter().chain(gc.iter()).chain(gr.iter()).map(|x| x*x).sum::<f32>().sqrt();
        if gn < 1e-8 { return (acc, step); }
        let old_p = net.params.clone();
        let old_c = net.c_params.clone();
        let old_r = net.rho_params.clone();
        let ol = net.mse(op);
        let mut improved = false;
        for att in 0..5 {
            for i in 0..net.params.len() { net.params[i] = old_p[i] - lr * g[i] / gn; }
            for i in 0..net.c_params.len() { net.c_params[i] = (old_c[i] - lr * gc[i] / gn).max(0.01); }
            for i in 0..net.rho_params.len() { net.rho_params[i] = (old_r[i] - lr * gr[i] / gn).max(0.0); }
            let nl = net.mse(op);
            if nl < ol { lr *= 1.1; if nl < best_loss - 1e-8 { best_loss = nl; stale = 0; improved = true; } break; }
            else { lr *= 0.5; if att == 4 { net.params = old_p.clone(); net.c_params = old_c.clone(); net.rho_params = old_r.clone(); } }
        }
        if !improved { stale += 1; }
        step += 1;
    }
}

/// Try to find integer scaling for a weight vector
fn find_int_scaling(weights: &[f32], max_int: i32) -> Vec<(f32, Vec<i32>, f64)> {
    let mut results = Vec::new();
    // Try different scale factors
    for scale_x10 in 1..=200 {
        let scale = scale_x10 as f32 * 0.1;
        let int_weights: Vec<i32> = weights.iter().map(|&w| {
            (w * scale).round() as i32
        }).collect();
        // Check if all within range
        if int_weights.iter().all(|&w| w.abs() <= max_int) {
            // Compute error
            let err: f64 = weights.iter().zip(int_weights.iter())
                .map(|(&w, &iw)| (w - iw as f32 / scale) as f64)
                .map(|e| e * e).sum::<f64>() / weights.len() as f64;
            results.push((scale, int_weights, err));
        }
    }
    results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    results.truncate(5);
    results
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn main() {
    println!("=== FLOAT → INTEGER EXTRACTION ===\n");
    println!("Train float, examine weights, find integer equivalent\n");

    let nc = 3;

    let tasks: Vec<(&str, fn(usize, usize) -> usize, usize)> = vec![
        ("ADD",   op_add,     3),
        ("MAX",   op_max,     3),
        ("MIN",   op_min,     3),
        ("|a-b|", op_sub_abs, 6),
        ("MUL",   op_mul,     6),
    ];

    for &(name, op, nw) in &tasks {
        println!("=== {} ({} workers) ===\n", name, nw);

        // Train 5 seeds, show weight analysis
        for seed in 1..=5u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = Net::new(nc, nw, &mut rng, 0.5);
            let (acc, steps) = optimize(&mut net, op);

            println!("  Seed {}: acc={:.0}% steps={}", seed, acc * 100.0, steps);
            println!("    C=[{}]  rho=[{}]",
                net.c_params.iter().map(|c| format!("{:.3}", c)).collect::<Vec<_>>().join(", "),
                net.rho_params.iter().map(|r| format!("{:.2}", r)).collect::<Vec<_>>().join(", "));

            for w in 0..nw {
                let weights = net.worker_weights(w);
                println!("    Worker {} float: [{}]", w,
                    weights.iter().map(|v| format!("{:+.3}", v)).collect::<Vec<_>>().join(", "));

                // Try integer scalings
                let scalings = find_int_scaling(weights, 3);
                if let Some(best) = scalings.first() {
                    println!("    Worker {} int (×{:.1}): [{}]  err={:.6}", w, best.0,
                        best.1.iter().map(|v| format!("{:+}", v)).collect::<Vec<_>>().join(","),
                        best.2);

                    // Test: does the integer version work?
                    let mut test_net = net.clone();
                    let o = test_net.offsets[w];
                    let nl = test_net.local_counts[w];
                    let np = INPUT_DIM + nl + nc + 1 + 1;
                    for i in 0..np {
                        test_net.params[o + i] = best.1[i] as f32 / best.0;
                    }
                    let int_acc = test_net.accuracy(op);
                    println!("    Worker {} int accuracy: {:.0}%", w, int_acc * 100.0);
                }
            }

            // Try quantizing ALL workers at once to int
            println!("\n    --- Full network integer quantization ---");
            for &max_int in &[1, 2, 3] {
                let mut qnet = net.clone();
                // Find best global scale
                let mut best_scale = 1.0f32;
                let mut best_acc = 0.0f64;
                for sx10 in 1..=100 {
                    let scale = sx10 as f32 * 0.1;
                    let mut test = net.clone();
                    for i in 0..test.params.len() {
                        let q = (test.params[i] * scale).round().max(-max_int as f32).min(max_int as f32);
                        test.params[i] = q / scale;
                    }
                    let a = test.accuracy(op);
                    if a > best_acc { best_acc = a; best_scale = scale; }
                }
                // Apply best scale
                for i in 0..qnet.params.len() {
                    let q = (qnet.params[i] * best_scale).round().max(-max_int as f32).min(max_int as f32);
                    qnet.params[i] = q / best_scale;
                }
                let qacc = qnet.accuracy(op);
                println!("    int±{} (scale={:.1}): {:.0}%", max_int, best_scale, qacc * 100.0);
            }
            println!();
        }
    }

    println!("=== DONE ===");
}
