//! Iterative quantization: float train → quantize one → retrain rest → repeat
//!
//! 1. Float train all → 100%
//! 2. Quantize neuron 0 (fraction extract, freeze)
//! 3. Retrain remaining float neurons to compensate
//! 4. Quantize neuron 1, retrain rest
//! 5. ... until all quantized
//!
//! Run: cargo run --example iterative_quantize --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const LOCAL_CAP: usize = 3;
const TICKS: usize = 2;

fn c19(x: f32, c: f32, rho: f32) -> f32 {
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

fn thermo_2(a: usize, b: usize) -> [f32; 8] {
    let mut v = [0.0f32; 8];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

#[derive(Clone)]
struct Net {
    nc: usize, nw: usize,
    params: Vec<f32>, c_params: Vec<f32>, rho_params: Vec<f32>,
    offsets: Vec<usize>, local_counts: Vec<usize>,
    frozen: Vec<bool>,  // per-worker: is this neuron quantized/frozen?
}

impl Net {
    fn new(nc: usize, nw: usize, rng: &mut StdRng, scale: f32) -> Self {
        let mut net = Net {
            nc, nw: 0, params: Vec::new(), c_params: Vec::new(), rho_params: Vec::new(),
            offsets: Vec::new(), local_counts: Vec::new(), frozen: Vec::new(),
        };
        for i in 0..nw {
            let nl = LOCAL_CAP.min(i);
            let np = INPUT_DIM + nl + nc + 1 + 1;
            net.offsets.push(net.params.len());
            net.local_counts.push(nl);
            net.params.extend((0..np).map(|_| rng.gen_range(-scale..scale)));
            net.c_params.push(1.0);
            net.rho_params.push(4.0);
            net.frozen.push(false);
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
                act[nc + i] = c19(s, self.c_params[i], self.rho_params[i]);
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

    /// Gradient only on NON-frozen params
    fn gradient_unfrozen(&mut self, op: fn(usize, usize) -> usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let eps = 1e-3f32;
        let n = self.params.len();
        let mut g = vec![0.0f32; n];

        for i in 0..self.nw {
            if self.frozen[i] { continue; } // skip frozen
            let o = self.offsets[i];
            let nl = self.local_counts[i];
            let np = INPUT_DIM + nl + self.nc + 1 + 1;
            for p in 0..np {
                let idx = o + p;
                let orig = self.params[idx];
                self.params[idx] = orig + eps; let lp = self.mse(op);
                self.params[idx] = orig - eps; let lm = self.mse(op);
                self.params[idx] = orig;
                g[idx] = ((lp - lm) / (2.0 * eps as f64)) as f32;
            }
        }

        let nw = self.nw;
        let mut gc = vec![0.0f32; nw];
        let mut gr = vec![0.0f32; nw];
        for i in 0..nw {
            if self.frozen[i] { continue; }
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

    /// Quantize one worker: fraction extract → snap params to integer grid
    fn quantize_worker(&mut self, worker_idx: usize) {
        let o = self.offsets[worker_idx];
        let nl = self.local_counts[worker_idx];
        let np = INPUT_DIM + nl + self.nc + 1 + 1;
        let slice = &self.params[o..o+np];

        // Find best denominator for this worker's params
        let mut best_d = 1u32;
        let mut best_err = f64::MAX;
        for d in 1..=30u32 {
            let err: f64 = slice.iter()
                .map(|&w| { let q = (w * d as f32).round() / d as f32; (w - q) as f64 })
                .map(|e| e * e).sum::<f64>() / np as f64;
            if err < best_err { best_err = err; best_d = d; }
        }

        // Snap to integer grid
        for p in 0..np {
            let idx = o + p;
            self.params[idx] = (self.params[idx] * best_d as f32).round() / best_d as f32;
        }
        self.frozen[worker_idx] = true;
    }
}

fn optimize_unfrozen(net: &mut Net, op: fn(usize, usize) -> usize, max_steps: usize) -> (f64, usize) {
    let mut lr = 0.01f32;
    let patience = 100;
    let mut stale = 0;
    let mut best_loss = net.mse(op);
    for step in 0..max_steps {
        let acc = net.accuracy(op);
        if acc >= 1.0 { return (acc, step); }
        if stale >= patience { return (acc, step); }

        let (g, gc, gr) = net.gradient_unfrozen(op);
        let gn: f32 = g.iter().chain(gc.iter()).chain(gr.iter()).map(|x| x*x).sum::<f32>().sqrt();
        if gn < 1e-8 { return (acc, step); }

        let old_p = net.params.clone();
        let old_c = net.c_params.clone();
        let old_r = net.rho_params.clone();
        let ol = net.mse(op);
        let mut improved = false;
        for att in 0..5 {
            for i in 0..net.params.len() {
                // Only update unfrozen
                let worker = net.offsets.iter().enumerate()
                    .rev().find(|(_, &o)| o <= i).map(|(w, _)| w).unwrap_or(0);
                if !net.frozen[worker] {
                    net.params[i] = old_p[i] - lr * g[i] / gn;
                }
            }
            for i in 0..net.nw {
                if !net.frozen[i] {
                    net.c_params[i] = (old_c[i] - lr * gc[i] / gn).max(0.01);
                    net.rho_params[i] = (old_r[i] - lr * gr[i] / gn).max(0.0);
                }
            }
            let nl = net.mse(op);
            if nl < ol { lr *= 1.1; if nl < best_loss - 1e-8 { best_loss = nl; stale = 0; improved = true; } break; }
            else {
                lr *= 0.5;
                if att == 4 {
                    for i in 0..net.params.len() { net.params[i] = old_p[i]; }
                    net.c_params = old_c.clone();
                    net.rho_params = old_r.clone();
                }
            }
        }
        if !improved { stale += 1; }
    }
    (net.accuracy(op), max_steps)
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn main() {
    println!("=== ITERATIVE QUANTIZATION ===\n");
    println!("Float train → quantize one neuron → retrain rest → repeat\n");

    let nc = 3;
    let n_seeds = 20;
    let seeds: Vec<u64> = (1..=n_seeds as u64).collect();

    let tasks: Vec<(&str, fn(usize, usize) -> usize, usize)> = vec![
        ("ADD",   op_add,     3),
        ("MAX",   op_max,     3),
        ("MIN",   op_min,     3),
        ("|a-b|", op_sub_abs, 6),
        ("MUL",   op_mul,     6),
    ];

    println!("{:>8} {:>4}  {:>10} {:>10} {:>10} {:>10}",
        "task", "nw", "float_100%", "all_quant", "iter_quant", "1shot_q");
    println!("{}", "=".repeat(60));

    for &(name, op, nw) in &tasks {
        let results: Vec<(bool, bool, bool)> = seeds.par_iter().map(|&seed| {
            // === Method A: Float train (baseline) ===
            let mut rng_a = StdRng::seed_from_u64(seed);
            let mut net_a = Net::new(nc, nw, &mut rng_a, 0.5);
            let (float_acc, _) = optimize_unfrozen(&mut net_a, op, 5000);
            let float_ok = float_acc >= 1.0;

            // === Method B: Iterative quantize ===
            let mut net_iter = net_a.clone(); // start from converged float
            let mut iter_ok = float_ok;

            if float_ok {
                for qi in 0..nw {
                    // Quantize worker qi
                    net_iter.quantize_worker(qi);
                    let after_q = net_iter.accuracy(op);

                    if after_q < 1.0 && qi < nw - 1 {
                        // Retrain unfrozen workers to compensate
                        let (retrain_acc, _) = optimize_unfrozen(&mut net_iter, op, 2000);
                        if retrain_acc < 1.0 {
                            // Try harder
                            optimize_unfrozen(&mut net_iter, op, 3000);
                        }
                    }
                }
                iter_ok = net_iter.accuracy(op) >= 1.0;
            }

            // === Method C: One-shot quantize all at once ===
            let mut net_1shot = net_a.clone();
            let mut oneshot_ok = false;
            if float_ok {
                for qi in 0..nw {
                    net_1shot.quantize_worker(qi);
                }
                oneshot_ok = net_1shot.accuracy(op) >= 1.0;
            }

            (float_ok, iter_ok, oneshot_ok)
        }).collect();

        let float_solved = results.iter().filter(|r| r.0).count();
        let iter_solved = results.iter().filter(|r| r.1).count();
        let oneshot_solved = results.iter().filter(|r| r.2).count();

        println!("{:>8} {:>4}  {:>7}/{:<2} {:>10} {:>7}/{:<2} {:>7}/{:<2}",
            name, nw,
            float_solved, n_seeds,
            "",
            iter_solved, float_solved,
            oneshot_solved, float_solved);
    }

    // Detailed per-step view for ADD
    println!("\n--- ADD detailed (seed 1): step-by-step ---\n");
    let mut rng = StdRng::seed_from_u64(1);
    let mut net = Net::new(nc, 3, &mut rng, 0.5);
    let (acc, steps) = optimize_unfrozen(&mut net, op_add, 5000);
    println!("  Float train: {:.0}% ({} steps)", acc * 100.0, steps);

    for qi in 0..3 {
        let before = net.accuracy(op_add);
        net.quantize_worker(qi);
        let after_q = net.accuracy(op_add);
        println!("  Quantize worker {}: {:.0}% → {:.0}%", qi, before * 100.0, after_q * 100.0);

        if after_q < 1.0 && qi < 2 {
            let (retrain_acc, retrain_steps) = optimize_unfrozen(&mut net, op_add, 3000);
            println!("  Retrain unfrozen: → {:.0}% ({} steps)", retrain_acc * 100.0, retrain_steps);
        }
    }
    let final_acc = net.accuracy(op_add);
    let all_frozen = net.frozen.iter().all(|&f| f);
    println!("\n  Final: {:.0}%, all_frozen={}", final_acc * 100.0, all_frozen);

    // Show the integer weights
    for i in 0..3 {
        let o = net.offsets[i];
        let nl = net.local_counts[i];
        let np = INPUT_DIM + nl + nc + 1 + 1;
        let weights = &net.params[o..o+np];
        // Find denom
        let mut best_d = 1u32;
        let mut best_err = f64::MAX;
        for d in 1..=30u32 {
            let err: f64 = weights.iter()
                .map(|&w| { let q = (w * d as f32).round() / d as f32; (w - q) as f64 })
                .map(|e| e * e).sum::<f64>();
            if err < best_err { best_err = err; best_d = d; }
        }
        let ints: Vec<i32> = weights.iter().map(|&w| (w * best_d as f32).round() as i32).collect();
        println!("  Worker {}: denom={} ints=[{}]", i, best_d,
            ints.iter().map(|v| format!("{:+}", v)).collect::<Vec<_>>().join(","));
    }

    println!("\n=== DONE ===");
}
