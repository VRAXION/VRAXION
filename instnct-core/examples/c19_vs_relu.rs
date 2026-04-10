//! C19 vs ReLU activation comparison in the connectome gradient pipeline
//!
//! C19: periodic parabolic wave with learnable C per-neuron (rho=4.0 fixed)
//!   core(x, C) = C * (sgn * h + rho * h^2)
//!   where scaled = x / C, n = floor(scaled), t = frac(scaled),
//!         h = t*(1-t), sgn = (-1)^n
//!   Outside [-6C, 6C]: linear pass-through (x - 6C or x + 6C)
//!
//! ReLU: max(0, x)
//!
//! Both trained with MSE gradient descent, native output.
//!
//! Run: cargo run --example c19_vs_relu --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const LOCAL_CAP: usize = 3;
const TICKS: usize = 2;
const RHO: f32 = 4.0;

fn relu(x: f32) -> f32 { x.max(0.0) }

fn c19(x: f32, c: f32) -> f32 {
    let c = c.max(0.1); // clamp C to avoid division by zero
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

fn thermo_2(a: usize, b: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; INPUT_DIM];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

#[derive(Clone)]
struct FlatNet {
    n_connectome: usize,
    n_workers: usize,
    params: Vec<f32>,        // network weights + biases
    c_params: Vec<f32>,      // per-worker C parameter for c19
    worker_param_offsets: Vec<usize>,
    worker_local_counts: Vec<usize>,
    use_c19: bool,
}

impl FlatNet {
    fn new_random(nc: usize, nw: usize, use_c19: bool, rng: &mut StdRng, scale: f32) -> Self {
        let mut net = FlatNet {
            n_connectome: nc, n_workers: 0,
            params: Vec::new(), c_params: Vec::new(),
            worker_param_offsets: Vec::new(),
            worker_local_counts: Vec::new(),
            use_c19,
        };
        for i in 0..nw {
            let nl = LOCAL_CAP.min(i);
            let np = INPUT_DIM + nl + nc + 1 + 1; // inputs + local + connectome + write_w + bias
            let init: Vec<f32> = (0..np).map(|_| rng.gen_range(-scale..scale)).collect();
            net.worker_param_offsets.push(net.params.len());
            net.worker_local_counts.push(nl);
            net.params.extend_from_slice(&init);
            net.c_params.push(1.0); // init C=1.0
            net.n_workers += 1;
        }
        net
    }

    fn forward(&self, a: usize, b: usize) -> f32 {
        let input = thermo_2(a, b);
        let nc = self.n_connectome;
        let nw = self.n_workers;
        let mut act = vec![0.0f32; nc + nw];

        for _t in 0..TICKS {
            let mut cc = vec![0.0f32; nc];
            for i in 0..nw {
                let o = self.worker_param_offsets[i];
                let nl = self.worker_local_counts[i];
                let ww = self.params[o + INPUT_DIM + nl + nc];
                let wi = i % nc.max(1);
                if wi < nc { cc[wi] += act[nc + i] * ww; }
            }
            for i in 0..nc { act[i] = cc[i]; }

            let old = act.clone();
            for i in 0..nw {
                let o = self.worker_param_offsets[i];
                let nl = self.worker_local_counts[i];
                let mut s = self.params[o + INPUT_DIM + nl + nc + 1]; // bias
                for j in 0..INPUT_DIM { s += input[j] * self.params[o + j]; }
                let ls = i.saturating_sub(nl);
                for (k, wi) in (ls..i).enumerate() {
                    s += old[nc + wi] * self.params[o + INPUT_DIM + k];
                }
                for k in 0..nc { s += old[k] * self.params[o + INPUT_DIM + nl + k]; }

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

    fn native_accuracy(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut c = 0;
        for a in 0..DIGITS { for b in 0..DIGITS {
            if (self.forward(a, b).round() as i32) == (op(a, b) as i32) { c += 1; }
        }}
        c as f64 / 25.0
    }

    /// Gradient over both params AND c_params (if c19)
    fn gradient(&mut self, op: fn(usize, usize) -> usize) -> (Vec<f32>, Vec<f32>) {
        let eps = 1e-3f32;

        // Gradient over network params
        let n = self.params.len();
        let mut g = vec![0.0f32; n];
        for i in 0..n {
            let orig = self.params[i];
            self.params[i] = orig + eps; let lp = self.mse_loss(op);
            self.params[i] = orig - eps; let lm = self.mse_loss(op);
            self.params[i] = orig;
            g[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }

        // Gradient over C params (only if c19)
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
}

/// Run gradient until 100% OR fully stalled (no loss improvement for `patience` steps)
fn optimize(net: &mut FlatNet, op: fn(usize, usize) -> usize) -> (f64, usize) {
    let mut lr = 0.01f32;
    let patience = 100; // stop after 100 steps with no loss improvement
    let mut stale = 0;
    let mut best_loss = net.mse_loss(op);
    let mut step = 0;

    loop {
        let acc = net.native_accuracy(op);
        if acc >= 1.0 { return (acc, step); }
        if stale >= patience { return (acc, step); }

        let (g, gc) = net.gradient(op);

        let gn: f32 = g.iter().chain(gc.iter()).map(|x| x * x).sum::<f32>().sqrt();
        if gn < 1e-8 { return (acc, step); }

        let old_params = net.params.clone();
        let old_c = net.c_params.clone();
        let ol = net.mse_loss(op);

        let mut improved = false;
        for att in 0..5 {
            for i in 0..net.params.len() { net.params[i] = old_params[i] - lr * g[i] / gn; }
            for i in 0..net.c_params.len() {
                net.c_params[i] = (old_c[i] - lr * gc[i] / gn).max(0.1);
            }
            let nl = net.mse_loss(op);
            if nl < ol {
                lr *= 1.1;
                if nl < best_loss - 1e-8 { best_loss = nl; stale = 0; improved = true; }
                break;
            } else {
                lr *= 0.5;
                if att == 4 { net.params = old_params.clone(); net.c_params = old_c.clone(); }
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
    println!("=== C19 vs ReLU — CONNECTOME GRADIENT PIPELINE ===\n");
    println!("C19: periodic parabolic wave, rho=4.0 fixed, C per-worker learnable");
    println!("ReLU: max(0, x)");
    println!("Both: MSE gradient, native output, 5 digits, 3 connectome neurons");
    println!("No step limit — runs until 100% or fully stalled (patience=100)\n");

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

    println!("{:>8} {:>6} {:>10} {:>8} {:>8} {:>8}  {:>10} {:>8} {:>8} {:>8}  {:>6}",
        "task", "workers", "ReLU_best", "mean", "solved", "avg_step",
        "C19_best", "mean", "solved", "avg_step", "winner");
    println!("{}", "=".repeat(115));

    for &(name, op, nw) in &tasks {
        let t1 = Instant::now();

        // ReLU
        let relu_results: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FlatNet::new_random(nc, nw, false, &mut rng, 0.5);
            optimize(&mut net, op)
        }).collect();

        // C19
        let c19_results: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FlatNet::new_random(nc, nw, true, &mut rng, 0.5);
            optimize(&mut net, op)
        }).collect();

        let relu_best = relu_results.iter().map(|r| r.0).fold(0.0f64, f64::max);
        let relu_mean: f64 = relu_results.iter().map(|r| r.0).sum::<f64>() / n_seeds as f64;
        let relu_solved = relu_results.iter().filter(|r| r.0 >= 1.0).count();
        let relu_avg_step: f64 = relu_results.iter().map(|r| r.1 as f64).sum::<f64>() / n_seeds as f64;

        let c19_best = c19_results.iter().map(|r| r.0).fold(0.0f64, f64::max);
        let c19_mean: f64 = c19_results.iter().map(|r| r.0).sum::<f64>() / n_seeds as f64;
        let c19_solved = c19_results.iter().filter(|r| r.0 >= 1.0).count();
        let c19_avg_step: f64 = c19_results.iter().map(|r| r.1 as f64).sum::<f64>() / n_seeds as f64;

        let winner = if c19_solved > relu_solved { "C19" }
            else if relu_solved > c19_solved { "ReLU" }
            else if c19_mean > relu_mean + 0.01 { "C19" }
            else if relu_mean > c19_mean + 0.01 { "ReLU" }
            else { "TIE" };

        println!("{:>8} {:>6} {:>9.0}% {:>7.0}% {:>5}/{:<2} {:>8.0}  {:>9.0}% {:>7.0}% {:>5}/{:<2} {:>8.0}  {:>6}",
            name, nw,
            relu_best * 100.0, relu_mean * 100.0, relu_solved, n_seeds, relu_avg_step,
            c19_best * 100.0, c19_mean * 100.0, c19_solved, n_seeds, c19_avg_step,
            winner);
    }

    // =========================================================
    // Detailed learned parameter analysis — ALL tasks
    // =========================================================
    println!("\n--- Learned C values per task (C19, 20 seeds, converged) ---\n");

    let detail_seeds: Vec<u64> = (1..=20u64).collect();

    let detail_tasks: Vec<(&str, fn(usize, usize) -> usize, usize)> = vec![
        ("ADD",   op_add,     3),
        ("MAX",   op_max,     3),
        ("MIN",   op_min,     3),
        ("|a-b|", op_sub_abs, 6),
        ("MUL",   op_mul,     6),
    ];

    for &(name, op, nw) in &detail_tasks {
        println!("  === {} ({} workers) ===", name, nw);

        let results: Vec<(f64, usize, Vec<f32>, f64)> = detail_seeds.iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FlatNet::new_random(nc, nw, true, &mut rng, 0.5);
            let (acc, steps) = optimize(&mut net, op);
            let final_mse = net.mse_loss(op);
            (acc, steps, net.c_params.clone(), final_mse)
        }).collect();

        for (i, (acc, steps, c_vals, mse)) in results.iter().enumerate() {
            let c_str: Vec<String> = c_vals.iter().map(|c| format!("{:.2}", c)).collect();
            println!("    seed {:>2}: acc={:>3.0}% steps={:>4} mse={:.4}  C=[{}]",
                i + 1, acc * 100.0, steps, mse, c_str.join(", "));
        }

        // Per-worker C stats
        for w in 0..nw {
            let vals: Vec<f32> = results.iter().map(|r| r.2[w]).collect();
            let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
            let std: f32 = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32).sqrt();
            let min = vals.iter().fold(f32::MAX, |a, &b| a.min(b));
            let max = vals.iter().fold(f32::MIN, |a, &b| a.max(b));
            println!("    Worker {}: C mean={:.2} std={:.2} range=[{:.2}, {:.2}]", w, mean, std, min, max);
        }

        let solved = results.iter().filter(|r| r.0 >= 1.0).count();
        let mean_steps: f64 = results.iter().map(|r| r.1 as f64).sum::<f64>() / results.len() as f64;
        println!("    Summary: {}/{} solved, avg {:.0} steps\n", solved, detail_seeds.len(), mean_steps);
    }

    // =========================================================
    // ReLU final MSE comparison (same seeds)
    // =========================================================
    println!("--- ReLU final MSE per task (20 seeds, converged) ---\n");

    for &(name, op, nw) in &detail_tasks {
        let results: Vec<(f64, usize, f64)> = detail_seeds.iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FlatNet::new_random(nc, nw, false, &mut rng, 0.5);
            let (acc, steps) = optimize(&mut net, op);
            let final_mse = net.mse_loss(op);
            (acc, steps, final_mse)
        }).collect();

        let solved = results.iter().filter(|r| r.0 >= 1.0).count();
        let mean_acc: f64 = results.iter().map(|r| r.0).sum::<f64>() / results.len() as f64;
        let mean_mse: f64 = results.iter().map(|r| r.2).sum::<f64>() / results.len() as f64;
        let mean_steps: f64 = results.iter().map(|r| r.1 as f64).sum::<f64>() / results.len() as f64;

        let unsolved_mse: Vec<f64> = results.iter().filter(|r| r.0 < 1.0).map(|r| r.2).collect();
        let unsolved_str = if unsolved_mse.is_empty() {
            "n/a".to_string()
        } else {
            format!("{:.4}", unsolved_mse.iter().sum::<f64>() / unsolved_mse.len() as f64)
        };

        println!("  {}: {}/{} solved, mean_acc={:.0}%, avg_mse={:.4}, unsolved_avg_mse={}, avg {:.0} steps",
            name, solved, detail_seeds.len(), mean_acc * 100.0, mean_mse, unsolved_str, mean_steps);
    }

    println!("\n=== DONE ===");
}
