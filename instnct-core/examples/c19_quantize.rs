//! C19 quantization test
//!
//! Train with float C19 → quantize both weights AND C parameter
//! How many bits does C need?
//!
//! Run: cargo run --example c19_quantize --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const LOCAL_CAP: usize = 3;
const TICKS: usize = 2;
const RHO: f32 = 4.0;

fn c19(x: f32, c: f32) -> f32 {
    let c = c.max(0.1);
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
    params: Vec<f32>,
    c_params: Vec<f32>,
    worker_param_offsets: Vec<usize>,
    worker_local_counts: Vec<usize>,
}

impl FlatNet {
    fn new_random(nc: usize, nw: usize, rng: &mut StdRng, scale: f32) -> Self {
        let mut net = FlatNet {
            n_connectome: nc, n_workers: 0,
            params: Vec::new(), c_params: Vec::new(),
            worker_param_offsets: Vec::new(),
            worker_local_counts: Vec::new(),
        };
        for i in 0..nw {
            let nl = LOCAL_CAP.min(i);
            let np = INPUT_DIM + nl + nc + 1 + 1;
            let init: Vec<f32> = (0..np).map(|_| rng.gen_range(-scale..scale)).collect();
            net.worker_param_offsets.push(net.params.len());
            net.worker_local_counts.push(nl);
            net.params.extend_from_slice(&init);
            net.c_params.push(1.0);
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
                let mut s = self.params[o + INPUT_DIM + nl + nc + 1];
                for j in 0..INPUT_DIM { s += input[j] * self.params[o + j]; }
                let ls = i.saturating_sub(nl);
                for (k, wi) in (ls..i).enumerate() {
                    s += old[nc + wi] * self.params[o + INPUT_DIM + k];
                }
                for k in 0..nc { s += old[k] * self.params[o + INPUT_DIM + nl + k]; }
                act[nc + i] = c19(s, self.c_params[i]);
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

    fn gradient(&mut self, op: fn(usize, usize) -> usize) -> (Vec<f32>, Vec<f32>) {
        let eps = 1e-3f32;
        let n = self.params.len();
        let mut g = vec![0.0f32; n];
        for i in 0..n {
            let orig = self.params[i];
            self.params[i] = orig + eps; let lp = self.mse_loss(op);
            self.params[i] = orig - eps; let lm = self.mse_loss(op);
            self.params[i] = orig;
            g[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        let nc = self.c_params.len();
        let mut gc = vec![0.0f32; nc];
        for i in 0..nc {
            let orig = self.c_params[i];
            self.c_params[i] = orig + eps; let lp = self.mse_loss(op);
            self.c_params[i] = orig - eps; let lm = self.mse_loss(op);
            self.c_params[i] = orig;
            gc[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        (g, gc)
    }

    /// Quantize weights to n_levels uniform steps in [min, max]
    fn quantize_weights(&mut self, n_levels: usize) {
        let min_w = self.params.iter().fold(f32::MAX, |a, &b| a.min(b));
        let max_w = self.params.iter().fold(f32::MIN, |a, &b| a.max(b));
        let step = (max_w - min_w) / (n_levels - 1) as f32;
        if step < 1e-10 { return; }
        for p in &mut self.params {
            let idx = ((*p - min_w) / step).round() as usize;
            *p = min_w + idx.min(n_levels - 1) as f32 * step;
        }
    }

    /// Quantize C params to n_levels uniform steps in [min, max]
    fn quantize_c(&mut self, n_levels: usize) {
        let min_c = self.c_params.iter().fold(f32::MAX, |a, &b| a.min(b));
        let max_c = self.c_params.iter().fold(f32::MIN, |a, &b| a.max(b));
        let range = max_c - min_c;
        if range < 1e-10 || n_levels <= 1 {
            // All same value, nothing to quantize
            return;
        }
        let step = range / (n_levels - 1) as f32;
        for c in &mut self.c_params {
            let idx = ((*c - min_c) / step).round() as usize;
            *c = (min_c + idx.min(n_levels - 1) as f32 * step).max(0.1);
        }
    }
}

fn optimize(net: &mut FlatNet, op: fn(usize, usize) -> usize) -> (f64, usize) {
    let mut lr = 0.01f32;
    let patience = 100;
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
    println!("=== C19 QUANTIZATION TEST ===\n");
    println!("Pipeline: float train → quantize weights + C → test accuracy");
    println!("Question: how many bits does C need?\n");

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

    // Weight quantization levels to test
    let w_levels: Vec<usize> = vec![256]; // i8 for weights (already proven)

    // C quantization levels to test
    let c_levels: Vec<usize> = vec![2, 3, 4, 8, 16, 32, 64, 256];

    // =========================================================
    // Test 1: Fix weights at i8, sweep C quantization
    // =========================================================
    println!("--- TEST 1: Weights=i8 (256 levels), sweep C quantization ---\n");
    println!("{:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "task", "float", "C=2lev", "C=3lev", "C=4lev", "C=8lev",
        "C=16lev", "C=32lev", "C=64lev", "C=256lev");
    println!("{}", "=".repeat(100));

    for &(name, op, nw) in &tasks {
        // Train all seeds first (float)
        let trained: Vec<FlatNet> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FlatNet::new_random(nc, nw, &mut rng, 0.5);
            optimize(&mut net, op);
            net
        }).collect();

        // Float baseline
        let float_solved = trained.iter().filter(|n| n.native_accuracy(op) >= 1.0).count();

        // Test each C quantization level (weights always i8)
        let mut results: Vec<usize> = Vec::new();
        for &cl in &c_levels {
            let solved: usize = trained.par_iter().map(|net| {
                let mut q = net.clone();
                q.quantize_weights(256); // i8 weights
                q.quantize_c(cl);
                if q.native_accuracy(op) >= 1.0 { 1 } else { 0 }
            }).sum();
            results.push(solved);
        }

        print!("{:>8} {:>5}/{:<2}", name, float_solved, n_seeds);
        for &s in &results {
            print!(" {:>7}/{:<2}", s, n_seeds);
        }
        println!();
    }

    // =========================================================
    // Test 2: Both weights AND C at various levels
    // =========================================================
    println!("\n--- TEST 2: Joint quantization (same levels for weights and C) ---\n");
    let joint_levels: Vec<usize> = vec![4, 8, 16, 32, 64, 128, 256];

    println!("{:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "task", "float", "4lev", "8lev", "16lev", "32lev", "64lev", "128lev", "256lev");
    println!("{}", "=".repeat(95));

    for &(name, op, nw) in &tasks {
        let trained: Vec<FlatNet> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FlatNet::new_random(nc, nw, &mut rng, 0.5);
            optimize(&mut net, op);
            net
        }).collect();

        let float_solved = trained.iter().filter(|n| n.native_accuracy(op) >= 1.0).count();

        let mut results: Vec<usize> = Vec::new();
        for &lev in &joint_levels {
            let solved: usize = trained.par_iter().map(|net| {
                let mut q = net.clone();
                q.quantize_weights(lev);
                q.quantize_c(lev);
                if q.native_accuracy(op) >= 1.0 { 1 } else { 0 }
            }).sum();
            results.push(solved);
        }

        print!("{:>8} {:>5}/{:<2}", name, float_solved, n_seeds);
        for &s in &results {
            print!(" {:>7}/{:<2}", s, n_seeds);
        }
        println!();
    }

    // =========================================================
    // Test 3: Fixed C values (no learning) — does C even need to be learned?
    // =========================================================
    println!("\n--- TEST 3: Fixed C (not learned) vs learned C ---\n");
    println!("{:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "task", "C=0.5", "C=1.0", "C=2.0", "C=3.0", "C=5.0", "learned");
    println!("{}", "=".repeat(70));

    let fixed_c_vals: Vec<f32> = vec![0.5, 1.0, 2.0, 3.0, 5.0];

    for &(name, op, nw) in &tasks {
        let mut results: Vec<usize> = Vec::new();

        for &fc in &fixed_c_vals {
            let solved: usize = seeds.par_iter().map(|&seed| {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut net = FlatNet::new_random(nc, nw, &mut rng, 0.5);
                // Fix C, don't learn it
                for c in &mut net.c_params { *c = fc; }
                // Train with gradient but only on weights (zero out C gradient)
                let mut lr = 0.01f32;
                let patience = 100;
                let mut stale = 0;
                let mut best_loss = net.mse_loss(op);
                for _ in 0..10000 {
                    let acc = net.native_accuracy(op);
                    if acc >= 1.0 { return 1; }
                    if stale >= patience { return 0; }
                    let (g, _gc) = net.gradient(op); // ignore C gradient
                    let gn: f32 = g.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if gn < 1e-8 { return 0; }
                    let old = net.params.clone();
                    let ol = net.mse_loss(op);
                    let mut improved = false;
                    for att in 0..5 {
                        for i in 0..net.params.len() { net.params[i] = old[i] - lr * g[i] / gn; }
                        let nl = net.mse_loss(op);
                        if nl < ol {
                            lr *= 1.1;
                            if nl < best_loss - 1e-8 { best_loss = nl; stale = 0; improved = true; }
                            break;
                        } else {
                            lr *= 0.5;
                            if att == 4 { net.params = old.clone(); }
                        }
                    }
                    if !improved { stale += 1; }
                }
                0
            }).sum();
            results.push(solved);
        }

        // Learned C (full pipeline)
        let learned_solved: usize = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FlatNet::new_random(nc, nw, &mut rng, 0.5);
            let (acc, _) = optimize(&mut net, op);
            if acc >= 1.0 { 1 } else { 0 }
        }).sum();

        print!("{:>8}", name);
        for &s in &results {
            print!(" {:>7}/{:<2}", s, n_seeds);
        }
        println!(" {:>7}/{:<2}", learned_solved, n_seeds);
    }

    println!("\n=== DONE ===");
}
