//! Int8 quantization: 256 levels in the weight range → does it survive?
//!
//! Run: cargo run --example quantize_int8 --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const LOCAL_CAP: usize = 3;
const TICKS: usize = 2;

fn relu(x: f32) -> f32 { x.max(0.0) }

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
    worker_param_offsets: Vec<usize>,
    worker_param_counts: Vec<usize>,
    worker_local_counts: Vec<usize>,
}

impl FlatNet {
    fn new_random(nc: usize, nw_target: usize, rng: &mut StdRng, scale: f32) -> Self {
        let mut net = FlatNet {
            n_connectome: nc, n_workers: 0, params: Vec::new(),
            worker_param_offsets: Vec::new(), worker_param_counts: Vec::new(),
            worker_local_counts: Vec::new(),
        };
        for i in 0..nw_target {
            let nl = LOCAL_CAP.min(i);
            let np = INPUT_DIM + nl + nc + 1 + 1;
            let init: Vec<f32> = (0..np).map(|_| rng.gen_range(-scale..scale)).collect();
            net.worker_param_offsets.push(net.params.len());
            net.worker_param_counts.push(np);
            net.worker_local_counts.push(nl);
            net.params.extend_from_slice(&init);
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
                for (k, wi) in (ls..i).enumerate() { s += old[nc + wi] * self.params[o + INPUT_DIM + k]; }
                for k in 0..nc { s += old[k] * self.params[o + INPUT_DIM + nl + k]; }
                act[nc + i] = relu(s);
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

    fn gradient(&mut self, op: fn(usize, usize) -> usize) -> Vec<f32> {
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
        g
    }

    /// Quantize to N uniform levels between [min_w, max_w]
    fn quantize_uniform(&mut self, n_levels: usize) {
        let min_w = self.params.iter().fold(f32::MAX, |a, &b| a.min(b));
        let max_w = self.params.iter().fold(f32::MIN, |a, &b| a.max(b));
        let range = max_w - min_w;
        if range < 1e-10 { return; }
        let step = range / (n_levels - 1) as f32;
        for p in &mut self.params {
            let idx = ((*p - min_w) / step).round() as usize;
            let idx = idx.min(n_levels - 1);
            *p = min_w + idx as f32 * step;
        }
    }

    /// Quantize to fixed set
    fn quantize_set(&mut self, levels: &[f32]) {
        for p in &mut self.params {
            *p = *levels.iter()
                .min_by(|a, b| (*a - *p).abs().partial_cmp(&(*b - *p).abs()).unwrap())
                .unwrap();
        }
    }
}

fn optimize(net: &mut FlatNet, op: fn(usize, usize) -> usize, steps: usize) {
    let mut lr = 0.01f32;
    for _ in 0..steps {
        if net.native_accuracy(op) >= 1.0 { break; }
        let g = net.gradient(op);
        let gn: f32 = g.iter().map(|x| x * x).sum::<f32>().sqrt();
        if gn < 1e-8 { break; }
        let old = net.params.clone();
        let ol = net.mse_loss(op);
        for att in 0..5 {
            for i in 0..net.params.len() { net.params[i] = old[i] - lr * g[i] / gn; }
            if net.mse_loss(op) < ol { lr *= 1.1; break; }
            else { lr *= 0.5; if att == 4 { net.params = old.clone(); } }
        }
    }
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn main() {
    let t0 = Instant::now();
    println!("=== INT8 QUANTIZATION TEST ===\n");

    let tasks: Vec<(&str, fn(usize, usize) -> usize)> = vec![
        ("ADD", op_add), ("MUL", op_mul), ("MAX", op_max),
        ("MIN", op_min), ("|a-b|", op_sub_abs),
    ];

    let nc = 3;
    let nw = 2;
    let n_seeds = 100u64;

    // Quantization levels to test
    let levels: Vec<(&str, usize)> = vec![
        ("3 (ternary)",    3),
        ("5 (quinary)",    5),
        ("8",              8),
        ("16 (int4)",     16),
        ("32",            32),
        ("64",            64),
        ("128",          128),
        ("256 (int8)",   256),
    ];

    println!("{:<8} {:>8} | {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "task", "float",
        "3", "5", "8", "16", "32", "64", "128", "256");
    println!("{}", "=".repeat(95));

    for &(task_name, task_op) in &tasks {
        // Train
        let trained: Vec<FlatNet> = (1..=n_seeds).into_par_iter().map(|seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FlatNet::new_random(nc, nw, &mut rng, 1.0);
            optimize(&mut net, task_op, 2000);
            net
        }).collect();

        let float_solved = trained.iter().filter(|n| n.native_accuracy(task_op) >= 1.0).count();

        print!("{:<8} {:>5}/100 |", task_name, float_solved);

        for &(_, n_lvl) in &levels {
            let q_solved: usize = trained.iter().map(|net| {
                let mut qnet = net.clone();
                qnet.quantize_uniform(n_lvl);
                if qnet.native_accuracy(task_op) >= 1.0 { 1 } else { 0 }
            }).sum();
            print!(" {:>7}/100", q_solved);
        }
        println!();
    }

    // Also: detailed view for ADD with weight distributions
    println!("\n--- ADD detailed: weight error per quantization level ---\n");
    let mut rng = StdRng::seed_from_u64(42);
    let mut net = FlatNet::new_random(nc, nw, &mut rng, 1.0);
    optimize(&mut net, op_add, 2000);
    let float_acc = net.native_accuracy(op_add);
    println!("  Float accuracy: {:.0}%", float_acc * 100.0);
    println!("  Float MSE:      {:.6}", net.mse_loss(op_add));
    let (wmin, wmax) = (
        net.params.iter().fold(f32::MAX, |a, &b| a.min(b)),
        net.params.iter().fold(f32::MIN, |a, &b| a.max(b)),
    );
    println!("  Weight range:   [{:.3}, {:.3}]", wmin, wmax);

    println!("\n  {:>6} {:>10} {:>10} {:>8} {:>8}",
        "levels", "step_size", "max_err", "acc", "mse");
    println!("  {}", "-".repeat(50));

    for &n in &[3, 5, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let mut qnet = net.clone();
        qnet.quantize_uniform(n);

        let max_err: f32 = net.params.iter().zip(qnet.params.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let step = (wmax - wmin) / (n - 1) as f32;

        println!("  {:>6} {:>10.6} {:>10.6} {:>7.0}% {:>8.4}",
            n, step, max_err, qnet.native_accuracy(op_add) * 100.0, qnet.mse_loss(op_add));
    }

    println!("\n=== DONE ({:.1}s) ===", t0.elapsed().as_secs_f64());
}
