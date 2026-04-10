//! Quantization test: train float → round to integer → does it still work?
//!
//! Pipeline:
//!   1. Random float init + gradient descent → 100% accuracy (float weights)
//!   2. Quantize weights to nearest {-2,-1,0,1,2} or {-1,0,1} or {-1,1}
//!   3. Check accuracy of quantized network
//!
//! This answers: can we TRAIN in float and DEPLOY in integer?
//!
//! Run: cargo run --example quantize_test --release

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
    fn new_random(n_connectome: usize, n_workers: usize, rng: &mut StdRng, init_scale: f32) -> Self {
        let mut net = FlatNet {
            n_connectome, n_workers: 0, params: Vec::new(),
            worker_param_offsets: Vec::new(), worker_param_counts: Vec::new(),
            worker_local_counts: Vec::new(),
        };
        for i in 0..n_workers {
            let n_local = LOCAL_CAP.min(i);
            let n_params = INPUT_DIM + n_local + n_connectome + 1 + 1;
            let init: Vec<f32> = (0..n_params).map(|_| rng.gen_range(-init_scale..init_scale)).collect();
            net.worker_param_offsets.push(net.params.len());
            net.worker_param_counts.push(n_params);
            net.worker_local_counts.push(n_local);
            net.params.extend_from_slice(&init);
            net.n_workers += 1;
        }
        net
    }

    fn forward(&self, a: usize, b: usize) -> f32 {
        let input = thermo_2(a, b);
        let nc = self.n_connectome;
        let nw = self.n_workers;
        let mut activations = vec![0.0f32; nc + nw];

        for _tick in 0..TICKS {
            let mut conn_charges = vec![0.0f32; nc];
            for i in 0..nw {
                let off = self.worker_param_offsets[i];
                let nl = self.worker_local_counts[i];
                let w_write = self.params[off + INPUT_DIM + nl + nc];
                let write_idx = i % nc.max(1);
                if write_idx < nc { conn_charges[write_idx] += activations[nc + i] * w_write; }
            }
            for i in 0..nc { activations[i] = conn_charges[i]; }

            let old_act = activations.clone();
            for i in 0..nw {
                let off = self.worker_param_offsets[i];
                let nl = self.worker_local_counts[i];
                let mut sum = self.params[off + INPUT_DIM + nl + nc + 1];
                for j in 0..INPUT_DIM { sum += input[j] * self.params[off + j]; }
                let local_start = i.saturating_sub(nl);
                for (k, widx) in (local_start..i).enumerate() {
                    sum += old_act[nc + widx] * self.params[off + INPUT_DIM + k];
                }
                for k in 0..nc { sum += old_act[k] * self.params[off + INPUT_DIM + nl + k]; }
                activations[nc + i] = relu(sum);
            }
        }
        activations[nc..].iter().sum()
    }

    fn mse_loss(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut loss = 0.0f64;
        for a in 0..DIGITS {
            for b in 0..DIGITS {
                let diff = self.forward(a, b) as f64 - op(a, b) as f64;
                loss += diff * diff;
            }
        }
        loss / 25.0
    }

    fn native_accuracy(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut correct = 0;
        for a in 0..DIGITS {
            for b in 0..DIGITS {
                if (self.forward(a, b).round() as i32) == (op(a, b) as i32) { correct += 1; }
            }
        }
        correct as f64 / 25.0
    }

    fn gradient(&mut self, op: fn(usize, usize) -> usize) -> Vec<f32> {
        let eps = 1e-3f32;
        let n = self.params.len();
        let mut grad = vec![0.0f32; n];
        for i in 0..n {
            let orig = self.params[i];
            self.params[i] = orig + eps;
            let lp = self.mse_loss(op);
            self.params[i] = orig - eps;
            let lm = self.mse_loss(op);
            self.params[i] = orig;
            grad[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        grad
    }

    /// Quantize all weights to nearest value in the given set
    fn quantize(&mut self, levels: &[f32]) {
        for p in &mut self.params {
            *p = *levels.iter()
                .min_by(|a, b| (*a - *p).abs().partial_cmp(&(*b - *p).abs()).unwrap())
                .unwrap();
        }
    }

    /// Show weight distribution
    fn weight_stats(&self) -> (f32, f32, f32, f32) {
        let min = self.params.iter().fold(f32::MAX, |a, &b| a.min(b));
        let max = self.params.iter().fold(f32::MIN, |a, &b| a.max(b));
        let mean = self.params.iter().sum::<f32>() / self.params.len() as f32;
        let std = (self.params.iter().map(|p| (p - mean).powi(2)).sum::<f32>() / self.params.len() as f32).sqrt();
        (min, max, mean, std)
    }
}

fn optimize_gradient(net: &mut FlatNet, op: fn(usize, usize) -> usize, steps: usize) {
    let mut lr = 0.01f32;
    for _ in 0..steps {
        if net.native_accuracy(op) >= 1.0 { break; }
        let grad = net.gradient(op);
        let grad_norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        if grad_norm < 1e-8 { break; }
        let old_params = net.params.clone();
        let old_loss = net.mse_loss(op);
        for attempt in 0..5 {
            for i in 0..net.params.len() {
                net.params[i] = old_params[i] - lr * grad[i] / grad_norm;
            }
            if net.mse_loss(op) < old_loss { lr *= 1.1; break; }
            else { lr *= 0.5; if attempt == 4 { net.params = old_params.clone(); } }
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
    println!("=== QUANTIZATION TEST: float train → integer deploy ===\n");

    let tasks: Vec<(&str, fn(usize, usize) -> usize)> = vec![
        ("ADD", op_add), ("MUL", op_mul), ("MAX", op_max),
        ("MIN", op_min), ("|a-b|", op_sub_abs),
    ];

    let nc = 3;
    let nw = 2;
    let n_seeds = 100;
    let seeds: Vec<u64> = (1..=n_seeds).collect();

    let quant_levels: Vec<(&str, Vec<f32>)> = vec![
        ("binary ±1",     vec![-1.0, 1.0]),
        ("ternary ±1,0",  vec![-1.0, 0.0, 1.0]),
        ("quinary ±2",    vec![-2.0, -1.0, 0.0, 1.0, 2.0]),
        ("7-level ±3",    vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]),
        ("int4 ±8",       (-8..=8).map(|i| i as f32).collect()),
        ("int8 ±128",     (-8..=8).map(|i| (i * 16) as f32).collect()),
    ];

    // =========================================================
    // For each task: train float, then quantize to each level
    // =========================================================
    for &(task_name, task_op) in &tasks {
        println!("--- {} ({} workers, {} connectome) ---\n", task_name, nw, nc);

        // Train all seeds in parallel
        let trained: Vec<FlatNet> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FlatNet::new_random(nc, nw, &mut rng, 1.0);
            optimize_gradient(&mut net, task_op, 2000);
            net
        }).collect();

        let float_solved = trained.iter().filter(|n| n.native_accuracy(task_op) >= 1.0).count();
        let float_mean: f64 = trained.iter().map(|n| n.native_accuracy(task_op)).sum::<f64>() / trained.len() as f64;

        // Weight stats from first solved network
        if let Some(solved_net) = trained.iter().find(|n| n.native_accuracy(task_op) >= 1.0) {
            let (wmin, wmax, wmean, wstd) = solved_net.weight_stats();
            println!("  Float weights: min={:.2} max={:.2} mean={:.2} std={:.2}",
                wmin, wmax, wmean, wstd);
            println!("  Weights: {:?}", solved_net.params.iter().map(|w| format!("{:.2}", w)).collect::<Vec<_>>());
        }

        println!("\n  {:<16} {:>8} {:>8} {:>10}", "quantization", "solved", "mean%", "vs_float");
        println!("  {}", "-".repeat(50));
        println!("  {:<16} {:>5}/{:>3} {:>7.0}% {:>10}",
            "float (no quant)", float_solved, n_seeds, float_mean * 100.0, "baseline");

        for (q_name, q_levels) in &quant_levels {
            let results: Vec<f64> = trained.iter().map(|net| {
                let mut qnet = net.clone();
                qnet.quantize(q_levels);
                qnet.native_accuracy(task_op)
            }).collect();

            let q_solved = results.iter().filter(|&&a| a >= 1.0).count();
            let q_mean: f64 = results.iter().sum::<f64>() / results.len() as f64;

            println!("  {:<16} {:>5}/{:>3} {:>7.0}% {:>+9.0}pp",
                q_name, q_solved, n_seeds, q_mean * 100.0,
                (q_mean - float_mean) * 100.0);
        }
        println!();
    }

    println!("=== DONE ({:.1}s) ===", t0.elapsed().as_secs_f64());
}
