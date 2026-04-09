//! Native output + gradient optimization: charge = answer, no readout.
//!
//! Pipeline:
//!   1. Binary exhaustive → good integer starting point
//!   2. MSE loss = Σ(charge - target)² over all 25 pairs
//!   3. Optimize: numerical gradient, Newton (Hessian), random perturbation
//!   4. Compare convergence speed and final accuracy
//!
//! Tasks: ADD first (proven native), then MAX, MIN, |a-b|, MUL
//!
//! Run: cargo run --example native_backprop --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const LOCAL_CAP: usize = 3;
const TICKS: usize = 2;

fn relu(x: f32) -> f32 { x.max(0.0) }
fn relu_grad(x: f32) -> f32 { if x > 0.0 { 1.0 } else { 0.0 } }

fn thermo_2(a: usize, b: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; INPUT_DIM];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

// ============================================================
// Flat parameter vector network — easy for gradient ops
// ============================================================
#[derive(Clone)]
struct FlatNet {
    n_connectome: usize,
    n_workers: usize,
    params: Vec<f32>,  // all weights flat: [worker0_params, worker1_params, ...]
    // Per worker: [w_input(8), w_local(up to 3), w_conn_read(n_conn), w_conn_write(1), bias(1)]
    worker_param_counts: Vec<usize>, // how many params per worker
    worker_param_offsets: Vec<usize>, // offset into params vec
    worker_local_counts: Vec<usize>, // how many local connections per worker
}

impl FlatNet {
    fn new(n_connectome: usize) -> Self {
        FlatNet {
            n_connectome, n_workers: 0,
            params: Vec::new(),
            worker_param_counts: Vec::new(),
            worker_param_offsets: Vec::new(),
            worker_local_counts: Vec::new(),
        }
    }

    fn add_worker(&mut self, init_params: &[f32]) {
        let n_local = LOCAL_CAP.min(self.n_workers);
        let expected = INPUT_DIM + n_local + self.n_connectome + 1 + 1;
        assert_eq!(init_params.len(), expected);

        self.worker_param_offsets.push(self.params.len());
        self.worker_param_counts.push(expected);
        self.worker_local_counts.push(n_local);
        self.params.extend_from_slice(init_params);
        self.n_workers += 1;
    }

    fn total_params(&self) -> usize { self.params.len() }

    /// Get worker i's params as slices
    fn worker_params(&self, i: usize) -> (&[f32], &[f32], &[f32], f32, f32) {
        let off = self.worker_param_offsets[i];
        let n_local = self.worker_local_counts[i];
        let nc = self.n_connectome;

        let w_input = &self.params[off..off + INPUT_DIM];
        let w_local = &self.params[off + INPUT_DIM..off + INPUT_DIM + n_local];
        let w_conn = &self.params[off + INPUT_DIM + n_local..off + INPUT_DIM + n_local + nc];
        let w_write = self.params[off + INPUT_DIM + n_local + nc];
        let bias = self.params[off + INPUT_DIM + n_local + nc + 1];

        (w_input, w_local, w_conn, w_write, bias)
    }

    /// Forward pass → returns charge (sum of worker activations)
    fn forward(&self, a: usize, b: usize) -> f32 {
        let input = thermo_2(a, b);
        let nc = self.n_connectome;
        let nw = self.n_workers;
        let n_total = nc + nw;

        let mut activations = vec![0.0f32; n_total];

        for _tick in 0..TICKS {
            // Phase 1: Passive connectome — accumulate from workers
            let mut conn_charges = vec![0.0f32; nc];
            for i in 0..nw {
                let (_, _, _, w_write, _) = self.worker_params(i);
                let write_idx = i % nc.max(1);
                if write_idx < nc {
                    conn_charges[write_idx] += activations[nc + i] * w_write;
                }
            }
            for i in 0..nc { activations[i] = conn_charges[i]; }

            // Phase 2: Workers compute
            let old_act = activations.clone();
            for i in 0..nw {
                let (w_input, w_local, w_conn, _, bias) = self.worker_params(i);
                let n_local = self.worker_local_counts[i];
                let mut sum = bias;

                // Input
                for (j, &w) in w_input.iter().enumerate() {
                    if j < input.len() { sum += input[j] * w; }
                }

                // Local neighbors
                let workers_before = i;
                let local_start = workers_before.saturating_sub(n_local);
                for (k, widx) in (local_start..workers_before).enumerate() {
                    if k < w_local.len() {
                        sum += old_act[nc + widx] * w_local[k];
                    }
                }

                // Connectome read
                for (k, &w) in w_conn.iter().enumerate() {
                    if k < nc { sum += old_act[k] * w; }
                }

                activations[nc + i] = relu(sum);
            }
        }

        // Native output: sum of worker activations
        activations[nc..].iter().sum()
    }

    /// MSE loss over all 25 input pairs
    fn mse_loss(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut loss = 0.0f64;
        for a in 0..DIGITS {
            for b in 0..DIGITS {
                let charge = self.forward(a, b) as f64;
                let target = op(a, b) as f64;
                loss += (charge - target).powi(2);
            }
        }
        loss / 25.0
    }

    /// Native accuracy: round(charge) == target
    fn native_accuracy(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut correct = 0;
        for a in 0..DIGITS {
            for b in 0..DIGITS {
                let charge = self.forward(a, b);
                let pred = charge.round() as i32;
                let target = op(a, b) as i32;
                if pred == target { correct += 1; }
            }
        }
        correct as f64 / 25.0
    }

    /// Numerical gradient (central differences)
    fn gradient(&mut self, op: fn(usize, usize) -> usize) -> Vec<f32> {
        let eps = 1e-3f32;
        let n = self.params.len();
        let mut grad = vec![0.0f32; n];

        for i in 0..n {
            let orig = self.params[i];

            self.params[i] = orig + eps;
            let loss_plus = self.mse_loss(op);

            self.params[i] = orig - eps;
            let loss_minus = self.mse_loss(op);

            self.params[i] = orig;
            grad[i] = ((loss_plus - loss_minus) / (2.0 * eps as f64)) as f32;
        }

        grad
    }

    /// Hessian matrix (numerical, central differences)
    fn hessian(&mut self, op: fn(usize, usize) -> usize) -> Vec<Vec<f32>> {
        let eps = 1e-2f32;
        let n = self.params.len();
        let mut H = vec![vec![0.0f32; n]; n];
        let f0 = self.mse_loss(op);

        for i in 0..n {
            for j in i..n {
                let orig_i = self.params[i];
                let orig_j = self.params[j];

                self.params[i] = orig_i + eps;
                self.params[j] = orig_j + eps;
                let fpp = self.mse_loss(op);

                self.params[i] = orig_i + eps;
                self.params[j] = orig_j - eps;
                let fpm = self.mse_loss(op);

                self.params[i] = orig_i - eps;
                self.params[j] = orig_j + eps;
                let fmp = self.mse_loss(op);

                self.params[i] = orig_i - eps;
                self.params[j] = orig_j - eps;
                let fmm = self.mse_loss(op);

                self.params[i] = orig_i;
                self.params[j] = orig_j;

                let h = ((fpp - fpm - fmp + fmm) / (4.0 * eps as f64 * eps as f64)) as f32;
                H[i][j] = h;
                H[j][i] = h;
            }
        }

        H
    }
}

// ============================================================
// Optimization methods
// ============================================================

/// Gradient descent with line search
fn optimize_gradient(net: &mut FlatNet, op: fn(usize, usize) -> usize, steps: usize) -> Vec<(usize, f64, f64)> {
    let mut history = Vec::new();
    let mut lr = 0.01f32;

    for step in 0..steps {
        let loss = net.mse_loss(op);
        let acc = net.native_accuracy(op);
        if step % 50 == 0 || acc >= 1.0 {
            history.push((step, loss, acc));
        }
        if acc >= 1.0 { break; }

        let grad = net.gradient(op);
        let grad_norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        if grad_norm < 1e-8 { break; } // converged

        // Simple line search
        let old_params = net.params.clone();
        let old_loss = loss;

        // Try lr, if worse halve it
        for attempt in 0..5 {
            for i in 0..net.params.len() {
                net.params[i] = old_params[i] - lr * grad[i] / grad_norm;
            }
            let new_loss = net.mse_loss(op);
            if new_loss < old_loss {
                lr *= 1.1; // speed up
                break;
            } else {
                lr *= 0.5; // slow down
                if attempt == 4 { net.params = old_params.clone(); }
            }
        }
    }

    history
}

/// Newton's method (Hessian)
fn optimize_newton(net: &mut FlatNet, op: fn(usize, usize) -> usize, steps: usize) -> Vec<(usize, f64, f64)> {
    let mut history = Vec::new();

    for step in 0..steps {
        let loss = net.mse_loss(op);
        let acc = net.native_accuracy(op);
        history.push((step, loss, acc));
        if acc >= 1.0 { break; }

        let grad = net.gradient(op);
        let hess = net.hessian(op);
        let n = net.params.len();

        // Solve H * delta = -grad using simple Gauss elimination
        // Add regularization to diagonal for numerical stability
        let mut aug = vec![vec![0.0f64; n + 1]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = hess[i][j] as f64;
            }
            aug[i][i] += 1e-4; // regularization
            aug[i][n] = -(grad[i] as f64);
        }

        // Gaussian elimination
        for i in 0..n {
            let mut max_row = i;
            for k in i + 1..n {
                if aug[k][i].abs() > aug[max_row][i].abs() { max_row = k; }
            }
            aug.swap(i, max_row);

            if aug[i][i].abs() < 1e-12 { continue; }

            for k in i + 1..n {
                let factor = aug[k][i] / aug[i][i];
                for j in i..=n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }

        // Back substitution
        let mut delta = vec![0.0f64; n];
        for i in (0..n).rev() {
            if aug[i][i].abs() < 1e-12 { continue; }
            delta[i] = aug[i][n];
            for j in i + 1..n {
                delta[i] -= aug[i][j] * delta[j];
            }
            delta[i] /= aug[i][i];
        }

        // Apply with damping
        let step_size = 0.5f64;
        let old_params = net.params.clone();
        for i in 0..n {
            net.params[i] += (step_size * delta[i]) as f32;
        }

        // Check if improved, revert if worse
        let new_loss = net.mse_loss(op);
        if new_loss > loss * 1.1 {
            net.params = old_params;
        }
    }

    history
}

/// Random perturbation (baseline comparison)
fn optimize_perturbation(net: &mut FlatNet, op: fn(usize, usize) -> usize, steps: usize, seed: u64) -> Vec<(usize, f64, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut history = Vec::new();
    let mut current_loss = net.mse_loss(op);

    for step in 0..steps {
        if step % 500 == 0 {
            let acc = net.native_accuracy(op);
            history.push((step, current_loss, acc));
            if acc >= 1.0 { break; }
        }

        let idx = rng.gen_range(0..net.params.len());
        let delta: f32 = rng.gen_range(-0.3..0.3);
        let old = net.params[idx];
        net.params[idx] += delta;

        let new_loss = net.mse_loss(op);
        if new_loss <= current_loss {
            current_loss = new_loss;
        } else {
            net.params[idx] = old;
        }
    }

    let acc = net.native_accuracy(op);
    history.push((steps, current_loss, acc));
    history
}

// ============================================================
// Binary exhaustive search for starting point
// ============================================================
fn find_binary_start(
    n_connectome: usize, n_workers: usize,
    op: fn(usize, usize) -> usize, seed: u64,
) -> FlatNet {
    let binary: Vec<f32> = vec![-1.0, 1.0];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut best_net = FlatNet::new(n_connectome);
    let mut best_loss = f64::MAX;

    for worker_i in 0..n_workers {
        let n_local = LOCAL_CAP.min(best_net.n_workers);
        let n_params = INPUT_DIM + n_local + n_connectome + 1 + 1;
        let total_configs = 2u64.pow(n_params as u32);

        let mut worker_best_loss = best_loss;
        let mut worker_best_params: Vec<f32> = vec![0.0; n_params];

        let use_exhaustive = total_configs <= 50_000_000;
        let sample_count = if use_exhaustive { total_configs } else { 5_000_000 };

        for sample in 0..sample_count {
            let mut c = if use_exhaustive { sample } else { rng.gen_range(0..total_configs) };
            let params: Vec<f32> = (0..n_params).map(|_| {
                let v = binary[(c % 2) as usize]; c /= 2; v
            }).collect();

            let mut test_net = best_net.clone();
            test_net.add_worker(&params);
            let loss = test_net.mse_loss(op);

            if loss < worker_best_loss {
                worker_best_loss = loss;
                worker_best_params = params;
            }
        }

        best_net.add_worker(&worker_best_params);
        best_loss = worker_best_loss;
    }

    best_net
}

// ============================================================
// Tasks
// ============================================================
fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

const SEEDS: &[u64] = &[42, 123, 777, 314, 999, 1337, 2024, 55, 101, 202,
                         303, 404, 505, 606, 707, 808, 9001, 1234, 5678, 31415];

fn main() {
    let t0 = Instant::now();
    println!("=== NATIVE OUTPUT + GRADIENT OPTIMIZATION ===");
    println!("charge = answer, MSE loss, no readout");
    println!("Passive {}-connectome, LOCAL_CAP={}, TICKS={}\n", 3, LOCAL_CAP, TICKS);

    let tasks: Vec<(&str, fn(usize, usize) -> usize)> = vec![
        ("ADD", op_add),
        ("MUL", op_mul),
        ("MAX", op_max),
        ("MIN", op_min),
        ("|a-b|", op_sub_abs),
    ];

    let n_connectome = 3;
    let n_workers = 6;

    // =========================================================
    // EXP 1: ADD — compare all 3 optimization methods
    // =========================================================
    println!("--- EXP 1: ADD — method comparison (20 seeds) ---\n");

    let methods = ["binary_only", "binary+gradient", "binary+newton", "binary+perturbation"];
    println!("{:<24} {:>8} {:>8} {:>10} {:>8}", "method", "best", "mean", "mean_loss", "time");
    println!("{}", "-".repeat(65));

    for method in &methods {
        let t1 = Instant::now();
        let results: Vec<(f64, f64)> = SEEDS.par_iter().map(|&seed| {
            let mut net = find_binary_start(n_connectome, n_workers, op_add, seed);
            let binary_acc = net.native_accuracy(op_add);
            let binary_loss = net.mse_loss(op_add);

            match *method {
                "binary_only" => (binary_acc, binary_loss),
                "binary+gradient" => {
                    optimize_gradient(&mut net, op_add, 500);
                    (net.native_accuracy(op_add), net.mse_loss(op_add))
                }
                "binary+newton" => {
                    optimize_newton(&mut net, op_add, 30);
                    (net.native_accuracy(op_add), net.mse_loss(op_add))
                }
                "binary+perturbation" => {
                    optimize_perturbation(&mut net, op_add, 50_000, seed + 1000);
                    (net.native_accuracy(op_add), net.mse_loss(op_add))
                }
                _ => unreachable!()
            }
        }).collect();

        let best_acc = results.iter().map(|r| r.0).fold(0.0f64, f64::max);
        let mean_acc: f64 = results.iter().map(|r| r.0).sum::<f64>() / results.len() as f64;
        let mean_loss: f64 = results.iter().map(|r| r.1).sum::<f64>() / results.len() as f64;

        println!("{:<24} {:>7.0}% {:>7.0}% {:>10.4} {:>7.1}s",
            method, best_acc * 100.0, mean_acc * 100.0, mean_loss, t1.elapsed().as_secs_f64());
    }

    // =========================================================
    // EXP 2: All tasks — binary + best optimizer
    // =========================================================
    println!("\n--- EXP 2: All tasks — binary exhaustive + gradient (20 seeds) ---\n");
    println!("{:<8} {:>8} {:>8} {:>10} {:>8} {:>10}", "task", "best", "mean", "mean_loss", "solved", "time");
    println!("{}", "-".repeat(60));

    for &(task_name, task_op) in &tasks {
        let t1 = Instant::now();
        let results: Vec<(f64, f64)> = SEEDS.par_iter().map(|&seed| {
            let mut net = find_binary_start(n_connectome, n_workers, task_op, seed);
            optimize_gradient(&mut net, task_op, 1000);
            (net.native_accuracy(task_op), net.mse_loss(task_op))
        }).collect();

        let best_acc = results.iter().map(|r| r.0).fold(0.0f64, f64::max);
        let mean_acc: f64 = results.iter().map(|r| r.0).sum::<f64>() / results.len() as f64;
        let mean_loss: f64 = results.iter().map(|r| r.1).sum::<f64>() / results.len() as f64;
        let solved = results.iter().filter(|r| r.0 >= 1.0).count();

        println!("{:<8} {:>7.0}% {:>7.0}% {:>10.4} {:>5}/{:>2} {:>7.1}s",
            task_name, best_acc * 100.0, mean_acc * 100.0, mean_loss,
            solved, SEEDS.len(), t1.elapsed().as_secs_f64());
    }

    // =========================================================
    // EXP 3: Worker count sweep (binary + Newton)
    // =========================================================
    println!("\n--- EXP 3: Worker count sweep on |a-b| (binary + gradient) ---\n");
    println!("{:<10} {:>8} {:>8} {:>8} {:>8}", "workers", "best", "mean", "solved", "time");
    println!("{}", "-".repeat(45));

    for &nw in &[1, 2, 3, 4, 6, 8, 10, 15, 20] {
        let t1 = Instant::now();
        let results: Vec<f64> = SEEDS.par_iter().map(|&seed| {
            let mut net = find_binary_start(n_connectome, nw, op_sub_abs, seed);
            optimize_gradient(&mut net, op_sub_abs, 1000);
            net.native_accuracy(op_sub_abs)
        }).collect();

        let best = results.iter().fold(0.0f64, |a, &b| a.max(b));
        let mean: f64 = results.iter().sum::<f64>() / results.len() as f64;
        let solved = results.iter().filter(|&&r| r >= 1.0).count();

        println!("{:<10} {:>7.0}% {:>7.0}% {:>5}/{:>2} {:>7.1}s",
            nw, best * 100.0, mean * 100.0, solved, SEEDS.len(), t1.elapsed().as_secs_f64());
    }

    // =========================================================
    // EXP 4: Convergence trace — single seed, detailed
    // =========================================================
    println!("\n--- EXP 4: Convergence trace (ADD, seed=42) ---\n");

    let mut net_grad = find_binary_start(n_connectome, n_workers, op_add, 42);
    let mut net_newton = net_grad.clone();

    println!("  Binary start: acc={:.0}% loss={:.4}",
        net_grad.native_accuracy(op_add) * 100.0, net_grad.mse_loss(op_add));

    println!("\n  Gradient descent:");
    let hist_grad = optimize_gradient(&mut net_grad, op_add, 500);
    for (step, loss, acc) in &hist_grad {
        println!("    step {:>4}: loss={:.6} acc={:.0}%", step, loss, acc * 100.0);
    }

    println!("\n  Newton:");
    let hist_newton = optimize_newton(&mut net_newton, op_add, 30);
    for (step, loss, acc) in &hist_newton {
        println!("    step {:>4}: loss={:.6} acc={:.0}%", step, loss, acc * 100.0);
    }

    println!("\n=== DONE ({:.1}s) ===", t0.elapsed().as_secs_f64());
}
