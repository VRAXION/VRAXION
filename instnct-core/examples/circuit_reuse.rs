//! Circuit Reuse Test: does a pre-trained ADD circuit get automatically used?
//!
//! Phase A: Train 2 workers on ADD(a,b), freeze
//! Phase B: Train NEW workers on tasks that NEED addition:
//!   - a + b + c  (3-input addition, needs to chain ADD)
//!   - 2a + b     (double + add)
//!   - max(a+b, c) (add then compare)
//!   - (a+b) > 3  (add then threshold)
//!
//! Compare: pre-trained ADD vs fresh start (no pre-training)
//!
//! Run: cargo run --example circuit_reuse --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const DIGITS: usize = 5; // 0..4
const TICKS: usize = 3;  // need more ticks for info to flow through frozen → new

fn relu(x: f32) -> f32 { x.max(0.0) }

fn thermo(val: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 4];
    for i in 0..val.min(4) { v[i] = 1.0; }
    v
}

fn thermo_3(a: usize, b: usize, c: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 12];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    for i in 0..c.min(4) { v[8 + i] = 1.0; }
    v
}

// ============================================================
// Network with frozen + trainable zones
// ============================================================
#[derive(Clone)]
struct CircuitNet {
    n_connectome: usize,
    // Frozen workers (pre-trained, weights don't change)
    frozen_params: Vec<Vec<f32>>,
    frozen_local_counts: Vec<usize>,
    // Trainable workers (new, gradient-optimized)
    trainable_params: Vec<f32>,  // flat
    trainable_offsets: Vec<usize>,
    trainable_local_counts: Vec<usize>,
    n_frozen: usize,
    n_trainable: usize,
    input_dim: usize,
}

impl CircuitNet {
    fn new(n_connectome: usize, input_dim: usize) -> Self {
        CircuitNet {
            n_connectome, frozen_params: Vec::new(), frozen_local_counts: Vec::new(),
            trainable_params: Vec::new(), trainable_offsets: Vec::new(),
            trainable_local_counts: Vec::new(),
            n_frozen: 0, n_trainable: 0, input_dim,
        }
    }

    fn total_workers(&self) -> usize { self.n_frozen + self.n_trainable }

    fn add_frozen_worker(&mut self, params: Vec<f32>, n_local: usize) {
        self.frozen_params.push(params);
        self.frozen_local_counts.push(n_local);
        self.n_frozen += 1;
    }

    fn add_trainable_worker(&mut self, params: Vec<f32>, n_local: usize) {
        self.trainable_offsets.push(self.trainable_params.len());
        self.trainable_local_counts.push(n_local);
        self.trainable_params.extend_from_slice(&params);
        self.n_trainable += 1;
    }

    fn forward(&self, input: &[f32]) -> f32 {
        let nc = self.n_connectome;
        let nf = self.n_frozen;
        let nt = self.n_trainable;
        let total = nc + nf + nt;
        let mut act = vec![0.0f32; total];

        for _tick in 0..TICKS {
            // Connectome: passive relay from all workers
            let mut cc = vec![0.0f32; nc];
            for i in 0..nf {
                let p = &self.frozen_params[i];
                let nl = self.frozen_local_counts[i];
                let w_write = p[self.input_dim + nl + nc];
                let wi = i % nc.max(1);
                if wi < nc { cc[wi] += act[nc + i] * w_write; }
            }
            for i in 0..nt {
                let off = self.trainable_offsets[i];
                let nl = self.trainable_local_counts[i];
                let w_write = self.trainable_params[off + self.input_dim + nl + nc];
                let wi = (nf + i) % nc.max(1);
                if wi < nc { cc[wi] += act[nc + nf + i] * w_write; }
            }
            for i in 0..nc { act[i] = cc[i]; }

            let old = act.clone();

            // Frozen workers
            for i in 0..nf {
                let p = &self.frozen_params[i];
                let nl = self.frozen_local_counts[i];
                let mut s = p[self.input_dim + nl + nc + 1]; // bias
                // Input
                for j in 0..self.input_dim.min(p.len()) {
                    if j < input.len() { s += input[j] * p[j]; }
                }
                // Local
                let ls = (nc + i).saturating_sub(nl);
                for (k, wi) in (ls..nc + i).enumerate() {
                    if k < nl && wi < total { s += old[wi] * p[self.input_dim + k]; }
                }
                // Connectome read
                for k in 0..nc {
                    s += old[k] * p[self.input_dim + nl + k];
                }
                act[nc + i] = relu(s);
            }

            // Trainable workers
            for i in 0..nt {
                let off = self.trainable_offsets[i];
                let nl = self.trainable_local_counts[i];
                let mut s = self.trainable_params[off + self.input_dim + nl + nc + 1]; // bias
                // Input
                for j in 0..self.input_dim {
                    if j < input.len() {
                        s += input[j] * self.trainable_params[off + j];
                    }
                }
                // Local (can see frozen workers too!)
                let global_idx = nc + nf + i;
                let ls = global_idx.saturating_sub(nl);
                for (k, wi) in (ls..global_idx).enumerate() {
                    if k < nl && wi < total {
                        s += old[wi] * self.trainable_params[off + self.input_dim + k];
                    }
                }
                // Connectome read
                for k in 0..nc {
                    s += old[k] * self.trainable_params[off + self.input_dim + nl + k];
                }
                act[nc + nf + i] = relu(s);
            }
        }

        // Output: sum of ALL worker activations (frozen + trainable)
        act[nc..].iter().sum()
    }

    fn mse_loss_3input(&self, op: fn(usize, usize, usize) -> usize) -> f64 {
        let mut loss = 0.0f64;
        let mut count = 0;
        for a in 0..DIGITS {
            for b in 0..DIGITS {
                for c in 0..DIGITS {
                    let input = thermo_3(a, b, c);
                    let charge = self.forward(&input) as f64;
                    let target = op(a, b, c) as f64;
                    loss += (charge - target).powi(2);
                    count += 1;
                }
            }
        }
        loss / count as f64
    }

    fn accuracy_3input(&self, op: fn(usize, usize, usize) -> usize) -> f64 {
        let mut correct = 0;
        let mut count = 0;
        for a in 0..DIGITS {
            for b in 0..DIGITS {
                for c in 0..DIGITS {
                    let input = thermo_3(a, b, c);
                    let charge = self.forward(&input);
                    if (charge.round() as i32) == (op(a, b, c) as i32) { correct += 1; }
                    count += 1;
                }
            }
        }
        correct as f64 / count as f64
    }

    /// Gradient on trainable params only
    fn gradient_3input(&mut self, op: fn(usize, usize, usize) -> usize) -> Vec<f32> {
        let eps = 1e-3f32;
        let n = self.trainable_params.len();
        let mut grad = vec![0.0f32; n];
        for i in 0..n {
            let orig = self.trainable_params[i];
            self.trainable_params[i] = orig + eps;
            let lp = self.mse_loss_3input(op);
            self.trainable_params[i] = orig - eps;
            let lm = self.mse_loss_3input(op);
            self.trainable_params[i] = orig;
            grad[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        grad
    }
}

fn optimize_trainable(net: &mut CircuitNet, op: fn(usize, usize, usize) -> usize, steps: usize) -> (f64, usize) {
    let mut lr = 0.01f32;
    for step in 0..steps {
        let acc = net.accuracy_3input(op);
        if acc >= 1.0 { return (acc, step); }

        let grad = net.gradient_3input(op);
        let gn: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        if gn < 1e-8 { return (acc, step); }

        let old = net.trainable_params.clone();
        let ol = net.mse_loss_3input(op);
        for att in 0..5 {
            for i in 0..net.trainable_params.len() {
                net.trainable_params[i] = old[i] - lr * grad[i] / gn;
            }
            if net.mse_loss_3input(op) < ol { lr *= 1.1; break; }
            else { lr *= 0.5; if att == 4 { net.trainable_params = old.clone(); } }
        }
    }
    (net.accuracy_3input(op), steps)
}

// ============================================================
// Phase A: Train ADD(a,b) circuit
// ============================================================
fn train_add_circuit(nc: usize, seed: u64) -> Vec<(Vec<f32>, usize)> {
    let input_dim = 12; // thermo_3 but we only use first 8
    let mut rng = StdRng::seed_from_u64(seed);

    // 2 workers for ADD(a,b)
    let mut best_params: Vec<(Vec<f32>, usize)> = Vec::new();

    // Simple: build a tiny network, train on a+b (ignore c)
    let mut net = CircuitNet::new(nc, input_dim);

    for wi in 0..2 {
        let nl = 3usize.min(wi);
        let np = input_dim + nl + nc + 1 + 1;
        let init: Vec<f32> = (0..np).map(|_| rng.gen_range(-0.5..0.5f32)).collect();
        net.add_trainable_worker(init, nl);
    }

    // Train on a+b (c ignored)
    let op_ab = |a: usize, b: usize, _c: usize| -> usize { a + b };
    optimize_trainable(&mut net, op_ab, 2000);

    // Extract trained params
    for i in 0..2 {
        let off = net.trainable_offsets[i];
        let nl = net.trainable_local_counts[i];
        let np = input_dim + nl + nc + 1 + 1;
        let params = net.trainable_params[off..off + np].to_vec();
        best_params.push((params, nl));
    }

    best_params
}

// ============================================================
// Tasks
// ============================================================
fn op_abc_sum(a: usize, b: usize, c: usize) -> usize { a + b + c }
fn op_2a_plus_b(a: usize, b: usize, _c: usize) -> usize { 2 * a + b }
fn op_max_ab_c(a: usize, b: usize, c: usize) -> usize { (a + b).max(c) }
fn op_ab_gt3(a: usize, b: usize, _c: usize) -> usize { if a + b > 3 { 1 } else { 0 } }

const SEEDS: &[u64] = &[42, 123, 777, 314, 999, 1337, 2024, 55, 101, 202,
                         303, 404, 505, 606, 707, 808, 9001, 1234, 5678, 31415];

fn main() {
    let t0 = Instant::now();
    println!("=== CIRCUIT REUSE TEST ===");
    println!("Does a pre-trained ADD circuit get automatically used?\n");

    let nc = 3;
    let input_dim = 12;
    let n_new_workers = 4; // new workers to add on top of frozen circuit

    let tasks: Vec<(&str, fn(usize, usize, usize) -> usize)> = vec![
        ("a+b+c", op_abc_sum),
        ("2a+b", op_2a_plus_b),
        ("max(a+b,c)", op_max_ab_c),
        ("(a+b)>3", op_ab_gt3),
    ];

    println!("{:<14} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "task", "WITH_ADD", "steps", "NO_ADD", "steps", "speedup");
    println!("{}", "=".repeat(70));

    for &(task_name, task_op) in &tasks {
        // =====================================================
        // A: WITH pre-trained ADD circuit (frozen)
        // =====================================================
        let results_with: Vec<(f64, usize)> = SEEDS.par_iter().map(|&seed| {
            // Train ADD circuit
            let add_circuit = train_add_circuit(nc, seed);

            // Build network with frozen ADD + new trainable workers
            let mut net = CircuitNet::new(nc, input_dim);

            // Add frozen ADD workers
            for (params, nl) in &add_circuit {
                net.add_frozen_worker(params.clone(), *nl);
            }

            // Add new trainable workers
            let mut rng = StdRng::seed_from_u64(seed + 10000);
            for i in 0..n_new_workers {
                let nl = 3usize.min(net.total_workers());
                let np = input_dim + nl + nc + 1 + 1;
                let init: Vec<f32> = (0..np).map(|_| rng.gen_range(-0.5..0.5f32)).collect();
                net.add_trainable_worker(init, nl);
            }

            optimize_trainable(&mut net, task_op, 2000)
        }).collect();

        // =====================================================
        // B: WITHOUT pre-trained ADD (fresh start, same total workers)
        // =====================================================
        let total_workers = 2 + n_new_workers; // same total neuron count

        let results_without: Vec<(f64, usize)> = SEEDS.par_iter().map(|&seed| {
            let mut net = CircuitNet::new(nc, input_dim);
            let mut rng = StdRng::seed_from_u64(seed);

            for i in 0..total_workers {
                let nl = 3usize.min(i);
                let np = input_dim + nl + nc + 1 + 1;
                let init: Vec<f32> = (0..np).map(|_| rng.gen_range(-0.5..0.5f32)).collect();
                net.add_trainable_worker(init, nl);
            }

            optimize_trainable(&mut net, task_op, 2000)
        }).collect();

        // Stats
        let with_solved = results_with.iter().filter(|r| r.0 >= 1.0).count();
        let with_mean_acc: f64 = results_with.iter().map(|r| r.0).sum::<f64>() / results_with.len() as f64;
        let with_mean_steps: f64 = results_with.iter().filter(|r| r.0 >= 1.0)
            .map(|r| r.1 as f64).sum::<f64>() / with_solved.max(1) as f64;

        let without_solved = results_without.iter().filter(|r| r.0 >= 1.0).count();
        let without_mean_acc: f64 = results_without.iter().map(|r| r.0).sum::<f64>() / results_without.len() as f64;
        let without_mean_steps: f64 = results_without.iter().filter(|r| r.0 >= 1.0)
            .map(|r| r.1 as f64).sum::<f64>() / without_solved.max(1) as f64;

        let speedup = if with_mean_steps > 0.0 && without_mean_steps > 0.0 {
            without_mean_steps / with_mean_steps
        } else { 0.0 };

        println!("{:<14} {:>4}/{:>2} ({:.0}%) {:>5.0}st  {:>4}/{:>2} ({:.0}%) {:>5.0}st  {:.1}x",
            task_name,
            with_solved, SEEDS.len(), with_mean_acc * 100.0, with_mean_steps,
            without_solved, SEEDS.len(), without_mean_acc * 100.0, without_mean_steps,
            speedup);
    }

    // =====================================================
    // Detailed trace: a+b+c with vs without, single seed
    // =====================================================
    println!("\n--- Detailed: a+b+c (seed=42) ---\n");

    // WITH ADD
    let add_circuit = train_add_circuit(nc, 42);
    let add_acc = {
        let mut test = CircuitNet::new(nc, input_dim);
        for (p, nl) in &add_circuit { test.add_frozen_worker(p.clone(), *nl); }
        // Dummy trainable for eval
        test.accuracy_3input(|a, b, _c| a + b)
    };
    println!("  Frozen ADD circuit accuracy on a+b: {:.0}%", add_acc * 100.0);

    let mut net_with = CircuitNet::new(nc, input_dim);
    for (p, nl) in &add_circuit { net_with.add_frozen_worker(p.clone(), *nl); }
    let mut rng = StdRng::seed_from_u64(10042);
    for i in 0..n_new_workers {
        let nl = 3usize.min(net_with.total_workers());
        let np = input_dim + nl + nc + 1 + 1;
        let init: Vec<f32> = (0..np).map(|_| rng.gen_range(-0.5..0.5f32)).collect();
        net_with.add_trainable_worker(init, nl);
    }

    println!("\n  WITH frozen ADD (training new workers on a+b+c):");
    let mut lr = 0.01f32;
    for step in 0..2000 {
        let acc = net_with.accuracy_3input(op_abc_sum);
        if step % 100 == 0 || acc >= 1.0 {
            let loss = net_with.mse_loss_3input(op_abc_sum);
            println!("    step {:>4}: acc={:.0}% loss={:.4}", step, acc * 100.0, loss);
        }
        if acc >= 1.0 { break; }

        let grad = net_with.gradient_3input(op_abc_sum);
        let gn: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        if gn < 1e-8 { break; }
        let old = net_with.trainable_params.clone();
        let ol = net_with.mse_loss_3input(op_abc_sum);
        for att in 0..5 {
            for i in 0..net_with.trainable_params.len() {
                net_with.trainable_params[i] = old[i] - lr * grad[i] / gn;
            }
            if net_with.mse_loss_3input(op_abc_sum) < ol { lr *= 1.1; break; }
            else { lr *= 0.5; if att == 4 { net_with.trainable_params = old.clone(); } }
        }
    }

    // WITHOUT ADD
    let mut net_without = CircuitNet::new(nc, input_dim);
    let mut rng2 = StdRng::seed_from_u64(42);
    for i in 0..6 {
        let nl = 3usize.min(i);
        let np = input_dim + nl + nc + 1 + 1;
        let init: Vec<f32> = (0..np).map(|_| rng2.gen_range(-0.5..0.5f32)).collect();
        net_without.add_trainable_worker(init, nl);
    }

    println!("\n  WITHOUT frozen ADD (training all 6 workers on a+b+c):");
    let mut lr2 = 0.01f32;
    for step in 0..2000 {
        let acc = net_without.accuracy_3input(op_abc_sum);
        if step % 100 == 0 || acc >= 1.0 {
            let loss = net_without.mse_loss_3input(op_abc_sum);
            println!("    step {:>4}: acc={:.0}% loss={:.4}", step, acc * 100.0, loss);
        }
        if acc >= 1.0 { break; }

        let grad = net_without.gradient_3input(op_abc_sum);
        let gn: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        if gn < 1e-8 { break; }
        let old = net_without.trainable_params.clone();
        let ol = net_without.mse_loss_3input(op_abc_sum);
        for att in 0..5 {
            for i in 0..net_without.trainable_params.len() {
                net_without.trainable_params[i] = old[i] - lr2 * grad[i] / gn;
            }
            if net_without.mse_loss_3input(op_abc_sum) < ol { lr2 *= 1.1; break; }
            else { lr2 *= 0.5; if att == 4 { net_without.trainable_params = old.clone(); } }
        }
    }

    println!("\n=== DONE ({:.1}s) ===", t0.elapsed().as_secs_f64());
}
