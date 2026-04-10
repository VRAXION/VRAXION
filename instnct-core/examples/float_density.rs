//! Float solution density: random init + gradient → how often 100%?
//!
//! No binary search, no exhaustive — just random float weights + gradient.
//! Measures the "basin of attraction" size for each task.
//!
//! Run: cargo run --example float_density --release

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
            n_connectome, n_workers: 0,
            params: Vec::new(),
            worker_param_offsets: Vec::new(),
            worker_param_counts: Vec::new(),
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
                if write_idx < nc {
                    conn_charges[write_idx] += activations[nc + i] * w_write;
                }
            }
            for i in 0..nc { activations[i] = conn_charges[i]; }

            let old_act = activations.clone();
            for i in 0..nw {
                let off = self.worker_param_offsets[i];
                let nl = self.worker_local_counts[i];
                let mut sum = self.params[off + INPUT_DIM + nl + nc + 1]; // bias

                for j in 0..INPUT_DIM { sum += input[j] * self.params[off + j]; }

                let local_start = i.saturating_sub(nl);
                for (k, widx) in (local_start..i).enumerate() {
                    sum += old_act[nc + widx] * self.params[off + INPUT_DIM + k];
                }

                for k in 0..nc {
                    sum += old_act[k] * self.params[off + INPUT_DIM + nl + k];
                }

                activations[nc + i] = relu(sum);
            }
        }
        activations[nc..].iter().sum()
    }

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
}

fn optimize_gradient(net: &mut FlatNet, op: fn(usize, usize) -> usize, steps: usize) -> (f64, usize) {
    let mut lr = 0.01f32;
    for step in 0..steps {
        let acc = net.native_accuracy(op);
        if acc >= 1.0 { return (acc, step); }

        let grad = net.gradient(op);
        let grad_norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        if grad_norm < 1e-8 { return (acc, step); }

        let old_params = net.params.clone();
        let old_loss = net.mse_loss(op);

        for attempt in 0..5 {
            for i in 0..net.params.len() {
                net.params[i] = old_params[i] - lr * grad[i] / grad_norm;
            }
            let new_loss = net.mse_loss(op);
            if new_loss < old_loss { lr *= 1.1; break; }
            else { lr *= 0.5; if attempt == 4 { net.params = old_params.clone(); } }
        }
    }
    (net.native_accuracy(op), steps)
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn main() {
    let t0 = Instant::now();
    println!("=== FLOAT SOLUTION DENSITY ===");
    println!("Random float init + gradient descent → how often 100%?\n");

    let tasks: Vec<(&str, fn(usize, usize) -> usize)> = vec![
        ("ADD", op_add),
        ("MUL", op_mul),
        ("MAX", op_max),
        ("MIN", op_min),
        ("|a-b|", op_sub_abs),
    ];

    let n_connectome = 3;
    let n_seeds = 200;

    // =========================================================
    // Sweep: worker count × init scale × task
    // =========================================================
    for &n_workers in &[1, 2, 3, 4, 6] {
        println!("--- {} workers, {} connectome, {} seeds ---\n",
            n_workers, n_connectome, n_seeds);

        for &init_scale in &[0.5, 1.0, 2.0] {
            print!("  scale={:.1}: ", init_scale);

            for &(task_name, task_op) in &tasks {
                let seeds: Vec<u64> = (0..n_seeds).map(|i| i as u64 + 1).collect();

                let results: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    let mut net = FlatNet::new_random(n_connectome, n_workers, &mut rng, init_scale);
                    optimize_gradient(&mut net, task_op, 2000)
                }).collect();

                let solved = results.iter().filter(|r| r.0 >= 1.0).count();
                let mean_acc: f64 = results.iter().map(|r| r.0).sum::<f64>() / results.len() as f64;
                let mean_steps: f64 = results.iter().filter(|r| r.0 >= 1.0)
                    .map(|r| r.1 as f64).sum::<f64>()
                    / solved.max(1) as f64;

                print!("{}:{:>3}/{} ({:.0}% ~{:.0}st)  ",
                    task_name, solved, n_seeds, mean_acc * 100.0, mean_steps);
            }
            println!();
        }
        println!();
    }

    println!("=== DONE ({:.1}s) ===", t0.elapsed().as_secs_f64());
}
