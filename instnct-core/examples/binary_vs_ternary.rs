//! Binary vs Ternary: which weight set works for the connectome architecture?
//!
//! Quick test: passive 3-connectome, 10 workers, 20 seeds, ALL exhaustive.
//! Compare binary (±1) vs ternary (±1,0) on all 5 tasks.
//!
//! Run: cargo run --example binary_vs_ternary --release

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

struct NearestMean { centroids: Vec<f32> }
impl NearestMean {
    fn fit(examples: &[(f32, usize)], n_classes: usize) -> Self {
        let mut sums = vec![0.0f32; n_classes];
        let mut counts = vec![0usize; n_classes];
        for &(s, c) in examples { sums[c] += s; counts[c] += 1; }
        NearestMean {
            centroids: (0..n_classes).map(|c| {
                if counts[c] > 0 { sums[c] / counts[c] as f32 } else { f32::NAN }
            }).collect()
        }
    }
    fn predict(&self, s: f32) -> usize {
        self.centroids.iter().enumerate()
            .filter(|(_, c)| !c.is_nan())
            .min_by(|a, b| (a.1 - s).abs().partial_cmp(&(b.1 - s).abs()).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }
}

#[derive(Clone)]
struct Neuron {
    w_input: Vec<f32>,
    w_local: Vec<f32>,
    local_indices: Vec<usize>,
    w_conn_read: Vec<f32>,
    w_conn_write: f32,
    conn_write_idx: usize,
    bias: f32,
}

#[derive(Clone)]
struct ConnectomeNet {
    neurons: Vec<Neuron>,
    activations: Vec<f32>,
    n_connectome: usize,
}

impl ConnectomeNet {
    fn new(n_connectome: usize) -> Self {
        let mut net = ConnectomeNet {
            neurons: Vec::new(), activations: Vec::new(), n_connectome,
        };
        for _ in 0..n_connectome {
            net.neurons.push(Neuron {
                w_input: vec![0.0; INPUT_DIM], w_local: vec![], local_indices: vec![],
                w_conn_read: vec![], w_conn_write: 0.0, conn_write_idx: 0, bias: 0.0,
            });
            net.activations.push(0.0);
        }
        net
    }

    fn n_workers(&self) -> usize { self.neurons.len() - self.n_connectome }

    fn add_worker(&mut self, w_input: Vec<f32>, w_local: Vec<f32>, local_indices: Vec<usize>,
        w_conn_read: Vec<f32>, w_conn_write: f32, conn_write_idx: usize, bias: f32) {
        self.neurons.push(Neuron {
            w_input, w_local, local_indices, w_conn_read, w_conn_write,
            conn_write_idx: conn_write_idx % self.n_connectome.max(1), bias,
        });
        self.activations.push(0.0);
    }

    fn reset(&mut self) { for a in &mut self.activations { *a = 0.0; } }

    fn tick(&mut self, input: &[f32]) {
        let n = self.neurons.len();
        let nc = self.n_connectome;
        let mut conn_charges = vec![0.0f32; nc];
        for i in nc..n {
            let neuron = &self.neurons[i];
            if neuron.conn_write_idx < nc {
                conn_charges[neuron.conn_write_idx] += self.activations[i] * neuron.w_conn_write;
            }
        }
        for i in 0..nc { self.activations[i] = conn_charges[i]; }

        let old_act = self.activations.clone();
        for i in nc..n {
            let neuron = &self.neurons[i];
            let mut sum = neuron.bias;
            for (j, &w) in neuron.w_input.iter().enumerate() {
                if j < input.len() { sum += input[j] * w; }
            }
            for (k, &idx) in neuron.local_indices.iter().enumerate() {
                if idx < n && k < neuron.w_local.len() { sum += old_act[idx] * neuron.w_local[k]; }
            }
            for (k, &w) in neuron.w_conn_read.iter().enumerate() {
                if k < nc { sum += old_act[k] * w; }
            }
            self.activations[i] = relu(sum);
        }
    }

    fn eval_pair(&mut self, a: usize, b: usize) -> f32 {
        self.reset();
        let input = thermo_2(a, b);
        for _ in 0..TICKS { self.tick(&input); }
        self.activations[self.n_connectome..].iter().sum()
    }
}

fn eval_accuracy(net: &mut ConnectomeNet, op: fn(usize, usize) -> usize, n_classes: usize) -> f64 {
    let mut examples = Vec::with_capacity(25);
    for a in 0..DIGITS {
        for b in 0..DIGITS {
            let target = op(a, b);
            let charge = net.eval_pair(a, b);
            if charge.is_nan() || charge.is_infinite() { return 0.0; }
            examples.push((charge, target));
        }
    }
    let readout = NearestMean::fit(&examples, n_classes);
    examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / 25.0
}

fn incremental_build(
    n_connectome: usize, max_workers: usize,
    op: fn(usize, usize) -> usize, n_classes: usize,
    seed: u64, weight_vals: &[f32],
) -> (f64, usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let base = weight_vals.len() as u64;
    let mut best_net = ConnectomeNet::new(n_connectome);
    let mut best_acc = eval_accuracy(&mut best_net, op, n_classes);

    for worker_i in 0..max_workers {
        if best_acc >= 1.0 { return (best_acc, worker_i); }

        let n_local = LOCAL_CAP.min(best_net.n_workers());
        let n_params = INPUT_DIM + n_local + n_connectome + 1 + 1;
        let total_configs = base.saturating_pow(n_params as u32);

        // Always exhaustive if feasible, else random
        let use_exhaustive = total_configs <= 50_000_000;
        let sample_count = if use_exhaustive { total_configs } else { 5_000_000 };

        let worker_start = n_connectome;
        let total_workers = best_net.n_workers();
        let local_indices: Vec<usize> = if total_workers == 0 { vec![] }
            else {
                let start = worker_start + total_workers.saturating_sub(n_local);
                (start..worker_start + total_workers).collect()
            };

        let mut worker_best_acc = best_acc;
        let mut worker_best_params: Option<(Vec<f32>, Vec<f32>, Vec<f32>, f32, f32)> = None;

        for sample in 0..sample_count {
            let mut c = if use_exhaustive { sample } else { rng.gen_range(0..total_configs) };
            let w_input: Vec<f32> = (0..INPUT_DIM).map(|_| { let v = weight_vals[(c % base) as usize]; c /= base; v }).collect();
            let w_local: Vec<f32> = (0..n_local).map(|_| { let v = weight_vals[(c % base) as usize]; c /= base; v }).collect();
            let w_conn_read: Vec<f32> = (0..n_connectome).map(|_| { let v = weight_vals[(c % base) as usize]; c /= base; v }).collect();
            let w_conn_write = weight_vals[(c % base) as usize]; c /= base;
            let bias = weight_vals[(c % base) as usize];
            let conn_write_idx = worker_i % n_connectome.max(1);

            let mut test_net = best_net.clone();
            test_net.add_worker(w_input.clone(), w_local.clone(), local_indices.clone(),
                w_conn_read.clone(), w_conn_write, conn_write_idx, bias);
            let acc = eval_accuracy(&mut test_net, op, n_classes);
            if acc > worker_best_acc {
                worker_best_acc = acc;
                worker_best_params = Some((w_input, w_local, w_conn_read, w_conn_write, bias));
            }
            if worker_best_acc >= 1.0 { break; }
        }

        if let Some((w_input, w_local, w_conn_read, w_conn_write, bias)) = worker_best_params {
            let conn_write_idx = worker_i % n_connectome.max(1);
            best_net.add_worker(w_input, w_local, local_indices, w_conn_read, w_conn_write, conn_write_idx, bias);
        } else {
            best_net.add_worker(
                vec![0.0; INPUT_DIM], vec![0.0; n_local], local_indices,
                vec![0.0; n_connectome], 0.0, 0, 0.0);
        }
        best_acc = worker_best_acc;
    }

    (best_acc, max_workers)
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

const SEEDS: &[u64] = &[42, 123, 777, 314, 999, 1337, 2024, 55, 101, 202, 303, 404, 505, 606, 707, 808, 9001, 1234, 5678, 31415];

fn main() {
    let t0 = Instant::now();
    println!("=== BINARY vs TERNARY: CONNECTOME ARCHITECTURE ===");
    println!("Passive 3-connectome, LOCAL_CAP={}, TICKS={}, 10 workers, {} seeds\n",
        LOCAL_CAP, TICKS, SEEDS.len());

    let tasks: Vec<(&str, fn(usize, usize) -> usize, usize)> = vec![
        ("ADD", op_add, 9),
        ("MUL", op_mul, 17),
        ("MAX", op_max, 5),
        ("MIN", op_min, 5),
        ("|a-b|", op_sub_abs, 5),
    ];

    let weight_sets: Vec<(&str, Vec<f32>)> = vec![
        ("binary ±1", vec![-1.0, 1.0]),
        ("ternary ±1,0", vec![-1.0, 0.0, 1.0]),
    ];

    println!("{:<16} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "weights", "ADD", "MUL", "MAX", "MIN", "|a-b|");
    println!("{}", "=".repeat(70));

    for (w_name, w_vals) in &weight_sets {
        let t1 = Instant::now();
        print!("{:<16}", w_name);

        for &(task_name, task_op, n_classes) in &tasks {
            let results: Vec<f64> = SEEDS.par_iter().map(|&seed| {
                let (acc, _) = incremental_build(3, 10, task_op, n_classes, seed, w_vals);
                acc
            }).collect();

            let best = results.iter().fold(0.0f64, |a, &b| a.max(b));
            let mean: f64 = results.iter().sum::<f64>() / results.len() as f64;
            let solved = results.iter().filter(|&&r| r >= 1.0).count();
            print!(" {:>3.0}%/{:.0}%", best * 100.0, mean * 100.0);
        }

        println!("  ({:.1}s)", t1.elapsed().as_secs_f64());
    }

    // Also test: binary with BIGGER neighborhoods
    println!("\n--- Binary with bigger neighborhoods ---\n");
    println!("{:<24} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6}",
        "config", "ADD", "MUL", "MAX", "MIN", "|a-b|", "time");
    println!("{}", "=".repeat(80));

    let binary = vec![-1.0f32, 1.0];
    let configs: Vec<(&str, usize, usize)> = vec![
        ("L=3 C=3 (baseline)", 3, 3),
        ("L=5 C=3", 5, 3),
        ("L=3 C=6", 3, 6),
        ("L=5 C=6", 5, 6),
        ("L=8 C=6", 8, 6),
        ("L=8 C=8", 8, 8),
    ];

    for (label, local_cap, n_conn) in &configs {
        let t1 = Instant::now();
        print!("{:<24}", label);

        for &(_task_name, task_op, n_classes) in &tasks {
            let lc = *local_cap;
            let nc = *n_conn;

            let results: Vec<f64> = SEEDS.par_iter().map(|&seed| {
                // Custom build with different LOCAL_CAP and n_connectome
                let w_vals = &binary;
                let base = w_vals.len() as u64;
                let mut rng = StdRng::seed_from_u64(seed);
                let mut best_net = ConnectomeNet::new(nc);
                let mut best_acc = eval_accuracy(&mut best_net, task_op, n_classes);

                for worker_i in 0..10 {
                    if best_acc >= 1.0 { break; }
                    let n_local = lc.min(best_net.n_workers());
                    let n_params = INPUT_DIM + n_local + nc + 1 + 1;
                    let total_configs = base.saturating_pow(n_params as u32);
                    let use_exhaustive = total_configs <= 50_000_000;
                    let sample_count = if use_exhaustive { total_configs } else { 5_000_000 };

                    let worker_start = nc;
                    let total_workers = best_net.n_workers();
                    let local_indices: Vec<usize> = if total_workers == 0 { vec![] }
                        else {
                            let start = worker_start + total_workers.saturating_sub(n_local);
                            (start..worker_start + total_workers).collect()
                        };

                    let mut worker_best_acc = best_acc;
                    let mut worker_best_params: Option<(Vec<f32>, Vec<f32>, Vec<f32>, f32, f32)> = None;

                    for sample in 0..sample_count {
                        let mut c = if use_exhaustive { sample } else { rng.gen_range(0..total_configs) };
                        let w_input: Vec<f32> = (0..INPUT_DIM).map(|_| { let v = w_vals[(c % base) as usize]; c /= base; v }).collect();
                        let w_local: Vec<f32> = (0..n_local).map(|_| { let v = w_vals[(c % base) as usize]; c /= base; v }).collect();
                        let w_conn_read: Vec<f32> = (0..nc).map(|_| { let v = w_vals[(c % base) as usize]; c /= base; v }).collect();
                        let w_conn_write = w_vals[(c % base) as usize]; c /= base;
                        let bias = w_vals[(c % base) as usize];
                        let conn_write_idx = worker_i % nc.max(1);

                        let mut test_net = best_net.clone();
                        test_net.add_worker(w_input.clone(), w_local.clone(), local_indices.clone(),
                            w_conn_read.clone(), w_conn_write, conn_write_idx, bias);
                        let acc = eval_accuracy(&mut test_net, task_op, n_classes);
                        if acc > worker_best_acc {
                            worker_best_acc = acc;
                            worker_best_params = Some((w_input, w_local, w_conn_read, w_conn_write, bias));
                        }
                        if worker_best_acc >= 1.0 { break; }
                    }

                    if let Some((w_input, w_local, w_conn_read, w_conn_write, bias)) = worker_best_params {
                        let conn_write_idx = worker_i % nc.max(1);
                        best_net.add_worker(w_input, w_local, local_indices, w_conn_read, w_conn_write, conn_write_idx, bias);
                    } else {
                        best_net.add_worker(
                            vec![0.0; INPUT_DIM], vec![0.0; n_local], local_indices,
                            vec![0.0; nc], 0.0, 0, 0.0);
                    }
                    best_acc = worker_best_acc;
                }
                best_acc
            }).collect();

            let best = results.iter().fold(0.0f64, |a, &b| a.max(b));
            let mean: f64 = results.iter().sum::<f64>() / results.len() as f64;
            print!(" {:>3.0}%/{:.0}%", best * 100.0, mean * 100.0);
        }
        println!(" {:>5.1}s", t1.elapsed().as_secs_f64());
    }

    println!("\n=== DONE ({:.1}s) ===", t0.elapsed().as_secs_f64());
}
