//! Deep connectome experiments: solve |a-b| reliably + explore design space.
//!
//! Experiments:
//!   1. More workers (20) + more seeds (20) on |a-b|
//!   2. Perturbation refinement after incremental build
//!   3. Connectome size sweep (1, 2, 3, 4, 6)
//!   4. Tick sweep (1, 2, 3, 4)
//!   5. All 5 tasks with best config
//!
//! Run: cargo run --example connectome_deep --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const LOCAL_CAP: usize = 3;

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

// ============================================================
// Network
// ============================================================
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
    ticks: usize,
}

impl ConnectomeNet {
    fn new(n_connectome: usize, ticks: usize) -> Self {
        let mut net = ConnectomeNet {
            neurons: Vec::new(),
            activations: Vec::new(),
            n_connectome,
            ticks,
        };
        for _ in 0..n_connectome {
            net.neurons.push(Neuron {
                w_input: vec![0.0; INPUT_DIM],
                w_local: vec![],
                local_indices: vec![],
                w_conn_read: vec![],
                w_conn_write: 0.0,
                conn_write_idx: 0,
                bias: 0.0,
            });
            net.activations.push(0.0);
        }
        net
    }

    fn n_workers(&self) -> usize { self.neurons.len() - self.n_connectome }

    fn add_worker(
        &mut self, w_input: Vec<f32>, w_local: Vec<f32>, local_indices: Vec<usize>,
        w_conn_read: Vec<f32>, w_conn_write: f32, conn_write_idx: usize, bias: f32,
    ) {
        self.neurons.push(Neuron {
            w_input, w_local, local_indices, w_conn_read, w_conn_write,
            conn_write_idx: conn_write_idx % self.n_connectome.max(1),
            bias,
        });
        self.activations.push(0.0);
    }

    fn reset(&mut self) {
        for a in &mut self.activations { *a = 0.0; }
    }

    fn tick(&mut self, input: &[f32]) {
        let n = self.neurons.len();
        let nc = self.n_connectome;

        // Passive connectome: accumulate from workers
        let mut conn_charges = vec![0.0f32; nc];
        for i in nc..n {
            let neuron = &self.neurons[i];
            let idx = neuron.conn_write_idx;
            if idx < nc { conn_charges[idx] += self.activations[i] * neuron.w_conn_write; }
        }
        for i in 0..nc { self.activations[i] = conn_charges[i]; }

        // Workers
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
        for _ in 0..self.ticks { self.tick(&input); }
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

// ============================================================
// Incremental build
// ============================================================
fn incremental_build(
    n_connectome: usize, ticks: usize, max_workers: usize,
    op: fn(usize, usize) -> usize, n_classes: usize, seed: u64,
    samples_per_worker: u64,
) -> (f64, usize, ConnectomeNet) {
    let ternary: Vec<f32> = vec![-1.0, 0.0, 1.0];
    let mut rng = StdRng::seed_from_u64(seed);

    let mut best_net = ConnectomeNet::new(n_connectome, ticks);
    let mut best_acc = eval_accuracy(&mut best_net, op, n_classes);

    for worker_i in 0..max_workers {
        if best_acc >= 1.0 { return (best_acc, worker_i, best_net); }

        let n_local = LOCAL_CAP.min(best_net.n_workers());
        let n_params = INPUT_DIM + n_local + n_connectome + 1 + 1;
        let total_configs = 3u64.saturating_pow(n_params as u32);
        let use_exhaustive = total_configs <= samples_per_worker;
        let sample_count = if use_exhaustive { total_configs } else { samples_per_worker };

        let worker_start = n_connectome;
        let total_workers = best_net.n_workers();
        let local_indices: Vec<usize> = if total_workers == 0 { vec![] }
            else {
                let start = worker_start + total_workers.saturating_sub(n_local);
                (start..worker_start + total_workers).collect()
            };

        let mut worker_best_acc = best_acc;
        let mut worker_best_params: Option<(Vec<f32>, Vec<f32>, Vec<f32>, f32, usize, f32)> = None;

        for sample in 0..sample_count {
            let mut c = if use_exhaustive { sample } else { rng.gen_range(0..total_configs) };

            let w_input: Vec<f32> = (0..INPUT_DIM).map(|_| { let v = ternary[(c % 3) as usize]; c /= 3; v }).collect();
            let w_local: Vec<f32> = (0..n_local).map(|_| { let v = ternary[(c % 3) as usize]; c /= 3; v }).collect();
            let w_conn_read: Vec<f32> = (0..n_connectome).map(|_| { let v = ternary[(c % 3) as usize]; c /= 3; v }).collect();
            let w_conn_write = ternary[(c % 3) as usize]; c /= 3;
            let bias = ternary[(c % 3) as usize];
            let conn_write_idx = worker_i % n_connectome.max(1);

            let mut test_net = best_net.clone();
            test_net.add_worker(
                w_input.clone(), w_local.clone(), local_indices.clone(),
                w_conn_read.clone(), w_conn_write, conn_write_idx, bias,
            );

            let acc = eval_accuracy(&mut test_net, op, n_classes);
            if acc > worker_best_acc {
                worker_best_acc = acc;
                worker_best_params = Some((w_input, w_local, w_conn_read, w_conn_write, conn_write_idx, bias));
            }
            if worker_best_acc >= 1.0 { break; }
        }

        if let Some((w_input, w_local, w_conn_read, w_conn_write, conn_write_idx, bias)) = worker_best_params {
            best_net.add_worker(w_input, w_local, local_indices, w_conn_read, w_conn_write, conn_write_idx, bias);
        } else {
            best_net.add_worker(
                vec![0.0; INPUT_DIM], vec![0.0; n_local], local_indices,
                vec![0.0; n_connectome], 0.0, 0, 0.0,
            );
        }
        best_acc = worker_best_acc;
    }

    (best_acc, max_workers, best_net)
}

// ============================================================
// Perturbation refinement
// ============================================================
fn perturbation_refine(
    net: &mut ConnectomeNet, op: fn(usize, usize) -> usize,
    n_classes: usize, steps: u64, seed: u64,
) -> f64 {
    let nc = net.n_connectome;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut current_acc = eval_accuracy(net, op, n_classes);

    // Collect all mutable param locations: (neuron_idx, param_type, sub_idx)
    // We only perturb worker neurons (nc..)
    let mut param_locs: Vec<(usize, u8, usize)> = Vec::new(); // (neuron, type, idx)
    for i in nc..net.neurons.len() {
        let n = &net.neurons[i];
        for j in 0..n.w_input.len() { param_locs.push((i, 0, j)); }
        for j in 0..n.w_local.len() { param_locs.push((i, 1, j)); }
        for j in 0..n.w_conn_read.len() { param_locs.push((i, 2, j)); }
        param_locs.push((i, 3, 0)); // w_conn_write
        param_locs.push((i, 4, 0)); // bias
    }

    if param_locs.is_empty() { return current_acc; }

    for _ in 0..steps {
        if current_acc >= 1.0 { break; }

        let loc = &param_locs[rng.gen_range(0..param_locs.len())];
        let delta: f32 = rng.gen_range(-0.5..0.5);

        // Apply perturbation
        let old_val = match loc.1 {
            0 => { let v = net.neurons[loc.0].w_input[loc.2]; net.neurons[loc.0].w_input[loc.2] += delta; v }
            1 => { let v = net.neurons[loc.0].w_local[loc.2]; net.neurons[loc.0].w_local[loc.2] += delta; v }
            2 => { let v = net.neurons[loc.0].w_conn_read[loc.2]; net.neurons[loc.0].w_conn_read[loc.2] += delta; v }
            3 => { let v = net.neurons[loc.0].w_conn_write; net.neurons[loc.0].w_conn_write += delta; v }
            4 => { let v = net.neurons[loc.0].bias; net.neurons[loc.0].bias += delta; v }
            _ => unreachable!(),
        };

        let new_acc = eval_accuracy(net, op, n_classes);
        if new_acc >= current_acc {
            current_acc = new_acc;
        } else {
            // Revert
            match loc.1 {
                0 => net.neurons[loc.0].w_input[loc.2] = old_val,
                1 => net.neurons[loc.0].w_local[loc.2] = old_val,
                2 => net.neurons[loc.0].w_conn_read[loc.2] = old_val,
                3 => net.neurons[loc.0].w_conn_write = old_val,
                4 => net.neurons[loc.0].bias = old_val,
                _ => unreachable!(),
            }
        }
    }

    current_acc
}

// ============================================================
// Tasks
// ============================================================
fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

struct Task { name: &'static str, op: fn(usize, usize) -> usize, n_classes: usize }
fn make_tasks() -> Vec<Task> {
    vec![
        Task { name: "ADD", op: op_add, n_classes: 9 },
        Task { name: "MUL", op: op_mul, n_classes: 17 },
        Task { name: "MAX", op: op_max, n_classes: 5 },
        Task { name: "MIN", op: op_min, n_classes: 5 },
        Task { name: "|a-b|", op: op_sub_abs, n_classes: 5 },
    ]
}

const SEEDS: &[u64] = &[
    42, 123, 777, 314, 999, 1337, 2024, 55,
    101, 202, 303, 404, 505, 606, 707, 808,
    9001, 1234, 5678, 31415,
]; // 20 seeds

fn main() {
    let t0 = Instant::now();
    println!("=== CONNECTOME DEEP EXPERIMENTS ===\n");

    // =========================================================
    // EXP 1: |a-b| with 20 workers, 20 seeds, passive connectome
    // =========================================================
    println!("--- EXP 1: |a-b| brute force (20 workers, 20 seeds, passive 3-conn) ---\n");

    let results: Vec<(f64, usize, u64)> = SEEDS.par_iter().map(|&seed| {
        let (acc, workers, _net) = incremental_build(
            3, 2, 20, op_sub_abs, 5, seed, 3_000_000,
        );
        (acc, workers, seed)
    }).collect();

    let solved: Vec<_> = results.iter().filter(|r| r.0 >= 1.0).collect();
    let best = results.iter().max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();
    let mean: f64 = results.iter().map(|r| r.0).sum::<f64>() / results.len() as f64;

    println!("  Solved: {}/{} seeds", solved.len(), SEEDS.len());
    println!("  Best: {:.0}% (seed={}, {} workers)", best.0 * 100.0, best.2, best.1);
    println!("  Mean: {:.0}%", mean * 100.0);
    for r in &results {
        print!("    seed={:>5}: {:.0}% ({} w) {}", r.2, r.0 * 100.0, r.1,
            if r.0 >= 1.0 { "SOLVED" } else { "" });
        println!();
    }

    // =========================================================
    // EXP 2: |a-b| incremental + perturbation refinement
    // =========================================================
    println!("\n--- EXP 2: |a-b| incremental(10w) + perturbation(500K steps) ---\n");

    let results2: Vec<(f64, f64, u64)> = SEEDS.par_iter().map(|&seed| {
        let (inc_acc, _workers, mut net) = incremental_build(
            3, 2, 10, op_sub_abs, 5, seed, 2_000_000,
        );
        let refined_acc = perturbation_refine(&mut net, op_sub_abs, 5, 500_000, seed + 1000);
        (inc_acc, refined_acc, seed)
    }).collect();

    let solved2: Vec<_> = results2.iter().filter(|r| r.1 >= 1.0).collect();
    let mean_inc: f64 = results2.iter().map(|r| r.0).sum::<f64>() / results2.len() as f64;
    let mean_ref: f64 = results2.iter().map(|r| r.1).sum::<f64>() / results2.len() as f64;

    println!("  Solved after perturbation: {}/{} seeds", solved2.len(), SEEDS.len());
    println!("  Mean incremental: {:.0}% → refined: {:.0}%", mean_inc * 100.0, mean_ref * 100.0);
    for r in &results2 {
        println!("    seed={:>5}: {:.0}% → {:.0}% {}",
            r.2, r.0 * 100.0, r.1 * 100.0,
            if r.1 >= 1.0 { "SOLVED" } else if r.1 > r.0 { "improved" } else { "" });
    }

    // =========================================================
    // EXP 3: Connectome size sweep on |a-b|
    // =========================================================
    println!("\n--- EXP 3: Connectome size sweep (|a-b|, 10 workers, ticks=2) ---\n");

    let conn_sizes = [0, 1, 2, 3, 4, 6, 8];
    println!("  {:>4} {:>8} {:>8} {:>8} {:>8}", "N_C", "best", "mean", "solved", "params/w");

    for &nc in &conn_sizes {
        let results: Vec<(f64, usize)> = SEEDS.par_iter().map(|&seed| {
            let (acc, w, _) = incremental_build(nc, 2, 10, op_sub_abs, 5, seed, 2_000_000);
            (acc, w)
        }).collect();

        let best = results.iter().map(|r| r.0).fold(0.0f64, f64::max);
        let mean: f64 = results.iter().map(|r| r.0).sum::<f64>() / results.len() as f64;
        let solved = results.iter().filter(|r| r.0 >= 1.0).count();
        let params = INPUT_DIM + LOCAL_CAP.min(10) + nc + 1 + 1;

        println!("  {:>4} {:>7.0}% {:>7.0}% {:>5}/{:>2} {:>8}",
            nc, best * 100.0, mean * 100.0, solved, SEEDS.len(), params);
    }

    // =========================================================
    // EXP 4: Tick sweep on |a-b| (passive, 3 connectome)
    // =========================================================
    println!("\n--- EXP 4: Tick sweep (|a-b|, passive 3-conn, 10 workers) ---\n");

    let tick_vals = [1, 2, 3, 4, 6];
    println!("  {:>5} {:>8} {:>8} {:>8}", "ticks", "best", "mean", "solved");

    for &t in &tick_vals {
        let results: Vec<f64> = SEEDS.par_iter().map(|&seed| {
            let (acc, _, _) = incremental_build(3, t, 10, op_sub_abs, 5, seed, 2_000_000);
            acc
        }).collect();

        let best = results.iter().fold(0.0f64, |a, &b| a.max(b));
        let mean: f64 = results.iter().sum::<f64>() / results.len() as f64;
        let solved = results.iter().filter(|&&r| r >= 1.0).count();

        println!("  {:>5} {:>7.0}% {:>7.0}% {:>5}/{:>2}",
            t, best * 100.0, mean * 100.0, solved, SEEDS.len());
    }

    // =========================================================
    // EXP 5: Best config on all tasks (with perturbation)
    // =========================================================
    println!("\n--- EXP 5: All tasks — passive 3-conn, 2 ticks, 10w + perturbation ---\n");

    let tasks = make_tasks();
    println!("  {:>6} {:>8} {:>8} {:>10} {:>10}", "task", "best", "mean", "best+pert", "mean+pert");

    for task in &tasks {
        let results: Vec<(f64, f64)> = SEEDS.par_iter().map(|&seed| {
            let (inc_acc, _, mut net) = incremental_build(
                3, 2, 10, task.op, task.n_classes, seed, 2_000_000,
            );
            let ref_acc = if inc_acc < 1.0 {
                perturbation_refine(&mut net, task.op, task.n_classes, 500_000, seed + 1000)
            } else { inc_acc };
            (inc_acc, ref_acc)
        }).collect();

        let best_inc = results.iter().map(|r| r.0).fold(0.0f64, f64::max);
        let mean_inc: f64 = results.iter().map(|r| r.0).sum::<f64>() / results.len() as f64;
        let best_ref = results.iter().map(|r| r.1).fold(0.0f64, f64::max);
        let mean_ref: f64 = results.iter().map(|r| r.1).sum::<f64>() / results.len() as f64;

        println!("  {:>6} {:>7.0}% {:>7.0}% {:>9.0}% {:>9.0}%",
            task.name, best_inc * 100.0, mean_inc * 100.0, best_ref * 100.0, mean_ref * 100.0);
    }

    println!("\n=== DONE ({:.1}s) ===", t0.elapsed().as_secs_f64());
}
