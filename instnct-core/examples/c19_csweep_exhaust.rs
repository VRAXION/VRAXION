//! Binary weights + fine C sweep — exhaustive guaranteed search
//!
//! Per neuron: sweep C (0.01 step) × binary weights exhaustive
//! Find the OPTIMAL (C, weights) per neuron, bake C as constant
//! Greedy: add neuron, find best, freeze, repeat
//!
//! Run: cargo run --example c19_csweep_exhaust --release

use rayon::prelude::*;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const LOCAL_CAP: usize = 3;
const TICKS: usize = 2;
const RHO: f32 = 4.0;

fn c19(x: f32, c: f32) -> f32 {
    let c = c.max(0.01);
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

fn relu(x: f32) -> f32 { x.max(0.0) }

fn thermo_2(a: usize, b: usize) -> [f32; 8] {
    let mut v = [0.0f32; 8];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

#[derive(Clone)]
struct GreedyNet {
    nc: usize,
    workers: Vec<(Vec<i8>, f32)>, // (binary weights, C value) per worker
}

impl GreedyNet {
    fn new(nc: usize) -> Self {
        GreedyNet { nc, workers: Vec::new() }
    }

    fn forward(&self, a: usize, b: usize) -> f32 {
        let input = thermo_2(a, b);
        let nc = self.nc;
        let nw = self.workers.len();
        let mut act = vec![0.0f32; nc + nw];

        for _t in 0..TICKS {
            let mut cc = vec![0.0f32; nc];
            for i in 0..nw {
                let (ref w, _c) = self.workers[i];
                let nl = LOCAL_CAP.min(i);
                let ww = w[INPUT_DIM + nl + nc] as f32;
                let slot = i % nc.max(1);
                if slot < nc { cc[slot] += act[nc + i] * ww; }
            }
            for k in 0..nc { act[k] = cc[k]; }
            let old = act.clone();
            for i in 0..nw {
                let (ref w, c_val) = self.workers[i];
                let nl = LOCAL_CAP.min(i);
                let mut s = w[INPUT_DIM + nl + nc + 1] as f32; // bias
                for j in 0..INPUT_DIM { s += input[j] * w[j] as f32; }
                let ls = i.saturating_sub(nl);
                for (k, wi) in (ls..i).enumerate() {
                    s += old[nc + wi] * w[INPUT_DIM + k] as f32;
                }
                for k in 0..nc {
                    s += old[k] * w[INPUT_DIM + nl + k] as f32;
                }
                act[nc + i] = c19(s, c_val);
            }
        }
        act[nc..].iter().sum()
    }

    fn accuracy(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut correct = 0;
        for a in 0..DIGITS { for b in 0..DIGITS {
            if (self.forward(a, b).round() as i32) == (op(a, b) as i32) { correct += 1; }
        }}
        correct as f64 / 25.0
    }

    fn mse(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut l = 0.0f64;
        for a in 0..DIGITS { for b in 0..DIGITS {
            let d = self.forward(a, b) as f64 - op(a, b) as f64;
            l += d * d;
        }}
        l / 25.0
    }

    /// Add best possible worker — exhaustive over binary weights × C sweep
    fn add_best_worker(&mut self, op: fn(usize, usize) -> usize, c_step: f32, c_max: f32) -> (f64, f32) {
        let worker_idx = self.workers.len();
        let nl = LOCAL_CAP.min(worker_idx);
        let n_params = INPUT_DIM + nl + self.nc + 1 + 1;
        let total_binary: u32 = 1 << n_params;

        // C values to sweep
        let c_steps: Vec<f32> = {
            let mut v = Vec::new();
            let mut c = c_step;
            while c <= c_max {
                v.push(c);
                c += c_step;
            }
            v
        };

        let base_net = self.clone();

        // Parallel over C values
        let results: Vec<(f64, f64, f32, Vec<i8>)> = c_steps.par_iter().map(|&c_val| {
            let mut best_acc = 0.0f64;
            let mut best_mse = f64::MAX;
            let mut best_weights: Vec<i8> = Vec::new();

            for config in 0..total_binary {
                let weights: Vec<i8> = (0..n_params).map(|bit| {
                    if (config >> bit) & 1 == 1 { 1i8 } else { -1i8 }
                }).collect();

                let mut test_net = base_net.clone();
                test_net.workers.push((weights.clone(), c_val));
                let acc = test_net.accuracy(op);
                let mse = test_net.mse(op);

                if acc > best_acc || (acc == best_acc && mse < best_mse) {
                    best_acc = acc;
                    best_mse = mse;
                    best_weights = weights;
                }
            }
            (best_acc, best_mse, c_val, best_weights)
        }).collect();

        // Find overall best
        let mut best = &results[0];
        for r in &results {
            if r.0 > best.0 || (r.0 == best.0 && r.1 < best.1) {
                best = r;
            }
        }

        let best_c = best.2;
        let best_weights = best.3.clone();
        let best_acc = best.0;

        self.workers.push((best_weights, best_c));
        (best_acc, best_c)
    }
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn main() {
    println!("=== GREEDY CONSTRUCTIVE: binary weights + C sweep exhaustive ===\n");
    println!("Per neuron: binary ±1 exhaustive × C sweep (0.05 step, 0.05-8.0)");
    println!("Greedy: add best neuron, freeze, repeat until 100% or no improvement\n");

    let nc = 3;
    let c_step = 0.05;
    let c_max = 8.0;
    let max_workers = 8;

    let ops: Vec<(&str, fn(usize, usize) -> usize)> = vec![
        ("ADD",   op_add),
        ("MAX",   op_max),
        ("MIN",   op_min),
        ("|a-b|", op_sub_abs),
        ("MUL",   op_mul),
    ];

    for &(name, op) in &ops {
        println!("--- {} ---", name);
        let mut net = GreedyNet::new(nc);

        for w in 0..max_workers {
            let n_params = INPUT_DIM + LOCAL_CAP.min(w) + nc + 1 + 1;
            let total_binary: u32 = 1 << n_params;
            let c_count = ((c_max - c_step) / c_step) as u32 + 1;
            let total_search = total_binary as u64 * c_count as u64;

            print!("  Worker {}: {} binary × {} C values = {} configs ... ",
                w, total_binary, c_count, total_search);

            let (acc, best_c) = net.add_best_worker(op, c_step, c_max);

            println!("acc={:.0}% C={:.2} mse={:.4}", acc * 100.0, best_c, net.mse(op));

            if acc >= 1.0 {
                println!("  SOLVED with {} workers!\n", w + 1);
                // Print solution
                println!("  Solution:");
                for (i, (weights, c)) in net.workers.iter().enumerate() {
                    let w_str: Vec<String> = weights.iter().map(|w| format!("{:+}", w)).collect();
                    println!("    Worker {}: C={:.2} weights=[{}]", i, c, w_str.join(","));
                }
                println!();
                break;
            }

            if w > 0 {
                // Check if this worker helped
                let mut prev = net.clone();
                prev.workers.pop();
                let prev_acc = prev.accuracy(op);
                if acc <= prev_acc {
                    println!("  No improvement, stopping at {} workers (best={:.0}%)\n", w, acc * 100.0);
                    break;
                }
            }

            if w == max_workers - 1 {
                println!("  Max workers reached, best={:.0}%\n", acc * 100.0);
            }
        }
    }

    // =========================================================
    // ReLU comparison: same greedy but with ReLU (no C search)
    // =========================================================
    println!("--- ReLU BASELINE (greedy, same setup, no C) ---\n");

    for &(name, op) in &ops {
        println!("  {}: ", name);
        let mut workers_relu: Vec<Vec<i8>> = Vec::new();

        for w in 0..max_workers {
            let nl = LOCAL_CAP.min(w);
            let n_params = INPUT_DIM + nl + nc + 1 + 1;
            let total_binary: u32 = 1 << n_params;

            let base_workers = workers_relu.clone();
            let nw = base_workers.len();

            let mut best_acc = 0.0f64;
            let mut best_mse = f64::MAX;
            let mut best_w: Vec<i8> = Vec::new();

            for config in 0..total_binary {
                let weights: Vec<i8> = (0..n_params).map(|bit| {
                    if (config >> bit) & 1 == 1 { 1i8 } else { -1i8 }
                }).collect();

                // Eval with ReLU
                let mut correct = 0;
                let mut mse = 0.0f64;
                for a in 0..DIGITS { for b in 0..DIGITS {
                    let input = thermo_2(a, b);
                    let total_n = nc + nw + 1;
                    let mut act = vec![0.0f32; total_n];
                    for _t in 0..TICKS {
                        let mut cc = vec![0.0f32; nc];
                        for i in 0..nw+1 {
                            let w_ref = if i < nw { &base_workers[i] } else { &weights };
                            let nli = LOCAL_CAP.min(i);
                            let ww = w_ref[INPUT_DIM + nli + nc] as f32;
                            let slot = i % nc.max(1);
                            if slot < nc { cc[slot] += act[nc + i] * ww; }
                        }
                        for k in 0..nc { act[k] = cc[k]; }
                        let old = act.clone();
                        for i in 0..nw+1 {
                            let w_ref = if i < nw { &base_workers[i] } else { &weights };
                            let nli = LOCAL_CAP.min(i);
                            let mut s = w_ref[INPUT_DIM + nli + nc + 1] as f32;
                            for j in 0..INPUT_DIM { s += input[j] * w_ref[j] as f32; }
                            let ls = i.saturating_sub(nli);
                            for (k, wi) in (ls..i).enumerate() {
                                s += old[nc + wi] * w_ref[INPUT_DIM + k] as f32;
                            }
                            for k in 0..nc { s += old[k] * w_ref[INPUT_DIM + nli + k] as f32; }
                            act[nc + i] = relu(s);
                        }
                    }
                    let output: f32 = act[nc..].iter().sum();
                    if output.round() as i32 == op(a, b) as i32 { correct += 1; }
                    mse += (output as f64 - op(a,b) as f64).powi(2);
                }}
                let acc = correct as f64 / 25.0;
                let mse_avg = mse / 25.0;
                if acc > best_acc || (acc == best_acc && mse_avg < best_mse) {
                    best_acc = acc; best_mse = mse_avg; best_w = weights;
                }
            }

            workers_relu.push(best_w);
            print!("    w{}: {:.0}%  ", w, best_acc * 100.0);
            if best_acc >= 1.0 { println!("SOLVED with {}w!", w+1); break; }
            if w == max_workers - 1 { println!("max reached"); }
        }
        println!();
    }

    println!("=== DONE ===");
}
