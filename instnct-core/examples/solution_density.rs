//! Solution density: how many winning configs exist in the search space?
//!
//! For each task, exhaustively scan ALL binary configs for worker neurons
//! and count how many achieve each accuracy level.
//!
//! Run: cargo run --example solution_density --release

use rayon::prelude::*;
use std::time::Instant;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const TICKS: usize = 2;

fn relu(x: f32) -> f32 { x.max(0.0) }

fn thermo_2(a: usize, b: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; INPUT_DIM];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

/// Minimal 1-worker + N connectome network, native output
fn eval_single_worker(
    params: &[f32], n_connectome: usize,
    op: fn(usize, usize) -> usize,
) -> (f64, f64) {
    // params = [w_input(8), w_conn_read(n_conn), w_conn_write(1), bias(1)]
    let n_params = INPUT_DIM + n_connectome + 1 + 1;
    assert_eq!(params.len(), n_params);

    let w_input = &params[0..INPUT_DIM];
    let w_conn = &params[INPUT_DIM..INPUT_DIM + n_connectome];
    let w_write = params[INPUT_DIM + n_connectome];
    let bias = params[INPUT_DIM + n_connectome + 1];

    let mut correct = 0;
    let mut mse = 0.0f64;

    for a in 0..DIGITS {
        for b in 0..DIGITS {
            let input = thermo_2(a, b);
            let target = op(a, b) as f64;

            let mut act = 0.0f32;
            let mut conn_charge = 0.0f32;

            for _tick in 0..TICKS {
                // Connectome = passive relay of worker
                conn_charge = act * w_write;

                // Worker computes
                let mut sum = bias;
                for (j, &w) in w_input.iter().enumerate() {
                    sum += input[j] * w;
                }
                // Read from all connectome (all have same charge for 1 worker)
                for &w in w_conn.iter() {
                    sum += conn_charge * w;
                }
                act = relu(sum);
            }

            let charge = act as f64;
            mse += (charge - target).powi(2);
            if (charge.round() as i32) == (target as i32) { correct += 1; }
        }
    }

    (correct as f64 / 25.0, mse / 25.0)
}

/// Eval 2-worker network
fn eval_two_workers(
    params0: &[f32], params1: &[f32], n_connectome: usize,
    op: fn(usize, usize) -> usize,
) -> (f64, f64) {
    let w_input0 = &params0[0..INPUT_DIM];
    let w_conn0 = &params0[INPUT_DIM..INPUT_DIM + n_connectome];
    let w_write0 = params0[INPUT_DIM + n_connectome];
    let bias0 = params0[INPUT_DIM + n_connectome + 1];

    // Worker 1 has 1 local connection (to worker 0)
    let w_input1 = &params1[0..INPUT_DIM];
    let w_local1 = params1[INPUT_DIM]; // 1 local weight
    let w_conn1 = &params1[INPUT_DIM + 1..INPUT_DIM + 1 + n_connectome];
    let w_write1 = params1[INPUT_DIM + 1 + n_connectome];
    let bias1 = params1[INPUT_DIM + 1 + n_connectome + 1];

    let mut correct = 0;
    let mut mse = 0.0f64;

    for a in 0..DIGITS {
        for b in 0..DIGITS {
            let input = thermo_2(a, b);
            let target = op(a, b) as f64;

            let mut act0 = 0.0f32;
            let mut act1 = 0.0f32;

            for _tick in 0..TICKS {
                let mut conn_charges = vec![0.0f32; n_connectome];
                let write_idx0 = 0 % n_connectome.max(1);
                let write_idx1 = 1 % n_connectome.max(1);
                if write_idx0 < n_connectome { conn_charges[write_idx0] += act0 * w_write0; }
                if write_idx1 < n_connectome { conn_charges[write_idx1] += act1 * w_write1; }

                let old_act0 = act0;

                // Worker 0
                let mut sum0 = bias0;
                for (j, &w) in w_input0.iter().enumerate() { sum0 += input[j] * w; }
                for (k, &w) in w_conn0.iter().enumerate() { if k < n_connectome { sum0 += conn_charges[k] * w; } }
                act0 = relu(sum0);

                // Worker 1
                let mut sum1 = bias1;
                for (j, &w) in w_input1.iter().enumerate() { sum1 += input[j] * w; }
                sum1 += old_act0 * w_local1; // local from worker 0
                for (k, &w) in w_conn1.iter().enumerate() { if k < n_connectome { sum1 += conn_charges[k] * w; } }
                act1 = relu(sum1);
            }

            let charge = (act0 + act1) as f64;
            mse += (charge - target).powi(2);
            if (charge.round() as i32) == (target as i32) { correct += 1; }
        }
    }

    (correct as f64 / 25.0, mse / 25.0)
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn main() {
    let t0 = Instant::now();
    println!("=== SOLUTION DENSITY: how many winners in the search space? ===\n");

    let n_connectome = 3;
    let binary: Vec<f32> = vec![-1.0, 1.0];

    let tasks: Vec<(&str, fn(usize, usize) -> usize)> = vec![
        ("ADD", op_add),
        ("MUL", op_mul),
        ("MAX", op_max),
        ("MIN", op_min),
        ("|a-b|", op_sub_abs),
    ];

    // =========================================================
    // 1-worker density
    // =========================================================
    println!("--- 1 WORKER (params={}, space=2^{}={}) ---\n",
        INPUT_DIM + n_connectome + 2, INPUT_DIM + n_connectome + 2,
        2u64.pow((INPUT_DIM + n_connectome + 2) as u32));

    let n_params_w0 = INPUT_DIM + n_connectome + 2; // 13
    let total_w0 = 2u64.pow(n_params_w0 as u32); // 8192

    println!("{:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "task", "100%", ">=96%", ">=80%", ">=60%", ">=40%", "mean%", "best_mse");
    println!("{}", "-".repeat(72));

    for &(task_name, task_op) in &tasks {
        let results: Vec<(f64, f64)> = (0..total_w0).into_par_iter().map(|config| {
            let mut c = config;
            let params: Vec<f32> = (0..n_params_w0).map(|_| {
                let v = binary[(c % 2) as usize]; c /= 2; v
            }).collect();
            eval_single_worker(&params, n_connectome, task_op)
        }).collect();

        let n100 = results.iter().filter(|r| r.0 >= 1.0).count();
        let n96 = results.iter().filter(|r| r.0 >= 0.96).count();
        let n80 = results.iter().filter(|r| r.0 >= 0.80).count();
        let n60 = results.iter().filter(|r| r.0 >= 0.60).count();
        let n40 = results.iter().filter(|r| r.0 >= 0.40).count();
        let mean_acc: f64 = results.iter().map(|r| r.0).sum::<f64>() / results.len() as f64;
        let best_mse = results.iter().map(|r| r.1).fold(f64::MAX, f64::min);

        println!("{:<8} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7.1}% {:>8.3}",
            task_name, n100, n96, n80, n60, n40, mean_acc * 100.0, best_mse);
    }

    println!("\n  (out of {} total configs)\n", total_w0);

    // =========================================================
    // 2-worker density (fix best worker 0, sweep worker 1)
    // =========================================================
    println!("--- 2 WORKERS (worker0=best fixed, sweep worker1) ---\n");

    let n_params_w1 = INPUT_DIM + 1 + n_connectome + 2; // 14 (has 1 local)
    let total_w1 = 2u64.pow(n_params_w1 as u32); // 16384

    println!("{:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "task", "100%", ">=96%", ">=80%", ">=60%", "w0_acc", "w0+w1_best");
    println!("{}", "-".repeat(66));

    for &(task_name, task_op) in &tasks {
        // First find best worker 0
        let mut best_w0_params = vec![0.0f32; n_params_w0];
        let mut best_w0_mse = f64::MAX;

        for config in 0..total_w0 {
            let mut c = config;
            let params: Vec<f32> = (0..n_params_w0).map(|_| {
                let v = binary[(c % 2) as usize]; c /= 2; v
            }).collect();
            let (_, mse) = eval_single_worker(&params, n_connectome, task_op);
            if mse < best_w0_mse {
                best_w0_mse = mse;
                best_w0_params = params;
            }
        }
        let w0_acc = eval_single_worker(&best_w0_params, n_connectome, task_op).0;

        // Sweep all worker 1 configs
        let results: Vec<(f64, f64)> = (0..total_w1).into_par_iter().map(|config| {
            let mut c = config;
            let params1: Vec<f32> = (0..n_params_w1).map(|_| {
                let v = binary[(c % 2) as usize]; c /= 2; v
            }).collect();
            eval_two_workers(&best_w0_params, &params1, n_connectome, task_op)
        }).collect();

        let n100 = results.iter().filter(|r| r.0 >= 1.0).count();
        let n96 = results.iter().filter(|r| r.0 >= 0.96).count();
        let n80 = results.iter().filter(|r| r.0 >= 0.80).count();
        let n60 = results.iter().filter(|r| r.0 >= 0.60).count();
        let best_2w = results.iter().map(|r| r.0).fold(0.0f64, f64::max);

        println!("{:<8} {:>7} {:>7} {:>7} {:>7} {:>7.0}% {:>7.0}%",
            task_name, n100, n96, n80, n60, w0_acc * 100.0, best_2w * 100.0);
    }

    println!("\n  (worker1: {} configs each)\n", total_w1);

    println!("=== DONE ({:.1}s) ===", t0.elapsed().as_secs_f64());
}
