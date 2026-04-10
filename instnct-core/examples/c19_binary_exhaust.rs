//! Binary weights + C19 exhaustive search
//!
//! Can binary ±1 weights + quantized C find native-output solutions?
//! ReLU with binary weights: 0/8192. Does C19 change that?
//!
//! Run: cargo run --example c19_binary_exhaust --release

use rayon::prelude::*;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const LOCAL_CAP: usize = 3;
const TICKS: usize = 2;
const RHO: f32 = 4.0;

fn relu(x: f32) -> f32 { x.max(0.0) }

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

fn thermo_2(a: usize, b: usize) -> [f32; 8] {
    let mut v = [0.0f32; 8];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

/// Single worker with binary weights, evaluate on all 25 pairs
fn eval_single_worker(
    weights: &[i8],  // INPUT_DIM + bias = 9 params
    c_val: f32,
    use_c19: bool,
    op: fn(usize, usize) -> usize,
) -> (f64, f64) {
    let mut correct = 0;
    let mut mse = 0.0f64;
    for a in 0..DIGITS {
        for b in 0..DIGITS {
            let input = thermo_2(a, b);
            let mut sum = weights[INPUT_DIM] as f32; // bias
            for j in 0..INPUT_DIM {
                sum += input[j] * weights[j] as f32;
            }
            let output = if use_c19 { c19(sum, c_val) } else { relu(sum) };
            let target = op(a, b) as f32;
            if output.round() as i32 == target as i32 { correct += 1; }
            mse += (output as f64 - target as f64).powi(2);
        }
    }
    (correct as f64 / 25.0, mse / 25.0)
}

/// Multi-worker network with binary weights
fn eval_network(
    n_workers: usize,
    nc: usize,
    // Per worker: [input_w(8) + local_w(0-3) + conn_read(nc) + conn_write(1) + bias(1)]
    all_weights: &[Vec<i8>],
    c_vals: &[f32],  // per worker
    use_c19: bool,
    op: fn(usize, usize) -> usize,
) -> f64 {
    let mut correct = 0;
    for a in 0..DIGITS {
        for b in 0..DIGITS {
            let input = thermo_2(a, b);
            let mut act = vec![0.0f32; nc + n_workers];

            for _t in 0..TICKS {
                let mut cc = vec![0.0f32; nc];
                for i in 0..n_workers {
                    let w = &all_weights[i];
                    let nl = LOCAL_CAP.min(i);
                    let ww = w[INPUT_DIM + nl + nc] as f32; // write weight
                    let slot = i % nc.max(1);
                    if slot < nc { cc[slot] += act[nc + i] * ww; }
                }
                for k in 0..nc { act[k] = cc[k]; }
                let old = act.clone();
                for i in 0..n_workers {
                    let w = &all_weights[i];
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
                    act[nc + i] = if use_c19 { c19(s, c_vals[i]) } else { relu(s) };
                }
            }
            let output: f32 = act[nc..].iter().sum();
            if output.round() as i32 == op(a, b) as i32 { correct += 1; }
        }
    }
    correct as f64 / 25.0
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn main() {
    println!("=== BINARY WEIGHTS + C19 EXHAUSTIVE SEARCH ===\n");

    // =========================================================
    // TEST 1: Single worker exhaustive — binary weights, sweep C
    // =========================================================
    println!("--- TEST 1: 1 worker, binary ±1 weights, exhaustive (2^9=512 configs) ---");
    println!("    Sweep C values, compare ReLU vs C19\n");

    let c_values: Vec<f32> = vec![0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0];
    let n_params_1w = INPUT_DIM + 1; // 8 input + 1 bias = 9
    let total_configs_1w: u32 = 1 << n_params_1w; // 2^9 = 512

    let ops: Vec<(&str, fn(usize, usize) -> usize)> = vec![
        ("ADD", op_add),
        ("MAX", op_max),
        ("MIN", op_min),
        ("|a-b|", op_sub_abs),
    ];

    // ReLU baseline
    println!("  ReLU (1 worker, no C):");
    for &(name, op) in &ops {
        let mut best_acc = 0.0f64;
        let mut n_perfect = 0u32;
        let mut n_above90 = 0u32;
        let mut n_above80 = 0u32;
        for config in 0..total_configs_1w {
            let weights: Vec<i8> = (0..n_params_1w).map(|bit| {
                if (config >> bit) & 1 == 1 { 1i8 } else { -1i8 }
            }).collect();
            let (acc, _) = eval_single_worker(&weights, 1.0, false, op);
            if acc > best_acc { best_acc = acc; }
            if acc >= 1.0 { n_perfect += 1; }
            if acc >= 0.9 { n_above90 += 1; }
            if acc >= 0.8 { n_above80 += 1; }
        }
        println!("    {}: best={:.0}%  100%={} >=90%={} >=80%={} (of {})",
            name, best_acc * 100.0, n_perfect, n_above90, n_above80, total_configs_1w);
    }

    // C19 sweep
    println!("\n  C19 (1 worker, sweep C, rho={}):", RHO);
    println!("  {:>6} {:>8} {:>8} {:>8} {:>8}", "C", "ADD", "MAX", "MIN", "|a-b|");
    println!("  {}", "=".repeat(45));

    for &c_val in &c_values {
        print!("  {:>6.1}", c_val);
        for &(name, op) in &ops {
            let mut best_acc = 0.0f64;
            let mut n_perfect = 0u32;
            for config in 0..total_configs_1w {
                let weights: Vec<i8> = (0..n_params_1w).map(|bit| {
                    if (config >> bit) & 1 == 1 { 1i8 } else { -1i8 }
                }).collect();
                let (acc, _) = eval_single_worker(&weights, c_val, true, op);
                if acc > best_acc { best_acc = acc; }
                if acc >= 1.0 { n_perfect += 1; }
            }
            print!(" {:>3.0}%/{:<3}", best_acc * 100.0, n_perfect);
        }
        println!();
    }

    // =========================================================
    // TEST 2: 2 workers, binary weights, C19 with best C values
    // =========================================================
    println!("\n--- TEST 2: 2 workers, binary ±1, exhaustive per-worker (greedy) ---");
    println!("    Worker 1: exhaustive 2^9=512. Freeze. Worker 2: exhaustive 2^12=4096.\n");

    // For each C value, do greedy 2-worker search
    let nc = 3;
    let best_c_vals: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5, 5.0];

    for &(name, op) in &ops {
        println!("  === {} ===", name);

        for &c_val in &best_c_vals {
            // Worker 0: no local, no connectome read effectively (first worker)
            let n_params_w0 = INPUT_DIM + 0 + nc + 1 + 1; // 8+0+3+1+1=13
            let total_w0: u32 = 1 << n_params_w0; // 2^13 = 8192

            let mut best_w0: Vec<i8> = Vec::new();
            let mut best_w0_acc = 0.0f64;

            // Search worker 0
            for config in 0..total_w0 {
                let w: Vec<i8> = (0..n_params_w0).map(|bit| {
                    if (config >> bit) & 1 == 1 { 1i8 } else { -1i8 }
                }).collect();
                let acc = eval_network(1, nc, &[w.clone()], &[c_val], true, op);
                if acc > best_w0_acc { best_w0_acc = acc; best_w0 = w; }
            }

            // Worker 1: has local connections to worker 0
            let nl_1 = LOCAL_CAP.min(1); // 1 local connection
            let n_params_w1 = INPUT_DIM + nl_1 + nc + 1 + 1; // 8+1+3+1+1=14
            let total_w1: u32 = 1 << n_params_w1; // 2^14 = 16384

            let mut best_combined_acc = best_w0_acc;

            // Search worker 1 with frozen worker 0
            let results: Vec<f64> = (0..total_w1).into_par_iter().map(|config| {
                let w1: Vec<i8> = (0..n_params_w1).map(|bit| {
                    if (config >> bit) & 1 == 1 { 1i8 } else { -1i8 }
                }).collect();
                eval_network(2, nc, &[best_w0.clone(), w1], &[c_val, c_val], true, op)
            }).collect();

            let w1_best = results.iter().fold(0.0f64, |a, &b| a.max(b));
            if w1_best > best_combined_acc { best_combined_acc = w1_best; }
            let n_perfect = results.iter().filter(|&&a| a >= 1.0).count();

            println!("    C={:.1}: w0={:.0}% → w0+w1={:.0}%  (100% configs: {})",
                c_val, best_w0_acc * 100.0, best_combined_acc * 100.0, n_perfect);
        }
        println!();
    }

    // =========================================================
    // TEST 3: Fine-grained C sweep on 1 worker (find the magic C)
    // =========================================================
    println!("--- TEST 3: Fine C sweep (1 worker, ADD, 0.1 step) ---\n");
    println!("  {:>6} {:>8} {:>8}", "C", "best%", "n_100%");
    println!("  {}", "=".repeat(25));

    for c_int in 1..=80 {
        let c_val = c_int as f32 * 0.1;
        let mut best_acc = 0.0f64;
        let mut n_perfect = 0u32;
        for config in 0..total_configs_1w {
            let weights: Vec<i8> = (0..n_params_1w).map(|bit| {
                if (config >> bit) & 1 == 1 { 1i8 } else { -1i8 }
            }).collect();
            let (acc, _) = eval_single_worker(&weights, c_val, true, op_add);
            if acc > best_acc { best_acc = acc; }
            if acc >= 1.0 { n_perfect += 1; }
        }
        if best_acc >= 0.8 || n_perfect > 0 {
            println!("  {:>6.1} {:>7.0}% {:>8}", c_val, best_acc * 100.0, n_perfect);
        }
    }

    println!("\n=== DONE ===");
}
