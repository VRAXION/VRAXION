//! Universal chip test: can recurrent ReLU solve ALL operations?
//!
//! For each operation, test:
//!   A) Per-neuron bias (3 params) — the minimal model
//!   B) Per-connection bias (N×(N+4) params) — if A fails
//!
//! Operations tested:
//!   Arithmetic: ADD, MUL, SUB, MAX, MIN
//!   Logic: AND, OR, XOR, NAND (binary, sequential)
//!   Comparison: a==b, |a-b|
//!
//! All recurrent: one input per tick, same W, test generalization.
//!
//! Run: cargo run --example chip_ops_sweep --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIGITS: usize = 5; // 0..4

fn thermo(val: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 4];
    for i in 0..val.min(4) { v[i] = 1.0; }
    v
}

fn relu(x: f32) -> f32 { x.max(0.0) }

struct NearestMean { centroids: Vec<f32> }
impl NearestMean {
    fn fit(examples: &[(f32, usize)], n_classes: usize) -> Self {
        let actual_classes = n_classes.max(examples.iter().map(|e| e.1 + 1).max().unwrap_or(1));
        let mut sums = vec![0.0f32; actual_classes];
        let mut counts = vec![0usize; actual_classes];
        for &(s, c) in examples { sums[c] += s; counts[c] += 1; }
        NearestMean {
            centroids: (0..n_classes).map(|c| if counts[c] > 0 { sums[c] / counts[c] as f32 } else { f32::NAN }).collect()
        }
    }
    fn predict(&self, s: f32) -> usize {
        self.centroids.iter().enumerate()
            .filter(|(_, c)| !c.is_nan())
            .min_by(|a, b| (a.1 - s).abs().partial_cmp(&(b.1 - s).abs()).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }
}

/// Recurrent forward with per-neuron bias.
fn recurrent_neuron_bias(
    digits: &[usize], w: &[Vec<f32>], bias: &[f32],
) -> Vec<f32> {
    let n = w.len();
    let mut act = vec![0.0f32; n];
    for &digit in digits {
        let t = thermo(digit);
        let mut input = Vec::with_capacity(n + 4);
        input.extend_from_slice(&act);
        input.extend_from_slice(&t);
        for i in 0..n {
            let mut sum = bias[i];
            for (j, &inp) in input.iter().enumerate() {
                if j < w[i].len() { sum += w[i][j] * inp; }
            }
            act[i] = relu(sum);
        }
    }
    act
}

/// Recurrent forward with per-connection bias.
fn recurrent_conn_bias(
    digits: &[usize], w: &[Vec<f32>], b: &[Vec<f32>],
) -> Vec<f32> {
    let n = w.len();
    let mut act = vec![0.0f32; n];
    for &digit in digits {
        let t = thermo(digit);
        let mut input = Vec::with_capacity(n + 4);
        input.extend_from_slice(&act);
        input.extend_from_slice(&t);
        for i in 0..n {
            let mut sum = 0.0f32;
            for (j, &inp) in input.iter().enumerate() {
                if j < w[i].len() { sum += inp * w[i][j] + b[i][j]; }
            }
            act[i] = relu(sum);
        }
    }
    act
}

fn gen_combos(n: usize) -> Vec<Vec<usize>> {
    let mut result = vec![vec![]];
    for _ in 0..n {
        let mut new_r = Vec::new();
        for combo in &result {
            for d in 0..DIGITS { let mut c = combo.clone(); c.push(d); new_r.push(c); }
        }
        result = new_r;
    }
    result
}

/// Generic eval: applies op to digit sequence, returns accuracy.
fn eval_op_neuron_bias(
    w: &[Vec<f32>], bias: &[f32], n_inputs: usize, n_classes: usize,
    op: &dyn Fn(&[usize]) -> usize,
) -> f64 {
    let combos = gen_combos(n_inputs);
    let mut examples = Vec::new();
    for combo in &combos {
        let target = op(combo);
        let act = recurrent_neuron_bias(combo, w, bias);
        let s: f32 = act.iter().sum();
        if s.is_nan() || s.is_infinite() { return 0.0; }
        examples.push((s, target));
    }
    let readout = NearestMean::fit(&examples, n_classes);
    examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / combos.len() as f64
}

fn eval_op_conn_bias(
    w: &[Vec<f32>], b: &[Vec<f32>], n_inputs: usize, n_classes: usize,
    op: &dyn Fn(&[usize]) -> usize,
) -> f64 {
    let combos = gen_combos(n_inputs);
    let mut examples = Vec::new();
    for combo in &combos {
        let target = op(combo);
        let act = recurrent_conn_bias(combo, w, b);
        let s: f32 = act.iter().sum();
        if s.is_nan() || s.is_infinite() { return 0.0; }
        examples.push((s, target));
    }
    let readout = NearestMean::fit(&examples, n_classes);
    examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / combos.len() as f64
}

/// Search with per-neuron bias
fn search_neuron_bias(
    n: usize, train_n: usize, n_classes: usize, seed: u64,
    op: &dyn Fn(&[usize]) -> usize,
) -> (Vec<Vec<f32>>, Vec<f32>, f64) {
    let input_dim = n + 4;
    let wr: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut rng = StdRng::seed_from_u64(seed);
    let total_params = n * input_dim + n;

    let mut best_acc = 0.0f64;
    let mut best_w = vec![vec![0.0f32; input_dim]; n];
    let mut best_bias = vec![0.0f32; n];

    // Random search
    for _ in 0..3_000_000u64 {
        let w: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| wr[rng.gen_range(0..wr.len())]).collect()).collect();
        let bias: Vec<f32> = (0..n).map(|_| wr[rng.gen_range(0..wr.len())]).collect();
        let acc = eval_op_neuron_bias(&w, &bias, train_n, n_classes, op);
        if acc > best_acc { best_acc = acc; best_w = w; best_bias = bias; }
        if best_acc >= 1.0 { return (best_w, best_bias, 1.0); }
    }

    // Perturbation
    let mut current = best_acc;
    for _ in 0..500_000u64 {
        let idx = rng.gen_range(0..total_params);
        let delta: f32 = rng.gen_range(-0.3..0.3);
        let (old, is_b, i, j) = if idx < n * input_dim {
            let i = idx / input_dim; let j = idx % input_dim;
            let old = best_w[i][j]; best_w[i][j] += delta; (old, false, i, j)
        } else {
            let i = idx - n * input_dim;
            let old = best_bias[i]; best_bias[i] += delta; (old, true, i, 0)
        };
        let acc = eval_op_neuron_bias(&best_w, &best_bias, train_n, n_classes, op);
        if acc >= current { current = acc; } else {
            if is_b { best_bias[i] = old; } else { best_w[i][j] = old; }
        }
        if current >= 1.0 { return (best_w, best_bias, 1.0); }
    }
    (best_w, best_bias, current)
}

/// Search with per-connection bias
fn search_conn_bias(
    n: usize, train_n: usize, n_classes: usize, seed: u64,
    op: &dyn Fn(&[usize]) -> usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, f64) {
    let input_dim = n + 4;
    let wr: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut rng = StdRng::seed_from_u64(seed);
    let total_params = n * input_dim * 2;

    let mut best_acc = 0.0f64;
    let mut best_w = vec![vec![0.0f32; input_dim]; n];
    let mut best_b = vec![vec![0.0f32; input_dim]; n];

    for _ in 0..3_000_000u64 {
        let w: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| wr[rng.gen_range(0..wr.len())]).collect()).collect();
        let b: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| wr[rng.gen_range(0..wr.len())]).collect()).collect();
        let acc = eval_op_conn_bias(&w, &b, train_n, n_classes, op);
        if acc > best_acc { best_acc = acc; best_w = w; best_b = b; }
        if best_acc >= 1.0 { return (best_w, best_b, 1.0); }
    }

    let mut current = best_acc;
    for _ in 0..500_000u64 {
        let idx = rng.gen_range(0..total_params);
        let delta: f32 = rng.gen_range(-0.3..0.3);
        let half = n * input_dim;
        let (old, is_b_mat, i, j) = if idx < half {
            let i = idx / input_dim; let j = idx % input_dim;
            let old = best_w[i][j]; best_w[i][j] += delta; (old, false, i, j)
        } else {
            let idx2 = idx - half;
            let i = idx2 / input_dim; let j = idx2 % input_dim;
            let old = best_b[i][j]; best_b[i][j] += delta; (old, true, i, j)
        };
        let acc = eval_op_conn_bias(&best_w, &best_b, train_n, n_classes, op);
        if acc >= current { current = acc; } else {
            if is_b_mat { best_b[i][j] = old; } else { best_w[i][j] = old; }
        }
        if current >= 1.0 { return (best_w, best_b, 1.0); }
    }
    (best_w, best_b, current)
}

fn main() {
    println!("=== UNIVERSAL CHIP TEST: recurrent ReLU on ALL operations ===");
    println!("Per-neuron bias first, per-connection bias if needed.\n");

    let n = 3;

    // Define operations (all reduce a sequence of digits)
    // For binary ops (AND, XOR, etc.), digits are 0 or 1 only
    struct Op {
        name: &'static str,
        op: Box<dyn Fn(&[usize]) -> usize>,
        n_classes: usize,
        train_n: usize,
        binary_input: bool, // if true, digits are 0/1 only
    }

    let ops = vec![
        Op { name: "ADD", op: Box::new(|d: &[usize]| d.iter().sum()), n_classes: 17, train_n: 3, binary_input: false },
        Op { name: "MUL", op: Box::new(|d: &[usize]| d.iter().product()), n_classes: 257, train_n: 3, binary_input: false }, // 4^4=256 max
        Op { name: "MAX", op: Box::new(|d: &[usize]| *d.iter().max().unwrap()), n_classes: 5, train_n: 3, binary_input: false },
        Op { name: "MIN", op: Box::new(|d: &[usize]| *d.iter().min().unwrap()), n_classes: 5, train_n: 3, binary_input: false },
        Op { name: "AND", op: Box::new(|d: &[usize]| d.iter().fold(1, |a, &b| a & b)), n_classes: 2, train_n: 3, binary_input: true },
        Op { name: "OR",  op: Box::new(|d: &[usize]| d.iter().fold(0, |a, &b| a | b)), n_classes: 2, train_n: 3, binary_input: true },
        Op { name: "XOR", op: Box::new(|d: &[usize]| d.iter().fold(0, |a, &b| a ^ b)), n_classes: 2, train_n: 3, binary_input: true },
        Op { name: "NAND", op: Box::new(|d: &[usize]| 1 - d.iter().fold(1, |a, &b| a & b)), n_classes: 2, train_n: 3, binary_input: true },
    ];

    println!("{:<8} {:>8} {:>8} | {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "Op", "neuron_b", "conn_b", "2-in", "3-in", "4-in", "5-in", "6-in", "7-in");
    println!("{}", "=".repeat(85));

    for op_def in &ops {
        // Determine digit range
        let digit_range = if op_def.binary_input { 2 } else { DIGITS };

        // Custom combo generator for binary inputs
        let gen_combos_bin = |n_inputs: usize, max_digit: usize| -> Vec<Vec<usize>> {
            let mut result = vec![vec![]];
            for _ in 0..n_inputs {
                let mut new_r = Vec::new();
                for combo in &result {
                    for d in 0..max_digit {
                        let mut c = combo.clone();
                        c.push(d);
                        new_r.push(c);
                    }
                }
                result = new_r;
            }
            result
        };

        // Custom eval for binary ops
        let eval_custom_neuron = |w: &[Vec<f32>], bias: &[f32], n_inputs: usize| -> f64 {
            let combos = gen_combos_bin(n_inputs, digit_range);
            let mut examples = Vec::new();
            for combo in &combos {
                let target = (op_def.op)(combo);
                let act = recurrent_neuron_bias(combo, w, bias);
                let s: f32 = act.iter().sum();
                if s.is_nan() || s.is_infinite() { return 0.0; }
                examples.push((s, target));
            }
            let readout = NearestMean::fit(&examples, op_def.n_classes);
            examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / combos.len() as f64
        };

        let eval_custom_conn = |w: &[Vec<f32>], b: &[Vec<f32>], n_inputs: usize| -> f64 {
            let combos = gen_combos_bin(n_inputs, digit_range);
            let mut examples = Vec::new();
            for combo in &combos {
                let target = (op_def.op)(combo);
                let act = recurrent_conn_bias(combo, w, b);
                let s: f32 = act.iter().sum();
                if s.is_nan() || s.is_infinite() { return 0.0; }
                examples.push((s, target));
            }
            let readout = NearestMean::fit(&examples, op_def.n_classes);
            examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / combos.len() as f64
        };

        // Search per-neuron bias
        let input_dim = n + 4;
        let wr: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut rng = StdRng::seed_from_u64(42);
        let total_params_n = n * input_dim + n;

        let mut best_acc_n = 0.0f64;
        let mut best_w_n = vec![vec![0.0f32; input_dim]; n];
        let mut best_bias_n = vec![0.0f32; n];

        for _ in 0..3_000_000u64 {
            let w: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| wr[rng.gen_range(0..wr.len())]).collect()).collect();
            let bias: Vec<f32> = (0..n).map(|_| wr[rng.gen_range(0..wr.len())]).collect();
            let acc = eval_custom_neuron(&w, &bias, op_def.train_n);
            if acc > best_acc_n { best_acc_n = acc; best_w_n = w; best_bias_n = bias; }
            if best_acc_n >= 1.0 { break; }
        }
        // Perturbation
        let mut current_n = best_acc_n;
        for _ in 0..500_000u64 {
            let idx = rng.gen_range(0..total_params_n);
            let delta: f32 = rng.gen_range(-0.3..0.3);
            let (old, is_b, i, j) = if idx < n * input_dim {
                let i = idx / input_dim; let j = idx % input_dim;
                let old = best_w_n[i][j]; best_w_n[i][j] += delta; (old, false, i, j)
            } else {
                let i = idx - n * input_dim;
                let old = best_bias_n[i]; best_bias_n[i] += delta; (old, true, i, 0)
            };
            let acc = eval_custom_neuron(&best_w_n, &best_bias_n, op_def.train_n);
            if acc >= current_n { current_n = acc; } else {
                if is_b { best_bias_n[i] = old; } else { best_w_n[i][j] = old; }
            }
            if current_n >= 1.0 { break; }
        }

        // Search per-connection bias
        let total_params_c = n * input_dim * 2;
        let mut rng2 = StdRng::seed_from_u64(777);

        let mut best_acc_c = 0.0f64;
        let mut best_w_c = vec![vec![0.0f32; input_dim]; n];
        let mut best_b_c = vec![vec![0.0f32; input_dim]; n];

        for _ in 0..3_000_000u64 {
            let w: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| wr[rng2.gen_range(0..wr.len())]).collect()).collect();
            let b: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| wr[rng2.gen_range(0..wr.len())]).collect()).collect();
            let acc = eval_custom_conn(&w, &b, op_def.train_n);
            if acc > best_acc_c { best_acc_c = acc; best_w_c = w; best_b_c = b; }
            if best_acc_c >= 1.0 { break; }
        }
        let mut current_c = best_acc_c;
        for _ in 0..500_000u64 {
            let idx = rng2.gen_range(0..total_params_c);
            let delta: f32 = rng2.gen_range(-0.3..0.3);
            let half = n * input_dim;
            let (old, is_b_mat, i, j) = if idx < half {
                let i = idx / input_dim; let j = idx % input_dim;
                let old = best_w_c[i][j]; best_w_c[i][j] += delta; (old, false, i, j)
            } else {
                let idx2 = idx - half;
                let i = idx2 / input_dim; let j = idx2 % input_dim;
                let old = best_b_c[i][j]; best_b_c[i][j] += delta; (old, true, i, j)
            };
            let acc = eval_custom_conn(&best_w_c, &best_b_c, op_def.train_n);
            if acc >= current_c { current_c = acc; } else {
                if is_b_mat { best_b_c[i][j] = old; } else { best_w_c[i][j] = old; }
            }
            if current_c >= 1.0 { break; }
        }

        // Pick winner and test generalization
        let use_conn = current_c > current_n;
        let winner_label = if use_conn { "conn" } else { "neuron" };

        print!("{:<8} {:>7.1}% {:>7.1}% |", op_def.name, current_n * 100.0, current_c * 100.0);

        let test_range = if op_def.binary_input { 2..=7 } else { 2..=7 };
        for n_inputs in test_range {
            let acc = if use_conn {
                eval_custom_conn(&best_w_c, &best_b_c, n_inputs)
            } else {
                eval_custom_neuron(&best_w_n, &best_bias_n, n_inputs)
            };
            print!(" {:>5.1}%", acc * 100.0);
        }
        println!("  [{}]", winner_label);
    }

    println!("\n=== DONE ===");
    println!("Legend: neuron_b = per-neuron bias trained accuracy, conn_b = per-connection bias");
    println!("Generalization columns use the better model (shown in brackets).");
}
