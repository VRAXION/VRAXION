//! Readout method sweep: find the universal readout that works for ALL ops.
//!
//! Methods:
//!   1. nearest-mean: centroid per class, predict by closest centroid
//!   2. threshold: charge > 0 → class 1 (binary only)
//!   3. boundary: sort charges, find optimal split points between classes
//!   4. max-margin: maximize gap between adjacent class boundaries
//!   5. median-boundary: use median of each class instead of mean
//!   6. min-overlap: boundary at min-overlap point between adjacent classes
//!
//! Test on ALL ops, train on 3-input, generalize to 2-8.
//!
//! Run: cargo run --example readout_sweep --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIGITS: usize = 5;

fn relu(x: f32) -> f32 { x.max(0.0) }

fn thermo(val: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 4];
    for i in 0..val.min(4) { v[i] = 1.0; }
    v
}

fn recurrent_forward(digits: &[usize], w: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
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

fn gen_combos(n_inputs: usize, max_d: usize) -> Vec<Vec<usize>> {
    let mut result = vec![vec![]];
    for _ in 0..n_inputs {
        let mut nr = Vec::new();
        for combo in &result { for d in 0..max_d { let mut c = combo.clone(); c.push(d); nr.push(c); } }
        result = nr;
    }
    result
}

// ============================================================
// Readout methods
// ============================================================

/// 1. Nearest-mean (baseline)
fn readout_nearest_mean(train: &[(f32, usize)], test_charge: f32) -> usize {
    let nc = train.iter().map(|e| e.1 + 1).max().unwrap_or(1);
    let mut sums = vec![0.0f32; nc];
    let mut counts = vec![0usize; nc];
    for &(s, c) in train { sums[c] += s; counts[c] += 1; }
    let centroids: Vec<f32> = (0..nc).map(|c| if counts[c] > 0 { sums[c] / counts[c] as f32 } else { f32::NAN }).collect();
    centroids.iter().enumerate().filter(|(_, c)| !c.is_nan())
        .min_by(|a, b| (a.1 - test_charge).abs().partial_cmp(&(b.1 - test_charge).abs()).unwrap())
        .map(|(i, _)| i).unwrap_or(0)
}

/// 2. Threshold (binary: charge > threshold → 1)
fn readout_threshold(train: &[(f32, usize)], test_charge: f32) -> usize {
    // Find optimal threshold that maximizes train accuracy
    let mut charges_0: Vec<f32> = train.iter().filter(|e| e.1 == 0).map(|e| e.0).collect();
    let mut charges_1: Vec<f32> = train.iter().filter(|e| e.1 == 1).map(|e| e.0).collect();
    if charges_0.is_empty() || charges_1.is_empty() { return readout_nearest_mean(train, test_charge); }
    charges_0.sort_by(|a, b| a.partial_cmp(b).unwrap());
    charges_1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let max_0 = charges_0.last().unwrap();
    let min_1 = charges_1.first().unwrap();
    let threshold = (max_0 + min_1) / 2.0;
    if test_charge > threshold { 1 } else { 0 }
}

/// 3. Optimal 1D boundary: find best split points using all training charges
fn readout_boundary(train: &[(f32, usize)], test_charge: f32) -> usize {
    let nc = train.iter().map(|e| e.1 + 1).max().unwrap_or(1);
    if nc <= 1 { return 0; }

    // Collect all unique charges with their classes
    let mut all: Vec<(f32, usize)> = train.to_vec();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // For each class, find min and max charge
    let mut class_min = vec![f32::MAX; nc];
    let mut class_max = vec![f32::MIN; nc];
    for &(charge, cls) in &all {
        if charge < class_min[cls] { class_min[cls] = charge; }
        if charge > class_max[cls] { class_max[cls] = charge; }
    }

    // Build ordered class boundaries: find the "best" class for each charge range
    // Simple approach: sort classes by their mean charge, then use midpoints
    let mut class_means: Vec<(usize, f32)> = (0..nc).filter_map(|c| {
        let vals: Vec<f32> = train.iter().filter(|e| e.1 == c).map(|e| e.0).collect();
        if vals.is_empty() { None } else { Some((c, vals.iter().sum::<f32>() / vals.len() as f32)) }
    }).collect();
    class_means.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Find which class interval the test charge falls in
    for i in 0..class_means.len() - 1 {
        let mid = (class_means[i].1 + class_means[i + 1].1) / 2.0;
        if test_charge <= mid { return class_means[i].0; }
    }
    class_means.last().map(|c| c.0).unwrap_or(0)
}

/// 4. Min-max boundary: use the GAP between max of class k and min of class k+1
fn readout_minmax_gap(train: &[(f32, usize)], test_charge: f32) -> usize {
    let nc = train.iter().map(|e| e.1 + 1).max().unwrap_or(1);

    let mut class_data: Vec<(usize, f32, f32)> = Vec::new(); // (class, min, max)
    for c in 0..nc {
        let vals: Vec<f32> = train.iter().filter(|e| e.1 == c).map(|e| e.0).collect();
        if vals.is_empty() { continue; }
        let min = vals.iter().cloned().fold(f32::MAX, f32::min);
        let max = vals.iter().cloned().fold(f32::MIN, f32::max);
        class_data.push((c, min, max));
    }
    class_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Boundaries at midpoint of (max_class_k, min_class_k+1)
    for i in 0..class_data.len() - 1 {
        let boundary = (class_data[i].2 + class_data[i + 1].1) / 2.0;
        if test_charge <= boundary { return class_data[i].0; }
    }
    class_data.last().map(|c| c.0).unwrap_or(0)
}

/// 5. Percentile boundary: use class medians instead of means
fn readout_median(train: &[(f32, usize)], test_charge: f32) -> usize {
    let nc = train.iter().map(|e| e.1 + 1).max().unwrap_or(1);

    let mut class_medians: Vec<(usize, f32)> = Vec::new();
    for c in 0..nc {
        let mut vals: Vec<f32> = train.iter().filter(|e| e.1 == c).map(|e| e.0).collect();
        if vals.is_empty() { continue; }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = vals[vals.len() / 2];
        class_medians.push((c, median));
    }
    class_medians.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for i in 0..class_medians.len() - 1 {
        let mid = (class_medians[i].1 + class_medians[i + 1].1) / 2.0;
        if test_charge <= mid { return class_medians[i].0; }
    }
    class_medians.last().map(|c| c.0).unwrap_or(0)
}

// ============================================================
// Generic eval with pluggable readout
// ============================================================
fn eval_with_readout(
    w: &[Vec<f32>], bias: &[f32],
    train_n: usize, test_n: usize, max_d: usize,
    op: &dyn Fn(&[usize]) -> usize,
    readout_fn: fn(&[(f32, usize)], f32) -> usize,
) -> f64 {
    // Build training data
    let train_combos = gen_combos(train_n, max_d);
    let mut train_data = Vec::new();
    for combo in &train_combos {
        let target = op(combo);
        let act = recurrent_forward(combo, w, bias);
        let s: f32 = act.iter().sum();
        if s.is_nan() || s.is_infinite() { return 0.0; }
        train_data.push((s, target));
    }

    // Evaluate on test data
    let test_combos = gen_combos(test_n, max_d);
    let mut correct = 0;
    for combo in &test_combos {
        let target = op(combo);
        let act = recurrent_forward(combo, w, bias);
        let s: f32 = act.iter().sum();
        let pred = readout_fn(&train_data, s);
        if pred == target { correct += 1; }
    }
    correct as f64 / test_combos.len() as f64
}

// ============================================================
// Search chip (same as before)
// ============================================================
fn search_chip(
    n: usize, train_n: usize, max_d: usize, seed: u64,
    op: &dyn Fn(&[usize]) -> usize,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let input_dim = n + 4;
    let wr: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut rng = StdRng::seed_from_u64(seed);
    let total_params = n * input_dim + n;

    // Use nearest-mean for search (then test all readouts)
    let eval = |w: &[Vec<f32>], bias: &[f32]| -> f64 {
        let combos = gen_combos(train_n, max_d);
        let mut examples = Vec::new();
        for combo in &combos {
            let target = op(combo);
            let act = recurrent_forward(combo, w, bias);
            let s: f32 = act.iter().sum();
            if s.is_nan() || s.is_infinite() { return 0.0; }
            examples.push((s, target));
        }
        let mut correct = 0;
        for &(s, t) in &examples {
            if readout_nearest_mean(&examples, s) == t { correct += 1; }
        }
        correct as f64 / examples.len() as f64
    };

    let mut best_acc = 0.0f64;
    let mut best_w = vec![vec![0.0f32; input_dim]; n];
    let mut best_bias = vec![0.0f32; n];

    for _ in 0..3_000_000u64 {
        let w: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| wr[rng.gen_range(0..wr.len())]).collect()).collect();
        let bias: Vec<f32> = (0..n).map(|_| wr[rng.gen_range(0..wr.len())]).collect();
        let acc = eval(&w, &bias);
        if acc > best_acc { best_acc = acc; best_w = w; best_bias = bias; }
        if best_acc >= 1.0 { break; }
    }

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
        let acc = eval(&best_w, &best_bias);
        if acc >= current { current = acc; } else {
            if is_b { best_bias[i] = old; } else { best_w[i][j] = old; }
        }
        if current >= 1.0 { break; }
    }

    (best_w, best_bias)
}

fn main() {
    println!("=== READOUT METHOD SWEEP ===");
    println!("Same chips, different readouts. Which is universal?\n");

    let readouts: Vec<(&str, fn(&[(f32, usize)], f32) -> usize)> = vec![
        ("nearest_mean", readout_nearest_mean),
        ("threshold",    readout_threshold),
        ("boundary",     readout_boundary),
        ("minmax_gap",   readout_minmax_gap),
        ("median",       readout_median),
    ];

    let ops: Vec<(&str, Box<dyn Fn(&[usize]) -> usize>, usize, usize)> = vec![
        ("ADD",  Box::new(|d: &[usize]| d.iter().sum()),                     DIGITS, 3),
        ("XOR",  Box::new(|d: &[usize]| d.iter().fold(0, |a, &b| a ^ b)),   2, 3),
        ("OR",   Box::new(|d: &[usize]| d.iter().fold(0, |a, &b| a | b)),   2, 3),
        ("AND",  Box::new(|d: &[usize]| d.iter().fold(1, |a, &b| a & b)),   2, 3),
        ("NAND", Box::new(|d: &[usize]| 1 - d.iter().fold(1, |a, &b| a & b)), 2, 3),
        ("MAX",  Box::new(|d: &[usize]| *d.iter().max().unwrap()),           DIGITS, 3),
        ("MIN",  Box::new(|d: &[usize]| *d.iter().min().unwrap()),           DIGITS, 3),
    ];

    let n = 3;

    for (op_name, op_fn, max_d, train_n) in &ops {
        println!("=== {} ===", op_name);

        // Find chip
        let (w, bias) = search_chip(n, *train_n, *max_d, 42, op_fn.as_ref());

        // Test each readout
        print!("{:<14}", "readout");
        for n_in in &[2, 3, 4, 5, 6, 7, 8] {
            print!(" {:>5}-in", n_in);
        }
        println!();
        println!("{}", "-".repeat(65));

        for (r_name, r_fn) in &readouts {
            print!("{:<14}", r_name);
            for &n_in in &[2, 3, 4, 5, 6, 7, 8] {
                let acc = eval_with_readout(&w, &bias, *train_n, n_in, *max_d, op_fn.as_ref(), *r_fn);
                let s = format!("{:.0}%", acc * 100.0);
                print!(" {:>7}", s);
            }
            println!();
        }
        println!();
    }

    println!("=== DONE ===");
}
