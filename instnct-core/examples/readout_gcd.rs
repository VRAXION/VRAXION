//! GCD/consensus readout: filter neurons by agreement, read only those.
//!
//! Idea: not all neurons contribute — only those that "agree" (share common factor).
//! This might help OR where charges scatter but "agreeing" neurons stay consistent.
//!
//! Run: cargo run --example readout_gcd --release

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

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

// ============================================================
// Readout variants — all work on the raw activation vector
// ============================================================

/// Feature extractors: each takes activations and returns a single f32 for readout
type FeatureFn = fn(&[f32]) -> f32;

fn feat_sum(act: &[f32]) -> f32 { act.iter().sum() }

fn feat_max(act: &[f32]) -> f32 { act.iter().cloned().fold(f32::MIN, f32::max) }

fn feat_min_nonzero(act: &[f32]) -> f32 {
    act.iter().cloned().filter(|&x| x > 0.001).fold(f32::MAX, f32::min)
}

fn feat_product(act: &[f32]) -> f32 {
    let p: f32 = act.iter().filter(|&&x| x > 0.001).product();
    if p == 0.0 { 0.0 } else { p.ln() } // log-product to avoid overflow
}

fn feat_gcd_filtered_sum(act: &[f32]) -> f32 {
    // Round to integers, find GCD of nonzero values, sum only those with GCD > 1
    let ints: Vec<u32> = act.iter().map(|&x| (x * 100.0).round().max(0.0) as u32).collect();
    let nonzero: Vec<u32> = ints.iter().cloned().filter(|&x| x > 0).collect();
    if nonzero.len() < 2 { return act.iter().sum(); }
    let g = nonzero.iter().cloned().reduce(gcd).unwrap_or(1);
    if g > 1 {
        // Only sum neurons that are multiples of GCD
        act.iter().zip(ints.iter()).filter(|(_, &i)| i > 0 && i % g == 0).map(|(a, _)| a).sum()
    } else {
        act.iter().sum() // fallback to sum
    }
}

fn feat_ratio(act: &[f32]) -> f32 {
    // Ratio of largest to smallest nonzero
    let nonzero: Vec<f32> = act.iter().cloned().filter(|&x| x > 0.001).collect();
    if nonzero.len() < 2 { return act.iter().sum(); }
    let max = nonzero.iter().cloned().fold(f32::MIN, f32::max);
    let min = nonzero.iter().cloned().fold(f32::MAX, f32::min);
    if min > 0.001 { max / min } else { max }
}

fn feat_n_active(act: &[f32]) -> f32 {
    // Count of neurons with activation > threshold
    act.iter().filter(|&&x| x > 0.1).count() as f32
}

fn feat_gcd_value(act: &[f32]) -> f32 {
    // The GCD itself as the feature
    let ints: Vec<u32> = act.iter().map(|&x| (x * 100.0).round().max(0.0) as u32).collect();
    let nonzero: Vec<u32> = ints.iter().cloned().filter(|&x| x > 0).collect();
    if nonzero.is_empty() { return 0.0; }
    nonzero.iter().cloned().reduce(gcd).unwrap_or(0) as f32
}

fn feat_per_neuron_concat(act: &[f32]) -> f32 {
    // Encode each neuron's activation as a "digit" in a base-100 number
    // This preserves per-neuron info without collapsing to sum
    let mut val = 0.0f32;
    for (i, &a) in act.iter().enumerate() {
        val += a * (1000.0f32).powi(i as i32);
    }
    val
}

// ============================================================
// Generic eval with feature extractor
// ============================================================
struct NearestMean { centroids: Vec<f32> }
impl NearestMean {
    fn fit(examples: &[(f32, usize)]) -> Self {
        let nc = examples.iter().map(|e| e.1 + 1).max().unwrap_or(1);
        let mut sums = vec![0.0f32; nc];
        let mut counts = vec![0usize; nc];
        for &(s, c) in examples { if !s.is_nan() && !s.is_infinite() { sums[c] += s; counts[c] += 1; } }
        NearestMean { centroids: (0..nc).map(|c| if counts[c] > 0 { sums[c] / counts[c] as f32 } else { f32::NAN }).collect() }
    }
    fn predict(&self, s: f32) -> usize {
        if s.is_nan() || s.is_infinite() { return 0; }
        self.centroids.iter().enumerate().filter(|(_, c)| !c.is_nan())
            .min_by(|a, b| (a.1 - s).abs().partial_cmp(&(b.1 - s).abs()).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }
}

fn eval_feature(
    w: &[Vec<f32>], bias: &[f32],
    train_n: usize, test_n: usize, max_d: usize,
    op: &dyn Fn(&[usize]) -> usize,
    feat: FeatureFn,
) -> f64 {
    let train_combos = gen_combos(train_n, max_d);
    let mut train_data = Vec::new();
    for combo in &train_combos {
        let target = op(combo);
        let act = recurrent_forward(combo, w, bias);
        let f = feat(&act);
        train_data.push((f, target));
    }
    let readout = NearestMean::fit(&train_data);

    let test_combos = gen_combos(test_n, max_d);
    let mut correct = 0;
    for combo in &test_combos {
        let target = op(combo);
        let act = recurrent_forward(combo, w, bias);
        let f = feat(&act);
        if readout.predict(f) == target { correct += 1; }
    }
    correct as f64 / test_combos.len() as f64
}

// Search chip
fn search_chip(n: usize, train_n: usize, max_d: usize, seed: u64, op: &dyn Fn(&[usize]) -> usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let input_dim = n + 4;
    let wr: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut rng = StdRng::seed_from_u64(seed);
    let total_params = n * input_dim + n;
    let mut best_acc = 0.0f64;
    let mut best_w = vec![vec![0.0f32; input_dim]; n];
    let mut best_bias = vec![0.0f32; n];

    let eval_nm = |w: &[Vec<f32>], bias: &[f32]| -> f64 {
        eval_feature(w, bias, train_n, train_n, max_d, op, feat_sum)
    };

    for _ in 0..3_000_000u64 {
        let w: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| wr[rng.gen_range(0..wr.len())]).collect()).collect();
        let bias: Vec<f32> = (0..n).map(|_| wr[rng.gen_range(0..wr.len())]).collect();
        let acc = eval_nm(&w, &bias);
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
        let acc = eval_nm(&best_w, &best_bias);
        if acc >= current { current = acc; } else {
            if is_b { best_bias[i] = old; } else { best_w[i][j] = old; }
        }
        if current >= 1.0 { break; }
    }
    (best_w, best_bias)
}

fn main() {
    println!("=== GCD / CONSENSUS READOUT TEST ===\n");

    let features: Vec<(&str, FeatureFn)> = vec![
        ("sum",            feat_sum),
        ("max",            feat_max),
        ("min_nonzero",    feat_min_nonzero),
        ("log_product",    feat_product),
        ("gcd_filter_sum", feat_gcd_filtered_sum),
        ("ratio",          feat_ratio),
        ("n_active",       feat_n_active),
        ("gcd_value",      feat_gcd_value),
        ("per_neuron",     feat_per_neuron_concat),
    ];

    let ops: Vec<(&str, Box<dyn Fn(&[usize]) -> usize>, usize)> = vec![
        ("ADD", Box::new(|d: &[usize]| d.iter().sum()), DIGITS),
        ("OR",  Box::new(|d: &[usize]| d.iter().fold(0, |a, &b| a | b)), 2),
        ("AND", Box::new(|d: &[usize]| d.iter().fold(1, |a, &b| a & b)), 2),
        ("XOR", Box::new(|d: &[usize]| d.iter().fold(0, |a, &b| a ^ b)), 2),
        ("MAX", Box::new(|d: &[usize]| *d.iter().max().unwrap()), DIGITS),
    ];

    let n = 3;

    for (op_name, op_fn, max_d) in &ops {
        println!("=== {} ===", op_name);
        let (w, bias) = search_chip(n, 3, *max_d, 42, op_fn.as_ref());

        // Also search with different seeds and pick best
        let mut best_w = w.clone();
        let mut best_b = bias.clone();
        let mut best_gen = 0.0f64;
        for seed in [42, 137, 314, 777, 999, 1234, 4242, 7777] {
            let (ws, bs) = search_chip(n, 3, *max_d, seed, op_fn.as_ref());
            let gen = eval_feature(&ws, &bs, 3, 6, *max_d, op_fn.as_ref(), feat_sum);
            if gen > best_gen { best_gen = gen; best_w = ws; best_b = bs; }
        }

        print!("{:<16}", "feature");
        for n_in in &[2, 3, 4, 5, 6, 8] {
            print!(" {:>6}-in", n_in);
        }
        println!();
        println!("{}", "-".repeat(60));

        for (f_name, f_fn) in &features {
            print!("{:<16}", f_name);
            for &n_in in &[2, 3, 4, 5, 6, 8] {
                let acc = eval_feature(&best_w, &best_b, 3, n_in, *max_d, op_fn.as_ref(), *f_fn);
                print!(" {:>6.0}%", acc * 100.0);
            }
            println!();
        }
        println!();
    }

    println!("=== DONE ===");
}
