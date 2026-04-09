//! Minimum viable chip: what's the absolute smallest config that works?
//!
//! Sweep: weight range × neuron count × bias type for each operation.
//! Find the minimum that achieves 100% on trained task AND generalizes.
//!
//! Run: cargo run --example chip_minimum --release

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

struct NearestMean { centroids: Vec<f32> }
impl NearestMean {
    fn fit(examples: &[(f32, usize)]) -> Self {
        let nc = examples.iter().map(|e| e.1 + 1).max().unwrap_or(1);
        let mut sums = vec![0.0f32; nc];
        let mut counts = vec![0usize; nc];
        for &(s, c) in examples { sums[c] += s; counts[c] += 1; }
        NearestMean { centroids: (0..nc).map(|c| if counts[c] > 0 { sums[c] / counts[c] as f32 } else { f32::NAN }).collect() }
    }
    fn predict(&self, s: f32) -> usize {
        self.centroids.iter().enumerate().filter(|(_, c)| !c.is_nan())
            .min_by(|a, b| (a.1 - s).abs().partial_cmp(&(b.1 - s).abs()).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }
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

fn eval_recurrent(w: &[Vec<f32>], bias: &[f32], n_inputs: usize, max_d: usize, op: &dyn Fn(&[usize]) -> usize) -> f64 {
    let combos = gen_combos(n_inputs, max_d);
    let mut examples = Vec::new();
    for combo in &combos {
        let target = op(combo);
        let act = recurrent_forward(combo, w, bias);
        let s: f32 = act.iter().sum();
        if s.is_nan() || s.is_infinite() { return 0.0; }
        examples.push((s, target));
    }
    let readout = NearestMean::fit(&examples);
    examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / combos.len() as f64
}

/// Exhaustive search over all weight+bias configs
fn exhaustive_search(
    n: usize, weight_vals: &[f32], bias_vals: &[f32],
    train_n: usize, max_d: usize,
    op: &dyn Fn(&[usize]) -> usize,
) -> (f64, Vec<Vec<f32>>, Vec<f32>) {
    let input_dim = n + 4;
    let n_w = n * input_dim;
    let n_b = n;
    let total = n_w + n_b;
    let w_card = weight_vals.len();
    let b_card = bias_vals.len();

    // Total configs = w_card^n_w * b_card^n_b
    let total_configs: u64 = (w_card as u64).pow(n_w as u32) * (b_card as u64).pow(n_b as u32);

    // Cap at 500M for practical reasons
    if total_configs > 500_000_000 {
        return random_search(n, weight_vals, bias_vals, train_n, max_d, op, 5_000_000);
    }

    let mut best_acc = 0.0f64;
    let mut best_w = vec![vec![0.0f32; input_dim]; n];
    let mut best_bias = vec![0.0f32; n];

    for config in 0..total_configs {
        let mut c = config;
        let mut w = vec![vec![0.0f32; input_dim]; n];
        let mut bias = vec![0.0f32; n];

        // Decode weights
        for i in 0..n {
            for j in 0..input_dim {
                w[i][j] = weight_vals[(c % w_card as u64) as usize];
                c /= w_card as u64;
            }
        }
        // Decode biases
        for i in 0..n {
            bias[i] = bias_vals[(c % b_card as u64) as usize];
            c /= b_card as u64;
        }

        let acc = eval_recurrent(&w, &bias, train_n, max_d, op);
        if acc > best_acc { best_acc = acc; best_w = w; best_bias = bias; }
        if best_acc >= 1.0 { break; }
    }

    (best_acc, best_w, best_bias)
}

fn random_search(
    n: usize, weight_vals: &[f32], bias_vals: &[f32],
    train_n: usize, max_d: usize,
    op: &dyn Fn(&[usize]) -> usize,
    samples: u64,
) -> (f64, Vec<Vec<f32>>, Vec<f32>) {
    let input_dim = n + 4;
    let mut rng = StdRng::seed_from_u64(42);
    let mut best_acc = 0.0f64;
    let mut best_w = vec![vec![0.0f32; input_dim]; n];
    let mut best_bias = vec![0.0f32; n];

    for _ in 0..samples {
        let w: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| weight_vals[rng.gen_range(0..weight_vals.len())]).collect()).collect();
        let bias: Vec<f32> = (0..n).map(|_| bias_vals[rng.gen_range(0..bias_vals.len())]).collect();
        let acc = eval_recurrent(&w, &bias, train_n, max_d, op);
        if acc > best_acc { best_acc = acc; best_w = w; best_bias = bias; }
        if best_acc >= 1.0 { break; }
    }
    (best_acc, best_w, best_bias)
}

fn main() {
    println!("=== MINIMUM VIABLE CHIP: what's the smallest that works? ===\n");

    // Weight ranges to test
    let ranges: Vec<(&str, Vec<f32>)> = vec![
        ("binary ±1",     vec![-1.0, 1.0]),
        ("binary 0,1",    vec![0.0, 1.0]),
        ("ternary ±1,0",  vec![-1.0, 0.0, 1.0]),
        ("quinary ±2",    vec![-2.0, -1.0, 0.0, 1.0, 2.0]),
    ];

    // Bias options to test
    let bias_opts: Vec<(&str, Vec<f32>)> = vec![
        ("no bias",       vec![0.0]),
        ("bias ±1",       vec![-1.0, 0.0, 1.0]),
        ("bias ±2",       vec![-2.0, -1.0, 0.0, 1.0, 2.0]),
    ];

    // Operations
    let ops: Vec<(&str, Box<dyn Fn(&[usize]) -> usize>, usize)> = vec![
        ("ADD", Box::new(|d: &[usize]| d.iter().sum()), DIGITS),
        ("XOR", Box::new(|d: &[usize]| d.iter().fold(0, |a, &b| a ^ b)), 2),
        ("OR",  Box::new(|d: &[usize]| d.iter().fold(0, |a, &b| a | b)), 2),
        ("AND", Box::new(|d: &[usize]| d.iter().fold(1, |a, &b| a & b)), 2),
        ("MAX", Box::new(|d: &[usize]| *d.iter().max().unwrap()), DIGITS),
    ];

    // Neuron counts to test
    let neuron_counts = [1, 2, 3];

    for (op_name, op_fn, max_d) in &ops {
        println!("=== {} (max_digit={}) ===", op_name, max_d);
        println!("{:<16} {:>4} {:>10} {:>8} {:>8} {:>8} {:>8}",
            "config", "n", "space", "train3", "gen4", "gen6", "gen8");
        println!("{}", "-".repeat(78));

        for &n in &neuron_counts {
            for (w_name, w_vals) in &ranges {
                for (b_name, b_vals) in &bias_opts {
                    let input_dim = n + 4;
                    let n_w = n * input_dim;
                    let n_b = n;
                    let total_space: u64 = (w_vals.len() as u64).saturating_pow(n_w as u32)
                        .saturating_mul((b_vals.len() as u64).saturating_pow(n_b as u32));

                    let label = format!("{} {}", w_name, b_name);

                    let (train_acc, best_w, best_bias) = exhaustive_search(
                        n, w_vals, b_vals, 3, *max_d, op_fn.as_ref(),
                    );

                    if train_acc < 0.5 {
                        // Skip configs that can't even train
                        println!("{:<16} {:>4} {:>10} {:>7.0}%       -       -       -",
                            label, n, if total_space > 1_000_000_000 { ">1B".to_string() } else { format!("{}", total_space) },
                            train_acc * 100.0);
                        continue;
                    }

                    // Test generalization
                    let gen4 = eval_recurrent(&best_w, &best_bias, 4, *max_d, op_fn.as_ref());
                    let gen6 = eval_recurrent(&best_w, &best_bias, 6, *max_d, op_fn.as_ref());
                    let gen8 = if *max_d <= 2 {
                        eval_recurrent(&best_w, &best_bias, 8, *max_d, op_fn.as_ref())
                    } else {
                        // Skip 8-input for large digit range (too slow)
                        -1.0
                    };

                    let perfect = train_acc >= 1.0 && gen4 >= 1.0 && gen6 >= 1.0 && (gen8 < 0.0 || gen8 >= 1.0);

                    println!("{:<16} {:>4} {:>10} {:>7.0}% {:>7.0}% {:>7.0}% {:>7}  {}",
                        label, n,
                        if total_space > 1_000_000_000 { ">1B".to_string() } else { format!("{}", total_space) },
                        train_acc * 100.0, gen4 * 100.0, gen6 * 100.0,
                        if gen8 >= 0.0 { format!("{:.0}%", gen8 * 100.0) } else { "skip".to_string() },
                        if perfect { "*** PERFECT ***" } else { "" });
                }
            }
        }
        println!();
    }

    println!("=== DONE ===");
}
