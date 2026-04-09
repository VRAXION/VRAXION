//! Recurrent chip: same W matrix, multiple ticks, one digit per tick.
//!
//! Instead of chaining separate chips, reuse ONE chip recurrently:
//!   tick 1: input = [zeros(N), thermo_a] → act = "a"
//!   tick 2: input = [act_prev, thermo_b] → act = "a+b"
//!   tick 3: input = [act_prev, thermo_c] → act = "a+b+c"
//!   ...
//!
//! W matrix: N × (N + 4) — recurrent columns + new-digit columns
//! Same W, same bias, every tick. Scales to arbitrary input count.
//!
//! Run: cargo run --example chip_recurrent --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIGITS: usize = 5; // 0..4

fn signed_square(x: f32) -> f32 {
    x * x.abs()
}

fn thermo(val: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 4];
    for i in 0..val.min(4) { v[i] = 1.0; }
    v
}

/// Run recurrent chip for a sequence of digits.
/// W: N × (N+4), bias: N
/// Returns final activations after all digits processed.
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
                if j < w[i].len() {
                    sum += w[i][j] * inp;
                }
            }
            act[i] = signed_square(sum);
        }
    }
    act
}

struct NearestMean { centroids: Vec<f32> }
impl NearestMean {
    fn fit(examples: &[(f32, usize)], n_classes: usize) -> Self {
        let mut sums = vec![0.0f32; n_classes];
        let mut counts = vec![0usize; n_classes];
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

/// Evaluate recurrent chip on n-input addition
fn eval_recurrent(w: &[Vec<f32>], bias: &[f32], n_inputs: usize) -> f64 {
    let max_sum = (DIGITS - 1) * n_inputs;
    let n_classes = max_sum + 1;

    // Generate all digit combinations
    let combos = gen_combos(n_inputs);
    let mut examples = Vec::new();

    for combo in &combos {
        let target: usize = combo.iter().sum();
        let act = recurrent_forward(combo, w, bias);
        let act_sum: f32 = act.iter().sum();
        examples.push((act_sum, target));
    }

    let readout = NearestMean::fit(&examples, n_classes);
    examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / combos.len() as f64
}

/// Generate all combinations of n_inputs digits (0..DIGITS)
fn gen_combos(n: usize) -> Vec<Vec<usize>> {
    if n == 0 { return vec![vec![]]; }
    let sub = gen_combos(n - 1);
    let mut result = Vec::new();
    for combo in sub {
        for d in 0..DIGITS {
            let mut c = combo.clone();
            c.push(d);
            result.push(c);
        }
    }
    result
}

fn main() {
    println!("=== RECURRENT CHIP: same W, multiple ticks ===");
    println!("W: N × (N+4), one digit per tick, signed square + nearest-mean\n");

    let weight_range: Vec<i8> = (-2..=2).collect();
    let n = 3; // neurons
    let input_dim = n + 4; // recurrent(3) + thermo(4) = 7

    // --- Phase 1: Random search on 2-input addition ---
    println!("--- Phase 1: Random search (train on 2-input ADD) ---");
    let mut rng = StdRng::seed_from_u64(42);
    let mut best_acc_2 = 0.0f64;
    let mut best_w = vec![vec![0.0f32; input_dim]; n];
    let mut best_bias = vec![0.0f32; n];
    let samples = 5_000_000u64;

    for iter in 0..samples {
        let w: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..input_dim).map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32).collect())
            .collect();
        let bias: Vec<f32> = (0..n)
            .map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32)
            .collect();

        let acc = eval_recurrent(&w, &bias, 2);
        if acc > best_acc_2 {
            best_acc_2 = acc;
            best_w = w;
            best_bias = bias;
            if best_acc_2 >= 1.0 {
                println!("  100% at iter {}", iter + 1);
                break;
            }
        }
    }
    println!("  Random search (2-input): {:.1}% ({}M samples)", best_acc_2 * 100.0, samples / 1_000_000);

    // --- Phase 2: Perturbation refinement ---
    if best_acc_2 < 1.0 {
        println!("\n--- Phase 2: Perturbation refinement (2-input) ---");
        let mut w_p = best_w.clone();
        let mut b_p = best_bias.clone();
        let mut current = best_acc_2;
        let total_params = n * input_dim + n;

        for step in 0..500_000u64 {
            let idx = rng.gen_range(0..total_params);
            let delta: f32 = rng.gen_range(-0.5..0.5);

            let (old, is_b, i, j) = if idx < n * input_dim {
                let i = idx / input_dim; let j = idx % input_dim;
                let old = w_p[i][j]; w_p[i][j] += delta; (old, false, i, j)
            } else {
                let i = idx - n * input_dim;
                let old = b_p[i]; b_p[i] += delta; (old, true, i, 0)
            };

            let acc = eval_recurrent(&w_p, &b_p, 2);
            if acc >= current { current = acc; } else {
                if is_b { b_p[i] = old; } else { w_p[i][j] = old; }
            }
            if current >= 1.0 {
                println!("  100% at step {}", step + 1);
                best_w = w_p.clone(); best_bias = b_p.clone();
                break;
            }
        }
        if current < 1.0 {
            println!("  Perturbation: {:.1}%", current * 100.0);
            best_w = w_p; best_bias = b_p;
        }
    }

    // --- Phase 3: Test generalization to more inputs ---
    println!("\n--- Phase 3: Generalization (same W, more ticks) ---");
    println!("  Trained on 2-input addition. Testing on 2,3,4,5,6 inputs:\n");

    let mut results = Vec::new();
    for n_inputs in 2..=6 {
        let acc = eval_recurrent(&best_w, &best_bias, n_inputs);
        let total_examples = DIGITS.pow(n_inputs as u32);
        let max_sum = (DIGITS - 1) * n_inputs;
        let correct = (acc * total_examples as f64).round() as usize;
        println!(
            "  {}-input (0..{}): {:.1}% ({}/{} examples, {} classes)",
            n_inputs, max_sum, acc * 100.0, correct, total_examples, max_sum + 1
        );
        results.push((n_inputs, acc));
    }

    // --- Phase 4: Train on 3-input, test generalization ---
    println!("\n--- Phase 4: Train on 3-input, test generalization ---");
    let mut rng2 = StdRng::seed_from_u64(999);
    let mut best_w3 = best_w.clone();
    let mut best_b3 = best_bias.clone();
    let mut best_acc_3 = eval_recurrent(&best_w, &best_bias, 3);
    println!("  Starting from 2-input weights: {:.1}% on 3-input", best_acc_3 * 100.0);

    // Perturbation on 3-input
    let total_params = n * input_dim + n;
    for step in 0..500_000u64 {
        let idx = rng2.gen_range(0..total_params);
        let delta: f32 = rng2.gen_range(-0.3..0.3);

        let (old, is_b, i, j) = if idx < n * input_dim {
            let i = idx / input_dim; let j = idx % input_dim;
            let old = best_w3[i][j]; best_w3[i][j] += delta; (old, false, i, j)
        } else {
            let i = idx - n * input_dim;
            let old = best_b3[i]; best_b3[i] += delta; (old, true, i, 0)
        };

        let acc = eval_recurrent(&best_w3, &best_b3, 3);
        if acc >= best_acc_3 { best_acc_3 = acc; } else {
            if is_b { best_b3[i] = old; } else { best_w3[i][j] = old; }
        }
        if best_acc_3 >= 1.0 {
            println!("  100% on 3-input at step {}", step + 1);
            break;
        }
    }
    if best_acc_3 < 1.0 {
        println!("  Perturbation on 3-input: {:.1}%", best_acc_3 * 100.0);
    }

    println!("\n  Generalization from 3-input trained weights:");
    for n_inputs in 2..=6 {
        let acc = eval_recurrent(&best_w3, &best_b3, n_inputs);
        let total_examples = DIGITS.pow(n_inputs as u32);
        println!("  {}-input: {:.1}% ({} examples)", n_inputs, acc * 100.0, total_examples);
    }

    // --- Show weights ---
    println!("\n--- Final weights ---");
    println!("  W ({} × {}):", n, input_dim);
    let col_labels: Vec<String> = (0..n).map(|i| format!("r{}", i)).chain((0..4).map(|i| format!("d{}", i))).collect();
    println!("  cols: [{}]", col_labels.join(", "));
    for (i, row) in best_w.iter().enumerate() {
        let s: Vec<String> = row.iter().map(|v| format!("{:>6.2}", v)).collect();
        println!("    n{}: [{}]  bias={:.2}", i, s.join(", "), best_bias[i]);
    }

    println!("\n=== VERDICT ===");
    let gen_works = results.iter().all(|&(_, acc)| acc >= 0.95);
    if gen_works {
        println!("  RECURRENT CHIP GENERALIZES across input counts!");
        println!("  Same 3-neuron chip, same W, arbitrary number of additions.");
    } else {
        println!("  Recurrent chip works for trained input count but generalization is limited.");
        println!("  The spread accumulates over ticks → need tighter abstraction.");
    }
}
