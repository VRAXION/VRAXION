//! Linear holographic model with per-connection bias.
//!
//! Model: output[i] = sum_j(input[j] * w[i][j] + b[i][j])
//! No activation function. Per-connection bias instead of per-neuron.
//! Nearest-mean readout.
//!
//! Key: without signed_square, recurrence won't explode.
//! Test: can this model compose/recur for multi-input addition?
//!
//! Run: cargo run --example chip_linear --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIGITS: usize = 5;

fn thermo(val: usize, size: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; size];
    for i in 0..val.min(size) { v[i] = 1.0; }
    v
}

fn thermo_2(a: usize, b: usize) -> Vec<f32> {
    let mut v = thermo(a, 4);
    v.extend_from_slice(&thermo(b, 4));
    v
}

/// Linear holographic forward: NO activation function.
/// output[i] = sum_j(input[j] * w[i][j] + b[i][j])
fn linear_forward(input: &[f32], w: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<f32> {
    let n = w.len();
    let mut act = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = 0.0f32;
        for (j, &inp) in input.iter().enumerate() {
            if j < w[i].len() {
                sum += inp * w[i][j] + b[i][j];
            }
        }
        act[i] = sum;
    }
    act
}

/// Recurrent linear forward: same W, one digit per tick.
/// act starts at zeros, each tick feeds [act, thermo(digit)].
fn recurrent_linear(digits: &[usize], w: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<f32> {
    let n = w.len();
    let mut act = vec![0.0f32; n];
    for &digit in digits {
        let t = thermo(digit, 4);
        let mut input = Vec::with_capacity(n + 4);
        input.extend_from_slice(&act);
        input.extend_from_slice(&t);
        act = linear_forward(&input, w, b);
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

fn gen_combos(n: usize) -> Vec<Vec<usize>> {
    if n == 0 { return vec![vec![]]; }
    let mut result = vec![vec![]];
    for _ in 0..n {
        let mut new_result = Vec::new();
        for combo in &result {
            for d in 0..DIGITS {
                let mut c = combo.clone();
                c.push(d);
                new_result.push(c);
            }
        }
        result = new_result;
    }
    result
}

/// Evaluate on n-input addition (recurrent mode)
fn eval_recurrent(w: &[Vec<f32>], b: &[Vec<f32>], n_inputs: usize) -> f64 {
    let max_sum = (DIGITS - 1) * n_inputs;
    let combos = gen_combos(n_inputs);
    let mut examples = Vec::new();
    for combo in &combos {
        let target: usize = combo.iter().sum();
        let act = recurrent_linear(combo, w, b);
        let s: f32 = act.iter().sum();
        examples.push((s, target));
    }
    let readout = NearestMean::fit(&examples, max_sum + 1);
    examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / combos.len() as f64
}

/// Evaluate on 2-input addition (direct, non-recurrent)
fn eval_direct_2input(w: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    let mut examples = Vec::new();
    for a in 0..DIGITS {
        for b_val in 0..DIGITS {
            let input = thermo_2(a, b_val);
            let act = linear_forward(&input, w, b);
            let s: f32 = act.iter().sum();
            examples.push((s, a + b_val));
        }
    }
    let readout = NearestMean::fit(&examples, 9);
    examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / 25.0
}

fn main() {
    println!("=== LINEAR HOLOGRAPHIC: per-connection bias, no activation ===");
    println!("output[i] = sum_j(input[j] * w[i][j] + b[i][j])");
    println!("No signed_square. No explosion on recurrence.\n");

    let weight_range: Vec<i8> = (-2..=2).collect();
    let n = 3; // neurons

    // ===========================================
    // TEST 1: Direct 2-input ADD (8 inputs → 3 neurons)
    // ===========================================
    println!("--- TEST 1: Direct 2-input ADD ---");
    let input_dim = 8;
    let total_params = n * input_dim * 2; // w + b per connection
    println!("  {} neurons × {} inputs × 2 (w+b) = {} params\n", n, input_dim, total_params);

    let mut rng = StdRng::seed_from_u64(42);
    let mut best_acc = 0.0f64;
    let mut best_w = vec![vec![0.0f32; input_dim]; n];
    let mut best_b = vec![vec![0.0f32; input_dim]; n];

    for _ in 0..5_000_000u64 {
        let w: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32).collect()).collect();
        let b: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32).collect()).collect();
        let acc = eval_direct_2input(&w, &b);
        if acc > best_acc { best_acc = acc; best_w = w; best_b = b; }
        if best_acc >= 1.0 { break; }
    }
    println!("  Random search: {:.1}%", best_acc * 100.0);

    // Perturbation
    let mut current = best_acc;
    for step in 0..500_000u64 {
        let idx = rng.gen_range(0..total_params);
        let delta: f32 = rng.gen_range(-0.5..0.5);
        let half = n * input_dim;
        let (old, is_b_mat, i, j) = if idx < half {
            let i = idx / input_dim; let j = idx % input_dim;
            let old = best_w[i][j]; best_w[i][j] += delta; (old, false, i, j)
        } else {
            let idx2 = idx - half;
            let i = idx2 / input_dim; let j = idx2 % input_dim;
            let old = best_b[i][j]; best_b[i][j] += delta; (old, true, i, j)
        };
        let acc = eval_direct_2input(&best_w, &best_b);
        if acc >= current { current = acc; } else {
            if is_b_mat { best_b[i][j] = old; } else { best_w[i][j] = old; }
        }
        if current >= 1.0 {
            println!("  Perturbation: 100% at step {}", step + 1);
            break;
        }
    }
    if current < 1.0 { println!("  Perturbation: {:.1}%", current * 100.0); }

    // ===========================================
    // TEST 2: Recurrent (same W, one digit per tick)
    // W: N × (N+4), b: N × (N+4)
    // ===========================================
    println!("\n--- TEST 2: Recurrent linear (one digit per tick) ---");
    let rec_input_dim = n + 4; // recurrent(3) + thermo(4) = 7
    let rec_params = n * rec_input_dim * 2;
    println!("  {} neurons × {} inputs × 2 (w+b) = {} params\n", n, rec_input_dim, rec_params);

    // Random search on 2-input
    let mut rng2 = StdRng::seed_from_u64(777);
    let mut best_acc_r = 0.0f64;
    let mut best_wr = vec![vec![0.0f32; rec_input_dim]; n];
    let mut best_br = vec![vec![0.0f32; rec_input_dim]; n];

    for _ in 0..5_000_000u64 {
        let w: Vec<Vec<f32>> = (0..n).map(|_| (0..rec_input_dim).map(|_| weight_range[rng2.gen_range(0..weight_range.len())] as f32).collect()).collect();
        let b: Vec<Vec<f32>> = (0..n).map(|_| (0..rec_input_dim).map(|_| weight_range[rng2.gen_range(0..weight_range.len())] as f32).collect()).collect();
        let acc = eval_recurrent(&w, &b, 2);
        if acc > best_acc_r { best_acc_r = acc; best_wr = w; best_br = b; }
        if best_acc_r >= 1.0 { break; }
    }
    println!("  Random search (2-input): {:.1}%", best_acc_r * 100.0);

    // Perturbation on 2-input
    let mut current_r = best_acc_r;
    for step in 0..500_000u64 {
        let idx = rng2.gen_range(0..rec_params);
        let delta: f32 = rng2.gen_range(-0.5..0.5);
        let half = n * rec_input_dim;
        let (old, is_b_mat, i, j) = if idx < half {
            let i = idx / rec_input_dim; let j = idx % rec_input_dim;
            let old = best_wr[i][j]; best_wr[i][j] += delta; (old, false, i, j)
        } else {
            let idx2 = idx - half;
            let i = idx2 / rec_input_dim; let j = idx2 % rec_input_dim;
            let old = best_br[i][j]; best_br[i][j] += delta; (old, true, i, j)
        };
        let acc = eval_recurrent(&best_wr, &best_br, 2);
        if acc >= current_r { current_r = acc; } else {
            if is_b_mat { best_br[i][j] = old; } else { best_wr[i][j] = old; }
        }
        if current_r >= 1.0 {
            println!("  Perturbation (2-input): 100% at step {}", step + 1);
            break;
        }
    }
    if current_r < 1.0 { println!("  Perturbation (2-input): {:.1}%", current_r * 100.0); }

    // Generalization: test on 2-7 inputs
    println!("\n  Generalization (trained on 2-input):");
    for n_inputs in 2..=8 {
        let acc = eval_recurrent(&best_wr, &best_br, n_inputs);
        let total = DIGITS.pow(n_inputs as u32);
        println!("    {}-input: {:.1}% ({} examples)", n_inputs, acc * 100.0, total);
    }

    // ===========================================
    // TEST 3: Train on 3-input, test generalization
    // ===========================================
    println!("\n--- TEST 3: Train recurrent on 3-input ---");
    let mut w3 = best_wr.clone();
    let mut b3 = best_br.clone();
    let mut best_3 = eval_recurrent(&w3, &b3, 3);
    println!("  Starting from 2-input weights: {:.1}% on 3-input", best_3 * 100.0);

    let mut rng3 = StdRng::seed_from_u64(999);
    for step in 0..500_000u64 {
        let idx = rng3.gen_range(0..rec_params);
        let delta: f32 = rng3.gen_range(-0.3..0.3);
        let half = n * rec_input_dim;
        let (old, is_b_mat, i, j) = if idx < half {
            let i = idx / rec_input_dim; let j = idx % rec_input_dim;
            let old = w3[i][j]; w3[i][j] += delta; (old, false, i, j)
        } else {
            let idx2 = idx - half;
            let i = idx2 / rec_input_dim; let j = idx2 % rec_input_dim;
            let old = b3[i][j]; b3[i][j] += delta; (old, true, i, j)
        };
        let acc = eval_recurrent(&w3, &b3, 3);
        if acc >= best_3 { best_3 = acc; } else {
            if is_b_mat { b3[i][j] = old; } else { w3[i][j] = old; }
        }
        if best_3 >= 1.0 {
            println!("  100% on 3-input at step {}", step + 1);
            break;
        }
    }
    if best_3 < 1.0 { println!("  Perturbation: {:.1}%", best_3 * 100.0); }

    println!("\n  Generalization (trained on 3-input):");
    for n_inputs in 2..=8 {
        let acc = eval_recurrent(&w3, &b3, n_inputs);
        println!("    {}-input: {:.1}%", n_inputs, acc * 100.0);
    }

    // ===========================================
    // TEST 4: Train on 4-input
    // ===========================================
    println!("\n--- TEST 4: Train recurrent on 4-input ---");
    let mut w4 = w3.clone();
    let mut b4 = b3.clone();
    let mut best_4 = eval_recurrent(&w4, &b4, 4);
    println!("  Starting from 3-input weights: {:.1}% on 4-input", best_4 * 100.0);

    let mut rng4 = StdRng::seed_from_u64(1234);
    for step in 0..500_000u64 {
        let idx = rng4.gen_range(0..rec_params);
        let delta: f32 = rng4.gen_range(-0.2..0.2);
        let half = n * rec_input_dim;
        let (old, is_b_mat, i, j) = if idx < half {
            let i = idx / rec_input_dim; let j = idx % rec_input_dim;
            let old = w4[i][j]; w4[i][j] += delta; (old, false, i, j)
        } else {
            let idx2 = idx - half;
            let i = idx2 / rec_input_dim; let j = idx2 % rec_input_dim;
            let old = b4[i][j]; b4[i][j] += delta; (old, true, i, j)
        };
        let acc = eval_recurrent(&w4, &b4, 4);
        if acc >= best_4 { best_4 = acc; } else {
            if is_b_mat { b4[i][j] = old; } else { w4[i][j] = old; }
        }
        if best_4 >= 1.0 {
            println!("  100% on 4-input at step {}", step + 1);
            break;
        }
    }
    if best_4 < 1.0 { println!("  Perturbation: {:.1}%", best_4 * 100.0); }

    println!("\n  Generalization (trained on 4-input):");
    for n_inputs in 2..=8 {
        let acc = eval_recurrent(&w4, &b4, n_inputs);
        println!("    {}-input: {:.1}%", n_inputs, acc * 100.0);
    }

    // Show final weights
    println!("\n--- Best recurrent weights (from 2-input training) ---");
    for (i, (wr, br)) in best_wr.iter().zip(best_br.iter()).enumerate() {
        let ws: Vec<String> = wr.iter().map(|v| format!("{:>6.2}", v)).collect();
        let bs: Vec<String> = br.iter().map(|v| format!("{:>6.2}", v)).collect();
        println!("  n{} W: [{}]", i, ws.join(", "));
        println!("  n{} B: [{}]", i, bs.join(", "));
    }

    println!("\n=== VERDICT ===");
    println!("  Linear (no activation) + per-connection bias + recurrent.");
    println!("  Does it generalize across input counts?");
}
