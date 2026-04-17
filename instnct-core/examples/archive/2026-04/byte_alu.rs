//! Byte-level neural ALU: 8 neurons = 8 bits = 1 byte.
//!
//! Input: binary encoded (8 bits per number)
//! Output: 8 neurons, threshold readout (charge > 0 → 1)
//! No nearest-mean, no centroids, no calibration.
//!
//! Test: ADD, XOR, AND, OR, SUB on 8-bit numbers.
//! Recurrent: accumulate one number per tick.
//!
//! Run: cargo run --example byte_alu --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BITS: usize = 8;

/// Number → binary LSB-first
fn to_bits(val: usize) -> Vec<f32> {
    (0..BITS).map(|i| if (val >> i) & 1 == 1 { 1.0 } else { 0.0 }).collect()
}

/// Binary LSB-first → number (threshold readout: charge > 0 → 1)
fn from_charges(charges: &[f32]) -> usize {
    let mut val = 0usize;
    for (i, &c) in charges.iter().enumerate() {
        if i < BITS && c > 0.0 {
            val |= 1 << i;
        }
    }
    val
}

fn relu(x: f32) -> f32 { x.max(0.0) }

/// Recurrent forward: one number per tick, 8 neurons.
/// W: 8 × (8+8) = 8×16 = 128 weights, bias: 8
fn recurrent_byte(
    numbers: &[usize],
    w: &[Vec<f32>],
    bias: &[f32],
) -> Vec<f32> {
    let n = BITS;
    let mut act = vec![0.0f32; n];
    for &num in numbers {
        let bits = to_bits(num);
        let mut input = Vec::with_capacity(n + BITS);
        input.extend_from_slice(&act);       // recurrent: previous 8 activations
        input.extend_from_slice(&bits);      // new: 8 input bits
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

/// Evaluate byte ALU on an operation
fn eval_byte_alu(
    w: &[Vec<f32>],
    bias: &[f32],
    n_inputs: usize,
    max_val: usize,
    op: &dyn Fn(&[usize]) -> usize,
) -> (f64, usize, usize) {
    let mut correct = 0usize;
    let mut total = 0usize;

    // Generate all combos (or sample if too many)
    let combos = gen_combos_bounded(n_inputs, max_val);

    for combo in &combos {
        let target = op(combo) & 0xFF; // mask to 8 bits
        let charges = recurrent_byte(combo, w, bias);
        let output = from_charges(&charges);
        if output == target { correct += 1; }
        total += 1;
    }
    (correct as f64 / total as f64, correct, total)
}

fn gen_combos_bounded(n_inputs: usize, max_val: usize) -> Vec<Vec<usize>> {
    // If total combos > 50K, sample randomly
    let total = (max_val + 1).pow(n_inputs as u32);
    if total <= 50_000 {
        let mut result = vec![vec![]];
        for _ in 0..n_inputs {
            let mut nr = Vec::new();
            for combo in &result {
                for d in 0..=max_val {
                    let mut c = combo.clone();
                    c.push(d);
                    nr.push(c);
                }
            }
            result = nr;
        }
        result
    } else {
        // Random sample 50K
        let mut rng = StdRng::seed_from_u64(12345);
        (0..50_000).map(|_| {
            (0..n_inputs).map(|_| rng.gen_range(0..=max_val)).collect()
        }).collect()
    }
}

/// Search byte ALU weights
fn search_byte_alu(
    max_val: usize,
    train_n: usize,
    op: &dyn Fn(&[usize]) -> usize,
    seed: u64,
    random_samples: u64,
    perturb_steps: u64,
) -> (Vec<Vec<f32>>, Vec<f32>, f64) {
    let n = BITS;
    let input_dim = BITS + BITS; // 8 recurrent + 8 input = 16
    let wr: Vec<f32> = vec![-1.0, 0.0, 1.0]; // ternary for smaller search space
    let mut rng = StdRng::seed_from_u64(seed);
    let total_params = n * input_dim + n; // 128 + 8 = 136

    let eval = |w: &[Vec<f32>], b: &[f32]| -> f64 {
        eval_byte_alu(w, b, train_n, max_val, op).0
    };

    let mut best_acc = 0.0f64;
    let mut best_w = vec![vec![0.0f32; input_dim]; n];
    let mut best_bias = vec![0.0f32; n];

    // Random search
    for _ in 0..random_samples {
        let w: Vec<Vec<f32>> = (0..n).map(|_| {
            (0..input_dim).map(|_| wr[rng.gen_range(0..wr.len())]).collect()
        }).collect();
        let bias: Vec<f32> = (0..n).map(|_| wr[rng.gen_range(0..wr.len())]).collect();
        let acc = eval(&w, &bias);
        if acc > best_acc {
            best_acc = acc;
            best_w = w;
            best_bias = bias;
            if best_acc >= 0.5 {
                println!("      random hit {:.1}%", best_acc * 100.0);
            }
        }
        if best_acc >= 1.0 { return (best_w, best_bias, 1.0); }
    }
    println!("      random best: {:.1}%", best_acc * 100.0);

    // Perturbation
    let mut current = best_acc;
    for step in 0..perturb_steps {
        let idx = rng.gen_range(0..total_params);
        let delta: f32 = rng.gen_range(-0.5..0.5);
        let (old, is_b, i, j) = if idx < n * input_dim {
            let i = idx / input_dim; let j = idx % input_dim;
            let old = best_w[i][j]; best_w[i][j] += delta; (old, false, i, j)
        } else {
            let i = idx - n * input_dim;
            let old = best_bias[i]; best_bias[i] += delta; (old, true, i, 0)
        };
        let acc = eval(&best_w, &best_bias);
        if acc >= current {
            current = acc;
            if current >= 1.0 {
                println!("      perturb 100% at step {}", step + 1);
                return (best_w, best_bias, 1.0);
            }
        } else {
            if is_b { best_bias[i] = old; } else { best_w[i][j] = old; }
        }
        if (step + 1) % 200_000 == 0 {
            println!("      perturb step {}: {:.1}%", step + 1, current * 100.0);
        }
    }
    println!("      perturb final: {:.1}%", current * 100.0);
    (best_w, best_bias, current)
}

fn main() {
    println!("=== BYTE-LEVEL NEURAL ALU ===");
    println!("8 neurons = 8 bits. Binary in, threshold out. No nearest-mean.\n");

    // Start small: 4-bit first (0..15), then try 8-bit
    for &(bits_label, max_val) in &[("4-bit (0..15)", 15), ("8-bit (0..255)", 255)] {
        println!("============================================================");
        println!("  {}", bits_label);
        println!("============================================================\n");

        let ops: Vec<(&str, Box<dyn Fn(&[usize]) -> usize>)> = vec![
            ("ADD", Box::new(|d: &[usize]| d.iter().sum::<usize>())),
            ("XOR", Box::new(|d: &[usize]| d.iter().fold(0, |a, &b| a ^ b))),
            ("AND", Box::new(|d: &[usize]| d.iter().fold(0xFF, |a, &b| a & b))),
            ("OR",  Box::new(|d: &[usize]| d.iter().fold(0, |a, &b| a | b))),
        ];

        let random_n = if max_val <= 15 { 3_000_000u64 } else { 1_000_000 };
        let perturb_n = if max_val <= 15 { 1_000_000u64 } else { 500_000 };

        for (op_name, op_fn) in &ops {
            println!("  --- {} ---", op_name);

            // Train on 2-input
            println!("    Training on 2-input:");
            let (w, bias, train_acc) = search_byte_alu(
                max_val, 2, op_fn.as_ref(), 42, random_n, perturb_n,
            );

            // Test generalization
            println!("    Generalization:");
            for &n_in in &[2, 3, 4] {
                let test_max = if max_val > 15 { 31 } else { max_val }; // limit for speed
                let (acc, correct, total) = eval_byte_alu(&w, &bias, n_in, test_max, op_fn.as_ref());
                println!("      {}-input (0..{}): {:.1}% ({}/{})", n_in, test_max, acc * 100.0, correct, total);
            }

            // Show a few examples
            if train_acc >= 0.5 {
                println!("    Examples:");
                let test_vals = if max_val <= 15 {
                    vec![(3, 5), (7, 8), (15, 1), (0, 0), (12, 3)]
                } else {
                    vec![(42, 13), (255, 1), (128, 127), (0, 0), (100, 200)]
                };
                for (a, b) in test_vals {
                    let charges = recurrent_byte(&[a, b], &w, &bias);
                    let output = from_charges(&charges);
                    let target = (op_fn)(&[a, b]) & 0xFF;
                    let ok = if output == target { "✓" } else { "✗" };
                    let bits_out: String = (0..8).map(|i| if (output >> i) & 1 == 1 { '1' } else { '0' }).collect();
                    println!("      {}({}, {}) = {} (expect {}) [{}] {}", op_name, a, b, output, target, bits_out, ok);
                }
            }
            println!();
        }
    }

    println!("=== DONE ===");
}
