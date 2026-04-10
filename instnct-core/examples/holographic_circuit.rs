//! Holographic one-shot learning on arithmetic circuits
//!
//! Only 25 patterns (5×5 digits) — well within holographic capacity.
//! Sweep: how many neurons (dimensions) needed per task?
//! No gradient, instant, pure matrix addition.
//!
//! Run: cargo run --example holographic_circuit --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIGITS: usize = 5;

/// Encode (a, b) as distributed vector using random projection
fn encode_input(a: usize, b: usize, dim: usize, proj: &Vec<Vec<f32>>) -> Vec<f32> {
    // Thermo encode a and b, then project to dim-dimensional space
    let mut thermo = vec![0.0f32; 8];
    for i in 0..a.min(4) { thermo[i] = 1.0; }
    for i in 0..b.min(4) { thermo[4 + i] = 1.0; }

    let mut v = vec![0.0f32; dim];
    for (j, &t) in thermo.iter().enumerate() {
        if t > 0.5 {
            for i in 0..dim { v[i] += proj[j][i]; }
        }
    }
    // Normalize
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 { for x in &mut v { *x /= norm; } }
    v
}

/// Encode output as one-hot-ish (target value at index)
fn encode_output(target: usize, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    if target < dim { v[target] = 1.0; }
    v
}

struct HoloCircuit {
    dim: usize,
    matrix: Vec<f32>,  // dim × dim
    proj: Vec<Vec<f32>>,
}

impl HoloCircuit {
    fn new(dim: usize, rng: &mut StdRng) -> Self {
        let proj: Vec<Vec<f32>> = (0..8).map(|_| {
            (0..dim).map(|_| rng.gen_range(-1.0..1.0f32)).collect()
        }).collect();
        HoloCircuit { dim, matrix: vec![0.0; dim * dim], proj }
    }

    fn store(&mut self, a: usize, b: usize, target: usize) {
        let inp = encode_input(a, b, self.dim, &self.proj);
        let out = encode_output(target, self.dim);
        for i in 0..self.dim {
            for j in 0..self.dim {
                self.matrix[i * self.dim + j] += inp[j] * out[i];
            }
        }
    }

    fn predict(&self, a: usize, b: usize) -> usize {
        let inp = encode_input(a, b, self.dim, &self.proj);
        let mut output = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                output[i] += self.matrix[i * self.dim + j] * inp[j];
            }
        }
        output.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }

    fn accuracy(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut correct = 0;
        for a in 0..DIGITS { for b in 0..DIGITS {
            if self.predict(a, b) == op(a, b) { correct += 1; }
        }}
        correct as f64 / 25.0
    }

    /// Quantize matrix, test accuracy
    fn quantized_accuracy(&self, op: fn(usize, usize) -> usize, levels: usize) -> f64 {
        let min_v = self.matrix.iter().fold(f32::MAX, |a, &b| a.min(b));
        let max_v = self.matrix.iter().fold(f32::MIN, |a, &b| a.max(b));
        if (max_v - min_v).abs() < 1e-10 { return 0.0; }
        let step = (max_v - min_v) / (levels - 1) as f32;

        let quantized: Vec<f32> = self.matrix.iter().map(|&v| {
            let idx = ((v - min_v) / step).round() as usize;
            min_v + idx.min(levels - 1) as f32 * step
        }).collect();

        let mut correct = 0;
        for a in 0..DIGITS { for b in 0..DIGITS {
            let inp = encode_input(a, b, self.dim, &self.proj);
            let mut output = vec![0.0f32; self.dim];
            for i in 0..self.dim {
                for j in 0..self.dim {
                    output[i] += quantized[i * self.dim + j] * inp[j];
                }
            }
            let pred = output.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if pred == op(a, b) { correct += 1; }
        }}
        correct as f64 / 25.0
    }
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }
fn op_xor(a: usize, b: usize) -> usize { a ^ b }
fn op_and(a: usize, b: usize) -> usize { a & b }

fn main() {
    println!("=== HOLOGRAPHIC CIRCUIT: one-shot arithmetic ===\n");
    println!("25 patterns (5×5), sweep dimensions, no gradient\n");

    let ops: Vec<(&str, fn(usize, usize) -> usize, usize)> = vec![
        ("ADD",   op_add,     9),  // max output = 8
        ("MUL",   op_mul,     17), // max output = 16
        ("MAX",   op_max,     5),  // max output = 4
        ("MIN",   op_min,     5),
        ("|a-b|", op_sub_abs, 5),
        ("XOR",   op_xor,     8),
        ("AND",   op_and,     5),
    ];

    let dims: Vec<usize> = vec![4, 8, 10, 12, 16, 20, 24, 32, 48, 64, 128];
    let n_seeds = 20;

    // =========================================================
    // TEST 1: Dimension sweep — min dim for 100%
    // =========================================================
    println!("--- Dimension sweep (20 seeds, float) ---\n");
    print!("{:>8}", "task");
    for &d in &dims { print!(" {:>5}", format!("d={}", d)); }
    println!("  min_100%");
    println!("{}", "=".repeat(8 + dims.len() * 6 + 10));

    for &(name, op, max_out) in &ops {
        print!("{:>8}", name);
        let mut min_dim_100 = 0usize;

        for &dim in &dims {
            if dim < max_out { print!("     -"); continue; }

            let mut solved = 0;
            for seed in 0..n_seeds {
                let mut rng = StdRng::seed_from_u64(seed as u64);
                let mut hc = HoloCircuit::new(dim, &mut rng);
                for a in 0..DIGITS { for b in 0..DIGITS {
                    hc.store(a, b, op(a, b));
                }}
                if hc.accuracy(op) >= 1.0 { solved += 1; }
            }
            let rate = solved as f64 / n_seeds as f64;
            print!(" {:>4.0}%", rate * 100.0);

            if rate >= 1.0 && min_dim_100 == 0 { min_dim_100 = dim; }
        }
        if min_dim_100 > 0 { println!("  d={}", min_dim_100); }
        else { println!("  >128"); }
    }

    // =========================================================
    // TEST 2: Quantization on best dim
    // =========================================================
    println!("\n--- Quantization (graceful degradation test) ---\n");
    println!("{:>8} {:>6}  {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "task", "dim", "float", "8bit", "4bit", "3bit", "2bit", "1bit");
    println!("{}", "=".repeat(65));

    for &(name, op, max_out) in &ops {
        // Use dim=32 for all
        let dim = 32;
        if dim < max_out { continue; }

        let mut float_acc = 0.0f64;
        let mut q_accs = vec![0.0f64; 5]; // 8,4,3,2,1 bit

        for seed in 0..n_seeds {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let mut hc = HoloCircuit::new(dim, &mut rng);
            for a in 0..DIGITS { for b in 0..DIGITS {
                hc.store(a, b, op(a, b));
            }}
            float_acc += hc.accuracy(op);
            for (qi, &levels) in [256, 16, 8, 4, 2].iter().enumerate() {
                q_accs[qi] += hc.quantized_accuracy(op, levels);
            }
        }
        float_acc /= n_seeds as f64;
        for q in &mut q_accs { *q /= n_seeds as f64; }

        println!("{:>8} {:>6}  {:>7.0}% {:>7.0}% {:>7.0}% {:>7.0}% {:>7.0}% {:>7.0}%",
            name, dim,
            float_acc * 100.0,
            q_accs[0] * 100.0, q_accs[1] * 100.0, q_accs[2] * 100.0,
            q_accs[3] * 100.0, q_accs[4] * 100.0);
    }

    // =========================================================
    // TEST 3: Scaling — 10 digits (100 patterns)
    // =========================================================
    println!("\n--- Scaling: 10 digits (100 patterns) ---\n");
    let digits_10 = 10;

    print!("{:>8}", "task");
    let big_dims = [16, 32, 64, 128, 256, 512];
    for &d in &big_dims { print!(" {:>5}", format!("d={}", d)); }
    println!();
    println!("{}", "=".repeat(8 + big_dims.len() * 6));

    let ops_10: Vec<(&str, fn(usize, usize) -> usize, usize)> = vec![
        ("ADD",   op_add,     19), // max = 18
        ("MAX",   op_max,     10),
        ("|a-b|", op_sub_abs, 10),
    ];

    for &(name, op, max_out) in &ops_10 {
        print!("{:>8}", name);
        for &dim in &big_dims {
            if dim < max_out { print!("     -"); continue; }
            let mut solved = 0;
            for seed in 0..n_seeds {
                let mut rng = StdRng::seed_from_u64(seed as u64);
                // Need bigger projections for 10 digits
                let proj: Vec<Vec<f32>> = (0..18).map(|_| {  // 9+9 thermo bits
                    (0..dim).map(|_| rng.gen_range(-1.0..1.0f32)).collect()
                }).collect();

                let mut matrix = vec![0.0f32; dim * dim];
                let mut correct = 0;

                // Store all 100 pairs
                for a in 0..digits_10 { for b in 0..digits_10 {
                    let target = op(a, b);
                    if target >= dim { continue; }

                    // Encode
                    let mut thermo = vec![0.0f32; 18];
                    for i in 0..a.min(9) { thermo[i] = 1.0; }
                    for i in 0..b.min(9) { thermo[9 + i] = 1.0; }
                    let mut inp = vec![0.0f32; dim];
                    for (j, &t) in thermo.iter().enumerate() {
                        if t > 0.5 && j < proj.len() {
                            for i in 0..dim { inp[i] += proj[j][i]; }
                        }
                    }
                    let norm: f32 = inp.iter().map(|x| x*x).sum::<f32>().sqrt();
                    if norm > 1e-8 { for x in &mut inp { *x /= norm; } }

                    // Store
                    for i in 0..dim {
                        if target < dim {
                            matrix[i * dim + target] += inp[i];
                        }
                    }
                }}

                // Test all 100
                for a in 0..digits_10 { for b in 0..digits_10 {
                    let target = op(a, b);
                    if target >= dim { continue; }

                    let mut thermo = vec![0.0f32; 18];
                    for i in 0..a.min(9) { thermo[i] = 1.0; }
                    for i in 0..b.min(9) { thermo[9 + i] = 1.0; }
                    let mut inp = vec![0.0f32; dim];
                    for (j, &t) in thermo.iter().enumerate() {
                        if t > 0.5 && j < proj.len() {
                            for i in 0..dim { inp[i] += proj[j][i]; }
                        }
                    }
                    let norm: f32 = inp.iter().map(|x| x*x).sum::<f32>().sqrt();
                    if norm > 1e-8 { for x in &mut inp { *x /= norm; } }

                    let mut output = vec![0.0f32; dim];
                    for i in 0..dim { for j in 0..dim {
                        output[i] += matrix[i * dim + j] * inp[j];
                    }}
                    let pred = output.iter().enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i).unwrap_or(0);
                    if pred == target { correct += 1; }
                }}
                let total = digits_10 * digits_10;
                if correct == total { solved += 1; }
            }
            print!(" {:>4.0}%", solved as f64 / n_seeds as f64 * 100.0);
        }
        println!();
    }

    println!("\n=== DONE ===");
}
