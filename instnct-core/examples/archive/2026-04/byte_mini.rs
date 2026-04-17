//! Mini byte ALU: 1-bit and 2-bit, EXHAUSTIVE search, ranked.
//!
//! 1-bit: 1 neuron × 2 input = 2 weights + 1 bias = 3 params → 3^3 = 27 (ternary)
//! 2-bit: 2 neurons × 4 input = 8 weights + 2 bias = 10 params → 3^10 = 59K
//! 3-bit: 3 neurons × 6 input = 18 weights + 3 bias = 21 params → 3^21 = 10B (sample)
//! 4-bit: 4 neurons × 8 input = 32 weights + 4 bias = 36 params → 3^36 (sample)
//!
//! ALL configs tested and ranked by accuracy.
//!
//! Run: cargo run --example byte_mini --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn relu(x: f32) -> f32 { x.max(0.0) }

fn to_bits(val: usize, bits: usize) -> Vec<f32> {
    (0..bits).map(|i| if (val >> i) & 1 == 1 { 1.0 } else { 0.0 }).collect()
}

fn from_charges(charges: &[f32]) -> usize {
    let mut val = 0usize;
    for (i, &c) in charges.iter().enumerate() {
        if c > 0.0 { val |= 1 << i; }
    }
    val
}

fn recurrent_byte(numbers: &[usize], w: &[Vec<f32>], bias: &[f32], bits: usize) -> Vec<f32> {
    let n = bits;
    let mut act = vec![0.0f32; n];
    for &num in numbers {
        let b = to_bits(num, bits);
        let mut input = Vec::with_capacity(n + bits);
        input.extend_from_slice(&act);
        input.extend_from_slice(&b);
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

fn eval_alu(w: &[Vec<f32>], bias: &[f32], bits: usize, n_inputs: usize, op: &dyn Fn(&[usize]) -> usize) -> (f64, usize, usize) {
    let max_val = (1 << bits) - 1;
    let combos = gen_combos(n_inputs, max_val);
    let mut correct = 0;
    for combo in &combos {
        let target = op(combo) & max_val;
        let charges = recurrent_byte(combo, w, bias, bits);
        let output = from_charges(&charges);
        if output == target { correct += 1; }
    }
    (correct as f64 / combos.len() as f64, correct, combos.len())
}

fn gen_combos(n_inputs: usize, max_val: usize) -> Vec<Vec<usize>> {
    let mut result = vec![vec![]];
    for _ in 0..n_inputs {
        let mut nr = Vec::new();
        for combo in &result {
            for d in 0..=max_val { let mut c = combo.clone(); c.push(d); nr.push(c); }
        }
        result = nr;
    }
    result
}

fn main() {
    println!("=== MINI BYTE ALU: exhaustive search ===\n");

    let ops: Vec<(&str, Box<dyn Fn(&[usize]) -> usize>)> = vec![
        ("ADD", Box::new(|d: &[usize]| d.iter().sum::<usize>())),
        ("XOR", Box::new(|d: &[usize]| d.iter().fold(0, |a, &b| a ^ b))),
        ("AND", Box::new(|d: &[usize]| d.iter().fold(0xFF, |a, &b| a & b))),
        ("OR",  Box::new(|d: &[usize]| d.iter().fold(0, |a, &b| a | b))),
        ("SUB", Box::new(|d: &[usize]| if d.len() >= 2 { d[0].wrapping_sub(d[1]) } else { d[0] })),
    ];

    let ternary: Vec<f32> = vec![-1.0, 0.0, 1.0];

    for &bits in &[1, 2, 3, 4] {
        let n = bits;
        let input_dim = bits * 2; // recurrent + input
        let total_params = n * input_dim + n;
        let total_configs = 3u64.pow(total_params as u32);
        let exhaustive = total_configs <= 100_000_000;

        println!("============================================================");
        println!("  {}-BIT ALU ({} neurons, {} params, 3^{} = {} configs) {}",
            bits, n, total_params, total_params, total_configs,
            if exhaustive { "EXHAUSTIVE" } else { "SAMPLED" });
        println!("============================================================\n");

        for (op_name, op_fn) in &ops {
            let max_val = (1 << bits) - 1;

            let mut best_acc = 0.0f64;
            let mut best_w = vec![vec![0.0f32; input_dim]; n];
            let mut best_bias = vec![0.0f32; n];
            let mut perfect_count = 0u64;
            let mut configs_checked = 0u64;

            let n_try = if exhaustive { total_configs } else { 10_000_000u64.min(total_configs) };

            let mut rng = StdRng::seed_from_u64(42);

            for iter in 0..n_try {
                let config = if exhaustive { iter } else { rng.gen_range(0..total_configs) };

                // Decode config
                let mut c = config;
                let mut w = vec![vec![0.0f32; input_dim]; n];
                let mut bias = vec![0.0f32; n];
                for i in 0..n {
                    for j in 0..input_dim {
                        w[i][j] = ternary[(c % 3) as usize];
                        c /= 3;
                    }
                }
                for i in 0..n {
                    bias[i] = ternary[(c % 3) as usize];
                    c /= 3;
                }

                let (acc, _, _) = eval_alu(&w, &bias, bits, 2, op_fn.as_ref());
                configs_checked += 1;

                if acc >= 1.0 { perfect_count += 1; }
                if acc > best_acc {
                    best_acc = acc;
                    best_w = w;
                    best_bias = bias;
                }
            }

            // Test generalization of best
            print!("  {:<5} train={:.0}%", op_name, best_acc * 100.0);
            if best_acc >= 0.99 {
                print!("  perfect={}", perfect_count);
                // Generalization
                for &n_in in &[2, 3, 4, 5, 6, 8] {
                    if bits <= 2 || n_in <= 4 {
                        let (acc, _, _) = eval_alu(&best_w, &best_bias, bits, n_in, op_fn.as_ref());
                        print!("  {}in:{:.0}%", n_in, acc * 100.0);
                    }
                }
            }
            println!();

            // Show best weights for perfect configs
            if best_acc >= 0.99 && bits <= 2 {
                println!("         W:");
                for (i, row) in best_w.iter().enumerate() {
                    let s: Vec<String> = row.iter().map(|v| format!("{:>2}", *v as i8)).collect();
                    println!("           n{}: [{}] bias={}", i, s.join(","), best_bias[i] as i8);
                }
            }
        }
        println!();
    }

    println!("=== DONE ===");
}
