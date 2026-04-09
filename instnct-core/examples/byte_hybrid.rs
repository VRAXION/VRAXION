//! Hybrid: thermometer input → binary output.
//! Best of both: thermo accumulation (proven) + binary readout (no calibration).
//!
//! Each tick: [8 recurrent bits, 4 thermo bits] → 8 neurons → threshold → 8 bits
//! The network must accumulate in binary, adding thermometer-encoded digits.
//!
//! Run: cargo run --example byte_hybrid --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn relu(x: f32) -> f32 { x.max(0.0) }

fn thermo(val: usize, bits: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; bits];
    for i in 0..val.min(bits) { v[i] = 1.0; }
    v
}

fn to_bits(val: usize, bits: usize) -> Vec<f32> {
    (0..bits).map(|i| if (val >> i) & 1 == 1 { 1.0 } else { 0.0 }).collect()
}

fn from_charges(charges: &[f32], bits: usize) -> usize {
    let mut val = 0usize;
    for i in 0..bits {
        if charges[i] > 0.0 { val |= 1 << i; }
    }
    val
}

/// Hybrid recurrent: thermo input, binary state/output
fn hybrid_forward(digits: &[usize], w: &[Vec<f32>], bias: &[f32], out_bits: usize, thermo_bits: usize) -> Vec<f32> {
    let n = out_bits;
    let mut act = vec![0.0f32; n];
    for &digit in digits {
        let t = thermo(digit, thermo_bits);
        let mut input = Vec::with_capacity(n + thermo_bits);
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

fn gen_combos(n_inputs: usize, max_digit: usize) -> Vec<Vec<usize>> {
    let mut result = vec![vec![]];
    for _ in 0..n_inputs {
        let mut nr = Vec::new();
        for combo in &result { for d in 0..max_digit { let mut c = combo.clone(); c.push(d); nr.push(c); } }
        result = nr;
    }
    result
}

fn eval_hybrid(
    w: &[Vec<f32>], bias: &[f32], out_bits: usize, thermo_bits: usize,
    n_inputs: usize, max_digit: usize,
    op: &dyn Fn(&[usize]) -> usize,
) -> f64 {
    let combos = gen_combos(n_inputs, max_digit);
    let mask = (1 << out_bits) - 1;
    let mut correct = 0;
    for combo in &combos {
        let target = op(combo) & mask;
        let charges = hybrid_forward(combo, w, bias, out_bits, thermo_bits);
        let output = from_charges(&charges, out_bits);
        if output == target { correct += 1; }
    }
    correct as f64 / combos.len() as f64
}

fn main() {
    println!("=== HYBRID: thermo input → binary output ===\n");

    let digits = 5; // 0..4
    let thermo_bits = 4;
    let ternary: Vec<f32> = vec![-1.0, 0.0, 1.0];

    let ops: Vec<(&str, Box<dyn Fn(&[usize]) -> usize>)> = vec![
        ("ADD", Box::new(|d: &[usize]| d.iter().sum::<usize>())),
        ("XOR", Box::new(|d: &[usize]| d.iter().fold(0, |a, &b| a ^ b))),
        ("OR",  Box::new(|d: &[usize]| d.iter().fold(0, |a, &b| a | b))),
        ("AND", Box::new(|d: &[usize]| d.iter().fold(0xF, |a, &b| a & b))),
        ("MAX", Box::new(|d: &[usize]| *d.iter().max().unwrap())),
    ];

    // Test different output bit widths
    for &out_bits in &[1, 2, 3, 4, 5] {
        let input_dim = out_bits + thermo_bits;
        let total_params = out_bits * input_dim + out_bits;
        let total_configs = 3u64.saturating_pow(total_params as u32);
        let exhaustive = total_configs <= 100_000_000;

        println!("============================================================");
        println!("  {} output bits, {} thermo input, {} params, 3^{} = {} {}",
            out_bits, thermo_bits, total_params, total_params,
            if total_configs > 1_000_000_000 { format!("{:.1e}", total_configs as f64) } else { format!("{}", total_configs) },
            if exhaustive { "EXHAUSTIVE" } else { "SAMPLED" });
        println!("============================================================\n");

        for (op_name, op_fn) in &ops {
            let mut rng = StdRng::seed_from_u64(42);
            let mut best_acc = 0.0f64;
            let mut best_w = vec![vec![0.0f32; input_dim]; out_bits];
            let mut best_bias = vec![0.0f32; out_bits];
            let mut perfect_count = 0u64;

            let n_try = if exhaustive { total_configs } else { 5_000_000u64.min(total_configs) };

            for iter in 0..n_try {
                let config = if exhaustive { iter } else { rng.gen_range(0..total_configs) };
                let mut c = config;
                let mut w = vec![vec![0.0f32; input_dim]; out_bits];
                let mut bias = vec![0.0f32; out_bits];
                for i in 0..out_bits {
                    for j in 0..input_dim {
                        w[i][j] = ternary[(c % 3) as usize]; c /= 3;
                    }
                }
                for i in 0..out_bits {
                    bias[i] = ternary[(c % 3) as usize]; c /= 3;
                }

                let acc = eval_hybrid(&w, &bias, out_bits, thermo_bits, 2, digits, op_fn.as_ref());
                if acc >= 1.0 { perfect_count += 1; }
                if acc > best_acc { best_acc = acc; best_w = w; best_bias = bias; }
            }

            // Perturbation if not perfect
            if best_acc < 1.0 && !exhaustive {
                let mut current = best_acc;
                for _ in 0..500_000u64 {
                    let idx = rng.gen_range(0..total_params);
                    let delta: f32 = rng.gen_range(-0.5..0.5);
                    let (old, is_b, i, j) = if idx < out_bits * input_dim {
                        let i = idx / input_dim; let j = idx % input_dim;
                        let old = best_w[i][j]; best_w[i][j] += delta; (old, false, i, j)
                    } else {
                        let i = idx - out_bits * input_dim;
                        let old = best_bias[i]; best_bias[i] += delta; (old, true, i, 0)
                    };
                    let acc = eval_hybrid(&best_w, &best_bias, out_bits, thermo_bits, 2, digits, op_fn.as_ref());
                    if acc >= current { current = acc; } else {
                        if is_b { best_bias[i] = old; } else { best_w[i][j] = old; }
                    }
                    if current >= 1.0 { best_acc = 1.0; break; }
                }
                if current > best_acc { best_acc = current; }
            }

            print!("  {:<5} train={:.0}%", op_name, best_acc * 100.0);
            if best_acc >= 0.99 {
                if exhaustive { print!("  perfect={}", perfect_count); }
                // Generalization
                for &n_in in &[2, 3, 4, 5, 6] {
                    let max_combo = digits.pow(n_in as u32);
                    if max_combo <= 100_000 {
                        let acc = eval_hybrid(&best_w, &best_bias, out_bits, thermo_bits, n_in, digits, op_fn.as_ref());
                        print!("  {}in:{:.0}%", n_in, acc * 100.0);
                    }
                }
            }
            println!();

            if best_acc >= 0.99 && out_bits <= 3 {
                for (i, row) in best_w.iter().enumerate() {
                    let s: Vec<String> = row.iter().map(|v| format!("{:>2}", *v as i8)).collect();
                    println!("         n{}: [{}] bias={}", i, s.join(","), best_bias[i] as i8);
                }
            }
        }
        println!();
    }

    println!("=== DONE ===");
}
