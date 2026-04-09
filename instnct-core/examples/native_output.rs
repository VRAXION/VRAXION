//! Native output test: charge = answer directly? No readout needed?
//!
//! For each op, find chip where charge literally equals the answer.
//! Compare: nearest-mean readout vs raw round(charge).
//!
//! Run: cargo run --example native_output --release

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

fn gen_combos(n_inputs: usize) -> Vec<Vec<usize>> {
    let mut result = vec![vec![]];
    for _ in 0..n_inputs {
        let mut nr = Vec::new();
        for combo in &result { for d in 0..DIGITS { let mut c = combo.clone(); c.push(d); nr.push(c); } }
        result = nr;
    }
    result
}

fn main() {
    println!("=== NATIVE OUTPUT TEST ===");
    println!("Does charge = answer directly? No readout needed?\n");

    // =====================================================
    // TEST 1: Handcrafted perfect ADD chip
    // =====================================================
    println!("--- TEST 1: Handcrafted ADD chip (1 neuron) ---\n");
    let w_add = vec![vec![1.0, 1.0, 1.0, 1.0, 1.0]]; // [recur, d0, d1, d2, d3]
    let b_add = vec![0.0];

    println!("  W = [1, 1, 1, 1, 1], bias = 0\n");
    println!("  {:>12} {:>8} {:>8} {:>8}", "input", "charge", "round", "target");
    println!("  {}", "-".repeat(42));

    let mut native_correct = 0;
    let mut total = 0;
    for combo in &gen_combos(3) {
        let target: usize = combo.iter().sum();
        let act = recurrent_forward(combo, &w_add, &b_add);
        let charge = act[0];
        let native = charge.round() as usize;
        let ok = native == target;
        if !ok || total < 10 {
            println!("  {:>12?} {:>8.2} {:>8} {:>8} {}", combo, charge, native, target, if ok { "✓" } else { "✗" });
        }
        if ok { native_correct += 1; }
        total += 1;
    }
    println!("  ...");
    println!("  Native accuracy (3-input): {}/{} = {:.1}%\n", native_correct, total, native_correct as f64 / total as f64 * 100.0);

    // Test generalization
    println!("  Generalization (charge = round(charge) == sum?):");
    for n_in in 2..=10 {
        let combos = gen_combos(n_in);
        let mut ok = 0;
        for combo in &combos {
            let target: usize = combo.iter().sum();
            let act = recurrent_forward(combo, &w_add, &b_add);
            if act[0].round() as usize == target { ok += 1; }
        }
        println!("    {}-input: {}/{} = {:.1}%", n_in, ok, combos.len(), ok as f64 / combos.len() as f64 * 100.0);
        if (ok as f64 / combos.len() as f64) < 0.99 { break; }
    }

    // =====================================================
    // TEST 2: Search for native chips (charge = answer)
    // =====================================================
    println!("\n--- TEST 2: Search native chips (1 neuron, charge = answer) ---\n");

    let ops: Vec<(&str, Box<dyn Fn(&[usize]) -> usize>)> = vec![
        ("ADD",   Box::new(|d: &[usize]| d.iter().sum())),
        ("MAX",   Box::new(|d: &[usize]| *d.iter().max().unwrap())),
        ("MIN",   Box::new(|d: &[usize]| *d.iter().min().unwrap())),
        ("COUNT", Box::new(|d: &[usize]| d.iter().filter(|&&x| x > 0).count())),
        ("OR",    Box::new(|d: &[usize]| if d.iter().any(|&x| x > 0) { 1 } else { 0 })),
        ("XOR",   Box::new(|d: &[usize]| d.iter().fold(0, |a, &b| a ^ b))),
    ];

    // Exhaustive search for 1-neuron (5 inputs = recur + 4 thermo)
    let ternary: Vec<f32> = vec![-1.0, 0.0, 1.0];
    let input_dim = 5;
    let total_configs = 3u64.pow(6); // 5 weights + 1 bias

    for (op_name, op_fn) in &ops {
        let mut best_native_acc = 0.0f64;
        let mut best_w = vec![vec![0.0f32; input_dim]];
        let mut best_bias = vec![0.0f32];

        for config in 0..total_configs {
            let mut c = config;
            let mut w = vec![vec![0.0f32; input_dim]];
            let mut bias = vec![0.0f32; 1];
            for j in 0..input_dim {
                w[0][j] = ternary[(c % 3) as usize]; c /= 3;
            }
            bias[0] = ternary[(c % 3) as usize];

            // Native eval: round(charge) == target?
            let mut ok = 0;
            let combos = gen_combos(2);
            for combo in &combos {
                let target = op_fn(combo);
                let act = recurrent_forward(combo, &w, &bias);
                if act[0].round() as i32 == target as i32 { ok += 1; }
            }
            let acc = ok as f64 / 25.0;
            if acc > best_native_acc {
                best_native_acc = acc;
                best_w = w; best_bias = bias;
            }
        }

        print!("  {:<7} native_2in={:.0}%", op_name, best_native_acc * 100.0);

        if best_native_acc >= 0.99 {
            // Test generalization
            for &n_in in &[3, 4, 5, 6, 8] {
                let combos = gen_combos(n_in);
                let mut ok = 0;
                for combo in &combos {
                    let target = op_fn(combo);
                    let act = recurrent_forward(combo, &best_w, &best_bias);
                    if act[0].round() as i32 == target as i32 { ok += 1; }
                }
                print!("  {}in:{:.0}%", n_in, ok as f64 / combos.len() as f64 * 100.0);
            }
            let ws: Vec<String> = best_w[0].iter().map(|v| format!("{}", *v as i8)).collect();
            print!("  W=[{}] b={}", ws.join(","), best_bias[0] as i8);
        }
        println!();
    }

    // =====================================================
    // TEST 3: Native vs nearest-mean comparison
    // =====================================================
    println!("\n--- TEST 3: Native round() vs nearest-mean comparison ---\n");

    // Use handcrafted ADD chip
    println!("  ADD chip W=[1,1,1,1,1] bias=0:");
    for n_in in 2..=8 {
        let combos = gen_combos(n_in);

        // Native: round(charge)
        let mut native_ok = 0;

        // Nearest-mean
        let mut examples = Vec::new();
        for combo in &combos {
            let target: usize = combo.iter().sum();
            let act = recurrent_forward(combo, &w_add, &b_add);
            let charge = act[0];
            if charge.round() as usize == target { native_ok += 1; }
            examples.push((charge, target));
        }

        // Build centroids
        let max_sum = (DIGITS - 1) * n_in;
        let mut sums = vec![0.0f32; max_sum + 1];
        let mut counts = vec![0usize; max_sum + 1];
        for &(c, t) in &examples { sums[t] += c; counts[t] += 1; }
        let centroids: Vec<f32> = (0..=max_sum).map(|c| if counts[c] > 0 { sums[c] / counts[c] as f32 } else { f32::NAN }).collect();

        let mut nm_ok = 0;
        for &(charge, target) in &examples {
            let pred = centroids.iter().enumerate().filter(|(_, c)| !c.is_nan())
                .min_by(|a, b| (a.1 - charge).abs().partial_cmp(&(b.1 - charge).abs()).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if pred == target { nm_ok += 1; }
        }

        println!("    {}-input: native={:.0}%  nearest_mean={:.0}%  ({}={})",
            n_in,
            native_ok as f64 / combos.len() as f64 * 100.0,
            nm_ok as f64 / combos.len() as f64 * 100.0,
            if native_ok == nm_ok { "SAME" } else { "DIFF" },
            if native_ok >= nm_ok { "native≥nm" } else { "nm>native" },
        );
    }

    println!("\n=== DONE ===");
}
