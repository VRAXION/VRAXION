//! Robustness tests for recurrent ReLU chip.
//!
//! 1. Multi-seed ADD: find consistent 100% configs
//! 2. Digit range scaling: 0..4, 0..9, 0..15
//! 3. Pipeline + recurrent combo: recurrent_ADD(a,b,c) → MUL(sum, d)
//!
//! Run: cargo run --example chip_robust --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn relu(x: f32) -> f32 { x.max(0.0) }

fn thermo(val: usize, bits: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; bits];
    for i in 0..val.min(bits) { v[i] = 1.0; }
    v
}

struct NearestMean { centroids: Vec<f32> }
impl NearestMean {
    fn fit(examples: &[(f32, usize)]) -> Self {
        let n_classes = examples.iter().map(|e| e.1 + 1).max().unwrap_or(1);
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

fn recurrent_forward(digits: &[usize], w: &[Vec<f32>], bias: &[f32], thermo_bits: usize) -> Vec<f32> {
    let n = w.len();
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
        let mut new_r = Vec::new();
        for combo in &result {
            for d in 0..max_digit { let mut c = combo.clone(); c.push(d); new_r.push(c); }
        }
        result = new_r;
    }
    result
}

fn eval_add(w: &[Vec<f32>], bias: &[f32], n_inputs: usize, max_digit: usize, thermo_bits: usize) -> f64 {
    let combos = gen_combos(n_inputs, max_digit);
    let mut examples = Vec::new();
    for combo in &combos {
        let target: usize = combo.iter().sum();
        let act = recurrent_forward(combo, w, bias, thermo_bits);
        let s: f32 = act.iter().sum();
        if s.is_nan() || s.is_infinite() { return 0.0; }
        examples.push((s, target));
    }
    let readout = NearestMean::fit(&examples);
    examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / combos.len() as f64
}

/// Search recurrent ADD chip
fn search_add(n: usize, train_n: usize, max_digit: usize, thermo_bits: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<f32>, f64) {
    let input_dim = n + thermo_bits;
    let wr: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut rng = StdRng::seed_from_u64(seed);
    let total_params = n * input_dim + n;

    let mut best_acc = 0.0f64;
    let mut best_w = vec![vec![0.0f32; input_dim]; n];
    let mut best_bias = vec![0.0f32; n];

    for _ in 0..3_000_000u64 {
        let w: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| wr[rng.gen_range(0..wr.len())]).collect()).collect();
        let bias: Vec<f32> = (0..n).map(|_| wr[rng.gen_range(0..wr.len())]).collect();
        let acc = eval_add(&w, &bias, train_n, max_digit, thermo_bits);
        if acc > best_acc { best_acc = acc; best_w = w; best_bias = bias; }
        if best_acc >= 1.0 { return (best_w, best_bias, 1.0); }
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
        let acc = eval_add(&best_w, &best_bias, train_n, max_digit, thermo_bits);
        if acc >= current { current = acc; } else {
            if is_b { best_bias[i] = old; } else { best_w[i][j] = old; }
        }
        if current >= 1.0 { return (best_w, best_bias, 1.0); }
    }
    (best_w, best_bias, current)
}

fn main() {
    println!("=== RECURRENT RELU ROBUSTNESS TESTS ===\n");

    // =========================================
    // TEST 1: Multi-seed ADD (find consistent 100%)
    // =========================================
    println!("--- TEST 1: Multi-seed ADD (3 neurons, digits 0..4, train on 3-input) ---\n");
    let n = 3;
    let max_digit = 5;
    let thermo_bits = 4;

    let mut perfect_count = 0;
    let mut best_global_w = vec![];
    let mut best_global_b = vec![];
    let mut best_global_min_gen = 0.0f64;

    for seed in 0..20u64 {
        let (w, bias, train_acc) = search_add(n, 3, max_digit, thermo_bits, seed * 137 + 1);
        if train_acc < 1.0 { continue; }

        // Test generalization
        let mut min_gen = 1.0f64;
        let mut accs = Vec::new();
        for n_in in 2..=8 {
            let acc = eval_add(&w, &bias, n_in, max_digit, thermo_bits);
            accs.push(acc);
            if acc < min_gen { min_gen = acc; }
        }

        let all_perfect = accs.iter().all(|&a| a >= 1.0);
        if all_perfect { perfect_count += 1; }
        if min_gen > best_global_min_gen {
            best_global_min_gen = min_gen;
            best_global_w = w;
            best_global_b = bias;
        }

        let label = if all_perfect { "*** PERFECT ***" } else { "" };
        println!("  seed {:>3}: 2:{:.0} 3:{:.0} 4:{:.0} 5:{:.0} 6:{:.0} 7:{:.0} 8:{:.0}  {}",
            seed, accs[0]*100.0, accs[1]*100.0, accs[2]*100.0, accs[3]*100.0,
            accs[4]*100.0, accs[5]*100.0, accs[6]*100.0, label);
    }

    println!("\n  Perfect seeds (100% on all 2-8): {}/20", perfect_count);
    println!("  Best worst-case generalization: {:.1}%", best_global_min_gen * 100.0);

    // Show best weights
    if !best_global_w.is_empty() {
        println!("\n  Best W ({}×{}):", n, n + thermo_bits);
        for (i, row) in best_global_w.iter().enumerate() {
            let s: Vec<String> = row.iter().map(|v| format!("{:>6.2}", v)).collect();
            println!("    n{}: [{}] bias={:.2}", i, s.join(", "), best_global_b[i]);
        }
    }

    // =========================================
    // TEST 2: Digit range scaling
    // =========================================
    println!("\n--- TEST 2: Digit range scaling (3 neurons, train on 3-input) ---\n");

    for &(max_d, bits) in &[(5, 4), (10, 9), (16, 15)] {
        println!("  Digits 0..{} ({} thermo bits):", max_d - 1, bits);
        let (w, bias, train_acc) = search_add(n, 3, max_d, bits, 42);

        if train_acc < 1.0 {
            println!("    Train: {:.1}% (not perfect)", train_acc * 100.0);
        } else {
            print!("    ");
            for n_in in 2..=6 {
                let acc = eval_add(&w, &bias, n_in, max_d, bits);
                print!(" {}-in:{:.0}%", n_in, acc * 100.0);
            }
            println!();
        }
    }

    // =========================================
    // TEST 3: Pipeline + recurrent combo
    // recurrent_ADD(a,b,c) → MUL_chip(sum, d)
    // =========================================
    println!("\n--- TEST 3: Pipeline combo: recurrent_ADD(a,b,...) → MUL(sum, d) ---\n");

    // First, find a working recurrent ADD chip
    let (add_w, add_b, _) = search_add(n, 3, max_digit, thermo_bits, 42);

    // Find MUL chip (non-recurrent, 2-input, holographic with ReLU)
    println!("  Finding 2-input MUL chip (ReLU, per-neuron bias)...");
    let mul_input_dim = 8; // thermo(a, 4) + thermo(b, 4)
    let wr: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut rng = StdRng::seed_from_u64(42);
    let mut best_mul_acc = 0.0f64;
    let mut mul_w = vec![vec![0.0f32; mul_input_dim]; n];
    let mut mul_bias = vec![0.0f32; n];

    for _ in 0..5_000_000u64 {
        let w: Vec<Vec<f32>> = (0..n).map(|_| (0..mul_input_dim).map(|_| wr[rng.gen_range(0..wr.len())]).collect()).collect();
        let bias: Vec<f32> = (0..n).map(|_| wr[rng.gen_range(0..wr.len())]).collect();
        // Eval MUL
        let mut examples = Vec::new();
        for a in 0..max_digit {
            for b in 0..max_digit {
                let mut input = thermo(a, 4);
                input.extend_from_slice(&thermo(b, 4));
                let mut act = vec![0.0f32; n];
                for i in 0..n {
                    let mut sum = bias[i];
                    for (j, &inp) in input.iter().enumerate() {
                        if j < w[i].len() { sum += w[i][j] * inp; }
                    }
                    act[i] = relu(sum);
                }
                let s: f32 = act.iter().sum();
                examples.push((s, a * b));
            }
        }
        let readout = NearestMean::fit(&examples);
        let acc = examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / examples.len() as f64;
        if acc > best_mul_acc { best_mul_acc = acc; mul_w = w; mul_bias = bias; }
        if best_mul_acc >= 1.0 { break; }
    }
    // Perturbation
    let total_mul_params = n * mul_input_dim + n;
    for _ in 0..500_000u64 {
        let idx = rng.gen_range(0..total_mul_params);
        let delta: f32 = rng.gen_range(-0.3..0.3);
        let (old, is_b, i, j) = if idx < n * mul_input_dim {
            let i = idx / mul_input_dim; let j = idx % mul_input_dim;
            let old = mul_w[i][j]; mul_w[i][j] += delta; (old, false, i, j)
        } else {
            let i = idx - n * mul_input_dim;
            let old = mul_bias[i]; mul_bias[i] += delta; (old, true, i, 0)
        };
        let mut examples = Vec::new();
        for a in 0..max_digit { for b in 0..max_digit {
            let mut input = thermo(a, 4); input.extend_from_slice(&thermo(b, 4));
            let mut act = vec![0.0f32; n];
            for i2 in 0..n { let mut sum = mul_bias[i2]; for (j2, &inp) in input.iter().enumerate() { if j2 < mul_w[i2].len() { sum += mul_w[i2][j2] * inp; } } act[i2] = relu(sum); }
            let s: f32 = act.iter().sum();
            examples.push((s, a * b));
        }}
        let readout = NearestMean::fit(&examples);
        let acc = examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / examples.len() as f64;
        if acc >= best_mul_acc { best_mul_acc = acc; } else {
            if is_b { mul_bias[i] = old; } else { mul_w[i][j] = old; }
        }
        if best_mul_acc >= 1.0 { break; }
    }
    println!("  MUL chip: {:.1}%", best_mul_acc * 100.0);

    // Now compose: recurrent_ADD(a,b,...,n-1) → wiring → MUL(add_result, last_digit)
    // The wiring takes [add_activations(3), thermo_d(4)] → 3 neurons
    println!("  Searching wiring: ADD_output → MUL_input...");
    let wire_input_dim = n + thermo_bits; // add_act(3) + thermo_d(4) = 7
    let mut best_wire_acc = 0.0f64;
    let mut wire_w = vec![vec![0.0f32; wire_input_dim]; n];
    let mut wire_bias = vec![0.0f32; n];
    let mut rng_w = StdRng::seed_from_u64(314);

    // Eval function for pipeline: ADD(a,b,...) * d
    let eval_pipeline = |ww: &[Vec<f32>], wb: &[f32], n_add_inputs: usize| -> f64 {
        let combos_add = gen_combos(n_add_inputs, max_digit);
        let mut examples = Vec::new();
        for add_combo in &combos_add {
            for d in 0..max_digit {
                let add_sum: usize = add_combo.iter().sum();
                let target = add_sum * d;
                // Step 1: recurrent ADD
                let add_act = recurrent_forward(add_combo, &add_w, &add_b, thermo_bits);
                // Step 2: wiring to combine with d
                let mut wire_input = add_act;
                wire_input.extend_from_slice(&thermo(d, thermo_bits));
                let mut act = vec![0.0f32; n];
                for i in 0..n {
                    let mut sum = wb[i];
                    for (j, &inp) in wire_input.iter().enumerate() {
                        if j < ww[i].len() { sum += ww[i][j] * inp; }
                    }
                    act[i] = relu(sum);
                }
                let s: f32 = act.iter().sum();
                if s.is_nan() || s.is_infinite() { return 0.0; }
                examples.push((s, target));
            }
        }
        let readout = NearestMean::fit(&examples);
        examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / examples.len() as f64
    };

    // Search wiring (train with 2-input ADD, i.e., (a+b)*d)
    for _ in 0..3_000_000u64 {
        let w: Vec<Vec<f32>> = (0..n).map(|_| (0..wire_input_dim).map(|_| wr[rng_w.gen_range(0..wr.len())]).collect()).collect();
        let bias: Vec<f32> = (0..n).map(|_| wr[rng_w.gen_range(0..wr.len())]).collect();
        let acc = eval_pipeline(&w, &bias, 2);
        if acc > best_wire_acc { best_wire_acc = acc; wire_w = w; wire_bias = bias; }
        if best_wire_acc >= 1.0 { break; }
    }
    // Perturbation
    let total_wire_params = n * wire_input_dim + n;
    let mut current_w = best_wire_acc;
    for _ in 0..500_000u64 {
        let idx = rng_w.gen_range(0..total_wire_params);
        let delta: f32 = rng_w.gen_range(-0.3..0.3);
        let (old, is_b, i, j) = if idx < n * wire_input_dim {
            let i = idx / wire_input_dim; let j = idx % wire_input_dim;
            let old = wire_w[i][j]; wire_w[i][j] += delta; (old, false, i, j)
        } else {
            let i = idx - n * wire_input_dim;
            let old = wire_bias[i]; wire_bias[i] += delta; (old, true, i, 0)
        };
        let acc = eval_pipeline(&wire_w, &wire_bias, 2);
        if acc >= current_w { current_w = acc; } else {
            if is_b { wire_bias[i] = old; } else { wire_w[i][j] = old; }
        }
        if current_w >= 1.0 { break; }
    }

    println!("  Pipeline (a+b)*d wiring: {:.1}%", current_w * 100.0);

    // Test generalization: (a+b+c)*d, (a+b+c+d)*e, etc.
    println!("\n  Generalization:");
    for n_add in 2..=5 {
        let acc = eval_pipeline(&wire_w, &wire_bias, n_add);
        let parts: Vec<String> = (0..n_add).map(|i| String::from((b'a' + i as u8) as char)).collect();
        let last = (b'a' + n_add as u8) as char;
        println!("    ({})*{}: {:.1}%", parts.join("+"), last, acc * 100.0);
    }

    println!("\n=== DONE ===");
}
