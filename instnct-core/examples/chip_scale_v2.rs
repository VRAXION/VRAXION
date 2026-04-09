//! Chip Composition Scaling v2: fix MUL chip + normalized recurrent.
//!
//! Part A: MUL chip to 100% (perturbation), then ADD(MUL(a,b), c)
//! Part B: Recurrent chip with normalization between ticks
//! Part C: Comparison table
//!
//! Run: cargo run --example chip_scale_v2 --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIGITS: usize = 5; // 0..4

fn signed_square(x: f32) -> f32 {
    x * x.abs()
}

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

fn holo_forward(input: &[f32], w: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    let n = w.len();
    let mut act = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = bias[i];
        for (j, &inp) in input.iter().enumerate() {
            if j < w[i].len() { sum += w[i][j] * inp; }
        }
        act[i] = signed_square(sum);
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

// ============================================================
// Generic chip finder: random search + perturbation
// ============================================================
fn find_chip_perfected(
    n_neurons: usize,
    input_dim: usize,
    weight_range: &[i8],
    examples_fn: &dyn Fn() -> Vec<(Vec<f32>, usize)>,
    n_classes: usize,
    seed: u64,
    name: &str,
) -> (Vec<Vec<f32>>, Vec<f32>, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let all_examples = examples_fn();

    let eval = |w: &[Vec<f32>], b: &[f32]| -> f64 {
        let mut rd = Vec::new();
        for (input, target) in &all_examples {
            let act = holo_forward(input, w, b);
            let s: f32 = act.iter().sum();
            rd.push((s, *target));
        }
        let readout = NearestMean::fit(&rd, n_classes);
        rd.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / all_examples.len() as f64
    };

    // Phase 1: Random search
    let mut best_acc = 0.0f64;
    let mut best_w = vec![vec![0.0f32; input_dim]; n_neurons];
    let mut best_bias = vec![0.0f32; n_neurons];
    let random_samples = 5_000_000u64;

    for _ in 0..random_samples {
        let w: Vec<Vec<f32>> = (0..n_neurons)
            .map(|_| (0..input_dim).map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32).collect())
            .collect();
        let bias: Vec<f32> = (0..n_neurons)
            .map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32)
            .collect();
        let acc = eval(&w, &bias);
        if acc > best_acc { best_acc = acc; best_w = w; best_bias = bias; }
        if best_acc >= 1.0 { break; }
    }
    println!("    {} random: {:.1}%", name, best_acc * 100.0);

    if best_acc >= 1.0 {
        return (best_w, best_bias, best_acc);
    }

    // Phase 2: Perturbation
    let total_params = n_neurons * input_dim + n_neurons;
    let mut current = best_acc;
    for step in 0..1_000_000u64 {
        let idx = rng.gen_range(0..total_params);
        let delta: f32 = rng.gen_range(-0.5..0.5);
        let (old, is_b, i, j) = if idx < n_neurons * input_dim {
            let i = idx / input_dim; let j = idx % input_dim;
            let old = best_w[i][j]; best_w[i][j] += delta; (old, false, i, j)
        } else {
            let i = idx - n_neurons * input_dim;
            let old = best_bias[i]; best_bias[i] += delta; (old, true, i, 0)
        };
        let acc = eval(&best_w, &best_bias);
        if acc >= current { current = acc; } else {
            if is_b { best_bias[i] = old; } else { best_w[i][j] = old; }
        }
        if current >= 1.0 {
            println!("    {} perturbation: 100% at step {}", name, step + 1);
            return (best_w, best_bias, 1.0);
        }
    }
    println!("    {} perturbation: {:.1}%", name, current * 100.0);
    (best_w, best_bias, current)
}

// ============================================================
// Wiring search (same as chip_scale.rs)
// ============================================================
fn search_wiring(
    n_neurons: usize,
    comp_input_dim: usize,
    weight_range: &[i8],
    eval_fn: &dyn Fn(&[Vec<f32>], &[f32]) -> f64,
    seed: u64,
    label: &str,
) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    let total_params = n_neurons * comp_input_dim + n_neurons;

    // Random search
    let mut best_acc = 0.0f64;
    let mut best_w = vec![vec![0.0f32; comp_input_dim]; n_neurons];
    let mut best_bias = vec![0.0f32; n_neurons];

    for _ in 0..5_000_000u64 {
        let w: Vec<Vec<f32>> = (0..n_neurons)
            .map(|_| (0..comp_input_dim).map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32).collect())
            .collect();
        let bias: Vec<f32> = (0..n_neurons)
            .map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32)
            .collect();
        let acc = eval_fn(&w, &bias);
        if acc > best_acc { best_acc = acc; best_w = w; best_bias = bias; }
        if best_acc >= 1.0 {
            println!("    {} random: 100%", label);
            return 1.0;
        }
    }
    println!("    {} random: {:.1}% ({} params)", label, best_acc * 100.0, total_params);

    // Perturbation
    let mut current = best_acc;
    for step in 0..1_000_000u64 {
        let idx = rng.gen_range(0..total_params);
        let delta: f32 = rng.gen_range(-0.5..0.5);
        let (old, is_b, i, j) = if idx < n_neurons * comp_input_dim {
            let i = idx / comp_input_dim; let j = idx % comp_input_dim;
            let old = best_w[i][j]; best_w[i][j] += delta; (old, false, i, j)
        } else {
            let i = idx - n_neurons * comp_input_dim;
            let old = best_bias[i]; best_bias[i] += delta; (old, true, i, 0)
        };
        let acc = eval_fn(&best_w, &best_bias);
        if acc >= current { current = acc; } else {
            if is_b { best_bias[i] = old; } else { best_w[i][j] = old; }
        }
        if current >= 1.0 {
            println!("    {} perturbation: 100% at step {}", label, step + 1);
            return 1.0;
        }
    }
    println!("    {} perturbation: {:.1}%", label, current * 100.0);
    current
}

// ============================================================
// PART A: Perfect MUL chip → ADD(MUL(a,b), c)
// ============================================================
fn part_a_mul_compose(weight_range: &[i8]) {
    println!("============================================================");
    println!("PART A: Perfect MUL chip → ADD(MUL(a,b), c)");
    println!("============================================================\n");

    let n = 3;
    let max_mul = (DIGITS - 1) * (DIGITS - 1); // 16
    let max_result = max_mul + (DIGITS - 1); // 20

    // Find perfect MUL chip
    let (mul_w, mul_b, mul_acc) = find_chip_perfected(
        n, 8, weight_range,
        &|| {
            (0..DIGITS).flat_map(|a| (0..DIGITS).map(move |b| (thermo_2(a, b), a * b))).collect()
        },
        max_mul + 1, 42, "MUL",
    );

    // Find perfect ADD chip (for comparison)
    let (add_w, add_b, add_acc) = find_chip_perfected(
        n, 8, weight_range,
        &|| {
            (0..DIGITS).flat_map(|a| (0..DIGITS).map(move |b| (thermo_2(a, b), a + b))).collect()
        },
        9, 42, "ADD",
    );

    println!("\n  MUL chip: {:.1}%, ADD chip: {:.1}%\n", mul_acc * 100.0, add_acc * 100.0);

    if mul_acc < 1.0 {
        println!("  MUL chip not perfect — composition will be limited.\n");
    }

    // Compose: ADD(MUL(a,b), c)
    let comp_input_dim = n + 4;
    println!("  Searching ADD(MUL(a,b), c) wiring...");
    let acc = search_wiring(n, comp_input_dim, weight_range,
        &|w_c, b_c| {
            let mut examples = Vec::new();
            for a in 0..DIGITS { for b in 0..DIGITS { for c in 0..DIGITS {
                let target = a * b + c;
                let act_mul = holo_forward(&thermo_2(a, b), &mul_w, &mul_b);
                let mut input = act_mul;
                input.extend_from_slice(&thermo(c, 4));
                let act = holo_forward(&input, w_c, b_c);
                let s: f32 = act.iter().sum();
                examples.push((s, target));
            }}}
            let readout = NearestMean::fit(&examples, max_result + 1);
            examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / 125.0
        },
        314, "ADD(MUL)",
    );

    println!("\n  RESULT: ADD(MUL(a,b), c) = {:.1}%", acc * 100.0);

    // Also try: MUL(a,b) + ADD(a,b) = mixed pipeline
    println!("\n  Searching SUB(MUL(a,b), ADD(a,b)) = a*b - (a+b) wiring...");
    // This tests: can we combine TWO different chip outputs?
    let comp2_input_dim = n * 2; // mul_out(3) + add_out(3) = 6
    // a*b - (a+b) ranges from 0*0-(0+0)=0 to 4*4-(4+4)=8, but can be negative
    // Actually: min = 0*1-(0+1) = -1, max = 4*4-(4+4) = 8
    // Let's offset: target = a*b - (a+b) + 8 to make it non-negative
    let offset = 8;
    let min_val: i32 = (0..DIGITS as i32).flat_map(|a| (0..DIGITS as i32).map(move |b| a*b - (a+b))).min().unwrap();
    let max_val: i32 = (0..DIGITS as i32).flat_map(|a| (0..DIGITS as i32).map(move |b| a*b - (a+b))).max().unwrap();
    let n_classes_sub = (max_val - min_val + 1) as usize;
    let sub_offset = (-min_val) as usize;

    println!("    Range: {}..{}, {} classes (offset by {})", min_val, max_val, n_classes_sub, sub_offset);

    let acc_sub = search_wiring(n, comp2_input_dim, weight_range,
        &|w_c, b_c| {
            let mut examples = Vec::new();
            for a in 0..DIGITS { for b in 0..DIGITS {
                let target = ((a * b) as i32 - (a + b) as i32 + sub_offset as i32) as usize;
                let act_mul = holo_forward(&thermo_2(a, b), &mul_w, &mul_b);
                let act_add = holo_forward(&thermo_2(a, b), &add_w, &add_b);
                let mut input = act_mul;
                input.extend_from_slice(&act_add);
                let act = holo_forward(&input, w_c, b_c);
                let s: f32 = act.iter().sum();
                examples.push((s, target));
            }}
            let readout = NearestMean::fit(&examples, n_classes_sub);
            examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / 25.0
        },
        555, "MUL-ADD",
    );

    println!("\n  RESULT: a*b-(a+b) from parallel MUL+ADD = {:.1}%", acc_sub * 100.0);
}

// ============================================================
// PART B: Normalized recurrent chip
// ============================================================
fn part_b_normalized_recurrent(weight_range: &[i8]) {
    println!("\n============================================================");
    println!("PART B: Normalized recurrent chip");
    println!("============================================================");
    println!("  Same W, normalize activations between ticks to prevent explosion.\n");

    let n = 3;
    let input_dim = n + 4;

    // Normalized recurrent forward
    let recurrent_norm = |digits: &[usize], w: &[Vec<f32>], bias: &[f32]| -> Vec<f32> {
        let mut act = vec![0.0f32; n];
        for &digit in digits {
            let t = thermo(digit, 4);
            let mut input = Vec::with_capacity(n + 4);
            input.extend_from_slice(&act);
            input.extend_from_slice(&t);

            for i in 0..n {
                let mut sum = bias[i];
                for (j, &inp) in input.iter().enumerate() {
                    if j < w[i].len() { sum += w[i][j] * inp; }
                }
                act[i] = signed_square(sum);
            }

            // NORMALIZE: scale activations to prevent explosion
            let max_abs = act.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            if max_abs > 1.0 {
                for v in &mut act { *v /= max_abs; }
            }
        }
        act
    };

    let gen_combos = |n_inputs: usize| -> Vec<Vec<usize>> {
        if n_inputs == 0 { return vec![vec![]]; }
        let mut result = vec![vec![]];
        for _ in 0..n_inputs {
            let mut new = Vec::new();
            for combo in &result {
                for d in 0..DIGITS {
                    let mut c = combo.clone();
                    c.push(d);
                    new.push(c);
                }
            }
            result = new;
        }
        result
    };

    let eval_norm = |w: &[Vec<f32>], bias: &[f32], n_inputs: usize| -> f64 {
        let max_sum = (DIGITS - 1) * n_inputs;
        let combos = gen_combos(n_inputs);
        let mut examples = Vec::new();
        for combo in &combos {
            let target: usize = combo.iter().sum();
            let act = recurrent_norm(combo, w, bias);
            let s: f32 = act.iter().sum();
            examples.push((s, target));
        }
        let readout = NearestMean::fit(&examples, max_sum + 1);
        examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / combos.len() as f64
    };

    // Try multiple normalization strategies
    for &train_n in &[2, 3, 4] {
        println!("  Training on {}-input addition (normalized recurrent):", train_n);

        let mut rng = StdRng::seed_from_u64(42 + train_n as u64);
        let mut best_acc = 0.0f64;
        let mut best_w = vec![vec![0.0f32; input_dim]; n];
        let mut best_bias = vec![0.0f32; n];

        // Random search
        for _ in 0..3_000_000u64 {
            let w: Vec<Vec<f32>> = (0..n)
                .map(|_| (0..input_dim).map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32).collect())
                .collect();
            let bias: Vec<f32> = (0..n)
                .map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32)
                .collect();
            let acc = eval_norm(&w, &bias, train_n);
            if acc > best_acc { best_acc = acc; best_w = w; best_bias = bias; }
            if best_acc >= 1.0 { break; }
        }
        println!("    random: {:.1}%", best_acc * 100.0);

        // Perturbation
        let total_params = n * input_dim + n;
        let mut current = best_acc;
        for step in 0..500_000u64 {
            let idx = rng.gen_range(0..total_params);
            let delta: f32 = rng.gen_range(-0.5..0.5);
            let (old, is_b, i, j) = if idx < n * input_dim {
                let i = idx / input_dim; let j = idx % input_dim;
                let old = best_w[i][j]; best_w[i][j] += delta; (old, false, i, j)
            } else {
                let i = idx - n * input_dim;
                let old = best_bias[i]; best_bias[i] += delta; (old, true, i, 0)
            };
            let acc = eval_norm(&best_w, &best_bias, train_n);
            if acc >= current { current = acc; } else {
                if is_b { best_bias[i] = old; } else { best_w[i][j] = old; }
            }
            if current >= 1.0 { println!("    perturbation: 100% at step {}", step + 1); break; }
        }
        if current < 1.0 { println!("    perturbation: {:.1}%", current * 100.0); }

        // Generalization
        println!("    Generalization:");
        for test_n in 2..=7 {
            let acc = eval_norm(&best_w, &best_bias, test_n);
            let mark = if test_n == train_n { " (trained)" } else { "" };
            println!("      {}-input: {:.1}%{}", test_n, acc * 100.0, mark);
        }
        println!();
    }
}

// ============================================================
// Main
// ============================================================
fn main() {
    println!("=== CHIP COMPOSITION SCALING v2 ===\n");
    let weight_range: Vec<i8> = (-2..=2).collect();

    part_a_mul_compose(&weight_range);
    part_b_normalized_recurrent(&weight_range);

    println!("============================================================");
    println!("=== DONE ===");
}
