//! Chip Composition Scaling: test composition limits.
//!
//! Experiments:
//!   1. 4-input addition: a+b+c+d via 3 chained ADD chips
//!   2. Mixed ops: ADD(MUL(a,b), c) — MUL chip → ADD chip
//!   3. 5-input addition: a+b+c+d+e via 4 chained chips
//!   4. Deeper mixed: MUL(ADD(a,b), ADD(c,d))
//!
//! All use holographic model: signed square + nearest-mean readout.
//! Every neuron connected to every other (all-to-all within chip).
//!
//! Run: cargo run --example chip_scale --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIGITS: usize = 5; // 0..4

// ============================================================
// Core: signed square + holographic forward
// ============================================================
fn signed_square(x: f32) -> f32 {
    x * x.abs()
}

fn holo_forward(input: &[f32], w: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    let n = w.len();
    let mut act = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = bias[i];
        for (j, &inp) in input.iter().enumerate() {
            if j < w[i].len() {
                sum += w[i][j] * inp;
            }
        }
        act[i] = signed_square(sum);
    }
    act
}

// ============================================================
// Nearest-mean readout
// ============================================================
struct NearestMean {
    centroids: Vec<f32>,
}

impl NearestMean {
    fn fit(examples: &[(f32, usize)], n_classes: usize) -> Self {
        let mut sums = vec![0.0f32; n_classes];
        let mut counts = vec![0usize; n_classes];
        for &(act_sum, cls) in examples {
            sums[cls] += act_sum;
            counts[cls] += 1;
        }
        let centroids: Vec<f32> = (0..n_classes)
            .map(|c| {
                if counts[c] > 0 { sums[c] / counts[c] as f32 } else { f32::NAN }
            })
            .collect();
        NearestMean { centroids }
    }

    fn predict(&self, act_sum: f32) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_nan())
            .min_by(|a, b| (a.1 - act_sum).abs().partial_cmp(&(b.1 - act_sum).abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ============================================================
// Chip: frozen holographic module
// ============================================================
#[derive(Clone)]
struct Chip {
    w: Vec<Vec<f32>>,
    bias: Vec<f32>,
    n_neurons: usize,
    input_dim: usize,
}

impl Chip {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        holo_forward(input, &self.w, &self.bias)
    }
}

// ============================================================
// Thermometer encoding
// ============================================================
fn thermo(val: usize, size: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; size];
    for i in 0..val.min(size) {
        v[i] = 1.0;
    }
    v
}

fn thermo_2(a: usize, b: usize) -> Vec<f32> {
    let mut v = thermo(a, 4);
    v.extend_from_slice(&thermo(b, 4));
    v
}

// ============================================================
// Find a chip that solves a 2-input operation via random search
// op: (a, b) -> target class
// ============================================================
fn find_chip(
    n_neurons: usize,
    input_dim: usize,
    weight_range: &[i8],
    n_samples: u64,
    seed: u64,
    examples_fn: &dyn Fn() -> Vec<(Vec<f32>, usize)>,
    n_classes: usize,
    name: &str,
) -> Chip {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut best_acc = 0.0f64;
    let mut best_w = vec![vec![0.0f32; input_dim]; n_neurons];
    let mut best_bias = vec![0.0f32; n_neurons];

    let all_examples = examples_fn();

    for _ in 0..n_samples {
        let w: Vec<Vec<f32>> = (0..n_neurons)
            .map(|_| (0..input_dim).map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32).collect())
            .collect();
        let bias: Vec<f32> = (0..n_neurons)
            .map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32)
            .collect();

        let mut readout_data = Vec::new();
        for (input, target) in &all_examples {
            let act = holo_forward(input, &w, &bias);
            let act_sum: f32 = act.iter().sum();
            readout_data.push((act_sum, *target));
        }

        let readout = NearestMean::fit(&readout_data, n_classes);
        let correct = readout_data.iter().filter(|&&(s, t)| readout.predict(s) == t).count();
        let acc = correct as f64 / all_examples.len() as f64;

        if acc > best_acc {
            best_acc = acc;
            best_w = w;
            best_bias = bias;
        }
        if best_acc >= 1.0 { break; }
    }

    println!("    {} chip: {} neurons, {:.1}% ({} samples)", name, n_neurons, best_acc * 100.0, n_samples);

    Chip { w: best_w, bias: best_bias, n_neurons, input_dim }
}

// ============================================================
// Wiring search: perturbation-based (try-keep-revert)
// Search float weights for W_compose (n_neurons × input_dim) + bias
// ============================================================
fn search_wiring(
    n_neurons: usize,
    comp_input_dim: usize,
    weight_range: &[i8],
    eval_fn: &dyn Fn(&[Vec<f32>], &[f32]) -> f64,
    seed: u64,
    label: &str,
) -> (Vec<Vec<f32>>, Vec<f32>, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let total_params = n_neurons * comp_input_dim + n_neurons;

    // Phase 1: Random integer search (coarse)
    let random_samples = 5_000_000u64;
    let mut best_acc = 0.0f64;
    let mut best_w = vec![vec![0.0f32; comp_input_dim]; n_neurons];
    let mut best_bias = vec![0.0f32; n_neurons];

    for iter in 0..random_samples {
        let w: Vec<Vec<f32>> = (0..n_neurons)
            .map(|_| (0..comp_input_dim).map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32).collect())
            .collect();
        let bias: Vec<f32> = (0..n_neurons)
            .map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32)
            .collect();

        let acc = eval_fn(&w, &bias);
        if acc > best_acc {
            best_acc = acc;
            best_w = w;
            best_bias = bias;
            if best_acc >= 1.0 {
                println!("    {} random search: 100% at iter {}", label, iter + 1);
                return (best_w, best_bias, best_acc);
            }
        }
    }
    println!("    {} random search: {:.1}% ({}M samples, {} params)", label, best_acc * 100.0, random_samples / 1_000_000, total_params);

    // Phase 2: Float perturbation refinement
    let mut w_perturb = best_w.clone();
    let mut b_perturb = best_bias.clone();
    let mut current_acc = best_acc;
    let perturb_steps = 1_000_000u64;

    for step in 0..perturb_steps {
        let param_idx = rng.gen_range(0..total_params);
        let delta: f32 = rng.gen_range(-0.5..0.5f32);

        let (old_val, is_bias, i, j) = if param_idx < n_neurons * comp_input_dim {
            let i = param_idx / comp_input_dim;
            let j = param_idx % comp_input_dim;
            let old = w_perturb[i][j];
            w_perturb[i][j] += delta;
            (old, false, i, j)
        } else {
            let i = param_idx - n_neurons * comp_input_dim;
            let old = b_perturb[i];
            b_perturb[i] += delta;
            (old, true, i, 0)
        };

        let new_acc = eval_fn(&w_perturb, &b_perturb);
        if new_acc >= current_acc {
            current_acc = new_acc;
        } else {
            if is_bias { b_perturb[i] = old_val; } else { w_perturb[i][j] = old_val; }
        }

        if current_acc >= 1.0 {
            println!("    {} perturbation: 100% at step {} (from {:.1}%)", label, step + 1, best_acc * 100.0);
            return (w_perturb, b_perturb, current_acc);
        }
    }

    println!("    {} perturbation: {:.1}% ({}M steps, from {:.1}%)", label, current_acc * 100.0, perturb_steps / 1_000_000, best_acc * 100.0);
    (w_perturb, b_perturb, current_acc)
}

// ============================================================
// Experiment 1: 4-input addition (a+b+c+d)
// Pipeline: chip1(a,b) → chip2(chip1_out, c) → chip3(chip2_out, d)
// ============================================================
fn exp_4input_add(add_chip: &Chip, weight_range: &[i8]) {
    println!("\n============================================================");
    println!("=== EXP 1: 4-INPUT ADDITION (a+b+c+d) ===");
    println!("Pipeline: chip1(a,b) → chip2(chip1_out, c) → chip3(chip2_out, d)");
    let n_b = add_chip.n_neurons; // 3

    // Stage 1: search wiring for chip2 = compose(chip1_out, c)
    let comp1_input_dim = n_b + 4; // act_chip1(3) + thermo_c(4) = 7
    let max_sum_3 = 12; // a+b+c max

    println!("\n  Stage 1: chip2 wiring (chip1→chip2 + c)");
    let (w2, b2, acc2) = search_wiring(
        n_b, comp1_input_dim, weight_range,
        &|w_c, b_c| {
            let mut examples = Vec::new();
            for a in 0..DIGITS {
                for b in 0..DIGITS {
                    for c in 0..DIGITS {
                        let target = a + b + c;
                        let act1 = add_chip.forward(&thermo_2(a, b));
                        let mut input2 = act1;
                        input2.extend_from_slice(&thermo(c, 4));
                        let act2 = holo_forward(&input2, w_c, b_c);
                        let s: f32 = act2.iter().sum();
                        examples.push((s, target));
                    }
                }
            }
            let readout = NearestMean::fit(&examples, max_sum_3 + 1);
            examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / examples.len() as f64
        },
        42, "stage1",
    );

    if acc2 < 0.95 {
        println!("  Stage 1 didn't reach high accuracy, continuing anyway...");
    }

    // Stage 2: search wiring for chip3 = compose(chip2_out, d)
    let comp2_input_dim = n_b + 4; // act_chip2(3) + thermo_d(4) = 7
    let max_sum_4 = 16; // a+b+c+d max

    println!("\n  Stage 2: chip3 wiring (chip2→chip3 + d)");
    let (w3, b3, acc3) = search_wiring(
        n_b, comp2_input_dim, weight_range,
        &|w_c, b_c| {
            let mut examples = Vec::new();
            let total = DIGITS * DIGITS * DIGITS * DIGITS;
            for a in 0..DIGITS {
                for b in 0..DIGITS {
                    for c in 0..DIGITS {
                        for d in 0..DIGITS {
                            let target = a + b + c + d;
                            // Pipeline: chip1 → chip2 → chip3
                            let act1 = add_chip.forward(&thermo_2(a, b));
                            let mut input2 = act1;
                            input2.extend_from_slice(&thermo(c, 4));
                            let act2 = holo_forward(&input2, &w2, &b2);
                            let mut input3 = act2;
                            input3.extend_from_slice(&thermo(d, 4));
                            let act3 = holo_forward(&input3, w_c, b_c);
                            let s: f32 = act3.iter().sum();
                            examples.push((s, target));
                        }
                    }
                }
            }
            let readout = NearestMean::fit(&examples, max_sum_4 + 1);
            examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / total as f64
        },
        777, "stage2",
    );

    println!("\n  RESULT EXP 1:");
    println!("    Stage 1 (a+b+c): {:.1}%", acc2 * 100.0);
    println!("    Stage 2 (a+b+c+d): {:.1}%", acc3 * 100.0);
    println!("    Total neurons: {} (3 chips × {})", n_b * 3, n_b);
    println!("    Pipeline depth: 3 stages");
}

// ============================================================
// Experiment 2: Mixed ops — ADD(MUL(a,b), c)
// Pipeline: mul_chip(a,b) → add_wiring(mul_out, c)
// ============================================================
fn exp_mixed_add_mul(weight_range: &[i8]) {
    println!("\n============================================================");
    println!("=== EXP 2: MIXED OPS — ADD(MUL(a,b), c) ===");
    println!("Pipeline: mul_chip(a,b) → wiring → add result + c");

    let n = 3;
    let max_mul = (DIGITS - 1) * (DIGITS - 1); // 4*4 = 16
    let max_result = max_mul + (DIGITS - 1); // 16 + 4 = 20
    let n_classes = max_result + 1;

    // Find MUL chip
    println!("\n  Finding MUL chip...");
    let mul_chip = find_chip(
        n, 8, weight_range, 5_000_000, 42,
        &|| {
            let mut examples = Vec::new();
            for a in 0..DIGITS {
                for b in 0..DIGITS {
                    examples.push((thermo_2(a, b), a * b));
                }
            }
            examples
        },
        max_mul + 1,
        "MUL",
    );

    // Search wiring for add stage
    let comp_input_dim = n + 4;
    println!("\n  Searching composition wiring for ADD(MUL(a,b), c)...");
    let (w_comp, b_comp, acc) = search_wiring(
        n, comp_input_dim, weight_range,
        &|w_c, b_c| {
            let mut examples = Vec::new();
            for a in 0..DIGITS {
                for b in 0..DIGITS {
                    for c in 0..DIGITS {
                        let target = a * b + c;
                        let act_mul = mul_chip.forward(&thermo_2(a, b));
                        let mut input = act_mul;
                        input.extend_from_slice(&thermo(c, 4));
                        let act = holo_forward(&input, w_c, b_c);
                        let s: f32 = act.iter().sum();
                        examples.push((s, target));
                    }
                }
            }
            let readout = NearestMean::fit(&examples, n_classes);
            examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / examples.len() as f64
        },
        314, "ADD(MUL)",
    );

    println!("\n  RESULT EXP 2:");
    println!("    ADD(MUL(a,b), c): {:.1}%", acc * 100.0);
    println!("    Classes: 0..{} ({} classes)", max_result, n_classes);
    println!("    Total neurons: {} (mul) + {} (add) = {}", n, n, n * 2);
}

// ============================================================
// Experiment 3: MUL(ADD(a,b), ADD(c,d)) — parallel + combine
// Pipeline: add_chip1(a,b) ↘
//                            → mul_wiring(add1_out, add2_out) → result
//           add_chip2(c,d) ↗
// ============================================================
fn exp_parallel_mul_of_adds(add_chip: &Chip, weight_range: &[i8]) {
    println!("\n============================================================");
    println!("=== EXP 3: MUL(ADD(a,b), ADD(c,d)) — parallel chips ===");
    println!("Pipeline: add1(a,b) + add2(c,d) → mul_wiring → result");

    let n = add_chip.n_neurons;
    let max_sum = (DIGITS - 1) * 2; // 8
    let max_result = max_sum * max_sum; // 64
    let n_classes = max_result + 1;

    // Wiring: takes [add1_out(3), add2_out(3)] = 6 inputs
    let comp_input_dim = n * 2;
    println!("  Wiring input: {} (add1_out={}, add2_out={})", comp_input_dim, n, n);
    println!("  Classes: 0..{}", max_result);

    let (_, _, acc) = search_wiring(
        n, comp_input_dim, weight_range,
        &|w_c, b_c| {
            let mut examples = Vec::new();
            for a in 0..DIGITS {
                for b in 0..DIGITS {
                    for c in 0..DIGITS {
                        for d in 0..DIGITS {
                            let target = (a + b) * (c + d);
                            let act1 = add_chip.forward(&thermo_2(a, b));
                            let act2 = add_chip.forward(&thermo_2(c, d));
                            let mut input: Vec<f32> = act1;
                            input.extend_from_slice(&act2);
                            let act = holo_forward(&input, w_c, b_c);
                            let s: f32 = act.iter().sum();
                            examples.push((s, target));
                        }
                    }
                }
            }
            let readout = NearestMean::fit(&examples, n_classes);
            examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / examples.len() as f64
        },
        555, "MUL(ADDs)",
    );

    println!("\n  RESULT EXP 3:");
    println!("    MUL(ADD(a,b), ADD(c,d)): {:.1}%", acc * 100.0);
    println!("    This tests PARALLEL composition + nonlinear combining (multiplication)");
    println!("    Total neurons: {} (add1) + {} (add2) + {} (mul) = {}", n, n, n, n * 3);
}

// ============================================================
// Experiment 4: Flat baselines for comparison
// ============================================================
fn exp_flat_baselines(weight_range: &[i8]) {
    println!("\n============================================================");
    println!("=== EXP 4: FLAT BASELINES (no chips) ===");

    let tasks: Vec<(&str, usize, usize, Box<dyn Fn(usize, usize, usize, usize) -> usize>)> = vec![
        ("a+b+c+d", 16, 17, Box::new(|a, b, c, d| a + b + c + d)),
        ("a*b+c", 12, 21, Box::new(|a, b, c, _d| a * b + c)),
        ("(a+b)*(c+d)", 12, 65, Box::new(|a, b, c, d| (a + b) * (c + d))),
    ];

    for (name, input_dim, n_classes, op) in &tasks {
        let mut rng = StdRng::seed_from_u64(42);
        let n_neurons = 6;
        let samples = 5_000_000u64;
        let mut best_acc = 0.0f64;

        for _ in 0..samples {
            let w: Vec<Vec<f32>> = (0..n_neurons)
                .map(|_| (0..*input_dim).map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32).collect())
                .collect();
            let bias: Vec<f32> = (0..n_neurons)
                .map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32)
                .collect();

            let mut examples = Vec::new();
            for a in 0..DIGITS {
                for b in 0..DIGITS {
                    for c in 0..DIGITS {
                        for d in 0..DIGITS {
                            let target = op(a, b, c, d);
                            let mut input = vec![0.0f32; *input_dim];
                            for i in 0..a { input[i] = 1.0; }
                            for i in 0..b { input[4 + i] = 1.0; }
                            if *input_dim > 8 {
                                for i in 0..c { input[8 + i] = 1.0; }
                            }
                            if *input_dim > 12 {
                                for i in 0..d { input[12 + i] = 1.0; }
                            }
                            let act = holo_forward(&input, &w, &bias);
                            let s: f32 = act.iter().sum();
                            examples.push((s, target));
                        }
                    }
                }
            }

            let readout = NearestMean::fit(&examples, *n_classes);
            let total = examples.len();
            let correct = examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count();
            let acc = correct as f64 / total as f64;

            if acc > best_acc { best_acc = acc; }
            if best_acc >= 1.0 { break; }
        }

        println!("    {} flat (6 neurons, 5M): {:.1}%", name, best_acc * 100.0);
    }
}

// ============================================================
// Main
// ============================================================
fn main() {
    println!("=== CHIP COMPOSITION SCALING EXPERIMENTS ===");
    println!("Holographic model, signed square, nearest-mean readout");
    println!("DIGITS = {} (0..4)\n", DIGITS);

    let weight_range: Vec<i8> = (-2..=2).collect();

    // Find the base ADD chip (reused across experiments)
    println!("--- Finding base ADD chip ---");
    let add_chip = find_chip(
        3, 8, &weight_range, 5_000_000, 42,
        &|| {
            let mut examples = Vec::new();
            for a in 0..DIGITS {
                for b in 0..DIGITS {
                    examples.push((thermo_2(a, b), a + b));
                }
            }
            examples
        },
        9,
        "ADD",
    );

    // Run experiments
    exp_4input_add(&add_chip, &weight_range);
    exp_mixed_add_mul(&weight_range);
    exp_parallel_mul_of_adds(&add_chip, &weight_range);
    exp_flat_baselines(&weight_range);

    // Summary
    println!("\n============================================================");
    println!("=== SCALING SUMMARY ===");
    println!("See per-experiment results above.");
    println!("Key question: does composition scale better than flat search?");
}
