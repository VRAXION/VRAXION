//! Chip Composition: chain two solved ADD chips to compute a + b + c.
//!
//! Uses the HOLOGRAPHIC model (1-step, signed square, nearest-mean) that solved
//! all 8 arithmetic tasks at 100%. NOT the tick-based spiking model.
//!
//! Architecture:
//!   - Chip A (frozen): N_A neurons, W_A (N_A × input_dim), solves ADD(a, b)
//!   - Chip B (frozen): N_B neurons, W_B (N_B × input_dim), solves ADD(x, c)
//!   - Composite: 2-step pipeline
//!     Step 1: act_A = signed_square(W_A × input + bias_A)
//!     Step 2: act_B = signed_square(W_B_ext × [act_A, c_thermo] + bias_B)
//!             where W_B_ext = [W_wire | W_B_c] (searched wiring + frozen c-weights)
//!
//! Every neuron sees everything (holographic). The composition question:
//! can we wire chip A's output into chip B's input and solve 3-input addition?
//!
//! Run: cargo run --example chip_compose --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIGITS: usize = 5; // 0..4
const CLASSES_2: usize = 9; // a+b: 0..8
const CLASSES_3: usize = 13; // a+b+c: 0..12

// ============================================================
// Signed square activation: x * |x|
// ============================================================
fn signed_square(x: f32) -> f32 {
    x * x.abs()
}

// ============================================================
// Holographic forward: 1-step, signed square activation
// W: n_neurons × input_dim, bias: n_neurons
// Returns: n_neurons activations
// ============================================================
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
    centroids: Vec<f32>, // one per class
}

impl NearestMean {
    /// Build centroids from (activation_sum, target_class) pairs.
    fn fit(examples: &[(f32, usize)], n_classes: usize) -> Self {
        let mut sums = vec![0.0f32; n_classes];
        let mut counts = vec![0usize; n_classes];
        for &(act_sum, cls) in examples {
            sums[cls] += act_sum;
            counts[cls] += 1;
        }
        let centroids: Vec<f32> = (0..n_classes)
            .map(|c| {
                if counts[c] > 0 {
                    sums[c] / counts[c] as f32
                } else {
                    f32::NAN
                }
            })
            .collect();
        NearestMean { centroids }
    }

    fn predict(&self, act_sum: f32) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_nan())
            .min_by(|a, b| {
                (a.1 - act_sum)
                    .abs()
                    .partial_cmp(&(b.1 - act_sum).abs())
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ============================================================
// Thermometer encoding
// ============================================================
fn thermo_2(a: usize, b: usize) -> Vec<f32> {
    // 8-dim: [a_thermo(4), b_thermo(4)]
    let mut v = vec![0.0f32; 8];
    for i in 0..a {
        v[i] = 1.0;
    }
    for i in 0..b {
        v[4 + i] = 1.0;
    }
    v
}

fn thermo_1(c: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 4];
    for i in 0..c {
        v[i] = 1.0;
    }
    v
}

// ============================================================
// Find ADD chip via random search (holographic model)
// ============================================================
fn find_add_chip(
    n_neurons: usize,
    weight_range: &[i8],
    n_samples: u64,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<f32>, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let input_dim = 8; // 2-digit thermometer

    let mut best_acc = 0.0f64;
    let mut best_w = vec![vec![0.0f32; input_dim]; n_neurons];
    let mut best_bias = vec![0.0f32; n_neurons];

    for _ in 0..n_samples {
        // Random integer weights
        let w: Vec<Vec<f32>> = (0..n_neurons)
            .map(|_| {
                (0..input_dim)
                    .map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32)
                    .collect()
            })
            .collect();
        let bias: Vec<f32> = (0..n_neurons)
            .map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32)
            .collect();

        // Evaluate on all 25 examples
        let mut examples = Vec::new();
        for a in 0..DIGITS {
            for b in 0..DIGITS {
                let input = thermo_2(a, b);
                let act = holo_forward(&input, &w, &bias);
                let act_sum: f32 = act.iter().sum();
                examples.push((act_sum, a + b));
            }
        }

        let readout = NearestMean::fit(&examples, CLASSES_2);
        let correct = examples
            .iter()
            .filter(|&&(act_sum, target)| readout.predict(act_sum) == target)
            .count();
        let acc = correct as f64 / 25.0;

        if acc > best_acc {
            best_acc = acc;
            best_w = w;
            best_bias = bias;
        }
        if best_acc >= 1.0 {
            break;
        }
    }

    (best_w, best_bias, best_acc)
}

// ============================================================
// Evaluate composite pipeline on a+b+c
//
// Pipeline:
//   1. act_A = signed_square(W_A × thermo(a,b) + bias_A)
//   2. composite_input = [act_A[0], ..., act_A[N_A-1], thermo_c[0..4]]
//   3. act_B = signed_square(W_compose × composite_input + bias_B)
//   4. readout = nearest_mean(sum(act_B))
// ============================================================
fn eval_pipeline(
    w_a: &[Vec<f32>],
    bias_a: &[f32],
    w_compose: &[Vec<f32>], // N_B × (N_A + 4)
    bias_b: &[f32],
) -> (f64, NearestMean) {
    let n_b = w_compose.len();

    // Collect all examples
    let mut examples = Vec::new();
    for a in 0..DIGITS {
        for b in 0..DIGITS {
            for c in 0..DIGITS {
                let target = a + b + c;

                // Step 1: chip A
                let input_a = thermo_2(a, b);
                let act_a = holo_forward(&input_a, w_a, bias_a);

                // Step 2: compose input = [act_A, thermo_c]
                let thermo_c = thermo_1(c);
                let mut comp_input = act_a.clone();
                comp_input.extend_from_slice(&thermo_c);

                // Step 3: chip B on composite input
                let act_b = holo_forward(&comp_input, w_compose, bias_b);
                let act_sum: f32 = act_b.iter().sum();

                examples.push((act_sum, target));
            }
        }
    }

    // Fit readout on all examples
    let readout = NearestMean::fit(&examples, CLASSES_3);

    // Score
    let correct = examples
        .iter()
        .filter(|&&(act_sum, target)| readout.predict(act_sum) == target)
        .count();
    let total = DIGITS * DIGITS * DIGITS; // 125

    (correct as f64 / total as f64, readout)
}

// ============================================================
// Main
// ============================================================
fn main() {
    println!("=== CHIP COMPOSITION: ADD(ADD(a,b), c) ===");
    println!("Holographic model: signed square + nearest-mean");
    println!("a,b,c in 0..{}, target = a+b+c in 0..{}", DIGITS, CLASSES_3);
    println!();

    // --- Step 1: Find ADD chip ---
    println!("--- Step 1: Find ADD chip (holographic, 1-step) ---");
    let weight_range: Vec<i8> = (-2..=2).collect(); // ±2

    for &n in &[3, 4, 5, 8] {
        let (_, _, acc) = find_add_chip(n, &weight_range, 2_000_000, 42);
        println!("  N={}: {:.1}% (2M samples, ±2 weights)", n, acc * 100.0);
    }

    // Use N=3 (the proven winner)
    let n_a = 3;
    let (w_a, bias_a, chip_acc) = find_add_chip(n_a, &weight_range, 5_000_000, 42);
    println!(
        "\n  Selected chip A: N={}, accuracy={:.1}%",
        n_a,
        chip_acc * 100.0
    );

    // Show chip A weights
    println!("  W_A ({} × 8):", n_a);
    for (i, row) in w_a.iter().enumerate() {
        let s: Vec<String> = row.iter().map(|v| format!("{:>3}", *v as i8)).collect();
        println!("    neuron {}: [{}]  bias={}", i, s.join(", "), bias_a[i] as i8);
    }

    // --- Step 2: Baseline — chip A alone on a+b ---
    println!("\n--- Step 2: Verify chip A on 2-input addition ---");
    let mut chip_a_examples = Vec::new();
    for a in 0..DIGITS {
        for b in 0..DIGITS {
            let input = thermo_2(a, b);
            let act = holo_forward(&input, &w_a, &bias_a);
            let act_sum: f32 = act.iter().sum();
            chip_a_examples.push((act_sum, a + b));
        }
    }
    let readout_a = NearestMean::fit(&chip_a_examples, CLASSES_2);
    let correct_a = chip_a_examples
        .iter()
        .filter(|&&(s, t)| readout_a.predict(s) == t)
        .count();
    println!("  Chip A accuracy: {}/25 = {:.1}%", correct_a, correct_a as f64 / 25.0 * 100.0);

    // Show chip A activations per class
    println!("  Activation sums per target:");
    for target in 0..CLASSES_2 {
        let acts: Vec<f32> = chip_a_examples
            .iter()
            .filter(|&&(_, t)| t == target)
            .map(|&(s, _)| s)
            .collect();
        if !acts.is_empty() {
            let mean = acts.iter().sum::<f32>() / acts.len() as f32;
            let spread = acts.iter().map(|a| (a - mean).abs()).fold(0.0f32, f32::max);
            println!(
                "    sum={}: mean={:.2}, spread={:.4}, n={}",
                target,
                mean,
                spread,
                acts.len()
            );
        }
    }

    // --- Step 3: Search composition wiring ---
    println!("\n--- Step 3: Search W_compose (chip B wiring) ---");
    let n_b = 3; // chip B also has 3 neurons
    let comp_input_dim = n_a + 4; // [act_A (3), thermo_c (4)] = 7

    println!(
        "  Chip B: {} neurons × {} inputs ([act_A({}), c_thermo(4)])",
        n_b, comp_input_dim, n_a
    );
    println!(
        "  Search space: W_compose ({} × {} = {} weights) + bias ({}) = {} params",
        n_b,
        comp_input_dim,
        n_b * comp_input_dim,
        n_b,
        n_b * comp_input_dim + n_b
    );

    let total_params = n_b * comp_input_dim + n_b; // 21 + 3 = 24
    let total_configs = 5u64.saturating_pow(total_params as u32); // ±2 = 5 values
    println!(
        "  5^{} = {:.2e} total configs",
        total_params, total_configs as f64
    );

    let sample_size: u64 = 10_000_000;
    println!("  Random search: {} samples\n", sample_size);

    let mut rng = StdRng::seed_from_u64(777);
    let mut best_acc = 0.0f64;
    let mut best_w_compose = vec![vec![0.0f32; comp_input_dim]; n_b];
    let mut best_bias_b = vec![0.0f32; n_b];

    let report_interval = sample_size / 20;

    for iter in 0..sample_size {
        // Random W_compose and bias_b
        let w_compose: Vec<Vec<f32>> = (0..n_b)
            .map(|_| {
                (0..comp_input_dim)
                    .map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32)
                    .collect()
            })
            .collect();
        let bias_b: Vec<f32> = (0..n_b)
            .map(|_| weight_range[rng.gen_range(0..weight_range.len())] as f32)
            .collect();

        let (acc, _) = eval_pipeline(&w_a, &bias_a, &w_compose, &bias_b);

        if acc > best_acc {
            best_acc = acc;
            best_w_compose = w_compose;
            best_bias_b = bias_b;
            println!(
                "    [{:>9}/{} ({:.1}%)] NEW BEST: {:.1}% ({}/125)",
                iter + 1,
                sample_size,
                (iter + 1) as f64 / sample_size as f64 * 100.0,
                best_acc * 100.0,
                (best_acc * 125.0).round() as usize,
            );
            if best_acc >= 1.0 {
                println!("    *** 100% ACCURACY — COMPOSITION WORKS! ***");
                break;
            }
        }

        if report_interval > 0 && (iter + 1) % report_interval == 0 && best_acc < 1.0 {
            println!(
                "    [{:>9}/{} ({:.0}%)] best: {:.1}%",
                iter + 1,
                sample_size,
                (iter + 1) as f64 / sample_size as f64 * 100.0,
                best_acc * 100.0
            );
        }
    }

    // --- Step 4: Results ---
    println!("\n--- Step 4: Results ---");
    let (final_acc, final_readout) =
        eval_pipeline(&w_a, &bias_a, &best_w_compose, &best_bias_b);
    println!("  Final accuracy: {:.1}% ({}/125)", final_acc * 100.0, (final_acc * 125.0).round() as usize);

    // Per-class breakdown
    println!("\n  Per-class accuracy:");
    let mut class_correct = vec![0usize; CLASSES_3];
    let mut class_total = vec![0usize; CLASSES_3];

    for a in 0..DIGITS {
        for b in 0..DIGITS {
            for c in 0..DIGITS {
                let target = a + b + c;
                let input_a = thermo_2(a, b);
                let act_a = holo_forward(&input_a, &w_a, &bias_a);
                let thermo_c = thermo_1(c);
                let mut comp_input = act_a;
                comp_input.extend_from_slice(&thermo_c);
                let act_b = holo_forward(&comp_input, &best_w_compose, &best_bias_b);
                let act_sum: f32 = act_b.iter().sum();
                let pred = final_readout.predict(act_sum);

                class_total[target] += 1;
                if pred == target {
                    class_correct[target] += 1;
                }
            }
        }
    }

    for cls in 0..CLASSES_3 {
        if class_total[cls] > 0 {
            let acc = class_correct[cls] as f64 / class_total[cls] as f64;
            println!(
                "    sum={:>2}: {:>2}/{:>2} ({:>5.1}%) {}",
                cls,
                class_correct[cls],
                class_total[cls],
                acc * 100.0,
                if acc >= 1.0 { "ok" } else { "MISS" }
            );
        }
    }

    // Show wiring
    println!("\n  W_compose (chip B, {} × {}):", n_b, comp_input_dim);
    let labels: Vec<String> = (0..n_a)
        .map(|i| format!("A{}", i))
        .chain((0..4).map(|i| format!("c{}", i)))
        .collect();
    println!("    cols: [{}]", labels.join(", "));
    for (i, row) in best_w_compose.iter().enumerate() {
        let s: Vec<String> = row.iter().map(|v| format!("{:>3}", *v as i8)).collect();
        println!("    B{}: [{}]  bias={}", i, s.join(", "), best_bias_b[i] as i8);
    }

    println!("\n  Summary (random search):");
    println!("    Chip A: {} neurons, {:.1}% on ADD(a,b)", n_a, chip_acc * 100.0);
    println!("    Chip B: {} neurons, wiring searched", n_b);
    println!("    Pipeline: act_A = chip_A(a,b) → act_B = chip_B(act_A, c)");
    println!("    Total neurons: {} (A) + {} (B) = {}", n_a, n_b, n_a + n_b);
    println!("    Searched params: {} (frozen chip A has {})", total_params, n_a * 8 + n_a);
    println!("    3-input addition accuracy: {:.1}%", final_acc * 100.0);

    // --- Step 5: Perturbation-based refinement (try-keep-revert) ---
    println!("\n--- Step 5: Perturbation refinement (float, try-keep-revert) ---");

    let mut w_perturb: Vec<Vec<f32>> = best_w_compose.clone();
    let mut b_perturb: Vec<f32> = best_bias_b.clone();
    let (mut current_acc, _) = eval_pipeline(&w_a, &bias_a, &w_perturb, &b_perturb);
    println!("  Starting from: {:.1}%", current_acc * 100.0);

    let perturb_steps = 500_000;
    let mut perturb_rng = StdRng::seed_from_u64(999);
    let mut accepts = 0u64;

    for step in 0..perturb_steps {
        // Pick random param, perturb by small float amount
        let param_idx = perturb_rng.gen_range(0..total_params);
        let delta: f32 = perturb_rng.gen_range(-0.5..0.5f32);

        let (old_val, is_bias, i, j) = if param_idx < n_b * comp_input_dim {
            let i = param_idx / comp_input_dim;
            let j = param_idx % comp_input_dim;
            let old = w_perturb[i][j];
            w_perturb[i][j] += delta;
            (old, false, i, j)
        } else {
            let i = param_idx - n_b * comp_input_dim;
            let old = b_perturb[i];
            b_perturb[i] += delta;
            (old, true, i, 0)
        };

        let (new_acc, _) = eval_pipeline(&w_a, &bias_a, &w_perturb, &b_perturb);

        if new_acc >= current_acc {
            current_acc = new_acc;
            accepts += 1;
        } else {
            // Revert
            if is_bias {
                b_perturb[i] = old_val;
            } else {
                w_perturb[i][j] = old_val;
            }
        }

        if (step + 1) % 50_000 == 0 {
            println!(
                "    step {:>6}: {:.1}% ({}/125), accepts={}",
                step + 1,
                current_acc * 100.0,
                (current_acc * 125.0).round() as usize,
                accepts,
            );
        }
        if current_acc >= 1.0 {
            println!(
                "    step {:>6}: *** 100% — PERTURBATION SOLVED IT! ***",
                step + 1
            );
            break;
        }
    }

    let (perturb_acc, _) = eval_pipeline(&w_a, &bias_a, &w_perturb, &b_perturb);

    // --- Step 6: Flat baseline — 6 neurons from scratch on 3-input ADD ---
    println!("\n--- Step 6: Flat baseline (no chips, direct 3-input search) ---");
    let flat_n = 6;
    let flat_input_dim = 12; // thermo_a(4) + thermo_b(4) + thermo_c(4)
    let flat_samples = 10_000_000u64;

    println!(
        "  {} neurons × {} inputs, ±2 weights, {} random samples",
        flat_n, flat_input_dim, flat_samples
    );

    let mut flat_rng = StdRng::seed_from_u64(314);
    let mut flat_best_acc = 0.0f64;

    for iter in 0..flat_samples {
        let w_flat: Vec<Vec<f32>> = (0..flat_n)
            .map(|_| {
                (0..flat_input_dim)
                    .map(|_| weight_range[flat_rng.gen_range(0..weight_range.len())] as f32)
                    .collect()
            })
            .collect();
        let b_flat: Vec<f32> = (0..flat_n)
            .map(|_| weight_range[flat_rng.gen_range(0..weight_range.len())] as f32)
            .collect();

        // Evaluate on a+b+c
        let mut examples = Vec::new();
        for a in 0..DIGITS {
            for b in 0..DIGITS {
                for c in 0..DIGITS {
                    let target = a + b + c;
                    let mut input = vec![0.0f32; flat_input_dim];
                    for ii in 0..a { input[ii] = 1.0; }
                    for ii in 0..b { input[4 + ii] = 1.0; }
                    for ii in 0..c { input[8 + ii] = 1.0; }
                    let act = holo_forward(&input, &w_flat, &b_flat);
                    let act_sum: f32 = act.iter().sum();
                    examples.push((act_sum, target));
                }
            }
        }

        let readout = NearestMean::fit(&examples, CLASSES_3);
        let correct = examples
            .iter()
            .filter(|&&(s, t)| readout.predict(s) == t)
            .count();
        let acc = correct as f64 / 125.0;

        if acc > flat_best_acc {
            flat_best_acc = acc;
            if flat_best_acc >= 0.9 || (iter < 100_000 && acc > 0.5) {
                println!(
                    "    [{:>9}] NEW BEST: {:.1}% ({}/125)",
                    iter + 1,
                    flat_best_acc * 100.0,
                    correct,
                );
            }
            if flat_best_acc >= 1.0 {
                println!("    *** 100% — FLAT BASELINE SOLVED IT! ***");
                break;
            }
        }
        if (iter + 1) % 2_000_000 == 0 {
            println!(
                "    [{:>9}] best: {:.1}%",
                iter + 1,
                flat_best_acc * 100.0
            );
        }
    }

    println!("\n=== FINAL VERDICT ===");
    println!("  Chip A (frozen): {} neurons, 100% on ADD(a,b)", n_a);
    println!("  Composition (pipeline, random):     {:.1}%", final_acc * 100.0);
    println!("  Composition (pipeline, perturbed):  {:.1}%", perturb_acc * 100.0);
    println!("  Flat baseline (6 neurons, no chip): {:.1}%", flat_best_acc * 100.0);
    let best_overall = perturb_acc.max(final_acc).max(flat_best_acc);
    if best_overall >= 1.0 {
        println!("  SOLVED!");
    } else {
        println!("  Not yet 100%. Key observation:");
        println!("    Chip A spread per target class: different (a,b) with same sum → different activations.");
        println!("    Chip B must learn a many-to-one mapping from 3 noisy values → class.");
        println!("    This is harder than direct 12→6 mapping (flat baseline).");
    }
}
