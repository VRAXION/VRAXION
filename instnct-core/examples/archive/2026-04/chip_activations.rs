//! Activation function sweep for recurrent generalization.
//!
//! The key problem: recurrent models don't generalize across tick counts.
//! We need an activation function where the same W works for any number of ticks.
//!
//! Candidates (all bounded, won't explode):
//!   1. tanh          — classic RNN, bounded [-1, 1]
//!   2. sigmoid       — bounded [0, 1]
//!   3. softsign      — x/(1+|x|), bounded [-1, 1]
//!   4. ReLU+clamp    — max(0, x) clamped to [0, C]
//!   5. clamp only    — linear but clamped to [-C, C]
//!   6. signed_sqrt   — sign(x)*sqrt(|x|), bounded growth
//!   7. step          — binary: 0 or 1 (like original INSTNCT spike)
//!
//! All use: per-neuron bias, holographic all-to-all, nearest-mean readout.
//! Train on 2-input ADD recurrently, test generalization to 3-8 inputs.
//!
//! Run: cargo run --example chip_activations --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIGITS: usize = 5;

fn thermo(val: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 4];
    for i in 0..val.min(4) { v[i] = 1.0; }
    v
}

// ============================================================
// Activation functions
// ============================================================
fn act_tanh(x: f32) -> f32 { x.tanh() }
fn act_sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn act_softsign(x: f32) -> f32 { x / (1.0 + x.abs()) }
fn act_relu_clamp(x: f32) -> f32 { x.max(0.0).min(4.0) }
fn act_clamp(x: f32) -> f32 { x.clamp(-4.0, 4.0) }
fn act_signed_sqrt(x: f32) -> f32 { x.signum() * x.abs().sqrt() }
fn act_step(x: f32) -> f32 { if x > 0.0 { 1.0 } else { 0.0 } }
fn act_signed_square(x: f32) -> f32 { x * x.abs() } // baseline (explodes)
fn act_linear(x: f32) -> f32 { x } // no activation
fn act_relu(x: f32) -> f32 { x.max(0.0) }
fn act_elu(x: f32) -> f32 { if x >= 0.0 { x.min(4.0) } else { 0.5 * (x.exp() - 1.0) } }
fn act_swish_clamp(x: f32) -> f32 { (x / (1.0 + (-x).exp())).clamp(-4.0, 4.0) }

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
    let mut result = vec![vec![]];
    for _ in 0..n {
        let mut new_r = Vec::new();
        for combo in &result {
            for d in 0..DIGITS {
                let mut c = combo.clone();
                c.push(d);
                new_r.push(c);
            }
        }
        result = new_r;
    }
    result
}

/// Recurrent forward: one digit per tick, with given activation function.
/// W: N × (N+4), bias: N (per-neuron)
fn recurrent_forward(
    digits: &[usize],
    w: &[Vec<f32>],
    bias: &[f32],
    act_fn: fn(f32) -> f32,
) -> Vec<f32> {
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
            act[i] = act_fn(sum);
        }
    }
    act
}

/// Evaluate recurrent model on n-input addition
fn eval_recurrent(
    w: &[Vec<f32>],
    bias: &[f32],
    n_inputs: usize,
    act_fn: fn(f32) -> f32,
) -> f64 {
    let max_sum = (DIGITS - 1) * n_inputs;
    let combos = gen_combos(n_inputs);
    let mut examples = Vec::new();
    for combo in &combos {
        let target: usize = combo.iter().sum();
        let act = recurrent_forward(combo, w, bias, act_fn);
        let s: f32 = act.iter().sum();
        if s.is_nan() || s.is_infinite() { return 0.0; }
        examples.push((s, target));
    }
    let readout = NearestMean::fit(&examples, max_sum + 1);
    examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / combos.len() as f64
}

/// Search: random + perturbation, training on train_n inputs
fn search_recurrent(
    n: usize,
    act_fn: fn(f32) -> f32,
    train_n: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<f32>, f64) {
    let input_dim = n + 4;
    let weight_range: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut rng = StdRng::seed_from_u64(seed);
    let total_params = n * input_dim + n; // W + bias

    // Random search
    let mut best_acc = 0.0f64;
    let mut best_w = vec![vec![0.0f32; input_dim]; n];
    let mut best_bias = vec![0.0f32; n];

    for _ in 0..3_000_000u64 {
        let w: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| weight_range[rng.gen_range(0..weight_range.len())]).collect()).collect();
        let bias: Vec<f32> = (0..n).map(|_| weight_range[rng.gen_range(0..weight_range.len())]).collect();
        let acc = eval_recurrent(&w, &bias, train_n, act_fn);
        if acc > best_acc { best_acc = acc; best_w = w; best_bias = bias; }
        if best_acc >= 1.0 { return (best_w, best_bias, 1.0); }
    }

    // Perturbation
    let mut current = best_acc;
    for _step in 0..500_000u64 {
        let idx = rng.gen_range(0..total_params);
        let delta: f32 = rng.gen_range(-0.3..0.3);
        let (old, is_b, i, j) = if idx < n * input_dim {
            let i = idx / input_dim; let j = idx % input_dim;
            let old = best_w[i][j]; best_w[i][j] += delta; (old, false, i, j)
        } else {
            let i = idx - n * input_dim;
            let old = best_bias[i]; best_bias[i] += delta; (old, true, i, 0)
        };
        let acc = eval_recurrent(&best_w, &best_bias, train_n, act_fn);
        if acc >= current { current = acc; } else {
            if is_b { best_bias[i] = old; } else { best_w[i][j] = old; }
        }
        if current >= 1.0 { return (best_w, best_bias, 1.0); }
    }

    (best_w, best_bias, current)
}

fn main() {
    println!("=== ACTIVATION FUNCTION SWEEP FOR RECURRENT GENERALIZATION ===");
    println!("All: 3 neurons, per-neuron bias, holographic, nearest-mean readout");
    println!("Train on 2-input ADD recurrently, test generalization to 3-8 inputs.\n");

    let activations: Vec<(&str, fn(f32) -> f32)> = vec![
        ("tanh",          act_tanh),
        ("sigmoid",       act_sigmoid),
        ("softsign",      act_softsign),
        ("relu_clamp4",   act_relu_clamp),
        ("clamp4",        act_clamp),
        ("signed_sqrt",   act_signed_sqrt),
        ("step",          act_step),
        ("signed_square", act_signed_square),
        ("linear",        act_linear),
        ("relu",          act_relu),
        ("elu_clamp4",    act_elu),
        ("swish_clamp4",  act_swish_clamp),
    ];

    let n = 3;

    // Phase 1: Train each on 2-input, test generalization
    println!("=== PHASE 1: Train on 2-input, test generalization ===\n");
    println!("{:<16} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "activation", "2-in", "3-in", "4-in", "5-in", "6-in", "7-in", "8-in");
    println!("{}", "-".repeat(80));

    let mut results_phase1: Vec<(&str, Vec<f64>)> = Vec::new();

    for (name, act_fn) in &activations {
        let (w, bias, train_acc) = search_recurrent(n, *act_fn, 2, 42);

        let mut accs = Vec::new();
        for n_inputs in 2..=8 {
            let acc = eval_recurrent(&w, &bias, n_inputs, *act_fn);
            accs.push(acc);
        }

        print!("{:<16}", name);
        for acc in &accs {
            print!(" {:>5.1}%", acc * 100.0);
        }
        println!();

        results_phase1.push((name, accs));
    }

    // Phase 2: Train best candidates on 3-input, retest
    println!("\n=== PHASE 2: Train on 3-input, test generalization ===\n");

    // Pick candidates that scored > 10% on 3-input in phase 1
    let candidates: Vec<(&str, fn(f32) -> f32)> = activations.iter()
        .zip(results_phase1.iter())
        .filter(|(_, (_, accs))| accs.len() > 1 && accs[1] > 0.10)
        .map(|((name, f), _)| (*name, *f))
        .collect();

    if candidates.is_empty() {
        println!("  No candidates with >10% on 3-input from 2-input training.");
    } else {
        println!("{:<16} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
            "activation", "2-in", "3-in", "4-in", "5-in", "6-in", "7-in", "8-in");
        println!("{}", "-".repeat(80));

        for (name, act_fn) in &candidates {
            let (w, bias, _) = search_recurrent(n, *act_fn, 3, 777);
            print!("{:<16}", name);
            for n_inputs in 2..=8 {
                let acc = eval_recurrent(&w, &bias, n_inputs, *act_fn);
                print!(" {:>5.1}%", acc * 100.0);
            }
            println!();
        }
    }

    // Phase 3: Try with more neurons (5, 8) on best activation
    println!("\n=== PHASE 3: More neurons (5, 8) with best activations ===\n");

    let phase3_acts: Vec<(&str, fn(f32) -> f32)> = vec![
        ("tanh", act_tanh),
        ("softsign", act_softsign),
        ("clamp4", act_clamp),
        ("relu_clamp4", act_relu_clamp),
    ];

    for &n_neurons in &[5, 8] {
        println!("  {} neurons, trained on 3-input:", n_neurons);
        println!("  {:<16} {:>6} {:>6} {:>6} {:>6} {:>6}",
            "activation", "2-in", "3-in", "4-in", "5-in", "6-in");
        println!("  {}", "-".repeat(60));

        for (name, act_fn) in &phase3_acts {
            let (w, bias, _) = search_recurrent(n_neurons, *act_fn, 3, 42);
            print!("  {:<16}", name);
            for n_inputs in 2..=6 {
                let acc = eval_recurrent(&w, &bias, n_inputs, *act_fn);
                print!(" {:>5.1}%", acc * 100.0);
            }
            println!();
        }
        println!();
    }

    // Phase 4: Multi-input training (train on 2+3+4 jointly)
    println!("=== PHASE 4: Joint training (2+3+4 input simultaneously) ===\n");

    let joint_acts: Vec<(&str, fn(f32) -> f32)> = vec![
        ("tanh", act_tanh),
        ("softsign", act_softsign),
        ("clamp4", act_clamp),
    ];

    for (name, act_fn) in &joint_acts {
        let input_dim = n + 4;
        let weight_range: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let total_params = n * input_dim + n;
        let mut rng = StdRng::seed_from_u64(42);

        // Joint eval: average accuracy on 2,3,4 inputs
        let joint_eval = |w: &[Vec<f32>], bias: &[f32]| -> f64 {
            let mut total = 0.0;
            for n_in in 2..=4 {
                total += eval_recurrent(w, bias, n_in, *act_fn);
            }
            total / 3.0
        };

        let mut best_acc = 0.0f64;
        let mut best_w = vec![vec![0.0f32; input_dim]; n];
        let mut best_bias = vec![0.0f32; n];

        for _ in 0..3_000_000u64 {
            let w: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| weight_range[rng.gen_range(0..weight_range.len())]).collect()).collect();
            let bias: Vec<f32> = (0..n).map(|_| weight_range[rng.gen_range(0..weight_range.len())]).collect();
            let acc = joint_eval(&w, &bias);
            if acc > best_acc { best_acc = acc; best_w = w; best_bias = bias; }
            if best_acc >= 1.0 { break; }
        }

        // Perturbation
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
            let acc = joint_eval(&best_w, &best_bias);
            if acc >= current { current = acc; } else {
                if is_b { best_bias[i] = old; } else { best_w[i][j] = old; }
            }
            if current >= 1.0 { break; }
        }

        print!("  {:<16} joint={:.1}% |", name, current * 100.0);
        for n_inputs in 2..=8 {
            let acc = eval_recurrent(&best_w, &best_bias, n_inputs, *act_fn);
            print!(" {}:{:.0}%", n_inputs, acc * 100.0);
        }
        println!();
    }

    println!("\n=== DONE ===");
}
