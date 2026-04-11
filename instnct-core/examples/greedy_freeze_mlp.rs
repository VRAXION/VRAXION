//! Greedy Freeze MLP: does layer-by-layer int8 freezing beat post-hoc quantization?
//!
//! Method A: Train float32 MLP with backprop, then quantize to int8
//! Method B: Build MLP layer-by-layer, freeze each to int8 immediately
//!
//! Tasks: XOR, 4-bit ADD (thermometer), 4-bit MUL (binary)
//! Configs: H={8,16,32}, depth={1,2,3}, act={C19(rho=8), ReLU}
//!
//! Run: cargo run --example greedy_freeze_mlp --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

// ============================================================
// Activation functions
// ============================================================

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor();
    let t = x - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

fn c19_deriv(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l || x <= -l { return 1.0; }
    let n = x.floor();
    let t = x - n;
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    (sgn + 2.0 * rho * (t * (1.0 - t))) * (1.0 - 2.0 * t)
}

fn relu(x: f32) -> f32 { x.max(0.0) }
fn relu_deriv(x: f32) -> f32 { if x > 0.0 { 1.0 } else { 0.0 } }

#[derive(Clone, Copy, PartialEq)]
enum Act { C19, ReLU }

fn activate(x: f32, act: Act) -> f32 {
    match act { Act::C19 => c19(x, 8.0), Act::ReLU => relu(x) }
}
fn activate_deriv(x: f32, act: Act) -> f32 {
    match act { Act::C19 => c19_deriv(x, 8.0), Act::ReLU => relu_deriv(x) }
}
fn act_name(act: Act) -> &'static str {
    match act { Act::C19 => "C19_rho8", Act::ReLU => "ReLU" }
}

// ============================================================
// Task data generation
// ============================================================

#[derive(Clone)]
struct Dataset {
    inputs: Vec<Vec<f32>>,
    outputs: Vec<Vec<f32>>,
    in_dim: usize,
    out_dim: usize,
}

fn make_xor() -> Dataset {
    let inputs = vec![
        vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0],
    ];
    let outputs = vec![
        vec![0.0], vec![1.0], vec![1.0], vec![0.0],
    ];
    Dataset { inputs, outputs, in_dim: 2, out_dim: 1 }
}

fn thermo4(val: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 4];
    for i in 0..val.min(4) { v[i] = 1.0; }
    v
}

fn make_add4() -> Dataset {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for a in 0..=4 {
        for b in 0..=4 {
            let mut inp = thermo4(a);
            inp.extend_from_slice(&thermo4(b));
            inputs.push(inp);
            let s = a + b; // 0..8
            let mut out = vec![0.0f32; 5]; // 5 bits for thermometer of sum
            for i in 0..s.min(5) { out[i] = 1.0; }
            // Actually encode sum as thermometer with 5 outputs (0-8 range)
            // Use binary encoding for cleaner targets
            let mut out = vec![0.0f32; 5];
            for bit in 0..5 {
                out[bit] = if (s >> bit) & 1 == 1 { 1.0 } else { 0.0 };
            }
            outputs.push(out);
        }
    }
    Dataset { inputs, outputs, in_dim: 8, out_dim: 5 }
}

fn make_mul4() -> Dataset {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for a in 0..16 {
        for b in 0..16 {
            let mut inp = vec![0.0f32; 8];
            for bit in 0..4 { inp[bit] = if (a >> bit) & 1 == 1 { 1.0 } else { 0.0 }; }
            for bit in 0..4 { inp[4 + bit] = if (b >> bit) & 1 == 1 { 1.0 } else { 0.0 }; }
            inputs.push(inp);
            let product = a * b; // 0..225
            let mut out = vec![0.0f32; 8];
            for bit in 0..8 { out[bit] = if (product >> bit) & 1 == 1 { 1.0 } else { 0.0 }; }
            outputs.push(out);
        }
    }
    Dataset { inputs, outputs, in_dim: 8, out_dim: 8 }
}

// ============================================================
// Method A: Standard MLP with backprop + post-hoc quantization
// ============================================================

#[derive(Clone)]
struct MLP {
    layers: Vec<MLPLayer>, // layer[i]: maps from prev to next
    act: Act,
}

#[derive(Clone)]
struct MLPLayer {
    w: Vec<f32>,   // [out_dim * in_dim] row-major
    b: Vec<f32>,   // [out_dim]
    in_dim: usize,
    out_dim: usize,
}

impl MLPLayer {
    fn new(in_dim: usize, out_dim: usize, rng: &mut StdRng) -> Self {
        let sc = (2.0 / in_dim as f32).sqrt();
        MLPLayer {
            w: (0..out_dim * in_dim).map(|_| rng.gen_range(-sc..sc)).collect(),
            b: vec![0.01; out_dim],
            in_dim,
            out_dim,
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.out_dim];
        for j in 0..self.out_dim {
            let mut s = self.b[j];
            let base = j * self.in_dim;
            for i in 0..self.in_dim {
                s += self.w[base + i] * input[i];
            }
            out[j] = s;
        }
        out
    }
}

impl MLP {
    fn new(dims: &[usize], act: Act, rng: &mut StdRng) -> Self {
        let mut layers = Vec::new();
        for i in 0..dims.len() - 1 {
            layers.push(MLPLayer::new(dims[i], dims[i + 1], rng));
        }
        MLP { layers, act }
    }

    fn forward(&self, input: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut pre_acts = Vec::new(); // before activation
        let mut post_acts = Vec::new(); // after activation
        let mut x = input.to_vec();
        post_acts.push(x.clone());

        for (li, layer) in self.layers.iter().enumerate() {
            let pre = layer.forward(&x);
            pre_acts.push(pre.clone());
            if li < self.layers.len() - 1 {
                // Hidden layer: apply activation
                x = pre.iter().map(|&v| activate(v, self.act)).collect();
            } else {
                // Output layer: sigmoid for binary targets
                x = pre.iter().map(|&v| sigmoid(v)).collect();
            }
            post_acts.push(x.clone());
        }
        (pre_acts, post_acts)
    }

    fn predict(&self, input: &[f32]) -> Vec<f32> {
        let (_, posts) = self.forward(input);
        posts.last().unwrap().clone()
    }

    fn train(&mut self, data: &Dataset, epochs: usize, lr: f32) {
        let n = data.inputs.len();
        for _epoch in 0..epochs {
            let mut total_loss = 0.0f32;
            for s in 0..n {
                let (pre_acts, post_acts) = self.forward(&data.inputs[s]);
                let out = post_acts.last().unwrap();

                // BCE loss gradient at output
                let mut delta: Vec<f32> = out.iter().zip(data.outputs[s].iter())
                    .map(|(&o, &t)| o - t) // sigmoid + BCE = simple gradient
                    .collect();

                for (&o, &t) in out.iter().zip(data.outputs[s].iter()) {
                    total_loss -= t * (o + 1e-7).ln() + (1.0 - t) * (1.0 - o + 1e-7).ln();
                }

                // Backprop through layers (reverse)
                for li in (0..self.layers.len()).rev() {
                    let input = &post_acts[li];
                    let layer = &self.layers[li];

                    // If hidden layer, multiply delta by activation derivative
                    if li < self.layers.len() - 1 {
                        // delta already has sigmoid derivative baked in for output
                        // For hidden layers coming from next layer's backprop:
                        // delta is already dL/d(post_act), need to multiply by act'
                    }

                    // Gradient for weights and biases
                    let mut dw = vec![0.0f32; layer.out_dim * layer.in_dim];
                    let mut db = vec![0.0f32; layer.out_dim];

                    for j in 0..layer.out_dim {
                        db[j] = delta[j];
                        let base = j * layer.in_dim;
                        for i in 0..layer.in_dim {
                            dw[base + i] = delta[j] * input[i];
                        }
                    }

                    // Propagate delta to previous layer
                    if li > 0 {
                        let mut new_delta = vec![0.0f32; layer.in_dim];
                        for i in 0..layer.in_dim {
                            let mut s = 0.0f32;
                            for j in 0..layer.out_dim {
                                s += layer.w[j * layer.in_dim + i] * delta[j];
                            }
                            // Multiply by activation derivative of the previous layer's pre-activation
                            new_delta[i] = s * activate_deriv(pre_acts[li - 1 + 1 - 1][i], self.act);
                        }
                        delta = new_delta;
                    }

                    // Update weights
                    let layer_mut = &mut self.layers[li];
                    for k in 0..layer_mut.w.len() {
                        layer_mut.w[k] -= lr * dw[k];
                    }
                    for k in 0..layer_mut.b.len() {
                        layer_mut.b[k] -= lr * db[k];
                    }
                }
            }
        }
    }

    fn accuracy(&self, data: &Dataset) -> f32 {
        let n = data.inputs.len();
        let mut correct = 0;
        for s in 0..n {
            let out = self.predict(&data.inputs[s]);
            let pred: Vec<u8> = out.iter().map(|&v| if v >= 0.5 { 1 } else { 0 }).collect();
            let target: Vec<u8> = data.outputs[s].iter().map(|&v| if v >= 0.5 { 1 } else { 0 }).collect();
            if pred == target { correct += 1; }
        }
        correct as f32 / n as f32
    }

    /// Record hidden activations for teacher targets
    fn hidden_activations(&self, data: &Dataset) -> Vec<Vec<Vec<f32>>> {
        // Returns: layer_idx -> sample_idx -> activation values
        let n = data.inputs.len();
        let mut all_hidden = Vec::new();
        for li in 0..self.layers.len() - 1 {
            let mut layer_acts = Vec::new();
            for s in 0..n {
                let (_, post_acts) = self.forward(&data.inputs[s]);
                layer_acts.push(post_acts[li + 1].clone()); // post-activation of hidden layer li
            }
            all_hidden.push(layer_acts);
        }
        all_hidden
    }
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ============================================================
// Int8 quantization for Method A
// ============================================================

#[derive(Clone)]
struct QuantizedLayer {
    w: Vec<i8>,
    scale: f32,   // multiply i8 by this to get float
    b: Vec<f32>,  // keep biases in float
    in_dim: usize,
    out_dim: usize,
}

fn quantize_weights(w: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = w.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs > 1e-10 { max_abs / 127.0 } else { 1.0 };
    let q: Vec<i8> = w.iter()
        .map(|&x| (x / scale).round().clamp(-128.0, 127.0) as i8)
        .collect();
    (q, scale)
}

fn quantized_forward(layers: &[QuantizedLayer], input: &[f32], act: Act) -> Vec<f32> {
    let mut x = input.to_vec();
    for (li, layer) in layers.iter().enumerate() {
        let mut out = vec![0.0f32; layer.out_dim];
        for j in 0..layer.out_dim {
            let mut s = layer.b[j];
            let base = j * layer.in_dim;
            for i in 0..layer.in_dim {
                s += (layer.w[base + i] as f32) * layer.scale * x[i];
            }
            if li < layers.len() - 1 {
                out[j] = activate(s, act);
            } else {
                out[j] = sigmoid(s);
            }
        }
        x = out;
    }
    x
}

fn quantize_mlp(mlp: &MLP) -> Vec<QuantizedLayer> {
    mlp.layers.iter().map(|layer| {
        let (qw, scale) = quantize_weights(&layer.w);
        QuantizedLayer {
            w: qw,
            scale,
            b: layer.b.clone(),
            in_dim: layer.in_dim,
            out_dim: layer.out_dim,
        }
    }).collect()
}

fn quantized_accuracy(qlayers: &[QuantizedLayer], data: &Dataset, act: Act) -> f32 {
    let n = data.inputs.len();
    let mut correct = 0;
    for s in 0..n {
        let out = quantized_forward(qlayers, &data.inputs[s], act);
        let pred: Vec<u8> = out.iter().map(|&v| if v >= 0.5 { 1 } else { 0 }).collect();
        let target: Vec<u8> = data.outputs[s].iter().map(|&v| if v >= 0.5 { 1 } else { 0 }).collect();
        if pred == target { correct += 1; }
    }
    correct as f32 / n as f32
}

// ============================================================
// Method B: Greedy layer-by-layer freeze
// ============================================================

/// For greedy search: try random weight vectors for a single neuron,
/// pick the one that minimizes MSE vs target on all training samples.
fn greedy_search_neuron(
    inputs: &[Vec<f32>],  // per-sample input to this neuron
    targets: &[f32],      // per-sample target for this neuron
    in_dim: usize,
    act: Act,
    is_output: bool,
    n_trials: usize,
    rng: &mut StdRng,
) -> (Vec<f32>, f32) {
    let n = inputs.len();
    let mut best_w = vec![0.0f32; in_dim + 1]; // weights + bias
    let mut best_mse = f32::MAX;

    for _ in 0..n_trials {
        // Random weights in [-2, 2] quantized to 0.25 steps
        let mut w: Vec<f32> = (0..in_dim + 1)
            .map(|_| {
                let v = rng.gen_range(-2.0f32..2.0);
                (v * 4.0).round() / 4.0 // quantize to 0.25 steps
            })
            .collect();

        // Evaluate MSE
        let mut mse = 0.0f32;
        for s in 0..n {
            let mut sum = w[in_dim]; // bias
            for i in 0..in_dim {
                sum += w[i] * inputs[s][i];
            }
            let out = if is_output { sigmoid(sum) } else { activate(sum, act) };
            let err = out - targets[s];
            mse += err * err;
        }
        mse /= n as f32;

        if mse < best_mse {
            best_mse = mse;
            best_w = w;
        }
    }

    (best_w, best_mse)
}

/// Build MLP greedily layer by layer with immediate int8 freezing.
/// target_mode: "random" = random projection of outputs, "teacher" = from trained MLP
fn greedy_freeze_build(
    data: &Dataset,
    layer_sizes: &[usize], // [in_dim, h1, h2, ..., out_dim]
    act: Act,
    target_mode: &str,
    teacher_hidden: Option<&Vec<Vec<Vec<f32>>>>, // layer -> sample -> values
    n_trials: usize,
    rng: &mut StdRng,
) -> Vec<QuantizedLayer> {
    let n_samples = data.inputs.len();
    let n_layers = layer_sizes.len() - 1;

    // Current inputs for each sample (starts as the dataset inputs)
    let mut current_inputs: Vec<Vec<f32>> = data.inputs.clone();

    let mut frozen_layers: Vec<QuantizedLayer> = Vec::new();

    for li in 0..n_layers {
        let in_dim = layer_sizes[li];
        let out_dim = layer_sizes[li + 1];
        let is_output = li == n_layers - 1;

        // Determine targets for each neuron in this layer
        let neuron_targets: Vec<Vec<f32>> = if is_output {
            // Output layer: target = dataset outputs
            (0..out_dim).map(|j| {
                (0..n_samples).map(|s| data.outputs[s][j]).collect()
            }).collect()
        } else if target_mode == "teacher" && teacher_hidden.is_some() {
            // Use teacher's hidden activations as targets
            let th = teacher_hidden.unwrap();
            if li < th.len() {
                (0..out_dim).map(|j| {
                    (0..n_samples).map(|s| {
                        if j < th[li][s].len() { th[li][s][j] } else { 0.0 }
                    }).collect()
                }).collect()
            } else {
                // Fallback to random projection
                make_random_targets(out_dim, n_samples, &data.outputs, rng)
            }
        } else {
            // Random projection of outputs
            make_random_targets(out_dim, n_samples, &data.outputs, rng)
        };

        // Search for best weights for each neuron
        let mut layer_w = vec![0.0f32; out_dim * in_dim];
        let mut layer_b = vec![0.0f32; out_dim];

        for j in 0..out_dim {
            let (best_w, _best_mse) = greedy_search_neuron(
                &current_inputs,
                &neuron_targets[j],
                in_dim,
                act,
                is_output,
                n_trials,
                rng,
            );

            // Store weights
            for i in 0..in_dim {
                layer_w[j * in_dim + i] = best_w[i];
            }
            layer_b[j] = best_w[in_dim]; // bias
        }

        // Quantize this layer to int8 immediately
        let (qw, scale) = quantize_weights(&layer_w);
        let qlayer = QuantizedLayer {
            w: qw,
            scale,
            b: layer_b.clone(),
            in_dim,
            out_dim,
        };

        // Compute this layer's outputs for all samples using quantized weights
        // These become inputs to the next layer
        let mut next_inputs = Vec::new();
        for s in 0..n_samples {
            let mut out = vec![0.0f32; out_dim];
            for j in 0..out_dim {
                let mut sum = layer_b[j];
                let base = j * in_dim;
                for i in 0..in_dim {
                    sum += (qlayer.w[base + i] as f32) * qlayer.scale * current_inputs[s][i];
                }
                if is_output {
                    out[j] = sigmoid(sum);
                } else {
                    out[j] = activate(sum, act);
                }
            }
            next_inputs.push(out);
        }

        frozen_layers.push(qlayer);
        current_inputs = next_inputs;
    }

    frozen_layers
}

fn make_random_targets(
    n_neurons: usize,
    n_samples: usize,
    outputs: &[Vec<f32>],
    rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    let out_dim = outputs[0].len();
    (0..n_neurons).map(|_| {
        // Random linear combo of output targets
        let proj: Vec<f32> = (0..out_dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        (0..n_samples).map(|s| {
            let mut v = 0.0f32;
            for k in 0..out_dim {
                v += proj[k] * outputs[s][k];
            }
            // Normalize to reasonable range
            activate(v, Act::ReLU) // non-negative target
        }).collect()
    }).collect()
}

// ============================================================
// Main experiment runner
// ============================================================

struct ExperimentResult {
    task: String,
    act: String,
    h: usize,
    depth: usize,
    target_type: String,
    float_acc: f32,
    quant_acc: f32,
    greedy_acc: f32,
}

fn run_config(
    task_name: &str,
    data: &Dataset,
    h: usize,
    depth: usize,
    act: Act,
    seed: u64,
) -> Vec<ExperimentResult> {
    let mut results = Vec::new();

    // Build layer sizes
    let mut dims = vec![data.in_dim];
    for _ in 0..depth {
        dims.push(h);
    }
    dims.push(data.out_dim);

    // Determine training params based on task difficulty
    let (epochs, lr, n_trials) = match task_name {
        "XOR" => (1000, 0.5, 20000usize),
        "ADD4" => (1500, 0.3, 15000),
        "MUL4" => (2000, 0.1, 10000),
        _ => (1000, 0.3, 10000),
    };

    // === Method A: Standard train + quantize ===
    let mut rng = StdRng::seed_from_u64(seed);
    let mut mlp = MLP::new(&dims, act, &mut rng);
    mlp.train(data, epochs, lr);
    let float_acc = mlp.accuracy(data);

    let qlayers = quantize_mlp(&mlp);
    let quant_acc = quantized_accuracy(&qlayers, data, act);

    // Get teacher hidden activations for Method B
    let teacher_hidden = mlp.hidden_activations(data);

    // === Method B: Greedy freeze with random projection targets ===
    let mut rng_greedy = StdRng::seed_from_u64(seed + 1000);
    let greedy_random = greedy_freeze_build(
        data, &dims, act, "random", None, n_trials, &mut rng_greedy,
    );
    let greedy_random_acc = quantized_accuracy(&greedy_random, data, act);

    results.push(ExperimentResult {
        task: task_name.to_string(),
        act: act_name(act).to_string(),
        h, depth,
        target_type: "random_proj".to_string(),
        float_acc,
        quant_acc,
        greedy_acc: greedy_random_acc,
    });

    // === Method B: Greedy freeze with teacher targets ===
    let mut rng_teacher = StdRng::seed_from_u64(seed + 2000);
    let greedy_teacher = greedy_freeze_build(
        data, &dims, act, "teacher", Some(&teacher_hidden), n_trials, &mut rng_teacher,
    );
    let greedy_teacher_acc = quantized_accuracy(&greedy_teacher, data, act);

    results.push(ExperimentResult {
        task: task_name.to_string(),
        act: act_name(act).to_string(),
        h, depth,
        target_type: "teacher".to_string(),
        float_acc,
        quant_acc,
        greedy_acc: greedy_teacher_acc,
    });

    results
}

fn main() {
    let t0 = Instant::now();
    println!("=== Greedy Freeze MLP Experiment ===");
    println!("Does layer-by-layer int8 freezing beat post-hoc quantization?\n");

    let xor_data = make_xor();
    let add_data = make_add4();
    let mul_data = make_mul4();

    let tasks: Vec<(&str, &Dataset)> = vec![
        ("XOR", &xor_data),
        ("ADD4", &add_data),
        ("MUL4", &mul_data),
    ];

    let h_values = [8, 16, 32];
    let depth_values = [1, 2, 3];
    let acts = [Act::C19, Act::ReLU];

    let mut all_results: Vec<ExperimentResult> = Vec::new();

    // Print header
    println!("{:<6} {:<10} {:>3} {:>5} {:<12} {:>9} {:>9} {:>9} {:>10} {:>13}",
        "task", "act", "H", "depth", "target", "float%", "quant%", "greedy%", "q_drop%", "greedy_vs_q");
    println!("{}", "-".repeat(100));

    let seed_base = 42u64;
    let mut config_idx = 0u64;

    for &(task_name, data) in &tasks {
        for &h in &h_values {
            for &depth in &depth_values {
                // Skip overly expensive configs: MUL4 with H=32, depth=3
                // (still run it but with fewer trials implicitly via run_config)
                for &act in &acts {
                    let results = run_config(
                        task_name, data, h, depth, act,
                        seed_base + config_idx * 100,
                    );

                    for r in &results {
                        let q_drop = r.float_acc - r.quant_acc;
                        let greedy_vs_q = r.greedy_acc - r.quant_acc;
                        let marker = if greedy_vs_q > 0.001 { "GREEDY+" }
                            else if greedy_vs_q < -0.001 { "QUANT+" }
                            else { "TIE" };

                        println!("{:<6} {:<10} {:>3} {:>5} {:<12} {:>8.1}% {:>8.1}% {:>8.1}% {:>9.1}% {:>8.1}% {}",
                            r.task, r.act, r.h, r.depth, r.target_type,
                            r.float_acc * 100.0, r.quant_acc * 100.0, r.greedy_acc * 100.0,
                            q_drop * 100.0, greedy_vs_q * 100.0, marker);
                    }

                    all_results.extend(results);
                    config_idx += 1;
                }
            }
        }
    }

    // ============================================================
    // Write TSV results
    // ============================================================

    let tsv_path = "greedy_freeze_results.tsv";
    let mut tsv = String::new();
    tsv.push_str("task\tact\tH\tdepth\ttarget_type\tfloat_acc\tquant_acc\tgreedy_acc\tquant_drop\tgreedy_vs_quant\n");

    for r in &all_results {
        let q_drop = r.float_acc - r.quant_acc;
        let g_vs_q = r.greedy_acc - r.quant_acc;
        tsv.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\n",
            r.task, r.act, r.h, r.depth, r.target_type,
            r.float_acc, r.quant_acc, r.greedy_acc, q_drop, g_vs_q));
    }
    std::fs::write(tsv_path, &tsv).expect("Failed to write TSV");
    println!("\nTSV written to: {}", tsv_path);

    // ============================================================
    // Summary statistics
    // ============================================================

    println!("\n=== SUMMARY ===");

    // Count wins
    let mut greedy_wins = 0;
    let mut quant_wins = 0;
    let mut ties = 0;
    let mut greedy_teacher_wins = 0;
    let mut greedy_random_wins = 0;
    let mut total_greedy_advantage = 0.0f32;
    let mut total_quant_drop = 0.0f32;
    let n_total = all_results.len();

    for r in &all_results {
        let q_drop = r.float_acc - r.quant_acc;
        let g_vs_q = r.greedy_acc - r.quant_acc;
        total_quant_drop += q_drop;
        total_greedy_advantage += g_vs_q;

        if g_vs_q > 0.001 { greedy_wins += 1; }
        else if g_vs_q < -0.001 { quant_wins += 1; }
        else { ties += 1; }

        if g_vs_q > 0.001 {
            if r.target_type == "teacher" { greedy_teacher_wins += 1; }
            else { greedy_random_wins += 1; }
        }
    }

    println!("Total configs tested: {}", n_total);
    println!("Greedy freeze wins: {} ({:.1}%)", greedy_wins, 100.0 * greedy_wins as f32 / n_total as f32);
    println!("Post-hoc quant wins: {} ({:.1}%)", quant_wins, 100.0 * quant_wins as f32 / n_total as f32);
    println!("Ties:                {} ({:.1}%)", ties, 100.0 * ties as f32 / n_total as f32);
    println!("Avg quantization drop: {:.2}%", 100.0 * total_quant_drop / n_total as f32);
    println!("Avg greedy advantage over quant: {:.2}%", 100.0 * total_greedy_advantage / n_total as f32);
    println!("Greedy wins with teacher targets: {}", greedy_teacher_wins);
    println!("Greedy wins with random targets:  {}", greedy_random_wins);

    // Per-task summary
    for task in &["XOR", "ADD4", "MUL4"] {
        let task_results: Vec<&ExperimentResult> = all_results.iter()
            .filter(|r| r.task == *task).collect();
        if task_results.is_empty() { continue; }

        let avg_float: f32 = task_results.iter().map(|r| r.float_acc).sum::<f32>() / task_results.len() as f32;
        let avg_quant: f32 = task_results.iter().map(|r| r.quant_acc).sum::<f32>() / task_results.len() as f32;
        let avg_greedy: f32 = task_results.iter().map(|r| r.greedy_acc).sum::<f32>() / task_results.len() as f32;
        let best_greedy = task_results.iter().map(|r| r.greedy_acc).fold(0.0f32, f32::max);
        let best_quant = task_results.iter().map(|r| r.quant_acc).fold(0.0f32, f32::max);

        println!("\n--- {} ---", task);
        println!("  Avg float: {:.1}%  Avg quant: {:.1}%  Avg greedy: {:.1}%",
            avg_float * 100.0, avg_quant * 100.0, avg_greedy * 100.0);
        println!("  Best quant: {:.1}%  Best greedy: {:.1}%",
            best_quant * 100.0, best_greedy * 100.0);
    }

    // Per-activation summary
    for act_str in &["C19_rho8", "ReLU"] {
        let act_results: Vec<&ExperimentResult> = all_results.iter()
            .filter(|r| r.act == *act_str).collect();
        if act_results.is_empty() { continue; }
        let g_wins = act_results.iter().filter(|r| r.greedy_acc - r.quant_acc > 0.001).count();
        let q_wins = act_results.iter().filter(|r| r.quant_acc - r.greedy_acc > 0.001).count();
        println!("\n--- {} ---  greedy_wins={} quant_wins={}", act_str, g_wins, q_wins);
    }

    let elapsed = t0.elapsed().as_secs_f32();
    println!("\nTotal time: {:.1}s", elapsed);
    println!("\n=== EXPERIMENT COMPLETE ===");
}
