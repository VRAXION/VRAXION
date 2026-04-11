//! Greedy Init -> Backprop Fine-tune experiment
//!
//! Four methods compared:
//!   A) Standard backprop from random init, then quantize
//!   B) Greedy freeze only (layer-by-layer, int8 immediately)
//!   C) Greedy init -> unfreeze -> backprop fine-tune -> re-quantize
//!   D) Greedy init -> unfreeze LAST layer only -> backprop -> re-quantize
//!
//! Tasks: XOR (2->H->1), ADD4 (8->H->5), MUL4 (8->H->H->8)
//!
//! Run: cargo run --example greedy_then_backprop --release

use std::time::Instant;

// ============================================================
// RNG (no external crates)
// ============================================================

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng {
            state: seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1),
        }
    }
    fn next(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 33) % 65536) as f32 / 65536.0
    }
    fn rf(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.f32() * (hi - lo)
    }
    #[allow(dead_code)]
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() as usize) % (i + 1);
            v.swap(i, j);
        }
    }
}

// ============================================================
// Activation functions
// ============================================================

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l {
        return x - l;
    }
    if x <= -l {
        return x + l;
    }
    let n = x.floor();
    let t = x - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

fn c19_deriv(x: f32, rho: f32) -> f32 {
    let l = 6.0;
    if x >= l || x <= -l {
        return 1.0;
    }
    let n = x.floor();
    let t = x - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    (sgn + 2.0 * rho * h) * (1.0 - 2.0 * t)
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn relu_deriv(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-30.0, 30.0)).exp())
}

#[derive(Clone, Copy, PartialEq)]
enum Act {
    C19,
    ReLU,
}

fn activate(x: f32, act: Act) -> f32 {
    match act {
        Act::C19 => c19(x, 8.0),
        Act::ReLU => relu(x),
    }
}

fn activate_deriv(x: f32, act: Act) -> f32 {
    match act {
        Act::C19 => c19_deriv(x, 8.0),
        Act::ReLU => relu_deriv(x),
    }
}

fn act_name(act: Act) -> &'static str {
    match act {
        Act::C19 => "C19_rho8",
        Act::ReLU => "ReLU",
    }
}

// ============================================================
// Int8 quantization
// ============================================================

fn quantize(w: &[f32]) -> (Vec<i8>, f32) {
    let mx = w.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-6);
    let scale = 127.0 / mx;
    (
        w.iter()
            .map(|&x| (x * scale).round().clamp(-128.0, 127.0) as i8)
            .collect(),
        scale,
    )
}

fn dequantize(q: &[i8], scale: f32) -> Vec<f32> {
    q.iter().map(|&x| x as f32 / scale).collect()
}

// ============================================================
// Dataset
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
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let outputs = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    Dataset {
        inputs,
        outputs,
        in_dim: 2,
        out_dim: 1,
    }
}

fn thermo4(val: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 4];
    for i in 0..val.min(4) {
        v[i] = 1.0;
    }
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
            let s = (a + b).min(5);
            // thermometer encoding for sum capped at 5
            let mut out = vec![0.0f32; 5];
            for i in 0..s {
                out[i] = 1.0;
            }
            outputs.push(out);
        }
    }
    Dataset {
        inputs,
        outputs,
        in_dim: 8,
        out_dim: 5,
    }
}

fn make_mul4() -> Dataset {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for a in 0..16u32 {
        for b in 0..16u32 {
            let mut inp = vec![0.0f32; 8];
            for bit in 0..4 {
                inp[bit] = if (a >> bit) & 1 == 1 { 1.0 } else { 0.0 };
            }
            for bit in 0..4 {
                inp[4 + bit] = if (b >> bit) & 1 == 1 { 1.0 } else { 0.0 };
            }
            inputs.push(inp);
            let product = (a * b) % 16; // mod 16
            let mut out = vec![0.0f32; 8];
            for bit in 0..8 {
                out[bit] = if (product >> bit) & 1 == 1 { 1.0 } else { 0.0 };
            }
            outputs.push(out);
        }
    }
    Dataset {
        inputs,
        outputs,
        in_dim: 8,
        out_dim: 8,
    }
}

// ============================================================
// MLP (float32, trainable)
// ============================================================

#[derive(Clone)]
struct Layer {
    w: Vec<f32>,  // [out_dim * in_dim] row-major
    b: Vec<f32>,  // [out_dim]
    in_dim: usize,
    out_dim: usize,
}

impl Layer {
    fn new_he(in_dim: usize, out_dim: usize, rng: &mut Rng) -> Self {
        let sc = (2.0 / in_dim as f32).sqrt();
        Layer {
            w: (0..out_dim * in_dim).map(|_| rng.rf(-sc, sc)).collect(),
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

#[derive(Clone)]
struct MLP {
    layers: Vec<Layer>,
    act: Act,
}

impl MLP {
    fn new(dims: &[usize], act: Act, rng: &mut Rng) -> Self {
        let mut layers = Vec::new();
        for i in 0..dims.len() - 1 {
            layers.push(Layer::new_he(dims[i], dims[i + 1], rng));
        }
        MLP { layers, act }
    }

    /// Forward returning pre-activations and post-activations per layer.
    /// post_acts[0] = input, post_acts[i+1] = after layer i.
    fn forward_full(&self, input: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut pre_acts = Vec::new();
        let mut post_acts = vec![input.to_vec()];
        let mut x = input.to_vec();

        for (li, layer) in self.layers.iter().enumerate() {
            let pre = layer.forward(&x);
            pre_acts.push(pre.clone());
            if li < self.layers.len() - 1 {
                x = pre.iter().map(|&v| activate(v, self.act)).collect();
            } else {
                x = pre.iter().map(|&v| sigmoid(v)).collect();
            }
            post_acts.push(x.clone());
        }
        (pre_acts, post_acts)
    }

    fn predict(&self, input: &[f32]) -> Vec<f32> {
        let (_, posts) = self.forward_full(input);
        posts.last().unwrap().clone()
    }

    fn accuracy(&self, data: &Dataset) -> f32 {
        let n = data.inputs.len();
        let mut correct = 0;
        for s in 0..n {
            let out = self.predict(&data.inputs[s]);
            let pred: Vec<u8> = out.iter().map(|&v| if v >= 0.5 { 1 } else { 0 }).collect();
            let tgt: Vec<u8> = data.outputs[s]
                .iter()
                .map(|&v| if v >= 0.5 { 1 } else { 0 })
                .collect();
            if pred == tgt {
                correct += 1;
            }
        }
        correct as f32 / n as f32
    }

    /// Train all layers with backprop (SGD + BCE).
    fn train(&mut self, data: &Dataset, epochs: usize, lr: f32) {
        let n = data.inputs.len();
        for _ep in 0..epochs {
            for s in 0..n {
                let (pre_acts, post_acts) = self.forward_full(&data.inputs[s]);
                let out = post_acts.last().unwrap();

                // Output delta: sigmoid + BCE => delta = output - target
                let mut delta: Vec<f32> = out
                    .iter()
                    .zip(data.outputs[s].iter())
                    .map(|(&o, &t)| o - t)
                    .collect();

                // Backprop through layers (reverse)
                for li in (0..self.layers.len()).rev() {
                    let inp = &post_acts[li];
                    let layer = &self.layers[li];

                    // Compute weight/bias gradients
                    let mut dw = vec![0.0f32; layer.out_dim * layer.in_dim];
                    let mut db = vec![0.0f32; layer.out_dim];
                    for j in 0..layer.out_dim {
                        db[j] = delta[j];
                        let base = j * layer.in_dim;
                        for i in 0..layer.in_dim {
                            dw[base + i] = delta[j] * inp[i];
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
                            new_delta[i] = s * activate_deriv(pre_acts[li - 1][i], self.act);
                        }
                        delta = new_delta;
                    }

                    // Update weights
                    let lm = &mut self.layers[li];
                    for k in 0..lm.w.len() {
                        lm.w[k] -= lr * dw[k];
                    }
                    for k in 0..lm.b.len() {
                        lm.b[k] -= lr * db[k];
                    }
                }
            }
        }
    }

    /// Train only the last layer with backprop (freeze all earlier layers).
    fn train_last_layer_only(&mut self, data: &Dataset, epochs: usize, lr: f32) {
        let n = data.inputs.len();
        let last = self.layers.len() - 1;
        for _ep in 0..epochs {
            for s in 0..n {
                let (_, post_acts) = self.forward_full(&data.inputs[s]);
                let out = post_acts.last().unwrap();

                let delta: Vec<f32> = out
                    .iter()
                    .zip(data.outputs[s].iter())
                    .map(|(&o, &t)| o - t)
                    .collect();

                let inp = &post_acts[last]; // input to the last layer
                let layer = &self.layers[last];

                let mut dw = vec![0.0f32; layer.out_dim * layer.in_dim];
                let mut db = vec![0.0f32; layer.out_dim];
                for j in 0..layer.out_dim {
                    db[j] = delta[j];
                    let base = j * layer.in_dim;
                    for i in 0..layer.in_dim {
                        dw[base + i] = delta[j] * inp[i];
                    }
                }

                let lm = &mut self.layers[last];
                for k in 0..lm.w.len() {
                    lm.w[k] -= lr * dw[k];
                }
                for k in 0..lm.b.len() {
                    lm.b[k] -= lr * db[k];
                }
            }
        }
    }
}

// ============================================================
// Quantized MLP (int8 weights, float bias)
// ============================================================

#[derive(Clone)]
struct QLay {
    w: Vec<i8>,
    scale: f32,
    b: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

fn quantize_mlp(mlp: &MLP) -> Vec<QLay> {
    mlp.layers
        .iter()
        .map(|l| {
            let (qw, sc) = quantize(&l.w);
            QLay {
                w: qw,
                scale: sc,
                b: l.b.clone(),
                in_dim: l.in_dim,
                out_dim: l.out_dim,
            }
        })
        .collect()
}

fn qforward(layers: &[QLay], input: &[f32], act: Act) -> Vec<f32> {
    let mut x = input.to_vec();
    for (li, l) in layers.iter().enumerate() {
        let mut out = vec![0.0f32; l.out_dim];
        for j in 0..l.out_dim {
            let mut s = l.b[j];
            let base = j * l.in_dim;
            for i in 0..l.in_dim {
                s += (l.w[base + i] as f32 / l.scale) * x[i];
            }
            out[j] = if li < layers.len() - 1 {
                activate(s, act)
            } else {
                sigmoid(s)
            };
        }
        x = out;
    }
    x
}

fn qacc(layers: &[QLay], data: &Dataset, act: Act) -> f32 {
    let n = data.inputs.len();
    let mut correct = 0;
    for s in 0..n {
        let out = qforward(layers, &data.inputs[s], act);
        let pred: Vec<u8> = out.iter().map(|&v| if v >= 0.5 { 1 } else { 0 }).collect();
        let tgt: Vec<u8> = data.outputs[s]
            .iter()
            .map(|&v| if v >= 0.5 { 1 } else { 0 })
            .collect();
        if pred == tgt {
            correct += 1;
        }
    }
    correct as f32 / n as f32
}

// ============================================================
// Greedy freeze: build layer-by-layer, int8 immediately
// ============================================================

/// Random projection targets for hidden neurons.
fn random_targets(
    n_neurons: usize,
    n_samples: usize,
    outputs: &[Vec<f32>],
    rng: &mut Rng,
) -> Vec<Vec<f32>> {
    let od = outputs[0].len();
    (0..n_neurons)
        .map(|_| {
            let proj: Vec<f32> = (0..od).map(|_| rng.rf(-1.0, 1.0)).collect();
            (0..n_samples)
                .map(|s| {
                    let mut v = 0.0f32;
                    for k in 0..od {
                        v += proj[k] * outputs[s][k];
                    }
                    relu(v)
                })
                .collect()
        })
        .collect()
}

/// Greedy search for a single neuron: try n_trials random weight vectors.
fn greedy_neuron(
    inputs: &[Vec<f32>],
    targets: &[f32],
    in_dim: usize,
    act: Act,
    is_output: bool,
    n_trials: usize,
    rng: &mut Rng,
) -> Vec<f32> {
    let n = inputs.len();
    let mut best_w = vec![0.0f32; in_dim + 1];
    let mut best_mse = f32::MAX;

    for _ in 0..n_trials {
        let w: Vec<f32> = (0..in_dim + 1)
            .map(|_| {
                let v = rng.rf(-2.0, 2.0);
                (v * 4.0).round() / 4.0
            })
            .collect();

        let mut mse = 0.0f32;
        for s in 0..n {
            let mut sum = w[in_dim];
            for i in 0..in_dim {
                sum += w[i] * inputs[s][i];
            }
            let out = if is_output {
                sigmoid(sum)
            } else {
                activate(sum, act)
            };
            let e = out - targets[s];
            mse += e * e;
        }
        mse /= n as f32;

        if mse < best_mse {
            best_mse = mse;
            best_w = w;
        }
    }
    best_w
}

/// Build greedy-frozen quantized layers. Returns Vec<QLay>.
fn greedy_build(
    data: &Dataset,
    dims: &[usize],
    act: Act,
    n_trials: usize,
    rng: &mut Rng,
) -> Vec<QLay> {
    let n_samples = data.inputs.len();
    let n_layers = dims.len() - 1;
    let mut current_inputs = data.inputs.clone();
    let mut frozen: Vec<QLay> = Vec::new();

    for li in 0..n_layers {
        let id = dims[li];
        let od = dims[li + 1];
        let is_out = li == n_layers - 1;

        let neuron_targets: Vec<Vec<f32>> = if is_out {
            (0..od)
                .map(|j| (0..n_samples).map(|s| data.outputs[s][j]).collect())
                .collect()
        } else {
            random_targets(od, n_samples, &data.outputs, rng)
        };

        let mut layer_w = vec![0.0f32; od * id];
        let mut layer_b = vec![0.0f32; od];

        for j in 0..od {
            let bw = greedy_neuron(&current_inputs, &neuron_targets[j], id, act, is_out, n_trials, rng);
            for i in 0..id {
                layer_w[j * id + i] = bw[i];
            }
            layer_b[j] = bw[id];
        }

        let (qw, sc) = quantize(&layer_w);
        let ql = QLay {
            w: qw,
            scale: sc,
            b: layer_b.clone(),
            in_dim: id,
            out_dim: od,
        };

        // Forward through quantized layer for next inputs
        let mut next = Vec::with_capacity(n_samples);
        for s in 0..n_samples {
            let mut out = vec![0.0f32; od];
            for j in 0..od {
                let mut sum = layer_b[j];
                let base = j * id;
                for i in 0..id {
                    sum += (ql.w[base + i] as f32 / ql.scale) * current_inputs[s][i];
                }
                out[j] = if is_out {
                    sigmoid(sum)
                } else {
                    activate(sum, act)
                };
            }
            next.push(out);
        }
        frozen.push(ql);
        current_inputs = next;
    }
    frozen
}

/// Convert quantized layers back to float MLP (for fine-tuning).
fn qlayers_to_mlp(qlayers: &[QLay], act: Act) -> MLP {
    let layers: Vec<Layer> = qlayers
        .iter()
        .map(|ql| {
            let w = dequantize(&ql.w, ql.scale);
            Layer {
                w,
                b: ql.b.clone(),
                in_dim: ql.in_dim,
                out_dim: ql.out_dim,
            }
        })
        .collect();
    MLP { layers, act }
}

// ============================================================
// Experiment result
// ============================================================

struct Res {
    task: String,
    act_s: String,
    h: usize,
    depth: usize,
    seed: u64,
    method: String,
    float_acc: f32,
    quant_acc: f32,
}

// ============================================================
// Main
// ============================================================

fn main() {
    let t0 = Instant::now();

    println!("=== Greedy Init -> Backprop Fine-tune Experiment ===");
    println!("Methods: A=std_backprop, B=greedy_freeze, C=greedy+backprop_all, D=greedy+backprop_last\n");

    let xor = make_xor();
    let add4 = make_add4();
    let mul4 = make_mul4();

    let tasks: Vec<(&str, &Dataset)> = vec![("XOR", &xor), ("ADD4", &add4), ("MUL4", &mul4)];
    let h_vals = [16, 32];
    let depth_vals = [1, 2];
    let acts = [Act::C19, Act::ReLU];
    let seeds: [u64; 3] = [42, 123, 7];

    let mut all: Vec<Res> = Vec::new();

    // Header
    println!(
        "{:<5} {:<9} {:>3} {:>5} {:>5} {:<22} {:>9} {:>9}",
        "task", "act", "H", "dep", "seed", "method", "float%", "quant%"
    );
    println!("{}", "-".repeat(80));

    let n_trials = 20000usize;

    for &(task_name, data) in &tasks {
        for &h in &h_vals {
            for &depth in &depth_vals {
                // MUL4 always depth=2
                let effective_depth = if task_name == "MUL4" && depth < 2 {
                    continue; // skip depth=1 for MUL4, only run depth=2
                } else {
                    depth
                };
                let _ = effective_depth; // suppress warning; we use `depth` below which is already >=2 for MUL4

                let mut dims = vec![data.in_dim];
                for _ in 0..depth {
                    dims.push(h);
                }
                dims.push(data.out_dim);

                for &act in &acts {
                    for &seed in &seeds {
                        // === Method A: Standard backprop ===
                        {
                            let mut rng = Rng::new(seed);
                            let mut mlp = MLP::new(&dims, act, &mut rng);
                            mlp.train(data, 1000, 0.01);
                            let fa = mlp.accuracy(data);
                            let ql = quantize_mlp(&mlp);
                            let qa = qacc(&ql, data, act);

                            let r = Res {
                                task: task_name.into(),
                                act_s: act_name(act).into(),
                                h,
                                depth,
                                seed,
                                method: "A_std_backprop".into(),
                                float_acc: fa,
                                quant_acc: qa,
                            };
                            println!(
                                "{:<5} {:<9} {:>3} {:>5} {:>5} {:<22} {:>8.1}% {:>8.1}%",
                                r.task, r.act_s, r.h, r.depth, r.seed, r.method,
                                fa * 100.0, qa * 100.0
                            );
                            all.push(r);
                        }

                        // === Method B: Greedy freeze only ===
                        let greedy_ql;
                        {
                            let mut rng = Rng::new(seed.wrapping_add(1000));
                            greedy_ql = greedy_build(data, &dims, act, n_trials, &mut rng);
                            let ga = qacc(&greedy_ql, data, act);

                            let r = Res {
                                task: task_name.into(),
                                act_s: act_name(act).into(),
                                h,
                                depth,
                                seed,
                                method: "B_greedy_only".into(),
                                float_acc: ga, // already int8
                                quant_acc: ga,
                            };
                            println!(
                                "{:<5} {:<9} {:>3} {:>5} {:>5} {:<22} {:>8.1}% {:>8.1}%",
                                r.task, r.act_s, r.h, r.depth, r.seed, r.method,
                                ga * 100.0, ga * 100.0
                            );
                            all.push(r);
                        }

                        // === Method C: Greedy init -> backprop ALL layers ===
                        {
                            let mut mlp_c = qlayers_to_mlp(&greedy_ql, act);
                            mlp_c.train(data, 500, 0.01);
                            let fa = mlp_c.accuracy(data);
                            let ql_c = quantize_mlp(&mlp_c);
                            let qa = qacc(&ql_c, data, act);

                            let r = Res {
                                task: task_name.into(),
                                act_s: act_name(act).into(),
                                h,
                                depth,
                                seed,
                                method: "C_greedy+bp_all".into(),
                                float_acc: fa,
                                quant_acc: qa,
                            };
                            println!(
                                "{:<5} {:<9} {:>3} {:>5} {:>5} {:<22} {:>8.1}% {:>8.1}%",
                                r.task, r.act_s, r.h, r.depth, r.seed, r.method,
                                fa * 100.0, qa * 100.0
                            );
                            all.push(r);
                        }

                        // === Method D: Greedy init -> backprop LAST layer only ===
                        {
                            let mut mlp_d = qlayers_to_mlp(&greedy_ql, act);
                            mlp_d.train_last_layer_only(data, 500, 0.01);
                            let fa = mlp_d.accuracy(data);
                            let ql_d = quantize_mlp(&mlp_d);
                            let qa = qacc(&ql_d, data, act);

                            let r = Res {
                                task: task_name.into(),
                                act_s: act_name(act).into(),
                                h,
                                depth,
                                seed,
                                method: "D_greedy+bp_last".into(),
                                float_acc: fa,
                                quant_acc: qa,
                            };
                            println!(
                                "{:<5} {:<9} {:>3} {:>5} {:>5} {:<22} {:>8.1}% {:>8.1}%",
                                r.task, r.act_s, r.h, r.depth, r.seed, r.method,
                                fa * 100.0, qa * 100.0
                            );
                            all.push(r);
                        }
                    }
                }
            }
        }
    }

    // ============================================================
    // Write TSV
    // ============================================================

    let tsv_path = "greedy_backprop_results.tsv";
    let mut tsv = String::from("task\tact\tH\tdepth\tseed\tmethod\tfloat_acc\tquant_acc\tbest_acc_int8\n");
    for r in &all {
        let best_i8 = r.quant_acc;
        tsv.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\n",
            r.task, r.act_s, r.h, r.depth, r.seed, r.method, r.float_acc, r.quant_acc, best_i8
        ));
    }
    std::fs::write(tsv_path, &tsv).expect("write TSV");
    println!("\nTSV written to: {}", tsv_path);

    // ============================================================
    // Summary table: average accuracy per method per task
    // ============================================================

    println!("\n=== SUMMARY: Average quant_acc (%) per method per task ===\n");
    let methods = ["A_std_backprop", "B_greedy_only", "C_greedy+bp_all", "D_greedy+bp_last"];
    let task_names = ["XOR", "ADD4", "MUL4"];

    println!(
        "{:<22} {:>8} {:>8} {:>8} {:>8}",
        "method", "XOR", "ADD4", "MUL4", "AVG"
    );
    println!("{}", "-".repeat(58));

    for m in &methods {
        let mut row_vals = Vec::new();
        for tn in &task_names {
            let vals: Vec<f32> = all
                .iter()
                .filter(|r| r.method == *m && r.task == *tn)
                .map(|r| r.quant_acc)
                .collect();
            let avg = if vals.is_empty() {
                0.0
            } else {
                vals.iter().sum::<f32>() / vals.len() as f32
            };
            row_vals.push(avg);
        }
        let overall = if row_vals.is_empty() {
            0.0
        } else {
            row_vals.iter().sum::<f32>() / row_vals.len() as f32
        };
        println!(
            "{:<22} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}%",
            m,
            row_vals[0] * 100.0,
            row_vals[1] * 100.0,
            row_vals[2] * 100.0,
            overall * 100.0
        );
    }

    // Also show float_acc for methods A, C, D
    println!("\n=== SUMMARY: Average float_acc (%) per method per task ===\n");
    println!(
        "{:<22} {:>8} {:>8} {:>8} {:>8}",
        "method", "XOR", "ADD4", "MUL4", "AVG"
    );
    println!("{}", "-".repeat(58));

    for m in &methods {
        let mut row_vals = Vec::new();
        for tn in &task_names {
            let vals: Vec<f32> = all
                .iter()
                .filter(|r| r.method == *m && r.task == *tn)
                .map(|r| r.float_acc)
                .collect();
            let avg = if vals.is_empty() {
                0.0
            } else {
                vals.iter().sum::<f32>() / vals.len() as f32
            };
            row_vals.push(avg);
        }
        let overall = if row_vals.is_empty() {
            0.0
        } else {
            row_vals.iter().sum::<f32>() / row_vals.len() as f32
        };
        println!(
            "{:<22} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}%",
            m,
            row_vals[0] * 100.0,
            row_vals[1] * 100.0,
            row_vals[2] * 100.0,
            overall * 100.0
        );
    }

    // Best config per task
    println!("\n=== BEST CONFIG per task (by quant_acc) ===\n");
    for tn in &task_names {
        let best = all
            .iter()
            .filter(|r| r.task == *tn)
            .max_by(|a, b| a.quant_acc.partial_cmp(&b.quant_acc).unwrap());
        if let Some(b) = best {
            println!(
                "{}: method={} act={} H={} depth={} seed={} -> quant_acc={:.1}% float_acc={:.1}%",
                tn, b.method, b.act_s, b.h, b.depth, b.seed,
                b.quant_acc * 100.0, b.float_acc * 100.0
            );
        }
    }

    let elapsed = t0.elapsed().as_secs_f32();
    println!("\nTotal time: {:.1}s", elapsed);
    println!("\n=== EXPERIMENT COMPLETE ===");
}
