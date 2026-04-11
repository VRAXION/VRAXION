//! VRAXION Capability Playground — Curriculum Dashboard
//! Trains progressively harder tasks, measures neuron budget + quantization
//! Tests C19 vs ReLU side by side, float vs int8
//!
//! Run: cargo run --example capability_playground --release
//!
//! Add new tasks by implementing the Task trait and adding to CURRICULUM.

use std::time::Instant;

// ── Activations ──

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

fn c19_deriv(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l || x <= -l { return 1.0; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    let dt = 1.0 - 2.0 * t;
    dt * (sgn + 2.0 * rho * h)
}

fn relu(x: f32) -> f32 { x.max(0.0) }
fn relu_deriv(x: f32) -> f32 { if x > 0.0 { 1.0 } else { 0.0 } }

const RHO: f32 = 8.0;

// ── PRNG ──

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.state }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }
    fn usize(&mut self, n: usize) -> usize { (self.next() as usize) % n }
    fn shuffle<T>(&mut self, v: &mut [T]) { for i in (1..v.len()).rev() { let j = self.usize(i + 1); v.swap(i, j); } }
}

// ── Task definition ──

struct TaskData {
    train: Vec<(Vec<f32>, Vec<f32>)>,  // (input, target) — target is float
    test: Vec<(Vec<f32>, Vec<f32>)>,
    in_dim: usize,
    out_dim: usize,
    is_classification: bool,  // if true, argmax accuracy; if false, MSE/PSNR
}

struct TaskSpec {
    name: &'static str,
    id: &'static str,
    difficulty: u8,      // 1-10
    min_hidden: usize,   // suggested starting hidden dim
    max_hidden: usize,   // max to try
    generate: fn(&mut Rng) -> TaskData,
}

// ── MLP (same as backprop upscaler but generalized) ──

struct Mlp {
    layers: Vec<usize>,
    weights: Vec<Vec<f32>>,
    biases: Vec<Vec<f32>>,
    use_c19: bool,
}

impl Mlp {
    fn new(layers: &[usize], use_c19: bool, rng: &mut Rng) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for l in 0..layers.len()-1 {
            let fan_in = layers[l];
            let fan_out = layers[l+1];
            let s = (2.0 / fan_in as f32).sqrt() * if use_c19 { 0.3 } else { 1.0 };
            weights.push((0..fan_out * fan_in).map(|_| rng.range_f32(-s, s)).collect());
            biases.push(vec![0.0; fan_out]);
        }
        Mlp { layers: layers.to_vec(), weights, biases, use_c19 }
    }

    fn act(&self, x: f32) -> f32 { if self.use_c19 { c19(x, RHO) } else { relu(x) } }
    fn act_d(&self, x: f32) -> f32 { if self.use_c19 { c19_deriv(x, RHO) } else { relu_deriv(x) } }

    fn forward(&self, input: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let n_layers = self.layers.len() - 1;
        let mut pres = Vec::with_capacity(n_layers);
        let mut acts = Vec::with_capacity(n_layers + 1);
        acts.push(input.to_vec());
        for l in 0..n_layers {
            let inp = &acts[l];
            let out_dim = self.layers[l+1];
            let in_dim = self.layers[l];
            let mut pre = vec![0.0f32; out_dim];
            for j in 0..out_dim {
                let mut s = self.biases[l][j];
                for i in 0..in_dim { s += self.weights[l][j * in_dim + i] * inp[i]; }
                pre[j] = s;
            }
            pres.push(pre.clone());
            // Last layer: sigmoid for classification, linear for regression
            let act: Vec<f32> = if l == n_layers - 1 {
                pre  // linear output — caller applies sigmoid/argmax if needed
            } else {
                pre.iter().map(|&x| self.act(x)).collect()
            };
            acts.push(act);
        }
        (pres, acts)
    }

    fn predict(&self, input: &[f32]) -> Vec<f32> {
        let (_, acts) = self.forward(input);
        acts.last().unwrap().clone()
    }

    fn backward(&self, pres: &[Vec<f32>], acts: &[Vec<f32>], target: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let n_layers = self.layers.len() - 1;
        let out_dim = self.layers[n_layers];
        let mut dw: Vec<Vec<f32>> = self.weights.iter().map(|w| vec![0.0; w.len()]).collect();
        let mut db: Vec<Vec<f32>> = self.biases.iter().map(|b| vec![0.0; b.len()]).collect();
        let out = &acts[n_layers];
        let mut delta: Vec<f32> = out.iter().zip(target).map(|(&o, &t)| 2.0 * (o - t) / out_dim as f32).collect();
        for l in (0..n_layers).rev() {
            let inp = &acts[l];
            let in_dim = self.layers[l];
            let o_dim = self.layers[l+1];
            if l < n_layers - 1 {
                for j in 0..o_dim { delta[j] *= self.act_d(pres[l][j]); }
            }
            for j in 0..o_dim {
                for i in 0..in_dim { dw[l][j * in_dim + i] += delta[j] * inp[i]; }
                db[l][j] += delta[j];
            }
            if l > 0 {
                let prev_dim = in_dim;
                let mut new_delta = vec![0.0f32; prev_dim];
                for i in 0..prev_dim {
                    for j in 0..o_dim { new_delta[i] += self.weights[l][j * in_dim + i] * delta[j]; }
                }
                delta = new_delta;
            }
        }
        (dw, db)
    }

    fn train(&mut self, data: &[(Vec<f32>, Vec<f32>)], lr: f32, epochs: usize, rng: &mut Rng, verbose: bool) {
        let mut idx: Vec<usize> = (0..data.len()).collect();
        let clip = 1.0f32;
        for ep in 0..epochs {
            let lr_e = if ep < 20 { lr * (ep as f32 + 1.0) / 20.0 } else { lr };
            rng.shuffle(&mut idx);
            for &i in &idx {
                let (x, y) = &data[i];
                let (pres, acts) = self.forward(x);
                let (dw, db) = self.backward(&pres, &acts, y);
                for l in 0..self.weights.len() {
                    for k in 0..self.weights[l].len() { self.weights[l][k] -= (lr_e * dw[l][k]).clamp(-clip, clip); }
                    for k in 0..self.biases[l].len() { self.biases[l][k] -= (lr_e * db[l][k]).clamp(-clip, clip); }
                }
            }
            if verbose && (ep % 200 == 0 || ep == epochs - 1) {
                let mut mse = 0.0f32;
                for (x, y) in data {
                    let pred = self.predict(x);
                    mse += pred.iter().zip(y).map(|(p, t)| (p - t) * (p - t)).sum::<f32>() / y.len() as f32;
                }
                mse /= data.len() as f32;
                println!("      ep {:4}: MSE={:.6}", ep, mse);
            }
        }
    }

    // Quantize weights to int8 (per-layer scale)
    fn quantize_int8(&self) -> Mlp {
        let mut qw = Vec::new();
        let mut qb = Vec::new();
        for l in 0..self.weights.len() {
            let w_max = self.weights[l].iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-8);
            let scale = 127.0 / w_max;
            qw.push(self.weights[l].iter().map(|&w| (w * scale).round() / scale).collect());
            let b_max = self.biases[l].iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-8);
            let b_scale = 127.0 / b_max;
            qb.push(self.biases[l].iter().map(|&b| (b * b_scale).round() / b_scale).collect());
        }
        Mlp { layers: self.layers.clone(), weights: qw, biases: qb, use_c19: self.use_c19 }
    }

    fn weight_count(&self) -> usize {
        self.weights.iter().zip(&self.biases).map(|(w,b)| w.len() + b.len()).sum()
    }
}

// ── Evaluation ──

fn eval_classification(net: &Mlp, data: &[(Vec<f32>, Vec<f32>)]) -> f32 {
    let mut correct = 0;
    for (x, y) in data {
        let pred = net.predict(x);
        if pred.iter().any(|v| v.is_nan()) { continue; }
        if y.len() == 1 {
            // Binary: threshold at 0.5
            let p = if pred[0] > 0.5 { 1.0 } else { 0.0 };
            if (p - y[0]).abs() < 0.01 { correct += 1; }
        } else {
            // Multi-bit: round each output to 0/1 and compare
            let mut all_match = true;
            for (p_val, t_val) in pred.iter().zip(y.iter()) {
                let pb = if *p_val > 0.5 { 1 } else { 0 };
                let tb = if *t_val > 0.5 { 1 } else { 0 };
                if pb != tb { all_match = false; break; }
            }
            if all_match { correct += 1; }
        }
    }
    correct as f32 / data.len() as f32 * 100.0
}

fn eval_regression_psnr(net: &Mlp, data: &[(Vec<f32>, Vec<f32>)]) -> f32 {
    let mut mse = 0.0f32;
    let mut n = 0;
    for (x, y) in data {
        let pred = net.predict(x);
        if pred.iter().any(|v| v.is_nan()) { continue; }
        mse += pred.iter().zip(y).map(|(p, t)| (p - t) * (p - t)).sum::<f32>() / y.len() as f32;
        n += 1;
    }
    if n == 0 { return 0.0; }
    mse /= n as f32;
    if mse > 1e-10 { -10.0 * mse.log10() } else { 99.0 }
}

// ── Task generators ──

fn bits_of(val: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| if val & (1 << i) != 0 { 1.0 } else { 0.0 }).collect()
}

fn one_hot(val: usize, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    if val < n { v[val] = 1.0; }
    v
}

// T01: AND gate (2→1)
fn gen_and(_rng: &mut Rng) -> TaskData {
    let mut all = Vec::new();
    for a in 0..2 { for b in 0..2 {
        all.push((vec![a as f32, b as f32], vec![if a==1 && b==1 { 1.0 } else { 0.0 }]));
    }}
    TaskData { train: all.clone(), test: all, in_dim: 2, out_dim: 1, is_classification: true }
}

// T02: XOR gate (2→1)
fn gen_xor(_rng: &mut Rng) -> TaskData {
    let mut all = Vec::new();
    for a in 0..2 { for b in 0..2 {
        all.push((vec![a as f32, b as f32], vec![if a != b { 1.0 } else { 0.0 }]));
    }}
    TaskData { train: all.clone(), test: all, in_dim: 2, out_dim: 1, is_classification: true }
}

// T03: 3-bit majority (3→1)
fn gen_maj3(_rng: &mut Rng) -> TaskData {
    let mut all = Vec::new();
    for v in 0..8u8 {
        let bits = bits_of(v as usize, 3);
        let sum: f32 = bits.iter().sum();
        all.push((bits, vec![if sum >= 2.0 { 1.0 } else { 0.0 }]));
    }
    TaskData { train: all.clone(), test: all, in_dim: 3, out_dim: 1, is_classification: true }
}

// T04: 4-bit popcount >2 (4→1)
fn gen_pop4(_rng: &mut Rng) -> TaskData {
    let mut all = Vec::new();
    for v in 0..16u8 {
        let bits = bits_of(v as usize, 4);
        let sum: f32 = bits.iter().sum();
        all.push((bits, vec![if sum > 2.0 { 1.0 } else { 0.0 }]));
    }
    TaskData { train: all.clone(), test: all, in_dim: 4, out_dim: 1, is_classification: true }
}

// T05: 8-bit popcount >4 (8→1)
fn gen_pop8(_rng: &mut Rng) -> TaskData {
    let mut all = Vec::new();
    for v in 0..256u16 {
        let bits = bits_of(v as usize, 8);
        let sum: f32 = bits.iter().sum();
        all.push((bits, vec![if sum > 4.0 { 1.0 } else { 0.0 }]));
    }
    TaskData { train: all.clone(), test: all, in_dim: 8, out_dim: 1, is_classification: true }
}

// T06: 4-bit parity (4→1)
fn gen_par4(_rng: &mut Rng) -> TaskData {
    let mut all = Vec::new();
    for v in 0..16u8 {
        let bits = bits_of(v as usize, 4);
        let sum: f32 = bits.iter().sum();
        all.push((bits, vec![if (sum as u8) % 2 == 1 { 1.0 } else { 0.0 }]));
    }
    TaskData { train: all.clone(), test: all, in_dim: 4, out_dim: 1, is_classification: true }
}

// T07: 8-bit parity (8→1)
fn gen_par8(_rng: &mut Rng) -> TaskData {
    let mut all = Vec::new();
    for v in 0..256u16 {
        let bits = bits_of(v as usize, 8);
        let sum: f32 = bits.iter().sum();
        all.push((bits, vec![if (sum as u8) % 2 == 1 { 1.0 } else { 0.0 }]));
    }
    TaskData { train: all.clone(), test: all, in_dim: 8, out_dim: 1, is_classification: true }
}

// T08: Nibble classifier (8→4 one-hot: which nibble is bigger)
fn gen_nibble(_rng: &mut Rng) -> TaskData {
    let mut all = Vec::new();
    for v in 0..256u16 {
        let bits = bits_of(v as usize, 8);
        let lo = (v & 0x0F) as usize;
        let hi = ((v >> 4) & 0x0F) as usize;
        let cls = if lo > hi { 0 } else if lo < hi { 1 } else if lo < 8 { 2 } else { 3 };
        all.push((bits, one_hot(cls, 4)));
    }
    TaskData { train: all.clone(), test: all, in_dim: 8, out_dim: 4, is_classification: true }
}

// T09: 4-bit addition (8→5, a+b encoded as 5 output bits)
fn gen_add4(_rng: &mut Rng) -> TaskData {
    let mut all = Vec::new();
    for a in 0..16u8 { for b in 0..16u8 {
        let mut inp = bits_of(a as usize, 4);
        inp.extend(bits_of(b as usize, 4));
        let sum = (a + b) as usize;
        all.push((inp, bits_of(sum, 5)));
    }}
    // Split: train on 80%, test on 20%
    let n = all.len();
    let split = n * 4 / 5;
    let test = all.split_off(split);
    TaskData { train: all, test, in_dim: 8, out_dim: 5, is_classification: true }
}

// T10: 4-bit multiply (8→8, a*b encoded as 8 output bits)
fn gen_mul4(_rng: &mut Rng) -> TaskData {
    let mut all = Vec::new();
    for a in 0..16u8 { for b in 0..16u8 {
        let mut inp = bits_of(a as usize, 4);
        inp.extend(bits_of(b as usize, 4));
        let prod = (a as usize) * (b as usize);
        all.push((inp, bits_of(prod, 8)));
    }}
    let n = all.len();
    let split = n * 4 / 5;
    let test = all.split_off(split);
    TaskData { train: all, test, in_dim: 8, out_dim: 8, is_classification: true }
}

// T11: Simple 1D regression (1→1, learn sin(x))
fn gen_sin(rng: &mut Rng) -> TaskData {
    let mut train = Vec::new();
    let mut test = Vec::new();
    for i in 0..200 {
        let x = i as f32 / 200.0 * 6.28;
        let y = x.sin() * 0.5;  // scale to [-0.5, 0.5]
        if i % 5 == 0 { test.push((vec![x / 6.28], vec![y])); }
        else { train.push((vec![x / 6.28], vec![y])); }
    }
    // Add some random samples
    for _ in 0..100 {
        let x = rng.f32() * 6.28;
        let y = x.sin() * 0.5;
        train.push((vec![x / 6.28], vec![y]));
    }
    TaskData { train, test, in_dim: 1, out_dim: 1, is_classification: false }
}

// T12: 2D regression (2→1, learn distance from center)
fn gen_dist(rng: &mut Rng) -> TaskData {
    let mut train = Vec::new();
    let mut test = Vec::new();
    for i in 0..500 {
        let x = rng.f32() * 2.0 - 1.0;
        let y = rng.f32() * 2.0 - 1.0;
        let d = (x*x + y*y).sqrt() / 1.414;  // normalized to [0, 1]
        if i % 5 == 0 { test.push((vec![x, y], vec![d])); }
        else { train.push((vec![x, y], vec![d])); }
    }
    TaskData { train, test, in_dim: 2, out_dim: 1, is_classification: false }
}

// T13: Pixel tile upscale (4→16, learn 2× upscale of 2×2→4×4)
fn gen_tile_up(rng: &mut Rng) -> TaskData {
    let mut train = Vec::new();
    let mut test = Vec::new();
    for i in 0..500 {
        // Random 4×4 HR tile
        let hr: Vec<f32> = (0..16).map(|_| rng.f32()).collect();
        // Downsample to 2×2
        let lr = vec![
            (hr[0]+hr[1]+hr[4]+hr[5])/4.0,
            (hr[2]+hr[3]+hr[6]+hr[7])/4.0,
            (hr[8]+hr[9]+hr[12]+hr[13])/4.0,
            (hr[10]+hr[11]+hr[14]+hr[15])/4.0,
        ];
        let target: Vec<f32> = hr.iter().map(|&x| x - 0.5).collect();
        let input: Vec<f32> = lr.iter().map(|&x| x - 0.5).collect();
        if i % 5 == 0 { test.push((input, target)); }
        else { train.push((input, target)); }
    }
    TaskData { train, test, in_dim: 4, out_dim: 16, is_classification: false }
}

// T14: Anomaly detection (8→1, normal=low variance, anomaly=spike)
fn gen_anomaly(rng: &mut Rng) -> TaskData {
    let mut all = Vec::new();
    for _ in 0..200 {
        // Normal: all values near 0.5
        let inp: Vec<f32> = (0..8).map(|_| 0.5 + rng.range_f32(-0.1, 0.1)).collect();
        all.push((inp, vec![0.0]));
    }
    for _ in 0..200 {
        // Anomaly: one value spiked
        let mut inp: Vec<f32> = (0..8).map(|_| 0.5 + rng.range_f32(-0.1, 0.1)).collect();
        let spike = rng.usize(8);
        inp[spike] = if rng.f32() > 0.5 { 0.95 } else { 0.05 };
        all.push((inp, vec![1.0]));
    }
    let split = all.len() * 4 / 5;
    let test = all.split_off(split);
    TaskData { train: all, test, in_dim: 8, out_dim: 1, is_classification: true }
}

// ── Curriculum ──

fn curriculum() -> Vec<TaskSpec> {
    vec![
        TaskSpec { name: "AND gate",       id: "T01", difficulty: 1, min_hidden: 2,  max_hidden: 8,    generate: gen_and },
        TaskSpec { name: "XOR gate",       id: "T02", difficulty: 2, min_hidden: 2,  max_hidden: 16,   generate: gen_xor },
        TaskSpec { name: "3-bit majority", id: "T03", difficulty: 2, min_hidden: 4,  max_hidden: 16,   generate: gen_maj3 },
        TaskSpec { name: "4-bit pop>2",    id: "T04", difficulty: 3, min_hidden: 4,  max_hidden: 32,   generate: gen_pop4 },
        TaskSpec { name: "4-bit parity",   id: "T05", difficulty: 4, min_hidden: 4,  max_hidden: 32,   generate: gen_par4 },
        TaskSpec { name: "8-bit pop>4",    id: "T06", difficulty: 5, min_hidden: 8,  max_hidden: 64,   generate: gen_pop8 },
        TaskSpec { name: "8-bit parity",   id: "T07", difficulty: 6, min_hidden: 8,  max_hidden: 64,   generate: gen_par8 },
        TaskSpec { name: "Nibble class",   id: "T08", difficulty: 5, min_hidden: 8,  max_hidden: 64,   generate: gen_nibble },
        TaskSpec { name: "4-bit add",      id: "T09", difficulty: 7, min_hidden: 16, max_hidden: 128,  generate: gen_add4 },
        TaskSpec { name: "4-bit multiply", id: "T10", difficulty: 9, min_hidden: 32, max_hidden: 256,  generate: gen_mul4 },
        TaskSpec { name: "sin(x) regr",    id: "T11", difficulty: 3, min_hidden: 8,  max_hidden: 64,   generate: gen_sin },
        TaskSpec { name: "dist regr",      id: "T12", difficulty: 4, min_hidden: 8,  max_hidden: 64,   generate: gen_dist },
        TaskSpec { name: "Tile 2x up",     id: "T13", difficulty: 8, min_hidden: 32, max_hidden: 128,  generate: gen_tile_up },
        TaskSpec { name: "Anomaly det",    id: "T14", difficulty: 4, min_hidden: 8,  max_hidden: 32,   generate: gen_anomaly },
    ]
}

// ── Run one task with one config ──

struct RunResult {
    hidden: usize,
    act_name: &'static str,
    train_score: f32,
    test_score: f32,
    q8_test_score: f32,
    train_time_ms: u64,
}

fn run_task(spec: &TaskSpec, hidden: usize, use_c19: bool, seed: u64) -> RunResult {
    let mut rng = Rng::new(seed);
    let data = (spec.generate)(&mut rng);
    let act_name = if use_c19 { "C19" } else { "ReLU" };

    let layers = vec![data.in_dim, hidden, data.out_dim];
    let mut rng2 = Rng::new(seed + 1000);
    let mut net = Mlp::new(&layers, use_c19, &mut rng2);

    let lr = if use_c19 { 0.002 } else { 0.005 };
    let epochs = if data.train.len() <= 16 { 3000 } else if data.train.len() <= 256 { 1500 } else { 800 };

    let t0 = Instant::now();
    net.train(&data.train, lr, epochs, &mut rng2, false);
    let train_time = t0.elapsed().as_millis() as u64;

    let q8 = net.quantize_int8();

    let (train_score, test_score, q8_test) = if data.is_classification {
        (eval_classification(&net, &data.train),
         eval_classification(&net, &data.test),
         eval_classification(&q8, &data.test))
    } else {
        (eval_regression_psnr(&net, &data.train),
         eval_regression_psnr(&net, &data.test),
         eval_regression_psnr(&q8, &data.test))
    };

    RunResult { hidden, act_name, train_score, test_score, q8_test_score: q8_test, train_time_ms: train_time }
}

// ── Find minimum hidden neurons for a task ──

fn find_min_neurons(spec: &TaskSpec, use_c19: bool, target: f32) -> (usize, RunResult) {
    let mut h = spec.min_hidden;
    let seeds = [42, 123, 7, 999, 314];
    loop {
        let mut best: Option<RunResult> = None;
        for &seed in &seeds {
            let res = run_task(spec, h, use_c19, seed);
            if res.test_score >= target {
                return (h, res);
            }
            if best.is_none() || res.test_score > best.as_ref().unwrap().test_score {
                best = Some(res);
            }
        }
        if h >= spec.max_hidden {
            return (h, best.unwrap());
        }
        h = (h * 3 / 2).max(h + 2).min(spec.max_hidden);
    }
}

fn main() {
    let total_t0 = Instant::now();

    println!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  VRAXION CAPABILITY PLAYGROUND — Curriculum Dashboard                          ║");
    println!("║  Backprop MLP, 1 hidden layer, C19 vs ReLU, float vs int8                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  ID  │ Task             │ Diff │ H(R) │ H(C) │ ReLU  │ C19   │ ReLU q8│ C19 q8║");
    println!("╠══════╪══════════════════╪══════╪══════╪══════╪═══════╪═══════╪════════╪═══════╣");

    let tasks = curriculum();
    let mut pass_relu = 0;
    let mut pass_c19 = 0;
    let mut pass_q8r = 0;
    let mut pass_q8c = 0;

    for spec in &tasks {
        let target = if spec.is_classification() { 95.0 } else { 15.0 };

        let (h_relu, res_relu) = find_min_neurons(spec, false, target);
        let (h_c19, res_c19) = find_min_neurons(spec, true, target);

        let unit = if spec.is_classification() { "%" } else { "dB" };

        let r_pass = res_relu.test_score >= target;
        let c_pass = res_c19.test_score >= target;
        let rq_pass = res_relu.q8_test_score >= target;
        let cq_pass = res_c19.q8_test_score >= target;
        if r_pass { pass_relu += 1; }
        if c_pass { pass_c19 += 1; }
        if rq_pass { pass_q8r += 1; }
        if cq_pass { pass_q8c += 1; }

        let mark = |pass: bool| if pass { " ✓" } else { " ✗" };

        println!("║ {} │ {:16} │  {:2}  │ {:4} │ {:4} │{:5.1}{}{} │{:5.1}{}{} │{:5.1}{}{} │{:5.1}{}{}║",
            spec.id, spec.name, spec.difficulty,
            h_relu, h_c19,
            res_relu.test_score, unit, mark(r_pass),
            res_c19.test_score, unit, mark(c_pass),
            res_relu.q8_test_score, unit, mark(rq_pass),
            res_c19.q8_test_score, unit, mark(cq_pass),
        );
    }

    let n = tasks.len();
    println!("╠══════╧══════════════════╧══════╧══════╧══════╧═══════╧═══════╧════════╧═══════╣");
    println!("║  PASS: ReLU {}/{}, C19 {}/{}, ReLU q8 {}/{}, C19 q8 {}/{}                        ║",
        pass_relu, n, pass_c19, n, pass_q8r, n, pass_q8c, n);
    println!("║  Total time: {:.1}s                                                             ║",
        total_t0.elapsed().as_secs_f64());
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝");
}

// Helper for TaskSpec (can't add methods in the vec! macro context)
impl TaskSpec {
    fn is_classification(&self) -> bool {
        let mut rng = Rng::new(0);
        (self.generate)(&mut rng).is_classification
    }
}
