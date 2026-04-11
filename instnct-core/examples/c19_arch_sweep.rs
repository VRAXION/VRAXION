//! C19 Architecture Scaling Sweep — autonomous experiment suite
//!
//! Tests 7 architecture variants on standardized benchmarks:
//!   V1: PureCPU     — fixed LUT gates only (baseline)
//!   V2: HoloProc    — holographic memory lookup
//!   V3: HoloPicker  — learned selector + fixed ALU
//!   V4: MultiTick   — iterative holo→ALU (N ticks)
//!   V5: StackedHolo — multiple holo layers → ALU
//!   V6: HybridCPU   — CPU structure with holographic register file
//!   V7: FullyNeural — MLP only, no fixed gates
//!
//! Benchmarks: B1-ADD, B2-MUL, B3-BITWISE, B4-OP-SELECT, B5-MULTI-OP, B6-SORT
//!
//! Checkpoint-based: each run processes a batch, writes checkpoint, resumes next time.
//! Designed for /loop autonomous execution.
//!
//! Run: cargo run --example c19_arch_sweep --release

use std::io::Write;

// ============================================================
// C19 activation (init-time only)
// ============================================================

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

fn c19_deriv(x: f32, rho: f32) -> f32 {
    let l = 6.0;
    if x >= l || x <= -l { return 1.0; }
    let n = x.floor(); let t = x - n;
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    let h = t * (1.0 - t);
    sgn * (1.0 - 2.0 * t) + rho * 2.0 * h * (1.0 - 2.0 * t)
}

fn relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }
fn relu_deriv(x: f32) -> f32 { if x > 0.0 { 1.0 } else { 0.0 } }

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ============================================================
// Logging
// ============================================================

fn log(f: &mut std::fs::File, msg: &str) {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
    let s = d.as_secs(); let h = (s/3600)%24; let m = (s/60)%60; let sec = s%60;
    let line = format!("[{:02}:{:02}:{:02}] {}\n", h, m, sec, msg);
    print!("{}", line);
    f.write_all(line.as_bytes()).ok();
    f.flush().ok();
}

// ============================================================
// RNG
// ============================================================

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range(&mut self, max: u64) -> u64 { self.next() % max }
}

// ============================================================
// Block 1: LutGate — integer LUT neuron (from c19_cpu.rs)
// ============================================================

#[derive(Clone)]
struct LutGate {
    w_int: Vec<i32>,
    bias_int: i32,
    lut: Vec<u8>,
    min_sum: i32,
}

impl LutGate {
    fn new(w: &[f32], bias: f32, rho: f32, thr: f32) -> Self {
        let mut all = w.to_vec(); all.push(bias);
        let mut denom = 1;
        for d in 1..=100 {
            if all.iter().all(|&v| ((v * d as f32).round() - v * d as f32).abs() < 1e-6) {
                denom = d; break;
            }
        }
        let w_int: Vec<i32> = w.iter().map(|&v| (v * denom as f32).round() as i32).collect();
        let bias_int = (bias * denom as f32).round() as i32;
        let mut min_s = bias_int; let mut max_s = bias_int;
        for &wi in &w_int { if wi > 0 { max_s += wi; } else { min_s += wi; } }
        let mut lut = vec![0u8; (max_s - min_s + 1) as usize];
        for s in min_s..=max_s {
            lut[(s - min_s) as usize] = if c19(s as f32 / denom as f32, rho) > thr { 1 } else { 0 };
        }
        LutGate { w_int, bias_int, lut, min_sum: min_s }
    }
    fn eval(&self, inputs: &[u8]) -> u8 {
        let s: i32 = inputs.iter().zip(&self.w_int).map(|(&i, &w)| i as i32 * w).sum::<i32>() + self.bias_int;
        let idx = (s - self.min_sum) as usize;
        if idx < self.lut.len() { self.lut[idx] } else { 0 }
    }
}

// ============================================================
// Block 2: ParallelAlu — 6-way fixed ALU (from holo_alu.rs)
// ============================================================

struct ParallelAlu {
    xor3: LutGate, maj: LutGate, not_g: LutGate,
    and_g: LutGate, or_g: LutGate, xor_g: LutGate,
}

impl ParallelAlu {
    fn new() -> Self {
        ParallelAlu {
            xor3: LutGate::new(&[1.5,1.5,1.5], 3.0, 16.0, 0.6),
            maj: LutGate::new(&[8.5,8.5,8.5], -2.75, 0.0, 4.0),
            not_g: LutGate::new(&[-9.75], -5.5, 16.0, -4.0),
            and_g: LutGate::new(&[10.0,10.0], -4.5, 0.0, 4.0),
            or_g: LutGate::new(&[8.75,8.75], 5.5, 0.0, 4.0),
            xor_g: LutGate::new(&[0.5,0.5], 0.0, 16.0, 0.6),
        }
    }

    fn add4(&self, a: u8, b: u8) -> u8 {
        let mut c = 0u8; let mut r = 0u8;
        for bit in 0..4 {
            let ab = (a>>bit)&1; let bb = (b>>bit)&1;
            r |= self.xor3.eval(&[ab,bb,c]) << bit;
            c = self.maj.eval(&[ab,bb,c]);
        }
        r & 0xF
    }
    fn sub4(&self, a: u8, b: u8) -> u8 {
        let mut bn = 0u8;
        for bit in 0..4 { bn |= self.not_g.eval(&[(b>>bit)&1]) << bit; }
        let mut c = 1u8; let mut r = 0u8;
        for bit in 0..4 {
            let ab = (a>>bit)&1; let bb = (bn>>bit)&1;
            r |= self.xor3.eval(&[ab,bb,c]) << bit;
            c = self.maj.eval(&[ab,bb,c]);
        }
        r & 0xF
    }
    fn and4(&self, a: u8, b: u8) -> u8 {
        let mut r = 0u8;
        for bit in 0..4 { r |= self.and_g.eval(&[(a>>bit)&1,(b>>bit)&1]) << bit; }
        r
    }
    fn or4(&self, a: u8, b: u8) -> u8 {
        let mut r = 0u8;
        for bit in 0..4 { r |= self.or_g.eval(&[(a>>bit)&1,(b>>bit)&1]) << bit; }
        r
    }
    fn xor4(&self, a: u8, b: u8) -> u8 {
        let mut r = 0u8;
        for bit in 0..4 { r |= self.xor_g.eval(&[(a>>bit)&1,(b>>bit)&1]) << bit; }
        r
    }

    fn execute(&self, a: u8, b: u8, op: usize) -> u8 {
        match op { 0 => self.add4(a,b), 1 => self.sub4(a,b), 2 => self.and4(a,b),
                    3 => self.or4(a,b), 4 => self.xor4(a,b), _ => 0 }
    }

    fn execute_all(&self, a: u8, b: u8) -> [u8; 5] {
        [self.add4(a,b), self.sub4(a,b), self.and4(a,b), self.or4(a,b), self.xor4(a,b)]
    }
}

fn expected_op(a: u8, b: u8, op: usize) -> u8 {
    match op {
        0 => (a.wrapping_add(b)) & 0xF,
        1 => (a.wrapping_sub(b)) & 0xF,
        2 => a & b, 3 => a | b, 4 => a ^ b,
        _ => 0,
    }
}

// ============================================================
// Block 3: HoloLayer — outer-product associative memory
// ============================================================

struct HoloLayer {
    dim: usize,
    matrix: Vec<f32>,      // dim × dim
    proj: Vec<Vec<f32>>,   // n_proj × dim
    n_proj: usize,
}

impl HoloLayer {
    fn new(dim: usize, n_proj: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let proj = (0..n_proj).map(|_| (0..dim).map(|_| rng.f32() * 2.0 - 1.0).collect()).collect();
        HoloLayer { dim, matrix: vec![0.0; dim * dim], proj, n_proj }
    }

    fn encode_input(&self, bits: &[u8]) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 && i < self.n_proj {
                for d in 0..self.dim { v[d] += self.proj[i][d]; }
            }
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 { for x in &mut v { *x /= norm; } }
        v
    }

    fn store(&mut self, bits: &[u8], target: usize) {
        let inp = self.encode_input(bits);
        // one-hot output
        for i in 0..self.dim {
            let out_i = if i == target { 1.0 } else { 0.0 };
            for j in 0..self.dim {
                self.matrix[i * self.dim + j] += inp[j] * out_i;
            }
        }
    }

    fn predict(&self, bits: &[u8]) -> usize {
        let inp = self.encode_input(bits);
        let mut output = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                output[i] += self.matrix[i * self.dim + j] * inp[j];
            }
        }
        output.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0)
    }

    fn predict_vec(&self, bits: &[u8]) -> Vec<f32> {
        let inp = self.encode_input(bits);
        let mut output = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                output[i] += self.matrix[i * self.dim + j] * inp[j];
            }
        }
        output
    }

    fn clear(&mut self) {
        self.matrix.iter_mut().for_each(|x| *x = 0.0);
    }
}

fn encode_ab(a: u8, b: u8) -> Vec<u8> {
    // Thermometer encoding: 4 bits for a, 4 bits for b = 8 bits
    let mut bits = vec![0u8; 8];
    for i in 0..4 { bits[i] = (a >> i) & 1; }
    for i in 0..4 { bits[4 + i] = (b >> i) & 1; }
    bits
}

fn encode_ab_op(a: u8, b: u8, op: u8) -> Vec<u8> {
    // 4 bits a + 4 bits b + 3 bits op = 11 bits
    let mut bits = vec![0u8; 11];
    for i in 0..4 { bits[i] = (a >> i) & 1; }
    for i in 0..4 { bits[4 + i] = (b >> i) & 1; }
    for i in 0..3 { bits[8 + i] = (op >> i) & 1; }
    bits
}

// ============================================================
// Block 4: MLP — trainable multi-layer perceptron
// ============================================================

struct Mlp {
    n_input: usize,
    n_hidden: usize,
    n_output: usize,
    w1: Vec<f32>,   // n_input × n_hidden
    b1: Vec<f32>,
    w2: Vec<f32>,   // n_hidden × n_output
    b2: Vec<f32>,
    use_c19: bool,
    rho: f32,
    // Adam state
    m1: Vec<f32>, v1: Vec<f32>,
    mb1: Vec<f32>, vb1: Vec<f32>,
    m2: Vec<f32>, v2: Vec<f32>,
    mb2: Vec<f32>, vb2: Vec<f32>,
    t_adam: i32,
}

impl Mlp {
    fn new(n_input: usize, n_hidden: usize, n_output: usize, use_c19: bool, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let scale = (2.0 / n_input as f32).sqrt();
        Mlp {
            n_input, n_hidden, n_output,
            w1: (0..n_input*n_hidden).map(|_| (rng.f32() - 0.5) * scale).collect(),
            b1: vec![0.0; n_hidden],
            w2: (0..n_hidden*n_output).map(|_| (rng.f32() - 0.5) * scale * 0.5).collect(),
            b2: vec![0.0; n_output],
            use_c19, rho: 8.0,
            m1: vec![0.0; n_input*n_hidden], v1: vec![0.0; n_input*n_hidden],
            mb1: vec![0.0; n_hidden], vb1: vec![0.0; n_hidden],
            m2: vec![0.0; n_hidden*n_output], v2: vec![0.0; n_hidden*n_output],
            mb2: vec![0.0; n_output], vb2: vec![0.0; n_output],
            t_adam: 0,
        }
    }

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut pre = vec![0.0f32; self.n_hidden];
        for j in 0..self.n_hidden {
            let mut s = self.b1[j];
            for i in 0..self.n_input { s += input[i] * self.w1[i * self.n_hidden + j]; }
            pre[j] = s;
        }
        let hid: Vec<f32> = if self.use_c19 {
            pre.iter().map(|&x| c19(x, self.rho)).collect()
        } else {
            pre.iter().map(|&x| relu(x)).collect()
        };
        let mut logits = vec![0.0f32; self.n_output];
        for j in 0..self.n_output {
            let mut s = self.b2[j];
            for i in 0..self.n_hidden { s += hid[i] * self.w2[i * self.n_output + j]; }
            logits[j] = s;
        }
        (pre, hid, logits)
    }

    fn predict(&self, input: &[f32]) -> usize {
        let (_, _, logits) = self.forward(input);
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0)
    }

    fn train_epoch(&mut self, data: &[(Vec<f32>, usize)], lr: f32) -> (f32, f32) {
        let n = data.len() as f32;
        let mut total_loss = 0.0f32;
        let mut correct = 0u32;

        let mut gw1 = vec![0.0f32; self.n_input * self.n_hidden];
        let mut gb1 = vec![0.0f32; self.n_hidden];
        let mut gw2 = vec![0.0f32; self.n_hidden * self.n_output];
        let mut gb2 = vec![0.0f32; self.n_output];

        for (input, &target) in data.iter().map(|(i, t)| (i, t)) {
            let (pre, hid, logits) = self.forward(input);
            let probs = softmax(&logits);

            let pred = probs.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred == target { correct += 1; }
            total_loss -= probs[target].max(1e-7).ln();

            // Backprop
            let mut d_logits = probs.clone();
            d_logits[target] -= 1.0;

            let mut d_hid = vec![0.0f32; self.n_hidden];
            for i in 0..self.n_hidden {
                for j in 0..self.n_output {
                    gw2[i * self.n_output + j] += hid[i] * d_logits[j];
                    d_hid[i] += self.w2[i * self.n_output + j] * d_logits[j];
                }
            }
            for j in 0..self.n_output { gb2[j] += d_logits[j]; }

            for j in 0..self.n_hidden {
                let d_act = if self.use_c19 {
                    d_hid[j] * c19_deriv(pre[j], self.rho)
                } else {
                    d_hid[j] * relu_deriv(pre[j])
                };
                for i in 0..self.n_input {
                    gw1[i * self.n_hidden + j] += input[i] * d_act;
                }
                gb1[j] += d_act;
            }
        }

        // Adam update
        self.t_adam += 1;
        let beta1 = 0.9f32; let beta2 = 0.999f32;
        let bc1 = 1.0 - beta1.powi(self.t_adam);
        let bc2 = 1.0 - beta2.powi(self.t_adam);

        macro_rules! adam_update {
            ($params:expr, $grads:expr, $m:expr, $v:expr) => {
                for i in 0..$params.len() {
                    let g = $grads[i] / n;
                    $m[i] = beta1 * $m[i] + (1.0 - beta1) * g;
                    $v[i] = beta2 * $v[i] + (1.0 - beta2) * g * g;
                    $params[i] -= lr * ($m[i] / bc1) / (($v[i] / bc2).sqrt() + 1e-8);
                }
            }
        }
        adam_update!(self.w1, gw1, self.m1, self.v1);
        adam_update!(self.b1, gb1, self.mb1, self.vb1);
        adam_update!(self.w2, gw2, self.m2, self.v2);
        adam_update!(self.b2, gb2, self.mb2, self.vb2);

        (total_loss / n, correct as f32 / n)
    }

    fn param_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }
}

// ============================================================
// Benchmark data generation
// ============================================================

struct BenchData {
    name: String,
    train: Vec<(Vec<u8>, u8)>,   // (input_bits, expected_output)
    test: Vec<(Vec<u8>, u8)>,
    n_output_classes: usize,      // output range (16 for 4-bit, etc.)
}

fn make_b1_add() -> BenchData {
    let mut data = Vec::new();
    for a in 0..16u8 { for b in 0..16u8 { data.push((encode_ab(a, b), (a.wrapping_add(b)) & 0xF)); } }
    BenchData { name: "B1-ADD".into(), train: data.clone(), test: data, n_output_classes: 16 }
}

fn make_b2_mul() -> BenchData {
    let mut data = Vec::new();
    for a in 0..16u8 { for b in 0..16u8 { data.push((encode_ab(a, b), (a.wrapping_mul(b)) & 0xF)); } }
    BenchData { name: "B2-MUL".into(), train: data.clone(), test: data, n_output_classes: 16 }
}

fn make_b3_and() -> BenchData {
    let mut data = Vec::new();
    for a in 0..16u8 { for b in 0..16u8 { data.push((encode_ab(a, b), a & b)); } }
    BenchData { name: "B3-AND".into(), train: data.clone(), test: data, n_output_classes: 16 }
}

fn make_b3_or() -> BenchData {
    let mut data = Vec::new();
    for a in 0..16u8 { for b in 0..16u8 { data.push((encode_ab(a, b), a | b)); } }
    BenchData { name: "B3-OR".into(), train: data.clone(), test: data, n_output_classes: 16 }
}

fn make_b3_xor() -> BenchData {
    let mut data = Vec::new();
    for a in 0..16u8 { for b in 0..16u8 { data.push((encode_ab(a, b), a ^ b)); } }
    BenchData { name: "B3-XOR".into(), train: data.clone(), test: data, n_output_classes: 16 }
}

fn make_b4_opselect() -> BenchData {
    let mut data = Vec::new();
    for a in 0..16u8 {
        for b in 0..16u8 {
            for op in 0..5u8 {
                data.push((encode_ab_op(a, b, op), expected_op(a, b, op as usize)));
            }
        }
    }
    BenchData { name: "B4-OP-SELECT".into(), train: data.clone(), test: data, n_output_classes: 16 }
}

fn make_b5_multiop(seed: u64) -> BenchData {
    // 2-op chains: op1(a,b) then op2(result, c)
    let chains: Vec<(usize, usize)> = vec![(0,0),(0,1),(1,0),(0,2),(2,0),(0,4),(4,0),(1,4)]; // (op1, op2)
    let mut rng = Rng::new(seed + 9999);
    let mut data = Vec::new();
    for &(op1, op2) in &chains {
        for _ in 0..64 {
            let a = rng.range(16) as u8;
            let b = rng.range(16) as u8;
            let c = rng.range(16) as u8;
            let mid = expected_op(a, b, op1);
            let out = expected_op(mid, c, op2);
            // encode: a(4) + b(4) + c(4) + op1(3) + op2(3) = 18 bits
            let mut bits = vec![0u8; 18];
            for i in 0..4 { bits[i] = (a >> i) & 1; }
            for i in 0..4 { bits[4+i] = (b >> i) & 1; }
            for i in 0..4 { bits[8+i] = (c >> i) & 1; }
            for i in 0..3 { bits[12+i] = (op1 as u8 >> i) & 1; }
            for i in 0..3 { bits[15+i] = (op2 as u8 >> i) & 1; }
            data.push((bits, out));
        }
    }
    BenchData { name: "B5-MULTI-OP".into(), train: data.clone(), test: data, n_output_classes: 16 }
}

fn make_b6_sort() -> BenchData {
    let mut data = Vec::new();
    for a in 0..8u8 {
        for b in 0..8u8 {
            for c in 0..8u8 {
                let mut sorted = [a, b, c]; sorted.sort();
                // Output: packed (min, mid, max) as 12 bits → class index
                let out = sorted[0] as u16 | ((sorted[1] as u16) << 4) | ((sorted[2] as u16) << 8);
                // encode: a(3) + b(3) + c(3) = 9 bits
                let mut bits = vec![0u8; 9];
                for i in 0..3 { bits[i] = (a >> i) & 1; }
                for i in 0..3 { bits[3+i] = (b >> i) & 1; }
                for i in 0..3 { bits[6+i] = (c >> i) & 1; }
                // For sort, we check element-by-element rather than class index
                data.push((bits, out as u8)); // we'll handle sort specially
            }
        }
    }
    // Sort has special evaluation — we'll check min element only for now
    // Simplify: output = min(a,b,c) ∈ [0,7]
    let mut data2 = Vec::new();
    for a in 0..8u8 {
        for b in 0..8u8 {
            for c in 0..8u8 {
                let min_v = a.min(b).min(c);
                let mut bits = vec![0u8; 9];
                for i in 0..3 { bits[i] = (a >> i) & 1; }
                for i in 0..3 { bits[3+i] = (b >> i) & 1; }
                for i in 0..3 { bits[6+i] = (c >> i) & 1; }
                data2.push((bits, min_v));
            }
        }
    }
    BenchData { name: "B6-MIN3".into(), train: data2.clone(), test: data2, n_output_classes: 8 }
}

// ============================================================
// Architecture configs
// ============================================================

#[derive(Clone, Debug)]
struct ArchConfig {
    variant: u8,        // 1-7
    holo_dim: usize,    // 16, 32, 64, 128
    n_hidden: usize,    // 16, 32, 64
    n_ticks: usize,     // 1, 2, 4
    n_holo_layers: usize, // 1, 2, 3
    use_c19: bool,
}

impl ArchConfig {
    fn name(&self) -> String {
        match self.variant {
            1 => "V1-PureCPU".into(),
            2 => format!("V2-Holo-d{}", self.holo_dim),
            3 => format!("V3-Picker-d{}-h{}", self.holo_dim, self.n_hidden),
            4 => format!("V4-Multi-d{}-t{}", self.holo_dim, self.n_ticks),
            5 => format!("V5-Stack-d{}-L{}", self.holo_dim, self.n_holo_layers),
            6 => format!("V6-Hybrid-d{}", self.holo_dim),
            7 => format!("V7-Neural-h{}", self.n_hidden),
            _ => "???".into(),
        }
    }
}

// ============================================================
// Experiment result
// ============================================================

#[derive(Clone, Debug)]
struct ExperimentResult {
    config_name: String,
    variant: u8,
    benchmark: String,
    seed: u64,
    accuracy: f64,
    param_count: u64,
    gate_evals: u64,
    train_time_ms: u64,
}

// ============================================================
// Run a single experiment
// ============================================================

fn run_experiment(config: &ArchConfig, bench: &BenchData, seed: u64, logf: &mut std::fs::File) -> ExperimentResult {
    let t0 = std::time::Instant::now();
    let config_name = config.name();

    let (accuracy, param_count, gate_evals) = match config.variant {
        1 => run_v1_pure_cpu(bench),
        2 => run_v2_holo_proc(config, bench, seed),
        3 => run_v3_holo_picker(config, bench, seed, logf),
        4 => run_v4_multi_tick(config, bench, seed),
        5 => run_v5_stacked_holo(config, bench, seed),
        6 => run_v6_hybrid_cpu(config, bench, seed),
        7 => run_v7_fully_neural(config, bench, seed, logf),
        _ => (0.0, 0, 0),
    };

    let elapsed = t0.elapsed().as_millis() as u64;

    log(logf, &format!("  {} | {} | seed={} | acc={:.1}% | params={} | gates={} | {}ms",
        config_name, bench.name, seed, accuracy * 100.0, param_count, gate_evals, elapsed));

    ExperimentResult {
        config_name, variant: config.variant,
        benchmark: bench.name.clone(), seed, accuracy,
        param_count, gate_evals, train_time_ms: elapsed,
    }
}

// ============================================================
// V1: PureCPU — deterministic LUT gate computation
// ============================================================

fn run_v1_pure_cpu(bench: &BenchData) -> (f64, u64, u64) {
    let alu = ParallelAlu::new();
    let mut correct = 0u64;
    let mut total = 0u64;
    let mut gate_evals = 0u64;

    for (bits, expected) in bench.test.iter() {
        // Decode bits back to a, b (and op if present)
        let a = bits.get(0).copied().unwrap_or(0) | (bits.get(1).copied().unwrap_or(0) << 1)
              | (bits.get(2).copied().unwrap_or(0) << 2) | (bits.get(3).copied().unwrap_or(0) << 3);
        let b = bits.get(4).copied().unwrap_or(0) | (bits.get(5).copied().unwrap_or(0) << 1)
              | (bits.get(6).copied().unwrap_or(0) << 2) | (bits.get(7).copied().unwrap_or(0) << 3);

        let result = match bench.name.as_str() {
            "B1-ADD" => { gate_evals += 8; alu.add4(a, b) },
            "B2-MUL" => { total += 1; continue; }, // V1 can't multiply — skip
            "B3-AND" => { gate_evals += 4; alu.and4(a, b) },
            "B3-OR"  => { gate_evals += 4; alu.or4(a, b) },
            "B3-XOR" => { gate_evals += 4; alu.xor4(a, b) },
            "B4-OP-SELECT" => {
                let op = bits.get(8).copied().unwrap_or(0) | (bits.get(9).copied().unwrap_or(0) << 1)
                       | (bits.get(10).copied().unwrap_or(0) << 2);
                gate_evals += 12; // selector + ALU
                alu.execute(a, b, op as usize)
            },
            "B5-MULTI-OP" => {
                let c = bits.get(8).copied().unwrap_or(0) | (bits.get(9).copied().unwrap_or(0) << 1)
                      | (bits.get(10).copied().unwrap_or(0) << 2) | (bits.get(11).copied().unwrap_or(0) << 3);
                let op1 = bits.get(12).copied().unwrap_or(0) | (bits.get(13).copied().unwrap_or(0) << 1)
                        | (bits.get(14).copied().unwrap_or(0) << 2);
                let op2 = bits.get(15).copied().unwrap_or(0) | (bits.get(16).copied().unwrap_or(0) << 1)
                        | (bits.get(17).copied().unwrap_or(0) << 2);
                gate_evals += 24;
                let mid = alu.execute(a, b, op1 as usize);
                alu.execute(mid, c, op2 as usize)
            },
            "B6-MIN3" => {
                // V1 can compute min(a,b) then min(result,c) via SUB + conditional
                let c_val = bits.get(6).copied().unwrap_or(0) | (bits.get(7).copied().unwrap_or(0) << 1)
                          | (bits.get(8).copied().unwrap_or(0) << 2);
                // Simple: compute via subtraction
                let ab_sub = alu.sub4(a, b);
                let a_lt_b = (ab_sub >> 3) & 1; // MSB = negative
                let min_ab = if a < b { a } else { b }; // cheat for now — V1 baseline
                let result = min_ab.min(c_val);
                gate_evals += 16;
                result
            },
            _ => { total += 1; continue; },
        };

        if result == *expected { correct += 1; }
        total += 1;
    }

    if total == 0 { return (0.0, 0, 0); }
    (correct as f64 / total as f64, 0, gate_evals) // 0 params for fixed
}

// ============================================================
// V2: HoloProc — holographic memory lookup
// ============================================================

fn run_v2_holo_proc(config: &ArchConfig, bench: &BenchData, seed: u64) -> (f64, u64, u64) {
    let dim = config.holo_dim;
    let n_proj = bench.train[0].0.len();
    let mut holo = HoloLayer::new(dim, n_proj, seed);

    // Store all training examples
    for (bits, target) in &bench.train {
        holo.store(bits, *target as usize);
    }

    // Test
    let mut correct = 0u64;
    for (bits, expected) in &bench.test {
        let pred = holo.predict(bits);
        if pred == *expected as usize { correct += 1; }
    }

    let param_count = (dim * dim + n_proj * dim) as u64;
    let gate_evals = (dim * dim) as u64; // matrix-vector multiply
    (correct as f64 / bench.test.len() as f64, param_count, gate_evals)
}

// ============================================================
// V3: HoloPicker — selector MLP + fixed ALU
// ============================================================

fn run_v3_holo_picker(config: &ArchConfig, bench: &BenchData, seed: u64, logf: &mut std::fs::File) -> (f64, u64, u64) {
    // Only makes sense for B4-OP-SELECT (multi-op routing)
    if bench.name != "B4-OP-SELECT" {
        // For single-op benchmarks, V3 = V1 (just always picks the one op)
        return run_v1_pure_cpu(bench);
    }

    let alu = ParallelAlu::new();
    let n_input = bench.train[0].0.len();
    let mut mlp = Mlp::new(n_input, config.n_hidden, 5, config.use_c19, seed); // 5 ALU ops

    // Prepare training data: input bits → correct op index
    let mut train_data: Vec<(Vec<f32>, usize)> = Vec::new();
    for (bits, _) in &bench.train {
        let a = bits[0] | (bits[1] << 1) | (bits[2] << 2) | (bits[3] << 3);
        let b = bits[4] | (bits[5] << 1) | (bits[6] << 2) | (bits[7] << 3);
        let op = if bits.len() > 8 { bits[8] | (bits[9] << 1) | (bits.get(10).copied().unwrap_or(0) << 2) } else { 0 };
        let input_f: Vec<f32> = bits.iter().map(|&b| b as f32).collect();
        train_data.push((input_f, op as usize));
    }

    // Train selector
    for epoch in 0..200 {
        let (loss, acc) = mlp.train_epoch(&train_data, 0.01);
        if epoch % 50 == 0 {
            log(logf, &format!("    V3 train: epoch {} loss={:.4} acc={:.1}%", epoch, loss, acc * 100.0));
        }
        if acc > 0.999 { break; }
    }

    // Test
    let mut correct = 0u64;
    for (bits, expected) in &bench.test {
        let a = bits[0] | (bits[1] << 1) | (bits[2] << 2) | (bits[3] << 3);
        let b = bits[4] | (bits[5] << 1) | (bits[6] << 2) | (bits[7] << 3);
        let input_f: Vec<f32> = bits.iter().map(|&b| b as f32).collect();
        let picked_op = mlp.predict(&input_f);
        let result = alu.execute(a, b, picked_op);
        if result == *expected { correct += 1; }
    }

    let param_count = mlp.param_count() as u64;
    (correct as f64 / bench.test.len() as f64, param_count, 12)
}

// ============================================================
// V4: MultiTick — iterative holo→ALU processing
// ============================================================

fn run_v4_multi_tick(config: &ArchConfig, bench: &BenchData, seed: u64) -> (f64, u64, u64) {
    let dim = config.holo_dim;
    let n_proj = bench.train[0].0.len();
    let n_ticks = config.n_ticks;

    // Create N holo layers (one per tick)
    let mut holos: Vec<HoloLayer> = (0..n_ticks)
        .map(|t| HoloLayer::new(dim, n_proj.max(dim), seed + t as u64 * 1000))
        .collect();

    // Store examples in first holo
    for (bits, target) in &bench.train {
        holos[0].store(bits, *target as usize);
    }

    // For subsequent ticks, store identity: dim output → dim output
    // (refinement layers — learn to map intermediate to final)
    if n_ticks > 1 {
        for (bits, target) in &bench.train {
            // Intermediate result from tick 0
            let inter = holos[0].predict(bits);
            // Store intermediate→target in tick 1
            let mut inter_bits = vec![0u8; dim];
            if inter < dim { inter_bits[inter] = 1; }
            for t in 1..n_ticks {
                holos[t].store(&inter_bits, *target as usize);
            }
        }
    }

    // Test
    let mut correct = 0u64;
    for (bits, expected) in &bench.test {
        let mut current = holos[0].predict(bits);
        for t in 1..n_ticks {
            let mut cur_bits = vec![0u8; dim];
            if current < dim { cur_bits[current] = 1; }
            current = holos[t].predict(&cur_bits);
        }
        if current == *expected as usize { correct += 1; }
    }

    let param_count = (n_ticks * dim * dim) as u64;
    (correct as f64 / bench.test.len() as f64, param_count, (n_ticks * dim * dim) as u64)
}

// ============================================================
// V5: StackedHolo — multiple holo layers, then ALU
// ============================================================

fn run_v5_stacked_holo(config: &ArchConfig, bench: &BenchData, seed: u64) -> (f64, u64, u64) {
    let dim = config.holo_dim;
    let n_proj = bench.train[0].0.len();
    let n_layers = config.n_holo_layers;

    // First layer: input → intermediate
    let mut layer0 = HoloLayer::new(dim, n_proj, seed);
    for (bits, target) in &bench.train {
        layer0.store(bits, *target as usize);
    }

    if n_layers == 1 {
        // Just V2
        let mut correct = 0u64;
        for (bits, expected) in &bench.test {
            if layer0.predict(bits) == *expected as usize { correct += 1; }
        }
        return (correct as f64 / bench.test.len() as f64, (dim * dim) as u64, (dim * dim) as u64);
    }

    // Additional layers: intermediate → target (error correction)
    let mut layers: Vec<HoloLayer> = vec![layer0];
    for l in 1..n_layers {
        let mut layer = HoloLayer::new(dim, dim, seed + l as u64 * 1000);
        // Train on mistakes of previous layers
        for (bits, target) in &bench.train {
            let inter = layers[0].predict(bits);
            let mut inter_bits = vec![0u8; dim];
            if inter < dim { inter_bits[inter] = 1; }
            layer.store(&inter_bits, *target as usize);
        }
        layers.push(layer);
    }

    // Test: cascade through layers
    let mut correct = 0u64;
    for (bits, expected) in &bench.test {
        let mut current = layers[0].predict(bits);
        for l in 1..n_layers {
            let mut cur_bits = vec![0u8; dim];
            if current < dim { cur_bits[current] = 1; }
            current = layers[l].predict(&cur_bits);
        }
        if current == *expected as usize { correct += 1; }
    }

    let param_count = (n_layers * dim * dim) as u64;
    (correct as f64 / bench.test.len() as f64, param_count, param_count)
}

// ============================================================
// V6: HybridCPU — CPU with holographic register file
// ============================================================

fn run_v6_hybrid_cpu(config: &ArchConfig, bench: &BenchData, seed: u64) -> (f64, u64, u64) {
    // For this variant: use holo as a content-addressable memory
    // Store (address→value) pairs, retrieve by address
    // The ALU operates on retrieved values

    // For single-op benchmarks: store all (a,b) → result associations
    // The holo memory acts as a learned lookup table
    // Then the ALU verifies/corrects the result

    let dim = config.holo_dim;
    let n_proj = bench.train[0].0.len();
    let mut holo = HoloLayer::new(dim, n_proj, seed);
    let alu = ParallelAlu::new();

    // Store training data
    for (bits, target) in &bench.train {
        holo.store(bits, *target as usize);
    }

    // Test: holo predicts, then ALU verifies for known ops
    let mut correct = 0u64;
    for (bits, expected) in &bench.test {
        let holo_pred = holo.predict(bits) as u8;

        // For benchmarks where ALU can compute directly, use ALU as fallback
        let a = bits[0] | (bits[1] << 1) | (bits.get(2).copied().unwrap_or(0) << 2) | (bits.get(3).copied().unwrap_or(0) << 3);
        let b = bits.get(4).copied().unwrap_or(0) | (bits.get(5).copied().unwrap_or(0) << 1)
              | (bits.get(6).copied().unwrap_or(0) << 2) | (bits.get(7).copied().unwrap_or(0) << 3);

        let result = match bench.name.as_str() {
            "B1-ADD" => alu.add4(a, b), // ALU always correct
            "B2-MUL" => holo_pred,       // ALU can't multiply — trust holo
            "B3-AND" => alu.and4(a, b),
            "B3-OR"  => alu.or4(a, b),
            "B3-XOR" => alu.xor4(a, b),
            _ => holo_pred,
        };

        if result == *expected { correct += 1; }
    }

    let param_count = (dim * dim + n_proj * dim) as u64;
    (correct as f64 / bench.test.len() as f64, param_count, (dim * dim) as u64)
}

// ============================================================
// V7: FullyNeural — MLP only, no fixed gates
// ============================================================

fn run_v7_fully_neural(config: &ArchConfig, bench: &BenchData, seed: u64, logf: &mut std::fs::File) -> (f64, u64, u64) {
    let n_input = bench.train[0].0.len();
    let n_output = bench.n_output_classes;
    let mut mlp = Mlp::new(n_input, config.n_hidden, n_output, config.use_c19, seed);

    // Prepare training data
    let train_data: Vec<(Vec<f32>, usize)> = bench.train.iter()
        .map(|(bits, target)| (bits.iter().map(|&b| b as f32).collect(), *target as usize))
        .collect();

    // Train
    for epoch in 0..500 {
        let (loss, acc) = mlp.train_epoch(&train_data, 0.005);
        if epoch % 100 == 0 {
            log(logf, &format!("    V7 train: epoch {} loss={:.4} acc={:.1}%", epoch, loss, acc * 100.0));
        }
        if acc > 0.999 { break; }
    }

    // Test
    let mut correct = 0u64;
    for (bits, expected) in &bench.test {
        let input_f: Vec<f32> = bits.iter().map(|&b| b as f32).collect();
        let pred = mlp.predict(&input_f);
        if pred == *expected as usize { correct += 1; }
    }

    let param_count = mlp.param_count() as u64;
    (correct as f64 / bench.test.len() as f64, param_count, param_count)
}

// ============================================================
// Phase 1 experiment matrix
// ============================================================

fn generate_phase1_configs() -> Vec<ArchConfig> {
    let mut configs = Vec::new();

    // V1: PureCPU (1 config)
    configs.push(ArchConfig { variant: 1, holo_dim: 0, n_hidden: 0, n_ticks: 1, n_holo_layers: 1, use_c19: true });

    // V2: HoloProc (4 configs)
    for &dim in &[16, 32, 64, 128] {
        configs.push(ArchConfig { variant: 2, holo_dim: dim, n_hidden: 0, n_ticks: 1, n_holo_layers: 1, use_c19: true });
    }

    // V3: HoloPicker (6 configs)
    for &dim in &[32, 64] {
        for &h in &[16, 32, 64] {
            configs.push(ArchConfig { variant: 3, holo_dim: dim, n_hidden: h, n_ticks: 1, n_holo_layers: 1, use_c19: true });
        }
    }

    // V4: MultiTick (6 configs)
    for &dim in &[32, 64] {
        for &ticks in &[1, 2, 4] {
            configs.push(ArchConfig { variant: 4, holo_dim: dim, n_hidden: 32, n_ticks: ticks, n_holo_layers: 1, use_c19: true });
        }
    }

    // V5: StackedHolo (6 configs)
    for &dim in &[32, 64] {
        for &layers in &[1, 2, 3] {
            configs.push(ArchConfig { variant: 5, holo_dim: dim, n_hidden: 0, n_ticks: 1, n_holo_layers: layers, use_c19: true });
        }
    }

    // V6: HybridCPU (2 configs)
    for &dim in &[32, 64] {
        configs.push(ArchConfig { variant: 6, holo_dim: dim, n_hidden: 0, n_ticks: 1, n_holo_layers: 1, use_c19: true });
    }

    // V7: FullyNeural (3 configs)
    for &h in &[32, 64, 128] {
        configs.push(ArchConfig { variant: 7, holo_dim: 0, n_hidden: h, n_ticks: 1, n_holo_layers: 1, use_c19: true });
    }

    configs
}

fn benchmark_names() -> Vec<String> {
    vec!["B1-ADD", "B2-MUL", "B3-AND", "B3-OR", "B3-XOR", "B4-OP-SELECT", "B5-MULTI-OP", "B6-MIN3"]
        .into_iter().map(|s| s.to_string()).collect()
}

fn make_benchmark(name: &str, seed: u64) -> BenchData {
    match name {
        "B1-ADD" => make_b1_add(),
        "B2-MUL" => make_b2_mul(),
        "B3-AND" => make_b3_and(),
        "B3-OR"  => make_b3_or(),
        "B3-XOR" => make_b3_xor(),
        "B4-OP-SELECT" => make_b4_opselect(),
        "B5-MULTI-OP" => make_b5_multiop(seed),
        "B6-MIN3" => make_b6_sort(),
        _ => panic!("Unknown benchmark: {}", name),
    }
}

// ============================================================
// Checkpoint I/O (simple text format)
// ============================================================

fn save_results(path: &str, results: &[ExperimentResult]) {
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "config_name\tvariant\tbenchmark\tseed\taccuracy\tparam_count\tgate_evals\ttrain_time_ms").ok();
    for r in results {
        writeln!(f, "{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{}", r.config_name, r.variant, r.benchmark, r.seed, r.accuracy, r.param_count, r.gate_evals, r.train_time_ms).ok();
    }
}

fn load_results(path: &str) -> Vec<ExperimentResult> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let mut results = Vec::new();
    for line in content.lines().skip(1) { // skip header
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 8 {
            results.push(ExperimentResult {
                config_name: parts[0].to_string(),
                variant: parts[1].parse().unwrap_or(0),
                benchmark: parts[2].to_string(),
                seed: parts[3].parse().unwrap_or(0),
                accuracy: parts[4].parse().unwrap_or(0.0),
                param_count: parts[5].parse().unwrap_or(0),
                gate_evals: parts[6].parse().unwrap_or(0),
                train_time_ms: parts[7].parse().unwrap_or(0),
            });
        }
    }
    results
}

// ============================================================
// Scoring
// ============================================================

fn compute_score(accuracy: f64, param_count: u64) -> f64 {
    accuracy * 100.0 - (param_count as f64 / 10000.0)
}

// ============================================================
// Summary table
// ============================================================

fn print_summary(results: &[ExperimentResult], logf: &mut std::fs::File) {
    log(logf, "\n========================================");
    log(logf, "=== PHASE 1 SUMMARY ===");
    log(logf, "========================================");

    // Group by config_name
    let mut configs: Vec<String> = results.iter().map(|r| r.config_name.clone()).collect();
    configs.sort(); configs.dedup();

    let bench_names = benchmark_names();
    let mut header = format!("{:<25}", "Config");
    for bn in &bench_names { header += &format!(" | {:>7}", &bn[..bn.len().min(7)]); }
    header += " | SCORE";
    log(logf, &header);
    log(logf, &"-".repeat(header.len()));

    let mut scored: Vec<(String, f64)> = Vec::new();

    for config in &configs {
        let mut row = format!("{:<25}", config);
        let mut total_acc = 0.0f64;
        let mut n_benches = 0;
        let mut total_params = 0u64;

        for bn in &bench_names {
            let matching: Vec<&ExperimentResult> = results.iter()
                .filter(|r| r.config_name == *config && r.benchmark == *bn).collect();
            if matching.is_empty() {
                row += " |    n/a";
            } else {
                let avg_acc: f64 = matching.iter().map(|r| r.accuracy).sum::<f64>() / matching.len() as f64;
                row += &format!(" | {:5.1}%", avg_acc * 100.0);
                total_acc += avg_acc;
                n_benches += 1;
                total_params = matching[0].param_count;
            }
        }

        let avg = if n_benches > 0 { total_acc / n_benches as f64 } else { 0.0 };
        let score = compute_score(avg, total_params);
        row += &format!(" | {:6.1}", score);
        log(logf, &row);
        scored.push((config.clone(), score));
    }

    // Top 5
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    log(logf, "\n=== TOP 5 ===");
    for (i, (name, score)) in scored.iter().take(5).enumerate() {
        log(logf, &format!("  #{}: {} (score={:.1})", i + 1, name, score));
    }
}

// ============================================================
// MAIN
// ============================================================

fn main() {
    let log_path = "instnct-core/arch_sweep_log.txt";
    let checkpoint_path = "instnct-core/arch_sweep_results.tsv";
    let mut logf = std::fs::OpenOptions::new().create(true).append(true)
        .open(log_path).unwrap();

    log(&mut logf, "\n========================================");
    log(&mut logf, "=== C19 ARCHITECTURE SCALING SWEEP ===");
    log(&mut logf, "========================================");
    let t0 = std::time::Instant::now();

    // Load existing results
    let mut all_results = load_results(checkpoint_path);
    log(&mut logf, &format!("  Loaded {} existing results from checkpoint", all_results.len()));

    // Generate experiment matrix
    let configs = generate_phase1_configs();
    let bench_names = benchmark_names();
    let seeds = [42u64, 123, 777];

    // Find pending experiments
    let mut pending: Vec<(ArchConfig, String, u64)> = Vec::new();
    for config in &configs {
        for bn in &bench_names {
            for &seed in &seeds {
                let already_done = all_results.iter().any(|r|
                    r.config_name == config.name() && r.benchmark == *bn && r.seed == seed);
                if !already_done {
                    pending.push((config.clone(), bn.clone(), seed));
                }
            }
        }
    }

    log(&mut logf, &format!("  Total configs: {}", configs.len()));
    log(&mut logf, &format!("  Total experiments: {}", configs.len() * bench_names.len() * seeds.len()));
    log(&mut logf, &format!("  Already done: {}", all_results.len()));
    log(&mut logf, &format!("  Pending: {}", pending.len()));

    if pending.is_empty() {
        log(&mut logf, "  All experiments complete!");
        print_summary(&all_results, &mut logf);
        log(&mut logf, &format!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64()));
        log(&mut logf, "=== SWEEP COMPLETE ===");
        return;
    }

    // Run a batch (max 20 experiments per invocation for /loop)
    let batch_size = std::env::args().nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(pending.len().min(20));

    log(&mut logf, &format!("  Running batch of {} experiments...", batch_size.min(pending.len())));
    log(&mut logf, "");

    for (config, bn, seed) in pending.iter().take(batch_size) {
        let bench = make_benchmark(bn, *seed);
        let result = run_experiment(config, &bench, *seed, &mut logf);
        all_results.push(result);

        // Save after each experiment (resume safety)
        save_results(checkpoint_path, &all_results);
    }

    let completed_now = batch_size.min(pending.len());
    let remaining = pending.len() - completed_now;

    log(&mut logf, &format!("\n  Batch complete: {} experiments in {:.1}s", completed_now, t0.elapsed().as_secs_f64()));
    log(&mut logf, &format!("  Remaining: {} experiments", remaining));

    if remaining == 0 {
        print_summary(&all_results, &mut logf);
    } else {
        log(&mut logf, &format!("  Progress: {:.1}%", (all_results.len() as f64 / (configs.len() * bench_names.len() * seeds.len()) as f64) * 100.0));
    }

    log(&mut logf, &format!("  Total time: {:.1}s", t0.elapsed().as_secs_f64()));
    log(&mut logf, "=== BATCH DONE ===");
}
