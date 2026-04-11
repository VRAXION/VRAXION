//! Cortex → ALU Control Kill Test
//!
//! THE QUESTION: Can an EP-trained C19 cortex INVOKE the right ALU operation
//! based on a non-mathematical "situation" input?
//!
//! Architecture:
//!   Sensor input (8 bit) → CORTEX (EP, C19 rho=8) → opcode (3 bit) → ALU (frozen) → result
//!
//! The cortex receives sensor data and must DECIDE which operation to perform.
//! It doesn't compute the result itself — it dispatches to the proven ALU.
//!
//! Scenario: Simple autonomous agent
//!   - 3 sensors: temperature(3bit), proximity(3bit), energy(2bit)
//!   - 6 possible actions mapped to ALU ops:
//!     ADD  = "approach" (add steps toward target)
//!     SUB  = "retreat"  (compute escape distance)
//!     AND  = "filter"   (mask sensor noise)
//!     OR   = "merge"    (combine sensor readings)
//!     XOR  = "toggle"   (flip behavior mode)
//!     MIN  = "conserve" (pick lowest-cost option)
//!
//! Kill criterion: <70% correct dispatch → KILL
//! Success: >90% correct dispatch + freeze preserves >85%
//!
//! Run: cargo run --example cortex_alu_kill1 --release

// ============================================================
// C19 activation
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

fn tanh_act(x: f32) -> f32 { x.tanh() }

// ============================================================
// RNG
// ============================================================

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() as usize) % (i + 1);
            v.swap(i, j);
        }
    }
}

// ============================================================
// Activation enum
// ============================================================

#[derive(Clone, Copy)]
enum Act { C19(f32), Tanh }

impl Act {
    fn apply(&self, x: f32) -> f32 {
        match self { Act::C19(rho) => c19(x, *rho), Act::Tanh => tanh_act(x) }
    }
    fn name(&self) -> String {
        match self { Act::C19(rho) => format!("C19_rho{}", rho), Act::Tanh => "tanh".into() }
    }
}

// ============================================================
// ALU operations (frozen compute — already proven in alu8bit.rs)
// These are the "body" the cortex controls
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AluOp { ADD, SUB, AND, OR, XOR, MIN }

impl AluOp {
    fn execute(&self, a: u8, b: u8) -> u8 {
        match self {
            AluOp::ADD => a.wrapping_add(b),
            AluOp::SUB => a.wrapping_sub(b),
            AluOp::AND => a & b,
            AluOp::OR  => a | b,
            AluOp::XOR => a ^ b,
            AluOp::MIN => a.min(b),
        }
    }
    fn index(&self) -> usize {
        match self { AluOp::ADD=>0, AluOp::SUB=>1, AluOp::AND=>2, AluOp::OR=>3, AluOp::XOR=>4, AluOp::MIN=>5 }
    }
    fn from_index(i: usize) -> Self {
        match i { 0=>AluOp::ADD, 1=>AluOp::SUB, 2=>AluOp::AND, 3=>AluOp::OR, 4=>AluOp::XOR, _=>AluOp::MIN }
    }
    fn label(&self) -> &'static str {
        match self { AluOp::ADD=>"ADD(approach)", AluOp::SUB=>"SUB(retreat)", AluOp::AND=>"AND(filter)",
                     AluOp::OR=>"OR(merge)", AluOp::XOR=>"XOR(toggle)", AluOp::MIN=>"MIN(conserve)" }
    }
}

const N_OPS: usize = 6;

// ============================================================
// Scenario: sensor → required action
//
// The mapping is NOT a simple threshold — it's a learned pattern:
//   temp=high(5-7) + prox=close(0-2)     → RETREAT (SUB)
//   temp=low(0-2)  + prox=far(5-7)       → APPROACH (ADD)
//   energy=low(0)  + any                  → CONSERVE (MIN)
//   temp=mid(3-4)  + prox=mid(3-4)       → FILTER (AND)
//   energy=high(3) + prox=close(0-2)     → TOGGLE (XOR)
//   otherwise                             → MERGE (OR)
// ============================================================

fn decide_action(temp: u8, prox: u8, energy: u8) -> AluOp {
    // Energy-low overrides everything
    if energy == 0 { return AluOp::MIN; }
    // Danger: hot + close → retreat
    if temp >= 5 && prox <= 2 { return AluOp::SUB; }
    // Safe + far → approach
    if temp <= 2 && prox >= 5 { return AluOp::ADD; }
    // High energy + close → toggle mode
    if energy == 3 && prox <= 2 { return AluOp::XOR; }
    // Mid-range → filter
    if temp >= 3 && temp <= 4 && prox >= 3 && prox <= 4 { return AluOp::AND; }
    // Default → merge
    AluOp::OR
}

fn encode_sensor(temp: u8, prox: u8, energy: u8) -> Vec<f32> {
    // Encode as normalized floats [0,1]
    vec![
        temp as f32 / 7.0,
        prox as f32 / 7.0,
        energy as f32 / 3.0,
        // Cross-features (helps the network learn interactions)
        (temp as f32 / 7.0) * (prox as f32 / 7.0),
        (temp as f32 / 7.0) * (energy as f32 / 3.0),
        (prox as f32 / 7.0) * (energy as f32 / 3.0),
        // Binary indicators
        if temp >= 5 { 1.0 } else { 0.0 },
        if prox <= 2 { 1.0 } else { 0.0 },
    ]
}

const IN_DIM: usize = 8;  // 3 raw + 3 cross + 2 binary

fn make_target(op: AluOp) -> Vec<f32> {
    let mut t = vec![0.0f32; N_OPS];
    t[op.index()] = 1.0;
    t
}

// ============================================================
// Generate all training data (8×8×4 = 256 scenarios)
// ============================================================

fn generate_data() -> Vec<(Vec<f32>, AluOp, Vec<f32>)> {
    let mut data = Vec::new();
    for temp in 0..8u8 {
        for prox in 0..8u8 {
            for energy in 0..4u8 {
                let op = decide_action(temp, prox, energy);
                let input = encode_sensor(temp, prox, energy);
                let target = make_target(op);
                data.push((input, op, target));
            }
        }
    }
    data
}

// ============================================================
// EP Network (1 hidden layer, multi-output for classification)
// ============================================================

struct EpNet {
    w1: Vec<f32>,  // h_dim x in_dim
    w2: Vec<f32>,  // out_dim x h_dim
    b1: Vec<f32>,
    b2: Vec<f32>,
    in_dim: usize,
    h_dim: usize,
    out_dim: usize,
}

impl EpNet {
    fn new(in_dim: usize, h_dim: usize, out_dim: usize, init_scale: f32, rng: &mut Rng) -> Self {
        let s1 = init_scale * (2.0 / in_dim as f32).sqrt();
        let s2 = init_scale * (2.0 / h_dim as f32).sqrt();
        EpNet {
            w1: (0..h_dim * in_dim).map(|_| rng.range_f32(-s1, s1)).collect(),
            w2: (0..out_dim * h_dim).map(|_| rng.range_f32(-s2, s2)).collect(),
            b1: vec![0.0; h_dim],
            b2: vec![0.0; out_dim],
            in_dim, h_dim, out_dim,
        }
    }

    fn freeze_i8(&self) -> FrozenNet {
        // Per-layer quantization scale (much better than global)
        let max_w1 = self.w1.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let max_w2 = self.w2.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let max_b1 = self.b1.iter().map(|b| b.abs()).fold(0.0f32, f32::max);
        let max_b2 = self.b2.iter().map(|b| b.abs()).fold(0.0f32, f32::max);

        let s1 = if max_w1 > 1e-8 { 127.0 / max_w1 } else { 1.0 };
        let s2 = if max_w2 > 1e-8 { 127.0 / max_w2 } else { 1.0 };
        let sb1 = if max_b1 > 1e-8 { 127.0 / max_b1 } else { 1.0 };
        let sb2 = if max_b2 > 1e-8 { 127.0 / max_b2 } else { 1.0 };

        FrozenNet {
            w1: self.w1.iter().map(|w| (w * s1).round().clamp(-127.0, 127.0) as i8).collect(),
            w2: self.w2.iter().map(|w| (w * s2).round().clamp(-127.0, 127.0) as i8).collect(),
            b1: self.b1.iter().map(|b| (b * sb1).round().clamp(-127.0, 127.0) as i8).collect(),
            b2: self.b2.iter().map(|b| (b * sb2).round().clamp(-127.0, 127.0) as i8).collect(),
            scale_w1: s1, scale_w2: s2, scale_b1: sb1, scale_b2: sb2,
            in_dim: self.in_dim, h_dim: self.h_dim, out_dim: self.out_dim,
        }
    }
}

// ============================================================
// Frozen int8 network (deployment form — zero FPU)
// ============================================================

struct FrozenNet {
    w1: Vec<i8>,
    w2: Vec<i8>,
    b1: Vec<i8>,
    b2: Vec<i8>,
    scale_w1: f32,
    scale_w2: f32,
    scale_b1: f32,
    scale_b2: f32,
    in_dim: usize,
    h_dim: usize,
    out_dim: usize,
}

impl FrozenNet {
    /// Integer-only forward pass. Returns argmax output index.
    fn infer(&self, x: &[f32], act: Act) -> usize {
        let (in_d, h, out_d) = (self.in_dim, self.h_dim, self.out_dim);

        // Quantize input to i8
        let x_i8: Vec<i8> = x.iter().map(|v| (v * 127.0).round().clamp(-127.0, 127.0) as i8).collect();

        // Hidden layer: W1 * x + b1
        let mut h_vals = vec![0i32; h];
        for j in 0..h {
            let mut sum = 0i32;
            for i in 0..in_d {
                sum += self.w1[j * in_d + i] as i32 * x_i8[i] as i32;
            }
            // Add bias (scaled to match: bias needs scale_b1/scale_w1 * 127 adjustment)
            let bias_scaled = (self.b1[j] as f32 / self.scale_b1 * self.scale_w1 * 127.0).round() as i32;
            sum += bias_scaled;
            h_vals[j] = sum;
        }

        // Apply activation (in real hardware: LUT lookup)
        let h_act: Vec<i8> = h_vals.iter().map(|&v| {
            let f = v as f32 / (127.0 * self.scale_w1);
            let a = act.apply(f);
            if a.is_nan() { 0i8 } else { (a * 127.0).round().clamp(-127.0, 127.0) as i8 }
        }).collect();

        // Output layer: W2 * act(h) + b2
        let mut out_vals = vec![0i32; out_d];
        for k in 0..out_d {
            let mut sum = 0i32;
            for j in 0..h {
                sum += self.w2[k * h + j] as i32 * h_act[j] as i32;
            }
            let bias_scaled = (self.b2[k] as f32 / self.scale_b2 * self.scale_w2 * 127.0).round() as i32;
            sum += bias_scaled;
            out_vals[k] = sum;
        }

        // Argmax (no activation needed — just pick the largest)
        out_vals.iter().enumerate().max_by_key(|(_, v)| *v).map(|(i, _)| i).unwrap()
    }
}

// ============================================================
// EP settle step (same as wave_ep_fixed.rs)
// ============================================================

fn settle_step(
    s_h: &[f32], s_out: &[f32],
    x: &[f32], net: &EpNet, dt: f32, act: Act, beta: f32, y: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let (in_d, h, out_d) = (net.in_dim, net.h_dim, net.out_dim);

    let mut new_h = vec![0.0f32; h];
    for j in 0..h {
        let mut drive = net.b1[j];
        for i in 0..in_d { drive += net.w1[j * in_d + i] * x[i]; }
        for k in 0..out_d { drive += net.w2[k * h + j] * act.apply(s_out[k]); }
        new_h[j] = s_h[j] + dt * (-s_h[j] + drive);
    }

    let mut new_out = vec![0.0f32; out_d];
    for k in 0..out_d {
        let mut drive = net.b2[k];
        for j in 0..h { drive += net.w2[k * h + j] * act.apply(s_h[j]); }
        let nudge = beta * (y[k] - act.apply(s_out[k]));
        new_out[k] = s_out[k] + dt * (-s_out[k] + drive + nudge);
    }

    (new_h, new_out)
}

// ============================================================
// Settle to equilibrium and return output activations
// ============================================================

fn settle(x: &[f32], y: &[f32], net: &EpNet, t_max: usize, dt: f32, act: Act, beta: f32)
    -> (Vec<f32>, Vec<f32>)  // (s_h, s_out)
{
    let mut s_h = vec![0.0f32; net.h_dim];
    let mut s_out = vec![0.0f32; net.out_dim];
    for _ in 0..t_max {
        let (nh, no) = settle_step(&s_h, &s_out, x, net, dt, act, beta, y);
        s_h = nh;
        s_out = no;
    }
    (s_h, s_out)
}

// ============================================================
// Evaluate accuracy: cortex picks op, ALU executes, check result
// ============================================================

fn evaluate(net: &EpNet, data: &[(Vec<f32>, AluOp, Vec<f32>)], t_max: usize, dt: f32, act: Act)
    -> (f32, usize, usize)  // (accuracy, correct, total)
{
    let mut correct = 0;
    for (x, expected_op, y) in data {
        let (_, s_out) = settle(x, y, net, t_max, dt, act, 0.0);
        let out_acts: Vec<f32> = s_out.iter().map(|s| act.apply(*s)).collect();
        if out_acts.iter().any(|v| v.is_nan()) { continue; }
        let predicted_idx = out_acts.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap();
        let predicted_op = AluOp::from_index(predicted_idx);
        if predicted_op == *expected_op { correct += 1; }
    }
    let acc = correct as f32 / data.len() as f32;
    (acc, correct, data.len())
}

// ============================================================
// Full pipeline test: cortex → ALU → verify result
// ============================================================

fn full_pipeline_test(net: &EpNet, data: &[(Vec<f32>, AluOp, Vec<f32>)],
                      t_max: usize, dt: f32, act: Act, label: &str, verbose: bool)
    -> (f32, f32)  // (dispatch_acc, result_acc)
{
    let mut dispatch_correct = 0;
    let mut result_correct = 0;
    let mut per_op_correct = [0usize; N_OPS];
    let mut per_op_total = [0usize; N_OPS];

    for (idx, (x, expected_op, y)) in data.iter().enumerate() {
        let (_, s_out) = settle(x, y, net, t_max, dt, act, 0.0);
        let out_acts: Vec<f32> = s_out.iter().map(|s| act.apply(*s)).collect();

        // Handle NaN: if any output is NaN, skip this sample (count as wrong)
        if out_acts.iter().any(|v| v.is_nan()) {
            per_op_total[expected_op.index()] += 1;
            continue;
        }

        let predicted_idx = out_acts.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap();
        let predicted_op = AluOp::from_index(predicted_idx);

        per_op_total[expected_op.index()] += 1;

        // Dispatch check: did cortex pick the right operation?
        let dispatch_ok = predicted_op == *expected_op;
        if dispatch_ok { dispatch_correct += 1; per_op_correct[expected_op.index()] += 1; }

        // Execute: ALU runs whatever cortex decided
        // Use operands derived from sensor (temp=a, prox=b)
        let a = (x[0] * 7.0).round() as u8;
        let b = (x[1] * 7.0).round() as u8;
        let cortex_result = predicted_op.execute(a, b);
        let expected_result = expected_op.execute(a, b);
        let result_ok = cortex_result == expected_result;
        if result_ok { result_correct += 1; }

        if verbose && !dispatch_ok && idx < 20 {
            println!("    MISS: sensor=({:.0},{:.0},{:.0}) expected={} got={}",
                x[0]*7.0, x[1]*7.0, x[2]*3.0, expected_op.label(), predicted_op.label());
        }
    }

    let dispatch_acc = dispatch_correct as f32 / data.len() as f32;
    let result_acc = result_correct as f32 / data.len() as f32;

    println!("  {} — dispatch={:.1}% ({}/{}), result={:.1}% ({}/{})",
        label, dispatch_acc*100.0, dispatch_correct, data.len(),
        result_acc*100.0, result_correct, data.len());

    // Per-operation breakdown
    println!("    Per-op accuracy:");
    for i in 0..N_OPS {
        if per_op_total[i] > 0 {
            let acc = per_op_correct[i] as f32 / per_op_total[i] as f32;
            println!("      {:15} {}/{} = {:.1}%",
                AluOp::from_index(i).label(), per_op_correct[i], per_op_total[i], acc*100.0);
        }
    }

    (dispatch_acc, result_acc)
}

// ============================================================
// Frozen pipeline test (integer-only inference)
// ============================================================

fn frozen_pipeline_test(frozen: &FrozenNet, data: &[(Vec<f32>, AluOp, Vec<f32>)], act: Act, label: &str)
    -> (f32, f32)
{
    let mut dispatch_correct = 0;
    let mut result_correct = 0;

    for (x, expected_op, _y) in data {
        let predicted_idx = frozen.infer(x, act);
        let predicted_op = AluOp::from_index(predicted_idx);

        let dispatch_ok = predicted_op == *expected_op;
        if dispatch_ok { dispatch_correct += 1; }

        let a = (x[0] * 7.0).round() as u8;
        let b = (x[1] * 7.0).round() as u8;
        let cortex_result = predicted_op.execute(a, b);
        let expected_result = expected_op.execute(a, b);
        if cortex_result == expected_result { result_correct += 1; }
    }

    let dispatch_acc = dispatch_correct as f32 / data.len() as f32;
    let result_acc = result_correct as f32 / data.len() as f32;

    println!("  {} — dispatch={:.1}% ({}/{}), result={:.1}% ({}/{})",
        label, dispatch_acc*100.0, dispatch_correct, data.len(),
        result_acc*100.0, result_correct, data.len());

    (dispatch_acc, result_acc)
}

// ============================================================
// Train EP cortex
// ============================================================

fn train_cortex(act: Act, h_dim: usize, beta: f32, t_max: usize, dt: f32,
                lr: f32, n_epochs: usize, seed: u64,
                data: &[(Vec<f32>, AluOp, Vec<f32>)])
    -> EpNet
{
    let in_dim = IN_DIM;
    let out_dim = N_OPS;
    let mut rng = Rng::new(seed);
    let mut net = EpNet::new(in_dim, h_dim, out_dim, 1.0, &mut rng);

    let mut indices: Vec<usize> = (0..data.len()).collect();

    for epoch in 0..n_epochs {
        let lr_eff = if epoch < 20 { lr * (epoch as f32 + 1.0) / 20.0 } else { lr };
        rng.shuffle(&mut indices);

        for &idx in &indices {
            let (x, _op, y) = &data[idx];

            // Free phase
            let (s_free_h, s_free_out) = settle(x, y, &net, t_max, dt, act, 0.0);

            // Nudged phase (from free equilibrium)
            let mut s_h = s_free_h.clone();
            let mut s_out = s_free_out.clone();
            for _ in 0..t_max {
                let (nh, no) = settle_step(&s_h, &s_out, x, &net, dt, act, beta, y);
                s_h = nh; s_out = no;
            }
            let s_nudge_h = s_h;
            let s_nudge_out = s_out;

            // Weight update (correct sign: +=)
            let inv_beta = 1.0 / beta;

            for j in 0..h_dim {
                let a_n = act.apply(s_nudge_h[j]);
                let a_f = act.apply(s_free_h[j]);
                for i in 0..in_dim {
                    net.w1[j * in_dim + i] += lr_eff * inv_beta * (a_n * x[i] - a_f * x[i]);
                }
                net.b1[j] += lr_eff * inv_beta * (a_n - a_f);
            }

            for k in 0..out_dim {
                let ao_n = act.apply(s_nudge_out[k]);
                let ao_f = act.apply(s_free_out[k]);
                for j in 0..h_dim {
                    let ah_n = act.apply(s_nudge_h[j]);
                    let ah_f = act.apply(s_free_h[j]);
                    net.w2[k * h_dim + j] += lr_eff * inv_beta * (ao_n * ah_n - ao_f * ah_f);
                }
                net.b2[k] += lr_eff * inv_beta * (ao_n - ao_f);
            }
        }

        // Log every 100 epochs
        if epoch % 100 == 0 || epoch == n_epochs - 1 {
            let (acc, correct, total) = evaluate(&net, data, t_max, dt, act);
            println!("    Epoch {:4} — dispatch acc = {:.1}% ({}/{})", epoch, acc*100.0, correct, total);
        }
    }

    net
}

// ============================================================
// MAIN
// ============================================================

fn main() {
    println!("================================================================");
    println!("  CORTEX → ALU CONTROL — Kill Test #1");
    println!("  Can EP-trained C19 cortex invoke the right ALU operation?");
    println!("================================================================");
    println!();

    // Generate all scenarios
    let data = generate_data();
    println!("Generated {} scenarios (8 temp × 8 prox × 4 energy)", data.len());

    // Count per-op distribution
    let mut op_counts = [0usize; N_OPS];
    for (_, op, _) in &data { op_counts[op.index()] += 1; }
    println!("Action distribution:");
    for i in 0..N_OPS {
        println!("  {:15} = {} ({:.1}%)", AluOp::from_index(i).label(),
            op_counts[i], op_counts[i] as f32 / data.len() as f32 * 100.0);
    }
    println!("Random baseline: {:.1}%", 100.0 / N_OPS as f32);
    println!();

    // ============================================================
    // Test configurations
    // ============================================================

    struct Config {
        act: Act,
        h_dim: usize,
        beta: f32,
        t_max: usize,
        dt: f32,
        lr: f32,
        epochs: usize,
        seeds: Vec<u64>,
    }

    let configs = vec![
        Config {
            act: Act::C19(8.0), h_dim: 24, beta: 0.5, t_max: 50, dt: 0.5,
            lr: 0.005, epochs: 500, seeds: vec![42, 123, 7],
        },
        Config {
            act: Act::C19(8.0), h_dim: 32, beta: 0.5, t_max: 50, dt: 0.5,
            lr: 0.005, epochs: 500, seeds: vec![42, 123, 7],
        },
        Config {
            act: Act::C19(8.0), h_dim: 48, beta: 0.5, t_max: 80, dt: 0.3,
            lr: 0.003, epochs: 800, seeds: vec![42, 123, 7],
        },
        Config {
            act: Act::Tanh, h_dim: 24, beta: 0.5, t_max: 50, dt: 0.5,
            lr: 0.005, epochs: 500, seeds: vec![42, 123, 7],
        },
        Config {
            act: Act::Tanh, h_dim: 32, beta: 0.5, t_max: 50, dt: 0.5,
            lr: 0.005, epochs: 500, seeds: vec![42, 123, 7],
        },
    ];

    println!("================================================================");
    println!("  TRAINING + EVALUATION");
    println!("================================================================");
    println!();

    let mut results: Vec<(String, f32, f32, f32, f32)> = Vec::new(); // (label, dispatch, result, frozen_dispatch, frozen_result)

    for (ci, cfg) in configs.iter().enumerate() {
        println!("━━━ Config {} — {} H={} β={} T={} dt={} lr={} ep={} ━━━",
            ci+1, cfg.act.name(), cfg.h_dim, cfg.beta, cfg.t_max, cfg.dt, cfg.lr, cfg.epochs);

        for &seed in &cfg.seeds {
            println!("  Seed {}:", seed);

            // Train
            let net = train_cortex(
                cfg.act, cfg.h_dim, cfg.beta, cfg.t_max, cfg.dt,
                cfg.lr, cfg.epochs, seed, &data
            );

            // Full pipeline test (float)
            let (d_acc, r_acc) = full_pipeline_test(
                &net, &data, cfg.t_max, cfg.dt, cfg.act,
                "FLOAT pipeline", ci == 0 && seed == 42
            );

            // Freeze to int8
            let frozen = net.freeze_i8();

            // Frozen pipeline test
            let (fd_acc, fr_acc) = frozen_pipeline_test(
                &frozen, &data, cfg.act, "INT8  pipeline"
            );

            let freeze_loss = (d_acc - fd_acc) * 100.0;
            println!("    Freeze loss: {:.1}pp dispatch", freeze_loss);
            println!();

            let label = format!("{} H={} s={}", cfg.act.name(), cfg.h_dim, seed);
            results.push((label, d_acc, r_acc, fd_acc, fr_acc));
        }
    }

    // ============================================================
    // Summary table
    // ============================================================

    println!("================================================================");
    println!("  SUMMARY TABLE");
    println!("================================================================");
    println!("  {:30} {:>8} {:>8} {:>8} {:>8}", "Config", "Disp%", "Res%", "FrzD%", "FrzR%");
    println!("  {}", "─".repeat(70));

    let mut best_dispatch = 0.0f32;
    let mut best_label = String::new();

    for (label, d, r, fd, fr) in &results {
        println!("  {:30} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}%",
            label, d*100.0, r*100.0, fd*100.0, fr*100.0);
        if *d > best_dispatch {
            best_dispatch = *d;
            best_label = label.clone();
        }
    }

    println!();

    // ============================================================
    // Kill/Continue decision
    // ============================================================

    println!("================================================================");
    println!("  KILL / CONTINUE DECISION");
    println!("================================================================");
    println!();

    let best_c19_dispatch = results.iter()
        .filter(|(l, _, _, _, _)| l.contains("C19"))
        .map(|(_, d, _, _, _)| *d)
        .fold(0.0f32, f32::max);

    let best_c19_frozen = results.iter()
        .filter(|(l, _, _, _, _)| l.contains("C19"))
        .map(|(_, _, _, fd, _)| *fd)
        .fold(0.0f32, f32::max);

    let best_tanh_dispatch = results.iter()
        .filter(|(l, _, _, _, _)| l.contains("tanh"))
        .map(|(_, d, _, _, _)| *d)
        .fold(0.0f32, f32::max);

    println!("  Best C19 dispatch:  {:.1}%", best_c19_dispatch * 100.0);
    println!("  Best C19 frozen:    {:.1}%", best_c19_frozen * 100.0);
    println!("  Best tanh dispatch: {:.1}%", best_tanh_dispatch * 100.0);
    println!("  Random baseline:    {:.1}%", 100.0 / N_OPS as f32);
    println!();

    if best_c19_dispatch < 0.70 {
        println!("  ██ VERDICT: KILL ██");
        println!("  C19 cortex cannot reliably dispatch ALU operations.");
        println!("  Best accuracy {:.1}% < 70% threshold.", best_c19_dispatch * 100.0);
    } else if best_c19_dispatch >= 0.90 && best_c19_frozen >= 0.85 {
        println!("  ██ VERDICT: STRONG CONTINUE ██");
        println!("  C19 cortex dispatches correctly AND survives int8 freeze.");
        println!("  This validates the VRAXION cortex → ALU architecture.");
    } else if best_c19_dispatch >= 0.70 {
        println!("  ██ VERDICT: CONTINUE (with caveats) ██");
        println!("  C19 cortex works but may need tuning.");
    }

    if best_c19_dispatch >= best_tanh_dispatch * 0.9 {
        println!("  C19 is competitive with tanh ({:.1}% vs {:.1}%).",
            best_c19_dispatch * 100.0, best_tanh_dispatch * 100.0);
    } else {
        println!("  WARNING: C19 lags behind tanh ({:.1}% vs {:.1}%).",
            best_c19_dispatch * 100.0, best_tanh_dispatch * 100.0);
    }

    println!();
    println!("  Champion: {}", best_label);
    println!("================================================================");
}
