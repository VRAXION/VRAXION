//! Wave EP Kill Experiment #1 v2: C19 + Equilibrium Propagation (SIGN FIX)
//!
//! Identical to wave_ep_kill1.rs except the weight update sign is FIXED:
//!   OLD (wrong): W -= lr * (1/beta) * (nudge - free)   [anti-learning]
//!   NEW (correct): W += lr * (1/beta) * (nudge - free) [proper EP]
//!
//! 3-stage experiment (swarm consensus plan, 3/3 judges READY):
//!   Stage 1: XOR kill test (36 configs, ~0.2s) — converges at all?
//!   Stage 2: Full sweep (1296 configs, ~43s) — learns anything?
//!   Stage 3: Depth bonus (12 configs, ~2s) — does depth help bidirectional?
//!
//! Run: cargo run --example wave_ep_kill1_v2 --release

use std::io::Write;

// ============================================================
// Activation functions
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
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    (sgn + 2.0 * rho * h) * (1.0 - 2.0 * t)
}

fn relu(x: f32) -> f32 { x.max(0.0) }
fn relu_deriv(x: f32) -> f32 { if x > 0.0 { 1.0 } else { 0.0 } }

fn tanh_act(x: f32) -> f32 { x.tanh() }
fn tanh_deriv(x: f32) -> f32 { let t = x.tanh(); 1.0 - t * t }

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
enum Act { C19(f32), ReLU, Tanh }

impl Act {
    fn apply(&self, x: f32) -> f32 {
        match self { Act::C19(rho) => c19(x, *rho), Act::ReLU => relu(x), Act::Tanh => tanh_act(x) }
    }
    fn deriv(&self, x: f32) -> f32 {
        match self { Act::C19(rho) => c19_deriv(x, *rho), Act::ReLU => relu_deriv(x), Act::Tanh => tanh_deriv(x) }
    }
    fn name(&self) -> String {
        match self { Act::C19(rho) => format!("c19_r{}", *rho as i32), Act::ReLU => "relu".into(), Act::Tanh => "tanh".into() }
    }
}

// ============================================================
// EP Network (1 hidden layer)
// ============================================================

struct EpNet {
    w1: Vec<f32>,  // h_dim x in_dim
    w2: Vec<f32>,  // out_dim x h_dim
    b1: Vec<f32>,  // h_dim
    b2: Vec<f32>,  // out_dim
    in_dim: usize,
    h_dim: usize,
    out_dim: usize,
}

impl EpNet {
    fn new(in_dim: usize, h_dim: usize, out_dim: usize, init_scale: f32, rng: &mut Rng) -> Self {
        let s1 = init_scale * (2.0 / in_dim as f32).sqrt();
        let s2 = init_scale * (2.0 / h_dim as f32).sqrt();
        let w1: Vec<f32> = (0..h_dim * in_dim).map(|_| rng.range_f32(-s1, s1)).collect();
        let w2: Vec<f32> = (0..out_dim * h_dim).map(|_| rng.range_f32(-s2, s2)).collect();
        EpNet { w1, w2, b1: vec![0.0; h_dim], b2: vec![0.0; out_dim], in_dim, h_dim, out_dim }
    }
}

// ============================================================
// EP Network (2 hidden layers, for Stage 3)
// ============================================================

struct EpNet2 {
    w1: Vec<f32>,  // h x in
    w2: Vec<f32>,  // h x h (symmetric: also h2->h1)
    w3: Vec<f32>,  // out x h
    b1: Vec<f32>,
    b2: Vec<f32>,
    b3: Vec<f32>,
    in_dim: usize,
    h_dim: usize,
    out_dim: usize,
}

impl EpNet2 {
    fn new(in_dim: usize, h_dim: usize, out_dim: usize, init_scale: f32, rng: &mut Rng) -> Self {
        let s1 = init_scale * (2.0 / in_dim as f32).sqrt();
        let s2 = init_scale * (2.0 / h_dim as f32).sqrt();
        let s3 = init_scale * (2.0 / h_dim as f32).sqrt();
        EpNet2 {
            w1: (0..h_dim * in_dim).map(|_| rng.range_f32(-s1, s1)).collect(),
            w2: (0..h_dim * h_dim).map(|_| rng.range_f32(-s2, s2)).collect(),
            w3: (0..out_dim * h_dim).map(|_| rng.range_f32(-s3, s3)).collect(),
            b1: vec![0.0; h_dim], b2: vec![0.0; h_dim], b3: vec![0.0; out_dim],
            in_dim, h_dim, out_dim,
        }
    }
}

// ============================================================
// Convergence info
// ============================================================

struct ConvInfo {
    converged: bool,
    final_delta: f32,
    converge_iter: Option<usize>,
    oscillating: bool,
    curve: Vec<f32>,
}

// ============================================================
// Settle step (1 hidden layer) — the bidirectional core
// ============================================================

fn settle_1h(
    s_h: &mut [f32], s_out: &mut [f32],
    x: &[f32], net: &EpNet, dt: f32, act: Act, beta: f32, y: &[f32],
) -> f32 {
    let (in_d, h, out_d) = (net.in_dim, net.h_dim, net.out_dim);
    let mut max_delta: f32 = 0.0;

    // Hidden layer: forward (W1*x) + backward (W2^T * act(s_out)) + bias
    for j in 0..h {
        let mut drive = net.b1[j];
        for i in 0..in_d { drive += net.w1[j * in_d + i] * x[i]; }
        for k in 0..out_d { drive += net.w2[k * h + j] * act.apply(s_out[k]); }
        let new_val = s_h[j] + dt * (-s_h[j] + drive);
        let delta = (new_val - s_h[j]).abs();
        if delta > max_delta { max_delta = delta; }
        s_h[j] = new_val;
    }

    // Output layer: forward (W2 * act(s_h)) + bias + nudge
    for k in 0..out_d {
        let mut drive = net.b2[k];
        for j in 0..h { drive += net.w2[k * h + j] * act.apply(s_h[j]); }
        let nudge = -beta * (act.apply(s_out[k]) - y[k]);
        let new_val = s_out[k] + dt * (-s_out[k] + drive + nudge);
        let delta = (new_val - s_out[k]).abs();
        if delta > max_delta { max_delta = delta; }
        s_out[k] = new_val;
    }

    max_delta
}

// ============================================================
// Settle step (2 hidden layers) — for Stage 3
// ============================================================

fn settle_2h(
    s_h1: &mut [f32], s_h2: &mut [f32], s_out: &mut [f32],
    x: &[f32], net: &EpNet2, dt: f32, act: Act, beta: f32, y: &[f32],
) -> f32 {
    let (in_d, h, out_d) = (net.in_dim, net.h_dim, net.out_dim);
    let mut max_delta: f32 = 0.0;

    // H1: forward (W1*x) + backward (W2^T * act(s_h2))
    for j in 0..h {
        let mut drive = net.b1[j];
        for i in 0..in_d { drive += net.w1[j * in_d + i] * x[i]; }
        for k in 0..h { drive += net.w2[k * h + j] * act.apply(s_h2[k]); }
        let new_val = s_h1[j] + dt * (-s_h1[j] + drive);
        let d = (new_val - s_h1[j]).abs(); if d > max_delta { max_delta = d; }
        s_h1[j] = new_val;
    }

    // H2: forward (W2 * act(s_h1)) + backward (W3^T * act(s_out))
    for j in 0..h {
        let mut drive = net.b2[j];
        for i in 0..h { drive += net.w2[j * h + i] * act.apply(s_h1[i]); }
        for k in 0..out_d { drive += net.w3[k * h + j] * act.apply(s_out[k]); }
        let new_val = s_h2[j] + dt * (-s_h2[j] + drive);
        let d = (new_val - s_h2[j]).abs(); if d > max_delta { max_delta = d; }
        s_h2[j] = new_val;
    }

    // Output: forward (W3 * act(s_h2)) + nudge
    for k in 0..out_d {
        let mut drive = net.b3[k];
        for j in 0..h { drive += net.w3[k * h + j] * act.apply(s_h2[j]); }
        let nudge = -beta * (act.apply(s_out[k]) - y[k]);
        let new_val = s_out[k] + dt * (-s_out[k] + drive + nudge);
        let d = (new_val - s_out[k]).abs(); if d > max_delta { max_delta = d; }
        s_out[k] = new_val;
    }

    max_delta
}

// ============================================================
// Run EP settling phase, return convergence info
// ============================================================

fn run_settle_1h(
    x: &[f32], y: &[f32], net: &EpNet, t_max: usize, dt: f32, act: Act, beta: f32,
) -> (Vec<f32>, Vec<f32>, ConvInfo) {
    let mut s_h = vec![0.0f32; net.h_dim];
    let mut s_out = vec![0.0f32; net.out_dim];
    let mut curve = Vec::with_capacity(t_max);
    let eps = 1e-3;
    let mut converge_iter = None;

    for t in 0..t_max {
        let delta = settle_1h(&mut s_h, &mut s_out, x, net, dt, act, beta, y);
        curve.push(delta);
        if delta < eps && converge_iter.is_none() { converge_iter = Some(t); }
    }

    let final_delta = *curve.last().unwrap_or(&f32::MAX);
    let oscillating = if curve.len() >= 20 {
        let tail = &curve[curve.len()-20..];
        tail.windows(2).filter(|w| w[1] > w[0]).count() >= 3
    } else { false };

    (s_h, s_out, ConvInfo { converged: final_delta < eps, final_delta, converge_iter, oscillating, curve })
}

fn run_settle_1h_from(
    s_h_init: &[f32], s_out_init: &[f32],
    x: &[f32], y: &[f32], net: &EpNet, t_max: usize, dt: f32, act: Act, beta: f32,
) -> (Vec<f32>, Vec<f32>, ConvInfo) {
    let mut s_h = s_h_init.to_vec();
    let mut s_out = s_out_init.to_vec();
    let mut curve = Vec::with_capacity(t_max);
    let eps = 1e-3;
    let mut converge_iter = None;

    for t in 0..t_max {
        let delta = settle_1h(&mut s_h, &mut s_out, x, net, dt, act, beta, y);
        curve.push(delta);
        if delta < eps && converge_iter.is_none() { converge_iter = Some(t); }
    }

    let final_delta = *curve.last().unwrap_or(&f32::MAX);
    let oscillating = if curve.len() >= 20 {
        let tail = &curve[curve.len()-20..];
        tail.windows(2).filter(|w| w[1] > w[0]).count() >= 3
    } else { false };

    (s_h, s_out, ConvInfo { converged: final_delta < eps, final_delta, converge_iter, oscillating, curve })
}

// ============================================================
// Spectral norm (power iteration, 5 iters)
// ============================================================

fn spectral_norm(w: &[f32], rows: usize, cols: usize) -> f32 {
    let mut v: Vec<f32> = (0..cols).map(|_| 1.0 / (cols as f32).sqrt()).collect();
    for _ in 0..5 {
        let mut u = vec![0.0f32; rows];
        for i in 0..rows { for j in 0..cols { u[i] += w[i * cols + j] * v[j]; } }
        let nu: f32 = u.iter().map(|x| x*x).sum::<f32>().sqrt();
        if nu < 1e-10 { return 0.0; }
        for x in &mut u { *x /= nu; }
        v = vec![0.0; cols];
        for j in 0..cols { for i in 0..rows { v[j] += w[i * cols + j] * u[i]; } }
        let nv: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
        if nv < 1e-10 { return 0.0; }
        for x in &mut v { *x /= nv; }
    }
    let mut wv = vec![0.0f32; rows];
    for i in 0..rows { for j in 0..cols { wv[i] += w[i * cols + j] * v[j]; } }
    wv.iter().map(|x| x*x).sum::<f32>().sqrt()
}

// ============================================================
// Segment crossing count (C19 specific)
// ============================================================

fn count_seg_crossings(curve: &[f32]) -> usize {
    let mut crossings = 0;
    for i in 1..curve.len() {
        if curve[i-1].floor() as i32 != curve[i].floor() as i32 { crossings += 1; }
    }
    crossings
}

// ============================================================
// Task data
// ============================================================

fn xor_data() -> Vec<(Vec<f32>, Vec<f32>)> {
    vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]
}

fn thermo(val: usize, bits: usize) -> Vec<f32> {
    (0..bits).map(|i| if i < val { 1.0 } else { 0.0 }).collect()
}

fn add_data() -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut data = Vec::new();
    for a in 0..5usize {
        for b in 0..5usize {
            let mut input = thermo(a, 4);
            input.extend(thermo(b, 4));
            let sum = (a + b).min(5);
            data.push((input, thermo(sum, 5)));
        }
    }
    data
}

// ============================================================
// Train + Evaluate (1 hidden layer)
// ============================================================

struct RunResult {
    conv_free: f32,
    conv_nudge: f32,
    accuracy: f32,
    mse: f32,
    mean_settle: f32,
    osc_ratio: f32,
    spectral_w1: f32,
    spectral_w2: f32,
    output_mag: f32,
    seg_crossings: f32,
    free_curves: Vec<Vec<f32>>,  // only populated for Stage 1
}

fn train_and_eval_1h(
    data: &[(Vec<f32>, Vec<f32>)],
    h_dim: usize, act: Act, beta: f32, t_max: usize, dt: f32,
    init_scale: f32, lr: f32, n_epochs: usize, seed: u64,
    log_curves: bool,
) -> RunResult {
    let in_dim = data[0].0.len();
    let out_dim = data[0].1.len();
    let mut rng = Rng::new(seed);
    let mut net = EpNet::new(in_dim, h_dim, out_dim, init_scale, &mut rng);

    let mut indices: Vec<usize> = (0..data.len()).collect();

    // Training
    for epoch in 0..n_epochs {
        let lr_eff = if epoch < 10 { lr * (epoch as f32 + 1.0) / 10.0 } else { lr };
        rng.shuffle(&mut indices);

        for &idx in &indices {
            let (x, y) = &data[idx];

            // Free phase
            let (s_free_h, s_free_out, _) = run_settle_1h(x, y, &net, t_max, dt, act, 0.0);

            // Nudged phase (from free equilibrium)
            let (s_nudge_h, s_nudge_out, _) = run_settle_1h_from(
                &s_free_h, &s_free_out, x, y, &net, t_max, dt, act, beta);

            // Weight update — FIX: += not -= (correct EP learning rule)
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
    }

    // Evaluation
    let mut conv_free_count = 0;
    let mut conv_nudge_count = 0;
    let mut correct = 0;
    let mut total_mse = 0.0f64;
    let mut total_settle = 0usize;
    let mut settle_count = 0;
    let mut osc_count = 0;
    let mut total_out_mag = 0.0f64;
    let mut total_seg = 0usize;
    let mut all_curves = Vec::new();

    for (x, y) in data {
        // Free phase eval
        let (s_h, s_out, conv) = run_settle_1h(x, y, &net, t_max, dt, act, 0.0);
        if conv.converged { conv_free_count += 1; }
        if conv.oscillating { osc_count += 1; }
        if let Some(ci) = conv.converge_iter { total_settle += ci; settle_count += 1; }

        // Segment crossings (track hidden states for C19)
        if let Act::C19(_) = act {
            for &s in &s_h { total_seg += count_seg_crossings(&conv.curve); }
        }

        if log_curves { all_curves.push(conv.curve); }

        // Nudged phase eval
        let (_, s_out_n, conv_n) = run_settle_1h_from(&s_h, &s_out, x, y, &net, t_max, dt, act, beta);
        if conv_n.converged { conv_nudge_count += 1; }

        // Accuracy + MSE (on free phase output)
        let mut sample_mse = 0.0f64;
        let mut sample_correct = true;
        for k in 0..out_dim {
            let pred = act.apply(s_out[k]);
            let target = y[k];
            sample_mse += (pred - target) as f64 * (pred - target) as f64;
            let pred_bin = if pred > 0.5 { 1.0 } else { 0.0 };
            if (pred_bin - target).abs() > 0.5 { sample_correct = false; }
            total_out_mag += pred.abs() as f64;
        }
        total_mse += sample_mse / out_dim as f64;
        if sample_correct { correct += 1; }
    }

    let n = data.len() as f32;
    RunResult {
        conv_free: conv_free_count as f32 / n,
        conv_nudge: conv_nudge_count as f32 / n,
        accuracy: correct as f32 / n,
        mse: (total_mse / data.len() as f64) as f32,
        mean_settle: if settle_count > 0 { total_settle as f32 / settle_count as f32 } else { t_max as f32 },
        osc_ratio: osc_count as f32 / n,
        spectral_w1: spectral_norm(&net.w1, h_dim, in_dim),
        spectral_w2: spectral_norm(&net.w2, out_dim, h_dim),
        output_mag: (total_out_mag / (data.len() * out_dim) as f64) as f32,
        seg_crossings: total_seg as f32 / n,
        free_curves: all_curves,
    }
}

// ============================================================
// Train + Evaluate (2 hidden layers, Stage 3 only)
// ============================================================

fn train_and_eval_2h(
    data: &[(Vec<f32>, Vec<f32>)],
    h_dim: usize, act: Act, beta: f32, t_max: usize, dt: f32,
    init_scale: f32, lr: f32, n_epochs: usize, seed: u64,
) -> (f32, f32) /* (conv_free, accuracy) */ {
    let in_dim = data[0].0.len();
    let out_dim = data[0].1.len();
    let mut rng = Rng::new(seed);
    let mut net = EpNet2::new(in_dim, h_dim, out_dim, init_scale, &mut rng);

    let mut indices: Vec<usize> = (0..data.len()).collect();

    for epoch in 0..n_epochs {
        let lr_eff = if epoch < 10 { lr * (epoch as f32 + 1.0) / 10.0 } else { lr };
        rng.shuffle(&mut indices);

        for &idx in &indices {
            let (x, y) = &data[idx];

            // Free phase
            let mut s_h1 = vec![0.0f32; h_dim];
            let mut s_h2 = vec![0.0f32; h_dim];
            let mut s_out = vec![0.0f32; out_dim];
            for _ in 0..t_max { settle_2h(&mut s_h1, &mut s_h2, &mut s_out, x, &net, dt, act, 0.0, y); }
            let (sf_h1, sf_h2, sf_out) = (s_h1.clone(), s_h2.clone(), s_out.clone());

            // Nudged phase
            let (mut s_h1, mut s_h2, mut s_out) = (sf_h1.clone(), sf_h2.clone(), sf_out.clone());
            for _ in 0..t_max { settle_2h(&mut s_h1, &mut s_h2, &mut s_out, x, &net, dt, act, beta, y); }

            // Weight updates — FIX: += not -= (correct EP learning rule)
            let inv_beta = 1.0 / beta;
            for j in 0..h_dim {
                let an = act.apply(s_h1[j]); let af = act.apply(sf_h1[j]);
                for i in 0..in_dim {
                    net.w1[j * in_dim + i] += lr_eff * inv_beta * (an * x[i] - af * x[i]);
                }
                net.b1[j] += lr_eff * inv_beta * (an - af);
            }
            for j in 0..h_dim {
                let an = act.apply(s_h2[j]); let af = act.apply(sf_h2[j]);
                for i in 0..h_dim {
                    let ahn = act.apply(s_h1[i]); let ahf = act.apply(sf_h1[i]);
                    net.w2[j * h_dim + i] += lr_eff * inv_beta * (an * ahn - af * ahf);
                }
                net.b2[j] += lr_eff * inv_beta * (an - af);
            }
            for k in 0..out_dim {
                let aon = act.apply(s_out[k]); let aof = act.apply(sf_out[k]);
                for j in 0..h_dim {
                    let ahn = act.apply(s_h2[j]); let ahf = act.apply(sf_h2[j]);
                    net.w3[k * h_dim + j] += lr_eff * inv_beta * (aon * ahn - aof * ahf);
                }
                net.b3[k] += lr_eff * inv_beta * (aon - aof);
            }
        }
    }

    // Eval
    let mut conv_count = 0;
    let mut correct = 0;
    for (x, y) in data {
        let mut s_h1 = vec![0.0f32; h_dim];
        let mut s_h2 = vec![0.0f32; h_dim];
        let mut s_out = vec![0.0f32; out_dim];
        let mut final_delta = f32::MAX;
        for _ in 0..t_max {
            final_delta = settle_2h(&mut s_h1, &mut s_h2, &mut s_out, x, &net, dt, act, 0.0, y);
        }
        if final_delta < 1e-3 { conv_count += 1; }
        let mut ok = true;
        for k in 0..out_dim {
            let p = if act.apply(s_out[k]) > 0.5 { 1.0 } else { 0.0 };
            if (p - y[k]).abs() > 0.5 { ok = false; }
        }
        if ok { correct += 1; }
    }

    (conv_count as f32 / data.len() as f32, correct as f32 / data.len() as f32)
}

// ============================================================
// Logger
// ============================================================

struct Log { file: std::fs::File }
impl Log {
    fn new(path: &str) -> Self { Log { file: std::fs::File::create(path).unwrap() } }
    fn w(&mut self, msg: &str) {
        let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
        let s = d.as_secs(); let h = (s/3600)%24; let m = (s/60)%60; let sec = s%60;
        let line = format!("[{:02}:{:02}:{:02}] {}\n", h, m, sec, msg);
        print!("{}", line);
        self.file.write_all(line.as_bytes()).ok();
        self.file.flush().ok();
    }
}

// ============================================================
// MAIN
// ============================================================

fn main() {
    let mut log = Log::new("instnct-core/wave_ep_v2_log.txt");
    let mut tsv = std::fs::File::create("instnct-core/wave_ep_v2_results.tsv").unwrap();
    let mut curves_file = std::fs::File::create("instnct-core/wave_ep_v2_curves.tsv").unwrap();

    writeln!(tsv, "stage\ttask\tact\trho\tH\tbeta\tT\tdt\tinit_scale\tseed\tconv_free\tconv_nudge\taccuracy\tmse\tmean_settle\tosc_ratio\tspectral_w1\tspectral_w2\toutput_mag\tseg_crossings").ok();
    writeln!(curves_file, "config_id\tsample_id\titer\tdelta").ok();

    log.w("======================================================");
    log.w("  WAVE EP KILL TEST v2: C19 + Equilibrium Propagation");
    log.w("  FIX APPLIED: weight update sign += (was -= in v1)");
    log.w("======================================================");

    let t0 = std::time::Instant::now();
    let xor = xor_data();
    let add = add_data();
    let seeds = [42u64, 123, 7];
    let lr = 0.05;

    // ========================================================
    // STAGE 1: XOR Kill Test
    // ========================================================
    log.w("");
    log.w("=== STAGE 1/3: XOR Kill Test (36 configs) ===");

    let dts = [0.5f32, 0.1];
    let init_scales = [0.1f32, 0.5, 1.0];
    let acts_s1: Vec<Act> = vec![Act::C19(8.0), Act::Tanh];

    let mut best_c19_conv: f32 = 0.0;
    let mut best_c19_dt: f32 = 0.5;
    let mut best_c19_scale: f32 = 0.1;
    let mut config_id = 0u32;
    let mut s1_idx = 0;

    for &act in &acts_s1 {
        for &dt in &dts {
            for &sc in &init_scales {
                for &seed in &seeds {
                    s1_idx += 1;
                    let r = train_and_eval_1h(&xor, 20, act, 0.5, 100, dt, sc, lr, 200, seed, true);

                    log.w(&format!("[{:>2}/36] {} dt={:.2} sc={:.1} s={} | conv={:.3} acc={:.3} mse={:.4} settle={:.0} osc={:.2}",
                        s1_idx, act.name(), dt, sc, seed, r.conv_free, r.accuracy, r.mse, r.mean_settle, r.osc_ratio));

                    writeln!(tsv, "1\tXOR\t{}\t{}\t20\t0.50\t100\t{:.2}\t{:.1}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.6}\t{:.1}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}",
                        act.name(), match act { Act::C19(r) => r as i32, _ => 0 }, dt, sc, seed,
                        r.conv_free, r.conv_nudge, r.accuracy, r.mse, r.mean_settle,
                        r.osc_ratio, r.spectral_w1, r.spectral_w2, r.output_mag, r.seg_crossings).ok();

                    // Log convergence curves for Stage 1
                    for (si, curve) in r.free_curves.iter().enumerate() {
                        for (t, &d) in curve.iter().enumerate() {
                            writeln!(curves_file, "{}\t{}\t{}\t{:.6}", config_id, si, t, d).ok();
                        }
                    }
                    config_id += 1;

                    if let Act::C19(_) = act {
                        if r.conv_free > best_c19_conv {
                            best_c19_conv = r.conv_free;
                            best_c19_dt = dt;
                            best_c19_scale = sc;
                        }
                    }
                }
            }
        }
    }

    log.w("");
    log.w(&format!("--- Stage 1 Summary ---"));
    log.w(&format!("  Best C19 convergence: {:.3} (dt={}, init_scale={})", best_c19_conv, best_c19_dt, best_c19_scale));

    if best_c19_conv < 0.5 {
        log.w("  *** KILL: C19 conv < 0.5 across all configs. C19+EP is DEAD. ***");
        log.w(&format!("  Total time: {:.2}s", t0.elapsed().as_secs_f64()));
        return;
    }
    log.w(&format!("  CONTINUE: C19 conv >= 0.5. Proceeding to Stage 2."));

    // ========================================================
    // STAGE 2: Full Sweep
    // ========================================================
    log.w("");
    log.w("=== STAGE 2/3: Full Sweep ===");

    let rhos = [1.0f32, 4.0, 8.0, 16.0];
    let hs = [10usize, 20, 50];
    let betas = [0.1f32, 0.5, 1.0];
    let ts = [20usize, 50, 100, 200];
    let tasks: Vec<(&str, &[(Vec<f32>, Vec<f32>)])> = vec![("XOR", &xor), ("ADD", &add)];

    let mut s2_idx = 0u32;
    let mut best_c19_acc = 0.0f32;
    let mut best_c19_rho = 8.0f32;
    let mut best_c19_beta = 0.5f32;
    let mut best_c19_t = 100usize;
    let mut best_tanh_acc = 0.0f32;
    let mut best_relu_acc = 0.0f32;

    let acts_s2: Vec<Act> = vec![Act::C19(1.0), Act::C19(4.0), Act::C19(8.0), Act::C19(16.0), Act::ReLU, Act::Tanh];

    let total_s2 = tasks.len() * acts_s2.len() * hs.len() * betas.len() * ts.len() * seeds.len();
    log.w(&format!("  Total configs: {} (estimated ~{}s)", total_s2, total_s2 / 30));

    for &(task_name, task_data) in &tasks {
        let n_epochs = 200;
        for &act in &acts_s2 {
            for &h in &hs {
                for &beta in &betas {
                    for &t in &ts {
                        for &seed in &seeds {
                            s2_idx += 1;
                            let (dt, sc) = match act {
                                Act::C19(_) => (best_c19_dt, best_c19_scale),
                                _ => (0.5, 1.0),
                            };

                            let r = train_and_eval_1h(task_data, h, act, beta, t, dt, sc, lr, n_epochs, seed, false);

                            let rho_val = match act { Act::C19(r) => r as i32, _ => 0 };

                            if s2_idx % 100 == 0 || s2_idx <= 5 {
                                log.w(&format!("[{}/{}] {} {} H={} b={:.1} T={} | conv={:.3} acc={:.3} mse={:.4}",
                                    s2_idx, total_s2, task_name, act.name(), h, beta, t,
                                    r.conv_free, r.accuracy, r.mse));
                            }

                            writeln!(tsv, "2\t{}\t{}\t{}\t{}\t{:.2}\t{}\t{:.2}\t{:.1}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.6}\t{:.1}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}",
                                task_name, act.name(), rho_val, h, beta, t, dt, sc, seed,
                                r.conv_free, r.conv_nudge, r.accuracy, r.mse, r.mean_settle,
                                r.osc_ratio, r.spectral_w1, r.spectral_w2, r.output_mag, r.seg_crossings).ok();

                            match act {
                                Act::C19(rho) => {
                                    if r.accuracy > best_c19_acc {
                                        best_c19_acc = r.accuracy;
                                        best_c19_rho = rho;
                                        best_c19_beta = beta;
                                        best_c19_t = t;
                                    }
                                },
                                Act::ReLU => { if r.accuracy > best_relu_acc { best_relu_acc = r.accuracy; } },
                                Act::Tanh => { if r.accuracy > best_tanh_acc { best_tanh_acc = r.accuracy; } },
                            }
                        }
                    }
                }
            }
        }
    }

    log.w("");
    log.w("--- Stage 2 Summary ---");
    log.w(&format!("  Best C19:  acc={:.3} (rho={}, beta={}, T={})", best_c19_acc, best_c19_rho as i32, best_c19_beta, best_c19_t));
    log.w(&format!("  Best ReLU: acc={:.3}", best_relu_acc));
    log.w(&format!("  Best tanh: acc={:.3}", best_tanh_acc));

    if best_c19_acc < 0.3 {
        log.w("  *** KILL: C19 acc < 30% (near random). Converges but doesn't learn. ***");
        log.w(&format!("  Total time: {:.2}s", t0.elapsed().as_secs_f64()));
        return;
    }

    let verdict = if best_c19_acc >= best_tanh_acc * 0.8 { "STRONG CONTINUE" }
                  else if best_c19_acc >= best_relu_acc * 0.9 { "CONTINUE" }
                  else if best_c19_acc < best_relu_acc * 0.7 { "WEAK" }
                  else { "CONTINUE" };
    log.w(&format!("  Stage 2 verdict: {}", verdict));

    // ========================================================
    // STAGE 3: Depth Bonus (1 vs 2 hidden layers)
    // ========================================================
    log.w("");
    log.w("=== STAGE 3/3: Depth Bonus (12 configs) ===");

    let depth_acts: Vec<Act> = vec![Act::C19(best_c19_rho), Act::Tanh];
    let h = 20;

    for &act in &depth_acts {
        for &seed in &seeds {
            // 1 hidden layer
            let r1 = train_and_eval_1h(&add, h, act, best_c19_beta, best_c19_t, best_c19_dt, best_c19_scale, lr, 200, seed, false);

            // 2 hidden layers
            let (conv2, acc2) = train_and_eval_2h(&add, h, act, best_c19_beta, best_c19_t, best_c19_dt, best_c19_scale, lr, 200, seed);

            log.w(&format!("  {} seed={}: 1-layer acc={:.3} conv={:.3} | 2-layer acc={:.3} conv={:.3} | depth_delta={:+.3}",
                act.name(), seed, r1.accuracy, r1.conv_free, acc2, conv2, acc2 - r1.accuracy));

            let rho_val = match act { Act::C19(r) => r as i32, _ => 0 };
            writeln!(tsv, "3\tADD\t{}\t{}\t{}\t{:.2}\t{}\t{:.2}\t{:.1}\t{}\t{:.4}\t-\t{:.4}\t-\t-\t-\t-\t-\t-\t-",
                act.name(), rho_val, h, best_c19_beta, best_c19_t, best_c19_dt, best_c19_scale, seed,
                r1.conv_free, r1.accuracy).ok();
            writeln!(tsv, "3\tADD_2L\t{}\t{}\t{}\t{:.2}\t{}\t{:.2}\t{:.1}\t{}\t{:.4}\t-\t{:.4}\t-\t-\t-\t-\t-\t-\t-",
                act.name(), rho_val, h, best_c19_beta, best_c19_t, best_c19_dt, best_c19_scale, seed,
                conv2, acc2).ok();
        }
    }

    // ========================================================
    // FINAL VERDICT
    // ========================================================
    log.w("");
    log.w("=======================================");
    log.w("  FINAL VERDICT");
    log.w("=======================================");
    log.w(&format!("  C19 convergence (Stage 1): {:.3}", best_c19_conv));
    log.w(&format!("  C19 accuracy (Stage 2):    {:.3}", best_c19_acc));
    log.w(&format!("  tanh accuracy (baseline):  {:.3}", best_tanh_acc));
    log.w(&format!("  ReLU accuracy (baseline):  {:.3}", best_relu_acc));
    log.w(&format!("  Verdict: {}", verdict));
    log.w(&format!("  Total time: {:.2}s", t0.elapsed().as_secs_f64()));
    log.w("=======================================");
}
