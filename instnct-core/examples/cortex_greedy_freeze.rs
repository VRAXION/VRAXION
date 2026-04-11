//! Greedy Freeze — Freeze neurons one-by-one with EP retraining
//!
//! THE QUESTION: Can we fix the catastrophic freeze problem by
//! freezing neurons incrementally instead of all at once?
//!
//! Method:
//!   1. Train full network (EP, float) → ~100%
//!   2. For each hidden neuron:
//!      a. Freeze it to int8 (lock its weights)
//!      b. Re-train remaining float neurons (EP, 100 epochs)
//!      c. Measure accuracy
//!   3. Finally freeze output layer
//!   4. If accuracy < target: add a neuron and repeat
//!
//! Test on: T3 NIBBLE CLASS (where global freeze = 0%)
//!          T2 POPCOUNT (where global freeze = ~83%)
//!          T4 BYTE ADD (where global freeze = 0%)
//!
//! Run: cargo run --example cortex_greedy_freeze --release

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

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
        for i in (1..v.len()).rev() { let j = (self.next() as usize) % (i + 1); v.swap(i, j); }
    }
}

#[derive(Clone, Copy)]
struct Act(f32);
impl Act { fn apply(self, x: f32) -> f32 { c19(x, self.0) } }

// ============================================================
// EP Network with per-neuron freeze support
// ============================================================

#[derive(Clone)]
struct EpNet {
    w1: Vec<f32>, w2: Vec<f32>, b1: Vec<f32>, b2: Vec<f32>,
    in_dim: usize, h_dim: usize, out_dim: usize,
    // Freeze state: per-neuron
    frozen_h: Vec<bool>,    // which hidden neurons are frozen
    frozen_w1: Vec<i8>,     // int8 weights for frozen hidden neurons
    frozen_b1: Vec<i8>,     // int8 biases for frozen hidden neurons
    scale_h: Vec<f32>,      // per-neuron quantization scale
    frozen_out: bool,       // output layer frozen?
    frozen_w2: Vec<i8>,
    frozen_b2: Vec<i8>,
    scale_out: f32,
}

impl EpNet {
    fn new(in_dim: usize, h_dim: usize, out_dim: usize, rng: &mut Rng) -> Self {
        let s1 = (2.0 / in_dim as f32).sqrt();
        let s2 = (2.0 / h_dim as f32).sqrt();
        EpNet {
            w1: (0..h_dim * in_dim).map(|_| rng.range_f32(-s1, s1)).collect(),
            w2: (0..out_dim * h_dim).map(|_| rng.range_f32(-s2, s2)).collect(),
            b1: vec![0.0; h_dim], b2: vec![0.0; out_dim],
            in_dim, h_dim, out_dim,
            frozen_h: vec![false; h_dim],
            frozen_w1: vec![0i8; h_dim * in_dim],
            frozen_b1: vec![0i8; h_dim],
            scale_h: vec![1.0; h_dim],
            frozen_out: false,
            frozen_w2: vec![0i8; out_dim * h_dim],
            frozen_b2: vec![0i8; out_dim],
            scale_out: 1.0,
        }
    }

    fn freeze_hidden(&mut self, j: usize) {
        // Find max abs for this neuron's weights
        let mut max_val = self.b1[j].abs();
        for i in 0..self.in_dim {
            max_val = max_val.max(self.w1[j * self.in_dim + i].abs());
        }
        let scale = if max_val > 1e-8 { 127.0 / max_val } else { 1.0 };

        // Quantize this neuron's weights
        for i in 0..self.in_dim {
            self.frozen_w1[j * self.in_dim + i] = (self.w1[j * self.in_dim + i] * scale)
                .round().clamp(-127.0, 127.0) as i8;
            // Write back the quantized value to float (so EP sees the true quantized weight)
            self.w1[j * self.in_dim + i] = self.frozen_w1[j * self.in_dim + i] as f32 / scale;
        }
        self.frozen_b1[j] = (self.b1[j] * scale).round().clamp(-127.0, 127.0) as i8;
        self.b1[j] = self.frozen_b1[j] as f32 / scale;

        self.scale_h[j] = scale;
        self.frozen_h[j] = true;
    }

    fn freeze_output(&mut self) {
        let mut max_val = 0.0f32;
        for &w in &self.w2 { max_val = max_val.max(w.abs()); }
        for &b in &self.b2 { max_val = max_val.max(b.abs()); }
        let scale = if max_val > 1e-8 { 127.0 / max_val } else { 1.0 };

        for i in 0..self.w2.len() {
            self.frozen_w2[i] = (self.w2[i] * scale).round().clamp(-127.0, 127.0) as i8;
            self.w2[i] = self.frozen_w2[i] as f32 / scale;
        }
        for i in 0..self.b2.len() {
            self.frozen_b2[i] = (self.b2[i] * scale).round().clamp(-127.0, 127.0) as i8;
            self.b2[i] = self.frozen_b2[i] as f32 / scale;
        }
        self.scale_out = scale;
        self.frozen_out = true;
    }

    fn n_frozen(&self) -> usize {
        self.frozen_h.iter().filter(|&&f| f).count()
    }

    #[allow(dead_code)]
    fn all_frozen(&self) -> bool {
        self.frozen_h.iter().all(|&f| f) && self.frozen_out
    }
}

// ============================================================
// EP Settle + Train (respects frozen neurons)
// ============================================================

fn settle_step(sh: &[f32], so: &[f32], x: &[f32], net: &EpNet,
               dt: f32, act: Act, beta: f32, y: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (id, h, od) = (net.in_dim, net.h_dim, net.out_dim);
    let mut nh = vec![0.0f32; h];
    for j in 0..h {
        let mut d = net.b1[j];
        for i in 0..id { d += net.w1[j * id + i] * x[i]; }
        for k in 0..od { d += net.w2[k * h + j] * act.apply(so[k]); }
        nh[j] = sh[j] + dt * (-sh[j] + d);
    }
    let mut no = vec![0.0f32; od];
    for k in 0..od {
        let mut d = net.b2[k];
        for j in 0..h { d += net.w2[k * h + j] * act.apply(sh[j]); }
        no[k] = so[k] + dt * (-so[k] + d + beta * (y[k] - act.apply(so[k])));
    }
    (nh, no)
}

fn settle(x: &[f32], y: &[f32], net: &EpNet, t: usize, dt: f32, act: Act, beta: f32) -> (Vec<f32>, Vec<f32>) {
    let mut sh = vec![0.0f32; net.h_dim];
    let mut so = vec![0.0f32; net.out_dim];
    for _ in 0..t { let (h2, o2) = settle_step(&sh, &so, x, net, dt, act, beta, y); sh = h2; so = o2; }
    (sh, so)
}

fn predict_bits(x: &[f32], net: &EpNet, t: usize, dt: f32, act: Act) -> Vec<u8> {
    let dy = vec![0.0f32; net.out_dim];
    let (_, so) = settle(x, &dy, net, t, dt, act, 0.0);
    so.iter().map(|s| { let a = act.apply(*s); if a.is_nan() { 0 } else if a > 0.5 { 1 } else { 0 } }).collect()
}

fn eval_exact(net: &EpNet, data: &[(Vec<f32>, Vec<f32>)], t: usize, dt: f32, act: Act) -> (f32, usize, usize) {
    let mut ok = 0;
    for (x, y) in data {
        let out = predict_bits(x, net, t, dt, act);
        let tgt: Vec<u8> = y.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).collect();
        if out == tgt { ok += 1; }
    }
    (ok as f32 / data.len() as f32, ok, data.len())
}

fn train_ep_frozen(net: &mut EpNet, data: &[(Vec<f32>, Vec<f32>)],
                   t: usize, dt: f32, act: Act, beta: f32, lr: f32,
                   epochs: usize, rng: &mut Rng) {
    let mut idx: Vec<usize> = (0..data.len()).collect();
    for ep in 0..epochs {
        let lr_e = if ep < 10 { lr * (ep as f32 + 1.0) / 10.0 } else { lr };
        rng.shuffle(&mut idx);
        for &i in &idx {
            let (x, y) = &data[i];
            let (sfh, sfo) = settle(x, y, net, t, dt, act, 0.0);
            let mut sh = sfh.clone(); let mut so = sfo.clone();
            for _ in 0..t { let (h2, o2) = settle_step(&sh, &so, x, net, dt, act, beta, y); sh = h2; so = o2; }
            let ib = 1.0 / beta;

            // Update W1/b1 ONLY for non-frozen hidden neurons
            for j in 0..net.h_dim {
                if net.frozen_h[j] { continue; } // SKIP frozen
                let an = act.apply(sh[j]); let af = act.apply(sfh[j]);
                for ii in 0..net.in_dim {
                    net.w1[j * net.in_dim + ii] += lr_e * ib * (an * x[ii] - af * x[ii]);
                }
                net.b1[j] += lr_e * ib * (an - af);
            }

            // Update W2/b2 ONLY if output not frozen
            if !net.frozen_out {
                for k in 0..net.out_dim {
                    let aon = act.apply(so[k]); let aof = act.apply(sfo[k]);
                    for j in 0..net.h_dim {
                        net.w2[k * net.h_dim + j] += lr_e * ib * (aon * act.apply(sh[j]) - aof * act.apply(sfh[j]));
                    }
                    net.b2[k] += lr_e * ib * (aon - aof);
                }
            }
        }
    }
}

// ============================================================
// Byte encoding
// ============================================================

fn byte_to_bits(b: u8) -> Vec<f32> {
    (0..8).map(|i| if b & (1 << i) != 0 { 1.0 } else { 0.0 }).collect()
}

// ============================================================
// Tasks
// ============================================================

fn gen_popcount() -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..=255u8).map(|b| {
        let input = byte_to_bits(b);
        let label = if b.count_ones() > 4 { 1.0 } else { 0.0 };
        (input, vec![label])
    }).collect()
}

fn gen_nibble() -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..=255u8).map(|b| {
        let input = byte_to_bits(b);
        let class = ((b & 0x0F) / 4) as usize;
        let mut target = vec![0.0f32; 4];
        target[class] = 1.0;
        (input, target)
    }).collect()
}

fn gen_byte_add() -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut data = Vec::new();
    for a in 0..16u8 {
        for b in 0..16u8 {
            let mut input = byte_to_bits(a);
            input.extend(byte_to_bits(b));
            let result = a.wrapping_add(b);
            let target: Vec<f32> = (0..5).map(|i| if result & (1 << i) != 0 { 1.0 } else { 0.0 }).collect();
            data.push((input, target));
        }
    }
    data
}

// ============================================================
// Greedy freeze pipeline
// ============================================================

fn greedy_freeze(
    name: &str, data: &[(Vec<f32>, Vec<f32>)],
    h_dim: usize, t: usize, dt: f32, act: Act,
    beta: f32, lr: f32, initial_epochs: usize, retrain_epochs: usize,
    seed: u64,
) {
    println!("  ━━━ {} (in={} h={} out={}) ━━━", name, data[0].0.len(), h_dim, data[0].1.len());

    let mut rng = Rng::new(seed);
    let mut net = EpNet::new(data[0].0.len(), h_dim, data[0].1.len(), &mut rng);

    // Phase 1: Full float training
    println!("    Phase 1: Full float training ({} epochs)...", initial_epochs);
    train_ep_frozen(&mut net, data, t, dt, act, beta, lr, initial_epochs, &mut rng);
    let (acc0, ok0, n0) = eval_exact(&net, data, t, dt, act);
    println!("    Float baseline: {:.1}% ({}/{})", acc0 * 100.0, ok0, n0);

    // Global freeze comparison
    let mut net_global = net.clone();
    for j in 0..h_dim { net_global.freeze_hidden(j); }
    net_global.freeze_output();
    let (gacc, gok, gn) = eval_exact(&net_global, data, t, dt, act);
    println!("    Global freeze:  {:.1}% ({}/{}) — the old way", gacc * 100.0, gok, gn);
    println!();

    // Phase 2: Greedy freeze, one neuron at a time
    println!("    Phase 2: Greedy freeze (neuron-by-neuron)...");
    for j in 0..h_dim {
        net.freeze_hidden(j);
        let (acc_pre, _, _) = eval_exact(&net, data, t, dt, act);

        // Retrain remaining neurons
        train_ep_frozen(&mut net, data, t, dt, act, beta, lr * 0.5, retrain_epochs, &mut rng);
        let (acc_post, ok_post, _) = eval_exact(&net, data, t, dt, act);

        println!("      Freeze h[{:2}] → pre={:.1}% → retrain → post={:.1}% ({}/{}) [frozen: {}/{}]",
            j, acc_pre * 100.0, acc_post * 100.0, ok_post, data.len(), net.n_frozen(), h_dim);
    }

    let (acc_h, ok_h, _) = eval_exact(&net, data, t, dt, act);
    println!("    All hidden frozen: {:.1}% ({}/{})", acc_h * 100.0, ok_h, data.len());

    // Phase 3: Freeze output
    println!("    Phase 3: Freeze output layer...");
    net.freeze_output();
    let (acc_final, ok_final, _) = eval_exact(&net, data, t, dt, act);
    println!("    FULLY FROZEN: {:.1}% ({}/{})", acc_final * 100.0, ok_final, data.len());

    println!();
    println!("    ┌─────────────────────────────────────────┐");
    println!("    │ Float baseline:    {:>5.1}% ({:>3}/{})   │", acc0 * 100.0, ok0, n0);
    println!("    │ Global freeze:     {:>5.1}% ({:>3}/{})   │", gacc * 100.0, gok, gn);
    println!("    │ Greedy freeze:     {:>5.1}% ({:>3}/{})   │", acc_final * 100.0, ok_final, data.len());
    println!("    │ Improvement:       {:>+5.1}pp vs global  │", (acc_final - gacc) * 100.0);
    println!("    └─────────────────────────────────────────┘");
    println!();
}

// ============================================================
// MAIN
// ============================================================

fn main() {
    println!("================================================================");
    println!("  GREEDY FREEZE — Neuron-by-Neuron with EP Retraining");
    println!("  Fix the catastrophic freeze problem");
    println!("================================================================\n");

    let act = Act(8.0);
    let seed = 123u64; // best seed from previous tests

    // T2: POPCOUNT (global freeze was ~83%)
    println!("── T2: POPCOUNT >4 ──");
    let t2 = gen_popcount();
    greedy_freeze("Popcount", &t2, 16, 50, 0.5, act, 0.5, 0.005, 800, 100, seed);

    // T3: NIBBLE CLASS (global freeze was 0%!)
    println!("── T3: NIBBLE CLASS ──");
    let t3 = gen_nibble();
    greedy_freeze("Nibble", &t3, 16, 50, 0.5, act, 0.5, 0.005, 800, 100, seed);

    // T4: BYTE ADD (global freeze was 0%!)
    println!("── T4: BYTE ADD ──");
    let t4 = gen_byte_add();
    greedy_freeze("ByteAdd", &t4, 32, 60, 0.4, act, 0.5, 0.003, 1000, 150, seed);

    println!("================================================================");
    println!("  VERDICT");
    println!("================================================================");
    println!("  If greedy >> global: neuron-by-neuron freeze WORKS");
    println!("  If greedy ≈ global: the problem is deeper than quantization order");
    println!("================================================================");
}
