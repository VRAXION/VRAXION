//! MLP → LutGate Freeze — Standard feedforward backprop,
//! freeze each neuron to exhaustive LUT one by one.
//!
//! No EP, no settle — plain MLP with C19 activation.
//! Each neuron baked to LUT = exhaustive verified by construction.
//!
//! Run: cargo run --example mlp_lutgate_freeze --release

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
    let n = x.floor(); let t = x - n;
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * (1.0 - 2.0 * t) + rho * 2.0 * t * (1.0 - t)
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

fn byte_to_bits(b: u8) -> Vec<f32> {
    (0..8).map(|i| if b & (1 << i) != 0 { 1.0 } else { 0.0 }).collect()
}

// ============================================================
// MLP with per-neuron freeze support
// ============================================================

const RHO: f32 = 8.0;
const THR: f32 = 0.5;

#[derive(Clone)]
struct Mlp {
    w1: Vec<f32>, b1: Vec<f32>,  // h × in
    w2: Vec<f32>, b2: Vec<f32>,  // out × h
    ind: usize, h: usize, outd: usize,
    // LutGate freeze state
    frozen_h: Vec<bool>,
    h_luts: Vec<Vec<u8>>,  // per hidden neuron: 2^ind entries
}

impl Mlp {
    fn new(ind: usize, h: usize, outd: usize, rng: &mut Rng) -> Self {
        let s1 = (2.0 / ind as f32).sqrt();
        let s2 = (2.0 / h as f32).sqrt();
        Mlp {
            w1: (0..h * ind).map(|_| rng.range_f32(-s1, s1)).collect(),
            b1: vec![0.0; h],
            w2: (0..outd * h).map(|_| rng.range_f32(-s2, s2)).collect(),
            b2: vec![0.0; outd],
            ind, h, outd,
            frozen_h: vec![false; h],
            h_luts: vec![vec![]; h],
        }
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // Returns: (h_pre, h_act, o_pre, o_act)
        let mut h_pre = vec![0.0f32; self.h];
        let mut h_act = vec![0.0f32; self.h];
        for j in 0..self.h {
            if self.frozen_h[j] {
                // Use LUT
                let mut idx = 0usize;
                for i in 0..self.ind {
                    if x[i] > 0.5 { idx |= 1 << i; }
                }
                h_act[j] = self.h_luts[j][idx] as f32;
                h_pre[j] = h_act[j]; // doesn't matter, frozen
            } else {
                let mut sum = self.b1[j];
                for i in 0..self.ind { sum += self.w1[j * self.ind + i] * x[i]; }
                h_pre[j] = sum;
                h_act[j] = c19(sum, RHO);
            }
        }

        let mut o_pre = vec![0.0f32; self.outd];
        let mut o_act = vec![0.0f32; self.outd];
        for k in 0..self.outd {
            let mut sum = self.b2[k];
            for j in 0..self.h { sum += self.w2[k * self.h + j] * h_act[j]; }
            o_pre[k] = sum;
            o_act[k] = c19(sum, RHO);
        }

        (h_pre, h_act, o_pre, o_act)
    }

    fn predict_bits(&self, x: &[f32]) -> Vec<u8> {
        let (_, _, _, o_act) = self.forward(x);
        o_act.iter().map(|&v| if v > THR { 1 } else { 0 }).collect()
    }

    fn freeze_neuron(&mut self, j: usize) {
        // Bake hidden neuron j into exhaustive LUT
        let n_entries = 1usize << self.ind;
        let mut lut = vec![0u8; n_entries];
        for pattern in 0..n_entries {
            let mut sum = self.b1[j];
            for i in 0..self.ind {
                let inp = if pattern & (1 << i) != 0 { 1.0f32 } else { 0.0 };
                sum += self.w1[j * self.ind + i] * inp;
            }
            lut[pattern] = if c19(sum, RHO) > THR { 1 } else { 0 };
        }
        self.h_luts[j] = lut;
        self.frozen_h[j] = true;
    }

    fn freeze_all_output(&mut self) -> Vec<Vec<u8>> {
        // Bake each output neuron into LUT over hidden outputs
        // Output neuron has h inputs (binary from frozen hidden)
        let n_entries = 1usize << self.h;
        let mut out_luts = Vec::new();
        for k in 0..self.outd {
            let mut lut = vec![0u8; n_entries];
            for pattern in 0..n_entries {
                let mut sum = self.b2[k];
                for j in 0..self.h {
                    let inp = if pattern & (1 << j) != 0 { 1.0f32 } else { 0.0 };
                    sum += self.w2[k * self.h + j] * inp;
                }
                lut[pattern] = if c19(sum, RHO) > THR { 1 } else { 0 };
            }
            out_luts.push(lut);
        }
        out_luts
    }
}

// ============================================================
// Backprop training (respects frozen neurons)
// ============================================================

fn train_backprop(mlp: &mut Mlp, data: &[(Vec<f32>, Vec<f32>)],
                  lr: f32, epochs: usize, rng: &mut Rng) {
    let mut idx: Vec<usize> = (0..data.len()).collect();

    for _ep in 0..epochs {
        rng.shuffle(&mut idx);
        for &i in &idx {
            let (x, y) = &data[i];
            let (h_pre, h_act, o_pre, o_act) = mlp.forward(x);

            // Output layer gradients
            let mut d_out = vec![0.0f32; mlp.outd];
            for k in 0..mlp.outd {
                let err = o_act[k] - y[k];
                d_out[k] = err * c19_deriv(o_pre[k], RHO);
            }

            // Hidden layer gradients
            let mut d_hid = vec![0.0f32; mlp.h];
            for j in 0..mlp.h {
                if mlp.frozen_h[j] { continue; } // skip frozen
                let mut sum = 0.0f32;
                for k in 0..mlp.outd { sum += d_out[k] * mlp.w2[k * mlp.h + j]; }
                d_hid[j] = sum * c19_deriv(h_pre[j], RHO);
            }

            // Update output weights
            for k in 0..mlp.outd {
                for j in 0..mlp.h {
                    mlp.w2[k * mlp.h + j] -= lr * d_out[k] * h_act[j];
                }
                mlp.b2[k] -= lr * d_out[k];
            }

            // Update hidden weights (skip frozen)
            for j in 0..mlp.h {
                if mlp.frozen_h[j] { continue; }
                for i in 0..mlp.ind {
                    mlp.w1[j * mlp.ind + i] -= lr * d_hid[j] * x[i];
                }
                mlp.b1[j] -= lr * d_hid[j];
            }
        }
    }
}

fn eval_exact(mlp: &Mlp, data: &[(Vec<f32>, Vec<f32>)]) -> (usize, usize) {
    let mut ok = 0;
    for (x, y) in data {
        let pred = mlp.predict_bits(x);
        let tgt: Vec<u8> = y.iter().map(|&v| if v > THR { 1 } else { 0 }).collect();
        if pred == tgt { ok += 1; }
    }
    (ok, data.len())
}

// ============================================================
// Full LutGate pipeline eval (all frozen, pure LUT)
// ============================================================

fn eval_full_lutgate(mlp: &Mlp, out_luts: &[Vec<u8>], data: &[(Vec<f32>, Vec<f32>)]) -> (usize, usize) {
    let mut ok = 0;
    for (x, y) in data {
        // Hidden: LUT lookups
        let mut h_idx = 0usize;
        for i in 0..mlp.ind {
            if x[i] > 0.5 { h_idx |= 1 << i; }
        }
        let mut h_pattern = 0usize;
        for j in 0..mlp.h {
            if mlp.h_luts[j][h_idx] != 0 { h_pattern |= 1 << j; }
        }
        // Output: LUT lookups
        let pred: Vec<u8> = out_luts.iter().map(|lut| lut[h_pattern]).collect();
        let tgt: Vec<u8> = y.iter().map(|&v| if v > THR { 1 } else { 0 }).collect();
        if pred == tgt { ok += 1; }
    }
    (ok, data.len())
}

// ============================================================
// Tasks
// ============================================================

fn gen_t2() -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..=255u8).map(|b| (byte_to_bits(b), vec![if b.count_ones() > 4 { 1.0 } else { 0.0 }])).collect()
}

fn gen_t3() -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..=255u8).map(|b| {
        let class = ((b & 0x0F) / 4) as usize;
        let mut t = vec![0.0f32; 4];
        t[class] = 1.0;
        (byte_to_bits(b), t)
    }).collect()
}

fn gen_t4() -> Vec<(Vec<f32>, Vec<f32>)> {
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
// Run one task with greedy LutGate freeze
// ============================================================

fn run_task(name: &str, data: &[(Vec<f32>, Vec<f32>)],
            h_dim: usize, lr: f32, initial_epochs: usize, retrain_epochs: usize, seed: u64) {
    let ind = data[0].0.len();
    let outd = data[0].1.len();
    println!("  ━━━ {} ({}→{}→{}, {} samples) ━━━", name, ind, h_dim, outd, data.len());

    let mut rng = Rng::new(seed);
    let mut mlp = Mlp::new(ind, h_dim, outd, &mut rng);

    // Phase 1: Full float training
    println!("    Phase 1: Float backprop ({} epochs)...", initial_epochs);
    train_backprop(&mut mlp, data, lr, initial_epochs, &mut rng);
    let (ok0, n0) = eval_exact(&mlp, data);
    println!("    Float baseline: {:.1}% ({}/{})", ok0 as f32 / n0 as f32 * 100.0, ok0, n0);

    // Phase 2: Greedy freeze, neuron by neuron
    println!("    Phase 2: Greedy LutGate freeze (neuron-by-neuron)...");
    for j in 0..h_dim {
        // Freeze neuron j → exhaustive LUT
        mlp.freeze_neuron(j);
        let (ok_pre, _) = eval_exact(&mlp, data);

        // Retrain remaining float neurons
        train_backprop(&mut mlp, data, lr * 0.5, retrain_epochs, &mut rng);
        let (ok_post, _) = eval_exact(&mlp, data);

        let lut_ones: usize = mlp.h_luts[j].iter().map(|&v| v as usize).sum();
        println!("      h[{:2}] → LUT({}→{}/{}) pre={:.1}% → retrain → post={:.1}% [{}/{}]",
            j, mlp.ind, lut_ones, 1 << mlp.ind,
            ok_pre as f32 / data.len() as f32 * 100.0,
            ok_post as f32 / data.len() as f32 * 100.0,
            j + 1, h_dim);
    }

    let (ok_h, _) = eval_exact(&mlp, data);
    println!("    All hidden frozen: {:.1}% ({}/{})", ok_h as f32 / data.len() as f32 * 100.0, ok_h, data.len());

    // Phase 3: Freeze output → full LutGate pipeline
    println!("    Phase 3: Freeze output layer → full LutGate...");
    let out_luts = mlp.freeze_all_output();

    // Exhaustive verify
    let (ok_final, n_final) = eval_full_lutgate(&mlp, &out_luts, data);
    let final_acc = ok_final as f32 / n_final as f32;
    println!("    FULLY FROZEN LutGate: {:.1}% ({}/{})", final_acc * 100.0, ok_final, n_final);

    // Memory
    let h_mem: usize = (0..h_dim).map(|_| 1usize << ind).sum();
    let o_mem: usize = (0..outd).map(|_| 1usize << h_dim).sum();
    println!("    Memory: hidden={}B + output={}B = {}B",
        h_mem, o_mem, h_mem + o_mem);
    println!("    Neurons: {} hidden + {} output = {}", h_dim, outd, h_dim + outd);

    println!();
    println!("    ┌────────────────────────────────────────────┐");
    println!("    │ Float baseline:    {:>5.1}% ({:>3}/{})      │", ok0 as f32/n0 as f32*100.0, ok0, n0);
    println!("    │ All hidden LUT:    {:>5.1}% ({:>3}/{})      │", ok_h as f32/data.len() as f32*100.0, ok_h, data.len());
    println!("    │ FULL LutGate:      {:>5.1}% ({:>3}/{})      │", final_acc*100.0, ok_final, n_final);
    println!("    │ ZERO FLOAT DEPLOY  {:>5}                  │", if final_acc > 0.99 { "✓ YES" } else { "✗ NO" });
    println!("    └────────────────────────────────────────────┘");
    println!();
}

// ============================================================
// MAIN
// ============================================================

fn main() {
    println!("================================================================");
    println!("  MLP → LutGate FREEZE — Neuron-by-Neuron Exhaustive Bake");
    println!("  Standard backprop, C19 activation, greedy LUT freeze");
    println!("================================================================\n");

    let seed = 123u64;

    println!("── T2: POPCOUNT >4 ──");
    let t2 = gen_t2();
    run_task("Popcount", &t2, 12, 0.01, 2000, 200, seed);

    println!("── T3: NIBBLE CLASS ──");
    let t3 = gen_t3();
    run_task("NibbleClass", &t3, 12, 0.01, 2000, 200, seed);

    // T4 with H=12 (so output LUT = 2^12 = 4096, feasible)
    println!("── T4: BYTE ADD (H=12) ──");
    let t4 = gen_t4();
    run_task("ByteAdd", &t4, 12, 0.005, 3000, 300, seed);

    println!("================================================================");
    println!("  VERDICT");
    println!("  If FULL LutGate ≈ Float: greedy freeze WORKS for deployment");
    println!("  Every neuron = exhaustive LUT = verified by construction");
    println!("  Zero float at runtime, pure table lookups");
    println!("================================================================");
}
