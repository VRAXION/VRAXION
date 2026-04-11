//! Byte-Level Cortex — EP + C19 with raw binary input
//!
//! THE QUESTION: Does EP+C19 work when input is raw bits (0/1),
//! not normalized floats?
//!
//! 4 tasks, increasing difficulty:
//!   T1: PARITY    — is the byte even? (trivial: just bit 0)
//!   T2: POPCOUNT  — are more than 4 bits set? (needs counting)
//!   T3: NIBBLE    — classify by nibble pattern (4-class)
//!   T4: BYTE OP   — 2 bytes in, 1 byte out (full byte compute)
//!
//! All inputs are BINARY: 0 or 1. No normalization.
//! All 256 byte values tested exhaustively.
//!
//! Run: cargo run --example cortex_byte --release

// ============================================================
// C19 + RNG + EP (same core as before)
// ============================================================

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
impl Act {
    fn apply(self, x: f32) -> f32 { c19(x, self.0) }
}

// ============================================================
// EP Network (reused)
// ============================================================

struct EpNet {
    w1: Vec<f32>, w2: Vec<f32>, b1: Vec<f32>, b2: Vec<f32>,
    in_dim: usize, h_dim: usize, out_dim: usize,
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
        }
    }
}

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

#[allow(dead_code)]
fn predict(x: &[f32], net: &EpNet, t: usize, dt: f32, act: Act, od: usize) -> usize {
    let dy = vec![0.0f32; od];
    let (_, so) = settle(x, &dy, net, t, dt, act, 0.0);
    let acts: Vec<f32> = so.iter().map(|s| act.apply(*s)).collect();
    if acts.iter().any(|v| v.is_nan()) { return 0; }
    acts.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap()
}

fn predict_bits(x: &[f32], net: &EpNet, t: usize, dt: f32, act: Act) -> Vec<u8> {
    let dy = vec![0.0f32; net.out_dim];
    let (_, so) = settle(x, &dy, net, t, dt, act, 0.0);
    so.iter().map(|s| {
        let a = act.apply(*s);
        if a.is_nan() { 0u8 } else if a > 0.5 { 1u8 } else { 0u8 }
    }).collect()
}

fn train_ep(net: &mut EpNet, data: &[(Vec<f32>, Vec<f32>)],
            t: usize, dt: f32, act: Act, beta: f32, lr: f32,
            epochs: usize, rng: &mut Rng, log_interval: usize)
{
    let mut idx: Vec<usize> = (0..data.len()).collect();
    for ep in 0..epochs {
        let lr_e = if ep < 20 { lr * (ep as f32 + 1.0) / 20.0 } else { lr };
        rng.shuffle(&mut idx);
        for &i in &idx {
            let (x, y) = &data[i];
            let (sfh, sfo) = settle(x, y, net, t, dt, act, 0.0);
            let mut sh = sfh.clone(); let mut so = sfo.clone();
            for _ in 0..t { let (h2, o2) = settle_step(&sh, &so, x, net, dt, act, beta, y); sh = h2; so = o2; }
            let ib = 1.0 / beta;
            for j in 0..net.h_dim {
                let an = act.apply(sh[j]); let af = act.apply(sfh[j]);
                for ii in 0..net.in_dim { net.w1[j * net.in_dim + ii] += lr_e * ib * (an * x[ii] - af * x[ii]); }
                net.b1[j] += lr_e * ib * (an - af);
            }
            for k in 0..net.out_dim {
                let aon = act.apply(so[k]); let aof = act.apply(sfo[k]);
                for j in 0..net.h_dim { net.w2[k * net.h_dim + j] += lr_e * ib * (aon * act.apply(sh[j]) - aof * act.apply(sfh[j])); }
                net.b2[k] += lr_e * ib * (aon - aof);
            }
        }
        if ep % log_interval == 0 || ep == epochs - 1 {
            let mut ok = 0;
            for (x, y) in data {
                let out = predict_bits(x, net, t, dt, act);
                let target: Vec<u8> = y.iter().map(|&v| if v > 0.5 { 1u8 } else { 0u8 }).collect();
                if out == target { ok += 1; }
            }
            println!("      Epoch {:4} — {:.1}% ({}/{})", ep, ok as f32 / data.len() as f32 * 100.0, ok, data.len());
        }
    }
}

// ============================================================
// Byte → bits (THE key function)
// ============================================================

fn byte_to_bits(b: u8) -> Vec<f32> {
    (0..8).map(|i| if b & (1 << i) != 0 { 1.0 } else { 0.0 }).collect()
}

#[allow(dead_code)]
fn bits_to_byte(bits: &[u8]) -> u8 {
    bits.iter().enumerate().map(|(i, &b)| (b & 1) << i).sum()
}

// ============================================================
// T1: PARITY — is byte even? (bit 0 = 0 → even)
// ============================================================

fn gen_t1() -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..=255u8).map(|b| {
        let input = byte_to_bits(b);
        let parity = (b & 1) as f32; // 0=even, 1=odd
        (input, vec![parity])
    }).collect()
}

// ============================================================
// T2: POPCOUNT > 4 — more than 4 bits set?
// ============================================================

fn gen_t2() -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..=255u8).map(|b| {
        let input = byte_to_bits(b);
        let pop = b.count_ones();
        let label = if pop > 4 { 1.0 } else { 0.0 };
        (input, vec![label])
    }).collect()
}

// ============================================================
// T3: NIBBLE CLASS — classify by low nibble pattern (4 class)
//   class 0: low nibble < 4
//   class 1: low nibble 4-7
//   class 2: low nibble 8-11
//   class 3: low nibble 12-15
// ============================================================

fn gen_t3() -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..=255u8).map(|b| {
        let input = byte_to_bits(b);
        let nibble = b & 0x0F;
        let class = (nibble / 4) as usize;
        let mut target = vec![0.0f32; 4];
        target[class] = 1.0;
        (input, target)
    }).collect()
}

// ============================================================
// T4: BYTE COMPUTE — 2 bytes in, 1 byte out
//   byte[0] = operand, byte[1] = modifier
//   output = (byte[0] + byte[1]) & 0xFF  (wrapping add)
//   BUT: we only test with small values (0-15) for learnability
//
//   Input: 16 bits (2 bytes)
//   Output: 8 bits (1 byte result)
// ============================================================

fn gen_t4() -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut data = Vec::new();
    // Small values only (4-bit each = 16×16 = 256 combos)
    for a in 0..16u8 {
        for b in 0..16u8 {
            let mut input = byte_to_bits(a);
            input.extend(byte_to_bits(b));
            // Only use low 4 bits of each byte → 8 meaningful bits
            // But we feed all 16 bits (high bits are 0)
            let result = a.wrapping_add(b); // result can be 0-30
            let target = byte_to_bits(result);
            // Only take first 5 bits (max value 30 = 11110)
            let target5: Vec<f32> = target[..5].to_vec();
            data.push((input, target5));
        }
    }
    data
}

// ============================================================
// Frozen int8
// ============================================================

struct FrozenNet {
    w1: Vec<i8>, w2: Vec<i8>, b1: Vec<i8>, b2: Vec<i8>,
    s1: f32, in_dim: usize, h_dim: usize, out_dim: usize,
}

impl FrozenNet {
    fn from(net: &EpNet) -> Self {
        let m1 = net.w1.iter().chain(net.b1.iter()).map(|w| w.abs()).fold(0.0f32, f32::max).max(1e-8);
        let m2 = net.w2.iter().chain(net.b2.iter()).map(|w| w.abs()).fold(0.0f32, f32::max).max(1e-8);
        let s1 = 127.0 / m1;
        let s2 = 127.0 / m2;
        FrozenNet {
            w1: net.w1.iter().map(|w| (w * s1).round().clamp(-127.0, 127.0) as i8).collect(),
            b1: net.b1.iter().map(|b| (b * s1).round().clamp(-127.0, 127.0) as i8).collect(),
            w2: net.w2.iter().map(|w| (w * s2).round().clamp(-127.0, 127.0) as i8).collect(),
            b2: net.b2.iter().map(|b| (b * s2).round().clamp(-127.0, 127.0) as i8).collect(),
            s1, in_dim: net.in_dim, h_dim: net.h_dim, out_dim: net.out_dim,
        }
    }

    fn infer_bits(&self, x: &[f32], act: Act) -> Vec<u8> {
        // Input is already 0/1, map to i8 as 0 or 127
        let xi: Vec<i8> = x.iter().map(|&v| if v > 0.5 { 127i8 } else { 0i8 }).collect();
        let mut hv = vec![0i32; self.h_dim];
        for j in 0..self.h_dim {
            let mut s = self.b1[j] as i32 * 127;
            for i in 0..self.in_dim { s += self.w1[j * self.in_dim + i] as i32 * xi[i] as i32; }
            hv[j] = s;
        }
        let ha: Vec<i8> = hv.iter().map(|&v| {
            let f = v as f32 / (127.0 * self.s1);
            let a = act.apply(f);
            if a.is_nan() { 0 } else { (a * 127.0).round().clamp(-127.0, 127.0) as i8 }
        }).collect();
        let mut ov = vec![0i32; self.out_dim];
        for k in 0..self.out_dim {
            let mut s = self.b2[k] as i32 * 127;
            for j in 0..self.h_dim { s += self.w2[k * self.h_dim + j] as i32 * ha[j] as i32; }
            ov[k] = s;
        }
        // Threshold at 0 for binary output
        ov.iter().map(|&v| if v > 0 { 1u8 } else { 0u8 }).collect()
    }
}

// ============================================================
// Run one task
// ============================================================

fn run_task(name: &str, data: &[(Vec<f32>, Vec<f32>)],
            h_dim: usize, t: usize, dt: f32, act: Act,
            beta: f32, lr: f32, epochs: usize, seeds: &[u64]) {
    println!("  ━━━ {} ({}→{}→{}, {} samples) ━━━",
        name, data[0].0.len(), h_dim, data[0].1.len(), data.len());

    for &seed in seeds {
        let mut rng = Rng::new(seed);
        let in_dim = data[0].0.len();
        let out_dim = data[0].1.len();
        let mut net = EpNet::new(in_dim, h_dim, out_dim, &mut rng);

        println!("    seed={}:", seed);
        train_ep(&mut net, data, t, dt, act, beta, lr, epochs, &mut rng, 200);

        // Float eval (exact bit match)
        let mut ok = 0;
        for (x, y) in data {
            let out = predict_bits(x, &net, t, dt, act);
            let target: Vec<u8> = y.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).collect();
            if out == target { ok += 1; }
        }
        let float_acc = ok as f32 / data.len() as f32;
        println!("      FLOAT exact: {:.1}% ({}/{})", float_acc * 100.0, ok, data.len());

        // Per-bit accuracy (more forgiving)
        let mut bit_ok = 0usize;
        let mut bit_total = 0usize;
        for (x, y) in data {
            let out = predict_bits(x, &net, t, dt, act);
            let target: Vec<u8> = y.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).collect();
            for (o, t) in out.iter().zip(target.iter()) {
                bit_total += 1;
                if o == t { bit_ok += 1; }
            }
        }
        let bit_acc = bit_ok as f32 / bit_total as f32;
        println!("      FLOAT per-bit: {:.1}% ({}/{})", bit_acc * 100.0, bit_ok, bit_total);

        // Frozen
        let frozen = FrozenNet::from(&net);
        let mut ok_f = 0;
        for (x, y) in data {
            let out = frozen.infer_bits(x, act);
            let target: Vec<u8> = y.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).collect();
            if out == target { ok_f += 1; }
        }
        let frozen_acc = ok_f as f32 / data.len() as f32;
        println!("      INT8  exact: {:.1}% ({}/{}), freeze loss: {:.1}pp",
            frozen_acc * 100.0, ok_f, data.len(), (float_acc - frozen_acc) * 100.0);
        println!();
    }
}

// ============================================================
// MAIN
// ============================================================

fn main() {
    println!("================================================================");
    println!("  BYTE-LEVEL CORTEX — Raw Binary Input Kill Test");
    println!("  Input: 0/1 bits (NOT normalized floats)");
    println!("  Does EP+C19 work with real byte-level data?");
    println!("================================================================\n");

    let act = Act(8.0);
    let seeds = [42u64, 123, 7];

    // T1: PARITY (trivial)
    println!("── T1: PARITY (is byte even?) ──");
    println!("   Input: 8 bits, Output: 1 bit, trivial (just bit 0)");
    let t1 = gen_t1();
    run_task("Parity", &t1, 8, 30, 0.5, act, 0.5, 0.01, 500, &seeds);

    // T2: POPCOUNT > 4 (needs counting)
    println!("── T2: POPCOUNT > 4 (are >4 bits set?) ──");
    println!("   Input: 8 bits, Output: 1 bit, requires counting");
    let t2 = gen_t2();
    run_task("Popcount>4", &t2, 16, 50, 0.5, act, 0.5, 0.005, 800, &seeds);

    // T3: NIBBLE CLASS (4-class pattern)
    println!("── T3: NIBBLE CLASS (low nibble → 4 classes) ──");
    println!("   Input: 8 bits, Output: 4 bits (one-hot), pattern recognition");
    let t3 = gen_t3();
    run_task("NibbleClass", &t3, 16, 50, 0.5, act, 0.5, 0.005, 800, &seeds);

    // T4: BYTE ADD (2 bytes → result bits)
    println!("── T4: BYTE ADD (4-bit a + 4-bit b → 5-bit result) ──");
    println!("   Input: 16 bits, Output: 5 bits, actual computation");
    let t4 = gen_t4();
    run_task("ByteAdd", &t4, 32, 60, 0.4, act, 0.5, 0.003, 1000, &seeds);

    // ================================================================
    // Summary
    // ================================================================
    println!("================================================================");
    println!("  SUMMARY");
    println!("================================================================");
    println!("  T1 Parity:    trivial (bit 0 check)");
    println!("  T2 Popcount:  counting (8 bits → threshold)");
    println!("  T3 Nibble:    pattern recognition (4-class)");
    println!("  T4 ByteAdd:   actual computation (a+b → result bits)");
    println!();
    println!("  If T1-T3 work: binary byte I/O is validated");
    println!("  If T4 works:   cortex can do byte-level compute");
    println!("  If T4 fails:   compute stays in ALU (which is fine!)");
    println!("================================================================");
}
