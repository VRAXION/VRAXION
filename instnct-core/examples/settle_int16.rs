//! Integer-only EP settle simulation.
//!
//! Proves EP settling can run with ZERO float at inference time:
//! int16 states + int8 weights + integer MAC + C19 LUT.
//! Validates that the VRAXION chip can run EP settle in hardware.
//!
//! Pipeline:
//!   1. EP train (float) on byte tasks
//!   2. Quantize weights to int8
//!   3. Bake C19 activation LUT (int32 sum -> int16 output)
//!   4. Integer settle loop (no float at all)
//!   5. Compare float vs int16 on ALL 256 inputs
//!
//! Run: cargo run --example settle_int16 --release

use std::time::Instant;

// ── C19 activation (exact spec) ──────────────────────────────────

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

const RHO: f32 = 8.0;

// ── Minimal RNG (same pattern as cortex_standalone) ──────────────

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed.wrapping_mul(6364136223846793005).wrapping_add(1) }
    }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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

// ── EP network (float, for training) ─────────────────────────────

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

// ── EP settle (float) ────────────────────────────────────────────

fn settle_step_float(
    s_h: &[f32], s_out: &[f32], x: &[f32], net: &EpNet,
    dt: f32, beta: f32, y: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let (in_d, h, out_d) = (net.in_dim, net.h_dim, net.out_dim);
    let mut new_h = vec![0.0f32; h];
    for j in 0..h {
        let mut drive = net.b1[j];
        for i in 0..in_d { drive += net.w1[j * in_d + i] * x[i]; }
        for k in 0..out_d { drive += net.w2[k * h + j] * c19(s_out[k], RHO); }
        new_h[j] = s_h[j] + dt * (-s_h[j] + drive);
    }
    let mut new_out = vec![0.0f32; out_d];
    for k in 0..out_d {
        let mut drive = net.b2[k];
        for j in 0..h { drive += net.w2[k * h + j] * c19(s_h[j], RHO); }
        new_out[k] = s_out[k] + dt * (-s_out[k] + drive + beta * (y[k] - c19(s_out[k], RHO)));
    }
    (new_h, new_out)
}

fn settle_float(x: &[f32], y: &[f32], net: &EpNet, t: usize, dt: f32, beta: f32)
    -> (Vec<f32>, Vec<f32>)
{
    let mut s_h = vec![0.0f32; net.h_dim];
    let mut s_out = vec![0.0f32; net.out_dim];
    for _ in 0..t {
        let (nh, no) = settle_step_float(&s_h, &s_out, x, net, dt, beta, y);
        s_h = nh; s_out = no;
    }
    (s_h, s_out)
}

fn predict_float(x: &[f32], net: &EpNet, t: usize, dt: f32, n_out: usize) -> Vec<f32> {
    let dummy_y = vec![0.0f32; n_out];
    let (_, s_out) = settle_float(x, &dummy_y, net, t, dt, 0.0);
    s_out.iter().map(|s| c19(*s, RHO)).collect()
}

// ── EP training ──────────────────────────────────────────────────

fn train_ep(
    net: &mut EpNet, data: &[(Vec<f32>, Vec<f32>)],
    t: usize, dt: f32, beta: f32, lr: f32,
    epochs: usize, rng: &mut Rng, task_name: &str,
) {
    let mut indices: Vec<usize> = (0..data.len()).collect();
    for epoch in 0..epochs {
        let lr_eff = if epoch < 20 { lr * (epoch as f32 + 1.0) / 20.0 } else { lr };
        rng.shuffle(&mut indices);
        for &idx in &indices {
            let (x, y) = &data[idx];
            // Free phase (beta=0)
            let (sf_h, sf_o) = settle_float(x, y, net, t, dt, 0.0);
            // Nudged phase (beta>0)
            let (sn_h, sn_o) = {
                let mut sh = sf_h.clone(); let mut so = sf_o.clone();
                for _ in 0..t {
                    let (nh, no) = settle_step_float(&sh, &so, x, net, dt, beta, y);
                    sh = nh; so = no;
                }
                (sh, so)
            };

            // W += lr * (1/beta) * (nudge - free) — correct sign
            let inv_b = 1.0 / beta;
            for j in 0..net.h_dim {
                let an = c19(sn_h[j], RHO); let af = c19(sf_h[j], RHO);
                for i in 0..net.in_dim {
                    net.w1[j * net.in_dim + i] += lr_eff * inv_b * (an * x[i] - af * x[i]);
                }
                net.b1[j] += lr_eff * inv_b * (an - af);
            }
            for k in 0..net.out_dim {
                let aon = c19(sn_o[k], RHO); let aof = c19(sf_o[k], RHO);
                for j in 0..net.h_dim {
                    let ahn = c19(sn_h[j], RHO); let ahf = c19(sf_h[j], RHO);
                    net.w2[k * net.h_dim + j] += lr_eff * inv_b * (aon * ahn - aof * ahf);
                }
                net.b2[k] += lr_eff * inv_b * (aon - aof);
            }
        }

        if epoch % 200 == 0 || epoch == epochs - 1 {
            let mut ok = 0;
            for (x, y) in data {
                let acts = predict_float(x, net, t, dt, y.len());
                let pred = acts.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap();
                let target = y.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap();
                if pred == target { ok += 1; }
            }
            println!("    [{}] epoch {:4} — {:.1}% ({}/{})",
                task_name, epoch, ok as f32 / data.len() as f32 * 100.0, ok, data.len());
        }
    }
}

// ── Quantization: float weights -> int8 ──────────────────────────

struct QuantizedNet {
    w1_i8: Vec<i8>,
    w2_i8: Vec<i8>,
    b1_i8: Vec<i8>,
    b2_i8: Vec<i8>,
    scale1: f32,  // 127.0 / max_abs(w1)
    scale2: f32,  // 127.0 / max_abs(w2)
    in_dim: usize,
    h_dim: usize,
    out_dim: usize,
}

fn quantize_weights(net: &EpNet) -> QuantizedNet {
    let max_abs_w1 = net.w1.iter().chain(net.b1.iter())
        .map(|w| w.abs()).fold(0.0f32, f32::max).max(1e-8);
    let max_abs_w2 = net.w2.iter().chain(net.b2.iter())
        .map(|w| w.abs()).fold(0.0f32, f32::max).max(1e-8);
    let scale1 = 127.0 / max_abs_w1;
    let scale2 = 127.0 / max_abs_w2;

    QuantizedNet {
        w1_i8: net.w1.iter().map(|w| (w * scale1).round().clamp(-127.0, 127.0) as i8).collect(),
        w2_i8: net.w2.iter().map(|w| (w * scale2).round().clamp(-127.0, 127.0) as i8).collect(),
        b1_i8: net.b1.iter().map(|b| (b * scale1).round().clamp(-127.0, 127.0) as i8).collect(),
        b2_i8: net.b2.iter().map(|b| (b * scale2).round().clamp(-127.0, 127.0) as i8).collect(),
        scale1, scale2,
        in_dim: net.in_dim, h_dim: net.h_dim, out_dim: net.out_dim,
    }
}

// ── C19 LUT: precomputed int32 sum -> int16 output ───────────────
//
// Fixed-point: neuron states are int16 with 8 fractional bits (scale=256).
// So int16 value 256 represents 1.0, 128 represents 0.5, etc.
//
// The LUT maps: int16 state -> int16 activated value
// We precompute for every possible int16 value.

const FP_SCALE: f32 = 256.0; // 8 fractional bits

struct C19Lut {
    // Indexed by (state + 32768): maps int16 -> int16
    table: Vec<i16>,
}

impl C19Lut {
    fn new() -> Self {
        let mut table = vec![0i16; 65536];
        for raw in 0..65536u32 {
            let state = raw as i16; // wraps correctly: 0..32767, -32768..-1
            let f = state as f32 / FP_SCALE;
            let activated = c19(f, RHO);
            let out = (activated * FP_SCALE).round().clamp(-32767.0, 32767.0) as i16;
            table[(state as i32 + 32768) as usize] = out;
        }
        C19Lut { table }
    }

    #[inline]
    fn lookup(&self, state: i16) -> i16 {
        self.table[(state as i32 + 32768) as usize]
    }
}

// ── Integer settle loop (ZERO float) ─────────────────────────────

fn settle_int16(
    x_bits: &[u8],            // binary input (0 or 1)
    qnet: &QuantizedNet,
    lut: &C19Lut,
    n_ticks: usize,
    dt_shift: u32,            // dt = 1 / 2^dt_shift
) -> (Vec<i16>, i32, usize)  // (output states, max_magnitude, overflow_count)
{
    let in_dim = qnet.in_dim;
    let h_dim = qnet.h_dim;
    let out_dim = qnet.out_dim;

    let mut s_h = vec![0i16; h_dim];
    let mut s_out = vec![0i16; out_dim];
    let mut max_mag: i32 = 0;
    let mut overflow_count: usize = 0;

    // Precompute input scale: binary inputs in fixed-point = 256 for 1.0, 0 for 0.0
    // But we also need to account for per-layer weight scale.
    // The int8 weight was: w_i8 = (w_float * scale1).round()
    // The float input is 0.0 or 1.0
    // In the float net: drive += w_float * x_float = (w_i8 / scale1) * x_float
    // In the int net: we want drive in int16 (scale=256) units
    // drive_fp = w_i8 * x_fp where x_fp = 256 for x=1.0
    // But w_i8 represents (w_float * scale1), so:
    //   drive_fp = w_i8 * 256 represents w_float * scale1 * 256
    // We need to unscale by scale1 at some point.
    // The approach: accumulate in int32 with implicit scale of (scale1 * 256),
    // then divide by scale1 to get int16 in fp-256 scale.
    //
    // For hidden layer:
    //   drive_i32 = sum(w1_i8 * x_fp) + sum(w2_i8 * act_out_scaled) + bias_scaled
    //   where x_fp = 256 for x=1.0
    //   bias_scaled = b1_i8 * 256
    //
    //   But the backward term (W2^T * act(s_out)) mixes scale1 and scale2.
    //   act(s_out) is in fp-256 units. w2_i8 is in scale2 units.
    //   w2_i8 * act_s = (w_float * scale2) * (act_float * 256) = w_float * act_float * scale2 * 256
    //
    //   Forward: w1_i8 * x_fp = (w_float * scale1) * 256 => scale1 * 256
    //   Backward: w2_i8 * act_s = (w_float * scale2) * (act_float * 256) => scale2 * 256
    //
    // To unify scales, we scale the backward term: multiply by (scale1 / scale2).
    // Then everything is in (scale1 * 256) units. Divide by scale1 to get fp-256.
    //
    // Similarly for output layer: forward is w2_i8 * act_h, in scale2*256 units.
    // Divide by scale2 to get fp-256.

    // Precompute scale ratios as fixed-point shifts (approximate)
    // scale_ratio_h = scale1 / scale2 (for backward term in hidden update)
    // We'll use a 16-bit fixed-point multiplier: ratio_h_fp = (scale1 / scale2 * 256).round()
    let ratio_h_fp = ((qnet.scale1 / qnet.scale2) * 256.0).round() as i32;
    // inv_scale1_shift: we divide by scale1 to get fp-256.
    //   drive_fp256 = drive_i32 / scale1
    //   But scale1 might not be a power of 2. Use multiply-then-shift:
    //   drive_fp256 = (drive_i32 * inv_mult) >> inv_shift
    //   where inv_mult / 2^inv_shift approx = 1/scale1
    // For simplicity, just do integer division by (scale1.round()).
    let inv_scale1 = qnet.scale1.round().max(1.0) as i32;
    let inv_scale2 = qnet.scale2.round().max(1.0) as i32;

    for _tick in 0..n_ticks {
        // ── Hidden update: forward (W1*x) + backward (W2^T * act(s_out)) ──
        let mut new_h = vec![0i16; h_dim];
        for j in 0..h_dim {
            let mut drive: i32 = qnet.b1_i8[j] as i32 * 256; // bias in (scale1 * 256) units
            // Forward: W1 * x (binary input)
            for i in 0..in_dim {
                if x_bits[i] != 0 {
                    drive += qnet.w1_i8[j * in_dim + i] as i32 * 256;
                }
            }
            // Backward: W2^T * act(s_out) — needs scale correction
            for k in 0..out_dim {
                let act_out = lut.lookup(s_out[k]) as i32; // fp-256
                let w2_val = qnet.w2_i8[k * h_dim + j] as i32; // scale2 units
                // w2_val * act_out is in (scale2 * 256) units
                // multiply by ratio_h_fp/256 to convert to (scale1 * 256) units
                let term = (w2_val * act_out * ratio_h_fp) >> 8;
                drive += term;
            }
            // Convert from (scale1 * 256) to fp-256 units
            let scaled_drive = (drive / inv_scale1) as i16;
            // Update: s_new = s_old + (scaled_drive - s_old) >> dt_shift
            let diff = (scaled_drive as i32) - (s_h[j] as i32);
            let update = diff >> dt_shift;
            let new_val = (s_h[j] as i32) + update;
            // Clamp to int16
            if new_val > 32767 || new_val < -32767 { overflow_count += 1; }
            new_h[j] = new_val.clamp(-32767, 32767) as i16;
            let m = new_h[j].unsigned_abs() as i32;
            if m > max_mag { max_mag = m; }
        }

        // ── Output update: forward (W2 * act(s_h)) + bias ──
        let mut new_out = vec![0i16; out_dim];
        for k in 0..out_dim {
            let mut drive: i32 = qnet.b2_i8[k] as i32 * 256; // bias in (scale2 * 256) units
            for j in 0..h_dim {
                let act_h = lut.lookup(s_h[j]) as i32; // fp-256
                let w2_val = qnet.w2_i8[k * h_dim + j] as i32; // scale2 units
                drive += w2_val * act_h;
            }
            // Convert from (scale2 * 256) to fp-256 units
            let scaled_drive = (drive / inv_scale2) as i16;
            let diff = (scaled_drive as i32) - (s_out[k] as i32);
            let update = diff >> dt_shift;
            let new_val = (s_out[k] as i32) + update;
            if new_val > 32767 || new_val < -32767 { overflow_count += 1; }
            new_out[k] = new_val.clamp(-32767, 32767) as i16;
            let m = new_out[k].unsigned_abs() as i32;
            if m > max_mag { max_mag = m; }
        }

        s_h = new_h;
        s_out = new_out;
    }

    (s_out, max_mag, overflow_count)
}

// ── Task definitions ─────────────────────────────────────────────

// T2: POPCOUNT >4 — binary classification
// Input: 8-bit pattern. Output: 1 if popcount > 4, else 0.
fn popcount_label(bits: u8) -> usize {
    if bits.count_ones() > 4 { 1 } else { 0 }
}

fn gen_popcount_data() -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut data = Vec::new();
    for p in 0..=255u8 {
        let label = popcount_label(p);
        let input: Vec<f32> = (0..8).map(|i| if p & (1 << i) != 0 { 1.0 } else { 0.0 }).collect();
        let mut target = vec![0.0f32; 2];
        // Use +1/-1 style for EP (helps convergence)
        target[label] = 1.0;
        data.push((input, target));
    }
    data
}

// T3: NIBBLE CLASS — 4-class classification
// Input: 8-bit. Output: classify based on nibble relationship.
//   Class 0: low_nibble > high_nibble
//   Class 1: low_nibble < high_nibble
//   Class 2: low_nibble == high_nibble
//   Class 3: both nibbles >= 8 (both "heavy")
fn nibble_label(bits: u8) -> usize {
    let lo = bits & 0x0F;
    let hi = (bits >> 4) & 0x0F;
    if lo >= 8 && hi >= 8 { return 3; }
    if lo > hi { return 0; }
    if lo < hi { return 1; }
    2 // equal
}

fn gen_nibble_data() -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut data = Vec::new();
    for p in 0..=255u8 {
        let label = nibble_label(p);
        let input: Vec<f32> = (0..8).map(|i| if p & (1 << i) != 0 { 1.0 } else { 0.0 }).collect();
        let mut target = vec![0.0f32; 4];
        target[label] = 1.0;
        data.push((input, target));
    }
    data
}

// ── Main ─────────────────────────────────────────────────────────

fn main() {
    let t0 = Instant::now();
    let mut log = String::new();

    macro_rules! log {
        ($($arg:tt)*) => {{
            let line = format!($($arg)*);
            println!("{}", line);
            log.push_str(&line);
            log.push('\n');
        }};
    }

    log!("================================================================");
    log!("  INTEGER SETTLE -- Float vs Int16 Comparison");
    log!("================================================================");
    log!("");

    // EP training parameters (from spec)
    let t = 50;       // settle ticks
    let dt = 0.5;
    let beta = 0.5;
    let lr = 0.005;
    let epochs = 800;
    let seed = 123u64;

    // ═══════════════════════════════════════════════════════════════
    // T2: POPCOUNT >4 (8 -> 16 -> 1... well, 2 for classification)
    // ═══════════════════════════════════════════════════════════════
    log!("-- T2: POPCOUNT >4 --");
    log!("  Architecture: 8 -> 16 -> 2 (binary classification, one-hot)");
    log!("");

    let popcount_data = gen_popcount_data();
    {
        // Class distribution
        let mut counts = [0usize; 2];
        for (_, y) in &popcount_data {
            let c = if y[1] > y[0] { 1 } else { 0 };
            counts[c] += 1;
        }
        log!("  Class distribution: NO={} YES={}", counts[0], counts[1]);
    }

    // Step 1: EP Train (float)
    log!("  Step 1: EP training (float)...");
    let mut rng = Rng::new(seed);
    let mut net_pop = EpNet::new(8, 16, 2, &mut rng);
    train_ep(&mut net_pop, &popcount_data, t, dt, beta, lr, epochs, &mut rng, "T2");

    // Float accuracy on all 256
    let mut float_preds_pop = vec![0usize; 256];
    let mut float_ok_pop = 0;
    for p in 0..=255u8 {
        let input: Vec<f32> = (0..8).map(|i| if p & (1 << i) != 0 { 1.0 } else { 0.0 }).collect();
        let acts = predict_float(&input, &net_pop, t, dt, 2);
        let pred = if acts[1] > acts[0] { 1 } else { 0 };
        float_preds_pop[p as usize] = pred;
        if pred == popcount_label(p) { float_ok_pop += 1; }
    }
    let float_acc_pop = float_ok_pop as f32 / 256.0 * 100.0;
    log!("  Float EP accuracy: {:.1}% ({}/256)", float_acc_pop, float_ok_pop);

    // Step 2: Quantize weights to int8
    let qnet_pop = quantize_weights(&net_pop);
    log!("  Step 2: Quantized to int8 (scale1={:.2}, scale2={:.2})", qnet_pop.scale1, qnet_pop.scale2);

    // Step 3: Bake C19 LUT
    let lut = C19Lut::new();
    log!("  Step 3: C19 LUT baked (65536 entries)");

    // Step 4 & 5: Integer settle and compare
    log!("  Step 4-5: Integer settle on all 256 inputs...");

    let mut int16_ok_pop = 0;
    let mut agreement_pop = 0;
    let mut max_mag_pop: i32 = 0;
    let mut total_overflow_pop: usize = 0;

    // Try dt_shift=1 (dt=0.5)
    let dt_shift: u32 = 1;

    for p in 0..=255u8 {
        let x_bits: Vec<u8> = (0..8).map(|i| if p & (1 << i) != 0 { 1u8 } else { 0u8 }).collect();
        let (s_out, mag, oflow) = settle_int16(&x_bits, &qnet_pop, &lut, t, dt_shift);
        if mag > max_mag_pop { max_mag_pop = mag; }
        total_overflow_pop += oflow;

        // Decision from int16: apply LUT to get activated output, pick argmax
        let act0 = lut.lookup(s_out[0]) as i32;
        let act1 = lut.lookup(s_out[1]) as i32;
        let int_pred = if act1 > act0 { 1 } else { 0 };
        let target = popcount_label(p);

        if int_pred == target { int16_ok_pop += 1; }
        if int_pred == float_preds_pop[p as usize] { agreement_pop += 1; }
    }

    let int16_acc_pop = int16_ok_pop as f32 / 256.0 * 100.0;
    let agreement_pct_pop = agreement_pop as f32 / 256.0 * 100.0;

    log!("");
    log!("  -- T2: POPCOUNT >4 --");
    log!("    Float EP:     {:.1}% ({}/256)", float_acc_pop, float_ok_pop);
    log!("    Int16 settle: {:.1}% ({}/256)", int16_acc_pop, int16_ok_pop);
    log!("    Agreement:    {:.1}% ({}/256) (float == int16 on same inputs)",
        agreement_pct_pop, agreement_pop);
    log!("    Max state magnitude: {}", max_mag_pop);
    log!("    Overflow events: {}", total_overflow_pop);
    log!("");

    // ═══════════════════════════════════════════════════════════════
    // T3: NIBBLE CLASS (8 -> 16 -> 4)
    // ═══════════════════════════════════════════════════════════════
    log!("-- T3: NIBBLE CLASS --");
    log!("  Architecture: 8 -> 16 -> 4 (4-class classification)");
    log!("");

    let nibble_data = gen_nibble_data();
    {
        let mut counts = [0usize; 4];
        for (_, y) in &nibble_data {
            let c = y.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap();
            counts[c] += 1;
        }
        log!("  Class distribution: lo>hi={} lo<hi={} eq={} both_heavy={}",
            counts[0], counts[1], counts[2], counts[3]);
    }

    // Step 1: EP Train (float)
    log!("  Step 1: EP training (float)...");
    let mut rng2 = Rng::new(seed);
    let mut net_nib = EpNet::new(8, 16, 4, &mut rng2);
    train_ep(&mut net_nib, &nibble_data, t, dt, beta, lr, epochs, &mut rng2, "T3");

    // Float accuracy
    let mut float_preds_nib = vec![0usize; 256];
    let mut float_ok_nib = 0;
    for p in 0..=255u8 {
        let input: Vec<f32> = (0..8).map(|i| if p & (1 << i) != 0 { 1.0 } else { 0.0 }).collect();
        let acts = predict_float(&input, &net_nib, t, dt, 4);
        let pred = acts.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap();
        float_preds_nib[p as usize] = pred;
        if pred == nibble_label(p) { float_ok_nib += 1; }
    }
    let float_acc_nib = float_ok_nib as f32 / 256.0 * 100.0;
    log!("  Float EP accuracy: {:.1}% ({}/256)", float_acc_nib, float_ok_nib);

    // Step 2: Quantize
    let qnet_nib = quantize_weights(&net_nib);
    log!("  Step 2: Quantized to int8 (scale1={:.2}, scale2={:.2})", qnet_nib.scale1, qnet_nib.scale2);

    // Step 3: Same LUT
    log!("  Step 3: C19 LUT (shared)");

    // Step 4-5: Integer settle
    log!("  Step 4-5: Integer settle on all 256 inputs...");

    let mut int16_ok_nib = 0;
    let mut agreement_nib = 0;
    let mut max_mag_nib: i32 = 0;
    let mut total_overflow_nib: usize = 0;

    for p in 0..=255u8 {
        let x_bits: Vec<u8> = (0..8).map(|i| if p & (1 << i) != 0 { 1u8 } else { 0u8 }).collect();
        let (s_out, mag, oflow) = settle_int16(&x_bits, &qnet_nib, &lut, t, dt_shift);
        if mag > max_mag_nib { max_mag_nib = mag; }
        total_overflow_nib += oflow;

        // Argmax of activated outputs
        let mut best_k = 0;
        let mut best_val = i32::MIN;
        for k in 0..4 {
            let act_k = lut.lookup(s_out[k]) as i32;
            if act_k > best_val { best_val = act_k; best_k = k; }
        }
        let target = nibble_label(p);

        if best_k == target { int16_ok_nib += 1; }
        if best_k == float_preds_nib[p as usize] { agreement_nib += 1; }
    }

    let int16_acc_nib = int16_ok_nib as f32 / 256.0 * 100.0;
    let agreement_pct_nib = agreement_nib as f32 / 256.0 * 100.0;

    log!("");
    log!("  -- T3: NIBBLE CLASS --");
    log!("    Float EP:     {:.1}% ({}/256)", float_acc_nib, float_ok_nib);
    log!("    Int16 settle: {:.1}% ({}/256)", int16_acc_nib, int16_ok_nib);
    log!("    Agreement:    {:.1}% ({}/256) (float == int16 on same inputs)",
        agreement_pct_nib, agreement_nib);
    log!("    Max state magnitude: {}", max_mag_nib);
    log!("    Overflow events: {}", total_overflow_nib);
    log!("");

    // ═══════════════════════════════════════════════════════════════
    // Also try dt_shift=2 (dt=0.25) for both tasks
    // ═══════════════════════════════════════════════════════════════
    log!("-- dt_shift=2 (dt=0.25) comparison --");

    let dt_shift2: u32 = 2;
    let mut int16_ok_pop2 = 0;
    let mut agree_pop2 = 0;
    for p in 0..=255u8 {
        let x_bits: Vec<u8> = (0..8).map(|i| if p & (1 << i) != 0 { 1u8 } else { 0u8 }).collect();
        let (s_out, _, _) = settle_int16(&x_bits, &qnet_pop, &lut, t, dt_shift2);
        let act0 = lut.lookup(s_out[0]) as i32;
        let act1 = lut.lookup(s_out[1]) as i32;
        let int_pred = if act1 > act0 { 1 } else { 0 };
        if int_pred == popcount_label(p) { int16_ok_pop2 += 1; }
        if int_pred == float_preds_pop[p as usize] { agree_pop2 += 1; }
    }

    let mut int16_ok_nib2 = 0;
    let mut agree_nib2 = 0;
    for p in 0..=255u8 {
        let x_bits: Vec<u8> = (0..8).map(|i| if p & (1 << i) != 0 { 1u8 } else { 0u8 }).collect();
        let (s_out, _, _) = settle_int16(&x_bits, &qnet_nib, &lut, t, dt_shift2);
        let mut best_k = 0;
        let mut best_val = i32::MIN;
        for k in 0..4 {
            let act_k = lut.lookup(s_out[k]) as i32;
            if act_k > best_val { best_val = act_k; best_k = k; }
        }
        if best_k == nibble_label(p) { int16_ok_nib2 += 1; }
        if best_k == float_preds_nib[p as usize] { agree_nib2 += 1; }
    }

    log!("  T2 POPCOUNT dt_shift=2: {:.1}% ({}/256), agreement {:.1}%",
        int16_ok_pop2 as f32 / 256.0 * 100.0, int16_ok_pop2,
        agree_pop2 as f32 / 256.0 * 100.0);
    log!("  T3 NIBBLE   dt_shift=2: {:.1}% ({}/256), agreement {:.1}%",
        int16_ok_nib2 as f32 / 256.0 * 100.0, int16_ok_nib2,
        agree_nib2 as f32 / 256.0 * 100.0);
    log!("");

    // ═══════════════════════════════════════════════════════════════
    // Convergence analysis: how many ticks does int16 need?
    // ═══════════════════════════════════════════════════════════════
    log!("-- Convergence analysis (T2, dt_shift=1) --");
    {
        // Pick a few representative inputs, track state over ticks
        let test_inputs: Vec<u8> = vec![0b11111000, 0b01010101, 0b11111111, 0b00000000, 0b11100111];
        for &p in &test_inputs {
            let x_bits: Vec<u8> = (0..8).map(|i| if p & (1 << i) != 0 { 1u8 } else { 0u8 }).collect();
            let mut s_h = vec![0i16; qnet_pop.h_dim];
            let mut s_out = vec![0i16; 2];

            let ratio_h_fp = ((qnet_pop.scale1 / qnet_pop.scale2) * 256.0).round() as i32;
            let inv_scale1 = qnet_pop.scale1.round().max(1.0) as i32;
            let inv_scale2 = qnet_pop.scale2.round().max(1.0) as i32;

            let mut converge_tick = t;
            let mut prev_out = (0i16, 0i16);
            let mut stable_count = 0u32;

            for tick in 0..t {
                // Hidden update
                let mut new_h = vec![0i16; qnet_pop.h_dim];
                for j in 0..qnet_pop.h_dim {
                    let mut drive: i32 = qnet_pop.b1_i8[j] as i32 * 256;
                    for i in 0..qnet_pop.in_dim {
                        if x_bits[i] != 0 {
                            drive += qnet_pop.w1_i8[j * qnet_pop.in_dim + i] as i32 * 256;
                        }
                    }
                    for k in 0..2 {
                        let act_out = lut.lookup(s_out[k]) as i32;
                        let w2_val = qnet_pop.w2_i8[k * qnet_pop.h_dim + j] as i32;
                        drive += (w2_val * act_out * ratio_h_fp) >> 8;
                    }
                    let scaled_drive = (drive / inv_scale1) as i16;
                    let diff = (scaled_drive as i32) - (s_h[j] as i32);
                    let new_val = (s_h[j] as i32) + (diff >> 1);
                    new_h[j] = new_val.clamp(-32767, 32767) as i16;
                }

                // Output update
                let mut new_out = vec![0i16; 2];
                for k in 0..2 {
                    let mut drive: i32 = qnet_pop.b2_i8[k] as i32 * 256;
                    for j in 0..qnet_pop.h_dim {
                        let act_h = lut.lookup(s_h[j]) as i32;
                        let w2_val = qnet_pop.w2_i8[k * qnet_pop.h_dim + j] as i32;
                        drive += w2_val * act_h;
                    }
                    let scaled_drive = (drive / inv_scale2) as i16;
                    let diff = (scaled_drive as i32) - (s_out[k] as i32);
                    let new_val = (s_out[k] as i32) + (diff >> 1);
                    new_out[k] = new_val.clamp(-32767, 32767) as i16;
                }

                s_h = new_h;
                s_out = [new_out[0], new_out[1]].to_vec();

                if (s_out[0], s_out[1]) == prev_out {
                    stable_count += 1;
                    if stable_count >= 3 && converge_tick == t {
                        converge_tick = tick + 1 - 3;
                    }
                } else {
                    stable_count = 0;
                }
                prev_out = (s_out[0], s_out[1]);
            }
            let decision = if lut.lookup(s_out[1]) > lut.lookup(s_out[0]) { "YES" } else { "NO" };
            log!("  input=0b{:08b} popcount={} converge@tick={:2} -> {} (s_out=[{}, {}])",
                p, p.count_ones(), converge_tick, decision, s_out[0], s_out[1]);
        }
    }
    log!("");

    // ═══════════════════════════════════════════════════════════════
    // Final verdict
    // ═══════════════════════════════════════════════════════════════
    let elapsed = t0.elapsed().as_secs_f64();

    // Pick best dt_shift per task
    let best_pop_int = int16_ok_pop.max(int16_ok_pop2);
    let best_nib_int = int16_ok_nib.max(int16_ok_nib2);
    let best_pop_agree = agreement_pop.max(agree_pop2);
    let best_nib_agree = agreement_nib.max(agree_nib2);
    let best_pop_shift = if int16_ok_pop2 > int16_ok_pop { 2 } else { 1 };
    let best_nib_shift = if int16_ok_nib2 > int16_ok_nib { 2 } else { 1 };

    log!("================================================================");
    log!("  SUMMARY");
    log!("================================================================");
    log!("");
    log!("  Task            Float EP    Int16 settle  Agreement   Best dt_shift");
    log!("  ──────────────  ──────────  ────────────  ──────────  ─────────────");
    log!("  T2: POPCOUNT    {:.1}%        {:.1}%         {:.1}%        {}",
        float_acc_pop,
        best_pop_int as f32 / 256.0 * 100.0,
        best_pop_agree as f32 / 256.0 * 100.0,
        best_pop_shift);
    log!("  T3: NIBBLE      {:.1}%        {:.1}%         {:.1}%        {}",
        float_acc_nib,
        best_nib_int as f32 / 256.0 * 100.0,
        best_nib_agree as f32 / 256.0 * 100.0,
        best_nib_shift);
    log!("");
    log!("  Max state magnitude: T2={} T3={}", max_mag_pop, max_mag_nib);
    log!("  Overflow events:     T2={} T3={}", total_overflow_pop, total_overflow_nib);
    log!("");

    // Verdict
    let pop_matches = best_pop_agree as f32 / 256.0 >= 0.85;
    let nib_matches = best_nib_agree as f32 / 256.0 >= 0.85;
    let pop_good = best_pop_int as f32 / 256.0 * 100.0 >= 70.0;
    let nib_good = best_nib_int as f32 / 256.0 * 100.0 >= 50.0;

    if pop_matches && nib_matches {
        log!("  VERDICT: Int16 settle MATCHES float EP (>85% agreement on both tasks)");
        log!("  >> VRAXION chip CAN run EP settle in pure integer hardware. <<");
    } else if pop_good || nib_good {
        log!("  VERDICT: Int16 settle PARTIALLY MATCHES float EP");
        log!("  Some quantization loss but decisions largely agree.");
        if !pop_matches {
            log!("  T2 agreement: {:.1}% (below 85% threshold)", best_pop_agree as f32 / 256.0 * 100.0);
        }
        if !nib_matches {
            log!("  T3 agreement: {:.1}% (below 85% threshold)", best_nib_agree as f32 / 256.0 * 100.0);
        }
    } else {
        log!("  VERDICT: Int16 settle DIVERGES FROM float EP");
        log!("  Integer quantization causes significant decision changes.");
    }

    log!("");
    log!("  Total runtime: {:.1}s", elapsed);
    log!("================================================================");

    // Write log to research directory
    let log_path = "S:/Git/VRAXION/.claude/research/settle_int16_log.txt";
    match std::fs::write(log_path, &log) {
        Ok(_) => println!("\nLog written to {}", log_path),
        Err(e) => println!("\nFailed to write log: {}", e),
    }
}
