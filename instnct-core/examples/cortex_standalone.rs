//! Cortex Standalone — Non-Mathematical Decision Making
//!
//! THE QUESTION: Can a C19 cortex make decisions that have NOTHING to do with math?
//! No ALU, no arithmetic — pure pattern recognition and classification.
//!
//! 3 Tasks (increasing difficulty):
//!
//!   Task 1: DANGER DETECTION (binary classification)
//!     8-bit sensor → "SAFE" or "DANGER"
//!     Not a threshold — complex interaction of multiple sensors
//!
//!   Task 2: PATTERN CATEGORY (4-class classification)
//!     8-bit pattern → which of 4 "species" is this?
//!     Overlapping features, no clean linear boundary
//!
//!   Task 3: SEQUENCE CONTEXT (temporal decision)
//!     Current sensor + 2-bit history → "STABLE/RISING/FALLING/ERRATIC"
//!     The cortex must consider time context, not just current value
//!
//! Kill criterion: <70% on any task → KILL
//! Success: >90% on all tasks + freeze >85%
//!
//! Run: cargo run --example cortex_standalone --release

// ============================================================
// C19 + RNG + EP (same as cortex_alu_kill1.rs)
// ============================================================

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

fn tanh_act(x: f32) -> f32 { x.tanh() }

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

    fn freeze_i8(&self) -> FrozenNet {
        let max_w1 = self.w1.iter().map(|w| w.abs()).fold(0.0f32, f32::max).max(1e-8);
        let max_w2 = self.w2.iter().map(|w| w.abs()).fold(0.0f32, f32::max).max(1e-8);
        let s1 = 127.0 / max_w1;
        let s2 = 127.0 / max_w2;
        FrozenNet {
            w1: self.w1.iter().map(|w| (w * s1).round().clamp(-127.0, 127.0) as i8).collect(),
            w2: self.w2.iter().map(|w| (w * s2).round().clamp(-127.0, 127.0) as i8).collect(),
            b1: self.b1.iter().map(|b| (b * s1).round().clamp(-127.0, 127.0) as i8).collect(),
            b2: self.b2.iter().map(|b| (b * s2).round().clamp(-127.0, 127.0) as i8).collect(),
            s1, s2, in_dim: self.in_dim, h_dim: self.h_dim, out_dim: self.out_dim,
        }
    }
}

#[allow(dead_code)]
struct FrozenNet {
    w1: Vec<i8>, w2: Vec<i8>, b1: Vec<i8>, b2: Vec<i8>,
    s1: f32, s2: f32,
    in_dim: usize, h_dim: usize, out_dim: usize,
}

impl FrozenNet {
    fn infer(&self, x: &[f32], act: Act) -> usize {
        let (in_d, h, out_d) = (self.in_dim, self.h_dim, self.out_dim);
        let x_i8: Vec<i8> = x.iter().map(|v| (v * 127.0).round().clamp(-127.0, 127.0) as i8).collect();

        let mut h_vals = vec![0i32; h];
        for j in 0..h {
            let mut sum = self.b1[j] as i32 * 127;
            for i in 0..in_d { sum += self.w1[j * in_d + i] as i32 * x_i8[i] as i32; }
            h_vals[j] = sum;
        }

        let h_act: Vec<i8> = h_vals.iter().map(|&v| {
            let f = v as f32 / (127.0 * self.s1);
            let a = act.apply(f);
            if a.is_nan() { 0i8 } else { (a * 127.0).round().clamp(-127.0, 127.0) as i8 }
        }).collect();

        let mut out_vals = vec![0i32; out_d];
        for k in 0..out_d {
            let mut sum = self.b2[k] as i32 * 127;
            for j in 0..h { sum += self.w2[k * h + j] as i32 * h_act[j] as i32; }
            out_vals[k] = sum;
        }
        out_vals.iter().enumerate().max_by_key(|(_, v)| *v).map(|(i, _)| i).unwrap()
    }
}

fn settle_step(
    s_h: &[f32], s_out: &[f32], x: &[f32], net: &EpNet,
    dt: f32, act: Act, beta: f32, y: &[f32],
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
        new_out[k] = s_out[k] + dt * (-s_out[k] + drive + beta * (y[k] - act.apply(s_out[k])));
    }
    (new_h, new_out)
}

fn settle(x: &[f32], y: &[f32], net: &EpNet, t: usize, dt: f32, act: Act, beta: f32)
    -> (Vec<f32>, Vec<f32>)
{
    let mut s_h = vec![0.0f32; net.h_dim];
    let mut s_out = vec![0.0f32; net.out_dim];
    for _ in 0..t {
        let (nh, no) = settle_step(&s_h, &s_out, x, net, dt, act, beta, y);
        s_h = nh; s_out = no;
    }
    (s_h, s_out)
}

fn predict(x: &[f32], net: &EpNet, t: usize, dt: f32, act: Act, n_out: usize) -> usize {
    let dummy_y = vec![0.0f32; n_out];
    let (_, s_out) = settle(x, &dummy_y, net, t, dt, act, 0.0);
    let acts: Vec<f32> = s_out.iter().map(|s| act.apply(*s)).collect();
    if acts.iter().any(|v| v.is_nan()) { return 0; }
    acts.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap()
}

fn train_ep(net: &mut EpNet, data: &[(Vec<f32>, Vec<f32>)],
            t: usize, dt: f32, act: Act, beta: f32, lr: f32,
            epochs: usize, rng: &mut Rng)
{
    let mut indices: Vec<usize> = (0..data.len()).collect();
    for epoch in 0..epochs {
        let lr_eff = if epoch < 20 { lr * (epoch as f32 + 1.0) / 20.0 } else { lr };
        rng.shuffle(&mut indices);
        for &idx in &indices {
            let (x, y) = &data[idx];
            let (sf_h, sf_o) = settle(x, y, net, t, dt, act, 0.0);
            let (sn_h, sn_o) = {
                let mut sh = sf_h.clone(); let mut so = sf_o.clone();
                for _ in 0..t {
                    let (nh, no) = settle_step(&sh, &so, x, net, dt, act, beta, y);
                    sh = nh; so = no;
                }
                (sh, so)
            };

            let inv_b = 1.0 / beta;
            for j in 0..net.h_dim {
                let an = act.apply(sn_h[j]); let af = act.apply(sf_h[j]);
                for i in 0..net.in_dim {
                    net.w1[j * net.in_dim + i] += lr_eff * inv_b * (an * x[i] - af * x[i]);
                }
                net.b1[j] += lr_eff * inv_b * (an - af);
            }
            for k in 0..net.out_dim {
                let aon = act.apply(sn_o[k]); let aof = act.apply(sf_o[k]);
                for j in 0..net.h_dim {
                    let ahn = act.apply(sn_h[j]); let ahf = act.apply(sf_h[j]);
                    net.w2[k * net.h_dim + j] += lr_eff * inv_b * (aon * ahn - aof * ahf);
                }
                net.b2[k] += lr_eff * inv_b * (aon - aof);
            }
        }

        if epoch % 100 == 0 || epoch == epochs - 1 {
            let mut ok = 0;
            for (x, y) in data {
                let pred = predict(x, net, t, dt, act, y.len());
                let target = y.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap();
                if pred == target { ok += 1; }
            }
            println!("    Epoch {:4} — acc = {:.1}% ({}/{})", epoch, ok as f32 / data.len() as f32 * 100.0, ok, data.len());
        }
    }
}

// ================================================================
// TASK 1: DANGER DETECTION
// "Is this situation dangerous?" — binary, non-linear boundary
//
// Sensors: light(3bit), sound(3bit), vibration(2bit)
// DANGER if:
//   (sound >= 6 AND vibration >= 2) — loud + shaking
//   (light <= 1 AND sound >= 4) — dark + noise
//   (vibration >= 3 AND light >= 6) — bright flash + shaking
//   (sound >= 5 AND light <= 2 AND vibration >= 1) — specific combo
// Everything else: SAFE
// ================================================================

fn danger_label(light: u8, sound: u8, vib: u8) -> usize {
    if sound >= 6 && vib >= 2 { return 1; } // loud + shaking
    if light <= 1 && sound >= 4 { return 1; } // dark + noise
    if vib >= 3 && light >= 6 { return 1; }   // flash + shaking
    if sound >= 5 && light <= 2 && vib >= 1 { return 1; } // specific combo
    0 // SAFE
}

fn gen_danger_data() -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut data = Vec::new();
    for light in 0..8u8 {
        for sound in 0..8u8 {
            for vib in 0..4u8 {
                let label = danger_label(light, sound, vib);
                let input = vec![
                    light as f32 / 7.0, sound as f32 / 7.0, vib as f32 / 3.0,
                    (light as f32 / 7.0) * (sound as f32 / 7.0),
                    (sound as f32 / 7.0) * (vib as f32 / 3.0),
                    (light as f32 / 7.0) * (vib as f32 / 3.0),
                ];
                let mut target = vec![0.0f32; 2];
                target[label] = 1.0;
                data.push((input, target));
            }
        }
    }
    data
}

// ================================================================
// TASK 2: PATTERN SPECIES (4-class, overlapping features)
//
// 8-bit pattern → 4 "species":
//   Species A: majority bits in positions 0-3 set (low nibble heavy)
//   Species B: majority bits in positions 4-7 set (high nibble heavy)
//   Species C: alternating pattern (bits 0,2,4,6 vs 1,3,5,7)
//   Species D: clustered (consecutive runs of 3+ same bits)
//
// When multiple match: priority A > B > C > D
// When none match: D (default)
// ================================================================

fn pattern_species(p: u8) -> usize {
    let low_count = (0..4).filter(|&i| p & (1 << i) != 0).count();
    let high_count = (4..8).filter(|&i| p & (1 << i) != 0).count();

    // Check alternating
    let even_bits = (0..4).filter(|&i| p & (1 << (i * 2)) != 0).count();
    let odd_bits = (0..4).filter(|&i| p & (1 << (i * 2 + 1)) != 0).count();
    let alternating = (even_bits >= 3 && odd_bits <= 1) || (odd_bits >= 3 && even_bits <= 1);

    // Check clustered (run of 3+ consecutive same bits)
    let mut max_run = 1;
    let mut cur_run = 1;
    for i in 1..8 {
        let prev = (p >> (i - 1)) & 1;
        let curr = (p >> i) & 1;
        if curr == prev { cur_run += 1; if cur_run > max_run { max_run = cur_run; } }
        else { cur_run = 1; }
    }
    let clustered = max_run >= 3;

    if low_count >= 3 && high_count <= 1 { return 0; } // Species A
    if high_count >= 3 && low_count <= 1 { return 1; } // Species B
    if alternating { return 2; } // Species C
    if clustered { return 3; }   // Species D
    3 // default
}

fn gen_pattern_data() -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut data = Vec::new();
    for p in 0..=255u8 {
        let label = pattern_species(p);
        let input: Vec<f32> = (0..8).map(|i| if p & (1 << i) != 0 { 1.0 } else { 0.0 }).collect();
        let mut target = vec![0.0f32; 4];
        target[label] = 1.0;
        data.push((input, target));
    }
    data
}

// ================================================================
// TASK 3: TEMPORAL CONTEXT (current + history → trend)
//
// Input: current_value(4bit) + prev_delta_sign(2bit) + prev_magnitude(2bit)
// Output: STABLE / RISING / FALLING / ERRATIC
//
// STABLE:  prev_delta small AND current in mid-range
// RISING:  prev_delta positive AND current > 8
// FALLING: prev_delta negative AND current < 8
// ERRATIC: prev_delta sign changed OR magnitude high
// ================================================================

fn temporal_label(current: u8, prev_sign: u8, prev_mag: u8) -> usize {
    // prev_sign: 0=zero, 1=positive, 2=negative, 3=sign-changed
    // prev_mag: 0=tiny, 1=small, 2=medium, 3=large
    if prev_sign == 3 || prev_mag == 3 { return 3; } // ERRATIC
    if prev_mag <= 1 && current >= 5 && current <= 11 { return 0; } // STABLE
    if prev_sign == 1 && current > 8 { return 1; } // RISING
    if prev_sign == 2 && current < 8 { return 2; } // FALLING
    if prev_mag == 0 { return 0; } // near-zero delta = stable
    3 // default: ERRATIC
}

fn gen_temporal_data() -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut data = Vec::new();
    for current in 0..16u8 {
        for prev_sign in 0..4u8 {
            for prev_mag in 0..4u8 {
                let label = temporal_label(current, prev_sign, prev_mag);
                let input = vec![
                    current as f32 / 15.0,
                    prev_sign as f32 / 3.0,
                    prev_mag as f32 / 3.0,
                    // Cross features
                    (current as f32 / 15.0) * (prev_sign as f32 / 3.0),
                    (current as f32 / 15.0) * (prev_mag as f32 / 3.0),
                    (prev_sign as f32 / 3.0) * (prev_mag as f32 / 3.0),
                ];
                let mut target = vec![0.0f32; 4];
                target[label] = 1.0;
                data.push((input, target));
            }
        }
    }
    data
}

// ================================================================
// Run one task
// ================================================================

fn run_task(
    _name: &str, class_names: &[&str],
    data: &[(Vec<f32>, Vec<f32>)],
    acts: &[Act], h_dim: usize,
    t: usize, dt: f32, beta: f32, lr: f32, epochs: usize,
    seeds: &[u64],
) -> Vec<(String, f32, f32)>  // (label, float_acc, frozen_acc)
{
    let n_classes = class_names.len();

    // Distribution
    let mut counts = vec![0usize; n_classes];
    for (_, y) in data {
        let c = y.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap();
        counts[c] += 1;
    }
    println!("  Classes:");
    for (i, name) in class_names.iter().enumerate() {
        println!("    {:10} = {} ({:.1}%)", name, counts[i], counts[i] as f32 / data.len() as f32 * 100.0);
    }
    let majority = *counts.iter().max().unwrap() as f32 / data.len() as f32;
    println!("  Majority baseline: {:.1}%", majority * 100.0);
    println!();

    let mut results = Vec::new();

    for act in acts {
        for &seed in seeds {
            let mut rng = Rng::new(seed);
            let out_dim = n_classes;
            let in_dim = data[0].0.len();
            let mut net = EpNet::new(in_dim, h_dim, out_dim, &mut rng);

            println!("  {} H={} seed={}:", act.name(), h_dim, seed);
            train_ep(&mut net, data, t, dt, *act, beta, lr, epochs, &mut rng);

            // Float eval
            let mut ok = 0;
            let mut per_class = vec![0usize; n_classes];
            let mut per_class_total = vec![0usize; n_classes];
            for (x, y) in data {
                let pred = predict(x, &net, t, dt, *act, out_dim);
                let target = y.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap();
                per_class_total[target] += 1;
                if pred == target { ok += 1; per_class[target] += 1; }
            }
            let float_acc = ok as f32 / data.len() as f32;
            println!("    FLOAT: {:.1}% ({}/{})", float_acc * 100.0, ok, data.len());
            for (i, cn) in class_names.iter().enumerate() {
                if per_class_total[i] > 0 {
                    println!("      {:10} {}/{} = {:.1}%", cn, per_class[i], per_class_total[i],
                        per_class[i] as f32 / per_class_total[i] as f32 * 100.0);
                }
            }

            // Frozen eval
            let frozen = net.freeze_i8();
            let mut ok_f = 0;
            for (x, y) in data {
                let pred = frozen.infer(x, *act);
                let target = y.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap();
                if pred == target { ok_f += 1; }
            }
            let frozen_acc = ok_f as f32 / data.len() as f32;
            let loss = (float_acc - frozen_acc) * 100.0;
            println!("    INT8:  {:.1}% ({}/{}) — freeze loss: {:.1}pp",
                frozen_acc * 100.0, ok_f, data.len(), loss);
            println!();

            let label = format!("{} s={}", act.name(), seed);
            results.push((label, float_acc, frozen_acc));
        }
    }

    results
}

// ================================================================
// MAIN
// ================================================================

fn main() {
    println!("================================================================");
    println!("  CORTEX STANDALONE — Non-Mathematical Decisions");
    println!("  No ALU. No arithmetic. Pure pattern recognition.");
    println!("================================================================");
    println!();

    let acts = vec![Act::C19(8.0), Act::Tanh];
    let seeds = vec![42u64, 123, 7];

    // TASK 1: DANGER DETECTION
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  TASK 1: DANGER DETECTION (binary)");
    println!("  'Is this situation dangerous?'");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let danger_data = gen_danger_data();
    println!("  {} scenarios", danger_data.len());
    let r1 = run_task("Danger", &["SAFE", "DANGER"], &danger_data,
        &acts, 16, 50, 0.5, 0.5, 0.005, 500, &seeds);

    // TASK 2: PATTERN SPECIES
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  TASK 2: PATTERN SPECIES (4-class)");
    println!("  'What type of pattern is this?'");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let pattern_data = gen_pattern_data();
    println!("  {} patterns", pattern_data.len());
    let r2 = run_task("Species", &["LowHeavy", "HighHeavy", "Alternat", "Cluster"], &pattern_data,
        &acts, 24, 50, 0.5, 0.5, 0.005, 500, &seeds);

    // TASK 3: TEMPORAL CONTEXT
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  TASK 3: TEMPORAL CONTEXT (4-class)");
    println!("  'Is the trend STABLE, RISING, FALLING, or ERRATIC?'");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let temporal_data = gen_temporal_data();
    println!("  {} samples", temporal_data.len());
    let r3 = run_task("Temporal", &["STABLE", "RISING", "FALLING", "ERRATIC"], &temporal_data,
        &acts, 20, 50, 0.5, 0.5, 0.005, 500, &seeds);

    // ================================================================
    // FINAL SUMMARY
    // ================================================================
    println!("================================================================");
    println!("  FINAL SUMMARY — Non-Mathematical Cortex");
    println!("================================================================");
    println!();
    println!("  {:25} {:>8} {:>8}", "Task / Config", "Float%", "Frozen%");
    println!("  {}", "─".repeat(50));

    let mut all_results = Vec::new();

    println!("  TASK 1: DANGER DETECTION");
    for (l, f, fr) in &r1 {
        println!("    {:23} {:>7.1}% {:>7.1}%", l, f*100.0, fr*100.0);
        all_results.push((*f, *fr, l.contains("C19")));
    }
    println!("  TASK 2: PATTERN SPECIES");
    for (l, f, fr) in &r2 {
        println!("    {:23} {:>7.1}% {:>7.1}%", l, f*100.0, fr*100.0);
        all_results.push((*f, *fr, l.contains("C19")));
    }
    println!("  TASK 3: TEMPORAL CONTEXT");
    for (l, f, fr) in &r3 {
        println!("    {:23} {:>7.1}% {:>7.1}%", l, f*100.0, fr*100.0);
        all_results.push((*f, *fr, l.contains("C19")));
    }

    println!();

    let c19_float_best = all_results.iter().filter(|(_, _, is_c19)| *is_c19)
        .map(|(f, _, _)| *f).fold(0.0f32, f32::max);
    let c19_frozen_best = all_results.iter().filter(|(_, _, is_c19)| *is_c19)
        .map(|(_, fr, _)| *fr).fold(0.0f32, f32::max);
    let c19_float_min = all_results.iter().filter(|(_, _, is_c19)| *is_c19)
        .map(|(f, _, _)| *f).fold(1.0f32, f32::min);
    let tanh_float_best = all_results.iter().filter(|(_, _, is_c19)| !*is_c19)
        .map(|(f, _, _)| *f).fold(0.0f32, f32::max);

    println!("  Best C19 float:  {:.1}%", c19_float_best * 100.0);
    println!("  Worst C19 float: {:.1}%", c19_float_min * 100.0);
    println!("  Best C19 frozen: {:.1}%", c19_frozen_best * 100.0);
    println!("  Best tanh float: {:.1}%", tanh_float_best * 100.0);
    println!();

    if c19_float_min >= 0.90 && c19_frozen_best >= 0.85 {
        println!("  ██ VERDICT: STRONG CONTINUE ██");
        println!("  C19 cortex handles non-mathematical decisions.");
    } else if c19_float_min >= 0.70 {
        println!("  ██ VERDICT: CONTINUE ██");
        println!("  C19 cortex works but needs tuning on some tasks.");
    } else {
        println!("  ██ VERDICT: PARTIAL ██");
        println!("  Some tasks work, some need improvement.");
    }

    if c19_float_best >= tanh_float_best * 0.95 {
        println!("  C19 is competitive with tanh.");
    }

    println!("================================================================");
}
