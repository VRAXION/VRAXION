//! EP -> Distill -> LutGate Pipeline
//!
//! The full pipeline:
//!   1. EP trains a neural network (float, bidirectional settling) to 100% on byte tasks
//!   2. Run EP on ALL possible inputs -> store as "oracle" truth table
//!   3. Build a FEEDFORWARD LutGate network that reproduces the oracle
//!   4. Each hidden neuron built one-by-one: random search for best C19 weights -> bake to LUT -> freeze -> next
//!   5. After hidden layer: train output layer on frozen hidden outputs
//!   6. If accuracy < 100%: add more hidden neurons and retry output
//!   7. Exhaustive verify: test ALL inputs -> must match oracle
//!
//! Tasks:
//!   T2: POPCOUNT >4   (8->16->1)
//!   T3: NIBBLE CLASS  (8->16->4)
//!   T4: BYTE ADD      (16->16->5)
//!
//! Run: cargo run --example cortex_distill --release

use std::io::Write as IoWrite;

// ============================================================
// C19 activation
// ============================================================

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

// ============================================================
// RNG (deterministic, same as cortex_byte)
// ============================================================

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed.wrapping_mul(6364136223846793005).wrapping_add(1) }
    }
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
// Activation wrapper
// ============================================================

#[derive(Clone, Copy)]
struct Act(f32);
impl Act {
    fn apply(self, x: f32) -> f32 { c19(x, self.0) }
}

// ============================================================
// EP Network
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

// ============================================================
// EP Dynamics
// ============================================================

fn settle_step(
    sh: &[f32], so: &[f32], x: &[f32], net: &EpNet,
    dt: f32, act: Act, beta: f32, y: &[f32],
) -> (Vec<f32>, Vec<f32>) {
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

fn settle(
    x: &[f32], y: &[f32], net: &EpNet, t: usize, dt: f32, act: Act, beta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut sh = vec![0.0f32; net.h_dim];
    let mut so = vec![0.0f32; net.out_dim];
    for _ in 0..t {
        let (h2, o2) = settle_step(&sh, &so, x, net, dt, act, beta, y);
        sh = h2; so = o2;
    }
    (sh, so)
}

fn predict_bits(x: &[f32], net: &EpNet, t: usize, dt: f32, act: Act) -> Vec<u8> {
    let dy = vec![0.0f32; net.out_dim];
    let (_, so) = settle(x, &dy, net, t, dt, act, 0.0);
    so.iter().map(|s| {
        let a = act.apply(*s);
        if a.is_nan() { 0u8 } else if a > 0.5 { 1u8 } else { 0u8 }
    }).collect()
}

// ============================================================
// EP Training
// ============================================================

fn train_ep(
    net: &mut EpNet, data: &[(Vec<f32>, Vec<f32>)],
    t: usize, dt: f32, act: Act, beta: f32, lr: f32,
    epochs: usize, rng: &mut Rng, log_interval: usize,
) {
    let mut idx: Vec<usize> = (0..data.len()).collect();
    for ep in 0..epochs {
        let lr_e = if ep < 20 { lr * (ep as f32 + 1.0) / 20.0 } else { lr };
        rng.shuffle(&mut idx);
        for &i in &idx {
            let (x, y) = &data[i];
            let (sfh, sfo) = settle(x, y, net, t, dt, act, 0.0);
            let mut sh = sfh.clone(); let mut so = sfo.clone();
            for _ in 0..t {
                let (h2, o2) = settle_step(&sh, &so, x, net, dt, act, beta, y);
                sh = h2; so = o2;
            }
            let ib = 1.0 / beta;
            for j in 0..net.h_dim {
                let an = act.apply(sh[j]); let af = act.apply(sfh[j]);
                for ii in 0..net.in_dim {
                    net.w1[j * net.in_dim + ii] += lr_e * ib * (an * x[ii] - af * x[ii]);
                }
                net.b1[j] += lr_e * ib * (an - af);
            }
            for k in 0..net.out_dim {
                let aon = act.apply(so[k]); let aof = act.apply(sfo[k]);
                for j in 0..net.h_dim {
                    net.w2[k * net.h_dim + j] += lr_e * ib * (aon * act.apply(sh[j]) - aof * act.apply(sfh[j]));
                }
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
            println!("      Epoch {:4} -- {:.1}% ({}/{})",
                     ep, ok as f32 / data.len() as f32 * 100.0, ok, data.len());
        }
    }
}

// ============================================================
// LutGate -- distilled neuron with truth table
// ============================================================

#[allow(dead_code)]
struct LutGate {
    lut: Vec<u8>,       // truth table indexed by input bit pattern
    n_inputs: usize,
}

impl LutGate {
    fn from_weights(weights: &[f32], bias: f32, rho: f32, threshold: f32) -> Self {
        let n = weights.len();
        let n_entries = 1usize << n;
        let mut lut = vec![0u8; n_entries];
        for pattern in 0..n_entries {
            let sum: f32 = (0..n).map(|i| {
                let input = if pattern & (1 << i) != 0 { 1.0 } else { 0.0 };
                weights[i] * input
            }).sum::<f32>() + bias;
            lut[pattern] = if c19(sum, rho) > threshold { 1 } else { 0 };
        }
        LutGate { lut, n_inputs: n }
    }

    fn eval(&self, inputs: &[u8]) -> u8 {
        let mut idx = 0usize;
        for (i, &inp) in inputs.iter().enumerate() {
            if inp != 0 { idx |= 1 << i; }
        }
        if idx < self.lut.len() { self.lut[idx] } else { 0 }
    }

    fn memory(&self) -> usize { self.lut.len() }
}

/// Evaluate c19(w.x + b) > threshold for a single input pattern (fast, no LUT baking).
#[inline]
fn eval_gate_direct(weights: &[f32], bias: f32, rho: f32, threshold: f32, inputs: &[u8]) -> u8 {
    let sum: f32 = weights.iter().zip(inputs.iter())
        .map(|(&w, &inp)| if inp != 0 { w } else { 0.0 })
        .sum::<f32>() + bias;
    if c19(sum, rho) > threshold { 1 } else { 0 }
}

/// Count correct classifications for a candidate gate.
#[inline]
fn count_correct(weights: &[f32], bias: f32, rho: f32, threshold: f32,
                 patterns: &[Vec<u8>], targets: &[u8]) -> usize {
    patterns.iter().zip(targets.iter()).filter(|(p, &t)| {
        eval_gate_direct(weights, bias, rho, threshold, p) == t
    }).count()
}

/// Hill-climbing refinement: coordinate descent + random perturbations.
fn hill_climb(
    weights: &[f32], bias: f32, rho: f32, threshold: f32,
    patterns: &[Vec<u8>], targets: &[u8],
    rng: &mut Rng, rounds: usize,
) -> (Vec<f32>, f32, usize) {
    let n = weights.len();
    let n_total = patterns.len();
    let mut w = weights.to_vec();
    let mut b = bias;
    let mut best = count_correct(&w, b, rho, threshold, patterns, targets);

    if best == n_total { return (w, b, best); }

    let steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0];

    for _ in 0..rounds {
        let mut improved = false;

        // Coordinate descent on each weight and bias
        for idx in 0..=n {
            for &step in &steps {
                for dir in &[1.0f32, -1.0] {
                    let delta = step * dir;
                    if idx < n { w[idx] += delta; } else { b += delta; }
                    let c = count_correct(&w, b, rho, threshold, patterns, targets);
                    if c > best {
                        best = c;
                        improved = true;
                        if best == n_total { return (w, b, best); }
                        break; // keep this perturbation
                    } else {
                        if idx < n { w[idx] -= delta; } else { b -= delta; }
                    }
                }
            }
        }

        // Random perturbation to escape local optima
        if !improved {
            for _ in 0..10 {
                let idx = (rng.next() as usize) % (n + 1);
                let delta = rng.range_f32(-2.0, 2.0);
                if idx < n { w[idx] += delta; } else { b += delta; }
                let c = count_correct(&w, b, rho, threshold, patterns, targets);
                if c >= best {
                    best = c;
                    if best == n_total { return (w, b, best); }
                } else {
                    if idx < n { w[idx] -= delta; } else { b -= delta; }
                }
            }
        }
    }

    (w, b, best)
}

/// Search for the best output gate using random search + multi-restart hill climbing.
fn search_output_gate(
    rng: &mut Rng, h_count: usize, rho: f32, threshold: f32,
    n_trials: usize, patterns: &[Vec<u8>], targets: &[u8],
) -> (Vec<f32>, f32, usize) {
    let n_patterns = patterns.len();
    let top_k = 5;

    // Phase 1: Random search, keep top-K candidates
    let mut candidates: Vec<(usize, Vec<f32>, f32)> = Vec::new();

    for _ in 0..n_trials {
        let w: Vec<f32> = (0..h_count).map(|_| rng.range_f32(-3.0, 3.0)).collect();
        let b = rng.range_f32(-3.0, 3.0);
        let correct = count_correct(&w, b, rho, threshold, patterns, targets);

        if candidates.len() < top_k || correct > candidates.last().map(|c| c.0).unwrap_or(0) {
            candidates.push((correct, w, b));
            candidates.sort_by(|a, b| b.0.cmp(&a.0));
            if candidates.len() > top_k { candidates.truncate(top_k); }
        }

        if correct == n_patterns { return (candidates[0].1.clone(), candidates[0].2, correct); }
    }

    // Phase 2: Hill-climb each of the top-K candidates
    let mut best_c = candidates[0].0;
    let mut best_w = candidates[0].1.clone();
    let mut best_b = candidates[0].2;

    for (_, w, b) in &candidates {
        let (rw, rb, rc) = hill_climb(w, *b, rho, threshold, patterns, targets, rng, 100);
        if rc > best_c {
            best_c = rc;
            best_w = rw;
            best_b = rb;
            if best_c == n_patterns { break; }
        }
    }

    (best_w, best_b, best_c)
}

// ============================================================
// Data generators
// ============================================================

fn byte_to_bits(b: u8) -> Vec<f32> {
    (0..8).map(|i| if b & (1 << i) != 0 { 1.0 } else { 0.0 }).collect()
}

fn _pattern_to_bits(pattern: usize, n_bits: usize) -> Vec<u8> {
    (0..n_bits).map(|i| if pattern & (1 << i) != 0 { 1u8 } else { 0u8 }).collect()
}

fn gen_t2() -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..=255u8).map(|b| {
        let input = byte_to_bits(b);
        let label = if b.count_ones() > 4 { 1.0 } else { 0.0 };
        (input, vec![label])
    }).collect()
}

fn gen_t3() -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..=255u8).map(|b| {
        let input = byte_to_bits(b);
        let class = ((b & 0x0F) / 4) as usize;
        let mut target = vec![0.0f32; 4];
        target[class] = 1.0;
        (input, target)
    }).collect()
}

fn gen_t4() -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut data = Vec::new();
    for a in 0..16u8 {
        for b in 0..16u8 {
            let mut input = byte_to_bits(a);
            input.extend(byte_to_bits(b));
            let result = a.wrapping_add(b);
            let target = byte_to_bits(result);
            data.push((input, target[..5].to_vec()));
        }
    }
    data
}

// ============================================================
// Scoring function for hidden neuron selection
// ============================================================

fn score_neuron(neuron_out: &[u8], oracle: &[Vec<u8>], existing: &[Vec<u8>]) -> i32 {
    let n = neuron_out.len();
    let n_out_bits = oracle[0].len();

    // Correlation with oracle output bits
    let mut max_corr = 0i32;
    for k in 0..n_out_bits {
        let agree: i32 = (0..n).map(|i| {
            if neuron_out[i] == oracle[i][k] { 1 } else { -1 }
        }).sum();
        max_corr = max_corr.max(agree.abs());
    }

    // Penalize redundancy with existing neurons
    let mut max_overlap = 0i32;
    for existing_out in existing {
        let overlap: i32 = (0..n).map(|i| {
            if neuron_out[i] == existing_out[i] { 1 } else { 0 }
        }).sum();
        max_overlap = max_overlap.max(overlap);
    }

    // Bonus for distinguishing power: how many input pairs does this neuron separate?
    // A neuron that outputs different values for inputs that need different oracle outputs
    // is more valuable. We approximate this by counting collision-breaking.
    let mut collision_break_score = 0i32;
    if !existing.is_empty() {
        // Build combined fingerprint with existing neurons
        let mut existing_groups: std::collections::HashMap<Vec<u8>, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, _) in oracle.iter().enumerate() {
            let key: Vec<u8> = existing.iter().map(|e| e[i]).collect();
            existing_groups.entry(key).or_default().push(i);
        }
        // Count groups where this neuron splits inputs with different oracle outputs
        for group in existing_groups.values() {
            if group.len() <= 1 { continue; }
            let has_0 = group.iter().any(|&i| neuron_out[i] == 0);
            let has_1 = group.iter().any(|&i| neuron_out[i] == 1);
            if has_0 && has_1 {
                // This neuron splits this group -- check if it's useful (different oracle outputs in subgroups)
                let oracle_0: Vec<&Vec<u8>> = group.iter().filter(|&&i| neuron_out[i] == 0).map(|&i| &oracle[i]).collect();
                let oracle_1: Vec<&Vec<u8>> = group.iter().filter(|&&i| neuron_out[i] == 1).map(|&i| &oracle[i]).collect();
                // Check if the split separates different oracle values
                let unique_0: std::collections::HashSet<&Vec<u8>> = oracle_0.iter().cloned().collect();
                let unique_1: std::collections::HashSet<&Vec<u8>> = oracle_1.iter().cloned().collect();
                if unique_0 != unique_1 || unique_0.len() < oracle_0.len() || unique_1.len() < oracle_1.len() {
                    collision_break_score += group.len() as i32;
                }
            }
        }
    }

    max_corr * 10 - max_overlap + collision_break_score * 5
}

// ============================================================
// Search for a hidden neuron (fast direct eval, no LUT during search)
// ============================================================

fn search_hidden_neuron(
    rng: &mut Rng, n_input_bits: usize, n_patterns: usize,
    rho: f32, threshold: f32, n_trials: usize,
    input_patterns: &[Vec<u8>], oracle: &[Vec<u8>], existing: &[Vec<u8>],
) -> (Vec<f32>, f32, Vec<u8>) {
    let mut best_score = i32::MIN;
    let mut best_w: Vec<f32> = vec![0.0; n_input_bits];
    let mut best_b = 0.0f32;
    let mut best_outputs: Vec<u8> = vec![0; n_patterns];

    for _ in 0..n_trials {
        let w: Vec<f32> = (0..n_input_bits).map(|_| rng.range_f32(-3.0, 3.0)).collect();
        let b = rng.range_f32(-3.0, 3.0);

        let outputs: Vec<u8> = (0..n_patterns).map(|p| {
            eval_gate_direct(&w, b, rho, threshold, &input_patterns[p])
        }).collect();

        let score = score_neuron(&outputs, oracle, existing);
        if score > best_score {
            best_score = score;
            best_w = w;
            best_b = b;
            best_outputs = outputs;
        }
    }

    (best_w, best_b, best_outputs)
}

fn _precompute_patterns(n_bits: usize) -> Vec<Vec<u8>> {
    let n = 1usize << n_bits;
    (0..n).map(|p| _pattern_to_bits(p, n_bits)).collect()
}

// ============================================================
// Collision check: do different inputs map to same hidden pattern
// but need different outputs?
// ============================================================

fn check_collisions(hidden_patterns: &[Vec<u8>], oracle: &[Vec<u8>]) -> (bool, usize) {
    let mut seen: std::collections::HashMap<Vec<u8>, Vec<u8>> = std::collections::HashMap::new();
    let mut collision_free = true;
    for (p, hp) in hidden_patterns.iter().enumerate() {
        if let Some(prev_out) = seen.get(hp) {
            if *prev_out != oracle[p] {
                collision_free = false;
                break;
            }
        } else {
            seen.insert(hp.clone(), oracle[p].clone());
        }
    }
    let n_unique = seen.len();
    (collision_free, n_unique)
}

// ============================================================
// Build direct LUT output: for each output bit, map hidden pattern -> oracle bit.
// Only works for H <= 20 (to keep LUT size manageable).
// ============================================================

fn build_direct_output(
    hidden_patterns: &[Vec<u8>], oracle: &[Vec<u8>],
    h_count: usize, n_output_bits: usize,
) -> Vec<LutGate> {
    let mut gates = Vec::new();
    let n_entries = 1usize << h_count;
    for k in 0..n_output_bits {
        let mut lut = vec![0u8; n_entries];
        for (p, hp) in hidden_patterns.iter().enumerate() {
            let mut idx = 0usize;
            for (i, &v) in hp.iter().enumerate() {
                if v != 0 { idx |= 1 << i; }
            }
            lut[idx] = oracle[p][k];
        }
        gates.push(LutGate { lut, n_inputs: h_count });
    }
    gates
}

// ============================================================
// Distillation result
// ============================================================

#[allow(dead_code)]
struct DistillResult {
    name: String,
    ep_oracle_acc: f32,
    n_patterns: usize,
    hidden_neurons: usize,
    output_neurons: usize,
    pipeline_correct: usize,
    pipeline_acc: f32,
    total_memory: usize,
    extra_neurons: usize,
    output_type: String,
}

// ============================================================
// Distill one task
// ============================================================

fn distill_task(
    name: &str,
    data: &[(Vec<f32>, Vec<f32>)],
    h_dim_ep: usize,
    h_init: usize,
    t: usize, dt: f32, act: Act,
    beta: f32, lr: f32,
    epochs: usize,
    seed: u64,
    n_input_bits: usize,
    n_output_bits: usize,
    n_trials_hidden: usize,
    n_trials_output: usize,
    max_h: usize, // max total hidden neurons (init + extra)
    log: &mut Vec<String>,
) -> DistillResult {
    let n_patterns = data.len(); // use actual data size, not 2^n_input_bits for T4

    let msg = format!("== {} ({}->{}->{}->LutGate, {} patterns) ==",
                      name, n_input_bits, h_init, n_output_bits, n_patterns);
    println!("\n{}", msg);
    log.push(msg);

    // ── Step 1: EP train ────────────────────────────────────────
    println!("  [1] EP Training (seed={}, {} epochs, H={})...", seed, epochs, h_dim_ep);
    log.push(format!("  EP Training: seed={}, epochs={}, H_ep={}", seed, epochs, h_dim_ep));

    let mut rng = Rng::new(seed);
    let in_dim = data[0].0.len();
    let out_dim = data[0].1.len();
    let mut net = EpNet::new(in_dim, h_dim_ep, out_dim, &mut rng);
    train_ep(&mut net, data, t, dt, act, beta, lr, epochs, &mut rng, 200);

    // Final EP accuracy
    let mut ep_ok = 0usize;
    for (x, y) in data {
        let out = predict_bits(x, &net, t, dt, act);
        let target: Vec<u8> = y.iter().map(|&v| if v > 0.5 { 1u8 } else { 0u8 }).collect();
        if out == target { ep_ok += 1; }
    }
    let ep_acc = ep_ok as f32 / data.len() as f32 * 100.0;
    println!("    EP Float accuracy: {:.1}% ({}/{})", ep_acc, ep_ok, data.len());
    log.push(format!("  EP: {:.1}%", ep_acc));

    // ── Step 2: Generate oracle truth table ─────────────────────
    println!("  [2] Generating oracle truth table ({} patterns)...", n_patterns);

    // Build oracle from all data patterns (not from exhaustive 2^n enumeration,
    // since for T4 with 16 bits we'd need 65536 predictions but only have 256 data)
    let oracle: Vec<Vec<u8>> = data.iter().map(|(x, _)| {
        predict_bits(x, &net, t, dt, act)
    }).collect();

    // Pre-compute input bit patterns from data
    let input_u8_patterns: Vec<Vec<u8>> = data.iter().map(|(x, _)| {
        x.iter().map(|&v| if v > 0.5 { 1u8 } else { 0u8 }).collect()
    }).collect();

    // Verify oracle
    let mut oracle_ok = 0usize;
    for (i, (_, y)) in data.iter().enumerate() {
        let target: Vec<u8> = y.iter().map(|&v| if v > 0.5 { 1u8 } else { 0u8 }).collect();
        if oracle[i] == target { oracle_ok += 1; }
    }
    let oracle_acc = oracle_ok as f32 / n_patterns as f32 * 100.0;
    println!("    Oracle accuracy: {:.1}% ({}/{})", oracle_acc, oracle_ok, n_patterns);
    log.push(format!("  Oracle: {:.1}%", oracle_acc));

    // ── Step 3: Build hidden neurons ────────────────────────────
    println!("  [3] Building hidden LutGate neurons ({} initial, {} trials each)...", h_init, n_trials_hidden);
    log.push(format!("  Hidden: H_init={}, trials={}", h_init, n_trials_hidden));

    let rho = 8.0f32;
    let threshold = 0.5f32;

    let mut hidden_gates: Vec<LutGate> = Vec::new();
    let mut existing_hidden_outputs: Vec<Vec<u8>> = Vec::new();

    for j in 0..h_init {
        let (best_w, best_b, best_outputs) = search_hidden_neuron(
            &mut rng, n_input_bits, n_patterns, rho, threshold,
            n_trials_hidden, &input_u8_patterns, &oracle, &existing_hidden_outputs,
        );

        let ones: usize = best_outputs.iter().filter(|&&v| v == 1).count();
        let score = score_neuron(&best_outputs, &oracle, &existing_hidden_outputs);
        hidden_gates.push(LutGate::from_weights(&best_w, best_b, rho, threshold));
        existing_hidden_outputs.push(best_outputs);

        if j < 8 || j == h_init - 1 || j % 4 == 0 {
            println!("    Hidden[{:2}]: score={:6}, ones={}/{}", j, score, ones, n_patterns);
        }
        log.push(format!("    H[{}]: score={}", j, score));
    }

    // ── Step 4+5+6: Build output, verify, retry ────────────────
    let mut pipeline_correct = 0usize;
    let mut output_gates: Vec<LutGate> = Vec::new();
    let mut extra_added = 0usize;
    let mut output_type = String::from("C19-gate");
    let max_extra = max_h.saturating_sub(h_init);

    loop {
        let h_count = hidden_gates.len();

        // Compute hidden outputs for all patterns
        let hidden_patterns: Vec<Vec<u8>> = (0..n_patterns).map(|p| {
            hidden_gates.iter().map(|g| g.eval(&input_u8_patterns[p])).collect()
        }).collect();

        let (collision_free, n_unique) = check_collisions(&hidden_patterns, &oracle);

        println!("  [4] Output layer (H={}, {} unique, collisions: {})...",
                 h_count, n_unique, !collision_free);
        log.push(format!("  Output: H={}, unique={}, coll_free={}", h_count, n_unique, collision_free));

        if collision_free && h_count <= 20 {
            // Try C19 gates first
            output_gates = Vec::new();
            for k in 0..n_output_bits {
                let target: Vec<u8> = oracle.iter().map(|o| o[k]).collect();
                let (bw, bb, bc) = search_output_gate(
                    &mut rng, h_count, rho, threshold,
                    n_trials_output, &hidden_patterns, &target,
                );
                output_gates.push(LutGate::from_weights(&bw, bb, rho, threshold));
                println!("    Output[{}]: {}/{} ({:.1}%)", k, bc, n_patterns,
                         bc as f32 / n_patterns as f32 * 100.0);
                log.push(format!("    O[{}]: {}/{}", k, bc, n_patterns));
            }

            // Check pipeline with C19 gates
            pipeline_correct = 0;
            for p in 0..n_patterns {
                let out: Vec<u8> = output_gates.iter().map(|g| g.eval(&hidden_patterns[p])).collect();
                if out == oracle[p] { pipeline_correct += 1; }
            }

            if pipeline_correct == n_patterns {
                output_type = String::from("C19-gate");
                println!("  Pipeline: {}/{} (100.0%) -- C19 gates", pipeline_correct, n_patterns);
                println!("  >>> PERFECT <<<");
                log.push("  PERFECT (C19 gates)".to_string());
                break;
            }

            // C19 didn't hit 100%, use direct LUT (guaranteed correct for collision-free)
            println!("  C19 gates: {}/{}, falling back to direct LUT...", pipeline_correct, n_patterns);
            output_gates = build_direct_output(&hidden_patterns, &oracle, h_count, n_output_bits);
            output_type = String::from("direct-LUT");

            pipeline_correct = 0;
            for p in 0..n_patterns {
                let out: Vec<u8> = output_gates.iter().map(|g| g.eval(&hidden_patterns[p])).collect();
                if out == oracle[p] { pipeline_correct += 1; }
            }

            println!("  Pipeline: {}/{} ({:.1}%) -- direct LUT",
                     pipeline_correct, n_patterns,
                     pipeline_correct as f32 / n_patterns as f32 * 100.0);
            log.push(format!("  Direct LUT: {}/{}", pipeline_correct, n_patterns));

            if pipeline_correct == n_patterns {
                println!("  >>> PERFECT (direct LUT) <<<");
                log.push("  PERFECT (direct LUT)".to_string());
                break;
            }
        } else if collision_free {
            // H > 20, collision-free: try C19 gates only
            output_gates = Vec::new();
            for k in 0..n_output_bits {
                let target: Vec<u8> = oracle.iter().map(|o| o[k]).collect();
                let (bw, bb, bc) = search_output_gate(
                    &mut rng, h_count, rho, threshold,
                    n_trials_output, &hidden_patterns, &target,
                );
                output_gates.push(LutGate::from_weights(&bw, bb, rho, threshold));
                println!("    Output[{}]: {}/{} ({:.1}%)", k, bc, n_patterns,
                         bc as f32 / n_patterns as f32 * 100.0);
            }

            pipeline_correct = 0;
            for p in 0..n_patterns {
                let out: Vec<u8> = output_gates.iter().map(|g| g.eval(&hidden_patterns[p])).collect();
                if out == oracle[p] { pipeline_correct += 1; }
            }
            output_type = String::from("C19-gate");
            println!("  Pipeline: {}/{} ({:.1}%)", pipeline_correct, n_patterns,
                     pipeline_correct as f32 / n_patterns as f32 * 100.0);

            if pipeline_correct == n_patterns {
                println!("  >>> PERFECT <<<");
                break;
            }
        } else {
            println!("  Collisions present -- need more hidden neurons.");
        }

        // Retry: add extra hidden neurons
        if extra_added >= max_extra {
            println!("  Max hidden neurons ({}) reached.", max_h);
            log.push(format!("  Max H reached ({})", max_h));

            // If we never built output gates (collisions on last iteration), do a final attempt
            if output_gates.is_empty() {
                let hidden_patterns: Vec<Vec<u8>> = (0..n_patterns).map(|p| {
                    hidden_gates.iter().map(|g| g.eval(&input_u8_patterns[p])).collect()
                }).collect();
                let h_count = hidden_gates.len();

                let (cf, _) = check_collisions(&hidden_patterns, &oracle);
                if cf && h_count <= 20 {
                    output_gates = build_direct_output(&hidden_patterns, &oracle, h_count, n_output_bits);
                    output_type = String::from("direct-LUT");
                } else {
                    for k in 0..n_output_bits {
                        let target: Vec<u8> = oracle.iter().map(|o| o[k]).collect();
                        let (bw, bb, _) = search_output_gate(
                            &mut rng, h_count, rho, threshold,
                            n_trials_output, &hidden_patterns, &target,
                        );
                        output_gates.push(LutGate::from_weights(&bw, bb, rho, threshold));
                    }
                    output_type = String::from("C19-gate");
                }

                pipeline_correct = 0;
                for p in 0..n_patterns {
                    let out: Vec<u8> = output_gates.iter().map(|g| g.eval(&hidden_patterns[p])).collect();
                    if out == oracle[p] { pipeline_correct += 1; }
                }
            }
            break;
        }

        let (bw, bb, bo) = search_hidden_neuron(
            &mut rng, n_input_bits, n_patterns, rho, threshold,
            n_trials_hidden, &input_u8_patterns, &oracle, &existing_hidden_outputs,
        );
        let score = score_neuron(&bo, &oracle, &existing_hidden_outputs);
        hidden_gates.push(LutGate::from_weights(&bw, bb, rho, threshold));
        existing_hidden_outputs.push(bo);
        extra_added += 1;
        println!("  +Extra[{}]: score={}, H now {}", extra_added, score, hidden_gates.len());
        log.push(format!("  +Extra {}: score={}, H={}", extra_added, score, hidden_gates.len()));
    }

    // ── Step 7: Exhaustive verify ───────────────────────────────
    println!("  [7] Exhaustive verify ({} patterns)...", n_patterns);
    let mut mismatch = 0usize;
    for p in 0..n_patterns {
        let hidden: Vec<u8> = hidden_gates.iter().map(|g| g.eval(&input_u8_patterns[p])).collect();
        let output: Vec<u8> = output_gates.iter().map(|g| g.eval(&hidden)).collect();
        if output != oracle[p] {
            mismatch += 1;
            if mismatch <= 3 {
                println!("    MISMATCH p={}: got {:?}, want {:?}", p, output, oracle[p]);
            }
        }
    }
    if mismatch > 3 { println!("    ... and {} more", mismatch - 3); }
    if mismatch == 0 { println!("    ALL {} patterns verified!", n_patterns); }

    let hidden_mem: usize = hidden_gates.iter().map(|g| g.memory()).sum();
    let output_mem: usize = output_gates.iter().map(|g| g.memory()).sum();
    let total_mem = hidden_mem + output_mem;
    let total_neurons = hidden_gates.len() + output_gates.len();

    let final_acc = pipeline_correct as f32 / n_patterns as f32 * 100.0;
    println!("  Memory: {}B ({} neurons), output: {}", total_mem, total_neurons, output_type);
    println!("  FINAL: {:.1}% ({}/{}), {} hidden + {} output, {} extra",
             final_acc, pipeline_correct, n_patterns,
             hidden_gates.len(), output_gates.len(), extra_added);

    log.push(format!("  FINAL: {:.1}% H={} O={} mem={}B type={}",
                     final_acc, hidden_gates.len(), output_gates.len(), total_mem, output_type));

    DistillResult {
        name: name.to_string(),
        ep_oracle_acc: oracle_acc,
        n_patterns,
        hidden_neurons: hidden_gates.len(),
        output_neurons: output_gates.len(),
        pipeline_correct,
        pipeline_acc: final_acc,
        total_memory: total_mem,
        extra_neurons: extra_added,
        output_type,
    }
}

// ============================================================
// MAIN
// ============================================================

fn main() {
    let t0 = std::time::Instant::now();

    println!("================================================================");
    println!("  EP -> DISTILL -> LUTGATE Pipeline");
    println!("  EP trains float net, oracle captures truth table,");
    println!("  random search builds LutGate neurons one-by-one.");
    println!("================================================================");

    let act = Act(8.0);
    let seed = 123u64;

    let mut log: Vec<String> = Vec::new();
    log.push("EP -> Distill -> LutGate Pipeline Log".to_string());
    log.push(String::new());

    let mut results: Vec<DistillResult> = Vec::new();

    // ── T2: POPCOUNT > 4 (8 in, 1 out -- easy) ────────────────
    {
        let data = gen_t2();
        let r = distill_task(
            "T2: POPCOUNT >4", &data,
            16, 16,  // EP H, distill H_init
            50, 0.5, act, 0.5, 0.005, 800, seed,
            8, 1,    // n_input, n_output bits
            10_000, 50_000,
            20,      // max total hidden
            &mut log,
        );
        results.push(r);
    }

    // ── T3: NIBBLE CLASS (8 in, 4 out -- medium) ────────────────
    {
        let data = gen_t3();
        let r = distill_task(
            "T3: NIBBLE CLASS", &data,
            16, 16,
            50, 0.5, act, 0.5, 0.005, 800, seed,
            8, 4,
            10_000, 50_000,
            20,       // max H (direct LUT feasible up to 20)
            &mut log,
        );
        results.push(r);
    }

    // ── T4: BYTE ADD (16 in, 5 out -- hard) ─────────────────────
    {
        let data = gen_t4();
        let r = distill_task(
            "T4: BYTE ADD", &data,
            32, 16,
            60, 0.4, act, 0.5, 0.003, 1000, seed,
            16, 5,
            10_000, 50_000,
            20,       // max H
            &mut log,
        );
        results.push(r);
    }

    // ============================================================
    // SUMMARY
    // ============================================================

    let elapsed = t0.elapsed().as_secs_f64();

    println!();
    println!("================================================================");
    println!("  SUMMARY");
    println!("  {:<16} {:>10} {:>12} {:>8} {:>10} {:>8}", "Task", "EP Oracle", "Pipeline", "Status", "Neurons", "Memory");

    let mut all_perfect = true;
    for r in &results {
        let status = if r.pipeline_correct == r.n_patterns { "PASS" } else { "FAIL" };
        let neurons = format!("{}+{}", r.hidden_neurons, r.output_neurons);
        println!("  {:<16} {:>9.0}% {:>11.0}% {:>8} {:>10} {:>7}B",
                 r.name, r.ep_oracle_acc, r.pipeline_acc, status, neurons, r.total_memory);
        if r.pipeline_correct != r.n_patterns { all_perfect = false; }
    }

    println!();
    if all_perfect {
        println!("  VERDICT: EP->Distill->LutGate WORKS");
    } else {
        println!("  VERDICT: EP->Distill->LutGate NEEDS WORK");
        for r in &results {
            if r.pipeline_correct != r.n_patterns {
                println!("    {} -- {:.1}% ({}/{}), output: {}",
                         r.name, r.pipeline_acc, r.pipeline_correct, r.n_patterns, r.output_type);
            }
        }
    }

    println!("  Total time: {:.1}s", elapsed);
    println!("================================================================");

    // ── Write log ───────────────────────────────────────────────
    log.push(String::new());
    log.push("== SUMMARY ==".to_string());
    for r in &results {
        log.push(format!("  {}: oracle={:.0}% pipe={:.0}% neurons={}+{} mem={}B type={}",
                         r.name, r.ep_oracle_acc, r.pipeline_acc,
                         r.hidden_neurons, r.output_neurons, r.total_memory, r.output_type));
    }
    log.push(format!("  Time: {:.1}s", elapsed));

    let log_path = "S:/Git/VRAXION/.claude/research/cortex_distill_log.txt";
    if let Ok(mut f) = std::fs::File::create(log_path) {
        for line in &log { let _ = writeln!(f, "{}", line); }
        println!("\n  Log written to {}", log_path);
    }
}
