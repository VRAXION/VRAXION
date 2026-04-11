//! Cortex LutGate -- EP Train -> Bake -> Exhaustive Verify
//!
//! Proves that an EP-trained cortex (float) can be baked into LutGate
//! neurons (binary LUT, zero float at runtime) with ZERO deployment loss.
//!
//! TWO baking strategies:
//!
//!   A) Per-neuron LutGate: bake each neuron individually into a LUT,
//!      then wire them as a 2-layer feedforward network.
//!      Works when EP has converged to a regime where a single forward
//!      pass through the weights matches the settled EP dynamics.
//!
//!   B) Whole-network functional bake: run the actual EP float network
//!      on EVERY possible input, record the binary output, store as LUT.
//!      Captures the exact EP behavior (bidirectional settle + threshold).
//!      GUARANTEED to match float accuracy on exhaustive inputs.
//!
//! Tasks:
//!   T2: POPCOUNT >4   (8->16->1)   -- 256 exhaustive inputs
//!   T3: NIBBLE CLASS  (8->16->4)   -- 256 exhaustive inputs
//!   T4: BYTE ADD      (16->16->5)  -- 65536 exhaustive inputs (H=16)
//!       also reports  (16->32->5)  -- float only (output LUT too large)
//!
//! Run: cargo run --example cortex_lutgate --release

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
            b1: vec![0.0; h_dim],
            b2: vec![0.0; out_dim],
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
        sh = h2;
        so = o2;
    }
    (sh, so)
}

/// EP predict: run settle dynamics (bidirectional), threshold output
fn predict_bits(x: &[f32], net: &EpNet, t: usize, dt: f32, act: Act) -> Vec<u8> {
    let dy = vec![0.0f32; net.out_dim];
    let (_, so) = settle(x, &dy, net, t, dt, act, 0.0);
    so.iter().map(|s| {
        let a = act.apply(*s);
        if a.is_nan() { 0u8 } else if a > 0.5 { 1u8 } else { 0u8 }
    }).collect()
}

/// Feedforward predict: single forward pass (no settle loop)
/// This is what per-neuron LutGate baking computes.
fn predict_bits_ff(x: &[f32], net: &EpNet, act: Act) -> Vec<u8> {
    let mut hidden = vec![0.0f32; net.h_dim];
    for j in 0..net.h_dim {
        let mut s = net.b1[j];
        for i in 0..net.in_dim {
            s += net.w1[j * net.in_dim + i] * x[i];
        }
        hidden[j] = act.apply(s);
    }
    // Threshold hidden to binary for output layer
    let hidden_bin: Vec<f32> = hidden.iter().map(|&h| if h > 0.5 { 1.0 } else { 0.0 }).collect();
    let mut output = vec![0.0f32; net.out_dim];
    for k in 0..net.out_dim {
        let mut s = net.b2[k];
        for j in 0..net.h_dim {
            s += net.w2[k * net.h_dim + j] * hidden_bin[j];
        }
        output[k] = act.apply(s);
    }
    output.iter().map(|&o| if o.is_nan() { 0u8 } else if o > 0.5 { 1u8 } else { 0u8 }).collect()
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
            let mut sh = sfh.clone();
            let mut so = sfo.clone();
            for _ in 0..t {
                let (h2, o2) = settle_step(&sh, &so, x, net, dt, act, beta, y);
                sh = h2;
                so = o2;
            }
            let ib = 1.0 / beta;
            for j in 0..net.h_dim {
                let an = act.apply(sh[j]);
                let af = act.apply(sfh[j]);
                for ii in 0..net.in_dim {
                    net.w1[j * net.in_dim + ii] += lr_e * ib * (an * x[ii] - af * x[ii]);
                }
                net.b1[j] += lr_e * ib * (an - af);
            }
            for k in 0..net.out_dim {
                let aon = act.apply(so[k]);
                let aof = act.apply(sfo[k]);
                for j in 0..net.h_dim {
                    net.w2[k * net.h_dim + j] +=
                        lr_e * ib * (aon * act.apply(sh[j]) - aof * act.apply(sfh[j]));
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
// LutGate -- pattern-indexed binary LUT, ZERO float at runtime
// ============================================================

#[allow(dead_code)]
struct LutGate {
    lut: Vec<u8>,    // truth table: index by input bit-pattern -> 0 or 1
    n_inputs: usize,
}

impl LutGate {
    /// Bake an EP-trained neuron into a LUT (per-neuron approach).
    /// Enumerates all 2^n_inputs input patterns, computes
    /// c19(dot_product + bias, rho) > threshold.
    fn from_ep_neuron(weights: &[f32], bias: f32, rho: f32, threshold: f32) -> Self {
        let n = weights.len();
        let n_entries = 1usize << n;
        let mut lut = vec![0u8; n_entries];
        for pattern in 0..n_entries {
            let sum: f32 = (0..n)
                .map(|i| {
                    let input = if pattern & (1 << i) != 0 { 1.0f32 } else { 0.0 };
                    weights[i] * input
                })
                .sum::<f32>()
                + bias;
            let activated = c19(sum, rho);
            lut[pattern] = if activated > threshold { 1 } else { 0 };
        }
        LutGate { lut, n_inputs: n }
    }

    /// Evaluate: integer bit-pattern lookup, ZERO float ops
    fn eval(&self, inputs: &[u8]) -> u8 {
        let mut idx = 0usize;
        for (i, &inp) in inputs.iter().enumerate() {
            if inp != 0 {
                idx |= 1 << i;
            }
        }
        self.lut[idx]
    }

    /// Memory in bytes
    fn memory(&self) -> usize { self.lut.len() }
}

// ============================================================
// Per-neuron LutGate network (Strategy A)
// ============================================================

struct LutNetwork {
    hidden: Vec<LutGate>,
    output: Vec<LutGate>,
}

impl LutNetwork {
    fn from_ep_net(net: &EpNet, rho: f32, threshold: f32) -> Self {
        let mut hidden = Vec::with_capacity(net.h_dim);
        for j in 0..net.h_dim {
            let w_start = j * net.in_dim;
            let weights = &net.w1[w_start..w_start + net.in_dim];
            hidden.push(LutGate::from_ep_neuron(weights, net.b1[j], rho, threshold));
        }
        let mut output = Vec::with_capacity(net.out_dim);
        for k in 0..net.out_dim {
            let w_start = k * net.h_dim;
            let weights = &net.w2[w_start..w_start + net.h_dim];
            output.push(LutGate::from_ep_neuron(weights, net.b2[k], rho, threshold));
        }
        LutNetwork { hidden, output }
    }

    /// Feedforward eval: input -> hidden LUTs -> output LUTs (zero float)
    fn eval(&self, input_bits: &[u8]) -> Vec<u8> {
        let hidden_bits: Vec<u8> = self.hidden.iter().map(|g| g.eval(input_bits)).collect();
        self.output.iter().map(|g| g.eval(&hidden_bits)).collect()
    }

    fn total_memory(&self) -> usize {
        self.hidden.iter().map(|g| g.memory()).sum::<usize>()
            + self.output.iter().map(|g| g.memory()).sum::<usize>()
    }
    fn hidden_memory(&self) -> usize {
        self.hidden.iter().map(|g| g.memory()).sum::<usize>()
    }
    fn output_memory(&self) -> usize {
        self.output.iter().map(|g| g.memory()).sum::<usize>()
    }
    fn neuron_count(&self) -> usize {
        self.hidden.len() + self.output.len()
    }
}

// ============================================================
// Whole-network functional LUT (Strategy B)
// Runs actual EP float inference on every input, stores result.
// GUARANTEED to match float -- zero information loss.
// ============================================================

#[allow(dead_code)]
struct FunctionalLut {
    /// For each output bit: a LUT mapping input pattern -> 0 or 1
    output_luts: Vec<Vec<u8>>,
    n_input_bits: usize,
    n_output_bits: usize,
}

impl FunctionalLut {
    /// Bake by running the actual EP network on every input pattern
    fn from_ep_net(
        net: &EpNet, t: usize, dt: f32, act: Act,
        n_input_bits: usize,
    ) -> Self {
        let n_patterns = 1usize << n_input_bits;
        let n_out = net.out_dim;
        let mut output_luts = vec![vec![0u8; n_patterns]; n_out];

        for pattern in 0..n_patterns {
            let input: Vec<f32> = (0..n_input_bits)
                .map(|i| if pattern & (1 << i) != 0 { 1.0f32 } else { 0.0 })
                .collect();
            let bits = predict_bits(&input, net, t, dt, act);
            for (k, &b) in bits.iter().enumerate() {
                output_luts[k][pattern] = b;
            }
        }

        FunctionalLut {
            output_luts,
            n_input_bits,
            n_output_bits: n_out,
        }
    }

    /// Evaluate: pure integer lookup, ZERO float
    fn eval(&self, input_bits: &[u8]) -> Vec<u8> {
        let mut idx = 0usize;
        for (i, &inp) in input_bits.iter().enumerate() {
            if inp != 0 { idx |= 1 << i; }
        }
        self.output_luts.iter().map(|lut| lut[idx]).collect()
    }

    /// Total memory in bytes
    fn total_memory(&self) -> usize {
        self.output_luts.iter().map(|l| l.len()).sum()
    }
}

// ============================================================
// Data generators (same as cortex_byte.rs)
// ============================================================

fn byte_to_bits(b: u8) -> Vec<f32> {
    (0..8).map(|i| if b & (1 << i) != 0 { 1.0 } else { 0.0 }).collect()
}

/// T2: POPCOUNT > 4
fn gen_t2() -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..=255u8).map(|b| {
        let input = byte_to_bits(b);
        let pop = b.count_ones();
        let label = if pop > 4 { 1.0 } else { 0.0 };
        (input, vec![label])
    }).collect()
}

/// T3: NIBBLE CLASS (one-hot 4 classes)
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

/// T4: BYTE ADD (4-bit + 4-bit -> 5-bit)
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
// Task result
// ============================================================

struct TaskResult {
    name: String,
    arch: String,
    // Float EP (settle-based)
    float_correct: usize,
    float_total: usize,
    // Feedforward float (single-pass, no settle)
    ff_correct: usize,
    // Per-neuron LutGate (Strategy A)
    pn_correct: usize,
    pn_total: usize,
    pn_match_float: usize,      // agrees with EP float
    pn_match_ff: usize,         // agrees with feedforward float
    // Functional LUT (Strategy B)
    fn_correct: usize,
    fn_total: usize,
    fn_match_float: usize,
    // Memory
    hidden_count: usize,
    output_count: usize,
    pn_hidden_mem: usize,
    pn_output_mem: usize,
    pn_total_mem: usize,
    fn_total_mem: usize,
    // Feasibility
    pn_feasible: bool,
}

// ============================================================
// Compute expected output for a given task
// ============================================================

fn compute_expected(task: &str, pattern: u32) -> Vec<u8> {
    match task {
        "T2: POPCOUNT >4" => {
            let byte = (pattern & 0xFF) as u8;
            vec![if byte.count_ones() > 4 { 1u8 } else { 0u8 }]
        }
        "T3: NIBBLE CLASS" => {
            let byte = (pattern & 0xFF) as u8;
            let class = (byte & 0x0F) / 4;
            let mut out = vec![0u8; 4];
            out[class as usize] = 1;
            out
        }
        "T4: BYTE ADD (H=16)" | "T4: BYTE ADD (H=32)" => {
            let a = (pattern & 0x0F) as u8;
            let b = ((pattern >> 8) & 0x0F) as u8;
            let result = a.wrapping_add(b);
            (0..5).map(|i| if result & (1 << i) != 0 { 1u8 } else { 0u8 }).collect()
        }
        _ => panic!("Unknown task: {}", task),
    }
}

// ============================================================
// Run one task
// ============================================================

fn run_task(
    name: &str,
    data: &[(Vec<f32>, Vec<f32>)],
    h_dim: usize,
    t: usize, dt: f32, act: Act,
    beta: f32, lr: f32,
    epochs: usize,
    seed: u64,
    exhaustive_bits: usize,
) -> TaskResult {
    let in_dim = data[0].0.len();
    let out_dim = data[0].1.len();
    let arch = format!("{}->{}->{}",in_dim, h_dim, out_dim);

    println!("\n  -- {} ({}) --", name, arch);
    println!("    EP Training (seed={}, {} epochs)...", seed, epochs);

    // --- 1. EP Train ---
    let mut rng = Rng::new(seed);
    let mut net = EpNet::new(in_dim, h_dim, out_dim, &mut rng);
    train_ep(&mut net, data, t, dt, act, beta, lr, epochs, &mut rng, 200);

    // --- 2. Float eval (EP settle-based, on training data) ---
    let mut float_ok = 0usize;
    for (x, y) in data {
        let out = predict_bits(x, &net, t, dt, act);
        let target: Vec<u8> = y.iter().map(|&v| if v > 0.5 { 1u8 } else { 0u8 }).collect();
        if out == target { float_ok += 1; }
    }
    println!("    EP Float (settle): {:.1}% ({}/{})",
             float_ok as f32 / data.len() as f32 * 100.0, float_ok, data.len());

    // --- 3. Feedforward float eval (single pass, no settle) ---
    let mut ff_ok = 0usize;
    for (x, y) in data {
        let out = predict_bits_ff(x, &net, act);
        let target: Vec<u8> = y.iter().map(|&v| if v > 0.5 { 1u8 } else { 0u8 }).collect();
        if out == target { ff_ok += 1; }
    }
    println!("    Feedforward float: {:.1}% ({}/{})",
             ff_ok as f32 / data.len() as f32 * 100.0, ff_ok, data.len());

    let total_patterns = 1u64 << exhaustive_bits;
    let pn_feasible = h_dim <= 20;

    // --- 4. Per-neuron LutGate (Strategy A) ---
    let (pn_correct, pn_match_float, pn_match_ff, pn_h_mem, pn_o_mem, pn_t_mem) =
    if pn_feasible {
        let rho = act.0;
        let threshold = 0.5;
        let t0_bake = std::time::Instant::now();
        let lut_net = LutNetwork::from_ep_net(&net, rho, threshold);
        let bake_ms = t0_bake.elapsed().as_millis();

        println!("    [A] Per-neuron LutGate baked in {}ms", bake_ms);
        println!("        Hidden: {} x {} = {} bytes",
                 lut_net.hidden.len(),
                 1usize << in_dim,
                 lut_net.hidden_memory());
        println!("        Output: {} x {} = {} bytes",
                 lut_net.output.len(),
                 1usize << h_dim,
                 lut_net.output_memory());
        println!("        Total:  {} bytes, {} neurons",
                 lut_net.total_memory(), lut_net.neuron_count());

        let mut ok = 0usize;
        let mut match_float = 0usize;
        let mut match_ff = 0usize;
        for pattern in 0..total_patterns {
            let ubits: Vec<u8> = (0..exhaustive_bits)
                .map(|i| if pattern & (1u64 << i) != 0 { 1u8 } else { 0u8 })
                .collect();
            let fbits: Vec<f32> = ubits.iter().map(|&b| b as f32).collect();
            let expected = compute_expected(name, pattern as u32);
            let lut_out = lut_net.eval(&ubits);
            if lut_out == expected { ok += 1; }
            let float_out = predict_bits(&fbits, &net, t, dt, act);
            if lut_out == float_out { match_float += 1; }
            let ff_out = predict_bits_ff(&fbits, &net, act);
            if lut_out == ff_out { match_ff += 1; }
        }
        println!("        Accuracy:     {:.1}% ({}/{})",
                 ok as f32 / total_patterns as f32 * 100.0, ok, total_patterns);
        println!("        vs EP float:  {:.1}% match",
                 match_float as f32 / total_patterns as f32 * 100.0);
        println!("        vs FF float:  {:.1}% match",
                 match_ff as f32 / total_patterns as f32 * 100.0);

        (ok, match_float, match_ff,
         lut_net.hidden_memory(), lut_net.output_memory(), lut_net.total_memory())
    } else {
        println!("    [A] Per-neuron: SKIPPED (H={}, output LUT = 2^{} entries)", h_dim, h_dim);
        (0, 0, 0, 0, 0, 0)
    };

    // --- 5. Functional LUT (Strategy B) ---
    println!("    [B] Functional LUT (bake entire EP network)...");
    let t0_fn = std::time::Instant::now();
    let fn_lut = FunctionalLut::from_ep_net(&net, t, dt, act, exhaustive_bits);
    let fn_bake_ms = t0_fn.elapsed().as_millis();
    println!("        Baked in {}ms ({} output LUTs x {} entries)",
             fn_bake_ms, fn_lut.n_output_bits, 1u64 << exhaustive_bits);
    println!("        Total memory: {} bytes", fn_lut.total_memory());

    let mut fn_ok = 0usize;
    let mut fn_match_float = 0usize;
    for pattern in 0..total_patterns {
        let ubits: Vec<u8> = (0..exhaustive_bits)
            .map(|i| if pattern & (1u64 << i) != 0 { 1u8 } else { 0u8 })
            .collect();
        let fbits: Vec<f32> = ubits.iter().map(|&b| b as f32).collect();
        let expected = compute_expected(name, pattern as u32);
        let fn_out = fn_lut.eval(&ubits);
        if fn_out == expected { fn_ok += 1; }
        let float_out = predict_bits(&fbits, &net, t, dt, act);
        if fn_out == float_out { fn_match_float += 1; }

        if total_patterns > 10000 && pattern % (total_patterns / 10) == 0 && pattern > 0 {
            print!("        {:.0}%.. ", pattern as f64 / total_patterns as f64 * 100.0);
        }
    }
    if total_patterns > 10000 { println!(); }

    println!("        Accuracy:     {:.1}% ({}/{})",
             fn_ok as f32 / total_patterns as f32 * 100.0, fn_ok, total_patterns);
    println!("        vs EP float:  {:.1}% match (must be 100%%)",
             fn_match_float as f32 / total_patterns as f32 * 100.0);

    if fn_match_float as u64 == total_patterns {
        println!("        >>> PERFECT: functional LUT == EP float on ALL inputs <<<");
    }

    TaskResult {
        name: name.to_string(), arch,
        float_correct: float_ok, float_total: data.len(),
        ff_correct: ff_ok,
        pn_correct, pn_total: total_patterns as usize,
        pn_match_float, pn_match_ff,
        fn_correct: fn_ok, fn_total: total_patterns as usize,
        fn_match_float,
        hidden_count: h_dim, output_count: out_dim,
        pn_hidden_mem: pn_h_mem, pn_output_mem: pn_o_mem,
        pn_total_mem: pn_t_mem,
        fn_total_mem: fn_lut.total_memory(),
        pn_feasible,
    }
}

// ============================================================
// MAIN
// ============================================================

fn main() {
    let t0 = std::time::Instant::now();

    println!("================================================================");
    println!("  CORTEX LUTGATE -- EP Train -> Bake -> Exhaustive Verify");
    println!("================================================================");
    println!();
    println!("  Strategy A: Per-neuron LutGate (feedforward bake)");
    println!("  Strategy B: Functional LUT (bake entire EP network)");
    println!("  Goal: PROVE zero deployment loss from float to LUT");
    println!();

    let act = Act(8.0);  // C19 rho=8
    let seed = 123u64;

    let mut results: Vec<TaskResult> = Vec::new();

    // -- T2: POPCOUNT >4 --
    {
        let data = gen_t2();
        let r = run_task("T2: POPCOUNT >4", &data, 16, 50, 0.5, act, 0.5, 0.005, 800, seed, 8);
        results.push(r);
    }

    // -- T3: NIBBLE CLASS --
    {
        let data = gen_t3();
        let r = run_task("T3: NIBBLE CLASS", &data, 16, 50, 0.5, act, 0.5, 0.005, 800, seed, 8);
        results.push(r);
    }

    // -- T4: BYTE ADD H=32 (float only, per-neuron LUT too large) --
    {
        let data = gen_t4();
        let r = run_task("T4: BYTE ADD (H=32)", &data, 32, 60, 0.4, act, 0.5, 0.003, 1000, seed, 16);
        results.push(r);
    }

    // -- T4: BYTE ADD H=16 (LUT-feasible) --
    {
        let data = gen_t4();
        let r = run_task("T4: BYTE ADD (H=16)", &data, 16, 60, 0.4, act, 0.5, 0.003, 1000, seed, 16);
        results.push(r);
    }

    // ============================================================
    // FINAL REPORT
    // ============================================================

    let elapsed = t0.elapsed().as_secs_f64();

    println!();
    println!("================================================================");
    println!("  CORTEX LUTGATE -- FINAL REPORT");
    println!("================================================================");

    for r in &results {
        println!();
        println!("-- {} ({}) --", r.name, r.arch);
        println!("  EP Float:     {:.1}% ({}/{})",
                 r.float_correct as f32 / r.float_total as f32 * 100.0,
                 r.float_correct, r.float_total);
        println!("  FF Float:     {:.1}% ({}/{})",
                 r.ff_correct as f32 / r.float_total as f32 * 100.0,
                 r.ff_correct, r.float_total);

        if r.pn_feasible {
            println!("  [A] Per-neuron LutGate:");
            println!("    Accuracy:   {:.1}% ({}/{}) <- exhaustive verified!",
                     r.pn_correct as f32 / r.pn_total as f32 * 100.0,
                     r.pn_correct, r.pn_total);
            println!("    vs EP:      {:.1}%    vs FF: {:.1}%",
                     r.pn_match_float as f32 / r.pn_total as f32 * 100.0,
                     r.pn_match_ff as f32 / r.pn_total as f32 * 100.0);
            println!("    Hidden:     {} x {} = {} bytes",
                     r.hidden_count,
                     r.pn_hidden_mem / r.hidden_count.max(1),
                     r.pn_hidden_mem);
            println!("    Output:     {} x {} = {} bytes",
                     r.output_count,
                     r.pn_output_mem / r.output_count.max(1),
                     r.pn_output_mem);
            println!("    Total mem:  {} bytes, {} neurons",
                     r.pn_total_mem, r.hidden_count + r.output_count);
        } else {
            println!("  [A] Per-neuron: SKIPPED (H={} too large for output LUT)", r.hidden_count);
        }

        println!("  [B] Functional LUT:");
        println!("    Accuracy:   {:.1}% ({}/{}) <- exhaustive verified!",
                 r.fn_correct as f32 / r.fn_total as f32 * 100.0,
                 r.fn_correct, r.fn_total);
        println!("    vs EP:      {:.1}% (must be 100%%)",
                 r.fn_match_float as f32 / r.fn_total as f32 * 100.0);
        println!("    Total mem:  {} bytes", r.fn_total_mem);
    }

    // Verdict
    println!();
    println!("================================================================");
    println!("  VERDICT");
    println!("================================================================");

    let all_fn_match = results.iter().all(|r| r.fn_match_float == r.fn_total);
    let pn_results: Vec<&TaskResult> = results.iter().filter(|r| r.pn_feasible).collect();
    let all_pn_ff_match = pn_results.iter().all(|r| r.pn_match_ff == r.pn_total);

    if all_fn_match {
        println!("  [B] Functional LUT == EP Float on ALL tasks (100%% match)");
        println!("      -> ZERO deployment loss! Baking is EXACT.");
        println!("      -> EP cortex CAN be deployed as pure integer LUT.");
    }

    if all_pn_ff_match {
        println!("  [A] Per-neuron LutGate == Feedforward Float (100%% match)");
        println!("      -> Per-neuron bake is correct for feedforward mode.");
    }

    let any_pn_ep_mismatch = pn_results.iter().any(|r| r.pn_match_float != r.pn_total);
    if any_pn_ep_mismatch && all_fn_match {
        println!();
        println!("  NOTE: Per-neuron LutGate != EP Float because EP uses");
        println!("  bidirectional settle dynamics (hidden <-> output feedback).");
        println!("  Functional LUT captures this exactly. Use Strategy B for");
        println!("  deployment, or retrain for feedforward-compatible weights.");
    }

    println!("================================================================");
    println!("  Total time: {:.1}s", elapsed);
    println!("================================================================");
}
