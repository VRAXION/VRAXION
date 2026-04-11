//! Holographic + ALU: learned selector drives parallel fixed ALUs
//!
//! Architecture:
//!   Input (a, b, op) → Holographic Layer → 6-way selector
//!   ALL 6 ALUs compute in parallel on (a, b)
//!   MUX picks the right ALU result based on selector
//!
//! The holographic layer ONLY learns WHICH ALU to use.
//! The ALUs are fixed, verified, integer-only.
//!
//! Run: cargo run --example holo_alu --release

use std::io::Write;

fn c19(x: f32, rho: f32) -> f32 {
    let c = 1.0f32; let l = 6.0;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
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

fn log(f: &mut std::fs::File, msg: &str) {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
    let s = d.as_secs(); let h = (s/3600)%24; let m = (s/60)%60; let sec = s%60;
    let line = format!("[{:02}:{:02}:{:02}] {}\n", h, m, sec, msg);
    print!("{}", line);
    f.write_all(line.as_bytes()).ok();
    f.flush().ok();
}

// ============================================================
// Fixed ALU (integer LUT gates)
// ============================================================

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
            if all.iter().all(|&v| ((v*d as f32).round() - v*d as f32).abs() < 1e-6) { denom = d; break; }
        }
        let w_int: Vec<i32> = w.iter().map(|&v| (v*denom as f32).round() as i32).collect();
        let bias_int = (bias*denom as f32).round() as i32;
        let mut min_s = bias_int; let mut max_s = bias_int;
        for &wi in &w_int { if wi > 0 { max_s += wi; } else { min_s += wi; } }
        let mut lut = vec![0u8; (max_s - min_s + 1) as usize];
        for s in min_s..=max_s {
            let y = c19(s as f32 / denom as f32, rho);
            lut[(s - min_s) as usize] = if y > thr { 1 } else { 0 };
        }
        LutGate { w_int, bias_int, lut, min_sum: min_s }
    }
    fn eval(&self, inputs: &[u8]) -> u8 {
        let s: i32 = inputs.iter().zip(&self.w_int).map(|(&i,&w)| i as i32 * w).sum::<i32>() + self.bias_int;
        self.lut[(s - self.min_sum) as usize]
    }
}

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

    // Returns ALL 6 results at once (parallel execution)
    fn execute_all(&self, a: u8, b: u8) -> [u8; 6] {
        [
            self.add4(a, b),
            self.sub4(a, b),
            self.and4(a, b),
            self.or4(a, b),
            self.xor4(a, b),
            self.cmp_gt(a, b),
        ]
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
    fn cmp_gt(&self, a: u8, b: u8) -> u8 {
        let mut bn = 0u8;
        for bit in 0..4 { bn |= self.not_g.eval(&[(b>>bit)&1]) << bit; }
        let mut c = 1u8;
        let mut r = 0u8;
        for bit in 0..4 {
            let ab = (a>>bit)&1; let bb = (bn>>bit)&1;
            r |= self.xor3.eval(&[ab,bb,c]) << bit;
            c = self.maj.eval(&[ab,bb,c]);
        }
        if c == 1 && (r & 0xF) != 0 { 1 } else { 0 }
    }

    fn op_name(op: usize) -> &'static str {
        ["ADD","SUB","AND","OR","XOR","CMP>"][op]
    }

    fn expected(a: u8, b: u8, op: usize) -> u8 {
        match op {
            0 => (a.wrapping_add(b)) & 0xF,
            1 => (a.wrapping_sub(b)) & 0xF,
            2 => a & b,
            3 => a | b,
            4 => a ^ b,
            5 => if a > b { 1 } else { 0 },
            _ => 0,
        }
    }
}

// ============================================================
// Selector Network: learns which ALU to pick
// ============================================================

struct Selector {
    n_input: usize,   // encoded input size
    n_hidden: usize,
    n_ops: usize,     // 6 ALU operations
    w1: Vec<f32>,     // n_input × n_hidden
    b1: Vec<f32>,
    w2: Vec<f32>,     // n_hidden × n_ops
    b2: Vec<f32>,
    use_c19: bool,
    rho: f32,
}

impl Selector {
    fn new(n_hidden: usize, use_c19: bool) -> Self {
        let n_input = 11; // 4(a) + 4(b) + 3(op bits)
        let n_ops = 6;
        let mut rng = 12345u64;
        let scale = 0.1;
        let mut rand_f = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = ((rng >> 33) % 65536) as f32; // safe range
            (bits / 65536.0 - 0.5) * scale
        };
        Selector {
            n_input, n_hidden, n_ops,
            w1: (0..n_input*n_hidden).map(|_| rand_f()).collect(),
            b1: vec![0.0; n_hidden],
            w2: (0..n_hidden*n_ops).map(|_| rand_f()).collect(),
            b2: vec![0.0; n_ops],
            use_c19, rho: 8.0,
        }
    }

    fn encode(a: u8, b: u8, op: u8) -> Vec<f32> {
        let mut v = vec![0.0f32; 11];
        for bit in 0..4 { v[bit] = ((a >> bit) & 1) as f32; }
        for bit in 0..4 { v[4+bit] = ((b >> bit) & 1) as f32; }
        for bit in 0..3 { v[8+bit] = ((op >> bit) & 1) as f32; }
        v
    }

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Hidden
        let mut pre = vec![0.0f32; self.n_hidden];
        for j in 0..self.n_hidden {
            let mut s = self.b1[j];
            for i in 0..self.n_input { s += input[i] * self.w1[i*self.n_hidden+j]; }
            pre[j] = s;
        }
        let hid: Vec<f32> = if self.use_c19 {
            pre.iter().map(|&x| c19(x, self.rho)).collect()
        } else {
            pre.iter().map(|&x| relu(x)).collect()
        };

        // Output logits (6-way)
        let mut logits = vec![0.0f32; self.n_ops];
        for j in 0..self.n_ops {
            let mut s = self.b2[j];
            for i in 0..self.n_hidden { s += hid[i] * self.w2[i*self.n_ops+j]; }
            logits[j] = s;
        }

        let probs = softmax(&logits);
        (pre, hid, probs)
    }
}

// ============================================================
// Training
// ============================================================

fn train_selector(logf: &mut std::fs::File, use_c19: bool) {
    let label = if use_c19 { "C19" } else { "ReLU" };
    let alu = ParallelAlu::new();
    let n_hidden = 32;
    let mut sel = Selector::new(n_hidden, use_c19);

    log(logf, &format!("\n=== {} Selector (hidden={}) ===", label, n_hidden));

    // Generate training data
    let mut data: Vec<(u8, u8, u8)> = Vec::new(); // (a, b, op)
    let mut rng = 42u64;
    for _ in 0..10000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let a = ((rng >> 16) & 0xF) as u8;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let b = ((rng >> 16) & 0xF) as u8;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let op = ((rng >> 16) % 6) as u8;
        data.push((a, b, op));
    }

    let lr = 0.01f32;
    let n_params = sel.w1.len() + sel.b1.len() + sel.w2.len() + sel.b2.len();
    let mut m_adam = vec![0.0f32; n_params];
    let mut v_adam = vec![0.0f32; n_params];
    let mut t_adam = 0i32;

    for epoch in 0..300 {
        let mut total_loss = 0.0f32;
        let mut correct_select = 0u32;
        let mut correct_result = 0u32;
        let mut total = 0u32;
        let mut grad = vec![0.0f32; n_params];

        for &(a, b, op) in &data {
            let input = Selector::encode(a, b, op);
            let (pre, hid, probs) = sel.forward(&input);

            // Target: one-hot for correct op
            let target_op = op as usize;

            // Cross-entropy loss
            let p = probs[target_op].max(1e-7);
            total_loss -= p.ln();

            // Did the selector pick the right op?
            let picked = probs.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if picked == target_op { correct_select += 1; }

            // Did we get the right result via the picked ALU?
            let all_results = alu.execute_all(a, b);
            let expected = ParallelAlu::expected(a, b, target_op);
            if all_results[picked] == expected { correct_result += 1; }
            total += 1;

            // Backprop: d_loss/d_logits for softmax + cross-entropy
            let mut d_logits = probs.clone();
            d_logits[target_op] -= 1.0; // softmax + CE gradient

            // w2 grad
            let w1_len = sel.w1.len();
            let b1_len = sel.b1.len();
            let w2_off = w1_len + b1_len;
            let b2_off = w2_off + sel.w2.len();

            let mut d_hid = vec![0.0f32; sel.n_hidden];
            for i in 0..sel.n_hidden {
                for j in 0..sel.n_ops {
                    grad[w2_off + i*sel.n_ops+j] += hid[i] * d_logits[j];
                    d_hid[i] += sel.w2[i*sel.n_ops+j] * d_logits[j];
                }
            }
            for j in 0..sel.n_ops {
                grad[b2_off + j] += d_logits[j];
            }

            // Hidden activation gradient
            for j in 0..sel.n_hidden {
                let d_act = if use_c19 {
                    d_hid[j] * c19_deriv(pre[j], sel.rho)
                } else {
                    d_hid[j] * relu_deriv(pre[j])
                };
                for i in 0..sel.n_input {
                    grad[i*sel.n_hidden+j] += input[i] * d_act;
                }
                grad[w1_len + j] += d_act;
            }
        }

        // Adam update
        t_adam += 1;
        let n = data.len() as f32;
        let mut params: Vec<&mut f32> = Vec::new();
        for p in sel.w1.iter_mut() { params.push(p); }
        for p in sel.b1.iter_mut() { params.push(p); }
        for p in sel.w2.iter_mut() { params.push(p); }
        for p in sel.b2.iter_mut() { params.push(p); }
        for (i, p) in params.iter_mut().enumerate() {
            let g = grad[i] / n;
            m_adam[i] = 0.9 * m_adam[i] + 0.1 * g;
            v_adam[i] = 0.999 * v_adam[i] + 0.001 * g * g;
            let mh = m_adam[i] / (1.0 - 0.9f32.powi(t_adam));
            let vh = v_adam[i] / (1.0 - 0.999f32.powi(t_adam));
            **p -= lr * mh / (vh.sqrt() + 1e-8);
        }

        if epoch % 30 == 0 || epoch == 299 {
            let grad_norm: f32 = grad.iter().map(|g| g*g).sum::<f32>().sqrt() / n;
            let w1_norm: f32 = sel.w1.iter().map(|w| w*w).sum::<f32>().sqrt();
            log(logf, &format!("  [{}] Epoch {:4} | loss={:.4} | select={:.1}% | result={:.1}% | grad={:.6} | w1={:.4}",
                label, epoch, total_loss/n,
                correct_select as f32/total as f32*100.0,
                correct_result as f32/total as f32*100.0,
                grad_norm, w1_norm));
        }
    }

    // Final eval on ALL inputs
    log(logf, &format!("\n  [{}] === Final Evaluation (all 16×16×6 = 1536) ===", label));
    let mut per_op = vec![(0u32, 0u32); 6]; // (correct, total)
    let mut select_ok = 0u32;
    for a in 0..16u8 {
        for b in 0..16u8 {
            for op in 0..6u8 {
                let input = Selector::encode(a, b, op);
                let (_, _, probs) = sel.forward(&input);
                let picked = probs.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
                if picked == op as usize { select_ok += 1; }

                let all_results = alu.execute_all(a, b);
                let expected = ParallelAlu::expected(a, b, op as usize);
                per_op[op as usize].1 += 1;
                if all_results[picked] == expected { per_op[op as usize].0 += 1; }
            }
        }
    }

    log(logf, &format!("  [{}] Selector accuracy: {}/1536 ({:.1}%)",
        label, select_ok, select_ok as f32/1536.0*100.0));
    for op in 0..6 {
        log(logf, &format!("  [{}] {}: {}/{} ({:.1}%)",
            label, ParallelAlu::op_name(op), per_op[op].0, per_op[op].1,
            per_op[op].0 as f32 / per_op[op].1 as f32 * 100.0));
    }
    let tc: u32 = per_op.iter().map(|p|p.0).sum();
    log(logf, &format!("  [{}] OVERALL result accuracy: {}/1536 ({:.1}%)", label, tc, tc as f32/1536.0*100.0));
}

fn main() {
    let log_path = "instnct-core/holo_alu_log.txt";
    let mut logf = std::fs::File::create(log_path).unwrap();

    log(&mut logf, "========================================");
    log(&mut logf, "=== HOLOGRAPHIC ALU SELECTOR TEST ===");
    log(&mut logf, "========================================");

    // ALU self-test
    let alu = ParallelAlu::new();
    log(&mut logf, "\n=== ALU Self-Test ===");
    for op in 0..6 {
        let mut ok = 0;
        for a in 0..16u8 {
            for b in 0..16u8 {
                let results = alu.execute_all(a, b);
                if results[op] == ParallelAlu::expected(a, b, op) { ok += 1; }
            }
        }
        log(&mut logf, &format!("  {}: {}/256 {}", ParallelAlu::op_name(op), ok,
            if ok == 256 { "OK" } else { "FAIL" }));
    }

    let t0 = std::time::Instant::now();

    // Test both activations
    train_selector(&mut logf, false); // ReLU
    train_selector(&mut logf, true);  // C19

    log(&mut logf, &format!("\nTotal time: {:.1}s", t0.elapsed().as_secs_f64()));
    log(&mut logf, "=== DONE ===");
}
