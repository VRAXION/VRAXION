//! Language prediction with ANALYTIC BACKPROP
//!
//! Same architecture as language_c19.rs but with chain-rule gradient.
//! ~4000x faster than numerical gradient → can run 10K+ steps.
//! Softmax + cross-entropy loss for proper classification.
//!
//! Run: cargo run --example language_backprop --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use std::time::Instant;

const CTX: usize = 4;
const N_BYTES: usize = 128;
const NC: usize = 8;

fn c19_forward(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01);
    let l = 6.0 * c;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let scaled = x / c;
    let n = scaled.floor();
    let t = scaled - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

fn c19_derivative(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01);
    let l = 6.0 * c;
    if x >= l || x <= -l { return 1.0; } // linear region
    let scaled = x / c;
    let n = scaled.floor();
    let t = scaled - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    (sgn + 2.0 * rho * h) * (1.0 - 2.0 * t)
}

fn relu_forward(x: f32) -> f32 { x.max(0.0) }
fn relu_derivative(x: f32) -> f32 { if x > 0.0 { 1.0 } else { 0.0 } }

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Encode context: for byte b, thermo[pos] = 1 if context[pos] == b
fn encode_context(context: &[u8], byte_idx: usize) -> [f32; CTX] {
    let mut v = [0.0f32; CTX];
    for (pos, &b) in context.iter().enumerate() {
        if (b as usize) == byte_idx { v[pos] = 1.0; }
    }
    v
}

struct ForwardCache {
    thermos: Vec<[f32; CTX]>,  // per input neuron
    input_sum: Vec<f32>,       // pre-activation sums
    input_act: Vec<f32>,       // post-activation
    conn: Vec<f32>,            // connectome values
    logits: Vec<f32>,          // raw output
    probs: Vec<f32>,           // softmax output
}

#[derive(Clone)]
struct BackpropNet {
    // Input: N_BYTES neurons, each [w0..w3, w_write, bias] = CTX+2
    inp_w: Vec<f32>,     // N_BYTES * CTX (thermo weights)
    inp_write: Vec<f32>, // N_BYTES (connectome write weight)
    inp_bias: Vec<f32>,  // N_BYTES
    inp_c: Vec<f32>,     // N_BYTES (C19 C param)
    inp_rho: Vec<f32>,   // N_BYTES (C19 rho param)
    // Output: N_BYTES neurons, each [w_conn0..w_conn7, bias] = NC+1
    out_w: Vec<f32>,     // N_BYTES * NC
    out_bias: Vec<f32>,  // N_BYTES
    use_c19: bool,
}

impl BackpropNet {
    fn new(use_c19: bool, rng: &mut StdRng, scale: f32) -> Self {
        BackpropNet {
            inp_w: (0..N_BYTES * CTX).map(|_| rng.gen_range(-scale..scale)).collect(),
            inp_write: (0..N_BYTES).map(|_| rng.gen_range(-scale..scale)).collect(),
            inp_bias: (0..N_BYTES).map(|_| rng.gen_range(-scale..scale)).collect(),
            inp_c: vec![1.0; N_BYTES],
            inp_rho: vec![4.0; N_BYTES],
            out_w: (0..N_BYTES * NC).map(|_| rng.gen_range(-scale..scale)).collect(),
            out_bias: (0..N_BYTES).map(|_| rng.gen_range(-scale..scale)).collect(),
            use_c19,
        }
    }

    fn forward_with_cache(&self, context: &[u8]) -> ForwardCache {
        // Phase 1: input byte detectors
        let mut thermos = Vec::with_capacity(N_BYTES);
        let mut input_sum = vec![0.0f32; N_BYTES];
        let mut input_act = vec![0.0f32; N_BYTES];

        for b in 0..N_BYTES {
            let thermo = encode_context(context, b);
            let mut s = self.inp_bias[b];
            for j in 0..CTX { s += thermo[j] * self.inp_w[b * CTX + j]; }
            input_sum[b] = s;
            input_act[b] = if self.use_c19 {
                c19_forward(s, self.inp_c[b], self.inp_rho[b])
            } else {
                relu_forward(s)
            };
            thermos.push(thermo);
        }

        // Phase 2: connectome
        let mut conn = vec![0.0f32; NC];
        for b in 0..N_BYTES {
            let slot = b % NC;
            conn[slot] += input_act[b] * self.inp_write[b];
        }

        // Phase 3: output
        let mut logits = vec![0.0f32; N_BYTES];
        for b in 0..N_BYTES {
            let mut s = self.out_bias[b];
            for k in 0..NC { s += conn[k] * self.out_w[b * NC + k]; }
            logits[b] = s;
        }

        let probs = softmax(&logits);

        ForwardCache { thermos, input_sum, input_act, conn, logits, probs }
    }

    fn backward(&self, cache: &ForwardCache, target: u8) -> Gradients {
        let mut g = Gradients::zeros(self.use_c19);

        // Step 5→4: d_loss/d_logits = probs - one_hot(target)
        // (softmax + cross-entropy gradient)
        let mut d_logits = cache.probs.clone();
        d_logits[target as usize] -= 1.0;

        // Step 4→3: output layer gradients
        let mut d_conn = vec![0.0f32; NC];
        for b in 0..N_BYTES {
            g.out_bias[b] += d_logits[b];
            for k in 0..NC {
                g.out_w[b * NC + k] += d_logits[b] * cache.conn[k];
                d_conn[k] += d_logits[b] * self.out_w[b * NC + k];
            }
        }

        // Step 3→2: connectome → input activation
        let mut d_input_act = vec![0.0f32; N_BYTES];
        for b in 0..N_BYTES {
            let slot = b % NC;
            d_input_act[b] += d_conn[slot] * self.inp_write[b];
            g.inp_write[b] += d_conn[slot] * cache.input_act[b];
        }

        // Step 2→1: activation derivative
        let mut d_input_sum = vec![0.0f32; N_BYTES];
        for b in 0..N_BYTES {
            let act_deriv = if self.use_c19 {
                c19_derivative(cache.input_sum[b], self.inp_c[b], self.inp_rho[b])
            } else {
                relu_derivative(cache.input_sum[b])
            };
            d_input_sum[b] = d_input_act[b] * act_deriv;
        }

        // Step 1→params: input weights
        for b in 0..N_BYTES {
            g.inp_bias[b] += d_input_sum[b];
            for j in 0..CTX {
                g.inp_w[b * CTX + j] += d_input_sum[b] * cache.thermos[b][j];
            }
        }

        // C19 extra: rho gradient
        if self.use_c19 {
            for b in 0..N_BYTES {
                let s = cache.input_sum[b];
                let c = self.inp_c[b].max(0.01);
                let l = 6.0 * c;
                if s > -l && s < l {
                    let scaled = s / c;
                    let t = scaled - scaled.floor();
                    let h = t * (1.0 - t);
                    // d(c19)/d(rho) = C * h^2
                    g.inp_rho[b] += d_input_act[b] * c * h * h;
                }
            }
        }

        g
    }

    fn predict(&self, context: &[u8]) -> u8 {
        let cache = self.forward_with_cache(context);
        cache.probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
    }

    fn cross_entropy_loss(&self, batch: &[(Vec<u8>, u8)]) -> f64 {
        let mut total = 0.0f64;
        for (ctx, target) in batch {
            let cache = self.forward_with_cache(ctx);
            let p = cache.probs[*target as usize].max(1e-10);
            total -= (p as f64).ln();
        }
        total / batch.len() as f64
    }

    fn accuracy(&self, batch: &[(Vec<u8>, u8)]) -> f64 {
        let mut correct = 0;
        for (ctx, target) in batch {
            if self.predict(ctx) == *target { correct += 1; }
        }
        correct as f64 / batch.len() as f64
    }

    fn top5_accuracy(&self, batch: &[(Vec<u8>, u8)]) -> f64 {
        let mut correct = 0;
        for (ctx, target) in batch {
            let cache = self.forward_with_cache(ctx);
            let mut indexed: Vec<(usize, f32)> = cache.probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            if indexed.iter().take(5).any(|(i, _)| *i == *target as usize) {
                correct += 1;
            }
        }
        correct as f64 / batch.len() as f64
    }

    fn param_count(&self) -> usize {
        self.inp_w.len() + self.inp_write.len() + self.inp_bias.len()
        + self.out_w.len() + self.out_bias.len()
        + if self.use_c19 { self.inp_c.len() + self.inp_rho.len() } else { 0 }
    }
}

struct Gradients {
    inp_w: Vec<f32>,
    inp_write: Vec<f32>,
    inp_bias: Vec<f32>,
    inp_rho: Vec<f32>,
    out_w: Vec<f32>,
    out_bias: Vec<f32>,
}

impl Gradients {
    fn zeros(has_c19: bool) -> Self {
        Gradients {
            inp_w: vec![0.0; N_BYTES * CTX],
            inp_write: vec![0.0; N_BYTES],
            inp_bias: vec![0.0; N_BYTES],
            inp_rho: vec![0.0; N_BYTES],
            out_w: vec![0.0; N_BYTES * NC],
            out_bias: vec![0.0; N_BYTES],
        }
    }

    fn add(&mut self, other: &Gradients) {
        for i in 0..self.inp_w.len() { self.inp_w[i] += other.inp_w[i]; }
        for i in 0..self.inp_write.len() { self.inp_write[i] += other.inp_write[i]; }
        for i in 0..self.inp_bias.len() { self.inp_bias[i] += other.inp_bias[i]; }
        for i in 0..self.inp_rho.len() { self.inp_rho[i] += other.inp_rho[i]; }
        for i in 0..self.out_w.len() { self.out_w[i] += other.out_w[i]; }
        for i in 0..self.out_bias.len() { self.out_bias[i] += other.out_bias[i]; }
    }

    fn scale(&mut self, s: f32) {
        for v in &mut self.inp_w { *v *= s; }
        for v in &mut self.inp_write { *v *= s; }
        for v in &mut self.inp_bias { *v *= s; }
        for v in &mut self.inp_rho { *v *= s; }
        for v in &mut self.out_w { *v *= s; }
        for v in &mut self.out_bias { *v *= s; }
    }

    fn norm(&self) -> f32 {
        let mut s = 0.0f32;
        for v in &self.inp_w { s += v * v; }
        for v in &self.inp_write { s += v * v; }
        for v in &self.inp_bias { s += v * v; }
        for v in &self.inp_rho { s += v * v; }
        for v in &self.out_w { s += v * v; }
        for v in &self.out_bias { s += v * v; }
        s.sqrt()
    }
}

fn apply_gradient(net: &mut BackpropNet, grad: &Gradients, lr: f32) {
    for i in 0..net.inp_w.len() { net.inp_w[i] -= lr * grad.inp_w[i]; }
    for i in 0..net.inp_write.len() { net.inp_write[i] -= lr * grad.inp_write[i]; }
    for i in 0..net.inp_bias.len() { net.inp_bias[i] -= lr * grad.inp_bias[i]; }
    for i in 0..net.out_w.len() { net.out_w[i] -= lr * grad.out_w[i]; }
    for i in 0..net.out_bias.len() { net.out_bias[i] -= lr * grad.out_bias[i]; }
    if net.use_c19 {
        for i in 0..net.inp_rho.len() {
            net.inp_rho[i] = (net.inp_rho[i] - lr * grad.inp_rho[i]).max(0.0);
        }
    }
}

fn train(net: &mut BackpropNet, train_data: &[(Vec<u8>, u8)],
         test_data: &[(Vec<u8>, u8)], n_steps: usize, batch_size: usize) {
    let mut lr = 0.01f32;
    let mut rng = StdRng::seed_from_u64(99);
    let mut shuffled = train_data.to_vec();

    for step in 0..n_steps {
        // Mini-batch
        shuffled.shuffle(&mut rng);
        let batch = &shuffled[..batch_size.min(shuffled.len())];

        // Accumulate gradients
        let mut grad = Gradients::zeros(net.use_c19);
        for (ctx, target) in batch {
            let cache = net.forward_with_cache(ctx);
            let g = net.backward(&cache, *target);
            grad.add(&g);
        }
        grad.scale(1.0 / batch.len() as f32);

        let gn = grad.norm();
        if gn < 1e-8 { continue; }
        grad.scale(1.0 / gn); // normalize

        // Simple SGD with adaptive lr
        let old_loss = net.cross_entropy_loss(batch);
        let old_net = net.clone();
        apply_gradient(net, &grad, lr);
        let new_loss = net.cross_entropy_loss(batch);

        if new_loss < old_loss {
            lr *= 1.05;
        } else {
            *net = old_net;
            lr *= 0.5;
        }

        // Report
        if step % 500 == 0 || step == n_steps - 1 {
            let train_acc = net.accuracy(train_data);
            let test_acc = net.accuracy(test_data);
            let top5 = net.top5_accuracy(test_data);
            let loss = net.cross_entropy_loss(test_data);
            println!("    step {:>5}: loss={:.3} train={:.1}% test={:.1}% top5={:.1}% lr={:.5}",
                step, loss, train_acc * 100.0, test_acc * 100.0, top5 * 100.0, lr);
        }
    }
}

/// Bigram baseline
fn bigram_baseline(train: &[(Vec<u8>, u8)], test: &[(Vec<u8>, u8)]) -> (f64, f64) {
    let mut counts = vec![vec![0u32; N_BYTES]; N_BYTES];
    for (ctx, target) in train {
        let last = *ctx.last().unwrap() as usize;
        if last < N_BYTES && (*target as usize) < N_BYTES {
            counts[last][*target as usize] += 1;
        }
    }
    let predict = |last: usize| -> usize {
        counts[last].iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(32)
    };
    let train_acc = train.iter().filter(|(ctx, t)| predict(*ctx.last().unwrap() as usize) == *t as usize).count() as f64 / train.len() as f64;
    let test_acc = test.iter().filter(|(ctx, t)| predict(*ctx.last().unwrap() as usize) == *t as usize).count() as f64 / test.len() as f64;
    (train_acc, test_acc)
}

fn main() {
    println!("=== LANGUAGE PREDICTION: ANALYTIC BACKPROP ===\n");

    let t0 = Instant::now();

    // Load text
    println!("Loading text data...");
    let raw_text: Vec<u8> = {
        let output = std::process::Command::new("python")
            .arg("-c")
            .arg(r#"
import pyarrow.parquet as pq
t = pq.read_table('S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/000_00000.parquet', columns=['text'])
col = t.column('text')
text = ''
for i in range(500):
    text += str(col[i]) + ' '
    if len(text) > 200000:
        break
import sys
sys.stdout.buffer.write(bytes([b for b in text.encode('ascii', errors='ignore') if 32 <= b < 127 or b == 10]))
"#)
            .output();
        match output {
            Ok(o) if o.stdout.len() > 1000 => { println!("  Loaded {} bytes from FineWeb", o.stdout.len()); o.stdout }
            _ => {
                println!("  Fallback text");
                "the quick brown fox jumps over the lazy dog ".repeat(500).bytes().collect()
            }
        }
    };

    let text: Vec<u8> = raw_text.iter().map(|&b| if b < 128 { b } else { 32 }).collect();
    println!("  Text: {} bytes", text.len());

    // Create pairs
    let mut all_pairs: Vec<(Vec<u8>, u8)> = Vec::new();
    for i in CTX..text.len() {
        let ctx: Vec<u8> = text[i-CTX..i].to_vec();
        let target = text[i];
        if (target as usize) < N_BYTES { all_pairs.push((ctx, target)); }
    }

    let mut rng = StdRng::seed_from_u64(42);
    all_pairs.shuffle(&mut rng);

    let train_size = 2000;
    let test_size = 1000;
    let train_data = all_pairs[..train_size].to_vec();
    let test_data = all_pairs[train_size..train_size + test_size].to_vec();
    println!("  Train: {}, Test: {}\n", train_size, test_size);

    // Baselines
    let (bi_train, bi_test) = bigram_baseline(&train_data, &test_data);
    println!("  Bigram: train={:.1}% test={:.1}%", bi_train * 100.0, bi_test * 100.0);
    println!("  Random: {:.2}%\n", 100.0 / N_BYTES as f64);

    // =========================================================
    // Gradient check: numerical vs analytic (1 sample)
    // =========================================================
    println!("--- Gradient check (1 sample) ---");
    {
        let mut net = BackpropNet::new(true, &mut StdRng::seed_from_u64(42), 0.1);
        let (ctx, target) = &train_data[0];
        let cache = net.forward_with_cache(ctx);
        let analytic = net.backward(&cache, *target);

        // Numerical check on first few params
        let eps = 1e-3f32;
        let mut max_diff = 0.0f32;
        for i in 0..5 {
            let orig = net.inp_w[i];
            net.inp_w[i] = orig + eps;
            let lp = net.cross_entropy_loss(&[(ctx.clone(), *target)]);
            net.inp_w[i] = orig - eps;
            let lm = net.cross_entropy_loss(&[(ctx.clone(), *target)]);
            net.inp_w[i] = orig;
            let numerical = ((lp - lm) / (2.0 * eps as f64)) as f32;
            let diff = (numerical - analytic.inp_w[i]).abs();
            if diff > max_diff { max_diff = diff; }
            println!("  inp_w[{}]: analytic={:.6} numerical={:.6} diff={:.6}",
                i, analytic.inp_w[i], numerical, diff);
        }
        println!("  Max diff: {:.6} {}\n", max_diff,
            if max_diff < 0.01 { "OK" } else { "WARNING: large!" });
    }

    // =========================================================
    // Speed comparison: numerical vs backprop
    // =========================================================
    println!("--- Speed comparison (100 samples) ---");
    {
        let small_batch = &train_data[..100];

        // Backprop
        let t1 = Instant::now();
        let mut net_bp = BackpropNet::new(false, &mut StdRng::seed_from_u64(42), 0.1);
        let mut grad = Gradients::zeros(false);
        for (ctx, target) in small_batch {
            let cache = net_bp.forward_with_cache(ctx);
            let g = net_bp.backward(&cache, *target);
            grad.add(&g);
        }
        let bp_time = t1.elapsed().as_secs_f64();

        // Numerical (just 10 params to estimate)
        let t2 = Instant::now();
        let eps = 1e-3f32;
        for i in 0..10 {
            let orig = net_bp.inp_w[i];
            net_bp.inp_w[i] = orig + eps; let _ = net_bp.cross_entropy_loss(small_batch);
            net_bp.inp_w[i] = orig - eps; let _ = net_bp.cross_entropy_loss(small_batch);
            net_bp.inp_w[i] = orig;
        }
        let num_time_10 = t2.elapsed().as_secs_f64();
        let num_estimated = num_time_10 / 10.0 * net_bp.param_count() as f64;

        println!("  Backprop (all params): {:.4}s", bp_time);
        println!("  Numerical (estimated): {:.1}s", num_estimated);
        println!("  Speedup: {:.0}x\n", num_estimated / bp_time);
    }

    // =========================================================
    // Main experiment: C19 vs ReLU, 5000 steps
    // =========================================================
    let n_steps = 5000;
    let batch_size = 200;

    for (label, use_c19) in [("ReLU", false), ("C19", true)] {
        println!("--- {} ({} steps, batch={}) ---", label, n_steps, batch_size);
        let t1 = Instant::now();
        let mut net = BackpropNet::new(use_c19, &mut StdRng::seed_from_u64(42), 0.1);
        println!("  Params: {}", net.param_count());

        train(&mut net, &train_data, &test_data, n_steps, batch_size);

        let elapsed = t1.elapsed().as_secs_f64();
        println!("  Time: {:.1}s ({:.2}ms/step)", elapsed, elapsed * 1000.0 / n_steps as f64);

        // Show predictions
        let samples = ["the ", "and ", "is a", "tion", "ing ", "for ", "ent ", "hat "];
        print!("  Predictions: ");
        for s in samples {
            let ctx: Vec<u8> = s.bytes().collect();
            let pred = net.predict(&ctx);
            let pred_ch = if pred >= 32 && pred < 127 { pred as char } else { '?' };
            print!("'{}'→'{}' ", s, pred_ch);
        }
        println!("\n");
    }

    println!("Total time: {:.1}s", t0.elapsed().as_secs_f64());
    println!("\n=== DONE ===");
}
