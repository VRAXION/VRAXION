//! Holographic: every neuron connects to every neuron via backprop
//!
//! 128 neurons = both input AND output (byte detectors + predictors)
//! Every neuron connected to every other = 128×128 connection matrix
//! Multiple ticks: neurons activate each other through connections
//! Prediction: most active neuron after ticks = next byte
//!
//! Previously IMPOSSIBLE with search (2^16384).
//! With backprop: trivial (16K params, 1 backward pass).
//!
//! Run: cargo run --example holographic_backprop --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use std::time::Instant;

const CTX: usize = 4;
const N: usize = 128;  // neurons = byte values

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

fn c19_deriv(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01);
    let l = 6.0 * c;
    if x >= l || x <= -l { return 1.0; }
    let scaled = x / c;
    let n = scaled.floor();
    let t = scaled - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    (sgn + 2.0 * rho * h) * (1.0 - 2.0 * t)
}

fn relu_fwd(x: f32) -> f32 { x.max(0.0) }
fn relu_deriv(x: f32) -> f32 { if x > 0.0 { 1.0 } else { 0.0 } }

fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn encode_context(context: &[u8], byte_idx: usize) -> [f32; CTX] {
    let mut v = [0.0f32; CTX];
    for (pos, &b) in context.iter().enumerate() {
        if (b as usize) == byte_idx { v[pos] = 1.0; }
    }
    v
}

/// Cache for backprop through ticks
struct TickCache {
    pre_act: Vec<Vec<f32>>,   // pre-activation per tick [tick][neuron]
    post_act: Vec<Vec<f32>>,  // post-activation per tick [tick][neuron]
}

#[derive(Clone)]
struct HoloNet {
    ticks: usize,
    // Per neuron: CTX thermo weights + bias = CTX+1
    thermo_w: Vec<f32>,    // N * CTX
    bias: Vec<f32>,        // N
    // Connection matrix: neuron i → neuron j
    conn: Vec<f32>,        // N * N (conn[i*N + j] = weight from i to j)
    // C19 params
    c_param: Vec<f32>,     // N
    rho_param: Vec<f32>,   // N
    use_c19: bool,
}

impl HoloNet {
    fn new(ticks: usize, use_c19: bool, rng: &mut StdRng, scale: f32) -> Self {
        HoloNet {
            ticks,
            thermo_w: (0..N * CTX).map(|_| rng.gen_range(-scale..scale)).collect(),
            bias: (0..N).map(|_| rng.gen_range(-scale..scale)).collect(),
            conn: (0..N * N).map(|_| rng.gen_range(-scale..scale)).collect(),
            c_param: vec![1.0; N],
            rho_param: vec![4.0; N],
            use_c19,
        }
    }

    fn forward_with_cache(&self, context: &[u8]) -> (Vec<f32>, TickCache) {
        let mut cache = TickCache {
            pre_act: Vec::new(),
            post_act: Vec::new(),
        };

        // Initial activation from thermo input
        let mut act = vec![0.0f32; N];

        // Tick 0: thermo input only
        let mut pre = vec![0.0f32; N];
        for i in 0..N {
            let thermo = encode_context(context, i);
            let mut s = self.bias[i];
            for j in 0..CTX { s += thermo[j] * self.thermo_w[i * CTX + j]; }
            pre[i] = s;
            act[i] = if self.use_c19 {
                c19_forward(s, self.c_param[i], self.rho_param[i])
            } else {
                relu_fwd(s)
            };
        }
        cache.pre_act.push(pre);
        cache.post_act.push(act.clone());

        // Subsequent ticks: neurons interact through connections
        for _t in 1..self.ticks {
            let prev = act.clone();
            let mut pre = vec![0.0f32; N];
            for i in 0..N {
                let thermo = encode_context(context, i);
                let mut s = self.bias[i];
                for j in 0..CTX { s += thermo[j] * self.thermo_w[i * CTX + j]; }
                // Read from ALL other neurons
                for j in 0..N {
                    s += prev[j] * self.conn[j * N + i]; // j writes to i
                }
                pre[i] = s;
                act[i] = if self.use_c19 {
                    c19_forward(s, self.c_param[i], self.rho_param[i])
                } else {
                    relu_fwd(s)
                };
            }
            cache.pre_act.push(pre);
            cache.post_act.push(act.clone());
        }

        // Output: softmax over final activations
        let probs = softmax(&act);
        (probs, cache)
    }

    fn backward(&self, context: &[u8], probs: &[f32], target: u8, cache: &TickCache) -> HoloGrad {
        let mut g = HoloGrad::zeros();

        // d_loss/d_act at final tick = probs - one_hot
        let mut d_act = probs.to_vec();
        d_act[target as usize] -= 1.0;

        // Backprop through ticks (reverse order)
        for t in (0..self.ticks).rev() {
            // d_act → d_pre (through activation derivative)
            let mut d_pre = vec![0.0f32; N];
            for i in 0..N {
                let deriv = if self.use_c19 {
                    c19_deriv(cache.pre_act[t][i], self.c_param[i], self.rho_param[i])
                } else {
                    relu_deriv(cache.pre_act[t][i])
                };
                d_pre[i] = d_act[i] * deriv;
            }

            // d_pre → param gradients
            let thermo_cache: Vec<[f32; CTX]> = (0..N).map(|i| encode_context(context, i)).collect();

            for i in 0..N {
                // Thermo weight gradients
                for j in 0..CTX {
                    g.thermo_w[i * CTX + j] += d_pre[i] * thermo_cache[i][j];
                }
                // Bias gradient
                g.bias[i] += d_pre[i];

                // C19 rho gradient
                if self.use_c19 {
                    let s = cache.pre_act[t][i];
                    let c = self.c_param[i].max(0.01);
                    let l = 6.0 * c;
                    if s > -l && s < l {
                        let scaled = s / c;
                        let tt = scaled - scaled.floor();
                        let h = tt * (1.0 - tt);
                        g.rho_param[i] += d_act[i] * c * h * h;
                    }
                }
            }

            // Connection gradients (tick > 0 only, since tick 0 has no prev activation)
            if t > 0 {
                let prev_act = &cache.post_act[t - 1];
                for i in 0..N {
                    for j in 0..N {
                        // conn[j*N + i] = weight from j to i
                        g.conn[j * N + i] += d_pre[i] * prev_act[j];
                    }
                }

                // Backprop d_act to previous tick
                let mut d_act_prev = vec![0.0f32; N];
                for j in 0..N {
                    for i in 0..N {
                        d_act_prev[j] += d_pre[i] * self.conn[j * N + i];
                    }
                }
                d_act = d_act_prev;
            }
        }

        g
    }

    fn predict(&self, context: &[u8]) -> u8 {
        let (probs, _) = self.forward_with_cache(context);
        probs.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u8).unwrap_or(0)
    }

    fn accuracy(&self, batch: &[(Vec<u8>, u8)]) -> f64 {
        batch.iter().filter(|(ctx, t)| self.predict(ctx) == *t).count() as f64 / batch.len() as f64
    }

    fn top5_accuracy(&self, batch: &[(Vec<u8>, u8)]) -> f64 {
        let mut correct = 0;
        for (ctx, target) in batch {
            let (probs, _) = self.forward_with_cache(ctx);
            let mut idx: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            if idx.iter().take(5).any(|(i, _)| *i == *target as usize) { correct += 1; }
        }
        correct as f64 / batch.len() as f64
    }

    fn cross_entropy(&self, batch: &[(Vec<u8>, u8)]) -> f64 {
        let mut total = 0.0f64;
        for (ctx, target) in batch {
            let (probs, _) = self.forward_with_cache(ctx);
            total -= (probs[*target as usize].max(1e-10) as f64).ln();
        }
        total / batch.len() as f64
    }

    fn param_count(&self) -> usize {
        N * CTX + N + N * N + if self.use_c19 { N * 2 } else { 0 }
    }
}

struct HoloGrad {
    thermo_w: Vec<f32>,
    bias: Vec<f32>,
    conn: Vec<f32>,
    rho_param: Vec<f32>,
}

impl HoloGrad {
    fn zeros() -> Self {
        HoloGrad {
            thermo_w: vec![0.0; N * CTX],
            bias: vec![0.0; N],
            conn: vec![0.0; N * N],
            rho_param: vec![0.0; N],
        }
    }
    fn add(&mut self, o: &HoloGrad) {
        for i in 0..self.thermo_w.len() { self.thermo_w[i] += o.thermo_w[i]; }
        for i in 0..self.bias.len() { self.bias[i] += o.bias[i]; }
        for i in 0..self.conn.len() { self.conn[i] += o.conn[i]; }
        for i in 0..self.rho_param.len() { self.rho_param[i] += o.rho_param[i]; }
    }
    fn scale(&mut self, s: f32) {
        for v in &mut self.thermo_w { *v *= s; }
        for v in &mut self.bias { *v *= s; }
        for v in &mut self.conn { *v *= s; }
        for v in &mut self.rho_param { *v *= s; }
    }
    fn norm(&self) -> f32 {
        let mut s = 0.0f32;
        for v in &self.thermo_w { s += v * v; }
        for v in &self.bias { s += v * v; }
        for v in &self.conn { s += v * v; }
        for v in &self.rho_param { s += v * v; }
        s.sqrt()
    }
}

fn apply_grad(net: &mut HoloNet, g: &HoloGrad, lr: f32) {
    for i in 0..net.thermo_w.len() { net.thermo_w[i] -= lr * g.thermo_w[i]; }
    for i in 0..net.bias.len() { net.bias[i] -= lr * g.bias[i]; }
    for i in 0..net.conn.len() { net.conn[i] -= lr * g.conn[i]; }
    if net.use_c19 {
        for i in 0..net.rho_param.len() {
            net.rho_param[i] = (net.rho_param[i] - lr * g.rho_param[i]).max(0.0);
        }
    }
}

fn main() {
    println!("=== HOLOGRAPHIC: every neuron ↔ every neuron, backprop ===\n");
    println!("Previously IMPOSSIBLE (2^16384 search space)");
    println!("Now: 1 backward pass, ~17K params, trivial\n");

    let t0 = Instant::now();

    // Load text
    println!("Loading text...");
    let raw: Vec<u8> = {
        let o = std::process::Command::new("python").arg("-c").arg(r#"
import pyarrow.parquet as pq
t = pq.read_table('S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/000_00000.parquet', columns=['text'])
c = t.column('text')
text = ''
for i in range(500):
    text += str(c[i]) + ' '
    if len(text) > 200000: break
import sys
sys.stdout.buffer.write(bytes([b for b in text.encode('ascii', errors='ignore') if 32 <= b < 127 or b == 10]))
"#).output();
        match o { Ok(o) if o.stdout.len() > 1000 => { println!("  {} bytes from FineWeb", o.stdout.len()); o.stdout }
            _ => { println!("  Fallback"); "the quick brown fox ".repeat(1000).bytes().collect() } }
    };
    let text: Vec<u8> = raw.iter().map(|&b| if b < 128 { b } else { 32 }).collect();

    let mut pairs: Vec<(Vec<u8>, u8)> = Vec::new();
    for i in CTX..text.len() {
        let ctx = text[i-CTX..i].to_vec();
        if (text[i] as usize) < N { pairs.push((ctx, text[i])); }
    }
    let mut rng = StdRng::seed_from_u64(42);
    pairs.shuffle(&mut rng);
    let train = pairs[..2000].to_vec();
    let test = pairs[2000..3000].to_vec();
    println!("  Train: {}, Test: {}\n", train.len(), test.len());

    // Bigram baseline
    let mut counts = vec![vec![0u32; N]; N];
    for (ctx, t) in &train { let l = *ctx.last().unwrap() as usize; if l < N { counts[l][*t as usize] += 1; } }
    let bi_test = test.iter().filter(|(ctx, t)| {
        let l = *ctx.last().unwrap() as usize;
        counts[l].iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0) == *t as usize
    }).count() as f64 / test.len() as f64;
    println!("  Bigram baseline: {:.1}%\n", bi_test * 100.0);

    let n_steps = 5000;
    let batch_size = 200;

    for &ticks in &[2, 3] {
        for (label, use_c19) in [("ReLU", false), ("C19", true)] {
            println!("--- {} ticks={} ({} steps) ---", label, ticks, n_steps);
            let t1 = Instant::now();
            let mut net = HoloNet::new(ticks, use_c19, &mut StdRng::seed_from_u64(42), 0.05);
            println!("  Params: {} ({} connections)", net.param_count(), N * N);

            let mut lr = 0.01f32;
            let mut shuffled = train.clone();

            for step in 0..n_steps {
                shuffled.shuffle(&mut rng);
                let batch = &shuffled[..batch_size.min(shuffled.len())];

                let mut grad = HoloGrad::zeros();
                for (ctx, target) in batch {
                    let (probs, cache) = net.forward_with_cache(ctx);
                    let g = net.backward(ctx, &probs, *target, &cache);
                    grad.add(&g);
                }
                grad.scale(1.0 / batch.len() as f32);
                let gn = grad.norm();
                if gn > 1e-8 { grad.scale(1.0 / gn); }

                let old = net.clone();
                let ol = net.cross_entropy(batch);
                apply_grad(&mut net, &grad, lr);
                if net.cross_entropy(batch) < ol { lr *= 1.02; }
                else { net = old; lr *= 0.5; }

                if step % 500 == 0 || step == n_steps - 1 {
                    let train_acc = net.accuracy(&train);
                    let test_acc = net.accuracy(&test);
                    let top5 = net.top5_accuracy(&test);
                    let loss = net.cross_entropy(&test);
                    println!("    step {:>5}: loss={:.3} train={:.1}% test={:.1}% top5={:.1}%",
                        step, loss, train_acc * 100.0, test_acc * 100.0, top5 * 100.0);
                }
            }

            let elapsed = t1.elapsed().as_secs_f64();
            println!("  Time: {:.1}s ({:.2}ms/step)", elapsed, elapsed * 1000.0 / n_steps as f64);

            let samples = ["the ", "and ", "is a", "tion", "ing ", "for ", "ent ", "hat "];
            print!("  Predictions: ");
            for s in samples {
                let ctx: Vec<u8> = s.bytes().collect();
                let pred = net.predict(&ctx);
                let ch = if pred >= 32 && pred < 127 { pred as char } else { '?' };
                print!("'{}'→'{}' ", s, ch);
            }
            println!("\n");
        }
    }

    println!("Total: {:.1}s", t0.elapsed().as_secs_f64());
    println!("\n=== DONE ===");
}
