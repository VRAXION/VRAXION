//! Language prediction: byte-detector neurons + connectome + C19
//!
//! 256 input neurons (one per byte), each gets thermo of positions
//! Connectome relay → output neurons → predict next byte
//! Compare: C19 vs ReLU, and vs bigram baseline
//!
//! Run: cargo run --example language_c19 --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rayon::prelude::*;
use std::time::Instant;

const CTX: usize = 4;       // context window (bytes)
const N_BYTES: usize = 128;  // ASCII printable (simplify to 0-127)
const NC: usize = 8;         // connectome relay neurons

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01);
    let rho = rho.max(0.0);
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

fn relu(x: f32) -> f32 { x.max(0.0) }

/// Encode context: for byte b, thermo[pos] = 1 if context[pos] == b
fn encode_context(context: &[u8], byte_idx: usize) -> [f32; CTX] {
    let mut v = [0.0f32; CTX];
    for (pos, &b) in context.iter().enumerate() {
        if (b as usize) == byte_idx { v[pos] = 1.0; }
    }
    v
}

/// Simple language model: byte detectors → connectome → output
#[derive(Clone)]
struct LangNet {
    // Per input byte: CTX weights + connectome_write(1) + bias(1) = CTX+2 params
    input_params: Vec<f32>,  // N_BYTES * (CTX + 2)
    input_c: Vec<f32>,       // N_BYTES
    input_rho: Vec<f32>,     // N_BYTES
    // Per output byte: NC read weights + bias(1) = NC+1 params
    output_params: Vec<f32>, // N_BYTES * (NC + 1)
    use_c19: bool,
}

impl LangNet {
    fn new(use_c19: bool, rng: &mut StdRng, scale: f32) -> Self {
        let inp_size = N_BYTES * (CTX + 2);
        let out_size = N_BYTES * (NC + 1);
        LangNet {
            input_params: (0..inp_size).map(|_| rng.gen_range(-scale..scale)).collect(),
            input_c: vec![1.0; N_BYTES],
            input_rho: vec![4.0; N_BYTES],
            output_params: (0..out_size).map(|_| rng.gen_range(-scale..scale)).collect(),
            use_c19,
        }
    }

    fn forward(&self, context: &[u8]) -> Vec<f32> {
        // Phase 1: input byte detectors
        let mut input_act = vec![0.0f32; N_BYTES];
        for b in 0..N_BYTES {
            let thermo = encode_context(context, b);
            let o = b * (CTX + 2);
            let mut s = self.input_params[o + CTX + 1]; // bias
            for j in 0..CTX { s += thermo[j] * self.input_params[o + j]; }
            input_act[b] = if self.use_c19 {
                c19(s, self.input_c[b], self.input_rho[b])
            } else {
                relu(s)
            };
        }

        // Phase 2: write to connectome
        let mut conn = vec![0.0f32; NC];
        for b in 0..N_BYTES {
            let o = b * (CTX + 2);
            let w_write = self.input_params[o + CTX]; // write weight
            let slot = b % NC;
            conn[slot] += input_act[b] * w_write;
        }

        // Phase 3: output neurons read connectome
        let mut output = vec![0.0f32; N_BYTES];
        for b in 0..N_BYTES {
            let o = b * (NC + 1);
            let mut s = self.output_params[o + NC]; // bias
            for k in 0..NC { s += conn[k] * self.output_params[o + k]; }
            output[b] = s; // raw logit (no activation on output)
        }

        output
    }

    fn predict(&self, context: &[u8]) -> u8 {
        let logits = self.forward(context);
        logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
    }

    /// Cross-entropy-ish loss: MSE on target byte's logit vs rest
    fn loss_on_batch(&self, batch: &[(Vec<u8>, u8)]) -> f64 {
        let mut total = 0.0f64;
        for (ctx, target) in batch {
            let logits = self.forward(ctx);
            // Simplified: MSE where target byte should be high, rest low
            for b in 0..N_BYTES {
                let target_val = if b == *target as usize { 1.0f32 } else { 0.0f32 };
                let d = logits[b] - target_val;
                total += (d * d) as f64;
            }
        }
        total / batch.len() as f64
    }

    fn accuracy_on_batch(&self, batch: &[(Vec<u8>, u8)]) -> f64 {
        let mut correct = 0;
        for (ctx, target) in batch {
            if self.predict(ctx) == *target { correct += 1; }
        }
        correct as f64 / batch.len() as f64
    }

    fn top5_accuracy(&self, batch: &[(Vec<u8>, u8)]) -> f64 {
        let mut correct = 0;
        for (ctx, target) in batch {
            let logits = self.forward(ctx);
            let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            if indexed.iter().take(5).any(|(i, _)| *i == *target as usize) {
                correct += 1;
            }
        }
        correct as f64 / batch.len() as f64
    }

    fn gradient_on_batch(&mut self, batch: &[(Vec<u8>, u8)]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let eps = 1e-3f32;

        let mut g_inp = vec![0.0f32; self.input_params.len()];
        for i in 0..self.input_params.len() {
            let orig = self.input_params[i];
            self.input_params[i] = orig + eps; let lp = self.loss_on_batch(batch);
            self.input_params[i] = orig - eps; let lm = self.loss_on_batch(batch);
            self.input_params[i] = orig;
            g_inp[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }

        let mut g_out = vec![0.0f32; self.output_params.len()];
        for i in 0..self.output_params.len() {
            let orig = self.output_params[i];
            self.output_params[i] = orig + eps; let lp = self.loss_on_batch(batch);
            self.output_params[i] = orig - eps; let lm = self.loss_on_batch(batch);
            self.output_params[i] = orig;
            g_out[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }

        let mut gc = vec![0.0f32; N_BYTES];
        let mut gr = vec![0.0f32; N_BYTES];
        if self.use_c19 {
            for i in 0..N_BYTES {
                let orig = self.input_c[i];
                self.input_c[i] = orig + eps; let lp = self.loss_on_batch(batch);
                self.input_c[i] = orig - eps; let lm = self.loss_on_batch(batch);
                self.input_c[i] = orig;
                gc[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
                let orig = self.input_rho[i];
                self.input_rho[i] = orig + eps; let lp = self.loss_on_batch(batch);
                self.input_rho[i] = orig - eps; let lm = self.loss_on_batch(batch);
                self.input_rho[i] = orig;
                gr[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
            }
        }

        (g_inp, g_out, gc, gr)
    }
}

fn train(net: &mut LangNet, train_batch: &[(Vec<u8>, u8)], steps: usize) -> Vec<(usize, f64, f64)> {
    let mut lr = 0.01f32;
    let mut history = Vec::new();

    for step in 0..steps {
        if step % 10 == 0 {
            let acc = net.accuracy_on_batch(train_batch);
            let loss = net.loss_on_batch(train_batch);
            history.push((step, acc, loss));
        }

        let (g_inp, g_out, gc, gr) = net.gradient_on_batch(train_batch);
        let gn: f32 = g_inp.iter().chain(g_out.iter()).chain(gc.iter()).chain(gr.iter())
            .map(|x| x * x).sum::<f32>().sqrt();
        if gn < 1e-8 { break; }

        let old_inp = net.input_params.clone();
        let old_out = net.output_params.clone();
        let old_c = net.input_c.clone();
        let old_r = net.input_rho.clone();
        let ol = net.loss_on_batch(train_batch);

        for att in 0..5 {
            for i in 0..net.input_params.len() { net.input_params[i] = old_inp[i] - lr * g_inp[i] / gn; }
            for i in 0..net.output_params.len() { net.output_params[i] = old_out[i] - lr * g_out[i] / gn; }
            if net.use_c19 {
                for i in 0..N_BYTES { net.input_c[i] = (old_c[i] - lr * gc[i] / gn).max(0.01); }
                for i in 0..N_BYTES { net.input_rho[i] = (old_r[i] - lr * gr[i] / gn).max(0.0); }
            }
            if net.loss_on_batch(train_batch) < ol { lr *= 1.1; break; }
            else {
                lr *= 0.5;
                if att == 4 {
                    net.input_params = old_inp.clone();
                    net.output_params = old_out.clone();
                    net.input_c = old_c.clone();
                    net.input_rho = old_r.clone();
                }
            }
        }
    }

    let acc = net.accuracy_on_batch(train_batch);
    let loss = net.loss_on_batch(train_batch);
    history.push((steps, acc, loss));
    history
}

/// Bigram baseline: P(next | last byte)
fn bigram_baseline(train: &[(Vec<u8>, u8)], test: &[(Vec<u8>, u8)]) -> (f64, f64) {
    let mut counts = vec![vec![0u32; N_BYTES]; N_BYTES];
    for (ctx, target) in train {
        let last = *ctx.last().unwrap() as usize;
        if last < N_BYTES && (*target as usize) < N_BYTES {
            counts[last][*target as usize] += 1;
        }
    }

    let predict = |last: usize| -> usize {
        counts[last].iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0)
    };

    let train_acc = train.iter().filter(|(ctx, target)| {
        predict(*ctx.last().unwrap() as usize) == *target as usize
    }).count() as f64 / train.len() as f64;

    let test_acc = test.iter().filter(|(ctx, target)| {
        predict(*ctx.last().unwrap() as usize) == *target as usize
    }).count() as f64 / test.len() as f64;

    (train_acc, test_acc)
}

fn main() {
    println!("=== LANGUAGE PREDICTION: byte detectors + connectome ===\n");

    let t0 = Instant::now();

    // Load text data from a simple source
    // Read raw bytes from parquet via a simpler approach: just read ASCII text
    println!("Loading text data...");

    // Read text directly - use a simple approach
    let raw_text: Vec<u8> = {
        // Try to read from parquet using a subprocess, fallback to generated text
        let output = std::process::Command::new("python")
            .arg("-c")
            .arg(r#"
import pyarrow.parquet as pq
t = pq.read_table('S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/000_00000.parquet', columns=['text'])
col = t.column('text')
text = ''
for i in range(200):
    text += str(col[i]) + ' '
    if len(text) > 50000:
        break
# Only printable ASCII
import sys
sys.stdout.buffer.write(bytes([b for b in text.encode('ascii', errors='ignore') if 32 <= b < 127 or b == 10]))
"#)
            .output();

        match output {
            Ok(o) if o.stdout.len() > 1000 => {
                println!("  Loaded {} bytes from FineWeb", o.stdout.len());
                o.stdout
            }
            _ => {
                // Fallback: English-like text
                println!("  Parquet load failed, using built-in English text");
                let text = "the quick brown fox jumps over the lazy dog and the cat sat on the mat while the bird flew over the tree the sun was shining bright and the wind was blowing gently through the leaves of the old oak tree that stood in the middle of the garden where children played every afternoon after school the teacher said that reading is important for learning new things and understanding the world around us she also mentioned that writing helps express thoughts and ideas clearly and effectively in both personal and professional settings the students listened carefully and took notes ";
                text.repeat(100).bytes().collect()
            }
        }
    };

    // Clamp to ASCII 0-127
    let text: Vec<u8> = raw_text.iter().map(|&b| if b < 128 { b } else { 32 }).collect();
    println!("  Text size: {} bytes", text.len());

    // Create training pairs: (context[CTX], next_byte)
    let mut all_pairs: Vec<(Vec<u8>, u8)> = Vec::new();
    for i in CTX..text.len() {
        let ctx: Vec<u8> = text[i-CTX..i].to_vec();
        let target = text[i];
        if (target as usize) < N_BYTES {
            all_pairs.push((ctx, target));
        }
    }
    println!("  Total pairs: {}", all_pairs.len());

    // Train/test split
    let mut rng = StdRng::seed_from_u64(42);
    all_pairs.shuffle(&mut rng);

    // Use small batches for gradient (numerical gradient is slow with many params)
    let train_size = 200;
    let test_size = 500;
    let train_batch: Vec<(Vec<u8>, u8)> = all_pairs[..train_size].to_vec();
    let test_batch: Vec<(Vec<u8>, u8)> = all_pairs[train_size..train_size+test_size].to_vec();

    println!("  Train: {}, Test: {}", train_size, test_size);

    // Bigram baseline
    let (bi_train, bi_test) = bigram_baseline(&train_batch, &test_batch);
    println!("\n  Bigram baseline: train={:.1}% test={:.1}%", bi_train * 100.0, bi_test * 100.0);

    // Random baseline
    let random_acc = 1.0 / N_BYTES as f64;
    println!("  Random baseline: {:.2}%", random_acc * 100.0);

    // =========================================================
    // Test 1: C19 vs ReLU
    // =========================================================
    println!("\n--- C19 vs ReLU (50 gradient steps, small batch) ---\n");

    let n_steps = 50;

    for (label, use_c19) in [("ReLU", false), ("C19", true)] {
        let t1 = Instant::now();
        let mut net = LangNet::new(use_c19, &mut StdRng::seed_from_u64(42), 0.1);

        let before_acc = net.accuracy_on_batch(&train_batch);
        let history = train(&mut net, &train_batch, n_steps);

        let train_acc = net.accuracy_on_batch(&train_batch);
        let test_acc = net.accuracy_on_batch(&test_batch);
        let top5_test = net.top5_accuracy(&test_batch);

        println!("  {}: train={:.1}% test={:.1}% top5={:.1}% ({:.1}s)",
            label, train_acc * 100.0, test_acc * 100.0, top5_test * 100.0,
            t1.elapsed().as_secs_f64());
        println!("    Progress: {:.1}% → {:.1}% ({}→{} steps)",
            before_acc * 100.0, train_acc * 100.0, 0, n_steps);

        // Show what it predicts
        let samples = ["the ", "and ", "is a", "tion", "ing "];
        print!("    Predictions: ");
        for s in samples {
            let ctx: Vec<u8> = s.bytes().collect();
            let pred = net.predict(&ctx);
            let pred_ch = if pred >= 32 && pred < 127 { pred as char } else { '?' };
            print!("'{}'→'{}' ", s, pred_ch);
        }
        println!();
    }

    // =========================================================
    // Test 2: More steps, check convergence
    // =========================================================
    println!("\n--- Longer training (200 steps) ---\n");

    for (label, use_c19) in [("ReLU", false), ("C19", true)] {
        let t1 = Instant::now();
        let mut net = LangNet::new(use_c19, &mut StdRng::seed_from_u64(42), 0.1);

        let history = train(&mut net, &train_batch, 200);

        let train_acc = net.accuracy_on_batch(&train_batch);
        let test_acc = net.accuracy_on_batch(&test_batch);
        let top5_test = net.top5_accuracy(&test_batch);

        println!("  {}: train={:.1}% test={:.1}% top5={:.1}% ({:.1}s)",
            label, train_acc * 100.0, test_acc * 100.0, top5_test * 100.0,
            t1.elapsed().as_secs_f64());

        // Training curve
        print!("    Curve: ");
        for (step, acc, _loss) in &history {
            if *step % 40 == 0 { print!("s{}={:.0}% ", step, acc * 100.0); }
        }
        println!();
    }

    // =========================================================
    // Param count summary
    // =========================================================
    println!("\n--- Architecture summary ---");
    let inp_params = N_BYTES * (CTX + 2);
    let out_params = N_BYTES * (NC + 1);
    let c_params = N_BYTES * 2; // C + rho
    println!("  Input neurons: {} (byte detectors)", N_BYTES);
    println!("  Input params: {} ({} per neuron × {})", inp_params, CTX + 2, N_BYTES);
    println!("  Connectome: {} relay neurons", NC);
    println!("  Output neurons: {} (byte predictors)", N_BYTES);
    println!("  Output params: {} ({} per neuron × {})", out_params, NC + 1, N_BYTES);
    println!("  C19 extra: {} (C + rho per input neuron)", c_params);
    println!("  Total: {} params (ReLU) / {} params (C19)",
        inp_params + out_params, inp_params + out_params + c_params);

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
    println!("\n=== DONE ===");
}
