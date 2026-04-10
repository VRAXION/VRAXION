//! TRUE holographic learning: one-shot, no gradient, no iteration
//!
//! Connection matrix = SUM of outer products of (context, next_byte) pairs.
//! No training loop. No loss function. Just matrix addition.
//! Retrieval: M × encode(context) → predicted next byte.
//!
//! Properties:
//!   - Distributed: every weight encodes ALL patterns
//!   - Superposition: N patterns in one fixed matrix
//!   - Graceful degradation: quantize → all patterns degrade, none breaks
//!   - One-shot: add new pattern = one matrix addition
//!
//! Run: cargo run --example holographic_learn --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use std::time::Instant;

const CTX: usize = 4;
const N: usize = 128;

/// Encode context as distributed vector using random projection
/// Each byte position gets a fixed random vector, combined by element-wise sum
fn encode_context_distributed(context: &[u8], projections: &Vec<Vec<f32>>) -> Vec<f32> {
    let mut v = vec![0.0f32; N];
    for (pos, &byte) in context.iter().enumerate() {
        let key = pos * 256 + byte as usize;
        if key < projections.len() {
            for i in 0..N { v[i] += projections[key][i]; }
        }
    }
    // Normalize
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 { for x in &mut v { *x /= norm; } }
    v
}

/// Simple one-hot for output byte
fn encode_output(byte: u8) -> Vec<f32> {
    let mut v = vec![0.0f32; N];
    if (byte as usize) < N { v[byte as usize] = 1.0; }
    v
}

/// Holographic memory: connection matrix = sum of outer products
struct HoloMemory {
    matrix: Vec<f32>,   // N × N
    projections: Vec<Vec<f32>>,  // random projections for encoding
    n_stored: usize,
}

impl HoloMemory {
    fn new(rng: &mut StdRng) -> Self {
        // Create random projections: CTX positions × 256 possible bytes
        let n_keys = CTX * 256;
        let projections: Vec<Vec<f32>> = (0..n_keys).map(|_| {
            (0..N).map(|_| rng.gen_range(-1.0..1.0f32)).collect()
        }).collect();

        HoloMemory {
            matrix: vec![0.0; N * N],
            projections,
            n_stored: 0,
        }
    }

    /// Store one pattern: M += encode(context) ⊗ encode(next_byte)
    fn store(&mut self, context: &[u8], next_byte: u8) {
        let ctx_vec = encode_context_distributed(context, &self.projections);
        let out_vec = encode_output(next_byte);

        // Outer product addition
        for i in 0..N {
            for j in 0..N {
                self.matrix[i * N + j] += ctx_vec[i] * out_vec[j];
            }
        }
        self.n_stored += 1;
    }

    /// Store all training data at once
    fn store_all(&mut self, data: &[(Vec<u8>, u8)]) {
        for (ctx, target) in data {
            self.store(ctx, *target);
        }
    }

    /// Retrieve: output = M × encode(context)
    fn retrieve(&self, context: &[u8]) -> Vec<f32> {
        let ctx_vec = encode_context_distributed(context, &self.projections);
        let mut output = vec![0.0f32; N];
        for i in 0..N {
            for j in 0..N {
                output[i] += self.matrix[i * N + j] * ctx_vec[j];
            }
        }
        output
    }

    fn predict(&self, context: &[u8]) -> u8 {
        let output = self.retrieve(context);
        output.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u8).unwrap_or(0)
    }

    fn accuracy(&self, data: &[(Vec<u8>, u8)]) -> f64 {
        data.iter().filter(|(ctx, t)| self.predict(ctx) == *t).count() as f64 / data.len() as f64
    }

    fn top5_accuracy(&self, data: &[(Vec<u8>, u8)]) -> f64 {
        let mut correct = 0;
        for (ctx, target) in data {
            let output = self.retrieve(ctx);
            let mut idx: Vec<(usize, f32)> = output.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            if idx.iter().take(5).any(|(i, _)| *i == *target as usize) { correct += 1; }
        }
        correct as f64 / data.len() as f64
    }

    /// Quantize matrix to int levels — test graceful degradation
    fn quantized_accuracy(&self, data: &[(Vec<u8>, u8)], levels: usize) -> f64 {
        let min_v = self.matrix.iter().fold(f32::MAX, |a, &b| a.min(b));
        let max_v = self.matrix.iter().fold(f32::MIN, |a, &b| a.max(b));
        let step = (max_v - min_v) / (levels - 1) as f32;

        let quantized: Vec<f32> = self.matrix.iter().map(|&v| {
            let idx = ((v - min_v) / step).round() as usize;
            min_v + idx.min(levels - 1) as f32 * step
        }).collect();

        // Predict with quantized matrix
        let mut correct = 0;
        for (ctx, target) in data {
            let ctx_vec = encode_context_distributed(ctx, &self.projections);
            let mut output = vec![0.0f32; N];
            for i in 0..N {
                for j in 0..N {
                    output[i] += quantized[i * N + j] * ctx_vec[j];
                }
            }
            let pred = output.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u8).unwrap_or(0);
            if pred == *target { correct += 1; }
        }
        correct as f64 / data.len() as f64
    }
}

/// Bigram baseline
fn bigram_baseline(train: &[(Vec<u8>, u8)], test: &[(Vec<u8>, u8)]) -> (f64, f64) {
    let mut counts = vec![vec![0u32; N]; N];
    for (ctx, t) in train {
        let l = *ctx.last().unwrap() as usize;
        if l < N { counts[l][*t as usize] += 1; }
    }
    let predict = |l: usize| counts[l].iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(32);
    let tr = train.iter().filter(|(c, t)| predict(*c.last().unwrap() as usize) == *t as usize).count() as f64 / train.len() as f64;
    let te = test.iter().filter(|(c, t)| predict(*c.last().unwrap() as usize) == *t as usize).count() as f64 / test.len() as f64;
    (tr, te)
}

fn main() {
    println!("=== TRUE HOLOGRAPHIC LEARNING ===\n");
    println!("No gradient. No iteration. Just M += ctx ⊗ next_byte.\n");

    let t0 = Instant::now();

    // Load text
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
        match o { Ok(o) if o.stdout.len() > 1000 => { println!("  {} bytes", o.stdout.len()); o.stdout }
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

    let (bi_tr, bi_te) = bigram_baseline(&train, &test);
    println!("  Bigram: train={:.1}% test={:.1}%\n", bi_tr * 100.0, bi_te * 100.0);

    // =========================================================
    // TEST 1: Holographic one-shot learning
    // =========================================================
    println!("--- Holographic one-shot ---");
    let t1 = Instant::now();
    let mut mem = HoloMemory::new(&mut rng);
    mem.store_all(&train);
    let store_time = t1.elapsed().as_secs_f64();

    let train_acc = mem.accuracy(&train);
    let test_acc = mem.accuracy(&test);
    let top5 = mem.top5_accuracy(&test);
    println!("  Stored {} patterns in {:.3}s", mem.n_stored, store_time);
    println!("  Train: {:.1}%  Test: {:.1}%  Top5: {:.1}%", train_acc * 100.0, test_acc * 100.0, top5 * 100.0);

    let samples = ["the ", "and ", "is a", "tion", "ing ", "for ", "ent ", "hat "];
    print!("  Predictions: ");
    for s in samples {
        let ctx: Vec<u8> = s.bytes().collect();
        let pred = mem.predict(&ctx);
        let ch = if pred >= 32 && pred < 127 { pred as char } else { '?' };
        print!("'{}'→'{}' ", s, ch);
    }
    println!("\n");

    // =========================================================
    // TEST 2: Graceful degradation — quantize and test
    // =========================================================
    println!("--- Graceful degradation (quantize matrix) ---\n");
    println!("  {:>8} {:>10} {:>10} {:>10}", "levels", "bits", "test_acc", "top5");
    println!("  {}", "=".repeat(42));

    for &levels in &[2, 4, 8, 16, 32, 64, 128, 256, 1024] {
        let bits = (levels as f64).log2().ceil() as u32;
        let q_acc = mem.quantized_accuracy(&test, levels);
        let q_top5 = {
            // Quick top5 for quantized
            let min_v = mem.matrix.iter().fold(f32::MAX, |a, &b| a.min(b));
            let max_v = mem.matrix.iter().fold(f32::MIN, |a, &b| a.max(b));
            let step = (max_v - min_v) / (levels - 1) as f32;
            let quantized: Vec<f32> = mem.matrix.iter().map(|&v| {
                let idx = ((v - min_v) / step).round() as usize;
                min_v + idx.min(levels - 1) as f32 * step
            }).collect();
            let mut correct = 0;
            for (ctx, target) in &test {
                let ctx_vec = encode_context_distributed(ctx, &mem.projections);
                let mut output = vec![0.0f32; N];
                for i in 0..N { for j in 0..N { output[i] += quantized[i * N + j] * ctx_vec[j]; } }
                let mut idx: Vec<(usize, f32)> = output.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                if idx.iter().take(5).any(|(i, _)| *i == *target as usize) { correct += 1; }
            }
            correct as f64 / test.len() as f64
        };
        println!("  {:>8} {:>8} bit {:>9.1}% {:>9.1}%", levels, bits, q_acc * 100.0, q_top5 * 100.0);
    }

    // =========================================================
    // TEST 3: Incremental learning — add patterns one by one
    // =========================================================
    println!("\n--- Incremental learning curve ---\n");
    let mut mem2 = HoloMemory::new(&mut StdRng::seed_from_u64(42));
    println!("  {:>8} {:>10} {:>10} {:>10}", "patterns", "train_acc", "test_acc", "top5");
    println!("  {}", "=".repeat(42));

    for (i, (ctx, target)) in train.iter().enumerate() {
        mem2.store(ctx, *target);
        if (i + 1) % 200 == 0 || i == 0 || i == train.len() - 1 {
            let tr = mem2.accuracy(&train[..i+1]);
            let te = mem2.accuracy(&test);
            let t5 = mem2.top5_accuracy(&test);
            println!("  {:>8} {:>9.1}% {:>9.1}% {:>9.1}%", i + 1, tr * 100.0, te * 100.0, t5 * 100.0);
        }
    }

    // =========================================================
    // TEST 4: Training set size sweep
    // =========================================================
    println!("\n--- Training size sweep ---\n");
    println!("  {:>8} {:>10} {:>10}", "n_train", "test_acc", "top5");
    println!("  {}", "=".repeat(32));

    for &n in &[100, 200, 500, 1000, 2000, 5000, 10000, 20000] {
        if n > pairs.len() - 1000 { break; }
        let tr = pairs[..n].to_vec();
        let te = pairs[n..n+1000].to_vec();
        let mut m = HoloMemory::new(&mut StdRng::seed_from_u64(42));
        m.store_all(&tr);
        let acc = m.accuracy(&te);
        let t5 = m.top5_accuracy(&te);
        println!("  {:>8} {:>9.1}% {:>9.1}%", n, acc * 100.0, t5 * 100.0);
    }

    // =========================================================
    // Comparison summary
    // =========================================================
    println!("\n--- SUMMARY ---\n");
    println!("  Method              Train   Test   Top5   Time      Gradient?");
    println!("  {}", "=".repeat(65));
    println!("  Bigram              {:.1}%  {:.1}%   -      instant   no", bi_tr * 100.0, bi_te * 100.0);
    println!("  Holographic 1-shot  {:.1}%  {:.1}%  {:.1}%  {:.3}s    NO",
        train_acc * 100.0, test_acc * 100.0, top5 * 100.0, store_time);
    println!("  Backprop ReLU       29.4%  25.8%  66.2%  237s      yes (5000 steps)");
    println!("  Backprop C19        25.7%  25.0%  62.7%  12s       yes (5000 steps)");

    println!("\n  Matrix size: {}×{} = {} weights = {} KB",
        N, N, N*N, N*N*4/1024);
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
    println!("\n=== DONE ===");
}
