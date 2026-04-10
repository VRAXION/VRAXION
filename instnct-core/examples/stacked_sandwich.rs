//! Stacked Sandwich: greedy layer-wise depth with checkpoints
//!
//! Phase 1: Train Layer1 (128S+32D, reads raw bytes) -> checkpoint
//! Phase 2: Freeze Layer1, train Layer2 (reads L1 dense output) -> checkpoint
//!
//! No BPTT between layers — each trained independently.
//! Key hypothesis: greedy stacking avoids tick-recurrence gradient issues.
//!
//! Run: cargo run --example stacked_sandwich --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use std::time::Instant;

const CTX: usize = 4;
const N_BYTES: usize = 128;

fn c19_fwd(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01);
    let l = 6.0 * c;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let s = x / c;
    let n = s.floor();
    let t = s - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

fn c19_deriv(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01);
    let l = 6.0 * c;
    if x >= l || x <= -l { return 1.0; }
    let s = x / c;
    let n = s.floor();
    let t = s - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    (sgn + 2.0 * rho * h) * (1.0 - 2.0 * t)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn encode_ctx(context: &[u8], byte_idx: usize) -> [f32; CTX] {
    let mut v = [0.0f32; CTX];
    for (pos, &b) in context.iter().enumerate() {
        if (b as usize) == byte_idx { v[pos] = 1.0; }
    }
    v
}

// ============================================================
// SandwichLayer: single t=1 pass (sparse -> dense -> output)
// Parameterized by sp_input_dim so same struct works for L1 and L2
// ============================================================

#[derive(Clone)]
struct SandwichLayer {
    n_sparse: usize,
    n_dense: usize,
    sp_input_dim: usize,  // CTX=4 for L1, k_bridge for L2
    k_sp: usize,          // how many sparse each dense reads

    sp_w: Vec<f32>,        // n_sparse * sp_input_dim
    sp_bias: Vec<f32>,
    sp_c: Vec<f32>,        // fixed at 1.0
    sp_rho: Vec<f32>,      // trained

    dn_sp: Vec<f32>,       // n_dense * k_sp
    dn_bias: Vec<f32>,
    dn_c: Vec<f32>,
    dn_rho: Vec<f32>,

    out_w: Vec<f32>,       // N_BYTES * n_dense
    out_bias: Vec<f32>,

    dn_reads_sp: Vec<Vec<usize>>,
}

struct FwdCache {
    sp_sum: Vec<f32>,
    sp_act: Vec<f32>,
    dn_sum: Vec<f32>,
    dn_act: Vec<f32>,
    probs: Vec<f32>,
}

struct LayerGrad {
    sp_w: Vec<f32>, sp_bias: Vec<f32>, sp_rho: Vec<f32>,
    dn_sp: Vec<f32>, dn_bias: Vec<f32>, dn_rho: Vec<f32>,
    out_w: Vec<f32>, out_bias: Vec<f32>,
}

impl LayerGrad {
    fn zeros(ns: usize, nd: usize, sp_dim: usize, k_sp: usize) -> Self {
        LayerGrad {
            sp_w: vec![0.0; ns * sp_dim], sp_bias: vec![0.0; ns], sp_rho: vec![0.0; ns],
            dn_sp: vec![0.0; nd * k_sp], dn_bias: vec![0.0; nd], dn_rho: vec![0.0; nd],
            out_w: vec![0.0; N_BYTES * nd], out_bias: vec![0.0; N_BYTES],
        }
    }
    fn add(&mut self, o: &LayerGrad) {
        for (a, b) in self.sp_w.iter_mut().zip(&o.sp_w) { *a += b; }
        for (a, b) in self.sp_bias.iter_mut().zip(&o.sp_bias) { *a += b; }
        for (a, b) in self.sp_rho.iter_mut().zip(&o.sp_rho) { *a += b; }
        for (a, b) in self.dn_sp.iter_mut().zip(&o.dn_sp) { *a += b; }
        for (a, b) in self.dn_bias.iter_mut().zip(&o.dn_bias) { *a += b; }
        for (a, b) in self.dn_rho.iter_mut().zip(&o.dn_rho) { *a += b; }
        for (a, b) in self.out_w.iter_mut().zip(&o.out_w) { *a += b; }
        for (a, b) in self.out_bias.iter_mut().zip(&o.out_bias) { *a += b; }
    }
    fn scale(&mut self, s: f32) {
        for v in &mut self.sp_w { *v *= s; } for v in &mut self.sp_bias { *v *= s; }
        for v in &mut self.sp_rho { *v *= s; } for v in &mut self.dn_sp { *v *= s; }
        for v in &mut self.dn_bias { *v *= s; } for v in &mut self.dn_rho { *v *= s; }
        for v in &mut self.out_w { *v *= s; } for v in &mut self.out_bias { *v *= s; }
    }
    fn norm(&self) -> f32 {
        let mut s = 0.0f32;
        for v in &self.sp_w { s += v*v; } for v in &self.sp_bias { s += v*v; }
        for v in &self.sp_rho { s += v*v; } for v in &self.dn_sp { s += v*v; }
        for v in &self.dn_bias { s += v*v; } for v in &self.dn_rho { s += v*v; }
        for v in &self.out_w { s += v*v; } for v in &self.out_bias { s += v*v; }
        s.sqrt()
    }
}

impl SandwichLayer {
    fn new(ns: usize, nd: usize, sp_dim: usize, k_sp: usize,
           seed: u64, rng: &mut StdRng) -> Self {
        let sc = 0.1f32;
        let k_sp = k_sp.min(ns);
        let dn_reads_sp: Vec<Vec<usize>> = (0..nd).map(|i| {
            let mut idx: Vec<usize> = (0..ns).collect();
            let mut r = StdRng::seed_from_u64(seed + i as u64 * 37 + 13);
            idx.shuffle(&mut r);
            idx[..k_sp].to_vec()
        }).collect();
        SandwichLayer {
            n_sparse: ns, n_dense: nd, sp_input_dim: sp_dim, k_sp,
            sp_w: (0..ns*sp_dim).map(|_| rng.gen_range(-sc..sc)).collect(),
            sp_bias: (0..ns).map(|_| rng.gen_range(-sc..sc)).collect(),
            sp_c: vec![1.0; ns], sp_rho: vec![4.0; ns],
            dn_sp: (0..nd*k_sp).map(|_| rng.gen_range(-sc..sc)).collect(),
            dn_bias: (0..nd).map(|_| rng.gen_range(-sc..sc)).collect(),
            dn_c: vec![1.0; nd], dn_rho: vec![4.0; nd],
            out_w: (0..N_BYTES*nd).map(|_| rng.gen_range(-sc..sc)).collect(),
            out_bias: (0..N_BYTES).map(|_| rng.gen_range(-sc..sc)).collect(),
            dn_reads_sp,
        }
    }

    fn forward(&self, sparse_input: &[f32]) -> FwdCache {
        let (ns, nd, dim) = (self.n_sparse, self.n_dense, self.sp_input_dim);
        let mut sp_sum = vec![0.0f32; ns];
        let mut sp_act = vec![0.0f32; ns];
        for i in 0..ns {
            let mut s = self.sp_bias[i];
            let off = i * dim;
            for j in 0..dim { s += sparse_input[off + j] * self.sp_w[off + j]; }
            sp_sum[i] = s;
            sp_act[i] = c19_fwd(s, self.sp_c[i], self.sp_rho[i]);
        }
        let mut dn_sum = vec![0.0f32; nd];
        let mut dn_act = vec![0.0f32; nd];
        for i in 0..nd {
            let mut s = self.dn_bias[i];
            for (ki, &si) in self.dn_reads_sp[i].iter().enumerate() {
                s += sp_act[si] * self.dn_sp[i * self.k_sp + ki];
            }
            dn_sum[i] = s;
            dn_act[i] = c19_fwd(s, self.dn_c[i], self.dn_rho[i]);
        }
        let mut logits = vec![0.0f32; N_BYTES];
        for b in 0..N_BYTES {
            let mut s = self.out_bias[b];
            for d in 0..nd { s += dn_act[d] * self.out_w[b * nd + d]; }
            logits[b] = s;
        }
        let probs = softmax(&logits);
        FwdCache { sp_sum, sp_act, dn_sum, dn_act, probs }
    }

    fn backward(&self, cache: &FwdCache, target: u8, sparse_input: &[f32]) -> LayerGrad {
        let (ns, nd, dim) = (self.n_sparse, self.n_dense, self.sp_input_dim);
        let mut g = LayerGrad::zeros(ns, nd, dim, self.k_sp);

        let mut d_logits = cache.probs.clone();
        d_logits[target as usize] -= 1.0;

        // Output -> dense
        let mut d_dn = vec![0.0f32; nd];
        for b in 0..N_BYTES {
            g.out_bias[b] = d_logits[b];
            for d in 0..nd {
                g.out_w[b*nd+d] = d_logits[b] * cache.dn_act[d];
                d_dn[d] += d_logits[b] * self.out_w[b*nd+d];
            }
        }

        // Dense -> sparse
        let mut d_sp = vec![0.0f32; ns];
        for i in 0..nd {
            let deriv = c19_deriv(cache.dn_sum[i], self.dn_c[i], self.dn_rho[i]);
            let d_sum = d_dn[i] * deriv;
            g.dn_bias[i] = d_sum;
            for (ki, &si) in self.dn_reads_sp[i].iter().enumerate() {
                g.dn_sp[i*self.k_sp+ki] = d_sum * cache.sp_act[si];
                d_sp[si] += d_sum * self.dn_sp[i*self.k_sp+ki];
            }
            // rho grad
            let s = cache.dn_sum[i]; let c = self.dn_c[i].max(0.01); let l = 6.0*c;
            if s > -l && s < l {
                let sc = s/c; let ft = sc - sc.floor(); let h = ft*(1.0-ft);
                g.dn_rho[i] = d_dn[i] * c * h * h;
            }
        }

        // Sparse -> input
        for i in 0..ns {
            let deriv = c19_deriv(cache.sp_sum[i], self.sp_c[i], self.sp_rho[i]);
            let d_sum = d_sp[i] * deriv;
            g.sp_bias[i] = d_sum;
            let off = i * dim;
            for j in 0..dim { g.sp_w[off+j] = d_sum * sparse_input[off+j]; }
            // rho grad
            let s = cache.sp_sum[i]; let c = self.sp_c[i].max(0.01); let l = 6.0*c;
            if s > -l && s < l {
                let sc = s/c; let ft = sc - sc.floor(); let h = ft*(1.0-ft);
                g.sp_rho[i] = d_sp[i] * c * h * h;
            }
        }
        g
    }

    fn apply_grad(&mut self, g: &LayerGrad, lr: f32) {
        for (w, gw) in self.sp_w.iter_mut().zip(&g.sp_w) { *w -= lr * gw; }
        for (w, gw) in self.sp_bias.iter_mut().zip(&g.sp_bias) { *w -= lr * gw; }
        for i in 0..self.n_sparse { self.sp_rho[i] = (self.sp_rho[i] - lr*g.sp_rho[i]).max(0.0); }
        for (w, gw) in self.dn_sp.iter_mut().zip(&g.dn_sp) { *w -= lr * gw; }
        for (w, gw) in self.dn_bias.iter_mut().zip(&g.dn_bias) { *w -= lr * gw; }
        for i in 0..self.n_dense { self.dn_rho[i] = (self.dn_rho[i] - lr*g.dn_rho[i]).max(0.0); }
        for (w, gw) in self.out_w.iter_mut().zip(&g.out_w) { *w -= lr * gw; }
        for (w, gw) in self.out_bias.iter_mut().zip(&g.out_bias) { *w -= lr * gw; }
    }

    fn predict(&self, inp: &[f32]) -> u8 {
        self.forward(inp).probs.iter().enumerate()
            .max_by(|a,b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i,_)| i as u8).unwrap_or(32)
    }

    fn accuracy(&self, inputs: &[Vec<f32>], targets: &[u8]) -> f64 {
        inputs.iter().zip(targets).filter(|(inp, &t)| self.predict(inp)==t).count() as f64 / targets.len() as f64
    }

    fn top5(&self, inputs: &[Vec<f32>], targets: &[u8]) -> f64 {
        let mut ok = 0;
        for (inp, &t) in inputs.iter().zip(targets) {
            let c = self.forward(inp);
            let mut idx: Vec<(usize,f32)> = c.probs.iter().enumerate().map(|(i,&v)|(i,v)).collect();
            idx.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
            if idx.iter().take(5).any(|(i,_)| *i==t as usize) { ok += 1; }
        }
        ok as f64 / targets.len() as f64
    }

    fn loss_batch(&self, inputs: &[Vec<f32>], targets: &[u8]) -> f64 {
        let mut l = 0.0f64;
        for (inp, &t) in inputs.iter().zip(targets) {
            let c = self.forward(inp);
            l -= (c.probs[t as usize].max(1e-10) as f64).ln();
        }
        l / targets.len() as f64
    }

    fn param_count(&self) -> usize {
        self.sp_w.len() + self.sp_bias.len() + self.sp_rho.len()
        + self.dn_sp.len() + self.dn_bias.len() + self.dn_rho.len()
        + self.out_w.len() + self.out_bias.len()
    }

    // ---- Checkpoint save/load ----

    fn save_checkpoint(&self, path: &str) {
        let mut data = Vec::new();
        data.extend_from_slice(b"CKPT");
        for &v in &[self.n_sparse as u32, self.n_dense as u32,
                     self.sp_input_dim as u32, self.k_sp as u32] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        for vec in [&self.sp_w, &self.sp_bias, &self.sp_c, &self.sp_rho,
                     &self.dn_sp, &self.dn_bias, &self.dn_c, &self.dn_rho,
                     &self.out_w, &self.out_bias] {
            for &v in vec.iter() { data.extend_from_slice(&v.to_le_bytes()); }
        }
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(path, &data).expect("save checkpoint failed");
    }

    fn load_checkpoint(path: &str, seed: u64) -> Option<Self> {
        let data = std::fs::read(path).ok()?;
        if data.len() < 20 || &data[0..4] != b"CKPT" { return None; }
        let r = |off: usize| u32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]) as usize;
        let ns = r(4); let nd = r(8); let dim = r(12); let k_sp = r(16);

        let dn_reads_sp: Vec<Vec<usize>> = (0..nd).map(|i| {
            let mut idx: Vec<usize> = (0..ns).collect();
            let mut r = StdRng::seed_from_u64(seed + i as u64 * 37 + 13);
            idx.shuffle(&mut r);
            idx[..k_sp].to_vec()
        }).collect();

        let mut off = 20usize;
        let mut read_vec = |n: usize| -> Vec<f32> {
            let v: Vec<f32> = (0..n).map(|i| {
                let o = off + i*4;
                f32::from_le_bytes([data[o], data[o+1], data[o+2], data[o+3]])
            }).collect();
            off += n * 4;
            v
        };
        let sp_w = read_vec(ns*dim);
        let sp_bias = read_vec(ns);
        let sp_c = read_vec(ns);
        let sp_rho = read_vec(ns);
        let dn_sp = read_vec(nd*k_sp);
        let dn_bias = read_vec(nd);
        let dn_c = read_vec(nd);
        let dn_rho = read_vec(nd);
        let out_w = read_vec(N_BYTES*nd);
        let out_bias = read_vec(N_BYTES);

        Some(SandwichLayer {
            n_sparse: ns, n_dense: nd, sp_input_dim: dim, k_sp,
            sp_w, sp_bias, sp_c, sp_rho,
            dn_sp, dn_bias, dn_c, dn_rho,
            out_w, out_bias, dn_reads_sp,
        })
    }

    fn train_layer(&mut self,
                   train_inp: &[Vec<f32>], train_tgt: &[u8],
                   test_inp: &[Vec<f32>], test_tgt: &[u8],
                   n_steps: usize, batch_size: usize, ckpt_prefix: &str) {
        let mut lr = 0.01f32;
        let mut rng = StdRng::seed_from_u64(99);
        let (ns, nd, dim) = (self.n_sparse, self.n_dense, self.sp_input_dim);
        let n = train_inp.len();
        let mut indices: Vec<usize> = (0..n).collect();

        for step in 0..n_steps {
            indices.shuffle(&mut rng);
            let batch = &indices[..batch_size.min(n)];

            let mut grad = LayerGrad::zeros(ns, nd, dim, self.k_sp);
            for &idx in batch {
                let cache = self.forward(&train_inp[idx]);
                let g = self.backward(&cache, train_tgt[idx], &train_inp[idx]);
                grad.add(&g);
            }
            grad.scale(1.0 / batch.len() as f32);

            let gn = grad.norm();
            if gn < 1e-8 { continue; }
            grad.scale(1.0 / gn);

            let old_loss: f64 = batch.iter().map(|&i| {
                let c = self.forward(&train_inp[i]);
                -(c.probs[train_tgt[i] as usize].max(1e-10) as f64).ln()
            }).sum::<f64>() / batch.len() as f64;

            let old_net = self.clone();
            self.apply_grad(&grad, lr);

            let new_loss: f64 = batch.iter().map(|&i| {
                let c = self.forward(&train_inp[i]);
                -(c.probs[train_tgt[i] as usize].max(1e-10) as f64).ln()
            }).sum::<f64>() / batch.len() as f64;

            if new_loss < old_loss { lr *= 1.05; } else { *self = old_net; lr *= 0.5; }

            if step % 500 == 0 || step == n_steps - 1 {
                let tr = self.accuracy(train_inp, train_tgt);
                let te = self.accuracy(test_inp, test_tgt);
                let t5 = self.top5(test_inp, test_tgt);
                let lo = self.loss_batch(test_inp, test_tgt);
                println!("    step {:>5}: loss={:.3} train={:.1}% test={:.1}% top5={:.1}% lr={:.5}",
                    step, lo, tr*100.0, te*100.0, t5*100.0, lr);
            }

            if step > 0 && step % 1000 == 0 {
                let path = format!("{}_{}.bin", ckpt_prefix, step);
                self.save_checkpoint(&path);
                println!("    [checkpoint saved: {}]", path);
            }
        }

        let path = format!("{}_final.bin", ckpt_prefix);
        self.save_checkpoint(&path);
        println!("    [checkpoint saved: {}]", path);
    }
}

// ============================================================
// Input preparation helpers
// ============================================================

fn make_l1_inputs(data: &[(Vec<u8>, u8)], sp_byte: &[usize]) -> (Vec<Vec<f32>>, Vec<u8>) {
    let ns = sp_byte.len();
    let inputs: Vec<Vec<f32>> = data.iter().map(|(ctx, _)| {
        let mut flat = vec![0.0f32; ns * CTX];
        for i in 0..ns {
            let thermo = encode_ctx(ctx, sp_byte[i]);
            flat[i*CTX..(i+1)*CTX].copy_from_slice(&thermo);
        }
        flat
    }).collect();
    let targets: Vec<u8> = data.iter().map(|(_, t)| *t).collect();
    (inputs, targets)
}

fn make_l2_inputs(l1_dense: &[Vec<f32>], bridge_idx: &[Vec<usize>]) -> Vec<Vec<f32>> {
    let ns = bridge_idx.len();
    let kb = bridge_idx[0].len();
    l1_dense.iter().map(|dn| {
        let mut flat = vec![0.0f32; ns * kb];
        for i in 0..ns {
            for (ki, &j) in bridge_idx[i].iter().enumerate() {
                flat[i*kb + ki] = dn[j];
            }
        }
        flat
    }).collect()
}

fn bigram_baseline(train: &[(Vec<u8>, u8)], test: &[(Vec<u8>, u8)]) -> (f64, f64) {
    let mut counts = vec![vec![0u32; N_BYTES]; N_BYTES];
    for (ctx, t) in train {
        let last = *ctx.last().unwrap() as usize;
        if last < N_BYTES { counts[last][*t as usize] += 1; }
    }
    let predict = |l: usize| counts[l].iter().enumerate().max_by_key(|(_,&c)|c).map(|(i,_)|i).unwrap_or(32);
    let tr = train.iter().filter(|(c,t)| predict(*c.last().unwrap() as usize)==*t as usize).count() as f64/train.len() as f64;
    let te = test.iter().filter(|(c,t)| predict(*c.last().unwrap() as usize)==*t as usize).count() as f64/test.len() as f64;
    (tr, te)
}

// ============================================================
// main
// ============================================================

fn main() {
    println!("=== STACKED SANDWICH: greedy layer-wise depth ===\n");
    let t0 = Instant::now();

    let ckpt_dir = "checkpoints/stacked_sandwich";

    // Load text
    println!("Loading FineWeb text...");
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
            _ => { println!("  Fallback"); "the quick brown fox jumps over the lazy dog ".repeat(500).bytes().collect() } }
    };
    let text: Vec<u8> = raw.iter().map(|&b| if b < 128 { b } else { 32 }).collect();

    let mut pairs: Vec<(Vec<u8>, u8)> = Vec::new();
    for i in CTX..text.len() {
        let ctx = text[i-CTX..i].to_vec();
        if (text[i] as usize) < N_BYTES { pairs.push((ctx, text[i])); }
    }
    let mut rng = StdRng::seed_from_u64(42);
    pairs.shuffle(&mut rng);
    let train = pairs[..2000].to_vec();
    let test = pairs[2000..3000].to_vec();
    println!("  Train: {}, Test: {}\n", train.len(), test.len());

    let (bi_tr, bi_te) = bigram_baseline(&train, &test);
    println!("  Bigram: train={:.1}% test={:.1}%", bi_tr*100.0, bi_te*100.0);
    println!("  Random: {:.2}%\n", 100.0 / N_BYTES as f64);

    // Config
    let ns = 128;
    let nd = 32;
    let k = 8;
    let k_bridge = 8; // each L2 sparse reads 8 of 32 L1 dense

    // L1 wiring
    let sp_byte: Vec<usize> = (0..ns).map(|i| i % N_BYTES).collect();

    // L2 bridge wiring (which L1 dense each L2 sparse reads)
    let bridge_idx: Vec<Vec<usize>> = (0..ns).map(|i| {
        let mut idx: Vec<usize> = (0..nd).collect();
        let mut r = StdRng::seed_from_u64(i as u64 * 43 + 17);
        idx.shuffle(&mut r);
        idx[..k_bridge.min(nd)].to_vec()
    }).collect();

    // Precompute L1 inputs
    let (train_l1_inp, train_tgt) = make_l1_inputs(&train, &sp_byte);
    let (test_l1_inp, test_tgt) = make_l1_inputs(&test, &sp_byte);

    // =========================================================
    // PHASE 1: Train Layer1 (reads raw bytes)
    // =========================================================
    println!("--- PHASE 1: Layer1 ({}S+{}D, input_dim={}, K={}) ---", ns, nd, CTX, k);

    let l1_seed = 1000u64;
    let l1_ckpt = format!("{}/l1_final.bin", ckpt_dir);
    let mut layer1 = if let Some(l) = SandwichLayer::load_checkpoint(&l1_ckpt, l1_seed) {
        println!("  Loaded from checkpoint!");
        let te = l.accuracy(&test_l1_inp, &test_tgt);
        let t5 = l.top5(&test_l1_inp, &test_tgt);
        println!("  Checkpoint accuracy: test={:.1}% top5={:.1}%", te*100.0, t5*100.0);
        l
    } else {
        let mut l = SandwichLayer::new(ns, nd, CTX, k, l1_seed, &mut rng);
        println!("  Params: {} (training...)", l.param_count());
        l.train_layer(&train_l1_inp, &train_tgt, &test_l1_inp, &test_tgt,
                       5000, 200, &format!("{}/l1", ckpt_dir));
        l
    };

    let l1_acc = layer1.accuracy(&test_l1_inp, &test_tgt);
    let l1_t5 = layer1.top5(&test_l1_inp, &test_tgt);
    println!("\n  => Layer1 alone: test={:.1}% top5={:.1}%\n", l1_acc*100.0, l1_t5*100.0);

    // =========================================================
    // PHASE 2: Freeze Layer1, train Layer2
    // =========================================================
    println!("--- PHASE 2: Freeze L1, train Layer2 ({}S+{}D, bridge={}) ---", ns, nd, k_bridge);

    // Precompute L1 dense for all data (frozen)
    println!("  Precomputing L1 dense activations...");
    let train_l1_dense: Vec<Vec<f32>> = train_l1_inp.iter()
        .map(|inp| layer1.forward(inp).dn_act.clone()).collect();
    let test_l1_dense: Vec<Vec<f32>> = test_l1_inp.iter()
        .map(|inp| layer1.forward(inp).dn_act.clone()).collect();

    let train_l2_inp = make_l2_inputs(&train_l1_dense, &bridge_idx);
    let test_l2_inp = make_l2_inputs(&test_l1_dense, &bridge_idx);

    let l2_seed = 2000u64;
    let l2_ckpt = format!("{}/l2_final.bin", ckpt_dir);
    let mut layer2 = if let Some(l) = SandwichLayer::load_checkpoint(&l2_ckpt, l2_seed) {
        println!("  Loaded from checkpoint!");
        let te = l.accuracy(&test_l2_inp, &test_tgt);
        let t5 = l.top5(&test_l2_inp, &test_tgt);
        println!("  Checkpoint accuracy: test={:.1}% top5={:.1}%", te*100.0, t5*100.0);
        l
    } else {
        let mut l = SandwichLayer::new(ns, nd, k_bridge, k, l2_seed, &mut rng);
        println!("  Params: {} (training...)", l.param_count());
        l.train_layer(&train_l2_inp, &train_tgt, &test_l2_inp, &test_tgt,
                       5000, 200, &format!("{}/l2", ckpt_dir));
        l
    };

    let l2_acc = layer2.accuracy(&test_l2_inp, &test_tgt);
    let l2_t5 = layer2.top5(&test_l2_inp, &test_tgt);
    println!("\n  => Stacked (L1->L2): test={:.1}% top5={:.1}%\n", l2_acc*100.0, l2_t5*100.0);

    // =========================================================
    // PHASE 3: Control — single fat layer (same total params)
    // =========================================================
    println!("--- PHASE 3: Control — single fat layer (~same params) ---");
    {
        // Layer1 has sp_w=512+bias=128+rho=128 + dn_sp=256+bias=32+rho=32 + out=4096+128 = ~5312
        // Layer2 has sp_w=1024+bias=128+rho=128 + dn_sp=256+bias=32+rho=32 + out=4096+128 = ~5824
        // Total stacked: ~11136. Single fat: 256S+64D with same K=8 should be similar.
        let fat_ns = 256;
        let fat_nd = 48; // ~11K params
        let fat_seed = 3000u64;
        let fat_ckpt = format!("{}/fat_final.bin", ckpt_dir);

        let fat_byte: Vec<usize> = (0..fat_ns).map(|i| i % N_BYTES).collect();
        let (fat_tr_inp, _) = make_l1_inputs(&train, &fat_byte);
        let (fat_te_inp, _) = make_l1_inputs(&test, &fat_byte);

        let mut fat = if let Some(l) = SandwichLayer::load_checkpoint(&fat_ckpt, fat_seed) {
            println!("  Loaded from checkpoint!");
            l
        } else {
            let mut l = SandwichLayer::new(fat_ns, fat_nd, CTX, k, fat_seed, &mut rng);
            println!("  Params: {} (training...)", l.param_count());
            l.train_layer(&fat_tr_inp, &train_tgt, &fat_te_inp, &test_tgt,
                           5000, 200, &format!("{}/fat", ckpt_dir));
            l
        };

        let fat_acc = fat.accuracy(&fat_te_inp, &test_tgt);
        let fat_t5 = fat.top5(&fat_te_inp, &test_tgt);
        println!("\n  => Fat single ({}S+{}D): test={:.1}% top5={:.1}%\n", fat_ns, fat_nd, fat_acc*100.0, fat_t5*100.0);
    }

    // =========================================================
    // COMPARISON
    // =========================================================
    println!("=== COMPARISON ===");
    println!("  Layer1 alone (128S+32D):  test={:.1}%  top5={:.1}%", l1_acc*100.0, l1_t5*100.0);
    println!("  Stacked L1->L2:           test={:.1}%  top5={:.1}%", l2_acc*100.0, l2_t5*100.0);
    println!("  Previous best (lang_sw):  test=22.1%  top5=57.8%");
    println!("  Bigram baseline:          test={:.1}%", bi_te*100.0);

    // Predictions comparison
    println!("\n  Layer1 predictions:");
    for s in &["the ", "and ", "is a", "tion", "ing ", "for ", "ent ", "hat "] {
        let ctx: Vec<u8> = s.bytes().collect();
        let mut flat = vec![0.0f32; ns * CTX];
        for i in 0..ns { let th = encode_ctx(&ctx, sp_byte[i]); flat[i*CTX..(i+1)*CTX].copy_from_slice(&th); }
        let pred = layer1.predict(&flat);
        let ch = if pred >= 32 && pred < 127 { pred as char } else { '?' };
        print!("'{}'->'{}'  ", s, ch);
    }
    println!();

    println!("  Stacked predictions:");
    for s in &["the ", "and ", "is a", "tion", "ing ", "for ", "ent ", "hat "] {
        let ctx: Vec<u8> = s.bytes().collect();
        let mut flat = vec![0.0f32; ns * CTX];
        for i in 0..ns { let th = encode_ctx(&ctx, sp_byte[i]); flat[i*CTX..(i+1)*CTX].copy_from_slice(&th); }
        let l1_dn = layer1.forward(&flat).dn_act;
        let mut l2_flat = vec![0.0f32; ns * k_bridge];
        for i in 0..ns { for (ki, &j) in bridge_idx[i].iter().enumerate() { l2_flat[i*k_bridge+ki] = l1_dn[j]; } }
        let pred = layer2.predict(&l2_flat);
        let ch = if pred >= 32 && pred < 127 { pred as char } else { '?' };
        print!("'{}'->'{}'  ", s, ch);
    }
    println!();

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
    println!("\n=== DONE ===");
}
