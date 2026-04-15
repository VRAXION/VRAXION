//! L1 Conv Feature Extractor — next char prediction on Alice
//!
//! Input: context bytes → LUT_2N → ctx × 2 int8 values (all side by side)
//! Architecture: Conv1D(k=3) → pool → dense → 27-way prediction
//! Task: predict the next character after the context
//! Eval: held-out 20% test set
//!
//! Run: cargo run --example l1_conv_nextchar --release

use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1)) }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn normal(&mut self) -> f32 {
        let u1 = (((self.next() >> 33) % 65536) as f32 / 65536.0).max(1e-7);
        let u2 = ((self.next() >> 33) % 65536) as f32 / 65536.0;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        if hi <= lo { lo } else { lo + (self.next() as usize % (hi - lo)) }
    }
}

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = std::fs::read(path).expect("read corpus");
    raw.iter().filter_map(|&b| match b {
        b'a'..=b'z' => Some(b - b'a'),
        b'A'..=b'Z' => Some(b - b'A'),
        b' ' | b'\n' | b'\t' | b'\r' => Some(26),
        _ => None,
    }).collect()
}

fn encode_ctx(corpus: &[u8], start: usize, ctx: usize) -> Vec<f32> {
    corpus[start..start + ctx].iter().flat_map(|&ch| {
        let e = LUT[ch as usize];
        [e[0] as f32, e[1] as f32]
    }).collect()
}

struct ConvBrain {
    // Conv: n_filters kernels, each k × channels
    conv_w: Vec<Vec<f32>>,
    conv_b: Vec<f32>,
    // Dense: pool_dim → hdim → 27
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    k: usize, nf: usize, ch: usize, ctx: usize, hdim: usize,
    pool_dim: usize,
}

impl ConvBrain {
    fn new(ctx: usize, ch: usize, k: usize, nf: usize, hdim: usize, rng: &mut Rng) -> Self {
        let fan_in = k * ch;
        let n_pos = ctx - k + 1;
        // Flatten all conv positions (works for small ctx, last-N for large)
        let pool_dim = n_pos * nf;
        let sc = (2.0 / fan_in as f32).sqrt();
        let s1 = (2.0 / pool_dim as f32).sqrt();
        let s2 = (2.0 / hdim as f32).sqrt();
        ConvBrain {
            conv_w: (0..nf).map(|_| (0..fan_in).map(|_| rng.normal() * sc).collect()).collect(),
            conv_b: vec![0.0; nf],
            w1: (0..hdim).map(|_| (0..pool_dim).map(|_| rng.normal() * s1).collect()).collect(),
            b1: vec![0.0; hdim],
            w2: (0..27).map(|_| (0..hdim).map(|_| rng.normal() * s2).collect()).collect(),
            b2: vec![0.0; 27],
            k, nf, ch, ctx, hdim, pool_dim,
        }
    }

    fn params(&self) -> usize {
        self.nf * self.k * self.ch + self.nf + self.pool_dim * self.hdim + self.hdim + self.hdim * 27 + 27
    }

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let n_pos = self.ctx - self.k + 1;

        // Conv + ReLU
        let mut conv_out = vec![0.0f32; n_pos * self.nf];
        for p in 0..n_pos {
            for f in 0..self.nf {
                let mut v = self.conv_b[f];
                for ki in 0..self.k {
                    for d in 0..self.ch {
                        v += self.conv_w[f][ki * self.ch + d] * input[(p + ki) * self.ch + d];
                    }
                }
                conv_out[p * self.nf + f] = v.max(0.0);
            }
        }

        // Flatten conv output directly (preserves position info)
        // Dense 1: conv_flat → hdim (ReLU)
        let mut h = vec![0.0f32; self.hdim];
        for k in 0..self.hdim {
            h[k] = self.b1[k];
            for j in 0..self.pool_dim { h[k] += self.w1[k][j] * conv_out[j]; }
            h[k] = h[k].max(0.0);
        }
        let pool = conv_out.clone(); // keep for backprop compatibility

        // Dense 2: hdim → 27
        let mut logits = vec![0.0f32; 27];
        for c in 0..27 {
            logits[c] = self.b2[c];
            for k in 0..self.hdim { logits[c] += self.w2[c][k] * h[k]; }
        }

        (conv_out, pool, h, logits)
    }

    fn train_step(&mut self, input: &[f32], target: u8, lr: f32) {
        let n_pos = self.ctx - self.k + 1;
        let (conv_out, pool, h, logits) = self.forward(input);

        // Softmax + cross-entropy gradient
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = vec![0.0f32; 27];
        let mut s = 0.0f32;
        for c in 0..27 { probs[c] = (logits[c] - mx).exp(); s += probs[c]; }
        for c in 0..27 { probs[c] /= s; }
        let mut dl = probs; dl[target as usize] -= 1.0;

        // Backprop dense 2
        let mut dh = vec![0.0f32; self.hdim];
        for c in 0..27 {
            for k in 0..self.hdim {
                dh[k] += dl[c] * self.w2[c][k];
                self.w2[c][k] -= lr * dl[c] * h[k];
            }
            self.b2[c] -= lr * dl[c];
        }

        // Backprop dense 1 (ReLU)
        let mut dp = vec![0.0f32; self.pool_dim];
        for k in 0..self.hdim {
            if h[k] <= 0.0 { continue; }
            for j in 0..self.pool_dim {
                dp[j] += dh[k] * self.w1[k][j];
                self.w1[k][j] -= lr * dh[k] * pool[j];
            }
            self.b1[k] -= lr * dh[k];
        }

        // Backprop through flatten (dp is gradient on conv_out directly)
        let mut d_conv = vec![0.0f32; n_pos * self.nf];
        for j in 0..self.pool_dim {
            if conv_out[j] > 0.0 { d_conv[j] = dp[j]; }
        }

        // Backprop conv
        for p in 0..n_pos {
            for f in 0..self.nf {
                let idx = p * self.nf + f;
                let dc = d_conv[idx];
                if dc == 0.0 { continue; }
                for ki in 0..self.k {
                    for d in 0..self.ch {
                        self.conv_w[f][ki * self.ch + d] -= lr * dc * input[(p + ki) * self.ch + d];
                    }
                }
                self.conv_b[f] -= lr * dc;
            }
        }
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split = corpus.len() * 80 / 100;

    println!("=== L1 CONV NEXT-CHAR PREDICTION ===\n");
    println!("  Corpus: {} chars ({} train, {} test)", corpus.len(), split, corpus.len() - split);
    println!("  Input: LUT_2N (2 int8/byte), Conv+Brain → predict next char\n");

    struct Cfg { ctx: usize, k: usize, nf: usize, hdim: usize, epochs: usize }
    let configs = vec![
        Cfg { ctx: 16,  k: 3, nf: 4,   hdim: 16,  epochs: 400 },
        Cfg { ctx: 16,  k: 3, nf: 8,   hdim: 32,  epochs: 400 },
        Cfg { ctx: 16,  k: 3, nf: 8,   hdim: 64,  epochs: 400 },
        Cfg { ctx: 16,  k: 3, nf: 16,  hdim: 32,  epochs: 400 },
        Cfg { ctx: 16,  k: 3, nf: 16,  hdim: 64,  epochs: 400 },
        Cfg { ctx: 16,  k: 3, nf: 16,  hdim: 128, epochs: 400 },
        Cfg { ctx: 16,  k: 3, nf: 32,  hdim: 64,  epochs: 400 },
        Cfg { ctx: 16,  k: 3, nf: 32,  hdim: 128, epochs: 300 },
    ];

    println!("  {:>6} {:>4} {:>4} {:>5} {:>8} {:>10} {:>10} {:>7}",
        "ctx", "k", "f", "h", "params", "train%", "test%", "time");
    println!("  {}", "-".repeat(65));

    for cfg in &configs {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let ch = 2; // LUT_2N channels
        let mut model = ConvBrain::new(cfg.ctx, ch, cfg.k, cfg.nf, cfg.hdim, &mut rng);

        let samples_per_ep = 15000.min(split / (cfg.ctx + 1));

        for ep in 0..cfg.epochs {
            let lr = 0.01 * (1.0 - ep as f32 / cfg.epochs as f32 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples_per_ep {
                let off = rt.range(0, split - cfg.ctx - 1);
                let input = encode_ctx(&corpus, off, cfg.ctx);
                let target = corpus[off + cfg.ctx];
                model.train_step(&input, target, lr);
            }
            if tc.elapsed().as_secs() > 120 { break; }
        }

        // Eval
        let eval = |start: usize, end: usize| -> f64 {
            let mut rng3 = Rng::new(999);
            let mut ok = 0usize; let mut tot = 0usize;
            let n_eval = 5000.min((end - start).saturating_sub(cfg.ctx + 1));
            for _ in 0..n_eval {
                if end < start + cfg.ctx + 1 { break; }
                let off = rng3.range(start, end - cfg.ctx - 1);
                let input = encode_ctx(&corpus, off, cfg.ctx);
                let (_, _, _, logits) = model.forward(&input);
                let pred = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
                if pred == corpus[off + cfg.ctx] as usize { ok += 1; }
                tot += 1;
            }
            if tot == 0 { 0.0 } else { ok as f64 / tot as f64 * 100.0 }
        };

        let tr = eval(0, split);
        let te = eval(split, corpus.len());
        let m = if te > 50.0 { " ***" } else if te > 40.0 { " **" } else if te > 30.0 { " *" } else { "" };

        println!("  {:>6} {:>4} {:>4} {:>5} {:>8} {:>9.1}% {:>9.1}% {:>6.1}s{}",
            cfg.ctx, cfg.k, cfg.nf, cfg.hdim, model.params(), tr, te, tc.elapsed().as_secs_f64(), m);
    }

    println!("\n  Baseline: random = {:.1}%", 100.0 / 27.0);
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
