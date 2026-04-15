//! L1 Feature Extractor — masked char prediction (BERT-style)
//!
//! Mask 1 byte in context, predict it from surrounding bytes.
//! Bidirectional: sees both left and right context.
//! Goal: 100% = features perfectly capture context patterns → freeze.
//!
//! Run: cargo run --example l1_masked_char --release

use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];
const MASK_VAL: [f32; 2] = [0.0, 0.0]; // masked position gets zeros

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

fn encode_masked(corpus: &[u8], start: usize, ctx: usize, mask_pos: usize) -> Vec<f32> {
    (0..ctx).flat_map(|i| {
        if i == mask_pos {
            MASK_VAL.to_vec()
        } else {
            let ch = corpus[start + i] as usize;
            vec![LUT[ch][0] as f32, LUT[ch][1] as f32]
        }
    }).collect()
}

struct ConvPredictor {
    // Conv layer
    conv_w: Vec<Vec<f32>>, conv_b: Vec<f32>,
    // Head: flatten conv around mask → hdim → 27
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    k: usize, nf: usize, ch: usize, ctx: usize, hdim: usize,
    head_dim: usize,
}

impl ConvPredictor {
    fn new(ctx: usize, ch: usize, k: usize, nf: usize, hdim: usize, rng: &mut Rng) -> Self {
        let fan_in = k * ch;
        // Take conv outputs from positions around the mask (window of conv positions)
        // For mask at center: take all conv positions (for small ctx)
        let n_conv = ctx - k + 1;
        let head_dim = n_conv * nf;
        let sc = (2.0 / fan_in as f32).sqrt();
        let s1 = (2.0 / head_dim as f32).sqrt();
        let s2 = (2.0 / hdim as f32).sqrt();
        ConvPredictor {
            conv_w: (0..nf).map(|_| (0..fan_in).map(|_| rng.normal() * sc).collect()).collect(),
            conv_b: vec![0.0; nf],
            w1: (0..hdim).map(|_| (0..head_dim).map(|_| rng.normal() * s1).collect()).collect(),
            b1: vec![0.0; hdim],
            w2: (0..27).map(|_| (0..hdim).map(|_| rng.normal() * s2).collect()).collect(),
            b2: vec![0.0; 27],
            k, nf, ch, ctx, hdim, head_dim,
        }
    }

    fn params(&self) -> usize {
        self.nf * self.k * self.ch + self.nf + self.head_dim * self.hdim + self.hdim + self.hdim * 27 + 27
    }

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n_conv = self.ctx - self.k + 1;
        // Conv + ReLU
        let mut co = vec![0.0f32; n_conv * self.nf];
        for p in 0..n_conv {
            for f in 0..self.nf {
                let mut v = self.conv_b[f];
                for ki in 0..self.k {
                    for d in 0..self.ch {
                        v += self.conv_w[f][ki * self.ch + d] * input[(p + ki) * self.ch + d];
                    }
                }
                co[p * self.nf + f] = v.max(0.0);
            }
        }

        // Dense 1: flatten → hdim (ReLU)
        let mut h = vec![0.0f32; self.hdim];
        for i in 0..self.hdim {
            h[i] = self.b1[i];
            for j in 0..self.head_dim { h[i] += self.w1[i][j] * co[j]; }
            h[i] = h[i].max(0.0);
        }

        // Dense 2: hdim → 27
        let mut logits = vec![0.0f32; 27];
        for c in 0..27 {
            logits[c] = self.b2[c];
            for i in 0..self.hdim { logits[c] += self.w2[c][i] * h[i]; }
        }

        (co, h, logits)
    }

    fn train_step(&mut self, input: &[f32], target: u8, lr: f32) {
        let n_conv = self.ctx - self.k + 1;
        let (co, h, logits) = self.forward(input);

        // Softmax + CE
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut p = vec![0.0f32; 27]; let mut s = 0.0f32;
        for c in 0..27 { p[c] = (logits[c] - mx).exp(); s += p[c]; }
        for c in 0..27 { p[c] /= s; }
        let mut dl = p; dl[target as usize] -= 1.0;

        // Backprop dense2
        let mut dh = vec![0.0f32; self.hdim];
        for c in 0..27 {
            for i in 0..self.hdim { dh[i] += dl[c] * self.w2[c][i]; self.w2[c][i] -= lr * dl[c] * h[i]; }
            self.b2[c] -= lr * dl[c];
        }

        // Backprop dense1 (ReLU)
        let mut dco = vec![0.0f32; self.head_dim];
        for i in 0..self.hdim {
            if h[i] <= 0.0 { continue; }
            for j in 0..self.head_dim { dco[j] += dh[i] * self.w1[i][j]; self.w1[i][j] -= lr * dh[i] * co[j]; }
            self.b1[i] -= lr * dh[i];
        }

        // Backprop conv
        for p in 0..n_conv {
            for f in 0..self.nf {
                let idx = p * self.nf + f;
                if co[idx] <= 0.0 { continue; }
                let dc = dco[idx];
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

    println!("=== L1 MASKED CHAR PREDICTION ===\n");
    println!("  Corpus: {} chars ({} train, {} test)", corpus.len(), split, corpus.len() - split);
    println!("  Task: mask 1 byte, predict from bidirectional context");
    println!("  Goal: 100% = perfect feature extraction\n");

    struct Cfg { ctx: usize, k: usize, nf: usize, hdim: usize, epochs: usize }
    let configs = vec![
        Cfg { ctx: 8,   k: 3, nf: 16,  hdim: 64,  epochs: 400 },
        Cfg { ctx: 16,  k: 3, nf: 16,  hdim: 64,  epochs: 400 },
        Cfg { ctx: 16,  k: 3, nf: 32,  hdim: 64,  epochs: 400 },
        Cfg { ctx: 16,  k: 3, nf: 32,  hdim: 128, epochs: 300 },
        Cfg { ctx: 16,  k: 5, nf: 32,  hdim: 128, epochs: 300 },
        Cfg { ctx: 32,  k: 3, nf: 16,  hdim: 64,  epochs: 300 },
        Cfg { ctx: 32,  k: 5, nf: 16,  hdim: 64,  epochs: 300 },
    ];

    println!("  {:>5} {:>3} {:>4} {:>5} {:>8} {:>10} {:>10} {:>7}",
        "ctx", "k", "f", "h", "params", "train%", "test%", "time");
    println!("  {}", "-".repeat(62));

    for cfg in &configs {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let ch = 2;
        let mut model = ConvPredictor::new(cfg.ctx, ch, cfg.k, cfg.nf, cfg.hdim, &mut rng);

        let samples_per_ep = 15000.min(split / cfg.ctx);

        for ep in 0..cfg.epochs {
            let lr = 0.01 * (1.0 - ep as f32 / cfg.epochs as f32 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples_per_ep {
                let off = rt.range(0, split - cfg.ctx);
                // Random mask position (not edges for better context)
                let mask_pos = rt.range(1, cfg.ctx - 1);
                let input = encode_masked(&corpus, off, cfg.ctx, mask_pos);
                let target = corpus[off + mask_pos];
                model.train_step(&input, target, lr);
            }
            if tc.elapsed().as_secs() > 120 { break; }
        }

        // Eval
        let eval = |start: usize, end: usize| -> f64 {
            let mut rng3 = Rng::new(999);
            let mut ok = 0usize; let mut tot = 0usize;
            let n = 5000.min((end - start).saturating_sub(cfg.ctx));
            for _ in 0..n {
                if end < start + cfg.ctx { break; }
                let off = rng3.range(start, end - cfg.ctx);
                let mask_pos = rng3.range(1, cfg.ctx - 1);
                let input = encode_masked(&corpus, off, cfg.ctx, mask_pos);
                let (_, _, logits) = model.forward(&input);
                let pred = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
                if pred == corpus[off + mask_pos] as usize { ok += 1; }
                tot += 1;
            }
            if tot == 0 { 0.0 } else { ok as f64 / tot as f64 * 100.0 }
        };

        let tr = eval(0, split);
        let te = eval(split, corpus.len());
        let m = if te > 90.0 { " ***" } else if te > 70.0 { " **" } else if te > 50.0 { " *" } else { "" };

        println!("  {:>5} {:>3} {:>4} {:>5} {:>8} {:>9.1}% {:>9.1}% {:>6.1}s{}",
            cfg.ctx, cfg.k, cfg.nf, cfg.hdim, model.params(), tr, te, tc.elapsed().as_secs_f64(), m);
    }

    println!("\n  Baseline: random = {:.1}%", 100.0 / 27.0);
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
