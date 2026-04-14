//! Fair A/B test: MLP (backprop) vs INSTNCT (evolution) at equal param count
//!
//! INSTNCT baseline: H=256, ~9K int8 params, 24.6% next-char accuracy (evolution, 30K steps)
//! MLP: frozen preprocessor → 1-hidden-layer → 27 output (softmax cross-entropy, backprop)
//!
//! Fair comparison:
//!   - Same task: Alice corpus, next-char argmax prediction
//!   - Same evaluation: random 2000-token windows
//!   - Sweep MLP sizes: 5K, 7.5K, 10K, 15K params
//!   - Also test: with vs without frozen preprocessor
//!   - Also test: int8 quantization of trained MLP
//!
//! Run: cargo run --example fair_ab_mlp_vs_instnct --release

use std::time::Instant;

struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn normal(&mut self) -> f32 { let u1 = self.f32().max(1e-7); let u2 = self.f32(); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() }
    fn range(&mut self, lo: usize, hi: usize) -> usize { if hi <= lo { lo } else { lo + (self.next() as usize % (hi - lo)) } }
}

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = std::fs::read(path).expect("read");
    let mut c = Vec::new();
    for &b in &raw { match b { b'a'..=b'z' => c.push(b-b'a'), b'A'..=b'Z' => c.push(b-b'A'), b' '|b'\n'|b'\t'|b'\r' => c.push(26), _ => {} } }
    c
}

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0*c;
    if x >= l { return x-l; } if x <= -l { return x+l; }
    let s = x/c; let n = s.floor(); let t = s-n; let h = t*(1.0-t);
    let sg = if (n as i32)%2==0 { 1.0 } else { -1.0 }; c*(sg*h+rho*h*h)
}

struct Preproc { w: [[i8;8];7], b: [i8;7], c: [f32;7], rho: [f32;7] }
impl Preproc {
    fn new() -> Self { Preproc {
        w: [[-1,1,1,-1,-1,1,1,-1],[1,-1,1,1,-1,-1,-1,-1],[-1,-1,1,-1,1,1,-1,-1],
            [-1,-1,-1,1,-1,-1,1,-1],[-1,1,1,1,-1,-1,-1,-1],[1,1,-1,-1,-1,-1,-1,-1],
            [-1,1,-1,1,1,1,-1,-1]],
        b: [1,1,1,1,1,1,1], c: [10.0;7], rho: [2.0,0.0,0.0,0.0,0.0,0.0,0.0],
    }}
    fn encode(&self, ch: u8) -> [f32;7] {
        let mut bits=[0.0f32;8]; for i in 0..8 { bits[i]=((ch>>i)&1) as f32; }
        let mut o=[0.0f32;7];
        for k in 0..7 { let mut d=self.b[k] as f32; for j in 0..8 { d+=self.w[k][j] as f32*bits[j]; } o[k]=c19(d,self.c[k],self.rho[k]); }
        o
    }
    fn encode_seq(&self, chars: &[u8]) -> Vec<f32> {
        chars.iter().flat_map(|&ch| self.encode(ch).to_vec()).collect()
    }
}

// ══════════════════════════════════════════════════════
// MLP with softmax cross-entropy — next char prediction
// ══════════════════════════════════════════════════════
struct MLP {
    w1: Vec<Vec<f32>>, b1: Vec<f32>,  // input → hidden
    w2: Vec<Vec<f32>>, b2: Vec<f32>,  // hidden → 27
    idim: usize, hdim: usize,
}

impl MLP {
    fn new(idim: usize, hdim: usize, rng: &mut Rng) -> Self {
        let s1 = (2.0/idim as f32).sqrt();
        let s2 = (2.0/hdim as f32).sqrt();
        MLP {
            w1: (0..hdim).map(|_| (0..idim).map(|_| rng.normal()*s1).collect()).collect(),
            b1: vec![0.0;hdim],
            w2: (0..27).map(|_| (0..hdim).map(|_| rng.normal()*s2).collect()).collect(),
            b2: vec![0.0;27],
            idim, hdim,
        }
    }

    fn param_count(&self) -> usize {
        self.idim * self.hdim + self.hdim + self.hdim * 27 + 27
    }

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // Hidden layer (ReLU)
        let mut h = vec![0.0f32; self.hdim];
        for k in 0..self.hdim {
            h[k] = self.b1[k];
            for j in 0..self.idim { h[k] += self.w1[k][j] * input[j]; }
            h[k] = h[k].max(0.0); // ReLU
        }
        // Output (softmax)
        let mut logits = vec![0.0f32; 27];
        for c in 0..27 {
            logits[c] = self.b2[c];
            for k in 0..self.hdim { logits[c] += self.w2[c][k] * h[k]; }
        }
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = vec![0.0f32; 27];
        let mut sum = 0.0f32;
        for c in 0..27 { probs[c] = (logits[c] - max_l).exp(); sum += probs[c]; }
        for c in 0..27 { probs[c] /= sum; }
        (h, probs)
    }

    fn train_step(&mut self, input: &[f32], target: u8, lr: f32) {
        let (h, probs) = self.forward(input);
        // Cross-entropy gradient: d_logits[c] = probs[c] - (c == target)
        let mut d_logits = probs.clone();
        d_logits[target as usize] -= 1.0;
        // Backprop to w2, b2
        let mut d_h = vec![0.0f32; self.hdim];
        for c in 0..27 {
            for k in 0..self.hdim {
                d_h[k] += d_logits[c] * self.w2[c][k];
                self.w2[c][k] -= lr * d_logits[c] * h[k];
            }
            self.b2[c] -= lr * d_logits[c];
        }
        // Backprop through ReLU to w1, b1
        for k in 0..self.hdim {
            if h[k] <= 0.0 { continue; } // ReLU gate
            let dh = d_h[k];
            for j in 0..self.idim { self.w1[k][j] -= lr * dh * input[j]; }
            self.b1[k] -= lr * dh;
        }
    }

    fn eval(&self, corpus: &[u8], ctx: usize, encode_fn: &dyn Fn(&[u8]) -> Vec<f32>,
            n_samples: usize, seed: u64) -> f64 {
        let mut rng = Rng::new(seed);
        let mut ok = 0usize; let mut tot = 0usize;
        for _ in 0..n_samples {
            if corpus.len() < ctx + 1 { break; }
            let off = rng.range(0, corpus.len() - ctx - 1);
            let input = encode_fn(&corpus[off..off+ctx]);
            let target = corpus[off + ctx];
            let (_, probs) = self.forward(&input);
            let pred = probs.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred == target as usize { ok += 1; }
            tot += 1;
        }
        ok as f64 / tot as f64 * 100.0
    }

    fn quantize_int8(&self) -> MLP_Int8 {
        // Find scale for each weight matrix
        let max1 = self.w1.iter().flat_map(|r| r.iter()).map(|x| x.abs()).fold(0.0f32, f32::max);
        let max2 = self.w2.iter().flat_map(|r| r.iter()).map(|x| x.abs()).fold(0.0f32, f32::max);
        let s1 = if max1 > 0.0 { 127.0 / max1 } else { 1.0 };
        let s2 = if max2 > 0.0 { 127.0 / max2 } else { 1.0 };

        MLP_Int8 {
            w1: self.w1.iter().map(|r| r.iter().map(|&x| (x * s1).round().max(-127.0).min(127.0) as i8).collect()).collect(),
            b1: self.b1.clone(),
            w2: self.w2.iter().map(|r| r.iter().map(|&x| (x * s2).round().max(-127.0).min(127.0) as i8).collect()).collect(),
            b2: self.b2.clone(),
            s1: 1.0/s1, s2: 1.0/s2,
            idim: self.idim, hdim: self.hdim,
        }
    }
}

#[allow(non_camel_case_types)]
struct MLP_Int8 {
    w1: Vec<Vec<i8>>, b1: Vec<f32>,
    w2: Vec<Vec<i8>>, b2: Vec<f32>,
    s1: f32, s2: f32,
    idim: usize, hdim: usize,
}

impl MLP_Int8 {
    fn eval(&self, corpus: &[u8], ctx: usize, encode_fn: &dyn Fn(&[u8]) -> Vec<f32>,
            n_samples: usize, seed: u64) -> f64 {
        let mut rng = Rng::new(seed);
        let mut ok = 0usize; let mut tot = 0usize;
        for _ in 0..n_samples {
            if corpus.len() < ctx + 1 { break; }
            let off = rng.range(0, corpus.len() - ctx - 1);
            let input = encode_fn(&corpus[off..off+ctx]);
            let target = corpus[off + ctx];
            // Int8 forward pass
            let mut h = vec![0.0f32; self.hdim];
            for k in 0..self.hdim {
                h[k] = self.b1[k];
                for j in 0..self.idim { h[k] += self.w1[k][j] as f32 * self.s1 * input[j]; }
                h[k] = h[k].max(0.0);
            }
            let mut logits = vec![0.0f32; 27];
            for c in 0..27 {
                logits[c] = self.b2[c];
                for k in 0..self.hdim { logits[c] += self.w2[c][k] as f32 * self.s2 * h[k]; }
            }
            let pred = logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred == target as usize { ok += 1; }
            tot += 1;
        }
        ok as f64 / tot as f64 * 100.0
    }
}

// Simple frequency baseline
fn frequency_baseline(corpus: &[u8]) -> (f64, u8) {
    let mut freq = [0usize; 27];
    for &c in corpus { if (c as usize) < 27 { freq[c as usize] += 1; } }
    let best = freq.iter().enumerate().max_by_key(|(_,&f)| f).unwrap().0 as u8;
    let acc = freq[best as usize] as f64 / corpus.len() as f64 * 100.0;
    (acc, best)
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let pp = Preproc::new();
    let ctx = 16usize;

    let (freq_acc, freq_ch) = frequency_baseline(&corpus);
    let freq_name = if freq_ch == 26 { "space".to_string() } else { ((freq_ch + b'a') as char).to_string() };

    println!("=== FAIR A/B: MLP (backprop) vs INSTNCT (evolution) ===\n");
    println!("  Task: next-char prediction, Alice corpus ({} chars)", corpus.len());
    println!("  Context: {} chars (frozen preprocessor → {} signals)", ctx, ctx * 7);
    println!("  Baselines:");
    println!("    Random:    {:.1}% (1/27)", 100.0/27.0);
    println!("    Frequency: {:.1}% (always '{}')", freq_acc, freq_name);
    println!("    INSTNCT:   24.6% (H=256, evolution, 30K steps, ~9K int8 params)");
    println!();

    // ── PART 1: MLP with frozen preprocessor ──
    println!("━━━ PART 1: MLP + frozen preprocessor (ctx={}) ━━━\n", ctx);
    println!("  {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "hidden", "params", "float%", "int8%", "Δint8", "time");
    println!("  {}", "─".repeat(52));

    let idim = ctx * 7; // 112

    // Sweep: hidden sizes chosen to bracket INSTNCT param count
    // Params = idim*H + H + H*27 + 27 = H*(idim+27) + H + 27 = H*(idim+28) + 27
    // INSTNCT ≈ 9K params
    for &hdim in &[16, 24, 36, 54, 72, 108] {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut mlp = MLP::new(idim, hdim, &mut rng);
        let params = mlp.param_count();

        // Train: 100 epochs, 10K samples/epoch
        let samples_per_ep = 10000.min(corpus.len() / (ctx + 1));
        for ep in 0..100 {
            let lr = 0.01 * (1.0 - ep as f32 / 100.0 * 0.7);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples_per_ep {
                let off = rt.range(0, corpus.len() - ctx - 1);
                let input = pp.encode_seq(&corpus[off..off+ctx]);
                let target = corpus[off + ctx];
                mlp.train_step(&input, target, lr);
            }
        }

        // Eval float
        let float_acc = mlp.eval(&corpus, ctx, &|chars| pp.encode_seq(chars), 5000, 999);
        // Eval int8
        let q = mlp.quantize_int8();
        let int8_acc = q.eval(&corpus, ctx, &|chars| pp.encode_seq(chars), 5000, 999);
        let delta = int8_acc - float_acc;

        let marker = if float_acc > 30.0 { " ★" } else { "" };
        let marker2 = if float_acc > 40.0 { "★" } else { "" };

        println!("  {:>6} {:>8} {:>7.1}% {:>7.1}% {:>+7.1}% {:>7.1}s{}{}",
            hdim, params, float_acc, int8_acc, delta, tc.elapsed().as_secs_f64(), marker, marker2);
    }

    // ── PART 2: Raw input (no preprocessor) for comparison ──
    println!("\n━━━ PART 2: MLP + raw one-hot input (NO preprocessor) ━━━\n");
    println!("  {:>6} {:>8} {:>8} {:>8}",
        "hidden", "params", "float%", "time");
    println!("  {}", "─".repeat(35));

    let raw_idim = ctx * 27; // one-hot: 432

    for &hdim in &[16, 36, 54] {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut mlp = MLP::new(raw_idim, hdim, &mut rng);
        let params = mlp.param_count();

        let samples_per_ep = 10000.min(corpus.len() / (ctx + 1));
        for ep in 0..100 {
            let lr = 0.01 * (1.0 - ep as f32 / 100.0 * 0.7);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples_per_ep {
                let off = rt.range(0, corpus.len() - ctx - 1);
                // One-hot encode
                let mut input = vec![0.0f32; raw_idim];
                for i in 0..ctx { input[i * 27 + corpus[off + i] as usize] = 1.0; }
                let target = corpus[off + ctx];
                mlp.train_step(&input, target, lr);
            }
        }

        let float_acc = mlp.eval(&corpus, ctx,
            &|chars| { let mut v = vec![0.0f32; raw_idim]; for i in 0..ctx { v[i*27+chars[i] as usize]=1.0; } v },
            5000, 999);

        println!("  {:>6} {:>8} {:>7.1}% {:>7.1}s",
            hdim, params, float_acc, tc.elapsed().as_secs_f64());
    }

    // ── PART 3: Summary ──
    println!("\n━━━ SUMMARY ━━━\n");
    println!("  INSTNCT:  24.6%  (~9K int8 params, evolution, 30K steps)");
    println!("  Compare MLP float/int8 accuracy at similar param counts above.");
    println!("  Preprocessor value: compare Part 1 vs Part 2 at same hidden size.");
    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
