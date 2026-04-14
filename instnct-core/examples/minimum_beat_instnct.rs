//! Minimum model to beat INSTNCT (24.6%) — how small can we go?
//!
//! Sweep: 1-hidden-layer MLP from H=1 to H=32
//! Input: frozen preprocessor (7 signals/char × ctx) OR raw one-hot (27/char × ctx)
//! Training: 200 epochs (enough to converge at small sizes)
//! Also test: binary {-1,+1} quantization of the minimum model
//!
//! Run: cargo run --example minimum_beat_instnct --release

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

struct MLP {
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
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
    fn param_count(&self) -> usize { self.idim * self.hdim + self.hdim + self.hdim * 27 + 27 }

    fn train_step(&mut self, input: &[f32], target: u8, lr: f32) {
        let mut h = vec![0.0f32; self.hdim];
        for k in 0..self.hdim { h[k] = self.b1[k]; for j in 0..self.idim { h[k] += self.w1[k][j] * input[j]; } h[k] = h[k].max(0.0); }
        let mut logits = vec![0.0f32; 27];
        for c in 0..27 { logits[c] = self.b2[c]; for k in 0..self.hdim { logits[c] += self.w2[c][k] * h[k]; } }
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = vec![0.0f32; 27];
        let mut sum = 0.0f32;
        for c in 0..27 { probs[c] = (logits[c] - max_l).exp(); sum += probs[c]; }
        for c in 0..27 { probs[c] /= sum; }
        let mut d = probs.clone(); d[target as usize] -= 1.0;
        let mut dh = vec![0.0f32; self.hdim];
        for c in 0..27 { for k in 0..self.hdim { dh[k] += d[c] * self.w2[c][k]; self.w2[c][k] -= lr * d[c] * h[k]; } self.b2[c] -= lr * d[c]; }
        for k in 0..self.hdim { if h[k] <= 0.0 { continue; } for j in 0..self.idim { self.w1[k][j] -= lr * dh[k] * input[j]; } self.b1[k] -= lr * dh[k]; }
    }

    fn eval_float(&self, corpus: &[u8], ctx: usize, encode: &dyn Fn(&[u8])->Vec<f32>, n: usize, seed: u64) -> f64 {
        let mut rng = Rng::new(seed); let mut ok = 0; let mut tot = 0;
        for _ in 0..n { if corpus.len() < ctx+1 { break; } let off = rng.range(0, corpus.len()-ctx-1);
            let input = encode(&corpus[off..off+ctx]);
            let mut logits = vec![0.0f32; 27];
            let mut h = vec![0.0f32; self.hdim];
            for k in 0..self.hdim { h[k] = self.b1[k]; for j in 0..self.idim { h[k] += self.w1[k][j] * input[j]; } h[k] = h[k].max(0.0); }
            for c in 0..27 { logits[c] = self.b2[c]; for k in 0..self.hdim { logits[c] += self.w2[c][k] * h[k]; } }
            let pred = logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred == corpus[off+ctx] as usize { ok += 1; } tot += 1;
        }
        ok as f64 / tot as f64 * 100.0
    }

    fn eval_int8(&self, corpus: &[u8], ctx: usize, encode: &dyn Fn(&[u8])->Vec<f32>, n: usize, seed: u64) -> f64 {
        let max1 = self.w1.iter().flat_map(|r| r.iter()).map(|x| x.abs()).fold(0.0f32, f32::max);
        let max2 = self.w2.iter().flat_map(|r| r.iter()).map(|x| x.abs()).fold(0.0f32, f32::max);
        let s1 = if max1 > 0.0 { 127.0 / max1 } else { 1.0 };
        let s2 = if max2 > 0.0 { 127.0 / max2 } else { 1.0 };
        let qw1: Vec<Vec<i8>> = self.w1.iter().map(|r| r.iter().map(|&x| (x*s1).round().max(-127.0).min(127.0) as i8).collect()).collect();
        let qw2: Vec<Vec<i8>> = self.w2.iter().map(|r| r.iter().map(|&x| (x*s2).round().max(-127.0).min(127.0) as i8).collect()).collect();
        let ds1 = 1.0/s1; let ds2 = 1.0/s2;
        let mut rng = Rng::new(seed); let mut ok = 0; let mut tot = 0;
        for _ in 0..n { if corpus.len() < ctx+1 { break; } let off = rng.range(0, corpus.len()-ctx-1);
            let input = encode(&corpus[off..off+ctx]);
            let mut h = vec![0.0f32; self.hdim];
            for k in 0..self.hdim { h[k] = self.b1[k]; for j in 0..self.idim { h[k] += qw1[k][j] as f32 * ds1 * input[j]; } h[k] = h[k].max(0.0); }
            let mut logits = vec![0.0f32; 27];
            for c in 0..27 { logits[c] = self.b2[c]; for k in 0..self.hdim { logits[c] += qw2[c][k] as f32 * ds2 * h[k]; } }
            let pred = logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred == corpus[off+ctx] as usize { ok += 1; } tot += 1;
        }
        ok as f64 / tot as f64 * 100.0
    }

    fn eval_binary(&self, corpus: &[u8], ctx: usize, encode: &dyn Fn(&[u8])->Vec<f32>, n: usize, seed: u64) -> f64 {
        let sign = |v: f32| -> f32 { if v >= 0.0 { 1.0 } else { -1.0 } };
        let qw1: Vec<Vec<f32>> = self.w1.iter().map(|r| r.iter().map(|&w| sign(w)).collect()).collect();
        let qw2: Vec<Vec<f32>> = self.w2.iter().map(|r| r.iter().map(|&w| sign(w)).collect()).collect();
        let mut rng = Rng::new(seed); let mut ok = 0; let mut tot = 0;
        for _ in 0..n { if corpus.len() < ctx+1 { break; } let off = rng.range(0, corpus.len()-ctx-1);
            let input = encode(&corpus[off..off+ctx]);
            let mut h = vec![0.0f32; self.hdim];
            for k in 0..self.hdim { h[k] = self.b1[k]; for j in 0..self.idim { h[k] += qw1[k][j] * input[j]; } h[k] = h[k].max(0.0); }
            let mut logits = vec![0.0f32; 27];
            for c in 0..27 { logits[c] = self.b2[c]; for k in 0..self.hdim { logits[c] += qw2[c][k] * h[k]; } }
            let pred = logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred == corpus[off+ctx] as usize { ok += 1; } tot += 1;
        }
        ok as f64 / tot as f64 * 100.0
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let pp = Preproc::new();
    let ctx = 16usize;

    println!("=== MINIMUM MODEL TO BEAT INSTNCT (24.6%) ===\n");
    println!("  Task: next-char prediction, Alice corpus ({} chars), ctx={}", corpus.len(), ctx);
    println!("  Target: beat INSTNCT 24.6% (evolution, ~9K int8 params)");
    println!("  Training: 200 epochs, 10K samples/epoch\n");

    // ── PART 1: With frozen preprocessor (7 signals/char) ──
    println!("━━━ PART 1: Preprocessor input ({}×7 = {} dim) ━━━\n", ctx, ctx*7);
    println!("  {:>3} {:>7} {:>8} {:>8} {:>8} {:>8}",
        "H", "params", "float%", "int8%", "binary%", "time");
    println!("  {}", "─".repeat(50));

    let idim_pp = ctx * 7;
    let mut found_pp = false;

    for &hdim in &[1, 2, 3, 4, 6, 8, 12, 16, 24, 32] {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut mlp = MLP::new(idim_pp, hdim, &mut rng);
        let params = mlp.param_count();

        for ep in 0..200 {
            let lr = 0.01 * (1.0 - ep as f32 / 200.0 * 0.7);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..10000.min(corpus.len() / (ctx + 1)) {
                let off = rt.range(0, corpus.len() - ctx - 1);
                mlp.train_step(&pp.encode_seq(&corpus[off..off+ctx]), corpus[off+ctx], lr);
            }
        }

        let fa = mlp.eval_float(&corpus, ctx, &|c| pp.encode_seq(c), 5000, 999);
        let ia = mlp.eval_int8(&corpus, ctx, &|c| pp.encode_seq(c), 5000, 999);
        let ba = mlp.eval_binary(&corpus, ctx, &|c| pp.encode_seq(c), 5000, 999);

        let mark = if fa > 24.6 && !found_pp { found_pp = true; " ← MINIMUM!" }
                   else if fa > 24.6 { " ★" } else { "" };

        println!("  {:>3} {:>7} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}s{}",
            hdim, params, fa, ia, ba, tc.elapsed().as_secs_f64(), mark);
    }

    // ── PART 2: With one-hot input (27/char) ──
    println!("\n━━━ PART 2: One-hot input ({}×27 = {} dim) ━━━\n", ctx, ctx*27);
    println!("  {:>3} {:>7} {:>8} {:>8} {:>8} {:>8}",
        "H", "params", "float%", "int8%", "binary%", "time");
    println!("  {}", "─".repeat(50));

    let idim_oh = ctx * 27;
    let mut found_oh = false;

    let encode_oh = |chars: &[u8]| -> Vec<f32> {
        let mut v = vec![0.0f32; idim_oh];
        for i in 0..ctx.min(chars.len()) { v[i*27 + chars[i] as usize] = 1.0; }
        v
    };

    for &hdim in &[1, 2, 3, 4, 6, 8, 12, 16] {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut mlp = MLP::new(idim_oh, hdim, &mut rng);
        let params = mlp.param_count();

        for ep in 0..200 {
            let lr = 0.01 * (1.0 - ep as f32 / 200.0 * 0.7);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..10000.min(corpus.len() / (ctx + 1)) {
                let off = rt.range(0, corpus.len() - ctx - 1);
                let mut input = vec![0.0f32; idim_oh];
                for i in 0..ctx { input[i*27 + corpus[off+i] as usize] = 1.0; }
                mlp.train_step(&input, corpus[off+ctx], lr);
            }
        }

        let fa = mlp.eval_float(&corpus, ctx, &encode_oh, 5000, 999);
        let ia = mlp.eval_int8(&corpus, ctx, &encode_oh, 5000, 999);
        let ba = mlp.eval_binary(&corpus, ctx, &encode_oh, 5000, 999);

        let mark = if fa > 24.6 && !found_oh { found_oh = true; " ← MINIMUM!" }
                   else if fa > 24.6 { " ★" } else { "" };

        println!("  {:>3} {:>7} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}s{}",
            hdim, params, fa, ia, ba, tc.elapsed().as_secs_f64(), mark);
    }

    // ── PART 3: No hidden layer — pure linear (logistic regression) ──
    println!("\n━━━ PART 3: No hidden layer (logistic regression) ━━━\n");
    {
        // Preprocessor input → linear → 27
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut w = vec![vec![0.0f32; idim_pp]; 27];
        let mut b = vec![0.0f32; 27];
        let s = (2.0/idim_pp as f32).sqrt();
        for c in 0..27 { for j in 0..idim_pp { w[c][j] = rng.normal() * s; } }

        for ep in 0..200 {
            let lr = 0.01 * (1.0 - ep as f32 / 200.0 * 0.7);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..10000.min(corpus.len() / (ctx + 1)) {
                let off = rt.range(0, corpus.len() - ctx - 1);
                let input = pp.encode_seq(&corpus[off..off+ctx]);
                let target = corpus[off+ctx] as usize;
                let mut logits = vec![0.0f32; 27];
                for c in 0..27 { logits[c] = b[c]; for j in 0..idim_pp { logits[c] += w[c][j] * input[j]; } }
                let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs = vec![0.0f32; 27]; let mut sum = 0.0f32;
                for c in 0..27 { probs[c] = (logits[c] - max_l).exp(); sum += probs[c]; }
                for c in 0..27 { probs[c] /= sum; }
                let mut d = probs; d[target] -= 1.0;
                for c in 0..27 { for j in 0..idim_pp { w[c][j] -= lr * d[c] * input[j]; } b[c] -= lr * d[c]; }
            }
        }

        // Eval
        let params = idim_pp * 27 + 27;
        let mut rng2 = Rng::new(999); let mut ok = 0; let mut tot = 0;
        for _ in 0..5000 { if corpus.len() < ctx+1 { break; } let off = rng2.range(0, corpus.len()-ctx-1);
            let input = pp.encode_seq(&corpus[off..off+ctx]);
            let mut logits = vec![0.0f32; 27];
            for c in 0..27 { logits[c] = b[c]; for j in 0..idim_pp { logits[c] += w[c][j] * input[j]; } }
            let pred = logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred == corpus[off+ctx] as usize { ok += 1; } tot += 1;
        }
        let acc = ok as f64 / tot as f64 * 100.0;
        println!("  Preprocessor → linear → 27: {:.1}% ({} params) {:.1}s",
            acc, params, tc.elapsed().as_secs_f64());
    }
    {
        // One-hot input → linear → 27
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut w = vec![vec![0.0f32; idim_oh]; 27];
        let mut b = vec![0.0f32; 27];
        let s = (2.0/idim_oh as f32).sqrt();
        for c in 0..27 { for j in 0..idim_oh { w[c][j] = rng.normal() * s; } }

        for ep in 0..200 {
            let lr = 0.01 * (1.0 - ep as f32 / 200.0 * 0.7);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..10000.min(corpus.len() / (ctx + 1)) {
                let off = rt.range(0, corpus.len() - ctx - 1);
                let mut input = vec![0.0f32; idim_oh];
                for i in 0..ctx { input[i*27 + corpus[off+i] as usize] = 1.0; }
                let target = corpus[off+ctx] as usize;
                let mut logits = vec![0.0f32; 27];
                for c in 0..27 { logits[c] = b[c]; for j in 0..idim_oh { logits[c] += w[c][j] * input[j]; } }
                let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs = vec![0.0f32; 27]; let mut sum = 0.0f32;
                for c in 0..27 { probs[c] = (logits[c] - max_l).exp(); sum += probs[c]; }
                for c in 0..27 { probs[c] /= sum; }
                let mut d = probs; d[target] -= 1.0;
                for c in 0..27 { for j in 0..idim_oh { w[c][j] -= lr * d[c] * input[j]; } b[c] -= lr * d[c]; }
            }
        }

        let params = idim_oh * 27 + 27;
        let mut rng2 = Rng::new(999); let mut ok = 0; let mut tot = 0;
        for _ in 0..5000 { if corpus.len() < ctx+1 { break; } let off = rng2.range(0, corpus.len()-ctx-1);
            let mut input = vec![0.0f32; idim_oh];
            for i in 0..ctx { input[i*27 + corpus[off+i] as usize] = 1.0; }
            let mut logits = vec![0.0f32; 27];
            for c in 0..27 { logits[c] = b[c]; for j in 0..idim_oh { logits[c] += w[c][j] * input[j]; } }
            let pred = logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred == corpus[off+ctx] as usize { ok += 1; } tot += 1;
        }
        let acc = ok as f64 / tot as f64 * 100.0;
        println!("  One-hot → linear → 27:      {:.1}% ({} params) {:.1}s",
            acc, params, tc.elapsed().as_secs_f64());
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
