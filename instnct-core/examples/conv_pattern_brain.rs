//! 3-stage hierarchical: byte encoder → conv pattern finder → brain
//!
//! Stage 1: Fixed binary preprocessor (byte → 7 signals, POPCOUNT)
//! Stage 2: Conv1D pattern finder (sliding window over encoded bytes)
//!          Shared kernel, extracts local patterns (bigrams, trigrams)
//! Stage 3: Brain (MLP on conv features → 27 prediction)
//!
//! Results (ctx=16, 200ep):
//!   k=3 f=64  h=512: 96.6% ★★★ NEW RECORD (475K params) — BEATS one-hot 93.8%!
//!   k=3 f=128 h=256: 95.3% ★★★
//!   k=2 f=64  h=256: 89.0% ★★
//!   k=2 f=32  h=256: 77.3%
//!   Without conv:     56.3% → conv adds +33pp!
//!
//! KEY: Conv pattern finder + preprocessor BEATS raw one-hot (96.6 > 93.8%)!
//!
//! Run: cargo run --example conv_pattern_brain --release

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

// Stage 1: Fixed binary preprocessor
struct ByteEncoder;
impl ByteEncoder {
    const W: [[i8;8];7] = [[-1,1,1,-1,-1,1,1,-1],[1,-1,1,1,-1,-1,-1,-1],[-1,-1,1,-1,1,1,-1,-1],
        [-1,-1,-1,1,-1,-1,1,-1],[-1,1,1,1,-1,-1,-1,-1],[1,1,-1,-1,-1,-1,-1,-1],
        [-1,1,-1,1,1,1,-1,-1]];
    const B: [i8;7] = [1,1,1,1,1,1,1];
    const C: [f32;7] = [10.0;7];
    const RHO: [f32;7] = [2.0,0.0,0.0,0.0,0.0,0.0,0.0];

    fn encode(ch: u8) -> [f32;7] {
        let mut bits=[0.0f32;8]; for i in 0..8 { bits[i]=((ch>>i)&1) as f32; }
        let mut o=[0.0f32;7];
        for k in 0..7 { let mut d=Self::B[k] as f32; for j in 0..8 { d+=Self::W[k][j] as f32*bits[j]; } o[k]=c19(d,Self::C[k],Self::RHO[k]); }
        o
    }
}

// Stage 2: Conv1D pattern finder
// kernel_size chars → n_filters features, shared across positions
struct ConvPatternFinder {
    // Conv weights: n_filters × (kernel_size × 7)
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
    kernel_size: usize,
    n_filters: usize,
}

impl ConvPatternFinder {
    fn new(kernel_size: usize, n_filters: usize, rng: &mut Rng) -> Self {
        let fan_in = kernel_size * 7;
        let s = (2.0 / fan_in as f32).sqrt();
        ConvPatternFinder {
            w: (0..n_filters).map(|_| (0..fan_in).map(|_| rng.normal()*s).collect()).collect(),
            b: vec![0.0; n_filters],
            kernel_size, n_filters,
        }
    }

    fn params(&self) -> usize { self.n_filters * self.kernel_size * 7 + self.n_filters }

    // Apply conv over encoded sequence, return (n_positions × n_filters) flattened
    fn forward(&self, encoded: &[[f32;7]], ctx: usize) -> Vec<f32> {
        let n_pos = ctx - self.kernel_size + 1;
        let mut out = vec![0.0f32; n_pos * self.n_filters];

        for pos in 0..n_pos {
            for f in 0..self.n_filters {
                let mut val = self.b[f];
                for ki in 0..self.kernel_size {
                    for d in 0..7 {
                        val += self.w[f][ki * 7 + d] * encoded[pos + ki][d];
                    }
                }
                out[pos * self.n_filters + f] = val.max(0.0); // ReLU
            }
        }
        out
    }
}

// Full 3-stage pipeline
struct HierPipeline {
    conv: ConvPatternFinder,
    // Brain: 2 layers (conv output is already features)
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    brain_idim: usize, h1: usize,
    ctx: usize,
}

impl HierPipeline {
    fn new(ctx: usize, kernel_size: usize, n_filters: usize, h1: usize, rng: &mut Rng) -> Self {
        let n_pos = ctx - kernel_size + 1;
        let brain_idim = n_pos * n_filters;
        let s1 = (2.0/brain_idim as f32).sqrt();
        let s2 = (2.0/h1 as f32).sqrt();
        HierPipeline {
            conv: ConvPatternFinder::new(kernel_size, n_filters, rng),
            w1: (0..h1).map(|_| (0..brain_idim).map(|_| rng.normal()*s1).collect()).collect(),
            b1: vec![0.0;h1],
            w2: (0..27).map(|_| (0..h1).map(|_| rng.normal()*s2).collect()).collect(),
            b2: vec![0.0;27],
            brain_idim, h1, ctx,
        }
    }

    fn total_params(&self) -> usize {
        self.conv.params() + self.brain_idim*self.h1+self.h1 + self.h1*27+27
    }

    fn forward_all(&self, chars: &[u8]) -> (Vec<[f32;7]>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // Stage 1: encode each byte
        let encoded: Vec<[f32;7]> = chars.iter().map(|&ch| ByteEncoder::encode(ch)).collect();
        // Stage 2: conv pattern finder
        let conv_out = self.conv.forward(&encoded, self.ctx);
        // Stage 3: brain
        let mut a1 = vec![0.0f32;self.h1];
        for k in 0..self.h1 { a1[k]=self.b1[k]; for j in 0..self.brain_idim { a1[k]+=self.w1[k][j]*conv_out[j]; } a1[k]=a1[k].max(0.0); }
        let mut logits = vec![0.0f32;27];
        for c in 0..27 { logits[c]=self.b2[c]; for k in 0..self.h1 { logits[c]+=self.w2[c][k]*a1[k]; } }
        (encoded, conv_out, a1, logits)
    }

    fn train_step(&mut self, chars: &[u8], target: u8, lr: f32) {
        let (encoded, conv_out, a1, logits) = self.forward_all(chars);

        // Softmax
        let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
        let mut p=vec![0.0f32;27]; let mut s=0.0f32;
        for c in 0..27 { p[c]=(logits[c]-mx).exp(); s+=p[c]; } for c in 0..27 { p[c]/=s; }
        let mut dl=p; dl[target as usize]-=1.0;

        // Backprop brain output layer
        let mut da1=vec![0.0f32;self.h1];
        for c in 0..27 { for k in 0..self.h1 { da1[k]+=dl[c]*self.w2[c][k]; self.w2[c][k]-=lr*dl[c]*a1[k]; } self.b2[c]-=lr*dl[c]; }

        // Backprop brain hidden
        let mut d_conv=vec![0.0f32;self.brain_idim];
        for k in 0..self.h1 { if a1[k]<=0.0{continue;} for j in 0..self.brain_idim { d_conv[j]+=da1[k]*self.w1[k][j]; self.w1[k][j]-=lr*da1[k]*conv_out[j]; } self.b1[k]-=lr*da1[k]; }

        // Backprop through conv (Stage 2)
        let n_pos = self.ctx - self.conv.kernel_size + 1;
        for pos in 0..n_pos {
            for f in 0..self.conv.n_filters {
                let idx = pos * self.conv.n_filters + f;
                if conv_out[idx] <= 0.0 { continue; } // ReLU gate
                let d = d_conv[idx];
                for ki in 0..self.conv.kernel_size {
                    for dim in 0..7 {
                        self.conv.w[f][ki*7+dim] -= lr * d * encoded[pos+ki][dim];
                    }
                }
                self.conv.b[f] -= lr * d;
            }
        }
    }

    fn eval(&self, corpus: &[u8], n: usize, seed: u64) -> f64 {
        let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
        for _ in 0..n { if corpus.len()<self.ctx+1{break;} let off=rng.range(0,corpus.len()-self.ctx-1);
            let (_, _, _, logits) = self.forward_all(&corpus[off..off+self.ctx]);
            let pred=logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred==corpus[off+self.ctx] as usize {ok+=1;} tot+=1;
        }
        ok as f64/tot as f64*100.0
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let ctx = 16usize;

    println!("=== 3-STAGE HIERARCHICAL: byte encoder → conv pattern → brain ===\n");
    println!("  Stage 1: Fixed binary preprocessor (byte→7, POPCOUNT)");
    println!("  Stage 2: Conv1D pattern finder (shared kernel, ReLU)");
    println!("  Stage 3: Brain MLP (features→27)\n");
    println!("  Corpus: {} chars, ctx={}", corpus.len(), ctx);
    println!("  Baselines: fix_pp=70.1%, learned_emb=81.2%, one-hot=93.8%\n");

    struct Cfg { name: &'static str, kernel: usize, filters: usize, h1: usize }
    let configs = vec![
        // Sweep kernel sizes (how many bytes does Stage 2 see?)
        Cfg { name: "k=2 f=32  h=256", kernel: 2, filters: 32, h1: 256 },
        Cfg { name: "k=3 f=32  h=256", kernel: 3, filters: 32, h1: 256 },
        Cfg { name: "k=4 f=32  h=256", kernel: 4, filters: 32, h1: 256 },
        Cfg { name: "k=2 f=64  h=256", kernel: 2, filters: 64, h1: 256 },
        Cfg { name: "k=3 f=64  h=256", kernel: 3, filters: 64, h1: 256 },
        // Bigger brain
        Cfg { name: "k=3 f=64  h=512", kernel: 3, filters: 64, h1: 512 },
        // Wider conv
        Cfg { name: "k=3 f=128 h=256", kernel: 3, filters: 128, h1: 256 },
        // Wide window
        Cfg { name: "k=5 f=64  h=256", kernel: 5, filters: 64, h1: 256 },
    ];

    println!("  {:>20} {:>6} {:>8} {:>8} {:>8}",
        "config", "conv_p", "total_p", "acc%", "time");
    println!("  {}", "─".repeat(55));

    for cfg in &configs {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut pipe = HierPipeline::new(ctx, cfg.kernel, cfg.filters, cfg.h1, &mut rng);
        let conv_params = pipe.conv.params();

        let samples = 15000.min(corpus.len() / (ctx+1));
        for ep in 0..200 {
            let lr = 0.01 * (1.0 - ep as f32 / 200.0 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples {
                let off = rt.range(0, corpus.len()-ctx-1);
                pipe.train_step(&corpus[off..off+ctx], corpus[off+ctx], lr);
            }
        }

        let acc = pipe.eval(&corpus, 5000, 999);
        let m = if acc>90.0{" ★★★"} else if acc>80.0{" ★★"} else if acc>60.0{" ★"} else {""};

        println!("  {:>20} {:>6} {:>8} {:>7.1}% {:>7.1}s{}",
            cfg.name, conv_params, pipe.total_params(), acc, tc.elapsed().as_secs_f64(), m);
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
