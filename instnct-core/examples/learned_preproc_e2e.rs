//! End-to-end learned preprocessor — backprop through the encoding layer
//!
//! Instead of fixed binary preprocessor, learn the byte→7 encoding
//! jointly with the brain via prediction loss (cross-entropy).
//!
//! Tests whether a LEARNED 7-signal encoding can close the gap
//! between fixed preprocessor (70.1%) and raw one-hot (93.8%).
//!
//! Architecture:
//!   byte → learned_preproc(byte→7, per-char, shared) → ctx×7 signals
//!   → brain (3L ReLU) → 27 output
//!
//! The learned preproc sees each byte independently (no context),
//! but its encoding is optimized for the prediction task.
//!
//! Run: cargo run --example learned_preproc_e2e --release

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

// Fixed binary preprocessor (baseline)
struct FixedPreproc { w: [[i8;8];7], b: [i8;7], c: [f32;7], rho: [f32;7] }
impl FixedPreproc {
    fn new() -> Self { FixedPreproc {
        w: [[-1,1,1,-1,-1,1,1,-1],[1,-1,1,1,-1,-1,-1,-1],[-1,-1,1,-1,1,1,-1,-1],
            [-1,-1,-1,1,-1,-1,1,-1],[-1,1,1,1,-1,-1,-1,-1],[1,1,-1,-1,-1,-1,-1,-1],
            [-1,1,-1,1,1,1,-1,-1]],
        b: [1,1,1,1,1,1,1], c: [10.0;7], rho: [2.0,0.0,0.0,0.0,0.0,0.0,0.0],
    }}
    fn encode_seq(&self, chars: &[u8]) -> Vec<f32> {
        chars.iter().flat_map(|&ch| {
            let mut bits=[0.0f32;8]; for i in 0..8 { bits[i]=((ch>>i)&1) as f32; }
            let mut o=[0.0f32;7];
            for k in 0..7 { let mut d=self.b[k] as f32; for j in 0..8 { d+=self.w[k][j] as f32*bits[j]; } o[k]=c19(d,self.c[k],self.rho[k]); }
            o.to_vec()
        }).collect()
    }
}

// Learned preprocessor: 27 → code_dim (lookup table, differentiable)
// Each of 27 chars gets a learned code_dim-dimensional embedding
struct LearnedPreproc {
    embeddings: Vec<Vec<f32>>, // 27 × code_dim
    code_dim: usize,
}

impl LearnedPreproc {
    fn new(code_dim: usize, rng: &mut Rng) -> Self {
        let s = (2.0 / code_dim as f32).sqrt();
        LearnedPreproc {
            embeddings: (0..27).map(|_| (0..code_dim).map(|_| rng.normal() * s).collect()).collect(),
            code_dim,
        }
    }

    fn encode_seq(&self, chars: &[u8]) -> Vec<f32> {
        chars.iter().flat_map(|&ch| self.embeddings[ch as usize].clone()).collect()
    }

    fn params(&self) -> usize { 27 * self.code_dim }
}

// Full pipeline: preprocessor + 3L brain, trained end-to-end
struct E2EPipeline {
    preproc: LearnedPreproc,
    // Brain: 3 layers
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    w3: Vec<Vec<f32>>, b3: Vec<f32>,
    w4: Vec<Vec<f32>>, b4: Vec<f32>,
    ctx: usize, idim: usize, h1: usize, h2: usize, h3: usize,
}

impl E2EPipeline {
    fn new(ctx: usize, code_dim: usize, h1: usize, h2: usize, h3: usize, rng: &mut Rng) -> Self {
        let idim = ctx * code_dim;
        let s1=(2.0/idim as f32).sqrt(); let s2=(2.0/h1 as f32).sqrt();
        let s3=(2.0/h2 as f32).sqrt(); let s4=(2.0/h3 as f32).sqrt();
        E2EPipeline {
            preproc: LearnedPreproc::new(code_dim, rng),
            w1: (0..h1).map(|_| (0..idim).map(|_| rng.normal()*s1).collect()).collect(), b1: vec![0.0;h1],
            w2: (0..h2).map(|_| (0..h1).map(|_| rng.normal()*s2).collect()).collect(), b2: vec![0.0;h2],
            w3: (0..h3).map(|_| (0..h2).map(|_| rng.normal()*s3).collect()).collect(), b3: vec![0.0;h3],
            w4: (0..27).map(|_| (0..h3).map(|_| rng.normal()*s4).collect()).collect(), b4: vec![0.0;27],
            ctx, idim, h1, h2, h3,
        }
    }

    fn total_params(&self) -> usize {
        self.preproc.params() + self.idim*self.h1+self.h1 + self.h1*self.h2+self.h2 + self.h2*self.h3+self.h3 + self.h3*27+27
    }

    fn train_step(&mut self, chars: &[u8], target: u8, lr: f32) {
        // Forward: encode
        let input = self.preproc.encode_seq(chars);

        // Layer 1
        let mut a1 = vec![0.0f32;self.h1];
        for k in 0..self.h1 { a1[k]=self.b1[k]; for j in 0..self.idim { a1[k]+=self.w1[k][j]*input[j]; } a1[k]=a1[k].max(0.0); }
        // Layer 2
        let mut a2 = vec![0.0f32;self.h2];
        for k in 0..self.h2 { a2[k]=self.b2[k]; for j in 0..self.h1 { a2[k]+=self.w2[k][j]*a1[j]; } a2[k]=a2[k].max(0.0); }
        // Layer 3
        let mut a3 = vec![0.0f32;self.h3];
        for k in 0..self.h3 { a3[k]=self.b3[k]; for j in 0..self.h2 { a3[k]+=self.w3[k][j]*a2[j]; } a3[k]=a3[k].max(0.0); }
        // Output
        let mut logits = vec![0.0f32;27];
        for c in 0..27 { logits[c]=self.b4[c]; for k in 0..self.h3 { logits[c]+=self.w4[c][k]*a3[k]; } }

        // Softmax
        let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
        let mut p=vec![0.0f32;27]; let mut s=0.0f32;
        for c in 0..27 { p[c]=(logits[c]-mx).exp(); s+=p[c]; } for c in 0..27 { p[c]/=s; }

        // Backprop
        let mut dl=p; dl[target as usize]-=1.0;

        // Layer 4 (output)
        let mut da3=vec![0.0f32;self.h3];
        for c in 0..27 { for k in 0..self.h3 { da3[k]+=dl[c]*self.w4[c][k]; self.w4[c][k]-=lr*dl[c]*a3[k]; } self.b4[c]-=lr*dl[c]; }
        // Layer 3
        let mut da2=vec![0.0f32;self.h2];
        for k in 0..self.h3 { if a3[k]<=0.0{continue;} for j in 0..self.h2 { da2[j]+=da3[k]*self.w3[k][j]; self.w3[k][j]-=lr*da3[k]*a2[j]; } self.b3[k]-=lr*da3[k]; }
        // Layer 2
        let mut da1=vec![0.0f32;self.h1];
        for k in 0..self.h2 { if a2[k]<=0.0{continue;} for j in 0..self.h1 { da1[j]+=da2[k]*self.w2[k][j]; self.w2[k][j]-=lr*da2[k]*a1[j]; } self.b2[k]-=lr*da2[k]; }
        // Layer 1
        let mut d_input=vec![0.0f32;self.idim];
        for k in 0..self.h1 { if a1[k]<=0.0{continue;} for j in 0..self.idim { d_input[j]+=da1[k]*self.w1[k][j]; self.w1[k][j]-=lr*da1[k]*input[j]; } self.b1[k]-=lr*da1[k]; }

        // Backprop through embeddings!
        let code_dim = self.preproc.code_dim;
        for i in 0..self.ctx {
            let ch = chars[i] as usize;
            for d in 0..code_dim {
                self.preproc.embeddings[ch][d] -= lr * d_input[i * code_dim + d];
            }
        }
    }

    fn eval(&self, corpus: &[u8], n: usize, seed: u64) -> f64 {
        let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
        for _ in 0..n { if corpus.len()<self.ctx+1{break;} let off=rng.range(0,corpus.len()-self.ctx-1);
            let input = self.preproc.encode_seq(&corpus[off..off+self.ctx]);
            let mut a1=vec![0.0f32;self.h1];
            for k in 0..self.h1 { a1[k]=self.b1[k]; for j in 0..self.idim { a1[k]+=self.w1[k][j]*input[j]; } a1[k]=a1[k].max(0.0); }
            let mut a2=vec![0.0f32;self.h2];
            for k in 0..self.h2 { a2[k]=self.b2[k]; for j in 0..self.h1 { a2[k]+=self.w2[k][j]*a1[j]; } a2[k]=a2[k].max(0.0); }
            let mut a3=vec![0.0f32;self.h3];
            for k in 0..self.h3 { a3[k]=self.b3[k]; for j in 0..self.h2 { a3[k]+=self.w3[k][j]*a2[j]; } a3[k]=a3[k].max(0.0); }
            let mut logits=vec![0.0f32;27];
            for c in 0..27 { logits[c]=self.b4[c]; for k in 0..self.h3 { logits[c]+=self.w4[c][k]*a3[k]; } }
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

    println!("=== END-TO-END LEARNED PREPROCESSOR ===\n");
    println!("  Corpus: {} chars, ctx={}", corpus.len(), ctx);
    println!("  Baselines: fixed preproc 70.1%, one-hot 93.8%, INSTNCT 24.6%\n");

    // Test different code dimensions with same brain architecture
    println!("  {:>30} {:>8} {:>8} {:>8}",
        "config", "params", "acc%", "time");
    println!("  {}", "─".repeat(60));

    let configs: Vec<(&str, usize, usize, usize, usize)> = vec![
        // (name, code_dim, h1, h2, h3)
        ("Learned-7  3L-512-256-128", 7, 512, 256, 128),   // same dim as fixed preproc
        ("Learned-14 3L-512-256-128", 14, 512, 256, 128),  // 2× wider encoding
        ("Learned-27 3L-512-256-128", 27, 512, 256, 128),  // same as one-hot dim
        ("Learned-7  3L-256-128-64",  7, 256, 128, 64),    // smaller brain
        ("Learned-5  3L-512-256-128", 5, 512, 256, 128),   // more compressed than fixed
    ];

    for (name, code_dim, h1, h2, h3) in &configs {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut pipe = E2EPipeline::new(ctx, *code_dim, *h1, *h2, *h3, &mut rng);

        let samples = 15000.min(corpus.len() / (ctx+1));
        for ep in 0..200 {
            let lr = 0.01 * (1.0 - ep as f32 / 200.0 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples {
                let off = rt.range(0, corpus.len()-ctx-1);
                pipe.train_step(&corpus[off..off+ctx], corpus[off+ctx], lr);
            }
            if (ep+1) % 50 == 0 {
                let a = pipe.eval(&corpus, 2000, 777 + ep as u64);
                print!("    ep{}: {:.1}%  ", ep+1, a);
            }
        }
        println!();

        let acc = pipe.eval(&corpus, 5000, 999);
        let m = if acc>90.0{" ★★★"} else if acc>80.0{" ★★"} else if acc>60.0{" ★"} else {""};

        println!("  {:>30} {:>8} {:>7.1}% {:>7.1}s{}",
            name, pipe.total_params(), acc, tc.elapsed().as_secs_f64(), m);

        // Show learned embeddings for first config
        if *code_dim == 7 && *h1 == 512 {
            println!("\n  Learned 7-dim embeddings (first 5 chars + space):");
            for &ch in &[0u8, 4, 8, 19, 25, 26] {
                let name_ch = if ch==26 { "space".to_string() } else { format!("'{}'", (ch+b'a') as char) };
                let emb = &pipe.preproc.embeddings[ch as usize];
                let vals: Vec<String> = emb.iter().map(|v| format!("{:+.2}", v)).collect();
                println!("    {:>5}: [{}]", name_ch, vals.join(", "));
            }
        }
        println!();
    }

    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
