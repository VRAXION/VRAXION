//! Full pipeline: frozen binary preprocessor → int8 brain → next-char prediction
//!
//! Architecture:
//!   128 chars → shared frozen preprocessor (7 binary neurons, C19) → 896 signals
//!   → Brain layer(s) (backprop float → freeze int8) → 27 output (softmax)
//!
//! Tests:
//!   1. Sweep brain sizes: 1L (896→H→27) at H=64,128,256,512
//!   2. 2-layer brain: 896→H1→H2→27
//!   3. Float vs int8 comparison at each size
//!   4. Compare preprocessor vs one-hot input at matched param count
//!
//! Run: cargo run --example int8_brain_pipeline --release

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
// 1-Layer Brain: input → H (ReLU) → 27 (softmax)
// ══════════════════════════════════════════════════════
struct Brain1L {
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    idim: usize, hdim: usize,
}

impl Brain1L {
    fn new(idim: usize, hdim: usize, rng: &mut Rng) -> Self {
        let s1 = (2.0/idim as f32).sqrt();
        let s2 = (2.0/hdim as f32).sqrt();
        Brain1L {
            w1: (0..hdim).map(|_| (0..idim).map(|_| rng.normal()*s1).collect()).collect(),
            b1: vec![0.0;hdim],
            w2: (0..27).map(|_| (0..hdim).map(|_| rng.normal()*s2).collect()).collect(),
            b2: vec![0.0;27],
            idim, hdim,
        }
    }
    fn params(&self) -> usize { self.idim*self.hdim + self.hdim + self.hdim*27 + 27 }

    fn train_step(&mut self, input: &[f32], target: u8, lr: f32) {
        let mut h = vec![0.0f32; self.hdim];
        for k in 0..self.hdim { h[k]=self.b1[k]; for j in 0..self.idim { h[k]+=self.w1[k][j]*input[j]; } h[k]=h[k].max(0.0); }
        let mut logits = vec![0.0f32;27];
        for c in 0..27 { logits[c]=self.b2[c]; for k in 0..self.hdim { logits[c]+=self.w2[c][k]*h[k]; } }
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut p = vec![0.0f32;27]; let mut s=0.0f32;
        for c in 0..27 { p[c]=(logits[c]-mx).exp(); s+=p[c]; }
        for c in 0..27 { p[c]/=s; }
        let mut d=p.clone(); d[target as usize]-=1.0;
        let mut dh=vec![0.0f32;self.hdim];
        for c in 0..27 { for k in 0..self.hdim { dh[k]+=d[c]*self.w2[c][k]; self.w2[c][k]-=lr*d[c]*h[k]; } self.b2[c]-=lr*d[c]; }
        for k in 0..self.hdim { if h[k]<=0.0{continue;} for j in 0..self.idim { self.w1[k][j]-=lr*dh[k]*input[j]; } self.b1[k]-=lr*dh[k]; }
    }

    fn eval_float(&self, corpus: &[u8], ctx: usize, enc: &dyn Fn(&[u8])->Vec<f32>, n: usize, seed: u64) -> f64 {
        let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
        for _ in 0..n { if corpus.len()<ctx+1{break;} let off=rng.range(0,corpus.len()-ctx-1);
            let input=enc(&corpus[off..off+ctx]);
            let mut h=vec![0.0f32;self.hdim];
            for k in 0..self.hdim { h[k]=self.b1[k]; for j in 0..self.idim { h[k]+=self.w1[k][j]*input[j]; } h[k]=h[k].max(0.0); }
            let mut logits=vec![0.0f32;27];
            for c in 0..27 { logits[c]=self.b2[c]; for k in 0..self.hdim { logits[c]+=self.w2[c][k]*h[k]; } }
            let pred=logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred==corpus[off+ctx] as usize {ok+=1;} tot+=1;
        }
        ok as f64/tot as f64*100.0
    }

    fn eval_int8(&self, corpus: &[u8], ctx: usize, enc: &dyn Fn(&[u8])->Vec<f32>, n: usize, seed: u64) -> f64 {
        let mx1=self.w1.iter().flat_map(|r|r.iter()).map(|x|x.abs()).fold(0.0f32,f32::max);
        let mx2=self.w2.iter().flat_map(|r|r.iter()).map(|x|x.abs()).fold(0.0f32,f32::max);
        let s1=if mx1>0.0{127.0/mx1}else{1.0}; let s2=if mx2>0.0{127.0/mx2}else{1.0};
        let q1:Vec<Vec<i8>>=self.w1.iter().map(|r|r.iter().map(|&x|(x*s1).round().max(-127.0).min(127.0) as i8).collect()).collect();
        let q2:Vec<Vec<i8>>=self.w2.iter().map(|r|r.iter().map(|&x|(x*s2).round().max(-127.0).min(127.0) as i8).collect()).collect();
        let ds1=1.0/s1; let ds2=1.0/s2;
        let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
        for _ in 0..n { if corpus.len()<ctx+1{break;} let off=rng.range(0,corpus.len()-ctx-1);
            let input=enc(&corpus[off..off+ctx]);
            let mut h=vec![0.0f32;self.hdim];
            for k in 0..self.hdim { h[k]=self.b1[k]; for j in 0..self.idim { h[k]+=q1[k][j] as f32*ds1*input[j]; } h[k]=h[k].max(0.0); }
            let mut logits=vec![0.0f32;27];
            for c in 0..27 { logits[c]=self.b2[c]; for k in 0..self.hdim { logits[c]+=q2[c][k] as f32*ds2*h[k]; } }
            let pred=logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred==corpus[off+ctx] as usize {ok+=1;} tot+=1;
        }
        ok as f64/tot as f64*100.0
    }
}

// ══════════════════════════════════════════════════════
// 2-Layer Brain: input → H1 (ReLU) → H2 (ReLU) → 27
// ══════════════════════════════════════════════════════
struct Brain2L {
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    w3: Vec<Vec<f32>>, b3: Vec<f32>,
    idim: usize, h1: usize, h2: usize,
}

impl Brain2L {
    fn new(idim: usize, h1: usize, h2: usize, rng: &mut Rng) -> Self {
        let s1=(2.0/idim as f32).sqrt(); let s2=(2.0/h1 as f32).sqrt(); let s3=(2.0/h2 as f32).sqrt();
        Brain2L {
            w1: (0..h1).map(|_| (0..idim).map(|_| rng.normal()*s1).collect()).collect(), b1: vec![0.0;h1],
            w2: (0..h2).map(|_| (0..h1).map(|_| rng.normal()*s2).collect()).collect(), b2: vec![0.0;h2],
            w3: (0..27).map(|_| (0..h2).map(|_| rng.normal()*s3).collect()).collect(), b3: vec![0.0;27],
            idim, h1, h2,
        }
    }
    fn params(&self) -> usize { self.idim*self.h1+self.h1 + self.h1*self.h2+self.h2 + self.h2*27+27 }

    fn train_step(&mut self, input: &[f32], target: u8, lr: f32) {
        // Forward
        let mut a1=vec![0.0f32;self.h1];
        for k in 0..self.h1 { a1[k]=self.b1[k]; for j in 0..self.idim { a1[k]+=self.w1[k][j]*input[j]; } a1[k]=a1[k].max(0.0); }
        let mut a2=vec![0.0f32;self.h2];
        for k in 0..self.h2 { a2[k]=self.b2[k]; for j in 0..self.h1 { a2[k]+=self.w2[k][j]*a1[j]; } a2[k]=a2[k].max(0.0); }
        let mut logits=vec![0.0f32;27];
        for c in 0..27 { logits[c]=self.b3[c]; for k in 0..self.h2 { logits[c]+=self.w3[c][k]*a2[k]; } }
        let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
        let mut p=vec![0.0f32;27]; let mut s=0.0f32;
        for c in 0..27 { p[c]=(logits[c]-mx).exp(); s+=p[c]; } for c in 0..27 { p[c]/=s; }
        // Backprop
        let mut dl=p.clone(); dl[target as usize]-=1.0;
        let mut da2=vec![0.0f32;self.h2];
        for c in 0..27 { for k in 0..self.h2 { da2[k]+=dl[c]*self.w3[c][k]; self.w3[c][k]-=lr*dl[c]*a2[k]; } self.b3[c]-=lr*dl[c]; }
        let mut da1=vec![0.0f32;self.h1];
        for k in 0..self.h2 { if a2[k]<=0.0{continue;} for j in 0..self.h1 { da1[j]+=da2[k]*self.w2[k][j]; self.w2[k][j]-=lr*da2[k]*a1[j]; } self.b2[k]-=lr*da2[k]; }
        for k in 0..self.h1 { if a1[k]<=0.0{continue;} for j in 0..self.idim { self.w1[k][j]-=lr*da1[k]*input[j]; } self.b1[k]-=lr*da1[k]; }
    }

    fn eval_float(&self, corpus: &[u8], ctx: usize, enc: &dyn Fn(&[u8])->Vec<f32>, n: usize, seed: u64) -> f64 {
        let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
        for _ in 0..n { if corpus.len()<ctx+1{break;} let off=rng.range(0,corpus.len()-ctx-1);
            let input=enc(&corpus[off..off+ctx]);
            let mut a1=vec![0.0f32;self.h1];
            for k in 0..self.h1 { a1[k]=self.b1[k]; for j in 0..self.idim { a1[k]+=self.w1[k][j]*input[j]; } a1[k]=a1[k].max(0.0); }
            let mut a2=vec![0.0f32;self.h2];
            for k in 0..self.h2 { a2[k]=self.b2[k]; for j in 0..self.h1 { a2[k]+=self.w2[k][j]*a1[j]; } a2[k]=a2[k].max(0.0); }
            let mut logits=vec![0.0f32;27];
            for c in 0..27 { logits[c]=self.b3[c]; for k in 0..self.h2 { logits[c]+=self.w3[c][k]*a2[k]; } }
            let pred=logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred==corpus[off+ctx] as usize {ok+=1;} tot+=1;
        }
        ok as f64/tot as f64*100.0
    }

    fn eval_int8(&self, corpus: &[u8], ctx: usize, enc: &dyn Fn(&[u8])->Vec<f32>, n: usize, seed: u64) -> f64 {
        let mx1=self.w1.iter().flat_map(|r|r.iter()).map(|x|x.abs()).fold(0.0f32,f32::max);
        let mx2=self.w2.iter().flat_map(|r|r.iter()).map(|x|x.abs()).fold(0.0f32,f32::max);
        let mx3=self.w3.iter().flat_map(|r|r.iter()).map(|x|x.abs()).fold(0.0f32,f32::max);
        let s1=if mx1>0.0{127.0/mx1}else{1.0}; let s2=if mx2>0.0{127.0/mx2}else{1.0}; let s3=if mx3>0.0{127.0/mx3}else{1.0};
        let q1:Vec<Vec<i8>>=self.w1.iter().map(|r|r.iter().map(|&x|(x*s1).round().max(-127.0).min(127.0) as i8).collect()).collect();
        let q2:Vec<Vec<i8>>=self.w2.iter().map(|r|r.iter().map(|&x|(x*s2).round().max(-127.0).min(127.0) as i8).collect()).collect();
        let q3:Vec<Vec<i8>>=self.w3.iter().map(|r|r.iter().map(|&x|(x*s3).round().max(-127.0).min(127.0) as i8).collect()).collect();
        let ds1=1.0/s1; let ds2=1.0/s2; let ds3=1.0/s3;
        let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
        for _ in 0..n { if corpus.len()<ctx+1{break;} let off=rng.range(0,corpus.len()-ctx-1);
            let input=enc(&corpus[off..off+ctx]);
            let mut a1=vec![0.0f32;self.h1];
            for k in 0..self.h1 { a1[k]=self.b1[k]; for j in 0..self.idim { a1[k]+=q1[k][j] as f32*ds1*input[j]; } a1[k]=a1[k].max(0.0); }
            let mut a2=vec![0.0f32;self.h2];
            for k in 0..self.h2 { a2[k]=self.b2[k]; for j in 0..self.h1 { a2[k]+=q2[k][j] as f32*ds2*a1[j]; } a2[k]=a2[k].max(0.0); }
            let mut logits=vec![0.0f32;27];
            for c in 0..27 { logits[c]=self.b3[c]; for k in 0..self.h2 { logits[c]+=q3[c][k] as f32*ds3*a2[k]; } }
            let pred=logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred==corpus[off+ctx] as usize {ok+=1;} tot+=1;
        }
        ok as f64/tot as f64*100.0
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let pp = Preproc::new();
    let ctx = 128usize;
    let idim = ctx * 7; // 896

    println!("=== INT8 BRAIN PIPELINE — frozen binary preproc → int8 brain ===\n");
    println!("  Corpus: {} chars, Context: {} chars, Input: {} signals", corpus.len(), ctx, idim);
    println!("  Baselines: random=3.7%, frequency=20.3%, INSTNCT=24.6%\n");

    let enc_pp = |chars: &[u8]| -> Vec<f32> { pp.encode_seq(chars) };

    // ── PART 1: 1-Layer brain sweep ──
    println!("━━━ PART 1: 1-Layer brain (896→H→27), 150 epochs ━━━\n");
    println!("  {:>5} {:>8} {:>8} {:>8} {:>7} {:>8}",
        "H", "params", "float%", "int8%", "Δ", "time");
    println!("  {}", "─".repeat(50));

    for &hdim in &[32, 64, 128, 256, 512] {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut brain = Brain1L::new(idim, hdim, &mut rng);

        let samples = 10000.min(corpus.len() / (ctx+1));
        for ep in 0..150 {
            let lr = 0.01 * (1.0 - ep as f32 / 150.0 * 0.7);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples {
                let off = rt.range(0, corpus.len()-ctx-1);
                brain.train_step(&enc_pp(&corpus[off..off+ctx]), corpus[off+ctx], lr);
            }
        }

        let fa = brain.eval_float(&corpus, ctx, &enc_pp, 5000, 999);
        let ia = brain.eval_int8(&corpus, ctx, &enc_pp, 5000, 999);
        let m = if fa > 50.0 { " ★★" } else if fa > 30.0 { " ★" } else { "" };

        println!("  {:>5} {:>8} {:>7.1}% {:>7.1}% {:>+6.1}% {:>7.1}s{}",
            hdim, brain.params(), fa, ia, ia-fa, tc.elapsed().as_secs_f64(), m);
    }

    // ── PART 2: 2-Layer brain ──
    println!("\n━━━ PART 2: 2-Layer brain (896→H1→H2→27), 150 epochs ━━━\n");
    println!("  {:>9} {:>8} {:>8} {:>8} {:>7} {:>8}",
        "arch", "params", "float%", "int8%", "Δ", "time");
    println!("  {}", "─".repeat(55));

    for &(h1, h2) in &[(256, 128), (512, 256), (512, 128)] {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut brain = Brain2L::new(idim, h1, h2, &mut rng);

        let samples = 10000.min(corpus.len() / (ctx+1));
        for ep in 0..150 {
            let lr = 0.01 * (1.0 - ep as f32 / 150.0 * 0.7);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples {
                let off = rt.range(0, corpus.len()-ctx-1);
                brain.train_step(&enc_pp(&corpus[off..off+ctx]), corpus[off+ctx], lr);
            }
        }

        let fa = brain.eval_float(&corpus, ctx, &enc_pp, 5000, 999);
        let ia = brain.eval_int8(&corpus, ctx, &enc_pp, 5000, 999);
        let arch = format!("{}→{}", h1, h2);
        let m = if fa > 50.0 { " ★★" } else if fa > 30.0 { " ★" } else { "" };

        println!("  {:>9} {:>8} {:>7.1}% {:>7.1}% {:>+6.1}% {:>7.1}s{}",
            arch, brain.params(), fa, ia, ia-fa, tc.elapsed().as_secs_f64(), m);
    }

    // ── PART 3: Context size comparison ──
    println!("\n━━━ PART 3: Context sweep (H=256, 1L, 100 epochs) ━━━\n");
    println!("  {:>5} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "ctx", "input", "params", "float%", "int8%", "time");
    println!("  {}", "─".repeat(50));

    for &c in &[16, 32, 64, 128] {
        let id = c * 7;
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut brain = Brain1L::new(id, 256, &mut rng);

        let samples = 10000.min(corpus.len() / (c+1));
        for ep in 0..100 {
            let lr = 0.01 * (1.0 - ep as f32 / 100.0 * 0.7);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples {
                let off = rt.range(0, corpus.len()-c-1);
                brain.train_step(&pp.encode_seq(&corpus[off..off+c]), corpus[off+c], lr);
            }
        }

        let fa = brain.eval_float(&corpus, c, &|ch| pp.encode_seq(ch), 5000, 999);
        let ia = brain.eval_int8(&corpus, c, &|ch| pp.encode_seq(ch), 5000, 999);
        let m = if fa > 50.0 { " ★★" } else if fa > 30.0 { " ★" } else { "" };

        println!("  {:>5} {:>6} {:>8} {:>7.1}% {:>7.1}% {:>7.1}s{}",
            c, id, brain.params(), fa, ia, tc.elapsed().as_secs_f64(), m);
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
