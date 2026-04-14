//! Push int8 brain pipeline to ceiling — combine all proven techniques
//!
//! Best known: abstract_core_v4 = 93.3% (3L 512→256→128, ctx=16, 200ep, 15K samples)
//! This test: frozen binary preprocessor → deep brain → int8 freeze
//!
//! Tests:
//!   A. 3L brain at ctx=16: 512→256→128→27 (match v4 architecture)
//!   B. Same with C19 activation (proven better on small models)
//!   C. 200 epochs, 15K samples (match v4 training budget)
//!   D. Int8 quantization of best model
//!
//! Run: cargo run --example int8_brain_push_ceiling --release

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

fn c19_deriv(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0*c;
    if x >= l || x <= -l { return 1.0; }
    let s = x/c; let n = s.floor(); let t = s-n;
    let sg = if (n as i32)%2==0 { 1.0 } else { -1.0 };
    (sg*(1.0-2.0*t) + rho*2.0*t*(1.0-t)) / c
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

#[derive(Clone, Copy, PartialEq)]
enum Act { ReLU, C19 }

// 3-Layer brain: input → H1 → H2 → H3 → 27
struct Brain3L {
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    w3: Vec<Vec<f32>>, b3: Vec<f32>,
    w4: Vec<Vec<f32>>, b4: Vec<f32>,
    idim: usize, h1: usize, h2: usize, h3: usize,
    act: Act,
    // C19 per-neuron params (learnable)
    c1: Vec<f32>, rho1: Vec<f32>,
    c2: Vec<f32>, rho2: Vec<f32>,
    c3: Vec<f32>, rho3: Vec<f32>,
}

impl Brain3L {
    fn new(idim: usize, h1: usize, h2: usize, h3: usize, act: Act, rng: &mut Rng) -> Self {
        let s1=(2.0/idim as f32).sqrt(); let s2=(2.0/h1 as f32).sqrt();
        let s3=(2.0/h2 as f32).sqrt(); let s4=(2.0/h3 as f32).sqrt();
        Brain3L {
            w1: (0..h1).map(|_| (0..idim).map(|_| rng.normal()*s1).collect()).collect(), b1: vec![0.0;h1],
            w2: (0..h2).map(|_| (0..h1).map(|_| rng.normal()*s2).collect()).collect(), b2: vec![0.0;h2],
            w3: (0..h3).map(|_| (0..h2).map(|_| rng.normal()*s3).collect()).collect(), b3: vec![0.0;h3],
            w4: (0..27).map(|_| (0..h3).map(|_| rng.normal()*s4).collect()).collect(), b4: vec![0.0;27],
            c1: vec![1.0;h1], rho1: vec![0.5;h1],
            c2: vec![1.0;h2], rho2: vec![0.5;h2],
            c3: vec![1.0;h3], rho3: vec![0.5;h3],
            idim, h1, h2, h3, act,
        }
    }

    fn params(&self) -> usize {
        let base = self.idim*self.h1+self.h1 + self.h1*self.h2+self.h2 + self.h2*self.h3+self.h3 + self.h3*27+27;
        if self.act == Act::C19 { base + (self.h1+self.h2+self.h3)*2 } else { base }
    }

    fn activate(&self, x: f32, c: f32, rho: f32) -> f32 {
        match self.act { Act::ReLU => x.max(0.0), Act::C19 => c19(x, c, rho) }
    }

    fn activate_deriv(&self, x: f32, a: f32, c: f32, rho: f32) -> f32 {
        match self.act { Act::ReLU => if a > 0.0 { 1.0 } else { 0.0 }, Act::C19 => c19_deriv(x, c, rho) }
    }

    fn train_step(&mut self, input: &[f32], target: u8, lr: f32) {
        // Forward
        let mut z1=vec![0.0f32;self.h1]; let mut a1=vec![0.0f32;self.h1];
        for k in 0..self.h1 { z1[k]=self.b1[k]; for j in 0..self.idim { z1[k]+=self.w1[k][j]*input[j]; } a1[k]=self.activate(z1[k],self.c1[k],self.rho1[k]); }
        let mut z2=vec![0.0f32;self.h2]; let mut a2=vec![0.0f32;self.h2];
        for k in 0..self.h2 { z2[k]=self.b2[k]; for j in 0..self.h1 { z2[k]+=self.w2[k][j]*a1[j]; } a2[k]=self.activate(z2[k],self.c2[k],self.rho2[k]); }
        let mut z3=vec![0.0f32;self.h3]; let mut a3=vec![0.0f32;self.h3];
        for k in 0..self.h3 { z3[k]=self.b3[k]; for j in 0..self.h2 { z3[k]+=self.w3[k][j]*a2[j]; } a3[k]=self.activate(z3[k],self.c3[k],self.rho3[k]); }
        let mut logits=vec![0.0f32;27];
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
        for k in 0..self.h3 { let d=da3[k]*self.activate_deriv(z3[k],a3[k],self.c3[k],self.rho3[k]);
            for j in 0..self.h2 { da2[j]+=d*self.w3[k][j]; self.w3[k][j]-=lr*d*a2[j]; } self.b3[k]-=lr*d; }

        // Layer 2
        let mut da1=vec![0.0f32;self.h1];
        for k in 0..self.h2 { let d=da2[k]*self.activate_deriv(z2[k],a2[k],self.c2[k],self.rho2[k]);
            for j in 0..self.h1 { da1[j]+=d*self.w2[k][j]; self.w2[k][j]-=lr*d*a1[j]; } self.b2[k]-=lr*d; }

        // Layer 1
        for k in 0..self.h1 { let d=da1[k]*self.activate_deriv(z1[k],a1[k],self.c1[k],self.rho1[k]);
            for j in 0..self.idim { self.w1[k][j]-=lr*d*input[j]; } self.b1[k]-=lr*d; }
    }

    fn eval(&self, corpus: &[u8], ctx: usize, enc: &dyn Fn(&[u8])->Vec<f32>, n: usize, seed: u64) -> f64 {
        let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
        for _ in 0..n { if corpus.len()<ctx+1{break;} let off=rng.range(0,corpus.len()-ctx-1);
            let input=enc(&corpus[off..off+ctx]);
            let mut a1=vec![0.0f32;self.h1];
            for k in 0..self.h1 { a1[k]=self.b1[k]; for j in 0..self.idim { a1[k]+=self.w1[k][j]*input[j]; } a1[k]=self.activate(a1[k],self.c1[k],self.rho1[k]); }
            let mut a2=vec![0.0f32;self.h2];
            for k in 0..self.h2 { a2[k]=self.b2[k]; for j in 0..self.h1 { a2[k]+=self.w2[k][j]*a1[j]; } a2[k]=self.activate(a2[k],self.c2[k],self.rho2[k]); }
            let mut a3=vec![0.0f32;self.h3];
            for k in 0..self.h3 { a3[k]=self.b3[k]; for j in 0..self.h2 { a3[k]+=self.w3[k][j]*a2[j]; } a3[k]=self.activate(a3[k],self.c3[k],self.rho3[k]); }
            let mut logits=vec![0.0f32;27];
            for c in 0..27 { logits[c]=self.b4[c]; for k in 0..self.h3 { logits[c]+=self.w4[c][k]*a3[k]; } }
            let pred=logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred==corpus[off+ctx] as usize {ok+=1;} tot+=1;
        }
        ok as f64/tot as f64*100.0
    }

    fn eval_int8(&self, corpus: &[u8], ctx: usize, enc: &dyn Fn(&[u8])->Vec<f32>, n: usize, seed: u64) -> f64 {
        // Quantize all 4 weight matrices
        let quantize = |w: &Vec<Vec<f32>>| -> (Vec<Vec<i8>>, f32) {
            let mx = w.iter().flat_map(|r|r.iter()).map(|x|x.abs()).fold(0.0f32,f32::max);
            let s = if mx>0.0{127.0/mx}else{1.0};
            (w.iter().map(|r|r.iter().map(|&x|(x*s).round().max(-127.0).min(127.0) as i8).collect()).collect(), 1.0/s)
        };
        let (q1,ds1)=quantize(&self.w1); let (q2,ds2)=quantize(&self.w2);
        let (q3,ds3)=quantize(&self.w3); let (q4,ds4)=quantize(&self.w4);

        let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
        for _ in 0..n { if corpus.len()<ctx+1{break;} let off=rng.range(0,corpus.len()-ctx-1);
            let input=enc(&corpus[off..off+ctx]);
            let mut a1=vec![0.0f32;self.h1];
            for k in 0..self.h1 { a1[k]=self.b1[k]; for j in 0..self.idim { a1[k]+=q1[k][j] as f32*ds1*input[j]; } a1[k]=self.activate(a1[k],self.c1[k],self.rho1[k]); }
            let mut a2=vec![0.0f32;self.h2];
            for k in 0..self.h2 { a2[k]=self.b2[k]; for j in 0..self.h1 { a2[k]+=q2[k][j] as f32*ds2*a1[j]; } a2[k]=self.activate(a2[k],self.c2[k],self.rho2[k]); }
            let mut a3=vec![0.0f32;self.h3];
            for k in 0..self.h3 { a3[k]=self.b3[k]; for j in 0..self.h2 { a3[k]+=q3[k][j] as f32*ds3*a2[j]; } a3[k]=self.activate(a3[k],self.c3[k],self.rho3[k]); }
            let mut logits=vec![0.0f32;27];
            for c in 0..27 { logits[c]=self.b4[c]; for k in 0..self.h3 { logits[c]+=q4[c][k] as f32*ds4*a3[k]; } }
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
    let ctx = 16usize;
    let idim = ctx * 7; // 112

    println!("=== PUSH TO CEILING — frozen preproc → deep int8 brain ===\n");
    println!("  Corpus: {} chars, ctx={}, input={}", corpus.len(), ctx, idim);
    println!("  Target: match abstract_core_v4 93.3% (3L 512→256→128, 200ep)");
    println!("  Baselines: freq=20.3%, INSTNCT=24.6%, v4=93.3%\n");

    let enc = |chars: &[u8]| -> Vec<f32> { pp.encode_seq(chars) };

    // Test configurations
    let configs: Vec<(&str, usize, usize, usize, Act, usize, usize)> = vec![
        // (name, h1, h2, h3, activation, epochs, samples)
        ("3L-512-256-128 ReLU 200ep", 512, 256, 128, Act::ReLU, 200, 15000),
        ("3L-512-256-128 C19  200ep", 512, 256, 128, Act::C19,  200, 15000),
        ("3L-256-128-64  ReLU 200ep", 256, 128, 64,  Act::ReLU, 200, 15000),
        ("3L-256-128-64  C19  200ep", 256, 128, 64,  Act::C19,  200, 15000),
        ("3L-128-64-32   ReLU 200ep", 128, 64,  32,  Act::ReLU, 200, 15000),
        ("3L-512-256-128 ReLU 400ep", 512, 256, 128, Act::ReLU, 400, 15000),
    ];

    println!("  {:>30} {:>8} {:>8} {:>8} {:>7} {:>8}",
        "config", "params", "float%", "int8%", "Δ", "time");
    println!("  {}", "─".repeat(75));

    for (name, h1, h2, h3, act, epochs, samples) in &configs {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut brain = Brain3L::new(idim, *h1, *h2, *h3, *act, &mut rng);

        let max_samples = (*samples).min(corpus.len() / (ctx+1));
        for ep in 0..*epochs {
            let lr = 0.01 * (1.0 - ep as f32 / *epochs as f32 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..max_samples {
                let off = rt.range(0, corpus.len()-ctx-1);
                brain.train_step(&enc(&corpus[off..off+ctx]), corpus[off+ctx], lr);
            }

            // Progress every 50 epochs
            if (ep+1) % 50 == 0 {
                let acc = brain.eval(&corpus, ctx, &enc, 2000, 777+ep as u64);
                print!("    ep{}: {:.1}%  ", ep+1, acc);
                if (ep+1) % 200 == 0 { println!(); }
            }
        }
        println!();

        let fa = brain.eval(&corpus, ctx, &enc, 5000, 999);
        let ia = brain.eval_int8(&corpus, ctx, &enc, 5000, 999);
        let m = if fa > 90.0 { " ★★★" } else if fa > 70.0 { " ★★" } else if fa > 50.0 { " ★" } else { "" };

        println!("  {:>30} {:>8} {:>7.1}% {:>7.1}% {:>+6.1}% {:>7.1}s{}",
            name, brain.params(), fa, ia, ia-fa, tc.elapsed().as_secs_f64(), m);
        println!();
    }

    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
