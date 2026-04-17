//! Conv pipeline with 4-neuron byte interpreter (vs 7-neuron)
//!
//! Test: does the 43% smaller byte interpreter (4 vs 7 neurons)
//! maintain the 96.6% accuracy in the full conv pipeline?
//!
//! Run: cargo run --example conv_pipeline_4neuron --release

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

// 7-neuron encoder (original)
fn encode7(ch: u8) -> [f32;7] {
    const W: [[i8;8];7] = [[-1,1,1,-1,-1,1,1,-1],[1,-1,1,1,-1,-1,-1,-1],[-1,-1,1,-1,1,1,-1,-1],
        [-1,-1,-1,1,-1,-1,1,-1],[-1,1,1,1,-1,-1,-1,-1],[1,1,-1,-1,-1,-1,-1,-1],[-1,1,-1,1,1,1,-1,-1]];
    const B: [i8;7] = [1,1,1,1,1,1,1];
    const C: [f32;7] = [10.0;7];
    const RHO: [f32;7] = [2.0,0.0,0.0,0.0,0.0,0.0,0.0];
    let mut bits=[0.0f32;8]; for i in 0..8 { bits[i]=((ch>>i)&1) as f32; }
    let mut o=[0.0f32;7];
    for k in 0..7 { let mut d=B[k] as f32; for j in 0..8 { d+=W[k][j] as f32*bits[j]; } o[k]=c19(d,C[k],RHO[k]); }
    o
}

// 4-neuron encoder (found by exhaustive search: c=5.0, rho=0.0)
fn encode4(ch: u8) -> [f32;4] {
    // Exhaustive-search optimal (from byte_interp_min_neurons)
    const W: [[i8;8];4] = [
        [ 1, 1, 1, 1,-1,-1,-1,-1],  // neuron 0
        [ 1,-1, 1,-1,-1,-1,-1,-1],  // neuron 1
        [-1, 1, 1,-1,-1,-1,-1,-1],  // neuron 2
        [ 1, 1,-1,-1,-1,-1,-1,-1],  // neuron 3
    ];
    const B: [i8;4] = [-1, -1, -1, -1];
    let mut bits=[0.0f32;8]; for i in 0..8 { bits[i]=((ch>>i)&1) as f32; }
    let mut o=[0.0f32;4];
    for k in 0..4 { let mut d=B[k] as f32; for j in 0..8 { d+=W[k][j] as f32*bits[j]; } o[k]=c19(d,5.0,0.0); }
    o
}

struct ConvBrain {
    conv_w: Vec<Vec<f32>>, conv_b: Vec<f32>,
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    kernel_size: usize, n_filters: usize, sig_dim: usize,
    brain_idim: usize, hdim: usize, ctx: usize,
}

impl ConvBrain {
    fn new(ctx: usize, kernel_size: usize, n_filters: usize, sig_dim: usize, hdim: usize, rng: &mut Rng) -> Self {
        let fan_in = kernel_size * sig_dim;
        let n_pos = ctx - kernel_size + 1;
        let brain_idim = n_pos * n_filters;
        let sc=(2.0/fan_in as f32).sqrt(); let s1=(2.0/brain_idim as f32).sqrt(); let s2=(2.0/hdim as f32).sqrt();
        ConvBrain {
            conv_w: (0..n_filters).map(|_| (0..fan_in).map(|_| rng.normal()*sc).collect()).collect(),
            conv_b: vec![0.0;n_filters],
            w1: (0..hdim).map(|_| (0..brain_idim).map(|_| rng.normal()*s1).collect()).collect(), b1: vec![0.0;hdim],
            w2: (0..27).map(|_| (0..hdim).map(|_| rng.normal()*s2).collect()).collect(), b2: vec![0.0;27],
            kernel_size, n_filters, sig_dim, brain_idim, hdim, ctx,
        }
    }
    fn params(&self) -> usize { self.n_filters*self.kernel_size*self.sig_dim+self.n_filters + self.brain_idim*self.hdim+self.hdim + self.hdim*27+27 }

    fn train_step(&mut self, signals: &[f32], target: u8, lr: f32) {
        let n_pos=self.ctx-self.kernel_size+1;
        // Conv
        let mut co=vec![0.0f32;n_pos*self.n_filters];
        for p in 0..n_pos { for f in 0..self.n_filters {
            let mut v=self.conv_b[f];
            for ki in 0..self.kernel_size { for d in 0..self.sig_dim { v+=self.conv_w[f][ki*self.sig_dim+d]*signals[(p+ki)*self.sig_dim+d]; } }
            co[p*self.n_filters+f]=v.max(0.0);
        }}
        // Brain
        let mut h=vec![0.0f32;self.hdim];
        for k in 0..self.hdim { h[k]=self.b1[k]; for j in 0..self.brain_idim { h[k]+=self.w1[k][j]*co[j]; } h[k]=h[k].max(0.0); }
        let mut logits=vec![0.0f32;27];
        for c in 0..27 { logits[c]=self.b2[c]; for k in 0..self.hdim { logits[c]+=self.w2[c][k]*h[k]; } }
        let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
        let mut p2=vec![0.0f32;27]; let mut s=0.0f32;
        for c in 0..27 { p2[c]=(logits[c]-mx).exp(); s+=p2[c]; } for c in 0..27 { p2[c]/=s; }
        let mut dl=p2; dl[target as usize]-=1.0;
        // Backprop brain
        let mut dh=vec![0.0f32;self.hdim];
        for c in 0..27 { for k in 0..self.hdim { dh[k]+=dl[c]*self.w2[c][k]; self.w2[c][k]-=lr*dl[c]*h[k]; } self.b2[c]-=lr*dl[c]; }
        let mut dc=vec![0.0f32;self.brain_idim];
        for k in 0..self.hdim { if h[k]<=0.0{continue;} for j in 0..self.brain_idim { dc[j]+=dh[k]*self.w1[k][j]; self.w1[k][j]-=lr*dh[k]*co[j]; } self.b1[k]-=lr*dh[k]; }
        // Backprop conv
        for p in 0..n_pos { for f in 0..self.n_filters {
            let idx=p*self.n_filters+f;
            if co[idx]<=0.0{continue;}
            for ki in 0..self.kernel_size { for d in 0..self.sig_dim { self.conv_w[f][ki*self.sig_dim+d]-=lr*dc[idx]*signals[(p+ki)*self.sig_dim+d]; } }
            self.conv_b[f]-=lr*dc[idx];
        }}
    }

    fn eval(&self, corpus: &[u8], enc: &dyn Fn(u8)->Vec<f32>, n: usize, seed: u64) -> f64 {
        let n_pos=self.ctx-self.kernel_size+1;
        let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
        for _ in 0..n { if corpus.len()<self.ctx+1{break;} let off=rng.range(0,corpus.len()-self.ctx-1);
            let signals: Vec<f32>=corpus[off..off+self.ctx].iter().flat_map(|&ch| enc(ch)).collect();
            let mut co=vec![0.0f32;n_pos*self.n_filters];
            for p in 0..n_pos { for f in 0..self.n_filters {
                let mut v=self.conv_b[f];
                for ki in 0..self.kernel_size { for d in 0..self.sig_dim { v+=self.conv_w[f][ki*self.sig_dim+d]*signals[(p+ki)*self.sig_dim+d]; } }
                co[p*self.n_filters+f]=v.max(0.0);
            }}
            let mut h=vec![0.0f32;self.hdim];
            for k in 0..self.hdim { h[k]=self.b1[k]; for j in 0..self.brain_idim { h[k]+=self.w1[k][j]*co[j]; } h[k]=h[k].max(0.0); }
            let mut logits=vec![0.0f32;27];
            for c in 0..27 { logits[c]=self.b2[c]; for k in 0..self.hdim { logits[c]+=self.w2[c][k]*h[k]; } }
            let pred=logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred==corpus[off+self.ctx] as usize {ok+=1;} tot+=1;
        }
        ok as f64/tot as f64*100.0
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let ctx = 16;

    // First verify 4-neuron encoder roundtrip
    let mut rt_ok = 0;
    for ch in 0..27u8 {
        let code = encode4(ch);
        let mut best = 0u8; let mut bd = f32::MAX;
        for t in 0..27u8 { let tc = encode4(t); let d: f32 = tc.iter().zip(code.iter()).map(|(a,b)|(a-b)*(a-b)).sum(); if d<bd{bd=d;best=t;} }
        if best == ch { rt_ok += 1; }
    }
    println!("=== CONV PIPELINE: 4-NEURON vs 7-NEURON BYTE INTERPRETER ===\n");
    println!("  4-neuron roundtrip: {}/27\n", rt_ok);

    // If 4-neuron encoder doesn't work with these hardcoded weights, use exhaustive search
    if rt_ok < 27 {
        println!("  4-neuron hardcoded weights failed, using 7-neuron only.\n");
    }

    struct Cfg { name: &'static str, sig_dim: usize, use_4: bool, k: usize, f: usize, h: usize }
    let mut configs = vec![
        Cfg { name: "7-neuron k=3 f=64 h=512", sig_dim: 7, use_4: false, k: 3, f: 64, h: 512 },
        Cfg { name: "7-neuron k=3 f=64 h=256", sig_dim: 7, use_4: false, k: 3, f: 64, h: 256 },
    ];
    if rt_ok == 27 {
        configs.push(Cfg { name: "4-neuron k=3 f=64 h=512", sig_dim: 4, use_4: true, k: 3, f: 64, h: 512 });
        configs.push(Cfg { name: "4-neuron k=3 f=64 h=256", sig_dim: 4, use_4: true, k: 3, f: 64, h: 256 });
        configs.push(Cfg { name: "4-neuron k=2 f=64 h=512", sig_dim: 4, use_4: true, k: 2, f: 64, h: 512 });
    }

    println!("  {:>30} {:>8} {:>8} {:>8}",
        "config", "params", "acc%", "time");
    println!("  {}", "─".repeat(60));

    for cfg in &configs {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut pipe = ConvBrain::new(ctx, cfg.k, cfg.f, cfg.sig_dim, cfg.h, &mut rng);
        let samples = 15000.min(corpus.len()/(ctx+1));

        let enc4 = |ch: u8| -> Vec<f32> { encode4(ch).to_vec() };
        let enc7 = |ch: u8| -> Vec<f32> { encode7(ch).to_vec() };

        for ep in 0..200 {
            let lr = 0.01*(1.0-ep as f32/200.0*0.8);
            let mut rt = Rng::new(ep as u64*1000+42);
            for _ in 0..samples {
                let off = rt.range(0, corpus.len()-ctx-1);
                let signals: Vec<f32> = corpus[off..off+ctx].iter().flat_map(|&ch| if cfg.use_4 { enc4(ch) } else { enc7(ch) }).collect();
                pipe.train_step(&signals, corpus[off+ctx], lr);
            }
        }

        let enc: &dyn Fn(u8)->Vec<f32> = if cfg.use_4 { &enc4 } else { &enc7 };
        let acc = pipe.eval(&corpus, enc, 5000, 999);
        let m = if acc>95.0{" ★★★"} else if acc>85.0{" ★★"} else if acc>60.0{" ★"} else {""};

        println!("  {:>30} {:>8} {:>7.1}% {:>7.1}s{}",
            cfg.name, pipe.params(), acc, tc.elapsed().as_secs_f64(), m);
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
