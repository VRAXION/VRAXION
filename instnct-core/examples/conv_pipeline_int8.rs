//! Int8 quantization of the winning conv pipeline + larger context test
//!
//! Best pipeline: C19 encoder → Conv1D(k=3,f=64) → Brain(h=512) = 96.6%
//! Test: does int8 quantization of conv+brain preserve this accuracy?
//! Also: test with ctx=32 and ctx=64 to see if conv helps at larger context.
//!
//! Run: cargo run --example conv_pipeline_int8 --release

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

fn encode_byte(ch: u8) -> [f32;7] {
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

struct ConvBrainPipeline {
    // Conv: n_filters × (kernel_size × 7)
    conv_w: Vec<Vec<f32>>, conv_b: Vec<f32>,
    kernel_size: usize, n_filters: usize,
    // Brain: 1 hidden layer
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    brain_idim: usize, hdim: usize,
    ctx: usize,
}

impl ConvBrainPipeline {
    fn new(ctx: usize, kernel_size: usize, n_filters: usize, hdim: usize, rng: &mut Rng) -> Self {
        let fan_in = kernel_size * 7;
        let n_pos = ctx - kernel_size + 1;
        let brain_idim = n_pos * n_filters;
        let sc = (2.0/fan_in as f32).sqrt();
        let s1 = (2.0/brain_idim as f32).sqrt();
        let s2 = (2.0/hdim as f32).sqrt();
        ConvBrainPipeline {
            conv_w: (0..n_filters).map(|_| (0..fan_in).map(|_| rng.normal()*sc).collect()).collect(),
            conv_b: vec![0.0;n_filters],
            kernel_size, n_filters,
            w1: (0..hdim).map(|_| (0..brain_idim).map(|_| rng.normal()*s1).collect()).collect(),
            b1: vec![0.0;hdim],
            w2: (0..27).map(|_| (0..hdim).map(|_| rng.normal()*s2).collect()).collect(),
            b2: vec![0.0;27],
            brain_idim, hdim, ctx,
        }
    }

    fn total_params(&self) -> usize {
        self.n_filters*(self.kernel_size*7) + self.n_filters + self.brain_idim*self.hdim+self.hdim + self.hdim*27+27
    }

    fn forward(&self, chars: &[u8]) -> (Vec<[f32;7]>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let encoded: Vec<[f32;7]> = chars.iter().map(|&ch| encode_byte(ch)).collect();
        let n_pos = self.ctx - self.kernel_size + 1;
        let mut conv_out = vec![0.0f32; n_pos * self.n_filters];
        for pos in 0..n_pos {
            for f in 0..self.n_filters {
                let mut v = self.conv_b[f];
                for ki in 0..self.kernel_size { for d in 0..7 { v += self.conv_w[f][ki*7+d]*encoded[pos+ki][d]; } }
                conv_out[pos*self.n_filters+f] = v.max(0.0);
            }
        }
        let mut h = vec![0.0f32;self.hdim];
        for k in 0..self.hdim { h[k]=self.b1[k]; for j in 0..self.brain_idim { h[k]+=self.w1[k][j]*conv_out[j]; } h[k]=h[k].max(0.0); }
        let mut logits = vec![0.0f32;27];
        for c in 0..27 { logits[c]=self.b2[c]; for k in 0..self.hdim { logits[c]+=self.w2[c][k]*h[k]; } }
        (encoded, conv_out, h, logits)
    }

    fn train_step(&mut self, chars: &[u8], target: u8, lr: f32) {
        let (encoded, conv_out, h, logits) = self.forward(chars);
        let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
        let mut p=vec![0.0f32;27]; let mut s=0.0f32;
        for c in 0..27 { p[c]=(logits[c]-mx).exp(); s+=p[c]; } for c in 0..27 { p[c]/=s; }
        let mut dl=p; dl[target as usize]-=1.0;
        // Brain output
        let mut dh=vec![0.0f32;self.hdim];
        for c in 0..27 { for k in 0..self.hdim { dh[k]+=dl[c]*self.w2[c][k]; self.w2[c][k]-=lr*dl[c]*h[k]; } self.b2[c]-=lr*dl[c]; }
        // Brain hidden
        let mut d_conv=vec![0.0f32;self.brain_idim];
        for k in 0..self.hdim { if h[k]<=0.0{continue;} for j in 0..self.brain_idim { d_conv[j]+=dh[k]*self.w1[k][j]; self.w1[k][j]-=lr*dh[k]*conv_out[j]; } self.b1[k]-=lr*dh[k]; }
        // Conv
        let n_pos = self.ctx - self.kernel_size + 1;
        for pos in 0..n_pos {
            for f in 0..self.n_filters {
                let idx=pos*self.n_filters+f;
                if conv_out[idx]<=0.0{continue;}
                let d=d_conv[idx];
                for ki in 0..self.kernel_size { for dim in 0..7 { self.conv_w[f][ki*7+dim]-=lr*d*encoded[pos+ki][dim]; } }
                self.conv_b[f]-=lr*d;
            }
        }
    }

    fn eval_float(&self, corpus: &[u8], n: usize, seed: u64) -> f64 {
        let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
        for _ in 0..n { if corpus.len()<self.ctx+1{break;} let off=rng.range(0,corpus.len()-self.ctx-1);
            let (_,_,_,logits)=self.forward(&corpus[off..off+self.ctx]);
            let pred=logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred==corpus[off+self.ctx] as usize {ok+=1;} tot+=1;
        }
        ok as f64/tot as f64*100.0
    }

    fn eval_int8(&self, corpus: &[u8], n: usize, seed: u64) -> f64 {
        let quantize = |w: &Vec<Vec<f32>>| -> (Vec<Vec<i8>>, f32) {
            let mx=w.iter().flat_map(|r|r.iter()).map(|x|x.abs()).fold(0.0f32,f32::max);
            let s=if mx>0.0{127.0/mx}else{1.0};
            (w.iter().map(|r|r.iter().map(|&x|(x*s).round().max(-127.0).min(127.0) as i8).collect()).collect(), 1.0/s)
        };
        let (qc,dsc)=quantize(&self.conv_w);
        let (q1,ds1)=quantize(&self.w1);
        let (q2,ds2)=quantize(&self.w2);
        let n_pos=self.ctx-self.kernel_size+1;

        let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
        for _ in 0..n { if corpus.len()<self.ctx+1{break;} let off=rng.range(0,corpus.len()-self.ctx-1);
            let encoded: Vec<[f32;7]>=corpus[off..off+self.ctx].iter().map(|&ch| encode_byte(ch)).collect();
            let mut conv_out=vec![0.0f32;n_pos*self.n_filters];
            for pos in 0..n_pos { for f in 0..self.n_filters {
                let mut v=self.conv_b[f];
                for ki in 0..self.kernel_size { for d in 0..7 { v+=qc[f][ki*7+d] as f32*dsc*encoded[pos+ki][d]; } }
                conv_out[pos*self.n_filters+f]=v.max(0.0);
            }}
            let mut h=vec![0.0f32;self.hdim];
            for k in 0..self.hdim { h[k]=self.b1[k]; for j in 0..self.brain_idim { h[k]+=q1[k][j] as f32*ds1*conv_out[j]; } h[k]=h[k].max(0.0); }
            let mut logits=vec![0.0f32;27];
            for c in 0..27 { logits[c]=self.b2[c]; for k in 0..self.hdim { logits[c]+=q2[c][k] as f32*ds2*h[k]; } }
            let pred=logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred==corpus[off+self.ctx] as usize {ok+=1;} tot+=1;
        }
        ok as f64/tot as f64*100.0
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");

    println!("=== CONV PIPELINE INT8 + CONTEXT SWEEP ===\n");

    struct Cfg { name: &'static str, ctx: usize, k: usize, f: usize, h: usize, ep: usize }
    let configs = vec![
        // Best config at ctx=16, int8 test
        Cfg { name: "ctx=16 k=3 f=64 h=512 200ep", ctx: 16, k: 3, f: 64, h: 512, ep: 200 },
        // Context sweep with conv
        Cfg { name: "ctx=32 k=3 f=64 h=512 200ep", ctx: 32, k: 3, f: 64, h: 512, ep: 200 },
        Cfg { name: "ctx=64 k=3 f=64 h=512 150ep", ctx: 64, k: 3, f: 64, h: 512, ep: 150 },
        // Compact model
        Cfg { name: "ctx=16 k=2 f=32 h=256 200ep", ctx: 16, k: 2, f: 32, h: 256, ep: 200 },
    ];

    println!("  {:>35} {:>8} {:>8} {:>8} {:>7} {:>8}",
        "config", "params", "float%", "int8%", "Δ", "time");
    println!("  {}", "─".repeat(80));

    for cfg in &configs {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut pipe = ConvBrainPipeline::new(cfg.ctx, cfg.k, cfg.f, cfg.h, &mut rng);
        let samples = 15000.min(corpus.len() / (cfg.ctx+1));

        for ep in 0..cfg.ep {
            let lr = 0.01 * (1.0 - ep as f32 / cfg.ep as f32 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples {
                let off = rt.range(0, corpus.len()-cfg.ctx-1);
                pipe.train_step(&corpus[off..off+cfg.ctx], corpus[off+cfg.ctx], lr);
            }
        }

        let fa = pipe.eval_float(&corpus, 5000, 999);
        let ia = pipe.eval_int8(&corpus, 5000, 999);
        let m = if fa>95.0{" ★★★"} else if fa>85.0{" ★★"} else if fa>60.0{" ★"} else {""};

        println!("  {:>35} {:>8} {:>7.1}% {:>7.1}% {:>+6.1}% {:>7.1}s{}",
            cfg.name, pipe.total_params(), fa, ia, ia-fa, tc.elapsed().as_secs_f64(), m);
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
