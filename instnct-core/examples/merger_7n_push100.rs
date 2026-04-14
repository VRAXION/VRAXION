//! 7-neuron merger push to EXACT 100% — find the minimum bottleneck
//!
//! 7-neuron byte interpreter has redundancy (7 vs 4.75 bits needed).
//! Previous: 86% ratio (96/112) = 99.8%. Push to exact 100%.
//! Eval on HELD-OUT data (last 20% of corpus not used for training).
//!
//! Run: cargo run --example merger_7n_push100 --release

use std::time::Instant;

struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn normal(&mut self) -> f32 { let u1 = (((self.next()>>33)%65536) as f32/65536.0).max(1e-7); let u2 = ((self.next()>>33)%65536) as f32/65536.0; (-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos() }
    fn range(&mut self, lo: usize, hi: usize) -> usize { if hi<=lo{lo}else{lo+(self.next() as usize%(hi-lo))} }
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

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

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

fn encode_seq7(chars: &[u8]) -> Vec<f32> {
    chars.iter().flat_map(|&ch| encode7(ch).to_vec()).collect()
}

struct Merger {
    w: Vec<Vec<f32>>, enc_b: Vec<f32>, dec_b: Vec<f32>,
    idim: usize, bneck: usize,
}

impl Merger {
    fn new(idim: usize, bneck: usize, rng: &mut Rng) -> Self {
        let s = (2.0/idim as f32).sqrt();
        Merger {
            w: (0..bneck).map(|_| (0..idim).map(|_| rng.normal()*s).collect()).collect(),
            enc_b: vec![0.0;bneck], dec_b: vec![0.0;idim], idim, bneck,
        }
    }
    fn train_step(&mut self, input: &[f32], lr: f32) {
        let mut h=vec![0.0f32;self.bneck];
        for k in 0..self.bneck { h[k]=self.enc_b[k]; for j in 0..self.idim { h[k]+=self.w[k][j]*input[j]; } h[k]=sigmoid(h[k]); }
        let mut o=vec![0.0f32;self.idim];
        for j in 0..self.idim { o[j]=self.dec_b[j]; for k in 0..self.bneck { o[j]+=self.w[k][j]*h[k]; } }
        let mut d_o=vec![0.0f32;self.idim];
        for j in 0..self.idim { d_o[j]=2.0*(o[j]-input[j])/self.idim as f32; }
        let mut d_h=vec![0.0f32;self.bneck];
        for j in 0..self.idim { for k in 0..self.bneck { d_h[k]+=d_o[j]*self.w[k][j]; self.w[k][j]-=lr*d_o[j]*h[k]; } self.dec_b[j]-=lr*d_o[j]; }
        for k in 0..self.bneck { let dh=d_h[k]*h[k]*(1.0-h[k]); for j in 0..self.idim { self.w[k][j]-=lr*dh*input[j]; } self.enc_b[k]-=lr*dh; }
    }

    fn check_accuracy(&self, corpus: &[u8], ctx: usize, start: usize, end: usize) -> (usize, usize) {
        let mut ok=0usize; let mut tot=0usize;
        let mut pos = start;
        while pos+ctx <= end {
            let sig=encode_seq7(&corpus[pos..pos+ctx]);
            let mut h=vec![0.0f32;self.bneck];
            for k in 0..self.bneck { h[k]=self.enc_b[k]; for j in 0..self.idim { h[k]+=self.w[k][j]*sig[j]; } h[k]=sigmoid(h[k]); }
            let mut o=vec![0.0f32;self.idim];
            for j in 0..self.idim { o[j]=self.dec_b[j]; for k in 0..self.bneck { o[j]+=self.w[k][j]*h[k]; } }
            for i in 0..ctx {
                let rs=&o[i*7..(i+1)*7];
                let mut best=0u8; let mut bd=f32::MAX;
                for ch in 0..27u8 { let code=encode7(ch); let d:f32=code.iter().zip(rs).map(|(a,b)|(a-b)*(a-b)).sum(); if d<bd{bd=d;best=ch;} }
                if best==corpus[pos+i]{ok+=1;}
                tot+=1;
            }
            pos+=1;
            if tot>300_000{break;}
        }
        (ok, tot)
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let ctx = 16usize;
    let idim = ctx * 7; // 112

    // Train/test split: 80/20
    let split = corpus.len() * 80 / 100;
    let train = &corpus[..split];
    let test = &corpus[split..];

    println!("=== 7-NEURON MERGER PUSH TO 100% ===\n");
    println!("  Input: ctx={} × 7 = {} signals", ctx, idim);
    println!("  Train: {} chars, Test: {} chars (held-out)\n", train.len(), test.len());

    println!("  {:>6} {:>5} {:>12} {:>12} {:>8}",
        "bneck", "ratio", "train_acc", "test_acc", "time");
    println!("  {}", "─".repeat(50));

    for &bneck in &[112, 108, 104, 100, 96, 90, 84, 80, 72, 64] {
        if bneck >= idim { continue; }
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut merger = Merger::new(idim, bneck, &mut rng);

        let samples = 15000.min(train.len() / ctx);
        for ep in 0..300 {
            let lr = 0.01 * (1.0 - ep as f32 / 300.0 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples {
                let off = rt.range(0, train.len() - ctx);
                merger.train_step(&encode_seq7(&train[off..off+ctx]), lr);
            }
        }

        let (tr_ok, tr_tot) = merger.check_accuracy(&corpus, ctx, 0, split);
        let (te_ok, te_tot) = merger.check_accuracy(&corpus, ctx, split, corpus.len());
        let tr_pct = tr_ok as f64 / tr_tot as f64 * 100.0;
        let te_pct = te_ok as f64 / te_tot as f64 * 100.0;
        let ratio = bneck as f64 / idim as f64 * 100.0;

        let tr_mark = if tr_ok==tr_tot {"100%!"} else {""};
        let te_mark = if te_ok==te_tot {"100%!"} else {""};

        println!("  {:>6} {:>4.0}% {:>7}/{:>7} {:>7}/{:>7} {:>7.1}s {} {}",
            bneck, ratio, tr_ok, tr_tot, te_ok, te_tot, tc.elapsed().as_secs_f64(), tr_mark, te_mark);
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
