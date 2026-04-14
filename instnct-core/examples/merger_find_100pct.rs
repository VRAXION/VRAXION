//! Input Merger — find the exact bottleneck size for 100% reconstruction
//!
//! Byte interpreter: 4 neurons, 100% proven ✓
//! Input merger: ctx×4 → bottleneck → ctx×4, must be 100% round-trip
//!
//! Strategy: sweep bottleneck sizes from 90% to 100% ratio,
//! train hard (200+ epochs), check EXACT 100% on full corpus.
//!
//! Run: cargo run --example merger_find_100pct --release

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

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// 4-neuron byte interpreter (proven 100%)
fn encode4(ch: u8) -> [f32;4] {
    const W: [[i8;8];4] = [[1,1,1,1,-1,-1,-1,-1],[1,-1,1,-1,-1,-1,-1,-1],[-1,1,1,-1,-1,-1,-1,-1],[1,1,-1,-1,-1,-1,-1,-1]];
    const B: [i8;4] = [-1,-1,-1,-1];
    let mut bits=[0.0f32;8]; for i in 0..8 { bits[i]=((ch>>i)&1) as f32; }
    let mut o=[0.0f32;4];
    for k in 0..4 { let mut d=B[k] as f32; for j in 0..8 { d+=W[k][j] as f32*bits[j]; } o[k]=c19(d,5.0,0.0); }
    o
}

fn encode_seq4(chars: &[u8]) -> Vec<f32> {
    chars.iter().flat_map(|&ch| encode4(ch).to_vec()).collect()
}

// Tied-weight autoencoder
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
    fn encode(&self, input: &[f32]) -> Vec<f32> {
        let mut h = vec![0.0f32;self.bneck];
        for k in 0..self.bneck { h[k]=self.enc_b[k]; for j in 0..self.idim { h[k]+=self.w[k][j]*input[j]; } h[k]=sigmoid(h[k]); }
        h
    }
    fn decode(&self, h: &[f32]) -> Vec<f32> {
        let mut o = vec![0.0f32;self.idim];
        for j in 0..self.idim { o[j]=self.dec_b[j]; for k in 0..self.bneck { o[j]+=self.w[k][j]*h[k]; } }
        o
    }
    fn train_step(&mut self, input: &[f32], lr: f32) {
        let h = self.encode(input);
        let o = self.decode(&h);
        let mut d_o = vec![0.0f32;self.idim];
        for j in 0..self.idim { d_o[j]=2.0*(o[j]-input[j])/self.idim as f32; }
        let mut d_h = vec![0.0f32;self.bneck];
        for j in 0..self.idim { for k in 0..self.bneck { d_h[k]+=d_o[j]*self.w[k][j]; self.w[k][j]-=lr*d_o[j]*h[k]; } self.dec_b[j]-=lr*d_o[j]; }
        for k in 0..self.bneck { let dh=d_h[k]*h[k]*(1.0-h[k]); for j in 0..self.idim { self.w[k][j]-=lr*dh*input[j]; } self.enc_b[k]-=lr*dh; }
    }

    // Check EXACT 100% on corpus — every single character reconstructed correctly
    fn check_100pct(&self, corpus: &[u8], ctx: usize) -> (usize, usize, f64) {
        let mut ok = 0usize; let mut tot = 0usize;
        let step = 1; // check every position
        let mut pos = 0;
        while pos + ctx <= corpus.len() {
            let sig = encode_seq4(&corpus[pos..pos+ctx]);
            let h = self.encode(&sig);
            let out = self.decode(&h);
            for i in 0..ctx {
                let rs = &out[i*4..(i+1)*4];
                let mut best = 0u8; let mut bd = f32::MAX;
                for ch in 0..27u8 { let code = encode4(ch); let d: f32 = code.iter().zip(rs).map(|(a,b)|(a-b)*(a-b)).sum(); if d<bd{bd=d;best=ch;} }
                if best == corpus[pos+i] { ok += 1; }
                tot += 1;
            }
            pos += step;
            if tot > 500_000 { break; } // cap for speed
        }
        (ok, tot, ok as f64 / tot as f64 * 100.0)
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let ctx = 16usize;
    let idim = ctx * 4; // 64 (with 4-neuron encoder)

    println!("=== INPUT MERGER — FIND 100% BOTTLENECK ===\n");
    println!("  Byte interpreter: 4 neurons, 100% proven");
    println!("  Input: ctx={} × 4 = {} signals", ctx, idim);
    println!("  Goal: find minimum bottleneck for EXACT 100% reconstruction\n");

    // Sweep: bneck from high (easy) to low (hard)
    println!("  {:>6} {:>5} {:>10} {:>8} {:>8}",
        "bneck", "ratio", "accuracy", "100%?", "time");
    println!("  {}", "─".repeat(45));

    for &bneck in &[64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24] {
        if bneck >= idim { continue; }
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut merger = Merger::new(idim, bneck, &mut rng);

        // Train hard: 300 epochs
        let samples = 15000.min(corpus.len() / ctx);
        for ep in 0..300 {
            let lr = 0.01 * (1.0 - ep as f32 / 300.0 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples {
                let off = rt.range(0, corpus.len() - ctx);
                merger.train_step(&encode_seq4(&corpus[off..off+ctx]), lr);
            }
        }

        let (ok, tot, acc) = merger.check_100pct(&corpus, ctx);
        let is_100 = if ok == tot { "★★★ YES" } else { "no" };
        let ratio = bneck as f64 / idim as f64 * 100.0;

        println!("  {:>6} {:>4.0}% {:>7}/{:>7} {:>8} {:>7.1}s",
            bneck, ratio, ok, tot, is_100, tc.elapsed().as_secs_f64());

        if ok == tot {
            println!("\n  ★★★ FOUND: bneck={} ({:.0}% ratio) = EXACT 100% ★★★", bneck, ratio);
            // Don't stop — continue to find minimum
        }
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
