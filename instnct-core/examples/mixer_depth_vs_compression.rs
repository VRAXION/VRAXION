//! Mixer depth vs compression: can deeper autoencoders hit 100% at lower bottleneck?
//!
//! Strategy: use the working 1L sigmoid AE as building block.
//! Stack multiple 1L autoencoders to compress progressively:
//!   Stage 1: 112 → 80 (sigmoid, tied)  → freeze
//!   Stage 2:  80 → 56 (sigmoid, tied)  → freeze
//!   Stage 3:  56 → 32 (sigmoid, tied)  → freeze
//! Each stage is simple and proven. The composition achieves deeper compression.
//!
//! Also test: single-layer with MORE training (200 epochs) at various bneck sizes.
//!
//! Run: cargo run --example mixer_depth_vs_compression --release

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

// Simple 1-layer tied AE (proven to work)
#[derive(Clone)]
struct AE1L {
    w: Vec<Vec<f32>>, enc_b: Vec<f32>, dec_b: Vec<f32>,
    idim: usize, bneck: usize,
}

impl AE1L {
    fn new(idim: usize, bneck: usize, rng: &mut Rng) -> Self {
        let s = (2.0/idim as f32).sqrt();
        AE1L { w: (0..bneck).map(|_| (0..idim).map(|_| rng.normal()*s).collect()).collect(),
               enc_b: vec![0.0;bneck], dec_b: vec![0.0;idim], idim, bneck }
    }
    fn encode(&self, input: &[f32]) -> Vec<f32> {
        let mut h=vec![0.0f32;self.bneck];
        for k in 0..self.bneck { h[k]=self.enc_b[k]; for j in 0..self.idim { h[k]+=self.w[k][j]*input[j]; } h[k]=sigmoid(h[k]); }
        h
    }
    fn decode(&self, h: &[f32]) -> Vec<f32> {
        let mut o=vec![0.0f32;self.idim];
        for j in 0..self.idim { o[j]=self.dec_b[j]; for k in 0..self.bneck { o[j]+=self.w[k][j]*h[k]; } }
        o
    }
    fn train_step(&mut self, input: &[f32], lr: f32) {
        let h = self.encode(input);
        let o = self.decode(&h);
        let mut d_o=vec![0.0f32;self.idim];
        for j in 0..self.idim { d_o[j]=2.0*(o[j]-input[j])/self.idim as f32; }
        let mut d_h=vec![0.0f32;self.bneck];
        for j in 0..self.idim { for k in 0..self.bneck { d_h[k]+=d_o[j]*self.w[k][j]; self.w[k][j]-=lr*d_o[j]*h[k]; } self.dec_b[j]-=lr*d_o[j]; }
        for k in 0..self.bneck { let dh=d_h[k]*h[k]*(1.0-h[k]); for j in 0..self.idim { self.w[k][j]-=lr*dh*input[j]; } self.enc_b[k]-=lr*dh; }
    }
}

fn eval_char_acc(pp: &Preproc, encode_fn: &dyn Fn(&[f32])->Vec<f32>, decode_fn: &dyn Fn(&[f32])->Vec<f32>,
                 corpus: &[u8], ctx: usize, n: usize, seed: u64) -> f64 {
    let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
    for _ in 0..n { if corpus.len()<ctx{break;} let off=rng.range(0,corpus.len()-ctx);
        let sig=pp.encode_seq(&corpus[off..off+ctx]);
        let h=encode_fn(&sig); let out=decode_fn(&h);
        for i in 0..ctx { let rs=&out[i*7..(i+1)*7];
            let mut best=0u8; let mut bd=f32::MAX;
            for ch in 0..27u8 { let code=pp.encode(ch); let d:f32=code.iter().zip(rs).map(|(a,b)|(a-b)*(a-b)).sum(); if d<bd{bd=d;best=ch;} }
            if best==corpus[off+i]{ok+=1;} tot+=1; } }
    ok as f64/tot as f64*100.0
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let pp = Preproc::new();
    let ctx = 16usize;
    let idim = ctx * 7;

    println!("=== DEPTH vs COMPRESSION — ctx={} (input={}) ===\n", ctx, idim);

    // ── PART 1: Single layer with MORE training ──
    println!("━━━ PART 1: Single layer, 200 epochs (find true 100% threshold) ━━━\n");
    println!("  {:>6} {:>6} {:>8} {:>8}", "bneck", "ratio", "char%", "time");
    println!("  {}", "─".repeat(35));

    for &bneck in &[32, 48, 56, 64, 72, 80, 96, 112] {
        if bneck >= idim { continue; }
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut ae = AE1L::new(idim, bneck, &mut rng);
        for ep in 0..200 { let lr=0.01*(1.0-ep as f32/200.0*0.8); let mut rt=Rng::new(ep as u64*1000+42);
            for _ in 0..8000 { if corpus.len()<ctx{break;} let off=rt.range(0,corpus.len()-ctx); ae.train_step(&pp.encode_seq(&corpus[off..off+ctx]),lr); } }
        let acc = eval_char_acc(&pp, &|x| ae.encode(x), &|h| ae.decode(h), &corpus, ctx, 3000, 999);
        let m = if acc>99.5{"★★★"} else if acc>95.0{"★★"} else if acc>90.0{"★"} else {""};
        println!("  {:>6} {:>5.0}% {:>7.1}% {:>7.1}s {}", bneck, bneck as f64/idim as f64*100.0, acc, tc.elapsed().as_secs_f64(), m);
    }

    // ── PART 2: Stacked progressive compression ──
    println!("\n━━━ PART 2: Stacked (112→80→56→32) — each stage trained separately ━━━\n");

    let stages: Vec<(usize,usize)> = vec![(112,80), (80,56), (56,32)];
    let mut stack: Vec<AE1L> = Vec::new();

    for (si, &(from, to)) in stages.iter().enumerate() {
        let tc = Instant::now();
        let mut rng = Rng::new(42 + si as u64 * 100);
        let mut ae = AE1L::new(from, to, &mut rng);

        // Train this stage
        for ep in 0..150 { let lr=0.01*(1.0-ep as f32/150.0*0.8); let mut rt=Rng::new(ep as u64*1000+42);
            for _ in 0..8000 { if corpus.len()<ctx{break;} let off=rt.range(0,corpus.len()-ctx);
                // Encode through all previous stages first
                let mut sig = pp.encode_seq(&corpus[off..off+ctx]);
                for prev_ae in &stack { sig = prev_ae.encode(&sig); }
                ae.train_step(&sig, lr);
            }
        }

        // Eval this stage alone
        let stage_acc = {
            let stack_ref = &stack;
            let ae_ref = &ae;
            eval_char_acc(&pp,
                &|x| { let mut s=x.to_vec(); for a in stack_ref { s=a.encode(&s); } ae_ref.encode(&s) },
                &|h| { let mut s=ae_ref.decode(h); for a in stack_ref.iter().rev() { s=a.decode(&s); } s },
                &corpus, ctx, 3000, 999)
        };

        println!("  Stage {}: {}→{} ({:.0}% of orig) char_acc={:.1}% time={:.1}s",
            si+1, from, to, to as f64/idim as f64*100.0, stage_acc, tc.elapsed().as_secs_f64());

        stack.push(ae);
    }

    // Final: full stack compression
    let final_acc = {
        let stack_ref = &stack;
        eval_char_acc(&pp,
            &|x| { let mut s=x.to_vec(); for a in stack_ref { s=a.encode(&s); } s },
            &|h| { let mut s=h.to_vec(); for a in stack_ref.iter().rev() { s=a.decode(&s); } s },
            &corpus, ctx, 3000, 999)
    };
    println!("\n  Full stack 112→80→56→32: {:.1}% at {:.0}% compression (32/{} = {:.0}%)",
        final_acc, (1.0 - 32.0/idim as f64)*100.0, idim, 32.0/idim as f64*100.0);

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
