//! Abstract Core v2 — Int8/binary quantization sweep + context size sweep
//!
//! Takes the best v1 config (2-layer 512→256) and tests:
//! 1. Quantization: float → int8 → int5 → int4 → binary
//! 2. Context sizes: 32, 64, 128, 256 chars
//! 3. C19 activation in mixer (instead of ReLU)
//!
//! Run: cargo run --example abstract_core_v2_quant --release

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
    let mut c = Vec::with_capacity(raw.len());
    for &b in &raw { match b { b'a'..=b'z' => c.push(b-b'a'), b'A'..=b'Z' => c.push(b-b'A'), b' '|b'\n'|b'\t'|b'\r' => c.push(26), _ => {} } }
    c
}

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0*c;
    if x >= l { return x-l; } if x <= -l { return x+l; }
    let s = x/c; let n = s.floor(); let t = s-n; let h = t*(1.0-t);
    let sg = if (n as i32)%2==0 { 1.0 } else { -1.0 }; c*(sg*h+rho*h*h)
}

struct FrozenPreprocessor { weights: [[i8;8];7], biases: [i8;7], c: [f32;7], rho: [f32;7] }
impl FrozenPreprocessor {
    fn new() -> Self {
        FrozenPreprocessor {
            weights: [[-1,1,1,-1,-1,1,1,-1],[1,-1,1,1,-1,-1,-1,-1],[-1,-1,1,-1,1,1,-1,-1],
                      [-1,-1,-1,1,-1,-1,1,-1],[-1,1,1,1,-1,-1,-1,-1],[1,1,-1,-1,-1,-1,-1,-1],
                      [-1,1,-1,1,1,1,-1,-1]],
            biases: [1,1,1,1,1,1,1],
            c: [10.0,10.0,10.0,10.0,10.0,10.0,10.0],
            rho: [2.0,0.0,0.0,0.0,0.0,0.0,0.0],
        }
    }
    fn encode(&self, ch: u8) -> [f32;7] {
        let mut bits = [0.0f32;8]; for i in 0..8 { bits[i] = ((ch>>i)&1) as f32; }
        let mut out = [0.0f32;7];
        for k in 0..7 { let mut d = self.biases[k] as f32; for j in 0..8 { d += self.weights[k][j] as f32 * bits[j]; } out[k] = c19(d, self.c[k], self.rho[k]); }
        out
    }
}

const N_CLASSES: usize = 27;
const PREPROC_OUT: usize = 7;

#[derive(Clone)]
struct Model {
    ctx: usize,
    mixer_in: usize,
    md1: usize, md2: usize,
    wm1: Vec<Vec<f32>>, bm1: Vec<f32>,
    wm2: Vec<Vec<f32>>, bm2: Vec<f32>,
    wo: Vec<Vec<f32>>, bo: Vec<f32>,
    use_c19_mixer: bool,
    c_m1: Vec<f32>, rho_m1: Vec<f32>,
    c_m2: Vec<f32>, rho_m2: Vec<f32>,
}

impl Model {
    fn new(ctx: usize, md1: usize, md2: usize, use_c19: bool, rng: &mut Rng) -> Self {
        let mi = ctx * PREPROC_OUT;
        let s1 = (2.0/mi as f32).sqrt(); let s2 = (2.0/md1 as f32).sqrt(); let so = (2.0/md2 as f32).sqrt();
        Model {
            ctx, mixer_in: mi, md1, md2,
            wm1: (0..md1).map(|_| (0..mi).map(|_| rng.normal()*s1).collect()).collect(),
            bm1: vec![0.0;md1],
            wm2: (0..md2).map(|_| (0..md1).map(|_| rng.normal()*s2).collect()).collect(),
            bm2: vec![0.0;md2],
            wo: (0..N_CLASSES).map(|_| (0..md2).map(|_| rng.normal()*so).collect()).collect(),
            bo: vec![0.0;N_CLASSES],
            use_c19_mixer: use_c19,
            c_m1: vec![3.0;md1], rho_m1: vec![1.0;md1],
            c_m2: vec![3.0;md2], rho_m2: vec![1.0;md2],
        }
    }

    fn act(&self, x: f32, c_val: f32, rho_val: f32) -> f32 {
        if self.use_c19_mixer { c19(x, c_val, rho_val) } else { x.max(0.0) }
    }

    fn forward(&self, signals: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut a1 = vec![0.0f32;self.md1];
        for i in 0..self.md1 { let mut s=self.bm1[i]; for j in 0..self.mixer_in { s+=self.wm1[i][j]*signals[j]; } a1[i]=self.act(s,self.c_m1[i],self.rho_m1[i]); }
        let mut a2 = vec![0.0f32;self.md2];
        for i in 0..self.md2 { let mut s=self.bm2[i]; for j in 0..self.md1 { s+=self.wm2[i][j]*a1[j]; } a2[i]=self.act(s,self.c_m2[i],self.rho_m2[i]); }
        let mut lo = vec![0.0f32;N_CLASSES];
        for i in 0..N_CLASSES { lo[i]=self.bo[i]; for j in 0..self.md2 { lo[i]+=self.wo[i][j]*a2[j]; } }
        (a1, a2, lo)
    }

    fn predict(&self, signals: &[f32]) -> usize {
        let (_,_,lo) = self.forward(signals);
        lo.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0)
    }

    fn train_step(&mut self, signals: &[f32], target: usize, lr: f32) -> f32 {
        let (a1,a2,lo) = self.forward(signals);
        let mx = lo.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = lo.iter().map(|&l| (l-mx).exp()).collect();
        let se: f32 = exp.iter().sum(); let sm: Vec<f32> = exp.iter().map(|&e| e/se).collect();
        let loss = -(sm[target].max(1e-7)).ln();
        let mut dl = sm; dl[target] -= 1.0;

        let mut da2 = vec![0.0f32;self.md2];
        for i in 0..N_CLASSES { for j in 0..self.md2 { da2[j]+=dl[i]*self.wo[i][j]; self.wo[i][j]-=lr*dl[i]*a2[j]; } self.bo[i]-=lr*dl[i]; }
        let mut d2 = vec![0.0f32;self.md2];
        for i in 0..self.md2 { d2[i] = if a2[i]>0.0 || self.use_c19_mixer { da2[i] } else { 0.0 }; }
        let mut da1 = vec![0.0f32;self.md1];
        for i in 0..self.md2 { for j in 0..self.md1 { da1[j]+=d2[i]*self.wm2[i][j]; self.wm2[i][j]-=lr*d2[i]*a1[j]; } self.bm2[i]-=lr*d2[i]; }
        let mut d1 = vec![0.0f32;self.md1];
        for i in 0..self.md1 { d1[i] = if a1[i]>0.0 || self.use_c19_mixer { da1[i] } else { 0.0 }; }
        for i in 0..self.md1 { for j in 0..self.mixer_in { self.wm1[i][j]-=lr*d1[i]*signals[j]; } self.bm1[i]-=lr*d1[i]; }
        loss
    }

    fn total_params(&self) -> usize {
        self.mixer_in*self.md1 + self.md1 + self.md1*self.md2 + self.md2 + self.md2*N_CLASSES + N_CLASSES
    }
}

fn make_signals(preproc: &FrozenPreprocessor, corpus: &[u8], off: usize, ctx: usize) -> Vec<f32> {
    let mut signals = vec![0.0f32; ctx * PREPROC_OUT];
    for i in 0..ctx {
        let enc = preproc.encode(corpus[off + i]);
        for k in 0..PREPROC_OUT { signals[i*PREPROC_OUT+k] = enc[k]; }
    }
    signals
}

fn eval_acc(preproc: &FrozenPreprocessor, model: &Model, corpus: &[u8], n: usize, seed: u64) -> f64 {
    let mut rng = Rng::new(seed); let mut ok=0usize; let mut tot=0;
    for _ in 0..n {
        if corpus.len() < model.ctx+1 { break; }
        let off = rng.range(0, corpus.len()-model.ctx-1);
        let signals = make_signals(preproc, corpus, off, model.ctx);
        if model.predict(&signals) == corpus[off+model.ctx] as usize { ok+=1; }
        tot+=1;
    }
    ok as f64/tot as f64*100.0
}

fn eval_quantized(preproc: &FrozenPreprocessor, model: &Model, corpus: &[u8], n: usize, seed: u64, bits: u32) -> f64 {
    let max_int: i32 = if bits==1 { 1 } else { (1i32<<(bits-1))-1 };
    let scale_mat = |m: &[Vec<f32>]| -> f32 { m.iter().flat_map(|r|r.iter()).map(|x|x.abs()).fold(0.0f32,f32::max).max(1e-7)/max_int as f32 };
    let scale_vec = |v: &[f32]| -> f32 { v.iter().map(|x|x.abs()).fold(0.0f32,f32::max).max(1e-7)/max_int as f32 };
    let q = |v:f32,s:f32| -> f32 { (v/s).round().max(-max_int as f32).min(max_int as f32)*s };

    let sw1=scale_mat(&model.wm1); let sw2=scale_mat(&model.wm2); let swo=scale_mat(&model.wo);
    let sb1=scale_vec(&model.bm1); let sb2=scale_vec(&model.bm2); let sbo=scale_vec(&model.bo);

    let mut rng = Rng::new(seed); let mut ok=0usize; let mut tot=0;
    for _ in 0..n {
        if corpus.len() < model.ctx+1 { break; }
        let off = rng.range(0, corpus.len()-model.ctx-1);
        let signals = make_signals(preproc, corpus, off, model.ctx);

        let mut a1 = vec![0.0f32;model.md1];
        for i in 0..model.md1 { let mut s=q(model.bm1[i],sb1); for j in 0..model.mixer_in { s+=q(model.wm1[i][j],sw1)*signals[j]; } a1[i]=s.max(0.0); }
        let mut a2 = vec![0.0f32;model.md2];
        for i in 0..model.md2 { let mut s=q(model.bm2[i],sb2); for j in 0..model.md1 { s+=q(model.wm2[i][j],sw2)*a1[j]; } a2[i]=s.max(0.0); }
        let mut lo = vec![0.0f32;N_CLASSES];
        for i in 0..N_CLASSES { lo[i]=q(model.bo[i],sbo); for j in 0..model.md2 { lo[i]+=q(model.wo[i][j],swo)*a2[j]; } }

        let pred = lo.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0);
        if pred == corpus[off+model.ctx] as usize { ok+=1; }
        tot+=1;
    }
    ok as f64/tot as f64*100.0
}

fn train_model(preproc: &FrozenPreprocessor, model: &mut Model, corpus: &[u8], epochs: usize, samples: usize, lr_init: f32) {
    for epoch in 0..epochs {
        let lr = lr_init * (1.0 - epoch as f32/epochs as f32 * 0.7);
        let mut rng = Rng::new(epoch as u64 * 1000 + 42);
        for _ in 0..samples {
            if corpus.len() < model.ctx+1 { break; }
            let off = rng.range(0, corpus.len()-model.ctx-1);
            let signals = make_signals(preproc, corpus, off, model.ctx);
            model.train_step(&signals, corpus[off+model.ctx] as usize, lr);
        }
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let preproc = FrozenPreprocessor::new();

    println!("=== ABSTRACT CORE v2 — Quantization + Context + C19 Sweep ===\n");

    // ── PART 1: Quantization sweep on best v1 config ──
    println!("━━━ PART 1: Quantization sweep (2-layer 512→256, ctx=128) ━━━\n");
    {
        let mut rng = Rng::new(42);
        let mut model = Model::new(128, 512, 256, false, &mut rng);
        train_model(&preproc, &mut model, &corpus, 30, 8000, 0.003);
        let float_acc = eval_acc(&preproc, &model, &corpus, 5000, 999);

        println!("  {:>8} {:>10} {:>10}", "quant", "accuracy", "vs float");
        println!("  {}", "─".repeat(35));
        println!("  {:>8} {:>9.1}% {:>10}", "float32", float_acc, "baseline");

        for &(label, bits) in &[("int8",8),("int6",6),("int5",5),("int4",4),("int3",3),("binary",1)] {
            let qacc = eval_quantized(&preproc, &model, &corpus, 5000, 999, bits);
            let delta = qacc - float_acc;
            println!("  {:>8} {:>9.1}% {:>+9.1}pp", label, qacc, delta);
        }
    }

    // ── PART 2: Context size sweep ──
    println!("\n━━━ PART 2: Context size sweep (2-layer 256→128, ReLU) ━━━\n");
    println!("  {:>6} {:>10} {:>10} {:>10} {:>10}", "ctx", "params", "float_acc", "int8_acc", "time");
    println!("  {}", "─".repeat(55));

    for &ctx in &[16, 32, 64, 128, 256] {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut model = Model::new(ctx, 256, 128, false, &mut rng);
        let epochs = if ctx <= 64 { 40 } else { 30 };
        train_model(&preproc, &mut model, &corpus, epochs, 8000, 0.003);
        let facc = eval_acc(&preproc, &model, &corpus, 5000, 999);
        let qacc = eval_quantized(&preproc, &model, &corpus, 5000, 999, 8);
        println!("  {:>6} {:>10} {:>9.1}% {:>9.1}% {:>8.1}s",
            ctx, model.total_params(), facc, qacc, tc.elapsed().as_secs_f64());
    }

    // ── PART 3: C19 mixer activation ──
    println!("\n━━━ PART 3: C19 vs ReLU mixer (2-layer 256→128, ctx=128) ━━━\n");
    println!("  {:>10} {:>10} {:>10} {:>10}", "activation", "float_acc", "int8_acc", "time");
    println!("  {}", "─".repeat(45));

    for &(label, use_c19) in &[("ReLU", false), ("C19", true)] {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut model = Model::new(128, 256, 128, use_c19, &mut rng);
        train_model(&preproc, &mut model, &corpus, 30, 8000, 0.003);
        let facc = eval_acc(&preproc, &model, &corpus, 5000, 999);
        let qacc = eval_quantized(&preproc, &model, &corpus, 5000, 999, 8);
        println!("  {:>10} {:>9.1}% {:>9.1}% {:>8.1}s",
            label, facc, qacc, tc.elapsed().as_secs_f64());
    }

    println!("\n  Baselines: frequency=20.3%, INSTNCT=24.6%");
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
