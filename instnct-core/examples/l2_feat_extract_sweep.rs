//! L2 Feature Extractor — activation sweep on masked char prediction
//!
//! Input: merger LUT output (1024 pairs × 2 int8 per chunk)
//! Task: mask 1 pair, predict both chars from bidirectional context
//! Sweep: activation functions × model sizes
//! Goal: 100% = features perfectly capture context patterns
//!
//! Run: cargo run --example l2_feat_extract_sweep --release

use std::time::Instant;

// ── L0+L1 LUT (merged byte pairs) ──
const LUT_2N: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];
const MW: [[i8;4];2] = [[-8,12,-7,-12],[-8,-6,-1,2]];
const MB: [i8;2] = [-1, 10];

fn merger_encode(a: u8, b: u8) -> [f32; 2] {
    let i = [LUT_2N[a as usize][0] as f32, LUT_2N[a as usize][1] as f32,
             LUT_2N[b as usize][0] as f32, LUT_2N[b as usize][1] as f32];
    let mut o = [0.0f32; 2];
    for k in 0..2 {
        o[k] = MB[k] as f32 + MW[k][0] as f32*i[0] + MW[k][1] as f32*i[1]
            + MW[k][2] as f32*i[2] + MW[k][3] as f32*i[3];
    }
    o
}

// ── RNG ──
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1)) }
    fn next(&mut self) -> u64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.0 }
    fn normal(&mut self) -> f32 {
        let u1 = (((self.next()>>33)%65536) as f32/65536.0).max(1e-7);
        let u2 = ((self.next()>>33)%65536) as f32/65536.0;
        (-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        if hi<=lo{lo}else{lo+(self.next() as usize%(hi-lo))}
    }
}

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = std::fs::read(path).expect("read corpus");
    raw.iter().filter_map(|&b| match b {
        b'a'..=b'z' => Some(b - b'a'),
        b'A'..=b'Z' => Some(b - b'A'),
        b' '|b'\n'|b'\t'|b'\r' => Some(26),
        _ => None,
    }).collect()
}

// ── Activation functions ──
fn act_relu(x: f32) -> f32 { x.max(0.0) }
fn act_lrelu(x: f32) -> f32 { if x > 0.0 { x } else { 0.01 * x } }
fn act_gelu(x: f32) -> f32 { x * 0.5 * (1.0 + (0.7978846 * (x + 0.044715 * x * x * x)).tanh()) }
fn act_swish(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn act_tanh(x: f32) -> f32 { x.tanh() }
fn act_c19(x: f32, c: f32) -> f32 {
    let c=c.max(0.1); let l=6.0*c;
    if x>=l{return x-l;} if x<=-l{return x+l;}
    let s=x/c; let n=s.floor(); let t=s-n; let h=t*(1.0-t);
    let sg=if(n as i32)%2==0{1.0}else{-1.0}; c*sg*h
}

fn apply_act(name: &str, x: f32) -> f32 {
    match name {
        "relu" => act_relu(x),
        "lrelu" => act_lrelu(x),
        "gelu" => act_gelu(x),
        "swish" => act_swish(x),
        "tanh" => act_tanh(x),
        "c19" => act_c19(x, 10.0),
        _ => act_relu(x),
    }
}

fn act_grad(name: &str, x: f32, y: f32) -> f32 {
    match name {
        "relu" => if x > 0.0 { 1.0 } else { 0.0 },
        "lrelu" => if x > 0.0 { 1.0 } else { 0.01 },
        "gelu" => {
            let s = 0.7978846 * (x + 0.044715*x*x*x);
            let t = s.tanh();
            0.5*(1.0+t) + x*0.5*(1.0-t*t)*0.7978846*(1.0+3.0*0.044715*x*x)
        },
        "swish" => { let s = 1.0/(1.0+(-x).exp()); s + x*s*(1.0-s) },
        "tanh" => 1.0 - y*y,
        "c19" => {
            let c=10.0; let l=6.0*c;
            if x>=l||x<=-l{1.0}else{
                let s=x/c;let n=s.floor();let t=s-n;
                let sg=if(n as i32)%2==0{1.0}else{-1.0};sg*(1.0-2.0*t)}
        },
        _ => if x > 0.0 { 1.0 } else { 0.0 },
    }
}

// ── Model ──
struct Model {
    conv_w: Vec<Vec<f32>>, conv_b: Vec<f32>,
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    wa: Vec<Vec<f32>>, ba: Vec<f32>, // head A: predict char_a (27)
    wb: Vec<Vec<f32>>, bb: Vec<f32>, // head B: predict char_b (27)
    k: usize, nf: usize, ch: usize, ctx: usize, hdim: usize, head_dim: usize,
    act: String,
}

impl Model {
    fn new(ctx: usize, ch: usize, k: usize, nf: usize, hdim: usize, act: &str, rng: &mut Rng) -> Self {
        let fi = k*ch; let nc = ctx-k+1; let hd = nc*nf;
        let sc=(2.0/fi as f32).sqrt(); let s1=(2.0/hd as f32).sqrt(); let s2=(2.0/hdim as f32).sqrt();
        Model {
            conv_w: (0..nf).map(|_|(0..fi).map(|_|rng.normal()*sc).collect()).collect(),
            conv_b: vec![0.0;nf],
            w1: (0..hdim).map(|_|(0..hd).map(|_|rng.normal()*s1).collect()).collect(),
            b1: vec![0.0;hdim],
            wa: (0..27).map(|_|(0..hdim).map(|_|rng.normal()*s2).collect()).collect(),
            ba: vec![0.0;27],
            wb: (0..27).map(|_|(0..hdim).map(|_|rng.normal()*s2).collect()).collect(),
            bb: vec![0.0;27],
            k,nf,ch,ctx,hdim,head_dim:hd,act:act.to_string(),
        }
    }

    fn params(&self)->usize {
        self.nf*self.k*self.ch+self.nf + self.head_dim*self.hdim+self.hdim + self.hdim*27*2+27*2
    }

    fn forward(&self, input: &[f32]) -> (Vec<f32>,Vec<f32>,Vec<f32>,Vec<f32>,Vec<f32>,Vec<f32>) {
        let nc=self.ctx-self.k+1;
        let mut co=vec![0.0f32;nc*self.nf];
        let mut co_pre=vec![0.0f32;nc*self.nf];
        for p in 0..nc{for f in 0..self.nf{
            let mut v=self.conv_b[f];
            for ki in 0..self.k{for d in 0..self.ch{
                v+=self.conv_w[f][ki*self.ch+d]*input[(p+ki)*self.ch+d];}}
            co_pre[p*self.nf+f]=v;
            co[p*self.nf+f]=apply_act(&self.act, v);
        }}
        let mut h=vec![0.0f32;self.hdim];
        let mut h_pre=vec![0.0f32;self.hdim];
        for i in 0..self.hdim{
            h_pre[i]=self.b1[i];
            for j in 0..self.head_dim{h_pre[i]+=self.w1[i][j]*co[j];}
            h[i]=apply_act(&self.act, h_pre[i]);
        }
        let mut la=vec![0.0f32;27]; let mut lb=vec![0.0f32;27];
        for c in 0..27{la[c]=self.ba[c]; lb[c]=self.bb[c];
            for i in 0..self.hdim{la[c]+=self.wa[c][i]*h[i]; lb[c]+=self.wb[c][i]*h[i];}}
        (co, co_pre, h, h_pre, la, lb)
    }

    fn train_step(&mut self, input: &[f32], tgt_a: u8, tgt_b: u8, lr: f32) {
        let nc=self.ctx-self.k+1;
        let (co, co_pre, h, h_pre, la, lb) = self.forward(input);

        // Softmax+CE for both heads
        let softmax = |logits: &[f32]| -> Vec<f32> {
            let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let mut p=vec![0.0f32;27];let mut s=0.0f32;
            for c in 0..27{p[c]=(logits[c]-mx).exp();s+=p[c];}
            for c in 0..27{p[c]/=s;} p
        };
        let mut da=softmax(&la); da[tgt_a as usize]-=1.0;
        let mut db=softmax(&lb); db[tgt_b as usize]-=1.0;

        // Backprop heads → dh
        let mut dh=vec![0.0f32;self.hdim];
        for c in 0..27{
            for i in 0..self.hdim{
                dh[i]+=da[c]*self.wa[c][i]+db[c]*self.wb[c][i];
                self.wa[c][i]-=lr*da[c]*h[i];
                self.wb[c][i]-=lr*db[c]*h[i];
            }
            self.ba[c]-=lr*da[c]; self.bb[c]-=lr*db[c];
        }

        // Backprop dense1
        let mut dco=vec![0.0f32;self.head_dim];
        for i in 0..self.hdim{
            let g=act_grad(&self.act, h_pre[i], h[i]);
            let dhi=dh[i]*g;
            for j in 0..self.head_dim{dco[j]+=dhi*self.w1[i][j]; self.w1[i][j]-=lr*dhi*co[j];}
            self.b1[i]-=lr*dhi;
        }

        // Backprop conv
        for p in 0..nc{for f in 0..self.nf{
            let idx=p*self.nf+f;
            let g=act_grad(&self.act, co_pre[idx], co[idx]);
            let dc=dco[idx]*g;
            if dc==0.0{continue;}
            for ki in 0..self.k{for d in 0..self.ch{
                self.conv_w[f][ki*self.ch+d]-=lr*dc*input[(p+ki)*self.ch+d];}}
            self.conv_b[f]-=lr*dc;
        }}
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split = corpus.len() * 80 / 100;

    // Pad to even
    let mut corp = corpus.clone();
    if corp.len() % 2 != 0 { corp.push(26); }

    // Pre-encode entire corpus with merger, scale to ~int4 range (÷16)
    let merged: Vec<[f32;2]> = corp.chunks(2).map(|p| {
        let r = merger_encode(p[0], if p.len()>1{p[1]}else{26});
        [r[0] / 16.0, r[1] / 16.0]
    }).collect();
    let mut mn=[f32::MAX;2]; let mut mx=[f32::MIN;2];
    for r in &merged { for k in 0..2 { if r[k]<mn[k]{mn[k]=r[k];} if r[k]>mx[k]{mx[k]=r[k];} }}
    println!("  Scaled range: [{:.1},{:.1}] x [{:.1},{:.1}] (÷16, ~int4)", mn[0],mx[0],mn[1],mx[1]);

    let n_pairs = merged.len();
    let split_p = split / 2;

    println!("=== L2 FEATURE EXTRACTOR — ACTIVATION SWEEP ===\n");
    println!("  Corpus: {} chars → {} merged pairs ({} train, {} test)",
        corp.len(), n_pairs, split_p, n_pairs - split_p);
    println!("  Task: mask 1 pair, predict both chars (bidirectional)");
    println!("  Goal: 100% = perfect feature extraction\n");

    let mask_val = [15.0f32, 15.0]; // outside ±8 data range — clearly visible as mask

    let activations = ["relu", "lrelu", "gelu", "swish", "tanh", "c19"];

    struct Cfg { ctx: usize, k: usize, nf: usize, hdim: usize, epochs: usize }
    let configs = vec![
        Cfg { ctx: 16, k: 3, nf: 32, hdim: 64, epochs: 400 },
        Cfg { ctx: 32, k: 3, nf: 32, hdim: 64, epochs: 300 },
    ];

    println!("  {:>6} {:>4} {:>4} {:>5} {:>7} {:>8} {:>10} {:>10} {:>7}",
        "act", "ctx", "k", "f", "h", "params", "train%", "test%", "time");
    println!("  {}", "-".repeat(72));

    for cfg in &configs {
        for &act_name in &activations {
            let tc = Instant::now();
            let mut rng = Rng::new(42);
            let ch = 2;
            let mut model = Model::new(cfg.ctx, ch, cfg.k, cfg.nf, cfg.hdim, act_name, &mut rng);

            let samples = 20000.min(split_p.saturating_sub(cfg.ctx + 1));

            for ep in 0..cfg.epochs {
                let lr = 0.01 * (1.0 - ep as f32 / cfg.epochs as f32 * 0.8);
                let mut rt = Rng::new(ep as u64 * 1000 + 42);

                for _ in 0..samples {
                    let off = rt.range(0, split_p.saturating_sub(cfg.ctx + 1));
                    let mask_pos = cfg.ctx / 2; // always center

                    // Build input with mask
                    let mut input = Vec::with_capacity(cfg.ctx * ch);
                    for i in 0..cfg.ctx {
                        if i == mask_pos {
                            input.extend_from_slice(&mask_val);
                        } else {
                            input.push(merged[off + i][0]);
                            input.push(merged[off + i][1]);
                        }
                    }

                    // Targets: the two chars at masked position
                    let pair_idx = off + mask_pos;
                    let tgt_a = corp[pair_idx * 2];
                    let tgt_b = corp[pair_idx * 2 + 1];

                    model.train_step(&input, tgt_a, tgt_b, lr);
                }

                if tc.elapsed().as_secs() > 90 { break; }
            }

            // Eval
            let eval = |start_p: usize, end_p: usize| -> (f64, f64) {
                let mut rng3 = Rng::new(999);
                let mut ok_both = 0usize;
                let mut ok_any = 0usize;
                let mut tot = 0usize;
                let n = 5000.min(end_p.saturating_sub(start_p + cfg.ctx + 1));
                for _ in 0..n {
                    if end_p < start_p + cfg.ctx + 1 { break; }
                    let off = rng3.range(start_p, end_p - cfg.ctx - 1);
                    let mask_pos = cfg.ctx / 2;

                    let mut input = Vec::with_capacity(cfg.ctx * ch);
                    for i in 0..cfg.ctx {
                        if i == mask_pos { input.extend_from_slice(&mask_val); }
                        else { input.push(merged[off+i][0]); input.push(merged[off+i][1]); }
                    }

                    let (_, _, _, _, la, lb) = model.forward(&input);
                    let pa = la.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|x|x.0).unwrap_or(0);
                    let pb = lb.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|x|x.0).unwrap_or(0);

                    let pi = off + mask_pos;
                    let ta = corp[pi*2] as usize;
                    let tb = corp[pi*2+1] as usize;
                    if pa==ta && pb==tb { ok_both+=1; }
                    if pa==ta || pb==tb { ok_any+=1; }
                    tot+=1;
                }
                if tot==0{(0.0,0.0)}else{
                    (ok_both as f64/tot as f64*100.0, ok_any as f64/tot as f64*100.0)
                }
            };

            let (tr_both, _) = eval(0, split_p);
            let (te_both, _) = eval(split_p, n_pairs);
            let m = if te_both>80.0{" ***"}else if te_both>50.0{" **"}else if te_both>30.0{" *"}else{""};

            println!("  {:>6} {:>4} {:>4} {:>5} {:>7} {:>8} {:>9.1}% {:>9.1}% {:>6.1}s{}",
                act_name, cfg.ctx, cfg.k, cfg.nf, cfg.hdim, model.params(),
                tr_both, te_both, tc.elapsed().as_secs_f64(), m);
        }
        println!();
    }

    println!("  Baseline: random = {:.2}% (both correct)", 100.0/27.0/27.0);
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
