//! Optimizer Sweep: norm_sgd vs momentum vs Adam vs AdamW
//! + Data scaling: same optimizer on 2K / 10K / 50K training samples
//!
//! Architecture: 384S+72D K=8 (sweet spot from width sweep)
//!
//! Run: cargo run --example optimizer_sweep --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use std::time::Instant;

const CTX: usize = 4;
const N_BYTES: usize = 128;

fn c19_fwd(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01); let l = 6.0 * c;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let s = x / c; let n = s.floor(); let t = s - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}
fn c19_deriv(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01); let l = 6.0 * c;
    if x >= l || x <= -l { return 1.0; }
    let s = x / c; let n = s.floor(); let t = s - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    (sgn + 2.0 * rho * h) * (1.0 - 2.0 * t)
}
fn softmax(logits: &[f32]) -> Vec<f32> {
    let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / s).collect()
}
fn encode_ctx(ctx: &[u8], byte_idx: usize) -> [f32; CTX] {
    let mut v = [0.0f32; CTX];
    for (p, &b) in ctx.iter().enumerate() { if (b as usize) == byte_idx { v[p] = 1.0; } }
    v
}

#[derive(Clone)]
struct Net {
    ns: usize, nd: usize, k: usize,
    sp_w: Vec<f32>, sp_bias: Vec<f32>, sp_c: Vec<f32>, sp_rho: Vec<f32>,
    sp_byte: Vec<usize>,
    dn_sp: Vec<f32>, dn_bias: Vec<f32>, dn_c: Vec<f32>, dn_rho: Vec<f32>,
    dn_reads_sp: Vec<Vec<usize>>,
    out_w: Vec<f32>, out_bias: Vec<f32>,
}

struct Cache { sp_sum: Vec<f32>, sp_act: Vec<f32>, dn_sum: Vec<f32>, dn_act: Vec<f32>, probs: Vec<f32> }

struct Grad {
    sp_w: Vec<f32>, sp_bias: Vec<f32>, sp_rho: Vec<f32>,
    dn_sp: Vec<f32>, dn_bias: Vec<f32>, dn_rho: Vec<f32>,
    out_w: Vec<f32>, out_bias: Vec<f32>,
}
impl Grad {
    fn zeros(ns: usize, nd: usize, k: usize) -> Self {
        Grad { sp_w: vec![0.0;ns*CTX], sp_bias: vec![0.0;ns], sp_rho: vec![0.0;ns],
               dn_sp: vec![0.0;nd*k], dn_bias: vec![0.0;nd], dn_rho: vec![0.0;nd],
               out_w: vec![0.0;N_BYTES*nd], out_bias: vec![0.0;N_BYTES] }
    }
    fn add(&mut self, o: &Grad) {
        for (a,b) in self.sp_w.iter_mut().zip(&o.sp_w){*a+=b;}
        for (a,b) in self.sp_bias.iter_mut().zip(&o.sp_bias){*a+=b;}
        for (a,b) in self.sp_rho.iter_mut().zip(&o.sp_rho){*a+=b;}
        for (a,b) in self.dn_sp.iter_mut().zip(&o.dn_sp){*a+=b;}
        for (a,b) in self.dn_bias.iter_mut().zip(&o.dn_bias){*a+=b;}
        for (a,b) in self.dn_rho.iter_mut().zip(&o.dn_rho){*a+=b;}
        for (a,b) in self.out_w.iter_mut().zip(&o.out_w){*a+=b;}
        for (a,b) in self.out_bias.iter_mut().zip(&o.out_bias){*a+=b;}
    }
    fn scale(&mut self, s: f32) {
        for v in &mut self.sp_w{*v*=s;} for v in &mut self.sp_bias{*v*=s;}
        for v in &mut self.sp_rho{*v*=s;} for v in &mut self.dn_sp{*v*=s;}
        for v in &mut self.dn_bias{*v*=s;} for v in &mut self.dn_rho{*v*=s;}
        for v in &mut self.out_w{*v*=s;} for v in &mut self.out_bias{*v*=s;}
    }
    fn norm(&self) -> f32 {
        let mut s=0.0f32;
        for v in &self.sp_w{s+=v*v;} for v in &self.sp_bias{s+=v*v;}
        for v in &self.sp_rho{s+=v*v;} for v in &self.dn_sp{s+=v*v;}
        for v in &self.dn_bias{s+=v*v;} for v in &self.dn_rho{s+=v*v;}
        for v in &self.out_w{s+=v*v;} for v in &self.out_bias{s+=v*v;}
        s.sqrt()
    }
    fn to_vec(&self) -> Vec<f32> {
        let mut v = Vec::new();
        v.extend(&self.sp_w); v.extend(&self.sp_bias); v.extend(&self.sp_rho);
        v.extend(&self.dn_sp); v.extend(&self.dn_bias); v.extend(&self.dn_rho);
        v.extend(&self.out_w); v.extend(&self.out_bias);
        v
    }
}

impl Net {
    fn new(ns: usize, nd: usize, k: usize, seed: u64, rng: &mut StdRng) -> Self {
        let sc = 0.1f32; let k = k.min(ns);
        let sp_byte: Vec<usize> = (0..ns).map(|i| i % N_BYTES).collect();
        let dn_reads_sp: Vec<Vec<usize>> = (0..nd).map(|i| {
            let mut idx: Vec<usize> = (0..ns).collect();
            let mut r = StdRng::seed_from_u64(seed + i as u64*37+13);
            idx.shuffle(&mut r); idx[..k].to_vec()
        }).collect();
        Net { ns, nd, k, sp_byte,
            sp_w: (0..ns*CTX).map(|_| rng.gen_range(-sc..sc)).collect(),
            sp_bias: (0..ns).map(|_| rng.gen_range(-sc..sc)).collect(),
            sp_c: vec![1.0;ns], sp_rho: vec![4.0;ns],
            dn_sp: (0..nd*k).map(|_| rng.gen_range(-sc..sc)).collect(),
            dn_bias: (0..nd).map(|_| rng.gen_range(-sc..sc)).collect(),
            dn_c: vec![1.0;nd], dn_rho: vec![4.0;nd], dn_reads_sp,
            out_w: (0..N_BYTES*nd).map(|_| rng.gen_range(-sc..sc)).collect(),
            out_bias: (0..N_BYTES).map(|_| rng.gen_range(-sc..sc)).collect(),
        }
    }

    fn forward(&self, ctx: &[u8]) -> Cache {
        let (ns,nd,k) = (self.ns, self.nd, self.k);
        let mut sp_sum = vec![0.0f32;ns]; let mut sp_act = vec![0.0f32;ns];
        for i in 0..ns {
            let mut s = self.sp_bias[i];
            let th = encode_ctx(ctx, self.sp_byte[i]);
            for j in 0..CTX { s += th[j] * self.sp_w[i*CTX+j]; }
            sp_sum[i] = s; sp_act[i] = c19_fwd(s, self.sp_c[i], self.sp_rho[i]);
        }
        let mut dn_sum = vec![0.0f32;nd]; let mut dn_act = vec![0.0f32;nd];
        for i in 0..nd {
            let mut s = self.dn_bias[i];
            for (ki,&si) in self.dn_reads_sp[i].iter().enumerate() {
                s += sp_act[si] * self.dn_sp[i*k+ki];
            }
            dn_sum[i] = s; dn_act[i] = c19_fwd(s, self.dn_c[i], self.dn_rho[i]);
        }
        let mut logits = vec![0.0f32;N_BYTES];
        for b in 0..N_BYTES {
            let mut s = self.out_bias[b];
            for d in 0..nd { s += dn_act[d] * self.out_w[b*nd+d]; }
            logits[b] = s;
        }
        Cache { sp_sum, sp_act, dn_sum, dn_act, probs: softmax(&logits) }
    }

    fn backward(&self, c: &Cache, ctx: &[u8], target: u8) -> Grad {
        let (ns,nd,k) = (self.ns, self.nd, self.k);
        let mut g = Grad::zeros(ns, nd, k);
        let mut dl = c.probs.clone(); dl[target as usize] -= 1.0;
        let mut d_dn = vec![0.0f32;nd];
        for b in 0..N_BYTES {
            g.out_bias[b] = dl[b];
            for d in 0..nd { g.out_w[b*nd+d] = dl[b]*c.dn_act[d]; d_dn[d] += dl[b]*self.out_w[b*nd+d]; }
        }
        let mut d_sp = vec![0.0f32;ns];
        for i in 0..nd {
            let dv = c19_deriv(c.dn_sum[i], self.dn_c[i], self.dn_rho[i]);
            let ds = d_dn[i]*dv; g.dn_bias[i] = ds;
            for (ki,&si) in self.dn_reads_sp[i].iter().enumerate() {
                g.dn_sp[i*k+ki] = ds*c.sp_act[si]; d_sp[si] += ds*self.dn_sp[i*k+ki];
            }
            let s=c.dn_sum[i]; let cv=self.dn_c[i].max(0.01); let l=6.0*cv;
            if s>-l&&s<l { let sc=s/cv; let ft=sc-sc.floor(); let h=ft*(1.0-ft); g.dn_rho[i]=d_dn[i]*cv*h*h; }
        }
        for i in 0..ns {
            let dv = c19_deriv(c.sp_sum[i], self.sp_c[i], self.sp_rho[i]);
            let ds = d_sp[i]*dv; g.sp_bias[i] = ds;
            let th = encode_ctx(ctx, self.sp_byte[i]);
            for j in 0..CTX { g.sp_w[i*CTX+j] = ds*th[j]; }
            let s=c.sp_sum[i]; let cv=self.sp_c[i].max(0.01); let l=6.0*cv;
            if s>-l&&s<l { let sc=s/cv; let ft=sc-sc.floor(); let h=ft*(1.0-ft); g.sp_rho[i]=d_sp[i]*cv*h*h; }
        }
        g
    }

    fn params(&self) -> usize {
        self.sp_w.len()+self.sp_bias.len()+self.sp_rho.len()
        +self.dn_sp.len()+self.dn_bias.len()+self.dn_rho.len()
        +self.out_w.len()+self.out_bias.len()
    }

    fn param_vec(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.params());
        v.extend(&self.sp_w); v.extend(&self.sp_bias); v.extend(&self.sp_rho);
        v.extend(&self.dn_sp); v.extend(&self.dn_bias); v.extend(&self.dn_rho);
        v.extend(&self.out_w); v.extend(&self.out_bias);
        v
    }

    fn set_param_vec(&mut self, v: &[f32]) {
        let mut o = 0;
        let n=self.sp_w.len(); self.sp_w.copy_from_slice(&v[o..o+n]); o+=n;
        let n=self.sp_bias.len(); self.sp_bias.copy_from_slice(&v[o..o+n]); o+=n;
        let n=self.sp_rho.len(); self.sp_rho.copy_from_slice(&v[o..o+n]); o+=n;
        let n=self.dn_sp.len(); self.dn_sp.copy_from_slice(&v[o..o+n]); o+=n;
        let n=self.dn_bias.len(); self.dn_bias.copy_from_slice(&v[o..o+n]); o+=n;
        let n=self.dn_rho.len(); self.dn_rho.copy_from_slice(&v[o..o+n]); o+=n;
        let n=self.out_w.len(); self.out_w.copy_from_slice(&v[o..o+n]); o+=n;
        let n=self.out_bias.len(); self.out_bias.copy_from_slice(&v[o..o+n]); o+=n;
        // Clamp rho >= 0
        for r in &mut self.sp_rho { *r = r.max(0.0); }
        for r in &mut self.dn_rho { *r = r.max(0.0); }
    }

    fn accuracy(&self, data: &[(Vec<u8>,u8)]) -> f64 {
        data.iter().filter(|(c,t)| {
            let ca=self.forward(c);
            ca.probs.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap()).unwrap().0==*t as usize
        }).count() as f64/data.len() as f64
    }
    fn top5(&self, data: &[(Vec<u8>,u8)]) -> f64 {
        let mut ok=0;
        for (ctx,t) in data {
            let c=self.forward(ctx);
            let mut idx:Vec<(usize,f32)>=c.probs.iter().enumerate().map(|(i,&v)|(i,v)).collect();
            idx.sort_by(|a,b|b.1.partial_cmp(&a.1).unwrap());
            if idx.iter().take(5).any(|(i,_)|*i==*t as usize){ok+=1;}
        }
        ok as f64/data.len() as f64
    }
    fn loss_samples(&self, data: &[(Vec<u8>,u8)]) -> f64 {
        let mut l=0.0f64;
        for (c,t) in data { let ca=self.forward(c); l-=(ca.probs[*t as usize].max(1e-10) as f64).ln(); }
        l/data.len() as f64
    }
}

// ============================================================
// Training with different optimizers
// ============================================================

fn train(net: &mut Net, train_data: &[(Vec<u8>,u8)], test: &[(Vec<u8>,u8)],
         steps: usize, batch: usize, opt: &str, label: &str) {
    let np = net.params();
    let mut m = vec![0.0f32; np];
    let mut v = vec![0.0f32; np];
    let mut rng = StdRng::seed_from_u64(99);
    let mut sh = train_data.to_vec();
    let mut lr_cur = 0.01f32;
    let test_sub = if test.len() > 2000 { &test[..2000] } else { test };

    println!("  [{}: {} params, {} train, {} test, {} steps]",
             label, np, train_data.len(), test_sub.len(), steps);

    for step in 1..=steps {
        sh.shuffle(&mut rng);
        let b = &sh[..batch.min(sh.len())];
        let mut grad = Grad::zeros(net.ns, net.nd, net.k);
        for (ctx,t) in b { let c=net.forward(ctx); let g=net.backward(&c,ctx,*t); grad.add(&g); }
        grad.scale(1.0 / b.len() as f32);

        match opt {
            "norm_sgd" => {
                let gn = grad.norm();
                if gn < 1e-8 { continue; }
                grad.scale(1.0 / gn);
                let old_loss = net.loss_samples(b);
                let old = net.clone();
                let gv = grad.to_vec();
                let mut pv = net.param_vec();
                for i in 0..np { pv[i] -= lr_cur * gv[i]; }
                net.set_param_vec(&pv);
                if net.loss_samples(b) < old_loss { lr_cur *= 1.05; } else { *net = old; lr_cur *= 0.5; }
            },
            "sgd_mom" => {
                let gv = grad.to_vec();
                let mut pv = net.param_vec();
                for i in 0..np {
                    m[i] = 0.9 * m[i] + gv[i];
                    pv[i] -= 0.01 * m[i];
                }
                net.set_param_vec(&pv);
            },
            "adam" | "adam_wd" => {
                let lr = 0.001f32;
                let wd: f32 = if opt == "adam_wd" { 0.01 } else { 0.0 };
                let gv = grad.to_vec();
                let mut pv = net.param_vec();
                let t = step as f32;
                let b1c = 1.0 - 0.9f32.powf(t);
                let b2c = 1.0 - 0.999f32.powf(t);
                for i in 0..np {
                    m[i] = 0.9 * m[i] + 0.1 * gv[i];
                    v[i] = 0.999 * v[i] + 0.001 * gv[i] * gv[i];
                    let mh = m[i] / b1c;
                    let vh = v[i] / b2c;
                    pv[i] -= lr * (mh / (vh.sqrt() + 1e-8) + wd * pv[i]);
                }
                net.set_param_vec(&pv);
            },
            "adam_fast" => {
                let lr = 0.003f32;
                let gv = grad.to_vec();
                let mut pv = net.param_vec();
                let t = step as f32;
                let b1c = 1.0 - 0.9f32.powf(t);
                let b2c = 1.0 - 0.999f32.powf(t);
                for i in 0..np {
                    m[i] = 0.9 * m[i] + 0.1 * gv[i];
                    v[i] = 0.999 * v[i] + 0.001 * gv[i] * gv[i];
                    let mh = m[i] / b1c;
                    let vh = v[i] / b2c;
                    pv[i] -= lr * mh / (vh.sqrt() + 1e-8);
                }
                net.set_param_vec(&pv);
            },
            _ => panic!("unknown optimizer: {}", opt),
        }

        if step % 500 == 0 || step == steps {
            let tr = net.accuracy(train_data);
            let te = net.accuracy(test_sub);
            let t5 = net.top5(test_sub);
            let lo = net.loss_samples(test_sub);
            println!("    step {:>5}: loss={:.3} train={:.1}% test={:.1}% top5={:.1}%",
                     step, lo, tr*100.0, te*100.0, t5*100.0);
        }
    }
}

fn bigram_baseline(tr: &[(Vec<u8>,u8)], te: &[(Vec<u8>,u8)]) -> f64 {
    let mut counts = vec![vec![0u32; N_BYTES]; N_BYTES];
    for (ctx,t) in tr {
        let last = *ctx.last().unwrap() as usize;
        if last < N_BYTES { counts[last][*t as usize] += 1; }
    }
    let predict = |l:usize| counts[l].iter().enumerate().max_by_key(|(_,&c)|c).map(|(i,_)|i).unwrap_or(32);
    te.iter().filter(|(c,t)| predict(*c.last().unwrap() as usize)==*t as usize).count() as f64/te.len() as f64
}

fn main() {
    println!("=== OPTIMIZER SWEEP: norm_sgd vs momentum vs Adam ===\n");
    let t0 = Instant::now();

    println!("Loading FineWeb text...");
    let raw: Vec<u8> = {
        let o = std::process::Command::new("python").arg("-c").arg(r#"
import pyarrow.parquet as pq
t = pq.read_table('S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/000_00000.parquet', columns=['text'])
c = t.column('text')
text = ''
for i in range(500):
    text += str(c[i]) + ' '
    if len(text) > 200000: break
import sys
sys.stdout.buffer.write(bytes([b for b in text.encode('ascii', errors='ignore') if 32 <= b < 127 or b == 10]))
"#).output();
        match o { Ok(o) if o.stdout.len()>1000 => { println!("  {} bytes",o.stdout.len()); o.stdout }
            _ => { println!("  Fallback"); "the quick brown fox jumps over the lazy dog ".repeat(500).bytes().collect() } }
    };
    let text: Vec<u8> = raw.iter().map(|&b| if b<128{b}else{32}).collect();
    let mut pairs: Vec<(Vec<u8>,u8)> = Vec::new();
    for i in CTX..text.len() {
        let ctx = text[i-CTX..i].to_vec();
        if (text[i] as usize) < N_BYTES { pairs.push((ctx, text[i])); }
    }
    let mut rng = StdRng::seed_from_u64(42);
    pairs.shuffle(&mut rng);
    println!("  Total pairs: {}\n", pairs.len());

    let test = pairs[60000..65000].to_vec();
    let train2k = pairs[..2000].to_vec();
    let train10k = pairs[..10000].to_vec();
    let train50k = pairs[..50000].to_vec();

    let bi = bigram_baseline(&train50k, &test);
    println!("  Bigram: {:.1}%\n", bi*100.0);

    let (ns, nd, k) = (384, 72, 8);
    let mut results: Vec<(String, f64, f64)> = Vec::new();

    // =========================================================
    // PHASE 1: Optimizer sweep on 2K data
    // =========================================================
    println!("========== PHASE 1: Optimizer sweep (2K data, 5K steps) ==========\n");

    for opt in &["norm_sgd", "sgd_mom", "adam", "adam_fast", "adam_wd"] {
        let label = format!("{} (2K)", opt);
        println!("--- {} ---", label);
        let mut net = Net::new(ns, nd, k, 42, &mut StdRng::seed_from_u64(77));
        let t1 = Instant::now();
        train(&mut net, &train2k, &test, 5000, 200, opt, &label);
        let acc = net.accuracy(&test[..2000]);
        let t5 = net.top5(&test[..2000]);
        println!("  => test={:.1}% top5={:.1}% ({:.1}s)\n", acc*100.0, t5*100.0, t1.elapsed().as_secs_f64());
        results.push((label, acc, t5));
    }

    // =========================================================
    // PHASE 2: Data scaling with best optimizer (Adam)
    // =========================================================
    println!("========== PHASE 2: Data scaling with Adam (10K steps) ==========\n");

    for (data, label_suffix) in &[
        (train2k.as_slice(), "2K"),
        (train10k.as_slice(), "10K"),
        (train50k.as_slice(), "50K"),
    ] {
        let label = format!("adam ({})", label_suffix);
        println!("--- {} ---", label);
        let mut net = Net::new(ns, nd, k, 42, &mut StdRng::seed_from_u64(77));
        let t1 = Instant::now();
        train(&mut net, data, &test, 10000, 200, "adam", &label);
        let acc = net.accuracy(&test[..2000]);
        let t5 = net.top5(&test[..2000]);
        println!("  => test={:.1}% top5={:.1}% ({:.1}s)\n", acc*100.0, t5*100.0, t1.elapsed().as_secs_f64());
        results.push((label, acc, t5));
    }

    // =========================================================
    // COMPARISON
    // =========================================================
    println!("=== FULL COMPARISON ===");
    println!("  {:>20} {:>8} {:>8}", "config", "test%", "top5%");
    println!("  {}", "=".repeat(40));
    for (label, acc, t5) in &results {
        println!("  {:>20} {:>7.1}% {:>7.1}%", label, acc*100.0, t5*100.0);
    }
    println!("  {:>20} {:>7.1}%", "bigram", bi*100.0);

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
    println!("\n=== DONE ===");
}
