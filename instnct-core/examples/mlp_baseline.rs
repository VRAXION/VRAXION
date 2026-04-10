//! MLP Baseline: vanilla ReLU MLP vs C19 sandwich on same task
//!
//! The one experiment that actually matters:
//! Is our architecture better/worse/same as a standard MLP with same params?
//!
//! Run: cargo run --example mlp_baseline --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use std::time::Instant;

const CTX: usize = 4;
const N_BYTES: usize = 128;
const IN: usize = CTX * N_BYTES; // 512 one-hot

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

fn one_hot(ctx: &[u8]) -> Vec<f32> {
    let mut v = vec![0.0f32; IN];
    for (p, &b) in ctx.iter().enumerate() {
        if (b as usize) < N_BYTES { v[p * N_BYTES + b as usize] = 1.0; }
    }
    v
}

// ============================================================
// Generic 1-hidden-layer net (configurable activation)
// ============================================================

#[derive(Clone)]
struct MLP {
    in_size: usize,
    h_size: usize,
    use_c19: bool,
    w1: Vec<f32>, b1: Vec<f32>,     // in → hidden
    w2: Vec<f32>, b2: Vec<f32>,     // hidden → 128 output
    c: Vec<f32>, rho: Vec<f32>,     // C19 params (only if use_c19)
}

struct MLPCache {
    input: Vec<f32>,
    h_pre: Vec<f32>,   // pre-activation
    h_act: Vec<f32>,   // post-activation
    probs: Vec<f32>,
}

impl MLP {
    fn new(in_size: usize, h_size: usize, use_c19: bool, rng: &mut StdRng) -> Self {
        let sc = (2.0 / in_size as f32).sqrt(); // He init
        MLP {
            in_size, h_size, use_c19,
            w1: (0..h_size*in_size).map(|_| rng.gen_range(-sc..sc)).collect(),
            b1: vec![0.0; h_size],
            w2: (0..N_BYTES*h_size).map(|_| rng.gen_range(-0.1..0.1)).collect(),
            b2: vec![0.0; N_BYTES],
            c: vec![1.0; h_size],
            rho: vec![4.0; h_size],
        }
    }

    fn params(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
            + if self.use_c19 { self.rho.len() } else { 0 }
    }

    fn forward(&self, input: &[f32]) -> MLPCache {
        let h = self.h_size;
        let mut h_pre = vec![0.0f32; h];
        let mut h_act = vec![0.0f32; h];
        for i in 0..h {
            let mut s = self.b1[i];
            for j in 0..self.in_size { s += input[j] * self.w1[i * self.in_size + j]; }
            h_pre[i] = s;
            h_act[i] = if self.use_c19 { c19_fwd(s, self.c[i], self.rho[i]) } else { s.max(0.0) };
        }
        let mut logits = vec![0.0f32; N_BYTES];
        for b in 0..N_BYTES {
            let mut s = self.b2[b];
            for i in 0..h { s += h_act[i] * self.w2[b * h + i]; }
            logits[b] = s;
        }
        MLPCache { input: input.to_vec(), h_pre, h_act, probs: softmax(&logits) }
    }

    fn backward(&self, c: &MLPCache, target: u8) -> Vec<f32> {
        let (h, ins) = (self.h_size, self.in_size);
        let np = self.params();
        let mut grad = vec![0.0f32; np];

        let mut dl = c.probs.clone();
        dl[target as usize] -= 1.0;

        // Output layer grads
        let w2_off = h * ins + h;  // offset in flat param vec
        let b2_off = w2_off + N_BYTES * h;
        let mut d_h = vec![0.0f32; h];
        for b in 0..N_BYTES {
            grad[b2_off + b] = dl[b];
            for i in 0..h {
                grad[w2_off + b * h + i] = dl[b] * c.h_act[i];
                d_h[i] += dl[b] * self.w2[b * h + i];
            }
        }

        // Hidden layer grads
        let w1_off = 0;
        let b1_off = h * ins;
        let rho_off = b2_off + N_BYTES;

        for i in 0..h {
            let deriv = if self.use_c19 {
                c19_deriv(c.h_pre[i], self.c[i], self.rho[i])
            } else {
                if c.h_pre[i] > 0.0 { 1.0 } else { 0.0 }
            };
            let d_pre = d_h[i] * deriv;
            grad[b1_off + i] = d_pre;
            for j in 0..ins {
                grad[w1_off + i * ins + j] = d_pre * c.input[j];
            }
            // C19 rho grad
            if self.use_c19 {
                let s = c.h_pre[i]; let cv = self.c[i].max(0.01); let l = 6.0 * cv;
                if s > -l && s < l {
                    let sc = s / cv; let ft = sc - sc.floor(); let hh = ft * (1.0 - ft);
                    grad[rho_off + i] = d_h[i] * cv * hh * hh;
                }
            }
        }
        grad
    }

    fn param_vec(&self) -> Vec<f32> {
        let mut v = Vec::new();
        v.extend(&self.w1); v.extend(&self.b1);
        v.extend(&self.w2); v.extend(&self.b2);
        if self.use_c19 { v.extend(&self.rho); }
        v
    }

    fn set_param_vec(&mut self, v: &[f32]) {
        let mut o = 0;
        let n=self.w1.len(); self.w1.copy_from_slice(&v[o..o+n]); o+=n;
        let n=self.b1.len(); self.b1.copy_from_slice(&v[o..o+n]); o+=n;
        let n=self.w2.len(); self.w2.copy_from_slice(&v[o..o+n]); o+=n;
        let n=self.b2.len(); self.b2.copy_from_slice(&v[o..o+n]); o+=n;
        if self.use_c19 {
            let n=self.rho.len(); self.rho.copy_from_slice(&v[o..o+n]);
            for r in &mut self.rho { *r = r.max(0.0); }
        }
    }

    fn train_adam(&mut self, data: &[(Vec<f32>, u8)], test: &[(Vec<f32>, u8)],
                  steps: usize, batch: usize, label: &str) {
        let np = self.params();
        let mut m = vec![0.0f32; np];
        let mut v = vec![0.0f32; np];
        let mut rng = StdRng::seed_from_u64(99);
        let mut sh: Vec<usize> = (0..data.len()).collect();
        let test_sub = if test.len() > 2000 { &test[..2000] } else { test };

        println!("  [{}] params={}, train={}, test={}", label, np, data.len(), test_sub.len());

        for step in 1..=steps {
            sh.shuffle(&mut rng);
            let batch_idx = &sh[..batch.min(data.len())];

            let mut grad_sum = vec![0.0f32; np];
            for &idx in batch_idx {
                let c = self.forward(&data[idx].0);
                let g = self.backward(&c, data[idx].1);
                for i in 0..np { grad_sum[i] += g[i]; }
            }
            let bs = batch_idx.len() as f32;
            for g in &mut grad_sum { *g /= bs; }

            let mut pv = self.param_vec();
            let t = step as f32;
            let b1c = 1.0 - 0.9f32.powf(t);
            let b2c = 1.0 - 0.999f32.powf(t);
            for i in 0..np {
                m[i] = 0.9 * m[i] + 0.1 * grad_sum[i];
                v[i] = 0.999 * v[i] + 0.001 * grad_sum[i] * grad_sum[i];
                pv[i] -= 0.001 * (m[i] / b1c) / ((v[i] / b2c).sqrt() + 1e-8);
            }
            self.set_param_vec(&pv);

            if step % 1000 == 0 || step == steps {
                let tr = self.accuracy(data);
                let te = self.accuracy(test_sub);
                let t5 = self.top5(test_sub);
                let lo = self.loss(test_sub);
                println!("    step {:>5}: loss={:.3} train={:.1}% test={:.1}% top5={:.1}%",
                         step, lo, tr*100.0, te*100.0, t5*100.0);
            }
        }
    }

    fn accuracy(&self, data: &[(Vec<f32>, u8)]) -> f64 {
        data.iter().filter(|(inp, t)| {
            let c = self.forward(inp);
            c.probs.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 == *t as usize
        }).count() as f64 / data.len() as f64
    }

    fn top5(&self, data: &[(Vec<f32>, u8)]) -> f64 {
        let mut ok = 0;
        for (inp, t) in data {
            let c = self.forward(inp);
            let mut idx: Vec<(usize,f32)> = c.probs.iter().enumerate().map(|(i,&v)|(i,v)).collect();
            idx.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
            if idx.iter().take(5).any(|(i,_)| *i == *t as usize) { ok += 1; }
        }
        ok as f64 / data.len() as f64
    }

    fn loss(&self, data: &[(Vec<f32>, u8)]) -> f64 {
        let mut l = 0.0f64;
        for (inp, t) in data {
            let c = self.forward(inp);
            l -= (c.probs[*t as usize].max(1e-10) as f64).ln();
        }
        l / data.len() as f64
    }
}

// ============================================================
// C19 Sandwich (same as before, for comparison)
// ============================================================

#[derive(Clone)]
struct Sandwich {
    ns: usize, nd: usize, k: usize,
    sp_w: Vec<f32>, sp_bias: Vec<f32>, sp_rho: Vec<f32>,
    sp_byte: Vec<usize>,
    dn_sp: Vec<f32>, dn_bias: Vec<f32>, dn_rho: Vec<f32>,
    dn_reads_sp: Vec<Vec<usize>>,
    out_w: Vec<f32>, out_bias: Vec<f32>,
}

impl Sandwich {
    fn new(ns: usize, nd: usize, k: usize, rng: &mut StdRng) -> Self {
        let sc = 0.1; let k = k.min(ns);
        let sp_byte: Vec<usize> = (0..ns).map(|i| i % N_BYTES).collect();
        let dn_reads_sp: Vec<Vec<usize>> = (0..nd).map(|i| {
            let mut idx: Vec<usize> = (0..ns).collect();
            let mut r = StdRng::seed_from_u64(42 + i as u64*37+13);
            idx.shuffle(&mut r); idx[..k].to_vec()
        }).collect();
        Sandwich { ns, nd, k, sp_byte,
            sp_w: (0..ns*CTX).map(|_| rng.gen_range(-sc..sc)).collect(),
            sp_bias: (0..ns).map(|_| rng.gen_range(-sc..sc)).collect(),
            sp_rho: vec![4.0; ns],
            dn_sp: (0..nd*k).map(|_| rng.gen_range(-sc..sc)).collect(),
            dn_bias: (0..nd).map(|_| rng.gen_range(-sc..sc)).collect(),
            dn_rho: vec![4.0; nd],
            dn_reads_sp, out_w: (0..N_BYTES*nd).map(|_| rng.gen_range(-sc..sc)).collect(),
            out_bias: (0..N_BYTES).map(|_| rng.gen_range(-sc..sc)).collect(),
        }
    }
    fn params(&self) -> usize {
        self.sp_w.len()+self.sp_bias.len()+self.sp_rho.len()
        +self.dn_sp.len()+self.dn_bias.len()+self.dn_rho.len()
        +self.out_w.len()+self.out_bias.len()
    }
    fn forward(&self, ctx: &[u8]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (ns,nd,k) = (self.ns,self.nd,self.k);
        let mut ss=vec![0.0f32;ns]; let mut sa=vec![0.0f32;ns];
        for i in 0..ns {
            let mut s=self.sp_bias[i];
            for (p,&b) in ctx.iter().enumerate() { if b as usize==self.sp_byte[i] { s+=self.sp_w[i*CTX+p]; } }
            ss[i]=s; sa[i]=c19_fwd(s,1.0,self.sp_rho[i]);
        }
        let mut ds=vec![0.0f32;nd]; let mut da=vec![0.0f32;nd];
        for i in 0..nd {
            let mut s=self.dn_bias[i];
            for (ki,&si) in self.dn_reads_sp[i].iter().enumerate() { s+=sa[si]*self.dn_sp[i*k+ki]; }
            ds[i]=s; da[i]=c19_fwd(s,1.0,self.dn_rho[i]);
        }
        let mut logits=vec![0.0f32;N_BYTES];
        for b in 0..N_BYTES { let mut s=self.out_bias[b]; for d in 0..nd{s+=da[d]*self.out_w[b*nd+d];} logits[b]=s; }
        (ss, sa, ds, da)
    }
    fn predict(&self, ctx: &[u8]) -> (Vec<f32>, u8) {
        let (_,_,_,da) = self.forward(ctx);
        let mut logits=vec![0.0f32;N_BYTES];
        for b in 0..N_BYTES { let mut s=self.out_bias[b]; for d in 0..self.nd{s+=da[d]*self.out_w[b*self.nd+d];} logits[b]=s; }
        let probs = softmax(&logits);
        let pred = probs.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u8;
        (probs, pred)
    }
    fn train_adam(&mut self, train: &[(Vec<u8>,u8)], test: &[(Vec<u8>,u8)], steps: usize, batch: usize) {
        let np = self.params();
        let mut mv = vec![0.0f32; np]; let mut vv = vec![0.0f32; np];
        let mut rng = StdRng::seed_from_u64(99);
        let mut sh = train.to_vec();
        let (ns,nd,k) = (self.ns,self.nd,self.k);
        let test_sub = if test.len()>2000{&test[..2000]}else{test};

        println!("  [C19 sandwich {}S+{}D] params={}", ns, nd, np);

        for step in 1..=steps {
            sh.shuffle(&mut rng);
            let b = &sh[..batch.min(sh.len())];
            let mut g = vec![0.0f32; np];
            for (ctx,target) in b {
                let (ss,sa,ds,da) = self.forward(ctx);
                let mut logits=vec![0.0f32;N_BYTES];
                for bb in 0..N_BYTES { let mut s=self.out_bias[bb]; for d in 0..nd{s+=da[d]*self.out_w[bb*nd+d];} logits[bb]=s; }
                let probs = softmax(&logits);
                let mut dl=probs; dl[*target as usize]-=1.0;
                // backward (accumulate into flat grad)
                let mut o=0;
                // sp_w grads
                let mut d_sp=vec![0.0f32;ns];
                let mut d_dn=vec![0.0f32;nd];
                for bb in 0..N_BYTES { for d in 0..nd { d_dn[d]+=dl[bb]*self.out_w[bb*nd+d]; } }
                for i in 0..nd {
                    let dv=c19_deriv(ds[i],1.0,self.dn_rho[i]); let dd=d_dn[i]*dv;
                    for (ki,&si) in self.dn_reads_sp[i].iter().enumerate() { d_sp[si]+=dd*self.dn_sp[i*k+ki]; }
                }
                for i in 0..ns {
                    let dv=c19_deriv(ss[i],1.0,self.sp_rho[i]); let dd=d_sp[i]*dv;
                    for p in 0..CTX {
                        let inp = if ctx[p] as usize==self.sp_byte[i]{1.0}else{0.0};
                        g[o+i*CTX+p] += dd*inp;
                    }
                }
                o+=ns*CTX;
                for i in 0..ns { let dv=c19_deriv(ss[i],1.0,self.sp_rho[i]); g[o+i]+=d_sp[i]*dv; } o+=ns;
                // sp_rho
                for i in 0..ns {
                    let s=ss[i]; let l=6.0; if s>-l&&s<l { let sc=s; let ft=sc-sc.floor(); let h=ft*(1.0-ft); g[o+i]+=d_sp[i]*h*h; }
                } o+=ns;
                // dn_sp
                for i in 0..nd {
                    let dv=c19_deriv(ds[i],1.0,self.dn_rho[i]); let dd=d_dn[i]*dv;
                    for (ki,&si) in self.dn_reads_sp[i].iter().enumerate() { g[o+i*k+ki]+=dd*sa[si]; }
                } o+=nd*k;
                for i in 0..nd { let dv=c19_deriv(ds[i],1.0,self.dn_rho[i]); g[o+i]+=d_dn[i]*dv; } o+=nd;
                // dn_rho
                for i in 0..nd {
                    let s=ds[i]; let l=6.0; if s>-l&&s<l { let sc=s; let ft=sc-sc.floor(); let h=ft*(1.0-ft); g[o+i]+=d_dn[i]*h*h; }
                } o+=nd;
                // out_w
                for bb in 0..N_BYTES { for d in 0..nd { g[o+bb*nd+d]+=dl[bb]*da[d]; } } o+=N_BYTES*nd;
                for bb in 0..N_BYTES { g[o+bb]+=dl[bb]; }
            }
            let bs=b.len() as f32;
            let mut pv = vec![0.0f32; np];
            { let mut o=0;
              for v in [&self.sp_w,&self.sp_bias,&self.sp_rho,&self.dn_sp,&self.dn_bias,&self.dn_rho,&self.out_w,&self.out_bias] {
                  pv[o..o+v.len()].copy_from_slice(v); o+=v.len();
              }
            }
            let t=step as f32; let b1c=1.0-0.9f32.powf(t); let b2c=1.0-0.999f32.powf(t);
            for i in 0..np {
                g[i]/=bs;
                mv[i]=0.9*mv[i]+0.1*g[i]; vv[i]=0.999*vv[i]+0.001*g[i]*g[i];
                pv[i]-=0.001*(mv[i]/b1c)/((vv[i]/b2c).sqrt()+1e-8);
            }
            { let mut o=0;
              let mut cp = |dst: &mut Vec<f32>| { let n=dst.len(); dst.copy_from_slice(&pv[o..o+n]); o+=n; };
              cp(&mut self.sp_w); cp(&mut self.sp_bias); cp(&mut self.sp_rho);
              cp(&mut self.dn_sp); cp(&mut self.dn_bias); cp(&mut self.dn_rho);
              cp(&mut self.out_w); cp(&mut self.out_bias);
            }
            for r in &mut self.sp_rho{*r=r.max(0.0);} for r in &mut self.dn_rho{*r=r.max(0.0);}

            if step%1000==0||step==steps {
                let tr=train.iter().filter(|(c,t)|self.predict(c).1==*t).count() as f64/train.len() as f64;
                let te=test_sub.iter().filter(|(c,t)|self.predict(c).1==*t).count() as f64/test_sub.len() as f64;
                let t5 = {let mut ok=0; for(c,t) in test_sub{let(p,_)=self.predict(c);let mut idx:Vec<(usize,f32)>=p.iter().enumerate().map(|(i,&v)|(i,v)).collect();idx.sort_by(|a,b|b.1.partial_cmp(&a.1).unwrap());if idx.iter().take(5).any(|(i,_)|*i==*t as usize){ok+=1;}} ok as f64/test_sub.len() as f64};
                let lo = {let mut l=0.0f64;for(c,t)in test_sub{let(p,_)=self.predict(c);l-=(p[*t as usize].max(1e-10) as f64).ln();}l/test_sub.len() as f64};
                println!("    step {:>5}: loss={:.3} train={:.1}% test={:.1}% top5={:.1}%",step,lo,tr*100.0,te*100.0,t5*100.0);
            }
        }
    }
}

fn main() {
    println!("=== MLP BASELINE: the experiment that matters ===\n");
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

    let mut pairs_raw: Vec<(Vec<u8>,u8)> = Vec::new();
    for i in CTX..text.len() {
        let ctx = text[i-CTX..i].to_vec();
        if (text[i] as usize) < N_BYTES { pairs_raw.push((ctx, text[i])); }
    }
    let mut rng = StdRng::seed_from_u64(42);
    pairs_raw.shuffle(&mut rng);
    let n_train = 10000;
    let train_raw = pairs_raw[..n_train].to_vec();
    let test_raw = pairs_raw[n_train..n_train+5000].to_vec();

    // Pre-encode for MLP (one-hot)
    let train_oh: Vec<(Vec<f32>,u8)> = train_raw.iter().map(|(c,t)|(one_hot(c),*t)).collect();
    let test_oh: Vec<(Vec<f32>,u8)> = test_raw.iter().map(|(c,t)|(one_hot(c),*t)).collect();

    println!("  Train: {}, Test: {}\n", n_train, test_raw.len());

    let steps = 10000;
    let batch = 200;
    let mut results: Vec<(String, usize, f64, f64)> = Vec::new();

    // =========================================================
    // 1. ReLU MLP (512 → 19 → 128) ~12.3K params
    // =========================================================
    println!("--- ReLU MLP (512 -> 19 -> 128) ---");
    {
        let mut net = MLP::new(IN, 19, false, &mut StdRng::seed_from_u64(77));
        let t1 = Instant::now();
        net.train_adam(&train_oh, &test_oh, steps, batch, "ReLU MLP");
        let acc = net.accuracy(&test_oh[..2000]);
        let t5 = net.top5(&test_oh[..2000]);
        println!("  => test={:.1}% top5={:.1}% ({:.1}s)\n", acc*100.0, t5*100.0, t1.elapsed().as_secs_f64());
        results.push(("ReLU MLP 512->19->128".into(), net.params(), acc, t5));
    }

    // =========================================================
    // 2. C19 MLP (512 → 19 → 128) — same but C19 activation
    // =========================================================
    println!("--- C19 MLP (512 -> 19 -> 128) ---");
    {
        let mut net = MLP::new(IN, 19, true, &mut StdRng::seed_from_u64(77));
        let t1 = Instant::now();
        net.train_adam(&train_oh, &test_oh, steps, batch, "C19 MLP");
        let acc = net.accuracy(&test_oh[..2000]);
        let t5 = net.top5(&test_oh[..2000]);
        println!("  => test={:.1}% top5={:.1}% ({:.1}s)\n", acc*100.0, t5*100.0, t1.elapsed().as_secs_f64());
        results.push(("C19 MLP 512->19->128".into(), net.params(), acc, t5));
    }

    // =========================================================
    // 3. ReLU MLP bigger (512 → 40 → 128) ~26K params
    // =========================================================
    println!("--- ReLU MLP big (512 -> 40 -> 128) ---");
    {
        let mut net = MLP::new(IN, 40, false, &mut StdRng::seed_from_u64(77));
        let t1 = Instant::now();
        net.train_adam(&train_oh, &test_oh, steps, batch, "ReLU MLP big");
        let acc = net.accuracy(&test_oh[..2000]);
        let t5 = net.top5(&test_oh[..2000]);
        println!("  => test={:.1}% top5={:.1}% ({:.1}s)\n", acc*100.0, t5*100.0, t1.elapsed().as_secs_f64());
        results.push(("ReLU MLP 512->40->128".into(), net.params(), acc, t5));
    }

    // =========================================================
    // 4. C19 Sandwich (384S+72D K=8) ~12.4K params
    // =========================================================
    println!("--- C19 Sandwich (384S+72D K=8) ---");
    {
        let mut net = Sandwich::new(384, 72, 8, &mut StdRng::seed_from_u64(77));
        let t1 = Instant::now();
        net.train_adam(&train_raw, &test_raw, steps, batch);
        let test_sub = &test_raw[..2000];
        let acc = test_sub.iter().filter(|(c,t)| net.predict(c).1==*t).count() as f64/2000.0;
        let t5 = {let mut ok=0;for(c,t)in test_sub{let(p,_)=net.predict(c);let mut idx:Vec<(usize,f32)>=p.iter().enumerate().map(|(i,&v)|(i,v)).collect();idx.sort_by(|a,b|b.1.partial_cmp(&a.1).unwrap());if idx.iter().take(5).any(|(i,_)|*i==*t as usize){ok+=1;}}ok as f64/2000.0};
        println!("  => test={:.1}% top5={:.1}% ({:.1}s)\n", acc*100.0, t5*100.0, t1.elapsed().as_secs_f64());
        results.push(("C19 Sandwich 384S+72D".into(), net.params(), acc, t5));
    }

    // =========================================================
    // VERDICT
    // =========================================================
    println!("=== VERDICT ===");
    println!("  {:>25} {:>8} {:>8} {:>8}", "model", "params", "test%", "top5%");
    println!("  {}", "=".repeat(55));
    for (label, params, acc, t5) in &results {
        println!("  {:>25} {:>8} {:>7.1}% {:>7.1}%", label, params, acc*100.0, t5*100.0);
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
    println!("\n=== DONE ===");
}
