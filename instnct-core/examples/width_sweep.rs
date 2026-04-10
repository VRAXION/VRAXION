//! Width Sweep: how does accuracy scale with neurons?
//!
//! t=1 only (proven best). Sweep: 64S+12D → 1024S+192D
//! Detailed telemetry: neuron utilization, activation stats, prediction diversity
//!
//! Run: cargo run --example width_sweep --release

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

// ============================================================
// Simple t=1 sandwich (no recurrence, no feedback)
// ============================================================

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

    fn apply(&mut self, g: &Grad, lr: f32) {
        for (w,gw) in self.sp_w.iter_mut().zip(&g.sp_w){*w-=lr*gw;}
        for (w,gw) in self.sp_bias.iter_mut().zip(&g.sp_bias){*w-=lr*gw;}
        for i in 0..self.ns { self.sp_rho[i]=(self.sp_rho[i]-lr*g.sp_rho[i]).max(0.0); }
        for (w,gw) in self.dn_sp.iter_mut().zip(&g.dn_sp){*w-=lr*gw;}
        for (w,gw) in self.dn_bias.iter_mut().zip(&g.dn_bias){*w-=lr*gw;}
        for i in 0..self.nd { self.dn_rho[i]=(self.dn_rho[i]-lr*g.dn_rho[i]).max(0.0); }
        for (w,gw) in self.out_w.iter_mut().zip(&g.out_w){*w-=lr*gw;}
        for (w,gw) in self.out_bias.iter_mut().zip(&g.out_bias){*w-=lr*gw;}
    }

    fn params(&self) -> usize {
        self.sp_w.len()+self.sp_bias.len()+self.sp_rho.len()
        +self.dn_sp.len()+self.dn_bias.len()+self.dn_rho.len()
        +self.out_w.len()+self.out_bias.len()
    }

    fn train(&mut self, train: &[(Vec<u8>,u8)], test: &[(Vec<u8>,u8)],
             steps: usize, batch: usize) {
        let mut lr = 0.01f32;
        let mut rng = StdRng::seed_from_u64(99);
        let mut sh = train.to_vec();
        for step in 0..steps {
            sh.shuffle(&mut rng);
            let b = &sh[..batch.min(sh.len())];
            let mut grad = Grad::zeros(self.ns, self.nd, self.k);
            for (ctx,t) in b { let c=self.forward(ctx); let g=self.backward(&c,ctx,*t); grad.add(&g); }
            grad.scale(1.0/b.len() as f32);
            let gn = grad.norm(); if gn<1e-8{continue;} grad.scale(1.0/gn);
            let old_loss = self.loss(b);
            let old = self.clone(); self.apply(&grad, lr);
            if self.loss(b) < old_loss { lr*=1.05; } else { *self=old; lr*=0.5; }
            if step%1000==0 || step==steps-1 {
                let tr=self.accuracy(train); let te=self.accuracy(test);
                let t5=self.top5(test); let lo=self.loss(test);
                println!("    step {:>5}: loss={:.3} train={:.1}% test={:.1}% top5={:.1}% lr={:.5}",
                    step, lo, tr*100.0, te*100.0, t5*100.0, lr);
            }
        }
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
    fn loss(&self, data: &[(Vec<u8>,u8)]) -> f64 {
        let mut l=0.0f64;
        for (c,t) in data { let ca=self.forward(c); l-=(ca.probs[*t as usize].max(1e-10) as f64).ln(); }
        l/data.len() as f64
    }

    // ---- Telemetry ----

    fn telemetry(&self, data: &[(Vec<u8>, u8)], n_samples: usize) {
        let samples = &data[..n_samples.min(data.len())];
        let (ns, nd) = (self.ns, self.nd);
        let n = samples.len();

        let mut sp_active_cnt = vec![0u32; ns]; // how often each sparse neuron is active
        let mut sp_act_abs_sum = vec![0.0f64; ns];
        let mut dn_active_cnt = vec![0u32; nd];
        let mut dn_act_abs_sum = vec![0.0f64; nd];
        let mut predictions = vec![0u32; N_BYTES];
        let mut per_tgt_correct = vec![0u32; N_BYTES];
        let mut per_tgt_count = vec![0u32; N_BYTES];

        for (ctx, target) in samples {
            let c = self.forward(ctx);
            for i in 0..ns {
                if c.sp_act[i].abs() > 0.01 { sp_active_cnt[i] += 1; }
                sp_act_abs_sum[i] += c.sp_act[i].abs() as f64;
            }
            for i in 0..nd {
                if c.dn_act[i].abs() > 0.01 { dn_active_cnt[i] += 1; }
                dn_act_abs_sum[i] += c.dn_act[i].abs() as f64;
            }
            let pred = c.probs.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            predictions[pred] += 1;
            per_tgt_count[*target as usize] += 1;
            if pred == *target as usize { per_tgt_correct[*target as usize] += 1; }
        }

        // Sparse stats
        let sp_ever = sp_active_cnt.iter().filter(|&&c| c > 0).count();
        let sp_usually = sp_active_cnt.iter().filter(|&&c| c as usize > n/2).count();
        let sp_never = ns - sp_ever;
        let sp_mean = sp_act_abs_sum.iter().sum::<f64>() / (ns * n) as f64;

        // Top sparse neurons by activation
        let mut sp_ranked: Vec<(usize, f64)> = sp_act_abs_sum.iter().enumerate()
            .map(|(i, &s)| (i, s / n as f64)).collect();
        sp_ranked.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

        // Dense stats
        let dn_ever = dn_active_cnt.iter().filter(|&&c| c > 0).count();
        let dn_mean = dn_act_abs_sum.iter().sum::<f64>() / (nd * n) as f64;

        // Output weight influence per dense neuron
        let mut dn_influence: Vec<f64> = (0..nd).map(|d| {
            (0..N_BYTES).map(|b| self.out_w[b*nd+d].abs() as f64).sum()
        }).collect();
        let max_inf = dn_influence.iter().cloned().fold(0.0, f64::max);
        let sig_dn = dn_influence.iter().filter(|&&v| v > max_inf * 0.1).count();

        // Prediction diversity
        let unique_preds = predictions.iter().filter(|&&c| c > 0).count();
        let entropy: f64 = predictions.iter().map(|&c| {
            if c == 0 { 0.0 } else { let p = c as f64 / n as f64; -p * p.ln() }
        }).sum::<f64>() / (2.0f64).ln();

        // Print
        println!("\n  TELEMETRY ({}S+{}D, {} samples):", ns, nd, n);

        println!("  Sparse ({} neurons):", ns);
        println!("    Ever active:    {}/{} ({:.0}%)  never: {}", sp_ever, ns, sp_ever as f64/ns as f64*100.0, sp_never);
        println!("    Usually active: {}/{} ({:.0}%)  — >50% of samples", sp_usually, ns, sp_usually as f64/ns as f64*100.0);
        println!("    Mean |act|:     {:.5}", sp_mean);
        print!("    Top 5 neurons:  ");
        for (i, (idx, val)) in sp_ranked.iter().take(5).enumerate() {
            if i > 0 { print!(", "); }
            let byte = self.sp_byte[*idx];
            let ch = if byte >= 32 && byte < 127 { byte as u8 as char } else { '?' };
            print!("sp{}(byte '{}')={:.3}", idx, ch, val);
        }
        println!();

        // Byte coverage: how many bytes have at least 1 dedicated sparse neuron?
        let mut byte_coverage = vec![false; N_BYTES];
        for &b in &self.sp_byte {
            if b < N_BYTES && sp_active_cnt[self.sp_byte.iter().position(|&x| x == b).unwrap_or(0)] > 0 {
                byte_coverage[b] = true;
            }
        }
        // Actually count how many neurons per byte
        let mut neurons_per_byte = vec![0u32; N_BYTES];
        for &b in &self.sp_byte { if b < N_BYTES { neurons_per_byte[b] += 1; } }
        let covered = neurons_per_byte.iter().filter(|&&c| c > 0).count();
        let max_per = *neurons_per_byte.iter().max().unwrap_or(&0);
        let min_per = *neurons_per_byte.iter().filter(|&&c| c > 0).min().unwrap_or(&0);
        println!("    Bytes covered:  {}/128 ({} neurons/byte: min={}, max={})",
                 covered, if ns >= N_BYTES { ns / N_BYTES } else { 0 }, min_per, max_per);

        println!("  Dense ({} neurons):", nd);
        println!("    Active:         {}/{} ({:.0}%)", dn_ever, nd, dn_ever as f64/nd as f64*100.0);
        println!("    Mean |act|:     {:.5}", dn_mean);
        println!("    Significant (output influence >10% of max): {}/{}", sig_dn, nd);

        println!("  Predictions:");
        println!("    Unique bytes:   {}/128", unique_preds);
        println!("    Entropy:        {:.2} bits (uniform={:.1})", entropy, (N_BYTES as f64).log2());
        let mut ps: Vec<(usize,u32)> = predictions.iter().enumerate().map(|(i,&c)|(i,c)).collect();
        ps.sort_by(|a,b| b.1.cmp(&a.1));
        print!("    Top preds:      ");
        for (i, (byte, count)) in ps.iter().take(6).enumerate() {
            let ch = if *byte >= 32 && *byte < 127 { *byte as u8 as char } else { '?' };
            if i > 0 { print!("  "); }
            print!("'{}'={:.1}%", ch, *count as f64 / n as f64 * 100.0);
        }
        println!();

        // Per-byte accuracy for top targets
        println!("  Per-byte accuracy:");
        let mut ts: Vec<(usize,u32)> = per_tgt_count.iter().enumerate()
            .filter(|(_,&c)| c > 0).map(|(i,&c)|(i,c)).collect();
        ts.sort_by(|a,b| b.1.cmp(&a.1));
        for (byte, count) in ts.iter().take(8) {
            let correct = per_tgt_correct[*byte];
            let ch = if *byte >= 32 && *byte < 127 { *byte as u8 as char } else { '?' };
            println!("    '{}': {}/{} ({:.1}%)", ch, correct, count, correct as f64 / *count as f64 * 100.0);
        }

        // Weight stats
        println!("  Weights:");
        let w_stat = |v: &[f32]| -> (f32, f32, f32) {
            let mean = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
            let max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let std = (v.iter().map(|x| (x.abs()-mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
            (mean, std, max)
        };
        let (m,s,x) = w_stat(&self.sp_w); println!("    sp_w:  |mean|={:.4} std={:.4} |max|={:.3}", m, s, x);
        let (m,s,x) = w_stat(&self.dn_sp); println!("    dn_sp: |mean|={:.4} std={:.4} |max|={:.3}", m, s, x);
        let (m,s,x) = w_stat(&self.out_w); println!("    out_w: |mean|={:.4} std={:.4} |max|={:.3}", m, s, x);

        // Rho stats
        let rho_mean_sp = self.sp_rho.iter().sum::<f32>() / ns as f32;
        let rho_mean_dn = self.dn_rho.iter().sum::<f32>() / nd as f32;
        println!("    sp_rho mean={:.2}  dn_rho mean={:.2}", rho_mean_sp, rho_mean_dn);
    }

    fn save(&self, path: &str) {
        let mut data = Vec::new();
        data.extend_from_slice(b"WSWP");
        for &v in &[self.ns as u32, self.nd as u32, self.k as u32] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        for vec in [&self.sp_w, &self.sp_bias, &self.sp_c, &self.sp_rho,
                     &self.dn_sp, &self.dn_bias, &self.dn_c, &self.dn_rho,
                     &self.out_w, &self.out_bias] {
            for &v in vec.iter() { data.extend_from_slice(&v.to_le_bytes()); }
        }
        if let Some(p) = std::path::Path::new(path).parent() { std::fs::create_dir_all(p).ok(); }
        std::fs::write(path, &data).expect("save failed");
    }

    fn load(path: &str) -> Option<Self> {
        let data = std::fs::read(path).ok()?;
        if data.len() < 16 || &data[0..4] != b"WSWP" { return None; }
        let r = |o: usize| u32::from_le_bytes([data[o],data[o+1],data[o+2],data[o+3]]) as usize;
        let ns = r(4); let nd = r(8); let k = r(12);
        let sp_byte: Vec<usize> = (0..ns).map(|i| i % N_BYTES).collect();
        let dn_reads_sp: Vec<Vec<usize>> = (0..nd).map(|i| {
            let mut idx: Vec<usize> = (0..ns).collect();
            let mut rv = StdRng::seed_from_u64(42 + i as u64*37+13);
            idx.shuffle(&mut rv); idx[..k].to_vec()
        }).collect();
        let mut off = 16usize;
        let mut rv = |n: usize| -> Vec<f32> {
            let v: Vec<f32> = (0..n).map(|i| {
                let o=off+i*4; f32::from_le_bytes([data[o],data[o+1],data[o+2],data[o+3]])
            }).collect(); off += n*4; v
        };
        Some(Net { ns, nd, k, sp_byte, dn_reads_sp,
            sp_w: rv(ns*CTX), sp_bias: rv(ns), sp_c: rv(ns), sp_rho: rv(ns),
            dn_sp: rv(nd*k), dn_bias: rv(nd), dn_c: rv(nd), dn_rho: rv(nd),
            out_w: rv(N_BYTES*nd), out_bias: rv(N_BYTES),
        })
    }
}

fn bigram_baseline(train: &[(Vec<u8>,u8)], test: &[(Vec<u8>,u8)]) -> f64 {
    let mut counts = vec![vec![0u32; N_BYTES]; N_BYTES];
    for (ctx,t) in train {
        let last = *ctx.last().unwrap() as usize;
        if last < N_BYTES { counts[last][*t as usize] += 1; }
    }
    let predict = |l:usize| counts[l].iter().enumerate().max_by_key(|(_,&c)|c).map(|(i,_)|i).unwrap_or(32);
    test.iter().filter(|(c,t)| predict(*c.last().unwrap() as usize)==*t as usize).count() as f64/test.len() as f64
}

fn main() {
    println!("=== WIDTH SWEEP: scaling sparse-dense sandwich ===\n");
    let t0 = Instant::now();
    let ckpt_dir = "checkpoints/width_sweep";

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
    let train = pairs[..2000].to_vec();
    let test = pairs[2000..3000].to_vec();
    println!("  Train: {}, Test: {}\n", train.len(), test.len());

    let bi = bigram_baseline(&train, &test);
    println!("  Bigram: {:.1}%\n", bi*100.0);

    // Width sweep configs (ns, nd)
    let configs: Vec<(usize, usize)> = vec![
        (64, 12),
        (128, 24),
        (256, 48),
        (384, 72),
        (512, 96),
        (768, 144),
        (1024, 192),
    ];
    let k = 8;
    let mut results: Vec<(usize, usize, usize, f64, f64, f64)> = Vec::new();

    for &(ns, nd) in &configs {
        let label = format!("{}s_{}d", ns, nd);
        println!("=== {}S+{}D (K={}) ===", ns, nd, k);
        let ckpt = format!("{}/{}.bin", ckpt_dir, label);
        let t1 = Instant::now();

        let net = if let Some(n) = Net::load(&ckpt) {
            println!("  Loaded from checkpoint");
            n
        } else {
            let mut n = Net::new(ns, nd, k, 42, &mut rng);
            println!("  Params: {} (training...)", n.params());
            n.train(&train, &test, 5000, 200);
            n.save(&ckpt);
            println!("  [saved: {}]", ckpt);
            n
        };

        let acc = net.accuracy(&test);
        let t5 = net.top5(&test);
        let secs = t1.elapsed().as_secs_f64();
        println!("  => test={:.1}% top5={:.1}% ({:.1}s)", acc*100.0, t5*100.0, secs);
        net.telemetry(&test, 500);
        results.push((ns, nd, net.params(), acc, t5, secs));
        println!();
    }

    // =========================================================
    // SCALING CURVE
    // =========================================================
    println!("=== SCALING CURVE ===");
    println!("  {:>6} {:>6} {:>8} {:>8} {:>8} {:>8}", "sparse", "dense", "params", "test%", "top5%", "time");
    println!("  {}", "=".repeat(52));
    for (ns, nd, params, acc, t5, secs) in &results {
        println!("  {:>6} {:>6} {:>8} {:>7.1}% {:>7.1}% {:>7.1}s", ns, nd, params, acc*100.0, t5*100.0, secs);
    }
    println!("  {:>6} {:>6} {:>8} {:>7.1}%", "bigram", "-", "-", bi*100.0);

    // Param efficiency
    println!("\n  Params/accuracy:");
    for (_, _, params, acc, _, _) in &results {
        println!("    {:>6} params -> {:.1}% ({:.1} params/pct)",
                 params, acc*100.0, *params as f64 / (acc*100.0));
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
    println!("\n=== DONE ===");
}
