//! Residual Sandwich: depth via residual tick recurrence + optional layer norm
//!
//! Key: state[t] = state[t-1] + correction (residual)
//!      state[t] = LayerNorm(state[t])       (optional LN, keeps state bounded)
//!
//! Compare: residual vs non-residual vs residual+LN at t=1,2,4
//! Telemetry: per-tick state norms, correction magnitudes, intermediate accuracy
//!
//! Run: cargo run --example residual_sandwich --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use std::time::Instant;

const CTX: usize = 4;
const N_BYTES: usize = 128;

fn c19_fwd(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01); let l = 6.0 * c;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
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
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn encode_ctx(context: &[u8], byte_idx: usize) -> [f32; CTX] {
    let mut v = [0.0f32; CTX];
    for (pos, &b) in context.iter().enumerate() {
        if (b as usize) == byte_idx { v[pos] = 1.0; }
    }
    v
}

// ============================================================
// ResidualSandwich with optional LayerNorm
// ============================================================

#[derive(Clone)]
struct ResidualSandwich {
    n_sparse: usize,
    n_dense: usize,
    k: usize,
    ticks: usize,
    residual: bool,
    use_ln: bool,

    sp_w: Vec<f32>, sp_fb: Vec<f32>, sp_bias: Vec<f32>,
    sp_c: Vec<f32>, sp_rho: Vec<f32>,
    sp_byte: Vec<usize>,
    sp_reads_dn: Vec<Vec<usize>>,

    dn_sp: Vec<f32>, dn_dn: Vec<f32>, dn_bias: Vec<f32>,
    dn_c: Vec<f32>, dn_rho: Vec<f32>,
    dn_reads_sp: Vec<Vec<usize>>,

    out_w: Vec<f32>, out_bias: Vec<f32>,

    // Layer norm params (per-element scale + shift)
    ln_gamma: Vec<f32>,  // n_dense, init 1.0
    ln_beta: Vec<f32>,   // n_dense, init 0.0
}

struct FwdCache {
    sp_sum: Vec<Vec<f32>>,
    sp_act: Vec<Vec<f32>>,
    dn_sum: Vec<Vec<f32>>,
    correction: Vec<Vec<f32>>,
    state: Vec<Vec<f32>>,        // [tick+1] after LN if use_ln
    thermos: Vec<[f32; CTX]>,
    probs: Vec<f32>,
    // LN cache (only filled if use_ln)
    ln_x_hat: Vec<Vec<f32>>,    // [tick] normalized values
    ln_std: Vec<f32>,            // [tick] std
}

struct Grad {
    sp_w: Vec<f32>, sp_fb: Vec<f32>, sp_bias: Vec<f32>, sp_rho: Vec<f32>,
    dn_sp: Vec<f32>, dn_dn: Vec<f32>, dn_bias: Vec<f32>, dn_rho: Vec<f32>,
    out_w: Vec<f32>, out_bias: Vec<f32>,
    ln_gamma: Vec<f32>, ln_beta: Vec<f32>,
}

impl Grad {
    fn zeros(ns: usize, nd: usize, k_dn: usize, k_sp: usize) -> Self {
        Grad {
            sp_w: vec![0.0; ns*CTX], sp_fb: vec![0.0; ns*k_dn],
            sp_bias: vec![0.0; ns], sp_rho: vec![0.0; ns],
            dn_sp: vec![0.0; nd*k_sp], dn_dn: vec![0.0; nd*nd],
            dn_bias: vec![0.0; nd], dn_rho: vec![0.0; nd],
            out_w: vec![0.0; N_BYTES*nd], out_bias: vec![0.0; N_BYTES],
            ln_gamma: vec![0.0; nd], ln_beta: vec![0.0; nd],
        }
    }
    fn add(&mut self, o: &Grad) {
        for (a,b) in self.sp_w.iter_mut().zip(&o.sp_w){*a+=b;}
        for (a,b) in self.sp_fb.iter_mut().zip(&o.sp_fb){*a+=b;}
        for (a,b) in self.sp_bias.iter_mut().zip(&o.sp_bias){*a+=b;}
        for (a,b) in self.sp_rho.iter_mut().zip(&o.sp_rho){*a+=b;}
        for (a,b) in self.dn_sp.iter_mut().zip(&o.dn_sp){*a+=b;}
        for (a,b) in self.dn_dn.iter_mut().zip(&o.dn_dn){*a+=b;}
        for (a,b) in self.dn_bias.iter_mut().zip(&o.dn_bias){*a+=b;}
        for (a,b) in self.dn_rho.iter_mut().zip(&o.dn_rho){*a+=b;}
        for (a,b) in self.out_w.iter_mut().zip(&o.out_w){*a+=b;}
        for (a,b) in self.out_bias.iter_mut().zip(&o.out_bias){*a+=b;}
        for (a,b) in self.ln_gamma.iter_mut().zip(&o.ln_gamma){*a+=b;}
        for (a,b) in self.ln_beta.iter_mut().zip(&o.ln_beta){*a+=b;}
    }
    fn scale(&mut self, s: f32) {
        for v in &mut self.sp_w{*v*=s;} for v in &mut self.sp_fb{*v*=s;}
        for v in &mut self.sp_bias{*v*=s;} for v in &mut self.sp_rho{*v*=s;}
        for v in &mut self.dn_sp{*v*=s;} for v in &mut self.dn_dn{*v*=s;}
        for v in &mut self.dn_bias{*v*=s;} for v in &mut self.dn_rho{*v*=s;}
        for v in &mut self.out_w{*v*=s;} for v in &mut self.out_bias{*v*=s;}
        for v in &mut self.ln_gamma{*v*=s;} for v in &mut self.ln_beta{*v*=s;}
    }
    fn norm(&self) -> f32 {
        let mut s=0.0f32;
        for v in &self.sp_w{s+=v*v;} for v in &self.sp_fb{s+=v*v;}
        for v in &self.sp_bias{s+=v*v;} for v in &self.sp_rho{s+=v*v;}
        for v in &self.dn_sp{s+=v*v;} for v in &self.dn_dn{s+=v*v;}
        for v in &self.dn_bias{s+=v*v;} for v in &self.dn_rho{s+=v*v;}
        for v in &self.out_w{s+=v*v;} for v in &self.out_bias{s+=v*v;}
        for v in &self.ln_gamma{s+=v*v;} for v in &self.ln_beta{s+=v*v;}
        s.sqrt()
    }
}

impl ResidualSandwich {
    fn new(ns: usize, nd: usize, k: usize, ticks: usize,
           residual: bool, use_ln: bool, seed: u64, rng: &mut StdRng) -> Self {
        let sc = 0.1f32;
        let k_dn = k.min(nd); let k_sp = k.min(ns);
        let sp_byte: Vec<usize> = (0..ns).map(|i| i % N_BYTES).collect();
        let sp_reads_dn: Vec<Vec<usize>> = (0..ns).map(|i| {
            let mut idx: Vec<usize> = (0..nd).collect();
            let mut r = StdRng::seed_from_u64(seed + i as u64 * 31 + 7);
            idx.shuffle(&mut r); idx[..k_dn].to_vec()
        }).collect();
        let dn_reads_sp: Vec<Vec<usize>> = (0..nd).map(|i| {
            let mut idx: Vec<usize> = (0..ns).collect();
            let mut r = StdRng::seed_from_u64(seed + i as u64 * 37 + 13);
            idx.shuffle(&mut r); idx[..k_sp].to_vec()
        }).collect();
        ResidualSandwich {
            n_sparse: ns, n_dense: nd, k, ticks, residual, use_ln,
            sp_w: (0..ns*CTX).map(|_| rng.gen_range(-sc..sc)).collect(),
            sp_fb: (0..ns*k_dn).map(|_| rng.gen_range(-sc..sc)).collect(),
            sp_bias: (0..ns).map(|_| rng.gen_range(-sc..sc)).collect(),
            sp_c: vec![1.0; ns], sp_rho: vec![4.0; ns],
            sp_byte, sp_reads_dn,
            dn_sp: (0..nd*k_sp).map(|_| rng.gen_range(-sc..sc)).collect(),
            dn_dn: (0..nd*nd).map(|_| rng.gen_range(-sc..sc)).collect(),
            dn_bias: (0..nd).map(|_| rng.gen_range(-sc..sc)).collect(),
            dn_c: vec![1.0; nd], dn_rho: vec![4.0; nd],
            dn_reads_sp,
            out_w: (0..N_BYTES*nd).map(|_| rng.gen_range(-sc..sc)).collect(),
            out_bias: (0..N_BYTES).map(|_| rng.gen_range(-sc..sc)).collect(),
            ln_gamma: vec![1.0; nd],
            ln_beta: vec![0.0; nd],
        }
    }

    fn forward(&self, context: &[u8]) -> FwdCache {
        let (ns, nd) = (self.n_sparse, self.n_dense);
        let k_dn = self.k.min(nd); let k_sp = self.k.min(ns);
        let thermos: Vec<[f32; CTX]> = (0..ns).map(|i| encode_ctx(context, self.sp_byte[i])).collect();

        let mut states = vec![vec![0.0f32; nd]];
        let mut sp_sum_all = Vec::new();
        let mut sp_act_all = Vec::new();
        let mut dn_sum_all = Vec::new();
        let mut corr_all = Vec::new();
        let mut ln_x_hat_all = Vec::new();
        let mut ln_std_all = Vec::new();

        for _t in 0..self.ticks {
            let prev = states.last().unwrap();

            // Sparse fires
            let mut sp_sum = vec![0.0f32; ns];
            let mut sp_act = vec![0.0f32; ns];
            for i in 0..ns {
                let mut s = self.sp_bias[i];
                for j in 0..CTX { s += thermos[i][j] * self.sp_w[i*CTX+j]; }
                for (ki, &di) in self.sp_reads_dn[i].iter().enumerate() {
                    s += prev[di] * self.sp_fb[i*k_dn+ki];
                }
                sp_sum[i] = s;
                sp_act[i] = c19_fwd(s, self.sp_c[i], self.sp_rho[i]);
            }

            // Dense fires -> correction
            let mut dn_sum = vec![0.0f32; nd];
            let mut corr = vec![0.0f32; nd];
            for i in 0..nd {
                let mut s = self.dn_bias[i];
                for (ki, &si) in self.dn_reads_sp[i].iter().enumerate() {
                    s += sp_act[si] * self.dn_sp[i*k_sp+ki];
                }
                for j in 0..nd {
                    if j != i { s += prev[j] * self.dn_dn[i*nd+j]; }
                }
                dn_sum[i] = s;
                corr[i] = c19_fwd(s, self.dn_c[i], self.dn_rho[i]);
            }

            // State update (residual or replace)
            let mut new_state: Vec<f32> = if self.residual {
                (0..nd).map(|i| prev[i] + corr[i]).collect()
            } else {
                corr.clone()
            };

            // Layer norm (post-residual)
            let (x_hat, std_val) = if self.use_ln {
                let nf = nd as f32;
                let mu: f32 = new_state.iter().sum::<f32>() / nf;
                let var: f32 = new_state.iter().map(|&v| (v-mu)*(v-mu)).sum::<f32>() / nf;
                let std = (var + 1e-5).sqrt();
                let x_hat: Vec<f32> = new_state.iter().map(|&v| (v - mu) / std).collect();
                for i in 0..nd {
                    new_state[i] = self.ln_gamma[i] * x_hat[i] + self.ln_beta[i];
                }
                (x_hat, std)
            } else {
                (vec![], 0.0)
            };

            sp_sum_all.push(sp_sum);
            sp_act_all.push(sp_act);
            dn_sum_all.push(dn_sum);
            corr_all.push(corr);
            ln_x_hat_all.push(x_hat);
            ln_std_all.push(std_val);
            states.push(new_state);
        }

        // Output from final state
        let final_state = states.last().unwrap();
        let mut logits = vec![0.0f32; N_BYTES];
        for b in 0..N_BYTES {
            let mut s = self.out_bias[b];
            for d in 0..nd { s += final_state[d] * self.out_w[b*nd+d]; }
            logits[b] = s;
        }
        let probs = softmax(&logits);

        FwdCache {
            sp_sum: sp_sum_all, sp_act: sp_act_all, dn_sum: dn_sum_all,
            correction: corr_all, state: states, thermos, probs,
            ln_x_hat: ln_x_hat_all, ln_std: ln_std_all,
        }
    }

    fn backward(&self, cache: &FwdCache, target: u8) -> Grad {
        let (ns, nd) = (self.n_sparse, self.n_dense);
        let k_dn = self.k.min(nd); let k_sp = self.k.min(ns);
        let mut g = Grad::zeros(ns, nd, k_dn, k_sp);

        let mut d_logits = cache.probs.clone();
        d_logits[target as usize] -= 1.0;

        // Output -> d_final_state
        let final_state = &cache.state[self.ticks];
        let mut d_state = vec![0.0f32; nd];
        for b in 0..N_BYTES {
            g.out_bias[b] += d_logits[b];
            for d in 0..nd {
                g.out_w[b*nd+d] += d_logits[b] * final_state[d];
                d_state[d] += d_logits[b] * self.out_w[b*nd+d];
            }
        }

        // BPTT backwards
        for t in (0..self.ticks).rev() {
            let prev = &cache.state[t];

            // Backprop through layer norm (if used)
            let d_raw = if self.use_ln {
                let x_hat = &cache.ln_x_hat[t];
                let std = cache.ln_std[t];
                let nf = nd as f32;
                // LN param grads
                for i in 0..nd {
                    g.ln_gamma[i] += d_state[i] * x_hat[i];
                    g.ln_beta[i] += d_state[i];
                }
                // d_x_hat = d_out * gamma
                let d_x_hat: Vec<f32> = (0..nd).map(|i| d_state[i] * self.ln_gamma[i]).collect();
                // Compact LN backward: d_input = (d_xh - mean(d_xh) - xh*mean(d_xh*xh)) / std
                let mean_dx: f32 = d_x_hat.iter().sum::<f32>() / nf;
                let mean_dx_xh: f32 = d_x_hat.iter().zip(x_hat).map(|(d,x)| d*x).sum::<f32>() / nf;
                (0..nd).map(|i| (d_x_hat[i] - mean_dx - x_hat[i] * mean_dx_xh) / std).collect()
            } else {
                d_state.clone()
            };

            // d_raw -> correction + prev (residual identity)
            let d_corr = d_raw.clone();
            let mut d_prev = if self.residual { d_raw } else { vec![0.0f32; nd] };

            // Backprop through correction = C19(dn_sum)
            let mut d_sp_act = vec![0.0f32; ns];
            for i in 0..nd {
                let deriv = c19_deriv(cache.dn_sum[t][i], self.dn_c[i], self.dn_rho[i]);
                let d_dn_sum = d_corr[i] * deriv;
                g.dn_bias[i] += d_dn_sum;
                for (ki, &si) in self.dn_reads_sp[i].iter().enumerate() {
                    g.dn_sp[i*k_sp+ki] += d_dn_sum * cache.sp_act[t][si];
                    d_sp_act[si] += d_dn_sum * self.dn_sp[i*k_sp+ki];
                }
                for j in 0..nd {
                    if j != i {
                        g.dn_dn[i*nd+j] += d_dn_sum * prev[j];
                        d_prev[j] += d_dn_sum * self.dn_dn[i*nd+j];
                    }
                }
                let s = cache.dn_sum[t][i]; let c = self.dn_c[i].max(0.01); let l = 6.0*c;
                if s > -l && s < l {
                    let sc = s/c; let ft = sc-sc.floor(); let h = ft*(1.0-ft);
                    g.dn_rho[i] += d_corr[i] * c * h * h;
                }
            }

            // Backprop through sparse
            for i in 0..ns {
                let deriv = c19_deriv(cache.sp_sum[t][i], self.sp_c[i], self.sp_rho[i]);
                let d_sp_sum = d_sp_act[i] * deriv;
                g.sp_bias[i] += d_sp_sum;
                for j in 0..CTX {
                    g.sp_w[i*CTX+j] += d_sp_sum * cache.thermos[i][j];
                }
                for (ki, &di) in self.sp_reads_dn[i].iter().enumerate() {
                    g.sp_fb[i*k_dn+ki] += d_sp_sum * prev[di];
                    d_prev[di] += d_sp_sum * self.sp_fb[i*k_dn+ki];
                }
                let s = cache.sp_sum[t][i]; let c = self.sp_c[i].max(0.01); let l = 6.0*c;
                if s > -l && s < l {
                    let sc = s/c; let ft = sc-sc.floor(); let h = ft*(1.0-ft);
                    g.sp_rho[i] += d_sp_act[i] * c * h * h;
                }
            }

            d_state = d_prev;
        }
        g
    }

    fn apply_grad(&mut self, g: &Grad, lr: f32) {
        for (w,gw) in self.sp_w.iter_mut().zip(&g.sp_w) { *w -= lr*gw; }
        for (w,gw) in self.sp_fb.iter_mut().zip(&g.sp_fb) { *w -= lr*gw; }
        for (w,gw) in self.sp_bias.iter_mut().zip(&g.sp_bias) { *w -= lr*gw; }
        for i in 0..self.n_sparse { self.sp_rho[i]=(self.sp_rho[i]-lr*g.sp_rho[i]).max(0.0); }
        for (w,gw) in self.dn_sp.iter_mut().zip(&g.dn_sp) { *w -= lr*gw; }
        for (w,gw) in self.dn_dn.iter_mut().zip(&g.dn_dn) { *w -= lr*gw; }
        for (w,gw) in self.dn_bias.iter_mut().zip(&g.dn_bias) { *w -= lr*gw; }
        for i in 0..self.n_dense { self.dn_rho[i]=(self.dn_rho[i]-lr*g.dn_rho[i]).max(0.0); }
        for (w,gw) in self.out_w.iter_mut().zip(&g.out_w) { *w -= lr*gw; }
        for (w,gw) in self.out_bias.iter_mut().zip(&g.out_bias) { *w -= lr*gw; }
        if self.use_ln {
            for (w,gw) in self.ln_gamma.iter_mut().zip(&g.ln_gamma) { *w -= lr*gw; }
            for (w,gw) in self.ln_beta.iter_mut().zip(&g.ln_beta) { *w -= lr*gw; }
        }
    }

    fn predict_from_state(&self, state: &[f32]) -> usize {
        let nd = self.n_dense;
        let mut logits = vec![0.0f32; N_BYTES];
        for b in 0..N_BYTES {
            let mut s = self.out_bias[b];
            for d in 0..nd { s += state[d] * self.out_w[b*nd+d]; }
            logits[b] = s;
        }
        logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    }

    fn accuracy(&self, data: &[(Vec<u8>, u8)]) -> f64 {
        data.iter().filter(|(c,t)| {
            let cache = self.forward(c);
            cache.probs.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 == *t as usize
        }).count() as f64 / data.len() as f64
    }

    fn top5(&self, data: &[(Vec<u8>, u8)]) -> f64 {
        let mut ok = 0;
        for (ctx, t) in data {
            let c = self.forward(ctx);
            let mut idx: Vec<(usize,f32)> = c.probs.iter().enumerate().map(|(i,&v)|(i,v)).collect();
            idx.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
            if idx.iter().take(5).any(|(i,_)| *i==*t as usize) { ok+=1; }
        }
        ok as f64 / data.len() as f64
    }

    fn loss(&self, data: &[(Vec<u8>, u8)]) -> f64 {
        let mut l = 0.0f64;
        for (ctx, t) in data {
            let c = self.forward(ctx);
            l -= (c.probs[*t as usize].max(1e-10) as f64).ln();
        }
        l / data.len() as f64
    }

    fn param_count(&self) -> usize {
        let base = self.sp_w.len()+self.sp_fb.len()+self.sp_bias.len()+self.sp_rho.len()
            +self.dn_sp.len()+self.dn_dn.len()+self.dn_bias.len()+self.dn_rho.len()
            +self.out_w.len()+self.out_bias.len();
        if self.use_ln { base + self.ln_gamma.len() + self.ln_beta.len() } else { base }
    }

    // ---- Checkpoint ----

    fn save_checkpoint(&self, path: &str) {
        let mut data = Vec::new();
        let magic = if self.use_ln { b"RSLN" } else { b"RSWC" };
        data.extend_from_slice(magic);
        for &v in &[self.n_sparse as u32, self.n_dense as u32,
                     self.k as u32, self.ticks as u32, self.residual as u32] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        for vec in [&self.sp_w, &self.sp_fb, &self.sp_bias, &self.sp_c, &self.sp_rho,
                     &self.dn_sp, &self.dn_dn, &self.dn_bias, &self.dn_c, &self.dn_rho,
                     &self.out_w, &self.out_bias] {
            for &v in vec.iter() { data.extend_from_slice(&v.to_le_bytes()); }
        }
        if self.use_ln {
            for &v in self.ln_gamma.iter() { data.extend_from_slice(&v.to_le_bytes()); }
            for &v in self.ln_beta.iter() { data.extend_from_slice(&v.to_le_bytes()); }
        }
        if let Some(p) = std::path::Path::new(path).parent() { std::fs::create_dir_all(p).ok(); }
        std::fs::write(path, &data).expect("save failed");
    }

    fn load_checkpoint(path: &str) -> Option<Self> {
        let data = std::fs::read(path).ok()?;
        if data.len() < 24 { return None; }
        let magic = &data[0..4];
        let has_ln = match magic {
            b"RSWC" => false,
            b"RSLN" => true,
            _ => return None,
        };
        let r = |off: usize| u32::from_le_bytes([data[off],data[off+1],data[off+2],data[off+3]]);
        let ns = r(4) as usize; let nd = r(8) as usize;
        let k = r(12) as usize; let ticks = r(16) as usize;
        let residual = r(20) != 0;
        let k_dn = k.min(nd); let k_sp = k.min(ns);
        let seed = 42u64;

        let sp_byte: Vec<usize> = (0..ns).map(|i| i % N_BYTES).collect();
        let sp_reads_dn: Vec<Vec<usize>> = (0..ns).map(|i| {
            let mut idx: Vec<usize> = (0..nd).collect();
            let mut r = StdRng::seed_from_u64(seed + i as u64 * 31 + 7);
            idx.shuffle(&mut r); idx[..k_dn].to_vec()
        }).collect();
        let dn_reads_sp: Vec<Vec<usize>> = (0..nd).map(|i| {
            let mut idx: Vec<usize> = (0..ns).collect();
            let mut r = StdRng::seed_from_u64(seed + i as u64 * 37 + 13);
            idx.shuffle(&mut r); idx[..k_sp].to_vec()
        }).collect();

        let mut off = 24usize;
        let mut rv = |n: usize| -> Vec<f32> {
            let v: Vec<f32> = (0..n).map(|i| {
                let o=off+i*4; f32::from_le_bytes([data[o],data[o+1],data[o+2],data[o+3]])
            }).collect(); off += n*4; v
        };
        let sp_w=rv(ns*CTX); let sp_fb=rv(ns*k_dn); let sp_bias=rv(ns);
        let sp_c=rv(ns); let sp_rho=rv(ns);
        let dn_sp=rv(nd*k_sp); let dn_dn=rv(nd*nd); let dn_bias=rv(nd);
        let dn_c=rv(nd); let dn_rho=rv(nd);
        let out_w=rv(N_BYTES*nd); let out_bias=rv(N_BYTES);

        let (ln_gamma, ln_beta) = if has_ln {
            (rv(nd), rv(nd))
        } else {
            (vec![1.0; nd], vec![0.0; nd])
        };

        Some(ResidualSandwich {
            n_sparse:ns, n_dense:nd, k, ticks, residual, use_ln: has_ln,
            sp_w,sp_fb,sp_bias,sp_c,sp_rho,sp_byte,sp_reads_dn,
            dn_sp,dn_dn,dn_bias,dn_c,dn_rho,dn_reads_sp,
            out_w,out_bias,ln_gamma,ln_beta,
        })
    }

    // ---- Training ----

    fn train(&mut self, train: &[(Vec<u8>,u8)], test: &[(Vec<u8>,u8)],
             n_steps: usize, batch_size: usize, ckpt_prefix: &str) {
        let mut lr = 0.01f32;
        let mut rng = StdRng::seed_from_u64(99);
        let (ns,nd) = (self.n_sparse, self.n_dense);
        let k_dn = self.k.min(nd); let k_sp = self.k.min(ns);
        let mut shuffled = train.to_vec();

        for step in 0..n_steps {
            shuffled.shuffle(&mut rng);
            let batch = &shuffled[..batch_size.min(shuffled.len())];

            let mut grad = Grad::zeros(ns, nd, k_dn, k_sp);
            for (ctx,target) in batch {
                let cache = self.forward(ctx);
                let g = self.backward(&cache, *target);
                grad.add(&g);
            }
            grad.scale(1.0 / batch.len() as f32);
            let gn = grad.norm();
            if gn < 1e-8 { continue; }
            grad.scale(1.0 / gn);

            let old_loss = self.loss(batch);
            let old_net = self.clone();
            self.apply_grad(&grad, lr);
            if self.loss(batch) < old_loss { lr *= 1.05; } else { *self = old_net; lr *= 0.5; }

            if step % 500 == 0 || step == n_steps-1 {
                let tr = self.accuracy(train); let te = self.accuracy(test);
                let t5 = self.top5(test); let lo = self.loss(test);
                println!("    step {:>5}: loss={:.3} train={:.1}% test={:.1}% top5={:.1}% lr={:.5}",
                    step, lo, tr*100.0, te*100.0, t5*100.0, lr);
            }
            if step > 0 && step % 1000 == 0 {
                let p = format!("{}_{}.bin", ckpt_prefix, step);
                self.save_checkpoint(&p);
                println!("    [ckpt: {}]", p);
            }
        }
        let p = format!("{}_final.bin", ckpt_prefix);
        self.save_checkpoint(&p);
        println!("    [ckpt: {}]", p);
    }

    // ---- Telemetry ----

    fn telemetry(&self, data: &[(Vec<u8>, u8)], n_samples: usize) {
        let samples = &data[..n_samples.min(data.len())];
        let t = self.ticks;
        let n = samples.len() as f32;

        let mut state_norms = vec![0.0f32; t];
        let mut corr_norms = vec![0.0f32; t];
        let mut sp_active = vec![0.0f32; t];
        let mut dn_mean = vec![0.0f32; t];
        let mut inter_correct = vec![0u32; t];
        let mut example_states: Vec<Vec<f32>> = Vec::new();

        for (si, (ctx, target)) in samples.iter().enumerate() {
            let cache = self.forward(ctx);
            if si == 0 { example_states = cache.state.clone(); }
            for tick in 0..t {
                let sn: f32 = cache.state[tick+1].iter().map(|v| v*v).sum::<f32>().sqrt();
                state_norms[tick] += sn;
                let cn: f32 = cache.correction[tick].iter().map(|v| v*v).sum::<f32>().sqrt();
                corr_norms[tick] += cn;
                let active = cache.sp_act[tick].iter().filter(|&&v| v.abs() > 0.01).count();
                sp_active[tick] += active as f32 / self.n_sparse as f32;
                dn_mean[tick] += cache.state[tick+1].iter().map(|v| v.abs()).sum::<f32>() / self.n_dense as f32;
                let pred = self.predict_from_state(&cache.state[tick+1]);
                if pred == *target as usize { inter_correct[tick] += 1; }
            }
        }

        let mode = match (self.residual, self.use_ln) {
            (false, _) => "standard",
            (true, false) => "residual",
            (true, true) => "res+LN",
        };
        println!("\n  TELEMETRY ({} samples, {}=t{}):", samples.len(), mode, self.ticks);
        println!("  {:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
                 "tick", "state_L2", "corr_L2", "corr/st", "sp_act%", "|dn|_mean", "inter_acc");
        println!("  {}", "-".repeat(70));

        for tick in 0..t {
            let sn = state_norms[tick]/n;
            let cn = corr_norms[tick]/n;
            let ratio = if sn > 1e-6 { cn/sn } else { 0.0 };
            println!("  {:>4} {:>10.4} {:>10.4} {:>10.3} {:>9.1}% {:>10.4} {:>9.1}%",
                tick+1, sn, cn, ratio,
                sp_active[tick]/n*100.0, dn_mean[tick]/n,
                inter_correct[tick] as f32/n*100.0);
        }

        if t > 1 {
            let show = 8.min(self.n_dense);
            println!("\n  Example dense state (first {} neurons, sample 0):", show);
            print!("  {:>6}", "");
            for d in 0..show { print!("  dn{:<4}", d); }
            println!();
            for tick in 0..=t {
                print!("  t={:<3}", tick);
                for d in 0..show { print!("  {:>+6.3}", example_states[tick][d]); }
                println!();
            }
        }
    }
}

fn bigram_baseline(train: &[(Vec<u8>,u8)], test: &[(Vec<u8>,u8)]) -> (f64,f64) {
    let mut counts = vec![vec![0u32; N_BYTES]; N_BYTES];
    for (ctx,t) in train {
        let last = *ctx.last().unwrap() as usize;
        if last < N_BYTES { counts[last][*t as usize] += 1; }
    }
    let predict = |l:usize| counts[l].iter().enumerate().max_by_key(|(_,&c)|c).map(|(i,_)|i).unwrap_or(32);
    let tr = train.iter().filter(|(c,t)| predict(*c.last().unwrap() as usize)==*t as usize).count() as f64/train.len() as f64;
    let te = test.iter().filter(|(c,t)| predict(*c.last().unwrap() as usize)==*t as usize).count() as f64/test.len() as f64;
    (tr, te)
}

// ============================================================

fn main() {
    println!("=== RESIDUAL SANDWICH: depth via residual ticks + LayerNorm ===\n");
    let t0 = Instant::now();
    let ckpt_dir = "checkpoints/residual_sandwich";

    // Load text
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

    let (_, bi_te) = bigram_baseline(&train, &test);
    println!("  Bigram: test={:.1}%", bi_te*100.0);
    println!("  Random: {:.2}%\n", 100.0/N_BYTES as f64);

    let ns = 256; let nd = 48; let k = 8;

    // Collect all results
    let mut results: Vec<(String, f64, f64)> = Vec::new();

    // =========================================================
    // Load previous results from checkpoints (already trained)
    // =========================================================
    println!("=== Loading previous checkpoints ===\n");

    // Baseline t=1
    if let Some(net) = ResidualSandwich::load_checkpoint(
        &format!("{}/base_t1_final.bin", ckpt_dir)) {
        let acc = net.accuracy(&test); let t5 = net.top5(&test);
        println!("  base_t1:  test={:.1}% top5={:.1}% (from checkpoint)", acc*100.0, t5*100.0);
        results.push(("base_t1".into(), acc, t5));
    }

    // Non-residual t=2,4
    for &ticks in &[2, 4] {
        let label = format!("std_t{}", ticks);
        if let Some(net) = ResidualSandwich::load_checkpoint(
            &format!("{}/{}_final.bin", ckpt_dir, label)) {
            let acc = net.accuracy(&test); let t5 = net.top5(&test);
            println!("  {}:  test={:.1}% top5={:.1}% (from checkpoint)", label, acc*100.0, t5*100.0);
            results.push((label, acc, t5));
        }
    }

    // Residual t=2,4 (no LN)
    for &ticks in &[2, 4] {
        let label = format!("res_t{}", ticks);
        if let Some(net) = ResidualSandwich::load_checkpoint(
            &format!("{}/{}_final.bin", ckpt_dir, label)) {
            let acc = net.accuracy(&test); let t5 = net.top5(&test);
            println!("  {}:  test={:.1}% top5={:.1}% (from checkpoint)", label, acc*100.0, t5*100.0);
            results.push((label, acc, t5));
        }
    }
    println!();

    // =========================================================
    // NEW: Residual + LayerNorm t=2,4
    // =========================================================
    for &ticks in &[2, 4, 8] {
        let label = format!("res_ln_t{}", ticks);
        println!("=== NEW: Residual + LayerNorm t={} ===", ticks);
        let ckpt = format!("{}/{}", ckpt_dir, label);

        let net = if let Some(n) = ResidualSandwich::load_checkpoint(
            &format!("{}_final.bin", ckpt)) {
            println!("  Loaded from checkpoint");
            n
        } else {
            let mut n = ResidualSandwich::new(ns, nd, k, ticks, true, true, 42, &mut rng);
            println!("  Params: {} (including LN: {})", n.param_count(), nd*2);
            n.train(&train, &test, 5000, 200, &ckpt);
            n
        };
        let acc = net.accuracy(&test); let t5 = net.top5(&test);
        println!("  => test={:.1}% top5={:.1}%", acc*100.0, t5*100.0);
        net.telemetry(&test, 200);
        results.push((label, acc, t5));
        println!();
    }

    // =========================================================
    // FULL COMPARISON
    // =========================================================
    println!("=== FULL COMPARISON ===");
    println!("  {:>14} {:>8} {:>8}", "config", "test%", "top5%");
    println!("  {}", "=".repeat(35));
    for (label, acc, t5) in &results {
        println!("  {:>14} {:>7.1}% {:>7.1}%", label, acc*100.0, t5*100.0);
    }
    println!("  {:>14} {:>7.1}%", "bigram", bi_te*100.0);

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
    println!("\n=== DONE ===");
}
