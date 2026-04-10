//! Language prediction: Sparse-Dense Sandwich with backprop
//!
//! Sparse layer: byte detectors (reads CTX positions + K dense feedback)
//! Dense layer: holographic all-to-all processing (reads K sparse + neighbors)
//! Output: 128 softmax (reads dense activations)
//! Tick recurrence: sparse→dense repeated T times
//!
//! Comparison: sandwich vs original connectome architecture vs bigram
//!
//! Run: cargo run --example language_sandwich --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use std::time::Instant;

const CTX: usize = 4;
const N_BYTES: usize = 128;

fn c19_fwd(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01);
    let l = 6.0 * c;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let s = x / c;
    let n = s.floor();
    let t = s - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

fn c19_deriv(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01);
    let l = 6.0 * c;
    if x >= l || x <= -l { return 1.0; }
    let s = x / c;
    let n = s.floor();
    let t = s - n;
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
// Sandwich Network for Language
// ============================================================

/// Sparse→Dense sandwich with tick recurrence + 128 output softmax
#[derive(Clone)]
struct SandwichLangNet {
    n_sparse: usize,    // sparse layer size
    n_dense: usize,     // dense layer size
    k: usize,           // cross-layer connections
    ticks: usize,

    // Sparse layer: each neuron detects byte patterns
    // weights: [thermo:CTX] [dense_feedback:K_eff] [bias:1]
    sp_w: Vec<f32>,        // n_sparse * CTX
    sp_fb: Vec<f32>,       // n_sparse * k (dense feedback weights)
    sp_bias: Vec<f32>,     // n_sparse
    sp_c: Vec<f32>,        // n_sparse
    sp_rho: Vec<f32>,      // n_sparse
    sp_byte: Vec<usize>,   // which byte each sparse neuron detects

    // Dense layer: holographic processing
    // weights: [sparse_read:K_eff] [dense_neighbors:n_dense-1] [bias:1]
    dn_sp: Vec<f32>,       // n_dense * k (sparse read weights)
    dn_dn: Vec<f32>,       // n_dense * n_dense (all-to-all, upper triangle used)
    dn_bias: Vec<f32>,     // n_dense
    dn_c: Vec<f32>,        // n_dense
    dn_rho: Vec<f32>,      // n_dense

    // Output layer: reads from dense → 128 logits
    out_w: Vec<f32>,       // N_BYTES * n_dense
    out_bias: Vec<f32>,    // N_BYTES

    // Wiring: which K dense neurons each sparse reads, and vice versa
    sp_reads_dn: Vec<Vec<usize>>,  // per sparse neuron
    dn_reads_sp: Vec<Vec<usize>>,  // per dense neuron
}

struct SandwichGrad {
    sp_w: Vec<f32>, sp_fb: Vec<f32>, sp_bias: Vec<f32>, sp_rho: Vec<f32>,
    dn_sp: Vec<f32>, dn_dn: Vec<f32>, dn_bias: Vec<f32>, dn_rho: Vec<f32>,
    out_w: Vec<f32>, out_bias: Vec<f32>,
}

impl SandwichGrad {
    fn zeros(ns: usize, nd: usize, k_dn: usize, k_sp: usize) -> Self {
        SandwichGrad {
            sp_w: vec![0.0; ns*CTX], sp_fb: vec![0.0; ns*k_dn],
            sp_bias: vec![0.0; ns], sp_rho: vec![0.0; ns],
            dn_sp: vec![0.0; nd*k_sp], dn_dn: vec![0.0; nd*nd],
            dn_bias: vec![0.0; nd], dn_rho: vec![0.0; nd],
            out_w: vec![0.0; N_BYTES*nd], out_bias: vec![0.0; N_BYTES],
        }
    }
    fn add(&mut self, o: &SandwichGrad) {
        for i in 0..self.sp_w.len() { self.sp_w[i]+=o.sp_w[i]; }
        for i in 0..self.sp_fb.len() { self.sp_fb[i]+=o.sp_fb[i]; }
        for i in 0..self.sp_bias.len() { self.sp_bias[i]+=o.sp_bias[i]; }
        for i in 0..self.sp_rho.len() { self.sp_rho[i]+=o.sp_rho[i]; }
        for i in 0..self.dn_sp.len() { self.dn_sp[i]+=o.dn_sp[i]; }
        for i in 0..self.dn_dn.len() { self.dn_dn[i]+=o.dn_dn[i]; }
        for i in 0..self.dn_bias.len() { self.dn_bias[i]+=o.dn_bias[i]; }
        for i in 0..self.dn_rho.len() { self.dn_rho[i]+=o.dn_rho[i]; }
        for i in 0..self.out_w.len() { self.out_w[i]+=o.out_w[i]; }
        for i in 0..self.out_bias.len() { self.out_bias[i]+=o.out_bias[i]; }
    }
    fn scale(&mut self, s: f32) {
        for v in &mut self.sp_w { *v*=s; } for v in &mut self.sp_fb { *v*=s; }
        for v in &mut self.sp_bias { *v*=s; } for v in &mut self.sp_rho { *v*=s; }
        for v in &mut self.dn_sp { *v*=s; } for v in &mut self.dn_dn { *v*=s; }
        for v in &mut self.dn_bias { *v*=s; } for v in &mut self.dn_rho { *v*=s; }
        for v in &mut self.out_w { *v*=s; } for v in &mut self.out_bias { *v*=s; }
    }
    fn norm(&self) -> f32 {
        let mut s = 0.0f32;
        for v in &self.sp_w { s+=v*v; } for v in &self.sp_fb { s+=v*v; }
        for v in &self.sp_bias { s+=v*v; } for v in &self.sp_rho { s+=v*v; }
        for v in &self.dn_sp { s+=v*v; } for v in &self.dn_dn { s+=v*v; }
        for v in &self.dn_bias { s+=v*v; } for v in &self.dn_rho { s+=v*v; }
        for v in &self.out_w { s+=v*v; } for v in &self.out_bias { s+=v*v; }
        s.sqrt()
    }
}

struct FwdCache {
    sp_sum: Vec<Vec<f32>>,    // [tick][neuron] pre-activation
    sp_act: Vec<Vec<f32>>,    // [tick][neuron] post-activation
    dn_sum: Vec<Vec<f32>>,    // [tick][neuron]
    dn_act: Vec<Vec<f32>>,    // [tick][neuron]
    thermos: Vec<[f32; CTX]>, // per sparse neuron
    logits: Vec<f32>,
    probs: Vec<f32>,
}

impl SandwichLangNet {
    fn new(n_sparse: usize, n_dense: usize, k: usize, ticks: usize, rng: &mut StdRng) -> Self {
        let sc = 0.1f32;
        let k_dn = k.min(n_dense);
        let k_sp = k.min(n_sparse);

        // Assign each sparse neuron a byte to detect (round-robin)
        let sp_byte: Vec<usize> = (0..n_sparse).map(|i| i % N_BYTES).collect();

        // Random cross-layer wiring
        let sp_reads_dn: Vec<Vec<usize>> = (0..n_sparse).map(|i| {
            let mut idx: Vec<usize> = (0..n_dense).collect();
            let mut r = StdRng::seed_from_u64(i as u64 * 31 + 7);
            idx.shuffle(&mut r);
            idx[..k_dn].to_vec()
        }).collect();

        let dn_reads_sp: Vec<Vec<usize>> = (0..n_dense).map(|i| {
            let mut idx: Vec<usize> = (0..n_sparse).collect();
            let mut r = StdRng::seed_from_u64(i as u64 * 37 + 13);
            idx.shuffle(&mut r);
            idx[..k_sp].to_vec()
        }).collect();

        SandwichLangNet {
            n_sparse, n_dense, k, ticks,
            sp_w: (0..n_sparse*CTX).map(|_| rng.gen_range(-sc..sc)).collect(),
            sp_fb: (0..n_sparse*k_dn).map(|_| rng.gen_range(-sc..sc)).collect(),
            sp_bias: (0..n_sparse).map(|_| rng.gen_range(-sc..sc)).collect(),
            sp_c: vec![1.0; n_sparse],
            sp_rho: vec![4.0; n_sparse],
            sp_byte,
            dn_sp: (0..n_dense*k_sp).map(|_| rng.gen_range(-sc..sc)).collect(),
            dn_dn: (0..n_dense*n_dense).map(|_| rng.gen_range(-sc..sc)).collect(),
            dn_bias: (0..n_dense).map(|_| rng.gen_range(-sc..sc)).collect(),
            dn_c: vec![1.0; n_dense],
            dn_rho: vec![4.0; n_dense],
            out_w: (0..N_BYTES*n_dense).map(|_| rng.gen_range(-sc..sc)).collect(),
            out_bias: (0..N_BYTES).map(|_| rng.gen_range(-sc..sc)).collect(),
            sp_reads_dn, dn_reads_sp,
        }
    }

    fn forward(&self, context: &[u8]) -> FwdCache {
        let ns = self.n_sparse;
        let nd = self.n_dense;
        let k_dn = self.k.min(nd);
        let k_sp = self.k.min(ns);

        // Encode thermo per sparse neuron
        let thermos: Vec<[f32; CTX]> = (0..ns).map(|i| encode_ctx(context, self.sp_byte[i])).collect();

        let mut sp_sum_all = Vec::new();
        let mut sp_act_all = Vec::new();
        let mut dn_sum_all = Vec::new();
        let mut dn_act_all = Vec::new();

        let mut sp_act = vec![0.0f32; ns];
        let mut dn_act = vec![0.0f32; nd];

        for _t in 0..self.ticks {
            // Sparse fires
            let mut sp_sum = vec![0.0f32; ns];
            for i in 0..ns {
                let mut s = self.sp_bias[i];
                for j in 0..CTX { s += thermos[i][j] * self.sp_w[i*CTX+j]; }
                for (ki, &di) in self.sp_reads_dn[i].iter().enumerate() {
                    s += dn_act[di] * self.sp_fb[i*k_dn+ki];
                }
                sp_sum[i] = s;
                sp_act[i] = c19_fwd(s, self.sp_c[i], self.sp_rho[i]);
            }
            sp_sum_all.push(sp_sum);
            sp_act_all.push(sp_act.clone());

            // Dense fires
            let old_dn = dn_act.clone();
            let mut dn_sum = vec![0.0f32; nd];
            for i in 0..nd {
                let mut s = self.dn_bias[i];
                for (ki, &si) in self.dn_reads_sp[i].iter().enumerate() {
                    s += sp_act[si] * self.dn_sp[i*k_sp+ki];
                }
                for j in 0..nd {
                    if j != i { s += old_dn[j] * self.dn_dn[i*nd+j]; }
                }
                dn_sum[i] = s;
                dn_act[i] = c19_fwd(s, self.dn_c[i], self.dn_rho[i]);
            }
            dn_sum_all.push(dn_sum);
            dn_act_all.push(dn_act.clone());
        }

        // Output logits
        let mut logits = vec![0.0f32; N_BYTES];
        for b in 0..N_BYTES {
            let mut s = self.out_bias[b];
            for d in 0..nd { s += dn_act[d] * self.out_w[b*nd+d]; }
            logits[b] = s;
        }
        let probs = softmax(&logits);

        FwdCache { sp_sum: sp_sum_all, sp_act: sp_act_all, dn_sum: dn_sum_all, dn_act: dn_act_all, thermos, logits, probs }
    }

    fn backward(&self, cache: &FwdCache, target: u8) -> SandwichGrad {
        let ns = self.n_sparse;
        let nd = self.n_dense;
        let k_dn = self.k.min(nd);
        let k_sp = self.k.min(ns);
        let mut g = SandwichGrad::zeros(ns, nd, k_dn, k_sp);

        // Step 7→5: d_logits = probs - one_hot(target)
        let mut d_logits = cache.probs.clone();
        d_logits[target as usize] -= 1.0;

        // Step 5: output layer gradients
        let last_dn = &cache.dn_act[self.ticks - 1];
        let mut d_dn_act = vec![0.0f32; nd];
        for b in 0..N_BYTES {
            g.out_bias[b] += d_logits[b];
            for d in 0..nd {
                g.out_w[b * nd + d] += d_logits[b] * last_dn[d];
                d_dn_act[d] += d_logits[b] * self.out_w[b * nd + d];
            }
        }

        // BPTT: unroll ticks backwards
        for t in (0..self.ticks).rev() {
            // d_dn_act → d_dn_sum (through C19 derivative)
            let mut d_dn_sum = vec![0.0f32; nd];
            for i in 0..nd {
                let deriv = c19_deriv(cache.dn_sum[t][i], self.dn_c[i], self.dn_rho[i]);
                d_dn_sum[i] = d_dn_act[i] * deriv;
                // rho gradient
                let s = cache.dn_sum[t][i];
                let c = self.dn_c[i].max(0.01);
                let l = 6.0 * c;
                if s > -l && s < l {
                    let scaled = s / c;
                    let ft = scaled - scaled.floor();
                    let h = ft * (1.0 - ft);
                    g.dn_rho[i] += d_dn_act[i] * c * h * h;
                }
            }

            // d_dn_sum → sparse act gradients + dense neighbor gradients
            let mut d_sp_act = vec![0.0f32; ns];
            let mut d_old_dn = vec![0.0f32; nd]; // gradient flowing to previous tick's dense

            for i in 0..nd {
                g.dn_bias[i] += d_dn_sum[i];
                // sparse read weights
                for (ki, &si) in self.dn_reads_sp[i].iter().enumerate() {
                    g.dn_sp[i * k_sp + ki] += d_dn_sum[i] * cache.sp_act[t][si];
                    d_sp_act[si] += d_dn_sum[i] * self.dn_sp[i * k_sp + ki];
                }
                // dense neighbor weights (from previous tick snapshot)
                for j in 0..nd {
                    if j != i {
                        let old_dn_j = if t > 0 { cache.dn_act[t-1][j] } else { 0.0 };
                        g.dn_dn[i * nd + j] += d_dn_sum[i] * old_dn_j;
                        d_old_dn[j] += d_dn_sum[i] * self.dn_dn[i * nd + j];
                    }
                }
            }

            // d_sp_act → d_sp_sum (through C19 derivative)
            let mut d_sp_sum = vec![0.0f32; ns];
            for i in 0..ns {
                let deriv = c19_deriv(cache.sp_sum[t][i], self.sp_c[i], self.sp_rho[i]);
                d_sp_sum[i] = d_sp_act[i] * deriv;
                // rho gradient
                let s = cache.sp_sum[t][i];
                let c = self.sp_c[i].max(0.01);
                let l = 6.0 * c;
                if s > -l && s < l {
                    let scaled = s / c;
                    let ft = scaled - scaled.floor();
                    let h = ft * (1.0 - ft);
                    g.sp_rho[i] += d_sp_act[i] * c * h * h;
                }
            }

            // d_sp_sum → input weights + feedback weights + d_dn_act for feedback
            let mut d_dn_feedback = vec![0.0f32; nd];
            for i in 0..ns {
                g.sp_bias[i] += d_sp_sum[i];
                for j in 0..CTX {
                    g.sp_w[i * CTX + j] += d_sp_sum[i] * cache.thermos[i][j];
                }
                // feedback weights: sparse reads from dense (previous tick's dn_act)
                for (ki, &di) in self.sp_reads_dn[i].iter().enumerate() {
                    let fb_dn = if t > 0 { cache.dn_act[t-1][di] } else { 0.0 };
                    g.sp_fb[i * k_dn + ki] += d_sp_sum[i] * fb_dn;
                    d_dn_feedback[di] += d_sp_sum[i] * self.sp_fb[i * k_dn + ki];
                }
            }

            // Propagate gradient to previous tick's dn_act
            if t > 0 {
                d_dn_act = vec![0.0f32; nd];
                for i in 0..nd {
                    d_dn_act[i] += d_old_dn[i] + d_dn_feedback[i];
                }
            }
        }

        g
    }

    fn train_analytic(&mut self, train: &[(Vec<u8>, u8)], test: &[(Vec<u8>, u8)],
                       n_steps: usize, batch_size: usize) {
        let mut lr = 0.01f32;
        let mut rng = StdRng::seed_from_u64(99);
        let mut shuffled = train.to_vec();
        let ns = self.n_sparse;
        let nd = self.n_dense;
        let k_dn = self.k.min(nd);
        let k_sp = self.k.min(ns);

        for step in 0..n_steps {
            shuffled.shuffle(&mut rng);
            let batch = &shuffled[..batch_size.min(shuffled.len())];

            let mut grad = SandwichGrad::zeros(ns, nd, k_dn, k_sp);
            for (ctx, target) in batch {
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

            if self.loss(batch) < old_loss { lr *= 1.05; }
            else { *self = old_net; lr *= 0.5; }

            if step % 500 == 0 || step == n_steps - 1 {
                let tr = self.accuracy(train);
                let te = self.accuracy(test);
                let t5 = self.top5(test);
                let lo = self.loss(test);
                println!("    step {:>5}: loss={:.3} train={:.1}% test={:.1}% top5={:.1}% lr={:.5}",
                    step, lo, tr*100.0, te*100.0, t5*100.0, lr);
            }
        }
    }

    fn apply_grad(&mut self, g: &SandwichGrad, lr: f32) {
        for i in 0..self.sp_w.len() { self.sp_w[i] -= lr * g.sp_w[i]; }
        for i in 0..self.sp_fb.len() { self.sp_fb[i] -= lr * g.sp_fb[i]; }
        for i in 0..self.sp_bias.len() { self.sp_bias[i] -= lr * g.sp_bias[i]; }
        for i in 0..self.sp_rho.len() { self.sp_rho[i] = (self.sp_rho[i] - lr * g.sp_rho[i]).max(0.0); }
        for i in 0..self.dn_sp.len() { self.dn_sp[i] -= lr * g.dn_sp[i]; }
        for i in 0..self.dn_dn.len() { self.dn_dn[i] -= lr * g.dn_dn[i]; }
        for i in 0..self.dn_bias.len() { self.dn_bias[i] -= lr * g.dn_bias[i]; }
        for i in 0..self.dn_rho.len() { self.dn_rho[i] = (self.dn_rho[i] - lr * g.dn_rho[i]).max(0.0); }
        for i in 0..self.out_w.len() { self.out_w[i] -= lr * g.out_w[i]; }
        for i in 0..self.out_bias.len() { self.out_bias[i] -= lr * g.out_bias[i]; }
    }

    fn predict(&self, ctx: &[u8]) -> u8 {
        let c = self.forward(ctx);
        c.probs.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i,_)| i as u8).unwrap_or(32)
    }

    fn accuracy(&self, data: &[(Vec<u8>, u8)]) -> f64 {
        data.iter().filter(|(c,t)| self.predict(c)==*t).count() as f64 / data.len() as f64
    }

    fn top5(&self, data: &[(Vec<u8>, u8)]) -> f64 {
        let mut ok = 0;
        for (ctx, t) in data {
            let c = self.forward(ctx);
            let mut idx: Vec<(usize,f32)> = c.probs.iter().enumerate().map(|(i,&v)|(i,v)).collect();
            idx.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
            if idx.iter().take(5).any(|(i,_)| *i==*t as usize) { ok += 1; }
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
        self.sp_w.len() + self.sp_fb.len() + self.sp_bias.len() + self.sp_c.len() + self.sp_rho.len()
        + self.dn_sp.len() + self.dn_dn.len() + self.dn_bias.len() + self.dn_c.len() + self.dn_rho.len()
        + self.out_w.len() + self.out_bias.len()
    }

    /// Numerical gradient training (simple but works for small nets)
    fn train_numerical(&mut self, train: &[(Vec<u8>, u8)], test: &[(Vec<u8>, u8)],
                       n_steps: usize, batch_size: usize) {
        let eps = 1e-3f32;
        let mut lr = 0.01f32;
        let mut rng = StdRng::seed_from_u64(99);
        let mut shuffled = train.to_vec();

        // Collect all params as mutable slices
        let n_params = self.param_count();
        println!("    Params: {}", n_params);

        for step in 0..n_steps {
            shuffled.shuffle(&mut rng);
            let batch = &shuffled[..batch_size.min(shuffled.len())];

            // Compute gradient numerically on a SUBSET of params
            // (full numerical gradient is too slow for all params)
            let base_loss = self.loss(batch);

            // Gather all params into flat vec
            let mut params = Vec::with_capacity(n_params);
            params.extend_from_slice(&self.sp_w);
            params.extend_from_slice(&self.sp_fb);
            params.extend_from_slice(&self.sp_bias);
            params.extend_from_slice(&self.sp_c);
            params.extend_from_slice(&self.sp_rho);
            params.extend_from_slice(&self.dn_sp);
            params.extend_from_slice(&self.dn_dn);
            params.extend_from_slice(&self.dn_bias);
            params.extend_from_slice(&self.dn_c);
            params.extend_from_slice(&self.dn_rho);
            params.extend_from_slice(&self.out_w);
            params.extend_from_slice(&self.out_bias);

            let mut grad = vec![0.0f32; n_params];

            // Stochastic param sampling: pick random subset to update
            let sample_size = 200.min(n_params);
            let mut indices: Vec<usize> = (0..n_params).collect();
            indices.shuffle(&mut rng);

            for &pi in indices[..sample_size].iter() {
                let orig = params[pi];
                self.set_param(pi, orig + eps);
                let lp = self.loss(batch);
                self.set_param(pi, orig - eps);
                let lm = self.loss(batch);
                self.set_param(pi, orig);
                grad[pi] = ((lp - lm) / (2.0 * eps as f64)) as f32;
            }

            // Normalize
            let gn: f32 = grad.iter().map(|g| g*g).sum::<f32>().sqrt();
            if gn < 1e-8 { continue; }

            // Apply
            let old = self.clone();
            for &pi in indices[..sample_size].iter() {
                let v = self.get_param(pi) - lr * grad[pi] / gn;
                self.set_param(pi, v);
            }

            // Clamp C and rho
            for i in 0..self.n_sparse { self.sp_c[i] = self.sp_c[i].max(0.01); self.sp_rho[i] = self.sp_rho[i].max(0.0); }
            for i in 0..self.n_dense { self.dn_c[i] = self.dn_c[i].max(0.01); self.dn_rho[i] = self.dn_rho[i].max(0.0); }

            let new_loss = self.loss(batch);
            if new_loss < base_loss { lr *= 1.05; }
            else { *self = old; lr *= 0.5; }

            if step % 100 == 0 || step == n_steps - 1 {
                let tr_acc = self.accuracy(train);
                let te_acc = self.accuracy(test);
                let t5 = self.top5(test);
                let te_loss = self.loss(test);
                println!("    step {:>5}: loss={:.3} train={:.1}% test={:.1}% top5={:.1}% lr={:.5}",
                    step, te_loss, tr_acc*100.0, te_acc*100.0, t5*100.0, lr);
            }
        }
    }

    fn get_param(&self, idx: usize) -> f32 {
        let mut i = idx;
        if i < self.sp_w.len() { return self.sp_w[i]; } i -= self.sp_w.len();
        if i < self.sp_fb.len() { return self.sp_fb[i]; } i -= self.sp_fb.len();
        if i < self.sp_bias.len() { return self.sp_bias[i]; } i -= self.sp_bias.len();
        if i < self.sp_c.len() { return self.sp_c[i]; } i -= self.sp_c.len();
        if i < self.sp_rho.len() { return self.sp_rho[i]; } i -= self.sp_rho.len();
        if i < self.dn_sp.len() { return self.dn_sp[i]; } i -= self.dn_sp.len();
        if i < self.dn_dn.len() { return self.dn_dn[i]; } i -= self.dn_dn.len();
        if i < self.dn_bias.len() { return self.dn_bias[i]; } i -= self.dn_bias.len();
        if i < self.dn_c.len() { return self.dn_c[i]; } i -= self.dn_c.len();
        if i < self.dn_rho.len() { return self.dn_rho[i]; } i -= self.dn_rho.len();
        if i < self.out_w.len() { return self.out_w[i]; } i -= self.out_w.len();
        self.out_bias[i]
    }

    fn set_param(&mut self, idx: usize, val: f32) {
        let mut i = idx;
        if i < self.sp_w.len() { self.sp_w[i] = val; return; } i -= self.sp_w.len();
        if i < self.sp_fb.len() { self.sp_fb[i] = val; return; } i -= self.sp_fb.len();
        if i < self.sp_bias.len() { self.sp_bias[i] = val; return; } i -= self.sp_bias.len();
        if i < self.sp_c.len() { self.sp_c[i] = val; return; } i -= self.sp_c.len();
        if i < self.sp_rho.len() { self.sp_rho[i] = val; return; } i -= self.sp_rho.len();
        if i < self.dn_sp.len() { self.dn_sp[i] = val; return; } i -= self.dn_sp.len();
        if i < self.dn_dn.len() { self.dn_dn[i] = val; return; } i -= self.dn_dn.len();
        if i < self.dn_bias.len() { self.dn_bias[i] = val; return; } i -= self.dn_bias.len();
        if i < self.dn_c.len() { self.dn_c[i] = val; return; } i -= self.dn_c.len();
        if i < self.dn_rho.len() { self.dn_rho[i] = val; return; } i -= self.dn_rho.len();
        if i < self.out_w.len() { self.out_w[i] = val; return; } i -= self.out_w.len();
        self.out_bias[i] = val;
    }
}

fn bigram_baseline(train: &[(Vec<u8>, u8)], test: &[(Vec<u8>, u8)]) -> (f64, f64) {
    let mut counts = vec![vec![0u32; N_BYTES]; N_BYTES];
    for (ctx, t) in train {
        let last = *ctx.last().unwrap() as usize;
        if last < N_BYTES { counts[last][*t as usize] += 1; }
    }
    let predict = |l: usize| counts[l].iter().enumerate().max_by_key(|(_,&c)|c).map(|(i,_)|i).unwrap_or(32);
    let tr = train.iter().filter(|(c,t)| predict(*c.last().unwrap() as usize)==*t as usize).count() as f64/train.len() as f64;
    let te = test.iter().filter(|(c,t)| predict(*c.last().unwrap() as usize)==*t as usize).count() as f64/test.len() as f64;
    (tr, te)
}

fn main() {
    println!("=== LANGUAGE SANDWICH: sparse→dense + tick recurrence ===\n");

    let t0 = Instant::now();

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
        match o { Ok(o) if o.stdout.len() > 1000 => { println!("  {} bytes", o.stdout.len()); o.stdout }
            _ => { println!("  Fallback"); "the quick brown fox jumps over the lazy dog ".repeat(500).bytes().collect() } }
    };
    let text: Vec<u8> = raw.iter().map(|&b| if b < 128 { b } else { 32 }).collect();

    let mut pairs: Vec<(Vec<u8>, u8)> = Vec::new();
    for i in CTX..text.len() {
        let ctx = text[i-CTX..i].to_vec();
        if (text[i] as usize) < N_BYTES { pairs.push((ctx, text[i])); }
    }
    let mut rng = StdRng::seed_from_u64(42);
    pairs.shuffle(&mut rng);
    let train = pairs[..2000].to_vec();
    let test = pairs[2000..3000].to_vec();
    println!("  Train: {}, Test: {}\n", train.len(), test.len());

    let (bi_tr, bi_te) = bigram_baseline(&train, &test);
    println!("  Bigram: train={:.1}% test={:.1}%", bi_tr*100.0, bi_te*100.0);
    println!("  Random: {:.2}%\n", 100.0/N_BYTES as f64);

    // =========================================================
    // Config 1: Small (128S+16D, K=4, t=2) — analytic backprop
    // =========================================================
    println!("--- Config 1: 128S+16D, K=4, ticks=2, ANALYTIC backprop ---");
    {
        let mut net = SandwichLangNet::new(128, 16, 4, 2, &mut rng);
        println!("  Params: {}", net.param_count());
        net.train_analytic(&train, &test, 5000, 200);

        println!("\n  Predictions:");
        for s in &["the ", "and ", "is a", "tion", "ing ", "for ", "ent ", "hat "] {
            let ctx: Vec<u8> = s.bytes().collect();
            let pred = net.predict(&ctx);
            let ch = if pred >= 32 && pred < 127 { pred as char } else { '?' };
            print!("'{}'->'{}'  ", s, ch);
        }
        println!("\n");
    }

    // =========================================================
    // Config 2: Bigger (128S+32D, K=8, t=2)
    // =========================================================
    println!("--- Config 2: 128S+32D, K=8, ticks=2 ---");
    {
        let mut net = SandwichLangNet::new(128, 32, 8, 2, &mut rng);
        println!("  Params: {}", net.param_count());
        net.train_analytic(&train, &test, 5000, 200);

        println!("\n  Predictions:");
        for s in &["the ", "and ", "is a", "tion", "ing ", "for ", "ent ", "hat "] {
            let ctx: Vec<u8> = s.bytes().collect();
            let pred = net.predict(&ctx);
            let ch = if pred >= 32 && pred < 127 { pred as char } else { '?' };
            print!("'{}'->'{}'  ", s, ch);
        }
        println!("\n");
    }

    // =========================================================
    // Config 3: Tick sweep (128S+32D, K=8, t=1,2,4)
    // =========================================================
    println!("--- Config 3: Tick sweep (128S+32D, K=8) ---\n");
    for &ticks in &[1, 2, 4] {
        let mut net = SandwichLangNet::new(128, 32, 8, ticks, &mut StdRng::seed_from_u64(42));
        println!("  ticks={}, params={}", ticks, net.param_count());
        net.train_analytic(&train, &test, 5000, 200);

        print!("  Predictions: ");
        for s in &["the ", "and ", "is a", "tion"] {
            let ctx: Vec<u8> = s.bytes().collect();
            let pred = net.predict(&ctx);
            let ch = if pred >= 32 && pred < 127 { pred as char } else { '?' };
            print!("'{}'->'{}'  ", s, ch);
        }
        println!("\n");
    }

    // =========================================================
    // Config 4: Train t=X, Inference t=Y cross-test
    // =========================================================
    println!("--- Config 4: Train vs Inference tick cross-test (128S+32D, K=8) ---\n");
    println!("  {:>10} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "train_t", "inf_t=1", "inf_t=2", "inf_t=4", "inf_t=8", "inf_t=16");
    println!("  {}", "=".repeat(50));

    for &train_ticks in &[1, 2, 4] {
        let mut net = SandwichLangNet::new(128, 32, 8, train_ticks, &mut StdRng::seed_from_u64(42));
        net.train_analytic(&train, &test, 5000, 200);

        print!("  {:>8}", format!("t={}", train_ticks));
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        for &inf_ticks in &[1, 2, 4, 8, 16] {
            net.ticks = inf_ticks;
            let acc = net.accuracy(&test);
            print!(" {:>7.1}%", acc * 100.0);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
        net.ticks = train_ticks; // restore
        println!();
    }

    // Top5 version
    println!("\n  Top5:");
    println!("  {:>10} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "train_t", "inf_t=1", "inf_t=2", "inf_t=4", "inf_t=8", "inf_t=16");
    println!("  {}", "=".repeat(50));

    for &train_ticks in &[1, 2, 4] {
        let mut net = SandwichLangNet::new(128, 32, 8, train_ticks, &mut StdRng::seed_from_u64(42));
        net.train_analytic(&train, &test, 5000, 200);

        print!("  {:>8}", format!("t={}", train_ticks));
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        for &inf_ticks in &[1, 2, 4, 8, 16] {
            net.ticks = inf_ticks;
            let t5 = net.top5(&test);
            print!(" {:>7.1}%", t5 * 100.0);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
        println!();
    }

    // =========================================================
    // Summary
    // =========================================================
    println!("\n--- SUMMARY ---");
    println!("  Previous best (language_backprop.rs connectome): C19 25.0% test, top5 62.7%");
    println!("  Bigram baseline: {:.1}%", bi_te * 100.0);
    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
    println!("\n=== DONE ===");
}
