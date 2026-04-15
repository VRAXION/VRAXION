//! L2 Conv — big context masked char prediction
//!
//! 1024 byte context, Conv stack → pool → MLP head
//! Sweep: depth (1-4 conv layers), swish activation
//! Compare with 16-byte MLP baseline (75%)
//!
//! Run: cargo run --example l2_conv_bigctx --release

use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

fn swish(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn swish_grad(x: f32) -> f32 { let s=1.0/(1.0+(-x).exp()); s+x*s*(1.0-s) }

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1)) }
    fn next(&mut self) -> u64 { self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.0 }
    fn normal(&mut self) -> f32 {
        let u1=(((self.next()>>33)%65536) as f32/65536.0).max(1e-7);
        let u2=((self.next()>>33)%65536) as f32/65536.0;
        (-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        if hi<=lo{lo}else{lo+(self.next() as usize%(hi-lo))}
    }
}

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = std::fs::read(path).expect("read");
    raw.iter().filter_map(|&b| match b {
        b'a'..=b'z' => Some(b-b'a'), b'A'..=b'Z' => Some(b-b'A'),
        b' '|b'\n'|b'\t'|b'\r' => Some(26), _ => None,
    }).collect()
}

// Multi-layer conv + pool + MLP head
struct ConvModel {
    // Conv layers: conv_w[layer][filter][kernel_pos * in_ch]
    conv_w: Vec<Vec<Vec<f32>>>,
    conv_b: Vec<Vec<f32>>,
    n_conv: usize,
    k: usize,
    channels: Vec<usize>, // channels per layer (input ch, then filter counts)
    // MLP head: pool_dim → hdim → 27
    hw: Vec<Vec<f32>>, hb: Vec<f32>,
    ow: Vec<Vec<f32>>, ob: Vec<f32>,
    hdim: usize,
    pool_dim: usize,
}

impl ConvModel {
    fn new(ctx: usize, in_ch: usize, k: usize, filters: &[usize], hdim: usize, rng: &mut Rng) -> Self {
        let mut conv_w = Vec::new();
        let mut conv_b = Vec::new();
        let mut channels = vec![in_ch];

        for (li, &nf) in filters.iter().enumerate() {
            let ch_in = channels[li];
            let fan = k * ch_in;
            let sc = (2.0/fan as f32).sqrt();
            conv_w.push((0..nf).map(|_|(0..fan).map(|_|rng.normal()*sc).collect()).collect());
            conv_b.push(vec![0.0; nf]);
            channels.push(nf);
        }

        let pool_dim = *filters.last().unwrap_or(&in_ch);
        let s1 = (2.0/pool_dim as f32).sqrt();
        let s2 = (2.0/hdim as f32).sqrt();

        ConvModel {
            conv_w, conv_b,
            n_conv: filters.len(), k, channels,
            hw: (0..hdim).map(|_|(0..pool_dim).map(|_|rng.normal()*s1).collect()).collect(),
            hb: vec![0.0; hdim],
            ow: (0..27).map(|_|(0..hdim).map(|_|rng.normal()*s2).collect()).collect(),
            ob: vec![0.0; 27],
            hdim, pool_dim,
        }
    }

    fn params(&self) -> usize {
        let conv_p: usize = self.conv_w.iter().zip(&self.conv_b)
            .map(|(w,b)| w.len()*w[0].len()+b.len()).sum();
        conv_p + self.pool_dim*self.hdim + self.hdim + self.hdim*27 + 27
    }

    fn forward(&self, input: &[f32], ctx: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // Conv layers
        let mut conv_pres: Vec<Vec<f32>> = Vec::new();
        let mut conv_posts: Vec<Vec<f32>> = Vec::new();
        let mut x = input.to_vec();
        let mut cur_len = ctx;
        let mut cur_ch = self.channels[0];

        for li in 0..self.n_conv {
            let nf = self.conv_w[li].len();
            let new_len = cur_len - self.k + 1;
            let mut pre = vec![0.0f32; new_len * nf];
            for p in 0..new_len {
                for f in 0..nf {
                    let mut v = self.conv_b[li][f];
                    for ki in 0..self.k {
                        for d in 0..cur_ch {
                            v += self.conv_w[li][f][ki*cur_ch+d] * x[(p+ki)*cur_ch+d];
                        }
                    }
                    pre[p*nf+f] = v;
                }
            }
            let post: Vec<f32> = pre.iter().map(|&v| swish(v)).collect();
            conv_pres.push(pre);
            conv_posts.push(post.clone());
            x = post;
            cur_len = new_len;
            cur_ch = nf;
        }

        // Global average pool → pool_dim
        let nf = cur_ch;
        let mut pool = vec![0.0f32; nf];
        for f in 0..nf {
            let mut s = 0.0f32;
            for p in 0..cur_len { s += x[p*nf+f]; }
            pool[f] = s / cur_len as f32;
        }

        // MLP head
        let mut h = vec![0.0f32; self.hdim];
        for i in 0..self.hdim {
            h[i] = self.hb[i];
            for j in 0..self.pool_dim { h[i] += self.hw[i][j]*pool[j]; }
            h[i] = swish(h[i]);
        }
        let mut logits = vec![0.0f32; 27];
        for c in 0..27 { logits[c] = self.ob[c];
            for i in 0..self.hdim { logits[c] += self.ow[c][i]*h[i]; } }

        (conv_pres, conv_posts, pool, h, logits)
    }

    fn train_step(&mut self, input: &[f32], ctx: usize, target: u8, lr: f32) -> f32 {
        let (conv_pres, conv_posts, pool, h, logits) = self.forward(input, ctx);

        // Softmax + CE
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut p = vec![0.0f32;27]; let mut s=0.0f32;
        for c in 0..27{p[c]=(logits[c]-mx).exp();s+=p[c];}
        for c in 0..27{p[c]/=s;}
        let loss = -(p[target as usize].max(1e-10).ln());
        p[target as usize] -= 1.0;

        // Backprop output layer
        let mut dh = vec![0.0f32; self.hdim];
        for c in 0..27 {
            for i in 0..self.hdim {
                dh[i] += p[c]*self.ow[c][i];
                self.ow[c][i] -= lr*p[c]*h[i];
            }
            self.ob[c] -= lr*p[c];
        }

        // Backprop hidden (swish)
        let mut dpool = vec![0.0f32; self.pool_dim];
        for i in 0..self.hdim {
            let pre_h = self.hb[i] + (0..self.pool_dim).map(|j| self.hw[i][j]*pool[j]).sum::<f32>();
            let g = dh[i] * swish_grad(pre_h);
            for j in 0..self.pool_dim {
                dpool[j] += g*self.hw[i][j];
                self.hw[i][j] -= lr*g*pool[j];
            }
            self.hb[i] -= lr*g;
        }

        // Backprop through pool → conv layers
        let mut cur_len = ctx;
        for _ in 0..self.n_conv { cur_len -= self.k - 1; }
        let nf = self.channels[self.n_conv];

        // dpool → d_conv_out (spread average)
        let inv_len = 1.0 / cur_len as f32;
        let mut dx = vec![0.0f32; cur_len * nf];
        for f in 0..nf {
            let g = dpool[f] * inv_len;
            for pp in 0..cur_len { dx[pp*nf+f] = g; }
        }

        // Apply swish grad to last conv layer output
        let last = self.n_conv - 1;
        for i in 0..dx.len() {
            dx[i] *= swish_grad(conv_pres[last][i]);
        }

        // Backprop conv layers (reverse)
        let mut all_inputs: Vec<&Vec<f32>> = Vec::new();
        // Reconstruct layer inputs
        let mut layer_input = input.to_vec();
        let mut layer_inputs = vec![input.to_vec()];
        let mut cl = ctx;
        for li in 0..self.n_conv {
            if li > 0 { layer_inputs.push(conv_posts[li-1].clone()); }
            cl -= self.k - 1;
        }

        for li in (0..self.n_conv).rev() {
            let ch_in = self.channels[li];
            let nf_cur = self.conv_w[li].len();
            let x_in = if li == 0 { input } else { &conv_posts[li-1] };
            let in_len = if li == 0 { ctx } else { ctx - li*(self.k-1) };
            let out_len = in_len - self.k + 1;

            let mut dx_prev = vec![0.0f32; in_len * ch_in];

            for pp in 0..out_len {
                for f in 0..nf_cur {
                    let dc = dx[pp*nf_cur+f];
                    if dc.abs() < 1e-12 { continue; }
                    for ki in 0..self.k {
                        for d in 0..ch_in {
                            let idx = (pp+ki)*ch_in+d;
                            if idx < x_in.len() {
                                self.conv_w[li][f][ki*ch_in+d] -= lr*dc*x_in[idx];
                                dx_prev[idx] += dc*self.conv_w[li][f][ki*ch_in+d];
                            }
                        }
                    }
                    self.conv_b[li][f] -= lr*dc;
                }
            }

            // Apply swish grad for previous layer (if not input)
            if li > 0 {
                for i in 0..dx_prev.len().min(conv_pres[li-1].len()) {
                    dx_prev[i] *= swish_grad(conv_pres[li-1][i]);
                }
            }

            dx = dx_prev;
        }

        loss
    }

    fn predict(&self, input: &[f32], ctx: usize) -> usize {
        let (_, _, _, _, logits) = self.forward(input, ctx);
        logits.iter().enumerate()
            .max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|v|v.0).unwrap_or(0)
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split = corpus.len() * 80 / 100;

    let encoded: Vec<[f32;2]> = corpus.iter().map(|&ch|
        [LUT[ch as usize][0] as f32 / 16.0, LUT[ch as usize][1] as f32 / 16.0]
    ).collect();

    let ctx = 1024usize;
    let mask_pos = ctx / 2;
    let mask_val = [1.0f32, 1.0];
    let in_ch = 2;

    println!("=== CONV BIG CONTEXT — 1024 bytes ===\n");

    struct Cfg { filters: Vec<usize>, k: usize, hdim: usize }
    let configs = vec![
        Cfg { filters: vec![32], k: 3, hdim: 64 },           // 1 conv
        Cfg { filters: vec![32, 32], k: 3, hdim: 64 },       // 2 conv
        Cfg { filters: vec![32, 32, 32], k: 3, hdim: 64 },   // 3 conv
        Cfg { filters: vec![32, 32, 32], k: 5, hdim: 64 },   // 3 conv k=5
    ];

    for cfg in &configs {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut model = ConvModel::new(ctx, in_ch, cfg.k, &cfg.filters, cfg.hdim, &mut rng);

        let rf = cfg.filters.len() * (cfg.k - 1) + 1; // receptive field
        println!("  --- {}×conv(k={},f={}) + mlp(h={}) | rf={} bytes | {} params ---\n",
            cfg.filters.len(), cfg.k, cfg.filters[0], cfg.hdim, rf, model.params());

        let samples = 1000;
        let mut best_test = 0.0f64;
        let mut plateau = 0u32;

        println!("  {:>5} {:>7} {:>8} {:>8} {:>6}", "epoch", "loss", "train%", "test%", "time");

        for ep in 0..5000 {
            let lr = 0.005 * (1.0 - (ep as f32 / 5000.0 * 0.5).min(0.9));
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            let mut tloss = 0.0f32; let mut n = 0u32;

            for _ in 0..samples {
                let off = rt.range(0, split.saturating_sub(ctx+1));
                let mut input = Vec::with_capacity(ctx*in_ch);
                for i in 0..ctx {
                    if i == mask_pos { input.extend_from_slice(&mask_val); }
                    else { input.push(encoded[off+i][0]); input.push(encoded[off+i][1]); }
                }
                let loss = model.train_step(&input, ctx, corpus[off+mask_pos], lr);
                if !loss.is_nan() { tloss += loss; n += 1; }
            }

            if ep % 20 == 0 {
                let eval = |start: usize, end: usize| -> f64 {
                    let mut rng3 = Rng::new(999);
                    let mut ok=0usize; let mut tot=0usize;
                    for _ in 0..300 {
                        if end < start+ctx+1{break;}
                        let off = rng3.range(start, end.saturating_sub(ctx+1));
                        let mut input = Vec::with_capacity(ctx*in_ch);
                        for i in 0..ctx {
                            if i==mask_pos{input.extend_from_slice(&mask_val);}
                            else{input.push(encoded[off+i][0]);input.push(encoded[off+i][1]);}
                        }
                        if model.predict(&input, ctx)==corpus[off+mask_pos] as usize{ok+=1;}
                        tot+=1;
                    }
                    if tot==0{0.0}else{ok as f64/tot as f64*100.0}
                };

                let tr = eval(0, split);
                let te = eval(split, corpus.len());
                println!("  {:>5} {:>7.3} {:>7.1}% {:>7.1}% {:>5.0}s",
                    ep, if n>0{tloss/n as f32}else{0.0}, tr, te, tc.elapsed().as_secs_f64());

                if te > best_test + 0.5 { best_test = te; plateau = 0; }
                else { plateau += 1; }
                if plateau >= 10 || te >= 99.5 { break; }
            }

            if tc.elapsed().as_secs() > 180 { break; }
        }
        println!("  → best test: {:.1}%\n", best_test);
    }

    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
