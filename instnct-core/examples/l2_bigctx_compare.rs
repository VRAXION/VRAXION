//! L2 Big Context — 1024 byte context, swish vs C19, 3-layer MLP
//!
//! Same architecture [2048, 128, 128, 128, 27] for both
//! (1024 bytes × 2 ch = 2048 input dim)
//! First: swish baseline, then C19 with learnable c,rho
//!
//! Run: cargo run --example l2_bigctx_compare --release

use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

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

fn swish(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn swish_grad(x: f32) -> f32 { let s=1.0/(1.0+(-x).exp()); s+x*s*(1.0-s) }

fn c19_act(x: f32, c: f32, rho: f32) -> f32 {
    let c=c.max(0.1); let rho=rho.max(0.0); let l=6.0*c;
    if x>=l{return x-l;} if x<=-l{return x+l;}
    let s=x/c; let n=s.floor(); let t=s-n; let h=t*(1.0-t);
    let sg=if(n as i32)%2==0{1.0}else{-1.0}; c*(sg*h+rho*h*h)
}

fn c19_grad(x: f32, c: f32, rho: f32) -> f32 {
    let c=c.max(0.1); let rho=rho.max(0.0); let l=6.0*c;
    if x>=l||x<=-l{return 1.0;}
    let s=x/c; let n=s.floor(); let t=s-n; let h=t*(1.0-t);
    let sg=if(n as i32)%2==0{1.0}else{-1.0};
    (sg+2.0*rho*h)*(1.0-2.0*t)
}

// MLP with per-neuron c,rho for C19 mode
struct MLP {
    w: Vec<Vec<Vec<f32>>>,
    b: Vec<Vec<f32>>,
    c: Vec<Vec<f32>>,   // per neuron per hidden layer
    rho: Vec<Vec<f32>>, // per neuron per hidden layer
    use_c19: bool,
}

impl MLP {
    fn new(sizes: &[usize], use_c19: bool, rng: &mut Rng) -> Self {
        let mut w = Vec::new(); let mut b = Vec::new();
        let mut c = Vec::new(); let mut rho = Vec::new();
        for i in 0..sizes.len()-1 {
            let sc = (2.0/sizes[i] as f32).sqrt();
            w.push((0..sizes[i+1]).map(|_|(0..sizes[i]).map(|_|rng.normal()*sc).collect()).collect());
            b.push(vec![0.0;sizes[i+1]]);
            if i < sizes.len()-2 { // hidden layers only
                c.push(vec![5.0f32; sizes[i+1]]);
                rho.push(vec![0.5f32; sizes[i+1]]);
            }
        }
        MLP{w,b,c,rho,use_c19}
    }

    fn params(&self) -> usize {
        let base: usize = self.w.iter().zip(&self.b).map(|(w,b)|w.len()*w[0].len()+b.len()).sum();
        if self.use_c19 {
            base + self.c.iter().map(|v|v.len()*2).sum::<usize>() // c + rho per neuron
        } else { base }
    }

    fn train_step(&mut self, input: &[f32], target: u8, lr: f32) -> f32 {
        let nl = self.w.len();

        // Forward with pre-activation storage
        let mut posts = vec![input.to_vec()];
        let mut pres: Vec<Vec<f32>> = Vec::new();
        let mut x = input.to_vec();

        for li in 0..nl {
            let mut z = self.b[li].clone();
            for j in 0..self.w[li].len() {
                for k in 0..x.len() { z[j] += self.w[li][j][k]*x[k]; }
            }
            pres.push(z.clone());
            if li < nl-1 {
                let a: Vec<f32> = if self.use_c19 {
                    z.iter().enumerate().map(|(j,&v)| c19_act(v, self.c[li][j], self.rho[li][j])).collect()
                } else {
                    z.iter().map(|&v| swish(v)).collect()
                };
                posts.push(a.clone()); x = a;
            } else {
                posts.push(z);
            }
        }

        // Loss
        let logits = posts.last().unwrap();
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut p = vec![0.0f32;27]; let mut s=0.0f32;
        for cc in 0..27{p[cc]=(logits[cc]-mx).exp();s+=p[cc];}
        for cc in 0..27{p[cc]/=s;}
        let loss = -(p[target as usize].max(1e-10).ln());
        p[target as usize] -= 1.0;
        let mut d = p;

        // Backward
        for li in (0..nl).rev() {
            let no = self.w[li].len();
            let ni = self.w[li][0].len();

            if li < nl-1 {
                for j in 0..no {
                    let pre = pres[li][j];
                    if self.use_c19 {
                        let g = c19_grad(pre, self.c[li][j], self.rho[li][j]);
                        d[j] *= g;
                        // c,rho gradients (finite diff)
                        let eps = 0.01;
                        let y = posts[li+1][j]; // current output
                        let dc = (c19_act(pre, self.c[li][j]+eps, self.rho[li][j])
                                - c19_act(pre, self.c[li][j]-eps, self.rho[li][j])) / (2.0*eps);
                        // d[j] already has upstream gradient divided by current grad, so use pre-division
                        // Approximate: just use sign of upstream * dc
                        self.c[li][j] -= lr * d[j] / g.max(0.01).min(100.0) * dc * 0.1;
                        self.c[li][j] = self.c[li][j].max(0.5).min(50.0);
                        let dr = (c19_act(pre, self.c[li][j], self.rho[li][j]+eps)
                                - c19_act(pre, self.c[li][j], self.rho[li][j]-eps)) / (2.0*eps);
                        self.rho[li][j] -= lr * d[j] / g.max(0.01).min(100.0) * dr * 0.1;
                        self.rho[li][j] = self.rho[li][j].max(0.0).min(5.0);
                    } else {
                        d[j] *= swish_grad(pre);
                    }
                }
            }

            let x_in = &posts[li];
            let mut d_prev = vec![0.0f32; ni];
            for j in 0..no {
                for k in 0..ni {
                    d_prev[k] += d[j] * self.w[li][j][k];
                    self.w[li][j][k] -= lr * d[j] * x_in[k];
                }
                self.b[li][j] -= lr * d[j];
            }
            d = d_prev;
        }
        loss
    }

    fn predict(&self, input: &[f32]) -> usize {
        let nl = self.w.len();
        let mut x = input.to_vec();
        for li in 0..nl {
            let mut z = self.b[li].clone();
            for j in 0..self.w[li].len() {
                for k in 0..x.len() { z[j] += self.w[li][j][k]*x[k]; }
            }
            if li < nl-1 {
                x = if self.use_c19 {
                    z.iter().enumerate().map(|(j,&v)| c19_act(v, self.c[li][j], self.rho[li][j])).collect()
                } else {
                    z.iter().map(|&v| swish(v)).collect()
                };
            } else {
                x = z;
            }
        }
        x.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
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
    let input_dim = ctx * 2; // 2048

    println!("=== BIG CONTEXT (1024 bytes) — SWISH vs C19 ===\n");
    println!("  {} chars, ctx={} bytes, input_dim={}", corpus.len(), ctx, input_dim);
    println!("  Architecture: [{}→128→128→128→27]\n", input_dim);

    let sizes = vec![input_dim, 128, 128, 128, 27];

    for &(name, use_c19) in &[("swish", false), ("c19", true)] {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut model = MLP::new(&sizes, use_c19, &mut rng);

        println!("  --- {} ({} params) ---\n", name, model.params());
        println!("  {:>5} {:>7} {:>8} {:>8} {:>6}",
            "epoch", "loss", "train%", "test%", "time");

        let samples = 2000; // fewer per epoch due to large input
        let mut best_test = 0.0f64;
        let mut plateau = 0u32;

        for ep in 0..5000 {
            let lr = 0.003 * (1.0 - (ep as f32 / 5000.0 * 0.5).min(0.9));
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            let mut tloss = 0.0f32; let mut n = 0u32;

            for _ in 0..samples {
                let off = rt.range(0, split.saturating_sub(ctx+1));
                let mut input = Vec::with_capacity(input_dim);
                for i in 0..ctx {
                    if i == mask_pos { input.extend_from_slice(&mask_val); }
                    else { input.push(encoded[off+i][0]); input.push(encoded[off+i][1]); }
                }
                let loss = model.train_step(&input, corpus[off+mask_pos], lr);
                if !loss.is_nan() { tloss += loss; n += 1; }
            }

            if ep % 20 == 0 {
                let eval = |start: usize, end: usize| -> f64 {
                    let mut rng3 = Rng::new(999);
                    let mut ok=0usize; let mut tot=0usize;
                    for _ in 0..500 {
                        if end < start+ctx+1{break;}
                        let off = rng3.range(start, end.saturating_sub(ctx+1));
                        let mut input = Vec::with_capacity(input_dim);
                        for i in 0..ctx {
                            if i==mask_pos{input.extend_from_slice(&mask_val);}
                            else{input.push(encoded[off+i][0]);input.push(encoded[off+i][1]);}
                        }
                        if model.predict(&input) == corpus[off+mask_pos] as usize {ok+=1;}
                        tot+=1;
                    }
                    if tot==0{0.0}else{ok as f64/tot as f64*100.0}
                };

                let tr = eval(0, split);
                let te = eval(split, corpus.len());
                let avg = if n>0{tloss/n as f32}else{0.0};
                println!("  {:>5} {:>7.3} {:>7.1}% {:>7.1}% {:>5.0}s",
                    ep, avg, tr, te, tc.elapsed().as_secs_f64());

                if te > best_test + 0.5 { best_test = te; plateau = 0; }
                else { plateau += 1; }
                if plateau >= 15 {
                    println!("  → Plateau at {:.1}%\n", best_test);
                    break;
                }
                if te >= 99.5 {
                    println!("  → *** 100% ***\n");
                    break;
                }
            }

            if tc.elapsed().as_secs() > 300 {
                println!("  → Time limit (best test: {:.1}%)\n", best_test);
                break;
            }
        }
    }

    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
