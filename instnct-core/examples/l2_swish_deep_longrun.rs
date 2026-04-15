//! L2 Swish 3-layer — long run to plateau
//!
//! 32→128→128→128→27, swish activation
//! Run until plateau or 100%
//!
//! Run: cargo run --example l2_swish_deep_longrun --release

use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

fn swish(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn swish_grad(x: f32) -> f32 { let s = 1.0/(1.0+(-x).exp()); s + x*s*(1.0-s) }

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

struct MLP {
    w: Vec<Vec<Vec<f32>>>,
    b: Vec<Vec<f32>>,
}

impl MLP {
    fn new(sizes: &[usize], rng: &mut Rng) -> Self {
        let mut w = Vec::new(); let mut b = Vec::new();
        for i in 0..sizes.len()-1 {
            let sc = (2.0/sizes[i] as f32).sqrt();
            w.push((0..sizes[i+1]).map(|_|(0..sizes[i]).map(|_|rng.normal()*sc).collect()).collect());
            b.push(vec![0.0;sizes[i+1]]);
        }
        MLP{w,b}
    }

    fn params(&self) -> usize {
        self.w.iter().zip(&self.b).map(|(w,b)|w.len()*w[0].len()+b.len()).sum()
    }

    fn forward(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let mut layers = vec![input.to_vec()];
        let mut x = input.to_vec();
        for li in 0..self.w.len() {
            let mut z = self.b[li].clone();
            for j in 0..self.w[li].len() {
                for k in 0..x.len() { z[j] += self.w[li][j][k]*x[k]; }
            }
            if li < self.w.len()-1 {
                let a: Vec<f32> = z.iter().map(|&v| swish(v)).collect();
                layers.push(a.clone()); x = a;
            } else {
                layers.push(z);
            }
        }
        layers
    }

    fn train_step(&mut self, input: &[f32], target: u8, lr: f32) -> f32 {
        let layers = self.forward(input);
        let logits = layers.last().unwrap();

        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut p = vec![0.0f32;27]; let mut s = 0.0f32;
        for c in 0..27 { p[c]=(logits[c]-mx).exp(); s+=p[c]; }
        for c in 0..27 { p[c]/=s; }
        let loss = -(p[target as usize].max(1e-10).ln());
        p[target as usize] -= 1.0;
        let mut d = p;

        let nl = self.w.len();
        for li in (0..nl).rev() {
            let no = self.w[li].len();
            let ni = self.w[li][0].len();

            if li < nl-1 {
                // Swish gradient using pre-activation from layers
                // layers[li+1] is post-activation, but we need pre-activation
                // Reconstruct: pre = inverse_swish(post) is hard, use gradient approximation
                // Actually we need to store pre-activations. Let me use a different approach:
                // d[j] *= swish_grad(pre[j]) where pre is before activation
                // Since we don't store pre, use: swish(x) = x*sigmoid(x)
                // If y = swish(x), we can approximate x from y for small values
                // Better: just recompute forward with pre-activation storage
                let x = &layers[li]; // input to this layer
                for j in 0..no {
                    let mut pre = self.b[li][j];
                    for k in 0..ni { pre += self.w[li][j][k] * x[k]; }
                    d[j] *= swish_grad(pre);
                }
            }

            let x = &layers[li];
            let mut d_prev = vec![0.0f32; ni];
            for j in 0..no {
                for k in 0..ni {
                    d_prev[k] += d[j] * self.w[li][j][k];
                    self.w[li][j][k] -= lr * d[j] * x[k];
                }
                self.b[li][j] -= lr * d[j];
            }
            d = d_prev;
        }
        loss
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split = corpus.len() * 80 / 100;

    let encoded: Vec<[f32;2]> = corpus.iter().map(|&ch|
        [LUT[ch as usize][0] as f32 / 16.0, LUT[ch as usize][1] as f32 / 16.0]
    ).collect();

    let ctx = 16usize;
    let mask_pos = ctx / 2;
    let mask_val = [1.0f32, 1.0];
    let input_dim = ctx * 2;

    let sizes = [input_dim, 128, 128, 128, 27];
    let mut rng = Rng::new(42);
    let mut model = MLP::new(&sizes, &mut rng);

    println!("=== SWISH 3-LAYER LONG RUN ===\n");
    println!("  {:?}, {} params", sizes, model.params());
    println!("  ctx={}, mask=center, 5K samples/epoch\n", ctx);

    let samples = 5000;
    let max_ep = 10000;
    let mut best_test = 0.0f64;
    let mut plateau_count = 0u32;

    println!("  {:>5} {:>7} {:>8} {:>8} {:>6}", "epoch", "loss", "train%", "test%", "time");
    println!("  {}", "-".repeat(42));

    for ep in 0..max_ep {
        let lr = 0.005 * (1.0 - (ep as f32 / max_ep as f32 * 0.5).min(0.9));
        let mut rt = Rng::new(ep as u64 * 1000 + 42);
        let mut total_loss = 0.0f32;
        let mut n = 0u32;

        for _ in 0..samples {
            let off = rt.range(0, split - ctx - 1);
            let mut input = Vec::with_capacity(input_dim);
            for i in 0..ctx {
                if i == mask_pos { input.extend_from_slice(&mask_val); }
                else { input.push(encoded[off+i][0]); input.push(encoded[off+i][1]); }
            }
            let loss = model.train_step(&input, corpus[off+mask_pos], lr);
            if !loss.is_nan() { total_loss += loss; n += 1; }
        }

        if ep % 50 == 0 {
            let eval = |start: usize, end: usize| -> f64 {
                let mut rng3 = Rng::new(999);
                let mut ok=0usize; let mut tot=0usize;
                for _ in 0..2000 {
                    if end < start+ctx+1{break;}
                    let off = rng3.range(start, end-ctx-1);
                    let mut input = Vec::with_capacity(input_dim);
                    for i in 0..ctx {
                        if i==mask_pos{input.extend_from_slice(&mask_val);}
                        else{input.push(encoded[off+i][0]);input.push(encoded[off+i][1]);}
                    }
                    let layers = model.forward(&input);
                    let logits = layers.last().unwrap();
                    let pred = logits.iter().enumerate()
                        .max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|x|x.0).unwrap_or(0);
                    if pred == corpus[off+mask_pos] as usize {ok+=1;}
                    tot+=1;
                }
                if tot==0{0.0}else{ok as f64/tot as f64*100.0}
            };

            let tr = eval(0, split);
            let te = eval(split, corpus.len());
            let avg_loss = if n>0{total_loss/n as f32}else{0.0};

            println!("  {:>5} {:>7.3} {:>7.1}% {:>7.1}% {:>5.0}s",
                ep, avg_loss, tr, te, t0.elapsed().as_secs_f64());

            // Plateau detection
            if te > best_test + 0.5 {
                best_test = te;
                plateau_count = 0;
            } else {
                plateau_count += 1;
            }

            if te >= 99.9 {
                println!("\n  *** 100% REACHED at epoch {} ***", ep);
                break;
            }

            if plateau_count >= 20 {
                println!("\n  Plateau detected at {:.1}% (no improvement for {} checks)", best_test, plateau_count);
                break;
            }
        }
    }

    // Final per-class
    println!("\n--- Final per-class test accuracy ---\n");
    let chars = "abcdefghijklmnopqrstuvwxyz ";
    let mut pc_ok = [0u32;27]; let mut pc_tot = [0u32;27];
    let mut rng3 = Rng::new(12345);
    for _ in 0..10000 {
        if corpus.len() < split+ctx+1{break;}
        let off = rng3.range(split, corpus.len()-ctx-1);
        let mut input = Vec::with_capacity(input_dim);
        for i in 0..ctx {
            if i==mask_pos{input.extend_from_slice(&mask_val);}
            else{input.push(encoded[off+i][0]);input.push(encoded[off+i][1]);}
        }
        let t = corpus[off+mask_pos] as usize;
        let layers = model.forward(&input);
        let logits = layers.last().unwrap();
        let pred = logits.iter().enumerate()
            .max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|x|x.0).unwrap_or(0);
        pc_tot[t]+=1; if pred==t{pc_ok[t]+=1;}
    }
    for c in 0..27 {
        let ch = chars.as_bytes()[c] as char;
        let acc = if pc_tot[c]>0{pc_ok[c] as f32/pc_tot[c] as f32*100.0}else{0.0};
        let bar: String = (0..(acc/5.0) as usize).map(|_|'#').collect();
        println!("  '{}': {:>5.1}% ({:>4}/{:>4}) {}", ch, acc, pc_ok[c], pc_tot[c], bar);
    }

    println!("\n  Best test: {:.1}%", best_test);
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
