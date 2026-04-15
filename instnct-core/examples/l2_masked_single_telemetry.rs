//! L2 Masked Single Char — with deep telemetry
//!
//! Mask 1 byte (not pair), predict it. 27-class (easier than 729).
//! Input: LUT_2N direct (skip merger for clarity).
//! Telemetry: loss, grad norms, activation stats, per-class accuracy.
//!
//! Run: cargo run --example l2_masked_single_telemetry --release

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

struct MLP {
    w: Vec<Vec<Vec<f32>>>,
    b: Vec<Vec<f32>>,
    sizes: Vec<usize>,
    act: String,
}

impl MLP {
    fn new(sizes: &[usize], act: &str, rng: &mut Rng) -> Self {
        let mut w = Vec::new(); let mut b = Vec::new();
        for i in 0..sizes.len()-1 {
            let sc = (2.0/sizes[i] as f32).sqrt();
            w.push((0..sizes[i+1]).map(|_|(0..sizes[i]).map(|_|rng.normal()*sc).collect()).collect());
            b.push(vec![0.0;sizes[i+1]]);
        }
        MLP{w,b,sizes:sizes.to_vec(),act:act.to_string()}
    }

    fn params(&self) -> usize {
        self.w.iter().zip(&self.b).map(|(w,b)|w.len()*w[0].len()+b.len()).sum()
    }

    fn forward(&self, input: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut pres = Vec::new(); let mut posts = vec![input.to_vec()];
        let mut x = input.to_vec();
        for li in 0..self.w.len() {
            let mut z = self.b[li].clone();
            for j in 0..self.w[li].len() {
                for k in 0..x.len() { z[j] += self.w[li][j][k]*x[k]; }
            }
            pres.push(z.clone());
            if li < self.w.len()-1 {
                let a: Vec<f32> = z.iter().map(|&v| if self.act=="swish"{swish(v)}else{v.tanh()}).collect();
                posts.push(a.clone()); x=a;
            } else {
                posts.push(z); // logits
            }
        }
        (pres, posts)
    }

    fn train_step(&mut self, input: &[f32], target: u8, lr: f32) -> (f32, Vec<f32>, Vec<(f32,f32,f32)>) {
        let (pres, posts) = self.forward(input);
        let logits = posts.last().unwrap();

        // Softmax + CE loss
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut p = vec![0.0f32;27]; let mut s=0.0f32;
        for c in 0..27{p[c]=(logits[c]-mx).exp();s+=p[c];}
        for c in 0..27{p[c]/=s;}
        let loss = -(p[target as usize].max(1e-10).ln());
        p[target as usize] -= 1.0;
        let mut d_out = p;

        let n_layers = self.w.len();
        let mut grad_norms = vec![0.0f32; n_layers];
        let mut act_stats: Vec<(f32,f32,f32)> = Vec::new(); // (mean, std, %saturated)

        for li in (0..n_layers).rev() {
            let n_out = self.w[li].len();
            let n_in = self.w[li][0].len();

            // Activation gradient (tanh) for hidden layers
            if li < n_layers - 1 {
                let a = &posts[li+1];
                // Collect activation stats
                let mean: f32 = a.iter().sum::<f32>() / a.len() as f32;
                let std: f32 = (a.iter().map(|&v|(v-mean)*(v-mean)).sum::<f32>()/a.len() as f32).sqrt();
                let saturated = if self.act=="swish" {
                    a.iter().filter(|&&v| v.abs() > 10.0).count() as f32 / a.len() as f32 * 100.0
                } else {
                    a.iter().filter(|&&v| v.abs() > 0.95).count() as f32 / a.len() as f32 * 100.0
                };
                act_stats.push((mean, std, saturated));

                for j in 0..n_out {
                    let y = posts[li+1][j];
                    d_out[j] *= if self.act=="swish"{swish_grad(pres[li][j])}else{1.0-y*y};
                }
            }

            // Compute gradient norm for this layer
            let mut gnorm = 0.0f32;
            let mut d_in = vec![0.0f32; n_in];
            for j in 0..n_out {
                for k in 0..n_in {
                    let g = d_out[j] * posts[li][k];
                    gnorm += g*g;
                    d_in[k] += d_out[j] * self.w[li][j][k];
                    self.w[li][j][k] -= lr * g;
                }
                self.b[li][j] -= lr * d_out[j];
            }
            grad_norms[li] = gnorm.sqrt();
            d_out = d_in;
        }
        act_stats.reverse();
        (loss, grad_norms, act_stats)
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split = corpus.len() * 80 / 100;

    // LUT encode corpus (÷16 scale)
    let encoded: Vec<[f32;2]> = corpus.iter().map(|&ch|
        [LUT[ch as usize][0] as f32 / 16.0, LUT[ch as usize][1] as f32 / 16.0]
    ).collect();

    let ctx = 16usize; // bytes
    let mask_pos = ctx / 2;
    let mask_val = [1.0f32, 1.0]; // outside data range (~[-0.6, 0.06])
    let input_dim = ctx * 2;

    println!("=== MASKED SINGLE CHAR + TELEMETRY ===\n");
    println!("  {} chars ({} train), ctx={} bytes, mask=center", corpus.len(), split, ctx);
    println!("  Predict 1 char (27-class), tanh MLP\n");

    struct Run { sizes: Vec<usize>, act: &'static str }
    let runs = vec![
        Run { sizes: vec![input_dim, 128, 128, 27], act: "swish" },
        Run { sizes: vec![input_dim, 256, 256, 27], act: "swish" },
        Run { sizes: vec![input_dim, 128, 128, 128, 27], act: "swish" },
    ];

    let samples = 5000.min(split - ctx - 1);
    let max_epochs = 2000;

    for run in &runs {
    let sizes = &run.sizes;
    let act_name = run.act;
    let mut rng = Rng::new(42);
    let mut model = MLP::new(sizes, act_name, &mut rng);
    let run_t = Instant::now();
    println!("\n  == {} {:?} ({} params) ==\n", act_name, sizes, model.params());

    println!("  {:>5} {:>7} {:>8} {:>8} {:>12} {:>12} {:>12} {:>8}",
        "epoch", "loss", "train%", "test%", "grad_L0", "grad_L1", "grad_L2", "sat%_L1");
    println!("  {}", "-".repeat(82));

    for ep in 0..max_epochs {
        let lr = 0.005 * (1.0 - ep as f32 / max_epochs as f32 * 0.8);
        let mut rt = Rng::new(ep as u64 * 1000 + 42);
        let mut total_loss = 0.0f32;
        let mut sum_gnorms = vec![0.0f32; sizes.len()-1];
        let mut sum_sat = 0.0f32;
        let mut n_samples = 0u32;

        for _ in 0..samples {
            let off = rt.range(0, split - ctx - 1);
            let mut input = Vec::with_capacity(input_dim);
            for i in 0..ctx {
                if i == mask_pos { input.extend_from_slice(&mask_val); }
                else { input.push(encoded[off+i][0]); input.push(encoded[off+i][1]); }
            }
            let target = corpus[off + mask_pos];
            let (loss, gnorms, act_stats) = model.train_step(&input, target, lr);

            if !loss.is_nan() {
                total_loss += loss;
                for l in 0..gnorms.len() { sum_gnorms[l] += gnorms[l]; }
                if !act_stats.is_empty() { sum_sat += act_stats.last().unwrap().2; }
                n_samples += 1;
            }
        }

        if n_samples == 0 { println!("  ep={}: all NaN!", ep); break; }

        // Log every 100 epochs
        if ep % 10 == 0 || ep == max_epochs - 1 {
            let avg_loss = total_loss / n_samples as f32;
            let inv = 1.0 / n_samples as f32;

            // Quick eval
            let eval = |start: usize, end: usize| -> f64 {
                let mut rng3 = Rng::new(999);
                let mut ok=0usize; let mut tot=0usize;
                let n = 1000.min(end.saturating_sub(start+ctx+1));
                for _ in 0..n {
                    if end < start+ctx+1{break;}
                    let off = rng3.range(start, end-ctx-1);
                    let mut input = Vec::with_capacity(input_dim);
                    for i in 0..ctx {
                        if i==mask_pos{input.extend_from_slice(&mask_val);}
                        else{input.push(encoded[off+i][0]);input.push(encoded[off+i][1]);}
                    }
                    let (_, posts) = model.forward(&input);
                    let logits = posts.last().unwrap();
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

            println!("  {:>5} {:>7.3} {:>7.1}% {:>7.1}% {:>12.6} {:>12.6} {:>12.6} {:>7.1}%",
                ep, avg_loss, tr, te,
                sum_gnorms[0]*inv, sum_gnorms[1]*inv,
                if sum_gnorms.len()>2{sum_gnorms[2]*inv}else{0.0},
                sum_sat*inv);
        }

            if run_t.elapsed().as_secs() > 180 { break; }
    }

    // Final per-class accuracy
    println!("\n--- Per-class test accuracy ---\n");
    let chars = "abcdefghijklmnopqrstuvwxyz ";
    let mut per_class_ok = [0u32; 27];
    let mut per_class_tot = [0u32; 27];
    let mut rng3 = Rng::new(12345);
    for _ in 0..10000 {
        let end = corpus.len();
        if end < split+ctx+1{break;}
        let off = rng3.range(split, end-ctx-1);
        let mut input = Vec::with_capacity(input_dim);
        for i in 0..ctx {
            if i==mask_pos{input.extend_from_slice(&mask_val);}
            else{input.push(encoded[off+i][0]);input.push(encoded[off+i][1]);}
        }
        let target = corpus[off+mask_pos] as usize;
        let (_, posts) = model.forward(&input);
        let logits = posts.last().unwrap();
        let pred = logits.iter().enumerate()
            .max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|x|x.0).unwrap_or(0);
        per_class_tot[target] += 1;
        if pred == target { per_class_ok[target] += 1; }
    }

    for c in 0..27 {
        let ch = chars.as_bytes()[c] as char;
        let acc = if per_class_tot[c]>0{per_class_ok[c] as f32/per_class_tot[c] as f32*100.0}else{0.0};
        let bar: String = (0..(acc/5.0) as usize).map(|_|'#').collect();
        println!("  '{}': {:>5.1}% ({:>4}/{:>4}) {}",
            ch, acc, per_class_ok[c], per_class_tot[c], bar);
    }

    } // end runs loop

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
