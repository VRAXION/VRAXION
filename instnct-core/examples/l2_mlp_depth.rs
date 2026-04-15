//! L2 MLP Depth Test — deeper networks for masked pair prediction
//!
//! Pure MLP (no conv), sweep depth × width × activation
//! Only tanh, swish, C19 (the ones that worked)
//!
//! Run: cargo run --example l2_mlp_depth --release

use std::time::Instant;

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
    for k in 0..2 { o[k] = MB[k] as f32 + MW[k][0] as f32*i[0] + MW[k][1] as f32*i[1]
        + MW[k][2] as f32*i[2] + MW[k][3] as f32*i[3]; }
    [o[0]/16.0, o[1]/16.0]
}

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

fn c19(x: f32) -> f32 {
    let c=10.0; let l=60.0;
    if x>=l{return x-l;} if x<=-l{return x+l;}
    let s=x/c; let n=s.floor(); let t=s-n; let h=t*(1.0-t);
    let sg=if(n as i32)%2==0{1.0}else{-1.0}; c*sg*h
}

fn apply(act: &str, x: f32) -> f32 {
    match act {
        "tanh" => x.tanh(),
        "swish" => x / (1.0 + (-x).exp()),
        "c19" => c19(x),
        _ => x.tanh(),
    }
}

fn grad(act: &str, x: f32, y: f32) -> f32 {
    match act {
        "tanh" => 1.0 - y*y,
        "swish" => { let s=1.0/(1.0+(-x).exp()); s+x*s*(1.0-s) },
        "c19" => {
            let c=10.0;let l=60.0;
            if x>=l||x<=-l{1.0}else{
                let s=x/c;let n=s.floor();let t=s-n;
                let sg=if(n as i32)%2==0{1.0}else{-1.0};sg*(1.0-2.0*t)}
        },
        _ => 1.0 - y*y,
    }
}

// Variable-depth MLP
struct MLP {
    // layers[i]: (weights, biases) — weights[out][in]
    layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
    act: String,
}

impl MLP {
    fn new(sizes: &[usize], act: &str, rng: &mut Rng) -> Self {
        let mut layers = Vec::new();
        for i in 0..sizes.len()-1 {
            let fan_in = sizes[i];
            let fan_out = sizes[i+1];
            let sc = (2.0/fan_in as f32).sqrt();
            let w: Vec<Vec<f32>> = (0..fan_out).map(|_|
                (0..fan_in).map(|_| rng.normal()*sc).collect()
            ).collect();
            let b = vec![0.0f32; fan_out];
            layers.push((w, b));
        }
        MLP { layers, act: act.to_string() }
    }

    fn params(&self) -> usize {
        self.layers.iter().map(|(w,b)| w.len()*w[0].len()+b.len()).sum()
    }

    fn forward(&self, input: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        // Returns (pre_acts, post_acts) for each layer
        let mut pre = Vec::new();
        let mut post = Vec::new();
        let mut x = input.to_vec();
        post.push(x.clone()); // input as layer 0 post

        for (li, (w, b)) in self.layers.iter().enumerate() {
            let mut z = b.clone();
            for j in 0..w.len() {
                for k in 0..x.len() { z[j] += w[j][k] * x[k]; }
            }
            pre.push(z.clone());

            // Last layer: no activation (logits)
            if li == self.layers.len() - 1 {
                post.push(z);
            } else {
                let a: Vec<f32> = z.iter().map(|&v| apply(&self.act, v)).collect();
                post.push(a.clone());
                x = a;
            }
        }
        (pre, post)
    }

    fn train_step(&mut self, input: &[f32], target_a: u8, target_b: u8, lr: f32) {
        let (pre, post) = self.forward(input);
        let logits = post.last().unwrap();

        // Split logits: first 27 = char_a, next 27 = char_b
        let la = &logits[..27];
        let lb = &logits[27..];

        let softmax_grad = |logits: &[f32], target: u8| -> Vec<f32> {
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut p = vec![0.0f32;27]; let mut s=0.0f32;
            for c in 0..27{p[c]=(logits[c]-mx).exp();s+=p[c];}
            for c in 0..27{p[c]/=s;}
            p[target as usize] -= 1.0;
            p
        };

        let da = softmax_grad(la, target_a);
        let db = softmax_grad(lb, target_b);
        let mut d_out: Vec<f32> = da.iter().chain(db.iter()).cloned().collect();

        // Backprop through layers (reverse)
        let n_layers = self.layers.len();
        for li in (0..n_layers).rev() {
            let n_out = self.layers[li].0.len();
            let n_in = self.layers[li].0[0].len();

            // Apply activation gradient (except last layer)
            if li < n_layers - 1 {
                for j in 0..n_out {
                    d_out[j] *= grad(&self.act, pre[li][j], post[li+1][j]);
                }
            }

            // Gradient for previous layer
            let mut d_in = vec![0.0f32; n_in];
            for j in 0..n_out {
                for k in 0..n_in {
                    d_in[k] += d_out[j] * self.layers[li].0[j][k];
                    self.layers[li].0[j][k] -= lr * d_out[j] * post[li][k];
                }
                self.layers[li].1[j] -= lr * d_out[j];
            }

            d_out = d_in;
        }
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let mut corp = corpus.clone();
    if corp.len()%2!=0{corp.push(26);}
    let merged: Vec<[f32;2]> = corp.chunks(2).map(|p|
        merger_encode(p[0], if p.len()>1{p[1]}else{26})
    ).collect();
    let n_pairs = merged.len();
    let split_p = n_pairs * 80 / 100;
    let ctx = 16usize; // pairs
    let mask_pos = ctx/2;
    let mask_val = [15.0f32, 15.0];
    let input_dim = ctx * 2; // 32

    println!("=== L2 MLP DEPTH TEST ===\n");
    println!("  {} merged pairs ({} train), ctx={} pairs, mask=center", n_pairs, split_p, ctx);
    println!("  Pure MLP (no conv), tanh/swish/c19\n");

    struct Cfg { layers: Vec<usize>, act: &'static str, epochs: usize }
    let configs = vec![
        // 1 hidden layer
        Cfg { layers: vec![32, 128, 54], act: "tanh", epochs: 500 },
        Cfg { layers: vec![32, 256, 54], act: "tanh", epochs: 500 },
        // 2 hidden layers
        Cfg { layers: vec![32, 128, 128, 54], act: "tanh", epochs: 500 },
        Cfg { layers: vec![32, 256, 256, 54], act: "tanh", epochs: 400 },
        // 3 hidden layers
        Cfg { layers: vec![32, 128, 128, 128, 54], act: "tanh", epochs: 400 },
        Cfg { layers: vec![32, 256, 256, 256, 54], act: "tanh", epochs: 300 },
        // 4 hidden layers
        Cfg { layers: vec![32, 128, 128, 128, 128, 54], act: "tanh", epochs: 300 },
        // Best configs with swish
        Cfg { layers: vec![32, 256, 256, 54], act: "swish", epochs: 400 },
        Cfg { layers: vec![32, 256, 256, 256, 54], act: "swish", epochs: 300 },
        // C19
        Cfg { layers: vec![32, 256, 256, 54], act: "c19", epochs: 400 },
    ];

    println!("  {:>6} {:>20} {:>8} {:>10} {:>10} {:>7}",
        "act", "architecture", "params", "train%", "test%", "time");
    println!("  {}", "-".repeat(68));

    for cfg in &configs {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut model = MLP::new(&cfg.layers, cfg.act, &mut rng);
        let samples = 20000.min(split_p.saturating_sub(ctx+1));

        for ep in 0..cfg.epochs {
            let lr = 0.005 * (1.0 - ep as f32 / cfg.epochs as f32 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);

            for _ in 0..samples {
                let off = rt.range(0, split_p.saturating_sub(ctx+1));
                let mut input = Vec::with_capacity(input_dim);
                for i in 0..ctx {
                    if i == mask_pos { input.extend_from_slice(&mask_val); }
                    else { input.push(merged[off+i][0]); input.push(merged[off+i][1]); }
                }
                let pi = off + mask_pos;
                let ta = corp[pi*2];
                let tb = corp[pi*2+1];
                model.train_step(&input, ta, tb, lr);
            }
            if tc.elapsed().as_secs() > 90 { break; }
        }

        // Eval
        let eval = |start: usize, end: usize| -> f64 {
            let mut rng3 = Rng::new(999);
            let mut ok=0usize; let mut tot=0usize;
            let n = 5000.min(end.saturating_sub(start+ctx+1));
            for _ in 0..n {
                if end < start+ctx+1{break;}
                let off = rng3.range(start, end-ctx-1);
                let mut input = Vec::with_capacity(input_dim);
                for i in 0..ctx {
                    if i==mask_pos{input.extend_from_slice(&mask_val);}
                    else{input.push(merged[off+i][0]);input.push(merged[off+i][1]);}
                }
                let (_, post) = model.forward(&input);
                let logits = post.last().unwrap();
                let pa = logits[..27].iter().enumerate()
                    .max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|x|x.0).unwrap_or(0);
                let pb = logits[27..].iter().enumerate()
                    .max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|x|x.0).unwrap_or(0);
                let pi = off+mask_pos;
                if pa==corp[pi*2] as usize && pb==corp[pi*2+1] as usize {ok+=1;}
                tot+=1;
            }
            if tot==0{0.0}else{ok as f64/tot as f64*100.0}
        };

        let tr = eval(0, split_p);
        let te = eval(split_p, n_pairs);
        let arch: String = cfg.layers.iter().map(|s|s.to_string()).collect::<Vec<_>>().join("→");
        let depth = cfg.layers.len()-2;
        let m = if te>50.0{" ***"}else if te>30.0{" **"}else if te>15.0{" *"}else{""};

        println!("  {:>6} {:>20} {:>8} {:>9.1}% {:>9.1}% {:>6.1}s{}",
            cfg.act, arch, model.params(), tr, te, tc.elapsed().as_secs_f64(), m);
    }

    println!("\n  Baseline: random = 0.14%");
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
