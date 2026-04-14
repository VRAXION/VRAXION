//! Final showdown: preprocessor vs one-hot at matched architecture
//!
//! Same brain (3L-512-256-128 ReLU), same training, both int8 quantized.
//! Tests whether the preprocessor's 3.86× compression is worth the accuracy cost.
//!
//! Also: wider brain (1024→512→256) to compensate for preprocessor info loss.
//! Also: full-corpus training (all windows, not random subset).
//!
//! Run: cargo run --example preproc_vs_onehot_final --release

use std::time::Instant;

struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn normal(&mut self) -> f32 { let u1 = self.f32().max(1e-7); let u2 = self.f32(); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() }
    fn range(&mut self, lo: usize, hi: usize) -> usize { if hi <= lo { lo } else { lo + (self.next() as usize % (hi - lo)) } }
}

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = std::fs::read(path).expect("read");
    let mut c = Vec::new();
    for &b in &raw { match b { b'a'..=b'z' => c.push(b-b'a'), b'A'..=b'Z' => c.push(b-b'A'), b' '|b'\n'|b'\t'|b'\r' => c.push(26), _ => {} } }
    c
}

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0*c;
    if x >= l { return x-l; } if x <= -l { return x+l; }
    let s = x/c; let n = s.floor(); let t = s-n; let h = t*(1.0-t);
    let sg = if (n as i32)%2==0 { 1.0 } else { -1.0 }; c*(sg*h+rho*h*h)
}

struct Preproc { w: [[i8;8];7], b: [i8;7], c: [f32;7], rho: [f32;7] }
impl Preproc {
    fn new() -> Self { Preproc {
        w: [[-1,1,1,-1,-1,1,1,-1],[1,-1,1,1,-1,-1,-1,-1],[-1,-1,1,-1,1,1,-1,-1],
            [-1,-1,-1,1,-1,-1,1,-1],[-1,1,1,1,-1,-1,-1,-1],[1,1,-1,-1,-1,-1,-1,-1],
            [-1,1,-1,1,1,1,-1,-1]],
        b: [1,1,1,1,1,1,1], c: [10.0;7], rho: [2.0,0.0,0.0,0.0,0.0,0.0,0.0],
    }}
    fn encode(&self, ch: u8) -> [f32;7] {
        let mut bits=[0.0f32;8]; for i in 0..8 { bits[i]=((ch>>i)&1) as f32; }
        let mut o=[0.0f32;7];
        for k in 0..7 { let mut d=self.b[k] as f32; for j in 0..8 { d+=self.w[k][j] as f32*bits[j]; } o[k]=c19(d,self.c[k],self.rho[k]); }
        o
    }
    fn encode_seq(&self, chars: &[u8]) -> Vec<f32> {
        chars.iter().flat_map(|&ch| self.encode(ch).to_vec()).collect()
    }
}

struct Brain {
    layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>, // (weights, biases) per layer
    dims: Vec<usize>, // input, h1, h2, ..., 27
}

impl Brain {
    fn new(dims: &[usize], rng: &mut Rng) -> Self {
        let mut layers = Vec::new();
        for i in 0..dims.len()-1 {
            let s = (2.0/dims[i] as f32).sqrt();
            let w: Vec<Vec<f32>> = (0..dims[i+1]).map(|_| (0..dims[i]).map(|_| rng.normal()*s).collect()).collect();
            let b = vec![0.0f32; dims[i+1]];
            layers.push((w, b));
        }
        Brain { layers, dims: dims.to_vec() }
    }

    fn params(&self) -> usize {
        self.layers.iter().map(|(w,b)| w.len()*w[0].len() + b.len()).sum()
    }

    fn forward(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let mut acts = vec![input.to_vec()];
        for (li, (w, b)) in self.layers.iter().enumerate() {
            let prev = &acts[li];
            let is_last = li == self.layers.len() - 1;
            let mut a = vec![0.0f32; w.len()];
            for k in 0..w.len() {
                a[k] = b[k];
                for j in 0..w[k].len() { a[k] += w[k][j] * prev[j]; }
                if !is_last { a[k] = a[k].max(0.0); } // ReLU except last
            }
            acts.push(a);
        }
        acts
    }

    fn train_step(&mut self, input: &[f32], target: u8, lr: f32) {
        let acts = self.forward(input);
        let logits = acts.last().unwrap();

        // Softmax
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut p: Vec<f32> = logits.iter().map(|&l| (l-mx).exp()).collect();
        let s: f32 = p.iter().sum(); for v in &mut p { *v /= s; }
        p[target as usize] -= 1.0;

        // Backprop
        let mut delta = p;
        let n_layers = self.layers.len();
        for li in (0..n_layers).rev() {
            let prev_act = acts[li].clone();
            let mut next_delta = vec![0.0f32; self.dims[li]];
            let (w, b) = &mut self.layers[li];
            for k in 0..w.len() {
                for j in 0..w[k].len() {
                    next_delta[j] += delta[k] * w[k][j];
                    w[k][j] -= lr * delta[k] * prev_act[j];
                }
                b[k] -= lr * delta[k];
            }
            if li > 0 {
                // ReLU derivative
                for j in 0..next_delta.len() {
                    if acts[li][j] <= 0.0 { next_delta[j] = 0.0; }
                }
            }
            delta = next_delta;
        }
    }

    fn eval(&self, corpus: &[u8], ctx: usize, enc: &dyn Fn(&[u8])->Vec<f32>, n: usize, seed: u64) -> f64 {
        let mut rng = Rng::new(seed); let mut ok = 0; let mut tot = 0;
        for _ in 0..n { if corpus.len()<ctx+1{break;} let off=rng.range(0,corpus.len()-ctx-1);
            let acts = self.forward(&enc(&corpus[off..off+ctx]));
            let logits = acts.last().unwrap();
            let pred = logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred == corpus[off+ctx] as usize { ok+=1; } tot+=1;
        }
        ok as f64/tot as f64*100.0
    }

    fn eval_int8(&self, corpus: &[u8], ctx: usize, enc: &dyn Fn(&[u8])->Vec<f32>, n: usize, seed: u64) -> f64 {
        // Quantize all layers
        let qlayers: Vec<(Vec<Vec<i8>>, Vec<f32>, f32)> = self.layers.iter().map(|(w, b)| {
            let mx = w.iter().flat_map(|r|r.iter()).map(|x|x.abs()).fold(0.0f32, f32::max);
            let s = if mx>0.0{127.0/mx}else{1.0};
            let qw = w.iter().map(|r| r.iter().map(|&x| (x*s).round().max(-127.0).min(127.0) as i8).collect()).collect();
            (qw, b.clone(), 1.0/s)
        }).collect();

        let mut rng = Rng::new(seed); let mut ok = 0; let mut tot = 0;
        for _ in 0..n { if corpus.len()<ctx+1{break;} let off=rng.range(0,corpus.len()-ctx-1);
            let mut a = enc(&corpus[off..off+ctx]);
            for (li, (qw, b, ds)) in qlayers.iter().enumerate() {
                let is_last = li == qlayers.len()-1;
                let mut next = vec![0.0f32; qw.len()];
                for k in 0..qw.len() { next[k] = b[k]; for j in 0..qw[k].len() { next[k] += qw[k][j] as f32 * ds * a[j]; }
                    if !is_last { next[k] = next[k].max(0.0); } }
                a = next;
            }
            let pred = a.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred == corpus[off+ctx] as usize { ok+=1; } tot+=1;
        }
        ok as f64/tot as f64*100.0
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let pp = Preproc::new();
    let ctx = 16usize;

    println!("=== PREPROCESSOR vs ONE-HOT — FINAL SHOWDOWN ===\n");
    println!("  Corpus: {} chars, ctx={}", corpus.len(), ctx);
    println!("  Baselines: freq=20.3%, INSTNCT=24.6%\n");

    let enc_pp = |chars: &[u8]| -> Vec<f32> { pp.encode_seq(chars) };
    let enc_oh = |chars: &[u8]| -> Vec<f32> {
        let mut v = vec![0.0f32; ctx*27];
        for i in 0..ctx.min(chars.len()) { v[i*27+chars[i] as usize] = 1.0; }
        v
    };

    struct Config { name: &'static str, dims: Vec<usize>, epochs: usize, samples: usize, input: &'static str }

    let configs = vec![
        // Matched architecture: 3L-512-256-128, 200ep
        Config { name: "PP  3L-512-256-128 200ep", dims: vec![112, 512, 256, 128, 27], epochs: 200, samples: 15000, input: "pp" },
        Config { name: "OH  3L-512-256-128 200ep", dims: vec![432, 512, 256, 128, 27], epochs: 200, samples: 15000, input: "oh" },
        // Wider brain for preprocessor to compensate
        Config { name: "PP  3L-1024-512-256 200ep", dims: vec![112, 1024, 512, 256, 27], epochs: 200, samples: 15000, input: "pp" },
        // Full-corpus training: more diversity
        Config { name: "PP  3L-512-256-128 200ep FULL", dims: vec![112, 512, 256, 128, 27], epochs: 200, samples: 50000, input: "pp" },
        Config { name: "OH  3L-512-256-128 200ep FULL", dims: vec![432, 512, 256, 128, 27], epochs: 200, samples: 50000, input: "oh" },
    ];

    println!("  {:>35} {:>8} {:>8} {:>8} {:>7} {:>8}",
        "config", "params", "float%", "int8%", "Δ", "time");
    println!("  {}", "─".repeat(80));

    for cfg in &configs {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut brain = Brain::new(&cfg.dims, &mut rng);
        let max_s = cfg.samples.min(corpus.len()/(ctx+1));

        for ep in 0..cfg.epochs {
            let lr = 0.01 * (1.0 - ep as f32 / cfg.epochs as f32 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..max_s {
                let off = rt.range(0, corpus.len()-ctx-1);
                let input = match cfg.input { "pp" => enc_pp(&corpus[off..off+ctx]), _ => enc_oh(&corpus[off..off+ctx]) };
                brain.train_step(&input, corpus[off+ctx], lr);
            }
            if (ep+1) % 50 == 0 {
                let enc: &dyn Fn(&[u8])->Vec<f32> = match cfg.input { "pp" => &enc_pp, _ => &enc_oh };
                let a = brain.eval(&corpus, ctx, enc, 2000, 777+ep as u64);
                print!("    ep{}: {:.1}%  ", ep+1, a);
            }
        }
        println!();

        let enc: &dyn Fn(&[u8])->Vec<f32> = match cfg.input { "pp" => &enc_pp, _ => &enc_oh };
        let fa = brain.eval(&corpus, ctx, enc, 5000, 999);
        let ia = brain.eval_int8(&corpus, ctx, enc, 5000, 999);
        let m = if fa>90.0{" ★★★"} else if fa>70.0{" ★★"} else if fa>50.0{" ★"} else {""};

        println!("  {:>35} {:>8} {:>7.1}% {:>7.1}% {:>+6.1}% {:>7.1}s{}",
            cfg.name, brain.params(), fa, ia, ia-fa, tc.elapsed().as_secs_f64(), m);
        println!();
    }

    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
