//! Abstract Core v3 — Deep dive on ctx=16 sweet spot
//!
//! ctx=16 gave 50.1% in v2 (best so far). Now:
//! 1. More epochs + lower LR to push accuracy higher
//! 2. Sweep model sizes (tiny to large) — find minimum for 50%+
//! 3. Deeper networks (3-4 layers)
//! 4. Int8 quant on all configs
//!
//! Run: cargo run --example abstract_core_v3_ctx16 --release

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
}

const CTX: usize = 16;
const PO: usize = 7;
const MI: usize = CTX * PO; // 112
const NC: usize = 27;

// Generic N-layer MLP
#[derive(Clone)]
struct Mlp {
    layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>, // (weights, biases) per layer
    dims: Vec<usize>, // input_dim, hidden1, hidden2, ..., output_dim
}

impl Mlp {
    fn new(dims: &[usize], rng: &mut Rng) -> Self {
        let mut layers = Vec::new();
        for i in 0..dims.len()-1 {
            let s = (2.0 / dims[i] as f32).sqrt();
            let w: Vec<Vec<f32>> = (0..dims[i+1]).map(|_| (0..dims[i]).map(|_| rng.normal()*s).collect()).collect();
            let b = vec![0.0f32; dims[i+1]];
            layers.push((w, b));
        }
        Mlp { layers, dims: dims.to_vec() }
    }

    fn total_params(&self) -> usize {
        self.layers.iter().map(|(w, b)| w.len() * w[0].len() + b.len()).sum()
    }

    fn forward(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let mut activations = vec![input.to_vec()];
        for (li, (w, b)) in self.layers.iter().enumerate() {
            let prev = activations.last().unwrap();
            let mut a = vec![0.0f32; w.len()];
            for i in 0..w.len() {
                a[i] = b[i];
                for j in 0..prev.len() { a[i] += w[i][j] * prev[j]; }
                if li < self.layers.len() - 1 { a[i] = a[i].max(0.0); } // ReLU for hidden, raw for output
            }
            activations.push(a);
        }
        activations
    }

    fn predict(&self, input: &[f32]) -> usize {
        let acts = self.forward(input);
        let logits = acts.last().unwrap();
        logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0)
    }

    fn train_step(&mut self, input: &[f32], target: usize, lr: f32) -> f32 {
        let acts = self.forward(input);
        let logits = acts.last().unwrap();

        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&l| (l-mx).exp()).collect();
        let se: f32 = exp.iter().sum();
        let sm: Vec<f32> = exp.iter().map(|&e| e/se).collect();
        let loss = -(sm[target].max(1e-7)).ln();

        // Backprop
        let mut delta: Vec<f32> = sm.clone();
        delta[target] -= 1.0;

        let n_layers = self.layers.len();
        for li in (0..n_layers).rev() {
            let prev_act = acts[li].clone();
            let (ref mut w, ref mut b) = self.layers[li];
            let mut d_prev = vec![0.0f32; prev_act.len()];

            for i in 0..w.len() {
                let d = if li < n_layers - 1 {
                    if acts[li+1][i] > 0.0 { delta[i] } else { 0.0 }
                } else { delta[i] };
                for j in 0..prev_act.len() {
                    d_prev[j] += d * w[i][j];
                    w[i][j] -= lr * d * prev_act[j];
                }
                b[i] -= lr * d;
            }
            delta = d_prev;
        }
        loss
    }
}

fn make_signals(pp: &Preproc, corpus: &[u8], off: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; MI];
    for i in 0..CTX { let e = pp.encode(corpus[off+i]); for k in 0..PO { s[i*PO+k] = e[k]; } }
    s
}

fn eval_acc(pp: &Preproc, m: &Mlp, corpus: &[u8], n: usize, seed: u64) -> f64 {
    let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
    for _ in 0..n { if corpus.len()<CTX+1{break;} let off=rng.range(0,corpus.len()-CTX-1);
        let s=make_signals(pp,corpus,off); if m.predict(&s)==corpus[off+CTX] as usize{ok+=1;} tot+=1; }
    ok as f64/tot as f64*100.0
}

fn eval_int8(pp: &Preproc, m: &Mlp, corpus: &[u8], n: usize, seed: u64) -> f64 {
    // Quantize each layer's weights to int8
    let scales: Vec<(f32,f32)> = m.layers.iter().map(|(w,b)| {
        let sw = w.iter().flat_map(|r|r.iter()).map(|x|x.abs()).fold(0.0f32,f32::max).max(1e-7)/127.0;
        let sb = b.iter().map(|x|x.abs()).fold(0.0f32,f32::max).max(1e-7)/127.0;
        (sw,sb)
    }).collect();

    let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
    for _ in 0..n {
        if corpus.len()<CTX+1{break;} let off=rng.range(0,corpus.len()-CTX-1);
        let mut act = make_signals(pp,corpus,off);
        for (li,(w,b)) in m.layers.iter().enumerate() {
            let (sw,sb) = scales[li];
            let mut next = vec![0.0f32;w.len()];
            for i in 0..w.len() {
                next[i] = (b[i]/sb).round().max(-127.0).min(127.0)*sb;
                for j in 0..act.len() { next[i] += (w[i][j]/sw).round().max(-127.0).min(127.0)*sw * act[j]; }
                if li < m.layers.len()-1 { next[i] = next[i].max(0.0); }
            }
            act = next;
        }
        let pred = act.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)|i).unwrap_or(0);
        if pred == corpus[off+CTX] as usize { ok+=1; } tot+=1;
    }
    ok as f64/tot as f64*100.0
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let pp = Preproc::new();

    println!("=== ABSTRACT CORE v3 — ctx=16 Deep Dive ===");
    println!("Corpus: {} chars, Context: {} chars, Input: {} signals\n", corpus.len(), CTX, MI);

    // ── PART 1: Model size sweep at ctx=16 ──
    println!("━━━ PART 1: Model size sweep (ctx=16, more epochs) ━━━\n");
    println!("  {:>30} {:>8} {:>10} {:>10} {:>8}", "config", "params", "float", "int8", "time");
    println!("  {}", "─".repeat(72));

    let configs: Vec<(&str, Vec<usize>)> = vec![
        ("tiny 112→32→27",           vec![MI,32,NC]),
        ("small 112→64→27",          vec![MI,64,NC]),
        ("med 112→128→27",           vec![MI,128,NC]),
        ("large 112→256→27",         vec![MI,256,NC]),
        ("2L 112→128→64→27",         vec![MI,128,64,NC]),
        ("2L 112→256→128→27",        vec![MI,256,128,NC]),
        ("3L 112→256→128→64→27",     vec![MI,256,128,64,NC]),
        ("3L 112→512→256→128→27",    vec![MI,512,256,128,NC]),
        ("wide 112→512→27",          vec![MI,512,NC]),
        ("narrow-deep 112→64→64→64→27", vec![MI,64,64,64,NC]),
    ];

    for (name, dims) in &configs {
        let tc = Instant::now();
        let mut best_float = 0.0f64;
        let mut best_int8 = 0.0f64;

        // Try multiple seeds
        for seed in 0..3 {
            let mut rng = Rng::new(42 + seed * 1000);
            let mut m = Mlp::new(dims, &mut rng);
            let epochs = 60;
            let samples = 10000;
            for epoch in 0..epochs {
                let lr = 0.005 * (1.0 - epoch as f32 / epochs as f32 * 0.8);
                let mut rng_t = Rng::new(epoch as u64 * 1000 + seed * 77 + 42);
                for _ in 0..samples {
                    if corpus.len() < CTX+1 { break; }
                    let off = rng_t.range(0, corpus.len()-CTX-1);
                    let s = make_signals(&pp, &corpus, off);
                    m.train_step(&s, corpus[off+CTX] as usize, lr);
                }
            }
            let fa = eval_acc(&pp, &m, &corpus, 5000, 999+seed);
            let ia = eval_int8(&pp, &m, &corpus, 5000, 999+seed);
            if fa > best_float { best_float = fa; }
            if ia > best_int8 { best_int8 = ia; }
        }

        let params = { let mut rng=Rng::new(0); Mlp::new(dims,&mut rng).total_params() };
        let marker = if best_float > 50.0 { " ★" } else { "" };
        println!("  {:>30} {:>8} {:>9.1}% {:>9.1}% {:>7.1}s{}",
            name, params, best_float, best_int8, tc.elapsed().as_secs_f64(), marker);
    }

    // ── PART 2: Extended training on best config ──
    println!("\n━━━ PART 2: Extended training (ctx=16, 2L 256→128, 100 epochs) ━━━\n");
    {
        let mut rng = Rng::new(42);
        let mut m = Mlp::new(&[MI, 256, 128, NC], &mut rng);
        let epochs = 100;
        let samples = 15000;

        for epoch in 0..epochs {
            let lr = 0.005 * (1.0 - epoch as f32 / epochs as f32 * 0.85);
            let mut rng_t = Rng::new(epoch as u64 * 1000 + 42);
            for _ in 0..samples {
                if corpus.len() < CTX+1 { break; }
                let off = rng_t.range(0, corpus.len()-CTX-1);
                let s = make_signals(&pp, &corpus, off);
                m.train_step(&s, corpus[off+CTX] as usize, lr);
            }
            if (epoch+1) % 10 == 0 {
                let fa = eval_acc(&pp, &m, &corpus, 5000, 999+epoch as u64);
                let ia = eval_int8(&pp, &m, &corpus, 5000, 999+epoch as u64);
                println!("  epoch {:>3}: float={:.1}% int8={:.1}%", epoch+1, fa, ia);
            }
        }
        let fa = eval_acc(&pp, &m, &corpus, 10000, 777);
        let ia = eval_int8(&pp, &m, &corpus, 10000, 777);
        println!("\n  FINAL (10K eval): float={:.1}% int8={:.1}%", fa, ia);
    }

    println!("\n  Baselines: frequency=20.3%, INSTNCT=24.6%");
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
