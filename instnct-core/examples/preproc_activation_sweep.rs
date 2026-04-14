//! Preprocessor activation sweep — which activation is best for encoding?
//!
//! Current: C19 (c=10, rho=2/0) with binary {-1,+1} weights → 100% round-trip
//! Test: ReLU, Sigmoid, Tanh, SiLU, LeakyReLU, abs, identity, C19 variants
//!
//! For each activation:
//!   1. Keep the proven binary weights OR do greedy search for new optimal weights
//!   2. Measure round-trip reconstruction accuracy (29 unique bytes)
//!   3. Train a brain on top → measure downstream prediction accuracy
//!
//! Run: cargo run --example preproc_activation_sweep --release

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

// ── Activation functions ──
fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0*c;
    if x >= l { return x-l; } if x <= -l { return x+l; }
    let s = x/c; let n = s.floor(); let t = s-n; let h = t*(1.0-t);
    let sg = if (n as i32)%2==0 { 1.0 } else { -1.0 }; c*(sg*h+rho*h*h)
}

fn relu(x: f32) -> f32 { x.max(0.0) }
fn leaky_relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.01 * x } }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn tanh_act(x: f32) -> f32 { x.tanh() }
fn silu(x: f32) -> f32 { x * sigmoid(x) }
fn abs_act(x: f32) -> f32 { x.abs() }
fn identity(x: f32) -> f32 { x }
fn step(x: f32) -> f32 { if x > 0.0 { 1.0 } else { -1.0 } }
fn softplus(x: f32) -> f32 { (1.0 + x.exp()).ln() }

// ── Preprocessor with configurable activation ──
struct FlexPreproc {
    w: [[i8;8];7],
    b: [i8;7],
    act_fn: fn(f32) -> f32,
    act_name: &'static str,
}

impl FlexPreproc {
    fn encode(&self, ch: u8) -> [f32;7] {
        let mut bits = [0.0f32;8];
        for i in 0..8 { bits[i] = ((ch>>i)&1) as f32; }
        let mut o = [0.0f32;7];
        for k in 0..7 {
            let mut d = self.b[k] as f32;
            for j in 0..8 { d += self.w[k][j] as f32 * bits[j]; }
            o[k] = (self.act_fn)(d);
        }
        o
    }

    fn encode_seq(&self, chars: &[u8]) -> Vec<f32> {
        chars.iter().flat_map(|&ch| self.encode(ch).to_vec()).collect()
    }

    fn roundtrip_accuracy(&self) -> (usize, usize) {
        // Test all 29 unique bytes (0-26 used in corpus, plus check a few more)
        let unique: Vec<u8> = (0..27).collect();
        let mut ok = 0;
        for &ch in &unique {
            let code = self.encode(ch);
            // Find nearest
            let mut best = 0u8; let mut bd = f32::MAX;
            for &test_ch in &unique {
                let tc = self.encode(test_ch);
                let d: f32 = tc.iter().zip(code.iter()).map(|(a,b)| (a-b)*(a-b)).sum();
                if d < bd { bd = d; best = test_ch; }
            }
            if best == ch { ok += 1; }
        }
        (ok, unique.len())
    }
}

// C19 preprocessor (original, with c/rho)
struct C19Preproc {
    w: [[i8;8];7], b: [i8;7], c: [f32;7], rho: [f32;7],
}
impl C19Preproc {
    fn new() -> Self { C19Preproc {
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

// ── Greedy binary weight search for a given activation ──
fn greedy_search_weights(act_fn: fn(f32) -> f32) -> ([[i8;8];7], [i8;7], usize) {
    let unique: Vec<u8> = (0..27).collect();
    let mut best_w = [[0i8;8];7];
    let mut best_b = [0i8;7];

    // Search one neuron at a time, greedily
    for neuron in 0..7 {
        let mut best_score = 0usize;
        let mut best_nw = [0i8;8];
        let mut best_nb = 0i8;

        // Exhaustive: 8 weights + 1 bias, each {-1, +1} = 2^9 = 512 combos
        for combo in 0..512u32 {
            let mut nw = [0i8;8];
            for j in 0..8 { nw[j] = if (combo>>j)&1==1 { 1 } else { -1 }; }
            let nb = if (combo>>8)&1==1 { 1i8 } else { -1 };

            // Temporarily set this neuron
            let mut w = best_w; w[neuron] = nw;
            let mut b = best_b; b[neuron] = nb;

            // Evaluate round-trip with neurons 0..=neuron
            let mut ok = 0;
            for &ch in &unique {
                let mut bits = [0.0f32;8];
                for i in 0..8 { bits[i] = ((ch>>i)&1) as f32; }
                let mut code = [0.0f32;7];
                for k in 0..=neuron {
                    let mut d = b[k] as f32;
                    for j in 0..8 { d += w[k][j] as f32 * bits[j]; }
                    code[k] = act_fn(d);
                }

                // Nearest neighbor in active dims
                let mut best_match = 0u8; let mut bd = f32::MAX;
                for &test_ch in &unique {
                    let mut tbits = [0.0f32;8];
                    for i in 0..8 { tbits[i] = ((test_ch>>i)&1) as f32; }
                    let mut tcode = [0.0f32;7];
                    for k in 0..=neuron {
                        let mut d = b[k] as f32;
                        for j in 0..8 { d += w[k][j] as f32 * tbits[j]; }
                        tcode[k] = act_fn(d);
                    }
                    let d: f32 = (0..=neuron).map(|k| (code[k]-tcode[k])*(code[k]-tcode[k])).sum();
                    if d < bd { bd = d; best_match = test_ch; }
                }
                if best_match == ch { ok += 1; }
            }

            if ok > best_score { best_score = ok; best_nw = nw; best_nb = nb; }
            if best_score == 27 { break; }
        }

        best_w[neuron] = best_nw;
        best_b[neuron] = best_nb;
    }

    // Final accuracy
    let pp = FlexPreproc { w: best_w, b: best_b, act_fn, act_name: "" };
    let (ok, tot) = pp.roundtrip_accuracy();
    (best_w, best_b, ok)
}

// ── Train brain and evaluate prediction ──
fn train_eval_brain(corpus: &[u8], ctx: usize, enc: &dyn Fn(&[u8])->Vec<f32>, idim: usize) -> f64 {
    let mut rng = Rng::new(42);
    let s1 = (2.0/idim as f32).sqrt(); let s2 = (2.0/128.0f32).sqrt();
    let hdim = 128;
    let mut w1: Vec<Vec<f32>> = (0..hdim).map(|_| (0..idim).map(|_| rng.normal()*s1).collect()).collect();
    let mut b1 = vec![0.0f32;hdim];
    let mut w2: Vec<Vec<f32>> = (0..27).map(|_| (0..hdim).map(|_| rng.normal()*s2).collect()).collect();
    let mut b2 = vec![0.0f32;27];

    let samples = 10000.min(corpus.len()/(ctx+1));
    for ep in 0..100 {
        let lr = 0.01 * (1.0 - ep as f32/100.0*0.7);
        let mut rt = Rng::new(ep as u64*1000+42);
        for _ in 0..samples {
            let off = rt.range(0, corpus.len()-ctx-1);
            let input = enc(&corpus[off..off+ctx]);
            let target = corpus[off+ctx] as usize;
            // Forward
            let mut h = vec![0.0f32;hdim];
            for k in 0..hdim { h[k]=b1[k]; for j in 0..idim { h[k]+=w1[k][j]*input[j]; } h[k]=h[k].max(0.0); }
            let mut logits = vec![0.0f32;27];
            for c in 0..27 { logits[c]=b2[c]; for k in 0..hdim { logits[c]+=w2[c][k]*h[k]; } }
            let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let mut p=vec![0.0f32;27]; let mut s=0.0f32;
            for c in 0..27 { p[c]=(logits[c]-mx).exp(); s+=p[c]; } for c in 0..27 { p[c]/=s; }
            let mut d=p; d[target]-=1.0;
            let mut dh=vec![0.0f32;hdim];
            for c in 0..27 { for k in 0..hdim { dh[k]+=d[c]*w2[c][k]; w2[c][k]-=lr*d[c]*h[k]; } b2[c]-=lr*d[c]; }
            for k in 0..hdim { if h[k]<=0.0{continue;} for j in 0..idim { w1[k][j]-=lr*dh[k]*input[j]; } b1[k]-=lr*dh[k]; }
        }
    }

    // Eval
    let mut rng2=Rng::new(999); let mut ok=0; let mut tot=0;
    for _ in 0..5000 { if corpus.len()<ctx+1{break;} let off=rng2.range(0,corpus.len()-ctx-1);
        let input=enc(&corpus[off..off+ctx]);
        let mut h=vec![0.0f32;hdim];
        for k in 0..hdim { h[k]=b1[k]; for j in 0..idim { h[k]+=w1[k][j]*input[j]; } h[k]=h[k].max(0.0); }
        let mut logits=vec![0.0f32;27];
        for c in 0..27 { logits[c]=b2[c]; for k in 0..hdim { logits[c]+=w2[c][k]*h[k]; } }
        let pred=logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        if pred==corpus[off+ctx] as usize {ok+=1;} tot+=1;
    }
    ok as f64/tot as f64*100.0
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let ctx = 16usize;

    println!("=== PREPROCESSOR ACTIVATION SWEEP ===\n");
    println!("  For each activation: greedy binary weight search → round-trip + prediction test");
    println!("  Brain: 1L H=128 ReLU, ctx={}, 100 epochs\n", ctx);

    let activations: Vec<(&str, fn(f32)->f32)> = vec![
        ("C19(10,2)", |x| c19(x, 10.0, 2.0)),
        ("C19(10,0)", |x| c19(x, 10.0, 0.0)),
        ("C19(1,0)",  |x| c19(x, 1.0, 0.0)),
        ("C19(1,2)",  |x| c19(x, 1.0, 2.0)),
        ("ReLU",      relu),
        ("LeakyReLU", leaky_relu),
        ("Sigmoid",   sigmoid),
        ("Tanh",      tanh_act),
        ("SiLU",      silu),
        ("Abs",       abs_act),
        ("Identity",  identity),
        ("Step",      step),
        ("Softplus",  softplus),
    ];

    println!("  {:>12} {:>8} {:>10} {:>8}",
        "activation", "rt_acc", "pred%", "time");
    println!("  {}", "─".repeat(45));

    // Also test the original C19 preprocessor with its mixed c/rho
    {
        let tc = Instant::now();
        let c19pp = C19Preproc::new();
        // Round-trip
        let unique: Vec<u8> = (0..27).collect();
        let mut rt_ok = 0;
        for &ch in &unique {
            let code = c19pp.encode(ch);
            let mut best = 0u8; let mut bd = f32::MAX;
            for &t in &unique { let tc2 = c19pp.encode(t); let d: f32 = tc2.iter().zip(code.iter()).map(|(a,b)|(a-b)*(a-b)).sum(); if d<bd{bd=d;best=t;} }
            if best == ch { rt_ok += 1; }
        }
        let pred = train_eval_brain(&corpus, ctx, &|chars| c19pp.encode_seq(chars), ctx*7);
        let m = if pred > 45.0 { " ★★" } else if pred > 35.0 { " ★" } else { "" };
        println!("  {:>12} {:>5}/{:>2} {:>7.1}% {:>7.1}s{}  (original mixed c/rho)",
            "C19-mixed", rt_ok, 27, pred, tc.elapsed().as_secs_f64(), m);
    }

    for (name, act_fn) in &activations {
        let tc = Instant::now();

        // Greedy search optimal binary weights for this activation
        let (w, b, rt_ok) = greedy_search_weights(*act_fn);

        // Train brain with this preprocessor encoding
        let pp = FlexPreproc { w, b, act_fn: *act_fn, act_name: name };
        let pred = train_eval_brain(&corpus, ctx, &|chars| pp.encode_seq(chars), ctx*7);
        let m = if pred > 45.0 { " ★★" } else if pred > 35.0 { " ★" } else { "" };

        println!("  {:>12} {:>5}/{:>2} {:>7.1}% {:>7.1}s{}",
            name, rt_ok, 27, pred, tc.elapsed().as_secs_f64(), m);
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
