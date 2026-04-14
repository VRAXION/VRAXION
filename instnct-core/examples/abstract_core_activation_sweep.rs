//! Activation function sweep in the mixer layer
//!
//! Tests: ReLU, LeakyReLU, GeLU, SiLU/Swish, Tanh, ELU, C19
//! All on the same config: ctx=16, 2L 256→128, 60 epochs
//!
//! Run: cargo run --example abstract_core_activation_sweep --release

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
const MI: usize = CTX * PO;
const NC: usize = 27;

// ══════════════════════════════════════════════════════
// ACTIVATION FUNCTIONS + DERIVATIVES
// ══════════════════════════════════════════════════════
#[derive(Clone, Copy)]
enum Act { ReLU, LeakyReLU, GeLU, SiLU, Tanh, ELU, C19Static }

fn act_forward(a: Act, x: f32) -> f32 {
    match a {
        Act::ReLU => x.max(0.0),
        Act::LeakyReLU => if x > 0.0 { x } else { 0.01 * x },
        Act::GeLU => 0.5 * x * (1.0 + ((2.0/std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh()),
        Act::SiLU => x / (1.0 + (-x).exp()),  // x * sigmoid(x)
        Act::Tanh => x.tanh(),
        Act::ELU => if x > 0.0 { x } else { (x.exp() - 1.0) },
        Act::C19Static => c19(x, 3.0, 1.0),  // fixed c=3, rho=1
    }
}

fn act_backward(a: Act, x: f32, out: f32) -> f32 {
    match a {
        Act::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
        Act::LeakyReLU => if x > 0.0 { 1.0 } else { 0.01 },
        Act::GeLU => {
            let eps = 0.001;
            (act_forward(Act::GeLU, x + eps) - act_forward(Act::GeLU, x - eps)) / (2.0 * eps)
        },
        Act::SiLU => {
            let sig = 1.0 / (1.0 + (-x).exp());
            sig + x * sig * (1.0 - sig)
        },
        Act::Tanh => 1.0 - out * out,
        Act::ELU => if x > 0.0 { 1.0 } else { out + 1.0 },
        Act::C19Static => {
            let eps = 0.001;
            (c19(x + eps, 3.0, 1.0) - c19(x - eps, 3.0, 1.0)) / (2.0 * eps)
        },
    }
}

fn act_name(a: Act) -> &'static str {
    match a { Act::ReLU => "ReLU", Act::LeakyReLU => "LeakyReLU", Act::GeLU => "GeLU",
              Act::SiLU => "SiLU/Swish", Act::Tanh => "Tanh", Act::ELU => "ELU",
              Act::C19Static => "C19(3,1)" }
}

// ══════════════════════════════════════════════════════
// MLP with configurable activation
// ══════════════════════════════════════════════════════
struct Mlp {
    layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
    act: Act,
}

impl Mlp {
    fn new(dims: &[usize], act: Act, rng: &mut Rng) -> Self {
        let mut layers = Vec::new();
        for i in 0..dims.len()-1 {
            let s = (2.0/dims[i] as f32).sqrt();
            layers.push(((0..dims[i+1]).map(|_| (0..dims[i]).map(|_| rng.normal()*s).collect()).collect(),
                         vec![0.0f32; dims[i+1]]));
        }
        Mlp { layers, act }
    }

    fn train_step(&mut self, input: &[f32], target: usize, lr: f32) -> f32 {
        let nl = self.layers.len();
        // Forward
        let mut pre_acts: Vec<Vec<f32>> = Vec::new(); // pre-activation (z)
        let mut post_acts: Vec<Vec<f32>> = vec![input.to_vec()]; // post-activation (a)
        for (li, (w, b)) in self.layers.iter().enumerate() {
            let prev = post_acts.last().unwrap();
            let mut z = vec![0.0f32; w.len()];
            let mut a = vec![0.0f32; w.len()];
            for i in 0..w.len() {
                z[i] = b[i]; for j in 0..prev.len() { z[i] += w[i][j] * prev[j]; }
                a[i] = if li < nl-1 { act_forward(self.act, z[i]) } else { z[i] }; // raw logits for output
            }
            pre_acts.push(z);
            post_acts.push(a);
        }

        // Softmax + loss
        let logits = post_acts.last().unwrap();
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&l| (l-mx).exp()).collect();
        let se: f32 = exp.iter().sum();
        let sm: Vec<f32> = exp.iter().map(|&e| e/se).collect();
        let loss = -(sm[target].max(1e-7)).ln();
        let mut delta: Vec<f32> = sm; delta[target] -= 1.0;

        // Backprop
        for li in (0..nl).rev() {
            let prev_act = post_acts[li].clone();
            let (ref mut w, ref mut b) = self.layers[li];
            let mut d_prev = vec![0.0f32; prev_act.len()];
            for i in 0..w.len() {
                let d = if li < nl-1 {
                    delta[i] * act_backward(self.act, pre_acts[li][i], post_acts[li+1][i])
                } else { delta[i] };
                for j in 0..prev_act.len() { d_prev[j] += d*w[i][j]; w[i][j] -= lr*d*prev_act[j]; }
                b[i] -= lr*d;
            }
            delta = d_prev;
        }
        loss
    }

    fn predict(&self, input: &[f32]) -> usize {
        let nl = self.layers.len();
        let mut act = input.to_vec();
        for (li, (w, b)) in self.layers.iter().enumerate() {
            let mut next = vec![0.0f32; w.len()];
            for i in 0..w.len() { next[i]=b[i]; for j in 0..act.len() { next[i]+=w[i][j]*act[j]; }
                if li<nl-1 { next[i]=act_forward(self.act, next[i]); } }
            act = next;
        }
        act.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0)
    }
}

fn make_signals(pp: &Preproc, corpus: &[u8], off: usize) -> Vec<f32> {
    let mut s = vec![0.0f32; MI];
    for i in 0..CTX { let e=pp.encode(corpus[off+i]); for k in 0..PO { s[i*PO+k]=e[k]; } } s
}

fn eval_acc(pp: &Preproc, m: &Mlp, corpus: &[u8], n: usize, seed: u64) -> f64 {
    let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
    for _ in 0..n { if corpus.len()<CTX+1{break;} let off=rng.range(0,corpus.len()-CTX-1);
        if m.predict(&make_signals(pp,corpus,off))==corpus[off+CTX] as usize{ok+=1;} tot+=1; }
    ok as f64/tot as f64*100.0
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let pp = Preproc::new();

    println!("=== ACTIVATION FUNCTION SWEEP (fast) ===");
    println!("Config: ctx=16, 1L 128→27, 30 epochs, 5000 samples/ep\n");

    let activations = [Act::ReLU, Act::LeakyReLU, Act::GeLU, Act::SiLU, Act::Tanh, Act::ELU, Act::C19Static];

    println!("  {:>12} {:>10} {:>8}", "activation", "accuracy", "time");
    println!("  {}", "─".repeat(35));

    for &act in &activations {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut m = Mlp::new(&[MI, 128, NC], act, &mut rng);
        for epoch in 0..30 {
            let lr = 0.01 * (1.0 - epoch as f32 / 30.0 * 0.7);
            let mut rng_t = Rng::new(epoch as u64 * 1000 + 42);
            for _ in 0..5000 {
                if corpus.len() < CTX+1 { break; }
                let off = rng_t.range(0, corpus.len()-CTX-1);
                m.train_step(&make_signals(&pp, &corpus, off), corpus[off+CTX] as usize, lr);
            }
        }
        let acc = eval_acc(&pp, &m, &corpus, 3000, 999);
        let marker = if acc >= 45.0 { " ★" } else { "" };
        println!("  {:>12} {:>9.1}% {:>7.1}s{}", act_name(act), acc, tc.elapsed().as_secs_f64(), marker);
    }

    println!("\n  Baselines: frequency=20.3%, INSTNCT=24.6%");
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
