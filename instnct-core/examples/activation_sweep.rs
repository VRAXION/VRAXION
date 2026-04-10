//! Activation Sweep: C19 vs ReLU vs tanh vs GELU vs swish vs C19 variants
//! Plain MLP (no sparse sandwich), 512→H→128, Adam, 10K data
//! Also test with tick recurrence (t=1,2,4) on best activations
//!
//! Run: cargo run --example activation_sweep --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use std::time::Instant;

const CTX: usize = 4;
const N_BYTES: usize = 128;
const IN: usize = CTX * N_BYTES;

fn c19_fwd(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01); let l = 6.0 * c;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let s = x / c; let n = s.floor(); let t = s - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}
fn c19_deriv(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01); let l = 6.0 * c;
    if x >= l || x <= -l { return 1.0; }
    let s = x / c; let n = s.floor(); let t = s - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    (sgn + 2.0 * rho * h) * (1.0 - 2.0 * t)
}

fn act_fwd(x: f32, kind: &str, rho: f32) -> f32 {
    match kind {
        "relu" => x.max(0.0),
        "leaky" => if x > 0.0 { x } else { 0.01 * x },
        "tanh" => x.tanh(),
        "gelu" => x * 0.5 * (1.0 + (0.7978846 * (x + 0.044715 * x * x * x)).tanh()),
        "swish" => x / (1.0 + (-x).exp()),
        "c19" => c19_fwd(x, 1.0, rho),
        "c19_r1" => c19_fwd(x, 1.0, 1.0),   // fixed rho=1
        "c19_r8" => c19_fwd(x, 1.0, 8.0),   // fixed rho=8
        _ => x,
    }
}

fn act_deriv(x: f32, kind: &str, rho: f32) -> f32 {
    match kind {
        "relu" => if x > 0.0 { 1.0 } else { 0.0 },
        "leaky" => if x > 0.0 { 1.0 } else { 0.01 },
        "tanh" => { let t = x.tanh(); 1.0 - t * t },
        "gelu" => {
            let cdf = 0.5 * (1.0 + (0.7978846 * (x + 0.044715 * x*x*x)).tanh());
            let pdf = (0.7978846 * (1.0 + 3.0*0.044715*x*x)) * (1.0 - (0.7978846*(x+0.044715*x*x*x)).tanh().powi(2));
            cdf + x * 0.5 * pdf
        },
        "swish" => {
            let sig = 1.0 / (1.0 + (-x).exp());
            sig + x * sig * (1.0 - sig)
        },
        "c19" => c19_deriv(x, 1.0, rho),
        "c19_r1" => c19_deriv(x, 1.0, 1.0),
        "c19_r8" => c19_deriv(x, 1.0, 8.0),
        _ => 1.0,
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / s).collect()
}

fn one_hot(ctx: &[u8]) -> Vec<f32> {
    let mut v = vec![0.0f32; IN];
    for (p, &b) in ctx.iter().enumerate() {
        if (b as usize) < N_BYTES { v[p * N_BYTES + b as usize] = 1.0; }
    }
    v
}

// ============================================================
// MLP with recurrence: hidden = act(W_in*input + W_rec*hidden_prev + bias)
// At t=1: no recurrence (W_rec unused since hidden starts at 0)
// At t>1: residual recurrence: hidden += act(W_in*input + W_rec*hidden + bias)
// ============================================================

#[derive(Clone)]
struct Net {
    h: usize, ticks: usize, act: String,
    w_in: Vec<f32>,   // h * IN
    w_rec: Vec<f32>,  // h * h (recurrent, only used at t>1)
    b: Vec<f32>,      // h
    w_out: Vec<f32>,  // N_BYTES * h
    b_out: Vec<f32>,  // N_BYTES
    rho: Vec<f32>,    // h (only for c19)
}

impl Net {
    fn new(h: usize, ticks: usize, act: &str, rng: &mut StdRng) -> Self {
        let sc_in = (2.0 / IN as f32).sqrt();
        let sc_rec = (2.0 / h as f32).sqrt();
        Net {
            h, ticks, act: act.to_string(),
            w_in: (0..h*IN).map(|_| rng.gen_range(-sc_in..sc_in)).collect(),
            w_rec: (0..h*h).map(|_| rng.gen_range(-sc_rec*0.1..sc_rec*0.1)).collect(),
            b: vec![0.0; h],
            w_out: (0..N_BYTES*h).map(|_| rng.gen_range(-0.1..0.1)).collect(),
            b_out: vec![0.0; N_BYTES],
            rho: vec![4.0; h],
        }
    }

    fn has_rho(&self) -> bool { self.act == "c19" }

    fn params(&self) -> usize {
        self.w_in.len() + self.b.len()
        + if self.ticks > 1 { self.w_rec.len() } else { 0 }
        + self.w_out.len() + self.b_out.len()
        + if self.has_rho() { self.rho.len() } else { 0 }
    }

    fn forward(&self, input: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
        let h = self.h;
        let mut state = vec![0.0f32; h];
        let mut all_pre = Vec::new();
        let mut all_act = Vec::new();

        for _t in 0..self.ticks {
            let mut pre = vec![0.0f32; h];
            for i in 0..h {
                let mut s = self.b[i];
                for j in 0..IN { s += input[j] * self.w_in[i * IN + j]; }
                if self.ticks > 1 {
                    for j in 0..h { if j != i { s += state[j] * self.w_rec[i * h + j]; } }
                }
                pre[i] = s;
            }
            // Activation + residual (if t>1)
            let mut new_state = vec![0.0f32; h];
            for i in 0..h {
                let a = act_fwd(pre[i], &self.act, self.rho[i]);
                new_state[i] = if self.ticks > 1 { state[i] + a } else { a }; // residual
            }
            all_pre.push(pre);
            all_act.push(new_state.clone());
            state = new_state;
        }

        let mut logits = vec![0.0f32; N_BYTES];
        for b in 0..N_BYTES {
            let mut s = self.b_out[b];
            for i in 0..h { s += state[i] * self.w_out[b * h + i]; }
            logits[b] = s;
        }
        (all_pre, all_act, softmax(&logits))
    }

    fn train_adam(&mut self, data: &[(Vec<f32>, u8)], test: &[(Vec<f32>, u8)],
                  steps: usize, batch_size: usize) {
        let h = self.h;
        let eps = 1e-3f32;
        let np = self.params();
        let mut m = vec![0.0f32; np];
        let mut v = vec![0.0f32; np];
        let mut rng = StdRng::seed_from_u64(99);
        let mut sh: Vec<usize> = (0..data.len()).collect();
        let test_sub = if test.len() > 2000 { &test[..2000] } else { test };

        for step in 1..=steps {
            sh.shuffle(&mut rng);
            let batch = &sh[..batch_size.min(data.len())];

            // Numerical gradient (fast enough for these sizes with 1-hidden MLP)
            // For t=1 this is simple; for t>1 BPTT would be better but numerical works
            let base_loss: f64 = batch.iter().map(|&idx| {
                let (_, _, probs) = self.forward(&data[idx].0);
                -(probs[data[idx].1 as usize].max(1e-10) as f64).ln()
            }).sum::<f64>() / batch.len() as f64;

            // Gather params
            let mut pv = Vec::with_capacity(np);
            pv.extend(&self.w_in); pv.extend(&self.b);
            if self.ticks > 1 { pv.extend(&self.w_rec); }
            pv.extend(&self.w_out); pv.extend(&self.b_out);
            if self.has_rho() { pv.extend(&self.rho); }

            // Stochastic param sampling for numerical grad
            let sample = 400.min(np);
            let mut indices: Vec<usize> = (0..np).collect();
            indices.shuffle(&mut rng);

            let mut grad = vec![0.0f32; np];
            for &pi in indices[..sample].iter() {
                let orig = pv[pi];
                self.set_param(pi, orig + eps);
                let lp: f64 = batch.iter().map(|&idx| {
                    let (_, _, probs) = self.forward(&data[idx].0);
                    -(probs[data[idx].1 as usize].max(1e-10) as f64).ln()
                }).sum::<f64>() / batch.len() as f64;
                self.set_param(pi, orig);
                grad[pi] = ((lp - base_loss) / eps as f64) as f32;
            }

            // Adam update
            let t = step as f32;
            let b1c = 1.0 - 0.9f32.powf(t);
            let b2c = 1.0 - 0.999f32.powf(t);
            for &pi in indices[..sample].iter() {
                m[pi] = 0.9 * m[pi] + 0.1 * grad[pi];
                v[pi] = 0.999 * v[pi] + 0.001 * grad[pi] * grad[pi];
                let mh = m[pi] / b1c;
                let vh = v[pi] / b2c;
                let new_val = pv[pi] - 0.001 * mh / (vh.sqrt() + 1e-8);
                self.set_param(pi, new_val);
            }
            // Clamp rho
            if self.has_rho() { for r in &mut self.rho { *r = r.max(0.0); } }

            if step % 500 == 0 || step == steps {
                let te = test_sub.iter().filter(|(inp, t)| {
                    let (_, _, p) = self.forward(inp);
                    p.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 == *t as usize
                }).count() as f64 / test_sub.len() as f64;
                let t5 = {
                    let mut ok = 0;
                    for (inp, tgt) in test_sub {
                        let (_, _, p) = self.forward(inp);
                        let mut idx: Vec<(usize,f32)> = p.iter().enumerate().map(|(i,&v)|(i,v)).collect();
                        idx.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
                        if idx.iter().take(5).any(|(i,_)| *i == *tgt as usize) { ok += 1; }
                    }
                    ok as f64 / test_sub.len() as f64
                };
                println!("    step {:>5}: test={:.1}% top5={:.1}%", step, te*100.0, t5*100.0);
            }
        }
    }

    fn set_param(&mut self, idx: usize, val: f32) {
        let mut i = idx;
        if i < self.w_in.len() { self.w_in[i] = val; return; } i -= self.w_in.len();
        if i < self.b.len() { self.b[i] = val; return; } i -= self.b.len();
        if self.ticks > 1 {
            if i < self.w_rec.len() { self.w_rec[i] = val; return; } i -= self.w_rec.len();
        }
        if i < self.w_out.len() { self.w_out[i] = val; return; } i -= self.w_out.len();
        if i < self.b_out.len() { self.b_out[i] = val; return; } i -= self.b_out.len();
        if self.has_rho() && i < self.rho.len() { self.rho[i] = val; }
    }

    fn accuracy(&self, data: &[(Vec<f32>, u8)]) -> f64 {
        data.iter().filter(|(inp, t)| {
            let (_, _, p) = self.forward(inp);
            p.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 == *t as usize
        }).count() as f64 / data.len() as f64
    }

    fn top5(&self, data: &[(Vec<f32>, u8)]) -> f64 {
        let mut ok = 0;
        for (inp, t) in data {
            let (_, _, p) = self.forward(inp);
            let mut idx: Vec<(usize,f32)> = p.iter().enumerate().map(|(i,&v)|(i,v)).collect();
            idx.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
            if idx.iter().take(5).any(|(i,_)| *i == *t as usize) { ok += 1; }
        }
        ok as f64 / data.len() as f64
    }
}

fn main() {
    println!("=== ACTIVATION SWEEP: what activation actually matters? ===\n");
    let t0 = Instant::now();

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
        match o { Ok(o) if o.stdout.len()>1000 => { println!("  {} bytes",o.stdout.len()); o.stdout }
            _ => { println!("  Fallback"); "the quick brown fox jumps over the lazy dog ".repeat(500).bytes().collect() } }
    };
    let text: Vec<u8> = raw.iter().map(|&b| if b<128{b}else{32}).collect();
    let mut pairs: Vec<(Vec<u8>,u8)> = Vec::new();
    for i in CTX..text.len() {
        let ctx = text[i-CTX..i].to_vec();
        if (text[i] as usize) < N_BYTES { pairs.push((ctx, text[i])); }
    }
    let mut rng = StdRng::seed_from_u64(42);
    pairs.shuffle(&mut rng);
    let train_oh: Vec<(Vec<f32>,u8)> = pairs[..10000].iter().map(|(c,t)|(one_hot(c),*t)).collect();
    let test_oh: Vec<(Vec<f32>,u8)> = pairs[60000..65000].iter().map(|(c,t)|(one_hot(c),*t)).collect();
    println!("  Train: {}, Test: {}\n", train_oh.len(), test_oh.len());

    let h = 19; // ~12K params to match previous experiments
    let steps = 3000; // numerical grad is slower, fewer steps
    let batch = 100;
    let mut results: Vec<(String, usize, f64, f64)> = Vec::new();

    // =========================================================
    // PHASE 1: Activation sweep (t=1, no recurrence)
    // =========================================================
    println!("===== PHASE 1: Activation sweep (512->{}-128, t=1) =====\n", h);

    for act in &["relu", "leaky", "tanh", "gelu", "swish", "c19", "c19_r1", "c19_r8"] {
        let label = format!("{} t=1", act);
        println!("--- {} ---", label);
        let mut net = Net::new(h, 1, act, &mut StdRng::seed_from_u64(77));
        println!("  params={}", net.params());
        let t1 = Instant::now();
        net.train_adam(&train_oh, &test_oh, steps, batch);
        let acc = net.accuracy(&test_oh[..2000]);
        let t5 = net.top5(&test_oh[..2000]);
        println!("  => {}: test={:.1}% top5={:.1}% ({:.1}s)\n",
                 act, acc*100.0, t5*100.0, t1.elapsed().as_secs_f64());
        results.push((label, net.params(), acc, t5));
    }

    // =========================================================
    // PHASE 2: Tick recurrence on best activations (t=2)
    // =========================================================
    println!("===== PHASE 2: Residual recurrence t=2 =====\n");

    for act in &["relu", "c19", "gelu"] {
        let label = format!("{} t=2 res", act);
        println!("--- {} ---", label);
        let mut net = Net::new(h, 2, act, &mut StdRng::seed_from_u64(77));
        println!("  params={}", net.params());
        let t1 = Instant::now();
        net.train_adam(&train_oh, &test_oh, steps, batch);
        let acc = net.accuracy(&test_oh[..2000]);
        let t5 = net.top5(&test_oh[..2000]);
        println!("  => {}: test={:.1}% top5={:.1}% ({:.1}s)\n",
                 label, acc*100.0, t5*100.0, t1.elapsed().as_secs_f64());
        results.push((label, net.params(), acc, t5));
    }

    // =========================================================
    // COMPARISON
    // =========================================================
    println!("=== FULL COMPARISON ===");
    println!("  {:>16} {:>8} {:>8} {:>8}", "activation", "params", "test%", "top5%");
    println!("  {}", "=".repeat(45));
    for (label, params, acc, t5) in &results {
        println!("  {:>16} {:>8} {:>7.1}% {:>7.1}%", label, params, acc*100.0, t5*100.0);
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
    println!("\n=== DONE ===");
}
