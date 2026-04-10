//! ALU Hybrid: learned neurons + frozen ALU circuits in the same network
//!
//! The learned layer figures out how to USE the fixed ALU neurons.
//! Test: does having ALU circuits help vs just more learned neurons?
//!
//! Run: cargo run --example alu_hybrid --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use std::time::Instant;

const CTX: usize = 4;
const N_BYTES: usize = 128;
const IN: usize = CTX * N_BYTES;

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}
fn c19d(x: f32, rho: f32) -> f32 {
    let l = 6.0;
    if x >= l || x <= -l { return 1.0; }
    let n = x.floor(); let t = x - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    (sgn + 2.0 * rho * h) * (1.0 - 2.0 * t)
}
fn softmax(v: &[f32]) -> Vec<f32> {
    let mx = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let e: Vec<f32> = v.iter().map(|&x| (x - mx).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|&x| x / s).collect()
}
fn one_hot(ctx: &[u8]) -> Vec<f32> {
    let mut v = vec![0.0f32; IN];
    for (p, &b) in ctx.iter().enumerate() { if (b as usize) < N_BYTES { v[p*N_BYTES+b as usize] = 1.0; } }
    v
}

// ============================================================
// Hybrid Net: learned neurons + frozen ALU neurons
//
// Architecture:
//   Input → [n_learned C19 neurons (trainable)]
//                ↕ (all-to-all connections)
//           [n_alu frozen neurons (fixed weights)]
//   Output ← reads ALL neurons (learned + ALU)
//
// ALU neurons: input from learned neurons, fixed weights, C19 activation
// They approximate ADD, MUL, |a-b| operations on learned representations
// ============================================================

#[derive(Clone)]
struct HybridNet {
    n_learned: usize,
    n_alu: usize,
    // Learned: input → hidden (trainable)
    w_in: Vec<f32>,      // n_learned * IN
    b_in: Vec<f32>,      // n_learned
    // ALU: reads from learned (FROZEN after init)
    w_alu: Vec<f32>,     // n_alu * n_learned (frozen)
    b_alu: Vec<f32>,     // n_alu (frozen)
    rho_alu: Vec<f32>,   // n_alu (frozen)
    // Output: reads from ALL (learned + ALU)
    w_out: Vec<f32>,     // N_BYTES * (n_learned + n_alu) (trainable)
    b_out: Vec<f32>,     // N_BYTES (trainable)
    rho: f32,            // shared rho for learned neurons
}

impl HybridNet {
    fn new(n_learned: usize, n_alu: usize, alu_mode: &str, rng: &mut StdRng) -> Self {
        let sc = (2.0 / IN as f32).sqrt();
        let total = n_learned + n_alu;

        // ALU weights: depends on mode
        let (w_alu, b_alu, rho_alu) = match alu_mode {
            "add" => {
                // Each ALU neuron sums 2 random learned neurons (approximate adder)
                let mut w = vec![0.0f32; n_alu * n_learned];
                let mut b = vec![0.0f32; n_alu];
                for i in 0..n_alu {
                    let a = (i * 7 + 3) % n_learned;
                    let bb = (i * 11 + 5) % n_learned;
                    w[i * n_learned + a] = 1.0;
                    w[i * n_learned + bb] = 1.0;
                }
                (w, b, vec![8.0; n_alu])
            },
            "mixed" => {
                // Mix of operations: half add-like, half mul-like (different rho)
                let mut w = vec![0.0f32; n_alu * n_learned];
                let mut b = vec![0.0f32; n_alu];
                let mut rho = vec![0.0f32; n_alu];
                for i in 0..n_alu {
                    let a = (i * 7 + 3) % n_learned;
                    let bb = (i * 11 + 5) % n_learned;
                    w[i * n_learned + a] = if i % 2 == 0 { 1.0 } else { 1.0 };
                    w[i * n_learned + bb] = if i % 2 == 0 { 1.0 } else { -1.0 }; // add vs subtract
                    rho[i] = if i % 3 == 0 { 1.0 } else if i % 3 == 1 { 4.0 } else { 8.0 };
                }
                (w, b, rho)
            },
            "random_frozen" => {
                // Random weights, frozen — control experiment
                let w: Vec<f32> = (0..n_alu * n_learned).map(|_| rng.gen_range(-0.5..0.5)).collect();
                let b = vec![0.0f32; n_alu];
                (w, b, vec![8.0; n_alu])
            },
            "dense_random" => {
                // Random weights from ALL inputs (not just learned) — like extra learned neurons but frozen
                // This one reads from input directly, not from learned
                let w: Vec<f32> = (0..n_alu * n_learned).map(|_| rng.gen_range(-0.3..0.3)).collect();
                let b = vec![0.0f32; n_alu];
                (w, b, vec![8.0; n_alu])
            },
            _ => {
                let w = vec![0.0f32; n_alu * n_learned];
                let b = vec![0.0f32; n_alu];
                (w, b, vec![8.0; n_alu])
            },
        };

        HybridNet {
            n_learned, n_alu,
            w_in: (0..n_learned*IN).map(|_| rng.gen_range(-sc..sc)).collect(),
            b_in: vec![0.0; n_learned],
            w_alu, b_alu, rho_alu,
            w_out: (0..N_BYTES*total).map(|_| rng.gen_range(-0.1..0.1)).collect(),
            b_out: vec![0.0; N_BYTES],
            rho: 8.0,
        }
    }

    fn total_h(&self) -> usize { self.n_learned + self.n_alu }

    fn trainable_params(&self) -> usize {
        self.w_in.len() + self.b_in.len() + self.w_out.len() + self.b_out.len()
    }

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (nl, na) = (self.n_learned, self.n_alu);
        // Learned neurons
        let mut h_pre = vec![0.0f32; nl];
        let mut h_act = vec![0.0f32; nl];
        for i in 0..nl {
            let mut s = self.b_in[i];
            for j in 0..IN { s += input[j] * self.w_in[i*IN+j]; }
            h_pre[i] = s;
            h_act[i] = c19(s, self.rho);
        }
        // ALU neurons (read from learned outputs)
        let mut alu_pre = vec![0.0f32; na];
        let mut alu_act = vec![0.0f32; na];
        for i in 0..na {
            let mut s = self.b_alu[i];
            for j in 0..nl { s += h_act[j] * self.w_alu[i*nl+j]; }
            alu_pre[i] = s;
            alu_act[i] = c19(s, self.rho_alu[i]);
        }
        // Concat all activations
        let mut all = Vec::with_capacity(nl + na);
        all.extend_from_slice(&h_act);
        all.extend_from_slice(&alu_act);
        // Output
        let total = nl + na;
        let mut logits = vec![0.0f32; N_BYTES];
        for b in 0..N_BYTES {
            let mut s = self.b_out[b];
            for i in 0..total { s += all[i] * self.w_out[b*total+i]; }
            logits[b] = s;
        }
        (all, vec![h_pre, alu_pre].concat(), softmax(&logits))
    }

    fn train_adam(&mut self, data: &[(Vec<f32>,u8)], test: &[(Vec<f32>,u8)],
                  steps: usize, batch: usize) {
        let nl = self.n_learned;
        let na = self.n_alu;
        let total = nl + na;
        let np = self.trainable_params();
        let mut mv = vec![0.0f32; np];
        let mut vv = vec![0.0f32; np];
        let mut rng = StdRng::seed_from_u64(99);
        let mut sh: Vec<usize> = (0..data.len()).collect();
        let test_sub = if test.len() > 2000 { &test[..2000] } else { test };

        for step in 1..=steps {
            sh.shuffle(&mut rng);
            let batch_idx = &sh[..batch.min(data.len())];
            let bs = batch_idx.len() as f32;

            let mut g_win = vec![0.0f32; nl*IN];
            let mut g_bin = vec![0.0f32; nl];
            let mut g_wout = vec![0.0f32; N_BYTES*total];
            let mut g_bout = vec![0.0f32; N_BYTES];

            for &idx in batch_idx {
                let (all_act, all_pre, probs) = self.forward(&data[idx].0);
                let mut dl = probs; dl[data[idx].1 as usize] -= 1.0;

                // Output grads
                let mut d_all = vec![0.0f32; total];
                for b in 0..N_BYTES {
                    g_bout[b] += dl[b];
                    for i in 0..total {
                        g_wout[b*total+i] += dl[b] * all_act[i];
                        d_all[i] += dl[b] * self.w_out[b*total+i];
                    }
                }

                // Backprop through ALU (frozen weights, but need d_learned from ALU)
                let mut d_learned_from_alu = vec![0.0f32; nl];
                for i in 0..na {
                    let deriv = c19d(all_pre[nl+i], self.rho_alu[i]);
                    let d_alu_pre = d_all[nl+i] * deriv;
                    // ALU reads from learned: propagate gradient
                    for j in 0..nl {
                        d_learned_from_alu[j] += d_alu_pre * self.w_alu[i*nl+j];
                    }
                    // NO gradient to w_alu (frozen)
                }

                // Backprop through learned neurons
                for i in 0..nl {
                    let d_h = d_all[i] + d_learned_from_alu[i];
                    let deriv = c19d(all_pre[i], self.rho);
                    let d_pre = d_h * deriv;
                    g_bin[i] += d_pre;
                    for j in 0..IN { g_win[i*IN+j] += d_pre * data[idx].0[j]; }
                }
            }

            // Flatten trainable grads
            let mut grad = Vec::with_capacity(np);
            for g in &g_win { grad.push(g / bs); }
            for g in &g_bin { grad.push(g / bs); }
            for g in &g_wout { grad.push(g / bs); }
            for g in &g_bout { grad.push(g / bs); }

            // Flatten trainable params
            let mut pv = Vec::with_capacity(np);
            pv.extend(&self.w_in); pv.extend(&self.b_in);
            pv.extend(&self.w_out); pv.extend(&self.b_out);

            let t = step as f32;
            let b1c = 1.0 - 0.9f32.powf(t);
            let b2c = 1.0 - 0.999f32.powf(t);
            for i in 0..np {
                mv[i] = 0.9*mv[i] + 0.1*grad[i];
                vv[i] = 0.999*vv[i] + 0.001*grad[i]*grad[i];
                pv[i] -= 0.001 * (mv[i]/b1c) / ((vv[i]/b2c).sqrt()+1e-8);
            }

            // Write back
            let mut o = 0;
            self.w_in.copy_from_slice(&pv[o..o+nl*IN]); o+=nl*IN;
            self.b_in.copy_from_slice(&pv[o..o+nl]); o+=nl;
            self.w_out.copy_from_slice(&pv[o..o+N_BYTES*total]); o+=N_BYTES*total;
            self.b_out.copy_from_slice(&pv[o..o+N_BYTES]);

            if step%1000==0 || step==steps {
                let te = self.accuracy(test_sub);
                let t5 = self.top5(test_sub);
                println!("    step {:>5}: test={:.1}% top5={:.1}%", step, te*100.0, t5*100.0);
            }
        }
    }

    fn accuracy(&self, data: &[(Vec<f32>,u8)]) -> f64 {
        data.iter().filter(|(inp,t)| {
            let (_,_,p) = self.forward(inp);
            p.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap()).unwrap().0 == *t as usize
        }).count() as f64 / data.len() as f64
    }
    fn top5(&self, data: &[(Vec<f32>,u8)]) -> f64 {
        let mut ok=0;
        for (inp,t) in data {
            let (_,_,p) = self.forward(inp);
            let mut idx:Vec<(usize,f32)>=p.iter().enumerate().map(|(i,&v)|(i,v)).collect();
            idx.sort_by(|a,b|b.1.partial_cmp(&a.1).unwrap());
            if idx.iter().take(5).any(|(i,_)|*i==*t as usize){ok+=1;}
        }
        ok as f64/data.len() as f64
    }
}

fn main() {
    println!("=== ALU HYBRID: learned + frozen ALU neurons ===\n");
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

    let steps = 10000;
    let batch = 200;
    let mut results: Vec<(String, usize, usize, f64, f64)> = Vec::new();

    // =========================================================
    // 1. Baseline: 19 learned, 0 ALU (pure C19 MLP)
    // =========================================================
    println!("--- Baseline: 19 learned + 0 ALU ---");
    {
        let mut net = HybridNet::new(19, 0, "", &mut StdRng::seed_from_u64(77));
        println!("  trainable={} total_h={}", net.trainable_params(), net.total_h());
        let t1 = Instant::now();
        net.train_adam(&train_oh, &test_oh, steps, batch);
        let acc = net.accuracy(&test_oh[..2000]); let t5 = net.top5(&test_oh[..2000]);
        println!("  => test={:.1}% top5={:.1}% ({:.1}s)\n", acc*100.0, t5*100.0, t1.elapsed().as_secs_f64());
        results.push(("19 learned".into(), net.trainable_params(), 19, acc, t5));
    }

    // =========================================================
    // 2. Hybrid: 15 learned + 8 ALU (mixed ops, frozen)
    // =========================================================
    println!("--- Hybrid: 15 learned + 8 ALU (mixed) ---");
    {
        let mut net = HybridNet::new(15, 8, "mixed", &mut StdRng::seed_from_u64(77));
        println!("  trainable={} total_h={} (8 ALU frozen)", net.trainable_params(), net.total_h());
        let t1 = Instant::now();
        net.train_adam(&train_oh, &test_oh, steps, batch);
        let acc = net.accuracy(&test_oh[..2000]); let t5 = net.top5(&test_oh[..2000]);
        println!("  => test={:.1}% top5={:.1}% ({:.1}s)\n", acc*100.0, t5*100.0, t1.elapsed().as_secs_f64());
        results.push(("15 learned + 8 ALU".into(), net.trainable_params(), 23, acc, t5));
    }

    // =========================================================
    // 3. Control: 15 learned + 8 random frozen
    // =========================================================
    println!("--- Control: 15 learned + 8 random frozen ---");
    {
        let mut net = HybridNet::new(15, 8, "random_frozen", &mut StdRng::seed_from_u64(77));
        println!("  trainable={} total_h={} (8 random frozen)", net.trainable_params(), net.total_h());
        let t1 = Instant::now();
        net.train_adam(&train_oh, &test_oh, steps, batch);
        let acc = net.accuracy(&test_oh[..2000]); let t5 = net.top5(&test_oh[..2000]);
        println!("  => test={:.1}% top5={:.1}% ({:.1}s)\n", acc*100.0, t5*100.0, t1.elapsed().as_secs_f64());
        results.push(("15 learned + 8 random".into(), net.trainable_params(), 23, acc, t5));
    }

    // =========================================================
    // 4. Hybrid: 15 learned + 8 ADD ALU
    // =========================================================
    println!("--- Hybrid: 15 learned + 8 ALU (add-only) ---");
    {
        let mut net = HybridNet::new(15, 8, "add", &mut StdRng::seed_from_u64(77));
        println!("  trainable={} total_h={} (8 ADD ALU frozen)", net.trainable_params(), net.total_h());
        let t1 = Instant::now();
        net.train_adam(&train_oh, &test_oh, steps, batch);
        let acc = net.accuracy(&test_oh[..2000]); let t5 = net.top5(&test_oh[..2000]);
        println!("  => test={:.1}% top5={:.1}% ({:.1}s)\n", acc*100.0, t5*100.0, t1.elapsed().as_secs_f64());
        results.push(("15 learned + 8 ADD".into(), net.trainable_params(), 23, acc, t5));
    }

    // =========================================================
    // 5. Fair comparison: 23 learned, 0 ALU (same total neurons)
    // =========================================================
    println!("--- Fair: 23 learned + 0 ALU (same total neurons) ---");
    {
        let mut net = HybridNet::new(23, 0, "", &mut StdRng::seed_from_u64(77));
        println!("  trainable={} total_h={}", net.trainable_params(), net.total_h());
        let t1 = Instant::now();
        net.train_adam(&train_oh, &test_oh, steps, batch);
        let acc = net.accuracy(&test_oh[..2000]); let t5 = net.top5(&test_oh[..2000]);
        println!("  => test={:.1}% top5={:.1}% ({:.1}s)\n", acc*100.0, t5*100.0, t1.elapsed().as_secs_f64());
        results.push(("23 learned".into(), net.trainable_params(), 23, acc, t5));
    }

    // =========================================================
    // 6. Big hybrid: 19 learned + 8 ALU (more total capacity)
    // =========================================================
    println!("--- Big hybrid: 19 learned + 8 ALU (mixed) ---");
    {
        let mut net = HybridNet::new(19, 8, "mixed", &mut StdRng::seed_from_u64(77));
        println!("  trainable={} total_h={}", net.trainable_params(), net.total_h());
        let t1 = Instant::now();
        net.train_adam(&train_oh, &test_oh, steps, batch);
        let acc = net.accuracy(&test_oh[..2000]); let t5 = net.top5(&test_oh[..2000]);
        println!("  => test={:.1}% top5={:.1}% ({:.1}s)\n", acc*100.0, t5*100.0, t1.elapsed().as_secs_f64());
        results.push(("19 learned + 8 ALU".into(), net.trainable_params(), 27, acc, t5));
    }

    // =========================================================
    // VERDICT
    // =========================================================
    println!("=== VERDICT ===");
    println!("  {:>25} {:>10} {:>6} {:>8} {:>8}", "config", "trainable", "total", "test%", "top5%");
    println!("  {}", "=".repeat(62));
    for (label, tp, th, acc, t5) in &results {
        println!("  {:>25} {:>10} {:>6} {:>7.1}% {:>7.1}%", label, tp, th, acc*100.0, t5*100.0);
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
    println!("\n=== DONE ===");
}
