//! ALU Math: compare architectures on arithmetic (where ALU SHOULD help)
//!
//! Tasks: ADD, MUL, |a-b| on inputs 0-15
//! Compare: ReLU MLP vs C19 MLP vs C19+ALU hybrid vs C19 tick=2
//!
//! Run: cargo run --example alu_math --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};

const RHO: f32 = 8.0;

fn c19(x: f32) -> f32 {
    let l = 6.0;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + RHO * h * h
}
fn c19d(x: f32) -> f32 {
    let l = 6.0;
    if x >= l || x <= -l { return 1.0; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    (sgn + 2.0 * RHO * h) * (1.0 - 2.0 * t)
}

// Simple regression MLP: 2 inputs → H hidden → 1 output
#[derive(Clone)]
struct Net {
    h: usize,
    n_alu: usize,
    ticks: usize,
    act: String, // "relu" or "c19"
    w1: Vec<f32>, b1: Vec<f32>,         // 2 → h (learned)
    w_alu: Vec<f32>, b_alu: Vec<f32>,   // h → n_alu (frozen)
    w_rec: Vec<f32>,                     // (h+n_alu) → h (recurrent, only if ticks>1)
    w2: Vec<f32>, b2: f32,              // (h+n_alu) → 1 output
}

impl Net {
    fn new(h: usize, n_alu: usize, ticks: usize, act: &str, rng: &mut StdRng) -> Self {
        let total = h + n_alu;
        let sc = 0.5;
        // ALU: simple fixed circuits reading pairs of learned neurons
        let mut w_alu = vec![0.0f32; n_alu * h];
        let mut b_alu = vec![0.0f32; n_alu];
        for i in 0..n_alu {
            let a = (i * 3 + 1) % h;
            let b = (i * 7 + 3) % h;
            match i % 4 {
                0 => { w_alu[i*h+a] = 1.0; w_alu[i*h+b] = 1.0; }       // add
                1 => { w_alu[i*h+a] = 1.0; w_alu[i*h+b] = -1.0; }      // subtract
                2 => { w_alu[i*h+a] = 2.0; w_alu[i*h+b] = 0.5; }       // scale
                _ => { w_alu[i*h+a] = 1.0; b_alu[i] = 0.5; }           // bias+activate
            }
        }
        Net {
            h, n_alu, ticks, act: act.to_string(),
            w1: (0..h*2).map(|_| rng.gen_range(-sc..sc)).collect(),
            b1: vec![0.0; h],
            w_alu, b_alu,
            w_rec: (0..h*total).map(|_| rng.gen_range(-0.1..0.1)).collect(),
            w2: (0..total).map(|_| rng.gen_range(-sc..sc)).collect(),
            b2: 0.0,
        }
    }

    fn activate(&self, x: f32) -> f32 {
        if self.act == "c19" { c19(x) } else { x.max(0.0) }
    }
    fn activate_d(&self, x: f32) -> f32 {
        if self.act == "c19" { c19d(x) } else { if x > 0.0 { 1.0 } else { 0.0 } }
    }

    fn forward(&self, a: f32, b: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
        let (h, na) = (self.h, self.n_alu);
        let total = h + na;
        let input = [a, b];
        let mut state = vec![0.0f32; h];
        let mut all_pre = Vec::new();

        for _t in 0..self.ticks {
            let mut pre = vec![0.0f32; h];
            for i in 0..h {
                let mut s = self.b1[i];
                s += input[0] * self.w1[i*2];
                s += input[1] * self.w1[i*2+1];
                if self.ticks > 1 {
                    // Recurrent: read from previous state + ALU
                    // (simplified: just read from state)
                    for j in 0..h { s += state[j] * self.w_rec[i*total+j]; }
                }
                pre[i] = s;
            }
            for i in 0..h { state[i] = if self.ticks > 1 { state[i] + self.activate(pre[i]) } else { self.activate(pre[i]) }; }
            all_pre.push(pre);
        }

        // ALU neurons (frozen, read from learned state)
        let mut alu_pre = vec![0.0f32; na];
        let mut alu_act = vec![0.0f32; na];
        for i in 0..na {
            let mut s = self.b_alu[i];
            for j in 0..h { s += state[j] * self.w_alu[i*h+j]; }
            alu_pre[i] = s;
            alu_act[i] = c19(s); // ALU always uses C19
        }

        // Output
        let mut all_act = Vec::with_capacity(total);
        all_act.extend_from_slice(&state);
        all_act.extend_from_slice(&alu_act);
        let mut out = self.b2;
        for i in 0..total { out += all_act[i] * self.w2[i]; }

        (all_act, [all_pre.concat(), alu_pre].concat(), vec![state.clone(), alu_act].concat(), out)
    }

    fn train(&mut self, data: &[(f32,f32,f32)], test: &[(f32,f32,f32)], steps: usize) {
        let h = self.h;
        let na = self.n_alu;
        let total = h + na;
        // Count trainable params
        let np = h*2 + h + total + 1 + if self.ticks > 1 { h*total } else { 0 };
        let mut mv = vec![0.0f32; np];
        let mut vv = vec![0.0f32; np];
        let mut rng = StdRng::seed_from_u64(99);
        let mut sh = data.to_vec();

        for step in 1..=steps {
            sh.shuffle(&mut rng);
            let bs = sh.len().min(64);
            let batch = &sh[..bs];

            let mut grad = vec![0.0f32; np];
            for &(a, b, target) in batch {
                let (all_act, all_pre, _, out) = self.forward(a, b);
                let d_out = 2.0 * (out - target); // MSE derivative

                let mut d_all = vec![0.0f32; total];
                let mut o = h*2 + h; // offset to w2 in param vec
                if self.ticks > 1 { // skip w_rec in param indexing
                    // w_rec params after b1
                    // Actually let me index properly
                }
                // Simpler: use direct param access
                // Param layout: w1(h*2), b1(h), [w_rec(h*total) if ticks>1], w2(total), b2(1)

                let w2_off = h*2 + h + if self.ticks > 1 { h*total } else { 0 };
                let b2_off = w2_off + total;

                // Output grad
                grad[b2_off] += d_out;
                for i in 0..total {
                    grad[w2_off + i] += d_out * all_act[i];
                    d_all[i] += d_out * self.w2[i];
                }

                // ALU backprop (frozen weights but propagate to learned)
                let mut d_state = vec![0.0f32; h];
                for i in 0..h { d_state[i] += d_all[i]; } // direct path
                for i in 0..na {
                    let deriv = c19d(all_pre[h + i]); // ALU always c19
                    let d_alu_pre = d_all[h + i] * deriv;
                    for j in 0..h { d_state[j] += d_alu_pre * self.w_alu[i*h+j]; }
                }

                // Learned neuron backprop (last tick only for simplicity)
                let pre_offset = if self.ticks > 1 { (self.ticks-1)*h } else { 0 };
                for i in 0..h {
                    let deriv = self.activate_d(all_pre[pre_offset + i]);
                    let d_pre = d_state[i] * deriv;
                    grad[i*2] += d_pre * a;     // w1
                    grad[i*2+1] += d_pre * b;   // w1
                    grad[h*2 + i] += d_pre;      // b1
                }
            }

            let bsf = bs as f32;
            let mut pv = Vec::with_capacity(np);
            pv.extend(&self.w1); pv.extend(&self.b1);
            if self.ticks > 1 { pv.extend(&self.w_rec); }
            pv.extend(&self.w2); pv.push(self.b2);

            let t = step as f32;
            let b1c = 1.0 - 0.9f32.powf(t); let b2c = 1.0 - 0.999f32.powf(t);
            for i in 0..np {
                grad[i] /= bsf;
                mv[i] = 0.9*mv[i] + 0.1*grad[i];
                vv[i] = 0.999*vv[i] + 0.001*grad[i]*grad[i];
                pv[i] -= 0.003 * (mv[i]/b1c) / ((vv[i]/b2c).sqrt()+1e-8);
            }

            let mut o = 0;
            self.w1.copy_from_slice(&pv[o..o+h*2]); o+=h*2;
            self.b1.copy_from_slice(&pv[o..o+h]); o+=h;
            if self.ticks > 1 { self.w_rec.copy_from_slice(&pv[o..o+h*total]); o+=h*total; }
            self.w2.copy_from_slice(&pv[o..o+total]); o+=total;
            self.b2 = pv[o];

            if step % 2000 == 0 || step == steps {
                let (mse, acc) = self.eval(test);
                println!("    step {:>5}: mse={:.4} acc={:.1}%", step, mse, acc*100.0);
            }
        }
    }

    fn eval(&self, data: &[(f32,f32,f32)]) -> (f64, f64) {
        let mut mse = 0.0f64;
        let mut correct = 0;
        for &(a, b, target) in data {
            let (_, _, _, out) = self.forward(a, b);
            let err = (out - target) as f64;
            mse += err * err;
            if (out - target).abs() < 0.5 { correct += 1; }
        }
        (mse / data.len() as f64, correct as f64 / data.len() as f64)
    }
}

fn make_data(op: &str, range: usize) -> Vec<(f32, f32, f32)> {
    let mut data = Vec::new();
    let scale = range as f32;
    let max_out = match op {
        "add" => (2 * (range-1)) as f32,
        "mul" => ((range-1)*(range-1)) as f32,
        "abs" => (range-1) as f32,
        "compound" => (2 * (range-1) * (range-1)) as f32, // (a+b)*|a-b|
        "poly" => {
            let r = (range-1) as f32;
            (r*r + 2.0*r*r + r) as f32 // a²+2ab+b = (a+b)²-b²+b
        },
        "div_mod" => (range-1) as f32, // (a*b) mod range
        "bitwise" => (range-1) as f32, // (a XOR b)
        _ => scale,
    };
    for a in 0..range {
        for b in 0..range {
            let result = match op {
                "add" => (a + b) as f32,
                "mul" => (a * b) as f32,
                "abs" => (a as i32 - b as i32).unsigned_abs() as f32,
                "compound" => ((a + b) as f32) * ((a as i32 - b as i32).unsigned_abs() as f32),
                "poly" => (a*a + 2*a*b + b) as f32,
                "div_mod" => ((a * b) % range) as f32,
                "bitwise" => (a ^ b) as f32,
                _ => 0.0,
            };
            data.push((a as f32 / scale, b as f32 / scale, result / max_out.max(1.0)));
        }
    }
    data
}

fn main() {
    println!("=== ALU MATH: architecture comparison on arithmetic ===\n");

    let range = 16; // 0-15
    let steps = 10000;

    // Complex tasks where multi-step computation should matter
    for op in &["compound", "poly", "div_mod", "bitwise"] {
        println!("======== TASK: {} (0-{}) ========", op.to_uppercase(), range-1);
        println!("  {}\n", match *op {
            "compound" => "(a+b)*|a-b| — needs ADD, MUL, ABS composed",
            "poly" => "a²+2ab+b — polynomial, needs MUL+ADD",
            "div_mod" => "(a*b) mod 16 — modular arithmetic",
            "bitwise" => "a XOR b — binary logic",
            _ => "",
        });

        let mut all_data = make_data(op, range);
        let mut rng = StdRng::seed_from_u64(42);
        all_data.shuffle(&mut rng);
        let split = all_data.len() * 3 / 4; // 75/25 split (harder)
        let train = all_data[..split].to_vec();
        let test = all_data[split..].to_vec();
        println!("  Train: {}, Test: {}\n", train.len(), test.len());

        let h = 12; // SMALL — force architectures to matter
        let configs: Vec<(&str, usize, usize, &str)> = vec![
            ("ReLU MLP",        h, 0, "relu"),
            ("C19 MLP",         h, 0, "c19"),
            ("C19 + 4 ALU",     h-4, 4, "c19"),   // fewer learned + ALU
            ("C19 + 8 ALU",     h-8, 8, "c19"),   // half learned, half ALU
            ("C19 t=2 (no ALU)", h, 0, "c19"),     // tick recurrence
            ("C19 t=2 + 4 ALU", h-4, 4, "c19"),   // ticks + ALU
        ];

        let mut results: Vec<(String, f64)> = Vec::new();

        for (name, n_h, n_alu, act) in &configs {
            let ticks = if name.contains("t=2") { 2 } else { 1 };
            // Run 3 seeds for robustness
            let mut accs = Vec::new();
            for seed in [77u64, 123, 456] {
            let mut net = Net::new(*n_h, *n_alu, ticks, act, &mut StdRng::seed_from_u64(seed));
            if seed == 77 { println!("  --- {} (h={}, alu={}, ticks={}) ---", name, n_h, n_alu, ticks); }
            net.train(&train, &test, steps);
            let (mse, acc) = net.eval(&test);
            accs.push(acc);
            if seed == 77 { println!("  => mse={:.4} acc={:.1}%", mse, acc*100.0); }
            }
            let avg = accs.iter().sum::<f64>() / accs.len() as f64;
            let min = accs.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = accs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            println!("  3-seed avg={:.1}% (min={:.1}% max={:.1}%)\n", avg*100.0, min*100.0, max*100.0);
            results.push((name.to_string(), avg));
        }

        println!("  SUMMARY ({}):", op.to_uppercase());
        println!("  {:>20} {:>8}", "model", "acc%");
        for (name, acc) in &results {
            println!("  {:>20} {:>7.1}%", name, acc*100.0);
        }
        println!();
    }

    println!("=== DONE ===");
}
