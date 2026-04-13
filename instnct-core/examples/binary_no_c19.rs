//! Binary signed exhaustive WITHOUT C19 — pure threshold neurons
//! Tests: do we actually need C19, or is threshold enough?
//!
//! Run: cargo run --example binary_no_c19 --release

use std::time::Instant;

fn load_unique_bytes(path: &str) -> Vec<u8> {
    let text = std::fs::read(path).expect("read");
    let mut seen = [false; 256]; for &b in &text { seen[b as usize] = true; }
    (0..=255u8).filter(|&b| seen[b as usize]).collect()
}
fn byte_to_bits(b: u8) -> [f32; 8] {
    let mut bits = [0.0f32; 8]; for i in 0..8 { bits[i] = ((b >> i) & 1) as f32; } bits
}
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

#[derive(Clone)]
struct N { w: [i32; 8], b: i32 }

impl N {
    // Pure threshold: output = continuous value (dot product as-is, no C19)
    fn eval_raw(&self, inp: &[f32; 8]) -> f32 {
        let mut d = self.b as f32;
        for i in 0..8 { d += self.w[i] as f32 * inp[i]; }
        d  // raw dot product — decoder handles the rest
    }
}

fn eval_rt(neurons: &[N], inputs: &[[f32; 8]]) -> usize {
    let n = inputs.len(); let k = neurons.len();
    if k == 0 { return 0; }
    let hid: Vec<Vec<f32>> = inputs.iter().map(|inp|
        neurons.iter().map(|n| n.eval_raw(inp)).collect()
    ).collect();
    let mut rz = vec![[0.0f32; 8]; n];
    for bi in 0..n { for j in 0..8 {
        let mut z = 0.0f32;
        for ki in 0..k { z += neurons[ki].w[j] as f32 * hid[bi][ki]; }
        rz[bi][j] = z;
    }}
    let mut bb = [0.0f32; 8];
    for j in 0..8 {
        let mut best_a = 0; let mut best_b = 0.0f32;
        let mut zt: Vec<(f32, f32)> = (0..n).map(|bi| (rz[bi][j], inputs[bi][j])).collect();
        zt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut cands: Vec<f32> = vec![-100.0, 100.0];
        for i in 0..zt.len() { cands.push(-zt[i].0); if i+1 < zt.len() { cands.push(-(zt[i].0 + zt[i+1].0) / 2.0); } }
        for &bias in &cands {
            let a = (0..n).filter(|&bi| (sigmoid(rz[bi][j] + bias) - inputs[bi][j]).abs() < 0.4).count();
            if a > best_a { best_a = a; best_b = bias; }
        }
        bb[j] = best_b;
    }
    let mut c = 0;
    for bi in 0..n { if (0..8).all(|j| (sigmoid(rz[bi][j] + bb[j]) - inputs[bi][j]).abs() < 0.4) { c += 1; } }
    c
}

fn main() {
    let t0 = Instant::now();
    let unique = load_unique_bytes("instnct-core/tests/fixtures/alice_corpus.txt");
    let inputs: Vec<[f32; 8]> = unique.iter().map(|&b| byte_to_bits(b)).collect();

    println!("=== BINARY SIGNED — NO C19 (pure dot product) ===");
    println!("{} unique bytes\n", unique.len());

    let vals = [-1i32, 1];
    // Also try threshold values for bias
    let bias_vals = [-1i32, 0, 1]; // ternary bias gives more options

    for &(blabel, bvals) in &[("bias{-1,+1}", &[-1i32, 1][..]), ("bias{-1,0,+1}", &[-1i32, 0, 1][..])] {
        let combos = 2usize.pow(8) * bvals.len();
        println!("━━━ signed{{-1,+1}} weights, {} — {} combos/neuron ━━━", blabel, combos);

        let mut neurons: Vec<N> = Vec::new();
        let mut acc = 0;

        for step in 0..20 {
            let mut best_n: Option<N> = None;
            let mut best_a = acc;
            let ts = Instant::now();

            for combo in 0..2u32.pow(8) {
                let mut w = [0i32; 8];
                for i in 0..8 { w[i] = if (combo >> i) & 1 == 1 { 1 } else { -1 }; }

                for &b in bvals.iter() {
                    let n = N { w, b };
                    let mut tn = neurons.clone(); tn.push(n.clone());
                    let a = eval_rt(&tn, &inputs);
                    if a > best_a { best_a = a; best_n = Some(n); if a == unique.len() { break; } }
                }
                if best_a == unique.len() { break; }
            }

            if let Some(n) = best_n {
                acc = best_a;
                let ws: Vec<String> = n.w.iter().map(|w| format!("{:>2}", w)).collect();
                println!("  N{}: {}/{} [{}] b={} ({:.2}s)",
                    step, acc, unique.len(), ws.join(","), n.b, ts.elapsed().as_secs_f64());
                neurons.push(n);
                if acc == unique.len() {
                    let w_bits = neurons.len() * 8; // 1 bit per weight
                    let b_bits = neurons.len() * 2; // bias: ternary = 2 bits
                    let total_bits = w_bits + b_bits;
                    println!("\n  ★★★ PERFECT: {} neurons ★★★", neurons.len());
                    println!("  Weights: {} bit (1 bit each)", w_bits);
                    println!("  Biases:  {} bit", b_bits);
                    println!("  TOTAL:   {} bit = {} byte", total_bits, (total_bits + 7) / 8);
                    println!("  NO C19. NO FLOAT. Pure binary + threshold.");
                    break;
                }
            } else {
                println!("  N{}: no improvement, stop", step);
                break;
            }
        }
        println!("  Final: {}/{}, {} neurons, {:.1}s\n", acc, unique.len(), neurons.len(), t0.elapsed().as_secs_f64());
    }

    println!("Total: {:.1}s", t0.elapsed().as_secs_f64());
}
