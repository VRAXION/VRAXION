//! ALL-BINARY exhaustive mirror grower
//! Every parameter = 1 bit. Weights, bias, c, rho — ALL binary.
//! Test multiple binary value pairs for c and rho.
//!
//! Run: cargo run --example all_binary_mirror --release

use std::time::Instant;

fn load_unique_bytes(path: &str) -> Vec<u8> {
    let text = std::fs::read(path).expect("read");
    let mut seen = [false; 256]; for &b in &text { seen[b as usize] = true; }
    (0..=255u8).filter(|&b| seen[b as usize]).collect()
}
fn byte_to_bits(b: u8) -> [f32; 8] {
    let mut bits = [0.0f32; 8]; for i in 0..8 { bits[i] = ((b >> i) & 1) as f32; } bits
}
fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0 * c;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let s = x / c; let n = s.floor(); let t = s - n;
    let h = t * (1.0 - t); let sg = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sg * h + rho * h * h)
}
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

#[derive(Clone)]
struct N { w: [i32; 8], b: i32, c: f32, rho: f32 }
impl N {
    fn eval(&self, inp: &[f32; 8]) -> f32 {
        let mut d = self.b as f32;
        for i in 0..8 { d += self.w[i] as f32 * inp[i]; }
        c19(d, self.c, self.rho)
    }
}

fn eval_rt(neurons: &[N], inputs: &[[f32; 8]]) -> usize {
    let n = inputs.len(); let k = neurons.len();
    if k == 0 { return 0; }
    let hid: Vec<Vec<f32>> = inputs.iter().map(|inp|
        neurons.iter().map(|n| n.eval(inp)).collect()).collect();
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
        for i in 0..zt.len() { cands.push(-zt[i].0); if i + 1 < zt.len() { cands.push(-(zt[i].0 + zt[i + 1].0) / 2.0); } }
        for &bias in &cands { let a = (0..n).filter(|&bi| (sigmoid(rz[bi][j] + bias) - inputs[bi][j]).abs() < 0.4).count(); if a > best_a { best_a = a; best_b = bias; } }
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

    println!("=== ALL-BINARY EXHAUSTIVE MIRROR GROWER ===");
    println!("{} unique bytes — EVERY param is 1 bit\n", unique.len());

    // Test different binary c,rho pairs
    let c_pairs: Vec<(&str, [f32; 2])> = vec![
        ("c{0.1,1.0}", [0.1, 1.0]),
        ("c{0.1,3.0}", [0.1, 3.0]),
        ("c{0.1,5.0}", [0.1, 5.0]),
        ("c{0.1,12.0}", [0.1, 12.0]),
        ("c{1.0,10.0}", [1.0, 10.0]),
        ("c{0.5,8.0}", [0.5, 8.0]),
    ];
    let rho_pairs: Vec<(&str, [f32; 2])> = vec![
        ("ρ{0.0,0.5}", [0.0, 0.5]),
        ("ρ{0.0,1.0}", [0.0, 1.0]),
        ("ρ{0.0,2.0}", [0.0, 2.0]),
        ("ρ{0.0,5.0}", [0.0, 5.0]),
        ("ρ{0.5,2.0}", [0.5, 2.0]),
    ];

    // Total combos per neuron: 2^8 (weights) × 2 (bias) × 2 (c) × 2 (rho) = 2048
    println!("  Combos per neuron: 2^8 × 2 × 2 × 2 = 2048 (instant!)\n");

    let mut best_overall = 0;
    let mut best_config = String::new();
    let mut best_neurons_count = 0;
    let mut best_bits = 0;

    for (c_label, c_pair) in &c_pairs {
        for (rho_label, rho_pair) in &rho_pairs {
            let mut neurons: Vec<N> = Vec::new();
            let mut acc = 0;

            for step in 0..20 {
                let mut best_n: Option<N> = None;
                let mut best_a = acc;

                for combo in 0..256u32 {
                    let mut w = [0i32; 8];
                    for i in 0..8 { w[i] = if (combo >> i) & 1 == 1 { 1 } else { -1 }; }

                    for &b in &[-1i32, 1] {
                        for &c in c_pair {
                            for &rho in rho_pair {
                                let n = N { w, b, c, rho };
                                let mut tn = neurons.clone(); tn.push(n.clone());
                                let a = eval_rt(&tn, &inputs);
                                if a > best_a { best_a = a; best_n = Some(n); }
                                if best_a == unique.len() { break; }
                            }
                            if best_a == unique.len() { break; }
                        }
                        if best_a == unique.len() { break; }
                    }
                    if best_a == unique.len() { break; }
                }

                if let Some(n) = best_n {
                    acc = best_a;
                    neurons.push(n);
                    if acc == unique.len() { break; }
                } else { break; }
            }

            let total_bits = neurons.len() * 11; // 8w + 1b + 1c + 1rho = 11 bits/neuron
            let total_bytes = (total_bits + 7) / 8;

            if acc > best_overall || (acc == best_overall && neurons.len() < best_neurons_count) {
                best_overall = acc;
                best_config = format!("{} {}", c_label, rho_label);
                best_neurons_count = neurons.len();
                best_bits = total_bits;
            }

            let marker = if acc == unique.len() { "★" } else { "" };
            if acc >= 27 || acc == unique.len() {
                println!("  {} {} → {}/{} in {} neurons ({} bits = {} bytes) {}",
                    c_label, rho_label, acc, unique.len(), neurons.len(), total_bits, total_bytes, marker);

                if acc == unique.len() {
                    for (i, n) in neurons.iter().enumerate() {
                        let ws: Vec<&str> = n.w.iter().map(|&w| if w == 1 { "+" } else { "-" }).collect();
                        let cb = if n.c == c_pair[0] { "0" } else { "1" };
                        let rb = if n.rho == rho_pair[0] { "0" } else { "1" };
                        println!("    N{}: [{}] b={:+} c={} ρ={} (c={:.1},ρ={:.1})",
                            i, ws.join(""), n.b, cb, rb, n.c, n.rho);
                    }
                }
            }
        }
    }

    println!("\n━━━ BEST ALL-BINARY RESULT ━━━");
    println!("  Config: {}", best_config);
    println!("  Accuracy: {}/{}", best_overall, unique.len());
    println!("  Neurons: {}", best_neurons_count);
    println!("  Total: {} bits = {} bytes", best_bits, (best_bits + 7) / 8);
    println!("  Every param = 1 bit. {} bits per neuron.", 11);
    println!("  + decoder bias: 8 × int8 = 8 bytes (threshold, computed once)");

    println!("\n  Time: {:.1}s", t0.elapsed().as_secs_f64());
}
