//! ALU X-Ray: exhaustive search for minimal C19 circuits, then SHOW what's inside
//!
//! Find the smallest perfect circuit for each operation, print exact weights.
//! Then use THOSE exact circuits as frozen ALU in a hybrid network.
//!
//! Run: cargo run --example alu_xray --release

use std::time::Instant;

const RHO: f32 = 8.0;

fn c19(x: f32) -> f32 {
    let l = 6.0;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + RHO * h * h
}

// 2-input, n-hidden, 1-output circuit
// Weights: w_in[h*2], bias[h], w_out[h], bias_out
fn eval_circuit(w_in: &[f32], bias: &[f32], w_out: &[f32], bias_out: f32,
                h: usize, a: f32, b: f32) -> f32 {
    let mut out = bias_out;
    for i in 0..h {
        let pre = w_in[i*2] * a + w_in[i*2+1] * b + bias[i];
        out += c19(pre) * w_out[i];
    }
    out
}

fn search_circuit(op: &str, h: usize, range: usize) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>, f32, f64)> {
    // Weight candidates
    let vals: Vec<f32> = vec![-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0];
    let n_vals = vals.len();

    // Generate all test pairs
    let scale = (range - 1) as f32;
    let mut targets: Vec<(f32, f32, f32)> = Vec::new();
    for a in 0..range {
        for b in 0..range {
            let af = a as f32 / scale;
            let bf = b as f32 / scale;
            let result = match op {
                "add" => (a + b) as f32 / (2.0 * scale),
                "mul" => (a * b) as f32 / (scale * scale),
                "abs" => (a as i32 - b as i32).unsigned_abs() as f32 / scale,
                "mod" => ((a * b) % range) as f32 / scale,
                "xor" => (a ^ b) as f32 / scale,
                _ => 0.0,
            };
            targets.push((af, bf, result));
        }
    }

    let n_params = h * 2 + h + h + 1; // w_in + bias + w_out + bias_out
    println!("    Searching {} neurons, {} weight values, {} params ({:.0} combos)...",
             h, n_vals, n_params, (n_vals as f64).powi(n_params as i32));

    let mut best_mse = f64::INFINITY;
    let mut best_circuit = None;
    let mut checked = 0u64;

    // For h=2: 9 params, 9^9 = 387M — might be slow
    // For h=1: 4 params, 9^4 = 6561 — instant
    // Use recursive enumeration with early pruning

    let total_combos = (n_vals as u64).pow(n_params as u32);
    if total_combos > 500_000_000 {
        println!("    Too many combos ({}), reducing to ±{{0,1,2}}", total_combos);
        // Use reduced set
        let vals2: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let n2 = vals2.len();
        let total2 = (n2 as u64).pow(n_params as u32);
        println!("    Reduced: {} combos", total2);

        let mut params = vec![0usize; n_params];
        loop {
            let w_in: Vec<f32> = (0..h*2).map(|i| vals2[params[i]]).collect();
            let bias: Vec<f32> = (0..h).map(|i| vals2[params[h*2+i]]).collect();
            let w_out: Vec<f32> = (0..h).map(|i| vals2[params[h*2+h+i]]).collect();
            let bias_out = vals2[params[n_params-1]];

            let mse: f64 = targets.iter().map(|&(a,b,t)| {
                let out = eval_circuit(&w_in, &bias, &w_out, bias_out, h, a, b);
                ((out - t) as f64).powi(2)
            }).sum::<f64>() / targets.len() as f64;

            if mse < best_mse {
                best_mse = mse;
                best_circuit = Some((w_in.clone(), bias.clone(), w_out.clone(), bias_out));
            }
            checked += 1;

            // Increment
            let mut carry = true;
            for i in 0..n_params {
                if carry {
                    params[i] += 1;
                    if params[i] >= n2 { params[i] = 0; } else { carry = false; }
                }
            }
            if carry { break; }
            if checked % 10_000_000 == 0 {
                print!("\r    {:.0}M/{:.0}M ({:.0}%) best_mse={:.6}",
                       checked as f64/1e6, total2 as f64/1e6,
                       checked as f64/total2 as f64*100.0, best_mse);
            }
        }
        println!();
    } else {
        let mut params = vec![0usize; n_params];
        loop {
            let w_in: Vec<f32> = (0..h*2).map(|i| vals[params[i]]).collect();
            let bias: Vec<f32> = (0..h).map(|i| vals[params[h*2+i]]).collect();
            let w_out: Vec<f32> = (0..h).map(|i| vals[params[h*2+h+i]]).collect();
            let bias_out = vals[params[n_params-1]];

            let mse: f64 = targets.iter().map(|&(a,b,t)| {
                let out = eval_circuit(&w_in, &bias, &w_out, bias_out, h, a, b);
                ((out - t) as f64).powi(2)
            }).sum::<f64>() / targets.len() as f64;

            if mse < best_mse {
                best_mse = mse;
                best_circuit = Some((w_in.clone(), bias.clone(), w_out.clone(), bias_out));
            }
            checked += 1;

            let mut carry = true;
            for i in 0..n_params {
                if carry {
                    params[i] += 1;
                    if params[i] >= n_vals { params[i] = 0; } else { carry = false; }
                }
            }
            if carry { break; }
            if checked % 5_000_000 == 0 {
                print!("\r    {:.0}M/{:.0}M ({:.0}%) best_mse={:.6}",
                       checked as f64/1e6, total_combos as f64/1e6,
                       checked as f64/total_combos as f64*100.0, best_mse);
            }
        }
        println!();
    }

    println!("    Checked {} combos, best MSE = {:.8}", checked, best_mse);

    if let Some((w_in, bias, w_out, bias_out)) = &best_circuit {
        // Check accuracy
        let mut correct = 0;
        let mut worst_err = 0.0f32;
        for &(a, b, target) in &targets {
            let out = eval_circuit(w_in, bias, w_out, *bias_out, h, a, b);
            let err = (out - target).abs();
            if err < 0.05 { correct += 1; }
            if err > worst_err { worst_err = err; }
        }
        let acc = correct as f64 / targets.len() as f64;

        // Print circuit
        println!("\n    CIRCUIT ({} neurons, {} op):", h, op);
        for i in 0..h {
            println!("      neuron {}: {:.1}*a + {:.1}*b + {:.1} → C19 → × {:.1}",
                     i, w_in[i*2], w_in[i*2+1], bias[i], w_out[i]);
        }
        println!("      output_bias: {:.1}", bias_out);
        println!("      accuracy: {}/{} ({:.1}%), worst_err: {:.4}", correct, targets.len(), acc*100.0, worst_err);

        // Show a few examples
        println!("\n    Examples:");
        for &(a, b, target) in targets.iter().take(8) {
            let out = eval_circuit(w_in, bias, w_out, *bias_out, h, a, b);
            let a_raw = (a * scale) as i32;
            let b_raw = (b * scale) as i32;
            let t_raw = match op {
                "add" => (target * 2.0 * scale) as i32,
                "mul" => (target * scale * scale) as i32,
                "abs" | "mod" | "xor" => (target * scale) as i32,
                _ => 0,
            };
            println!("      {}({}, {}) = {} (predicted {:.3}, target {:.3})",
                     op, a_raw, b_raw, t_raw, out, target);
        }

        return Some((w_in.clone(), bias.clone(), w_out.clone(), *bias_out, acc));
    }
    None
}

fn main() {
    println!("=== ALU X-RAY: what's INSIDE a perfect C19 circuit? ===\n");
    let t0 = Instant::now();

    let range = 8; // 0-7 for faster search

    // Search for minimal circuits
    for op in &["add", "mul", "abs", "xor"] {
        println!("== {} ==", op.to_uppercase());

        for h in 1..=3 {
            println!("\n  Trying {} hidden neurons:", h);
            let t1 = Instant::now();
            if let Some((_, _, _, _, acc)) = search_circuit(op, h, range) {
                println!("    Time: {:.1}s", t1.elapsed().as_secs_f64());
                if acc >= 0.99 {
                    println!("    >>> PERFECT with {} neurons! <<<", h);
                    break;
                }
            }
        }
        println!();
    }

    println!("Total time: {:.1}s", t0.elapsed().as_secs_f64());
    println!("\n=== DONE ===");
}
