//! ALU All-in-One: exhaustive search for complete circuits WITHOUT gate-by-gate decomposition
//!
//! Can a full-adder be done in 3 neurons instead of 5?
//! Can a 2-bit adder be done in fewer neurons than 2×5=10?
//! Raw analog flow between neurons — no binarization.
//!
//! Run: cargo run --example alu_allinone --release

fn c19(x: f32, rho: f32) -> f32 {
    let c = 1.0f32; let l = 6.0;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

/// N-neuron circuit with 3 binary inputs (a, b, cin) and 2 binary outputs (sum, cout)
/// Each neuron reads all 3 inputs + all previous neurons' RAW outputs (analog, no binarize)
/// Last 2 neurons are thresholded for sum and cout
fn eval_full_adder(
    weights: &[f32], // per neuron: [w_a, w_b, w_cin, w_prev0, w_prev1, ..., bias, rho]
    n: usize, a: f32, b: f32, cin: f32
) -> (f32, f32, Vec<f32>) { // (sum_raw, cout_raw, all_outputs)
    let params_per = 3 + n + 2; // 3 inputs + n-1 prev neurons max (padded to n) + bias + rho
    // Actually: neuron i reads 3 inputs + i previous neurons + bias + rho
    // But for uniform indexing: all neurons have slots for n prev (unused ones = 0 weight)

    let mut outputs = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * params_per;
        let mut s = weights[off + 3 + n]; // bias
        s += weights[off + 0] * a;
        s += weights[off + 1] * b;
        s += weights[off + 2] * cin;
        for j in 0..i {
            s += weights[off + 3 + j] * outputs[j];
        }
        let rho = weights[off + 3 + n + 1];
        outputs.push(c19(s, rho));
    }
    // Last 2 outputs → sum, cout (thresholded externally)
    let sum_raw = if n >= 2 { outputs[n-2] } else { outputs[0] };
    let cout_raw = outputs[n-1];
    (sum_raw, cout_raw, outputs)
}

/// Exhaustive search for N-neuron full adder
fn search_full_adder(n: usize) -> bool {
    println!("=== Full Adder: {} neurons (analog chaining) ===", n);

    let truth: Vec<(u8,u8,u8,u8,u8)> = vec![
        // a, b, cin, sum, cout
        (0,0,0, 0,0),
        (0,0,1, 1,0),
        (0,1,0, 1,0),
        (0,1,1, 0,1),
        (1,0,0, 1,0),
        (1,0,1, 0,1),
        (1,1,0, 0,1),
        (1,1,1, 1,1),
    ];

    // Weight candidates (coarser for larger n)
    let vals: Vec<f32> = if n <= 3 {
        vec![-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]  // 9 values
    } else {
        vec![-2.0, -1.0, 0.0, 1.0, 2.0] // 5 values for speed
    };
    let rhos: Vec<f32> = vec![0.0, 4.0, 8.0, 16.0];
    let thresholds: Vec<f32> = vec![-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0];

    let params_per = 3 + n + 2; // inputs + prev_slots + bias + rho
    let n_weight_params = n * (3 + n); // weight+bias slots per neuron (excl rho)
    let total_configs = (vals.len() as f64).powf(n_weight_params as f64)
                        * (rhos.len() as f64).powf(n as f64)
                        * (thresholds.len() as f64).powf(2.0); // 2 thresholds for sum,cout

    println!("  {} params/neuron, ~{:.0} total weight combos (estimated)", params_per, total_configs);

    if total_configs > 5e9 {
        println!("  Too many combos, skipping full exhaustive. Trying random search...");
        return random_search_full_adder(n, &truth, 50_000_000);
    }

    // For small n, try exhaustive
    let mut best_correct = 0u32;
    let mut best_weights: Option<Vec<f32>> = None;
    let mut best_thrs = (0.0f32, 0.0f32);
    let mut checked = 0u64;

    // Recursive enumeration
    let n_wb = n * (3 + n + 1); // weight+bias params (no rho)
    let mut params = vec![0usize; n_wb];
    let nv = vals.len();

    'outer: loop {
        // For each weight combo, try all rho combos
        for rho_mask in 0..(rhos.len().pow(n as u32)) {
            let mut weights = vec![0.0f32; n * params_per];
            let mut pi = 0;
            for i in 0..n {
                let off = i * params_per;
                // Input weights
                for j in 0..3 { weights[off + j] = vals[params[pi]]; pi += 1; }
                // Prev neuron weights
                for j in 0..n {
                    if j < i { weights[off + 3 + j] = vals[params[pi]]; }
                    pi += 1;
                }
                // Bias
                weights[off + 3 + n] = vals[params[pi]]; pi += 1;
                // Rho
                let ri = (rho_mask / rhos.len().pow(i as u32)) % rhos.len();
                weights[off + 3 + n + 1] = rhos[ri];
            }

            // Eval all 8 truth table entries
            let mut outputs: Vec<(f32, f32)> = Vec::new();
            for &(a,b,cin,_,_) in &truth {
                let (sr, cr, _) = eval_full_adder(&weights, n, a as f32, b as f32, cin as f32);
                outputs.push((sr, cr));
            }

            // Try threshold combos
            for &thr_s in &thresholds {
                for &thr_c in &thresholds {
                    let correct: u32 = truth.iter().zip(&outputs).map(|(&(_,_,_,es,ec), &(sr,cr))| {
                        let s_ok = (if sr > thr_s {1u8} else {0}) == es;
                        let c_ok = (if cr > thr_c {1u8} else {0}) == ec;
                        if s_ok && c_ok { 1u32 } else { 0 }
                    }).sum();

                    if correct > best_correct {
                        best_correct = correct;
                        best_weights = Some(weights.clone());
                        best_thrs = (thr_s, thr_c);
                        if correct == 8 {
                            println!("  PERFECT found after {} checks!", checked);
                            print_circuit(&weights, n, params_per, thr_s, thr_c, &truth);
                            return true;
                        }
                    }
                }
            }
            checked += 1;
            if checked % 1_000_000 == 0 {
                print!("\r  {}M checked, best={}/8   ", checked/1_000_000, best_correct);
            }
        }

        // Increment weight params
        let mut carry = true;
        for i in 0..n_wb {
            if carry {
                params[i] += 1;
                if params[i] >= nv { params[i] = 0; } else { carry = false; }
            }
        }
        if carry { break 'outer; }
    }

    println!("\n  Exhaustive done: {} checks, best={}/8", checked, best_correct);
    if let Some(w) = &best_weights {
        print_circuit(w, n, params_per, best_thrs.0, best_thrs.1, &truth);
    }
    best_correct == 8
}

fn random_search_full_adder(n: usize, truth: &[(u8,u8,u8,u8,u8)], attempts: u64) -> bool {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let params_per = 3 + n + 2;
    let vals: Vec<f32> = vec![-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0];
    let rhos: Vec<f32> = vec![0.0, 4.0, 8.0, 16.0];
    let thresholds: Vec<f32> = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

    let mut best_correct = 0u32;
    let mut best_weights: Option<Vec<f32>> = None;
    let mut best_thrs = (0.0f32, 0.0f32);

    for attempt in 0..attempts {
        // Generate random weights using hash
        let mut weights = vec![0.0f32; n * params_per];
        let mut h = DefaultHasher::new();
        attempt.hash(&mut h);
        let hash = h.finish();

        for i in 0..n {
            let off = i * params_per;
            for j in 0..(3 + n + 1) {
                let idx = ((hash.wrapping_mul(6364136223846793005).wrapping_add((i*100+j) as u64)) >> 32) as usize % vals.len();
                weights[off + j] = vals[idx];
            }
            let ri = ((hash >> (i * 4)) as usize) % rhos.len();
            weights[off + 3 + n + 1] = rhos[ri];
        }

        let mut outputs: Vec<(f32,f32)> = Vec::new();
        for &(a,b,cin,_,_) in truth {
            let (sr, cr, _) = eval_full_adder(&weights, n, a as f32, b as f32, cin as f32);
            outputs.push((sr, cr));
        }

        for &thr_s in &thresholds {
            for &thr_c in &thresholds {
                let correct: u32 = truth.iter().zip(&outputs).map(|(&(_,_,_,es,ec), &(sr,cr))| {
                    let s_ok = (if sr > thr_s {1u8} else {0}) == es;
                    let c_ok = (if cr > thr_c {1u8} else {0}) == ec;
                    if s_ok && c_ok { 1 } else { 0 }
                }).sum();

                if correct > best_correct {
                    best_correct = correct;
                    best_weights = Some(weights.clone());
                    best_thrs = (thr_s, thr_c);
                    if correct == 8 {
                        println!("  PERFECT found at attempt {}!", attempt);
                        print_circuit(&weights, n, params_per, thr_s, thr_c, truth);
                        return true;
                    }
                }
            }
        }

        if attempt % 5_000_000 == 0 && attempt > 0 {
            println!("  {}M attempts, best={}/8", attempt/1_000_000, best_correct);
        }
    }

    println!("  Random search done: {}M attempts, best={}/8", attempts/1_000_000, best_correct);
    if let Some(w) = &best_weights {
        print_circuit(w, n, params_per, best_thrs.0, best_thrs.1, truth);
    }
    best_correct == 8
}

fn print_circuit(weights: &[f32], n: usize, params_per: usize, thr_s: f32, thr_c: f32,
                 truth: &[(u8,u8,u8,u8,u8)]) {
    println!("\n  CIRCUIT ({} neurons, analog chain):", n);
    for i in 0..n {
        let off = i * params_per;
        let wa = weights[off]; let wb = weights[off+1]; let wc = weights[off+2];
        let bias = weights[off + 3 + n];
        let rho = weights[off + 3 + n + 1];
        print!("    n{}: {:+.1}*a {:+.1}*b {:+.1}*cin", i, wa, wb, wc);
        for j in 0..i {
            let wp = weights[off + 3 + j];
            if wp != 0.0 { print!(" {:+.1}*n{}", wp, j); }
        }
        println!(" {:+.1}(bias) rho={:.0} → C19", bias, rho);
    }
    if n >= 2 {
        println!("    sum = n{} > {:.1},  cout = n{} > {:.1}", n-2, thr_s, n-1, thr_c);
    }

    println!("  Verify:");
    for &(a,b,cin,es,ec) in truth {
        let (sr, cr, outs) = eval_full_adder(weights, n, a as f32, b as f32, cin as f32);
        let s = if sr > thr_s {1u8} else {0};
        let c = if cr > thr_c {1u8} else {0};
        let ok = s == es && c == ec;
        println!("    {}+{}+{} = {}:{} (exp {}:{}) raw=[{}] {}",
                 a, b, cin, c, s, ec, es,
                 outs.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>().join(", "),
                 if ok {"✓"} else {"✗"});
    }
}

fn main() {
    println!("=== ALU ALL-IN-ONE: can we beat gate-by-gate? ===\n");
    println!("Gate-by-gate full adder: 5 neurons (2×XOR + 2×AND + OR)");
    println!("Question: can we do it in 2, 3, or 4 neurons with analog chaining?\n");

    let t0 = std::time::Instant::now();

    for n in 2..=5 {
        let found = search_full_adder(n);
        if found {
            println!("  >>> FULL ADDER in {} neurons! (vs 5 gate-by-gate) <<<", n);
            if n < 5 { println!("  >>> BETTER than gate decomposition! <<<"); }
        }
        println!();
    }

    println!("Total time: {:.1}s", t0.elapsed().as_secs_f64());
    println!("\n=== DONE ===");
}
