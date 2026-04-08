//! Hardcoded SUM circuit — no clip, no search. Does it generalize?
//!
//! RUNNING: sum_hardcoded
//! Run: cargo run --example sum_hardcoded --release

fn forward(input: &[f32], w: &[Vec<f32>], h: usize) -> Vec<f32> {
    let mut charge = vec![0.0f32; h];
    let mut act = vec![0.0f32; h];
    for t in 0..8 {
        if t == 0 { for i in 0..input.len().min(h) { act[i] = input[i]; } }
        let mut raw = vec![0.0f32; h];
        for i in 0..h { for j in 0..h { raw[i] += act[j] * w[j][i]; } }
        for i in 0..h { charge[i] += raw[i] * 0.3; charge[i] *= 0.85; }
        for i in 0..h { act[i] = (charge[i] - 0.1).max(0.0); }
        let total: f32 = act.iter().sum();
        if total > 0.0 { let d = 1.0 + 0.05 * total; for i in 0..h { act[i] /= d; } }
        // NO CLIP
    }
    charge
}

fn main() {
    let h = 18;
    let mut w = vec![vec![0.0f32; h]; h];
    for i in 0..8 { w[i][8] = 1.0; } // all inputs → SUM neuron
    // SUM → outputs: each output has DIFFERENT weight from SUM
    // Output[k] should respond maximally when SUM charge ≈ level for k active inputs
    // Weight encodes "preferred count": output[0] gets negative (inhibited by activity)
    // output[8] gets strong positive (excited by high activity)
    for k in 0..9usize {
        // Graded weight: output[k] prefers SUM level k
        // w = (k - 4) * 0.5 → output[0]=-2.0, output[4]=0, output[8]=2.0
        w[8][9 + k] = (k as f32 - 4.0) * 0.5;
    }
    // Also: lateral inhibition between output neurons (winner-take-all)
    for k in 0..9usize {
        for j in 0..9usize {
            if k != j { w[9+k][9+j] = -0.3; } // outputs inhibit each other
        }
    }

    println!("=== HARDCODED SUM + NO CLIP ===");
    println!("RUNNING: sum_hardcoded\n");

    println!("n_active → SUM_charge:");
    let mut sum_charges = Vec::new();
    for n in 0..=8usize {
        let mut input = vec![0.0f32; h]; for i in 0..n { input[i] = 1.0; }
        let c = forward(&input, &w, h);
        println!("  {} → {:.4}", n, c[8]);
        sum_charges.push(c[8]);
    }
    let distinct = sum_charges.windows(2).filter(|w| (w[1] - w[0]).abs() > 0.001).count();
    println!("  Distinct levels: {}/8\n", distinct);

    println!("Same sum=4, different digits:");
    for (a,b) in [(0usize,4usize),(1,3),(2,2),(3,1),(4,0)] {
        let mut input = vec![0.0f32; h]; for i in 0..a { input[i]=1.0; } for i in 0..b { input[4+i]=1.0; }
        println!("  {}+{} → SUM={:.4}", a, b, forward(&input, &w, h)[8]);
    }

    println!("\nAll 25 examples:");
    let mut correct = 0; let mut total = 0;
    let mut test_ok = 0; let mut test_n = 0;
    for a in 0..5usize { for b in 0..5usize {
        let target = a + b;
        let mut input = vec![0.0f32; h]; for i in 0..a { input[i]=1.0; } for i in 0..b { input[4+i]=1.0; }
        let c = forward(&input, &w, h);
        let pred = (0..9usize).max_by(|&i,&j| c[9+i].partial_cmp(&c[9+j]).unwrap()).unwrap();
        let ok = pred == target;
        if ok { correct += 1; }
        if target == 4 { test_n += 1; if ok { test_ok += 1; } }
        total += 1;
        println!("  {}+{}={} pred={} {}", a, b, target, pred, if ok { "✓" } else { "✗ MISS" });
    }}
    println!("\nAll: {}/{} = {:.0}%", correct, total, 100.0 * correct as f64 / total as f64);
    println!("Test (sum=4): {}/{} = {:.0}%", test_ok, test_n, 100.0 * test_ok as f64 / test_n as f64);
}
