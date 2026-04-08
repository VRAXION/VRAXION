//! Bidirectional pool: every neuron is BOTH input AND output.
//! No zones, no separation. Input injected into all, readout from all.
//! Use the PROVEN SUM circuit (uniform weights) but bidirectional.
//!
//! RUNNING: bidirectional_sum
//!
//! Run: cargo run --example bidirectional_sum --release

const TICKS: usize = 8;
const CHARGE_RATE: f32 = 0.3;
const LEAK: f32 = 0.85;
const THRESHOLD: f32 = 0.1;
const DIGITS: usize = 5;
const SUMS: usize = 9;

fn forward(input: &[f32], w: &[Vec<f32>], h: usize) -> Vec<f32> {
    let mut charge = vec![0.0f32; h];
    let mut act = vec![0.0f32; h];
    for t in 0..TICKS {
        if t == 0 {
            for i in 0..input.len().min(h) { act[i] = input[i]; }
        }
        let mut raw = vec![0.0f32; h];
        for i in 0..h { for j in 0..h { raw[i] += act[j] * w[j][i]; } }
        for i in 0..h { charge[i] += raw[i] * CHARGE_RATE; charge[i] *= LEAK; }
        for i in 0..h { act[i] = (charge[i] - THRESHOLD).max(0.0); }
        let total: f32 = act.iter().sum();
        if total > 0.0 { let d = 1.0 + 0.05 * total; for i in 0..h { act[i] /= d; } }
    }
    charge
}

fn main() {
    println!("=== BIDIRECTIONAL POOL: every neuron is I/O ===");
    println!("RUNNING: bidirectional_sum\n");

    // Setup A: 9 neurons. ALL are both input AND output.
    // Thermometer: digit A injects into neurons 0-3, digit B into 4-7
    // Readout: ALL 9 neurons' charge → nearest class
    // The "SUM" is implicit: uniform weights mean every neuron accumulates total activity
    let h = 9;

    println!("--- A: 9 neurons, uniform weight=1.0, all↔all ---\n");
    {
        let mut w = vec![vec![0.0f32; h]; h];
        // ALL neurons connect to ALL others with weight 1.0
        for i in 0..h { for j in 0..h { if i != j { w[i][j] = 1.0; } } }

        // Each input goes into a neuron, readout from ALL
        let mut correct = 0; let mut total = 0;
        let mut test_ok = 0; let mut test_n = 0;
        for a in 0..DIGITS { for b in 0..DIGITS {
            let target = a + b;
            let mut input = vec![0.0f32; h];
            // Thermometer inject: first a neurons + next b neurons get input
            for i in 0..a.min(4) { input[i] = 1.0; }
            for i in 0..b.min(4) { input[4 + i] = 1.0; }

            let charge = forward(&input, &w, h);
            // Readout: use TOTAL charge across all neurons as the "sum signal"
            let total_charge: f32 = charge.iter().sum();
            // Map total_charge to class via nearest
            let charge_per_count = {
                // Calibrate: what's the charge for 1 active input?
                let mut inp1 = vec![0.0f32; h]; inp1[0] = 1.0;
                let c1: f32 = forward(&inp1, &w, h).iter().sum();
                c1
            };
            let pred = if charge_per_count.abs() < 0.0001 { 0 }
                else { ((total_charge / charge_per_count).round() as usize).min(8) };
            let ok = pred == target;
            if ok { correct += 1; }
            if target == 4 { test_n += 1; if ok { test_ok += 1; } }
            total += 1;
            println!("  {}+{}={} total_charge={:.3} pred={} {}", a, b, target, total_charge, pred, if ok {"✓"} else {"✗"});
        }}
        println!("\n  All: {}/{} = {:.0}%  Test(sum=4): {}/{} = {:.0}%\n",
            correct, total, 100.0*correct as f64/total as f64,
            test_ok, test_n, 100.0*test_ok as f64/test_n as f64);
    }

    // Setup B: same but check if individual neuron charges differ per input pattern
    println!("--- B: Do individual neurons differentiate patterns? ---\n");
    {
        let mut w = vec![vec![0.0f32; h]; h];
        for i in 0..h { for j in 0..h { if i != j { w[i][j] = 1.0; } } }

        println!("{:>5} {:>5} | neuron charges [0..8]", "A", "B");
        for (a, b) in [(0,4), (1,3), (2,2), (3,1), (4,0)] {
            let mut input = vec![0.0f32; h];
            for i in 0..a.min(4) { input[i] = 1.0; }
            for i in 0..b.min(4) { input[4+i] = 1.0; }
            let charge = forward(&input, &w, h);
            let cs: Vec<String> = charge.iter().map(|c| format!("{:.3}", c)).collect();
            println!("  {}+{} | [{}]", a, b, cs.join(", "));
        }
    }

    // Setup C: NON-uniform weights — does it still work?
    println!("\n--- C: Exhaustive over 9-neuron pool (ternary self-connections only) ---\n");
    {
        // With 9 neurons fully connected, there are 9*8=72 edges → 3^72 = impossible
        // BUT: if we fix uniform inter-neuron weight=1.0 and only vary SELF-connection weight:
        // 9 self-weights × ternary = 3^9 = 19683 configs
        let ternary = [-1i8, 0, 1];
        let mut best_test = 0.0f64;
        let mut best_self_w = vec![0i8; h];
        let mut generalizes = 0u32;

        for config in 0..19683u32 {
            let mut self_w = vec![0i8; h];
            let mut c = config;
            for i in 0..h { self_w[i] = ternary[(c % 3) as usize]; c /= 3; }

            let mut w = vec![vec![0.0f32; h]; h];
            for i in 0..h { for j in 0..h {
                if i == j { w[i][j] = self_w[i] as f32 * 0.5; }
                else { w[i][j] = 1.0; }
            }}

            // Eval all 25 examples
            let charge_per_count = {
                let mut inp1 = vec![0.0f32; h]; inp1[0] = 1.0;
                forward(&inp1, &w, h).iter().sum::<f32>()
            };
            if charge_per_count.abs() < 0.0001 { continue; }

            let mut correct = 0; let mut test_ok = 0;
            for a in 0..DIGITS { for b in 0..DIGITS {
                let target = a + b;
                let mut input = vec![0.0f32; h];
                for i in 0..a.min(4) { input[i] = 1.0; }
                for i in 0..b.min(4) { input[4+i] = 1.0; }
                let charge = forward(&input, &w, h);
                let total_charge: f32 = charge.iter().sum();
                let pred = ((total_charge / charge_per_count).round() as usize).min(8);
                if pred == target { correct += 1; if target == 4 { test_ok += 1; } }
            }}
            let test_acc = test_ok as f64 / 5.0;
            if correct == 25 && test_ok == 5 { generalizes += 1; }
            if test_acc > best_test { best_test = test_acc; best_self_w = self_w.clone(); }
        }

        println!("  19683 configs tested");
        println!("  Generalizing (100% all): {}", generalizes);
        println!("  Best test: {:.0}%", best_test * 100.0);
        println!("  Best self-weights: {:?}", best_self_w);
    }
}
