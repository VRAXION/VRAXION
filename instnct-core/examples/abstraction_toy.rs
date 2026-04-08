//! Toy abstraction test: can a minimal network learn to COUNT active inputs?
//!
//! Hand-build the "correct" circuit, then observe what happens.
//! Then try mutation search to find it.
//!
//! RUNNING: abstraction_toy
//!
//! Run: cargo run --example abstraction_toy --release

const TICKS: usize = 8;
const CHARGE_RATE: f32 = 0.3;
const LEAK: f32 = 0.85;
const THRESHOLD: f32 = 0.5;
const DIVNORM_ALPHA: f32 = 0.1;

fn forward(input: &[f32], w: &[Vec<f32>], h: usize, use_divnorm: bool) -> Vec<f32> {
    let mut charge = vec![0.0f32; h];
    let mut act = vec![0.0f32; h];

    for t in 0..TICKS {
        if t == 0 {
            for i in 0..input.len().min(h) { act[i] = input[i]; }
        }
        // Matrix multiply
        let mut raw = vec![0.0f32; h];
        for i in 0..h {
            for j in 0..h {
                raw[i] += act[j] * w[j][i];
            }
        }
        // Charge dynamics
        for i in 0..h {
            charge[i] += raw[i] * CHARGE_RATE;
            charge[i] *= LEAK;
        }
        // ReLU
        for i in 0..h {
            act[i] = (charge[i] - THRESHOLD).max(0.0);
        }
        // Divnorm
        if use_divnorm {
            let total: f32 = act.iter().sum();
            if total > 0.0 {
                let denom = 1.0 + DIVNORM_ALPHA * total;
                for i in 0..h { act[i] /= denom; }
            }
        }
        charge.iter_mut().for_each(|c| *c = c.clamp(-1.0, 1.0));
    }
    charge
}

fn main() {
    println!("=== ABSTRACTION TOY ===");
    println!("RUNNING: abstraction_toy\n");

    // Minimal network: 8 input neurons + 1 SUM neuron + 9 output neurons = 18 total
    // Input [0..8]: thermometer A (4 bits) + thermometer B (4 bits)
    // SUM neuron [8]: connects FROM all 8 inputs with weight=1 → counts total
    // Output [9..18]: 9 neurons, one per possible sum (0-8)
    //   SUM→output[k] with weight that peaks at k

    let h = 18;
    let mut w = vec![vec![0.0f32; h]; h];

    // Input → SUM neuron: all inputs connect to neuron 8 with weight 1.0
    for i in 0..8 { w[i][8] = 1.0; }

    // SUM neuron → output neurons: weight encodes "preferred sum"
    // Output neuron k (at index 9+k) should fire most when SUM = k
    // Simple: SUM neuron connects to ALL output neurons, but each output
    // has a different threshold (encoded as negative self-weight)
    // Actually simpler: use different weights from SUM to each output
    for k in 0..9usize {
        // Weight from SUM(8) to output(9+k) = positive
        w[8][9 + k] = 1.0;
        // Self-inhibition on output = -(8-k) so higher k needs more activation
        // Actually let's just trace and see what happens
    }

    println!("--- Test 1: Hand-built SUM circuit (no learning) ---");
    println!("  Input[0..8] →(w=1.0)→ SUM[8] →(w=1.0)→ Output[9..18]");
    println!("  Each output neuron receives same signal from SUM\n");

    println!("{:>5} {:>5} | {:>8} {:>8} | output charges [0..8]",
        "A", "B", "SUM_chg", "SUM_act");
    println!("{:-<5} {:-<5}-+-{:-<8} {:-<8}-+----", "", "", "", "");

    for a in 0..5usize {
        for b in 0..5usize {
            let mut input = vec![0.0f32; h];
            for i in 0..a { input[i] = 1.0; }
            for i in 0..b { input[4 + i] = 1.0; }

            let charge = forward(&input, &w, h, true);

            let sum_charge = charge[8];
            let sum_act = (charge[8] - THRESHOLD).max(0.0);
            let out: Vec<String> = (0..9).map(|k| format!("{:.2}", charge[9 + k])).collect();

            if b == 0 || a + b <= 5 {
                println!("{:>5} {:>5} | {:>8.3} {:>8.3} | [{}]",
                    a, b, sum_charge, sum_act, out.join(", "));
            }
        }
    }

    // Test 2: Does SUM neuron charge scale with input count?
    println!("\n--- Test 2: SUM neuron charge vs total active inputs ---\n");
    println!("{:>8} {:>10} {:>10}", "n_active", "SUM_charge", "distinct?");
    println!("{:-<8} {:-<10} {:-<10}", "", "", "");

    let mut prev = -999.0f32;
    let mut all_distinct = true;
    for n in 0..=8 {
        let mut input = vec![0.0f32; h];
        for i in 0..n { input[i] = 1.0; }
        let charge = forward(&input, &w, h, true);
        let distinct = (charge[8] - prev).abs() > 0.001;
        if !distinct && n > 0 { all_distinct = false; }
        println!("{:>8} {:>10.4} {:>10}", n, charge[8], if distinct || n == 0 { "✓" } else { "✗ SAME" });
        prev = charge[8];
    }
    println!("\n  {} → SUM neuron {} distinguishes input counts\n",
        if all_distinct { "ALL DISTINCT" } else { "NOT ALL DISTINCT" },
        if all_distinct { "CAN" } else { "CANNOT" });

    // Test 3: Same sum, different digits — does SUM see the same?
    println!("--- Test 3: Same sum (=4), different digits — is SUM charge same? ---\n");
    let combos = vec![(0,4), (1,3), (2,2), (3,1), (4,0)];
    println!("{:>5} {:>5} {:>10}", "A", "B", "SUM_charge");
    for &(a, b) in &combos {
        let mut input = vec![0.0f32; h];
        for i in 0..a { input[i] = 1.0; }
        for i in 0..b { input[4 + i] = 1.0; }
        let charge = forward(&input, &w, h, true);
        println!("{:>5} {:>5} {:>10.4}", a, b, charge[8]);
    }
    let charges: Vec<f32> = combos.iter().map(|&(a,b)| {
        let mut input = vec![0.0f32; h]; for i in 0..a { input[i] = 1.0; } for i in 0..b { input[4+i] = 1.0; }
        forward(&input, &w, h, true)[8]
    }).collect();
    let same = charges.windows(2).all(|w| (w[0] - w[1]).abs() < 0.001);
    println!("\n  All same? {} → {}\n",
        if same { "YES" } else { "NO" },
        if same { "SUM neuron abstracts perfectly — digit identity lost, only count matters" }
        else { "SUM neuron DOESN'T abstract — digit identity leaks through" });

    // Test 4: Can mutation search FIND this circuit?
    println!("--- Test 4: Can mutation search find the SUM circuit? ---");
    println!("  Starting from random weights, try-keep-revert, 50K steps\n");

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let all_examples: Vec<(usize,usize,usize)> = (0..5).flat_map(|a| (0..5).map(move |b| (a, b, a+b))).collect();
    let train: Vec<_> = all_examples.iter().filter(|&&(_,_,s)| s != 4).cloned().collect();
    let test: Vec<_> = all_examples.iter().filter(|&&(_,_,s)| s == 4).cloned().collect();

    for &seed in &[42u64, 1042, 2042] {
        let mut rng = StdRng::seed_from_u64(seed);
        // Random init
        let mut w_mut: Vec<Vec<f32>> = (0..h).map(|_| (0..h).map(|_| {
            let r: f32 = rng.gen_range(-0.5..0.5);
            if rng.gen::<f32>() > 0.9 { r } else { 0.0 } // 10% density
        }).collect()).collect();

        let eval = |w: &[Vec<f32>], examples: &[(usize,usize,usize)]| -> f64 {
            let mut correct = 0;
            for &(a, b, target) in examples {
                let mut input = vec![0.0f32; h]; for i in 0..a { input[i] = 1.0; } for i in 0..b { input[4+i] = 1.0; }
                let charge = forward(&input, w, h, true);
                let mut scores = vec![0.0f32; 9];
                for k in 0..9 { scores[k] = charge[9 + k]; }
                let pred = scores.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0);
                if pred == target { correct += 1; }
            }
            correct as f64 / examples.len() as f64
        };

        for step in 0..50000 {
            let before = eval(&w_mut, &train);
            // Mutate one weight
            let i = rng.gen_range(0..h); let j = rng.gen_range(0..h);
            let old = w_mut[i][j];
            w_mut[i][j] = rng.gen_range(-2.0..2.0f32);

            let after = eval(&w_mut, &train);
            if after <= before { w_mut[i][j] = old; } // revert

            if step % 10000 == 9999 {
                let train_acc = eval(&w_mut, &train);
                let test_acc = eval(&w_mut, &test);

                // Check: does neuron 8 act as a SUM neuron?
                let mut sum_charges = Vec::new();
                for n in 0..=8 {
                    let mut input = vec![0.0f32; h]; for i in 0..n { input[i] = 1.0; }
                    sum_charges.push(forward(&input, &w_mut, h, true)[8]);
                }
                let monotonic = sum_charges.windows(2).all(|w| w[1] >= w[0] - 0.01);

                print!("  seed {} step {}: train={:.0}% test={:.0}%", seed, step+1, train_acc*100.0, test_acc*100.0);
                println!(" | n8_mono={}", if monotonic { "✓" } else { "✗" });
            }
        }
    }
}
