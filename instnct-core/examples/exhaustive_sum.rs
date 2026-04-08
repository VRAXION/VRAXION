//! Exhaustive search: 8 input → 1 hidden (SUM) → 9 output
//! Try ALL possible ternary weight configs for input→hidden.
//! Readout: charge-based nearest class.
//!
//! RUNNING: exhaustive_sum
//!
//! Run: cargo run --example exhaustive_sum --release

const TICKS: usize = 8;
const CHARGE_RATE: f32 = 0.3;
const LEAK: f32 = 0.85;
const THRESHOLD: f32 = 0.1;
const DIGITS: usize = 5;
const SUMS: usize = 9;

fn eval_config(weights: &[i8; 8]) -> (f64, f64, Vec<(usize,usize,usize,usize)>) {
    // Forward: 8 input → 1 hidden (SUM) → charge readout
    // hidden charge = sum(input[i] * weight[i]) accumulated over ticks

    let all: Vec<(usize,usize,usize)> = (0..DIGITS).flat_map(|a| (0..DIGITS).map(move |b| (a, b, a+b))).collect();
    let train: Vec<_> = all.iter().filter(|&&(_,_,s)| s != 4).cloned().collect();
    let test: Vec<_> = all.iter().filter(|&&(_,_,s)| s == 4).cloned().collect();

    // Run all 25 examples, collect hidden charge per example
    let mut charges: Vec<(usize, f32)> = Vec::new(); // (target_sum, hidden_charge)

    for &(a, b, target) in &all {
        let mut input = [0.0f32; 8];
        for i in 0..a { input[i] = 1.0; }
        for i in 0..b { input[4 + i] = 1.0; }

        let mut charge = 0.0f32;
        let mut act = 0.0f32;

        for t in 0..TICKS {
            let mut raw = 0.0f32;
            if t == 0 {
                for i in 0..8 { raw += input[i] * weights[i] as f32; }
            } else {
                for i in 0..8 { raw += input[i] * weights[i] as f32 * act; }
            }
            charge += raw * CHARGE_RATE;
            charge *= LEAK;
            act = (charge - THRESHOLD).max(0.0);
        }

        charges.push((target, charge));
    }

    // Build readout: for each sum level, what's the average charge?
    let mut sum_to_charges: Vec<Vec<f32>> = vec![Vec::new(); SUMS];
    for &(target, charge) in &charges {
        sum_to_charges[target].push(charge);
    }

    // Check: same sum → same charge? (abstraction test)
    let abstracts = sum_to_charges.iter().all(|charges| {
        if charges.len() <= 1 { return true; }
        let first = charges[0];
        charges.iter().all(|&c| (c - first).abs() < 0.001)
    });

    if !abstracts { return (0.0, 0.0, Vec::new()); }

    // Readout: nearest-neighbor based on mean charge per sum
    let centroids: Vec<f32> = sum_to_charges.iter().map(|cs| {
        if cs.is_empty() { 0.0 } else { cs.iter().sum::<f32>() / cs.len() as f32 }
    }).collect();

    // Check if centroids are monotonically increasing (or decreasing)
    let mono_inc = centroids.windows(2).all(|w| w[1] >= w[0] - 0.001);
    let mono_dec = centroids.windows(2).all(|w| w[1] <= w[0] + 0.001);
    if !mono_inc && !mono_dec { return (0.0, 0.0, Vec::new()); }

    // Predict each example
    let mut results = Vec::new();
    let mut train_correct = 0usize;
    let mut test_correct = 0usize;

    for &(target, charge) in &charges {
        let pred = centroids.iter().enumerate()
            .min_by(|a, b| (a.1 - charge).abs().partial_cmp(&(b.1 - charge).abs()).unwrap())
            .map(|(i, _)| i).unwrap_or(0);

        let is_train = target != 4;
        let ok = pred == target;
        if is_train && ok { train_correct += 1; }
        if !is_train && ok { test_correct += 1; }
        results.push((target, pred, charge as usize, if ok { 1 } else { 0 }));
    }

    let train_acc = train_correct as f64 / train.len() as f64;
    let test_acc = test_correct as f64 / test.len() as f64;
    (train_acc, test_acc, results)
}

fn main() {
    println!("=== EXHAUSTIVE SEARCH: 8 input → 1 hidden → readout ===");
    println!("RUNNING: exhaustive_sum");
    println!("Ternary weights (-1, 0, +1) on 8 input→hidden edges");
    println!("3^8 = 6561 configurations to try\n");

    let ternary = [-1i8, 0, 1];
    let mut total_configs = 0u64;
    let mut abstracts = 0u64;
    let mut perfect_train = 0u64;
    let mut generalizes = 0u64;
    let mut best_test = 0.0f64;
    let mut best_weights = [0i8; 8];

    // Exhaustive: try all 3^8 = 6561 configs
    for w0 in &ternary { for w1 in &ternary { for w2 in &ternary { for w3 in &ternary {
    for w4 in &ternary { for w5 in &ternary { for w6 in &ternary { for w7 in &ternary {
        let weights = [*w0, *w1, *w2, *w3, *w4, *w5, *w6, *w7];
        total_configs += 1;

        let (train_acc, test_acc, _) = eval_config(&weights);

        if train_acc > 0.0 { abstracts += 1; }
        if train_acc >= 1.0 { perfect_train += 1; }
        if test_acc >= 1.0 { generalizes += 1; }
        if test_acc > best_test {
            best_test = test_acc;
            best_weights = weights;
        }
    }}}}}}}};

    println!("Results:");
    println!("  Total configs:      {}", total_configs);
    println!("  Abstracts (>0%):    {}", abstracts);
    println!("  Perfect train:      {}", perfect_train);
    println!("  GENERALIZES (100%): {}", generalizes);
    println!("  Best test acc:      {:.0}%", best_test * 100.0);
    println!("  Best weights:       {:?}\n", best_weights);

    // Show the best config details
    if best_test > 0.0 {
        let (train, test, results) = eval_config(&best_weights);
        println!("Best config detail:");
        println!("  Weights: {:?}", best_weights);
        println!("  Train: {:.0}%  Test: {:.0}%", train * 100.0, test * 100.0);

        // Show charges per sum
        println!("\n  Sum → hidden charge:");
        for s in 0..SUMS {
            let ex: Vec<_> = results.iter().filter(|r| r.0 == s).collect();
            if !ex.is_empty() {
                println!("    sum={}: charge~{}, pred={} {}", s, ex[0].2, ex[0].1,
                    if ex.iter().all(|r| r.3 == 1) { "✓ all correct" } else { "✗ some wrong" });
            }
        }
    }

    // Also try wider range: -2..2 (5 values)
    println!("\n--- Wider range: weights in -2..2 (5^8 = 390,625 configs) ---\n");
    let range5 = [-2i8, -1, 0, 1, 2];
    let mut gen5 = 0u64;
    let mut best5_test = 0.0f64;
    let mut best5_w = [0i8; 8];

    for w0 in &range5 { for w1 in &range5 { for w2 in &range5 { for w3 in &range5 {
    for w4 in &range5 { for w5 in &range5 { for w6 in &range5 { for w7 in &range5 {
        let weights = [*w0, *w1, *w2, *w3, *w4, *w5, *w6, *w7];
        let (_, test_acc, _) = eval_config(&weights);
        if test_acc >= 1.0 { gen5 += 1; }
        if test_acc > best5_test { best5_test = test_acc; best5_w = weights; }
    }}}}}}}};

    println!("  GENERALIZES: {} / 390625", gen5);
    println!("  Best test: {:.0}%", best5_test * 100.0);
    println!("  Best weights: {:?}", best5_w);

    if best5_test >= 1.0 {
        let (train, test, _) = eval_config(&best5_w);
        println!("  Train: {:.0}%  Test: {:.0}% → *** GENERALIZATION FOUND ***", train*100.0, test*100.0);
    }
}

// Show ALL generalizing configs
