//! Diagnostic: train seq_5x5 addition, then dissect what the network learned.
//!
//! After training, prints:
//! 1. Per-problem prediction matrix (which sums correct/wrong)
//! 2. Charge patterns per input — do different A values create different traces?
//! 3. Neuron activation analysis — are there digit-specialized neurons?
//! 4. Edge topology stats for the trained vs random network
//!
//! Run: cargo run --example addition_diag --release

use instnct_core::{
    apply_mutation, build_network, cosine_to_onehot, InitConfig, Int8Projection, Network, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const INPUT_SYMBOLS: usize = 5; // digits 0-4
const OUTPUT_CLASSES: usize = 9; // sums 0-8
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 50_000;

fn all_problems() -> Vec<(usize, usize, usize)> {
    let mut p = Vec::new();
    for a in 0..5 { for b in 0..5 { p.push((a, b, a + b)); } }
    p
}

fn eval_fitness(
    net: &mut Network, proj: &Int8Projection, sdr: &SdrTable,
    init: &InitConfig, problems: &[(usize, usize, usize)], rng: &mut StdRng,
) -> f64 {
    let mut total = 0.0f64;
    for _ in 0..20 {
        let &(a, b, sum) = &problems[rng.gen_range(0..problems.len())];
        net.reset();
        net.propagate(sdr.pattern(a), &init.propagation).unwrap();
        net.propagate(sdr.pattern(b), &init.propagation).unwrap();
        let scores = proj.raw_scores(&net.charge()[init.output_start()..init.neuron_count]);
        total += cosine_to_onehot(&scores, sum);
    }
    total / 20.0
}

fn main() {
    let seed = 8042u64; // best seed from the sweep (64%)
    let problems = all_problems();
    let init = InitConfig::phi(256);
    let edge_cap = init.edge_cap();

    println!("=== Addition Diagnostic ===");
    println!("  Training seq_5x5 with seed={}, 50K steps, jackpot=9\n", seed);

    // --- Also build a RANDOM (untrained) network for comparison ---
    let mut rng_random = StdRng::seed_from_u64(seed);
    let net_random = build_network(&init, &mut rng_random);

    // --- Train ---
    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, OUTPUT_CLASSES,
        &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let sdr = SdrTable::new(INPUT_SYMBOLS, init.neuron_count, init.input_end(),
        SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 100)).unwrap();

    let start = Instant::now();
    let mut accepted = 0u32;

    for step in 0..STEPS {
        let snap = eval_rng.clone();
        let before = eval_fitness(&mut net, &proj, &sdr, &init, &problems, &mut eval_rng);
        eval_rng = snap;

        let parent_net = net.save_state();
        let parent_proj = proj.clone();
        let edges_before = net.edge_count();

        let mut best_delta = f64::NEG_INFINITY;
        let mut best_net = None;
        let mut best_proj = None;
        let mut any = false;

        for c in 0..9 {
            net.restore_state(&parent_net);
            proj = parent_proj.clone();
            let mut cr = StdRng::seed_from_u64(seed.wrapping_add(300).wrapping_add(step as u64 * 100 + c as u64));
            if !apply_mutation(&mut net, &mut proj, &mut cr) { continue; }
            any = true;
            let cs = eval_rng.clone();
            let after = eval_fitness(&mut net, &proj, &sdr, &init, &problems, &mut eval_rng);
            eval_rng = cs;
            let eg = net.edge_count() > edges_before;
            let cap = !eg || net.edge_count() <= edge_cap;
            let d = after - before;
            if d > best_delta && cap { best_delta = d; best_net = Some(net.save_state()); best_proj = Some(proj.clone()); }
        }
        net.restore_state(&parent_net); proj = parent_proj.clone();
        let _ = eval_fitness(&mut net, &proj, &sdr, &init, &problems, &mut eval_rng);
        if !any { net.restore_state(&parent_net); proj = parent_proj; continue; }
        if best_delta > 0.0 {
            if let (Some(ns), Some(ps)) = (best_net, best_proj) { net.restore_state(&ns); proj = ps; accepted += 1; }
        } else { net.restore_state(&parent_net); proj = parent_proj; }

        if (step + 1) % 10_000 == 0 {
            let mut correct = 0u32;
            for &(a, b, sum) in &problems {
                net.reset();
                net.propagate(sdr.pattern(a), &init.propagation).unwrap();
                net.propagate(sdr.pattern(b), &init.propagation).unwrap();
                if proj.predict(&net.charge()[init.output_start()..init.neuron_count]) == sum { correct += 1; }
            }
            println!("  step {:>5}: acc={}/25 ({:.0}%) edges={} accepted={}",
                step + 1, correct, correct as f64 / 25.0 * 100.0, net.edge_count(), accepted);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("\n  Training done in {:.0}s, {} accepted\n", elapsed, accepted);

    // ========================================
    // DIAGNOSTIC 1: Per-problem prediction grid
    // ========================================
    println!("=== 1. PREDICTION GRID (rows=A, cols=B) ===\n");
    println!("  Target grid (A+B):");
    print!("     ");
    for b in 0..5 { print!("  B={}", b); }
    println!();
    for a in 0..5 {
        print!("  A={}", a);
        for b in 0..5 { print!("   {:>2}", a + b); }
        println!();
    }

    println!("\n  Predicted grid:");
    print!("     ");
    for b in 0..5 { print!("  B={}", b); }
    println!();
    let mut grid_correct = 0u32;
    for a in 0..5 {
        print!("  A={}", a);
        for b in 0..5 {
            net.reset();
            net.propagate(sdr.pattern(a), &init.propagation).unwrap();
            net.propagate(sdr.pattern(b), &init.propagation).unwrap();
            let pred = proj.predict(&net.charge()[init.output_start()..init.neuron_count]);
            let ok = pred == a + b;
            if ok { grid_correct += 1; }
            let mark = if ok { "+" } else { "X" };
            print!("  {:>2}{}", pred, mark);
        }
        println!();
    }
    println!("\n  Accuracy: {}/25 ({:.0}%)", grid_correct, grid_correct as f64 / 25.0 * 100.0);

    // ========================================
    // DIAGNOSTIC 2: Charge fingerprints per (A,B)
    // ========================================
    println!("\n=== 2. CHARGE FINGERPRINTS ===\n");

    // Collect charge vectors for all 25 problems
    let os = init.output_start();
    let nc = init.neuron_count;
    let mut charges: Vec<Vec<u32>> = Vec::new();
    for a in 0..5 {
        for b in 0..5 {
            net.reset();
            net.propagate(sdr.pattern(a), &init.propagation).unwrap();
            net.propagate(sdr.pattern(b), &init.propagation).unwrap();
            charges.push(net.charge()[os..nc].to_vec());
        }
    }

    // How many neurons have non-zero charge per problem?
    println!("  Active neurons in output zone (of {}):", nc - os);
    for a in 0..5 {
        print!("  A={}: ", a);
        for b in 0..5 {
            let idx = a * 5 + b;
            let active = charges[idx].iter().filter(|&&c| c > 0).count();
            print!(" B{}={:<3}", b, active);
        }
        println!();
    }

    // Cosine similarity between charge vectors for same-sum pairs
    println!("\n  Cosine similarity between pairs with SAME SUM:");
    for target_sum in 0..=8 {
        let pairs: Vec<(usize, usize)> = (0..5)
            .flat_map(|a| (0..5).map(move |b| (a, b)))
            .filter(|&(a, b)| a + b == target_sum)
            .collect();
        if pairs.len() < 2 { continue; }
        let mut sims = Vec::new();
        for i in 0..pairs.len() {
            for j in (i + 1)..pairs.len() {
                let idx_i = pairs[i].0 * 5 + pairs[i].1;
                let idx_j = pairs[j].0 * 5 + pairs[j].1;
                let ci = &charges[idx_i];
                let cj = &charges[idx_j];
                let dot: f64 = ci.iter().zip(cj).map(|(&a, &b)| a as f64 * b as f64).sum();
                let ni: f64 = ci.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
                let nj: f64 = cj.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
                let cos = if ni * nj < 1e-12 { 0.0 } else { dot / (ni * nj) };
                sims.push(cos);
            }
        }
        let mean_sim: f64 = sims.iter().sum::<f64>() / sims.len() as f64;
        let pair_str: String = pairs.iter().map(|(a, b)| format!("{}+{}", a, b)).collect::<Vec<_>>().join(", ");
        println!("  sum={}: cos={:.3} (pairs: {})", target_sum, mean_sim, pair_str);
    }

    // Cosine between DIFFERENT sums
    println!("\n  Cosine between DIFFERENT sums (should be lower):");
    let mut diff_sims = Vec::new();
    for i in 0..25 {
        for j in (i + 1)..25 {
            let (a1, b1) = (i / 5, i % 5);
            let (a2, b2) = (j / 5, j % 5);
            if a1 + b1 == a2 + b2 { continue; }
            let ci = &charges[i]; let cj = &charges[j];
            let dot: f64 = ci.iter().zip(cj).map(|(&a, &b)| a as f64 * b as f64).sum();
            let ni: f64 = ci.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
            let nj: f64 = cj.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = if ni * nj < 1e-12 { 0.0 } else { dot / (ni * nj) };
            diff_sims.push(cos);
        }
    }
    let mean_diff: f64 = diff_sims.iter().sum::<f64>() / diff_sims.len().max(1) as f64;
    println!("  mean cross-sum cosine: {:.3}", mean_diff);

    // ========================================
    // DIAGNOSTIC 3: Do different A values leave different traces?
    // ========================================
    println!("\n=== 3. A-VALUE TRACE ANALYSIS ===\n");
    println!("  Does the charge after (A, B=0) differ by A?");

    // Charge after A,0 for each A
    let mut a_traces: Vec<Vec<u32>> = Vec::new();
    for a in 0..5 {
        net.reset();
        net.propagate(sdr.pattern(a), &init.propagation).unwrap();
        net.propagate(sdr.pattern(0), &init.propagation).unwrap();
        a_traces.push(net.charge()[os..nc].to_vec());
    }

    println!("  Pairwise cosine of charge(A, B=0):");
    for i in 0..5 {
        for j in (i + 1)..5 {
            let ci = &a_traces[i]; let cj = &a_traces[j];
            let dot: f64 = ci.iter().zip(cj).map(|(&a, &b)| a as f64 * b as f64).sum();
            let ni: f64 = ci.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
            let nj: f64 = cj.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = if ni * nj < 1e-12 { 0.0 } else { dot / (ni * nj) };
            println!("    A={} vs A={}: cos={:.3}", i, j, cos);
        }
    }

    // ========================================
    // DIAGNOSTIC 4: Topology comparison (trained vs random)
    // ========================================
    println!("\n=== 4. TOPOLOGY: TRAINED vs RANDOM ===\n");
    println!("  Trained:  {} edges", net.edge_count());
    println!("  Random:   {} edges", net_random.edge_count());

    // Count edges in different zones
    let input_end = init.input_end();
    let output_start = init.output_start();
    let count_zone_edges = |network: &Network| -> (usize, usize, usize, usize) {
        let mut input_to_input = 0usize;
        let mut input_to_output = 0usize;
        let mut output_to_output = 0usize;
        let mut other = 0usize;
        for edge in network.graph().iter_edges() {
            let s = edge.source as usize;
            let t = edge.target as usize;
            let s_input = s < input_end;
            let t_output = t >= output_start;
            if s_input && !t_output { input_to_input += 1; }
            else if s_input && t_output { input_to_output += 1; }
            else if !s_input && t_output { output_to_output += 1; }
            else { other += 1; }
        }
        (input_to_input, input_to_output, output_to_output, other)
    };

    let (ii_t, io_t, oo_t, ot_t) = count_zone_edges(&net);
    let (ii_r, io_r, oo_r, ot_r) = count_zone_edges(&net_random);

    println!("\n  Zone edge counts:");
    println!("  {:>20} {:>8} {:>8}", "Zone", "Trained", "Random");
    println!("  {:>20} {:>8} {:>8}", "input→non-output", ii_t, ii_r);
    println!("  {:>20} {:>8} {:>8}", "input→output", io_t, io_r);
    println!("  {:>20} {:>8} {:>8}", "non-input→output", oo_t, oo_r);
    println!("  {:>20} {:>8} {:>8}", "other", ot_t, ot_r);

    // ========================================
    // DIAGNOSTIC 5: W projection analysis
    // ========================================
    println!("\n=== 5. W PROJECTION ANALYSIS ===\n");

    // For each output class, which neurons have the strongest weights?
    println!("  Top-3 neurons per output class (by |weight|):");
    for cls in 0..OUTPUT_CLASSES {
        let mut neuron_weights: Vec<(usize, i8)> = Vec::new();
        for n in 0..init.phi_dim {
            // raw_scores computes sum(charge * weight), so we need the weight column for this class
            // Weights are row-major: weight[neuron][class]
            // We can get them indirectly: set charge to one-hot and read raw_scores
            let mut onehot = vec![0u32; init.phi_dim];
            onehot[n] = 1;
            let scores = proj.raw_scores(&onehot);
            neuron_weights.push((n, scores[cls] as i8));
        }
        neuron_weights.sort_by_key(|&(_, w)| -(w.unsigned_abs() as i32));
        let top3: String = neuron_weights.iter().take(3)
            .map(|(n, w)| format!("n{}={:+}", n + output_start, w))
            .collect::<Vec<_>>().join(", ");
        println!("  sum={}: {}", cls, top3);
    }

    println!("\n  Done.");
}
