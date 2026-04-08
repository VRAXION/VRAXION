//! Diagnose: what does the 60%/72% network actually predict? Which examples fail?
//!
//! Run: cargo run --example addition_diagnose --release

use instnct_core::{
    build_network, evolution_step, EvolutionConfig, InitConfig, Int8Projection,
    Network, SdrTable,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

const DIGITS: usize = 5;
const SUMS: usize = 9;
const H: usize = 256;
const SDR_ACTIVE_PCT: usize = 20;
const TRAIN_STEPS: usize = 50_000;

fn make_examples() -> Vec<(usize, usize, usize)> {
    let mut ex = Vec::new();
    for a in 0..DIGITS { for b in 0..DIGITS { ex.push((a, b, a + b)); } }
    ex
}

fn main() {
    let examples = make_examples();

    println!("=== ADDITION DIAGNOSE ===\n");

    for &seed in &[42u64, 1042, 2042] {
        let mut rng = StdRng::seed_from_u64(seed);
        let init = InitConfig::empty(H);
        let mut net = build_network(&init, &mut rng);
        let sdr_a = SdrTable::new(DIGITS, H, init.input_end() / 2, SDR_ACTIVE_PCT,
            &mut StdRng::seed_from_u64(seed + 100)).unwrap();
        let sdr_b = SdrTable::new(DIGITS, H, init.input_end(), SDR_ACTIVE_PCT,
            &mut StdRng::seed_from_u64(seed + 200)).unwrap();
        let mut proj = Int8Projection::new(init.phi_dim, SUMS, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
        let evo_config = EvolutionConfig { edge_cap: 300, accept_ties: false };
        let prop_cfg = init.propagation.clone();
        let output_start = init.output_start();
        let neuron_count = init.neuron_count;

        // Train
        for _ in 0..TRAIN_STEPS {
            evolution_step(
                &mut net, &mut proj, &mut rng, &mut eval_rng,
                |net, proj, _| {
                    let mut correct = 0;
                    for &(a, b, target) in &examples {
                        net.reset();
                        let pa = sdr_a.pattern(a); let pb = sdr_b.pattern(b);
                        let mut combined = vec![0i32; neuron_count];
                        for i in 0..neuron_count { combined[i] = pa[i] + pb[i]; }
                        for _ in 0..6 { let _ = net.propagate(&combined, &prop_cfg); }
                        if proj.predict(&net.charge_vec(output_start..neuron_count)) == target { correct += 1; }
                    }
                    correct as f64 / examples.len() as f64
                },
                &evo_config,
            );
        }

        // Diagnose every example
        println!("--- seed {} ({} edges) ---", seed, net.edge_count());
        println!("{:>3} {:>3} {:>6} {:>6} {:>8} {:>30}",
            "a", "b", "target", "pred", "correct", "scores (top 3)");
        println!("{:-<3} {:-<3} {:-<6} {:-<6} {:-<8} {:-<30}",
            "", "", "", "", "", "");

        let mut correct_count = 0;
        let mut pred_histogram = vec![0usize; SUMS];
        let mut confusion: Vec<Vec<usize>> = vec![vec![0; SUMS]; SUMS]; // [target][pred]

        for &(a, b, target) in &examples {
            net.reset();
            let pa = sdr_a.pattern(a); let pb = sdr_b.pattern(b);
            let mut combined = vec![0i32; neuron_count];
            for i in 0..neuron_count { combined[i] = pa[i] + pb[i]; }
            for _ in 0..6 { let _ = net.propagate(&combined, &prop_cfg); }

            let charge = net.charge_vec(output_start..neuron_count);
            // Get raw scores
            let phi_dim = init.phi_dim;
            let pred = proj.predict(&charge);

            let is_correct = pred == target;
            if is_correct { correct_count += 1; }
            pred_histogram[pred] += 1;
            confusion[target][pred] += 1;

            let mark = if is_correct { "OK" } else { "MISS" };
            println!("{:>3} {:>3} {:>6} {:>6} {:>8}",
                a, b, target, pred, mark);
        }

        let acc = correct_count as f64 / examples.len() as f64;
        println!("\nAccuracy: {}/{} = {:.0}%", correct_count, examples.len(), acc * 100.0);

        // Prediction distribution
        println!("\nPrediction histogram:");
        for s in 0..SUMS {
            if pred_histogram[s] > 0 {
                println!("  sum={}: predicted {} times", s, pred_histogram[s]);
            }
        }

        // Which sums are never correctly predicted?
        println!("\nPer-sum accuracy:");
        for target_sum in 0..SUMS {
            let examples_with_sum: Vec<_> = examples.iter().filter(|&&(a,b,t)| t == target_sum).collect();
            let correct_for_sum = confusion[target_sum][target_sum];
            let total_for_sum = examples_with_sum.len();
            if total_for_sum > 0 {
                println!("  sum={}: {}/{} ({:.0}%) — examples: {:?}",
                    target_sum, correct_for_sum, total_for_sum,
                    correct_for_sum as f64 / total_for_sum as f64 * 100.0,
                    examples_with_sum.iter().map(|&&(a,b,_)| format!("{}+{}", a, b)).collect::<Vec<_>>().join(", "));
            }
        }
        println!();
    }
}
