//! Deep parameter tuning experiment: many parameter mutations between
//! each topology change. Tests the hypothesis that parameters need to be
//! well-tuned to the current topology before structural changes are useful.
//!
//! Schedule per cycle:
//!   - N parameter steps (theta/channel/polarity)
//!   - 1 topology step (add/remove/rewire/reverse)
//!
//! Run: cargo run --example deep_tune --release

use instnct_core::{Int8Projection, Network, PropagationConfig, SdrTable, StepOutcome};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs;

const CHARS: usize = 27;
const NEURON_COUNT: usize = 256;
const PHI_DIM: usize = 158;
const INPUT_END: usize = PHI_DIM;
const OUTPUT_START: usize = NEURON_COUNT - PHI_DIM;
const SDR_ACTIVE_PCT: usize = 20;
const EDGE_CAP: usize = NEURON_COUNT * NEURON_COUNT * 7 / 100;

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = fs::read(path).expect("cannot read corpus");
    raw.iter()
        .filter_map(|&b| {
            if b.is_ascii_lowercase() { Some(b - b'a') }
            else if b.is_ascii_uppercase() { Some(b.to_ascii_lowercase() - b'a') }
            else if b == b' ' || b == b'\n' || b == b'\t' { Some(26) }
            else { None }
        })
        .collect()
}

fn sample_eval_offset(corpus_len: usize, len: usize, rng: &mut StdRng) -> Option<usize> {
    if corpus_len <= len { return None; }
    let max_offset = corpus_len - len - 1;
    Some(rng.gen_range(0..=max_offset))
}

fn eval_accuracy(
    net: &mut Network,
    projection: &Int8Projection,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    config: &PropagationConfig,
) -> f64 {
    let Some(off) = sample_eval_offset(corpus.len(), len, rng) else { return 0.0; };
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        if projection.predict(&net.charge()[OUTPUT_START..NEURON_COUNT]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

/// One paired-eval step with a specific mutation.
#[allow(clippy::too_many_arguments)]
fn paired_step(
    net: &mut Network,
    projection: &mut Int8Projection,
    mutation_rng: &mut StdRng,
    eval_rng: &mut StdRng,
    corpus: &[u8],
    sdr: &SdrTable,
    prop_config: &PropagationConfig,
    mutation_fn: impl FnOnce(&mut Network, &mut Int8Projection, &mut StdRng) -> bool,
) -> StepOutcome {
    // Paired eval: before
    let eval_snapshot = eval_rng.clone();
    let before = eval_accuracy(net, projection, corpus, 100, eval_rng, sdr, prop_config);
    *eval_rng = eval_snapshot;

    let net_snapshot = net.save_state();
    let mutated = mutation_fn(net, projection, mutation_rng);

    if !mutated {
        let _ = eval_accuracy(net, projection, corpus, 100, eval_rng, sdr, prop_config);
        return StepOutcome::Skipped;
    }

    let after = eval_accuracy(net, projection, corpus, 100, eval_rng, sdr, prop_config);

    let accepted = if net.edge_count() < EDGE_CAP {
        after >= before
    } else {
        after > before
    };

    if accepted {
        StepOutcome::Accepted
    } else {
        net.restore_state(&net_snapshot);
        StepOutcome::Rejected
    }
}

fn build_network(rng: &mut StdRng) -> Network {
    let mut net = Network::new(NEURON_COUNT);
    let target_edges = NEURON_COUNT * NEURON_COUNT * 5 / 100;
    for _ in 0..target_edges * 3 {
        net.mutate_add_edge(rng);
        if net.edge_count() >= target_edges { break; }
    }
    for i in 0..NEURON_COUNT {
        net.threshold_mut()[i] = rng.gen_range(0..=7);
        net.channel_mut()[i] = rng.gen_range(1..=8);
        if rng.gen_ratio(1, 10) { net.polarity_mut()[i] = -1; }
    }
    net
}

fn bar(val: f64, max_val: f64, width: usize) -> String {
    let filled = ((val / max_val) * width as f64).round().min(width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
}

fn main() {
    let param_steps_per_cycle = 50;  // parameter mutations per topology change
    let total_cycles = 2000;         // number of topology changes
    let total_steps = total_cycles * (param_steps_per_cycle + 1);
    let seed = 42u64;

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path);
    println!("  {} chars\n", corpus.len());

    let prop_config = PropagationConfig {
        ticks_per_token: 6,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
        use_refractory: false,
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(&mut rng);
    let mut projection = Int8Projection::new(PHI_DIM, CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let sdr = SdrTable::new(CHARS, NEURON_COUNT, INPUT_END, SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(seed + 100)).unwrap();

    println!("=== Deep Tune: {param_steps_per_cycle} param steps per topology change ===");
    println!("seed={seed}, {total_cycles} cycles = {total_steps} total steps\n");

    let mut param_accepted = 0u32;
    let mut param_rejected = 0u32;
    let mut topo_accepted = 0u32;
    let mut topo_rejected = 0u32;
    let mut topo_skipped = 0u32;

    for cycle in 0..total_cycles {
        // Phase 1: deep parameter tuning
        for _ in 0..param_steps_per_cycle {
            let outcome = {
                let roll = rng.gen_range(0..100u32);
                match roll {
                    0..35 => paired_step(&mut net, &mut projection, &mut rng, &mut eval_rng,
                        &corpus, &sdr, &prop_config,
                        |net, _proj, rng| net.mutate_theta(rng)),
                    35..65 => paired_step(&mut net, &mut projection, &mut rng, &mut eval_rng,
                        &corpus, &sdr, &prop_config,
                        |net, _proj, rng| net.mutate_channel(rng)),
                    65..95 => paired_step(&mut net, &mut projection, &mut rng, &mut eval_rng,
                        &corpus, &sdr, &prop_config,
                        |net, _proj, rng| net.mutate_polarity(rng)),
                    _ => paired_step(&mut net, &mut projection, &mut rng, &mut eval_rng,
                        &corpus, &sdr, &prop_config,
                        |_net, proj, rng| { let _ = proj.mutate_one(rng); true }),
                }
            };
            match outcome {
                StepOutcome::Accepted => param_accepted += 1,
                StepOutcome::Rejected => param_rejected += 1,
                StepOutcome::Skipped => {}
            }
        }

        // Phase 2: one topology mutation
        let topo_outcome = {
            let roll = rng.gen_range(0..100u32);
            match roll {
                0..35 => paired_step(&mut net, &mut projection, &mut rng, &mut eval_rng,
                    &corpus, &sdr, &prop_config,
                    |net, _proj, rng| net.mutate_add_edge(rng)),
                35..55 => paired_step(&mut net, &mut projection, &mut rng, &mut eval_rng,
                    &corpus, &sdr, &prop_config,
                    |net, _proj, rng| net.mutate_remove_edge(rng)),
                55..75 => paired_step(&mut net, &mut projection, &mut rng, &mut eval_rng,
                    &corpus, &sdr, &prop_config,
                    |net, _proj, rng| net.mutate_rewire(rng)),
                75..90 => paired_step(&mut net, &mut projection, &mut rng, &mut eval_rng,
                    &corpus, &sdr, &prop_config,
                    |net, _proj, rng| net.mutate_reverse(rng)),
                _ => paired_step(&mut net, &mut projection, &mut rng, &mut eval_rng,
                    &corpus, &sdr, &prop_config,
                    |net, _proj, rng| net.mutate_mirror(rng)),
            }
        };
        match topo_outcome {
            StepOutcome::Accepted => topo_accepted += 1,
            StepOutcome::Rejected => topo_rejected += 1,
            StepOutcome::Skipped => topo_skipped += 1,
        }

        // Report every 200 cycles
        if (cycle + 1) % 200 == 0 {
            let full = eval_accuracy(&mut net, &projection, &corpus, 2000, &mut eval_rng, &sdr, &prop_config);
            let step = (cycle + 1) * (param_steps_per_cycle + 1);
            let param_total = param_accepted + param_rejected;
            let param_rate = if param_total > 0 { param_accepted as f64 / param_total as f64 * 100.0 } else { 0.0 };
            let topo_total = topo_accepted + topo_rejected;
            let topo_rate = if topo_total > 0 { topo_accepted as f64 / topo_total as f64 * 100.0 } else { 0.0 };
            println!(
                "  cycle {:>5} (step {:>6}): |{}| {:.1}%  edges={}  param_acc={:.0}%  topo_acc={:.0}%  topo_skip={}",
                cycle + 1, step, bar(full, 0.30, 30), full * 100.0,
                net.edge_count(), param_rate, topo_rate, topo_skipped
            );
        }
    }

    let final_acc = eval_accuracy(&mut net, &projection, &corpus, 5000, &mut eval_rng, &sdr, &prop_config);
    println!("\n=== FINAL ===");
    println!("  Accuracy: {:.1}%  (5K chars)", final_acc * 100.0);
    println!("  Edges: {}", net.edge_count());
    println!("  Param steps: {} accepted / {} rejected ({:.0}%)",
        param_accepted, param_rejected,
        param_accepted as f64 / (param_accepted + param_rejected).max(1) as f64 * 100.0);
    println!("  Topo steps:  {} accepted / {} rejected / {} skipped ({:.0}%)",
        topo_accepted, topo_rejected, topo_skipped,
        topo_accepted as f64 / (topo_accepted + topo_rejected).max(1) as f64 * 100.0);
    println!("  Total steps: {}", total_steps);
}
