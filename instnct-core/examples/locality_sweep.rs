//! Locality sweep: does distance-biased edge creation help cache AND accuracy?
//!
//! Tests mutate_add_edge with different locality biases (σ parameter).
//! Nearby neuron connections → cache-friendly scatter-add → faster propagation.
//! Also tests whether locality-biased evolution improves or hurts accuracy.
//!
//! Run: cargo run --example locality_sweep --release -- <corpus-path>

use instnct_core::{load_corpus, InitConfig, Int8Projection, Network, PropagationConfig, SdrTable};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;

// ---------------------------------------------------------------------------
// Locality-biased mutations
// ---------------------------------------------------------------------------

/// Add edge with target biased toward source. σ controls locality:
/// σ=∞ → uniform random (like mutate_add_edge), σ=1 → very local.
fn local_add_edge(net: &mut Network, sigma: f64, rng: &mut impl Rng) -> bool {
    let h = net.neuron_count();
    if h < 2 { return false; }
    let src = rng.gen_range(0..h);
    let tgt = if sigma >= h as f64 {
        rng.gen_range(0..h)
    } else {
        // Uniform offset in [-sigma, sigma], wrapped modular
        let offset = (rng.gen::<f64>() * 2.0 - 1.0) * sigma;
        ((src as f64 + offset).round() as isize).rem_euclid(h as isize) as usize
    };
    net.graph_mut().add_edge(src as u16, tgt as u16)
}

/// Build a network with locality-biased edges instead of random.
fn build_local_network(init: &InitConfig, sigma: f64, rng: &mut StdRng) -> Network {
    let h = init.neuron_count;
    let mut net = Network::new(h);

    // Chain highways (same as InitConfig)
    let overlap_start = init.output_start();
    let overlap_end = init.input_end();
    if init.chain_count > 0 && overlap_end > overlap_start + 1 {
        let overlap_mid = (overlap_start + overlap_end) / 2;
        for _ in 0..init.chain_count {
            let src = rng.gen_range(0..overlap_start) as u16;
            let hub1 = rng.gen_range(overlap_start..overlap_mid) as u16;
            let hub2 = rng.gen_range(overlap_mid..overlap_end) as u16;
            let tgt = rng.gen_range(overlap_end..h) as u16;
            net.graph_mut().add_edge(src, hub1);
            net.graph_mut().add_edge(hub1, hub2);
            net.graph_mut().add_edge(hub2, tgt);
        }
    }

    // Fill to target density with LOCALITY-BIASED edges
    let target_edges = h * h * init.density_pct / 100;
    for _ in 0..target_edges * 3 {
        local_add_edge(&mut net, sigma, rng);
        if net.edge_count() >= target_edges { break; }
    }

    // Random params
    for i in 0..h {
        net.threshold_mut()[i] = rng.gen_range(0..=init.threshold_max);
        net.channel_mut()[i] = rng.gen_range(1..=init.channel_max);
        if init.inhibitory_pct > 0 && rng.gen_ratio(init.inhibitory_pct, 100) {
            net.polarity_mut()[i] = -1;
        }
    }
    net
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------


fn sample_eval_offset(corpus_len: usize, len: usize, rng: &mut StdRng) -> Option<usize> {
    if corpus_len <= len { return None; }
    Some(rng.gen_range(0..=corpus_len - len - 1))
}

#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network,
    projection: &Int8Projection,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    config: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
) -> f64 {
    let Some(off) = sample_eval_offset(corpus.len(), len, rng) else { return 0.0; };
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        if projection.predict(&net.charge()[output_start..neuron_count]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

#[allow(dead_code)]
fn bar(val: f64, max_val: f64, width: usize) -> String {
    let filled = ((val / max_val) * width as f64).round().min(width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
}

/// Measure average "jump distance" between consecutive edge targets per source.
fn measure_locality(net: &Network) -> f64 {
    let h = net.neuron_count();
    let mut edges_by_src: Vec<Vec<u16>> = vec![Vec::new(); h];
    for e in net.graph().iter_edges() {
        edges_by_src[e.source as usize].push(e.target);
    }
    let mut total_jump = 0u64;
    let mut count = 0u64;
    for targets in &mut edges_by_src {
        targets.sort_unstable();
        for w in targets.windows(2) {
            total_jump += (w[1] as i32 - w[0] as i32).unsigned_abs() as u64;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { total_jump as f64 / count as f64 }
}

/// Measure propagation speed (ns per token).
fn measure_prop_speed(net: &mut Network, sdr: &SdrTable, config: &PropagationConfig) -> u64 {
    let iters = 500;
    net.reset();
    // Warmup
    for _ in 0..10 {
        net.propagate(sdr.pattern(0), config).unwrap();
    }
    net.reset();
    let t0 = Instant::now();
    for i in 0..iters {
        net.propagate(sdr.pattern(i % CHARS), config).unwrap();
    }
    t0.elapsed().as_nanos() as u64 / iters as u64
}

// ---------------------------------------------------------------------------
// Evolution with locality-biased mutations
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_evolution(
    corpus: &[u8],
    init: &InitConfig,
    sigma: f64,
    seed: u64,
    steps: usize,
) -> (f64, usize, f64, u64) {
    let output_start = init.output_start();
    let neuron_count = init.neuron_count;
    let edge_cap = init.edge_cap();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_local_network(init, sigma, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let sdr = SdrTable::new(CHARS, neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(seed + 100)).unwrap();

    for _ in 0..steps {
        // Paired eval: before
        let eval_snapshot = eval_rng.clone();
        let before = eval_accuracy(
            &mut net, &proj, corpus, 100, &mut eval_rng, &sdr,
            &init.propagation, output_start, neuron_count,
        );
        eval_rng = eval_snapshot;

        let snapshot = net.save_state();

        // Mutate: locality-biased add, standard everything else
        let roll = rng.gen_range(0..100u32);
        let mut weight_backup = None;
        let mutated = match roll {
            0..25 => local_add_edge(&mut net, sigma, &mut rng),
            25..40 => net.mutate_remove_edge(&mut rng),
            40..50 => net.mutate_rewire(&mut rng),
            50..65 => net.mutate_reverse(&mut rng),
            65..72 => net.mutate_mirror(&mut rng),
            72..80 => net.mutate_enhance(&mut rng),
            80..85 => net.mutate_theta(&mut rng),
            85..90 => net.mutate_channel(&mut rng),
            _ => { weight_backup = Some(proj.mutate_one(&mut rng)); true }
        };

        if !mutated {
            let _ = eval_accuracy(
                &mut net, &proj, corpus, 100, &mut eval_rng, &sdr,
                &init.propagation, output_start, neuron_count,
            );
            continue;
        }

        let after = eval_accuracy(
            &mut net, &proj, corpus, 100, &mut eval_rng, &sdr,
            &init.propagation, output_start, neuron_count,
        );

        let accepted = if net.edge_count() < edge_cap { after >= before } else { after > before };
        if !accepted {
            net.restore_state(&snapshot);
            if let Some(backup) = weight_backup {
                proj.rollback(backup);
            }
        }
    }

    let locality = measure_locality(&net);
    let prop_speed = measure_prop_speed(&mut net, &sdr, &init.propagation);
    let final_acc = eval_accuracy(
        &mut net, &proj, corpus, 5000, &mut eval_rng, &sdr,
        &init.propagation, output_start, neuron_count,
    );

    (final_acc, net.edge_count(), locality, prop_speed)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

struct SweepConfig {
    sigma: f64,
    label: &'static str,
}

fn main() {
    let init = InitConfig::phi(256);
    let steps = 15000;
    let seeds = [42u64, 123, 7];

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let h = init.neuron_count as f64;
    let sigmas = [
        SweepConfig { sigma: 99999.0, label: "random" },
        SweepConfig { sigma: h / 2.0, label: "σ=H/2" },
        SweepConfig { sigma: h / 4.0, label: "σ=H/4" },
        SweepConfig { sigma: h / 8.0, label: "σ=H/8" },
        SweepConfig { sigma: h / 16.0, label: "σ=H/16" },
    ];

    println!("=== Locality Sweep: H={}, {} steps, {} seeds ===\n",
        init.neuron_count, steps, seeds.len());

    struct Result {
        label: &'static str,
        sigma: f64,
        mean_acc: f64,
        mean_edges: usize,
        mean_locality: f64,
        mean_prop_ns: u64,
    }

    let mut results = Vec::new();

    for sc in &sigmas {
        let runs: Vec<(f64, usize, f64, u64)> = seeds.par_iter()
            .map(|&seed| run_evolution(&corpus, &init, sc.sigma, seed, steps))
            .collect();

        let mean_acc = runs.iter().map(|r| r.0).sum::<f64>() / runs.len() as f64;
        let mean_edges = runs.iter().map(|r| r.1).sum::<usize>() / runs.len();
        let mean_locality = runs.iter().map(|r| r.2).sum::<f64>() / runs.len() as f64;
        let mean_prop = runs.iter().map(|r| r.3).sum::<u64>() / runs.len() as u64;

        for (i, (acc, edges, loc, prop)) in runs.iter().enumerate() {
            println!("  {:<8} seed={:<5} -> {:.1}%  edges={}  jump={:.1}  prop={}ns",
                sc.label, seeds[i], acc * 100.0, edges, loc, prop);
        }

        results.push(Result {
            label: sc.label, sigma: sc.sigma, mean_acc, mean_edges,
            mean_locality, mean_prop_ns: mean_prop,
        });
    }

    println!("\n=== SUMMARY ===\n");
    println!("{:<10} {:>6} {:>8} {:>8} {:>8} {:>10}", "strategy", "σ", "mean%", "edges", "jump", "prop_ns");
    println!("{}", "-".repeat(55));
    for r in &results {
        println!("{:<10} {:>6.0} {:>7.1}% {:>8} {:>7.1} {:>9}ns",
            r.label, r.sigma, r.mean_acc * 100.0, r.mean_edges, r.mean_locality, r.mean_prop_ns);
    }
}
