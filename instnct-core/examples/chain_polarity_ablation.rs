//! Chain-50 + polarity mutation ablation.
//!
//! The mutation_profile (24.2% peak) has: no chain-50 init, polarity mutation (5%).
//! The library evolution_step (19.1% peak) has: chain-50 init, no polarity mutation.
//!
//! This tests all 4 combinations to isolate the effect:
//!   A: chain-50 + no polarity  (current library behavior)
//!   B: no chain + no polarity
//!   C: chain-50 + polarity
//!   D: no chain + polarity     (mutation_profile behavior)
//!
//! Run: cargo run --example chain_polarity_ablation --release -- <corpus-path>

use instnct_core::{load_corpus,
    eval_accuracy, InitConfig, Int8Projection, Network, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;

/// Build network with optional chain-50 init.
fn build_net(init: &InitConfig, use_chains: bool, rng: &mut impl Rng) -> Network {
    let h = init.neuron_count;
    let mut net = Network::new(h);

    // Phase 1: optional chain highways
    if use_chains {
        let os = init.output_start();
        let oe = init.input_end();
        if oe > os + 1 {
            let om = (os + oe) / 2;
            for _ in 0..50 {
                let src = rng.gen_range(0..os) as u16;
                let h1 = rng.gen_range(os..om) as u16;
                let h2 = rng.gen_range(om..oe) as u16;
                let tgt = rng.gen_range(oe..h) as u16;
                net.graph_mut().add_edge(src, h1);
                net.graph_mut().add_edge(h1, h2);
                net.graph_mut().add_edge(h2, tgt);
            }
        }
    }

    // Phase 2: fill to 5% density
    let target = h * h * init.density_pct / 100;
    for _ in 0..target * 3 {
        net.mutate_add_edge(rng);
        if net.edge_count() >= target { break; }
    }

    // Phase 3: random params
    for i in 0..h {
        net.threshold_mut()[i] = rng.gen_range(0..=init.threshold_max);
        net.channel_mut()[i] = rng.gen_range(1..=init.channel_max);
        if init.inhibitory_pct > 0 && rng.gen_ratio(init.inhibitory_pct, 100) {
            net.polarity_mut()[i] = -1;
        }
    }
    net
}

/// Mutation schedule. With polarity: theta 5%, channel 5%, polarity 5%, projection 5%.
/// Without polarity: theta 5%, channel 5%, projection 10% (library default).
fn do_mutation(
    net: &mut Network,
    proj: &mut Int8Projection,
    rng: &mut impl Rng,
    use_polarity: bool,
) -> (bool, Option<instnct_core::WeightBackup>) {
    let roll = rng.gen_range(0..100u32);

    if use_polarity {
        // mutation_profile schedule: 5% polarity, 5% projection
        match roll {
            0..25 => (net.mutate_add_edge(rng), None),
            25..40 => (net.mutate_remove_edge(rng), None),
            40..50 => (net.mutate_rewire(rng), None),
            50..65 => (net.mutate_reverse(rng), None),
            65..72 => (net.mutate_mirror(rng), None),
            72..80 => (net.mutate_enhance(rng), None),
            80..85 => (net.mutate_theta(rng), None),
            85..90 => (net.mutate_channel(rng), None),
            90..95 => (net.mutate_polarity(rng), None),
            _ => (true, Some(proj.mutate_one(rng))),
        }
    } else {
        // library schedule: 0% polarity, 10% projection
        match roll {
            0..25 => (net.mutate_add_edge(rng), None),
            25..40 => (net.mutate_remove_edge(rng), None),
            40..50 => (net.mutate_rewire(rng), None),
            50..65 => (net.mutate_reverse(rng), None),
            65..72 => (net.mutate_mirror(rng), None),
            72..80 => (net.mutate_enhance(rng), None),
            80..85 => (net.mutate_theta(rng), None),
            85..90 => (net.mutate_channel(rng), None),
            _ => (true, Some(proj.mutate_one(rng))),
        }
    }
}

#[allow(dead_code)]
struct RunResult {
    label: String,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    peak_step: usize,
    final_edges: usize,
}

fn run_one(
    use_chains: bool,
    use_polarity: bool,
    seed: u64,
    steps: usize,
    corpus: &[u8],
) -> RunResult {
    let init = InitConfig::phi(256);
    let output_start = init.output_start();
    let neuron_count = init.neuron_count;
    let edge_cap = init.edge_cap();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_net(&init, use_chains, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let sdr = SdrTable::new(CHARS, neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(seed + 100)).unwrap();

    let mut peak_acc = 0.0f64;
    let mut peak_step = 0usize;

    let label = format!("{}{}",
        if use_chains { "chain" } else { "nochain" },
        if use_polarity { "+pol" } else { "" });

    for step in 0..steps {
        // Paired eval: before
        let snap = eval_rng.clone();
        let before = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng,
            &sdr, &init.propagation, output_start, neuron_count);
        eval_rng = snap;

        let state = net.save_state();
        let (mutated, wb) = do_mutation(&mut net, &mut proj, &mut rng, use_polarity);

        if !mutated {
            let _ = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng,
                &sdr, &init.propagation, output_start, neuron_count);
            continue;
        }

        let after = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng,
            &sdr, &init.propagation, output_start, neuron_count);

        let accepted = if net.edge_count() < edge_cap { after >= before } else { after > before };
        if !accepted {
            net.restore_state(&state);
            if let Some(b) = wb { proj.rollback(b); }
        }

        if (step + 1) % 10_000 == 0 {
            let mut check_rng = StdRng::seed_from_u64(seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut check_rng,
                &sdr, &init.propagation, output_start, neuron_count);
            if acc > peak_acc {
                peak_acc = acc;
                peak_step = step + 1;
            }
            println!("  {:<12} seed={:<4} step {:>5}: {:.1}%  edges={}",
                label, seed, step + 1, acc * 100.0, net.edge_count());
        }
    }

    let mut final_rng = StdRng::seed_from_u64(seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut final_rng,
        &sdr, &init.propagation, output_start, neuron_count);

    println!("  {:<12} seed={:<4} FINAL: {:.1}%  peak={:.1}% @{}  edges={}",
        label, seed, final_acc * 100.0, peak_acc * 100.0, peak_step, net.edge_count());

    RunResult {
        label, seed, final_acc, peak_acc, peak_step,
        final_edges: net.edge_count(),
    }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let seeds = [42u64, 123, 7];
    let steps = 50_000;

    // All 4 combinations: (use_chains, use_polarity)
    let variants: Vec<(bool, bool, &str)> = vec![
        (true,  false, "chain"),          // A: current library
        (false, false, "nochain"),        // B: no chain, no polarity
        (true,  true,  "chain+pol"),      // C: chain + polarity
        (false, true,  "nochain+pol"),    // D: mutation_profile behavior
    ];

    let mut configs: Vec<(bool, bool, u64, String)> = Vec::new();
    for &(chains, pol, label) in &variants {
        for &seed in &seeds {
            configs.push((chains, pol, seed, label.to_string()));
        }
    }

    println!("=== Chain+Polarity Ablation: {} configs, {} steps ===\n", configs.len(), steps);

    let results: Vec<RunResult> = configs.par_iter()
        .map(|(chains, pol, seed, _)| run_one(*chains, *pol, *seed, steps, &corpus))
        .collect();

    println!("\n=== SUMMARY ===\n");
    println!("{:<14} {:>8} {:>8} {:>8} {:>8}",
        "variant", "mean%", "best%", "peak%", "edges");
    println!("{}", "-".repeat(52));

    for &(chains, pol, label) in &variants {
        let full_label = label.to_string();
        let group: Vec<_> = results.iter()
            .filter(|r| r.label == format!("{}{}",
                if chains { "chain" } else { "nochain" },
                if pol { "+pol" } else { "" }))
            .collect();
        let mean = group.iter().map(|r| r.final_acc).sum::<f64>() / group.len() as f64;
        let best = group.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = group.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = group.iter().map(|r| r.final_edges).sum::<usize>() / group.len();
        println!("{:<14} {:>7.1}% {:>7.1}% {:>7.1}% {:>8}",
            full_label, mean * 100.0, best * 100.0, peak * 100.0, edges);
    }

    println!("\nPer-seed detail:");
    println!("{:<14} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "variant", "seed", "final%", "peak%", "p_step", "edges");
    println!("{}", "-".repeat(56));
    for r in &results {
        println!("{:<14} {:>6} {:>7.1}% {:>7.1}% {:>8} {:>8}",
            r.label, r.seed, r.final_acc * 100.0, r.peak_acc * 100.0,
            r.peak_step, r.final_edges);
    }
}
