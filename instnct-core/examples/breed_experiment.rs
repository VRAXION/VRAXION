//! Breed experiment: consensus edge merge from independently trained networks.
//!
//! The Python recipe's 24.4% peak came from breeding. The 1+1 ES ceiling is ~17-18%.
//! Breed creates an offspring with only edges present in BOTH parents (consensus AND),
//! then continues training from that cleaner starting point.
//!
//! Pipeline: Train N parents (15K steps) → Select top 2 → Consensus merge → Train offspring (15K)
//!
//! Run: cargo run --example breed_experiment --release -- <corpus-path>

use instnct_core::{load_corpus, 
    build_network, evolution_step, InitConfig, Int8Projection, Network, PropagationConfig,
    SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;


#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network,
    proj: &Int8Projection,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    config: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
) -> f64 {
    if corpus.len() <= len { return 0.0; }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        if proj.predict(&net.charge_vec(output_start..neuron_count)) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

fn bar(val: f64, max_val: f64, width: usize) -> String {
    let filled = ((val / max_val) * width as f64).round().min(width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
}

/// Train a single parent network.
fn train_parent(
    seed: u64,
    steps: usize,
    corpus: &[u8],
    init: &InitConfig,
    sdr: &SdrTable,
) -> (Network, Int8Projection, f64) {
    let output_start = init.output_start();
    let neuron_count = init.neuron_count;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let evo_config = init.evolution_config();

    for step in 0..steps {
        let _ = evolution_step(
            &mut net, &mut proj, &mut rng, &mut eval_rng,
            |net, proj, eval_rng| {
                eval_accuracy(net, proj, corpus, 100, eval_rng, sdr,
                    &init.propagation, output_start, neuron_count)
            },
            &evo_config,
        );

        if (step + 1) % 5000 == 0 {
            let mut check_rng = StdRng::seed_from_u64(seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut check_rng,
                sdr, &init.propagation, output_start, neuron_count);
            println!("    parent seed={:<5} step {:>5}: |{}| {:.1}%  edges={}",
                seed, step + 1, bar(acc, 0.30, 20), acc * 100.0, net.edge_count());
        }
    }

    // Final eval on 5K chars
    let mut final_rng = StdRng::seed_from_u64(seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut final_rng,
        sdr, &init.propagation, output_start, neuron_count);
    println!("    parent seed={:<5} FINAL: {:.1}%  edges={}", seed, final_acc * 100.0, net.edge_count());

    (net, proj, final_acc)
}

/// Breed offspring from two parent networks via consensus AND.
/// An edge exists in offspring iff it exists in BOTH parents.
fn breed_consensus(
    parent_a: &Network,
    parent_b: &Network,
    neuron_count: usize,
) -> Network {
    let mut offspring = Network::new(neuron_count);

    // Consensus AND: only keep edges present in both parents
    for edge in parent_a.graph().iter_edges() {
        if parent_b.graph().has_edge(edge.source, edge.target) {
            offspring.graph_mut().add_edge(edge.source, edge.target);
        }
    }

    // Copy parameters from parent_a (the better parent)
    for i in 0..neuron_count {
        offspring.spike_data_mut()[i].threshold = parent_a.threshold_at(i);
        offspring.spike_data_mut()[i].channel = parent_a.channel_at(i);
        offspring.polarity_mut()[i] = parent_a.polarity()[i];
    }

    offspring
}

/// Continue training a bred offspring.
#[allow(clippy::too_many_arguments)]
fn train_offspring(
    net: &mut Network,
    proj: &mut Int8Projection,
    seed: u64,
    steps: usize,
    corpus: &[u8],
    init: &InitConfig,
    sdr: &SdrTable,
    label: &str,
) -> f64 {
    let output_start = init.output_start();
    let neuron_count = init.neuron_count;
    let evo_config = init.evolution_config();

    let mut rng = StdRng::seed_from_u64(seed + 3000);
    let mut eval_rng = StdRng::seed_from_u64(seed + 4000);

    let mut accepted = 0u32;
    let mut total = 0u32;

    for step in 0..steps {
        let outcome = evolution_step(
            net, proj, &mut rng, &mut eval_rng,
            |net, proj, eval_rng| {
                eval_accuracy(net, proj, corpus, 100, eval_rng, sdr,
                    &init.propagation, output_start, neuron_count)
            },
            &evo_config,
        );
        match outcome {
            StepOutcome::Accepted => { accepted += 1; total += 1; }
            StepOutcome::Rejected => { total += 1; }
            StepOutcome::Skipped => {}
        }

        if (step + 1) % 5000 == 0 {
            let mut check_rng = StdRng::seed_from_u64(seed + 7000 + step as u64);
            let acc = eval_accuracy(net, proj, corpus, 2000, &mut check_rng,
                sdr, &init.propagation, output_start, neuron_count);
            let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };
            println!("    {} step {:>5}: |{}| {:.1}%  edges={}  accept={:.0}%",
                label, step + 1, bar(acc, 0.30, 20), acc * 100.0, net.edge_count(), rate);
        }
    }

    let mut final_rng = StdRng::seed_from_u64(seed + 9999);
    let final_acc = eval_accuracy(net, proj, corpus, 5000, &mut final_rng,
        sdr, &init.propagation, output_start, neuron_count);
    let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };
    println!("    {} FINAL: {:.1}%  edges={}  accept={:.0}%", label, final_acc * 100.0, net.edge_count(), rate);

    final_acc
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let init = InitConfig::phi(256);
    let parent_seeds = [42u64, 123, 7, 1042, 2042, 4042];
    let parent_steps = 15_000;
    let offspring_steps = 15_000;

    // Shared SDR for all networks (same encoding)
    let sdr = SdrTable::new(CHARS, init.neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(999)).unwrap();

    // === Phase 1: Train parents in parallel ===
    println!("=== Phase 1: Training {} parents ({} steps each) ===\n", parent_seeds.len(), parent_steps);

    let parents: Vec<(u64, Network, Int8Projection, f64)> = parent_seeds.par_iter()
        .map(|&seed| {
            let (net, proj, acc) = train_parent(seed, parent_steps, &corpus, &init, &sdr);
            (seed, net, proj, acc)
        })
        .collect();

    println!("\n--- Parent results ---");
    for (seed, net, _, acc) in &parents {
        println!("  seed={:<5} acc={:.1}%  edges={}", seed, acc * 100.0, net.edge_count());
    }

    // === Phase 2: Select top 2 ===
    let mut sorted: Vec<_> = parents.iter().collect();
    sorted.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    let (seed_a, net_a, proj_a, acc_a) = sorted[0];
    let (seed_b, net_b, _, acc_b) = sorted[1];

    println!("\n=== Phase 2: Breeding top 2 ===");
    println!("  Parent A: seed={}, acc={:.1}%, edges={}", seed_a, acc_a * 100.0, net_a.edge_count());
    println!("  Parent B: seed={}, acc={:.1}%, edges={}", seed_b, acc_b * 100.0, net_b.edge_count());

    // Edge overlap analysis
    let mut shared = 0usize;
    let mut a_only = 0usize;
    let mut b_only = 0usize;
    for edge in net_a.graph().iter_edges() {
        if net_b.graph().has_edge(edge.source, edge.target) {
            shared += 1;
        } else {
            a_only += 1;
        }
    }
    for edge in net_b.graph().iter_edges() {
        if !net_a.graph().has_edge(edge.source, edge.target) {
            b_only += 1;
        }
    }
    let total_union = shared + a_only + b_only;
    println!("  Edge overlap: {} shared / {} union ({:.0}% Jaccard)",
        shared, total_union, shared as f64 / total_union as f64 * 100.0);
    println!("  A-only: {}, B-only: {}", a_only, b_only);

    // === Phase 3: Create offspring via consensus AND ===
    let mut offspring_net = breed_consensus(net_a, net_b, init.neuron_count);
    let mut offspring_proj = proj_a.clone(); // W from best parent

    println!("\n  Offspring: {} consensus edges (from {} + {} parents)",
        offspring_net.edge_count(), net_a.edge_count(), net_b.edge_count());

    // Eval offspring before training
    let mut pre_rng = StdRng::seed_from_u64(8888);
    let pre_acc = eval_accuracy(&mut offspring_net, &offspring_proj, &corpus, 5000, &mut pre_rng,
        &sdr, &init.propagation, init.output_start(), init.neuron_count);
    println!("  Offspring pre-train: {:.1}%\n", pre_acc * 100.0);

    // === Phase 4: Train offspring ===
    println!("=== Phase 3: Training offspring ({} steps) ===\n", offspring_steps);
    let breed_acc = train_offspring(
        &mut offspring_net, &mut offspring_proj,
        seed_a.wrapping_add(seed_b.wrapping_mul(7)), // combined seed
        offspring_steps, &corpus, &init, &sdr, "breed",
    );

    // === Phase 5: Control — best parent continues training for same steps ===
    println!("\n=== Control: Best parent continues ({} more steps) ===\n", offspring_steps);
    let mut control_net = net_a.clone();
    let mut control_proj = proj_a.clone();
    let control_acc = train_offspring(
        &mut control_net, &mut control_proj,
        *seed_a, offspring_steps, &corpus, &init, &sdr, "control",
    );

    // === Summary ===
    println!("\n=== SUMMARY ===\n");
    println!("  Best parent:     {:.1}%  ({} edges, seed={})", acc_a * 100.0, net_a.edge_count(), seed_a);
    println!("  Offspring pre:   {:.1}%  ({} edges, consensus AND)", pre_acc * 100.0, offspring_net.edge_count() - (offspring_net.edge_count() - shared));
    println!("  Offspring final: {:.1}%  ({} edges, +{} steps)", breed_acc * 100.0, offspring_net.edge_count(), offspring_steps);
    println!("  Control final:   {:.1}%  ({} edges, +{} steps)", control_acc * 100.0, control_net.edge_count(), offspring_steps);
    println!("  Delta (breed - control): {:+.1}pp", (breed_acc - control_acc) * 100.0);
}
