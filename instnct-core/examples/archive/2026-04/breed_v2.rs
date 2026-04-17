//! Breed v2: OR + Crystallize.
//!
//! Breed v1 (consensus AND) failed: only 4% Jaccard overlap, offspring worse than parents.
//! v2 takes the UNION of both parents' edges, then batch-crystallizes to remove dead weight.
//!
//! Pipeline:
//!   1. Train 6 parents (15K steps)
//!   2. Select top 2
//!   3. OR merge (union of all edges)
//!   4. Batch crystallize (score each edge, remove non-contributing)
//!   5. Train offspring (15K steps)
//!   6. Compare to control (best parent + 15K)
//!
//! Run: cargo run --example breed_v2 --release -- <corpus-path>

use instnct_core::{load_corpus, 
    build_network, evolution_step, DirectedEdge, InitConfig, Int8Projection, Network,
    PropagationConfig, SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;


#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network, proj: &Int8Projection, corpus: &[u8], len: usize,
    rng: &mut StdRng, sdr: &SdrTable, config: &PropagationConfig,
    output_start: usize, neuron_count: usize,
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

/// Train a single parent.
fn train_parent(
    seed: u64, steps: usize, corpus: &[u8], init: &InitConfig, sdr: &SdrTable,
) -> (Network, Int8Projection, f64) {
    let os = init.output_start();
    let nc = init.neuron_count;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let evo = init.evolution_config();

    for step in 0..steps {
        let _ = evolution_step(&mut net, &mut proj, &mut rng, &mut eval_rng,
            |n, p, e| eval_accuracy(n, p, corpus, 100, e, sdr, &init.propagation, os, nc),
            &evo);
        if (step + 1) % 5000 == 0 {
            let mut cr = StdRng::seed_from_u64(seed + 6000 + step as u64);
            let a = eval_accuracy(&mut net, &proj, corpus, 2000, &mut cr, sdr, &init.propagation, os, nc);
            println!("    parent seed={:<5} step {:>5}: |{}| {:.1}%  edges={}",
                seed, step + 1, bar(a, 0.30, 20), a * 100.0, net.edge_count());
        }
    }
    let mut fr = StdRng::seed_from_u64(seed + 9999);
    let fa = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr, sdr, &init.propagation, os, nc);
    println!("    parent seed={:<5} FINAL: {:.1}%  edges={}", seed, fa * 100.0, net.edge_count());
    (net, proj, fa)
}

/// Iterative crystallize: remove worst 30% per round until accuracy drops.
///
/// v1 bug: single-edge removal on 2000-char eval never changes accuracy for
/// 8878-edge networks, so delta≥0 removes ALL edges. Fix: iterative pruning
/// with rank-ordered removal and accuracy guard.
#[allow(clippy::too_many_arguments)]
fn iterative_crystallize(
    net: &mut Network, proj: &Int8Projection, corpus: &[u8], eval_len: usize,
    sdr: &SdrTable, config: &PropagationConfig, os: usize, nc: usize, seed: u64,
) -> (usize, f64, f64) {
    let initial_edges = net.edge_count();
    let mut br = StdRng::seed_from_u64(seed);
    let initial_acc = eval_accuracy(&mut net.clone(), proj, corpus, eval_len, &mut br, sdr, config, os, nc);
    let mut current_acc = initial_acc;
    let mut total_removed = 0usize;
    let max_rounds = 5;

    for round in 0..max_rounds {
        let edges: Vec<DirectedEdge> = net.graph().iter_edges().collect();
        if edges.len() < 100 { break; } // don't prune below 100 edges

        // Score each edge: delta = acc_without - acc_with
        let round_seed = seed + round as u64 * 111;
        let deltas: Vec<f64> = edges.par_iter().map(|edge| {
            let mut tn = net.clone();
            tn.graph_mut().remove_edge(edge.source, edge.target);
            let mut er = StdRng::seed_from_u64(round_seed);
            let a = eval_accuracy(&mut tn, proj, corpus, eval_len, &mut er, sdr, config, os, nc);
            a - current_acc
        }).collect();

        // Rank edges by delta (most negative = most important)
        let mut ranked: Vec<(usize, f64)> = deltas.iter().enumerate().map(|(i, &d)| (i, d)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // highest delta first (least important)

        // Remove top 30% (least important), but only those with delta >= -0.005
        let remove_count = (edges.len() * 30 / 100).max(1);
        let snapshot = net.save_state();
        let mut round_removed = 0usize;
        for &(idx, delta) in ranked.iter().take(remove_count) {
            if delta >= -0.005 { // only if removal doesn't hurt by more than 0.5pp
                let e = &edges[idx];
                net.graph_mut().remove_edge(e.source, e.target);
                round_removed += 1;
            }
        }

        // Verify accuracy didn't drop
        let mut vr = StdRng::seed_from_u64(round_seed);
        let after = eval_accuracy(&mut net.clone(), proj, corpus, eval_len, &mut vr, sdr, config, os, nc);

        if after < initial_acc - 0.02 {
            net.restore_state(&snapshot);
            println!("    crystal round {}: STOP (would drop to {:.1}%)", round, after * 100.0);
            break;
        }

        total_removed += round_removed;
        current_acc = after;
        println!("    crystal round {}: -{} edges ({} remain), acc: {:.1}%",
            round, round_removed, net.edge_count(), after * 100.0);

        if round_removed == 0 { break; } // nothing to remove
    }

    let mut fr = StdRng::seed_from_u64(seed);
    let final_acc = eval_accuracy(&mut net.clone(), proj, corpus, eval_len, &mut fr, sdr, config, os, nc);
    println!("    crystallize total: {}/{} removed, {:.1}% -> {:.1}%",
        total_removed, initial_edges, initial_acc * 100.0, final_acc * 100.0);
    (total_removed, initial_acc, final_acc)
}

/// Continue training.
#[allow(clippy::too_many_arguments)]
fn train_offspring(
    net: &mut Network, proj: &mut Int8Projection, seed: u64, steps: usize,
    corpus: &[u8], init: &InitConfig, sdr: &SdrTable, label: &str,
) -> f64 {
    let os = init.output_start();
    let nc = init.neuron_count;
    let evo = init.evolution_config();
    let mut rng = StdRng::seed_from_u64(seed + 3000);
    let mut eval_rng = StdRng::seed_from_u64(seed + 4000);
    let mut acc = 0u32;
    let mut tot = 0u32;

    for step in 0..steps {
        match evolution_step(net, proj, &mut rng, &mut eval_rng,
            |n, p, e| eval_accuracy(n, p, corpus, 100, e, sdr, &init.propagation, os, nc),
            &evo) {
            StepOutcome::Accepted => { acc += 1; tot += 1; }
            StepOutcome::Rejected => { tot += 1; }
            StepOutcome::Skipped => {}
        }
        if (step + 1) % 5000 == 0 {
            let mut cr = StdRng::seed_from_u64(seed + 7000 + step as u64);
            let a = eval_accuracy(net, proj, corpus, 2000, &mut cr, sdr, &init.propagation, os, nc);
            let r = if tot > 0 { acc as f64 / tot as f64 * 100.0 } else { 0.0 };
            println!("    {} step {:>5}: |{}| {:.1}%  edges={}  accept={:.0}%",
                label, step + 1, bar(a, 0.30, 20), a * 100.0, net.edge_count(), r);
        }
    }
    let mut fr = StdRng::seed_from_u64(seed + 9999);
    let fa = eval_accuracy(net, proj, corpus, 5000, &mut fr, sdr, &init.propagation, os, nc);
    let r = if tot > 0 { acc as f64 / tot as f64 * 100.0 } else { 0.0 };
    println!("    {} FINAL: {:.1}%  edges={}  accept={:.0}%", label, fa * 100.0, net.edge_count(), r);
    fa
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let init = InitConfig::phi(256);
    let os = init.output_start();
    let nc = init.neuron_count;
    let parent_seeds = [42u64, 123, 7, 1042, 2042, 4042];
    let parent_steps = 15_000;
    let offspring_steps = 15_000;

    let sdr = SdrTable::new(CHARS, nc, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(999)).unwrap();

    // === Phase 1: Train parents ===
    println!("=== Phase 1: Training {} parents ({} steps) ===\n", parent_seeds.len(), parent_steps);
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

    println!("\n=== Phase 2: OR merge + Crystallize ===");
    println!("  Parent A: seed={}, acc={:.1}%, edges={}", seed_a, acc_a * 100.0, net_a.edge_count());
    println!("  Parent B: seed={}, acc={:.1}%, edges={}", seed_b, acc_b * 100.0, net_b.edge_count());

    // Edge overlap analysis
    let mut shared = 0usize;
    let mut a_only = 0usize;
    for edge in net_a.graph().iter_edges() {
        if net_b.graph().has_edge(edge.source, edge.target) { shared += 1; }
        else { a_only += 1; }
    }
    let mut b_only = 0usize;
    for edge in net_b.graph().iter_edges() {
        if !net_a.graph().has_edge(edge.source, edge.target) { b_only += 1; }
    }
    let union_count = shared + a_only + b_only;
    println!("  Edge overlap: {} shared / {} union ({:.0}% Jaccard)",
        shared, union_count, shared as f64 / union_count as f64 * 100.0);

    // === Phase 3: OR merge — union of all edges ===
    let mut offspring = Network::new(nc);

    // Add ALL edges from parent A
    for edge in net_a.graph().iter_edges() {
        offspring.graph_mut().add_edge(edge.source, edge.target);
    }
    // Add edges from parent B that aren't already there
    for edge in net_b.graph().iter_edges() {
        offspring.graph_mut().add_edge(edge.source, edge.target); // add_edge ignores duplicates
    }

    // Copy params from best parent
    for i in 0..nc {
        offspring.spike_data_mut()[i].threshold = net_a.threshold_at(i);
        offspring.spike_data_mut()[i].channel = net_a.channel_at(i);
        offspring.polarity_mut()[i] = net_a.polarity()[i];
    }
    let mut offspring_proj = proj_a.clone();

    println!("\n  OR offspring: {} edges (union)", offspring.edge_count());

    // Eval pre-crystallize
    let mut pre_rng = StdRng::seed_from_u64(8888);
    let pre_acc = eval_accuracy(&mut offspring, &offspring_proj, &corpus, 5000, &mut pre_rng,
        &sdr, &init.propagation, os, nc);
    println!("  Pre-crystallize accuracy: {:.1}%\n", pre_acc * 100.0);

    // === Phase 4: Iterative crystallize ===
    println!("  --- Iterative crystallize (eval_len=3000, max 5 rounds) ---");
    let (removed, _base, _after) = iterative_crystallize(
        &mut offspring, &offspring_proj, &corpus, 3000,
        &sdr, &init.propagation, os, nc, 7777,
    );

    let mut post_rng = StdRng::seed_from_u64(8889);
    let post_acc = eval_accuracy(&mut offspring, &offspring_proj, &corpus, 5000, &mut post_rng,
        &sdr, &init.propagation, os, nc);
    println!("  Post-crystallize: {:.1}%  edges={} ({} removed)\n",
        post_acc * 100.0, offspring.edge_count(), removed);

    // === Phase 5: Train offspring ===
    println!("=== Phase 3: Training offspring ({} steps) ===\n", offspring_steps);
    let breed_acc = train_offspring(
        &mut offspring, &mut offspring_proj,
        seed_a.wrapping_add(seed_b.wrapping_mul(7)),
        offspring_steps, &corpus, &init, &sdr, "breed-v2",
    );

    // === Control: best parent continues ===
    println!("\n=== Control: Best parent +{} steps ===\n", offspring_steps);
    let mut ctrl_net = net_a.clone();
    let mut ctrl_proj = proj_a.clone();
    let ctrl_acc = train_offspring(
        &mut ctrl_net, &mut ctrl_proj, *seed_a,
        offspring_steps, &corpus, &init, &sdr, "control",
    );

    // === Summary ===
    println!("\n=== SUMMARY ===\n");
    println!("  Best parent:        {:.1}%  ({} edges)", acc_a * 100.0, net_a.edge_count());
    println!("  OR merge:           {:.1}%  ({} edges)", pre_acc * 100.0, union_count);
    println!("  Post-crystallize:   {:.1}%  ({} edges)", post_acc * 100.0, offspring.edge_count());
    println!("  Breed v2 final:     {:.1}%  ({} edges, +{} steps)", breed_acc * 100.0, offspring.edge_count(), offspring_steps);
    println!("  Control final:      {:.1}%  ({} edges, +{} steps)", ctrl_acc * 100.0, ctrl_net.edge_count(), offspring_steps);
    println!("  Delta (v2 - ctrl):  {:+.1}pp", (breed_acc - ctrl_acc) * 100.0);
    println!("\n  Breed v1 (AND) was: 16.0% — this is the bar to beat");
}
