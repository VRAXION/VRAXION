//! BUILD→CRYSTALLIZE ratchet: v3 specialized workers + batch edge pruning.
//!
//! Cycle: BUILD (7 jackpot workers) → CRYSTALLIZE (batch edge + W prune) → repeat.
//!
//! Crystallize uses batch importance scoring: each edge evaluated independently
//! against the same baseline, removing only edges whose absence doesn't hurt.
//! This avoids the sequential cascade problem from earlier attempts.
//!
//! Run: cargo run --example ratchet_evolution --release -- <corpus-path>

use instnct_core::{
    build_network, DirectedEdge, InitConfig, Int8Projection, Network, PropagationConfig, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::fs;
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const N_WORKERS: usize = 7;

// ---------------------------------------------------------------------------
// Helpers (same as evolve_language / clustered_evolution)
// ---------------------------------------------------------------------------

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

fn bar(val: f64, max_val: f64, width: usize) -> String {
    let filled = ((val / max_val) * width as f64).round().min(width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
}

fn apply_mutation(
    worker_id: usize,
    net: &mut Network,
    proj: &mut Int8Projection,
    rng: &mut impl Rng,
) -> bool {
    match worker_id {
        0 => net.mutate_add_edge(rng),
        1 => net.mutate_remove_edge(rng),
        2 => net.mutate_rewire(rng),
        3 => net.mutate_reverse(rng),
        4 => net.mutate_mirror(rng),
        5 => {
            let sub = rng.gen_range(0..3u32);
            match sub {
                0 => net.mutate_theta(rng),
                1 => net.mutate_channel(rng),
                _ => net.mutate_polarity(rng),
            }
        }
        _ => { let _ = proj.mutate_one(rng); true }
    }
}

// ---------------------------------------------------------------------------
// BUILD phase: v3 specialized workers with jackpot
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_phase(
    jackpot_net: &mut Network,
    jackpot_proj: &mut Int8Projection,
    corpus: &[u8],
    init: &InitConfig,
    sdr: &SdrTable,
    rounds: usize,
    seed: u64,
    round_offset: usize,
) {
    let output_start = init.output_start();
    let neuron_count = init.neuron_count;
    let edge_cap = init.edge_cap();

    for round in 0..rounds {
        let abs_round = round_offset + round;
        let eval_seed = seed.wrapping_add(5000).wrapping_add(abs_round as u64);

        let proposals: Vec<(Network, Int8Projection, f64)> = (0..N_WORKERS)
            .into_par_iter()
            .map(|wid| {
                let mut net = jackpot_net.clone();
                let mut proj = jackpot_proj.clone();
                let mut mrng = StdRng::seed_from_u64(
                    seed.wrapping_add(9000).wrapping_add(abs_round as u64 * 100 + wid as u64),
                );

                let mut erng = StdRng::seed_from_u64(eval_seed);
                let before = eval_accuracy(
                    &mut net, &proj, corpus, 100, &mut erng, sdr,
                    &init.propagation, output_start, neuron_count,
                );

                apply_mutation(wid, &mut net, &mut proj, &mut mrng);

                let mut erng2 = StdRng::seed_from_u64(eval_seed);
                let after = eval_accuracy(
                    &mut net, &proj, corpus, 100, &mut erng2, sdr,
                    &init.propagation, output_start, neuron_count,
                );

                (net, proj, after - before)
            })
            .collect();

        let best = proposals.iter().enumerate()
            .filter(|(_, (net, _, delta))| {
                if net.edge_count() < edge_cap { *delta >= 0.0 } else { *delta > 0.0 }
            })
            .max_by(|a, b| a.1 .2.partial_cmp(&b.1 .2).unwrap());

        if let Some((_, (net, proj, _))) = best {
            *jackpot_net = net.clone();
            *jackpot_proj = proj.clone();
        }

        if (round + 1) % 2500 == 0 {
            let mut erng = StdRng::seed_from_u64(seed.wrapping_add(6000).wrapping_add(abs_round as u64));
            let full = eval_accuracy(
                jackpot_net, jackpot_proj, corpus, 2000, &mut erng,
                sdr, &init.propagation, output_start, neuron_count,
            );
            println!(
                "    build step {:>6}: |{}| {:.1}%  edges={}",
                round + 1, bar(full, 0.30, 30), full * 100.0, jackpot_net.edge_count(),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// CRYSTALLIZE: iterative edge pruning (fixed from one-shot delta>=0 bug)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn crystallize_edges(
    net: &mut Network,
    proj: &Int8Projection,
    corpus: &[u8],
    eval_len: usize,
    sdr: &SdrTable,
    config: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    seed: u64,
) -> (usize, f64, f64) {
    let initial_edges = net.edge_count();
    let mut br = StdRng::seed_from_u64(seed);
    let initial_acc = eval_accuracy(
        &mut net.clone(), proj, corpus, eval_len, &mut br,
        sdr, config, output_start, neuron_count,
    );
    let mut current_acc = initial_acc;
    let mut total_removed = 0usize;

    for round in 0..5u64 {
        let edges: Vec<DirectedEdge> = net.graph().iter_edges().collect();
        if edges.len() < 100 { break; }

        let round_seed = seed + round * 111;
        let deltas: Vec<f64> = edges.par_iter().map(|edge| {
            let mut tn = net.clone();
            tn.graph_mut().remove_edge(edge.source, edge.target);
            let mut er = StdRng::seed_from_u64(round_seed);
            let a = eval_accuracy(&mut tn, proj, corpus, eval_len, &mut er,
                sdr, config, output_start, neuron_count);
            a - current_acc
        }).collect();

        let mut ranked: Vec<(usize, f64)> = deltas.iter().enumerate().map(|(i, &d)| (i, d)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let remove_count = (edges.len() * 30 / 100).max(1);
        let snapshot = net.save_state();
        let mut round_removed = 0usize;
        for &(idx, delta) in ranked.iter().take(remove_count) {
            if delta >= -0.005 {
                let e = &edges[idx];
                net.graph_mut().remove_edge(e.source, e.target);
                round_removed += 1;
            }
        }

        let mut vr = StdRng::seed_from_u64(round_seed);
        let after = eval_accuracy(&mut net.clone(), proj, corpus, eval_len, &mut vr,
            sdr, config, output_start, neuron_count);

        if after < initial_acc - 0.02 {
            net.restore_state(&snapshot);
            println!("    crystal round {}: STOP (would drop to {:.1}%)", round, after * 100.0);
            break;
        }

        total_removed += round_removed;
        current_acc = after;
        println!("    crystal round {}: -{} edges ({} remain), acc: {:.1}%",
            round, round_removed, net.edge_count(), after * 100.0);
        if round_removed == 0 { break; }
    }

    let mut fr = StdRng::seed_from_u64(seed);
    let final_acc = eval_accuracy(&mut net.clone(), proj, corpus, eval_len, &mut fr,
        sdr, config, output_start, neuron_count);
    println!("    edge crystallize total: {}/{} removed, {:.1}% -> {:.1}%",
        total_removed, initial_edges, initial_acc * 100.0, final_acc * 100.0);
    (total_removed, initial_acc, final_acc)
}

// ---------------------------------------------------------------------------
// CRYSTALLIZE: W matrix sparsification
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn crystallize_weights(
    net: &mut Network,
    proj: &mut Int8Projection,
    corpus: &[u8],
    eval_len: usize,
    threshold: u8,
    sdr: &SdrTable,
    config: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    seed: u64,
) -> (usize, f64, f64) {
    let mut baseline_rng = StdRng::seed_from_u64(seed);
    let baseline_acc = eval_accuracy(
        &mut net.clone(), proj, corpus, eval_len, &mut baseline_rng,
        sdr, config, output_start, neuron_count,
    );

    let proj_backup = proj.clone();
    let zeroed = proj.sparsify(threshold);

    let mut verify_rng = StdRng::seed_from_u64(seed);
    let after_acc = eval_accuracy(
        &mut net.clone(), proj, corpus, eval_len, &mut verify_rng,
        sdr, config, output_start, neuron_count,
    );

    if after_acc < baseline_acc - 0.005 {
        *proj = proj_backup;
        println!("    W crystallize: rollback (threshold={threshold} would lose {:.1}pp)", (baseline_acc - after_acc) * 100.0);
        return (0, baseline_acc, baseline_acc);
    }

    println!("    W crystallize: {zeroed} zeroed, nz={}/{}, acc: {:.1}% -> {:.1}%",
        proj.nonzero_count(), proj.weight_count(), baseline_acc * 100.0, after_acc * 100.0);

    (zeroed, baseline_acc, after_acc)
}

// ---------------------------------------------------------------------------
// Plain v3 control (no crystallize)
// ---------------------------------------------------------------------------

fn run_plain_v3(
    corpus: &[u8],
    init: &InitConfig,
    sdr: &SdrTable,
    rounds: usize,
    seed: u64,
) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));

    build_phase(&mut net, &mut proj, corpus, init, sdr, rounds, seed, 0);

    let mut erng = StdRng::seed_from_u64(seed + 9999);
    eval_accuracy(
        &mut net, &proj, corpus, 5000, &mut erng,
        sdr, &init.propagation, init.output_start(), init.neuron_count,
    )
}

// ---------------------------------------------------------------------------
// Main: ratchet sweep + comparison
// ---------------------------------------------------------------------------

struct RatchetConfig {
    build_rounds: usize,
    n_cycles: usize,
    crystallize_eval_len: usize,
    w_threshold: u8,
    seed: u64,
}

fn main() {
    let h: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let flat = std::env::args().nth(3).map(|s| s == "flat").unwrap_or(false);
    let mut init = InitConfig::phi(h);
    if flat {
        // Fixed I/O (same as H=256): input 0..158, output (H-158)..H, no overlap
        init.phi_dim = 158;
    }

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path);
    println!("  {} chars\n", corpus.len());

    println!("H={}, phi_dim={}, input_end={}, output_start={}, edge_cap={}, chains={}",
        h, init.phi_dim, init.input_end(), init.output_start(), init.edge_cap(), init.chain_count);

    let sdr = SdrTable::new(CHARS, init.neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(42 + 100)).unwrap();

    let configs = [
        RatchetConfig { build_rounds: 10_000, n_cycles: 4, crystallize_eval_len: 2000, w_threshold: 1, seed: 42 },
    ];

    let output_start = init.output_start();
    let neuron_count = init.neuron_count;

    for (ci, cfg) in configs.iter().enumerate() {
        let total_build = cfg.build_rounds * cfg.n_cycles;
        println!("=== Ratchet config {}: {} cycles × {} build rounds = {} total ===\n",
            ci, cfg.n_cycles, cfg.build_rounds, total_build);

        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let mut net = build_network(&init, &mut rng);
        let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(cfg.seed + 200));

        let t0 = Instant::now();

        for cycle in 0..cfg.n_cycles {
            println!("  --- cycle {} ---", cycle);

            // BUILD
            let round_offset = cycle * cfg.build_rounds;
            build_phase(&mut net, &mut proj, &corpus, &init, &sdr,
                cfg.build_rounds, cfg.seed, round_offset);

            let mut erng = StdRng::seed_from_u64(cfg.seed + 6000 + cycle as u64 * 100);
            let build_acc = eval_accuracy(
                &mut net, &proj, &corpus, cfg.crystallize_eval_len, &mut erng,
                &sdr, &init.propagation, output_start, neuron_count,
            );
            println!("  BUILD done: {:.1}%  edges={}  W_nz={}/{}",
                build_acc * 100.0, net.edge_count(), proj.nonzero_count(), proj.weight_count());

            // CRYSTALLIZE EDGES
            crystallize_edges(
                &mut net, &proj, &corpus, cfg.crystallize_eval_len,
                &sdr, &init.propagation, output_start, neuron_count,
                cfg.seed + 7000 + cycle as u64 * 100,
            );

            // CRYSTALLIZE W
            crystallize_weights(
                &mut net, &mut proj, &corpus, cfg.crystallize_eval_len,
                cfg.w_threshold, &sdr, &init.propagation, output_start, neuron_count,
                cfg.seed + 8000 + cycle as u64 * 100,
            );

            println!();
        }

        let mut final_rng = StdRng::seed_from_u64(cfg.seed + 9999);
        let final_acc = eval_accuracy(
            &mut net, &proj, &corpus, 5000, &mut final_rng,
            &sdr, &init.propagation, output_start, neuron_count,
        );
        let elapsed = t0.elapsed();
        println!("  RATCHET FINAL: {:.1}%  edges={}  W_nz={}/{}  ({:.1}s)\n",
            final_acc * 100.0, net.edge_count(), proj.nonzero_count(), proj.weight_count(),
            elapsed.as_secs_f64());

        // --- Control: plain v3 same total rounds ---
        println!("  --- Control: plain v3, {} rounds ---", total_build);
        let t1 = Instant::now();
        let plain_acc = run_plain_v3(&corpus, &init, &sdr, total_build, cfg.seed);
        let plain_time = t1.elapsed();
        println!("\n  PLAIN V3 FINAL: {:.1}%  ({:.1}s)", plain_acc * 100.0, plain_time.as_secs_f64());
        println!("  Delta (ratchet - plain): {:+.1}pp\n", (final_acc - plain_acc) * 100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_mutation_all_types_no_panic() {
        let init = InitConfig::phi(64);
        let mut rng = StdRng::seed_from_u64(42);
        let mut net = build_network(&init, &mut rng);
        let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut rng);
        for wid in 0..N_WORKERS {
            let _ = apply_mutation(wid, &mut net, &mut proj, &mut rng);
        }
    }
}
