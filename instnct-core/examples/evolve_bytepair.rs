//! ABC↔Brain direct integration: symmetric I/O via C embeddings.
//!
//! The canonical pipeline test:
//!   Input:  bytes → A → B → pair_id → C embed → quantize → Brain input (32 neurons)
//!   Brain:  spiking propagation through sparse recurrent topology
//!   Output: Brain output charges → dequantize → C⁻¹ nearest → pair_id → (B⁻¹ → A⁻¹ → bytes)
//!
//! No SdrTable. No Int8Projection. The C embedding table IS the interface.
//!
//! Brain geometry: H=52, phi_dim=32 = C.E
//!   Input zone:  neurons 0..31   (32 neurons = C's 32 dimensions)
//!   Output zone: neurons 20..51  (32 neurons = C's 32 dimensions)
//!   Overlap:     neurons 20..31  (12 neurons, both I and O)
//!
//! Run:
//! ```
//! cargo run --release --example evolve_bytepair -- <corpus.txt> <packed.bin> [--steps N]
//! ```

use instnct_core::{
    build_bytepair_bigram, build_network, eval_bytepair_accuracy,
    eval_bytepair_smooth_bigram, evolution_step_jackpot, InitConfig, Int8Projection, Network,
    StepOutcome, VcbpTable,
};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::VecDeque;
use std::env;
use std::path::Path;
use std::time::Instant;

// ── Rooted Pathway: BFS-based IO signal guarantee ──────────────────
//
// Ported from Python ab_rooted_pathway.py (commit 4adf079).
// Builds loops anchored to IO-reachable neurons so signal is guaranteed
// to flow from input zone → hidden → output zone.

/// BFS forward from `start` nodes, return all reachable neurons within `max_hops`.
fn bfs_forward(net: &Network, starts: &[usize], max_hops: usize) -> Vec<bool> {
    let h = net.neuron_count();
    let mut reached = vec![false; h];
    let mut queue = VecDeque::new();
    for &s in starts {
        reached[s] = true;
        queue.push_back((s, 0usize));
    }
    // Build adjacency from edge list for BFS
    let mut adj: Vec<Vec<u16>> = vec![Vec::new(); h];
    for edge in net.graph().iter_edges() {
        adj[edge.source as usize].push(edge.target);
    }
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_hops {
            continue;
        }
        for &tgt in &adj[node] {
            if !reached[tgt as usize] {
                reached[tgt as usize] = true;
                queue.push_back((tgt as usize, depth + 1));
            }
        }
    }
    reached
}

/// BFS reverse: which neurons can REACH any of `ends` within `max_hops`.
fn bfs_reverse(net: &Network, ends: &[usize], max_hops: usize) -> Vec<bool> {
    let h = net.neuron_count();
    let mut reached = vec![false; h];
    let mut queue = VecDeque::new();
    for &e in ends {
        reached[e] = true;
        queue.push_back((e, 0usize));
    }
    // Reverse adjacency
    let mut rev_adj: Vec<Vec<u16>> = vec![Vec::new(); h];
    for edge in net.graph().iter_edges() {
        rev_adj[edge.target as usize].push(edge.source);
    }
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_hops {
            continue;
        }
        for &src in &rev_adj[node] {
            if !reached[src as usize] {
                reached[src as usize] = true;
                queue.push_back((src as usize, depth + 1));
            }
        }
    }
    reached
}

/// Build N rooted pathways: loops anchored to IO-reachable neurons.
/// Returns number of pathways successfully built.
fn seed_rooted_pathways(
    net: &mut Network,
    input_end: usize,
    output_start: usize,
    n_pathways: usize,
    rng: &mut impl Rng,
) -> usize {
    let h = net.neuron_count();
    let input_zone: Vec<usize> = (0..input_end).collect();
    let output_zone: Vec<usize> = (output_start..h).collect();

    let from_input = bfs_forward(net, &input_zone, 4);
    let to_output = bfs_reverse(net, &output_zone, 4);

    // Candidate anchors: reachable from input but not in output zone
    let input_anchors: Vec<usize> = (0..h)
        .filter(|&n| from_input[n] && n < output_start)
        .collect();
    // Candidate anchors: can reach output but not in input zone
    let output_anchors: Vec<usize> = (0..h)
        .filter(|&n| to_output[n] && n >= input_end)
        .collect();

    if input_anchors.is_empty() || output_anchors.is_empty() {
        // Fallback: direct edges from input boundary to output boundary
        for _ in 0..n_pathways.min(5) {
            let src = rng.gen_range(0..input_end) as u16;
            let tgt = rng.gen_range(output_start..h) as u16;
            net.graph_mut().add_edge(src, tgt);
        }
        return 0;
    }

    let mut built = 0;
    for _ in 0..n_pathways {
        let anchor_in = input_anchors[rng.gen_range(0..input_anchors.len())];
        let anchor_out = output_anchors[rng.gen_range(0..output_anchors.len())];

        // Pick 2-3 intermediate neurons
        let used: Vec<usize> = vec![anchor_in, anchor_out];
        let available: Vec<usize> = (0..h)
            .filter(|n| !used.contains(n))
            .collect();
        if available.len() < 2 {
            continue;
        }
        let n_mid = rng.gen_range(2..=3.min(available.len()));
        let mut mids = Vec::new();
        let mut pool = available.clone();
        for _ in 0..n_mid {
            let idx = rng.gen_range(0..pool.len());
            mids.push(pool.swap_remove(idx));
        }

        // Chain: anchor_in → mid1 → mid2 → ... → anchor_out
        let mut chain = vec![anchor_in];
        chain.extend(&mids);
        chain.push(anchor_out);

        let mut added = false;
        for w in chain.windows(2) {
            let (s, t) = (w[0] as u16, w[1] as u16);
            if net.graph_mut().add_edge(s, t) {
                added = true;
            }
        }
        // Close loop: anchor_out → anchor_in
        if net.graph_mut().add_edge(anchor_out as u16, anchor_in as u16) {
            added = true;
        }

        // Lower threshold on loop neurons for easier firing
        if added {
            for &n in &chain {
                let sd = &mut net.spike_data_mut()[n];
                if sd.threshold > 1 {
                    sd.threshold -= 1;
                }
            }
            built += 1;
        }
    }
    built
}

const DEFAULT_STEPS: usize = 10_000;
const DEFAULT_SEEDS: usize = 3;
const DEFAULT_EVAL_LEN: usize = 200;
const DEFAULT_FULL_LEN: usize = 1_000;
const PROGRESS_INTERVAL: usize = 1_000;
const MAX_CHARGE: u8 = 7; // quantization range [0, 7] — conservative for threshold compat

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: evolve_bytepair <corpus.txt> <packed.bin> [--steps N] [--seeds N] [--max-charge N]"
        );
        std::process::exit(1);
    }

    let corpus_path = &args[0];
    let packed_path = &args[1];
    let mut steps = DEFAULT_STEPS;
    let mut seed_count = DEFAULT_SEEDS;
    let mut max_charge = MAX_CHARGE;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                i += 1;
                steps = args[i].parse().expect("--steps: invalid number");
            }
            "--seeds" => {
                i += 1;
                seed_count = args[i].parse().expect("--seeds: invalid number");
            }
            "--max-charge" => {
                i += 1;
                max_charge = args[i].parse().expect("--max-charge: invalid number");
            }
            other => panic!("unknown flag: {other}"),
        }
        i += 1;
    }

    // ── Load C embedding table ──
    println!("Loading C embedding table from {packed_path}...");
    let table = VcbpTable::from_packed(Path::new(packed_path)).expect("failed to load packed.bin");
    println!("  {table}");
    assert_eq!(table.e, 32, "expected E=32, got E={}", table.e);

    // ── Load corpus as raw bytes ──
    println!("Loading corpus from {corpus_path}...");
    let corpus = std::fs::read(corpus_path).expect("failed to read corpus");
    let n_pairs = corpus.len() / 2;
    println!("  {} bytes = {} byte-pairs", corpus.len(), n_pairs);

    // ── Baselines ──
    let random_baseline = 1.0 / table.n_hot as f64;
    // Frequency baseline: most common byte-pair
    let mut freq = vec![0u32; 65536];
    for chunk in corpus.chunks_exact(2) {
        let id = ((chunk[0] as usize) << 8) | chunk[1] as usize;
        freq[id] += 1;
    }
    let most_common = freq.iter().enumerate().max_by_key(|(_, &c)| c).unwrap();
    let freq_baseline = most_common.1.to_owned() as f64 / n_pairs as f64;
    let mc_hi = (most_common.0 >> 8) as u8;
    let mc_lo = (most_common.0 & 0xFF) as u8;
    println!(
        "  Random: {:.3}%  Freq('{}{}'): {:.1}%",
        random_baseline * 100.0,
        mc_hi as char,
        mc_lo as char,
        freq_baseline * 100.0
    );

    // ── Brain geometry ──
    // H=128: phi_dim=79, plenty of room for signal propagation.
    // C's 32 dims map into the first 32 input neurons and last 32 output neurons.
    let h = 128usize;
    let mut init = InitConfig::phi(h);
    let e = table.e; // 32
    println!(
        "\n=== ABC↔Brain: H={}, phi_dim={}, input=0..{}, output={}..{} ===",
        h, init.phi_dim, init.input_end(), init.output_start(), h,
    );
    println!(
        "  C.E={} mapped to first {} input + last {} output neurons",
        e, e, e
    );
    println!(
        "  {} steps, {} seeds, max_charge={}, eval_len={}",
        steps, seed_count, max_charge, DEFAULT_EVAL_LEN
    );

    // Build byte-pair bigram table for smooth fitness
    println!("\nBuilding byte-pair bigram table...");
    let (hot_ids, bigram) = build_bytepair_bigram(&corpus, &table);
    let mut hot_to_idx = vec![usize::MAX; 65536];
    for (i, &hid) in hot_ids.iter().enumerate() {
        hot_to_idx[hid as usize] = i;
    }
    println!("  {} hot pairs in bigram table", hot_ids.len());

    let evo_config = init.evolution_config();

    // ── Run seeds ──
    let t_start = Instant::now();

    for seed_idx in 0..seed_count {
        let seed = 42 + seed_idx as u64 * 1000;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net = build_network(&init, &mut rng);

        // Seed rooted pathways: BFS-guaranteed IO signal flow
        let n_paths = seed_rooted_pathways(
            &mut net,
            init.input_end(),
            init.output_start(),
            30, // 30 pathways for H=128
            &mut rng,
        );
        println!(
            "  [seed={seed}] Rooted pathways: {n_paths} built, edges={}",
            net.edge_count()
        );

        // Dummy projection — evolution_step_jackpot requires one, but we don't use it
        let mut dummy_proj = Int8Projection::new(1, 1, &mut StdRng::seed_from_u64(seed + 200));
        let mut eval_rng = StdRng::seed_from_u64(seed + 1000);

        let mut accepted = 0u32;
        let mut rejected = 0u32;
        let mut peak_acc = 0.0f64;

        for step in 0..steps {
            let outcome = evolution_step_jackpot(
                &mut net,
                &mut dummy_proj,
                &mut rng,
                &mut eval_rng,
                |net_ref, _proj_ref, eval_rng_ref| {
                    eval_bytepair_smooth_bigram(
                        net_ref,
                        &table,
                        &corpus,
                        DEFAULT_EVAL_LEN,
                        eval_rng_ref,
                        &init.propagation,
                        init.output_start(),
                        init.neuron_count,
                        max_charge,
                        &hot_ids,
                        &bigram,
                        &hot_to_idx,
                    )
                },
                &evo_config,
                9, // 9 candidates (jackpot)
            );
            match outcome {
                StepOutcome::Accepted => accepted += 1,
                StepOutcome::Rejected => rejected += 1,
                StepOutcome::Skipped => {}
            }

            if (step + 1) % PROGRESS_INTERVAL == 0 {
                let acc = eval_bytepair_accuracy(
                    &mut net,
                    &table,
                    &corpus,
                    DEFAULT_FULL_LEN,
                    &mut eval_rng,
                    &init.propagation,
                    init.output_start(),
                    init.neuron_count,
                    max_charge,
                );
                peak_acc = peak_acc.max(acc);
                let tot = accepted + rejected;
                let rate = if tot > 0 {
                    accepted as f64 / tot as f64 * 100.0
                } else {
                    0.0
                };
                println!(
                    "  [seed={}] step {:>5}: acc={:.2}%  peak={:.2}%  accept={:.0}%  edges={}",
                    seed,
                    step + 1,
                    acc * 100.0,
                    peak_acc * 100.0,
                    rate,
                    net.edge_count()
                );
            }
        }

        // Final eval
        let final_acc = eval_bytepair_accuracy(
            &mut net,
            &table,
            &corpus,
            DEFAULT_FULL_LEN.min(n_pairs / 2),
            &mut eval_rng,
            &init.propagation,
            init.output_start(),
            init.neuron_count,
            max_charge,
        );
        peak_acc = peak_acc.max(final_acc);
        let rate = accepted as f64 / (accepted + rejected).max(1) as f64 * 100.0;
        println!(
            "  [seed={}] FINAL: {:.2}%  peak={:.2}%  edges={}  accept={:.0}%",
            seed,
            final_acc * 100.0,
            peak_acc * 100.0,
            net.edge_count(),
            rate
        );
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    println!("\nRuntime: {elapsed:.1}s");
}
