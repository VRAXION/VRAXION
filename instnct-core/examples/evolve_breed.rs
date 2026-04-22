//! BREED — Competitive coevolution between multiple Brains.
//!
//! Theoretical basis: a single Brain playing "solitaire" against a static
//! fitness landscape gets stuck in single-attractor local optima (frequency
//! baseline trap). An OPPONENT (competing Brain) destabilizes local optima
//! by creating competitive pressure.
//!
//! Architecture:
//! 1. Start N=3 Brains (different seeds) simultaneously
//! 2. Each Brain evolves for CYCLE_LEN=3000 steps with smooth cosine bigram fitness
//! 3. After each cycle: EVALUATE all Brains, RANK by accuracy
//! 4. BREED: create child from top-2 parents via consensus topology merge
//! 5. CRYSTALLIZE the child: exhaustive edge pruning
//! 6. The child REPLACES the worst Brain
//! 7. Repeat for N_CYCLES cycles
//!
//! Run:
//! ```
//! cargo run --release --example evolve_breed -- <corpus.txt> <packed.bin>
//! ```

use instnct_core::{
    build_network, cosine_similarity, evolution_step_jackpot, save_checkpoint,
    softmax, CheckpointMeta, InitConfig, Int8Projection, Network,
    PropagationConfig, StepOutcome, VcbpTable,
};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::VecDeque;
use std::env;
use std::path::Path;
use std::time::Instant;

// ── Constants ───────────────────────────────────────────────────────
const N_BRAINS: usize = 3;
const CYCLE_LEN: usize = 3000;
const N_CYCLES: usize = 10;
const EVAL_LEN: usize = 100;
const FULL_EVAL_LEN: usize = 1_000;
const MAX_CHARGE: i32 = 7;
const PROGRESS_INTERVAL: usize = 1000;
const CRYSTALLIZE_SAMPLES: usize = 200;

// ── Dual-pipeline input (reused from evolve_bytepair_proj) ──────────

fn build_dual_input(
    table: &VcbpTable,
    cur_id: u16,
    prev_id: u16,
    neuron_count: usize,
) -> Vec<i32> {
    let mut input = vec![0i32; neuron_count];
    let e = table.e;
    let emb_cur = table.embed_id(cur_id);
    table.quantize_to_input(emb_cur, &mut input[..e], MAX_CHARGE);
    let emb_prev = table.embed_id(prev_id);
    table.quantize_to_input(emb_prev, &mut input[e..e * 2], MAX_CHARGE);
    input
}

fn build_freq_classes(pair_ids: &[u16]) -> Vec<u8> {
    let mut freq = vec![0u32; 65536];
    for &pid in pair_ids {
        freq[pid as usize] += 1;
    }
    let max_freq = *freq.iter().max().unwrap_or(&1) as f64;
    freq.iter()
        .map(|&f| {
            if f == 0 {
                0u8
            } else {
                ((f as f64 / max_freq).sqrt() * 7.0).round().min(7.0) as u8
            }
        })
        .collect()
}

// ── Rooted pathways ─────────────────────────────────────────────────

fn bfs_forward(net: &Network, starts: &[usize], max_hops: usize) -> Vec<bool> {
    let h = net.neuron_count();
    let mut reached = vec![false; h];
    let mut queue = VecDeque::new();
    for &s in starts {
        reached[s] = true;
        queue.push_back((s, 0usize));
    }
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

fn bfs_reverse(net: &Network, ends: &[usize], max_hops: usize) -> Vec<bool> {
    let h = net.neuron_count();
    let mut reached = vec![false; h];
    let mut queue = VecDeque::new();
    for &e in ends {
        reached[e] = true;
        queue.push_back((e, 0usize));
    }
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

fn seed_rooted_pathways(
    net: &mut Network,
    input_end: usize,
    output_start: usize,
    n_pathways: usize,
    rng: &mut impl Rng,
) -> usize {
    let h = net.neuron_count();
    let from_input = bfs_forward(net, &(0..input_end).collect::<Vec<_>>(), 4);
    let to_output = bfs_reverse(net, &(output_start..h).collect::<Vec<_>>(), 4);
    let input_anchors: Vec<usize> = (0..h)
        .filter(|&n| from_input[n] && n < output_start)
        .collect();
    let output_anchors: Vec<usize> = (0..h)
        .filter(|&n| to_output[n] && n >= input_end)
        .collect();
    if input_anchors.is_empty() || output_anchors.is_empty() {
        for _ in 0..n_pathways.min(5) {
            net.graph_mut()
                .add_edge(rng.gen_range(0..input_end) as u16, rng.gen_range(output_start..h) as u16);
        }
        return 0;
    }
    let mut built = 0;
    for _ in 0..n_pathways {
        let ai = input_anchors[rng.gen_range(0..input_anchors.len())];
        let ao = output_anchors[rng.gen_range(0..output_anchors.len())];
        let avail: Vec<usize> = (0..h).filter(|n| *n != ai && *n != ao).collect();
        if avail.len() < 2 {
            continue;
        }
        let nm = rng.gen_range(2..=3.min(avail.len()));
        let mut mids = Vec::new();
        let mut pool = avail;
        for _ in 0..nm {
            let idx = rng.gen_range(0..pool.len());
            mids.push(pool.swap_remove(idx));
        }
        let mut chain = vec![ai];
        chain.extend(&mids);
        chain.push(ao);
        let mut added = false;
        for w in chain.windows(2) {
            if net.graph_mut().add_edge(w[0] as u16, w[1] as u16) {
                added = true;
            }
        }
        if net.graph_mut().add_edge(ao as u16, ai as u16) {
            added = true;
        }
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

// ── Corpus helpers ──────────────────────────────────────────────────

fn build_corpus_pairs(
    corpus: &[u8],
    table: &VcbpTable,
    max_classes: usize,
) -> (Vec<u16>, Vec<usize>, Vec<u16>, usize) {
    let n_pairs = corpus.len() / 2;
    let mut pair_ids = Vec::with_capacity(n_pairs);
    let mut freq = vec![0u32; 65536];
    for i in 0..n_pairs {
        let pid = VcbpTable::pair_id(corpus[i * 2], corpus[i * 2 + 1]);
        pair_ids.push(pid);
        freq[pid as usize] += 1;
    }

    let mut hot_freq: Vec<(u16, u32)> = (0..65536u32)
        .filter(|&v| table.is_hot(v as u16) && freq[v as usize] > 0)
        .map(|v| (v as u16, freq[v as usize]))
        .collect();
    hot_freq.sort_by(|a, b| b.1.cmp(&a.1));
    let n_classes = hot_freq.len().min(max_classes);
    hot_freq.truncate(n_classes);

    let hot_ids: Vec<u16> = hot_freq.iter().map(|&(id, _)| id).collect();
    let mut top_to_idx = vec![usize::MAX; 65536];
    for (i, &id) in hot_ids.iter().enumerate() {
        top_to_idx[id as usize] = i;
    }

    (pair_ids, top_to_idx, hot_ids, n_classes)
}

fn build_pair_bigram(pair_ids: &[u16], hot_to_idx: &[usize], n_hot: usize) -> Vec<Vec<f64>> {
    let mut counts = vec![vec![0u32; n_hot]; n_hot];
    for i in 0..pair_ids.len().saturating_sub(1) {
        let ci = hot_to_idx[pair_ids[i] as usize];
        let ni = hot_to_idx[pair_ids[i + 1] as usize];
        if ci != usize::MAX && ni != usize::MAX {
            counts[ci][ni] += 1;
        }
    }
    counts
        .iter()
        .map(|row| {
            let total: f64 = row.iter().map(|&c| c as f64).sum();
            if total < 1.0 {
                vec![1.0 / n_hot as f64; n_hot]
            } else {
                row.iter().map(|&c| c as f64 / total).collect()
            }
        })
        .collect()
}

// ── Eval functions ──────────────────────────────────────────────────

/// Smooth cosine bigram fitness (THE CHAMPION).
fn eval_smooth_proj(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    bigram: &[Vec<f64>],
    len: usize,
    rng: &mut StdRng,
    propagation: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    _freq_classes: &[u8],
) -> f64 {
    let n = pair_ids.len();
    if n <= len + 1 {
        return 0.0;
    }
    let off = rng.gen_range(1..=n - len - 1);
    net.reset();
    let mut total_cos = 0.0f64;
    let mut counted = 0usize;
    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let cur_idx = hot_to_idx[cur_id as usize];
        let prev_id = if i > 0 { pair_ids[off + i - 1] } else { cur_id };

        let input = build_dual_input(table, cur_id, prev_id, neuron_count);
        net.propagate(&input, propagation).unwrap();

        if cur_idx == usize::MAX {
            continue;
        }

        let scores = proj.raw_scores(&net.charge_vec(output_start..neuron_count));
        let probs = softmax(&scores);
        let target = &bigram[cur_idx];
        total_cos += cosine_similarity(&probs, target);
        counted += 1;
    }
    if counted == 0 {
        0.0
    } else {
        total_cos / counted as f64
    }
}

/// Accuracy eval (next-token prediction).
fn eval_accuracy_proj(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    len: usize,
    rng: &mut StdRng,
    propagation: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    _freq_classes: &[u8],
) -> f64 {
    let n = pair_ids.len();
    if n <= len + 1 {
        return 0.0;
    }
    let off = rng.gen_range(1..=n - len - 1);
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let prev_id = if i > 0 { pair_ids[off + i - 1] } else { cur_id };
        let tgt_id = pair_ids[off + i + 1];
        let tgt_idx = hot_to_idx[tgt_id as usize];

        let input = build_dual_input(table, cur_id, prev_id, neuron_count);
        net.propagate(&input, propagation).unwrap();

        let predicted_idx = proj.predict(&net.charge_vec(output_start..neuron_count));
        if tgt_idx != usize::MAX && predicted_idx == tgt_idx {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

// ── BREED: Consensus topology merge ─────────────────────────────────
//
// child edges = edges present in BOTH parents (consensus)
// child edges += edges from better parent only (winner bonus)
// child neuron params (threshold, channel, polarity) = copy from better parent
// child projection = copy from better parent
// All edge weights = 1 (binary, proven best)

/// Build a child Brain by merging two parent topologies.
///
/// `better` = higher-ranked parent, `worse` = lower-ranked parent.
/// Returns a new (Network, Int8Projection) pair for the child.
fn consensus_merge(
    better: &Network,
    worse: &Network,
    better_proj: &Int8Projection,
    h: usize,
) -> (Network, Int8Projection) {
    // Start with an empty child network
    let mut child = Network::new(h);

    // Step 1: Consensus edges — edges present in BOTH parents
    for edge in better.graph().iter_edges() {
        if worse.graph().has_edge(edge.source, edge.target) {
            child.graph_mut().add_edge(edge.source, edge.target);
        }
    }
    let consensus_count = child.edge_count();

    // Step 2: Winner bonus — edges only in the better parent
    for edge in better.graph().iter_edges() {
        if !worse.graph().has_edge(edge.source, edge.target) {
            child.graph_mut().add_edge(edge.source, edge.target);
        }
    }
    let winner_bonus = child.edge_count() - consensus_count;

    println!(
        "    MERGE: consensus={}, winner_bonus={}, total={}",
        consensus_count, winner_bonus, child.edge_count()
    );

    // Step 3: Copy neuron params from better parent
    for i in 0..h {
        child.spike_data_mut()[i].threshold = better.spike_data()[i].threshold;
        child.spike_data_mut()[i].channel = better.spike_data()[i].channel;
        child.polarity_mut()[i] = better.polarity()[i];
    }

    // Step 4: Clone the better parent's projection
    let child_proj = better_proj.clone();

    (child, child_proj)
}

// ── CRYSTALLIZE: exhaustive edge pruning ────────────────────────────
//
// Test each sampled edge: remove it, measure fitness. If fitness
// doesn't drop, the edge is dead weight — keep it removed.
// Uses eval_rng snapshot so ALL tests use the SAME corpus segment.

fn crystallize(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    bigram: &[Vec<f64>],
    freq_classes: &[u8],
    eval_rng: &mut StdRng,
    propagation: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    sample_count: usize,
    mut_rng: &mut StdRng,
) -> usize {
    // Snapshot eval_rng so every fitness test uses the same corpus segment
    let eval_snapshot = eval_rng.clone();

    // Baseline fitness on a longer window for stability
    *eval_rng = eval_snapshot.clone();
    let baseline = eval_smooth_proj(
        net,
        proj,
        table,
        pair_ids,
        hot_to_idx,
        bigram,
        EVAL_LEN * 2,
        eval_rng,
        propagation,
        output_start,
        neuron_count,
        freq_classes,
    );

    let mut pruned = 0u32;
    let tests = sample_count.min(net.edge_count());

    for _ in 0..tests {
        if net.edge_count() < 50 {
            break;
        }
        let edge_idx = mut_rng.gen_range(0..net.edge_count());
        let edge = net.graph().iter_edges().nth(edge_idx);
        if let Some(edge) = edge {
            let src = edge.source;
            let tgt = edge.target;
            net.graph_mut().remove_edge(src, tgt);

            // Eval with snapshot rng (same segment)
            *eval_rng = eval_snapshot.clone();
            let new_fitness = eval_smooth_proj(
                net,
                proj,
                table,
                pair_ids,
                hot_to_idx,
                bigram,
                EVAL_LEN,
                eval_rng,
                propagation,
                output_start,
                neuron_count,
                freq_classes,
            );

            if new_fitness >= baseline - 0.0001 {
                pruned += 1; // edge was dead weight, leave it removed
            } else {
                net.graph_mut().add_edge(src, tgt); // edge matters, restore
            }
        }
    }

    // Restore eval_rng to post-baseline state for downstream use
    *eval_rng = eval_snapshot;
    let _ = eval_smooth_proj(
        net,
        proj,
        table,
        pair_ids,
        hot_to_idx,
        bigram,
        EVAL_LEN * 2,
        eval_rng,
        propagation,
        output_start,
        neuron_count,
        freq_classes,
    );

    pruned as usize
}

// ── Brain container ─────────────────────────────────────────────────

struct BrainState {
    net: Network,
    proj: Int8Projection,
    rng: StdRng,
    eval_rng: StdRng,
    seed: u64,
    accepted: u32,
    rejected: u32,
    peak: f64,
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: evolve_breed <corpus.txt> <packed.bin> [--cycles N] [--cycle-len N] [--brains N]"
        );
        std::process::exit(1);
    }

    let corpus_path = &args[0];
    let packed_path = &args[1];
    let mut n_cycles = N_CYCLES;
    let mut cycle_len = CYCLE_LEN;
    let mut n_brains = N_BRAINS;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--cycles" => {
                i += 1;
                n_cycles = args[i].parse().unwrap();
            }
            "--cycle-len" => {
                i += 1;
                cycle_len = args[i].parse().unwrap();
            }
            "--brains" => {
                i += 1;
                n_brains = args[i].parse().unwrap();
            }
            other => panic!("unknown flag: {other}"),
        }
        i += 1;
    }

    // ── Load corpus + C embedding ──
    println!("Loading C embedding table...");
    let table = VcbpTable::from_packed(Path::new(packed_path)).unwrap();
    println!("  {table}");

    println!("Loading corpus...");
    let corpus = std::fs::read(corpus_path).unwrap();
    let max_classes = 397;
    let (pair_ids, hot_to_idx, hot_ids, n_classes) =
        build_corpus_pairs(&corpus, &table, max_classes);
    println!(
        "  {} bytes = {} byte-pairs, {} output classes (top-{})",
        corpus.len(),
        pair_ids.len(),
        n_classes,
        max_classes
    );

    println!("Building byte-pair bigram...");
    let bigram = build_pair_bigram(&pair_ids, &hot_to_idx, n_classes);
    let freq_classes = build_freq_classes(&pair_ids);

    // Baselines
    let random_baseline = 1.0 / n_classes as f64;
    let mut freq = vec![0u32; n_classes];
    for &pid in &pair_ids {
        let idx = hot_to_idx[pid as usize];
        if idx != usize::MAX {
            freq[idx] += 1;
        }
    }
    let most = freq.iter().enumerate().max_by_key(|(_, &c)| c).unwrap();
    let freq_baseline = *most.1 as f64 / pair_ids.len() as f64;
    let most_pair = hot_ids[most.0];
    let (mh, ml) = VcbpTable::pair_bytes(most_pair);
    println!(
        "  Random: {:.3}%  Freq('{}{}'): {:.1}%",
        random_baseline * 100.0,
        if mh.is_ascii_graphic() || mh == b' ' {
            mh as char
        } else {
            '.'
        },
        if ml.is_ascii_graphic() || ml == b' ' {
            ml as char
        } else {
            '.'
        },
        freq_baseline * 100.0
    );

    // ── Brain config ──
    let h = 256usize;
    let init = InitConfig::phi(h);
    let evo_config = init.evolution_config();

    println!(
        "\n=== BREED: Competitive Coevolution with {} Brains ===",
        n_brains
    );
    println!(
        "  H={}, {} cycles x {} steps, Int8Projection({}, {})\n",
        h, n_cycles, cycle_len, init.phi_dim, n_classes
    );

    let t_start = Instant::now();
    let checkpoint_dir = Path::new("output/breed_checkpoints");
    std::fs::create_dir_all(checkpoint_dir).ok();

    // ── Initialize N Brains ──
    let mut brains: Vec<BrainState> = Vec::with_capacity(n_brains);
    for b in 0..n_brains {
        let seed = 42 + b as u64 * 1000;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net = build_network(&init, &mut rng);

        let n_paths = seed_rooted_pathways(
            &mut net,
            init.input_end(),
            init.output_start(),
            30,
            &mut rng,
        );
        println!(
            "  Brain[{}] seed={}: rooted pathways={}, edges={}",
            b,
            seed,
            n_paths,
            net.edge_count()
        );

        let proj = Int8Projection::new(init.phi_dim, n_classes, &mut StdRng::seed_from_u64(seed + 200));
        let eval_rng = StdRng::seed_from_u64(seed + 1000);

        brains.push(BrainState {
            net,
            proj,
            rng,
            eval_rng,
            seed,
            accepted: 0,
            rejected: 0,
            peak: 0.0,
        });
    }

    // ── Global peak tracking ──
    let mut global_peak = 0.0f64;
    let mut global_peak_brain = 0usize;

    // ── Competitive coevolution loop ──
    for cycle in 0..n_cycles {
        let cycle_start_time = Instant::now();
        println!(
            "\n╔══════════════════════════════════════════════════════╗"
        );
        println!(
            "║  CYCLE {}/{}: evolve {} brains x {} steps             ║",
            cycle + 1,
            n_cycles,
            n_brains,
            cycle_len
        );
        println!(
            "╚══════════════════════════════════════════════════════╝"
        );

        // ── Phase 1: EVOLVE each Brain independently ──
        for b in 0..brains.len() {
            let brain = &mut brains[b];
            for step in 0..cycle_len {
                let outcome = evolution_step_jackpot(
                    &mut brain.net,
                    &mut brain.proj,
                    &mut brain.rng,
                    &mut brain.eval_rng,
                    |n, p, er| {
                        eval_smooth_proj(
                            n,
                            p,
                            &table,
                            &pair_ids,
                            &hot_to_idx,
                            &bigram,
                            EVAL_LEN,
                            er,
                            &init.propagation,
                            init.output_start(),
                            h,
                            &freq_classes,
                        )
                    },
                    &evo_config,
                    9,
                );
                match outcome {
                    StepOutcome::Accepted => brain.accepted += 1,
                    StepOutcome::Rejected => brain.rejected += 1,
                    StepOutcome::Skipped => {}
                }

                if (step + 1) % PROGRESS_INTERVAL == 0 {
                    let acc = eval_accuracy_proj(
                        &mut brain.net,
                        &brain.proj,
                        &table,
                        &pair_ids,
                        &hot_to_idx,
                        FULL_EVAL_LEN,
                        &mut brain.eval_rng,
                        &init.propagation,
                        init.output_start(),
                        h,
                        &freq_classes,
                    );
                    if acc > brain.peak {
                        brain.peak = acc;
                    }
                    let rate = brain.accepted as f64
                        / (brain.accepted + brain.rejected).max(1) as f64
                        * 100.0;
                    println!(
                        "  Brain[{}] step {:>5}: acc={:.2}%  peak={:.2}%  accept={:.0}%  edges={}",
                        b,
                        step + 1,
                        acc * 100.0,
                        brain.peak * 100.0,
                        rate,
                        brain.net.edge_count()
                    );
                }
            }
        }

        // ── Phase 2: EVALUATE all Brains, RANK by accuracy ──
        println!("\n  --- EVALUATION ---");
        let mut scores: Vec<(usize, f64)> = Vec::with_capacity(brains.len());
        for (b, brain) in brains.iter_mut().enumerate() {
            let acc = eval_accuracy_proj(
                &mut brain.net,
                &brain.proj,
                &table,
                &pair_ids,
                &hot_to_idx,
                FULL_EVAL_LEN,
                &mut brain.eval_rng,
                &init.propagation,
                init.output_start(),
                h,
                &freq_classes,
            );
            if acc > brain.peak {
                brain.peak = acc;
            }
            scores.push((b, acc));
            println!(
                "  Brain[{}] seed={}: acc={:.2}%  peak={:.2}%  edges={}",
                b,
                brain.seed,
                acc * 100.0,
                brain.peak * 100.0,
                brain.net.edge_count()
            );
        }

        // Sort by accuracy (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let best_idx = scores[0].0;
        let second_idx = scores[1].0;
        let worst_idx = scores[scores.len() - 1].0;
        let best_acc = scores[0].1;

        println!(
            "\n  RANKING: best=Brain[{}] ({:.2}%), second=Brain[{}] ({:.2}%), worst=Brain[{}] ({:.2}%)",
            best_idx,
            scores[0].1 * 100.0,
            second_idx,
            scores[1].1 * 100.0,
            worst_idx,
            scores[scores.len() - 1].1 * 100.0
        );

        // Track global peak
        if best_acc > global_peak {
            global_peak = best_acc;
            global_peak_brain = best_idx;
            let cp_path = checkpoint_dir.join("breed_global_peak.ckpt");
            save_checkpoint(
                &cp_path,
                &brains[best_idx].net,
                &brains[best_idx].proj,
                CheckpointMeta {
                    step: (cycle + 1) * cycle_len,
                    accuracy: best_acc,
                    label: format!(
                        "breed_cycle{}_brain{}_seed{}",
                        cycle + 1,
                        best_idx,
                        brains[best_idx].seed
                    ),
                },
            )
            .ok();
            println!(
                "  NEW GLOBAL PEAK: {:.2}% (Brain[{}], saved checkpoint)",
                global_peak * 100.0, best_idx
            );
        }

        // ── Phase 3: BREED — consensus merge of top-2 parents ──
        if brains.len() >= 3 {
            println!("\n  --- BREED: Brain[{}] x Brain[{}] ---", best_idx, second_idx);

            let edges_before_merge;
            let (child_net, child_proj) = {
                let better = &brains[best_idx];
                let worse = &brains[second_idx];
                edges_before_merge = better.net.edge_count();
                consensus_merge(&better.net, &worse.net, &better.proj, h)
            };

            // ── Phase 4: CRYSTALLIZE the child ──
            println!(
                "  --- CRYSTALLIZE: testing {} edges ---",
                CRYSTALLIZE_SAMPLES
            );

            let child_seed = brains[worst_idx].seed + 500 * (cycle as u64 + 1);
            let mut child_rng = StdRng::seed_from_u64(child_seed);
            let mut child_eval_rng = StdRng::seed_from_u64(child_seed + 1000);

            // We need owned copies since crystallize takes &mut
            let mut cryst_net = child_net;
            let cryst_proj = child_proj;

            let pruned = crystallize(
                &mut cryst_net,
                &cryst_proj,
                &table,
                &pair_ids,
                &hot_to_idx,
                &bigram,
                &freq_classes,
                &mut child_eval_rng,
                &init.propagation,
                init.output_start(),
                h,
                CRYSTALLIZE_SAMPLES,
                &mut child_rng,
            );

            let child_acc = eval_accuracy_proj(
                &mut cryst_net,
                &cryst_proj,
                &table,
                &pair_ids,
                &hot_to_idx,
                FULL_EVAL_LEN,
                &mut child_eval_rng,
                &init.propagation,
                init.output_start(),
                h,
                &freq_classes,
            );

            println!(
                "  CRYSTALLIZED: pruned {}/{} edges ({}->{}), child acc={:.2}%",
                pruned,
                CRYSTALLIZE_SAMPLES,
                edges_before_merge,
                cryst_net.edge_count(),
                child_acc * 100.0
            );

            // Seed rooted pathways in the child to ensure reachability
            let n_paths = seed_rooted_pathways(
                &mut cryst_net,
                init.input_end(),
                init.output_start(),
                15,
                &mut child_rng,
            );
            println!(
                "  Child rooted pathways: {}, final edges={}",
                n_paths,
                cryst_net.edge_count()
            );

            // ── Phase 5: REPLACE worst Brain with child ──
            println!(
                "  REPLACE: Brain[{}] (worst) <- child (from Brain[{}] x Brain[{}])",
                worst_idx, best_idx, second_idx
            );

            let child_peak = child_acc;
            brains[worst_idx] = BrainState {
                net: cryst_net,
                proj: cryst_proj,
                rng: child_rng,
                eval_rng: child_eval_rng,
                seed: child_seed,
                accepted: 0,
                rejected: 0,
                peak: child_peak,
            };
        }

        // ── Per-cycle summary table ──
        println!("\n  ┌─────────────────────────────────────────────────────────┐");
        println!(
            "  │  CYCLE {}/{} SUMMARY  ({:.1}s)                              │",
            cycle + 1,
            n_cycles,
            cycle_start_time.elapsed().as_secs_f64()
        );
        println!("  ├─────────┬──────────┬──────────┬──────────┬──────────────┤");
        println!("  │  Brain  │  Acc%    │  Peak%   │  Edges   │  Accept%     │");
        println!("  ├─────────┼──────────┼──────────┼──────────┼──────────────┤");
        for (b, brain) in brains.iter_mut().enumerate() {
            let acc = eval_accuracy_proj(
                &mut brain.net,
                &brain.proj,
                &table,
                &pair_ids,
                &hot_to_idx,
                FULL_EVAL_LEN,
                &mut brain.eval_rng,
                &init.propagation,
                init.output_start(),
                h,
                &freq_classes,
            );
            let rate = brain.accepted as f64
                / (brain.accepted + brain.rejected).max(1) as f64
                * 100.0;
            println!(
                "  │  [{:>1}]    │  {:>5.2}%  │  {:>5.2}%  │  {:>5}   │  {:>5.1}%       │",
                b,
                acc * 100.0,
                brain.peak * 100.0,
                brain.net.edge_count(),
                rate
            );
        }
        println!("  └─────────┴──────────┴──────────┴──────────┴──────────────┘");
        println!(
            "  Global peak: {:.2}% (Brain[{}])",
            global_peak * 100.0, global_peak_brain
        );
    }

    // ── Final results ──
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  BREED FINAL RESULTS                                ║");
    println!("╚══════════════════════════════════════════════════════╝");

    for (b, brain) in brains.iter_mut().enumerate() {
        let final_acc = eval_accuracy_proj(
            &mut brain.net,
            &brain.proj,
            &table,
            &pair_ids,
            &hot_to_idx,
            FULL_EVAL_LEN.min(pair_ids.len() / 2),
            &mut brain.eval_rng,
            &init.propagation,
            init.output_start(),
            h,
            &freq_classes,
        );
        brain.peak = brain.peak.max(final_acc);
        let rate = brain.accepted as f64
            / (brain.accepted + brain.rejected).max(1) as f64
            * 100.0;
        println!(
            "  Brain[{}] seed={}: FINAL={:.2}%  peak={:.2}%  edges={}  accept={:.0}%",
            b,
            brain.seed,
            final_acc * 100.0,
            brain.peak * 100.0,
            brain.net.edge_count(),
            rate
        );

        // Save final checkpoint for each brain
        let cp_path = checkpoint_dir.join(format!("breed_brain{}_final.ckpt", b));
        save_checkpoint(
            &cp_path,
            &brain.net,
            &brain.proj,
            CheckpointMeta {
                step: n_cycles * cycle_len,
                accuracy: final_acc,
                label: format!("breed_brain{}_seed{}_final", b, brain.seed),
            },
        )
        .ok();
    }

    println!(
        "\n  Global peak: {:.2}% (Brain[{}])",
        global_peak * 100.0, global_peak_brain
    );
    println!("  Runtime: {:.1}s", t_start.elapsed().as_secs_f64());
}
