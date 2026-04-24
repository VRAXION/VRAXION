//! ABC↔Brain with Int8Projection output for byte-pair prediction.
//!
//! Input:  byte-pair → C embed → quantize [0,7] → Brain input neurons
//! Output: Int8Projection(phi_dim, n_hot) → softmax → predicted byte-pair
//!
//! This uses the PROVEN output mechanism (co-evolved Int8Projection + smooth
//! bigram fitness) but scales to 3386 byte-pair classes instead of 27 chars.
//!
//! Run:
//! ```
//! cargo run --release --example evolve_bytepair_proj -- <corpus.txt> <packed.bin>
//! ```

use instnct_core::{
    build_network, cosine_similarity, evolution_step_jackpot, save_checkpoint, softmax,
    InitConfig, Int8Projection, Network, PropagationConfig, StepOutcome, VcbpTable,
};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::VecDeque;
use std::env;
use std::path::Path;
use std::time::Instant;

const DEFAULT_STEPS: usize = 20_000;
const DEFAULT_SEEDS: usize = 3;
const DEFAULT_EVAL_LEN: usize = 100;
const DEFAULT_FULL_LEN: usize = 1_000;
const PROGRESS_INTERVAL: usize = 2_000;
const MAX_CHARGE: i32 = 7;

// ── Multi-channel input builder ──────────────────────────────────
//
// Instead of only feeding C embedding (32 dims) into 158 input neurons,
// fill multiple "sensory channels" with different views of the same input.
// This creates richer, more diverse input patterns that are harder for
// the Brain to collapse into a single attractor.

/// Build dual-pipeline input: current + previous byte-pair C embeddings
/// in separate brain regions. Like two eyes on consecutive text positions.
///   0..31:  C embedding of current pair (token t)
///   32..63: C embedding of previous pair (token t-1)
fn build_dual_input(
    table: &VcbpTable,
    cur_id: u16,
    prev_id: u16,
    neuron_count: usize,
) -> Vec<i32> {
    let mut input = vec![0i32; neuron_count];
    let e = table.e; // 32

    // Pipeline 1: current token → neurons 0..31
    let emb_cur = table.embed_id(cur_id);
    table.quantize_to_input(emb_cur, &mut input[..e], MAX_CHARGE);

    // Pipeline 2: previous token → neurons 32..63
    let emb_prev = table.embed_id(prev_id);
    table.quantize_to_input(emb_prev, &mut input[e..e * 2], MAX_CHARGE);

    input
}

/// Pre-compute frequency class for each pair_id (0-7 buckets by corpus frequency).
fn build_freq_classes(pair_ids: &[u16]) -> Vec<u8> {
    let mut freq = vec![0u32; 65536];
    for &pid in pair_ids { freq[pid as usize] += 1; }
    let max_freq = *freq.iter().max().unwrap_or(&1) as f64;
    freq.iter().map(|&f| {
        if f == 0 { 0u8 }
        else { ((f as f64 / max_freq).sqrt() * 7.0).round().min(7.0) as u8 }
    }).collect()
}

// ── Rooted pathway (reused from evolve_bytepair.rs) ──

fn bfs_forward(net: &Network, starts: &[usize], max_hops: usize) -> Vec<bool> {
    let h = net.neuron_count();
    let mut reached = vec![false; h];
    let mut queue = VecDeque::new();
    for &s in starts { reached[s] = true; queue.push_back((s, 0usize)); }
    let mut adj: Vec<Vec<u16>> = vec![Vec::new(); h];
    for edge in net.graph().iter_edges() { adj[edge.source as usize].push(edge.target); }
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_hops { continue; }
        for &tgt in &adj[node] {
            if !reached[tgt as usize] { reached[tgt as usize] = true; queue.push_back((tgt as usize, depth + 1)); }
        }
    }
    reached
}

fn bfs_reverse(net: &Network, ends: &[usize], max_hops: usize) -> Vec<bool> {
    let h = net.neuron_count();
    let mut reached = vec![false; h];
    let mut queue = VecDeque::new();
    for &e in ends { reached[e] = true; queue.push_back((e, 0usize)); }
    let mut rev_adj: Vec<Vec<u16>> = vec![Vec::new(); h];
    for edge in net.graph().iter_edges() { rev_adj[edge.target as usize].push(edge.source); }
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_hops { continue; }
        for &src in &rev_adj[node] {
            if !reached[src as usize] { reached[src as usize] = true; queue.push_back((src as usize, depth + 1)); }
        }
    }
    reached
}

fn seed_rooted_pathways(net: &mut Network, input_end: usize, output_start: usize, n_pathways: usize, rng: &mut impl Rng) -> usize {
    let h = net.neuron_count();
    let from_input = bfs_forward(net, &(0..input_end).collect::<Vec<_>>(), 4);
    let to_output = bfs_reverse(net, &(output_start..h).collect::<Vec<_>>(), 4);
    let input_anchors: Vec<usize> = (0..h).filter(|&n| from_input[n] && n < output_start).collect();
    let output_anchors: Vec<usize> = (0..h).filter(|&n| to_output[n] && n >= input_end).collect();
    if input_anchors.is_empty() || output_anchors.is_empty() {
        for _ in 0..n_pathways.min(5) { net.graph_mut().add_edge(rng.gen_range(0..input_end) as u16, rng.gen_range(output_start..h) as u16); }
        return 0;
    }
    let mut built = 0;
    for _ in 0..n_pathways {
        let ai = input_anchors[rng.gen_range(0..input_anchors.len())];
        let ao = output_anchors[rng.gen_range(0..output_anchors.len())];
        let avail: Vec<usize> = (0..h).filter(|n| *n != ai && *n != ao).collect();
        if avail.len() < 2 { continue; }
        let nm = rng.gen_range(2..=3.min(avail.len()));
        let mut mids = Vec::new();
        let mut pool = avail;
        for _ in 0..nm { let idx = rng.gen_range(0..pool.len()); mids.push(pool.swap_remove(idx)); }
        let mut chain = vec![ai]; chain.extend(&mids); chain.push(ao);
        let mut added = false;
        for w in chain.windows(2) { if net.graph_mut().add_edge(w[0] as u16, w[1] as u16) { added = true; } }
        if net.graph_mut().add_edge(ao as u16, ai as u16) { added = true; }
        if added { for &n in &chain { let sd = &mut net.spike_data_mut()[n]; if sd.threshold > 1 { sd.threshold -= 1; } } built += 1; }
    }
    built
}

// ── Byte-pair corpus helpers ──

/// Build mapping: raw corpus byte-pairs → top-N pair indices by frequency.
/// Returns (pair_ids, top_to_idx, idx_to_top, n_classes).
fn build_corpus_pairs(corpus: &[u8], table: &VcbpTable, max_classes: usize) -> (Vec<u16>, Vec<usize>, Vec<u16>, usize) {
    let n_pairs = corpus.len() / 2;
    let mut pair_ids = Vec::with_capacity(n_pairs);
    let mut freq = vec![0u32; 65536];
    for i in 0..n_pairs {
        let pid = VcbpTable::pair_id(corpus[i * 2], corpus[i * 2 + 1]);
        pair_ids.push(pid);
        freq[pid as usize] += 1;
    }

    // Rank hot pairs by frequency, take top-N
    let mut hot_freq: Vec<(u16, u32)> = (0..65536u32)
        .filter(|&v| table.is_hot(v as u16) && freq[v as usize] > 0)
        .map(|v| (v as u16, freq[v as usize]))
        .collect();
    hot_freq.sort_by(|a, b| b.1.cmp(&a.1));
    let n_classes = hot_freq.len().min(max_classes);
    hot_freq.truncate(n_classes);

    let mut top_ids: Vec<u16> = hot_freq.iter().map(|&(id, _)| id).collect();
    let mut top_to_idx = vec![usize::MAX; 65536];
    for (i, &id) in top_ids.iter().enumerate() {
        top_to_idx[id as usize] = i;
    }

    (pair_ids, top_to_idx, top_ids, n_classes)
}

/// Build byte-pair bigram: P(next_hot | current_hot).
fn build_pair_bigram(pair_ids: &[u16], hot_to_idx: &[usize], n_hot: usize) -> Vec<Vec<f64>> {
    let mut counts = vec![vec![0u32; n_hot]; n_hot];
    for i in 0..pair_ids.len().saturating_sub(1) {
        let ci = hot_to_idx[pair_ids[i] as usize];
        let ni = hot_to_idx[pair_ids[i + 1] as usize];
        if ci != usize::MAX && ni != usize::MAX {
            counts[ci][ni] += 1;
        }
    }
    counts.iter().map(|row| {
        let total: f64 = row.iter().map(|&c| c as f64).sum();
        if total < 1.0 { vec![1.0 / n_hot as f64; n_hot] }
        else { row.iter().map(|&c| c as f64 / total).collect() }
    }).collect()
}

// ── Eval functions using Int8Projection ──

fn eval_accuracy_proj(
    net: &mut Network, proj: &Int8Projection, table: &VcbpTable,
    pair_ids: &[u16], hot_to_idx: &[usize],
    len: usize, rng: &mut StdRng,
    propagation: &PropagationConfig, output_start: usize, neuron_count: usize,
    _freq_classes: &[u8],
) -> f64 {
    let n = pair_ids.len();
    if n <= len + 1 { return 0.0; }
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

/// Smooth cosine + boredom penalty: rewards accuracy AND output diversity.
/// If the Brain always predicts the same class → diversity≈0 → fitness stays low.
/// If the Brain differentiates inputs → diversity grows → fitness amplified.
fn eval_cosine_boredom(
    net: &mut Network, proj: &Int8Projection, table: &VcbpTable,
    pair_ids: &[u16], hot_to_idx: &[usize], bigram: &[Vec<f64>],
    len: usize, rng: &mut StdRng,
    propagation: &PropagationConfig, output_start: usize, neuron_count: usize,
) -> f64 {
    let n = pair_ids.len();
    if n <= len { return 0.0; }
    let off = rng.gen_range(0..=n - len - 1);
    net.reset();
    let e = table.e;
    let mut total_cos = 0.0f64;
    let mut counted = 0usize;
    let mut predictions: Vec<usize> = Vec::with_capacity(len);

    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let cur_idx = hot_to_idx[cur_id as usize];

        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
        net.propagate(&input, propagation).unwrap();

        if cur_idx == usize::MAX { continue; }

        let scores = proj.raw_scores(&net.charge_vec(output_start..neuron_count));
        let probs = softmax(&scores);
        let target = &bigram[cur_idx];
        total_cos += cosine_similarity(&probs, target);

        // Track argmax prediction for diversity
        let pred = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        predictions.push(pred);
        counted += 1;
    }

    if counted == 0 { return 0.0; }

    let mean_cos = total_cos / counted as f64;

    // Diversity: fraction of unique predictions
    let mut unique = predictions.clone();
    unique.sort();
    unique.dedup();
    let diversity = unique.len() as f64 / counted as f64;

    // Boredom penalty: multiplicative — diversity amplifies cosine
    // λ=0.5: if diversity=0.01 (constant) → 1.005× (almost no boost)
    //         if diversity=0.5 (varied)    → 1.25× (25% boost)
    let lambda = 0.5;
    mean_cos * (1.0 + lambda * diversity)
}

/// Top-N masked cosine: only compare against the N most likely next pairs.
/// Keeps linear cosine (no log info loss) but removes tail noise.
/// N annealing: large N = smooth landscape, small N = focused signal.
fn eval_topn_cosine(
    net: &mut Network, proj: &Int8Projection, table: &VcbpTable,
    pair_ids: &[u16], hot_to_idx: &[usize], bigram: &[Vec<f64>],
    len: usize, rng: &mut StdRng,
    propagation: &PropagationConfig, output_start: usize, neuron_count: usize,
    top_n: usize,
) -> f64 {
    let n = pair_ids.len();
    if n <= len { return 0.0; }
    let off = rng.gen_range(0..=n - len - 1);
    net.reset();
    let e = table.e;
    let n_classes = bigram[0].len();
    let mut total_cos = 0.0f64;
    let mut counted = 0usize;

    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let cur_idx = hot_to_idx[cur_id as usize];

        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
        net.propagate(&input, propagation).unwrap();

        if cur_idx == usize::MAX { continue; }

        let scores = proj.raw_scores(&net.charge_vec(output_start..neuron_count));
        let probs = softmax(&scores);
        let target = &bigram[cur_idx];

        // Find top-N indices of the bigram target
        let mut indices: Vec<(usize, f64)> = target.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_indices: Vec<usize> = indices.iter().take(top_n).map(|&(i, _)| i).collect();

        // Mask: only keep top-N, zero rest, renormalize
        let mut masked_target = vec![0.0f64; n_classes];
        let mut masked_probs = vec![0.0f64; n_classes];
        for &idx in &top_indices {
            masked_target[idx] = target[idx];
            masked_probs[idx] = probs[idx];
        }
        // Renormalize
        let sum_t: f64 = masked_target.iter().sum();
        let sum_p: f64 = masked_probs.iter().sum();
        if sum_t > 0.0 { for v in masked_target.iter_mut() { *v /= sum_t; } }
        if sum_p > 0.0 { for v in masked_probs.iter_mut() { *v /= sum_p; } }

        total_cos += cosine_similarity(&masked_probs, &masked_target);
        counted += 1;
    }
    if counted == 0 { 0.0 } else { total_cos / counted as f64 }
}

/// Ultra-smooth blend: linear cosine → log cosine.
/// blend=0: pure linear cosine (smooth, proven 7.1%).
/// blend=0.3: 70% linear + 30% log (gentle sharpening).
/// The log component makes peaks more prominent WITHOUT destroying the smooth landscape.
fn eval_blended_proj(
    net: &mut Network, proj: &Int8Projection, table: &VcbpTable,
    pair_ids: &[u16], hot_to_idx: &[usize], bigram: &[Vec<f64>],
    len: usize, rng: &mut StdRng,
    propagation: &PropagationConfig, output_start: usize, neuron_count: usize,
    blend: f64,
) -> f64 {
    let n = pair_ids.len();
    if n <= len { return 0.0; }
    let off = rng.gen_range(0..=n - len - 1);
    net.reset();
    let e = table.e;
    let eps = 1e-10;
    let mut total = 0.0f64;
    let mut counted = 0usize;
    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let cur_idx = hot_to_idx[cur_id as usize];

        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
        net.propagate(&input, propagation).unwrap();

        if cur_idx == usize::MAX { continue; }

        let scores = proj.raw_scores(&net.charge_vec(output_start..neuron_count));
        let probs = softmax(&scores);
        let target = &bigram[cur_idx];

        // Linear cosine
        let cos_linear = cosine_similarity(&probs, target);

        // Log cosine (peaks amplified, tail compressed)
        let log_probs: Vec<f64> = probs.iter().map(|&p| (p + eps).ln()).collect();
        let log_target: Vec<f64> = target.iter().map(|&t| (t + eps).ln()).collect();
        let cos_log = cosine_similarity(&log_probs, &log_target);

        // Blend
        total += (1.0 - blend) * cos_linear + blend * cos_log;
        counted += 1;
    }
    if counted == 0 { 0.0 } else { total / counted as f64 }
}

/// Mean P(target): smooth accuracy — the probability assigned to the correct next pair.
/// Smooth (continuous in P), directional (only correct target matters),
/// not too steep (no log), penalizes constant output.
fn eval_mean_p_target(
    net: &mut Network, proj: &Int8Projection, table: &VcbpTable,
    pair_ids: &[u16], hot_to_idx: &[usize],
    len: usize, rng: &mut StdRng,
    propagation: &PropagationConfig, output_start: usize, neuron_count: usize,
) -> f64 {
    let n = pair_ids.len();
    if n <= len + 1 { return 0.0; }
    let off = rng.gen_range(0..=n - len - 2);
    net.reset();
    let e = table.e;
    let mut total_p = 0.0f64;
    let mut counted = 0usize;
    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let tgt_id = pair_ids[off + i + 1];
        let tgt_idx = hot_to_idx[tgt_id as usize];

        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
        net.propagate(&input, propagation).unwrap();

        if tgt_idx == usize::MAX { continue; }

        let scores = proj.raw_scores(&net.charge_vec(output_start..neuron_count));
        let probs = softmax(&scores);
        total_p += probs[tgt_idx]; // P(correct target)
        counted += 1;
    }
    if counted == 0 { 0.0 } else { total_p / counted as f64 }
}

/// Cosine to bigram distribution (original smooth cosine, linear space).
fn eval_smooth_proj(
    net: &mut Network, proj: &Int8Projection, table: &VcbpTable,
    pair_ids: &[u16], hot_to_idx: &[usize], bigram: &[Vec<f64>],
    len: usize, rng: &mut StdRng,
    propagation: &PropagationConfig, output_start: usize, neuron_count: usize,
    _freq_classes: &[u8],
) -> f64 {
    let n = pair_ids.len();
    if n <= len + 1 { return 0.0; }
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

        if cur_idx == usize::MAX { continue; }

        let scores = proj.raw_scores(&net.charge_vec(output_start..neuron_count));
        let probs = softmax(&scores);
        let target = &bigram[cur_idx];
        total_cos += cosine_similarity(&probs, target);
        counted += 1;
    }
    if counted == 0 { 0.0 } else { total_cos / counted as f64 }
}

/// Smooth cosine with temperature-controlled softmax.
/// T=1.0: standard softmax (soft, easy to climb)
/// T→0: sharper softmax (peaky, punishes wrong predictions harder)
fn eval_smooth_proj_temp(
    net: &mut Network, proj: &Int8Projection, table: &VcbpTable,
    pair_ids: &[u16], hot_to_idx: &[usize], bigram: &[Vec<f64>],
    len: usize, rng: &mut StdRng,
    propagation: &PropagationConfig, output_start: usize, neuron_count: usize,
    temperature: f64,
) -> f64 {
    let n = pair_ids.len();
    if n <= len { return 0.0; }
    let off = rng.gen_range(0..=n - len - 1);
    net.reset();
    let e = table.e;
    let mut total_cos = 0.0f64;
    let mut counted = 0usize;
    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let cur_idx = hot_to_idx[cur_id as usize];

        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
        net.propagate(&input, propagation).unwrap();

        if cur_idx == usize::MAX { continue; }

        let raw_scores = proj.raw_scores(&net.charge_vec(output_start..neuron_count));
        // Temperature-scaled softmax: divide scores by T before exp
        let scaled: Vec<f64> = raw_scores.iter().map(|&s| s as f64 / temperature).collect();
        let max_s = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = scaled.iter().map(|&s| (s - max_s).exp()).collect();
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 { for p in probs.iter_mut() { *p /= sum; } }
        let target = &bigram[cur_idx];
        total_cos += cosine_similarity(&probs, target);
        counted += 1;
    }
    if counted == 0 { 0.0 } else { total_cos / counted as f64 }
}

/// Hybrid fitness: smooth cosine bigram + CE target sharpening.
///
/// `fitness = α * cosine_to_bigram + (1-α) * normalized_log_p_target`
///
/// Cosine alone → converges to frequency baseline (smooth but lazy).
/// CE alone → too steep for mutation search (sharp but unclimbable).
/// Hybrid: cosine provides smooth gradient, CE sharpens toward correct target.
fn eval_hybrid_proj(
    net: &mut Network, proj: &Int8Projection, table: &VcbpTable,
    pair_ids: &[u16], hot_to_idx: &[usize], bigram: &[Vec<f64>],
    len: usize, rng: &mut StdRng,
    propagation: &PropagationConfig, output_start: usize, neuron_count: usize,
    alpha: f64,  // blend: 1.0 = pure cosine, 0.0 = pure CE
) -> f64 {
    let n = pair_ids.len();
    if n <= len { return 0.0; }
    let off = rng.gen_range(0..=n - len - 1);
    net.reset();
    let e = table.e;
    let n_classes = bigram[0].len();
    let eps = 1e-10;
    // log(1/N) is the worst-case CE (uniform prediction), use for normalization
    let worst_ce = (1.0 / n_classes as f64).ln(); // e.g., ln(1/397) ≈ -5.98

    let mut total_cos = 0.0f64;
    let mut total_ce_norm = 0.0f64;
    let mut counted = 0usize;

    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let cur_idx = hot_to_idx[cur_id as usize];
        let tgt_id = pair_ids[(off + i + 1).min(n - 1)];
        let tgt_idx = hot_to_idx[tgt_id as usize];

        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
        net.propagate(&input, propagation).unwrap();

        if cur_idx == usize::MAX || tgt_idx == usize::MAX { continue; }

        let scores = proj.raw_scores(&net.charge_vec(output_start..neuron_count));
        let probs = softmax(&scores);

        // Component 1: cosine similarity to bigram distribution
        let target_dist = &bigram[cur_idx];
        total_cos += cosine_similarity(&probs, target_dist);

        // Component 2: normalized log P(target) — maps [worst_ce, 0] → [0, 1]
        let log_p = probs[tgt_idx].max(eps).ln();
        let ce_normalized = 1.0 - (log_p / worst_ce); // 0 = worst (uniform), 1 = perfect
        total_ce_norm += ce_normalized;

        counted += 1;
    }
    if counted == 0 {
        return 0.0;
    }
    let mean_cos = total_cos / counted as f64;
    let mean_ce = total_ce_norm / counted as f64;
    alpha * mean_cos + (1.0 - alpha) * mean_ce
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("Usage: evolve_bytepair_proj <corpus.txt> <packed.bin> [--steps N] [--seeds N]");
        std::process::exit(1);
    }

    let corpus_path = &args[0];
    let packed_path = &args[1];
    let mut steps = DEFAULT_STEPS;
    let mut seed_count = DEFAULT_SEEDS;
    let mut cli_h: usize = 256;
    let mut cli_seed: Option<u64> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => { i += 1; steps = args[i].parse().unwrap(); }
            "--seeds" => { i += 1; seed_count = args[i].parse().unwrap(); }
            "--seed" => { i += 1; cli_seed = Some(args[i].parse().unwrap()); }
            "--H" => { i += 1; cli_h = args[i].parse().unwrap(); }
            other => panic!("unknown flag: {other}"),
        }
        i += 1;
    }
    // --seed overrides the multi-seed loop: single run with the given seed.
    if cli_seed.is_some() {
        seed_count = 1;
    }

    // Load
    println!("Loading C embedding table...");
    let table = VcbpTable::from_packed(Path::new(packed_path)).unwrap();
    println!("  {table}");

    println!("Loading corpus...");
    let corpus = std::fs::read(corpus_path).unwrap();
    let max_classes = 397; // back to full byte-pair with CE fitness
    let (pair_ids, hot_to_idx, hot_ids, n_classes) = build_corpus_pairs(&corpus, &table, max_classes);
    println!("  {} bytes = {} byte-pairs, {} output classes (top-{})", corpus.len(), pair_ids.len(), n_classes, max_classes);

    println!("Building byte-pair bigram...");
    let bigram = build_pair_bigram(&pair_ids, &hot_to_idx, n_classes);
    let freq_classes = build_freq_classes(&pair_ids);

    // Baselines
    let random_baseline = 1.0 / n_classes as f64;
    let mut freq = vec![0u32; n_classes];
    for &pid in &pair_ids {
        let idx = hot_to_idx[pid as usize];
        if idx != usize::MAX { freq[idx] += 1; }
    }
    let most = freq.iter().enumerate().max_by_key(|(_, &c)| c).unwrap();
    let freq_baseline = *most.1 as f64 / pair_ids.len() as f64;
    let most_pair = hot_ids[most.0];
    let (mh, ml) = VcbpTable::pair_bytes(most_pair);
    println!("  Random: {:.3}%  Freq('{}{}'): {:.1}%",
        random_baseline * 100.0,
        if mh.is_ascii_graphic() || mh == b' ' { mh as char } else { '.' },
        if ml.is_ascii_graphic() || ml == b' ' { ml as char } else { '.' },
        freq_baseline * 100.0);

    // Brain config — scaled up
    let h = cli_h;
    let init = InitConfig::phi(h); // proven phi-overlap + chain-50
    let evo_config = init.evolution_config();

    println!("\n=== Byte-Pair Prediction with Int8Projection({}, {}) ===", init.phi_dim, n_classes);
    println!("  H={}, {} steps, {} seeds, max_charge={}\n", h, steps, seed_count, MAX_CHARGE);

    let t_start = Instant::now();

    for seed_idx in 0..seed_count {
        let seed = cli_seed.unwrap_or(42 + seed_idx as u64 * 1000);
        let seed_start = Instant::now();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net = build_network(&init, &mut rng);

        let n_paths = seed_rooted_pathways(&mut net, init.input_end(), init.output_start(), 30, &mut rng);
        println!("  [seed={seed}] Rooted pathways: {n_paths}, edges={}", net.edge_count());

        // Int8Projection sized for n_hot output classes
        let mut proj = Int8Projection::new(init.phi_dim, n_classes, &mut StdRng::seed_from_u64(seed + 200));
        let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
        let mut accepted = 0u32;
        let mut rejected = 0u32;
        let mut peak = 0.0f64;
        let checkpoint_dir = Path::new("output/bytepair_proj_checkpoints");
        std::fs::create_dir_all(checkpoint_dir).ok();

        // ── GROW-PRUNE CYCLE: evolve → crystallize → evolve → ... ──
        let cycle_len = 5000usize;
        let n_cycles = steps / cycle_len;
        let crystallize_samples = 200; // edges to test per crystallize phase

        for cycle in 0..n_cycles {
            let cycle_start = cycle * cycle_len;
            println!("  [seed={seed}] === CYCLE {}/{} (steps {}..{}) ===",
                cycle + 1, n_cycles, cycle_start, cycle_start + cycle_len);

            // Phase 1: EVOLVE — standard mutation with smooth cosine fitness
            for step in cycle_start..cycle_start + cycle_len {
                let outcome = evolution_step_jackpot(
                    &mut net, &mut proj, &mut rng, &mut eval_rng,
                    |n, p, er| eval_smooth_proj(n, p, &table, &pair_ids, &hot_to_idx, &bigram,
                        DEFAULT_EVAL_LEN, er, &init.propagation, init.output_start(), h, &freq_classes),
                    &evo_config, 9,
                );
                match outcome {
                    StepOutcome::Accepted => accepted += 1,
                    StepOutcome::Rejected => rejected += 1,
                    StepOutcome::Skipped => {}
                }
                if (step + 1) % PROGRESS_INTERVAL == 0 {
                    let acc = eval_accuracy_proj(&mut net, &proj, &table, &pair_ids, &hot_to_idx,
                        DEFAULT_FULL_LEN, &mut eval_rng, &init.propagation, init.output_start(), h, &freq_classes);
                    if acc > peak {
                        peak = acc;
                        let cp_path = checkpoint_dir.join(format!("seed{seed}_peak.bin"));
                        save_checkpoint(&cp_path, &net, &proj, instnct_core::CheckpointMeta {
                            step: step + 1, accuracy: acc,
                            label: format!("seed{seed}_peak"),
                        }).ok();
                    }
                    let rate = accepted as f64 / (accepted + rejected).max(1) as f64 * 100.0;
                    println!("  [seed={seed}] step {:>5}: acc={:.2}%  peak={:.2}%  accept={:.0}%  edges={}",
                        step + 1, acc * 100.0, peak * 100.0, rate, net.edge_count());
                }
            }

            // Phase 2: CRYSTALLIZE — prune edges that don't contribute to fitness
            if cycle < n_cycles - 1 {
                let edges_before = net.edge_count();
                let baseline_fitness = eval_smooth_proj(
                    &mut net, &proj, &table, &pair_ids, &hot_to_idx, &bigram,
                    DEFAULT_EVAL_LEN * 2, &mut eval_rng, &init.propagation,
                    init.output_start(), h, &freq_classes);

                let mut pruned = 0u32;
                let sample_count = 200usize.min(net.edge_count());

                // Sample 200 random edges — the "noisy but effective" crystallize
                for _ in 0..sample_count {
                    if net.edge_count() < 100 { break; }
                    let edge_idx = rng.gen_range(0..net.edge_count());
                    let edge = net.graph().iter_edges().nth(edge_idx);
                    if let Some(edge) = edge {
                        let src = edge.source;
                        let tgt = edge.target;
                        net.graph_mut().remove_edge(src, tgt);
                        let new_fitness = eval_smooth_proj(
                            &mut net, &proj, &table, &pair_ids, &hot_to_idx, &bigram,
                            DEFAULT_EVAL_LEN, &mut eval_rng, &init.propagation,
                            init.output_start(), h, &freq_classes);
                        if new_fitness >= baseline_fitness - 0.0001 {
                            pruned += 1;
                        } else {
                            net.graph_mut().add_edge(src, tgt);
                        }
                    }
                }

                let acc_after = eval_accuracy_proj(&mut net, &proj, &table, &pair_ids, &hot_to_idx,
                    DEFAULT_FULL_LEN, &mut eval_rng, &init.propagation, init.output_start(), h, &freq_classes);
                println!("  [seed={seed}] CRYSTALLIZE: pruned {pruned}/{sample_count} edges ({edges_before}→{}), acc={:.2}%",
                    net.edge_count(), acc_after * 100.0);
            }
        }

        let final_acc = eval_accuracy_proj(&mut net, &proj, &table, &pair_ids, &hot_to_idx,
            DEFAULT_FULL_LEN.min(pair_ids.len() / 2), &mut eval_rng, &init.propagation, init.output_start(), h, &freq_classes);
        peak = peak.max(final_acc);
        let rate = accepted as f64 / (accepted + rejected).max(1) as f64 * 100.0;
        println!("  [seed={seed}] FINAL: {:.2}%  peak={:.2}%  edges={}  accept={:.0}%",
            final_acc * 100.0, peak * 100.0, net.edge_count(), rate);

        // ── Topology analysis ──
        println!("\n  [seed={seed}] === TOPOLOGY ANALYSIS ===");
        let edges = net.edge_count();
        let density = edges as f64 / (h * h) as f64 * 100.0;
        println!("  Edges: {edges}, density: {density:.2}%");

        // In-degree / out-degree distribution
        let mut in_deg = vec![0u32; h];
        let mut out_deg = vec![0u32; h];
        let mut bidir_count = 0u32;
        for edge in net.graph().iter_edges() {
            out_deg[edge.source as usize] += 1;
            in_deg[edge.target as usize] += 1;
            if net.graph().has_edge(edge.target, edge.source) {
                bidir_count += 1;
            }
        }
        bidir_count /= 2; // each bidir pair counted twice

        let max_in = *in_deg.iter().max().unwrap_or(&0);
        let max_out = *out_deg.iter().max().unwrap_or(&0);
        let avg_deg = edges as f64 / h as f64;
        let dead = in_deg.iter().zip(out_deg.iter()).filter(|(&i, &o)| i == 0 && o == 0).count();
        println!("  Avg degree: {avg_deg:.1}, max_in: {max_in}, max_out: {max_out}");
        println!("  Bidirectional pairs: {bidir_count} ({:.1}% of edges)", bidir_count as f64 / edges.max(1) as f64 * 100.0);
        println!("  Dead neurons (no edges): {dead}/{h}");

        // Zone analysis
        let input_end = init.input_end();
        let output_start = init.output_start();
        let mut cross_io = 0u32; // edges from input zone to output zone
        let mut within_input = 0u32;
        let mut within_output = 0u32;
        let mut within_hidden = 0u32;
        for edge in net.graph().iter_edges() {
            let s = edge.source as usize;
            let t = edge.target as usize;
            let s_in = s < input_end;
            let s_out = s >= output_start;
            let t_in = t < input_end;
            let t_out = t >= output_start;
            if s_in && t_out { cross_io += 1; }
            if s_in && t_in { within_input += 1; }
            if s_out && t_out { within_output += 1; }
            if !s_in && !s_out && !t_in && !t_out { within_hidden += 1; }
        }
        println!("  Input→Output edges: {cross_io}");
        println!("  Within-input: {within_input}, within-output: {within_output}, within-hidden: {within_hidden}");

        // Loop detection (triangles)
        let mut triangles = 0u32;
        for edge in net.graph().iter_edges() {
            let a = edge.source;
            let b = edge.target;
            // Check if any c exists: b→c and c→a
            for edge2 in net.graph().iter_edges() {
                if edge2.source == b && net.graph().has_edge(edge2.target, a) {
                    triangles += 1;
                }
            }
        }
        triangles /= 3; // each triangle counted 3 times
        println!("  Triangles: {triangles}");

        // Polarity distribution
        let n_inhibitory = (0..h).filter(|&i| net.polarity()[i] < 0).count();
        println!("  Inhibitory neurons: {n_inhibitory}/{h} ({:.0}%)", n_inhibitory as f64 / h as f64 * 100.0);

        // Threshold distribution
        let thresholds: Vec<u8> = net.spike_data().iter().map(|s| s.threshold).collect();
        let avg_th = thresholds.iter().map(|&t| t as f64).sum::<f64>() / h as f64;
        let min_th = *thresholds.iter().min().unwrap_or(&0);
        let max_th = *thresholds.iter().max().unwrap_or(&0);
        println!("  Threshold: avg={avg_th:.1}, min={min_th}, max={max_th}");

        // BFS reachability: how many output neurons reachable from input?
        let from_input = bfs_forward(&net, &(0..input_end).collect::<Vec<_>>(), 6);
        let reachable_output = (output_start..h).filter(|&n| from_input[n]).count();
        println!("  Output reachable from input (6 hops): {reachable_output}/{}", h - output_start);

        // Alive-frac mean over 20 deterministically-spaced corpus pairs (for SUMMARY)
        let alive_samples = 20usize.min(pair_ids.len().max(1));
        let mut alive_frac_sum = 0.0f64;
        let out_zone = h - init.output_start();
        for k in 0..alive_samples {
            let pid = pair_ids[(k * pair_ids.len()) / alive_samples];
            net.reset();
            let emb = table.embed_id(pid);
            let mut input_buf = vec![0i32; h];
            table.quantize_to_input(emb, &mut input_buf[..table.e], MAX_CHARGE);
            net.propagate(&input_buf, &init.propagation).unwrap();
            let charges = net.charge_vec(init.output_start()..h);
            let alive = charges.iter().filter(|&&c| c > 0).count();
            alive_frac_sum += alive as f64 / out_zone.max(1) as f64;
        }
        let alive_frac_mean = alive_frac_sum / alive_samples as f64;
        let seed_wall_clock_s = seed_start.elapsed().as_secs_f64();

        // Machine-readable summary line for multi-seed drivers.
        println!(
            "SUMMARY {{\"fixture\":\"bytepair_proj\",\"seed\":{},\"H\":{},\"phi_dim\":{},\"peak_acc\":{:.6},\"final_acc\":{:.6},\"accept_rate_pct\":{:.4},\"alive_frac_mean\":{:.6},\"edges\":{},\"wall_clock_s\":{:.3}}}",
            seed, h, init.phi_dim, peak, final_acc, rate, alive_frac_mean,
            net.edge_count(), seed_wall_clock_s
        );
    }

    println!("\nRuntime: {:.1}s", t_start.elapsed().as_secs_f64());
}
