//! Byte-pair prediction evaluation for ABC↔Brain direct integration.
//!
//! Replaces the character-level eval functions (`eval_accuracy`, `eval_smooth`)
//! with byte-pair-level equivalents that use `VcbpTable` for I/O instead of
//! `SdrTable` + `Int8Projection`.
//!
//! Three eval modes:
//! - `eval_bytepair_accuracy`: hard argmax accuracy (nearest neighbor)
//! - `eval_bytepair_smooth`: cosine similarity to single target embedding
//! - `eval_bytepair_smooth_bigram`: cosine similarity to bigram distribution
//!   over output distances (proven approach from char-level, adapted to C space)

use crate::fitness::cosine_similarity;
use crate::propagation::PropagationConfig;
use crate::vcbp_io::VcbpTable;
use crate::Network;
use rand::rngs::StdRng;
use rand::Rng;

/// Argmax accuracy: fraction of correctly predicted next byte-pairs.
///
/// For each consecutive byte-pair in the corpus:
/// 1. Embed current pair via C → quantize → inject into Brain input neurons
/// 2. Propagate one token through the spiking network
/// 3. Read output neuron charges → dequantize → nearest-neighbor in C → predicted pair
/// 4. Compare to actual next pair
#[allow(clippy::too_many_arguments)]
pub fn eval_bytepair_accuracy(
    net: &mut Network,
    table: &VcbpTable,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    propagation: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    max_charge: u8,
) -> f64 {
    let n_pairs = corpus.len() / 2;
    if n_pairs <= len {
        return 0.0;
    }
    // Random even-aligned offset
    let max_start = n_pairs - len - 1;
    let pair_off = rng.gen_range(0..=max_start);
    let byte_off = pair_off * 2;

    net.reset();
    let mut correct = 0u32;
    let e = table.e;
    let mc = max_charge as i32;

    for i in 0..len {
        let idx = byte_off + i * 2;
        let current_id = VcbpTable::pair_id(corpus[idx], corpus[idx + 1]);
        let target_id = VcbpTable::pair_id(corpus[idx + 2], corpus[idx + 3]);

        // Input: embed current pair → quantize → inject
        let emb = table.embed_id(current_id);
        let mut input = vec![0i32; neuron_count];
        table.quantize_to_input(emb, &mut input[..e], mc);

        net.propagate(&input, propagation).unwrap();

        // Output: read charges → dequantize → nearest neighbor
        let charges = net.charge_vec(output_start..output_start + e);
        let query = table.dequantize_output(&charges, max_charge);
        let predicted_id = table.nearest_hot(&query);

        if predicted_id == target_id {
            correct += 1;
        }
    }

    correct as f64 / len as f64
}

/// Smooth fitness: mean cosine similarity between output embedding and target embedding.
///
/// Instead of hard argmax, computes cosine similarity between the Brain's
/// dequantized output vector and the actual next byte-pair's C embedding.
/// This gives continuous gradient signal for evolution — a mutation that
/// shifts the output *toward* the target embedding is rewarded even if the
/// nearest-neighbor argmax doesn't flip.
#[allow(clippy::too_many_arguments)]
pub fn eval_bytepair_smooth(
    net: &mut Network,
    table: &VcbpTable,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    propagation: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    max_charge: u8,
) -> f64 {
    let n_pairs = corpus.len() / 2;
    if n_pairs <= len {
        return 0.0;
    }
    let max_start = n_pairs - len - 1;
    let pair_off = rng.gen_range(0..=max_start);
    let byte_off = pair_off * 2;

    net.reset();
    let mut total_cos = 0.0f64;
    let e = table.e;
    let mc = max_charge as i32;

    for i in 0..len {
        let idx = byte_off + i * 2;
        let current_id = VcbpTable::pair_id(corpus[idx], corpus[idx + 1]);
        let target_id = VcbpTable::pair_id(corpus[idx + 2], corpus[idx + 3]);

        // Input
        let emb = table.embed_id(current_id);
        let mut input = vec![0i32; neuron_count];
        table.quantize_to_input(emb, &mut input[..e], mc);

        net.propagate(&input, propagation).unwrap();

        // Output → cosine similarity with target embedding
        let charges = net.charge_vec(output_start..output_start + e);
        let query = table.dequantize_output(&charges, max_charge);
        let target_emb = table.embed_id(target_id);

        // Convert to f64 for cosine
        let q64: Vec<f64> = query.iter().map(|&v| v as f64).collect();
        let t64: Vec<f64> = target_emb.iter().map(|&v| v as f64).collect();
        total_cos += cosine_similarity(&q64, &t64);
    }

    total_cos / len as f64
}

/// Byte-pair bigram table: for each hot pair, distribution over likely next hot pairs.
///
/// Returns `(hot_ids, bigram)` where `bigram[i]` is a probability distribution
/// (sums to ~1.0) over hot pair indices, given that the current pair is `hot_ids[i]`.
pub fn build_bytepair_bigram(
    corpus: &[u8],
    table: &VcbpTable,
) -> (Vec<u16>, Vec<Vec<f64>>) {
    // Collect hot pair IDs in order
    let mut hot_ids: Vec<u16> = Vec::with_capacity(table.n_hot);
    let mut hot_to_idx = vec![usize::MAX; table.vocab_size];
    for vid in 0..table.vocab_size {
        if table.is_hot(vid as u16) {
            hot_to_idx[vid] = hot_ids.len();
            hot_ids.push(vid as u16);
        }
    }
    let n = hot_ids.len();

    // Count bigram transitions between hot pairs
    let mut counts = vec![vec![0u32; n]; n];
    let n_pairs = corpus.len() / 2;
    for i in 0..n_pairs.saturating_sub(1) {
        let idx = i * 2;
        let cur = ((corpus[idx] as usize) << 8) | corpus[idx + 1] as usize;
        let nxt = ((corpus[idx + 2] as usize) << 8) | corpus[idx + 3] as usize;
        let ci = hot_to_idx[cur];
        let ni = hot_to_idx[nxt];
        if ci != usize::MAX && ni != usize::MAX {
            counts[ci][ni] += 1;
        }
    }

    // Normalize to probabilities
    let bigram: Vec<Vec<f64>> = counts
        .iter()
        .map(|row| {
            let total: f64 = row.iter().map(|&c| c as f64).sum();
            if total < 1.0 {
                vec![1.0 / n as f64; n] // uniform fallback
            } else {
                row.iter().map(|&c| c as f64 / total).collect()
            }
        })
        .collect();

    (hot_ids, bigram)
}

/// Smooth fitness via bigram distribution: the proven approach from char-level.
///
/// For each token:
/// 1. Dequantize output charges → query vector (32-dim)
/// 2. Compute negative L2 distance from query to EVERY hot embedding → scores
/// 3. Softmax over scores → predicted distribution over hot pairs
/// 4. Cosine similarity between predicted distribution and bigram distribution
///
/// This gives much richer signal than single-target cosine because:
/// - Partial credit: output near ANY likely next pair gets rewarded
/// - Distribution captures full statistics, not just argmax
/// - Smooth landscape for mutation search
#[allow(clippy::too_many_arguments)]
pub fn eval_bytepair_smooth_bigram(
    net: &mut Network,
    table: &VcbpTable,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    propagation: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    max_charge: u8,
    hot_ids: &[u16],
    bigram: &[Vec<f64>],
    hot_to_idx: &[usize],
) -> f64 {
    let n_pairs = corpus.len() / 2;
    if n_pairs <= len {
        return 0.0;
    }
    let max_start = n_pairs - len - 1;
    let pair_off = rng.gen_range(0..=max_start);
    let byte_off = pair_off * 2;
    let n_hot = hot_ids.len();
    let e = table.e;
    let mc = max_charge as i32;

    net.reset();
    let mut total_cos = 0.0f64;
    let mut counted = 0usize;

    for i in 0..len {
        let idx = byte_off + i * 2;
        let cur_id = ((corpus[idx] as usize) << 8) | corpus[idx + 1] as usize;

        // Only eval on hot current pairs (we have bigram data for them)
        let cur_hot_idx = hot_to_idx[cur_id];
        if cur_hot_idx == usize::MAX {
            // Still propagate to maintain network state
            let emb = table.embed_id(cur_id as u16);
            let mut input = vec![0i32; neuron_count];
            table.quantize_to_input(emb, &mut input[..e], mc);
            net.propagate(&input, propagation).unwrap();
            continue;
        }

        // Input
        let emb = table.embed_id(cur_id as u16);
        let mut input = vec![0i32; neuron_count];
        table.quantize_to_input(emb, &mut input[..e], mc);
        net.propagate(&input, propagation).unwrap();

        // Output → distance to all hot embeddings → softmax → distribution
        let charges = net.charge_vec(output_start..output_start + e);
        let query = table.dequantize_output(&charges, max_charge);

        // Compute negative L2 distances (scores) to all hot embeddings
        let mut scores = vec![0.0f64; n_hot];
        for (j, &hid) in hot_ids.iter().enumerate() {
            let h_emb = table.embed_id(hid);
            let mut dist_sq = 0.0f64;
            for d in 0..e {
                let diff = query[d] as f64 - h_emb[d] as f64;
                dist_sq += diff * diff;
            }
            scores[j] = -dist_sq; // negative distance = higher score for closer
        }

        // Softmax (with temperature to prevent overflow)
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut exp_sum = 0.0f64;
        let mut probs = vec![0.0f64; n_hot];
        for j in 0..n_hot {
            // Temperature scaling: divide by E to normalize
            probs[j] = ((scores[j] - max_score) / e as f64).exp();
            exp_sum += probs[j];
        }
        if exp_sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= exp_sum;
            }
        }

        // Cosine similarity between predicted distribution and bigram target
        let target = &bigram[cur_hot_idx];
        total_cos += cosine_similarity(&probs, target);
        counted += 1;
    }

    if counted == 0 {
        0.0
    } else {
        total_cos / counted as f64
    }
}
