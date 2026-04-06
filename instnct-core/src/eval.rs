//! Standard evaluation functions for language prediction.
//!
//! These are the canonical eval loops used in all INSTNCT language experiments:
//! propagate a corpus segment token-by-token, then measure accuracy or smooth
//! cosine fitness against the bigram distribution.

use crate::fitness::{cosine_similarity, softmax};
use crate::projection::Int8Projection;
use crate::propagation::PropagationConfig;
use crate::sdr::SdrTable;
use crate::Network;
use rand::rngs::StdRng;
use rand::Rng;

/// Argmax accuracy: fraction of correct next-character predictions.
///
/// Propagates `len` tokens from a random corpus offset, comparing
/// `proj.predict(charge)` to the actual next character.
/// Returns 0.0 if the corpus is too short.
///
/// # Example
///
/// ```no_run
/// use instnct_core::*;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let init = InitConfig::phi(256);
/// let mut rng = StdRng::seed_from_u64(42);
/// let mut net = build_network(&init, &mut rng);
/// let proj = Int8Projection::new(init.phi_dim, 27, &mut rng);
/// let sdr = SdrTable::new(27, 256, 158, 20, &mut rng).unwrap();
/// let corpus = vec![0u8; 200];
/// let mut eval_rng = StdRng::seed_from_u64(99);
///
/// let acc = eval_accuracy(
///     &mut net, &proj, &corpus, 100, &mut eval_rng,
///     &sdr, &init.propagation, init.output_start(), init.neuron_count,
/// );
/// assert!(acc >= 0.0 && acc <= 1.0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn eval_accuracy(
    net: &mut Network,
    proj: &Int8Projection,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    propagation: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
) -> f64 {
    if corpus.len() <= len {
        return 0.0;
    }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), propagation)
            .unwrap();
        if proj.predict(&net.charge()[output_start..neuron_count]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

/// Smooth fitness: mean cosine similarity to bigram distribution.
///
/// For each token, computes softmax over the raw projection scores and
/// measures cosine similarity to the true bigram distribution
/// `P(next | current)`. This gives continuous feedback — a mutation
/// that shifts the output toward the correct distribution is rewarded
/// even if the argmax doesn't flip.
///
/// Proven +2.6pp peak over argmax accuracy (A/B test 2026-04-06).
#[allow(clippy::too_many_arguments)]
pub fn eval_smooth(
    net: &mut Network,
    proj: &Int8Projection,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    propagation: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    bigram: &[Vec<f64>],
) -> f64 {
    if corpus.len() <= len {
        return 0.0;
    }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut total_cos = 0.0f64;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), propagation)
            .unwrap();
        let scores = proj.raw_scores(&net.charge()[output_start..neuron_count]);
        let probs = softmax(&scores);
        let target = &bigram[seg[i] as usize];
        total_cos += cosine_similarity(&probs, target);
    }
    total_cos / len as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_network, InitConfig, SdrTable};
    use rand::SeedableRng;

    fn setup() -> (InitConfig, Network, Int8Projection, SdrTable, StdRng) {
        let init = InitConfig::phi(256);
        let mut rng = StdRng::seed_from_u64(42);
        let net = build_network(&init, &mut rng);
        let proj = Int8Projection::new(init.phi_dim, 27, &mut StdRng::seed_from_u64(200));
        let sdr = SdrTable::new(27, init.neuron_count, init.input_end(), 20, &mut StdRng::seed_from_u64(100)).unwrap();
        let eval_rng = StdRng::seed_from_u64(1000);
        (init, net, proj, sdr, eval_rng)
    }

    #[test]
    fn accuracy_returns_zero_for_short_corpus() {
        let (init, mut net, proj, sdr, mut eval_rng) = setup();
        let corpus = vec![0u8; 5];
        let acc = eval_accuracy(&mut net, &proj, &corpus, 100, &mut eval_rng, &sdr,
            &init.propagation, init.output_start(), init.neuron_count);
        assert_eq!(acc, 0.0);
    }

    #[test]
    fn accuracy_deterministic_with_same_rng() {
        let (init, mut net, proj, sdr, _) = setup();
        let corpus: Vec<u8> = (0..500).map(|i| (i % 27) as u8).collect();

        let mut rng1 = StdRng::seed_from_u64(999);
        let a1 = eval_accuracy(&mut net, &proj, &corpus, 100, &mut rng1, &sdr,
            &init.propagation, init.output_start(), init.neuron_count);

        let mut rng2 = StdRng::seed_from_u64(999);
        let a2 = eval_accuracy(&mut net, &proj, &corpus, 100, &mut rng2, &sdr,
            &init.propagation, init.output_start(), init.neuron_count);

        assert_eq!(a1, a2);
    }

    #[test]
    fn accuracy_in_valid_range() {
        let (init, mut net, proj, sdr, mut eval_rng) = setup();
        let corpus: Vec<u8> = (0..500).map(|i| (i % 27) as u8).collect();
        let acc = eval_accuracy(&mut net, &proj, &corpus, 100, &mut eval_rng, &sdr,
            &init.propagation, init.output_start(), init.neuron_count);
        assert!((0.0..=1.0).contains(&acc), "acc={acc} out of range");
    }

    #[test]
    fn smooth_returns_zero_for_short_corpus() {
        let (init, mut net, proj, sdr, mut eval_rng) = setup();
        let corpus = vec![0u8; 5];
        let bigram = crate::build_bigram_table(&corpus, 27);
        let s = eval_smooth(&mut net, &proj, &corpus, 100, &mut eval_rng, &sdr,
            &init.propagation, init.output_start(), init.neuron_count, &bigram);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn smooth_non_negative() {
        let (init, mut net, proj, sdr, mut eval_rng) = setup();
        let corpus: Vec<u8> = (0..500).map(|i| (i % 27) as u8).collect();
        let bigram = crate::build_bigram_table(&corpus, 27);
        let s = eval_smooth(&mut net, &proj, &corpus, 100, &mut eval_rng, &sdr,
            &init.propagation, init.output_start(), init.neuron_count, &bigram);
        assert!(s >= 0.0, "smooth fitness = {s}, expected >= 0");
    }
}
