//! Fitness primitives: softmax, cosine similarity, and smooth fitness helpers.
//!
//! These functions convert raw integer scores from [`Int8Projection::raw_scores`]
//! into probability distributions and fitness signals for evolution.

/// Compute softmax over integer scores, returning a probability distribution.
///
/// Returns a `Vec<f64>` that sums to 1.0 (or uniform if all scores overflow).
///
/// # Example
///
/// ```
/// let probs = instnct_core::softmax(&[100, 50, 0]);
/// assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
/// assert!(probs[0] > probs[1]); // higher score → higher prob
/// ```
pub fn softmax(scores: &[i32]) -> Vec<f64> {
    if scores.is_empty() {
        return vec![];
    }
    let max = scores.iter().copied().max().unwrap_or(0) as f64;
    let mut out: Vec<f64> = scores.iter().map(|&s| ((s as f64) - max).exp()).collect();
    let sum: f64 = out.iter().sum();
    if sum < 1e-30 {
        let u = 1.0 / out.len() as f64;
        out.fill(u);
    } else {
        for v in out.iter_mut() {
            *v /= sum;
        }
    }
    out
}

/// Cosine similarity between two equal-length slices.
///
/// Returns 0.0 if either vector has zero norm.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
///
/// # Example
///
/// ```
/// let v = vec![1.0, 0.0, 0.0];
/// assert!((instnct_core::cosine_similarity(&v, &v) - 1.0).abs() < 1e-10);
/// ```
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine_similarity: length mismatch");
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (ai, bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

/// Cosine of the softmax distribution to a one-hot target vector.
///
/// Useful for deterministic tasks (e.g., addition) where the target is a
/// single correct class. Equivalent to `cosine_similarity(softmax(scores), one_hot(target))`
/// but avoids allocating the one-hot vector.
///
/// # Example
///
/// ```
/// // Score heavily favors class 1 → high cosine to one-hot(1)
/// let c = instnct_core::cosine_to_onehot(&[0, 100, 0], 1);
/// assert!(c > 0.9);
/// ```
pub fn cosine_to_onehot(scores: &[i32], target: usize) -> f64 {
    let probs = softmax(scores);
    if probs.is_empty() || target >= probs.len() {
        return 0.0;
    }
    let norm_sq: f64 = probs.iter().map(|p| p * p).sum();
    if norm_sq < 1e-30 {
        return 0.0;
    }
    probs[target] / norm_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let probs = softmax(&[10, 20, 30, 5, -10]);
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "softmax sums to {sum}, expected 1.0"
        );
    }

    #[test]
    fn softmax_uniform_on_equal_scores() {
        let probs = softmax(&[5, 5, 5, 5]);
        for &p in &probs {
            assert!(
                (p - 0.25).abs() < 1e-10,
                "expected 0.25 for equal scores, got {p}"
            );
        }
    }

    #[test]
    fn softmax_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn softmax_single() {
        let probs = softmax(&[42]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn softmax_large_scores_no_overflow() {
        let probs = softmax(&[1000, 999, 998]);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(probs[0] > probs[1]);
    }

    #[test]
    fn cosine_self_is_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn cosine_opposite_is_negative() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn cosine_length_mismatch_panics() {
        cosine_similarity(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn onehot_correct_target_high() {
        // Strongly favors class 2
        let c = cosine_to_onehot(&[0, 0, 100, 0], 2);
        assert!(c > 0.9, "expected > 0.9, got {c}");
    }

    #[test]
    fn onehot_wrong_target_low() {
        // Strongly favors class 2, but target is class 0
        let c = cosine_to_onehot(&[0, 0, 100, 0], 0);
        assert!(c < 0.1, "expected < 0.1, got {c}");
    }

    #[test]
    fn onehot_uniform_scores() {
        let c = cosine_to_onehot(&[5, 5, 5, 5], 0);
        // Uniform → all probs equal → cosine to one-hot = 1/(sqrt(N)*1) = 0.5 for N=4
        assert!(c > 0.4 && c < 0.6, "expected ~0.5 for uniform, got {c}");
    }

    #[test]
    fn onehot_target_out_of_range() {
        assert_eq!(cosine_to_onehot(&[1, 2, 3], 99), 0.0);
    }

    #[test]
    fn onehot_empty_scores() {
        assert_eq!(cosine_to_onehot(&[], 0), 0.0);
    }
}
