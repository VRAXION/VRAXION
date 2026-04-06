//! Rollback-safe int8 output projection for spiking networks.
//!
//! Converts output-zone charge values into class predictions via a
//! learnable integer weight matrix. The [`WeightBackup`] type enforces
//! correct mutation rollback at the type level — the pattern that
//! previously caused a +7.1pp bug when done manually.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Learnable int8 projection: `charge_slice @ weights -> argmax`.
///
/// Weights are stored row-major: `input_dim` rows of `output_classes` columns.
/// Each weight is an `i8` in `[-127, 127]` — sufficient for gradient-free
/// evolution (Navigable Infinity principle).
///
/// # Example
///
/// ```
/// use instnct_core::Int8Projection;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let mut proj = Int8Projection::new(158, 27, &mut rng);
///
/// let charge = vec![0u32; 158]; // output zone charge
/// let predicted_class = proj.predict(&charge);
/// assert!(predicted_class < 27);
///
/// let backup = proj.mutate_one(&mut rng);
/// // ... evaluate fitness ...
/// proj.rollback(backup); // undo if rejected
/// ```
#[derive(Clone, Debug)]
pub struct Int8Projection {
    weights: Vec<i8>,      // input_dim × output_classes, row-major
    input_dim: usize,      // output zone neuron count (e.g. 158)
    output_classes: usize,  // number of prediction classes (e.g. 27)
}

/// Private wire DTO for serialization. Validated on deserialize.
#[derive(Serialize, Deserialize)]
struct ProjectionDisk {
    weights: Vec<i8>,      // input_dim × output_classes, row-major
    input_dim: usize,      // output zone neuron count
    output_classes: usize, // prediction class count
}

impl Serialize for Int8Projection {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ProjectionDisk {
            weights: self.weights.clone(),
            input_dim: self.input_dim,
            output_classes: self.output_classes,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Int8Projection {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let disk = ProjectionDisk::deserialize(deserializer)?;
        let expected_len = disk.input_dim * disk.output_classes;
        if disk.weights.len() != expected_len {
            return Err(serde::de::Error::custom(format!(
                "weights length {} != input_dim {} * output_classes {} = {}",
                disk.weights.len(),
                disk.input_dim,
                disk.output_classes,
                expected_len
            )));
        }
        Ok(Self {
            weights: disk.weights,
            input_dim: disk.input_dim,
            output_classes: disk.output_classes,
        })
    }
}

/// Backup token returned by [`Int8Projection::mutate_one`].
///
/// Pass to [`Int8Projection::rollback`] to undo the mutation.
/// Drop without calling rollback to accept the mutation.
#[derive(Clone, Debug)]
pub struct WeightBackup {
    index: usize,
    old_value: i8,
}

impl Int8Projection {
    /// Create a projection with random weights in `[-127, 127]`.
    pub fn new(input_dim: usize, output_classes: usize, rng: &mut impl Rng) -> Self {
        let weights = (0..input_dim * output_classes)
            .map(|_| rng.gen_range(-127..=127i8))
            .collect();
        Self {
            weights,
            input_dim,
            output_classes,
        }
    }

    /// Predict class from output-zone charge values.
    ///
    /// Computes `scores[c] = sum_i(charge[i] * weights[i][c])` for each class,
    /// returns the argmax. Skips neurons with zero charge.
    ///
    /// # Panics
    ///
    /// Panics if `charge_slice.len() != input_dim`.
    pub fn predict(&self, charge_slice: &[u32]) -> usize {
        assert_eq!(
            charge_slice.len(),
            self.input_dim,
            "charge_slice length {} != input_dim {}",
            charge_slice.len(),
            self.input_dim
        );
        let mut scores = vec![0i32; self.output_classes];
        for (neuron_idx, &charge) in charge_slice.iter().enumerate() {
            if charge == 0 {
                continue;
            }
            let charge_value = charge as i32;
            let row_start = neuron_idx * self.output_classes;
            let row = &self.weights[row_start..row_start + self.output_classes];
            for (score, &weight) in scores.iter_mut().zip(row.iter()) {
                *score += charge_value * weight as i32;
            }
        }
        scores
            .iter()
            .enumerate()
            .max_by_key(|&(_, &s)| s)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Mutate one random weight. Returns a [`WeightBackup`] for rollback.
    ///
    /// The new weight is sampled uniformly from `[-127, 127]`.
    pub fn mutate_one(&mut self, rng: &mut impl Rng) -> WeightBackup {
        let index = rng.gen_range(0..self.weights.len());
        let old_value = self.weights[index];
        self.weights[index] = rng.gen_range(-127..=127i8);
        WeightBackup { index, old_value }
    }

    /// Undo a mutation using the backup token.
    pub fn rollback(&mut self, backup: WeightBackup) {
        self.weights[backup.index] = backup.old_value;
    }

    /// Total number of weights (`input_dim × output_classes`).
    #[inline]
    pub fn weight_count(&self) -> usize {
        self.weights.len()
    }

    /// Number of input neurons (output zone width).
    #[inline]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Number of prediction classes.
    #[inline]
    pub fn output_classes(&self) -> usize {
        self.output_classes
    }

    /// Count weights with non-zero value.
    #[inline]
    pub fn nonzero_count(&self) -> usize {
        self.weights.iter().filter(|&&w| w != 0).count()
    }

    /// Zero weights with absolute value at or below `threshold`. Returns count zeroed.
    ///
    /// Used during crystallize phases to prune weak projection weights.
    pub fn sparsify(&mut self, threshold: u8) -> usize {
        let mut count = 0usize;
        for w in self.weights.iter_mut() {
            if w.unsigned_abs() <= threshold {
                *w = 0;
                count += 1;
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn new_creates_correct_dimensions() {
        let mut rng = StdRng::seed_from_u64(42);
        let proj = Int8Projection::new(158, 27, &mut rng);
        assert_eq!(proj.weight_count(), 158 * 27);
        assert_eq!(proj.input_dim(), 158);
        assert_eq!(proj.output_classes(), 27);
    }

    #[test]
    fn predict_returns_valid_class() {
        let mut rng = StdRng::seed_from_u64(42);
        let proj = Int8Projection::new(16, 5, &mut rng);
        let charge = vec![3u32; 16];
        let class = proj.predict(&charge);
        assert!(class < 5, "predicted class {class} >= 5");
    }

    #[test]
    fn predict_matches_manual_computation() {
        // 3 input neurons, 2 classes
        let proj = Int8Projection {
            weights: vec![
                10, -5, // neuron 0: class 0 = 10, class 1 = -5
                3, 7,   // neuron 1: class 0 = 3, class 1 = 7
                -2, 4,  // neuron 2: class 0 = -2, class 1 = 4
            ],
            input_dim: 3,
            output_classes: 2,
        };
        let charge = vec![2u32, 0, 3]; // neuron 1 silent
        // scores[0] = 2*10 + 3*(-2) = 20 - 6 = 14
        // scores[1] = 2*(-5) + 3*4 = -10 + 12 = 2
        assert_eq!(proj.predict(&charge), 0); // class 0 wins with 14
    }

    #[test]
    fn mutate_rollback_exact_restore() {
        let mut rng = StdRng::seed_from_u64(77);
        let mut proj = Int8Projection::new(32, 10, &mut rng);
        let original_weights = proj.weights.clone();

        for _ in 0..100 {
            let backup = proj.mutate_one(&mut rng);
            proj.rollback(backup);
            assert_eq!(proj.weights, original_weights, "rollback leaked weight drift");
        }
    }

    #[test]
    fn mutate_accept_changes_one_weight() {
        let mut rng = StdRng::seed_from_u64(99);
        let mut proj = Int8Projection::new(32, 10, &mut rng);
        let before = proj.weights.clone();

        let _backup = proj.mutate_one(&mut rng); // accept: don't rollback

        let diffs: usize = proj
            .weights
            .iter()
            .zip(before.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(diffs <= 1, "mutate_one changed {diffs} weights, expected 0 or 1");
    }

    #[test]
    fn deserialize_rejects_dimension_mismatch() {
        // Craft a payload where weights.len() != input_dim * output_classes
        let bad_bytes = bincode::serialize(&ProjectionDisk {
            weights: vec![1, 2, 3],
            input_dim: 4,
            output_classes: 2,
        })
        .unwrap();
        let result: Result<Int8Projection, _> = bincode::deserialize(&bad_bytes);
        assert!(result.is_err(), "should reject mismatched dimensions");
    }

    #[test]
    fn predict_supports_more_than_256_classes() {
        let mut rng = StdRng::seed_from_u64(42);
        let proj = Int8Projection::new(4, 300, &mut rng);
        let charge = vec![1u32; 4];
        let class = proj.predict(&charge);
        assert!(class < 300, "predicted class {class} >= 300");
        // Specifically: result should NOT be truncated to u8 range
        // (old bug: 299 as u8 == 43)
    }

    #[test]
    fn sparsify_zeroes_small_weights() {
        let mut proj = Int8Projection {
            weights: vec![0, 1, -1, 2, -2, 127, -127, 0],
            input_dim: 4,
            output_classes: 2,
        };
        let zeroed = proj.sparsify(1);
        assert_eq!(zeroed, 4); // 0, 1, -1, 0
        assert_eq!(proj.weights, vec![0, 0, 0, 2, -2, 127, -127, 0]);
    }

    #[test]
    fn sparsify_zero_threshold_only_existing_zeros() {
        let mut proj = Int8Projection {
            weights: vec![0, 1, -1, 0],
            input_dim: 2,
            output_classes: 2,
        };
        let zeroed = proj.sparsify(0);
        assert_eq!(zeroed, 2);
        assert_eq!(proj.weights, vec![0, 1, -1, 0]);
    }

    #[test]
    fn nonzero_count_correct() {
        let proj = Int8Projection {
            weights: vec![0, 1, -1, 0, 5, 0],
            input_dim: 3,
            output_classes: 2,
        };
        assert_eq!(proj.nonzero_count(), 3);
    }
}
