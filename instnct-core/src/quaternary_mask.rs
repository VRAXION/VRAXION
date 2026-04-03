//! Quaternary upper-triangle connection mask.
//!
//! Encoding per neuron pair (i,j) where i < j:
//! - 0 = no connection
//! - 1 = i -> j (forward)
//! - 2 = j -> i (backward)
//! - 3 = bidirectional (both)
//!
//! Storage: flat `Vec<u8>` of length H*(H-1)/2.

/// Quaternary connection mask for a network of H neurons.
#[derive(Clone)]
pub struct QuaternaryMask {
    pub h: usize,
    pub data: Vec<u8>,
}

impl QuaternaryMask {
    /// Create a new empty mask for H neurons.
    pub fn new(h: usize) -> Self {
        let n_pairs = h * (h - 1) / 2;
        Self {
            h,
            data: vec![0u8; n_pairs],
        }
    }

    /// Number of neuron pairs.
    #[inline]
    pub fn n_pairs(&self) -> usize {
        self.data.len()
    }

    /// Linear index for pair (i, j). Canonicalizes so i < j.
    #[inline]
    pub fn pair_index(&self, i: usize, j: usize) -> usize {
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        lo * self.h - lo * (lo + 1) / 2 + hi - lo - 1
    }

    /// Get the quaternary value for a pair, from the caller's perspective.
    /// If i > j, values 1 and 2 are swapped (reversed perspective).
    #[inline]
    pub fn get_pair(&self, i: usize, j: usize) -> u8 {
        let swapped = i > j;
        let idx = self.pair_index(i, j);
        let val = self.data[idx];
        if swapped && (val == 1 || val == 2) {
            3 - val
        } else {
            val
        }
    }

    /// Set the quaternary value for a pair, from the caller's perspective.
    #[inline]
    pub fn set_pair(&mut self, i: usize, j: usize, mut val: u8) {
        let swapped = i > j;
        if swapped && (val == 1 || val == 2) {
            val = 3 - val;
        }
        let idx = self.pair_index(i, j);
        self.data[idx] = val;
    }

    /// Total directed edges. val 1/2 -> 1 edge, val 3 -> 2 edges.
    pub fn count_edges(&self) -> usize {
        self.data.iter().map(|&v| match v {
            1 | 2 => 1,
            3 => 2,
            _ => 0,
        }).sum()
    }

    /// Count of bidirectional pairs (val == 3).
    pub fn count_bidir(&self) -> usize {
        self.data.iter().filter(|&&v| v == 3).count()
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    /// Extract directed edge list as (sources, targets) vectors.
    /// This is the sparse cache for the forward pass.
    pub fn to_directed_edges(&self) -> (Vec<usize>, Vec<usize>) {
        let mut sources = Vec::new();
        let mut targets = Vec::new();

        let mut idx = 0;
        for i in 0..self.h {
            for j in (i + 1)..self.h {
                let val = self.data[idx];
                if val == 1 || val == 3 {
                    sources.push(i);
                    targets.push(j);
                }
                if val == 2 || val == 3 {
                    sources.push(j);
                    targets.push(i);
                }
                idx += 1;
            }
        }
        (sources, targets)
    }

    /// Create from a dense H x H boolean adjacency matrix (row-major).
    pub fn from_bool_matrix(h: usize, matrix: &[bool]) -> Self {
        let mut mask = Self::new(h);
        let mut idx = 0;
        for i in 0..h {
            for j in (i + 1)..h {
                let fwd = matrix[i * h + j]; // i -> j
                let bwd = matrix[j * h + i]; // j -> i
                mask.data[idx] = match (fwd, bwd) {
                    (false, false) => 0,
                    (true, false) => 1,
                    (false, true) => 2,
                    (true, true) => 3,
                };
                idx += 1;
            }
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let mask = QuaternaryMask::new(4);
        assert_eq!(mask.n_pairs(), 6); // 4*3/2
        assert_eq!(mask.count_edges(), 0);
        assert_eq!(mask.count_bidir(), 0);
    }

    #[test]
    fn test_set_get_forward() {
        let mut mask = QuaternaryMask::new(4);
        mask.set_pair(0, 2, 1); // 0 -> 2
        assert_eq!(mask.get_pair(0, 2), 1); // forward from 0's perspective
        assert_eq!(mask.get_pair(2, 0), 2); // backward from 2's perspective
        assert_eq!(mask.count_edges(), 1);
    }

    #[test]
    fn test_bidir() {
        let mut mask = QuaternaryMask::new(4);
        mask.set_pair(1, 3, 3); // bidir
        assert_eq!(mask.get_pair(1, 3), 3);
        assert_eq!(mask.get_pair(3, 1), 3);
        assert_eq!(mask.count_edges(), 2);
        assert_eq!(mask.count_bidir(), 1);
    }

    #[test]
    fn test_directed_edges() {
        let mut mask = QuaternaryMask::new(4);
        mask.set_pair(0, 1, 1); // 0 -> 1
        mask.set_pair(2, 3, 3); // bidir 2 <-> 3
        let (src, tgt) = mask.to_directed_edges();
        assert_eq!(src.len(), 3); // 0->1, 2->3, 3->2
        assert!(src.contains(&0) && tgt.contains(&1));
    }

    #[test]
    fn test_roundtrip_bool_matrix() {
        let h = 8;
        let mut matrix = vec![false; h * h];
        matrix[0 * h + 3] = true; // 0 -> 3
        matrix[3 * h + 0] = true; // 3 -> 0 (bidir with above)
        matrix[2 * h + 5] = true; // 2 -> 5
        let mask = QuaternaryMask::from_bool_matrix(h, &matrix);
        assert_eq!(mask.get_pair(0, 3), 3); // bidir
        assert_eq!(mask.get_pair(2, 5), 1); // forward
        assert_eq!(mask.count_edges(), 3); // 0->3, 3->0, 2->5
    }

    #[test]
    fn test_memory_savings() {
        let h = 256;
        let mask = QuaternaryMask::new(h);
        let bool_bytes = h * h; // 1 byte per bool
        let quat_bytes = mask.memory_bytes();
        assert!(quat_bytes < bool_bytes / 2 + 100); // ~50% savings
    }
}
