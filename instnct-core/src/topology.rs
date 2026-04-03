//! # Network Topology — Quaternary Connection Mask
//!
//! Stores the directed graph structure of a spiking network using a compact
//! upper-triangle encoding. For a network of H neurons, the full H x H
//! boolean adjacency matrix requires H^2 bytes. This module stores the same
//! information in H*(H-1)/2 bytes — a **50% memory reduction** — by encoding
//! each neuron pair (i, j) where i < j as a single quaternary value:
//!
//! | Value | Meaning | Directed edges |
//! |-------|---------|----------------|
//! | 0 | No connection | — |
//! | 1 | Forward | i -> j |
//! | 2 | Backward | j -> i |
//! | 3 | Bidirectional | i -> j and j -> i |
//!
//! ## Design Rationale
//!
//! The encoding enables **O(1) atomic direction reversal**: flipping an edge
//! from i->j to j->i is a single byte write (1 <-> 2), whereas the boolean
//! matrix requires two writes and cannot represent the swap atomically during
//! mutation undo. Bidirectional pairs (value 3) are detected without scanning
//! the matrix transpose.
//!
//! ## Indexing
//!
//! Pairs are stored in standard upper-triangle order. The linear index for
//! pair (i, j) where i < j is:
//!
//! ```text
//! index = i * H - i * (i + 1) / 2 + j - i - 1
//! ```
//!
//! When accessing with i > j, the module transparently swaps the perspective:
//! value 1 (forward from i's view) becomes value 2 (backward from j's view).

/// The four possible connection states between a neuron pair.
pub mod connection_state {
    /// No connection between the pair.
    pub const NONE: u8 = 0;
    /// Directed edge from the lower-indexed to the higher-indexed neuron.
    pub const FORWARD: u8 = 1;
    /// Directed edge from the higher-indexed to the lower-indexed neuron.
    pub const BACKWARD: u8 = 2;
    /// Bidirectional: edges in both directions.
    pub const BIDIRECTIONAL: u8 = 3;
}

use connection_state::*;

/// Compact directed-graph representation for a spiking network.
///
/// Stores one quaternary value (0-3) per neuron pair in a flat byte array.
/// Total storage: `H * (H - 1) / 2` bytes, where H is the neuron count.
///
/// # Example
///
/// ```
/// use instnct_core::topology::ConnectionMask;
///
/// let mut mask = ConnectionMask::new(256);
/// mask.set_connection(10, 42, 1); // directed edge: neuron 10 -> neuron 42
/// assert_eq!(mask.get_connection(10, 42), 1); // forward from 10's perspective
/// assert_eq!(mask.get_connection(42, 10), 2); // backward from 42's perspective
/// ```
#[derive(Clone, Debug)]
pub struct ConnectionMask {
    /// Number of neurons in the network.
    pub neuron_count: usize,
    /// Flat upper-triangle storage. Length = neuron_count * (neuron_count - 1) / 2.
    pub pairs: Vec<u8>,
}

impl ConnectionMask {
    /// Create an empty connection mask for `neuron_count` neurons (no edges).
    pub fn new(neuron_count: usize) -> Self {
        let pair_count = neuron_count * (neuron_count - 1) / 2;
        Self {
            neuron_count,
            pairs: vec![NONE; pair_count],
        }
    }

    /// Total number of storable neuron pairs.
    #[inline]
    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    /// Compute the linear index into [`pairs`] for the neuron pair (i, j).
    ///
    /// Automatically canonicalizes so the lower index comes first.
    /// Panics in debug mode if i == j (self-connections are undefined).
    #[inline]
    pub fn pair_index(&self, i: usize, j: usize) -> usize {
        debug_assert_ne!(i, j, "self-connections are not stored");
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        lo * self.neuron_count - lo * (lo + 1) / 2 + hi - lo - 1
    }

    /// Read the connection state between neurons `i` and `j`.
    ///
    /// The returned value is from neuron `i`'s perspective:
    /// - [`FORWARD`] (1) means i -> j
    /// - [`BACKWARD`] (2) means j -> i
    /// - [`BIDIRECTIONAL`] (3) is symmetric
    ///
    /// When `i > j`, the stored value is transparently perspective-swapped.
    #[inline]
    pub fn get_connection(&self, i: usize, j: usize) -> u8 {
        let is_reversed = i > j;
        let idx = self.pair_index(i, j);
        let stored = self.pairs[idx];
        if is_reversed && (stored == FORWARD || stored == BACKWARD) {
            3 - stored // swap perspective: 1 <-> 2
        } else {
            stored
        }
    }

    /// Set the connection state between neurons `i` and `j`.
    ///
    /// The value is interpreted from neuron `i`'s perspective:
    /// - [`FORWARD`] (1) creates edge i -> j
    /// - [`BACKWARD`] (2) creates edge j -> i
    /// - [`BIDIRECTIONAL`] (3) creates both
    #[inline]
    pub fn set_connection(&mut self, i: usize, j: usize, mut value: u8) {
        let is_reversed = i > j;
        if is_reversed && (value == FORWARD || value == BACKWARD) {
            value = 3 - value; // canonicalize to storage perspective
        }
        let idx = self.pair_index(i, j);
        self.pairs[idx] = value;
    }

    /// Count total directed edges in the network.
    ///
    /// Each unidirectional pair (value 1 or 2) contributes 1 edge.
    /// Each bidirectional pair (value 3) contributes 2 edges.
    pub fn directed_edge_count(&self) -> usize {
        self.pairs.iter().map(|&v| match v {
            FORWARD | BACKWARD => 1,
            BIDIRECTIONAL => 2,
            _ => 0,
        }).sum()
    }

    /// Count neuron pairs connected in both directions.
    pub fn bidirectional_pair_count(&self) -> usize {
        self.pairs.iter().filter(|&&v| v == BIDIRECTIONAL).count()
    }

    /// Storage footprint in bytes (the `pairs` vector only).
    pub fn storage_bytes(&self) -> usize {
        self.pairs.len()
    }

    /// Extract all directed edges as parallel source/target index vectors.
    ///
    /// Returns `(sources, targets)` where `sources[k] -> targets[k]` is the
    /// k-th directed edge. The ordering is deterministic: pairs are visited
    /// in ascending (i, j) order, with forward edges emitted before backward
    /// edges for bidirectional pairs.
    ///
    /// These vectors serve as the **sparse cache** consumed by the forward
    /// pass propagation step.
    pub fn to_directed_edges(&self) -> (Vec<usize>, Vec<usize>) {
        let estimated_capacity = self.directed_edge_count();
        let mut sources = Vec::with_capacity(estimated_capacity);
        let mut targets = Vec::with_capacity(estimated_capacity);

        let mut pair_idx = 0;
        for i in 0..self.neuron_count {
            for j in (i + 1)..self.neuron_count {
                let state = self.pairs[pair_idx];
                if state == FORWARD || state == BIDIRECTIONAL {
                    sources.push(i);
                    targets.push(j);
                }
                if state == BACKWARD || state == BIDIRECTIONAL {
                    sources.push(j);
                    targets.push(i);
                }
                pair_idx += 1;
            }
        }
        (sources, targets)
    }

    /// Construct a [`ConnectionMask`] from a dense H x H boolean adjacency
    /// matrix stored in row-major order.
    ///
    /// `adjacency[i * h + j] == true` means there is a directed edge i -> j.
    pub fn from_adjacency_matrix(neuron_count: usize, adjacency: &[bool]) -> Self {
        assert_eq!(
            adjacency.len(),
            neuron_count * neuron_count,
            "adjacency matrix must be H x H"
        );
        let mut mask = Self::new(neuron_count);
        let mut pair_idx = 0;
        for i in 0..neuron_count {
            for j in (i + 1)..neuron_count {
                let has_forward = adjacency[i * neuron_count + j];
                let has_backward = adjacency[j * neuron_count + i];
                mask.pairs[pair_idx] = match (has_forward, has_backward) {
                    (false, false) => NONE,
                    (true, false) => FORWARD,
                    (false, true) => BACKWARD,
                    (true, true) => BIDIRECTIONAL,
                };
                pair_idx += 1;
            }
        }
        mask
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mask_has_no_edges() {
        let mask = ConnectionMask::new(4);
        assert_eq!(mask.pair_count(), 6); // C(4,2) = 6
        assert_eq!(mask.directed_edge_count(), 0);
        assert_eq!(mask.bidirectional_pair_count(), 0);
    }

    #[test]
    fn forward_edge_perspective_is_consistent() {
        let mut mask = ConnectionMask::new(4);
        mask.set_connection(0, 2, FORWARD); // edge: 0 -> 2
        assert_eq!(mask.get_connection(0, 2), FORWARD);  // "I send to 2"
        assert_eq!(mask.get_connection(2, 0), BACKWARD);  // "0 sends to me"
        assert_eq!(mask.directed_edge_count(), 1);
    }

    #[test]
    fn bidirectional_pair_is_symmetric() {
        let mut mask = ConnectionMask::new(4);
        mask.set_connection(1, 3, BIDIRECTIONAL);
        assert_eq!(mask.get_connection(1, 3), BIDIRECTIONAL);
        assert_eq!(mask.get_connection(3, 1), BIDIRECTIONAL);
        assert_eq!(mask.directed_edge_count(), 2);
        assert_eq!(mask.bidirectional_pair_count(), 1);
    }

    #[test]
    fn directed_edge_extraction_is_complete() {
        let mut mask = ConnectionMask::new(4);
        mask.set_connection(0, 1, FORWARD);       // 0 -> 1
        mask.set_connection(2, 3, BIDIRECTIONAL);  // 2 <-> 3
        let (sources, targets) = mask.to_directed_edges();
        assert_eq!(sources.len(), 3); // 0->1, 2->3, 3->2
        let has_0_to_1 = sources.iter().zip(targets.iter())
            .any(|(&s, &t)| s == 0 && t == 1);
        assert!(has_0_to_1);
    }

    #[test]
    fn adjacency_matrix_roundtrip_preserves_edges() {
        let h = 8;
        let mut adjacency = vec![false; h * h];
        adjacency[0 * h + 3] = true; // 0 -> 3
        adjacency[3 * h + 0] = true; // 3 -> 0 (forms bidir pair with above)
        adjacency[2 * h + 5] = true; // 2 -> 5

        let mask = ConnectionMask::from_adjacency_matrix(h, &adjacency);
        assert_eq!(mask.get_connection(0, 3), BIDIRECTIONAL);
        assert_eq!(mask.get_connection(2, 5), FORWARD);
        assert_eq!(mask.directed_edge_count(), 3);
    }

    #[test]
    fn storage_is_half_of_dense_matrix() {
        let h = 256;
        let mask = ConnectionMask::new(h);
        let dense_bytes = h * h;
        let compact_bytes = mask.storage_bytes();
        assert!(compact_bytes < dense_bytes / 2 + 100);
    }
}
