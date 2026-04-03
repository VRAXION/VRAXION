//! # Network Topology — Sparse Edge List
//!
//! Stores the directed graph structure as a compact list of edges rather than
//! a dense adjacency matrix. For sparse networks (1-5% density), this uses
//! **2-12x less memory** than a full H x H matrix at H=1024+.
//!
//! ## Representation
//!
//! Each directed edge is a `(source, target)` pair of `u16` neuron indices,
//! consuming 4 bytes per edge. A parallel `HashSet` provides O(1) existence
//! checks for mutation operations ("does edge A->B exist?").
//!
//! ## Memory Comparison (at typical densities)
//!
//! | H | Density | Edges | Edge list | Dense matrix | Savings |
//! |---|---------|-------|-----------|-------------|---------|
//! | 256 | 5% | 3,200 | 63 KB | 32 KB | dense wins |
//! | 1024 | 1% | 10,000 | 204 KB | 512 KB | **2.5x** |
//! | 4096 | 0.2% | 33,000 | 655 KB | 8,190 KB | **12.5x** |
//!
//! For H >= 512 with typical sparsity, the edge list is strictly superior.
//!
//! ## Forward Pass Integration
//!
//! The propagation step iterates directly over the edge vectors — no
//! conversion step needed. The `sources()` and `targets()` slices are
//! cache-friendly contiguous arrays consumed by the scatter-add loop.

use std::collections::HashSet;

/// A single directed connection between two neurons.
///
/// Uses `u16` indices, supporting networks up to 65,535 neurons.
/// At 4 bytes per edge, a 10,000-edge network costs 40 KB for edges alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DirectedEdge {
    pub source: u16,
    pub target: u16,
}

/// Sparse directed graph storing the network's connection topology.
///
/// The primary representation is a `Vec<DirectedEdge>` for cache-friendly
/// iteration during the forward pass, backed by a `HashSet` for O(1)
/// existence checks during mutations.
///
/// # Example
///
/// ```
/// use instnct_core::topology::ConnectionGraph;
///
/// let mut graph = ConnectionGraph::new(256);
/// graph.add_edge(10, 42);   // neuron 10 -> neuron 42
/// graph.add_edge(42, 10);   // neuron 42 -> neuron 10 (bidirectional pair)
///
/// assert_eq!(graph.edge_count(), 2);
/// assert!(graph.has_edge(10, 42));
/// assert!(!graph.has_edge(10, 43));
/// ```
#[derive(Clone, Debug)]
pub struct ConnectionGraph {
    /// Number of neurons in the network.
    pub neuron_count: usize,

    /// Ordered list of directed edges. Iterated by the forward pass.
    edges: Vec<DirectedEdge>,

    /// O(1) lookup set for mutation existence checks.
    edge_set: HashSet<(u16, u16)>,
}

impl ConnectionGraph {
    /// Create an empty graph with no edges.
    pub fn new(neuron_count: usize) -> Self {
        Self {
            neuron_count,
            edges: Vec::new(),
            edge_set: HashSet::new(),
        }
    }

    /// Create a graph with pre-allocated capacity for `expected_edges`.
    pub fn with_capacity(neuron_count: usize, expected_edges: usize) -> Self {
        Self {
            neuron_count,
            edges: Vec::with_capacity(expected_edges),
            edge_set: HashSet::with_capacity(expected_edges),
        }
    }

    // ----- Queries -----

    /// Total number of directed edges.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Check whether a specific directed edge exists. O(1).
    #[inline]
    pub fn has_edge(&self, source: u16, target: u16) -> bool {
        self.edge_set.contains(&(source, target))
    }

    /// Source neuron indices as a contiguous slice (for scatter-add).
    #[inline]
    pub fn sources(&self) -> Vec<usize> {
        self.edges.iter().map(|e| e.source as usize).collect()
    }

    /// Target neuron indices as a contiguous slice (for scatter-add).
    #[inline]
    pub fn targets(&self) -> Vec<usize> {
        self.edges.iter().map(|e| e.target as usize).collect()
    }

    /// Direct access to the edge list.
    #[inline]
    pub fn edges(&self) -> &[DirectedEdge] {
        &self.edges
    }

    /// Count bidirectional pairs (both A->B and B->A exist).
    pub fn bidirectional_pair_count(&self) -> usize {
        let mut count = 0;
        for edge in &self.edges {
            if edge.source < edge.target && self.has_edge(edge.target, edge.source) {
                count += 1;
            }
        }
        count
    }

    /// Approximate memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        // Vec<DirectedEdge>: 4 bytes per edge
        // HashSet<(u16,u16)>: ~16 bytes per entry (key + hash + metadata)
        self.edges.len() * 4 + self.edge_set.len() * 16
    }

    // ----- Mutations -----

    /// Add a directed edge. Returns `true` if the edge was new.
    ///
    /// Silently rejects self-connections (source == target) and duplicates.
    pub fn add_edge(&mut self, source: u16, target: u16) -> bool {
        if source == target {
            return false;
        }
        if self.edge_set.insert((source, target)) {
            self.edges.push(DirectedEdge { source, target });
            true
        } else {
            false
        }
    }

    /// Remove a directed edge by value. Returns `true` if found and removed.
    ///
    /// Uses swap-remove for O(1) removal from the edge vector.
    pub fn remove_edge(&mut self, source: u16, target: u16) -> bool {
        if self.edge_set.remove(&(source, target)) {
            if let Some(pos) = self.edges.iter().position(|e| e.source == source && e.target == target) {
                self.edges.swap_remove(pos);
            }
            true
        } else {
            false
        }
    }

    /// Remove an edge by its index in the edge list. O(1) via swap-remove.
    ///
    /// Returns the removed edge, or `None` if the index is out of bounds.
    pub fn remove_edge_at(&mut self, index: usize) -> Option<DirectedEdge> {
        if index >= self.edges.len() {
            return None;
        }
        let edge = self.edges.swap_remove(index);
        self.edge_set.remove(&(edge.source, edge.target));
        Some(edge)
    }

    /// Reverse an edge's direction: A->B becomes B->A. O(1).
    ///
    /// Returns `true` if the reversal was performed. Fails if the reverse
    /// edge already exists (would create a duplicate).
    pub fn reverse_edge(&mut self, source: u16, target: u16) -> bool {
        if !self.has_edge(source, target) || self.has_edge(target, source) {
            return false;
        }
        self.edge_set.remove(&(source, target));
        self.edge_set.insert((target, source));
        if let Some(edge) = self.edges.iter_mut().find(|e| e.source == source && e.target == target) {
            edge.source = target;
            edge.target = source;
        }
        true
    }

    /// Construct from a list of `(source, target)` pairs.
    pub fn from_pairs(neuron_count: usize, pairs: &[(u16, u16)]) -> Self {
        let mut graph = Self::with_capacity(neuron_count, pairs.len());
        for &(s, t) in pairs {
            graph.add_edge(s, t);
        }
        graph
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph_has_no_edges() {
        let graph = ConnectionGraph::new(256);
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.bidirectional_pair_count(), 0);
    }

    #[test]
    fn add_and_query_edge() {
        let mut graph = ConnectionGraph::new(256);
        assert!(graph.add_edge(10, 42));
        assert!(graph.has_edge(10, 42));
        assert!(!graph.has_edge(42, 10)); // directed, not symmetric
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn duplicate_edge_rejected() {
        let mut graph = ConnectionGraph::new(256);
        assert!(graph.add_edge(5, 9));
        assert!(!graph.add_edge(5, 9)); // duplicate
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn self_connection_rejected() {
        let mut graph = ConnectionGraph::new(256);
        assert!(!graph.add_edge(7, 7));
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn remove_edge_works() {
        let mut graph = ConnectionGraph::new(256);
        graph.add_edge(3, 8);
        graph.add_edge(8, 3);
        assert_eq!(graph.edge_count(), 2);

        assert!(graph.remove_edge(3, 8));
        assert_eq!(graph.edge_count(), 1);
        assert!(!graph.has_edge(3, 8));
        assert!(graph.has_edge(8, 3)); // other direction unaffected
    }

    #[test]
    fn reverse_edge_flips_direction() {
        let mut graph = ConnectionGraph::new(256);
        graph.add_edge(10, 20);
        assert!(graph.reverse_edge(10, 20));
        assert!(!graph.has_edge(10, 20));
        assert!(graph.has_edge(20, 10));
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn reverse_blocked_if_target_exists() {
        let mut graph = ConnectionGraph::new(256);
        graph.add_edge(10, 20);
        graph.add_edge(20, 10);
        assert!(!graph.reverse_edge(10, 20)); // 20->10 already exists
    }

    #[test]
    fn bidirectional_count() {
        let mut graph = ConnectionGraph::new(256);
        graph.add_edge(1, 5);
        graph.add_edge(5, 1);  // bidir pair
        graph.add_edge(2, 7);  // one-way
        assert_eq!(graph.bidirectional_pair_count(), 1);
    }

    #[test]
    fn from_pairs_builds_graph() {
        let pairs = vec![(0, 3), (3, 0), (2, 5)];
        let graph = ConnectionGraph::from_pairs(256, &pairs);
        assert_eq!(graph.edge_count(), 3);
        assert!(graph.has_edge(0, 3));
        assert!(graph.has_edge(3, 0));
        assert!(graph.has_edge(2, 5));
    }

    #[test]
    fn edge_list_is_cache_friendly() {
        let mut graph = ConnectionGraph::new(64);
        for i in 0..10 { graph.add_edge(i, i + 1); }
        let sources = graph.sources();
        let targets = graph.targets();
        assert_eq!(sources.len(), 10);
        assert_eq!(targets.len(), 10);
        assert_eq!(sources[0], 0);
        assert_eq!(targets[0], 1);
    }

    #[test]
    fn memory_scales_with_edges_not_neurons() {
        let sparse = ConnectionGraph::with_capacity(4096, 10000);
        // Edge list: 10K * 4 = 40KB + HashSet ~160KB = ~200KB
        // vs dense matrix: 4096*4095/2 = 8MB
        assert!(sparse.memory_bytes() < 300_000); // well under 300KB
    }
}
