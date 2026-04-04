//! # Network Topology — Sparse Edge List
//!
//! Stores the directed graph as a compact edge list rather than a dense
//! adjacency matrix. For sparse networks (1-5% density), this uses
//! **2-12x less memory** at H=1024+.
//!
//! ## Representation
//!
//! Each directed edge is a `(source, target)` pair of `u16` neuron indices (4 bytes).
//! A parallel `HashSet` provides O(1) existence checks for mutations.
//! Separate `sources`/`targets` `Vec<usize>` caches feed the propagation hot path.
//!
//! ## Forward Pass Integration
//!
//! The propagation scatter-add loop consumes `sources()` and `targets()` directly —
//! contiguous, cache-friendly `usize` slices. No conversion needed at runtime.

use std::collections::HashSet;

/// A single directed connection between two neurons.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DirectedEdge {
    /// Source neuron index.
    pub source: u16,
    /// Target neuron index.
    pub target: u16,
}

/// Sparse directed graph storing the network's connection topology.
///
/// Three parallel structures, each optimized for a different access pattern:
/// - `edges: Vec<DirectedEdge>` — canonical list, iterated by mutations
/// - `edge_set: HashSet` — O(1) existence check (`has_edge`)
/// - `sources/targets: Vec<usize>` — hot path scatter-add (propagation)
///
/// # Example
///
/// ```
/// use instnct_core::ConnectionGraph;
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
    edges: Vec<DirectedEdge>,      // canonical edge list (4 bytes/edge)
    edge_set: HashSet<(u16, u16)>, // O(1) lookup for mutations (~16 bytes/edge)
    neuron_count: usize,
    sources: Vec<usize>,           // hot path cache (8 bytes/edge, usize for scatter-add)
    targets: Vec<usize>,           // hot path cache (8 bytes/edge, usize for scatter-add)
}

impl ConnectionGraph {
    /// Create an empty graph with no edges.
    pub fn new(neuron_count: usize) -> Self {
        Self {
            edges: Vec::new(),
            edge_set: HashSet::new(),
            neuron_count,
            sources: Vec::new(),
            targets: Vec::new(),
        }
    }

    /// Create a graph with pre-allocated capacity for `expected_edges`.
    pub fn with_capacity(neuron_count: usize, expected_edges: usize) -> Self {
        Self {
            edges: Vec::with_capacity(expected_edges),
            edge_set: HashSet::with_capacity(expected_edges),
            neuron_count,
            sources: Vec::with_capacity(expected_edges),
            targets: Vec::with_capacity(expected_edges),
        }
    }

    // ----- Queries -----

    /// Total number of neurons in the graph.
    #[inline]
    pub fn neuron_count(&self) -> usize { self.neuron_count }

    /// Total number of directed edges.
    #[inline]
    pub fn edge_count(&self) -> usize { self.edges.len() }

    /// Check whether a specific directed edge exists. O(1) HashSet lookup.
    #[inline]
    pub fn has_edge(&self, source: u16, target: u16) -> bool {
        self.edge_set.contains(&(source, target))
    }

    /// Source neuron indices as a contiguous slice (for scatter-add).
    #[inline]
    #[cfg(test)]
    pub(crate) fn sources(&self) -> &[usize] { &self.sources }

    /// Target neuron indices as a contiguous slice (for scatter-add).
    #[inline]
    #[cfg(test)]
    pub(crate) fn targets(&self) -> &[usize] { &self.targets }

    /// Paired source/target slices used by the propagation core.
    #[inline]
    pub(crate) fn edge_endpoints(&self) -> (&[usize], &[usize]) {
        (&self.sources, &self.targets)
    }

    /// Direct access to the edge list.
    #[inline]
    pub fn edges(&self) -> &[DirectedEdge] { &self.edges }

    /// Count bidirectional pairs (both A->B and B->A exist). O(edges).
    pub fn bidirectional_pair_count(&self) -> usize {
        self.edges.iter()
            .filter(|e| e.source < e.target && self.has_edge(e.target, e.source))
            .count()
    }

    /// Approximate memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        let edge_list = self.edges.len() * 4;           // Vec<DirectedEdge>: 4 bytes/edge
        let hash_set = self.edge_set.len() * 16;        // HashSet: ~16 bytes/entry
        let endpoint_cache = (self.sources.len() + self.targets.len()) * size_of::<usize>(); // 8 bytes/edge each
        edge_list + hash_set + endpoint_cache
    }

    // ----- Mutations -----

    /// Add a directed edge. Returns `true` if the edge was new.
    /// Rejects self-connections, duplicates, and out-of-range endpoints. O(1).
    pub fn add_edge(&mut self, source: u16, target: u16) -> bool {
        if source == target
            || source as usize >= self.neuron_count
            || target as usize >= self.neuron_count
        { return false; }
        if self.edge_set.insert((source, target)) {
            self.edges.push(DirectedEdge { source, target });
            self.sources.push(source as usize);
            self.targets.push(target as usize);
            true
        } else {
            false
        }
    }

    /// Remove a directed edge by value. Returns `true` if found and removed.
    /// O(n) linear scan to find the edge index, then O(1) swap-remove.
    pub fn remove_edge(&mut self, source: u16, target: u16) -> bool {
        if self.edge_set.remove(&(source, target)) {
            if let Some(pos) = self.edges.iter()
                .position(|e| e.source == source && e.target == target)
            {
                self.edges.swap_remove(pos);
                self.sources.swap_remove(pos);
                self.targets.swap_remove(pos);
            }
            true
        } else {
            false
        }
    }

    /// Remove an edge by its index in the edge list. O(1) via swap-remove.
    pub fn remove_edge_at(&mut self, index: usize) -> Option<DirectedEdge> {
        if index >= self.edges.len() { return None; }
        let edge = self.edges.swap_remove(index);
        self.sources.swap_remove(index);
        self.targets.swap_remove(index);
        self.edge_set.remove(&(edge.source, edge.target));
        Some(edge)
    }

    /// Reverse an edge's direction: A->B becomes B->A.
    /// O(n) linear scan to find the edge index, then O(1) swap.
    /// Fails if the reverse edge already exists (would create a duplicate).
    pub fn reverse_edge(&mut self, source: u16, target: u16) -> bool {
        if !self.has_edge(source, target) || self.has_edge(target, source) {
            return false;
        }
        self.edge_set.remove(&(source, target));
        self.edge_set.insert((target, source));
        if let Some(pos) = self.edges.iter()
            .position(|e| e.source == source && e.target == target)
        {
            self.edges[pos] = DirectedEdge { source: target, target: source };
            self.sources[pos] = target as usize;
            self.targets[pos] = source as usize;
        }
        true
    }

    /// Construct from a list of `(source, target)` pairs.
    pub fn from_pairs(neuron_count: usize, pairs: &[(u16, u16)]) -> Self {
        let mut graph = Self::with_capacity(neuron_count, pairs.len());
        for &(s, t) in pairs { graph.add_edge(s, t); }
        graph
    }

    #[cfg(test)]
    pub(crate) fn from_raw_parts_for_tests(
        neuron_count: usize,
        edges: Vec<DirectedEdge>,
        sources: Vec<usize>,
        targets: Vec<usize>,
    ) -> Self {
        let edge_set = edges.iter().map(|e| (e.source, e.target)).collect();
        Self { edges, edge_set, neuron_count, sources, targets }
    }
}

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
        assert!(!graph.add_edge(5, 9));
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
        assert!(graph.has_edge(8, 3));
    }

    #[test]
    fn reverse_edge_flips_direction() {
        let mut graph = ConnectionGraph::new(256);
        graph.add_edge(10, 20);
        assert!(graph.reverse_edge(10, 20));
        assert!(!graph.has_edge(10, 20));
        assert!(graph.has_edge(20, 10));
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.sources(), &[20]);
        assert_eq!(graph.targets(), &[10]);
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
        graph.add_edge(5, 1);
        graph.add_edge(2, 7); // one-way only
        assert_eq!(graph.bidirectional_pair_count(), 1);
    }

    #[test]
    fn from_pairs_builds_graph() {
        let graph = ConnectionGraph::from_pairs(256, &[(0, 3), (3, 0), (2, 5)]);
        assert_eq!(graph.edge_count(), 3);
        assert!(graph.has_edge(0, 3));
        assert!(graph.has_edge(3, 0));
        assert!(graph.has_edge(2, 5));
    }

    #[test]
    fn edge_list_is_cache_friendly() {
        let mut graph = ConnectionGraph::new(64);
        for i in 0..10 { graph.add_edge(i, i + 1); }
        let (src, tgt) = (graph.sources(), graph.targets());
        assert_eq!(src.len(), 10);
        assert_eq!(tgt.len(), 10);
        assert_eq!(src[0], 0);
        assert_eq!(tgt[0], 1);
    }

    #[test]
    fn remove_edge_keeps_endpoint_cache_in_sync() {
        let mut graph = ConnectionGraph::new(8);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        assert!(graph.remove_edge(1, 2));
        assert_eq!(graph.edge_count(), graph.sources().len());
        assert_eq!(graph.edge_count(), graph.targets().len());
        assert!(!graph.has_edge(1, 2));
        assert!(graph.edges().iter()
            .zip(graph.sources().iter().zip(graph.targets().iter()))
            .all(|(edge, (&s, &t))| edge.source as usize == s && edge.target as usize == t));
    }

    #[test]
    fn out_of_range_endpoints_rejected() {
        let mut graph = ConnectionGraph::new(4);
        assert!(!graph.add_edge(99, 1)); // source out of range
        assert!(!graph.add_edge(1, 99)); // target out of range
        assert!(!graph.add_edge(4, 0));  // exactly at boundary
        assert!(graph.add_edge(3, 0));   // max valid index
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn memory_scales_with_edges_not_neurons() {
        let graph = ConnectionGraph::with_capacity(4096, 10000);
        assert!(graph.memory_bytes() < 300_000); // well under 300KB vs 8MB dense
    }
}
