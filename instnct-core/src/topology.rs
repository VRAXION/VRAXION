//! # Network Topology — Sparse Edge List
//!
//! Stores the directed graph as a compact edge list rather than a dense
//! adjacency matrix. For sparse networks (1-5% density), this uses
//! **2-12x less memory** at H=1024+.
//!
//! ## Representation
//!
//! Two parallel structures, each optimized for a different access pattern:
//! - `edge_set: HashSet<(u16,u16)>` — O(1) existence checks for mutations
//! - `sources/targets: Vec<u16>` — cache-friendly slices for the propagation hot path
//!
//! Public edge iteration is exposed via `iter_edges()`, which reconstructs
//! `DirectedEdge` values on the fly without allocating. The `edges()` helper
//! remains available for compatibility and collects those values into a `Vec`.
//!
//! ## Forward Pass Integration
//!
//! The propagation scatter-add loop consumes `edge_endpoints()` directly —
//! paired contiguous `u16` slices. No conversion needed at runtime.

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
/// Two parallel structures:
/// - `edge_set: HashSet` — O(1) existence check (`has_edge`)
/// - `sources/targets: Vec<u16>` — hot path scatter-add (propagation)
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
    edge_set: HashSet<(u16, u16)>, // O(1) lookup for mutations (~16 bytes/edge)
    neuron_count: usize,
    sources: Vec<u16>, // hot path cache (2 bytes/edge, compact for scatter-add)
    targets: Vec<u16>, // hot path cache (2 bytes/edge, compact for scatter-add)
}

impl ConnectionGraph {
    /// Create an empty graph with no edges.
    pub fn new(neuron_count: usize) -> Self {
        Self {
            edge_set: HashSet::new(),
            neuron_count,
            sources: Vec::new(),
            targets: Vec::new(),
        }
    }

    /// Create a graph with pre-allocated capacity for `expected_edges`.
    pub fn with_capacity(neuron_count: usize, expected_edges: usize) -> Self {
        Self {
            edge_set: HashSet::with_capacity(expected_edges),
            neuron_count,
            sources: Vec::with_capacity(expected_edges),
            targets: Vec::with_capacity(expected_edges),
        }
    }

    // ----- Queries -----

    /// Total number of neurons in the graph.
    #[inline]
    pub fn neuron_count(&self) -> usize {
        self.neuron_count
    }

    /// Total number of directed edges.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.sources.len()
    }

    /// Check whether a specific directed edge exists. O(1) HashSet lookup.
    #[inline]
    pub fn has_edge(&self, source: u16, target: u16) -> bool {
        self.edge_set.contains(&(source, target))
    }

    /// Source neuron indices as a contiguous slice (for scatter-add).
    #[inline]
    #[cfg(test)]
    pub(crate) fn sources(&self) -> &[u16] {
        &self.sources
    }

    /// Target neuron indices as a contiguous slice (for scatter-add).
    #[inline]
    #[cfg(test)]
    pub(crate) fn targets(&self) -> &[u16] {
        &self.targets
    }

    /// Paired source/target slices used by the propagation core.
    #[inline]
    pub(crate) fn edge_endpoints(&self) -> (&[u16], &[u16]) {
        (&self.sources, &self.targets)
    }

    /// Public edge endpoints for benchmarks. Same as `edge_endpoints`.
    #[cfg(feature = "benchmarks")]
    #[doc(hidden)]
    #[inline]
    pub fn edge_endpoints_pub(&self) -> (&[u16], &[u16]) {
        (&self.sources, &self.targets)
    }

    /// Iterate over directed edges without allocating.
    #[inline]
    pub fn iter_edges(&self) -> impl Iterator<Item = DirectedEdge> + '_ {
        self.sources
            .iter()
            .zip(self.targets.iter())
            .map(|(&source, &target)| DirectedEdge { source, target })
    }

    /// Reconstruct the edge list into a newly allocated `Vec`.
    #[inline]
    pub fn edges(&self) -> Vec<DirectedEdge> {
        self.iter_edges().collect()
    }

    /// Count bidirectional pairs (both A->B and B->A exist). O(edges).
    pub fn bidirectional_pair_count(&self) -> usize {
        self.sources
            .iter()
            .zip(self.targets.iter())
            .filter(|(&s, &t)| s < t && self.edge_set.contains(&(t, s))) // check REVERSE exists
            .count()
    }

    /// Approximate reserved memory in bytes.
    ///
    /// This is a capacity-based estimate, not an exact heap-footprint measurement.
    /// The `HashSet` term uses a coarse per-entry heuristic because std does not
    /// expose bucket-level allocation size.
    pub fn memory_bytes(&self) -> usize {
        let hash_set = self.edge_set.capacity() * 16; // HashSet: coarse reserved-bytes heuristic
        let endpoint_cache =
            (self.sources.capacity() + self.targets.capacity()) * size_of::<u16>();
        hash_set + endpoint_cache
    }

    // ----- Mutations -----

    /// Add a directed edge. Returns `true` if the edge was new.
    /// Rejects self-connections, duplicates, and out-of-range endpoints. O(1).
    pub fn add_edge(&mut self, source: u16, target: u16) -> bool {
        if source == target
            || source as usize >= self.neuron_count
            || target as usize >= self.neuron_count
        {
            return false;
        }
        if self.edge_set.insert((source, target)) {
            self.sources.push(source);
            self.targets.push(target);
            self.debug_assert_invariants();
            true
        } else {
            false
        }
    }

    /// Remove a directed edge by value. Returns `true` if found and removed.
    /// O(n) linear scan to find the edge index, then O(1) swap-remove.
    pub fn remove_edge(&mut self, source: u16, target: u16) -> bool {
        if self.edge_set.remove(&(source, target)) {
            let pos = self
                .sources
                .iter()
                .zip(self.targets.iter())
                .position(|(&src, &tgt)| src == source && tgt == target);
            debug_assert!(
                pos.is_some(),
                "edge_set and endpoint caches diverged after removing edge"
            );
            if let Some(pos) = pos {
                self.sources.swap_remove(pos);
                self.targets.swap_remove(pos);
            }
            self.debug_assert_invariants();
            true
        } else {
            false
        }
    }

    /// Remove an edge by its index in the edge list. O(1) via swap-remove.
    pub fn remove_edge_at(&mut self, index: usize) -> Option<DirectedEdge> {
        if index >= self.sources.len() {
            return None;
        }
        let edge = DirectedEdge {
            source: self.sources[index],
            target: self.targets[index],
        };
        self.sources.swap_remove(index);
        self.targets.swap_remove(index);
        self.edge_set.remove(&(edge.source, edge.target));
        self.debug_assert_invariants();
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
        let pos = self
            .sources
            .iter()
            .zip(self.targets.iter())
            .position(|(&src, &tgt)| src == source && tgt == target);
        debug_assert!(
            pos.is_some(),
            "edge_set and endpoint caches diverged after reversing edge"
        );
        if let Some(pos) = pos {
            self.sources[pos] = target;
            self.targets[pos] = source;
        }
        self.debug_assert_invariants();
        true
    }

    /// Sort edges by target index for cache-friendly scatter-add writes.
    /// Does not affect correctness (addition is commutative), only performance.
    #[cfg(feature = "benchmarks")]
    #[doc(hidden)]
    pub fn sort_edges_by_target(&mut self) {
        let mut indices: Vec<usize> = (0..self.sources.len()).collect();
        indices.sort_by_key(|&i| self.targets[i]);
        let old_sources = self.sources.clone();
        let old_targets = self.targets.clone();
        for (new_pos, &old_pos) in indices.iter().enumerate() {
            self.sources[new_pos] = old_sources[old_pos];
            self.targets[new_pos] = old_targets[old_pos];
        }
    }

    /// Construct from pre-validated edge vectors (used by genome deserialization).
    ///
    /// The caller must guarantee: `sources.len() == targets.len()`, all endpoints
    /// `< neuron_count`, no self-loops. These are NOT checked here.
    pub(crate) fn from_validated_edges(
        neuron_count: usize,
        sources: Vec<u16>,
        targets: Vec<u16>,
    ) -> Self {
        let mut edge_set = HashSet::with_capacity(sources.len());
        for (&s, &t) in sources.iter().zip(&targets) {
            edge_set.insert((s, t));
        }
        Self {
            edge_set,
            neuron_count,
            sources,
            targets,
        }
    }

    /// Construct from a list of `(source, target)` pairs.
    pub fn from_pairs(neuron_count: usize, pairs: &[(u16, u16)]) -> Self {
        let mut graph = Self::with_capacity(neuron_count, pairs.len());
        for &(s, t) in pairs {
            graph.add_edge(s, t);
        }
        graph
    }

    #[cfg(test)]
    pub(crate) fn from_raw_parts_for_tests(
        neuron_count: usize,
        _edges: Vec<DirectedEdge>, // kept for API compat, ignored
        sources: Vec<u16>,
        targets: Vec<u16>,
    ) -> Self {
        let edge_set = sources
            .iter()
            .zip(targets.iter())
            .map(|(&s, &t)| (s, t))
            .collect();
        Self {
            edge_set,
            neuron_count,
            sources,
            targets,
        }
    }

    #[inline]
    fn debug_assert_invariants(&self) {
        debug_assert_eq!(self.sources.len(), self.targets.len());
        debug_assert_eq!(self.edge_set.len(), self.sources.len());
        for (&source, &target) in self.sources.iter().zip(self.targets.iter()) {
            debug_assert!(self.edge_set.contains(&(source, target)));
        }
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
        for i in 0..10 {
            graph.add_edge(i, i + 1);
        }
        let (src, tgt) = (graph.sources(), graph.targets());
        assert_eq!(src.len(), 10);
        assert_eq!(tgt.len(), 10);
        assert_eq!(src[0], 0);
        assert_eq!(tgt[0], 1);
    }

    #[test]
    fn iter_edges_matches_endpoint_cache() {
        let mut graph = ConnectionGraph::new(64);
        graph.add_edge(3, 8);
        graph.add_edge(8, 3);
        graph.add_edge(5, 13);

        let iterated: Vec<_> = graph.iter_edges().collect();
        let expected: Vec<_> = graph
            .sources()
            .iter()
            .zip(graph.targets().iter())
            .map(|(&source, &target)| DirectedEdge {
                source,
                target,
            })
            .collect();

        assert_eq!(iterated, expected);
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
        // Verify sources/targets match edge_set
        for (&s, &t) in graph.sources().iter().zip(graph.targets().iter()) {
            assert!(graph.has_edge(s, t));
        }
    }

    #[test]
    fn out_of_range_endpoints_rejected() {
        let mut graph = ConnectionGraph::new(4);
        assert!(!graph.add_edge(99, 1)); // source out of range
        assert!(!graph.add_edge(1, 99)); // target out of range
        assert!(!graph.add_edge(4, 0)); // exactly at boundary
        assert!(graph.add_edge(3, 0)); // max valid index
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn edges_helper_matches_iter_edges() {
        let mut graph = ConnectionGraph::new(256);
        graph.add_edge(10, 42);
        graph.add_edge(42, 10);
        graph.add_edge(5, 99);

        assert_eq!(graph.edges(), graph.iter_edges().collect::<Vec<_>>());
    }

    #[test]
    fn memory_scales_with_edges_not_neurons() {
        let graph = ConnectionGraph::with_capacity(4096, 10000);
        let bytes = graph.memory_bytes();
        assert!(bytes > 150_000);
        assert!(bytes < 300_000);
    }

    #[test]
    fn new_graph_memory_starts_small() {
        let graph = ConnectionGraph::new(4096);
        assert_eq!(graph.memory_bytes(), 0);
    }
}
