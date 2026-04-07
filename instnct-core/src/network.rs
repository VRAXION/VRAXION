//! # Network — Self-Contained Spiking Network
//!
//! Bundles a [`ConnectionGraph`], per-neuron learned parameters (threshold,
//! channel, polarity), ephemeral tick state (activation, charge), and a
//! reusable scratch workspace into one coherent object.
//!
//! This is the primary user-facing type for the crate. Create a `Network`,
//! wire edges via [`graph_mut()`](Network::graph_mut), adjust parameters,
//! then call [`propagate()`](Network::propagate) to simulate one token.

use crate::parameters::{
    GLOBAL_PHASE_CHANNEL_COUNT, GLOBAL_PHASE_TICKS_PER_PERIOD, LIMIT_MAX_CHARGE,
};
use crate::propagation::{PropagationConfig, PropagationError, PropagationWorkspace};
use crate::topology::ConnectionGraph;
use rand::Rng;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

mod disk;

// ---- SpikeData ----

/// Per-neuron spike-decision data packed for cache locality.
/// 4 bytes per neuron (padded to power-of-2 stride for fast indexing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct SpikeData {
    /// Accumulated charge [0,15]. Reset to 0 on fire.
    pub charge: u8,
    /// Firing threshold, stored [0,15], effective = stored+1 → [1,16].
    pub threshold: u8,
    /// Phase gating channel [1,8].
    pub channel: u8,
    /// Padding to 4-byte alignment for efficient indexing.
    pub _pad: u8,
}

// ---- Error ----

/// Errors from [`Network`] operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum NetworkError {
    /// Propagation validation failed (slice lengths, value ranges, edge bounds).
    Propagation(PropagationError),
    /// File I/O error during genome save/load.
    Io(io::Error),
    /// Malformed genome file (invalid edges, parameters, or version).
    Genome(String),
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Propagation(e) => write!(f, "propagation error: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Genome(msg) => write!(f, "malformed genome: {msg}"),
        }
    }
}

impl Error for NetworkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Propagation(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::Genome(_) => None,
        }
    }
}

impl From<PropagationError> for NetworkError {
    fn from(e: PropagationError) -> Self {
        Self::Propagation(e)
    }
}

// ---- Snapshot ----

/// Frozen copy of a [`Network`]'s mutable state for rollback.
///
/// Cheaper than a full `Network` clone — does not allocate a workspace buffer.
/// Created by [`Network::save_state`], borrowed by [`Network::restore_state`].
/// A single snapshot can be restored from multiple times.
#[derive(Clone, Debug)]
pub struct NetworkSnapshot {
    graph: ConnectionGraph,
    spike: Vec<SpikeData>,
    polarity: Vec<i8>,
    activation: Vec<i8>,
    refractory: Vec<u8>,
}

impl NetworkSnapshot {
    /// Number of neurons in the snapshot.
    #[inline]
    pub fn neuron_count(&self) -> usize {
        self.graph.neuron_count()
    }

    /// Number of directed edges in the snapshot.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Immutable reference to the snapshot's connection graph.
    #[inline]
    pub fn graph(&self) -> &ConnectionGraph {
        &self.graph
    }

    /// Snapshot spike data (charge, threshold, channel per neuron).
    #[inline]
    pub fn spike_data(&self) -> &[SpikeData] {
        &self.spike
    }

    /// Snapshot polarity values.
    #[inline]
    pub fn polarity(&self) -> &[i8] {
        &self.polarity
    }

    /// Snapshot activation values.
    #[inline]
    pub fn activation(&self) -> &[i8] {
        &self.activation
    }

    /// Snapshot threshold value at `idx`.
    #[inline]
    pub fn threshold_at(&self, idx: usize) -> u8 {
        self.spike[idx].threshold
    }

    /// Snapshot channel value at `idx`.
    #[inline]
    pub fn channel_at(&self, idx: usize) -> u8 {
        self.spike[idx].channel
    }

    /// Snapshot charge value at `idx`.
    #[inline]
    pub fn charge_at(&self, idx: usize) -> u8 {
        self.spike[idx].charge
    }
}

// ---- Mutation Undo ----

/// Lightweight undo token for a single mutation. O(1) storage.
///
/// Instead of cloning the entire network (O(H) spike vec + O(E) graph),
/// each mutation records only what it changed. On reject, [`Network::apply_undo`]
/// reverses that one change. Since `eval_accuracy` calls [`Network::reset()`]
/// at the start, ephemeral state (charge, activation, refractory) does not
/// need to be saved — only topology and learned parameters.
///
/// This makes save = O(1) and restore = O(1), vs O(H+E) for full snapshots.
/// Since ~99% of mutations are rejected, this eliminates nearly all clone overhead.
#[derive(Clone, Debug)]
pub enum MutationUndo {
    /// An edge was added — undo by removing it.
    AddedEdge {
        /// Source neuron of the added edge.
        source: u16,
        /// Target neuron of the added edge.
        target: u16,
    },
    /// An edge was removed — undo by re-adding it.
    RemovedEdge {
        /// Source neuron of the removed edge.
        source: u16,
        /// Target neuron of the removed edge.
        target: u16,
    },
    /// An edge was reversed (A->B became B->A) — undo by reversing back.
    ReversedEdge {
        /// Source neuron after the reversal.
        new_source: u16,
        /// Target neuron after the reversal.
        new_target: u16,
    },
    /// An edge was rewired: old edge removed, new edge added.
    Rewired {
        /// Source of the original edge.
        old_source: u16,
        /// Target of the original edge.
        old_target: u16,
        /// Source of the replacement edge.
        new_source: u16,
        /// Target of the replacement edge.
        new_target: u16,
    },
    /// A loop of edges was added — undo by removing all of them.
    AddedLoop {
        /// The (source, target) pairs of all edges in the loop.
        edges: Vec<(u16, u16)>,
    },
    /// A neuron's threshold was changed.
    Theta {
        /// Neuron index whose threshold was changed.
        index: usize,
        /// Threshold value before the mutation.
        old_value: u8,
    },
    /// A neuron's channel was changed.
    Channel {
        /// Neuron index whose channel was changed.
        index: usize,
        /// Channel value before the mutation.
        old_value: u8,
    },
    /// A neuron's polarity was flipped.
    Polarity {
        /// Neuron index whose polarity was flipped.
        index: usize,
    },
    /// No mutation was applied (mutation returned false).
    Noop,
}

// ---- Network ----

/// Self-contained spiking network owning topology, parameters, and state.
///
/// # Example
///
/// ```
/// use instnct_core::{Network, PropagationConfig};
///
/// let mut net = Network::new(256);
/// net.graph_mut().add_edge(10, 42);
///
/// let input = vec![0i32; 256];
/// net.propagate(&input, &PropagationConfig::default()).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct Network {
    graph: ConnectionGraph,
    spike: Vec<SpikeData>, // per-neuron AoS: charge, threshold, channel (cache-hot)
    polarity: Vec<i8>,   // per-neuron, +1 excitatory, -1 inhibitory
    activation: Vec<i8>,   // ephemeral, set to polarity or 0 by spike stage; transient mid-tick
    refractory: Vec<u8>,  // per-neuron, 0 = ready, 1 = cooling (1-tick refractory after fire)
    workspace: PropagationWorkspace, // reusable scratch buffer
    csr_offsets: Vec<u32>, // CSR: per-neuron edge start index
    csr_targets: Vec<u16>, // CSR: compact target indices
    csr_dirty: bool,     // true when graph changed since last CSR rebuild
    active_neurons: Vec<u16>, // scratch: indices of neurons with charge > 0 (spike skip-inactive)
    dirty_set: Vec<u16>,   // sparse tick: indices of neurons that may have charge>0 or activation!=0
    dirty_member: Vec<bool>, // dirty_member[i] == true iff i is in dirty_set
}

impl Network {
    // ---- Construction ----

    /// Create a network with `neuron_count` neurons, no edges, and default parameters.
    ///
    /// Defaults: threshold=0 (effective 1), channel=1, polarity=+1 (all excitatory).
    /// Activation and charge start at zero.
    pub fn new(neuron_count: usize) -> Self {
        Self {
            graph: ConnectionGraph::new(neuron_count),
            spike: vec![SpikeData { charge: 0, threshold: 0, channel: 1, _pad: 0 }; neuron_count],
            polarity: vec![1i8; neuron_count],
            activation: vec![0i8; neuron_count],
            refractory: vec![0u8; neuron_count],
            workspace: PropagationWorkspace::new(neuron_count),
            csr_offsets: vec![0u32; neuron_count + 1],
            csr_targets: Vec::new(),
            csr_dirty: true,
            active_neurons: Vec::with_capacity(neuron_count),
            dirty_set: Vec::new(),
            dirty_member: vec![false; neuron_count],
        }
    }

    // ---- Propagation ----

    /// Rebuild CSR (Compressed Sparse Row) cache from the connection graph.
    /// Called automatically by [`propagate`](Self::propagate) when dirty.
    fn rebuild_csr(&mut self) {
        let neuron_count = self.graph.neuron_count();
        self.csr_offsets.clear();
        self.csr_offsets.reserve(neuron_count + 1);
        self.csr_targets.clear();
        self.csr_targets.reserve(self.graph.edge_count());

        // Count outgoing edges per neuron
        let mut outgoing_edge_counts = vec![0u32; neuron_count];
        for edge in self.graph.iter_edges() {
            outgoing_edge_counts[edge.source as usize] += 1;
        }
        // Build offsets (prefix sum)
        let mut offset = 0u32;
        for &count in &outgoing_edge_counts {
            self.csr_offsets.push(offset);
            offset += count;
        }
        self.csr_offsets.push(offset);

        // Fill targets
        self.csr_targets.resize(self.graph.edge_count(), 0);
        let mut write_pos = self.csr_offsets.clone();
        for edge in self.graph.iter_edges() {
            let src = edge.source as usize;
            let pos = write_pos[src] as usize;
            self.csr_targets[pos] = edge.target;
            write_pos[src] += 1;
        }

        self.csr_dirty = false;
    }

    /// Run one token's forward pass using CSR skip-inactive scatter-add.
    ///
    /// Validates input length and parameter ranges inline, then runs the
    /// optimized propagation loop that skips silent neurons.
    ///
    /// # Errors
    ///
    /// Returns [`NetworkError::Propagation`] wrapping the underlying
    /// [`PropagationError`] if any validation check fails.
    pub fn propagate(
        &mut self,
        input: &[i32],
        config: &PropagationConfig,
    ) -> Result<(), NetworkError> {
        // Validate input length and parameter ranges
        let neuron_count = self.graph.neuron_count();
        if input.len() != neuron_count {
            return Err(PropagationError::InputLengthMismatch {
                expected: neuron_count,
                actual: input.len(),
            }
            .into());
        }
        for i in 0..neuron_count {
            if self.spike[i].threshold > 15 {
                return Err(PropagationError::ThresholdOutOfRange {
                    index: i,
                    value: self.spike[i].threshold,
                }
                .into());
            }
            if !(1..=GLOBAL_PHASE_CHANNEL_COUNT as u8).contains(&self.spike[i].channel) {
                return Err(PropagationError::ChannelOutOfRange {
                    index: i,
                    value: self.spike[i].channel,
                }
                .into());
            }
            let p: i8 = self.polarity[i];
            if p != 1 && p != -1 {
                return Err(PropagationError::PolarityOutOfRange { index: i, value: p }.into());
            }
        }

        // Rebuild CSR if graph changed
        if self.csr_dirty {
            self.rebuild_csr();
        }

        // CSR skip-inactive propagation (the actual computation)
        let neuron_count = self.graph.neuron_count();
        let phase_base: [u8; GLOBAL_PHASE_TICKS_PER_PERIOD] = [7, 8, 10, 12, 13, 12, 10, 8]; // cosine gating curve (x10 scale, peak at ch offset)
        let incoming = &mut self.workspace.incoming_scratch_mut()[..neuron_count];

        for tick in 0..config.ticks_per_token {
            // 1. Charge decay — O(dirty) instead of O(H)
            if config.decay_interval_ticks > 0 && tick % config.decay_interval_ticks == 0 {
                for &idx in &self.dirty_set {
                    self.spike[idx as usize].charge = self.spike[idx as usize].charge.saturating_sub(1);
                }
            }

            // 2. Input injection — O(H) only during input ticks (unavoidable)
            //    BUT: add injected neurons to dirty set
            if tick < config.input_duration_ticks {
                for i in 0..neuron_count {
                    if input[i] != 0 {
                        self.activation[i] = self.activation[i].saturating_add(input[i] as i8);
                        if !self.dirty_member[i] {
                            self.dirty_member[i] = true;
                            self.dirty_set.push(i as u16);
                        }
                    }
                }
            }

            // 3. CSR scatter-add — iterate dirty_set (not 0..neuron_count)
            //    Add targets to dirty set during scatter
            let scatter_end = self.dirty_set.len(); // snapshot before new additions
            for di in 0..scatter_end {
                let idx = self.dirty_set[di] as usize;
                let act = self.activation[idx];
                if act == 0 { continue; }
                let start = self.csr_offsets[idx] as usize;
                let end = self.csr_offsets[idx + 1] as usize;
                for &target in &self.csr_targets[start..end] {
                    incoming[target as usize] += act as i16;
                    if !self.dirty_member[target as usize] {
                        self.dirty_member[target as usize] = true;
                        self.dirty_set.push(target);
                    }
                }
            }

            // 4. Charge accumulation — O(dirty)
            for &idx in &self.dirty_set {
                let i = idx as usize;
                let val = (self.spike[i].charge as i16) + incoming[i];
                self.spike[i].charge = val.clamp(0, LIMIT_MAX_CHARGE as i16) as u8;
            }

            // 5. Build active set — O(dirty)
            self.active_neurons.clear();
            for &idx in &self.dirty_set {
                if self.spike[idx as usize].charge > 0 {
                    self.active_neurons.push(idx);
                }
            }

            // 6. Clear activation — O(dirty)
            for &idx in &self.dirty_set {
                self.activation[idx as usize] = 0;
            }

            // 7. Spike decision — O(active), existing logic unchanged
            let phase_tick = tick % GLOBAL_PHASE_TICKS_PER_PERIOD;
            for i in 0..self.active_neurons.len() {
                let neuron_idx = self.active_neurons[i] as usize;
                // Refractory gate: neuron in cooldown cannot fire
                if config.use_refractory && self.refractory[neuron_idx] > 0 {
                    self.refractory[neuron_idx] -= 1;
                    continue;
                }
                let channel_idx = self.spike[neuron_idx].channel as usize;
                let phase_mult: u16 = if (1..=GLOBAL_PHASE_CHANNEL_COUNT).contains(&channel_idx) {
                    phase_base[(phase_tick + 9 - channel_idx) & 7] as u16
                } else {
                    10
                };
                let charge_x10: u16 = self.spike[neuron_idx].charge as u16 * 10;
                let threshold_x10: u16 = (self.spike[neuron_idx].threshold as u16 + 1) * phase_mult;
                if charge_x10 >= threshold_x10 {
                    self.activation[neuron_idx] = self.polarity[neuron_idx];
                    self.spike[neuron_idx].charge = 0;
                    if config.use_refractory {
                        self.refractory[neuron_idx] = 1;
                    }
                }
            }

            // 8. Sparse clear incoming — O(dirty)
            for &idx in &self.dirty_set {
                incoming[idx as usize] = 0;
            }

            // 9. Prune dirty set — remove quiescent neurons
            {
                let spike = &self.spike;
                let activation = &self.activation;
                let dirty_member = &mut self.dirty_member;
                self.dirty_set.retain(|&idx| {
                    let i = idx as usize;
                    if spike[i].charge > 0 || activation[i] != 0 {
                        true
                    } else {
                        dirty_member[i] = false;
                        false
                    }
                });
            }
        }
        Ok(())
    }

    /// Run one token's forward pass using sparse input injection.
    ///
    /// Identical to [`propagate()`](Self::propagate) except the input is provided
    /// as sparse `(indices, values)` pairs instead of a dense `&[i32]` slice.
    /// Only iterates the provided indices during input injection (step 2),
    /// making it O(k) instead of O(H) where k = number of active neurons.
    ///
    /// # Errors
    ///
    /// Returns [`NetworkError::Propagation`] if any validation check fails.
    pub fn propagate_sparse(
        &mut self,
        indices: &[u16],
        values: &[i8],
        config: &PropagationConfig,
    ) -> Result<(), NetworkError> {
        let neuron_count = self.graph.neuron_count();

        // Validate sparse input
        if indices.len() != values.len() {
            return Err(PropagationError::InputLengthMismatch {
                expected: indices.len(),
                actual: values.len(),
            }
            .into());
        }
        for &idx in indices {
            if idx as usize >= neuron_count {
                return Err(PropagationError::InputLengthMismatch {
                    expected: neuron_count,
                    actual: idx as usize + 1,
                }
                .into());
            }
        }

        // Validate neuron parameters (same as propagate)
        for i in 0..neuron_count {
            if self.spike[i].threshold > 15 {
                return Err(PropagationError::ThresholdOutOfRange {
                    index: i,
                    value: self.spike[i].threshold,
                }
                .into());
            }
            if !(1..=GLOBAL_PHASE_CHANNEL_COUNT as u8).contains(&self.spike[i].channel) {
                return Err(PropagationError::ChannelOutOfRange {
                    index: i,
                    value: self.spike[i].channel,
                }
                .into());
            }
            let p: i8 = self.polarity[i];
            if p != 1 && p != -1 {
                return Err(PropagationError::PolarityOutOfRange { index: i, value: p }.into());
            }
        }

        // Rebuild CSR if graph changed
        if self.csr_dirty {
            self.rebuild_csr();
        }

        // CSR skip-inactive propagation (identical to propagate except step 2)
        let neuron_count = self.graph.neuron_count();
        let phase_base: [u8; GLOBAL_PHASE_TICKS_PER_PERIOD] = [7, 8, 10, 12, 13, 12, 10, 8];
        let incoming = &mut self.workspace.incoming_scratch_mut()[..neuron_count];

        for tick in 0..config.ticks_per_token {
            // 1. Charge decay — O(dirty)
            if config.decay_interval_ticks > 0 && tick % config.decay_interval_ticks == 0 {
                for &idx in &self.dirty_set {
                    self.spike[idx as usize].charge = self.spike[idx as usize].charge.saturating_sub(1);
                }
            }

            // 2. Sparse input injection — O(k) instead of O(H)
            if tick < config.input_duration_ticks {
                for (&idx, &val) in indices.iter().zip(values.iter()) {
                    let i = idx as usize;
                    self.activation[i] = self.activation[i].saturating_add(val);
                    if !self.dirty_member[i] {
                        self.dirty_member[i] = true;
                        self.dirty_set.push(idx);
                    }
                }
            }

            // 3. CSR scatter-add — iterate dirty_set
            let scatter_end = self.dirty_set.len();
            for di in 0..scatter_end {
                let idx = self.dirty_set[di] as usize;
                let act = self.activation[idx];
                if act == 0 { continue; }
                let start = self.csr_offsets[idx] as usize;
                let end = self.csr_offsets[idx + 1] as usize;
                for &target in &self.csr_targets[start..end] {
                    incoming[target as usize] += act as i16;
                    if !self.dirty_member[target as usize] {
                        self.dirty_member[target as usize] = true;
                        self.dirty_set.push(target);
                    }
                }
            }

            // 4. Charge accumulation — O(dirty)
            for &idx in &self.dirty_set {
                let i = idx as usize;
                let val = (self.spike[i].charge as i16) + incoming[i];
                self.spike[i].charge = val.clamp(0, LIMIT_MAX_CHARGE as i16) as u8;
            }

            // 5. Build active set — O(dirty)
            self.active_neurons.clear();
            for &idx in &self.dirty_set {
                if self.spike[idx as usize].charge > 0 {
                    self.active_neurons.push(idx);
                }
            }

            // 6. Clear activation — O(dirty)
            for &idx in &self.dirty_set {
                self.activation[idx as usize] = 0;
            }

            // 7. Spike decision — O(active)
            let phase_tick = tick % GLOBAL_PHASE_TICKS_PER_PERIOD;
            for i in 0..self.active_neurons.len() {
                let neuron_idx = self.active_neurons[i] as usize;
                if config.use_refractory && self.refractory[neuron_idx] > 0 {
                    self.refractory[neuron_idx] -= 1;
                    continue;
                }
                let channel_idx = self.spike[neuron_idx].channel as usize;
                let phase_mult: u16 = if (1..=GLOBAL_PHASE_CHANNEL_COUNT).contains(&channel_idx) {
                    phase_base[(phase_tick + 9 - channel_idx) & 7] as u16
                } else {
                    10
                };
                let charge_x10: u16 = self.spike[neuron_idx].charge as u16 * 10;
                let threshold_x10: u16 = (self.spike[neuron_idx].threshold as u16 + 1) * phase_mult;
                if charge_x10 >= threshold_x10 {
                    self.activation[neuron_idx] = self.polarity[neuron_idx];
                    self.spike[neuron_idx].charge = 0;
                    if config.use_refractory {
                        self.refractory[neuron_idx] = 1;
                    }
                }
            }

            // 8. Sparse clear incoming — O(dirty)
            for &idx in &self.dirty_set {
                incoming[idx as usize] = 0;
            }

            // 9. Prune dirty set
            {
                let spike = &self.spike;
                let activation = &self.activation;
                let dirty_member = &mut self.dirty_member;
                self.dirty_set.retain(|&idx| {
                    let i = idx as usize;
                    if spike[i].charge > 0 || activation[i] != 0 {
                        true
                    } else {
                        dirty_member[i] = false;
                        false
                    }
                });
            }
        }
        Ok(())
    }

    // ---- State management ----

    /// Zero all activation and charge values. Does not touch topology or learned parameters.
    pub fn reset(&mut self) {
        self.activation.fill(0);
        for s in self.spike.iter_mut() {
            s.charge = 0;
        }
        self.refractory.fill(0);
        for &idx in &self.dirty_set {
            self.dirty_member[idx as usize] = false;
        }
        self.dirty_set.clear();
    }

    /// Snapshot the current topology, parameters, and state for later rollback.
    /// Does not copy the workspace (it is scratch space, zeroed every tick).
    #[must_use]
    pub fn save_state(&self) -> NetworkSnapshot {
        NetworkSnapshot {
            graph: self.graph.clone(),
            spike: self.spike.clone(),
            polarity: self.polarity.clone(),
            activation: self.activation.clone(),
            refractory: self.refractory.clone(),
        }
    }

    /// Restore topology, parameters, and state from a previous snapshot.
    ///
    /// # Panics
    ///
    /// Panics if `snapshot.neuron_count() != self.neuron_count()`.
    pub fn restore_state(&mut self, snapshot: &NetworkSnapshot) {
        assert_eq!(
            self.graph.neuron_count(),
            snapshot.graph.neuron_count(),
            "snapshot neuron_count mismatch: network={}, snapshot={}",
            self.graph.neuron_count(),
            snapshot.graph.neuron_count(),
        );
        self.graph.clone_from(&snapshot.graph);
        self.spike.copy_from_slice(&snapshot.spike);
        self.polarity.copy_from_slice(&snapshot.polarity);
        self.activation.copy_from_slice(&snapshot.activation);
        self.refractory.copy_from_slice(&snapshot.refractory);
        self.workspace
            .ensure_neuron_count(self.graph.neuron_count());
        self.csr_dirty = true;
        // Rebuild dirty set from restored state
        for &idx in &self.dirty_set {
            self.dirty_member[idx as usize] = false;
        }
        self.dirty_set.clear();
        for i in 0..self.graph.neuron_count() {
            if self.spike[i].charge > 0 || self.activation[i] != 0 {
                self.dirty_member[i] = true;
                self.dirty_set.push(i as u16);
            }
        }
    }

    // ---- Mutation undo ----

    /// Reverse a single mutation using a lightweight undo token.
    ///
    /// After calling this, call [`reset()`](Self::reset) to zero ephemeral state
    /// (the undo only restores topology and learned parameters).
    pub fn apply_undo(&mut self, undo: &MutationUndo) {
        match undo {
            MutationUndo::AddedEdge { source, target } => {
                self.graph.remove_edge(*source, *target);
                self.csr_dirty = true;
            }
            MutationUndo::RemovedEdge { source, target } => {
                self.graph.add_edge(*source, *target);
                self.csr_dirty = true;
            }
            MutationUndo::ReversedEdge {
                new_source,
                new_target,
            } => {
                // The mutation turned old→ into new_source→new_target,
                // so reverse it back.
                self.graph.reverse_edge(*new_source, *new_target);
                self.csr_dirty = true;
            }
            MutationUndo::Rewired {
                old_source,
                old_target,
                new_source,
                new_target,
            } => {
                // Remove the new edge, re-add the old edge
                self.graph.remove_edge(*new_source, *new_target);
                self.graph.add_edge(*old_source, *old_target);
                self.csr_dirty = true;
            }
            MutationUndo::AddedLoop { edges } => {
                for &(src, tgt) in edges {
                    self.graph.remove_edge(src, tgt);
                }
                self.csr_dirty = true;
            }
            MutationUndo::Theta { index, old_value } => {
                self.spike[*index].threshold = *old_value;
            }
            MutationUndo::Channel { index, old_value } => {
                self.spike[*index].channel = *old_value;
            }
            MutationUndo::Polarity { index } => {
                self.polarity[*index] = -self.polarity[*index];
            }
            MutationUndo::Noop => {}
        }
    }

    /// Add a random edge, returning an undo token.
    pub fn mutate_add_edge_undo(&mut self, rng: &mut impl Rng) -> (bool, MutationUndo) {
        let neuron_count = self.graph.neuron_count();
        if neuron_count < 2 {
            return (false, MutationUndo::Noop);
        }
        let source = rng.gen_range(0..neuron_count) as u16;
        let target = rng.gen_range(0..neuron_count) as u16;
        let added = self.graph.add_edge(source, target);
        if added {
            self.csr_dirty = true;
            (true, MutationUndo::AddedEdge { source, target })
        } else {
            (false, MutationUndo::Noop)
        }
    }

    /// Remove a random edge, returning an undo token.
    pub fn mutate_remove_edge_undo(&mut self, rng: &mut impl Rng) -> (bool, MutationUndo) {
        let edge_count = self.graph.edge_count();
        if edge_count == 0 {
            return (false, MutationUndo::Noop);
        }
        let index = rng.gen_range(0..edge_count);
        let edge = self.graph.remove_edge_at(index).unwrap();
        self.csr_dirty = true;
        (
            true,
            MutationUndo::RemovedEdge {
                source: edge.source,
                target: edge.target,
            },
        )
    }

    /// Rewire a random edge, returning an undo token.
    pub fn mutate_rewire_undo(&mut self, rng: &mut impl Rng) -> (bool, MutationUndo) {
        let edge_count = self.graph.edge_count();
        let neuron_count = self.graph.neuron_count();
        if edge_count == 0 || neuron_count < 2 {
            return (false, MutationUndo::Noop);
        }
        let index = rng.gen_range(0..edge_count);
        let old_edge = self.graph.remove_edge_at(index).unwrap();
        let new_target = rng.gen_range(0..neuron_count) as u16;
        if !self.graph.add_edge(old_edge.source, new_target) {
            self.graph.add_edge(old_edge.source, old_edge.target);
            return (false, MutationUndo::Noop);
        }
        self.csr_dirty = true;
        (
            true,
            MutationUndo::Rewired {
                old_source: old_edge.source,
                old_target: old_edge.target,
                new_source: old_edge.source,
                new_target,
            },
        )
    }

    /// Reverse a random edge, returning an undo token.
    pub fn mutate_reverse_undo(&mut self, rng: &mut impl Rng) -> (bool, MutationUndo) {
        let edge_count = self.graph.edge_count();
        if edge_count == 0 {
            return (false, MutationUndo::Noop);
        }
        let index = rng.gen_range(0..edge_count);
        let edges: Vec<_> = self.graph.iter_edges().collect();
        let edge = edges[index];
        let reversed = self.graph.reverse_edge(edge.source, edge.target);
        if reversed {
            self.csr_dirty = true;
            (
                true,
                MutationUndo::ReversedEdge {
                    new_source: edge.target,
                    new_target: edge.source,
                },
            )
        } else {
            (false, MutationUndo::Noop)
        }
    }

    /// Mirror a random edge (add its reverse), returning an undo token.
    pub fn mutate_mirror_undo(&mut self, rng: &mut impl Rng) -> (bool, MutationUndo) {
        let edge_count = self.graph.edge_count();
        if edge_count == 0 {
            return (false, MutationUndo::Noop);
        }
        let index = rng.gen_range(0..edge_count);
        let edges: Vec<_> = self.graph.iter_edges().collect();
        let edge = edges[index];
        let added = self.graph.add_edge(edge.target, edge.source);
        if added {
            self.csr_dirty = true;
            (
                true,
                MutationUndo::AddedEdge {
                    source: edge.target,
                    target: edge.source,
                },
            )
        } else {
            (false, MutationUndo::Noop)
        }
    }

    /// Enhance (add edge to high-degree neuron), returning an undo token.
    pub fn mutate_enhance_undo(&mut self, rng: &mut impl Rng) -> (bool, MutationUndo) {
        let neuron_count = self.graph.neuron_count();
        if neuron_count < 2 {
            return (false, MutationUndo::Noop);
        }
        if self.graph.edge_count() == 0 {
            return self.mutate_add_edge_undo(rng);
        }
        let mut in_degree = vec![0u32; neuron_count];
        for edge in self.graph.iter_edges() {
            in_degree[edge.target as usize] += 1;
        }
        let mut sorted_degrees = in_degree.clone();
        sorted_degrees.sort_unstable();
        let top_quartile_idx = neuron_count - neuron_count / 4;
        let threshold = sorted_degrees[top_quartile_idx.min(neuron_count - 1)];
        let high_indegree_neurons: Vec<u16> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, &d)| d >= threshold)
            .map(|(i, _)| i as u16)
            .collect();
        if high_indegree_neurons.is_empty() {
            return self.mutate_add_edge_undo(rng);
        }
        let target = high_indegree_neurons[rng.gen_range(0..high_indegree_neurons.len())];
        let source = rng.gen_range(0..neuron_count) as u16;
        let added = self.graph.add_edge(source, target);
        if added {
            self.csr_dirty = true;
            (true, MutationUndo::AddedEdge { source, target })
        } else {
            (false, MutationUndo::Noop)
        }
    }

    /// Add a loop of edges, returning an undo token.
    pub fn mutate_add_loop_undo(
        &mut self,
        rng: &mut impl Rng,
        len: usize,
    ) -> (bool, MutationUndo) {
        let neuron_count = self.graph.neuron_count();
        if neuron_count < 2 || len < 2 {
            return (false, MutationUndo::Noop);
        }
        let len = len.min(neuron_count);

        let mut nodes = Vec::with_capacity(len);
        for _ in 0..len * 10 {
            let n = rng.gen_range(0..neuron_count) as u16;
            if !nodes.contains(&n) {
                nodes.push(n);
            }
            if nodes.len() == len {
                break;
            }
        }
        if nodes.len() < len {
            return (false, MutationUndo::Noop);
        }

        for i in 0..len {
            let src = nodes[i];
            let tgt = nodes[(i + 1) % len];
            if self.graph.has_edge(src, tgt) {
                return (false, MutationUndo::Noop);
            }
        }

        let mut edges = Vec::with_capacity(len);
        for i in 0..len {
            let src = nodes[i];
            let tgt = nodes[(i + 1) % len];
            self.graph.add_edge(src, tgt);
            edges.push((src, tgt));
        }
        self.csr_dirty = true;
        (true, MutationUndo::AddedLoop { edges })
    }

    /// Mutate threshold, returning an undo token.
    pub fn mutate_theta_undo(&mut self, rng: &mut impl Rng) -> (bool, MutationUndo) {
        let neuron_count = self.graph.neuron_count();
        if neuron_count == 0 {
            return (false, MutationUndo::Noop);
        }
        let index = rng.gen_range(0..neuron_count);
        let new_value = rng.gen_range(0..=15u8);
        let old_value = self.spike[index].threshold;
        if old_value == new_value {
            return (false, MutationUndo::Noop);
        }
        self.spike[index].threshold = new_value;
        (true, MutationUndo::Theta { index, old_value })
    }

    /// Mutate channel, returning an undo token.
    pub fn mutate_channel_undo(&mut self, rng: &mut impl Rng) -> (bool, MutationUndo) {
        let neuron_count = self.graph.neuron_count();
        if neuron_count == 0 {
            return (false, MutationUndo::Noop);
        }
        let index = rng.gen_range(0..neuron_count);
        let new_value = rng.gen_range(1..=8u8);
        let old_value = self.spike[index].channel;
        if old_value == new_value {
            return (false, MutationUndo::Noop);
        }
        self.spike[index].channel = new_value;
        (true, MutationUndo::Channel { index, old_value })
    }

    /// Flip polarity, returning an undo token.
    pub fn mutate_polarity_undo(&mut self, rng: &mut impl Rng) -> (bool, MutationUndo) {
        let neuron_count = self.graph.neuron_count();
        if neuron_count == 0 {
            return (false, MutationUndo::Noop);
        }
        let index = rng.gen_range(0..neuron_count);
        self.polarity[index] = -self.polarity[index];
        (true, MutationUndo::Polarity { index })
    }

    // ---- Genome persistence ----

    /// Serialize network genome to bytes (bincode format).
    ///
    /// Ephemeral state (activation, charge, refractory) is **not** included.
    /// Use [`genome_from_bytes`](Self::genome_from_bytes) to reconstruct.
    pub fn genome_to_bytes(&self) -> Vec<u8> {
        let (sources, targets) = self.graph.edge_endpoints();
        let dto = disk::NetworkDiskV1 {
            version: disk::CURRENT_VERSION,
            graph: disk::ConnectionGraphDiskV1 {
                neuron_count: self.graph.neuron_count(),
                sources: sources.iter().map(|&s| s as usize).collect(),
                targets: targets.iter().map(|&t| t as usize).collect(),
            },
            threshold: self.spike.iter().map(|s| s.threshold as u32).collect(),
            channel: self.spike.iter().map(|s| s.channel).collect(),
            polarity: self.polarity.iter().map(|&p| p as i32).collect(),
        };
        bincode::serialize(&dto).expect("genome serialization cannot fail")
    }

    /// Deserialize network genome from bytes.
    ///
    /// Returns a fresh network with zeroed ephemeral state, like after
    /// [`reset()`](Self::reset).
    ///
    /// # Errors
    ///
    /// Returns [`NetworkError::Genome`] if the bytes are malformed.
    pub fn genome_from_bytes(bytes: &[u8]) -> Result<Self, NetworkError> {
        let dto: disk::NetworkDiskV1 =
            bincode::deserialize(bytes).map_err(|e| NetworkError::Genome(e.to_string()))?;
        if dto.version != disk::CURRENT_VERSION {
            return Err(NetworkError::Genome(format!(
                "unsupported version {}, expected {}",
                dto.version,
                disk::CURRENT_VERSION
            )));
        }
        disk::validate(&dto).map_err(NetworkError::Genome)?;

        let n = dto.graph.neuron_count;
        let sources: Vec<u16> = dto.graph.sources.iter().map(|&s| s as u16).collect();
        let targets: Vec<u16> = dto.graph.targets.iter().map(|&t| t as u16).collect();
        let graph = ConnectionGraph::from_validated_edges(n, sources, targets);

        let threshold_vec: Vec<u8> = dto.threshold.iter().map(|&t| t as u8).collect();
        let channel_vec: Vec<u8> = dto.channel;
        let spike: Vec<SpikeData> = threshold_vec
            .iter()
            .zip(channel_vec.iter())
            .map(|(&t, &c)| SpikeData { charge: 0, threshold: t, channel: c, _pad: 0 })
            .collect();

        Ok(Self {
            graph,
            spike,
            polarity: dto.polarity.iter().map(|&p| p as i8).collect(),
            activation: vec![0; n],
            refractory: vec![0; n],
            workspace: PropagationWorkspace::new(n),
            csr_offsets: vec![0u32; n + 1],
            csr_targets: Vec::new(),
            csr_dirty: true,
            active_neurons: Vec::with_capacity(n),
            dirty_set: Vec::new(),
            dirty_member: vec![false; n],
        })
    }

    /// Save network genome (topology + learned parameters) to disk.
    ///
    /// Ephemeral state (activation, charge, refractory) is **not** saved.
    /// The file is written atomically (temp + rename) to avoid corruption.
    pub fn save_genome(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = self.genome_to_bytes();
        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, bytes)?;
        fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load network genome from disk. Returns a fresh network with zeroed
    /// ephemeral state (activation, charge, refractory), like after [`reset()`](Self::reset).
    ///
    /// # Errors
    ///
    /// Returns [`NetworkError::Io`] on file errors or deserialization failure.
    /// Returns [`NetworkError::Genome`] if the file contents are malformed
    /// (edge bounds, parameter ranges, array lengths).
    pub fn load_genome(path: impl AsRef<Path>) -> Result<Self, NetworkError> {
        let bytes = fs::read(path).map_err(NetworkError::Io)?;
        Self::genome_from_bytes(&bytes).map_err(|e| match e {
            NetworkError::Genome(msg) => NetworkError::Genome(msg),
            other => other,
        })
    }

    // ---- Mutations ----

    /// Add a random directed edge between two neurons.
    ///
    /// Picks a random source and target uniformly. Returns `true` if an edge
    /// was added, `false` if the pick was a self-loop, duplicate, or the
    /// network has fewer than 2 neurons.
    pub fn mutate_add_edge(&mut self, rng: &mut impl Rng) -> bool {
        let neuron_count = self.graph.neuron_count();
        if neuron_count < 2 {
            return false;
        }
        let source = rng.gen_range(0..neuron_count) as u16;
        let target = rng.gen_range(0..neuron_count) as u16;
        let added = self.graph.add_edge(source, target);
        if added {
            self.csr_dirty = true;
        }
        added
    }

    /// Remove a random existing edge.
    ///
    /// Picks a random index from the edge list and swap-removes it. O(1).
    /// Returns `true` if an edge was removed, `false` if the graph is empty.
    ///
    /// In a mutation schedule, remove should come **last** — after add/rewire
    /// have built structure, pruning removes what doesn't help.
    pub fn mutate_remove_edge(&mut self, rng: &mut impl Rng) -> bool {
        let edge_count = self.graph.edge_count();
        if edge_count == 0 {
            return false;
        }
        let index = rng.gen_range(0..edge_count);
        self.graph.remove_edge_at(index);
        self.csr_dirty = true;
        true
    }

    /// Rewire a random edge to a new random target. Keeps the source, changes the target.
    ///
    /// Picks a random existing edge, removes it, then adds a new edge from the
    /// same source to a random different target. Returns `true` if the rewire
    /// succeeded, `false` if the graph is empty or the new target was invalid
    /// (self-loop, duplicate).
    pub fn mutate_rewire(&mut self, rng: &mut impl Rng) -> bool {
        let edge_count = self.graph.edge_count();
        let neuron_count = self.graph.neuron_count();
        if edge_count == 0 || neuron_count < 2 {
            return false;
        }
        let index = rng.gen_range(0..edge_count);
        let old_edge = self.graph.remove_edge_at(index).unwrap();
        let new_target = rng.gen_range(0..neuron_count) as u16;
        if !self.graph.add_edge(old_edge.source, new_target) {
            self.graph.add_edge(old_edge.source, old_edge.target);
            return false;
        }
        self.csr_dirty = true;
        true
    }

    /// Reverse a random edge's direction: A->B becomes B->A.
    ///
    /// Picks a random existing edge and flips it. Fails if the reverse
    /// already exists (would create a duplicate) or the graph is empty.
    pub fn mutate_reverse(&mut self, rng: &mut impl Rng) -> bool {
        let edge_count = self.graph.edge_count();
        if edge_count == 0 {
            return false;
        }
        let index = rng.gen_range(0..edge_count);
        let edges: Vec<_> = self.graph.iter_edges().collect();
        let edge = edges[index];
        let reversed = self.graph.reverse_edge(edge.source, edge.target);
        if reversed {
            self.csr_dirty = true;
        }
        reversed
    }

    /// Mirror a random edge: if A->B exists, also add B->A (bidirectional pair).
    ///
    /// Picks a random existing edge and adds the reverse. Fails if the reverse
    /// already exists or the graph is empty.
    pub fn mutate_mirror(&mut self, rng: &mut impl Rng) -> bool {
        let edge_count = self.graph.edge_count();
        if edge_count == 0 {
            return false;
        }
        let index = rng.gen_range(0..edge_count);
        let edges: Vec<_> = self.graph.iter_edges().collect();
        let edge = edges[index];
        let added = self.graph.add_edge(edge.target, edge.source);
        if added {
            self.csr_dirty = true;
        }
        added
    }

    /// Add an edge targeting a high in-degree neuron (top 25%).
    ///
    /// Computes in-degree for all neurons, picks a target from the top quartile,
    /// picks a random source. Falls back to random add if no edges exist yet.
    pub fn mutate_enhance(&mut self, rng: &mut impl Rng) -> bool {
        let neuron_count = self.graph.neuron_count();
        if neuron_count < 2 {
            return false;
        }
        if self.graph.edge_count() == 0 {
            return self.mutate_add_edge(rng); // fallback: no topology to enhance
        }
        // Compute in-degree
        let mut in_degree = vec![0u32; neuron_count];
        for edge in self.graph.iter_edges() {
            in_degree[edge.target as usize] += 1;
        }
        // Find top 25% threshold
        let mut sorted_degrees = in_degree.clone();
        sorted_degrees.sort_unstable();
        let top_quartile_idx = neuron_count - neuron_count / 4;
        let threshold = sorted_degrees[top_quartile_idx.min(neuron_count - 1)];
        // Collect high in-degree neurons
        let high_indegree_neurons: Vec<u16> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, &d)| d >= threshold)
            .map(|(i, _)| i as u16)
            .collect();
        if high_indegree_neurons.is_empty() {
            return self.mutate_add_edge(rng);
        }
        let target = high_indegree_neurons[rng.gen_range(0..high_indegree_neurons.len())];
        let source = rng.gen_range(0..neuron_count) as u16;
        let added = self.graph.add_edge(source, target);
        if added {
            self.csr_dirty = true;
        }
        added
    }

    /// Add an edge preferring a same-channel target (fire together, wire together).
    ///
    /// Picks a random source neuron, finds neurons with the same channel,
    /// and adds an edge to one of them. Falls back to random if no same-channel
    /// neuron exists.
    pub fn mutate_add_affinity(&mut self, rng: &mut impl Rng) -> bool {
        let neuron_count = self.graph.neuron_count();
        if neuron_count < 2 {
            return false;
        }
        let source_idx = rng.gen_range(0..neuron_count);
        let source_channel = self.spike[source_idx].channel;
        // Find same-channel neurons
        let same_ch: Vec<u16> = self
            .spike
            .iter()
            .enumerate()
            .filter(|(i, s)| s.channel == source_channel && *i != source_idx)
            .map(|(i, _)| i as u16)
            .collect();
        let target = if same_ch.is_empty() {
            rng.gen_range(0..neuron_count) as u16 // fallback: random
        } else {
            same_ch[rng.gen_range(0..same_ch.len())]
        };
        let added = self.graph.add_edge(source_idx as u16, target);
        if added {
            self.csr_dirty = true;
        }
        added
    }

    /// Add a complete directed loop of `len` random neurons as a single atomic mutation.
    ///
    /// A loop of length N creates N edges: A→B→C→...→A.
    /// All edges must be new (no duplicates) and no self-loops.
    /// If any edge collides, the entire loop is aborted (returns `false`).
    ///
    /// Loop mutations are valuable for sparse/empty networks because they create
    /// recurrent circuits in one step — a single `add_edge` cannot form a cycle.
    ///
    /// `len` is clamped to `[2, neuron_count]`. Returns `true` if all edges were added.
    pub fn mutate_add_loop(&mut self, rng: &mut impl Rng, len: usize) -> bool {
        let neuron_count = self.graph.neuron_count();
        if neuron_count < 2 || len < 2 {
            return false;
        }
        let len = len.min(neuron_count);

        // Pick `len` distinct random neurons
        let mut nodes = Vec::with_capacity(len);
        for _ in 0..len * 10 {
            let n = rng.gen_range(0..neuron_count) as u16;
            if !nodes.contains(&n) {
                nodes.push(n);
            }
            if nodes.len() == len {
                break;
            }
        }
        if nodes.len() < len {
            return false; // couldn't find enough distinct neurons
        }

        // Check all edges are free
        for i in 0..len {
            let src = nodes[i];
            let tgt = nodes[(i + 1) % len];
            if self.graph.has_edge(src, tgt) {
                return false; // collision, abort
            }
        }

        // Commit all edges atomically
        for i in 0..len {
            let src = nodes[i];
            let tgt = nodes[(i + 1) % len];
            self.graph.add_edge(src, tgt);
        }
        self.csr_dirty = true;
        true
    }

    /// Mutate one random neuron's threshold to a random value in `[0, 15]`.
    ///
    /// Returns `true` if the value changed, `false` if the random pick
    /// landed on the same value or the network is empty.
    pub fn mutate_theta(&mut self, rng: &mut impl Rng) -> bool {
        let neuron_count = self.graph.neuron_count();
        if neuron_count == 0 {
            return false;
        }
        let index = rng.gen_range(0..neuron_count);
        let new_value = rng.gen_range(0..=15u8);
        if self.spike[index].threshold == new_value {
            return false;
        }
        self.spike[index].threshold = new_value;
        true
    }

    /// Mutate one random neuron's channel to a random value in `[1, 8]`.
    ///
    /// Returns `true` if the value changed, `false` if the random pick
    /// landed on the same value or the network is empty.
    pub fn mutate_channel(&mut self, rng: &mut impl Rng) -> bool {
        let neuron_count = self.graph.neuron_count();
        if neuron_count == 0 {
            return false;
        }
        let index = rng.gen_range(0..neuron_count);
        let new_value = rng.gen_range(1..=8u8);
        if self.spike[index].channel == new_value {
            return false;
        }
        self.spike[index].channel = new_value;
        true
    }

    /// Flip one random neuron's polarity: +1 becomes -1, -1 becomes +1.
    ///
    /// Always returns `true` unless the network is empty.
    pub fn mutate_polarity(&mut self, rng: &mut impl Rng) -> bool {
        let neuron_count = self.graph.neuron_count();
        if neuron_count == 0 {
            return false;
        }
        let index = rng.gen_range(0..neuron_count);
        self.polarity[index] = -self.polarity[index];
        true
    }

    // ---- Queries ----

    /// Number of neurons in the network.
    #[inline]
    pub fn neuron_count(&self) -> usize {
        self.graph.neuron_count()
    }

    /// Number of directed edges.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    // ---- Topology access ----

    /// Immutable reference to the connection graph.
    #[inline]
    pub fn graph(&self) -> &ConnectionGraph {
        &self.graph
    }

    /// Mutable reference to the connection graph for adding/removing edges.
    /// Marks the CSR cache as dirty (rebuilt on next propagate).
    #[inline]
    pub fn graph_mut(&mut self) -> &mut ConnectionGraph {
        self.csr_dirty = true;
        &mut self.graph
    }

    // ---- Spike data access (AoS: charge, threshold, channel) ----

    /// Per-neuron spike-decision data (charge, threshold, channel).
    #[inline]
    pub fn spike_data(&self) -> &[SpikeData] {
        &self.spike
    }

    /// Mutable access to per-neuron spike-decision data.
    #[inline]
    pub fn spike_data_mut(&mut self) -> &mut [SpikeData] {
        &mut self.spike
    }

    /// Threshold value for neuron at `idx` (stored `[0,15]`, effective = stored+1).
    #[inline]
    pub fn threshold_at(&self, idx: usize) -> u8 {
        self.spike[idx].threshold
    }

    /// Set threshold for neuron at `idx`. Valid range: `[0,15]`.
    #[inline]
    pub fn set_threshold(&mut self, idx: usize, val: u8) {
        self.spike[idx].threshold = val;
    }

    /// Set all neuron thresholds to `val`.
    #[inline]
    pub fn set_all_threshold(&mut self, val: u8) {
        for s in self.spike.iter_mut() {
            s.threshold = val;
        }
    }

    /// Channel value for neuron at `idx`. Range: `[1,8]`.
    #[inline]
    pub fn channel_at(&self, idx: usize) -> u8 {
        self.spike[idx].channel
    }

    /// Set channel for neuron at `idx`. Valid range: `[1,8]`.
    #[inline]
    pub fn set_channel(&mut self, idx: usize, val: u8) {
        self.spike[idx].channel = val;
    }

    /// Set all neuron channels to `val`.
    #[inline]
    pub fn set_all_channel(&mut self, val: u8) {
        for s in self.spike.iter_mut() {
            s.channel = val;
        }
    }

    /// Charge value for neuron at `idx`. Range: `[0, LIMIT_MAX_CHARGE]`.
    #[inline]
    pub fn charge_at(&self, idx: usize) -> u8 {
        self.spike[idx].charge
    }

    /// Collect charge values for a range of neurons into a `Vec<u8>`.
    ///
    /// Useful for feeding output-zone charges to a projection layer.
    #[inline]
    pub fn charge_vec(&self, range: std::ops::Range<usize>) -> Vec<u8> {
        self.spike[range].iter().map(|s| s.charge).collect()
    }

    // ---- Parameter access (polarity stays separate) ----

    /// Per-neuron polarity (`+1` excitatory, `-1` inhibitory).
    #[inline]
    pub fn polarity(&self) -> &[i8] {
        &self.polarity
    }

    /// Mutable access to per-neuron polarity. Valid values: `+1` or `-1`.
    #[inline]
    pub fn polarity_mut(&mut self) -> &mut [i8] {
        &mut self.polarity
    }

    // ---- State access (read-only) ----

    /// Current activation values (read-only). Values: `+1`, `-1`, or `0`.
    #[inline]
    pub fn activation(&self) -> &[i8] {
        &self.activation
    }

    /// Length of the dirty set (for testing/diagnostics).
    #[cfg(test)]
    pub(crate) fn dirty_set_len(&self) -> usize {
        self.dirty_set.len()
    }
}

#[cfg(test)]
mod audit;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::LIMIT_MAX_CHARGE;
    use crate::propagation::PropagationError;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn default_config() -> PropagationConfig {
        PropagationConfig::default()
    }

    #[test]
    fn new_creates_correct_defaults() {
        let net = Network::new(256);
        assert_eq!(net.neuron_count(), 256);
        assert_eq!(net.edge_count(), 0);
        assert!(net.spike_data().iter().all(|s| s.threshold == 0));
        assert!(net.spike_data().iter().all(|s| s.channel == 1));
        assert!(net.polarity().iter().all(|&p| p == 1));
        assert!(net.activation().iter().all(|&a| a == 0));
        assert!(net.spike_data().iter().all(|s| s.charge == 0));
    }

    #[test]
    fn new_zero_neurons() {
        let net = Network::new(0);
        assert_eq!(net.neuron_count(), 0);
        assert_eq!(net.edge_count(), 0);
        assert!(net.spike_data().is_empty());
    }

    #[test]
    fn propagate_no_edges_no_panic() {
        let mut net = Network::new(64);
        let input = vec![0i32; 64];
        assert!(net.propagate(&input, &default_config()).is_ok());
    }

    #[test]
    fn propagate_rejects_wrong_input_length() {
        let mut net = Network::new(256);
        let short_input = vec![0i32; 100];
        let err = net.propagate(&short_input, &default_config()).unwrap_err();
        assert!(
            matches!(
                err,
                NetworkError::Propagation(PropagationError::InputLengthMismatch {
                    expected: 256,
                    actual: 100
                })
            ),
            "expected InputLengthMismatch, got: {err}"
        );
    }

    #[test]
    fn propagate_accepts_correct_input() {
        let mut net = Network::new(64);
        let input = vec![0i32; 64];
        assert!(net.propagate(&input, &default_config()).is_ok());
    }

    #[test]
    fn reset_zeros_state() {
        let mut net = Network::new(64);
        net.graph_mut().add_edge(0, 1);
        net.set_threshold(0, 5);

        let mut input = vec![0i32; 64];
        input[0] = 1;
        net.propagate(&input, &default_config()).unwrap();

        net.reset();
        assert!(net.activation().iter().all(|&a| a == 0));
        assert!(net.spike_data().iter().all(|s| s.charge == 0));
        // parameters and edges untouched
        assert_eq!(net.threshold_at(0), 5);
        assert_eq!(net.edge_count(), 1);
    }

    #[test]
    fn graph_mut_add_edges() {
        let mut net = Network::new(64);
        assert!(net.graph_mut().add_edge(10, 42));
        assert!(net.graph_mut().add_edge(42, 10));
        assert_eq!(net.edge_count(), 2);
        assert!(net.graph().has_edge(10, 42));
    }

    #[test]
    fn set_threshold_works() {
        let mut net = Network::new(16);
        net.set_threshold(5, 12);
        assert_eq!(net.threshold_at(5), 12);
    }

    #[test]
    fn set_channel_works() {
        let mut net = Network::new(16);
        net.set_channel(3, 7);
        assert_eq!(net.channel_at(3), 7);
    }

    #[test]
    fn polarity_mut_works() {
        let mut net = Network::new(16);
        net.polarity_mut()[0] = -1;
        assert_eq!(net.polarity()[0], -1);
    }

    #[test]
    fn excitatory_chain_propagates() {
        let mut net = Network::new(3);
        net.graph_mut().add_edge(0, 1);
        net.graph_mut().add_edge(1, 2);

        // Input at neuron 0, 2 ticks so signal is still in transit
        let input = vec![1i32, 0, 0];
        let config = PropagationConfig {
            ticks_per_token: 2,
            input_duration_ticks: 2,
            decay_interval_ticks: 6,
            use_refractory: false,
        };
        net.propagate(&input, &config).unwrap();

        // After 2 ticks: neuron 1 fired (from 0→1) and neuron 2 fired (from 1→2)
        assert_eq!(net.activation()[2], 1, "signal should reach neuron 2");
    }

    #[test]
    fn inhibitory_suppresses_charge() {
        // Run two networks with identical topology but different polarity at neuron 1.
        // Excitatory network: neuron 1 fires +1 → neuron 2 gains charge.
        // Inhibitory network: neuron 1 fires -1 → neuron 2 gains nothing.
        let config = PropagationConfig {
            ticks_per_token: 2,
            input_duration_ticks: 2,
            decay_interval_ticks: 0,
            use_refractory: false,
        };
        let input = vec![1i32, 0, 0];

        // Excitatory control
        let mut exc = Network::new(3);
        exc.graph_mut().add_edge(0, 1);
        exc.graph_mut().add_edge(1, 2);
        exc.propagate(&input, &config).unwrap();

        // Inhibitory variant
        let mut inh = Network::new(3);
        inh.graph_mut().add_edge(0, 1);
        inh.graph_mut().add_edge(1, 2);
        inh.polarity_mut()[1] = -1;
        inh.propagate(&input, &config).unwrap();

        // Excitatory: neuron 2 receives +1 → fires → activation=1
        // Inhibitory: neuron 2 receives -1 → charge stays 0 → silent → activation=0
        assert_eq!(exc.activation()[2], 1, "excitatory neuron 2 should fire");
        assert_eq!(
            inh.activation()[2],
            0,
            "inhibitory signal should prevent neuron 2 from firing"
        );
    }

    #[test]
    fn charge_bounded_extreme_input() {
        let mut net = Network::new(4);
        net.graph_mut().add_edge(0, 1);
        net.graph_mut().add_edge(2, 1);
        net.graph_mut().add_edge(3, 1);

        let input = vec![1i32; 4];
        let config = PropagationConfig {
            ticks_per_token: 24,
            input_duration_ticks: 24,
            decay_interval_ticks: 0, // no decay
            use_refractory: false,
        };
        net.propagate(&input, &config).unwrap();

        assert!(
            net.spike_data().iter().all(|s| s.charge <= LIMIT_MAX_CHARGE),
            "charge must not exceed LIMIT_MAX_CHARGE"
        );
    }

    #[test]
    fn multiple_propagate_accumulates() {
        // High threshold (effective=6) so neurons accumulate charge without firing.
        // After 2 ticks with edge 0→1 and input at 0: charge[1] increases by 2 per propagate.
        let mut net = Network::new(4);
        net.graph_mut().add_edge(0, 1);
        net.set_all_threshold(5); // effective = 6

        let input = vec![1i32, 0, 0, 0];
        let config = PropagationConfig {
            ticks_per_token: 2,
            input_duration_ticks: 2,
            decay_interval_ticks: 0,
            use_refractory: false,
        };

        net.propagate(&input, &config).unwrap();
        let charge_after_one = net.charge_at(1);

        // Second propagation without reset — charge carries over
        net.propagate(&input, &config).unwrap();
        let charge_after_two = net.charge_at(1);

        assert!(
            charge_after_two > charge_after_one,
            "charge should accumulate: after_one={charge_after_one}, after_two={charge_after_two}"
        );
    }

    #[test]
    fn propagate_after_reset_matches_fresh() {
        let mut net = Network::new(4);
        net.graph_mut().add_edge(0, 1);
        net.graph_mut().add_edge(1, 2);

        let input = vec![1i32, 0, 0, 0];
        let config = default_config();

        // First: propagate on dirty state, then reset and propagate again
        net.propagate(&input, &config).unwrap();
        net.reset();
        net.propagate(&input, &config).unwrap();
        let after_reset = (net.activation().to_vec(), net.spike_data().to_vec());

        // Second: fresh network, propagate once
        let mut fresh = Network::new(4);
        fresh.graph_mut().add_edge(0, 1);
        fresh.graph_mut().add_edge(1, 2);
        fresh.propagate(&input, &config).unwrap();
        let after_fresh = (fresh.activation().to_vec(), fresh.spike_data().to_vec());

        assert_eq!(
            after_reset, after_fresh,
            "reset should restore to fresh state"
        );
    }

    #[test]
    fn network_error_display() {
        let inner = PropagationError::InputLengthMismatch {
            expected: 256,
            actual: 100,
        };
        let err = NetworkError::Propagation(inner);
        let msg = format!("{err}");
        assert!(msg.contains("256"), "should mention expected length");
        assert!(msg.contains("100"), "should mention actual length");
    }

    // --- Adversarial review findings (#16): invalid _mut() + propagate round-trip ---

    #[test]
    fn propagate_rejects_invalid_threshold() {
        let mut net = Network::new(4);
        net.spike_data_mut()[0].threshold = 99; // out of [0,15]
        let input = vec![0i32; 4];
        let err = net.propagate(&input, &default_config()).unwrap_err();
        assert!(
            matches!(
                err,
                NetworkError::Propagation(PropagationError::ThresholdOutOfRange {
                    index: 0,
                    value: 99
                })
            ),
            "expected ThresholdOutOfRange, got: {err}"
        );
    }

    #[test]
    fn propagate_rejects_invalid_channel() {
        let mut net = Network::new(4);
        net.spike_data_mut()[2].channel = 0; // out of [1,8]
        let input = vec![0i32; 4];
        let err = net.propagate(&input, &default_config()).unwrap_err();
        assert!(
            matches!(
                err,
                NetworkError::Propagation(PropagationError::ChannelOutOfRange {
                    index: 2,
                    value: 0
                })
            ),
            "expected ChannelOutOfRange, got: {err}"
        );
    }

    #[test]
    fn propagate_rejects_invalid_polarity() {
        let mut net = Network::new(4);
        net.polarity_mut()[1] = 2; // not ±1
        let input = vec![0i32; 4];
        let err = net.propagate(&input, &default_config()).unwrap_err();
        assert!(
            matches!(
                err,
                NetworkError::Propagation(PropagationError::PolarityOutOfRange {
                    index: 1,
                    value: 2
                })
            ),
            "expected PolarityOutOfRange, got: {err}"
        );
    }

    // --- Adversarial review finding #19: 0-neuron propagate ---

    #[test]
    fn zero_neuron_propagate_succeeds() {
        let mut net = Network::new(0);
        let input: Vec<i32> = vec![];
        assert!(net.propagate(&input, &default_config()).is_ok());
    }

    // --- Adversarial review finding #15: ticks_per_token = 0 ---

    #[test]
    fn zero_ticks_propagate_is_noop() {
        let mut net = Network::new(4);
        net.graph_mut().add_edge(0, 1);
        let input = vec![1i32, 0, 0, 0];
        let config = PropagationConfig {
            ticks_per_token: 0,
            input_duration_ticks: 0,
            decay_interval_ticks: 0,
            use_refractory: false,
        };
        net.propagate(&input, &config).unwrap();
        // Zero ticks = no simulation, state unchanged
        assert!(net.activation().iter().all(|&a| a == 0));
        assert!(net.spike_data().iter().all(|s| s.charge == 0));
    }

    // --- Deep research finding: Network should be Clone ---

    #[test]
    fn network_clone_is_independent() {
        let mut net = Network::new(4);
        net.graph_mut().add_edge(0, 1);
        net.set_threshold(0, 5);

        let mut cloned = net.clone();
        cloned.set_threshold(0, 10);
        cloned.graph_mut().add_edge(2, 3);

        // Original unaffected
        assert_eq!(net.threshold_at(0), 5);
        assert_eq!(net.edge_count(), 1);
        // Clone has its own state
        assert_eq!(cloned.threshold_at(0), 10);
        assert_eq!(cloned.edge_count(), 2);
    }

    // --- Snapshot/rollback tests (step 2) ---

    #[test]
    fn save_captures_defaults() {
        let net = Network::new(16);
        let snap = net.save_state();
        assert_eq!(snap.neuron_count(), 16);
        assert_eq!(snap.edge_count(), 0);
        assert!(snap.spike_data().iter().all(|s| s.threshold == 0));
        assert!(snap.spike_data().iter().all(|s| s.channel == 1));
        assert!(snap.polarity().iter().all(|&p| p == 1));
        assert!(snap.activation().iter().all(|&a| a == 0));
        assert!(snap.spike_data().iter().all(|s| s.charge == 0));
    }

    #[test]
    fn save_captures_modified_params() {
        let mut net = Network::new(8);
        net.set_threshold(0, 12);
        net.set_channel(1, 7);
        net.polarity_mut()[2] = -1;
        let snap = net.save_state();
        assert_eq!(snap.threshold_at(0), 12);
        assert_eq!(snap.channel_at(1), 7);
        assert_eq!(snap.polarity()[2], -1);
    }

    #[test]
    fn save_captures_edges() {
        let mut net = Network::new(8);
        net.graph_mut().add_edge(0, 1);
        net.graph_mut().add_edge(2, 3);
        let snap = net.save_state();
        assert_eq!(snap.edge_count(), 2);
        assert!(snap.graph().has_edge(0, 1));
        assert!(snap.graph().has_edge(2, 3));
    }

    #[test]
    fn save_captures_propagated_state() {
        let mut net = Network::new(4);
        net.graph_mut().add_edge(0, 1);
        let input = vec![1i32, 0, 0, 0];
        let config = PropagationConfig {
            ticks_per_token: 2,
            input_duration_ticks: 2,
            decay_interval_ticks: 0,
            use_refractory: false,
        };
        net.propagate(&input, &config).unwrap();
        let snap = net.save_state();
        assert_eq!(snap.activation(), net.activation());
        assert_eq!(snap.spike_data(), net.spike_data());
    }

    #[test]
    fn restore_reverts_params() {
        let mut net = Network::new(8);
        net.set_threshold(3, 10);
        let snap = net.save_state();

        net.set_threshold(3, 0);
        assert_eq!(net.threshold_at(3), 0);

        net.restore_state(&snap);
        assert_eq!(net.threshold_at(3), 10);
    }

    #[test]
    fn restore_reverts_edges() {
        let mut net = Network::new(8);
        net.graph_mut().add_edge(0, 1);
        let snap = net.save_state();

        net.graph_mut().add_edge(2, 3);
        assert_eq!(net.edge_count(), 2);

        net.restore_state(&snap);
        assert_eq!(net.edge_count(), 1);
        assert!(net.graph().has_edge(0, 1));
        assert!(!net.graph().has_edge(2, 3));
    }

    #[test]
    fn restore_reverts_state() {
        let mut net = Network::new(4);
        net.graph_mut().add_edge(0, 1);
        let snap = net.save_state(); // activation=0, charge=0

        let input = vec![1i32, 0, 0, 0];
        net.propagate(&input, &default_config()).unwrap();

        net.restore_state(&snap);
        assert!(net.activation().iter().all(|&a| a == 0));
        assert!(net.spike_data().iter().all(|s| s.charge == 0));
    }

    #[test]
    fn restore_twice_from_same_snapshot() {
        let mut net = Network::new(4);
        net.set_threshold(0, 8);
        let snap = net.save_state();

        net.set_threshold(0, 0);
        net.restore_state(&snap);
        assert_eq!(net.threshold_at(0), 8);

        net.set_threshold(0, 15);
        net.restore_state(&snap);
        assert_eq!(net.threshold_at(0), 8);
    }

    #[test]
    fn snapshot_independent_of_network() {
        let mut net = Network::new(4);
        net.set_threshold(0, 5);
        let snap = net.save_state();

        net.set_threshold(0, 99);
        net.graph_mut().add_edge(0, 1);

        assert_eq!(snap.threshold_at(0), 5);
        assert_eq!(snap.edge_count(), 0);
    }

    #[test]
    fn restore_then_propagate_matches_original() {
        let mut net = Network::new(4);
        net.graph_mut().add_edge(0, 1);
        net.graph_mut().add_edge(1, 2);
        let snap = net.save_state();

        let input = vec![1i32, 0, 0, 0];
        let config = PropagationConfig {
            ticks_per_token: 2,
            input_duration_ticks: 2,
            decay_interval_ticks: 0,
            use_refractory: false,
        };
        net.propagate(&input, &config).unwrap();
        let first_result = (net.activation().to_vec(), net.spike_data().to_vec());

        // Dirty the state with a different input
        net.propagate(&[0, 1, 0, 0], &config).unwrap();

        // Restore and re-propagate with original input
        net.restore_state(&snap);
        net.propagate(&input, &config).unwrap();
        let second_result = (net.activation().to_vec(), net.spike_data().to_vec());

        assert_eq!(first_result, second_result);
    }

    #[test]
    #[should_panic(expected = "snapshot neuron_count mismatch")]
    fn restore_panics_wrong_neuron_count() {
        let net_big = Network::new(256);
        let snap = net_big.save_state();

        let mut net_small = Network::new(128);
        net_small.restore_state(&snap);
    }

    #[test]
    fn snapshot_clone_is_independent() {
        let mut net = Network::new(4);
        net.set_threshold(0, 7);
        let snap = net.save_state();
        let snap_clone = snap.clone();

        net.set_threshold(0, 0);
        net.restore_state(&snap);
        assert_eq!(net.threshold_at(0), 7);
        assert_eq!(snap_clone.threshold_at(0), 7);
    }

    #[test]
    fn mixed_rollback_restores_everything() {
        let mut net = Network::new(8);
        net.graph_mut().add_edge(0, 1);
        net.graph_mut().add_edge(1, 2);
        net.set_threshold(3, 10);
        net.set_channel(4, 5);
        net.polarity_mut()[5] = -1;
        let snap = net.save_state();

        // Mutate EVERYTHING at once
        net.graph_mut().add_edge(3, 4);
        net.graph_mut().remove_edge(0, 1);
        net.set_threshold(3, 0);
        net.set_channel(4, 8);
        net.polarity_mut()[5] = 1;
        let input = vec![1i32; 8];
        let config = PropagationConfig {
            ticks_per_token: 2,
            input_duration_ticks: 2,
            decay_interval_ticks: 0,
            use_refractory: false,
        };
        net.propagate(&input, &config).unwrap();

        // Restore
        net.restore_state(&snap);

        // Everything reverted
        assert_eq!(net.edge_count(), 2);
        assert!(net.graph().has_edge(0, 1));
        assert!(!net.graph().has_edge(3, 4));
        assert_eq!(net.threshold_at(3), 10);
        assert_eq!(net.channel_at(4), 5);
        assert_eq!(net.polarity()[5], -1);
        assert!(net.activation().iter().all(|&a| a == 0));
        assert!(net.spike_data().iter().all(|s| s.charge == 0));
    }

    // --- Mutation tests (deterministic, fixed seed) ---

    #[test]
    fn mutate_add_edge_increases_count() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(64);
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..10 {
            net.mutate_add_edge(&mut rng);
        }
        assert!(
            net.edge_count() > 0,
            "10 attempts on 64 neurons should add edges"
        );
    }

    #[test]
    fn mutate_add_edge_deterministic() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..5 {
            net.mutate_add_edge(&mut rng);
        }
        // seed=42, 8 neurons, 5 attempts: 1 self-loop rejected, 4 edges added
        assert_eq!(net.edge_count(), 4);
        assert!(net.graph().has_edge(5, 3));
        assert!(net.graph().has_edge(0, 3));
        assert!(net.graph().has_edge(1, 0));
        assert!(net.graph().has_edge(7, 4));
    }

    #[test]
    fn mutate_add_edge_no_self_loops() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(16);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..100 {
            net.mutate_add_edge(&mut rng);
        }
        for edge in net.graph().iter_edges() {
            assert_ne!(edge.source, edge.target, "self-loop found: {}", edge.source);
        }
    }

    #[test]
    fn mutate_add_edge_saturates_small_graph() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(4); // max edges = 4 * 3 = 12
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..200 {
            net.mutate_add_edge(&mut rng);
        }
        assert_eq!(
            net.edge_count(),
            12,
            "4-neuron graph should saturate at 12 edges"
        );
    }

    #[test]
    fn mutate_add_edge_single_neuron() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(1);
        let mut rng = StdRng::seed_from_u64(0);
        assert!(
            !net.mutate_add_edge(&mut rng),
            "1-neuron network cannot add edges"
        );
        assert_eq!(net.edge_count(), 0);
    }

    #[test]
    fn mutate_add_edge_rollback() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        let snap = net.save_state();

        let mut rng = StdRng::seed_from_u64(42);
        net.mutate_add_edge(&mut rng);
        net.mutate_add_edge(&mut rng);
        assert!(net.edge_count() > 0);

        net.restore_state(&snap);
        assert_eq!(net.edge_count(), 0, "rollback should remove mutated edges");
    }

    // --- mutate_remove_edge tests ---

    #[test]
    fn mutate_remove_edge_decreases_count() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        net.graph_mut().add_edge(0, 1);
        net.graph_mut().add_edge(1, 2);
        net.graph_mut().add_edge(2, 3);
        assert_eq!(net.edge_count(), 3);

        let mut rng = StdRng::seed_from_u64(42);
        assert!(net.mutate_remove_edge(&mut rng));
        assert_eq!(net.edge_count(), 2);
    }

    #[test]
    fn mutate_remove_edge_empty_graph() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        let mut rng = StdRng::seed_from_u64(42);
        assert!(
            !net.mutate_remove_edge(&mut rng),
            "empty graph should return false"
        );
    }

    #[test]
    fn mutate_remove_edge_drains_all() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(4);
        net.graph_mut().add_edge(0, 1);
        net.graph_mut().add_edge(1, 2);
        net.graph_mut().add_edge(2, 3);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..3 {
            assert!(net.mutate_remove_edge(&mut rng));
        }
        assert_eq!(net.edge_count(), 0);
        assert!(
            !net.mutate_remove_edge(&mut rng),
            "should return false when empty"
        );
    }

    #[test]
    fn mutate_remove_edge_rollback() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        net.graph_mut().add_edge(0, 1);
        net.graph_mut().add_edge(1, 2);
        let snap = net.save_state();

        let mut rng = StdRng::seed_from_u64(42);
        net.mutate_remove_edge(&mut rng);
        assert_eq!(net.edge_count(), 1);

        net.restore_state(&snap);
        assert_eq!(net.edge_count(), 2, "rollback should restore removed edge");
        assert!(net.graph().has_edge(0, 1));
        assert!(net.graph().has_edge(1, 2));
    }

    #[test]
    fn mutate_add_then_remove_net_zero() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        let mut rng = StdRng::seed_from_u64(42);

        // Add 5 edges
        for _ in 0..10 {
            net.mutate_add_edge(&mut rng);
        }
        let edges_after_add = net.edge_count();
        assert!(edges_after_add > 0);

        // Remove all
        while net.edge_count() > 0 {
            net.mutate_remove_edge(&mut rng);
        }
        assert_eq!(net.edge_count(), 0);
    }

    // --- mutate_rewire tests ---

    #[test]
    fn mutate_rewire_changes_target() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        net.graph_mut().add_edge(0, 1);
        let mut rng = StdRng::seed_from_u64(42);
        let rewired = net.mutate_rewire(&mut rng);
        // Either rewired successfully (target changed) or failed (self-loop/dup)
        assert_eq!(net.edge_count(), 1, "rewire should not change edge count");
        if rewired {
            assert!(
                !net.graph().has_edge(0, 1),
                "old edge should be gone after rewire"
            );
        }
    }

    #[test]
    fn mutate_rewire_preserves_edge_count() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(16);
        for i in 0..5 {
            net.graph_mut().add_edge(i, i + 1);
        }
        assert_eq!(net.edge_count(), 5);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..20 {
            net.mutate_rewire(&mut rng);
        }
        assert_eq!(
            net.edge_count(),
            5,
            "rewire should never change total edge count"
        );
    }

    #[test]
    fn mutate_rewire_empty_graph() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        let mut rng = StdRng::seed_from_u64(42);
        assert!(
            !net.mutate_rewire(&mut rng),
            "empty graph should return false"
        );
    }

    #[test]
    fn mutate_rewire_no_self_loops() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        for i in 0..7 {
            net.graph_mut().add_edge(i, i + 1);
        }
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..100 {
            net.mutate_rewire(&mut rng);
        }
        for edge in net.graph().iter_edges() {
            assert_ne!(edge.source, edge.target, "self-loop found after rewire");
        }
    }

    #[test]
    fn mutate_rewire_rollback() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        net.graph_mut().add_edge(0, 1);
        net.graph_mut().add_edge(2, 3);
        let snap = net.save_state();

        let mut rng = StdRng::seed_from_u64(42);
        net.mutate_rewire(&mut rng);

        net.restore_state(&snap);
        assert!(net.graph().has_edge(0, 1));
        assert!(net.graph().has_edge(2, 3));
        assert_eq!(net.edge_count(), 2);
    }

    // --- mutate_theta tests ---

    #[test]
    fn mutate_theta_changes_value() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        assert!(net.spike_data().iter().all(|s| s.threshold == 0)); // default
        let mut rng = StdRng::seed_from_u64(42);
        net.mutate_theta(&mut rng);
        assert!(
            net.spike_data().iter().any(|s| s.threshold != 0),
            "at least one theta should change"
        );
    }

    #[test]
    fn mutate_theta_stays_in_range() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(16);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..200 {
            net.mutate_theta(&mut rng);
        }
        assert!(
            net.spike_data().iter().all(|s| s.threshold <= 15),
            "theta must stay in [0,15]"
        );
    }

    #[test]
    fn mutate_theta_rollback() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        let snap = net.save_state();
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..10 {
            net.mutate_theta(&mut rng);
        }
        net.restore_state(&snap);
        assert!(net.spike_data().iter().all(|s| s.threshold == 0));
    }

    // --- mutate_channel tests ---

    #[test]
    fn mutate_channel_changes_value() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        assert!(net.spike_data().iter().all(|s| s.channel == 1)); // default
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..10 {
            net.mutate_channel(&mut rng);
        }
        assert!(
            net.spike_data().iter().any(|s| s.channel != 1),
            "at least one channel should change"
        );
    }

    #[test]
    fn mutate_channel_stays_in_range() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(16);
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..200 {
            net.mutate_channel(&mut rng);
        }
        assert!(
            net.spike_data().iter().all(|s| (1..=8).contains(&s.channel)),
            "channel must stay in [1,8]"
        );
    }

    // --- mutate_polarity tests ---

    #[test]
    fn mutate_polarity_flips() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        assert!(net.polarity().iter().all(|&p| p == 1)); // default: all excitatory
        let mut rng = StdRng::seed_from_u64(42);
        net.mutate_polarity(&mut rng);
        assert!(
            net.polarity().contains(&-1),
            "one neuron should flip to inhibitory"
        );
    }

    #[test]
    fn mutate_polarity_double_flip_restores() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(4);
        // Use seed that picks the same neuron twice
        let mut rng = StdRng::seed_from_u64(0);
        let idx1 = {
            let mut r = StdRng::seed_from_u64(0);
            r.gen_range(0..4usize)
        };
        net.mutate_polarity(&mut rng); // flip once
        assert_eq!(net.polarity()[idx1], -1);
        // Manually flip back the same neuron
        net.polarity_mut()[idx1] = -net.polarity()[idx1];
        assert_eq!(
            net.polarity()[idx1],
            1,
            "double flip should restore original"
        );
    }

    #[test]
    fn mutate_polarity_empty_network() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(0);
        let mut rng = StdRng::seed_from_u64(42);
        assert!(!net.mutate_polarity(&mut rng));
    }

    // --- mutate_reverse tests ---

    #[test]
    fn mutate_reverse_flips_direction() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        net.graph_mut().add_edge(0, 1);
        let mut rng = StdRng::seed_from_u64(42);
        let ok = net.mutate_reverse(&mut rng);
        if ok {
            assert!(!net.graph().has_edge(0, 1));
            assert!(net.graph().has_edge(1, 0));
        }
        assert_eq!(net.edge_count(), 1, "reverse should not change edge count");
    }

    #[test]
    fn mutate_reverse_empty_graph() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        let mut rng = StdRng::seed_from_u64(42);
        assert!(!net.mutate_reverse(&mut rng));
    }

    #[test]
    fn mutate_reverse_no_self_loops() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        for i in 0..7 {
            net.graph_mut().add_edge(i, i + 1);
        }
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..50 {
            net.mutate_reverse(&mut rng);
        }
        for edge in net.graph().iter_edges() {
            assert_ne!(edge.source, edge.target);
        }
    }

    // --- mutate_mirror tests ---

    #[test]
    fn mutate_mirror_adds_reverse() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        net.graph_mut().add_edge(0, 1);
        let mut rng = StdRng::seed_from_u64(42);
        let ok = net.mutate_mirror(&mut rng);
        if ok {
            assert!(net.graph().has_edge(0, 1), "original edge should remain");
            assert!(net.graph().has_edge(1, 0), "reverse edge should be added");
            assert_eq!(net.edge_count(), 2);
        }
    }

    #[test]
    fn mutate_mirror_already_bidirectional() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(4);
        net.graph_mut().add_edge(0, 1);
        net.graph_mut().add_edge(1, 0);
        let mut rng = StdRng::seed_from_u64(42);
        // Already bidirectional — mirror should fail (duplicate)
        assert!(!net.mutate_mirror(&mut rng));
        assert_eq!(net.edge_count(), 2);
    }

    #[test]
    fn mutate_mirror_empty_graph() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        let mut rng = StdRng::seed_from_u64(42);
        assert!(!net.mutate_mirror(&mut rng));
    }

    // --- mutate_enhance tests ---

    #[test]
    fn mutate_enhance_targets_high_degree() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        // Neuron 3 gets high in-degree
        net.graph_mut().add_edge(0, 3);
        net.graph_mut().add_edge(1, 3);
        net.graph_mut().add_edge(2, 3);
        net.graph_mut().add_edge(4, 5); // other edges
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..10 {
            net.mutate_enhance(&mut rng);
        }
        // Neuron 3 should have gained more incoming edges
        let in_deg_3 = net.graph().iter_edges().filter(|e| e.target == 3).count();
        assert!(
            in_deg_3 >= 3,
            "high in-degree neuron should keep/gain edges, got {in_deg_3}"
        );
    }

    #[test]
    fn mutate_enhance_empty_fallback() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        let mut rng = StdRng::seed_from_u64(42);
        // Empty graph → falls back to mutate_add_edge
        net.mutate_enhance(&mut rng);
        // Should have added at least tried to add an edge
        // (might fail if self-loop, that's ok)
    }

    // --- mutate_add_affinity tests ---

    #[test]
    fn mutate_add_affinity_prefers_same_channel() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        // Set channels: 0-3 = channel 1, 4-7 = channel 5
        for i in 0..4 {
            net.set_channel(i, 1);
        }
        for i in 4..8 {
            net.set_channel(i, 5);
        }
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..20 {
            net.mutate_add_affinity(&mut rng);
        }
        // Count same-channel edges
        let same_ch_edges = net
            .graph()
            .iter_edges()
            .filter(|e| net.channel_at(e.source as usize) == net.channel_at(e.target as usize))
            .count();
        let total = net.edge_count();
        assert!(
            same_ch_edges as f64 / total.max(1) as f64 > 0.5,
            "majority of edges should connect same-channel neurons, got {same_ch_edges}/{total}"
        );
    }

    #[test]
    fn mutate_add_affinity_no_self_loops() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut net = Network::new(8);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..50 {
            net.mutate_add_affinity(&mut rng);
        }
        for edge in net.graph().iter_edges() {
            assert_ne!(edge.source, edge.target);
        }
    }

    // ---- Genome persistence tests ----

    #[test]
    fn genome_roundtrip_preserves_topology_and_params() {
        let mut net = Network::new(16);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..20 {
            net.mutate_add_edge(&mut rng);
        }
        for i in 0..16 {
            net.spike_data_mut()[i].threshold = rng.gen_range(0..=15);
            net.spike_data_mut()[i].channel = rng.gen_range(1..=8);
            if rng.gen_ratio(1, 4) {
                net.polarity_mut()[i] = -1;
            }
        }

        let path = std::env::temp_dir().join("instnct_test_roundtrip.bin");
        net.save_genome(&path).unwrap();
        let loaded = Network::load_genome(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.edge_count(), net.edge_count());
        assert_eq!(loaded.spike_data(), net.spike_data());
        assert_eq!(loaded.polarity(), net.polarity());
        assert_eq!(loaded.neuron_count(), net.neuron_count());
        // Verify edges match
        let mut orig: Vec<_> = net.graph().iter_edges().map(|e| (e.source, e.target)).collect();
        let mut load: Vec<_> = loaded.graph().iter_edges().map(|e| (e.source, e.target)).collect();
        orig.sort();
        load.sort();
        assert_eq!(orig, load);
    }

    #[test]
    fn genome_roundtrip_zeroes_ephemeral_state() {
        let mut net = Network::new(8);
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..5 {
            net.mutate_add_edge(&mut rng);
        }
        // Propagate to create non-zero state; use high threshold so charge accumulates
        net.set_all_threshold(15); // effective 16, won't fire
        let input: Vec<i32> = (0..8).map(|i| if i < 3 { 2 } else { 0 }).collect();
        net.propagate(&input, &PropagationConfig::default()).unwrap();
        // At least charge or activation should be non-zero
        let has_state = net.spike_data().iter().any(|s| s.charge > 0) || net.activation().iter().any(|&a| a != 0);
        assert!(has_state, "propagation should leave some non-zero state");

        let path = std::env::temp_dir().join("instnct_test_ephemeral.bin");
        net.save_genome(&path).unwrap();
        let loaded = Network::load_genome(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(loaded.activation().iter().all(|&a| a == 0));
        assert!(loaded.spike_data().iter().all(|s| s.charge == 0));
    }

    #[test]
    fn genome_load_then_propagate_matches_fresh() {
        let mut net = Network::new(8);
        let mut rng = StdRng::seed_from_u64(77);
        for _ in 0..10 {
            net.mutate_add_edge(&mut rng);
        }
        net.set_threshold(0, 2);
        net.set_channel(1, 3);

        let path = std::env::temp_dir().join("instnct_test_propagate.bin");
        net.save_genome(&path).unwrap();
        let mut loaded = Network::load_genome(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let input: Vec<i32> = (0..8).map(|i| if i < 3 { 1 } else { 0 }).collect();
        let config = PropagationConfig::default();
        net.reset();
        net.propagate(&input, &config).unwrap();
        loaded.propagate(&input, &config).unwrap();

        assert_eq!(loaded.spike_data(), net.spike_data());
        assert_eq!(loaded.activation(), net.activation());
    }

    #[test]
    fn genome_load_rejects_edge_length_mismatch() {
        // Craft a malformed file: sources.len != targets.len
        let dto = disk::NetworkDiskV1 {
            version: disk::CURRENT_VERSION,
            graph: disk::ConnectionGraphDiskV1 {
                neuron_count: 4,
                sources: vec![0, 1],
                targets: vec![1],  // mismatch!
            },
            threshold: vec![0; 4],
            channel: vec![1; 4],
            polarity: vec![1; 4],
        };
        let bytes = bincode::serialize(&dto).unwrap();
        let path = std::env::temp_dir().join("instnct_test_malformed.bin");
        std::fs::write(&path, bytes).unwrap();

        let result = Network::load_genome(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("mismatch"), "error should mention mismatch: {err}");
    }

    #[test]
    fn genome_load_rejects_out_of_bounds_endpoint() {
        let dto = disk::NetworkDiskV1 {
            version: disk::CURRENT_VERSION,
            graph: disk::ConnectionGraphDiskV1 {
                neuron_count: 4,
                sources: vec![0, 99],  // 99 >= 4
                targets: vec![1, 2],
            },
            threshold: vec![0; 4],
            channel: vec![1; 4],
            polarity: vec![1; 4],
        };
        let bytes = bincode::serialize(&dto).unwrap();
        let path = std::env::temp_dir().join("instnct_test_oob.bin");
        std::fs::write(&path, bytes).unwrap();

        let result = Network::load_genome(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("source 99"), "error should mention source 99: {err}");
    }

    #[test]
    fn genome_load_rejects_duplicate_edge() {
        let dto = disk::NetworkDiskV1 {
            version: disk::CURRENT_VERSION,
            graph: disk::ConnectionGraphDiskV1 {
                neuron_count: 4,
                sources: vec![0, 0],
                targets: vec![1, 1], // duplicate (0,1)
            },
            threshold: vec![0; 4],
            channel: vec![1; 4],
            polarity: vec![1; 4],
        };
        let bytes = bincode::serialize(&dto).unwrap();
        let path = std::env::temp_dir().join("instnct_test_dup_edge.bin");
        std::fs::write(&path, bytes).unwrap();

        let result = Network::load_genome(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("duplicate"), "error should mention duplicate: {err}");
    }

    // ---- Spike skip-inactive adversarial tests ----

    #[test]
    fn spike_skip_matches_full_scan_on_dense_network() {
        // Dense network where almost every neuron is active.
        // Verifies skip-inactive produces deterministic results.
        let mut rng = StdRng::seed_from_u64(0xDE45E);
        let mut net = Network::new(128);
        for _ in 0..2000 {
            net.mutate_add_edge(&mut rng);
        }
        // Set low thresholds so everyone fires
        for i in 0..128 {
            net.spike_data_mut()[i].threshold = 0;
        }
        let input: Vec<i32> = (0..128).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        let config = PropagationConfig::default();

        // Run and capture state
        net.propagate(&input, &config).unwrap();
        let act1 = net.activation().to_vec();
        let charge1: Vec<u8> = net.spike_data().iter().map(|s| s.charge).collect();

        // Run again from same initial state — should be deterministic
        net.reset();
        net.propagate(&input, &config).unwrap();
        let act2 = net.activation().to_vec();
        let charge2: Vec<u8> = net.spike_data().iter().map(|s| s.charge).collect();

        assert_eq!(act1, act2, "spike skip not deterministic on dense network");
        assert_eq!(charge1, charge2);
    }

    #[test]
    fn spike_skip_matches_full_scan_on_sparse_network() {
        // Very sparse network — the target scenario for skip-inactive optimization
        let mut rng = StdRng::seed_from_u64(0x5BAB5E);
        let mut net = Network::new(4096);
        // Only 30 edges
        for _ in 0..30 {
            net.mutate_add_edge(&mut rng);
        }
        let input: Vec<i32> = {
            let mut v = vec![0i32; 4096];
            v[0] = 1;
            v[100] = 1;
            v[200] = 1;
            v
        };
        let config = PropagationConfig::default();

        net.propagate(&input, &config).unwrap();
        // Verify most neurons are inactive
        let active_count = net.activation().iter().filter(|&&a| a != 0).count();
        // With 30 edges and 3 input neurons, very few should fire
        assert!(
            active_count < 100,
            "too many active neurons in sparse network: {}",
            active_count
        );
    }

    #[test]
    fn spike_skip_zero_charge_never_fires() {
        // Verify the core invariant: charge=0 neurons CANNOT fire
        // regardless of threshold, channel, or phase tick.
        // effective threshold = (0+1) * phase_mult >= 7, and charge*10 = 0
        let mut net = Network::new(8);
        net.graph_mut().add_edge(0, 1);
        for i in 0..8 {
            net.spike_data_mut()[i].charge = 0;
            net.spike_data_mut()[i].threshold = 0;
        }
        // Don't inject any input
        let input = vec![0i32; 8];
        let config = PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 0,
            decay_interval_ticks: 0,
            use_refractory: false,
        };
        net.propagate(&input, &config).unwrap();
        assert!(
            net.activation().iter().all(|&a| a == 0),
            "charge=0 neuron fired!"
        );
    }

    #[test]
    fn spike_skip_preserves_semantics_across_random_mutations() {
        // Compare skip-inactive Network::propagate vs standalone propagate_token.
        // Focused on sparse networks.
        use crate::propagation::{propagate_token, PropagationParameters, PropagationState};
        use rand::Rng;

        let mut rng = StdRng::seed_from_u64(0xADD7EB5);
        let mut net = Network::new(512);
        // Add only ~50 edges (sparse)
        for _ in 0..50 {
            net.mutate_add_edge(&mut rng);
        }

        let input: Vec<i32> = (0..512)
            .map(|_| if rng.gen_bool(0.05) { 1 } else { 0 })
            .collect();
        let config = PropagationConfig::default();

        // Network path (CSR + spike skip)
        let net_clone = net.clone();
        net.propagate(&input, &config).unwrap();

        // Standalone path (propagate_token)
        let threshold: Vec<u8> = net_clone.spike_data().iter().map(|s| s.threshold).collect();
        let channel: Vec<u8> = net_clone.spike_data().iter().map(|s| s.channel).collect();
        let mut activation = vec![0i8; 512];
        let mut charge = vec![0u8; 512];
        let mut refractory = vec![0u8; 512];
        let mut workspace = PropagationWorkspace::new(512);

        propagate_token(
            &input,
            net_clone.graph(),
            &PropagationParameters {
                threshold: &threshold,
                channel: &channel,
                polarity: net_clone.polarity(),
            },
            &mut PropagationState {
                activation: &mut activation,
                charge: &mut charge,
                refractory: &mut refractory,
            },
            &config,
            &mut workspace,
        )
        .unwrap();

        // Must match exactly
        assert_eq!(
            net.activation(),
            &activation[..],
            "activation diverged: Network vs propagate_token"
        );
        let net_charge: Vec<u8> = net.spike_data().iter().map(|s| s.charge).collect();
        assert_eq!(
            net_charge, charge,
            "charge diverged: Network vs propagate_token"
        );
    }

    #[test]
    fn spike_skip_handles_refractory_correctly() {
        // With refractory enabled, a neuron that fires gets 1-tick cooldown.
        // The skip must still process refractory neurons (they need decrement).
        let mut net = Network::new(4);
        net.graph_mut().add_edge(0, 1);
        net.graph_mut().add_edge(1, 2);
        for i in 0..4 {
            net.spike_data_mut()[i].threshold = 0; // easy to fire
        }

        let input = vec![1i32, 0, 0, 0];
        let config = PropagationConfig {
            ticks_per_token: 4,
            input_duration_ticks: 1,
            decay_interval_ticks: 0,
            use_refractory: true,
        };

        // Should not panic and should produce valid state
        net.propagate(&input, &config).unwrap();
        // Neuron 0 received input, should have fired and entered refractory
        // All charges must be within bounds
        assert!(net.spike_data().iter().all(|s| s.charge <= LIMIT_MAX_CHARGE));
    }

    // ---- Sparse tick adversarial tests ----

    #[test]
    fn sparse_tick_matches_dense_on_random_networks() {
        // Proves sparse tick (CSR path) produces identical results to dense
        // standalone propagate_token across many random configurations.
        use crate::propagation::{propagate_token, PropagationParameters, PropagationState};
        use rand::Rng;

        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let neuron_count = rng.gen_range(4..=200);
            let mut net = Network::new(neuron_count);

            // Random edges (sparse to dense)
            let edge_count = rng.gen_range(0..neuron_count * 2);
            for _ in 0..edge_count {
                net.mutate_add_edge(&mut rng);
            }

            // Random parameters
            for i in 0..neuron_count {
                net.spike_data_mut()[i].threshold = rng.gen_range(0..=15);
                net.spike_data_mut()[i].channel = rng.gen_range(1..=8);
            }
            for i in 0..neuron_count {
                if rng.gen_bool(0.2) {
                    net.polarity_mut()[i] = -1;
                }
            }

            // Random input
            let input: Vec<i32> = (0..neuron_count).map(|_| rng.gen_range(-1..=1)).collect();
            let config = PropagationConfig {
                ticks_per_token: rng.gen_range(1..=12),
                input_duration_ticks: rng.gen_range(0..=3),
                decay_interval_ticks: rng.gen_range(0..=8),
                use_refractory: rng.gen_bool(0.3),
            };

            // Run Network::propagate (uses sparse tick + CSR)
            let mut net_sparse = net.clone();
            net_sparse.propagate(&input, &config).unwrap();

            // Run standalone propagate_token (reference implementation)
            let threshold: Vec<u8> = net.spike_data().iter().map(|s| s.threshold).collect();
            let channel: Vec<u8> = net.spike_data().iter().map(|s| s.channel).collect();
            let mut act_ref = vec![0i8; neuron_count];
            let mut charge_ref = vec![0u8; neuron_count];
            let mut refr_ref = vec![0u8; neuron_count];
            let mut ws = PropagationWorkspace::new(neuron_count);

            propagate_token(
                &input,
                net.graph(),
                &PropagationParameters {
                    threshold: &threshold,
                    channel: &channel,
                    polarity: net.polarity(),
                },
                &mut PropagationState {
                    activation: &mut act_ref,
                    charge: &mut charge_ref,
                    refractory: &mut refr_ref,
                },
                &config,
                &mut ws,
            )
            .unwrap();

            // MUST MATCH EXACTLY
            assert_eq!(
                net_sparse.activation(),
                &act_ref[..],
                "activation diverged at seed={seed}, H={neuron_count}, E={}",
                net.edge_count()
            );
            let sparse_charge: Vec<u8> =
                net_sparse.spike_data().iter().map(|s| s.charge).collect();
            assert_eq!(
                sparse_charge, charge_ref,
                "charge diverged at seed={seed}, H={neuron_count}, E={}",
                net.edge_count()
            );
        }
    }

    #[test]
    fn dirty_set_shrinks_when_activity_stops() {
        let mut net = Network::new(1000);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            net.mutate_add_edge(&mut rng);
        }

        // Inject input once
        let mut input = vec![0i32; 1000];
        input[0] = 1;
        input[50] = 1;
        input[100] = 1;
        let config = PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 1,
            decay_interval_ticks: 1,
            use_refractory: false,
        };

        // First propagation creates activity
        net.propagate(&input, &config).unwrap();

        // Subsequent propagations with zero input should let activity die
        let zero_input = vec![0i32; 1000];
        for _ in 0..50 {
            net.propagate(&zero_input, &config).unwrap();
        }

        // After 50 tokens with no input and decay=1, dirty set should be empty
        assert_eq!(
            net.dirty_set_len(),
            0,
            "dirty set should be empty after extended quiescence, got {}",
            net.dirty_set_len()
        );
    }

    #[test]
    fn sparse_tick_empty_network() {
        let mut net = Network::new(0);
        let config = PropagationConfig::default();
        net.propagate(&[], &config).unwrap();
        // Should not panic
    }

    #[test]
    fn sparse_tick_all_neurons_fire() {
        let mut net = Network::new(100);
        let mut rng = StdRng::seed_from_u64(0xF);
        // Dense: add lots of edges
        for _ in 0..5000 {
            net.mutate_add_edge(&mut rng);
        }
        // Low threshold so everyone fires
        for i in 0..100 {
            net.spike_data_mut()[i].threshold = 0;
        }

        let input: Vec<i32> = vec![1; 100];
        let config = PropagationConfig::default();
        net.propagate(&input, &config).unwrap();

        // Most neurons should be active
        let active = net.activation().iter().filter(|&&a| a != 0).count();
        assert!(
            active > 50,
            "expected most neurons active in dense network, got {}",
            active
        );
    }

    #[test]
    fn decay_only_affects_charged_neurons_sparse() {
        let mut net = Network::new(100);

        let mut input = vec![0i32; 100];
        input[5] = 1;
        let config = PropagationConfig {
            ticks_per_token: 1,
            input_duration_ticks: 1,
            decay_interval_ticks: 1,
            use_refractory: false,
        };

        // Propagate to establish activity at neuron 5
        net.propagate(&input, &config).unwrap();

        // Now run with zero input, charge should decay
        let zero_input = vec![0i32; 100];
        for _ in 0..20 {
            net.propagate(&zero_input, &config).unwrap();
        }

        // All charge should be 0 after enough decay
        for i in 0..100 {
            assert_eq!(
                net.spike_data()[i].charge, 0,
                "neuron {} should have charge 0 after decay",
                i
            );
        }
    }

    #[test]
    fn sparse_propagate_matches_dense() {
        // Build a network with some edges and varied parameters
        let mut rng = StdRng::seed_from_u64(42);
        let neuron_count = 64;
        let mut net = Network::new(neuron_count);
        for _ in 0..50 {
            net.mutate_add_edge(&mut rng);
        }
        for i in 0..neuron_count {
            net.spike_data_mut()[i].threshold = rng.gen_range(0..=7);
            net.spike_data_mut()[i].channel = rng.gen_range(1..=8);
        }

        // Build a dense input pattern with some active neurons
        let mut dense_input = vec![0i32; neuron_count];
        let active_indices: Vec<u16> = vec![3, 7, 12, 25, 40, 55];
        let active_values: Vec<i8> = vec![1, 1, 1, 1, 1, 1];
        for (&idx, &val) in active_indices.iter().zip(active_values.iter()) {
            dense_input[idx as usize] = val as i32;
        }

        let config = default_config();

        // Run dense propagation
        let mut net_dense = net.clone();
        net_dense.propagate(&dense_input, &config).unwrap();
        let dense_charges: Vec<u8> = net_dense.spike_data().iter().map(|s| s.charge).collect();
        let dense_activation: Vec<i8> = net_dense.activation().to_vec();

        // Run sparse propagation on a fresh clone
        let mut net_sparse = net.clone();
        net_sparse
            .propagate_sparse(&active_indices, &active_values, &config)
            .unwrap();
        let sparse_charges: Vec<u8> = net_sparse.spike_data().iter().map(|s| s.charge).collect();
        let sparse_activation: Vec<i8> = net_sparse.activation().to_vec();

        // They must be identical
        assert_eq!(
            dense_charges, sparse_charges,
            "sparse propagation produced different charges than dense"
        );
        assert_eq!(
            dense_activation, sparse_activation,
            "sparse propagation produced different activation than dense"
        );
    }

    #[test]
    fn mutation_undo_restores_topology_and_params() {
        let mut rng = StdRng::seed_from_u64(42);
        let neuron_count = 64;
        let mut net = Network::new(neuron_count);
        // Seed with some edges
        for _ in 0..50 {
            net.mutate_add_edge(&mut rng);
        }
        for i in 0..neuron_count {
            net.spike_data_mut()[i].threshold = rng.gen_range(0..=7);
            net.spike_data_mut()[i].channel = rng.gen_range(1..=8);
        }

        // Test each undo mutation type
        let mutation_fns: Vec<Box<dyn Fn(&mut Network, &mut StdRng) -> (bool, MutationUndo)>> = vec![
            Box::new(|net, rng| net.mutate_add_edge_undo(rng)),
            Box::new(|net, rng| net.mutate_remove_edge_undo(rng)),
            Box::new(|net, rng| net.mutate_rewire_undo(rng)),
            Box::new(|net, rng| net.mutate_reverse_undo(rng)),
            Box::new(|net, rng| net.mutate_mirror_undo(rng)),
            Box::new(|net, rng| net.mutate_enhance_undo(rng)),
            Box::new(|net, rng| net.mutate_add_loop_undo(rng, 2)),
            Box::new(|net, rng| net.mutate_add_loop_undo(rng, 3)),
            Box::new(|net, rng| net.mutate_theta_undo(rng)),
            Box::new(|net, rng| net.mutate_channel_undo(rng)),
            Box::new(|net, rng| net.mutate_polarity_undo(rng)),
        ];

        for (fn_idx, mutate_fn) in mutation_fns.iter().enumerate() {
            // Try several times to get a successful mutation
            for attempt in 0..20 {
                let mut test_net = net.clone();
                let edges_before: Vec<_> = test_net.graph().iter_edges().collect();
                let spike_before: Vec<SpikeData> = test_net.spike_data().to_vec();
                let polarity_before: Vec<i8> = test_net.polarity().to_vec();

                let mut attempt_rng = StdRng::seed_from_u64(1000 + fn_idx as u64 * 100 + attempt);
                let (mutated, undo) = mutate_fn(&mut test_net, &mut attempt_rng);
                if !mutated {
                    continue;
                }

                // Apply undo
                test_net.apply_undo(&undo);

                // Verify topology restored
                let edges_after: Vec<_> = test_net.graph().iter_edges().collect();
                assert_eq!(
                    edges_before.len(),
                    edges_after.len(),
                    "mutation fn {fn_idx}: edge count mismatch after undo"
                );
                // Edge sets should match (order may differ for swap-remove mutations)
                let mut before_sorted: Vec<_> = edges_before
                    .iter()
                    .map(|e| (e.source, e.target))
                    .collect();
                let mut after_sorted: Vec<_> = edges_after
                    .iter()
                    .map(|e| (e.source, e.target))
                    .collect();
                before_sorted.sort();
                after_sorted.sort();
                assert_eq!(
                    before_sorted, after_sorted,
                    "mutation fn {fn_idx}: edge set mismatch after undo"
                );

                // Verify parameters restored
                assert_eq!(
                    test_net.spike_data().iter().map(|s| s.threshold).collect::<Vec<_>>(),
                    spike_before.iter().map(|s| s.threshold).collect::<Vec<_>>(),
                    "mutation fn {fn_idx}: threshold mismatch after undo"
                );
                assert_eq!(
                    test_net.spike_data().iter().map(|s| s.channel).collect::<Vec<_>>(),
                    spike_before.iter().map(|s| s.channel).collect::<Vec<_>>(),
                    "mutation fn {fn_idx}: channel mismatch after undo"
                );
                assert_eq!(
                    test_net.polarity(),
                    polarity_before.as_slice(),
                    "mutation fn {fn_idx}: polarity mismatch after undo"
                );

                break; // success — move to next mutation type
            }
        }
    }
}
