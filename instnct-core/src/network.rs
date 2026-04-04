//! # Network — Self-Contained Spiking Network
//!
//! Bundles a [`ConnectionGraph`], per-neuron learned parameters (threshold,
//! channel, polarity), ephemeral tick state (activation, charge), and a
//! reusable scratch workspace into one coherent object.
//!
//! This is the primary user-facing type for the crate. Create a `Network`,
//! wire edges via [`graph_mut()`](Network::graph_mut), adjust parameters,
//! then call [`propagate()`](Network::propagate) to simulate one token.

use crate::propagation::{
    propagate_token, PropagationConfig, PropagationError, PropagationParameters, PropagationState,
    PropagationWorkspace,
};
use crate::topology::ConnectionGraph;
use std::error::Error;
use std::fmt;

// ---- Error ----

/// Errors from [`Network`] operations.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NetworkError {
    /// Propagation validation failed (slice lengths, value ranges, edge bounds).
    Propagation(PropagationError),
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Propagation(e) => write!(f, "propagation error: {e}"),
        }
    }
}

impl Error for NetworkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Propagation(e) => Some(e),
        }
    }
}

impl From<PropagationError> for NetworkError {
    fn from(e: PropagationError) -> Self {
        Self::Propagation(e)
    }
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
    threshold: Vec<u32>, // per-neuron, stored [0,15]; effective = stored+1 → [1,16]
    channel: Vec<u8>,    // per-neuron, phase gating channel [1,8]
    polarity: Vec<i32>,  // per-neuron, +1 excitatory, -1 inhibitory
    activation: Vec<i32>, // ephemeral, +1, -1, or 0
    charge: Vec<u32>,    // ephemeral, [0, LIMIT_MAX_CHARGE]
    workspace: PropagationWorkspace, // reusable scratch buffer
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
            threshold: vec![0u32; neuron_count],
            channel: vec![1u8; neuron_count],
            polarity: vec![1i32; neuron_count],
            activation: vec![0i32; neuron_count],
            charge: vec![0u32; neuron_count],
            workspace: PropagationWorkspace::new(neuron_count),
        }
    }

    // ---- Propagation ----

    /// Run one token's forward pass through the spiking network.
    ///
    /// Delegates to the checked [`propagate_token`](crate::propagate_token),
    /// which validates all slice lengths, parameter ranges, and edge bounds.
    /// This ensures safety even when parameters are modified via `_mut()` accessors.
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
        propagate_token(
            input,
            &self.graph,
            &PropagationParameters {
                threshold: &self.threshold,
                channel: &self.channel,
                polarity: &self.polarity,
            },
            &mut PropagationState {
                activation: &mut self.activation,
                charge: &mut self.charge,
            },
            config,
            &mut self.workspace,
        )?;
        Ok(())
    }

    // ---- State management ----

    /// Zero all activation and charge values. Does not touch topology or learned parameters.
    pub fn reset(&mut self) {
        self.activation.fill(0);
        self.charge.fill(0);
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
    #[inline]
    pub fn graph_mut(&mut self) -> &mut ConnectionGraph {
        &mut self.graph
    }

    // ---- Parameter access ----

    /// Per-neuron threshold values (stored range `[0,15]`, effective = stored+1).
    #[inline]
    pub fn threshold(&self) -> &[u32] {
        &self.threshold
    }

    /// Mutable access to per-neuron thresholds. Valid range: `[0,15]`.
    #[inline]
    pub fn threshold_mut(&mut self) -> &mut [u32] {
        &mut self.threshold
    }

    /// Per-neuron phase gating channel. Valid range: `[1,8]`.
    #[inline]
    pub fn channel(&self) -> &[u8] {
        &self.channel
    }

    /// Mutable access to per-neuron channels. Valid range: `[1,8]`.
    #[inline]
    pub fn channel_mut(&mut self) -> &mut [u8] {
        &mut self.channel
    }

    /// Per-neuron polarity (`+1` excitatory, `-1` inhibitory).
    #[inline]
    pub fn polarity(&self) -> &[i32] {
        &self.polarity
    }

    /// Mutable access to per-neuron polarity. Valid values: `+1` or `-1`.
    #[inline]
    pub fn polarity_mut(&mut self) -> &mut [i32] {
        &mut self.polarity
    }

    // ---- State access (read-only) ----

    /// Current activation values (read-only). Values: `+1`, `-1`, or `0`.
    #[inline]
    pub fn activation(&self) -> &[i32] {
        &self.activation
    }

    /// Current charge values (read-only). Range: `[0, LIMIT_MAX_CHARGE]` (currently 15).
    #[inline]
    pub fn charge(&self) -> &[u32] {
        &self.charge
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::LIMIT_MAX_CHARGE;
    use crate::propagation::PropagationError;

    fn default_config() -> PropagationConfig {
        PropagationConfig::default()
    }

    #[test]
    fn new_creates_correct_defaults() {
        let net = Network::new(256);
        assert_eq!(net.neuron_count(), 256);
        assert_eq!(net.edge_count(), 0);
        assert!(net.threshold().iter().all(|&t| t == 0));
        assert!(net.channel().iter().all(|&c| c == 1));
        assert!(net.polarity().iter().all(|&p| p == 1));
        assert!(net.activation().iter().all(|&a| a == 0));
        assert!(net.charge().iter().all(|&c| c == 0));
    }

    #[test]
    fn new_zero_neurons() {
        let net = Network::new(0);
        assert_eq!(net.neuron_count(), 0);
        assert_eq!(net.edge_count(), 0);
        assert!(net.threshold().is_empty());
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
        net.threshold_mut()[0] = 5;

        let mut input = vec![0i32; 64];
        input[0] = 1;
        net.propagate(&input, &default_config()).unwrap();

        net.reset();
        assert!(net.activation().iter().all(|&a| a == 0));
        assert!(net.charge().iter().all(|&c| c == 0));
        // parameters and edges untouched
        assert_eq!(net.threshold()[0], 5);
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
    fn threshold_mut_works() {
        let mut net = Network::new(16);
        net.threshold_mut()[5] = 12;
        assert_eq!(net.threshold()[5], 12);
    }

    #[test]
    fn channel_mut_works() {
        let mut net = Network::new(16);
        net.channel_mut()[3] = 7;
        assert_eq!(net.channel()[3], 7);
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
        };
        net.propagate(&input, &config).unwrap();

        assert!(
            net.charge().iter().all(|&c| c <= LIMIT_MAX_CHARGE),
            "charge must not exceed LIMIT_MAX_CHARGE"
        );
    }

    #[test]
    fn multiple_propagate_accumulates() {
        // High threshold (effective=6) so neurons accumulate charge without firing.
        // After 2 ticks with edge 0→1 and input at 0: charge[1] increases by 2 per propagate.
        let mut net = Network::new(4);
        net.graph_mut().add_edge(0, 1);
        net.threshold_mut().fill(5); // effective = 6

        let input = vec![1i32, 0, 0, 0];
        let config = PropagationConfig {
            ticks_per_token: 2,
            input_duration_ticks: 2,
            decay_interval_ticks: 0,
        };

        net.propagate(&input, &config).unwrap();
        let charge_after_one = net.charge()[1];

        // Second propagation without reset — charge carries over
        net.propagate(&input, &config).unwrap();
        let charge_after_two = net.charge()[1];

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
        let after_reset = (net.activation().to_vec(), net.charge().to_vec());

        // Second: fresh network, propagate once
        let mut fresh = Network::new(4);
        fresh.graph_mut().add_edge(0, 1);
        fresh.graph_mut().add_edge(1, 2);
        fresh.propagate(&input, &config).unwrap();
        let after_fresh = (fresh.activation().to_vec(), fresh.charge().to_vec());

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
        net.threshold_mut()[0] = 999; // out of [0,15]
        let input = vec![0i32; 4];
        let err = net.propagate(&input, &default_config()).unwrap_err();
        assert!(
            matches!(
                err,
                NetworkError::Propagation(PropagationError::ThresholdOutOfRange {
                    index: 0,
                    value: 999
                })
            ),
            "expected ThresholdOutOfRange, got: {err}"
        );
    }

    #[test]
    fn propagate_rejects_invalid_channel() {
        let mut net = Network::new(4);
        net.channel_mut()[2] = 0; // out of [1,8]
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
        net.polarity_mut()[1] = 42; // not ±1
        let input = vec![0i32; 4];
        let err = net.propagate(&input, &default_config()).unwrap_err();
        assert!(
            matches!(
                err,
                NetworkError::Propagation(PropagationError::PolarityOutOfRange {
                    index: 1,
                    value: 42
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
        };
        net.propagate(&input, &config).unwrap();
        // Zero ticks = no simulation, state unchanged
        assert!(net.activation().iter().all(|&a| a == 0));
        assert!(net.charge().iter().all(|&c| c == 0));
    }

    // --- Deep research finding: Network should be Clone ---

    #[test]
    fn network_clone_is_independent() {
        let mut net = Network::new(4);
        net.graph_mut().add_edge(0, 1);
        net.threshold_mut()[0] = 5;

        let mut cloned = net.clone();
        cloned.threshold_mut()[0] = 10;
        cloned.graph_mut().add_edge(2, 3);

        // Original unaffected
        assert_eq!(net.threshold()[0], 5);
        assert_eq!(net.edge_count(), 1);
        // Clone has its own state
        assert_eq!(cloned.threshold()[0], 10);
        assert_eq!(cloned.edge_count(), 2);
    }
}
