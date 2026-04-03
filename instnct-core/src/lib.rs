//! # INSTNCT Core
//!
//! Gradient-free self-wiring spiking network engine.
//!
//! ## Architectural Overview
//!
//! INSTNCT learns by **mutating its own directed graph topology** rather than
//! adjusting continuous weights via backpropagation. The network is a recurrent
//! substrate of spiking neurons connected by directed binary edges. At each
//! time step ("tick"), neurons accumulate incoming charge via scatter-add,
//! compare against a per-neuron wave-gated threshold, and emit binary spikes
//! that propagate along outgoing edges.
//!
//! ### Key Design Choices
//!
//! - **Passive I/O**: Input and output projections are fixed random matrices,
//!   not learned. All learning occurs in the hidden graph.
//! - **Quaternary edge encoding**: Each neuron pair stores one of four states
//!   (none / forward / backward / bidirectional) in 2 bits, halving memory
//!   compared to a full boolean adjacency matrix.
//! - **Multiply-free propagation**: Spikes are binary (+1/-1 via polarity),
//!   so the forward pass core loop is pure scatter-add with no floating-point
//!   multiplies.
//! - **Evolution, not gradient descent**: Training proceeds by proposing
//!   single-edge mutations, evaluating fitness on text data, and accepting
//!   improvements. Crystallization (greedy pruning) removes dead-weight edges.
//!
//! ## Module Index
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`parameters`] | Centralized hyperparameter registry — single source of truth |
//! | [`topology`] | Quaternary connection mask — the learnable graph structure |
//! | [`propagation`] | Spiking forward pass — the performance-critical inner loop |

pub mod parameters;
pub mod topology;
pub mod propagation;
