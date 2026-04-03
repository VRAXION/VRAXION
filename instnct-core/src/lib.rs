//! INSTNCT Core — gradient-free self-wiring spiking network engine.
//!
//! # Architecture
//!
//! The network is a directed graph of spiking neurons. Connections are stored
//! in a quaternary upper-triangle mask (0=none, 1=fwd, 2=bwd, 3=bidir).
//! Learning happens through mutation + selection, not gradient descent.
//!
//! # Modules
//!
//! - [`quaternary_mask`]: Connection storage with 50% memory vs bool matrix
//! - [`forward`]: Spiking forward pass (the hot path)
//! - [`mutations`]: Graph mutations (add, remove, reverse, flip, loop3..8)
//! - [`eval`]: Bigram cosine fitness evaluation
//! - [`crystal`]: Greedy edge pruning

pub mod quaternary_mask;
pub mod forward;
