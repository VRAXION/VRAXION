//! VRAXION Rust deploy SDK.
//!
//! Minimal, deploy-ready Rust API over the frozen byte-level pipeline.
//!
//! Currently exposes:
//!   - [`ByteEncoder`] — Block A (byte unit, L0, 8 → 16 → 8 tied-mirror autoencoder)
//!   - [`L1Merger`]    — Block B (byte-pair merger, L1, 32 → 81 → 32 single-W mirror)
//!
//! See also the parallel `Python/` deploy SDK; both read the same champion
//! artifacts from the repo's `output/` directory.

pub mod block_a_byte_unit;
pub mod block_b_merger;

pub use block_a_byte_unit::ByteEncoder;
pub use block_b_merger::L1Merger;
