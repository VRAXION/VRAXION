//! VRAXION Rust deploy SDK.
//!
//! Minimal, deploy-ready Rust API over the frozen byte-level pipeline.
//!
//! Currently exposes:
//!   - [`ByteEncoder`] — Block A (byte unit, L0)
//!
//! See also the parallel `Python/` deploy SDK; both read the same champion
//! artifacts from the repo's `output/` directory.

pub mod byte_encoder;

pub use byte_encoder::ByteEncoder;
