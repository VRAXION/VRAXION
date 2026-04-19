//! Block A — L0 Byte Unit encoder/decoder.
//!
//! Binary + C19 + H=16 tied-mirror autoencoder; 100% lossless on all 256 bytes.
//! Reads the champion artifacts from `output/byte_unit_champion_binary_c19_h16/`.

pub mod byte_encoder;

pub use byte_encoder::ByteEncoder;
