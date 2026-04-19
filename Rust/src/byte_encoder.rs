//! VRAXION byte encoder (Block A) — Rust deploy SDK.
//!
//! Reads the frozen binary+C19+H=16 champion LUT and provides:
//!
//! ```ignore
//! use vraxion::ByteEncoder;
//! let enc = ByteEncoder::load_default()?;
//! let vec = enc.encode(0x41);            // 16-dim f32
//! let byte_back = enc.decode(&vec);      // recover byte
//! ```
//!
//! The encoder uses the baked 256-entry int8 LUT. The decoder uses the saved
//! binary weight matrices (W1, W2) from the winner JSON.
//!
//! 100% lossless on all 256 bytes. Zero ML framework dependency — pure Rust.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

pub const LUT_DIM: usize = 16;
pub const INPUT_BITS: usize = 8;
pub const HIDDEN: usize = 16;

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Parse(serde_json::Error),
    BadShape(&'static str),
    BadFormat(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Parse(e) => write!(f, "parse: {e}"),
            Self::BadShape(m) => write!(f, "bad shape: {m}"),
            Self::BadFormat(m) => write!(f, "bad format: {m}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for LoadError {
    fn from(e: serde_json::Error) -> Self {
        Self::Parse(e)
    }
}

#[derive(Deserialize)]
struct LutBlob {
    format: String,
    scale: f32,
    lut: Vec<Vec<i8>>,
}

#[derive(Deserialize)]
struct WeightsBlob {
    precision: String,
    #[serde(rename = "W1_binary_idx")]
    w1_idx: Vec<Vec<i64>>,
    #[serde(rename = "W1_levels")]
    w1_levels: Vec<f32>,
    #[serde(rename = "W2_binary_idx")]
    w2_idx: Vec<Vec<i64>>,
    #[serde(rename = "W2_levels")]
    w2_levels: Vec<f32>,
}

/// Deploy-ready byte encoder for Block A (binary + C19 + H=16, 100% lossless).
pub struct ByteEncoder {
    /// Decoded float LUT (256 × LUT_DIM). Row `b` is the latent for byte `b`.
    lut_f32: [[f32; LUT_DIM]; 256],
    /// W1 weights (INPUT_BITS × HIDDEN), already multiplied by alpha1.
    w1: [[f32; HIDDEN]; INPUT_BITS],
    /// W2 weights (HIDDEN × LUT_DIM), already multiplied by alpha2.
    w2: [[f32; LUT_DIM]; HIDDEN],
}

impl ByteEncoder {
    /// Load from the repo's default champion paths.
    ///
    /// Resolves relative to the crate directory (`CARGO_MANIFEST_DIR/..`) so
    /// `cargo test` works from any working directory. For runtime deploy that
    /// ships the champion artifacts next to the binary, use [`Self::from_paths`].
    pub fn load_default() -> Result<Self, LoadError> {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("Rust/ parent")
            .to_path_buf();
        let base = repo_root.join("output").join("byte_unit_champion_binary_c19_h16");
        Self::from_paths(
            base.join("byte_embedder_lut_int8.json"),
            base.join("byte_unit_winner_binary.json"),
        )
    }

    /// Load from explicit paths.
    pub fn from_paths<P: AsRef<Path>, Q: AsRef<Path>>(
        lut_path: P,
        weights_path: Q,
    ) -> Result<Self, LoadError> {
        let lut_text = fs::read_to_string(lut_path.as_ref())?;
        let lut: LutBlob = serde_json::from_str(&lut_text)?;
        if lut.format != "int8_lut" {
            return Err(LoadError::BadFormat(format!(
                "expected int8_lut, got {}",
                lut.format
            )));
        }
        if lut.lut.len() != 256 {
            return Err(LoadError::BadShape("LUT rows != 256"));
        }
        if lut.lut[0].len() != LUT_DIM {
            return Err(LoadError::BadShape("LUT cols != 16"));
        }

        let mut lut_f32 = [[0f32; LUT_DIM]; 256];
        for (b, row) in lut.lut.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                lut_f32[b][j] = (v as f32) * lut.scale;
            }
        }

        let w_text = fs::read_to_string(weights_path.as_ref())?;
        let w: WeightsBlob = serde_json::from_str(&w_text)?;
        if w.precision != "binary_scaled" {
            return Err(LoadError::BadFormat(format!(
                "expected binary_scaled weights, got {}",
                w.precision
            )));
        }
        if w.w1_idx.len() != INPUT_BITS || w.w1_idx[0].len() != HIDDEN {
            return Err(LoadError::BadShape("W1 shape != (8, 16)"));
        }
        if w.w2_idx.len() != HIDDEN || w.w2_idx[0].len() != LUT_DIM {
            return Err(LoadError::BadShape("W2 shape != (16, 16)"));
        }

        let mut w1 = [[0f32; HIDDEN]; INPUT_BITS];
        for i in 0..INPUT_BITS {
            for j in 0..HIDDEN {
                let idx = w.w1_idx[i][j] as usize;
                w1[i][j] = w.w1_levels[idx];
            }
        }
        let mut w2 = [[0f32; LUT_DIM]; HIDDEN];
        for i in 0..HIDDEN {
            for j in 0..LUT_DIM {
                let idx = w.w2_idx[i][j] as usize;
                w2[i][j] = w.w2_levels[idx];
            }
        }

        Ok(Self { lut_f32, w1, w2 })
    }

    /// Byte (0..=255) -> 16-dim f32 latent. O(1) LUT lookup.
    pub fn encode(&self, byte: u8) -> [f32; LUT_DIM] {
        self.lut_f32[byte as usize]
    }

    /// Vectorized encode: bytes -> matrix (one row per byte).
    pub fn encode_bytes(&self, data: &[u8]) -> Vec<[f32; LUT_DIM]> {
        data.iter().map(|&b| self.lut_f32[b as usize]).collect()
    }

    /// (16,) latent -> byte via tied-mirror decode: latent @ W2^T @ W1^T -> signs -> bits.
    pub fn decode(&self, latent: &[f32; LUT_DIM]) -> u8 {
        // Step 1: h = latent @ W2^T, shape (HIDDEN,)
        let mut h = [0f32; HIDDEN];
        for k in 0..HIDDEN {
            let mut acc = 0f32;
            for j in 0..LUT_DIM {
                acc += latent[j] * self.w2[k][j];
            }
            h[k] = acc;
        }

        // Step 2: x_hat = h @ W1^T, shape (INPUT_BITS,)
        let mut x_hat = [0f32; INPUT_BITS];
        for i in 0..INPUT_BITS {
            let mut acc = 0f32;
            for k in 0..HIDDEN {
                acc += h[k] * self.w1[i][k];
            }
            x_hat[i] = acc;
        }

        // Step 3: pack signs into byte (bit i = bit position i)
        let mut byte: u8 = 0;
        for i in 0..INPUT_BITS {
            if x_hat[i] > 0.0 {
                byte |= 1 << i;
            }
        }
        byte
    }

    /// Vectorized decode: latents matrix -> byte sequence.
    pub fn decode_bytes(&self, latents: &[[f32; LUT_DIM]]) -> Vec<u8> {
        latents.iter().map(|l| self.decode(l)).collect()
    }

    /// Encode all 256 bytes, decode them back, return (matches, 256).
    pub fn verify_lossless(&self) -> (usize, usize) {
        let mut matches = 0;
        for b in 0u16..256 {
            let vec = self.lut_f32[b as usize];
            let out = self.decode(&vec);
            if out == b as u8 {
                matches += 1;
            }
        }
        (matches, 256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_for_test() -> ByteEncoder {
        ByteEncoder::load_default().expect("load champion")
    }

    #[test]
    fn default_load_self_verifies_lossless() {
        let enc = load_for_test();
        let (m, t) = enc.verify_lossless();
        assert_eq!((m, t), (256, 256));
    }

    #[test]
    fn encode_shape_is_16_floats() {
        let enc = load_for_test();
        let v = enc.encode(0x41);
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn round_trip_all_256_bytes() {
        let enc = load_for_test();
        for b in 0u16..256 {
            let v = enc.encode(b as u8);
            let back = enc.decode(&v);
            assert_eq!(back, b as u8, "byte {b} round-tripped to {back}");
        }
    }

    #[test]
    fn vectorized_round_trip_ascii() {
        let enc = load_for_test();
        let text = b"The quick brown fox jumps over the lazy dog. 0123456789!@#$%^&*()";
        let latents = enc.encode_bytes(text);
        let back = enc.decode_bytes(&latents);
        assert_eq!(back, text);
    }

    #[test]
    fn vectorized_round_trip_utf8() {
        let enc = load_for_test();
        let text = "Péter szépen éneklő bárány 中文 العربية 🐈".as_bytes();
        let latents = enc.encode_bytes(text);
        let back = enc.decode_bytes(&latents);
        assert_eq!(back, text);
    }
}
