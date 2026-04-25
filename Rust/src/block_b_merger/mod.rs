//! Block B — L1 Byte-Pair Merger (single-W mirror-tied).
//!
//! # Architecture
//!
//! Forward pass: `y = C19(x @ W + b1) @ W.T + b2`
//!
//! Single weight matrix W (32×81). Encoder and decoder share W entirely
//! (W-transpose for the return leg). C19 activation with per-channel `c`
//! and `rho` parameters.
//!
//! 100% lossless on all 65,536 byte pairs.
//!
//! # Load format
//!
//! Reads `output/merger_single_w_huffman_pack/packed_model.bin` — a 3,440 B
//! Huffman-packed binary produced by `diag_byte_single_w_huffman_pack.py`
//! (archived to tag `archives/tools-cleanup-20260425`).
//!
//! Format: magic `VGH1` + 5 components in order: W, b1, b2, c19_c, c19_rho.
//! Each non-raw component uses: G fp16 generators, mode bitmap (1=encoded),
//! sign bitmap for encoded cells, nibble-packed Huffman code lengths (coef
//! symbols 1..=7 and generator-index symbols 0..G-1), blob-length pair
//! (2×u16-le), Huffman-coded coef stream, Huffman-coded index stream, fp16
//! fallback stream for non-encoded cells.
//! `b2` is raw fp16 only (no Huffman layer).
//!
//! This is a direct Rust port of `unpack_component` from
//! `diag_byte_single_w_huffman_pack.py` (see archive tag noted above).
//! No JSON fallback needed;
//! the binary unpacks cleanly in ~50 lines.

pub mod merger;

pub use merger::L1Merger;
