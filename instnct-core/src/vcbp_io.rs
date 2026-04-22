//! VCBP packed binary reader — loads Block C byte-pair embeddings for direct
//! Brain I/O without SDR encoding or Int8Projection.
//!
//! Reads the `packed.bin` artifact (VCBP v1 format, ~62 KB) and provides:
//! - `embed_id(pair_id)` → E-dim float32 embedding
//! - `quantize_to_input(embedding)` → charge values [0, max_charge] for Brain input
//! - `dequantize_output(charges)` → approximate float32 embedding
//! - `nearest_hot(query)` → closest byte-pair ID via L2 nearest neighbor
//!
//! No ML framework dependency — pure Rust, no unsafe.

use std::fmt;
use std::fs;
use std::path::Path;

/// Errors from VCBP loading.
#[derive(Debug)]
pub enum VcbpError {
    /// File I/O error.
    Io(std::io::Error),
    /// Invalid format (bad magic, wrong version, etc.).
    Format(String),
}

impl fmt::Display for VcbpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "VCBP I/O error: {e}"),
            Self::Format(s) => write!(f, "VCBP format error: {s}"),
        }
    }
}

impl From<std::io::Error> for VcbpError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Decode an fp16 value (IEEE 754 half-precision) to f32.
fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        // subnormal or zero
        let val = f32::from_bits((sign << 31) | 0) + (frac as f32) * (2.0f32).powi(-24);
        if sign == 1 { -val.abs() } else { val }
    } else if exp == 31 {
        // inf / nan
        f32::from_bits((sign << 31) | 0x7F800000 | (frac << 13))
    } else {
        // normal: rebias exponent from 15-bias to 127-bias
        let new_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (new_exp << 23) | (frac << 13))
    }
}

/// Read a little-endian u16 from a byte slice at the given offset.
fn read_u16_le(data: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([data[off], data[off + 1]])
}

/// Read a little-endian u32 from a byte slice at the given offset.
fn read_u32_le(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

/// Block C byte-pair embedding table loaded from VCBP v1 packed binary.
pub struct VcbpTable {
    hot_float: Vec<f32>,   // n_hot * e, row-major (dequantized)
    oov: Vec<f32>,         // e-length shared OOV vector
    row_map: Vec<i32>,     // vocab_size-length, -1 for cold, else index into hot rows
    /// Embedding dimension (32).
    pub e: usize,
    /// Vocabulary size (65536).
    pub vocab_size: usize,
    /// Number of hot (frequently seen) byte-pairs.
    pub n_hot: usize,
    // Per-channel min/max for quantization.
    chan_min: Vec<f32>,
    chan_max: Vec<f32>,
}

impl VcbpTable {
    /// Load from a VCBP v1 packed binary file.
    pub fn from_packed(path: &Path) -> Result<Self, VcbpError> {
        let data = fs::read(path)?;
        Self::from_bytes(&data)
    }

    /// Parse from raw bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, VcbpError> {
        if data.len() < 32 {
            return Err(VcbpError::Format("file too short for header".into()));
        }
        if &data[0..4] != b"VCBP" {
            return Err(VcbpError::Format(format!(
                "bad magic {:?}, expected VCBP",
                &data[0..4]
            )));
        }
        let version = data[4];
        if version != 1 {
            return Err(VcbpError::Format(format!("unsupported version {version}")));
        }
        let _scheme = data[5]; // 0 = cold_shared
        let vocab_size = read_u32_le(data, 8) as usize;
        let e = read_u32_le(data, 12) as usize;
        let n_hot = read_u32_le(data, 16) as usize;

        if e == 0 || e % 2 != 0 {
            return Err(VcbpError::Format(format!("E={e} must be even and > 0")));
        }

        let mut off = 32usize;

        // Scales: E x fp16
        let mut scales = vec![0.0f32; e];
        for d in 0..e {
            scales[d] = fp16_to_f32(read_u16_le(data, off + d * 2));
        }
        off += e * 2;

        // Shared OOV: E x fp16
        let mut oov = vec![0.0f32; e];
        for d in 0..e {
            oov[d] = fp16_to_f32(read_u16_le(data, off + d * 2));
        }
        off += e * 2;

        // Hot bitmap: (V+7)/8 bytes, LSB-first
        let bitmap_bytes = (vocab_size + 7) / 8;
        let bitmap_raw = &data[off..off + bitmap_bytes];
        let mut hot_mask = vec![false; vocab_size];
        let mut hot_count = 0usize;
        for i in 0..vocab_size {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            if (bitmap_raw[byte_idx] >> bit_idx) & 1 == 1 {
                hot_mask[i] = true;
                hot_count += 1;
            }
        }
        if hot_count != n_hot {
            return Err(VcbpError::Format(format!(
                "bitmap popcount {hot_count} != n_hot {n_hot}"
            )));
        }
        off += bitmap_bytes;

        // Hot rows: n_hot x (E/2) bytes, int4 packed
        let row_bytes = e / 2;
        let mut hot_float = vec![0.0f32; n_hot * e];
        for row in 0..n_hot {
            let row_start = off + row * row_bytes;
            for col_pair in 0..(e / 2) {
                let packed = data[row_start + col_pair];
                let low_raw = packed & 0x0F;
                let high_raw = (packed >> 4) & 0x0F;
                // Sign-extend 4-bit 2's-complement
                let low_i8: i8 = if low_raw >= 8 {
                    low_raw as i8 - 16
                } else {
                    low_raw as i8
                };
                let high_i8: i8 = if high_raw >= 8 {
                    high_raw as i8 - 16
                } else {
                    high_raw as i8
                };
                // Interleave: even columns = low, odd columns = high
                let d_even = col_pair * 2;
                let d_odd = col_pair * 2 + 1;
                hot_float[row * e + d_even] = low_i8 as f32 * scales[d_even];
                hot_float[row * e + d_odd] = high_i8 as f32 * scales[d_odd];
            }
        }
        off += n_hot * row_bytes;

        if off != data.len() {
            return Err(VcbpError::Format(format!(
                "{} trailing bytes after decode",
                data.len() - off
            )));
        }

        // Build row_map: vocab_id -> hot row index (or -1)
        let mut row_map = vec![-1i32; vocab_size];
        let mut hot_idx = 0i32;
        for i in 0..vocab_size {
            if hot_mask[i] {
                row_map[i] = hot_idx;
                hot_idx += 1;
            }
        }

        // Pre-compute per-channel min/max for quantization
        let mut chan_min = vec![f32::MAX; e];
        let mut chan_max = vec![f32::MIN; e];
        for row in 0..n_hot {
            for d in 0..e {
                let v = hot_float[row * e + d];
                if v < chan_min[d] {
                    chan_min[d] = v;
                }
                if v > chan_max[d] {
                    chan_max[d] = v;
                }
            }
        }
        // Protect against zero-range channels
        for d in 0..e {
            if (chan_max[d] - chan_min[d]).abs() < 1e-8 {
                chan_max[d] = chan_min[d] + 1.0;
            }
        }

        Ok(Self {
            hot_float,
            oov,
            row_map,
            e,
            vocab_size,
            n_hot,
            chan_min,
            chan_max,
        })
    }

    /// Return the E-dim float32 embedding for a byte-pair ID.
    pub fn embed_id(&self, pair_id: u16) -> &[f32] {
        let idx = pair_id as usize;
        let row = self.row_map[idx];
        if row >= 0 {
            let start = row as usize * self.e;
            &self.hot_float[start..start + self.e]
        } else {
            &self.oov
        }
    }

    /// Check if a byte-pair ID is in the hot set.
    pub fn is_hot(&self, pair_id: u16) -> bool {
        self.row_map[pair_id as usize] >= 0
    }

    /// Quantize a float32 embedding to charge values for Brain input.
    /// Output values are in `[0, max_charge]` (typically 7 or 15).
    pub fn quantize_to_input(&self, embedding: &[f32], out: &mut [i32], max_charge: i32) {
        let mc = max_charge as f32;
        for d in 0..self.e {
            let range = self.chan_max[d] - self.chan_min[d];
            let norm = (embedding[d] - self.chan_min[d]) / range;
            out[d] = (norm * mc).round().clamp(0.0, mc) as i32;
        }
    }

    /// Dequantize Brain output charges back to approximate float32 embedding.
    pub fn dequantize_output(&self, charges: &[u8], max_charge: u8) -> Vec<f32> {
        let mc = max_charge as f32;
        let mut out = vec![0.0f32; self.e];
        for d in 0..self.e {
            let norm = charges[d] as f32 / mc;
            out[d] = self.chan_min[d] + norm * (self.chan_max[d] - self.chan_min[d]);
        }
        out
    }

    /// Find the nearest hot byte-pair ID to a query embedding (L2 distance).
    pub fn nearest_hot(&self, query: &[f32]) -> u16 {
        let mut best_dist = f32::MAX;
        let mut best_id = 0u16;
        let mut hot_idx = 0usize;
        for vid in 0..self.vocab_size {
            if self.row_map[vid] < 0 {
                continue;
            }
            let row_start = hot_idx * self.e;
            let mut dist = 0.0f32;
            for d in 0..self.e {
                let diff = query[d] - self.hot_float[row_start + d];
                dist += diff * diff;
            }
            if dist < best_dist {
                best_dist = dist;
                best_id = vid as u16;
            }
            hot_idx += 1;
        }
        best_id
    }

    /// Build a byte-pair ID from two bytes: `(hi << 8) | lo`.
    pub fn pair_id(hi: u8, lo: u8) -> u16 {
        (hi as u16) << 8 | lo as u16
    }

    /// Split a byte-pair ID into `(hi, lo)` bytes.
    pub fn pair_bytes(id: u16) -> (u8, u8) {
        ((id >> 8) as u8, (id & 0xFF) as u8)
    }
}

impl fmt::Display for VcbpTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VcbpTable(E={}, vocab={}, n_hot={})",
            self.e, self.vocab_size, self.n_hot
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn packed_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("output/block_c_bytepair_champion/packed.bin")
    }

    #[test]
    fn load_packed_bin() {
        let path = packed_path();
        if !path.exists() {
            eprintln!("skipping: packed.bin not found at {}", path.display());
            return;
        }
        let table = VcbpTable::from_packed(&path).unwrap();
        assert_eq!(table.e, 32);
        assert_eq!(table.vocab_size, 65536);
        assert_eq!(table.n_hot, 3386);
    }

    #[test]
    fn embed_hot_vs_cold() {
        let path = packed_path();
        if !path.exists() {
            return;
        }
        let table = VcbpTable::from_packed(&path).unwrap();
        // 'th' = (0x74 << 8) | 0x68 = 29800 — should be hot
        let th_id = VcbpTable::pair_id(b't', b'h');
        assert!(table.is_hot(th_id), "'th' should be hot");
        let emb = table.embed_id(th_id);
        assert_eq!(emb.len(), 32);

        // Some rare pair like 0xFF 0xFE — should be cold
        let rare_id = VcbpTable::pair_id(0xFF, 0xFE);
        assert!(!table.is_hot(rare_id));
        let cold_emb = table.embed_id(rare_id);
        assert_eq!(cold_emb.len(), 32);
        // Cold should equal OOV
        assert_eq!(cold_emb, table.oov.as_slice());
    }

    #[test]
    fn quantize_roundtrip() {
        let path = packed_path();
        if !path.exists() {
            return;
        }
        let table = VcbpTable::from_packed(&path).unwrap();
        let th_id = VcbpTable::pair_id(b't', b'h');
        let emb = table.embed_id(th_id).to_vec();

        let mut charges_i32 = vec![0i32; 32];
        table.quantize_to_input(&emb, &mut charges_i32, 15);
        let charges_u8: Vec<u8> = charges_i32.iter().map(|&c| c as u8).collect();
        let recovered = table.dequantize_output(&charges_u8, 15);

        // Should be approximate (int4 quantization)
        for d in 0..32 {
            let err = (emb[d] - recovered[d]).abs();
            let range = table.chan_max[d] - table.chan_min[d];
            assert!(
                err < range * 0.15,
                "dim {d}: err={err:.4} range={range:.4}"
            );
        }
    }

    #[test]
    fn nearest_hot_identity() {
        let path = packed_path();
        if !path.exists() {
            return;
        }
        let table = VcbpTable::from_packed(&path).unwrap();
        // The embedding of 'th' should be nearest to itself
        let th_id = VcbpTable::pair_id(b't', b'h');
        let emb = table.embed_id(th_id).to_vec();
        let found = table.nearest_hot(&emb);
        assert_eq!(found, th_id, "nearest_hot should find exact match");
    }
}
