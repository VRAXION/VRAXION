//! L1Merger — Block B deploy implementation.
//!
//! Pure Rust, zero ML-framework dependencies. Only std + serde_json (for the
//! Block A LUT used in `verify_lossless`).

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

// --------------------------------------------------------------------------- constants

pub const IN_DIM: usize = 32;
pub const HIDDEN: usize = 81;
pub const LUT_DIM: usize = 16;

// Component descriptors match the Python COMPONENTS list exactly:
// (name, n, raw_only)
const COMPONENTS: [(&str, usize, bool); 5] = [
    ("W", 2592, false),
    ("b1", 81, false),
    ("b2", 32, true),
    ("c19_c", 81, false),
    ("c19_rho", 81, false),
];

// --------------------------------------------------------------------------- error type

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Parse(serde_json::Error),
    BadMagic,
    BadShape(&'static str),
    BadFormat(String),
    BadStream(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Parse(e) => write!(f, "parse: {e}"),
            Self::BadMagic => write!(f, "bad magic (expected VGH1)"),
            Self::BadShape(m) => write!(f, "bad shape: {m}"),
            Self::BadFormat(m) => write!(f, "bad format: {m}"),
            Self::BadStream(m) => write!(f, "bad bitstream: {m}"),
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

// --------------------------------------------------------------------------- fp16 helper

/// Decode a little-endian IEEE 754 half-precision (fp16) word to f32.
/// No crate needed — just bit-manipulation per the IEEE 754 spec.
fn fp16_to_f32(bits: u16) -> f32 {
    let sign: u32 = ((bits as u32) >> 15) & 1;
    let exp: u32 = ((bits as u32) >> 10) & 0x1F;
    let mant: u32 = (bits as u32) & 0x3FF;

    let f: u32 = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal fp16 -> normalised fp32
            let mut e = 0u32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF;
            (sign << 31) | (((127 - 15 - e) + 1) << 23) | (m << 13)
        }
    } else if exp == 31 {
        // inf / nan
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(f)
}

// --------------------------------------------------------------------------- bit reader

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u32,
    nbits: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, acc: 0, nbits: 0 }
    }

    /// Read `length` bits (MSB first) and return as a u32.
    fn read(&mut self, mut length: u32) -> Result<u32, LoadError> {
        let mut out: u32 = 0;
        while length > 0 {
            if self.nbits == 0 {
                if self.pos >= self.data.len() {
                    return Err(LoadError::BadStream("bitstream exhausted".into()));
                }
                self.acc = self.data[self.pos] as u32;
                self.pos += 1;
                self.nbits = 8;
            }
            let take = self.nbits.min(length);
            let shift = self.nbits - take;
            let chunk = (self.acc >> shift) & ((1 << take) - 1);
            out = (out << take) | chunk;
            self.acc &= if self.nbits > take { (1 << (self.nbits - take)) - 1 } else { 0 };
            self.nbits -= take;
            length -= take;
        }
        Ok(out)
    }
}

// --------------------------------------------------------------------------- nibble helpers

fn unpack_nibbles(data: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for &b in data {
        if out.len() == n { break; }
        out.push(b & 0xF);
        if out.len() == n { break; }
        out.push((b >> 4) & 0xF);
    }
    out
}

// --------------------------------------------------------------------------- canonical Huffman decode

/// Build a decode table: (code, code_length) -> symbol, from a symbol->length map.
///
/// Matches the canonical code assignment in the Python packer exactly:
/// sort by (length, symbol), assign codes starting from 0 shifting up on
/// length increases.
fn build_decode_table(lengths: &[(usize, u8)]) -> Vec<(u32, u32, usize)> {
    // lengths: Vec<(symbol, bit_length)> with bit_length > 0
    let mut items: Vec<(u8, usize)> = lengths
        .iter()
        .filter(|&&(_, ln)| ln > 0)
        .map(|&(sym, ln)| (ln, sym))
        .collect();
    items.sort_by_key(|&(ln, sym)| (ln, sym));

    let mut table: Vec<(u32, u32, usize)> = Vec::with_capacity(items.len());
    let mut code: u32 = 0;
    let mut prev_len: u32 = if items.is_empty() { 0 } else { items[0].0 as u32 };
    for (ln, sym) in &items {
        let ln = *ln as u32;
        code <<= ln - prev_len;
        table.push((code, ln, *sym));
        code += 1;
        prev_len = ln;
    }
    table
}

/// Decode `n` symbols from a BitReader using a prebuilt table.
fn decode_symbols(
    reader: &mut BitReader<'_>,
    n: usize,
    table: &[(u32, u32, usize)],
) -> Result<Vec<usize>, LoadError> {
    if n == 0 {
        return Ok(vec![]);
    }
    let max_len = table.iter().map(|&(_, ln, _)| ln).max().unwrap_or(0);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut code: u32 = 0;
        let mut matched = false;
        for ln in 1..=max_len {
            code = (code << 1) | reader.read(1)?;
            for &(c, l, sym) in table {
                if l == ln && c == code {
                    out.push(sym);
                    matched = true;
                    break;
                }
            }
            if matched { break; }
        }
        if !matched {
            return Err(LoadError::BadStream("invalid Huffman code".into()));
        }
    }
    Ok(out)
}

// --------------------------------------------------------------------------- component unpacker

/// Unpack one component from `payload[offset..]`.
/// Returns (decoded_f32_values, new_offset).
fn unpack_component(
    n: usize,
    raw_only: bool,
    payload: &[u8],
    mut offset: usize,
) -> Result<(Vec<f32>, usize), LoadError> {
    if raw_only {
        // b2: plain fp16 array, no Huffman
        let end = offset + n * 2;
        if end > payload.len() {
            return Err(LoadError::BadStream("truncated raw fp16 stream".into()));
        }
        let arr: Vec<f32> = (0..n)
            .map(|i| {
                let lo = payload[offset + i * 2] as u16;
                let hi = payload[offset + i * 2 + 1] as u16;
                fp16_to_f32(lo | (hi << 8))
            })
            .collect();
        return Ok((arr, end));
    }

    // Number of generators G (1 byte)
    let g = payload[offset] as usize;
    offset += 1;

    // G fp16 generator values
    let gens_end = offset + g * 2;
    if gens_end > payload.len() {
        return Err(LoadError::BadStream("truncated generators".into()));
    }
    let gens: Vec<f32> = (0..g)
        .map(|i| {
            let lo = payload[offset + i * 2] as u16;
            let hi = payload[offset + i * 2 + 1] as u16;
            fp16_to_f32(lo | (hi << 8))
        })
        .collect();
    offset = gens_end;

    // Mode bitmap: ceil(n/8) bytes, n bits (1=encoded, 0=fallback)
    let mode_nbytes = (n + 7) / 8;
    if offset + mode_nbytes > payload.len() {
        return Err(LoadError::BadStream("truncated mode bitmap".into()));
    }
    let mode_bits: Vec<bool> = {
        let mut r = BitReader::new(&payload[offset..offset + mode_nbytes]);
        (0..n).map(|_| r.read(1).map(|b| b == 1)).collect::<Result<Vec<_>, _>>()?
    };
    offset += mode_nbytes;
    let n_enc = mode_bits.iter().filter(|&&b| b).count();
    let n_fb = n - n_enc;

    // Sign bitmap: ceil(n_enc/8) bytes, n_enc bits
    let sign_nbytes = (n_enc + 7) / 8;
    if offset + sign_nbytes > payload.len() {
        return Err(LoadError::BadStream("truncated sign bitmap".into()));
    }
    let sign_bits: Vec<i32> = {
        let mut r = BitReader::new(&payload[offset..offset + sign_nbytes]);
        (0..n_enc).map(|_| r.read(1).map(|b| if b == 1 { 1 } else { -1 })).collect::<Result<Vec<_>, _>>()?
    };
    offset += sign_nbytes;

    // Coef lengths: symbols 1..=7 packed as 7 nibbles -> ceil(7/2) = 4 bytes
    let coef_len_nbytes = 4; // ceil(7/2)
    if offset + coef_len_nbytes > payload.len() {
        return Err(LoadError::BadStream("truncated coef lengths".into()));
    }
    let coef_lens_raw = unpack_nibbles(&payload[offset..offset + coef_len_nbytes], 7);
    offset += coef_len_nbytes;
    // symbol = 1..=7, len = coef_lens_raw[i]
    let coef_lengths: Vec<(usize, u8)> = (0..7)
        .map(|i| (i + 1, coef_lens_raw[i]))
        .collect();

    // Idx lengths: symbols 0..G packed as G nibbles -> ceil(G/2) bytes
    let idx_len_nbytes = (g + 1) / 2;
    if offset + idx_len_nbytes > payload.len() {
        return Err(LoadError::BadStream("truncated idx lengths".into()));
    }
    let idx_lens_raw = unpack_nibbles(&payload[offset..offset + idx_len_nbytes], g);
    offset += idx_len_nbytes;
    let idx_lengths: Vec<(usize, u8)> = (0..g)
        .map(|i| (i, idx_lens_raw[i]))
        .collect();

    // Blob length pair: 2 × u16-le
    if offset + 4 > payload.len() {
        return Err(LoadError::BadStream("truncated blob lengths".into()));
    }
    let coef_nbytes = u16::from_le_bytes([payload[offset], payload[offset + 1]]) as usize;
    let idx_nbytes = u16::from_le_bytes([payload[offset + 2], payload[offset + 3]]) as usize;
    offset += 4;

    // Coef blob
    if offset + coef_nbytes > payload.len() {
        return Err(LoadError::BadStream("truncated coef blob".into()));
    }
    let coef_blob = &payload[offset..offset + coef_nbytes];
    offset += coef_nbytes;

    // Idx blob
    if offset + idx_nbytes > payload.len() {
        return Err(LoadError::BadStream("truncated idx blob".into()));
    }
    let idx_blob = &payload[offset..offset + idx_nbytes];
    offset += idx_nbytes;

    // Decode coef and idx symbol streams
    let coef_table = build_decode_table(&coef_lengths);
    let idx_table = build_decode_table(&idx_lengths);
    let coef_syms = decode_symbols(&mut BitReader::new(coef_blob), n_enc, &coef_table)?;
    let idx_syms = decode_symbols(&mut BitReader::new(idx_blob), n_enc, &idx_table)?;

    // Fallback fp16 stream
    let fb_end = offset + n_fb * 2;
    if fb_end > payload.len() {
        return Err(LoadError::BadStream("truncated fallback stream".into()));
    }
    let fallback: Vec<f32> = (0..n_fb)
        .map(|i| {
            let lo = payload[offset + i * 2] as u16;
            let hi = payload[offset + i * 2 + 1] as u16;
            fp16_to_f32(lo | (hi << 8))
        })
        .collect();
    offset = fb_end;

    // Reconstruct array
    let mut arr = vec![0f32; n];
    let mut ie = 0usize;
    let mut ifb = 0usize;
    for (i, &is_enc) in mode_bits.iter().enumerate() {
        if is_enc {
            let s = sign_bits[ie] as f32;
            let c = coef_syms[ie] as f32;
            let gi = idx_syms[ie];
            arr[i] = s * c * gens[gi];
            ie += 1;
        } else {
            arr[i] = fallback[ifb];
            ifb += 1;
        }
    }

    Ok((arr, offset))
}

// --------------------------------------------------------------------------- C19 activation

/// Per-channel C19 activation (piecewise polynomial).
///
/// For each channel i:
///   c  = max(c_raw[i], 0.1)
///   rho = max(rho_raw[i], 0.0)
///   L  = 6 * c
///   scaled = x / c
///   n  = floor(scaled)
///   t  = scaled - n
///   h  = t * (1 - t)
///   sgn = +1 if n is even else -1
///   interior = c * (sgn * h + rho * h * h)
///   output = if x >= L: x - L
///            elif x <= -L: x + L
///            else: interior
#[allow(dead_code)]
fn c19_activate(x: &[f32; IN_DIM], c_raw: &[f32; HIDDEN], rho_raw: &[f32; HIDDEN], out: &mut [f32; HIDDEN]) {
    for k in 0..HIDDEN {
        let c = c_raw[k].max(0.1);
        let rho = rho_raw[k].max(0.0);
        let l = 6.0 * c;
        let xk = x[k]; // x is the hidden pre-activation, indexed by k
        if xk >= l {
            out[k] = xk - l;
        } else if xk <= -l {
            out[k] = xk + l;
        } else {
            let scaled = xk / c;
            let n = scaled.floor();
            let t = scaled - n;
            let h = t * (1.0 - t);
            let sgn = if (n as i64).rem_euclid(2) == 0 { 1.0f32 } else { -1.0f32 };
            out[k] = c * (sgn * h + rho * h * h);
        }
    }
}

// --------------------------------------------------------------------------- L1Merger

/// Block B: L1 Byte-Pair Merger (single-W mirror-tied, 100% lossless).
///
/// Forward: `y = C19(x @ W + b1) @ W.T + b2`
pub struct L1Merger {
    /// W matrix (IN_DIM × HIDDEN) — row-major, i.e. W[i][k].
    w: [[f32; HIDDEN]; IN_DIM],
    /// b1 bias (HIDDEN,)
    b1: [f32; HIDDEN],
    /// b2 bias (IN_DIM,)
    b2: [f32; IN_DIM],
    /// C19 c_raw parameters (HIDDEN,)
    c19_c: [f32; HIDDEN],
    /// C19 rho_raw parameters (HIDDEN,)
    c19_rho: [f32; HIDDEN],
}

impl L1Merger {
    /// Load from the repo's default champion artifact.
    ///
    /// Path resolved from `CARGO_MANIFEST_DIR/../output/merger_single_w_huffman_pack/packed_model.bin`.
    pub fn load_default() -> Result<Self, LoadError> {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("Rust/ parent must exist")
            .to_path_buf();
        let bin_path = repo_root
            .join("output")
            .join("merger_single_w_huffman_pack")
            .join("packed_model.bin");
        Self::from_bin_path(&bin_path)
    }

    /// Load from an explicit path to the packed binary.
    pub fn from_bin_path(path: &std::path::Path) -> Result<Self, LoadError> {
        let payload = fs::read(path)?;
        if payload.len() < 4 || &payload[..4] != b"VGH1" {
            return Err(LoadError::BadMagic);
        }
        Self::from_bytes(&payload)
    }

    /// Decode from a VGH1-magic byte slice.
    pub fn from_bytes(payload: &[u8]) -> Result<Self, LoadError> {
        if payload.len() < 4 || &payload[..4] != b"VGH1" {
            return Err(LoadError::BadMagic);
        }
        let mut offset = 4usize;
        let mut arrays: [Option<Vec<f32>>; 5] = [None, None, None, None, None];
        for (ci, &(_, n, raw_only)) in COMPONENTS.iter().enumerate() {
            let (arr, new_offset) = unpack_component(n, raw_only, payload, offset)?;
            if arr.len() != n {
                return Err(LoadError::BadShape("component length mismatch"));
            }
            arrays[ci] = Some(arr);
            offset = new_offset;
        }
        if offset != payload.len() {
            return Err(LoadError::BadFormat(format!(
                "{} trailing bytes after decode",
                payload.len() - offset
            )));
        }

        // COMPONENTS order: W(2592), b1(81), b2(32), c19_c(81), c19_rho(81)
        let w_flat = arrays[0].take().unwrap();
        let b1_flat = arrays[1].take().unwrap();
        let b2_flat = arrays[2].take().unwrap();
        let c_flat  = arrays[3].take().unwrap();
        let rho_flat = arrays[4].take().unwrap();

        // W is stored flat with shape (IN_DIM=32, HIDDEN=81) -> row-major
        if w_flat.len() != IN_DIM * HIDDEN {
            return Err(LoadError::BadShape("W size != 32*81"));
        }
        let mut w = [[0f32; HIDDEN]; IN_DIM];
        for i in 0..IN_DIM {
            for k in 0..HIDDEN {
                w[i][k] = w_flat[i * HIDDEN + k];
            }
        }

        let mut b1 = [0f32; HIDDEN];
        let mut b2 = [0f32; IN_DIM];
        let mut c19_c = [0f32; HIDDEN];
        let mut c19_rho = [0f32; HIDDEN];

        b1.copy_from_slice(&b1_flat);
        b2.copy_from_slice(&b2_flat);
        c19_c.copy_from_slice(&c_flat);
        c19_rho.copy_from_slice(&rho_flat);

        Ok(Self { w, b1, b2, c19_c, c19_rho })
    }

    /// Forward pass: `y = C19(x @ W + b1) @ W.T + b2`.
    ///
    /// x: &[f32; 32]  (two 16-dim Block A latents concatenated)
    /// returns [f32; 32]
    pub fn forward(&self, x: &[f32; IN_DIM]) -> [f32; IN_DIM] {
        // Step 1: pre_h = x @ W + b1   shape (HIDDEN,)
        let mut pre_h = [0f32; HIDDEN];
        for k in 0..HIDDEN {
            let mut acc = self.b1[k];
            for i in 0..IN_DIM {
                acc += x[i] * self.w[i][k];
            }
            pre_h[k] = acc;
        }

        // Step 2: h = C19(pre_h)   shape (HIDDEN,)
        // C19 takes x indexed over HIDDEN channels; re-use the same function
        // by treating pre_h as the "x" over HIDDEN dims.
        let mut h = [0f32; HIDDEN];
        // Inline C19 over HIDDEN channels (pre_h plays role of x[k])
        for k in 0..HIDDEN {
            let c = self.c19_c[k].max(0.1);
            let rho = self.c19_rho[k].max(0.0);
            let l = 6.0 * c;
            let xk = pre_h[k];
            h[k] = if xk >= l {
                xk - l
            } else if xk <= -l {
                xk + l
            } else {
                let scaled = xk / c;
                let n = scaled.floor();
                let t = scaled - n;
                let hv = t * (1.0 - t);
                let sgn = if (n as i64).rem_euclid(2) == 0 { 1.0f32 } else { -1.0f32 };
                c * (sgn * hv + rho * hv * hv)
            };
        }

        // Step 3: y = h @ W.T + b2   shape (IN_DIM,)
        let mut y = [0f32; IN_DIM];
        for i in 0..IN_DIM {
            let mut acc = self.b2[i];
            for k in 0..HIDDEN {
                acc += h[k] * self.w[i][k]; // W.T[k][i] = W[i][k]
            }
            y[i] = acc;
        }
        y
    }

    /// Verify 100% lossless on all 65,536 byte pairs.
    ///
    /// The merger was trained on `tools/byte_embedder_lut_int8_nozero.json`
    /// (scale ≈ 0.132, different from the Block A champion LUT). This method
    /// loads that file, constructs all 65,536 (a, b) pair inputs by
    /// concatenating their 16-dim latents, runs forward(), and checks
    /// sign(y[i]) == sign(x[i]) for all 32 dims.
    ///
    /// Returns (matching_pairs, total_pairs). Expected: (65536, 65536).
    pub fn verify_lossless(&self) -> Result<(usize, usize), LoadError> {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("Rust/ parent must exist")
            .to_path_buf();
        // The merger was trained on the nozero variant of the LUT (tools/),
        // NOT the Block A champion LUT in output/.
        let lut_path = repo_root
            .join("tools")
            .join("byte_embedder_lut_int8_nozero.json");

        let lut_text = fs::read_to_string(&lut_path)?;
        let lut_blob: LutBlob = serde_json::from_str(&lut_text)?;

        if lut_blob.lut.len() != 256 {
            return Err(LoadError::BadShape("LUT rows != 256"));
        }
        if lut_blob.lut[0].len() != LUT_DIM {
            return Err(LoadError::BadShape("LUT cols != 16"));
        }

        // Build float LUT
        let mut lut_f32 = [[0f32; LUT_DIM]; 256];
        for (b, row) in lut_blob.lut.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                lut_f32[b][j] = (v as f32) * lut_blob.scale;
            }
        }

        let mut matches = 0usize;
        let total = 65536usize;

        let mut x = [0f32; IN_DIM];
        for a in 0u16..256 {
            for b in 0u16..256 {
                // Concatenate latent(a) || latent(b)
                x[..LUT_DIM].copy_from_slice(&lut_f32[a as usize]);
                x[LUT_DIM..].copy_from_slice(&lut_f32[b as usize]);

                let y = self.forward(&x);

                // Check sign match on all 32 dims
                let all_match = x.iter().zip(y.iter()).all(|(&xi, &yi)| {
                    // sign(xi) == sign(yi): both positive, both negative, or both zero
                    // Zero is treated as its own sign class.
                    sign_f32(xi) == sign_f32(yi)
                });
                if all_match {
                    matches += 1;
                }
            }
        }
        Ok((matches, total))
    }
}

/// Returns -1, 0, or 1 as an i8 (mirrors Python's numpy.sign).
#[inline]
fn sign_f32(x: f32) -> i8 {
    if x > 0.0 { 1 } else if x < 0.0 { -1 } else { 0 }
}

// --------------------------------------------------------------------------- LUT deserialization

#[derive(Deserialize)]
struct LutBlob {
    scale: f32,
    lut: Vec<Vec<i8>>,
}

// --------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_default_succeeds() {
        let m = L1Merger::load_default().expect("load champion packed_model.bin");
        // Spot-check a couple of parameters for sanity
        // b1 mean should be in roughly ±3 range (Python says ~-0.83)
        let b1_mean: f32 = m.b1.iter().sum::<f32>() / m.b1.len() as f32;
        assert!(b1_mean.abs() < 5.0, "b1 mean out of range: {b1_mean}");
    }

    #[test]
    fn forward_output_shape() {
        let m = L1Merger::load_default().expect("load champion");
        let x = [0.1f32; IN_DIM];
        let y = m.forward(&x);
        assert_eq!(y.len(), IN_DIM);
    }

    #[test]
    fn verify_lossless_all_pairs() {
        let m = L1Merger::load_default().expect("load champion");
        let (matches, total) = m.verify_lossless().expect("verify lossless");
        assert_eq!(total, 65536, "expected 65536 total pairs");
        assert_eq!(
            matches, 65536,
            "expected 65536 lossless pairs, got {matches}/{total}"
        );
    }
}
