//! Small deterministic bit helpers shared by ingress and tests.

use crate::binary_ingress::CRC_BITS;

pub fn bits_from_int(value: u8, width: usize) -> Vec<u8> {
    (0..width).rev().map(|shift| (value >> shift) & 1).collect()
}

pub fn int_from_bits(bits: &[u8]) -> u8 {
    bits.iter().fold(0u8, |acc, bit| (acc << 1) | (*bit & 1))
}

pub fn checksum(bits: &[u8]) -> Vec<u8> {
    let mut acc: u8 = 0x2A;
    for (idx, bit) in bits.iter().enumerate() {
        acc ^= (((idx + 3) * 17 + (*bit as usize) * 29) & 0x3F) as u8;
        acc = ((acc << 1) | (acc >> 5)) & 0x3F;
    }
    bits_from_int(acc, CRC_BITS)
}

pub fn safe_filler(length: usize) -> Vec<u8> {
    let pattern = [0, 0, 1, 0, 1, 0, 0];
    (0..length)
        .map(|idx| pattern[idx % pattern.len()])
        .collect()
}

pub fn insert_bit(bits: &[u8], pos: usize, bit: u8) -> Vec<u8> {
    let pos = pos.min(bits.len());
    let mut out = Vec::with_capacity(bits.len() + 1);
    out.extend_from_slice(&bits[..pos]);
    out.push(bit & 1);
    out.extend_from_slice(&bits[pos..]);
    out
}

pub fn drop_bit(bits: &[u8], pos: usize) -> Vec<u8> {
    if bits.is_empty() {
        return Vec::new();
    }
    let pos = pos.min(bits.len() - 1);
    let mut out = Vec::with_capacity(bits.len() - 1);
    out.extend_from_slice(&bits[..pos]);
    out.extend_from_slice(&bits[pos + 1..]);
    out
}
