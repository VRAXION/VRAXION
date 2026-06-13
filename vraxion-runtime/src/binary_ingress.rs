//! Bit-slip tolerant binary ingress.
//!
//! Runtime rule: byte/bit input is only evidence after frame reassembly,
//! integrity validation, requested-feature matching, and ambiguity checks.

use crate::bit_codec::{bits_from_int, checksum, int_from_bits, safe_filler};
use crate::Action;

pub const START_SYNC: [u8; 8] = [1, 1, 0, 1, 0, 0, 1, 1];
pub const END_SYNC: [u8; 8] = [0, 1, 0, 0, 1, 1, 0, 1];
pub const LENGTH_BITS: usize = 6;
pub const PAYLOAD_BITS: usize = 11;
pub const CRC_BITS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeReason {
    ValidRequestedFrame,
    NoValidRequestedFrame,
    ConflictingRequestedFrames,
    TruncatedOrInvalid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeResult {
    pub action: Action,
    pub reason: DecodeReason,
    pub selected_offset: Option<usize>,
    pub selected_feature: Option<u8>,
    pub selected_value: Option<u8>,
    pub candidate_count: usize,
    pub crc_pass_count: usize,
    pub requested_match_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidate {
    offset: usize,
    feature_id: u8,
    value: u8,
    trust: u8,
    crc_ok: bool,
    end_ok: bool,
    valid: bool,
}

pub fn encode_frame(feature_id: u8, value: u8, trust: u8, nonce: u8) -> Vec<u8> {
    let mut payload = bits_from_int(feature_id & 31, 5);
    payload.push(value & 1);
    payload.push(trust & 1);
    payload.extend(bits_from_int(nonce & 15, 4));

    let length_bits = bits_from_int(payload.len() as u8, LENGTH_BITS);
    let mut crc_input = length_bits.clone();
    crc_input.extend(payload.iter().copied());
    let crc = checksum(&crc_input);

    let mut frame = START_SYNC.to_vec();
    frame.extend(length_bits);
    frame.extend(payload);
    frame.extend(crc);
    frame.extend(END_SYNC);
    frame
}

pub fn corrupt_crc(frame: &[u8]) -> Vec<u8> {
    let mut out = frame.to_vec();
    let crc_start = START_SYNC.len() + LENGTH_BITS + PAYLOAD_BITS;
    if let Some(bit) = out.get_mut(crc_start) {
        *bit ^= 1;
    }
    out
}

pub fn corrupt_length(frame: &[u8]) -> Vec<u8> {
    let mut out = frame.to_vec();
    let length_start = START_SYNC.len();
    if let Some(bit) = out.get_mut(length_start) {
        *bit ^= 1;
    }
    out
}

fn parse_frame_at(stream: &[u8], offset: usize) -> Option<Candidate> {
    let min_len = START_SYNC.len() + LENGTH_BITS + CRC_BITS + END_SYNC.len();
    if offset + min_len > stream.len() {
        return None;
    }
    if stream[offset..offset + START_SYNC.len()] != START_SYNC {
        return None;
    }
    let length_start = offset + START_SYNC.len();
    let length_bits = &stream[length_start..length_start + LENGTH_BITS];
    let payload_len = int_from_bits(length_bits) as usize;
    let payload_start = length_start + LENGTH_BITS;
    let payload_end = payload_start + payload_len;
    let crc_end = payload_end + CRC_BITS;
    let end_end = crc_end + END_SYNC.len();
    if payload_len != PAYLOAD_BITS || end_end > stream.len() {
        return Some(Candidate {
            offset,
            feature_id: 0,
            value: 0,
            trust: 0,
            crc_ok: false,
            end_ok: false,
            valid: false,
        });
    }
    let payload = &stream[payload_start..payload_end];
    let observed_crc = &stream[payload_end..crc_end];
    let mut crc_input = length_bits.to_vec();
    crc_input.extend_from_slice(payload);
    let expected_crc = checksum(&crc_input);
    let crc_ok = observed_crc == expected_crc.as_slice();
    let end_ok = stream[crc_end..end_end] == END_SYNC;
    let feature_id = int_from_bits(&payload[..5]);
    let value = payload[5];
    let trust = payload[6];
    Some(Candidate {
        offset,
        feature_id,
        value,
        trust,
        crc_ok,
        end_ok,
        valid: crc_ok && end_ok && trust == 1,
    })
}

pub fn reassemble_requested_frame(stream: &[u8], requested_feature: u8) -> DecodeResult {
    let mut candidate_count = 0usize;
    let mut crc_pass_count = 0usize;
    let mut requested: Vec<Candidate> = Vec::new();

    for offset in 0..=stream.len().saturating_sub(START_SYNC.len()) {
        let Some(candidate) = parse_frame_at(stream, offset) else {
            continue;
        };
        candidate_count += 1;
        if candidate.crc_ok && candidate.end_ok {
            crc_pass_count += 1;
        }
        if candidate.valid && candidate.feature_id == (requested_feature & 31) {
            requested.push(candidate);
        }
    }

    if requested.is_empty() {
        return DecodeResult {
            action: Action::Defer,
            reason: if candidate_count == 0 {
                DecodeReason::NoValidRequestedFrame
            } else {
                DecodeReason::TruncatedOrInvalid
            },
            selected_offset: None,
            selected_feature: None,
            selected_value: None,
            candidate_count,
            crc_pass_count,
            requested_match_count: 0,
        };
    }

    let first_value = requested[0].value;
    if requested
        .iter()
        .any(|candidate| candidate.value != first_value)
    {
        return DecodeResult {
            action: Action::Defer,
            reason: DecodeReason::ConflictingRequestedFrames,
            selected_offset: None,
            selected_feature: None,
            selected_value: None,
            candidate_count,
            crc_pass_count,
            requested_match_count: requested.len(),
        };
    }

    let selected = requested[0];
    DecodeResult {
        action: Action::CommitEvidence,
        reason: DecodeReason::ValidRequestedFrame,
        selected_offset: Some(selected.offset),
        selected_feature: Some(selected.feature_id),
        selected_value: Some(selected.value),
        candidate_count,
        crc_pass_count,
        requested_match_count: requested.len(),
    }
}

pub fn demo_case_insert_before_frame() -> (Vec<u8>, u8, u8) {
    let requested_feature = 9;
    let value = 1;
    let valid = encode_frame(requested_feature, value, 1, 3);
    let mut stream = safe_filler(16);
    stream.push(0);
    stream.extend(valid);
    stream.extend(safe_filler(16));
    (stream, requested_feature, value)
}
