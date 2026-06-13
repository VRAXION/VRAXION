//! Minimal locked VRAXION runtime kernel.
//!
//! This crate is intentionally small and deterministic. It is not a training
//! system. It captures the currently locked runtime rules:
//!
//! - raw binary input is reassembled before commit;
//! - text input uses a small set of Agency-selected modes;
//! - pockets emit proposals, not truth;
//! - Agency is the only commit boundary;
//! - output is rendered from committed state only.

pub const START_SYNC: [u8; 8] = [1, 1, 0, 1, 0, 0, 1, 1];
pub const END_SYNC: [u8; 8] = [0, 1, 0, 0, 1, 1, 0, 1];
pub const LENGTH_BITS: usize = 6;
pub const PAYLOAD_BITS: usize = 11;
pub const CRC_BITS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    CommitEvidence,
    Reject,
    Defer,
    AskOrMultiCycle,
    AnswerReady,
}

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
pub enum TextMode {
    FastDefault,
    LongCapped,
    CleanLong,
    AskOrMultiCycle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextProfile {
    pub byte_len: usize,
    pub evidence_available: bool,
    pub boundary_risk: u8,
    pub integrity_risk: u8,
    pub requires_clean_long: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposalKind {
    EvidenceWrite,
    OutputIntent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Proposal {
    pub kind: ProposalKind,
    pub cycle_id: u32,
    pub source_pocket_id: u32,
    pub trace_valid: bool,
    pub ground_compatible: bool,
    pub target_feature: Option<u8>,
    pub value: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgencyState {
    pub cycle_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitRecord {
    pub feature_id: u8,
    pub value: u8,
    pub source_pocket_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EgressMode {
    CompactAction,
    ShortText,
    LongText,
    MultiResolution,
    NeedMoreInfo,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedOutput {
    pub compact: String,
    pub short: Option<String>,
    pub long: Option<String>,
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

pub fn safe_filler(length: usize) -> Vec<u8> {
    let pattern = [0, 0, 1, 0, 1, 0, 0];
    (0..length)
        .map(|idx| pattern[idx % pattern.len()])
        .collect()
}

pub fn corrupt_crc(frame: &[u8]) -> Vec<u8> {
    let mut out = frame.to_vec();
    let crc_start = START_SYNC.len() + LENGTH_BITS + PAYLOAD_BITS;
    out[crc_start] ^= 1;
    out
}

pub fn insert_bit(bits: &[u8], pos: usize, bit: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len() + 1);
    out.extend_from_slice(&bits[..pos]);
    out.push(bit & 1);
    out.extend_from_slice(&bits[pos..]);
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

pub fn select_text_mode(profile: TextProfile) -> TextMode {
    if !profile.evidence_available {
        return TextMode::AskOrMultiCycle;
    }
    if profile.byte_len <= 416 && profile.boundary_risk <= 1 && profile.integrity_risk <= 1 {
        return TextMode::FastDefault;
    }
    if profile.byte_len <= 1024 && profile.integrity_risk <= 2 && !profile.requires_clean_long {
        return TextMode::LongCapped;
    }
    if profile.byte_len <= 1664 && profile.integrity_risk <= 3 {
        return TextMode::CleanLong;
    }
    TextMode::AskOrMultiCycle
}

pub fn agency_decide(state: AgencyState, proposal: Proposal) -> (Action, Option<CommitRecord>) {
    if proposal.cycle_id != state.cycle_id || !proposal.trace_valid || !proposal.ground_compatible {
        return (Action::Reject, None);
    }
    match proposal.kind {
        ProposalKind::EvidenceWrite => {
            let (Some(feature_id), Some(value)) = (proposal.target_feature, proposal.value) else {
                return (Action::Defer, None);
            };
            (
                Action::CommitEvidence,
                Some(CommitRecord {
                    feature_id,
                    value,
                    source_pocket_id: proposal.source_pocket_id,
                }),
            )
        }
        ProposalKind::OutputIntent => (Action::AnswerReady, None),
    }
}

pub fn render_output(mode: EgressMode, record: Option<CommitRecord>) -> RenderedOutput {
    match (mode, record) {
        (EgressMode::NeedMoreInfo, _) | (_, None) => RenderedOutput {
            compact: "NEED_MORE_INFO".to_string(),
            short: Some("The committed state is unresolved; request more evidence.".to_string()),
            long: None,
        },
        (EgressMode::CompactAction, Some(_)) => RenderedOutput {
            compact: "COMMIT_EVIDENCE".to_string(),
            short: None,
            long: None,
        },
        (EgressMode::ShortText, Some(record)) => RenderedOutput {
            compact: "COMMIT_EVIDENCE".to_string(),
            short: Some(format!("Feature {} is committed as {}.", record.feature_id, record.value)),
            long: None,
        },
        (EgressMode::LongText, Some(record)) => RenderedOutput {
            compact: "COMMIT_EVIDENCE".to_string(),
            short: None,
            long: Some(format!(
                "Trace: Agency committed feature {} = {} from validated pocket {} after the proposal passed cycle, trace, ground, and ingress guards.",
                record.feature_id, record.value, record.source_pocket_id
            )),
        },
        (EgressMode::MultiResolution, Some(record)) => RenderedOutput {
            compact: "COMMIT_EVIDENCE".to_string(),
            short: Some(format!("Feature {} is committed as {}.", record.feature_id, record.value)),
            long: Some(format!(
                "Trace: Agency committed feature {} = {} from validated pocket {} after the proposal passed cycle, trace, ground, and ingress guards.",
                record.feature_id, record.value, record.source_pocket_id
            )),
        },
    }
}

pub fn ingress_to_proposal(
    cycle_id: u32,
    source_pocket_id: u32,
    requested_feature: u8,
    stream: &[u8],
) -> (DecodeResult, Proposal) {
    let decoded = reassemble_requested_frame(stream, requested_feature);
    let proposal = Proposal {
        kind: ProposalKind::EvidenceWrite,
        cycle_id,
        source_pocket_id,
        trace_valid: decoded.action == Action::CommitEvidence,
        ground_compatible: decoded.action == Action::CommitEvidence,
        target_feature: decoded.selected_feature,
        value: decoded.selected_value,
    };
    (decoded, proposal)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn framed_stream(feature: u8, value: u8) -> Vec<u8> {
        let mut stream = safe_filler(12);
        stream.extend(encode_frame(feature, value, 1, 5));
        stream.extend(safe_filler(12));
        stream
    }

    #[test]
    fn bitslip_insert_is_recovered() {
        let (stream, feature, value) = demo_case_insert_before_frame();
        let decoded = reassemble_requested_frame(&stream, feature);
        assert_eq!(decoded.action, Action::CommitEvidence);
        assert_eq!(decoded.selected_feature, Some(feature));
        assert_eq!(decoded.selected_value, Some(value));
    }

    #[test]
    fn wrong_feature_valid_crc_is_not_committed() {
        let stream = framed_stream(11, 1);
        let decoded = reassemble_requested_frame(&stream, 12);
        assert_eq!(decoded.action, Action::Defer);
        assert_eq!(decoded.selected_feature, None);
    }

    #[test]
    fn corrupted_crc_is_not_committed() {
        let frame = corrupt_crc(&encode_frame(4, 1, 1, 2));
        let mut stream = safe_filler(8);
        stream.extend(frame);
        let decoded = reassemble_requested_frame(&stream, 4);
        assert_eq!(decoded.action, Action::Defer);
    }

    #[test]
    fn conflicting_duplicate_requested_frames_defer() {
        let mut stream = safe_filler(8);
        stream.extend(encode_frame(3, 0, 1, 1));
        stream.extend(safe_filler(6));
        stream.extend(encode_frame(3, 1, 1, 2));
        let decoded = reassemble_requested_frame(&stream, 3);
        assert_eq!(decoded.action, Action::Defer);
        assert_eq!(decoded.reason, DecodeReason::ConflictingRequestedFrames);
    }

    #[test]
    fn end_marker_alone_is_not_enough() {
        let mut stream = safe_filler(12);
        stream.extend(END_SYNC);
        let decoded = reassemble_requested_frame(&stream, 1);
        assert_eq!(decoded.action, Action::Defer);
    }

    #[test]
    fn text_mode_selector_uses_smallest_safe_mode() {
        assert_eq!(
            select_text_mode(TextProfile {
                byte_len: 200,
                evidence_available: true,
                boundary_risk: 0,
                integrity_risk: 0,
                requires_clean_long: false,
            }),
            TextMode::FastDefault
        );
        assert_eq!(
            select_text_mode(TextProfile {
                byte_len: 900,
                evidence_available: true,
                boundary_risk: 2,
                integrity_risk: 2,
                requires_clean_long: false,
            }),
            TextMode::LongCapped
        );
        assert_eq!(
            select_text_mode(TextProfile {
                byte_len: 1400,
                evidence_available: true,
                boundary_risk: 3,
                integrity_risk: 3,
                requires_clean_long: true,
            }),
            TextMode::CleanLong
        );
        assert_eq!(
            select_text_mode(TextProfile {
                byte_len: 2000,
                evidence_available: true,
                boundary_risk: 4,
                integrity_risk: 4,
                requires_clean_long: true,
            }),
            TextMode::AskOrMultiCycle
        );
    }

    #[test]
    fn agency_rejects_stale_or_unverified_proposals() {
        let state = AgencyState { cycle_id: 10 };
        let stale = Proposal {
            kind: ProposalKind::EvidenceWrite,
            cycle_id: 9,
            source_pocket_id: 7,
            trace_valid: true,
            ground_compatible: true,
            target_feature: Some(2),
            value: Some(1),
        };
        assert_eq!(agency_decide(state, stale), (Action::Reject, None));

        let unverified = Proposal {
            cycle_id: 10,
            trace_valid: false,
            ..stale
        };
        assert_eq!(agency_decide(state, unverified), (Action::Reject, None));
    }

    #[test]
    fn agency_committed_state_can_render_multi_resolution() {
        let record = CommitRecord {
            feature_id: 2,
            value: 1,
            source_pocket_id: 99,
        };
        let rendered = render_output(EgressMode::MultiResolution, Some(record));
        assert_eq!(rendered.compact, "COMMIT_EVIDENCE");
        assert!(rendered.short.unwrap().contains("Feature 2"));
        assert!(rendered.long.unwrap().contains("Agency committed"));
    }
}
