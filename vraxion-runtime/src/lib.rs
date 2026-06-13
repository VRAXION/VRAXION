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

pub mod agency;
pub mod binary_ingress;
pub mod bit_codec;
pub mod body;
pub mod curriculum;
pub mod egress;
pub mod library;
pub mod manager;
pub mod next_mutation;
pub mod pocket;
pub mod proposal;
pub mod text_field;

pub use agency::{agency_decide, Action, AgencyState, CommitRecord};
pub use binary_ingress::{
    corrupt_crc, corrupt_length, demo_case_insert_before_frame, encode_frame,
    reassemble_requested_frame, DecodeReason, DecodeResult, END_SYNC, LENGTH_BITS, PAYLOAD_BITS,
    START_SYNC,
};
pub use bit_codec::{bits_from_int, checksum, drop_bit, insert_bit, int_from_bits, safe_filler};
pub use body::{
    flow_cell_for, ground_cell_for, proposal_record_bits, AgencyView, BodyConfig, FieldKind,
    FieldMatrix, LockedBodyRuntime, ProposalField, ProposalFieldError, RuntimeStep, DEFAULT_BODY,
    EXTENDED_BODY, OVERCAPACITY_AVOID_DEFAULT, PROPOSAL_WIDTH_64_CONTROL, RESEARCH_CEILING_BODY,
};
pub use curriculum::{
    CurriculumBlockReason, CurriculumLesson, CurriculumVerdict, RustCurriculumRunner,
};
pub use egress::{render_output, EgressMode, RenderedOutput};
pub use library::{
    PocketLibraryStore, StoreDecision, StoreGuardReason, StoreMutationStats,
    StorePromotionCandidate, StoreSnapshot, StoredPocketArtifact,
};
pub use manager::{
    evaluate_promotion, ChallengerEvidence, PromotionBlockReason, PromotionEvidence,
    PromotionLevel, PromotionVerdict, SafetyGate, ScoreVector,
};
pub use next_mutation::{
    evaluate_next_mutation_lifecycle, GoldenDiscRecord, MutationBlockReason,
    MutationLifecycleStage, MutationStats, NextMutationEvidence, NextMutationVerdict,
    S_RANK_QUALITY_THRESHOLD, UNIQUE_VALUE_THRESHOLD,
};
pub use pocket::{
    active_pocket_set, resolve_pocket_call, ActivePocket, LoadBlockReason, LoadDecision,
    PocketLifecycle, PocketRegistryEntry, PocketToken,
};
pub use proposal::{ingress_to_proposal, Proposal, ProposalKind};
pub use text_field::{select_text_mode, TextMode, TextProfile};

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
    fn corrupted_length_is_not_committed() {
        let frame = corrupt_length(&encode_frame(4, 1, 1, 2));
        let mut stream = safe_filler(8);
        stream.extend(frame);
        let decoded = reassemble_requested_frame(&stream, 4);
        assert_eq!(decoded.action, Action::Defer);
    }

    #[test]
    fn untrusted_requested_frame_is_not_committed() {
        let mut stream = safe_filler(8);
        stream.extend(encode_frame(4, 1, 0, 2));
        let decoded = reassemble_requested_frame(&stream, 4);
        assert_eq!(decoded.action, Action::Defer);
        assert_eq!(decoded.selected_feature, None);
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
    fn dropped_corrupt_frame_can_recover_from_later_repeat() {
        let requested = 6;
        let value = 1;
        let frame = encode_frame(requested, value, 1, 3);
        let dropped = drop_bit(&frame, 18);
        let mut stream = safe_filler(8);
        stream.extend(dropped);
        stream.extend(safe_filler(5));
        stream.extend(frame);
        let decoded = reassemble_requested_frame(&stream, requested);
        assert_eq!(decoded.action, Action::CommitEvidence);
        assert_eq!(decoded.selected_value, Some(value));
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
        assert!(rendered
            .short
            .as_deref()
            .is_some_and(|text| text.contains("Feature 2")));
        assert!(rendered
            .long
            .as_deref()
            .is_some_and(|text| text.contains("Agency committed")));
    }

    #[test]
    fn default_body_matches_e64_lock() {
        assert_eq!(DEFAULT_BODY.flow_side, 28);
        assert_eq!(DEFAULT_BODY.ground_side, 32);
        assert_eq!(DEFAULT_BODY.proposal_slots, 20);
        assert_eq!(DEFAULT_BODY.proposal_bits, 80);
        assert_eq!(DEFAULT_BODY.agency_view_bits, 896);
        assert_eq!(DEFAULT_BODY.flow_cells(), 784);
        assert_eq!(DEFAULT_BODY.ground_cells(), 1024);
        assert_eq!(DEFAULT_BODY.proposal_capacity_bits(), 1600);
    }

    #[test]
    fn proposal_width_64_control_is_too_narrow_for_full_evidence_record() {
        let proposal = Proposal {
            kind: ProposalKind::EvidenceWrite,
            cycle_id: 1,
            source_pocket_id: 42,
            trace_valid: true,
            ground_compatible: true,
            target_feature: Some(7),
            value: Some(1),
        };
        assert!(proposal_record_bits(&proposal) > PROPOSAL_WIDTH_64_CONTROL.proposal_bits);
        let mut field = ProposalField::new(PROPOSAL_WIDTH_64_CONTROL);
        assert_eq!(
            field.push(proposal),
            Err(ProposalFieldError::SlotWidthTooSmall)
        );
    }

    #[test]
    fn default_body_runs_full_ingress_to_egress_cycle() {
        let mut runtime = LockedBodyRuntime::default_body();
        let requested = 9;
        let value = 1;
        let stream = framed_stream(requested, value);
        let step = runtime.process_binary_evidence(42, requested, &stream);
        assert_eq!(step.action, Action::CommitEvidence);
        assert_eq!(
            step.committed.map(|record| record.feature_id),
            Some(requested)
        );
        assert_eq!(step.committed.map(|record| record.value), Some(value));
        assert_eq!(step.proposal_slots_used, 1);
        assert_eq!(
            runtime.flow.read(flow_cell_for(requested, DEFAULT_BODY)),
            Some(value)
        );
        assert_eq!(
            runtime
                .ground
                .read(ground_cell_for(requested, DEFAULT_BODY)),
            Some(value)
        );
        assert_eq!(step.rendered.compact, "COMMIT_EVIDENCE");
        assert!(step
            .rendered
            .long
            .as_deref()
            .is_some_and(|text| text.contains("Agency committed")));
    }

    #[test]
    fn default_body_rejects_invalid_ingress_without_field_mutation() {
        let mut runtime = LockedBodyRuntime::default_body();
        let requested = 4;
        let wrong = 5;
        let mut stream = safe_filler(8);
        stream.extend(encode_frame(wrong, 1, 1, 1));
        let step = runtime.process_binary_evidence(42, requested, &stream);
        assert_ne!(step.action, Action::CommitEvidence);
        assert_eq!(step.committed, None);
        assert_eq!(runtime.flow.active_count(), 0);
        assert_eq!(runtime.ground.active_count(), 0);
        assert_eq!(step.rendered.compact, "NEED_MORE_INFO");
    }

    #[test]
    fn proposal_field_enforces_slot_capacity() {
        let mut field = ProposalField::new(DEFAULT_BODY);
        let proposal = Proposal {
            kind: ProposalKind::EvidenceWrite,
            cycle_id: 1,
            source_pocket_id: 42,
            trace_valid: true,
            ground_compatible: true,
            target_feature: Some(1),
            value: Some(1),
        };
        for _ in 0..DEFAULT_BODY.proposal_slots {
            assert_eq!(field.push(proposal), Ok(()));
        }
        assert_eq!(field.push(proposal), Err(ProposalFieldError::SlotOverflow));
    }
}
