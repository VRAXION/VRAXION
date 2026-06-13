//! Proposal ABI.
//!
//! Pocket output is a temporary proposal, not committed state.

use crate::binary_ingress::{reassemble_requested_frame, DecodeResult};
use crate::Action;

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
