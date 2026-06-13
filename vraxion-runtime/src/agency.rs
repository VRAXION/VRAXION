//! Agency commit boundary.

use crate::proposal::{Proposal, ProposalKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    CommitEvidence,
    Reject,
    Defer,
    AskOrMultiCycle,
    AnswerReady,
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
