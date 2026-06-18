//! Agency commit boundary.

use crate::proposal::{Proposal, ProposalKind};
use std::collections::{BTreeMap, BTreeSet};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AtomicCommitAction {
    CommitSingle,
    CommitMulti,
    CommitChunk,
    Reject,
    Defer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AtomicProposalRole {
    Primary,
    Rollback,
    HeldChallenger,
    HeldLineage,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AtomicRejectReason {
    StaleSnapshot,
    TraceOrGroundInvalid,
    ChecksumInvalid,
    DirectFlowWrite,
    UnsafeAnswer,
    PrimaryRegression,
    MissingTarget,
    AmbiguousSameRegion,
    AtomicBatchConflict,
    ProposalCapacityExceeded,
    MultiWritePolicyBlocked,
    NoValidProposal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicCommitPolicy {
    pub reject_direct_flow_write: bool,
    pub reject_stale_snapshot: bool,
    pub reject_checksum_tamper: bool,
    pub reject_ambiguous_same_region: bool,
    pub enable_multi_write: bool,
    pub enable_chunk_commit: bool,
    pub enable_rollback_fallback: bool,
    pub enable_rollback_audit_write: bool,
    pub hold_held_variants: bool,
    pub stable_write_order: bool,
    pub max_multi_write: usize,
    pub chunk_min_support: usize,
    pub require_whole_group_for_chunk: bool,
}

impl AtomicCommitPolicy {
    pub const fn e136p_preview() -> Self {
        Self {
            reject_direct_flow_write: true,
            reject_stale_snapshot: true,
            reject_checksum_tamper: true,
            reject_ambiguous_same_region: true,
            enable_multi_write: true,
            enable_chunk_commit: true,
            enable_rollback_fallback: true,
            enable_rollback_audit_write: true,
            hold_held_variants: true,
            stable_write_order: true,
            max_multi_write: 3,
            chunk_min_support: 3,
            require_whole_group_for_chunk: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicCommitProposal {
    pub cycle_id: u32,
    pub source_pocket_id: u32,
    pub role: AtomicProposalRole,
    pub relation_group: u8,
    pub trace_valid: bool,
    pub ground_compatible: bool,
    pub checksum_valid: bool,
    pub direct_flow_write: bool,
    pub unsupported_answer: bool,
    pub hard_negative: bool,
    pub primary_regression_signal: bool,
    pub target_feature: Option<u8>,
    pub value: Option<u8>,
    pub confidence: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicRejectedProposal {
    pub source_pocket_id: u32,
    pub reason: AtomicRejectReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicCommitDecision {
    pub action: AtomicCommitAction,
    pub records: Vec<CommitRecord>,
    pub rejected: Vec<AtomicRejectedProposal>,
    pub reject_reason: Option<AtomicRejectReason>,
    pub requires_child_check: bool,
    pub partial_write_prevented: bool,
    pub order_independent: bool,
    pub runtime_direct_write_count: usize,
    pub oracle_plan_feature_use_count: usize,
}

impl AtomicCommitDecision {
    pub fn rejected(reason: AtomicRejectReason) -> Self {
        Self {
            action: AtomicCommitAction::Reject,
            records: Vec::new(),
            rejected: Vec::new(),
            reject_reason: Some(reason),
            requires_child_check: false,
            partial_write_prevented: true,
            order_independent: true,
            runtime_direct_write_count: 0,
            oracle_plan_feature_use_count: 0,
        }
    }

    pub fn committed(&self) -> bool {
        matches!(
            self.action,
            AtomicCommitAction::CommitSingle
                | AtomicCommitAction::CommitMulti
                | AtomicCommitAction::CommitChunk
        )
    }
}

fn atomic_reject_reason(
    state: AgencyState,
    policy: AtomicCommitPolicy,
    proposal: AtomicCommitProposal,
) -> Option<AtomicRejectReason> {
    if policy.reject_stale_snapshot && proposal.cycle_id != state.cycle_id {
        return Some(AtomicRejectReason::StaleSnapshot);
    }
    if !(proposal.trace_valid && proposal.ground_compatible) {
        return Some(AtomicRejectReason::TraceOrGroundInvalid);
    }
    if policy.reject_checksum_tamper && !proposal.checksum_valid {
        return Some(AtomicRejectReason::ChecksumInvalid);
    }
    if policy.reject_direct_flow_write && proposal.direct_flow_write {
        return Some(AtomicRejectReason::DirectFlowWrite);
    }
    if proposal.unsupported_answer || proposal.hard_negative {
        return Some(AtomicRejectReason::UnsafeAnswer);
    }
    if proposal.primary_regression_signal {
        return Some(AtomicRejectReason::PrimaryRegression);
    }
    if proposal.target_feature.is_none() || proposal.value.is_none() {
        return Some(AtomicRejectReason::MissingTarget);
    }
    None
}

fn highest_confidence(mut proposals: Vec<AtomicCommitProposal>) -> AtomicCommitProposal {
    proposals.sort_by_key(|proposal| {
        (
            proposal.confidence,
            proposal.source_pocket_id,
            proposal.target_feature.unwrap_or_default(),
        )
    });
    *proposals
        .last()
        .expect("highest_confidence requires a non-empty proposal set")
}

fn record_for(proposal: AtomicCommitProposal) -> CommitRecord {
    CommitRecord {
        feature_id: proposal
            .target_feature
            .expect("valid atomic proposal has target_feature"),
        value: proposal.value.expect("valid atomic proposal has value"),
        source_pocket_id: proposal.source_pocket_id,
    }
}

fn finalize_decision(
    action: AtomicCommitAction,
    mut records: Vec<CommitRecord>,
    rejected: Vec<AtomicRejectedProposal>,
    reject_reason: Option<AtomicRejectReason>,
    requires_child_check: bool,
    runtime_direct_write_count: usize,
) -> AtomicCommitDecision {
    if matches!(
        action,
        AtomicCommitAction::Reject | AtomicCommitAction::Defer
    ) {
        records.clear();
    }
    let mut seen: BTreeMap<u8, BTreeSet<u8>> = BTreeMap::new();
    for record in &records {
        seen.entry(record.feature_id)
            .or_default()
            .insert(record.value);
    }
    let conflict = seen.values().any(|values| values.len() > 1);
    if conflict {
        return AtomicCommitDecision {
            action: AtomicCommitAction::Reject,
            records: Vec::new(),
            rejected,
            reject_reason: Some(AtomicRejectReason::AtomicBatchConflict),
            requires_child_check,
            partial_write_prevented: true,
            order_independent: true,
            runtime_direct_write_count,
            oracle_plan_feature_use_count: 0,
        };
    }
    AtomicCommitDecision {
        action,
        partial_write_prevented: matches!(action, AtomicCommitAction::Reject) && records.is_empty(),
        order_independent: true,
        records,
        rejected,
        reject_reason,
        requires_child_check,
        runtime_direct_write_count,
        oracle_plan_feature_use_count: 0,
    }
}

pub fn agency_decide_atomic_batch(
    state: AgencyState,
    policy: AtomicCommitPolicy,
    proposals: &[AtomicCommitProposal],
) -> AtomicCommitDecision {
    let mut rejected = Vec::new();
    let mut valid = Vec::new();
    let runtime_direct_write_count = 0;
    let requires_child_check = proposals.iter().any(|proposal| {
        matches!(
            proposal.role,
            AtomicProposalRole::HeldChallenger | AtomicProposalRole::HeldLineage
        )
    });

    for proposal in proposals {
        if let Some(reason) = atomic_reject_reason(state, policy, *proposal) {
            rejected.push(AtomicRejectedProposal {
                source_pocket_id: proposal.source_pocket_id,
                reason,
            });
        } else {
            valid.push(*proposal);
        }
    }

    let mut by_feature: BTreeMap<u8, Vec<AtomicCommitProposal>> = BTreeMap::new();
    for proposal in valid {
        by_feature
            .entry(proposal.target_feature.expect("valid proposal has feature"))
            .or_default()
            .push(proposal);
    }

    let mut primary_candidates = Vec::new();
    let mut rollback_candidates = Vec::new();
    let mut held_candidates = Vec::new();
    for proposals_for_feature in by_feature.values() {
        let primaries: Vec<_> = proposals_for_feature
            .iter()
            .copied()
            .filter(|proposal| proposal.role == AtomicProposalRole::Primary)
            .collect();
        let primary_values: BTreeSet<u8> = primaries
            .iter()
            .filter_map(|proposal| proposal.value)
            .collect();
        if policy.reject_ambiguous_same_region && primary_values.len() > 1 {
            return finalize_decision(
                AtomicCommitAction::Reject,
                Vec::new(),
                rejected,
                Some(AtomicRejectReason::AmbiguousSameRegion),
                requires_child_check,
                runtime_direct_write_count,
            );
        }
        if !primaries.is_empty() {
            primary_candidates.push(highest_confidence(primaries));
            continue;
        }

        let rollbacks: Vec<_> = proposals_for_feature
            .iter()
            .copied()
            .filter(|proposal| proposal.role == AtomicProposalRole::Rollback)
            .collect();
        if !rollbacks.is_empty() && policy.enable_rollback_fallback {
            rollback_candidates.push(highest_confidence(rollbacks));
            continue;
        }

        let held: Vec<_> = proposals_for_feature
            .iter()
            .copied()
            .filter(|proposal| {
                matches!(
                    proposal.role,
                    AtomicProposalRole::HeldChallenger | AtomicProposalRole::HeldLineage
                )
            })
            .collect();
        if !held.is_empty() {
            held_candidates.push(highest_confidence(held));
        }
    }

    if !primary_candidates.is_empty() {
        primary_candidates.sort_by_key(|proposal| {
            (
                proposal.relation_group,
                proposal.target_feature.unwrap_or_default(),
                proposal.source_pocket_id,
            )
        });
        let whole_group = primary_candidates
            .iter()
            .map(|proposal| proposal.relation_group)
            .collect::<BTreeSet<_>>()
            .len()
            == 1;
        if policy.enable_chunk_commit
            && whole_group
            && primary_candidates.len() >= policy.chunk_min_support
        {
            let selected = primary_candidates
                .into_iter()
                .take(policy.max_multi_write)
                .map(record_for)
                .collect();
            return finalize_decision(
                AtomicCommitAction::CommitChunk,
                selected,
                rejected,
                None,
                requires_child_check,
                runtime_direct_write_count,
            );
        }
        if policy.enable_multi_write && primary_candidates.len() >= 2 {
            let selected = primary_candidates
                .into_iter()
                .take(policy.max_multi_write)
                .map(record_for)
                .collect();
            return finalize_decision(
                AtomicCommitAction::CommitMulti,
                selected,
                rejected,
                None,
                requires_child_check,
                runtime_direct_write_count,
            );
        }
        let selected = vec![record_for(primary_candidates[0])];
        return finalize_decision(
            AtomicCommitAction::CommitSingle,
            selected,
            rejected,
            None,
            requires_child_check,
            runtime_direct_write_count,
        );
    }

    if !rollback_candidates.is_empty() {
        rollback_candidates.sort_by_key(|proposal| {
            (
                proposal.target_feature.unwrap_or_default(),
                proposal.source_pocket_id,
            )
        });
        return finalize_decision(
            AtomicCommitAction::CommitSingle,
            vec![record_for(rollback_candidates[0])],
            rejected,
            None,
            requires_child_check,
            runtime_direct_write_count,
        );
    }

    if !held_candidates.is_empty() && !policy.hold_held_variants {
        held_candidates.sort_by_key(|proposal| {
            (
                proposal.target_feature.unwrap_or_default(),
                proposal.source_pocket_id,
            )
        });
        return finalize_decision(
            AtomicCommitAction::CommitSingle,
            vec![record_for(held_candidates[0])],
            rejected,
            None,
            requires_child_check,
            runtime_direct_write_count,
        );
    }

    finalize_decision(
        AtomicCommitAction::Defer,
        Vec::new(),
        rejected,
        Some(AtomicRejectReason::NoValidProposal),
        requires_child_check,
        runtime_direct_write_count,
    )
}

#[cfg(test)]
mod atomic_tests {
    use super::*;

    fn primary(feature: u8, value: u8, source: u32, group: u8) -> AtomicCommitProposal {
        AtomicCommitProposal {
            cycle_id: 7,
            source_pocket_id: source,
            role: AtomicProposalRole::Primary,
            relation_group: group,
            trace_valid: true,
            ground_compatible: true,
            checksum_valid: true,
            direct_flow_write: false,
            unsupported_answer: false,
            hard_negative: false,
            primary_regression_signal: false,
            target_feature: Some(feature),
            value: Some(value),
            confidence: 200,
        }
    }

    #[test]
    fn atomic_batch_commits_disjoint_primary_records() {
        let decision = agency_decide_atomic_batch(
            AgencyState { cycle_id: 7 },
            AtomicCommitPolicy::e136p_preview(),
            &[primary(1, 1, 10, 1), primary(2, 0, 11, 2)],
        );
        assert_eq!(decision.action, AtomicCommitAction::CommitMulti);
        assert_eq!(decision.records.len(), 2);
        assert!(decision.order_independent);
        assert_eq!(decision.oracle_plan_feature_use_count, 0);
    }

    #[test]
    fn atomic_batch_rejects_ambiguous_same_region_without_partial_write() {
        let decision = agency_decide_atomic_batch(
            AgencyState { cycle_id: 7 },
            AtomicCommitPolicy::e136p_preview(),
            &[primary(1, 0, 10, 1), primary(1, 1, 11, 1)],
        );
        assert_eq!(decision.action, AtomicCommitAction::Reject);
        assert_eq!(
            decision.reject_reason,
            Some(AtomicRejectReason::AmbiguousSameRegion)
        );
        assert!(decision.records.is_empty());
        assert!(decision.partial_write_prevented);
    }

    #[test]
    fn atomic_batch_holds_held_variants_by_default() {
        let mut held = primary(3, 1, 30, 1);
        held.role = AtomicProposalRole::HeldChallenger;
        let decision = agency_decide_atomic_batch(
            AgencyState { cycle_id: 7 },
            AtomicCommitPolicy::e136p_preview(),
            &[held],
        );
        assert_eq!(decision.action, AtomicCommitAction::Defer);
        assert!(decision.records.is_empty());
        assert!(decision.requires_child_check);
    }
}
