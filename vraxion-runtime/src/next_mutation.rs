//! Next Mutation slot lifecycle.
//!
//! This module captures the E51 rule: exactly one active candidate may move
//! from sandboxed light probe through mutation/rollback refinement and
//! challenger-backed S-rank before a frozen Golden Disc record can be saved.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationLifecycleStage {
    NextMutation,
    LightProbePass,
    ActiveRefinement,
    Stable,
    SRank,
    GoldenDisc,
    Discard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationBlockReason {
    None,
    EmptySlot,
    MultipleActiveSlots,
    SandboxViolation,
    DirectFlowWrite,
    LightProbeFailed,
    MissingMutationEvidence,
    RollbackMismatch,
    RefinementDidNotImprove,
    PruneUnstable,
    UniqueValueTooLow,
    ChallengerFoundBetterMutation,
    TraceReplayFailed,
    WrongCommit,
    QualityBelowSRank,
    MissingFrozenIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MutationStats {
    pub attempts: u32,
    pub accepted: u32,
    pub rejected: u32,
    pub rollback_count: u32,
    pub attempts_to_s_rank: Option<u32>,
}

impl MutationStats {
    pub fn has_refinement_evidence(self) -> bool {
        self.attempts > 0
            && self.accepted > 0
            && self.rejected > 0
            && self.attempts == self.accepted + self.rejected
    }

    pub fn rollback_consistent(self) -> bool {
        self.rollback_count == self.rejected
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NextMutationEvidence {
    pub active_slot_count: u8,
    pub sandbox_only: bool,
    pub proposal_only: bool,
    pub light_probe_passed: bool,
    pub initial_quality: f32,
    pub refined_quality: f32,
    pub golden_disc_quality: f32,
    pub unique_value_score: f32,
    pub mutation_stats: MutationStats,
    pub prune_stability_passed: bool,
    pub challenger_defense_passed: bool,
    pub trace_replay_passed: bool,
    pub wrong_commit_rate: f32,
    pub direct_flow_write_violation_rate: f32,
    pub pocket_uid_present: bool,
    pub content_digest_present: bool,
    pub token_metadata_present: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldenDiscRecord {
    pub pocket_uid: &'static str,
    pub lifecycle: &'static str,
    pub frozen_anchor: bool,
    pub mutable_working_copy_allowed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NextMutationVerdict {
    pub stage: MutationLifecycleStage,
    pub reason: MutationBlockReason,
    pub golden_disc: Option<GoldenDiscRecord>,
}

pub const S_RANK_QUALITY_THRESHOLD: f32 = 0.999;
pub const UNIQUE_VALUE_THRESHOLD: f32 = 0.05;

pub fn evaluate_next_mutation_lifecycle(evidence: NextMutationEvidence) -> NextMutationVerdict {
    if evidence.active_slot_count == 0 {
        return discard(MutationBlockReason::EmptySlot);
    }
    if evidence.active_slot_count > 1 {
        return discard(MutationBlockReason::MultipleActiveSlots);
    }
    if !evidence.sandbox_only {
        return discard(MutationBlockReason::SandboxViolation);
    }
    if !evidence.proposal_only || evidence.direct_flow_write_violation_rate > 0.0 {
        return discard(MutationBlockReason::DirectFlowWrite);
    }
    if !evidence.light_probe_passed {
        return discard(MutationBlockReason::LightProbeFailed);
    }
    if !evidence.mutation_stats.has_refinement_evidence() {
        return NextMutationVerdict {
            stage: MutationLifecycleStage::LightProbePass,
            reason: MutationBlockReason::MissingMutationEvidence,
            golden_disc: None,
        };
    }
    if !evidence.mutation_stats.rollback_consistent() {
        return NextMutationVerdict {
            stage: MutationLifecycleStage::ActiveRefinement,
            reason: MutationBlockReason::RollbackMismatch,
            golden_disc: None,
        };
    }
    if evidence.refined_quality <= evidence.initial_quality {
        return NextMutationVerdict {
            stage: MutationLifecycleStage::ActiveRefinement,
            reason: MutationBlockReason::RefinementDidNotImprove,
            golden_disc: None,
        };
    }
    if !evidence.prune_stability_passed {
        return NextMutationVerdict {
            stage: MutationLifecycleStage::Stable,
            reason: MutationBlockReason::PruneUnstable,
            golden_disc: None,
        };
    }
    if evidence.unique_value_score < UNIQUE_VALUE_THRESHOLD {
        return NextMutationVerdict {
            stage: MutationLifecycleStage::Stable,
            reason: MutationBlockReason::UniqueValueTooLow,
            golden_disc: None,
        };
    }
    if !evidence.challenger_defense_passed {
        return NextMutationVerdict {
            stage: MutationLifecycleStage::SRank,
            reason: MutationBlockReason::ChallengerFoundBetterMutation,
            golden_disc: None,
        };
    }
    if !evidence.trace_replay_passed {
        return NextMutationVerdict {
            stage: MutationLifecycleStage::SRank,
            reason: MutationBlockReason::TraceReplayFailed,
            golden_disc: None,
        };
    }
    if evidence.wrong_commit_rate > 0.0 {
        return NextMutationVerdict {
            stage: MutationLifecycleStage::SRank,
            reason: MutationBlockReason::WrongCommit,
            golden_disc: None,
        };
    }
    if evidence.golden_disc_quality < S_RANK_QUALITY_THRESHOLD {
        return NextMutationVerdict {
            stage: MutationLifecycleStage::SRank,
            reason: MutationBlockReason::QualityBelowSRank,
            golden_disc: None,
        };
    }
    if !(evidence.pocket_uid_present
        && evidence.content_digest_present
        && evidence.token_metadata_present)
    {
        return NextMutationVerdict {
            stage: MutationLifecycleStage::SRank,
            reason: MutationBlockReason::MissingFrozenIdentity,
            golden_disc: None,
        };
    }

    NextMutationVerdict {
        stage: MutationLifecycleStage::GoldenDisc,
        reason: MutationBlockReason::None,
        golden_disc: Some(GoldenDiscRecord {
            pocket_uid: "gold_candidate",
            lifecycle: "golden_disc",
            frozen_anchor: true,
            mutable_working_copy_allowed: true,
        }),
    }
}

fn discard(reason: MutationBlockReason) -> NextMutationVerdict {
    NextMutationVerdict {
        stage: MutationLifecycleStage::Discard,
        reason,
        golden_disc: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_stats() -> MutationStats {
        MutationStats {
            attempts: 648,
            accepted: 11,
            rejected: 637,
            rollback_count: 637,
            attempts_to_s_rank: Some(37),
        }
    }

    fn good_evidence() -> NextMutationEvidence {
        NextMutationEvidence {
            active_slot_count: 1,
            sandbox_only: true,
            proposal_only: true,
            light_probe_passed: true,
            initial_quality: 0.87,
            refined_quality: 0.999,
            golden_disc_quality: 1.0,
            unique_value_score: 0.132,
            mutation_stats: good_stats(),
            prune_stability_passed: true,
            challenger_defense_passed: true,
            trace_replay_passed: true,
            wrong_commit_rate: 0.0,
            direct_flow_write_violation_rate: 0.0,
            pocket_uid_present: true,
            content_digest_present: true,
            token_metadata_present: true,
        }
    }

    #[test]
    fn primary_path_saves_one_golden_disc() {
        let verdict = evaluate_next_mutation_lifecycle(good_evidence());
        assert_eq!(verdict.stage, MutationLifecycleStage::GoldenDisc);
        assert!(verdict.golden_disc.is_some());
    }

    #[test]
    fn parallel_candidate_spam_is_discarded() {
        let mut evidence = good_evidence();
        evidence.active_slot_count = 4;
        evidence.direct_flow_write_violation_rate = 0.857;
        let verdict = evaluate_next_mutation_lifecycle(evidence);
        assert_eq!(verdict.stage, MutationLifecycleStage::Discard);
        assert_eq!(verdict.reason, MutationBlockReason::MultipleActiveSlots);
    }

    #[test]
    fn light_probe_only_cannot_promote() {
        let mut evidence = good_evidence();
        evidence.mutation_stats = MutationStats {
            attempts: 0,
            accepted: 0,
            rejected: 0,
            rollback_count: 0,
            attempts_to_s_rank: None,
        };
        let verdict = evaluate_next_mutation_lifecycle(evidence);
        assert_eq!(verdict.stage, MutationLifecycleStage::LightProbePass);
        assert_eq!(verdict.reason, MutationBlockReason::MissingMutationEvidence);
        assert!(verdict.golden_disc.is_none());
    }

    #[test]
    fn rollback_mismatch_blocks_refinement() {
        let mut evidence = good_evidence();
        evidence.mutation_stats.rollback_count = 636;
        let verdict = evaluate_next_mutation_lifecycle(evidence);
        assert_eq!(verdict.stage, MutationLifecycleStage::ActiveRefinement);
        assert_eq!(verdict.reason, MutationBlockReason::RollbackMismatch);
    }

    #[test]
    fn uniqueness_is_required_before_s_rank() {
        let mut evidence = good_evidence();
        evidence.unique_value_score = 0.031;
        let verdict = evaluate_next_mutation_lifecycle(evidence);
        assert_eq!(verdict.stage, MutationLifecycleStage::Stable);
        assert_eq!(verdict.reason, MutationBlockReason::UniqueValueTooLow);
    }

    #[test]
    fn direct_flow_write_discarded_before_quality() {
        let mut evidence = good_evidence();
        evidence.direct_flow_write_violation_rate = 0.001;
        evidence.golden_disc_quality = 1.0;
        let verdict = evaluate_next_mutation_lifecycle(evidence);
        assert_eq!(verdict.stage, MutationLifecycleStage::Discard);
        assert_eq!(verdict.reason, MutationBlockReason::DirectFlowWrite);
    }
}
