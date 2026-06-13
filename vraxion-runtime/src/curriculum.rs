//! Deterministic curriculum runner glue.
//!
//! E70 connects the already locked Rust runtime pieces into one row-level
//! curriculum step:
//!
//! Pocket Library active set -> guarded load -> LockedBodyRuntime evidence
//! commit -> trace-backed egress -> Next Mutation lifecycle -> safe store
//! promotion.

use crate::{
    active_pocket_set, encode_frame, flow_cell_for, ground_cell_for, safe_filler, Action,
    BodyConfig, LockedBodyRuntime, NextMutationEvidence, PocketLibraryStore, PromotionEvidence,
    StoreGuardReason, StorePromotionCandidate, DEFAULT_BODY,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurriculumBlockReason {
    None,
    NoActivePocket,
    GuardedLoadBlocked(StoreGuardReason),
    RuntimeDidNotCommit,
    FlowGroundMismatch,
    TraceRenderMissing,
    PromotionBlocked(StoreGuardReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurriculumLesson {
    pub requested_feature: u8,
    pub value: u8,
    pub source_pocket_id: u32,
    pub nonce: u8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurriculumVerdict {
    pub passed: bool,
    pub reason: CurriculumBlockReason,
    pub action: Action,
    pub active_pocket_count: usize,
    pub proposal_slots_used: usize,
    pub flow_value: Option<u8>,
    pub ground_value: Option<u8>,
    pub promoted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurriculumQueueLesson {
    pub family: &'static str,
    pub lesson: CurriculumLesson,
    pub candidate: StorePromotionCandidate,
    pub lifecycle_evidence: NextMutationEvidence,
    pub promotion_evidence: PromotionEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurriculumQueueReport {
    pub lesson_count: usize,
    pub passed_count: usize,
    pub promoted_count: usize,
    pub flow_ground_sync_count: usize,
    pub proposal_boundary_count: usize,
    pub reload_match: bool,
    pub quality_delta: f32,
    pub first_failure: CurriculumBlockReason,
}

impl CurriculumQueueReport {
    pub fn passed(self) -> bool {
        self.lesson_count > 0
            && self.lesson_count == self.passed_count
            && self.lesson_count == self.promoted_count
            && self.lesson_count == self.flow_ground_sync_count
            && self.lesson_count == self.proposal_boundary_count
            && self.reload_match
            && self.quality_delta > 0.0
            && self.first_failure == CurriculumBlockReason::None
    }
}

#[derive(Debug, Clone)]
pub struct RustCurriculumRunner {
    pub body: LockedBodyRuntime,
    pub store: PocketLibraryStore,
    pub active_set_limit: usize,
}

impl RustCurriculumRunner {
    pub fn new(store: PocketLibraryStore, config: BodyConfig, active_set_limit: usize) -> Self {
        Self {
            body: LockedBodyRuntime::new(config),
            store,
            active_set_limit,
        }
    }

    pub fn default_body(store: PocketLibraryStore) -> Self {
        Self::new(store, DEFAULT_BODY, 4)
    }

    pub fn run_binary_lesson(
        &mut self,
        lesson: CurriculumLesson,
        candidate: StorePromotionCandidate,
        lifecycle_evidence: NextMutationEvidence,
        promotion_evidence: PromotionEvidence,
    ) -> CurriculumVerdict {
        let mut stream = safe_filler(8);
        stream.extend(encode_frame(
            lesson.requested_feature,
            lesson.value,
            1,
            lesson.nonce,
        ));
        stream.extend(safe_filler(8));
        self.run_binary_lesson_with_stream(
            lesson,
            &stream,
            candidate,
            lifecycle_evidence,
            promotion_evidence,
        )
    }

    pub fn run_binary_lesson_with_stream(
        &mut self,
        lesson: CurriculumLesson,
        stream: &[u8],
        candidate: StorePromotionCandidate,
        lifecycle_evidence: NextMutationEvidence,
        promotion_evidence: PromotionEvidence,
    ) -> CurriculumVerdict {
        let active = active_pocket_set(
            &self.store.tokens,
            &self.store.registry,
            self.active_set_limit,
        );
        if active.is_empty() {
            return CurriculumVerdict {
                passed: false,
                reason: CurriculumBlockReason::NoActivePocket,
                action: Action::Defer,
                active_pocket_count: 0,
                proposal_slots_used: 0,
                flow_value: None,
                ground_value: None,
                promoted: false,
            };
        }
        let selected_uid = active[0].pocket_uid;
        let Some(selected_token) = self
            .store
            .tokens
            .iter()
            .copied()
            .find(|token| token.pocket_uid == selected_uid)
        else {
            return CurriculumVerdict {
                passed: false,
                reason: CurriculumBlockReason::NoActivePocket,
                action: Action::Defer,
                active_pocket_count: active.len(),
                proposal_slots_used: 0,
                flow_value: None,
                ground_value: None,
                promoted: false,
            };
        };
        let load = self.store.guarded_load(selected_token);
        if !load.allowed {
            return CurriculumVerdict {
                passed: false,
                reason: CurriculumBlockReason::GuardedLoadBlocked(load.reason),
                action: Action::Reject,
                active_pocket_count: active.len(),
                proposal_slots_used: 0,
                flow_value: None,
                ground_value: None,
                promoted: false,
            };
        }

        let step = self.body.process_binary_evidence(
            lesson.source_pocket_id,
            lesson.requested_feature,
            stream,
        );
        let flow_value = self
            .body
            .flow
            .read(flow_cell_for(lesson.requested_feature, self.body.config));
        let ground_value = self
            .body
            .ground
            .read(ground_cell_for(lesson.requested_feature, self.body.config));
        if step.action != Action::CommitEvidence || step.committed.is_none() {
            return CurriculumVerdict {
                passed: false,
                reason: CurriculumBlockReason::RuntimeDidNotCommit,
                action: step.action,
                active_pocket_count: active.len(),
                proposal_slots_used: step.proposal_slots_used,
                flow_value,
                ground_value,
                promoted: false,
            };
        }
        if flow_value != Some(lesson.value) || ground_value != Some(lesson.value) {
            return CurriculumVerdict {
                passed: false,
                reason: CurriculumBlockReason::FlowGroundMismatch,
                action: step.action,
                active_pocket_count: active.len(),
                proposal_slots_used: step.proposal_slots_used,
                flow_value,
                ground_value,
                promoted: false,
            };
        }
        let trace_ok = step
            .rendered
            .long
            .as_deref()
            .is_some_and(|text| text.contains("Agency committed"));
        if !trace_ok {
            return CurriculumVerdict {
                passed: false,
                reason: CurriculumBlockReason::TraceRenderMissing,
                action: step.action,
                active_pocket_count: active.len(),
                proposal_slots_used: step.proposal_slots_used,
                flow_value,
                ground_value,
                promoted: false,
            };
        }

        let promotion =
            self.store
                .promote_candidate(candidate, lifecycle_evidence, promotion_evidence);
        self.body.cycle_id = self.body.cycle_id.saturating_add(1);
        if !promotion.allowed {
            return CurriculumVerdict {
                passed: false,
                reason: CurriculumBlockReason::PromotionBlocked(promotion.reason),
                action: step.action,
                active_pocket_count: active.len(),
                proposal_slots_used: step.proposal_slots_used,
                flow_value,
                ground_value,
                promoted: false,
            };
        }

        CurriculumVerdict {
            passed: true,
            reason: CurriculumBlockReason::None,
            action: step.action,
            active_pocket_count: active.len(),
            proposal_slots_used: step.proposal_slots_used,
            flow_value,
            ground_value,
            promoted: true,
        }
    }

    pub fn run_queue(&mut self, lessons: &[CurriculumQueueLesson]) -> CurriculumQueueReport {
        let mut passed_count = 0;
        let mut promoted_count = 0;
        let mut flow_ground_sync_count = 0;
        let mut proposal_boundary_count = 0;
        let mut first_failure = CurriculumBlockReason::None;

        for queue_lesson in lessons {
            let verdict = self.run_binary_lesson(
                queue_lesson.lesson,
                queue_lesson.candidate,
                queue_lesson.lifecycle_evidence,
                queue_lesson.promotion_evidence,
            );
            passed_count += verdict.passed as usize;
            promoted_count += verdict.promoted as usize;
            flow_ground_sync_count += (verdict.flow_value == verdict.ground_value
                && verdict.flow_value == Some(queue_lesson.lesson.value))
                as usize;
            proposal_boundary_count += (verdict.proposal_slots_used == 1) as usize;
            if first_failure == CurriculumBlockReason::None && !verdict.passed {
                first_failure = verdict.reason;
            }
        }

        let snapshot = self.store.snapshot();
        CurriculumQueueReport {
            lesson_count: lessons.len(),
            passed_count,
            promoted_count,
            flow_ground_sync_count,
            proposal_boundary_count,
            reload_match: self.store.reload_matches(snapshot),
            quality_delta: self.store.quality_delta(),
            first_failure,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ChallengerEvidence, MutationStats, PocketLifecycle, PocketRegistryEntry, PocketToken,
        SafetyGate, ScoreVector, StoredPocketArtifact,
    };

    fn token(uid: &'static str, lifecycle_score: f32) -> PocketToken {
        PocketToken {
            pocket_uid: uid,
            token_version: 4,
            min_token_version: 3,
            token_hash: "tok_ingress",
            content_digest: "digest_ingress",
            abi_version: "PocketABI-v1",
            capability_signature: "binary_ingress",
            utility_score: lifecycle_score,
            safety_score: 0.99,
            reuse_score: 0.82,
            cost_score: 0.06,
        }
    }

    fn entry(uid: &'static str, lifecycle: PocketLifecycle) -> PocketRegistryEntry {
        PocketRegistryEntry {
            pocket_uid: uid,
            human_alias: "binary_ingress",
            artifact_path: "persistent_library/artifacts/ingress.json",
            content_digest: "digest_ingress",
            token_hash: "tok_ingress",
            abi_version: "PocketABI-v1",
            capability_signature: "binary_ingress",
            lifecycle,
        }
    }

    fn artifact(uid: &'static str) -> StoredPocketArtifact {
        StoredPocketArtifact {
            pocket_uid: uid,
            content_digest: "digest_ingress",
            token_hash: "tok_ingress",
            abi_version: "PocketABI-v1",
            quality_delta: 0.0,
            generation: 1,
        }
    }

    fn store(lifecycle: PocketLifecycle) -> PocketLibraryStore {
        let mut store = PocketLibraryStore::new();
        store.insert_pocket(
            entry("pkt_ingress", lifecycle),
            token("pkt_ingress", 0.94),
            artifact("pkt_ingress"),
        );
        store
    }

    fn lifecycle() -> NextMutationEvidence {
        NextMutationEvidence {
            active_slot_count: 1,
            sandbox_only: true,
            proposal_only: true,
            light_probe_passed: true,
            initial_quality: 0.88,
            refined_quality: 0.999,
            golden_disc_quality: 1.0,
            unique_value_score: 0.12,
            mutation_stats: MutationStats {
                attempts: 128,
                accepted: 8,
                rejected: 120,
                rollback_count: 120,
                attempts_to_s_rank: Some(24),
            },
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

    fn promotion() -> PromotionEvidence {
        PromotionEvidence {
            score: ScoreVector {
                utility: 0.96,
                safety: 0.99,
                eligible_activation: 0.90,
                generality: 0.95,
                uniqueness: 0.94,
                transfer: 0.95,
                robustness: 0.96,
                cost: 0.08,
                stability: 0.98,
                scope_clarity: 0.95,
            },
            safety: SafetyGate {
                trace_safe: true,
                no_direct_flow_write: true,
                no_credit_hijack: true,
                no_delayed_poison: true,
                no_negative_transfer: true,
                no_unsafe_high_utility: true,
                no_scope_violation: true,
            },
            challenger: ChallengerEvidence {
                challenger_sweep_passed: true,
                counterfactual_unique: true,
                reload_shadow_import_passed: true,
                long_horizon_no_harm: true,
                redundant_clone_rejected: true,
                rare_critical_preserved: true,
            },
            rare_critical: false,
            global_scope_allowed: false,
        }
    }

    fn candidate(uid: &'static str) -> StorePromotionCandidate {
        StorePromotionCandidate {
            pocket_uid: uid,
            human_alias: "curriculum_promoted",
            content_digest: "digest_promoted",
            token_hash: "tok_promoted",
            capability_signature: "binary_ingress_repair",
            quality_delta: 0.044,
        }
    }

    fn queue_lesson(uid: &'static str, feature: u8, value: u8) -> CurriculumQueueLesson {
        CurriculumQueueLesson {
            family: "binary_ingress_queue",
            lesson: CurriculumLesson {
                requested_feature: feature,
                value,
                source_pocket_id: 42,
                nonce: feature.wrapping_mul(3),
            },
            candidate: candidate(uid),
            lifecycle_evidence: lifecycle(),
            promotion_evidence: promotion(),
        }
    }

    #[test]
    fn curriculum_row_loads_commits_renders_and_promotes() {
        let mut runner = RustCurriculumRunner::default_body(store(PocketLifecycle::Stable));
        let verdict = runner.run_binary_lesson(
            CurriculumLesson {
                requested_feature: 7,
                value: 1,
                source_pocket_id: 42,
                nonce: 3,
            },
            candidate("gold_curriculum"),
            lifecycle(),
            promotion(),
        );
        assert!(verdict.passed);
        assert_eq!(verdict.action, Action::CommitEvidence);
        assert_eq!(verdict.flow_value, Some(1));
        assert_eq!(verdict.ground_value, Some(1));
        assert!(verdict.promoted);
        assert_eq!(runner.store.snapshot().artifact_count, 2);
    }

    #[test]
    fn no_active_pocket_blocks_before_runtime() {
        let mut runner = RustCurriculumRunner::default_body(store(PocketLifecycle::Quarantine));
        let verdict = runner.run_binary_lesson(
            CurriculumLesson {
                requested_feature: 7,
                value: 1,
                source_pocket_id: 42,
                nonce: 3,
            },
            candidate("gold_curriculum"),
            lifecycle(),
            promotion(),
        );
        assert!(!verdict.passed);
        assert_eq!(verdict.reason, CurriculumBlockReason::NoActivePocket);
        assert_eq!(runner.body.flow.active_count(), 0);
    }

    #[test]
    fn bad_frame_does_not_promote() {
        let mut runner = RustCurriculumRunner::default_body(store(PocketLifecycle::Stable));
        let mut stream = safe_filler(8);
        stream.extend(encode_frame(8, 1, 1, 3));
        let verdict = runner.run_binary_lesson_with_stream(
            CurriculumLesson {
                requested_feature: 7,
                value: 1,
                source_pocket_id: 42,
                nonce: 3,
            },
            &stream,
            candidate("gold_curriculum"),
            lifecycle(),
            promotion(),
        );
        assert!(!verdict.passed);
        assert_eq!(verdict.reason, CurriculumBlockReason::RuntimeDidNotCommit);
        assert_eq!(runner.store.snapshot().artifact_count, 1);
    }

    #[test]
    fn curriculum_queue_promotes_multiple_lessons_with_reload_match() {
        let mut runner = RustCurriculumRunner::default_body(store(PocketLifecycle::Stable));
        let lessons = [
            queue_lesson("gold_queue_a", 3, 1),
            queue_lesson("gold_queue_b", 4, 0),
            queue_lesson("gold_queue_c", 9, 1),
        ];
        let report = runner.run_queue(&lessons);
        assert!(report.passed());
        assert_eq!(report.lesson_count, 3);
        assert_eq!(report.promoted_count, 3);
        assert!(report.reload_match);
        assert!(report.quality_delta > 0.0);
        assert_eq!(runner.store.snapshot().artifact_count, 4);
    }
}
