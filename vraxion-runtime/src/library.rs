//! Persistent Pocket Library store model.
//!
//! E69 moves the E54 Python reference store into the Rust runtime surface. The
//! module is intentionally small: it models the persistent registry, tokens,
//! artifacts, ledgers, guarded load, stale-write blocking, and safe promotion
//! pipeline without becoming a training executor.

use crate::{
    evaluate_next_mutation_lifecycle, evaluate_promotion, next_mutation::NextMutationEvidence,
    resolve_pocket_call, LoadBlockReason, PocketLifecycle, PocketRegistryEntry, PocketToken,
    PromotionEvidence, PromotionLevel,
};

pub use crate::next_mutation::MutationStats as StoreMutationStats;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreGuardReason {
    None,
    LoadBlocked(LoadBlockReason),
    ArtifactMissing,
    DirectArtifactTamper,
    ConcurrentStaleWrite,
    UnsafePromotion,
    PromotionEvidenceInsufficient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreDecision {
    pub allowed: bool,
    pub pocket_uid: &'static str,
    pub reason: StoreGuardReason,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StoredPocketArtifact {
    pub pocket_uid: &'static str,
    pub content_digest: &'static str,
    pub token_hash: &'static str,
    pub abi_version: &'static str,
    pub quality_delta: f32,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LibraryLedgers {
    pub lifecycle_rows: u64,
    pub access_rows: u64,
    pub promotion_rows: u64,
    pub score_rows: u64,
}

impl LibraryLedgers {
    pub fn complete(self) -> bool {
        self.lifecycle_rows > 0
            && self.access_rows > 0
            && self.promotion_rows > 0
            && self.score_rows > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreSnapshot {
    pub registry_entry_count: usize,
    pub token_count: usize,
    pub artifact_count: usize,
    pub generation: u64,
    pub ledger_complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StorePromotionCandidate {
    pub pocket_uid: &'static str,
    pub human_alias: &'static str,
    pub content_digest: &'static str,
    pub token_hash: &'static str,
    pub capability_signature: &'static str,
    pub quality_delta: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PocketLibraryStore {
    pub registry: Vec<PocketRegistryEntry>,
    pub tokens: Vec<PocketToken>,
    pub artifacts: Vec<StoredPocketArtifact>,
    pub ledgers: LibraryLedgers,
    pub generation: u64,
}

impl PocketLibraryStore {
    pub fn new() -> Self {
        Self {
            registry: Vec::new(),
            tokens: Vec::new(),
            artifacts: Vec::new(),
            ledgers: LibraryLedgers::default(),
            generation: 0,
        }
    }

    pub fn insert_pocket(
        &mut self,
        entry: PocketRegistryEntry,
        pocket_descriptor: PocketToken,
        artifact: StoredPocketArtifact,
    ) {
        if let Some(position) = self
            .registry
            .iter()
            .position(|existing| existing.pocket_uid == entry.pocket_uid)
        {
            self.registry[position] = entry;
            self.tokens[position] = pocket_descriptor;
            self.artifacts[position] = artifact;
        } else {
            self.registry.push(entry);
            self.tokens.push(pocket_descriptor);
            self.artifacts.push(artifact);
        }
        self.generation += 1;
        self.ledgers.lifecycle_rows += 1;
        self.ledgers.score_rows += 1;
    }

    pub fn rename_alias(&mut self, pocket_uid: &'static str, alias: &'static str) -> bool {
        let Some(entry) = self
            .registry
            .iter_mut()
            .find(|entry| entry.pocket_uid == pocket_uid)
        else {
            return false;
        };
        entry.human_alias = alias;
        self.generation += 1;
        self.ledgers.lifecycle_rows += 1;
        true
    }

    pub fn guarded_load(&mut self, pocket_descriptor: PocketToken) -> StoreDecision {
        self.ledgers.access_rows += 1;
        let load = resolve_pocket_call(pocket_descriptor, &self.registry);
        if !load.allowed {
            return StoreDecision {
                allowed: false,
                pocket_uid: pocket_descriptor.pocket_uid,
                reason: StoreGuardReason::LoadBlocked(
                    load.reason.unwrap_or(LoadBlockReason::UidMissing),
                ),
            };
        }
        let Some(entry) = self
            .registry
            .iter()
            .find(|entry| entry.pocket_uid == pocket_descriptor.pocket_uid)
        else {
            return StoreDecision {
                allowed: false,
                pocket_uid: pocket_descriptor.pocket_uid,
                reason: StoreGuardReason::LoadBlocked(LoadBlockReason::UidMissing),
            };
        };
        let Some(artifact) = self
            .artifacts
            .iter()
            .find(|artifact| artifact.pocket_uid == pocket_descriptor.pocket_uid)
        else {
            return StoreDecision {
                allowed: false,
                pocket_uid: pocket_descriptor.pocket_uid,
                reason: StoreGuardReason::ArtifactMissing,
            };
        };
        if artifact.content_digest != entry.content_digest
            || artifact.content_digest != pocket_descriptor.content_digest
            || artifact.token_hash != entry.token_hash
        {
            return StoreDecision {
                allowed: false,
                pocket_uid: pocket_descriptor.pocket_uid,
                reason: StoreGuardReason::DirectArtifactTamper,
            };
        }
        StoreDecision {
            allowed: true,
            pocket_uid: pocket_descriptor.pocket_uid,
            reason: StoreGuardReason::None,
        }
    }

    pub fn concurrent_write_guard(&self, expected_generation: u64) -> StoreDecision {
        if expected_generation == self.generation {
            StoreDecision {
                allowed: true,
                pocket_uid: "store",
                reason: StoreGuardReason::None,
            }
        } else {
            StoreDecision {
                allowed: false,
                pocket_uid: "store",
                reason: StoreGuardReason::ConcurrentStaleWrite,
            }
        }
    }

    pub fn promote_candidate(
        &mut self,
        candidate: StorePromotionCandidate,
        lifecycle_evidence: NextMutationEvidence,
        promotion_evidence: PromotionEvidence,
    ) -> StoreDecision {
        let lifecycle = evaluate_next_mutation_lifecycle(lifecycle_evidence);
        let promotion = evaluate_promotion(promotion_evidence);
        self.ledgers.promotion_rows += 1;
        self.ledgers.score_rows += 1;
        if lifecycle.golden_disc.is_none() {
            return StoreDecision {
                allowed: false,
                pocket_uid: candidate.pocket_uid,
                reason: StoreGuardReason::UnsafePromotion,
            };
        }
        if !promotion.level.is_core_or_above()
            && !matches!(
                promotion.level,
                PromotionLevel::LocalGolden | PromotionLevel::SemiPerma
            )
        {
            return StoreDecision {
                allowed: false,
                pocket_uid: candidate.pocket_uid,
                reason: StoreGuardReason::PromotionEvidenceInsufficient,
            };
        }
        let lifecycle = if promotion.level.is_core_or_above() {
            PocketLifecycle::Core
        } else {
            PocketLifecycle::Stable
        };
        let entry = PocketRegistryEntry {
            pocket_uid: candidate.pocket_uid,
            human_alias: candidate.human_alias,
            artifact_path: "persistent_library/artifacts/promoted.json",
            content_digest: candidate.content_digest,
            token_hash: candidate.token_hash,
            abi_version: "PocketABI-v1",
            capability_signature: candidate.capability_signature,
            lifecycle,
        };
        let pocket_descriptor = PocketToken {
            pocket_uid: candidate.pocket_uid,
            token_version: 4,
            min_token_version: 3,
            token_hash: candidate.token_hash,
            content_digest: candidate.content_digest,
            abi_version: "PocketABI-v1",
            capability_signature: candidate.capability_signature,
            utility_score: 0.96,
            safety_score: 0.99,
            reuse_score: 0.82,
            cost_score: 0.07,
        };
        let artifact = StoredPocketArtifact {
            pocket_uid: candidate.pocket_uid,
            content_digest: candidate.content_digest,
            token_hash: candidate.token_hash,
            abi_version: "PocketABI-v1",
            quality_delta: candidate.quality_delta,
            generation: self.generation + 1,
        };
        self.insert_pocket(entry, pocket_descriptor, artifact);
        StoreDecision {
            allowed: true,
            pocket_uid: candidate.pocket_uid,
            reason: StoreGuardReason::None,
        }
    }

    pub fn snapshot(&self) -> StoreSnapshot {
        StoreSnapshot {
            registry_entry_count: self.registry.len(),
            token_count: self.tokens.len(),
            artifact_count: self.artifacts.len(),
            generation: self.generation,
            ledger_complete: self.ledgers.complete(),
        }
    }

    pub fn reload_matches(&self, snapshot: StoreSnapshot) -> bool {
        self.snapshot() == snapshot
    }

    pub fn quality_delta(&self) -> f32 {
        self.artifacts
            .iter()
            .map(|artifact| artifact.quality_delta)
            .sum()
    }
}

impl Default for PocketLibraryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChallengerEvidence, NextMutationEvidence, SafetyGate, ScoreVector};

    fn token(uid: &'static str) -> PocketToken {
        PocketToken {
            pocket_uid: uid,
            token_version: 4,
            min_token_version: 3,
            token_hash: "tok_a",
            content_digest: "digest_a",
            abi_version: "PocketABI-v1",
            capability_signature: "binary_ingress",
            utility_score: 0.90,
            safety_score: 0.99,
            reuse_score: 0.80,
            cost_score: 0.05,
        }
    }

    fn entry(uid: &'static str, lifecycle: PocketLifecycle) -> PocketRegistryEntry {
        PocketRegistryEntry {
            pocket_uid: uid,
            human_alias: "alias",
            artifact_path: "persistent_library/artifacts/a.json",
            content_digest: "digest_a",
            token_hash: "tok_a",
            abi_version: "PocketABI-v1",
            capability_signature: "binary_ingress",
            lifecycle,
        }
    }

    fn artifact(uid: &'static str) -> StoredPocketArtifact {
        StoredPocketArtifact {
            pocket_uid: uid,
            content_digest: "digest_a",
            token_hash: "tok_a",
            abi_version: "PocketABI-v1",
            quality_delta: 0.05,
            generation: 1,
        }
    }

    fn good_lifecycle() -> NextMutationEvidence {
        NextMutationEvidence {
            active_slot_count: 1,
            sandbox_only: true,
            proposal_only: true,
            light_probe_passed: true,
            initial_quality: 0.87,
            refined_quality: 0.999,
            golden_disc_quality: 1.0,
            unique_value_score: 0.132,
            mutation_stats: StoreMutationStats {
                attempts: 648,
                accepted: 11,
                rejected: 637,
                rollback_count: 637,
                attempts_to_s_rank: Some(37),
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

    fn good_promotion() -> PromotionEvidence {
        PromotionEvidence {
            score: ScoreVector {
                utility: 0.96,
                safety: 0.99,
                eligible_activation: 0.90,
                generality: 0.96,
                uniqueness: 0.95,
                transfer: 0.96,
                robustness: 0.97,
                cost: 0.08,
                stability: 0.98,
                scope_clarity: 0.96,
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
            human_alias: "promoted",
            content_digest: "digest_promoted",
            token_hash: "tok_promoted",
            capability_signature: "commit_guard",
            quality_delta: 0.055,
        }
    }

    #[test]
    fn valid_load_survives_alias_rename() {
        let mut store = PocketLibraryStore::new();
        store.insert_pocket(
            entry("pkt_a", PocketLifecycle::Stable),
            token("pkt_a"),
            artifact("pkt_a"),
        );
        assert!(store.rename_alias("pkt_a", "renamed"));
        assert!(store.guarded_load(token("pkt_a")).allowed);
    }

    #[test]
    fn direct_artifact_tamper_blocks_load() {
        let mut store = PocketLibraryStore::new();
        store.insert_pocket(
            entry("pkt_a", PocketLifecycle::Stable),
            token("pkt_a"),
            artifact("pkt_a"),
        );
        store.artifacts[0].content_digest = "tampered";
        assert_eq!(
            store.guarded_load(token("pkt_a")).reason,
            StoreGuardReason::DirectArtifactTamper
        );
    }

    #[test]
    fn concurrent_stale_write_is_blocked() {
        let mut store = PocketLibraryStore::new();
        let before = store.generation;
        store.insert_pocket(
            entry("pkt_a", PocketLifecycle::Stable),
            token("pkt_a"),
            artifact("pkt_a"),
        );
        assert_eq!(
            store.concurrent_write_guard(before).reason,
            StoreGuardReason::ConcurrentStaleWrite
        );
    }

    #[test]
    fn safe_promotion_persists_artifact_and_ledgers() {
        let mut store = PocketLibraryStore::new();
        let decision =
            store.promote_candidate(candidate("gold_a"), good_lifecycle(), good_promotion());
        assert!(decision.allowed);
        assert!(store.guarded_load(store.tokens[0]).allowed);
        let snapshot = store.snapshot();
        assert_eq!(snapshot.registry_entry_count, 1);
        assert!(snapshot.ledger_complete);
        assert!(store.reload_matches(snapshot));
        assert!(store.quality_delta() > 0.0);
    }

    #[test]
    fn repeated_uid_insert_updates_canonical_record_without_duplicate_growth() {
        let mut store = PocketLibraryStore::new();
        store.insert_pocket(
            entry("pkt_a", PocketLifecycle::Stable),
            token("pkt_a"),
            artifact("pkt_a"),
        );
        let mut updated = entry("pkt_a", PocketLifecycle::Core);
        updated.human_alias = "updated_alias";
        store.insert_pocket(updated, token("pkt_a"), artifact("pkt_a"));

        let snapshot = store.snapshot();
        assert_eq!(snapshot.registry_entry_count, 1);
        assert_eq!(snapshot.token_count, 1);
        assert_eq!(snapshot.artifact_count, 1);
        assert_eq!(snapshot.generation, 2);
        assert_eq!(store.registry[0].human_alias, "updated_alias");
        assert_eq!(store.registry[0].lifecycle, PocketLifecycle::Core);
    }

    #[test]
    fn unsafe_promotion_is_blocked() {
        let mut lifecycle = good_lifecycle();
        lifecycle.unique_value_score = 0.0;
        let mut store = PocketLibraryStore::new();
        let decision = store.promote_candidate(candidate("bad"), lifecycle, good_promotion());
        assert!(!decision.allowed);
        assert_eq!(decision.reason, StoreGuardReason::UnsafePromotion);
        assert_eq!(store.registry.len(), 0);
    }
}
