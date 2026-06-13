//! PocketToken, Registry, and runtime governance.
//!
//! Human aliases are documentation only. Runtime calls resolve by immutable
//! pocket uid, content digest, token binding hash, ABI, token freshness, and
//! lifecycle gate.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PocketLifecycle {
    Candidate,
    Active,
    Stable,
    Specialist,
    Core,
    Quarantine,
    Deprecated,
    Banned,
}

impl PocketLifecycle {
    pub fn load_allowed(self) -> bool {
        matches!(
            self,
            PocketLifecycle::Active
                | PocketLifecycle::Stable
                | PocketLifecycle::Specialist
                | PocketLifecycle::Core
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PocketToken {
    pub pocket_uid: &'static str,
    pub token_version: u32,
    pub min_token_version: u32,
    pub token_hash: &'static str,
    pub content_digest: &'static str,
    pub abi_version: &'static str,
    pub capability_signature: &'static str,
    pub utility_score: f32,
    pub safety_score: f32,
    pub reuse_score: f32,
    pub cost_score: f32,
}

impl PocketToken {
    pub fn routing_score(self) -> f32 {
        self.utility_score + self.safety_score + self.reuse_score - self.cost_score
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PocketRegistryEntry {
    pub pocket_uid: &'static str,
    pub human_alias: &'static str,
    pub artifact_path: &'static str,
    pub content_digest: &'static str,
    pub token_hash: &'static str,
    pub abi_version: &'static str,
    pub capability_signature: &'static str,
    pub lifecycle: PocketLifecycle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBlockReason {
    UidMissing,
    ContentDigestMismatch,
    TokenBindingMismatch,
    AbiMismatch,
    CapabilityMismatch,
    StaleToken,
    LifecycleBlocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadDecision {
    pub allowed: bool,
    pub pocket_uid: &'static str,
    pub reason: Option<LoadBlockReason>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivePocket {
    pub pocket_uid: &'static str,
    pub routing_score: f32,
}

pub fn resolve_pocket_call(token: PocketToken, registry: &[PocketRegistryEntry]) -> LoadDecision {
    let Some(entry) = registry
        .iter()
        .find(|entry| entry.pocket_uid == token.pocket_uid)
    else {
        return LoadDecision {
            allowed: false,
            pocket_uid: token.pocket_uid,
            reason: Some(LoadBlockReason::UidMissing),
        };
    };
    let reason = if token.content_digest != entry.content_digest {
        Some(LoadBlockReason::ContentDigestMismatch)
    } else if token.token_hash != entry.token_hash {
        Some(LoadBlockReason::TokenBindingMismatch)
    } else if token.abi_version != entry.abi_version {
        Some(LoadBlockReason::AbiMismatch)
    } else if token.capability_signature != entry.capability_signature {
        Some(LoadBlockReason::CapabilityMismatch)
    } else if token.token_version < token.min_token_version {
        Some(LoadBlockReason::StaleToken)
    } else if !entry.lifecycle.load_allowed() {
        Some(LoadBlockReason::LifecycleBlocked)
    } else {
        None
    };
    LoadDecision {
        allowed: reason.is_none(),
        pocket_uid: token.pocket_uid,
        reason,
    }
}

pub fn active_pocket_set(
    tokens: &[PocketToken],
    registry: &[PocketRegistryEntry],
    limit: usize,
) -> Vec<ActivePocket> {
    let mut active: Vec<ActivePocket> = tokens
        .iter()
        .copied()
        .filter(|token| resolve_pocket_call(*token, registry).allowed)
        .map(|token| ActivePocket {
            pocket_uid: token.pocket_uid,
            routing_score: token.routing_score(),
        })
        .collect();
    active.sort_by(|a, b| {
        b.routing_score
            .partial_cmp(&a.routing_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.pocket_uid.cmp(b.pocket_uid))
    });
    active.truncate(limit);
    active
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(uid: &'static str) -> PocketToken {
        PocketToken {
            pocket_uid: uid,
            token_version: 4,
            min_token_version: 3,
            token_hash: "tok_hash_a",
            content_digest: "digest_a",
            abi_version: "PocketABI-v1",
            capability_signature: "binary_ingress",
            utility_score: 0.94,
            safety_score: 0.99,
            reuse_score: 0.82,
            cost_score: 0.07,
        }
    }

    fn entry(
        uid: &'static str,
        alias: &'static str,
        lifecycle: PocketLifecycle,
    ) -> PocketRegistryEntry {
        PocketRegistryEntry {
            pocket_uid: uid,
            human_alias: alias,
            artifact_path: "pockets/pkt_0101.bin",
            content_digest: "digest_a",
            token_hash: "tok_hash_a",
            abi_version: "PocketABI-v1",
            capability_signature: "binary_ingress",
            lifecycle,
        }
    }

    #[test]
    fn alias_rename_does_not_break_uid_resolution() {
        let token = token("pkt_0101");
        let old = [entry(
            "pkt_0101",
            "binary_frame_codec",
            PocketLifecycle::Stable,
        )];
        let renamed = [entry(
            "pkt_0101",
            "protocol_framing_ingress",
            PocketLifecycle::Stable,
        )];
        assert!(resolve_pocket_call(token, &old).allowed);
        assert!(resolve_pocket_call(token, &renamed).allowed);
    }

    #[test]
    fn digest_mismatch_is_blocked() {
        let token = token("pkt_0101");
        let mut bad = entry("pkt_0101", "binary_frame_codec", PocketLifecycle::Stable);
        bad.content_digest = "digest_b";
        assert_eq!(
            resolve_pocket_call(token, &[bad]).reason,
            Some(LoadBlockReason::ContentDigestMismatch)
        );
    }

    #[test]
    fn token_swap_is_blocked() {
        let token = token("pkt_0101");
        let mut swapped = entry("pkt_0101", "binary_frame_codec", PocketLifecycle::Stable);
        swapped.token_hash = "tok_hash_b";
        assert_eq!(
            resolve_pocket_call(token, &[swapped]).reason,
            Some(LoadBlockReason::TokenBindingMismatch)
        );
    }

    #[test]
    fn unsafe_lifecycle_is_blocked() {
        for lifecycle in [
            PocketLifecycle::Candidate,
            PocketLifecycle::Quarantine,
            PocketLifecycle::Deprecated,
            PocketLifecycle::Banned,
        ] {
            let decision =
                resolve_pocket_call(token("pkt_0101"), &[entry("pkt_0101", "alias", lifecycle)]);
            assert_eq!(decision.reason, Some(LoadBlockReason::LifecycleBlocked));
        }
    }

    #[test]
    fn stale_token_requires_reaudit() {
        let stale = PocketToken {
            token_version: 2,
            min_token_version: 3,
            ..token("pkt_0101")
        };
        assert_eq!(
            resolve_pocket_call(
                stale,
                &[entry("pkt_0101", "alias", PocketLifecycle::Stable)]
            )
            .reason,
            Some(LoadBlockReason::StaleToken)
        );
    }

    #[test]
    fn active_set_filters_and_sorts_loadable_pockets() {
        let mut t2 = token("pkt_0102");
        t2.token_hash = "tok_hash_b";
        t2.content_digest = "digest_b";
        t2.utility_score = 0.80;
        let mut t3 = token("pkt_0103");
        t3.token_hash = "tok_hash_c";
        t3.content_digest = "digest_c";
        t3.utility_score = 0.99;
        t3.cost_score = 0.50;
        let registry = [
            entry("pkt_0101", "a", PocketLifecycle::Stable),
            PocketRegistryEntry {
                pocket_uid: "pkt_0102",
                human_alias: "b",
                artifact_path: "pockets/b.bin",
                content_digest: "digest_b",
                token_hash: "tok_hash_b",
                abi_version: "PocketABI-v1",
                capability_signature: "binary_ingress",
                lifecycle: PocketLifecycle::Quarantine,
            },
            PocketRegistryEntry {
                pocket_uid: "pkt_0103",
                human_alias: "c",
                artifact_path: "pockets/c.bin",
                content_digest: "digest_c",
                token_hash: "tok_hash_c",
                abi_version: "PocketABI-v1",
                capability_signature: "binary_ingress",
                lifecycle: PocketLifecycle::Core,
            },
        ];
        let active = active_pocket_set(&[token("pkt_0101"), t2, t3], &registry, 4);
        assert_eq!(active.len(), 2);
        assert_eq!(active[0].pocket_uid, "pkt_0101");
        assert_eq!(active[1].pocket_uid, "pkt_0103");
    }
}
