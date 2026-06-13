//! Pocket Manager scoring and promotion policy.
//!
//! This module captures the E52 rule: final answers, immediate score,
//! popularity, and scalar averages are not enough for Core or Golden status.
//! Promotion is gate-first and challenger-backed.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionLevel {
    Candidate,
    Active,
    Stable,
    LocalGolden,
    SemiPerma,
    Core,
    TrueGoldenDisc,
    Quarantine,
    Deprecated,
}

impl PromotionLevel {
    pub fn is_core_or_above(self) -> bool {
        matches!(self, PromotionLevel::Core | PromotionLevel::TrueGoldenDisc)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionBlockReason {
    None,
    SafetyGateFailed,
    CreditHijackRisk,
    DelayedPoisonRisk,
    NegativeTransfer,
    UnsafeHighUtility,
    ScopeViolation,
    ReloadOrShadowImportFailed,
    LongHorizonHarm,
    RedundantClone,
    ChallengerRequired,
    VectorScoreTooLow,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreVector {
    pub utility: f32,
    pub safety: f32,
    pub eligible_activation: f32,
    pub generality: f32,
    pub uniqueness: f32,
    pub transfer: f32,
    pub robustness: f32,
    pub cost: f32,
    pub stability: f32,
    pub scope_clarity: f32,
}

impl ScoreVector {
    pub fn weighted_score(self, rare_critical: bool) -> f32 {
        let activation = if rare_critical {
            self.eligible_activation.max(0.85)
        } else {
            self.eligible_activation
        };
        let numerator = self.utility * 1.10
            + self.safety * 1.65
            + activation * 0.65
            + self.generality * 0.85
            + self.uniqueness * 1.15
            + self.transfer * 1.00
            + self.robustness * 1.15
            + self.stability * 1.00
            + self.scope_clarity * 0.85
            - self.cost * 0.80;
        let denominator = 1.10 + 1.65 + 0.65 + 0.85 + 1.15 + 1.00 + 1.15 + 1.00 + 0.85;
        (numerator / denominator).clamp(0.0, 1.0)
    }

    pub fn hard_floor_passes(self) -> bool {
        self.safety >= 0.95
            && self.utility >= 0.50
            && self.robustness >= 0.75
            && self.transfer >= 0.70
            && self.stability >= 0.75
            && self.scope_clarity >= 0.80
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafetyGate {
    pub trace_safe: bool,
    pub no_direct_flow_write: bool,
    pub no_credit_hijack: bool,
    pub no_delayed_poison: bool,
    pub no_negative_transfer: bool,
    pub no_unsafe_high_utility: bool,
    pub no_scope_violation: bool,
}

impl SafetyGate {
    pub fn pass(self) -> Result<(), PromotionBlockReason> {
        if !self.no_unsafe_high_utility {
            Err(PromotionBlockReason::UnsafeHighUtility)
        } else if !self.no_credit_hijack {
            Err(PromotionBlockReason::CreditHijackRisk)
        } else if !self.no_delayed_poison {
            Err(PromotionBlockReason::DelayedPoisonRisk)
        } else if !self.no_negative_transfer {
            Err(PromotionBlockReason::NegativeTransfer)
        } else if !self.no_scope_violation {
            Err(PromotionBlockReason::ScopeViolation)
        } else if !(self.trace_safe && self.no_direct_flow_write) {
            Err(PromotionBlockReason::SafetyGateFailed)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChallengerEvidence {
    pub challenger_sweep_passed: bool,
    pub counterfactual_unique: bool,
    pub reload_shadow_import_passed: bool,
    pub long_horizon_no_harm: bool,
    pub redundant_clone_rejected: bool,
    pub rare_critical_preserved: bool,
}

impl ChallengerEvidence {
    pub fn core_ready(self, rare_critical: bool) -> Result<(), PromotionBlockReason> {
        if !self.reload_shadow_import_passed {
            Err(PromotionBlockReason::ReloadOrShadowImportFailed)
        } else if !self.long_horizon_no_harm {
            Err(PromotionBlockReason::LongHorizonHarm)
        } else if !self.redundant_clone_rejected || !self.counterfactual_unique {
            Err(PromotionBlockReason::RedundantClone)
        } else if rare_critical && !self.rare_critical_preserved {
            Err(PromotionBlockReason::SafetyGateFailed)
        } else if !self.challenger_sweep_passed {
            Err(PromotionBlockReason::ChallengerRequired)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PromotionEvidence {
    pub score: ScoreVector,
    pub safety: SafetyGate,
    pub challenger: ChallengerEvidence,
    pub rare_critical: bool,
    pub global_scope_allowed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PromotionVerdict {
    pub level: PromotionLevel,
    pub weighted_score: f32,
    pub reason: PromotionBlockReason,
}

pub fn evaluate_promotion(evidence: PromotionEvidence) -> PromotionVerdict {
    if let Err(reason) = evidence.safety.pass() {
        return PromotionVerdict {
            level: PromotionLevel::Quarantine,
            weighted_score: 0.0,
            reason,
        };
    }
    if !evidence.score.hard_floor_passes() {
        return PromotionVerdict {
            level: PromotionLevel::Active,
            weighted_score: evidence.score.weighted_score(evidence.rare_critical),
            reason: PromotionBlockReason::VectorScoreTooLow,
        };
    }

    let weighted_score = evidence.score.weighted_score(evidence.rare_critical);
    let core_ready = evidence.challenger.core_ready(evidence.rare_critical);
    match core_ready {
        Ok(()) if evidence.global_scope_allowed && weighted_score >= 0.965 => PromotionVerdict {
            level: PromotionLevel::TrueGoldenDisc,
            weighted_score,
            reason: PromotionBlockReason::None,
        },
        Ok(()) if weighted_score >= 0.925 => PromotionVerdict {
            level: PromotionLevel::Core,
            weighted_score,
            reason: PromotionBlockReason::None,
        },
        Ok(()) if weighted_score >= 0.875 => PromotionVerdict {
            level: PromotionLevel::SemiPerma,
            weighted_score,
            reason: PromotionBlockReason::None,
        },
        Ok(()) if weighted_score >= 0.810 => PromotionVerdict {
            level: PromotionLevel::LocalGolden,
            weighted_score,
            reason: PromotionBlockReason::None,
        },
        Ok(()) if weighted_score >= 0.700 => PromotionVerdict {
            level: PromotionLevel::Stable,
            weighted_score,
            reason: PromotionBlockReason::None,
        },
        Ok(()) if weighted_score >= 0.550 => PromotionVerdict {
            level: PromotionLevel::Active,
            weighted_score,
            reason: PromotionBlockReason::None,
        },
        Ok(()) => PromotionVerdict {
            level: PromotionLevel::Candidate,
            weighted_score,
            reason: PromotionBlockReason::VectorScoreTooLow,
        },
        Err(reason) if weighted_score >= 0.875 => PromotionVerdict {
            level: PromotionLevel::Stable,
            weighted_score,
            reason,
        },
        Err(reason) if weighted_score >= 0.700 => PromotionVerdict {
            level: PromotionLevel::Active,
            weighted_score,
            reason,
        },
        Err(reason) => PromotionVerdict {
            level: PromotionLevel::Candidate,
            weighted_score,
            reason,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strong_score() -> ScoreVector {
        ScoreVector {
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
        }
    }

    fn safe_gate() -> SafetyGate {
        SafetyGate {
            trace_safe: true,
            no_direct_flow_write: true,
            no_credit_hijack: true,
            no_delayed_poison: true,
            no_negative_transfer: true,
            no_unsafe_high_utility: true,
            no_scope_violation: true,
        }
    }

    fn challenger() -> ChallengerEvidence {
        ChallengerEvidence {
            challenger_sweep_passed: true,
            counterfactual_unique: true,
            reload_shadow_import_passed: true,
            long_horizon_no_harm: true,
            redundant_clone_rejected: true,
            rare_critical_preserved: true,
        }
    }

    fn evidence() -> PromotionEvidence {
        PromotionEvidence {
            score: strong_score(),
            safety: safe_gate(),
            challenger: challenger(),
            rare_critical: false,
            global_scope_allowed: false,
        }
    }

    #[test]
    fn strong_challenged_candidate_reaches_core() {
        let verdict = evaluate_promotion(evidence());
        assert_eq!(verdict.level, PromotionLevel::Core);
        assert_eq!(verdict.reason, PromotionBlockReason::None);
    }

    #[test]
    fn true_golden_requires_global_scope() {
        let mut evidence = evidence();
        evidence.global_scope_allowed = true;
        evidence.score = ScoreVector {
            utility: 1.0,
            safety: 1.0,
            eligible_activation: 1.0,
            generality: 1.0,
            uniqueness: 1.0,
            transfer: 1.0,
            robustness: 1.0,
            cost: 0.0,
            stability: 1.0,
            scope_clarity: 1.0,
        };
        let verdict = evaluate_promotion(evidence);
        assert_eq!(verdict.level, PromotionLevel::TrueGoldenDisc);
    }

    #[test]
    fn unsafe_high_utility_is_quarantined_before_score() {
        let mut evidence = evidence();
        evidence.safety.no_unsafe_high_utility = false;
        evidence.score.utility = 1.0;
        let verdict = evaluate_promotion(evidence);
        assert_eq!(verdict.level, PromotionLevel::Quarantine);
        assert_eq!(verdict.reason, PromotionBlockReason::UnsafeHighUtility);
    }

    #[test]
    fn challenger_is_required_for_core() {
        let mut evidence = evidence();
        evidence.challenger.challenger_sweep_passed = false;
        let verdict = evaluate_promotion(evidence);
        assert!(!verdict.level.is_core_or_above());
        assert_eq!(verdict.reason, PromotionBlockReason::ChallengerRequired);
    }

    #[test]
    fn rare_critical_can_survive_low_activation() {
        let mut evidence = evidence();
        evidence.rare_critical = true;
        evidence.score.eligible_activation = 0.05;
        let verdict = evaluate_promotion(evidence);
        assert!(matches!(
            verdict.level,
            PromotionLevel::SemiPerma | PromotionLevel::Core | PromotionLevel::TrueGoldenDisc
        ));
    }

    #[test]
    fn redundant_clone_cannot_core_promote() {
        let mut evidence = evidence();
        evidence.challenger.counterfactual_unique = false;
        let verdict = evaluate_promotion(evidence);
        assert!(!verdict.level.is_core_or_above());
        assert_eq!(verdict.reason, PromotionBlockReason::RedundantClone);
    }
}
