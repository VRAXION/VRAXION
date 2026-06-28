//! Eval action proposal primitives.
//!
//! This module is intentionally not an Eval Coordinator runtime. It only
//! defines the safe shape of an eval-related action proposal:
//!
//! ```text
//! coordinator signal
//!   -> EvalProposal
//!   -> Agency/runtime validation
//!   -> actual eval/mutation/promotion outside core
//! ```
//!
//! The core rule is that action proposals must be compatible with the
//! completion gate state and to the runtime evidence snapshot that Agency
//! checked. This prevents the coordinator from using stale "done" state,
//! reusing admission context across proposals, or launching evals for
//! already-complete tasks.

use core::fmt;

use crate::fabric::ConfidencePpm;
use crate::ids::{ArtifactDigest, OperatorId, PrismionId, ProposalId};
use crate::progress::CompletionDecision;

/// E157 default Agency admission confidence floor in parts per million.
pub const E157_ADMISSION_CONFIDENCE_FLOOR_PPM: u32 = 340_000;

/// E151 default deterministic execution-budget ceiling for admitted eval work.
///
/// Raw [`EvalProposal`] values may describe a larger request, but the default
/// public-candidate admission policy refuses to admit requests above this
/// bound. This keeps absurd values such as `u64::MAX` from looking like a
/// sensible workload.
pub const E151_MAX_EXECUTION_BUDGET_UNITS: u64 = 1_000_000_000;

/// Stable eval-proposal error codes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EvalError {
    /// A non-defer proposal had no deterministic budget.
    ZeroBudget,
    /// An action proposal was based on stale completion-gate state.
    StaleCompletionGate,
    /// `RUN_EVAL` was requested when the completion gate was not incomplete.
    RunEvalWithoutIncompleteGate,
    /// `PROMOTE` was requested before the completion gate was complete.
    PromoteWithoutCompleteGate,
    /// An admission policy was configured with no executable budget.
    InvalidBudgetLimit,
}

impl EvalError {
    /// Returns the stable machine-readable error code.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ZeroBudget => "zero_budget",
            Self::StaleCompletionGate => "stale_completion_gate",
            Self::RunEvalWithoutIncompleteGate => "run_eval_without_incomplete_gate",
            Self::PromoteWithoutCompleteGate => "promote_without_complete_gate",
            Self::InvalidBudgetLimit => "invalid_budget_limit",
        }
    }
}

impl fmt::Display for EvalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::error::Error for EvalError {}

/// Bounded eval/coordinator action kind.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EvalActionKind {
    /// Run an evaluation against an incomplete requirement.
    RunEval,
    /// Propose a mutation search.
    Mutate,
    /// Propose a pruning/minimization pass.
    Prune,
    /// Propose promotion after evidence completion.
    Promote,
    /// Propose quarantine.
    Quarantine,
    /// Take no execution action.
    Defer,
}

/// Target of an eval action proposal.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EvalTarget {
    /// A Prismion rule.
    Prismion(PrismionId),
    /// A canonical operator definition.
    Operator(OperatorId),
}

/// Snapshot proof binding used by eval admission.
///
/// The binding couples the completion-gate revision with the evidence artifact
/// digest Agency checked. Proposals and admission contexts must carry the same
/// binding before runtime execution can be materialized.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EvalSnapshotBinding {
    gate_revision: u64,
    evidence_digest: ArtifactDigest,
}

impl EvalSnapshotBinding {
    /// Creates a snapshot binding from a completion-gate revision and evidence digest.
    #[must_use]
    pub const fn new(gate_revision: u64, evidence_digest: ArtifactDigest) -> Self {
        Self {
            gate_revision,
            evidence_digest,
        }
    }

    /// Returns the completion-gate revision bound to this snapshot.
    #[must_use]
    pub const fn gate_revision(&self) -> u64 {
        self.gate_revision
    }

    /// Returns the evidence artifact digest bound to this snapshot.
    #[must_use]
    pub const fn evidence_digest(&self) -> &ArtifactDigest {
        &self.evidence_digest
    }
}

/// Coordinator-originated eval action proposal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvalProposal {
    proposal_id: ProposalId,
    target: EvalTarget,
    action: EvalActionKind,
    completion_decision: CompletionDecision,
    snapshot: EvalSnapshotBinding,
    confidence: ConfidencePpm,
    budget_units: u64,
}

/// Stable reason an eval proposal was rejected by public-candidate admission.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EvalAdmissionRejectReason {
    /// Execution action had no deterministic budget.
    ZeroBudget,
    /// Execution action exceeded the admission policy budget ceiling.
    BudgetExceeded,
    /// Execution action carried stale completion-gate state.
    StaleCompletionGate,
    /// Proposal confidence was below the policy floor.
    LowConfidence,
    /// Action does not match the completion-gate state.
    WrongGateForAction,
}

impl EvalAdmissionRejectReason {
    /// Returns the stable machine-readable reason code.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ZeroBudget => "zero_budget",
            Self::BudgetExceeded => "budget_exceeded",
            Self::StaleCompletionGate => "stale_completion_gate",
            Self::LowConfidence => "low_confidence",
            Self::WrongGateForAction => "wrong_gate_for_action",
        }
    }
}

/// Agency admission decision for an eval proposal.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EvalAdmissionDecision {
    /// Proposal is admitted for runtime execution by a later executor.
    Admit,
    /// Proposal intentionally performs no runtime execution.
    Defer,
    /// Proposal is rejected before execution.
    Reject {
        /// Stable reject reason.
        reason: EvalAdmissionRejectReason,
    },
}

/// Minimal non-authoritative admission policy for eval proposals.
///
/// This type returns a pure recommendation about whether a proposal is
/// well-formed enough for a later runtime to consider. It does not issue an
/// execution ticket and must not be treated as permission to run eval, mutation,
/// prune, promotion, or quarantine work.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EvalAdmissionPolicy {
    confidence_floor: ConfidencePpm,
    max_budget_units: u64,
}

impl EvalAdmissionPolicy {
    /// Creates an admission policy with the public-candidate budget ceiling.
    #[must_use]
    pub const fn new(confidence_floor: ConfidencePpm) -> Self {
        Self {
            confidence_floor,
            max_budget_units: E151_MAX_EXECUTION_BUDGET_UNITS,
        }
    }

    /// Creates an admission policy with an explicit deterministic budget ceiling.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError::InvalidBudgetLimit`] when the ceiling is zero. A
    /// zero ceiling would reject every execution action while still looking like
    /// a configured execution policy.
    pub const fn with_max_budget_units(
        confidence_floor: ConfidencePpm,
        max_budget_units: u64,
    ) -> Result<Self, EvalError> {
        if max_budget_units == 0 {
            return Err(EvalError::InvalidBudgetLimit);
        }

        Ok(Self {
            confidence_floor,
            max_budget_units,
        })
    }

    /// Returns the confidence floor.
    #[must_use]
    pub const fn confidence_floor(self) -> ConfidencePpm {
        self.confidence_floor
    }

    /// Returns the maximum deterministic execution budget admitted by policy.
    #[must_use]
    pub const fn max_budget_units(self) -> u64 {
        self.max_budget_units
    }

    /// Decides whether an eval proposal is compatible with this pure policy.
    ///
    /// This is not an executor and not an authority token. It only checks the
    /// proposal's completion-gate state, confidence, and deterministic budget.
    #[must_use]
    pub fn admit(self, proposal: &EvalProposal) -> EvalAdmissionDecision {
        if proposal.action == EvalActionKind::Defer {
            return EvalAdmissionDecision::Defer;
        }
        if proposal.budget_units == 0 {
            return EvalAdmissionDecision::Reject {
                reason: EvalAdmissionRejectReason::ZeroBudget,
            };
        }
        if proposal.budget_units > self.max_budget_units {
            return EvalAdmissionDecision::Reject {
                reason: EvalAdmissionRejectReason::BudgetExceeded,
            };
        }
        if matches!(
            proposal.completion_decision,
            CompletionDecision::StaleRevision { .. }
        ) {
            return EvalAdmissionDecision::Reject {
                reason: EvalAdmissionRejectReason::StaleCompletionGate,
            };
        }
        if proposal.confidence < self.confidence_floor {
            return EvalAdmissionDecision::Reject {
                reason: EvalAdmissionRejectReason::LowConfidence,
            };
        }

        match proposal.action {
            EvalActionKind::RunEval | EvalActionKind::Mutate => {
                if matches!(
                    proposal.completion_decision,
                    CompletionDecision::Incomplete { .. }
                ) {
                    EvalAdmissionDecision::Admit
                } else {
                    EvalAdmissionDecision::Reject {
                        reason: EvalAdmissionRejectReason::WrongGateForAction,
                    }
                }
            }
            EvalActionKind::Prune | EvalActionKind::Promote => {
                if matches!(proposal.completion_decision, CompletionDecision::Complete) {
                    EvalAdmissionDecision::Admit
                } else {
                    EvalAdmissionDecision::Reject {
                        reason: EvalAdmissionRejectReason::WrongGateForAction,
                    }
                }
            }
            EvalActionKind::Quarantine => EvalAdmissionDecision::Admit,
            EvalActionKind::Defer => EvalAdmissionDecision::Defer,
        }
    }
}

impl EvalProposal {
    /// Creates an eval action proposal.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError`] when an execution action has zero budget or when
    /// the request violates a constructor-level invariant such as stale gate
    /// state, `RUN_EVAL` without missing evidence, or `PROMOTE` before
    /// completion.
    ///
    /// Broader policy compatibility remains an Agency-admission responsibility:
    /// for example, shape-valid `MUTATE` and `PRUNE` proposals may still be
    /// rejected later when their action does not match the current gate state.
    pub fn new(
        proposal_id: ProposalId,
        target: EvalTarget,
        action: EvalActionKind,
        completion_decision: CompletionDecision,
        snapshot: EvalSnapshotBinding,
        confidence: ConfidencePpm,
        budget_units: u64,
    ) -> Result<Self, EvalError> {
        if action != EvalActionKind::Defer && budget_units == 0 {
            return Err(EvalError::ZeroBudget);
        }
        if action != EvalActionKind::Defer
            && matches!(
                completion_decision,
                CompletionDecision::StaleRevision { .. }
            )
        {
            return Err(EvalError::StaleCompletionGate);
        }
        if action == EvalActionKind::RunEval
            && !matches!(completion_decision, CompletionDecision::Incomplete { .. })
        {
            return Err(EvalError::RunEvalWithoutIncompleteGate);
        }
        if action == EvalActionKind::Promote
            && !matches!(completion_decision, CompletionDecision::Complete)
        {
            return Err(EvalError::PromoteWithoutCompleteGate);
        }

        Ok(Self {
            proposal_id,
            target,
            action,
            completion_decision,
            snapshot,
            confidence,
            budget_units,
        })
    }

    /// Returns the proposal ID.
    #[must_use]
    pub fn proposal_id(&self) -> &ProposalId {
        &self.proposal_id
    }

    /// Returns the eval target.
    #[must_use]
    pub const fn target(&self) -> &EvalTarget {
        &self.target
    }

    /// Returns the action kind.
    #[must_use]
    pub const fn action(&self) -> EvalActionKind {
        self.action
    }

    /// Returns the completion-gate decision used to construct the proposal.
    #[must_use]
    pub const fn completion_decision(&self) -> CompletionDecision {
        self.completion_decision
    }

    /// Returns the evidence/gate snapshot bound to this proposal.
    #[must_use]
    pub const fn snapshot(&self) -> &EvalSnapshotBinding {
        &self.snapshot
    }

    /// Returns the completion-gate revision checked by Agency.
    #[must_use]
    pub const fn checked_revision(&self) -> u64 {
        self.snapshot.gate_revision()
    }

    /// Returns the evidence snapshot digest checked by Agency.
    #[must_use]
    pub const fn evidence_digest(&self) -> &ArtifactDigest {
        self.snapshot.evidence_digest()
    }

    /// Returns proposal confidence.
    #[must_use]
    pub const fn confidence(&self) -> ConfidencePpm {
        self.confidence
    }

    /// Returns deterministic budget units.
    #[must_use]
    pub const fn budget_units(&self) -> u64 {
        self.budget_units
    }
}

#[cfg(test)]
mod tests {
    use crate::fabric::ConfidencePpm;
    use crate::ids::{ArtifactDigest, OperatorId, ProposalId};
    use crate::progress::CompletionDecision;

    use super::{
        E151_MAX_EXECUTION_BUDGET_UNITS, E157_ADMISSION_CONFIDENCE_FLOOR_PPM, EvalActionKind,
        EvalAdmissionDecision, EvalAdmissionPolicy, EvalAdmissionRejectReason, EvalError,
        EvalProposal, EvalSnapshotBinding, EvalTarget,
    };

    const PROPOSAL_HEX: &str = "0123456789abcdef0123456789abcdef";
    const OPERATOR_HEX: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const EVIDENCE_HEX: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const CHECKED_REVISION: u64 = 7;

    fn proposal_id() -> ProposalId {
        ProposalId::parse(PROPOSAL_HEX).expect("valid proposal ID")
    }

    fn operator_target() -> EvalTarget {
        EvalTarget::Operator(OperatorId::parse(OPERATOR_HEX).expect("valid operator ID"))
    }

    fn evidence_digest() -> ArtifactDigest {
        ArtifactDigest::parse(EVIDENCE_HEX).expect("valid evidence digest")
    }

    fn snapshot() -> EvalSnapshotBinding {
        EvalSnapshotBinding::new(CHECKED_REVISION, evidence_digest())
    }

    fn confidence() -> ConfidencePpm {
        ConfidencePpm::new(900_000).expect("valid confidence")
    }

    fn floor_confidence() -> ConfidencePpm {
        ConfidencePpm::new(E157_ADMISSION_CONFIDENCE_FLOOR_PPM).expect("valid E157 floor")
    }

    fn policy() -> EvalAdmissionPolicy {
        EvalAdmissionPolicy::new(floor_confidence())
    }

    fn default_run_eval() -> EvalProposal {
        EvalProposal::new(
            proposal_id(),
            operator_target(),
            EvalActionKind::RunEval,
            CompletionDecision::Incomplete { missing_count: 1 },
            snapshot(),
            confidence(),
            10,
        )
        .expect("shape-valid run eval")
    }

    #[test]
    fn eval_error_codes_are_unique() {
        let codes = [
            EvalError::ZeroBudget.as_str(),
            EvalError::StaleCompletionGate.as_str(),
            EvalError::RunEvalWithoutIncompleteGate.as_str(),
            EvalError::PromoteWithoutCompleteGate.as_str(),
            EvalError::InvalidBudgetLimit.as_str(),
        ];

        for (index, left) in codes.iter().enumerate() {
            for right in &codes[index + 1..] {
                assert_ne!(left, right);
            }
        }
    }

    #[test]
    fn eval_admission_reject_reason_codes_are_unique() {
        let codes = [
            EvalAdmissionRejectReason::ZeroBudget.as_str(),
            EvalAdmissionRejectReason::BudgetExceeded.as_str(),
            EvalAdmissionRejectReason::StaleCompletionGate.as_str(),
            EvalAdmissionRejectReason::LowConfidence.as_str(),
            EvalAdmissionRejectReason::WrongGateForAction.as_str(),
        ];

        for (index, left) in codes.iter().enumerate() {
            for right in &codes[index + 1..] {
                assert_ne!(left, right);
            }
        }
    }

    #[test]
    fn run_eval_requires_incomplete_completion_gate() {
        let proposal = EvalProposal::new(
            proposal_id(),
            operator_target(),
            EvalActionKind::RunEval,
            CompletionDecision::Incomplete { missing_count: 2 },
            snapshot(),
            confidence(),
            10,
        )
        .expect("run eval is valid for incomplete gate");

        assert_eq!(proposal.action(), EvalActionKind::RunEval);
        assert_eq!(proposal.checked_revision(), CHECKED_REVISION);
        assert_eq!(proposal.evidence_digest(), &evidence_digest());
        assert_eq!(proposal.budget_units(), 10);
    }

    #[test]
    fn run_eval_rejects_complete_gate() {
        assert_eq!(
            EvalProposal::new(
                proposal_id(),
                operator_target(),
                EvalActionKind::RunEval,
                CompletionDecision::Complete,
                snapshot(),
                confidence(),
                10,
            )
            .expect_err("run eval rejected for complete gate"),
            EvalError::RunEvalWithoutIncompleteGate
        );
    }

    #[test]
    fn promote_requires_complete_gate() {
        assert_eq!(
            EvalProposal::new(
                proposal_id(),
                operator_target(),
                EvalActionKind::Promote,
                CompletionDecision::Incomplete { missing_count: 1 },
                snapshot(),
                confidence(),
                10,
            )
            .expect_err("premature promote rejected"),
            EvalError::PromoteWithoutCompleteGate
        );
    }

    #[test]
    fn stale_completion_gate_rejects_execution_action() {
        assert_eq!(
            EvalProposal::new(
                proposal_id(),
                operator_target(),
                EvalActionKind::Mutate,
                CompletionDecision::StaleRevision {
                    current: 2,
                    checked: 1
                },
                EvalSnapshotBinding::new(1, evidence_digest()),
                confidence(),
                10,
            )
            .expect_err("stale gate rejected"),
            EvalError::StaleCompletionGate
        );
    }

    #[test]
    fn execution_action_requires_budget() {
        assert_eq!(
            EvalProposal::new(
                proposal_id(),
                operator_target(),
                EvalActionKind::Prune,
                CompletionDecision::Incomplete { missing_count: 1 },
                snapshot(),
                confidence(),
                0,
            )
            .expect_err("zero budget rejected"),
            EvalError::ZeroBudget
        );
    }

    #[test]
    fn defer_may_carry_zero_budget_and_stale_gate() {
        let proposal = EvalProposal::new(
            proposal_id(),
            operator_target(),
            EvalActionKind::Defer,
            CompletionDecision::StaleRevision {
                current: 3,
                checked: 2,
            },
            EvalSnapshotBinding::new(2, evidence_digest()),
            confidence(),
            0,
        )
        .expect("defer is safe with stale gate");

        assert_eq!(proposal.action(), EvalActionKind::Defer);
    }

    #[test]
    fn agency_admission_accepts_valid_run_eval() {
        let proposal = default_run_eval();

        assert_eq!(policy().admit(&proposal), EvalAdmissionDecision::Admit);
        assert_eq!(policy().max_budget_units(), E151_MAX_EXECUTION_BUDGET_UNITS);
    }

    #[test]
    fn agency_admission_rejects_budget_above_policy_ceiling() {
        let proposal = EvalProposal::new(
            proposal_id(),
            operator_target(),
            EvalActionKind::RunEval,
            CompletionDecision::Incomplete { missing_count: 1 },
            snapshot(),
            confidence(),
            11,
        )
        .expect("proposal shape does not know admission budget ceiling");
        let policy =
            EvalAdmissionPolicy::with_max_budget_units(floor_confidence(), 10).expect("valid cap");

        assert_eq!(
            policy.admit(&proposal),
            EvalAdmissionDecision::Reject {
                reason: EvalAdmissionRejectReason::BudgetExceeded
            }
        );
    }

    #[test]
    fn agency_admission_accepts_exact_budget_ceiling() {
        let proposal = EvalProposal::new(
            proposal_id(),
            operator_target(),
            EvalActionKind::RunEval,
            CompletionDecision::Incomplete { missing_count: 1 },
            snapshot(),
            confidence(),
            10,
        )
        .expect("shape-valid run eval");
        let policy =
            EvalAdmissionPolicy::with_max_budget_units(floor_confidence(), 10).expect("valid cap");

        assert_eq!(policy.admit(&proposal), EvalAdmissionDecision::Admit);
    }

    #[test]
    fn admission_policy_rejects_zero_budget_ceiling() {
        assert_eq!(
            EvalAdmissionPolicy::with_max_budget_units(floor_confidence(), 0)
                .expect_err("zero cap rejected"),
            EvalError::InvalidBudgetLimit
        );
    }

    #[test]
    fn agency_admission_rejects_low_confidence() {
        let proposal = EvalProposal::new(
            proposal_id(),
            operator_target(),
            EvalActionKind::RunEval,
            CompletionDecision::Incomplete { missing_count: 1 },
            snapshot(),
            ConfidencePpm::new(E157_ADMISSION_CONFIDENCE_FLOOR_PPM - 1)
                .expect("valid low confidence"),
            10,
        )
        .expect("shape-valid run eval");

        assert_eq!(
            policy().admit(&proposal),
            EvalAdmissionDecision::Reject {
                reason: EvalAdmissionRejectReason::LowConfidence
            }
        );
    }

    #[test]
    fn agency_admission_rejects_shape_valid_mutate_on_complete_gate() {
        let proposal = EvalProposal::new(
            proposal_id(),
            operator_target(),
            EvalActionKind::Mutate,
            CompletionDecision::Complete,
            snapshot(),
            confidence(),
            10,
        )
        .expect("mutate is shape-valid before admission");

        assert_eq!(
            policy().admit(&proposal),
            EvalAdmissionDecision::Reject {
                reason: EvalAdmissionRejectReason::WrongGateForAction
            }
        );
    }

    #[test]
    fn agency_admission_rejects_shape_valid_prune_on_incomplete_gate() {
        let proposal = EvalProposal::new(
            proposal_id(),
            operator_target(),
            EvalActionKind::Prune,
            CompletionDecision::Incomplete { missing_count: 1 },
            snapshot(),
            confidence(),
            10,
        )
        .expect("prune is shape-valid before admission");

        assert_eq!(
            policy().admit(&proposal),
            EvalAdmissionDecision::Reject {
                reason: EvalAdmissionRejectReason::WrongGateForAction
            }
        );
    }

    #[test]
    fn agency_admission_accepts_quarantine_only_with_hazard() {
        let proposal = EvalProposal::new(
            proposal_id(),
            operator_target(),
            EvalActionKind::Quarantine,
            CompletionDecision::Blocked { blocker_count: 1 },
            snapshot(),
            confidence(),
            10,
        )
        .expect("quarantine is shape-valid before admission");

        assert_eq!(policy().admit(&proposal), EvalAdmissionDecision::Admit);
    }

    #[test]
    fn agency_admission_preserves_defer_as_no_action() {
        let proposal = EvalProposal::new(
            proposal_id(),
            operator_target(),
            EvalActionKind::Defer,
            CompletionDecision::StaleRevision {
                current: 3,
                checked: 2,
            },
            EvalSnapshotBinding::new(2, evidence_digest()),
            confidence(),
            0,
        )
        .expect("defer is shape-valid");

        assert_eq!(policy().admit(&proposal), EvalAdmissionDecision::Defer);
    }
}
