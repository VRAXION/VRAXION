//! Shape-independent Prismion rules over anonymous observation matrices.
//!
//! A Prismion rule reads [`ObservationMatrix`] cells through predicates. If all
//! predicates match, it emits a [`PrismionProposal`] with Proposal Field patches.
//! Reads and writes are shape-independent while stored, then validated at the
//! evaluation/admission boundary.

use crate::ids::{PrismionId, ProposalId};

use super::confidence::ConfidencePpm;
use super::error::FabricError;
use super::matrix::{MatrixAddress, MatrixShape, ObservationMatrix};
use super::proposal::{PrismionProposal, ProposalPatch};
use super::{MAX_PRISMION_CONDITIONS, MAX_PRISMION_PATCHES};

/// Byte comparison used by one Prismion predicate.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CellComparison {
    /// `left == right`.
    Eq,
    /// `left != right`.
    Ne,
    /// `left < right`.
    Lt,
    /// `left <= right`.
    Lte,
    /// `left > right`.
    Gt,
    /// `left >= right`.
    Gte,
}

impl CellComparison {
    #[must_use]
    const fn evaluate(self, left: u8, right: u8) -> bool {
        match self {
            Self::Eq => left == right,
            Self::Ne => left != right,
            Self::Lt => left < right,
            Self::Lte => left <= right,
            Self::Gt => left > right,
            Self::Gte => left >= right,
        }
    }
}

/// Shape-independent matrix read: address + comparison + byte literal.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CellPredicate {
    address: MatrixAddress,
    comparison: CellComparison,
    value: u8,
}

impl CellPredicate {
    /// Creates a predicate without binding it to a concrete matrix shape.
    #[must_use]
    pub const fn new(address: MatrixAddress, comparison: CellComparison, value: u8) -> Self {
        Self {
            address,
            comparison,
            value,
        }
    }

    /// Returns the matrix read address.
    #[must_use]
    pub const fn address(self) -> MatrixAddress {
        self.address
    }

    /// Returns the comparison operator.
    #[must_use]
    pub const fn comparison(self) -> CellComparison {
        self.comparison
    }

    /// Returns the comparison literal.
    #[must_use]
    pub const fn value(self) -> u8 {
        self.value
    }

    fn matches(self, matrix: &ObservationMatrix) -> Result<bool, FabricError> {
        Ok(self
            .comparison
            .evaluate(matrix.get(self.address)?, self.value))
    }
}

/// Canonical anonymous matrix-reader rule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrismionRule {
    source: PrismionId,
    conditions: Vec<CellPredicate>,
    patches: Vec<ProposalPatch>,
    confidence: ConfidencePpm,
}

impl PrismionRule {
    /// Creates a bounded rule and canonicalizes predicate/patch order.
    ///
    /// This constructor validates rule shape only: non-empty lists, caps,
    /// duplicate predicates, and duplicate patch targets. Concrete matrix bounds
    /// are checked later by [`Self::evaluate`]. Different predicates may read
    /// the same address when their comparison/value pair differs.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError`] when the rule is empty, too large, or contains
    /// duplicate predicates or duplicate proposal targets.
    pub fn new(
        source: PrismionId,
        mut conditions: Vec<CellPredicate>,
        mut patches: Vec<ProposalPatch>,
        confidence: ConfidencePpm,
    ) -> Result<Self, FabricError> {
        if conditions.is_empty() {
            return Err(FabricError::EmptyConditionSet);
        }

        if patches.is_empty() {
            return Err(FabricError::EmptyPatchSet);
        }

        if conditions.len() > MAX_PRISMION_CONDITIONS {
            return Err(FabricError::TooManyConditions);
        }

        if patches.len() > MAX_PRISMION_PATCHES {
            return Err(FabricError::TooManyPatches);
        }

        conditions.sort_unstable();
        patches.sort_unstable();

        if conditions.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(FabricError::DuplicatePredicate);
        }

        if patches
            .windows(2)
            .any(|pair| pair[0].target() == pair[1].target())
        {
            return Err(FabricError::DuplicatePatchTarget);
        }

        Ok(Self {
            source,
            conditions,
            patches,
            confidence,
        })
    }

    /// Returns the Prismion source ID.
    #[must_use]
    pub fn source(&self) -> &PrismionId {
        &self.source
    }

    /// Returns canonical predicates in deterministic order.
    #[must_use]
    pub fn conditions(&self) -> &[CellPredicate] {
        &self.conditions
    }

    /// Returns canonical Proposal Field patches in deterministic order.
    #[must_use]
    pub fn patches(&self) -> &[ProposalPatch] {
        &self.patches
    }

    /// Returns the proposal confidence carried by this rule.
    #[must_use]
    pub const fn confidence(&self) -> ConfidencePpm {
        self.confidence
    }

    /// Evaluates this rule against one observation matrix and proposal shape.
    ///
    /// Evaluation has three stages:
    ///
    /// ```text
    /// 1. validate all predicate addresses against the observation matrix
    /// 2. validate all patch targets against the Proposal Field shape
    /// 3. run predicates; all true emits a PrismionProposal, any false emits none
    /// ```
    ///
    /// Invalid geometry is rejected before activation, so a false earlier
    /// predicate cannot hide a later out-of-bounds read or write.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::CellOutOfBounds`] when any predicate references
    /// a cell outside the observation matrix, or
    /// [`FabricError::ProposalPatchOutOfBounds`] when a proposal patch target
    /// is outside `proposal_shape`.
    pub fn evaluate(
        &self,
        proposal_id: ProposalId,
        matrix: &ObservationMatrix,
        proposal_shape: MatrixShape,
    ) -> Result<Option<PrismionProposal>, FabricError> {
        if self
            .conditions
            .iter()
            .any(|condition| !matrix.shape().contains(condition.address))
        {
            return Err(FabricError::CellOutOfBounds);
        }

        if self
            .patches
            .iter()
            .any(|patch| !proposal_shape.contains(patch.target()))
        {
            return Err(FabricError::ProposalPatchOutOfBounds);
        }

        for condition in &self.conditions {
            if !condition.matches(matrix)? {
                return Ok(None);
            }
        }

        let mut evidence_cells = Vec::with_capacity(self.conditions.len());
        for condition in &self.conditions {
            if evidence_cells.last().copied() != Some(condition.address) {
                evidence_cells.push(condition.address);
            }
        }

        Ok(Some(PrismionProposal::new(
            proposal_id,
            self.source.clone(),
            self.confidence,
            evidence_cells,
            self.patches.clone(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use crate::ids::{PrismionId, ProposalId};

    use super::super::confidence::ConfidencePpm;
    use super::super::error::FabricError;
    use super::super::matrix::{MatrixAddress, MatrixShape, ObservationMatrix};
    use super::super::proposal::ProposalPatch;
    use super::super::{MAX_PRISMION_CONDITIONS, MAX_PRISMION_PATCHES};
    use super::{CellComparison, CellPredicate, PrismionRule};

    const PRISMION_HEX: &str = "0123456789abcdef0123456789abcdef";
    const PROPOSAL_HEX: &str = "fedcba9876543210fedcba9876543210";

    fn address(plane: u16, row: u16, column: u16) -> MatrixAddress {
        MatrixAddress::new(plane, row, column)
    }

    fn prismion_id() -> PrismionId {
        PrismionId::parse(PRISMION_HEX).expect("valid Prismion ID")
    }

    fn proposal_id() -> ProposalId {
        ProposalId::parse(PROPOSAL_HEX).expect("valid proposal ID")
    }

    fn confidence() -> ConfidencePpm {
        ConfidencePpm::new(700_000).expect("valid confidence")
    }

    fn predicate(column: u16, comparison: CellComparison, value: u8) -> CellPredicate {
        CellPredicate::new(address(0, 0, column), comparison, value)
    }

    fn patch(column: u16, value: u8) -> ProposalPatch {
        ProposalPatch::new(address(0, 0, column), value)
    }

    fn matrix(cells: &[u8]) -> ObservationMatrix {
        let column_count = u16::try_from(cells.len()).expect("test matrix width fits u16");
        ObservationMatrix::from_dense(
            MatrixShape::new(1, 1, column_count).expect("valid shape"),
            cells.to_vec(),
        )
        .expect("valid dense matrix")
    }

    #[test]
    fn cell_comparison_truth_table_covers_edges() {
        assert!(CellComparison::Eq.evaluate(7, 7));
        assert!(!CellComparison::Eq.evaluate(7, 8));
        assert!(CellComparison::Ne.evaluate(7, 8));
        assert!(!CellComparison::Ne.evaluate(7, 7));
        assert!(CellComparison::Lt.evaluate(0, 1));
        assert!(!CellComparison::Lt.evaluate(1, 1));
        assert!(CellComparison::Lte.evaluate(1, 1));
        assert!(CellComparison::Gt.evaluate(u8::MAX, 0));
        assert!(!CellComparison::Gt.evaluate(0, u8::MAX));
        assert!(CellComparison::Gte.evaluate(u8::MAX, u8::MAX));
    }

    #[test]
    fn rule_constructor_rejects_empty_oversized_and_duplicate_inputs() {
        let condition = predicate(0, CellComparison::Eq, 1);
        let patch = patch(0, 9);

        assert_eq!(
            PrismionRule::new(prismion_id(), vec![], vec![patch], confidence()).unwrap_err(),
            FabricError::EmptyConditionSet
        );
        assert_eq!(
            PrismionRule::new(prismion_id(), vec![condition], vec![], confidence()).unwrap_err(),
            FabricError::EmptyPatchSet
        );
        assert_eq!(
            PrismionRule::new(
                prismion_id(),
                vec![condition; MAX_PRISMION_CONDITIONS + 1],
                vec![patch],
                confidence(),
            )
            .unwrap_err(),
            FabricError::TooManyConditions
        );
        assert_eq!(
            PrismionRule::new(
                prismion_id(),
                vec![condition],
                vec![patch; MAX_PRISMION_PATCHES + 1],
                confidence(),
            )
            .unwrap_err(),
            FabricError::TooManyPatches
        );
        assert_eq!(
            PrismionRule::new(
                prismion_id(),
                vec![condition, condition],
                vec![patch],
                confidence(),
            )
            .unwrap_err(),
            FabricError::DuplicatePredicate
        );
        assert_eq!(
            PrismionRule::new(
                prismion_id(),
                vec![condition],
                vec![patch, ProposalPatch::new(patch.target(), 3)],
                confidence(),
            )
            .unwrap_err(),
            FabricError::DuplicatePatchTarget
        );
    }

    #[test]
    fn rule_constructor_canonicalizes_conditions_and_patches() {
        let left_condition = predicate(0, CellComparison::Eq, 1);
        let right_condition = predicate(1, CellComparison::Eq, 2);
        let left_patch = patch(0, 7);
        let right_patch = patch(1, 8);

        let rule = PrismionRule::new(
            prismion_id(),
            vec![right_condition, left_condition],
            vec![right_patch, left_patch],
            confidence(),
        )
        .expect("valid rule");

        assert_eq!(rule.conditions(), &[left_condition, right_condition]);
        assert_eq!(rule.patches(), &[left_patch, right_patch]);
    }

    #[test]
    fn inactive_rule_with_valid_patch_emits_no_proposal() {
        let rule = PrismionRule::new(
            prismion_id(),
            vec![predicate(0, CellComparison::Eq, 9)],
            vec![patch(0, 7)],
            confidence(),
        )
        .expect("valid rule");

        let result = rule
            .evaluate(
                proposal_id(),
                &matrix(&[1]),
                MatrixShape::new(1, 1, 1).expect("valid proposal shape"),
            )
            .expect("valid inactive evaluation");

        assert!(result.is_none());
    }

    #[test]
    fn invalid_patch_is_rejected_even_when_rule_would_be_inactive() {
        let rule = PrismionRule::new(
            prismion_id(),
            vec![predicate(0, CellComparison::Eq, 9)],
            vec![patch(1, 7)],
            confidence(),
        )
        .expect("valid shape-independent rule");

        assert_eq!(
            rule.evaluate(
                proposal_id(),
                &matrix(&[1]),
                MatrixShape::new(1, 1, 1).expect("valid proposal shape"),
            )
            .unwrap_err(),
            FabricError::ProposalPatchOutOfBounds
        );
    }

    #[test]
    fn out_of_bounds_condition_is_not_hidden_by_false_earlier_condition() {
        let rule = PrismionRule::new(
            prismion_id(),
            vec![
                predicate(0, CellComparison::Eq, 9),
                predicate(1, CellComparison::Eq, 0),
            ],
            vec![patch(0, 7)],
            confidence(),
        )
        .expect("valid shape-independent rule");

        assert_eq!(
            rule.evaluate(
                proposal_id(),
                &matrix(&[1]),
                MatrixShape::new(1, 1, 1).expect("valid proposal shape"),
            )
            .unwrap_err(),
            FabricError::CellOutOfBounds
        );
    }

    #[test]
    fn active_rule_emits_canonical_deduped_evidence_and_patches() {
        let shared = address(0, 0, 0);
        let other = address(0, 0, 1);
        let rule = PrismionRule::new(
            prismion_id(),
            vec![
                CellPredicate::new(shared, CellComparison::Gte, 1),
                CellPredicate::new(shared, CellComparison::Lte, 1),
                CellPredicate::new(other, CellComparison::Eq, 2),
            ],
            vec![patch(0, 7)],
            confidence(),
        )
        .expect("valid rule");

        let proposal = rule
            .evaluate(
                proposal_id(),
                &matrix(&[1, 2]),
                MatrixShape::new(1, 1, 1).expect("valid proposal shape"),
            )
            .expect("valid active evaluation")
            .expect("proposal emitted");

        assert_eq!(proposal.evidence_cells(), &[shared, other]);
        assert_eq!(proposal.patches(), &[patch(0, 7)]);
        assert_eq!(proposal.confidence(), confidence());
        assert_eq!(proposal.source(), rule.source());
    }
}
