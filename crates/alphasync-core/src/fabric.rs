//! Anonymous matrix fabric primitives.
//!
//! This module is the semantic-name-free path of the `AlphaSync` core:
//! Prismions read an [`ObservationMatrix`] by cell address and emit
//! [`PrismionProposal`] values. They do not write to Flow or Ground directly.
//!
//! # Design Contract
//!
//! The fabric intentionally avoids hand-labeled feature IDs. Runtime learning
//! sees anonymous `u8` cells only:
//!
//! ```text
//! ObservationMatrix
//!   -> PrismionRule conditions over cell addresses
//!   -> PrismionProposal patches into the Proposal Field
//!   -> later Agency arbitration outside this module
//! ```
//!
//! The code in this module is therefore both data structure and safety
//! boundary: it keeps observation data bounded, canonical, deterministic, and
//! unable to become a direct Flow/Ground write by accident.
//!
//! # File Layout
//!
//! ```text
//! matrix.rs:
//!   MatrixShape, CellAddress, CellSample, ObservationMatrix
//!
//! prismion.rs:
//!   CellComparison, CellPredicate, PrismionRule
//!
//! proposal.rs:
//!   ProposalPatch, PrismionProposal, ProposalField
//!
//! evolution.rs:
//!   EvolutionField-adjacent mutation proposal primitives
//!
//! confidence.rs:
//!   deterministic parts-per-million confidence values
//!
//! consensus.rs:
//!   passive consensus field contract with Agency ingress/egress gates
//!
//! profile.rs:
//!   named matrix-shape profiles for canary and frontier runs
//!
//! error.rs:
//!   stable machine-readable FabricError codes
//! ```
//!
//! # Changeability Rule
//!
//! Comments marked `Changeability:` record how expensive a future change is:
//!
//! ```text
//! easy:
//!   local cap or policy; still needs regression probes
//! medium:
//!   cross-component or migration-sensitive
//! hard:
//!   ABI / persisted-meaning / benchmark-comparability surface
//! locked-or-replace-only:
//!   add a new versioned path instead of mutating in place
//! ```

pub mod confidence;
pub mod consensus;
pub mod error;
pub mod evolution;
pub mod evolution_router;
pub mod matrix;
pub mod prismion;
pub mod profile;
pub mod proposal;

pub use confidence::ConfidencePpm;
pub use consensus::{
    ConsensusDecision, ConsensusPolicy, ConsensusRecommendation, ConsensusSourceKind,
    ConsensusVote, ConsensusVoteKind,
};
pub use error::FabricError;
pub use evolution::{
    EvolutionLane, EvolutionPatch, EvolutionProposal, MAX_EVOLUTION_PATCHES, MutationKind,
    MutationTarget,
};
pub use evolution_router::{EvolutionLaneStats, EvolutionRouterDecision, EvolutionRouterPolicy};
pub use matrix::{CellAddress, CellSample, MatrixShape, ObservationMatrix};
pub use prismion::{CellComparison, CellPredicate, PrismionRule};
pub use profile::{FabricMatrixLayer, FabricShapeProfile, FabricShapeProfileKind};
pub use proposal::{PrismionProposal, ProposalField, ProposalPatch};

/// Maximum number of cells accepted in one anonymous observation matrix.
///
/// Changeability: easy schema/runtime constant. Raising it is mechanically
/// simple, but must be re-benchmarked because search cost grows with the
/// active cell window, not just with memory.
pub const MAX_MATRIX_CELLS: usize = 4_194_304;

/// Maximum number of predicates in one Prismion rule.
///
/// Changeability: easy runtime cap. Keep the normal search target much lower
/// than this cap; raise only if probes show useful Prismions hitting the limit.
pub const MAX_PRISMION_CONDITIONS: usize = 64;

/// Maximum number of proposal patches emitted by one Prismion rule.
///
/// Changeability: easy runtime cap. Raising it increases proposal fanout and
/// Agency load, so it needs a conflict/false-commit regression check.
pub const MAX_PRISMION_PATCHES: usize = 64;

/// Maximum number of Prismion proposals materialized into one Proposal Field.
///
/// Changeability: easy runtime cap. This is not a search target; it is a
/// denial-of-service boundary so malformed callers cannot force unbounded
/// per-cycle allocation before Agency can arbitrate.
pub const MAX_PROPOSALS_PER_FIELD: usize = 4_096;

/// Maximum aggregate proposal patches materialized into one Proposal Field.
///
/// Changeability: easy runtime cap. The value matches
/// `MAX_PROPOSALS_PER_FIELD * MAX_PRISMION_PATCHES`, keeping the worst-case
/// materialization bounded while preserving the full per-rule fanout envelope.
pub const MAX_PROPOSAL_PATCHES_PER_FIELD: usize = MAX_PROPOSALS_PER_FIELD * MAX_PRISMION_PATCHES;

/// Maximum aggregate evidence cells accepted from one runtime cycle.
///
/// Changeability: easy runtime cap. Evidence can be sparse and duplicated
/// across proposals, so this caps aggregate accounting before report/consensus
/// construction rather than relying on matrix size alone.
pub const MAX_EVIDENCE_CELLS_PER_CYCLE: usize = MAX_PROPOSALS_PER_FIELD * MAX_PRISMION_CONDITIONS;

/// Parts-per-million denominator for confidence values.
///
/// Changeability: hard ABI choice. Changing it would reinterpret stored
/// confidence values, so prefer adding a new versioned confidence type instead.
pub const CONFIDENCE_PPM_DENOMINATOR: u32 = 1_000_000;

#[cfg(test)]
mod tests {
    use crate::ids::{OperatorId, PrismionId, ProposalId};

    use super::{
        CellAddress, CellComparison, CellPredicate, CellSample, ConfidencePpm, ConsensusPolicy,
        ConsensusRecommendation, ConsensusSourceKind, ConsensusVote, ConsensusVoteKind,
        EvolutionLane, EvolutionLaneStats, EvolutionPatch, EvolutionProposal,
        EvolutionRouterDecision, EvolutionRouterPolicy, FabricError, FabricMatrixLayer,
        FabricShapeProfile, FabricShapeProfileKind, MAX_EVOLUTION_PATCHES, MAX_MATRIX_CELLS,
        MAX_PROPOSAL_PATCHES_PER_FIELD, MAX_PROPOSALS_PER_FIELD, MatrixShape, MutationKind,
        MutationTarget, ObservationMatrix, PrismionRule, ProposalField, ProposalPatch,
    };

    const OPAQUE_HEX: &str = "0123456789abcdef0123456789abcdef";
    const OTHER_OPAQUE_HEX: &str = "fedcba9876543210fedcba9876543210";
    const FULL_HEX: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn prismion_id() -> PrismionId {
        PrismionId::parse(OPAQUE_HEX).expect("valid Prismion ID")
    }

    fn proposal_id() -> ProposalId {
        ProposalId::parse(OTHER_OPAQUE_HEX).expect("valid proposal ID")
    }

    fn operator_id() -> OperatorId {
        OperatorId::parse(FULL_HEX).expect("valid operator ID")
    }

    fn address(plane: u16, row: u16, column: u16) -> CellAddress {
        CellAddress::new(plane, row, column)
    }

    #[test]
    fn fabric_error_codes_are_unique() {
        let codes = [
            FabricError::ZeroDimension.as_str(),
            FabricError::MatrixTooLarge.as_str(),
            FabricError::DenseCellCountMismatch.as_str(),
            FabricError::CellOutOfBounds.as_str(),
            FabricError::DuplicateSparseCell.as_str(),
            FabricError::TooManySparseCells.as_str(),
            FabricError::NonCanonicalZeroSparseCell.as_str(),
            FabricError::EmptyConditionSet.as_str(),
            FabricError::EmptyPatchSet.as_str(),
            FabricError::TooManyConditions.as_str(),
            FabricError::TooManyPatches.as_str(),
            FabricError::DuplicatePredicate.as_str(),
            FabricError::DuplicatePatchTarget.as_str(),
            FabricError::ProposalPatchOutOfBounds.as_str(),
            FabricError::InvalidConfidence.as_str(),
            FabricError::EmptyEvolutionPatchSet.as_str(),
            FabricError::TooManyEvolutionPatches.as_str(),
            FabricError::DuplicateEvolutionPatchSlot.as_str(),
            FabricError::EmptyEvolutionRouterStats.as_str(),
            FabricError::DuplicateEvolutionLaneStats.as_str(),
            FabricError::EmptyEvolutionLaneStats.as_str(),
            FabricError::ZeroEvolutionLaneCost.as_str(),
            FabricError::InvalidEvolutionLaneStats.as_str(),
            FabricError::InvalidConsensusThreshold.as_str(),
            FabricError::InvalidConsensusPolicy.as_str(),
            FabricError::TooManyConsensusVotes.as_str(),
            FabricError::DuplicateConsensusSource.as_str(),
        ];

        for (index, left) in codes.iter().enumerate() {
            for right in &codes[index + 1..] {
                assert_ne!(left, right);
            }
        }
    }

    #[test]
    fn cell_address_is_shape_independent_until_checked() {
        let shape = MatrixShape::new(1, 2, 3).expect("valid matrix shape");
        let last_valid = address(0, 1, 2);

        assert_eq!(last_valid.plane(), 0);
        assert_eq!(last_valid.row(), 1);
        assert_eq!(last_valid.column(), 2);
        assert!(shape.contains(last_valid));

        assert!(!shape.contains(address(1, 0, 0)));
        assert!(!shape.contains(address(0, 2, 0)));
        assert!(!shape.contains(address(0, 0, 3)));
    }

    #[test]
    fn cell_sample_preserves_nonzero_sparse_atom() {
        let sample = CellSample::new(address(u16::MAX, 7, 11), u8::MAX)
            .expect("non-zero sparse sample accepted before shape binding");

        assert_eq!(sample.address(), address(u16::MAX, 7, 11));
        assert_eq!(sample.value(), u8::MAX);
    }

    #[test]
    fn matrix_shape_rejects_zero_axes() {
        assert_eq!(
            MatrixShape::new(0, 1, 1).expect_err("zero plane rejected"),
            FabricError::ZeroDimension
        );
        assert_eq!(
            MatrixShape::new(1, 0, 1).expect_err("zero row rejected"),
            FabricError::ZeroDimension
        );
        assert_eq!(
            MatrixShape::new(1, 1, 0).expect_err("zero column rejected"),
            FabricError::ZeroDimension
        );
    }

    #[test]
    fn matrix_shape_accepts_exact_cell_cap() {
        let shape = MatrixShape::new(16, 512, 512).expect("exact cap accepted");

        assert_eq!(shape.cell_count(), MAX_MATRIX_CELLS);
        assert_eq!(shape.planes(), 16);
        assert_eq!(shape.rows(), 512);
        assert_eq!(shape.columns(), 512);
    }

    #[test]
    fn matrix_shape_rejects_cell_cap_overflow() {
        assert_eq!(
            MatrixShape::new(16, 512, 513).expect_err("over cap rejected"),
            FabricError::MatrixTooLarge
        );
    }

    #[test]
    fn matrix_shape_rejects_adversarial_u16_max_axes() {
        assert_eq!(
            MatrixShape::new(1, u16::MAX, u16::MAX).expect_err("huge shape rejected"),
            FabricError::MatrixTooLarge
        );
    }

    #[test]
    fn logic_iq_profile_matches_canary_shapes() {
        let profile = FabricShapeProfile::logic_iq_canary().expect("valid canary profile");

        assert_eq!(profile.kind(), FabricShapeProfileKind::LogicIqCanary);
        assert_eq!(profile.observation(), MatrixShape::new(1, 16, 16).unwrap());
        assert_eq!(profile.proposal(), MatrixShape::new(1, 4, 4).unwrap());
        assert_eq!(profile.flow(), MatrixShape::new(1, 4, 4).unwrap());
        assert_eq!(profile.ground(), MatrixShape::new(1, 16, 16).unwrap());
        assert_eq!(
            profile.layer(FabricMatrixLayer::Observation),
            profile.observation()
        );
        assert_eq!(
            profile.layer(FabricMatrixLayer::Proposal),
            profile.proposal()
        );
        assert_eq!(profile.layer(FabricMatrixLayer::Flow), profile.flow());
        assert_eq!(profile.layer(FabricMatrixLayer::Ground), profile.ground());
    }

    #[test]
    fn frontier_local_profile_locks_high_searchable_shapes() {
        let profile = FabricShapeProfile::frontier_local().expect("valid frontier profile");

        assert_eq!(profile.kind(), FabricShapeProfileKind::FrontierLocal);
        assert_eq!(
            profile.observation(),
            MatrixShape::new(16, 512, 512).unwrap()
        );
        assert_eq!(profile.proposal(), MatrixShape::new(4, 256, 256).unwrap());
        assert_eq!(profile.flow(), MatrixShape::new(4, 512, 512).unwrap());
        assert_eq!(profile.ground(), MatrixShape::new(4, 512, 512).unwrap());
        assert_eq!(profile.observation().cell_count(), MAX_MATRIX_CELLS);
        assert!(profile.flow().cell_count() < profile.observation().cell_count());
        assert!(profile.proposal().cell_count() < profile.flow().cell_count());
    }

    #[test]
    fn shape_profile_kind_resolves_and_has_stable_codes() {
        let canary = FabricShapeProfile::from_kind(FabricShapeProfileKind::LogicIqCanary).unwrap();
        let frontier =
            FabricShapeProfile::from_kind(FabricShapeProfileKind::FrontierLocal).unwrap();

        assert_eq!(
            canary.kind().as_str(),
            FabricShapeProfileKind::LogicIqCanary.as_str()
        );
        assert_eq!(
            FabricShapeProfileKind::LogicIqCanary.as_str(),
            "logic-iq-canary"
        );
        assert_eq!(frontier.kind().as_str(), "frontier-local");
        assert!(frontier.observation().cell_count() > canary.observation().cell_count());
    }

    #[test]
    fn sparse_observation_matrix_is_anonymous_and_canonical() {
        let shape = MatrixShape::new(1, 2, 3).expect("valid matrix shape");
        let matrix = ObservationMatrix::from_sparse(
            shape,
            vec![
                CellSample::new(address(0, 1, 2), 9).expect("non-zero sample"),
                CellSample::new(address(0, 0, 1), 4).expect("non-zero sample"),
            ],
        )
        .expect("valid sparse matrix");

        assert_eq!(matrix.get(address(0, 0, 0)).expect("in bounds"), 0);
        assert_eq!(matrix.get(address(0, 0, 1)).expect("in bounds"), 4);
        assert_eq!(matrix.get(address(0, 1, 2)).expect("in bounds"), 9);
        assert_eq!(matrix.dense_cells(), &[0, 4, 0, 0, 0, 9]);
    }

    #[test]
    fn sparse_matrix_rejects_noncanonical_zero_samples() {
        assert_eq!(
            CellSample::new(address(0, 0, 0), 0).expect_err("zero sample rejected"),
            FabricError::NonCanonicalZeroSparseCell
        );
    }

    #[test]
    fn sparse_matrix_rejects_too_many_samples_before_sorting() {
        let shape = MatrixShape::new(1, 1, 1).expect("valid matrix shape");

        assert_eq!(
            ObservationMatrix::from_sparse(
                shape,
                vec![
                    CellSample::new(address(0, 0, 0), 1).expect("non-zero sample"),
                    CellSample::new(address(0, 0, 1), 2).expect("non-zero sample"),
                ],
            )
            .expect_err("overlength sparse input rejected"),
            FabricError::TooManySparseCells
        );
    }

    #[test]
    fn sparse_matrix_rejects_duplicate_sparse_addresses() {
        let shape = MatrixShape::new(1, 2, 2).expect("valid matrix shape");

        assert_eq!(
            ObservationMatrix::from_sparse(
                shape,
                vec![
                    CellSample::new(address(0, 0, 0), 1).expect("non-zero sample"),
                    CellSample::new(address(0, 0, 0), 2).expect("non-zero sample"),
                ],
            )
            .expect_err("duplicate sparse address rejected"),
            FabricError::DuplicateSparseCell
        );
    }

    #[test]
    fn sparse_matrix_rejects_out_of_bounds_sample() {
        let shape = MatrixShape::new(1, 1, 1).expect("valid matrix shape");

        assert_eq!(
            ObservationMatrix::from_sparse(
                shape,
                vec![CellSample::new(address(0, 0, 1), 1).expect("non-zero sample")],
            )
            .expect_err("out-of-bounds sparse sample rejected"),
            FabricError::CellOutOfBounds
        );
    }

    #[test]
    fn dense_matrix_requires_exact_shape_size() {
        let shape = MatrixShape::new(1, 2, 2).expect("valid matrix shape");

        assert_eq!(
            ObservationMatrix::from_dense(shape, vec![1, 2, 3])
                .expect_err("wrong dense count rejected"),
            FabricError::DenseCellCountMismatch
        );
    }

    #[test]
    fn prismion_emits_proposal_from_anonymous_matrix_pattern() {
        let shape = MatrixShape::new(1, 2, 2).expect("valid matrix shape");
        let matrix = ObservationMatrix::from_sparse(
            shape,
            vec![
                CellSample::new(address(0, 0, 0), 7).expect("non-zero sample"),
                CellSample::new(address(0, 0, 1), 3).expect("non-zero sample"),
            ],
        )
        .expect("valid sparse matrix");

        let rule = PrismionRule::new(
            prismion_id(),
            vec![
                CellPredicate::new(address(0, 0, 0), CellComparison::Gte, 5),
                CellPredicate::new(address(0, 0, 1), CellComparison::Eq, 3),
            ],
            vec![ProposalPatch::new(address(0, 1, 1), 1)],
            ConfidencePpm::new(900_000).expect("valid confidence"),
        )
        .expect("valid Prismion rule");

        let proposal = rule
            .evaluate(proposal_id(), &matrix, shape)
            .expect("evaluation succeeds")
            .expect("proposal emitted");

        assert_eq!(proposal.source(), rule.source());
        assert_eq!(proposal.confidence().as_ppm(), 900_000);
        assert_eq!(
            proposal.evidence_cells(),
            &[address(0, 0, 0), address(0, 0, 1)]
        );
        assert_eq!(
            proposal.patches(),
            &[ProposalPatch::new(address(0, 1, 1), 1)]
        );
    }

    #[test]
    fn prismion_does_not_emit_when_pattern_is_absent() {
        let shape = MatrixShape::new(1, 1, 1).expect("valid matrix shape");
        let matrix = ObservationMatrix::zeroed(shape);
        let rule = PrismionRule::new(
            prismion_id(),
            vec![CellPredicate::new(address(0, 0, 0), CellComparison::Gt, 0)],
            vec![ProposalPatch::new(address(0, 0, 0), 1)],
            ConfidencePpm::new(700_000).expect("valid confidence"),
        )
        .expect("valid Prismion rule");

        assert_eq!(
            rule.evaluate(proposal_id(), &matrix, shape)
                .expect("evaluation succeeds"),
            None
        );
    }

    #[test]
    fn prismion_rejects_out_of_bounds_proposal_patch_on_evaluate() {
        let shape = MatrixShape::new(1, 1, 1).expect("valid matrix shape");
        let matrix = ObservationMatrix::from_sparse(
            shape,
            vec![CellSample::new(address(0, 0, 0), 7).expect("non-zero sample")],
        )
        .expect("valid sparse matrix");
        let rule = PrismionRule::new(
            prismion_id(),
            vec![CellPredicate::new(address(0, 0, 0), CellComparison::Eq, 7)],
            vec![ProposalPatch::new(address(0, 0, 1), 1)],
            ConfidencePpm::new(700_000).expect("valid confidence"),
        )
        .expect("valid Prismion rule before proposal-shape binding");

        assert_eq!(
            rule.evaluate(proposal_id(), &matrix, shape)
                .expect_err("out-of-bounds proposal patch rejected"),
            FabricError::ProposalPatchOutOfBounds
        );
    }

    #[test]
    fn prismion_rejects_out_of_bounds_predicate_on_evaluate() {
        let shape = MatrixShape::new(1, 1, 1).expect("valid matrix shape");
        let matrix = ObservationMatrix::zeroed(shape);
        let rule = PrismionRule::new(
            prismion_id(),
            vec![CellPredicate::new(address(0, 0, 1), CellComparison::Eq, 0)],
            vec![ProposalPatch::new(address(0, 0, 0), 1)],
            ConfidencePpm::new(700_000).expect("valid confidence"),
        )
        .expect("valid Prismion rule before observation-shape binding");

        assert_eq!(
            rule.evaluate(proposal_id(), &matrix, shape)
                .expect_err("out-of-bounds predicate rejected"),
            FabricError::CellOutOfBounds
        );
    }

    #[test]
    fn prismion_rejects_directly_ambiguous_duplicate_targets() {
        let target = address(0, 0, 0);

        assert_eq!(
            PrismionRule::new(
                prismion_id(),
                vec![CellPredicate::new(target, CellComparison::Eq, 1)],
                vec![ProposalPatch::new(target, 1), ProposalPatch::new(target, 2)],
                ConfidencePpm::new(500_000).expect("valid confidence"),
            )
            .expect_err("duplicate patch target rejected"),
            FabricError::DuplicatePatchTarget
        );
    }

    #[test]
    fn prismion_rejects_duplicate_exact_predicates() {
        let target = address(0, 0, 0);

        assert_eq!(
            PrismionRule::new(
                prismion_id(),
                vec![
                    CellPredicate::new(target, CellComparison::Eq, 1),
                    CellPredicate::new(target, CellComparison::Eq, 1),
                ],
                vec![ProposalPatch::new(target, 1)],
                ConfidencePpm::new(500_000).expect("valid confidence"),
            )
            .expect_err("duplicate exact predicate rejected"),
            FabricError::DuplicatePredicate
        );
    }

    #[test]
    fn prismion_deduplicates_same_cell_evidence() {
        let shape = MatrixShape::new(1, 1, 2).expect("valid matrix shape");
        let matrix = ObservationMatrix::from_sparse(
            shape,
            vec![CellSample::new(address(0, 0, 0), 7).expect("non-zero sample")],
        )
        .expect("valid sparse matrix");
        let rule = PrismionRule::new(
            prismion_id(),
            vec![
                CellPredicate::new(address(0, 0, 0), CellComparison::Eq, 7),
                CellPredicate::new(address(0, 0, 0), CellComparison::Gte, 5),
            ],
            vec![ProposalPatch::new(address(0, 0, 1), 1)],
            ConfidencePpm::new(500_000).expect("valid confidence"),
        )
        .expect("same-address compatible predicates are valid");

        let proposal = rule
            .evaluate(proposal_id(), &matrix, shape)
            .expect("evaluation succeeds")
            .expect("proposal emitted");

        assert_eq!(proposal.evidence_cells(), &[address(0, 0, 0)]);
    }

    #[test]
    fn proposal_field_materializes_sparse_patch_matrix() {
        let shape = MatrixShape::new(1, 2, 2).expect("valid proposal shape");
        let matrix = ObservationMatrix::from_sparse(
            shape,
            vec![CellSample::new(address(0, 0, 0), 7).expect("non-zero sample")],
        )
        .expect("valid matrix");
        let rule = PrismionRule::new(
            prismion_id(),
            vec![CellPredicate::new(address(0, 0, 0), CellComparison::Eq, 7)],
            vec![
                ProposalPatch::new(address(0, 0, 1), 0),
                ProposalPatch::new(address(0, 1, 1), 9),
            ],
            ConfidencePpm::new(700_000).expect("valid confidence"),
        )
        .expect("valid rule");
        let proposal = rule
            .evaluate(proposal_id(), &matrix, shape)
            .expect("evaluation succeeds")
            .expect("proposal emitted");

        let field = ProposalField::from_proposals(shape, &[proposal]).expect("field materializes");

        assert_eq!(field.shape(), shape);
        assert_eq!(field.patch_count(), 2);
        assert_eq!(field.occupied_cell_count(), 2);
        assert_eq!(field.collision_cell_count(), 0);
        assert_eq!(field.conflicting_patch_count(), 0);
        assert_eq!(field.get(address(0, 0, 0)).expect("in bounds"), None);
        assert_eq!(field.get(address(0, 0, 1)).expect("in bounds"), Some(0));
        assert!(field.is_occupied(address(0, 0, 1)).expect("in bounds"));
        assert_eq!(field.get(address(0, 1, 1)).expect("in bounds"), Some(9));
        assert_eq!(field.dense_cells(), &[0, 0, 0, 9]);
    }

    #[test]
    fn proposal_field_merges_agreement_and_records_conflict() {
        let shape = MatrixShape::new(1, 1, 2).expect("valid proposal shape");
        let target = address(0, 0, 0);
        let other = address(0, 0, 1);
        let proposal_a = super::PrismionProposal::new(
            proposal_id(),
            prismion_id(),
            ConfidencePpm::new(900_000).expect("valid confidence"),
            vec![target],
            vec![ProposalPatch::new(target, 3), ProposalPatch::new(other, 7)],
        );
        let proposal_b = super::PrismionProposal::new(
            ProposalId::parse(OPAQUE_HEX).expect("valid proposal ID"),
            PrismionId::parse(OTHER_OPAQUE_HEX).expect("valid Prismion ID"),
            ConfidencePpm::new(800_000).expect("valid confidence"),
            vec![target],
            vec![ProposalPatch::new(target, 3)],
        );
        let proposal_c = super::PrismionProposal::new(
            ProposalId::parse("00000000000000000000000000000009").expect("valid proposal ID"),
            PrismionId::parse("0000000000000000000000000000000a").expect("valid Prismion ID"),
            ConfidencePpm::new(1_000_000).expect("valid confidence"),
            vec![target],
            vec![ProposalPatch::new(target, 4)],
        );

        let field = ProposalField::from_proposals(shape, &[proposal_c, proposal_a, proposal_b])
            .expect("field materializes deterministically");

        assert_eq!(field.patch_count(), 4);
        assert_eq!(field.occupied_cell_count(), 2);
        assert_eq!(field.collision_cell_count(), 1);
        assert_eq!(field.conflicting_patch_count(), 3);
        assert_eq!(field.get(target).expect("in bounds"), Some(3));
        assert_eq!(field.get(other).expect("in bounds"), Some(7));
    }

    #[test]
    fn proposal_field_rejects_out_of_shape_patch() {
        let shape = MatrixShape::new(1, 1, 1).expect("valid proposal shape");
        let proposal = super::PrismionProposal::new(
            proposal_id(),
            prismion_id(),
            ConfidencePpm::new(900_000).expect("valid confidence"),
            vec![address(0, 0, 0)],
            vec![ProposalPatch::new(address(0, 0, 1), 3)],
        );

        assert_eq!(
            ProposalField::from_proposals(shape, &[proposal])
                .expect_err("out-of-shape proposal patch rejected"),
            FabricError::ProposalPatchOutOfBounds
        );
    }

    #[test]
    fn proposal_field_rejects_too_many_proposals_before_materialization() {
        let shape = MatrixShape::new(1, 1, 1).expect("valid proposal shape");
        let proposal = super::PrismionProposal::new(
            proposal_id(),
            prismion_id(),
            ConfidencePpm::new(900_000).expect("valid confidence"),
            vec![address(0, 0, 0)],
            vec![ProposalPatch::new(address(0, 0, 0), 3)],
        );
        let proposals = vec![proposal; MAX_PROPOSALS_PER_FIELD + 1];

        assert_eq!(
            ProposalField::from_proposals(shape, &proposals)
                .expect_err("proposal count cap rejected"),
            FabricError::TooManyProposals
        );
    }

    #[test]
    fn proposal_field_rejects_aggregate_patch_cap_before_sorting() {
        let shape = MatrixShape::new(1, 1, 1).expect("valid proposal shape");
        let proposal = super::PrismionProposal::new(
            proposal_id(),
            prismion_id(),
            ConfidencePpm::new(900_000).expect("valid confidence"),
            vec![address(0, 0, 0)],
            vec![ProposalPatch::new(address(0, 0, 0), 3); MAX_PROPOSAL_PATCHES_PER_FIELD + 1],
        );

        assert_eq!(
            ProposalField::from_proposals(shape, &[proposal])
                .expect_err("aggregate patch cap rejected"),
            FabricError::TooManyProposalPatches
        );
    }

    #[test]
    fn confidence_rejects_above_parts_per_million_cap() {
        assert_eq!(
            ConfidencePpm::new(1_000_001).expect_err("confidence over cap rejected"),
            FabricError::InvalidConfidence
        );
    }

    #[test]
    fn evolution_proposal_canonicalizes_mutation_patches() {
        let proposal = EvolutionProposal::new(
            proposal_id(),
            EvolutionLane::Hybrid,
            MutationTarget::Prismion(prismion_id()),
            MutationKind::NumericThresholdStep,
            ConfidencePpm::new(850_000).expect("valid confidence"),
            vec![EvolutionPatch::new(7, 3), EvolutionPatch::new(2, 9)],
        )
        .expect("valid evolution proposal");

        assert_eq!(proposal.lane(), EvolutionLane::Hybrid);
        assert_eq!(proposal.kind(), MutationKind::NumericThresholdStep);
        assert_eq!(proposal.confidence().as_ppm(), 850_000);
        assert_eq!(
            proposal.patches(),
            &[EvolutionPatch::new(2, 9), EvolutionPatch::new(7, 3)]
        );
    }

    #[test]
    fn evolution_proposal_rejects_duplicate_patch_slots() {
        assert_eq!(
            EvolutionProposal::new(
                proposal_id(),
                EvolutionLane::Predicted,
                MutationTarget::Operator(operator_id()),
                MutationKind::Prune,
                ConfidencePpm::new(750_000).expect("valid confidence"),
                vec![EvolutionPatch::new(1, 8), EvolutionPatch::new(1, 9)],
            )
            .expect_err("duplicate evolution patch slot rejected"),
            FabricError::DuplicateEvolutionPatchSlot
        );
    }

    #[test]
    fn evolution_proposal_rejects_empty_and_too_large_patch_sets() {
        assert_eq!(
            EvolutionProposal::new(
                proposal_id(),
                EvolutionLane::Random,
                MutationTarget::Prismion(prismion_id()),
                MutationKind::WidenTrigger,
                ConfidencePpm::new(500_000).expect("valid confidence"),
                Vec::new(),
            )
            .expect_err("empty evolution proposal rejected"),
            FabricError::EmptyEvolutionPatchSet
        );

        let patches = (0..=MAX_EVOLUTION_PATCHES)
            .map(|slot| EvolutionPatch::new(u16::try_from(slot).expect("slot fits u16"), 1))
            .collect();

        assert_eq!(
            EvolutionProposal::new(
                proposal_id(),
                EvolutionLane::Guided,
                MutationTarget::Prismion(prismion_id()),
                MutationKind::SpecializeTrigger,
                ConfidencePpm::new(500_000).expect("valid confidence"),
                patches,
            )
            .expect_err("too many evolution patches rejected"),
            FabricError::TooManyEvolutionPatches
        );
    }

    #[test]
    fn evolution_router_selects_cheapest_lane_that_preserves_quality() {
        let policy = EvolutionRouterPolicy::new(
            ConfidencePpm::new(1_000_000).expect("valid pass-rate gate"),
            ConfidencePpm::new(950_000).expect("valid score gate"),
        );
        let stats = [
            EvolutionLaneStats::new(
                EvolutionLane::Binary,
                4,
                4,
                ConfidencePpm::new(1_000_000).expect("valid score"),
                ConfidencePpm::new(1_000_000).expect("valid score"),
                100,
            )
            .expect("valid binary lane stats"),
            EvolutionLaneStats::new(
                EvolutionLane::Guided,
                4,
                4,
                ConfidencePpm::new(1_000_000).expect("valid score"),
                ConfidencePpm::new(1_000_000).expect("valid score"),
                600,
            )
            .expect("valid guided lane stats"),
        ];

        let decision = policy.choose(&stats).expect("router can select a lane");

        assert_eq!(decision.lane(), EvolutionLane::Binary);
        assert_eq!(
            decision,
            EvolutionRouterDecision::cheapest_quality_lane(EvolutionLane::Binary)
        );
    }

    #[test]
    fn evolution_router_falls_back_to_best_score_when_no_lane_passes_gate() {
        let policy = EvolutionRouterPolicy::new(
            ConfidencePpm::new(1_000_000).expect("valid pass-rate gate"),
            ConfidencePpm::new(990_000).expect("valid score gate"),
        );
        let stats = [
            EvolutionLaneStats::new(
                EvolutionLane::Random,
                4,
                2,
                ConfidencePpm::new(700_000).expect("valid score"),
                ConfidencePpm::new(600_000).expect("valid score"),
                400,
            )
            .expect("valid random lane stats"),
            EvolutionLaneStats::new(
                EvolutionLane::Hybrid,
                4,
                3,
                ConfidencePpm::new(900_000).expect("valid score"),
                ConfidencePpm::new(800_000).expect("valid score"),
                900,
            )
            .expect("valid hybrid lane stats"),
        ];

        let decision = policy.choose(&stats).expect("router can fall back");

        assert_eq!(decision.lane(), EvolutionLane::Hybrid);
        assert_eq!(
            decision,
            EvolutionRouterDecision::fallback_best_score(EvolutionLane::Hybrid)
        );
    }

    #[test]
    fn evolution_router_rejects_empty_stats() {
        let policy = EvolutionRouterPolicy::new(
            ConfidencePpm::new(1_000_000).expect("valid pass-rate gate"),
            ConfidencePpm::new(950_000).expect("valid score gate"),
        );

        assert_eq!(
            policy.choose(&[]).expect_err("empty router stats rejected"),
            FabricError::EmptyEvolutionRouterStats
        );
    }

    #[test]
    fn evolution_router_rejects_duplicate_lane_stats() {
        let policy = EvolutionRouterPolicy::new(
            ConfidencePpm::new(1_000_000).expect("valid pass-rate gate"),
            ConfidencePpm::new(950_000).expect("valid score gate"),
        );
        let stats = [
            EvolutionLaneStats::new(
                EvolutionLane::Guided,
                4,
                4,
                ConfidencePpm::new(1_000_000).expect("valid score"),
                ConfidencePpm::new(1_000_000).expect("valid score"),
                100,
            )
            .expect("valid guided stats"),
            EvolutionLaneStats::new(
                EvolutionLane::Guided,
                4,
                4,
                ConfidencePpm::new(1_000_000).expect("valid score"),
                ConfidencePpm::new(1_000_000).expect("valid score"),
                200,
            )
            .expect("valid guided stats"),
        ];

        assert_eq!(
            policy
                .choose(&stats)
                .expect_err("duplicate lane stats rejected"),
            FabricError::DuplicateEvolutionLaneStats
        );
    }

    #[test]
    fn evolution_router_tie_break_is_explicit_and_deterministic() {
        let policy = EvolutionRouterPolicy::new(
            ConfidencePpm::new(1_000_000).expect("valid pass-rate gate"),
            ConfidencePpm::new(1_000_000).expect("valid score gate"),
        );
        let stats = [
            EvolutionLaneStats::new(
                EvolutionLane::Hybrid,
                4,
                4,
                ConfidencePpm::new(1_000_000).expect("valid score"),
                ConfidencePpm::new(1_000_000).expect("valid score"),
                100,
            )
            .expect("valid hybrid stats"),
            EvolutionLaneStats::new(
                EvolutionLane::Guided,
                4,
                4,
                ConfidencePpm::new(1_000_000).expect("valid score"),
                ConfidencePpm::new(1_000_000).expect("valid score"),
                100,
            )
            .expect("valid guided stats"),
        ];

        assert_eq!(
            policy
                .choose(&stats)
                .expect("router can break quality/cost tie")
                .lane(),
            EvolutionLane::Guided
        );
    }

    #[test]
    fn evolution_lane_stats_floor_rounds_pass_rate() {
        let stats = EvolutionLaneStats::new(
            EvolutionLane::Guided,
            3,
            2,
            ConfidencePpm::new(700_000).expect("valid score"),
            ConfidencePpm::new(600_000).expect("valid score"),
            9,
        )
        .expect("valid lane stats");

        assert_eq!(stats.pass_rate_ppm(), 666_666);
    }

    #[test]
    fn evolution_lane_stats_reject_invalid_counters() {
        assert_eq!(
            EvolutionLaneStats::new(
                EvolutionLane::Guided,
                0,
                0,
                ConfidencePpm::new(0).expect("valid score"),
                ConfidencePpm::new(0).expect("valid score"),
                0,
            )
            .expect_err("empty lane history rejected"),
            FabricError::EmptyEvolutionLaneStats
        );

        assert_eq!(
            EvolutionLaneStats::new(
                EvolutionLane::Guided,
                4,
                5,
                ConfidencePpm::new(1_000_000).expect("valid score"),
                ConfidencePpm::new(1_000_000).expect("valid score"),
                1,
            )
            .expect_err("pass count above evaluated count rejected"),
            FabricError::InvalidEvolutionLaneStats
        );

        assert_eq!(
            EvolutionLaneStats::new(
                EvolutionLane::Guided,
                4,
                4,
                ConfidencePpm::new(1_000_000).expect("valid score"),
                ConfidencePpm::new(1_000_000).expect("valid score"),
                0,
            )
            .expect_err("zero deterministic cost rejected"),
            FabricError::ZeroEvolutionLaneCost
        );
    }

    fn consensus_vote(source_slot: u16, kind: ConsensusVoteKind, confidence: u32) -> ConsensusVote {
        ConsensusVote::new(
            u64::from(source_slot),
            consensus_source_kind(source_slot),
            kind,
            ConfidencePpm::new(900_000).expect("valid reputation"),
            ConfidencePpm::new(confidence).expect("valid confidence"),
            ConfidencePpm::new(900_000).expect("valid evidence quality"),
            0,
        )
    }

    fn consensus_vote_with_reputation(
        source_slot: u16,
        kind: ConsensusVoteKind,
        reputation: u32,
        confidence: u32,
    ) -> ConsensusVote {
        ConsensusVote::new(
            u64::from(source_slot),
            ConsensusSourceKind::PrismionProposal,
            kind,
            ConfidencePpm::new(reputation).expect("valid reputation"),
            ConfidencePpm::new(confidence).expect("valid confidence"),
            ConfidencePpm::new(900_000).expect("valid evidence quality"),
            0,
        )
    }

    const fn consensus_source_kind(_source_slot: u16) -> ConsensusSourceKind {
        ConsensusSourceKind::PrismionProposal
    }

    #[test]
    fn agency_consensus_rejects_zero_worker_policy() {
        assert_eq!(
            ConsensusPolicy::new(
                0,
                ConfidencePpm::new(340_000).expect("valid accept threshold"),
                ConfidencePpm::new(340_000).expect("valid reject threshold"),
                ConfidencePpm::new(320_000).expect("valid conflict threshold"),
                ConfidencePpm::new(440_000).expect("valid stale threshold"),
                32,
                ConfidencePpm::new(160_000).expect("valid signal floor"),
            )
            .expect_err("zero-worker consensus policy rejected"),
            FabricError::InvalidConsensusPolicy
        );
    }

    #[test]
    fn agency_consensus_commits_clean_supported_votes() {
        let policy = ConsensusPolicy::e150_best_safe().expect("valid E150 policy");
        let decision = policy
            .decide(&[
                consensus_vote(1, ConsensusVoteKind::Support, 950_000),
                consensus_vote(2, ConsensusVoteKind::Support, 900_000),
            ])
            .expect("valid consensus vote set");

        assert_eq!(decision.action(), ConsensusRecommendation::RecommendCommit);
        assert_eq!(decision.allowed_votes(), 2);
        assert_eq!(decision.blocked_votes(), 0);
        assert!(decision.score_ppm() >= 340_000);
    }

    #[test]
    fn agency_consensus_blocks_stale_ingress_before_accumulation() {
        let policy = ConsensusPolicy::e150_best_safe().expect("valid E150 policy");
        let stale = ConsensusVote::new(
            1,
            ConsensusSourceKind::TraceContext,
            ConsensusVoteKind::Support,
            ConfidencePpm::new(1_000_000).expect("valid reputation"),
            ConfidencePpm::new(1_000_000).expect("valid confidence"),
            ConfidencePpm::new(1_000_000).expect("valid evidence quality"),
            33,
        );
        let valid = consensus_vote(2, ConsensusVoteKind::Reject, 900_000);

        let decision = policy
            .decide(&[stale, valid])
            .expect("valid consensus vote set");

        assert_eq!(decision.action(), ConsensusRecommendation::RecommendDefer);
        assert_eq!(decision.allowed_votes(), 1);
        assert_eq!(decision.blocked_votes(), 1);
    }

    #[test]
    fn agency_consensus_blocks_stale_or_weak_signal_votes() {
        let policy = ConsensusPolicy::e150_best_safe().expect("valid E150 policy");
        let stale = ConsensusVote::new(
            1,
            ConsensusSourceKind::TraceContext,
            ConsensusVoteKind::Support,
            ConfidencePpm::new(1_000_000).expect("valid reputation"),
            ConfidencePpm::new(1_000_000).expect("valid confidence"),
            ConfidencePpm::new(1_000_000).expect("valid evidence quality"),
            33,
        );
        let weak = ConsensusVote::new(
            2,
            ConsensusSourceKind::GroundContext,
            ConsensusVoteKind::Support,
            ConfidencePpm::new(1_000_000).expect("valid reputation"),
            ConfidencePpm::new(300_000).expect("valid confidence"),
            ConfidencePpm::new(300_000).expect("valid evidence quality"),
            0,
        );

        let decision = policy
            .decide(&[stale, weak])
            .expect("stale and weak votes are valid but blocked");

        assert_eq!(decision.action(), ConsensusRecommendation::RecommendDefer);
        assert_eq!(decision.allowed_votes(), 0);
        assert_eq!(decision.blocked_votes(), 2);
        assert_eq!(decision.score_ppm(), 0);
    }

    #[test]
    fn agency_consensus_uses_policy_stale_cutoff() {
        let strict_policy = ConsensusPolicy::new_with_source_diversity_and_stale_after(
            2,
            1,
            ConfidencePpm::new(340_000).expect("valid accept threshold"),
            ConfidencePpm::new(340_000).expect("valid reject threshold"),
            ConfidencePpm::new(320_000).expect("valid conflict threshold"),
            ConfidencePpm::new(100_000).expect("valid stale threshold"),
            32,
            4,
            ConfidencePpm::new(160_000).expect("valid signal floor"),
        )
        .expect("valid strict stale policy");
        let relaxed_policy = ConsensusPolicy::new_with_source_diversity_and_stale_after(
            2,
            1,
            ConfidencePpm::new(340_000).expect("valid accept threshold"),
            ConfidencePpm::new(340_000).expect("valid reject threshold"),
            ConfidencePpm::new(320_000).expect("valid conflict threshold"),
            ConfidencePpm::new(100_000).expect("valid stale threshold"),
            32,
            8,
            ConfidencePpm::new(160_000).expect("valid signal floor"),
        )
        .expect("valid relaxed stale policy");
        let aged_votes = [
            ConsensusVote::new(
                10,
                ConsensusSourceKind::PrismionProposal,
                ConsensusVoteKind::Support,
                ConfidencePpm::new(1_000_000).expect("valid reputation"),
                ConfidencePpm::new(1_000_000).expect("valid confidence"),
                ConfidencePpm::new(1_000_000).expect("valid evidence quality"),
                5,
            ),
            ConsensusVote::new(
                11,
                ConsensusSourceKind::PrismionProposal,
                ConsensusVoteKind::Support,
                ConfidencePpm::new(1_000_000).expect("valid reputation"),
                ConfidencePpm::new(1_000_000).expect("valid confidence"),
                ConfidencePpm::new(1_000_000).expect("valid evidence quality"),
                5,
            ),
        ];

        let strict_decision = strict_policy
            .decide(&aged_votes)
            .expect("aged votes are valid under strict policy");
        let relaxed_decision = relaxed_policy
            .decide(&aged_votes)
            .expect("aged votes are valid under relaxed policy");

        assert_eq!(
            strict_decision.action(),
            ConsensusRecommendation::RecommendDefer
        );
        assert!(strict_decision.stale_ppm() > strict_policy.max_stale().as_ppm());
        assert_eq!(
            relaxed_decision.action(),
            ConsensusRecommendation::RecommendCommit
        );
        assert_eq!(relaxed_decision.stale_ppm(), 0);
        assert_eq!(relaxed_policy.stale_after_ticks(), 8);
    }

    #[test]
    fn agency_consensus_rejects_duplicate_source_votes() {
        let policy = ConsensusPolicy::e150_best_safe().expect("valid E150 policy");

        assert_eq!(
            policy
                .decide(&[
                    consensus_vote(1, ConsensusVoteKind::Support, 900_000),
                    consensus_vote(1, ConsensusVoteKind::Support, 800_000),
                ])
                .expect_err("duplicate source vote rejected"),
            FabricError::DuplicateConsensusSource
        );
    }

    #[test]
    fn agency_consensus_requires_independent_prismion_sources() {
        let policy = ConsensusPolicy::e150_best_safe().expect("valid E150 policy");
        let context_only_votes = [
            ConsensusVote::new(
                0,
                ConsensusSourceKind::ProposalField,
                ConsensusVoteKind::Support,
                ConfidencePpm::new(1_000_000).expect("valid reputation"),
                ConfidencePpm::new(1_000_000).expect("valid confidence"),
                ConfidencePpm::new(1_000_000).expect("valid evidence quality"),
                0,
            ),
            ConsensusVote::new(
                1,
                ConsensusSourceKind::TraceContext,
                ConsensusVoteKind::Support,
                ConfidencePpm::new(1_000_000).expect("valid reputation"),
                ConfidencePpm::new(1_000_000).expect("valid confidence"),
                ConfidencePpm::new(1_000_000).expect("valid evidence quality"),
                0,
            ),
            ConsensusVote::new(
                2,
                ConsensusSourceKind::GroundContext,
                ConsensusVoteKind::Support,
                ConfidencePpm::new(1_000_000).expect("valid reputation"),
                ConfidencePpm::new(1_000_000).expect("valid confidence"),
                ConfidencePpm::new(1_000_000).expect("valid evidence quality"),
                0,
            ),
        ];

        let decision = policy
            .decide(&context_only_votes)
            .expect("context votes are valid but cannot satisfy source quorum");

        assert_eq!(decision.action(), ConsensusRecommendation::RecommendDefer);
        assert_eq!(decision.allowed_votes(), 3);
        assert_eq!(decision.allowed_source_kinds(), 3);
        assert_eq!(decision.allowed_independent_sources(), 0);
        assert_eq!(decision.winner_independent_sources(), 0);
    }

    #[test]
    fn agency_consensus_requires_winner_side_independent_quorum() {
        let policy = ConsensusPolicy::new(
            2,
            ConfidencePpm::new(100_000).expect("valid accept threshold"),
            ConfidencePpm::new(100_000).expect("valid reject threshold"),
            ConfidencePpm::new(1_000_000).expect("valid conflict threshold"),
            ConfidencePpm::new(1_000_000).expect("valid stale threshold"),
            32,
            ConfidencePpm::new(0).expect("valid signal floor"),
        )
        .expect("valid consensus policy");
        let decision = policy
            .decide(&[
                consensus_vote(1, ConsensusVoteKind::Support, 1_000_000),
                consensus_vote(2, ConsensusVoteKind::Reject, 100_000),
                ConsensusVote::new(
                    3,
                    ConsensusSourceKind::ProposalField,
                    ConsensusVoteKind::Support,
                    ConfidencePpm::new(1_000_000).expect("valid reputation"),
                    ConfidencePpm::new(1_000_000).expect("valid confidence"),
                    ConfidencePpm::new(1_000_000).expect("valid evidence quality"),
                    0,
                ),
            ])
            .expect("valid consensus vote set");

        assert_eq!(decision.action(), ConsensusRecommendation::RecommendDefer);
        assert_eq!(decision.allowed_independent_sources(), 2);
        assert_eq!(decision.winner_independent_sources(), 1);
    }

    #[test]
    fn agency_consensus_zero_weight_vote_does_not_count_toward_quorum() {
        let policy = ConsensusPolicy::new(
            2,
            ConfidencePpm::new(100_000).expect("valid accept threshold"),
            ConfidencePpm::new(100_000).expect("valid reject threshold"),
            ConfidencePpm::new(1_000_000).expect("valid conflict threshold"),
            ConfidencePpm::new(1_000_000).expect("valid stale threshold"),
            32,
            ConfidencePpm::new(0).expect("valid signal floor"),
        )
        .expect("valid consensus policy");
        let decision = policy
            .decide(&[
                consensus_vote_with_reputation(1, ConsensusVoteKind::Support, 900_000, 1_000_000),
                consensus_vote_with_reputation(2, ConsensusVoteKind::Support, 0, 1_000_000),
                ConsensusVote::new(
                    3,
                    ConsensusSourceKind::TraceContext,
                    ConsensusVoteKind::Support,
                    ConfidencePpm::new(1_000_000).expect("valid reputation"),
                    ConfidencePpm::new(1_000_000).expect("valid confidence"),
                    ConfidencePpm::new(1_000_000).expect("valid evidence quality"),
                    0,
                ),
            ])
            .expect("valid consensus vote set");

        assert_eq!(decision.action(), ConsensusRecommendation::RecommendDefer);
        assert_eq!(decision.blocked_votes(), 1);
        assert_eq!(decision.allowed_independent_sources(), 1);
        assert_eq!(decision.winner_independent_sources(), 1);
    }

    #[test]
    fn agency_consensus_rejects_unrepresentable_vote_count() {
        let policy = ConsensusPolicy::e150_best_safe().expect("valid E150 policy");
        let votes: Vec<_> = (0..=u16::MAX)
            .map(|source| consensus_vote(source, ConsensusVoteKind::Support, 900_000))
            .collect();

        assert_eq!(
            policy
                .decide(&votes)
                .expect_err("vote count above compact counter range rejected"),
            FabricError::TooManyConsensusVotes
        );
    }

    #[test]
    fn agency_consensus_defers_high_conflict_outputs() {
        let policy = ConsensusPolicy::new(
            2,
            ConfidencePpm::new(100_000).expect("valid threshold"),
            ConfidencePpm::new(100_000).expect("valid threshold"),
            ConfidencePpm::new(200_000).expect("valid conflict cap"),
            ConfidencePpm::new(1_000_000).expect("valid stale cap"),
            32,
            ConfidencePpm::new(0).expect("valid signal floor"),
        )
        .expect("valid consensus policy");

        let decision = policy
            .decide(&[
                consensus_vote(1, ConsensusVoteKind::Support, 1_000_000),
                consensus_vote(2, ConsensusVoteKind::Support, 1_000_000),
                consensus_vote(3, ConsensusVoteKind::Reject, 700_000),
            ])
            .expect("valid consensus vote set");

        assert_eq!(decision.action(), ConsensusRecommendation::RecommendDefer);
        assert!(decision.conflict_ppm() > 200_000);
    }
}
