//! Stable fabric error codes.

use core::fmt;

macro_rules! fabric_errors {
    ($($name:ident => $code:literal, $doc:literal;)+) => {
        /// Stable error codes for anonymous fabric primitives.
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub enum FabricError { $(#[doc = $doc] $name,)+ }

        impl FabricError {
            /// All stable fabric errors in declaration order.
            pub const ALL: &[Self] = &[$(Self::$name,)+];

            /// Returns the stable machine-readable error code.
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self { $(Self::$name => $code,)+ }
            }
        }
    };
}

fabric_errors! {
    ZeroDimension => "zero_dimension", "A matrix axis was zero.";
    MatrixTooLarge => "matrix_too_large", "A matrix shape exceeded the bounded cell limit.";
    DenseCellCountMismatch => "dense_cell_count_mismatch", "Dense cell count did not match the matrix shape.";
    CellOutOfBounds => "cell_out_of_bounds", "A matrix address was outside the relevant matrix.";
    DuplicateSparseCell => "duplicate_sparse_cell", "Sparse input contained the same address more than once.";
    TooManySparseCells => "too_many_sparse_cells", "Sparse input contained more samples than the shape can hold.";
    NonCanonicalZeroSparseCell => "noncanonical_zero_sparse_cell", "Sparse input explicitly encoded a default zero value.";
    EmptyConditionSet => "empty_condition_set", "A Prismion had no conditions.";
    EmptyPatchSet => "empty_patch_set", "A Prismion had no proposal patches.";
    TooManyConditions => "too_many_conditions", "A Prismion exceeded the condition cap.";
    TooManyPatches => "too_many_patches", "A Prismion exceeded the patch cap.";
    DuplicatePredicate => "duplicate_predicate", "A Prismion repeated an identical predicate.";
    DuplicatePatchTarget => "duplicate_patch_target", "A Prismion proposed two writes to the same proposal-field target.";
    ProposalPatchOutOfBounds => "proposal_patch_out_of_bounds", "A Prismion proposed a patch outside the provided proposal-field shape.";
    TooManyProposals => "too_many_proposals", "One Proposal Field materialization received too many proposals.";
    TooManyProposalPatches => "too_many_proposal_patches", "One Proposal Field materialization received too many aggregate patches.";
    TooManyEvidenceCells => "too_many_evidence_cells", "One runtime cycle accumulated too many aggregate evidence cells.";
    InvalidConfidence => "invalid_confidence", "Confidence was outside the accepted range.";
    EmptyEvolutionPatchSet => "empty_evolution_patch_set", "An evolution proposal did not contain any mutation patch.";
    TooManyEvolutionPatches => "too_many_evolution_patches", "An evolution proposal exceeded the mutation patch cap.";
    DuplicateEvolutionPatchSlot => "duplicate_evolution_patch_slot", "An evolution proposal repeated a mutation patch slot.";
    EmptyEvolutionRouterStats => "empty_evolution_router_stats", "The router received no lane statistics.";
    DuplicateEvolutionLaneStats => "duplicate_evolution_lane_stats", "The router received more than one statistic for the same lane.";
    EmptyEvolutionLaneStats => "empty_evolution_lane_stats", "One lane statistic had no evaluated history.";
    ZeroEvolutionLaneCost => "zero_evolution_lane_cost", "One evaluated lane statistic reported zero deterministic work.";
    InvalidEvolutionLaneStats => "invalid_evolution_lane_stats", "One lane statistic had impossible counters or score ordering.";
    InvalidConsensusThreshold => "invalid_consensus_threshold", "A consensus policy threshold was outside the valid confidence range.";
    InvalidConsensusPolicy => "invalid_consensus_policy", "A consensus policy had internally contradictory thresholds.";
    TooManyConsensusVotes => "too_many_consensus_votes", "A consensus vote set cannot be represented by compact vote counters.";
    DuplicateConsensusSource => "duplicate_consensus_source", "A consensus vote set repeated the same source slot.";
}

impl fmt::Display for FabricError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::error::Error for FabricError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::FabricError;

    #[test]
    fn fabric_error_codes_are_unique_and_machine_safe() {
        let mut seen = BTreeSet::new();
        for &error in FabricError::ALL {
            let code = error.as_str();
            assert!(seen.insert(code), "duplicate FabricError code: {code}");
            assert!(!code.is_empty() && !code.starts_with('_') && !code.ends_with('_'));
            assert!(!code.contains("__"));
            assert!(code.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'
            }));
            assert_eq!(error.to_string(), code);
        }
        assert_eq!(seen.len(), FabricError::ALL.len());
    }
}
