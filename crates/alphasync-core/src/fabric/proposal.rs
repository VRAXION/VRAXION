//! Proposal-field primitives emitted by Prismions.

use crate::ids::{PrismionId, ProposalId};

use super::error::FabricError;
use super::matrix::{MatrixAddress, MatrixShape};
use super::{MAX_PROPOSAL_PATCHES_PER_FIELD, MAX_PROPOSALS_PER_FIELD, confidence::ConfidencePpm};

/// One patch a Prismion proposes to the proposal field.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProposalPatch {
    target: MatrixAddress,
    value: u8,
}

impl ProposalPatch {
    /// Creates one shape-independent proposal-field patch.
    ///
    /// The target is proposal-field-local, not a direct Flow/Ground address.
    /// This constructor deliberately does not know the concrete Proposal Field
    /// shape. A patch is admitted only when a later evaluation/admission path
    /// checks it against the active proposal shape.
    #[must_use]
    pub const fn new(target: MatrixAddress, value: u8) -> Self {
        Self { target, value }
    }

    /// Returns the proposal-field target cell.
    #[must_use]
    pub const fn target(self) -> MatrixAddress {
        self.target
    }

    /// Returns the proposed byte value.
    #[must_use]
    pub const fn value(self) -> u8 {
        self.value
    }
}

/// Proposal emitted by a Prismion after reading an observation matrix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrismionProposal {
    proposal_id: ProposalId,
    source: PrismionId,
    confidence: ConfidencePpm,
    evidence_cells: Vec<MatrixAddress>,
    patches: Vec<ProposalPatch>,
}

impl PrismionProposal {
    pub(super) fn new(
        proposal_id: ProposalId,
        source: PrismionId,
        confidence: ConfidencePpm,
        evidence_cells: Vec<MatrixAddress>,
        patches: Vec<ProposalPatch>,
    ) -> Self {
        Self {
            proposal_id,
            source,
            confidence,
            evidence_cells,
            patches,
        }
    }

    /// Returns the proposal ID supplied by the caller.
    #[must_use]
    pub fn proposal_id(&self) -> &ProposalId {
        &self.proposal_id
    }

    /// Returns the source Prismion ID.
    #[must_use]
    pub fn source(&self) -> &PrismionId {
        &self.source
    }

    /// Returns the proposal confidence.
    #[must_use]
    pub const fn confidence(&self) -> ConfidencePpm {
        self.confidence
    }

    /// Returns the observation cells that triggered the proposal.
    #[must_use]
    pub fn evidence_cells(&self) -> &[MatrixAddress] {
        &self.evidence_cells
    }

    /// Returns proposal-field patches. These are not direct Flow writes.
    #[must_use]
    pub fn patches(&self) -> &[ProposalPatch] {
        &self.patches
    }
}

/// Materialized one-cycle Proposal Field matrix.
///
/// This is the standard fabric-node view of [`PrismionProposal`] output:
///
/// ```text
/// PrismionProposal patch-list
///   -> ProposalField::from_proposals
///   -> dense proposal-value canvas + occupied mask + collision counters
/// ```
///
/// The byte canvas is useful for normal matrix rendering and downstream
/// anonymous fabric processing. The occupied mask is required because a
/// proposal may intentionally write byte value `0`, which would otherwise be
/// indistinguishable from a cell that received no proposal at all.
///
/// Collisions are not silently overwritten. If multiple values target the same
/// address, the materialized cell uses a deterministic local winner
/// `(support_count, confidence_sum, lowest_value)`, while collision counters
/// preserve that the field was contested. That makes the matrix usable without
/// hiding conflict from Agency/Consensus.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProposalField {
    shape: MatrixShape,
    cells: Vec<u8>,
    occupied: Vec<bool>,
    patch_count: usize,
    occupied_cell_count: usize,
    collision_cell_count: usize,
    conflicting_patch_count: usize,
}

impl ProposalField {
    /// Materializes a Proposal Field from emitted Prismion proposals.
    ///
    /// Proposal order does not affect the final field. All patches are sorted
    /// by target and value before local winners are selected.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::ProposalPatchOutOfBounds`] when any patch target
    /// falls outside `shape`.
    pub fn from_proposals(
        shape: MatrixShape,
        proposals: &[PrismionProposal],
    ) -> Result<Self, FabricError> {
        if proposals.len() > MAX_PROPOSALS_PER_FIELD {
            return Err(FabricError::TooManyProposals);
        }

        let mut aggregate_patch_count = 0usize;
        for proposal in proposals {
            aggregate_patch_count = aggregate_patch_count
                .checked_add(proposal.patches().len())
                .ok_or(FabricError::TooManyProposalPatches)?;
            if aggregate_patch_count > MAX_PROPOSAL_PATCHES_PER_FIELD {
                return Err(FabricError::TooManyProposalPatches);
            }
        }

        let mut atoms = Vec::new();
        atoms
            .try_reserve_exact(aggregate_patch_count)
            .map_err(|_error| FabricError::TooManyProposalPatches)?;
        for proposal in proposals {
            for patch in proposal.patches() {
                if !shape.contains(patch.target()) {
                    return Err(FabricError::ProposalPatchOutOfBounds);
                }
                atoms.push(ProposalFieldAtom {
                    target: patch.target(),
                    value: patch.value(),
                    confidence_ppm: proposal.confidence().as_ppm(),
                });
            }
        }

        atoms.sort_unstable_by_key(|atom| (atom.target, atom.value));

        let mut cells = vec![0; shape.cell_count()];
        let mut occupied = vec![false; shape.cell_count()];
        let mut occupied_cell_count = 0usize;
        let mut collision_cell_count = 0usize;
        let mut conflicting_patch_count = 0usize;

        let mut index = 0usize;
        while index < atoms.len() {
            let target = atoms[index].target;
            let start = index;
            while index < atoms.len() && atoms[index].target == target {
                index += 1;
            }

            let group = &atoms[start..index];
            let outcome = choose_cell_outcome(group);
            let cell_index = shape
                .dense_index(target)
                .ok_or(FabricError::ProposalPatchOutOfBounds)?;
            cells[cell_index] = outcome.value;
            occupied[cell_index] = true;
            occupied_cell_count += 1;
            if outcome.distinct_value_count > 1 {
                collision_cell_count += 1;
                conflicting_patch_count += group.len();
            }
        }

        Ok(Self {
            shape,
            cells,
            occupied,
            patch_count: atoms.len(),
            occupied_cell_count,
            collision_cell_count,
            conflicting_patch_count,
        })
    }

    /// Returns the Proposal Field matrix shape.
    #[must_use]
    pub const fn shape(&self) -> MatrixShape {
        self.shape
    }

    /// Returns dense proposal values in canonical plane-row-column order.
    ///
    /// Use [`Self::is_occupied`] or [`Self::get`] when absence and explicit
    /// zero writes must be distinguished.
    #[must_use]
    pub fn dense_cells(&self) -> &[u8] {
        &self.cells
    }

    /// Returns whether a cell received at least one proposal patch.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::CellOutOfBounds`] when `address` is outside this
    /// Proposal Field shape.
    pub fn is_occupied(&self, address: MatrixAddress) -> Result<bool, FabricError> {
        let index = self
            .shape
            .dense_index(address)
            .ok_or(FabricError::CellOutOfBounds)?;
        Ok(self.occupied[index])
    }

    /// Reads one proposed cell value.
    ///
    /// `Ok(None)` means no proposal targeted the cell. `Ok(Some(0))` means the
    /// cell was explicitly proposed as zero.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::CellOutOfBounds`] when `address` is outside this
    /// Proposal Field shape.
    pub fn get(&self, address: MatrixAddress) -> Result<Option<u8>, FabricError> {
        let index = self
            .shape
            .dense_index(address)
            .ok_or(FabricError::CellOutOfBounds)?;
        if self.occupied[index] {
            Ok(Some(self.cells[index]))
        } else {
            Ok(None)
        }
    }

    /// Returns the total number of proposal patches materialized.
    #[must_use]
    pub const fn patch_count(&self) -> usize {
        self.patch_count
    }

    /// Returns the number of unique cells that received proposal patches.
    #[must_use]
    pub const fn occupied_cell_count(&self) -> usize {
        self.occupied_cell_count
    }

    /// Returns the number of cells that received more than one proposed value.
    #[must_use]
    pub const fn collision_cell_count(&self) -> usize {
        self.collision_cell_count
    }

    /// Returns the number of patches participating in contested cells.
    #[must_use]
    pub const fn conflicting_patch_count(&self) -> usize {
        self.conflicting_patch_count
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct ProposalFieldAtom {
    target: MatrixAddress,
    value: u8,
    confidence_ppm: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct ProposalCellOutcome {
    value: u8,
    distinct_value_count: usize,
}

fn choose_cell_outcome(group: &[ProposalFieldAtom]) -> ProposalCellOutcome {
    let mut value_index = 0usize;
    let mut best_value = group[0].value;
    let mut best_count = 0usize;
    let mut best_confidence_sum = 0u64;
    let mut distinct_value_count = 0usize;

    while value_index < group.len() {
        let value = group[value_index].value;
        let mut count = 0usize;
        let mut confidence_sum = 0u64;
        while value_index < group.len() && group[value_index].value == value {
            count += 1;
            confidence_sum =
                confidence_sum.saturating_add(u64::from(group[value_index].confidence_ppm));
            value_index += 1;
        }
        distinct_value_count += 1;

        if count > best_count
            || (count == best_count && confidence_sum > best_confidence_sum)
            || (count == best_count && confidence_sum == best_confidence_sum && value < best_value)
        {
            best_value = value;
            best_count = count;
            best_confidence_sum = confidence_sum;
        }
    }

    ProposalCellOutcome {
        value: best_value,
        distinct_value_count,
    }
}

#[cfg(test)]
mod tests {
    use crate::ids::{PrismionId, ProposalId};

    use super::super::confidence::ConfidencePpm;
    use super::super::error::FabricError;
    use super::super::matrix::{MatrixAddress, MatrixShape};
    use super::super::{MAX_PROPOSAL_PATCHES_PER_FIELD, MAX_PROPOSALS_PER_FIELD};
    use super::{PrismionProposal, ProposalField, ProposalPatch};

    const PRISMION_HEX: &str = "0123456789abcdef0123456789abcdef";
    const PROPOSAL_HEX: &str = "fedcba9876543210fedcba9876543210";

    fn address(column: u16) -> MatrixAddress {
        MatrixAddress::new(0, 0, column)
    }

    fn shape(columns: u16) -> MatrixShape {
        MatrixShape::new(1, 1, columns).expect("valid shape")
    }

    fn confidence(ppm: u32) -> ConfidencePpm {
        ConfidencePpm::new(ppm).expect("valid confidence")
    }

    fn prismion_id() -> PrismionId {
        PrismionId::parse(PRISMION_HEX).expect("valid Prismion ID")
    }

    fn proposal_id() -> ProposalId {
        ProposalId::parse(PROPOSAL_HEX).expect("valid proposal ID")
    }

    fn patch(column: u16, value: u8) -> ProposalPatch {
        ProposalPatch::new(address(column), value)
    }

    fn proposal(confidence_ppm: u32, patches: Vec<ProposalPatch>) -> PrismionProposal {
        PrismionProposal::new(
            proposal_id(),
            prismion_id(),
            confidence(confidence_ppm),
            vec![address(0)],
            patches,
        )
    }

    #[test]
    fn empty_proposal_field_is_zeroed_and_unoccupied() {
        let field = ProposalField::from_proposals(shape(3), &[]).expect("valid empty field");

        assert_eq!(field.patch_count(), 0);
        assert_eq!(field.occupied_cell_count(), 0);
        assert_eq!(field.collision_cell_count(), 0);
        assert_eq!(field.conflicting_patch_count(), 0);
        assert_eq!(field.dense_cells(), &[0, 0, 0]);
        assert_eq!(field.is_occupied(address(0)), Ok(false));
        assert_eq!(field.get(address(0)), Ok(None));
    }

    #[test]
    fn explicit_zero_write_is_distinguishable_from_absence() {
        let proposals = [proposal(500_000, vec![patch(0, 0)])];
        let field = ProposalField::from_proposals(shape(2), &proposals)
            .expect("valid zero-valued proposal");

        assert_eq!(field.dense_cells(), &[0, 0]);
        assert_eq!(field.patch_count(), 1);
        assert_eq!(field.occupied_cell_count(), 1);
        assert_eq!(field.collision_cell_count(), 0);
        assert_eq!(field.conflicting_patch_count(), 0);
        assert_eq!(field.is_occupied(address(0)), Ok(true));
        assert_eq!(field.get(address(0)), Ok(Some(0)));
        assert_eq!(field.is_occupied(address(1)), Ok(false));
        assert_eq!(field.get(address(1)), Ok(None));
    }

    #[test]
    fn out_of_bounds_patch_is_rejected_before_materialization() {
        let proposals = [proposal(500_000, vec![patch(1, 9)])];

        assert_eq!(
            ProposalField::from_proposals(shape(1), &proposals).unwrap_err(),
            FabricError::ProposalPatchOutOfBounds
        );
    }

    #[test]
    fn proposal_count_cap_is_checked_before_patch_scan() {
        let proposals = vec![proposal(500_000, vec![]); MAX_PROPOSALS_PER_FIELD + 1];

        assert_eq!(
            ProposalField::from_proposals(shape(1), &proposals).unwrap_err(),
            FabricError::TooManyProposals
        );
    }

    #[test]
    fn aggregate_patch_cap_is_checked_before_materialization() {
        let patches = vec![patch(0, 1); MAX_PROPOSAL_PATCHES_PER_FIELD + 1];
        let proposals = [proposal(500_000, patches)];

        assert_eq!(
            ProposalField::from_proposals(shape(1), &proposals).unwrap_err(),
            FabricError::TooManyProposalPatches
        );
    }

    #[test]
    fn same_value_support_is_not_a_collision() {
        let proposals = [
            proposal(100_000, vec![patch(0, 7)]),
            proposal(900_000, vec![patch(0, 7)]),
        ];
        let field =
            ProposalField::from_proposals(shape(1), &proposals).expect("valid same-value support");

        assert_eq!(field.patch_count(), 2);
        assert_eq!(field.occupied_cell_count(), 1);
        assert_eq!(field.collision_cell_count(), 0);
        assert_eq!(field.conflicting_patch_count(), 0);
        assert_eq!(field.get(address(0)), Ok(Some(7)));
    }

    #[test]
    fn materialization_is_order_independent_and_resolves_by_support() {
        let first = proposal(500_000, vec![patch(0, 9)]);
        let second = proposal(100_000, vec![patch(0, 5)]);
        let third = proposal(100_000, vec![patch(0, 5)]);
        let forward = [first.clone(), second.clone(), third.clone()];
        let reverse = [third, second, first];

        let forward_field =
            ProposalField::from_proposals(shape(1), &forward).expect("valid forward field");
        let reverse_field =
            ProposalField::from_proposals(shape(1), &reverse).expect("valid reverse field");

        assert_eq!(forward_field, reverse_field);
        assert_eq!(forward_field.patch_count(), 3);
        assert_eq!(forward_field.occupied_cell_count(), 1);
        assert_eq!(forward_field.collision_cell_count(), 1);
        assert_eq!(forward_field.conflicting_patch_count(), 3);
        assert_eq!(forward_field.get(address(0)), Ok(Some(5)));
    }

    #[test]
    fn collision_ties_break_by_confidence_sum_then_lowest_value() {
        let proposals = [
            proposal(100_000, vec![patch(0, 4)]),
            proposal(900_000, vec![patch(0, 9)]),
            proposal(500_000, vec![patch(1, 8)]),
            proposal(500_000, vec![patch(1, 2)]),
        ];
        let field =
            ProposalField::from_proposals(shape(2), &proposals).expect("valid contested field");

        assert_eq!(field.patch_count(), 4);
        assert_eq!(field.occupied_cell_count(), 2);
        assert_eq!(field.collision_cell_count(), 2);
        assert_eq!(field.conflicting_patch_count(), 4);
        assert_eq!(field.get(address(0)), Ok(Some(9)));
        assert_eq!(field.get(address(1)), Ok(Some(2)));
    }

    #[test]
    fn get_and_occupancy_reject_out_of_bounds_addresses() {
        let field = ProposalField::from_proposals(shape(1), &[]).expect("valid empty field");

        assert_eq!(
            field.is_occupied(address(1)).unwrap_err(),
            FabricError::CellOutOfBounds
        );
        assert_eq!(
            field.get(address(1)).unwrap_err(),
            FabricError::CellOutOfBounds
        );
    }
}
