//! Evolution-field primitives for proposed mutation search.
//!
//! The evolution field is the same architectural pattern as the normal
//! proposal field: it is a place to write candidate next actions, not a place
//! to mutate the live library directly.
//!
//! ```text
//! predicted / guided / hybrid / random lane
//!   -> EvolutionProposal
//!   -> external Mutation Manager / Agency Guard
//!   -> shadow candidate
//!   -> future-only validation
//! ```

use crate::ids::{OperatorId, PrismionId, ProposalId};

use super::confidence::ConfidencePpm;
use super::error::FabricError;

/// Maximum number of mutation patches accepted in one evolution proposal.
///
/// Changeability: easy runtime cap. Raising it increases mutation fanout and
/// should be rechecked against challenger spam and rollback tests.
pub const MAX_EVOLUTION_PATCHES: usize = 64;

/// Source lane that produced a mutation proposal.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EvolutionLane {
    /// Zero/non-zero or explicit binary lane for cases where a binary view is
    /// sufficient and cheaper than full-range u8 search.
    Binary,
    /// Network-predicted next-best mutation lane.
    Predicted,
    /// Survivor-guided / bucketed mutation lane.
    Guided,
    /// Hybrid lane that mixes guided search with a random escape lane.
    Hybrid,
    /// Pure random exploration lane.
    Random,
}

/// Bounded mutation operation proposed by the evolution field.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MutationKind {
    /// Add a guard-like condition.
    AddGuard,
    /// Remove a guard-like condition.
    RemoveGuard,
    /// Make a trigger narrower.
    SpecializeTrigger,
    /// Make a trigger wider.
    WidenTrigger,
    /// Remove a noisy condition or reduce boolean structure.
    SimplifyCondition,
    /// Move a numeric threshold by one allowed grid step.
    NumericThresholdStep,
    /// Split one operator/Prismion into multiple candidates.
    Split,
    /// Merge compatible siblings.
    Merge,
    /// Prune redundant structure while preserving behavior.
    Prune,
}

/// Live artifact targeted by a mutation proposal.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MutationTarget {
    /// Mutate a Prismion matrix-reader rule.
    Prismion(PrismionId),
    /// Mutate a canonical operator definition.
    Operator(OperatorId),
}

/// One byte-sized mutation patch inside an evolution proposal.
///
/// The slot is local to the target's versioned mutation ABI. This type does
/// not interpret the slot as a field name; it only carries the bounded patch
/// that a guarded mutation manager can decode later.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EvolutionPatch {
    slot: u16,
    value: u8,
}

impl EvolutionPatch {
    /// Creates one bounded mutation patch.
    #[must_use]
    pub const fn new(slot: u16, value: u8) -> Self {
        Self { slot, value }
    }

    /// Returns the mutation-ABI-local slot.
    #[must_use]
    pub const fn slot(self) -> u16 {
        self.slot
    }

    /// Returns the proposed byte value for the slot.
    #[must_use]
    pub const fn value(self) -> u8 {
        self.value
    }
}

/// Proposed next mutation written to the evolution field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvolutionProposal {
    proposal_id: ProposalId,
    lane: EvolutionLane,
    target: MutationTarget,
    kind: MutationKind,
    confidence: ConfidencePpm,
    patches: Vec<EvolutionPatch>,
}

impl EvolutionProposal {
    /// Creates a bounded evolution proposal.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError`] when the patch list is empty, too large, or
    /// contains duplicate mutation slots.
    pub fn new(
        proposal_id: ProposalId,
        lane: EvolutionLane,
        target: MutationTarget,
        kind: MutationKind,
        confidence: ConfidencePpm,
        mut patches: Vec<EvolutionPatch>,
    ) -> Result<Self, FabricError> {
        if patches.is_empty() {
            return Err(FabricError::EmptyEvolutionPatchSet);
        }

        if patches.len() > MAX_EVOLUTION_PATCHES {
            return Err(FabricError::TooManyEvolutionPatches);
        }

        patches.sort_unstable();
        if patches.windows(2).any(|pair| pair[0].slot == pair[1].slot) {
            return Err(FabricError::DuplicateEvolutionPatchSlot);
        }

        Ok(Self {
            proposal_id,
            lane,
            target,
            kind,
            confidence,
            patches,
        })
    }

    /// Returns the evolution proposal ID.
    #[must_use]
    pub fn proposal_id(&self) -> &ProposalId {
        &self.proposal_id
    }

    /// Returns the source lane.
    #[must_use]
    pub const fn lane(&self) -> EvolutionLane {
        self.lane
    }

    /// Returns the mutation target.
    #[must_use]
    pub const fn target(&self) -> &MutationTarget {
        &self.target
    }

    /// Returns the mutation operation.
    #[must_use]
    pub const fn kind(&self) -> MutationKind {
        self.kind
    }

    /// Returns the proposal confidence.
    #[must_use]
    pub const fn confidence(&self) -> ConfidencePpm {
        self.confidence
    }

    /// Returns canonical mutation patches.
    #[must_use]
    pub fn patches(&self) -> &[EvolutionPatch] {
        &self.patches
    }
}
