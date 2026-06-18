//! Locked VRAXION body v1.
//!
//! This module consolidates the current runtime body locks from E64 into the
//! Rust kernel. It is intentionally mechanical: fields are capacity-bounded
//! state containers, pockets still emit proposals, and only Agency commits can
//! change Flow/Ground state.

use crate::agency::{
    agency_decide, agency_decide_atomic_batch, Action, AgencyState, AtomicCommitDecision,
    AtomicCommitPolicy, AtomicCommitProposal, AtomicRejectReason, CommitRecord,
};
use crate::egress::{render_output, EgressMode, RenderedOutput};
use crate::proposal::{ingress_to_proposal, Proposal};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BodyConfig {
    pub name: &'static str,
    pub flow_side: usize,
    pub ground_side: usize,
    pub proposal_slots: usize,
    pub proposal_bits: usize,
    pub agency_view_bits: usize,
}

impl BodyConfig {
    pub const fn flow_cells(self) -> usize {
        self.flow_side * self.flow_side
    }

    pub const fn ground_cells(self) -> usize {
        self.ground_side * self.ground_side
    }

    pub const fn proposal_capacity_bits(self) -> usize {
        self.proposal_slots * self.proposal_bits
    }

    pub const fn total_work_cells(self) -> usize {
        self.flow_cells()
            + self.ground_cells()
            + self.proposal_capacity_bits()
            + self.agency_view_bits
    }
}

pub const DEFAULT_BODY: BodyConfig = BodyConfig {
    name: "near_28f_32g_20x80_default",
    flow_side: 28,
    ground_side: 32,
    proposal_slots: 20,
    proposal_bits: 80,
    agency_view_bits: 896,
};

pub const EXTENDED_BODY: BodyConfig = BodyConfig {
    name: "wide_32x32_20x80",
    flow_side: 32,
    ground_side: 32,
    proposal_slots: 20,
    proposal_bits: 80,
    agency_view_bits: 1024,
};

pub const RESEARCH_CEILING_BODY: BodyConfig = BodyConfig {
    name: "large_48x48_24x80",
    flow_side: 48,
    ground_side: 48,
    proposal_slots: 24,
    proposal_bits: 80,
    agency_view_bits: 1536,
};

pub const OVERCAPACITY_AVOID_DEFAULT: BodyConfig = BodyConfig {
    name: "oversized_64x64_32x80",
    flow_side: 64,
    ground_side: 64,
    proposal_slots: 32,
    proposal_bits: 80,
    agency_view_bits: 2048,
};

pub const PROPOSAL_WIDTH_64_CONTROL: BodyConfig = BodyConfig {
    name: "proposal_width_64_control",
    flow_side: 24,
    ground_side: 32,
    proposal_slots: 16,
    proposal_bits: 64,
    agency_view_bits: 768,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    Flow,
    Ground,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldMatrix {
    kind: FieldKind,
    side: usize,
    cells: Vec<u8>,
}

impl FieldMatrix {
    pub fn new(kind: FieldKind, side: usize) -> Self {
        Self {
            kind,
            side,
            cells: vec![0; side * side],
        }
    }

    pub fn side(&self) -> usize {
        self.side
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn read(&self, index: usize) -> Option<u8> {
        self.cells.get(index).copied()
    }

    pub fn write_committed(&mut self, index: usize, value: u8) -> bool {
        if let Some(cell) = self.cells.get_mut(index) {
            *cell = value & 1;
            return true;
        }
        false
    }

    pub fn active_count(&self) -> usize {
        self.cells.iter().filter(|cell| **cell != 0).count()
    }

    pub fn kind(&self) -> FieldKind {
        self.kind
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProposalFieldError {
    SlotOverflow,
    SlotWidthTooSmall,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposalField {
    config: BodyConfig,
    slots: Vec<Proposal>,
}

impl ProposalField {
    pub fn new(config: BodyConfig) -> Self {
        Self {
            config,
            slots: Vec::with_capacity(config.proposal_slots),
        }
    }

    pub fn push(&mut self, proposal: Proposal) -> Result<(), ProposalFieldError> {
        if self.slots.len() >= self.config.proposal_slots {
            return Err(ProposalFieldError::SlotOverflow);
        }
        if proposal_record_bits(&proposal) > self.config.proposal_bits {
            return Err(ProposalFieldError::SlotWidthTooSmall);
        }
        self.slots.push(proposal);
        Ok(())
    }

    pub fn clear_cycle(&mut self) {
        self.slots.clear();
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

pub fn proposal_record_bits(proposal: &Proposal) -> usize {
    let base = 8 // action code + schema flags
        + 12     // source pocket id compact digest
        + 12     // cycle id compact digest
        + 8      // target/value envelope
        + 16     // trace/evidence/ground compatibility refs
        + 16; // read/write footprint digest
    let payload =
        usize::from(proposal.target_feature.is_some()) * 5 + usize::from(proposal.value.is_some());
    base + payload
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgencyView {
    bits: usize,
}

impl AgencyView {
    pub fn new(config: BodyConfig) -> Self {
        Self {
            bits: config.agency_view_bits,
        }
    }

    pub fn bits(&self) -> usize {
        self.bits
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeStep {
    pub action: Action,
    pub committed: Option<CommitRecord>,
    pub rendered: RenderedOutput,
    pub proposal_slots_used: usize,
    pub flow_active_cells: usize,
    pub ground_active_cells: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicRuntimeStep {
    pub decision: AtomicCommitDecision,
    pub committed: Vec<CommitRecord>,
    pub proposal_slots_used: usize,
    pub flow_active_cells: usize,
    pub ground_active_cells: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicOverlayCanaryConfig {
    pub canary_overlay_active: bool,
    pub rollback_snapshot_required: bool,
    pub production_apply_allowed_now: bool,
}

impl AtomicOverlayCanaryConfig {
    pub const fn e136q_canary() -> Self {
        Self {
            canary_overlay_active: true,
            rollback_snapshot_required: true,
            production_apply_allowed_now: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicOverlayCanaryStep {
    pub canary_overlay_active: bool,
    pub production_apply_allowed_now: bool,
    pub rollback_snapshot_taken: bool,
    pub default_route_unchanged: bool,
    pub default_flow_active_cells_before: usize,
    pub default_flow_active_cells_after: usize,
    pub default_ground_active_cells_before: usize,
    pub default_ground_active_cells_after: usize,
    pub overlay_step: AtomicRuntimeStep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicDefaultRouteSwitchCanaryConfig {
    pub switch_canary_active: bool,
    pub default_route_apply_allowed: bool,
    pub rollback_snapshot_required: bool,
    pub require_preview_match: bool,
    pub production_apply_allowed_now: bool,
}

impl AtomicDefaultRouteSwitchCanaryConfig {
    pub const fn e136s_switch_canary() -> Self {
        Self {
            switch_canary_active: true,
            default_route_apply_allowed: true,
            rollback_snapshot_required: true,
            require_preview_match: true,
            production_apply_allowed_now: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicDefaultRouteSwitchCanaryStep {
    pub switch_canary_active: bool,
    pub default_route_apply_allowed: bool,
    pub production_apply_allowed_now: bool,
    pub rollback_snapshot_taken: bool,
    pub rollback_ready: bool,
    pub preview_checked: bool,
    pub preview_guard_passed: bool,
    pub preview_match: bool,
    pub default_route_applied: bool,
    pub default_route_rolled_back: bool,
    pub default_route_unchanged_on_block: bool,
    pub held_variant_promoted: bool,
    pub default_flow_active_cells_before: usize,
    pub default_flow_active_cells_after: usize,
    pub default_ground_active_cells_before: usize,
    pub default_ground_active_cells_after: usize,
    pub preview_step: AtomicRuntimeStep,
    pub applied_step: Option<AtomicRuntimeStep>,
}

#[derive(Debug, Clone)]
pub struct LockedBodyRuntime {
    pub config: BodyConfig,
    pub flow: FieldMatrix,
    pub ground: FieldMatrix,
    pub proposal_field: ProposalField,
    pub agency_view: AgencyView,
    pub cycle_id: u32,
}

impl LockedBodyRuntime {
    pub fn new(config: BodyConfig) -> Self {
        Self {
            config,
            flow: FieldMatrix::new(FieldKind::Flow, config.flow_side),
            ground: FieldMatrix::new(FieldKind::Ground, config.ground_side),
            proposal_field: ProposalField::new(config),
            agency_view: AgencyView::new(config),
            cycle_id: 1,
        }
    }

    pub fn default_body() -> Self {
        Self::new(DEFAULT_BODY)
    }

    pub fn process_binary_evidence(
        &mut self,
        source_pocket_id: u32,
        requested_feature: u8,
        stream: &[u8],
    ) -> RuntimeStep {
        self.proposal_field.clear_cycle();
        let (_decoded, proposal) =
            ingress_to_proposal(self.cycle_id, source_pocket_id, requested_feature, stream);
        let field_result = self.proposal_field.push(proposal);
        let (action, committed) = match field_result {
            Ok(()) => agency_decide(
                AgencyState {
                    cycle_id: self.cycle_id,
                },
                proposal,
            ),
            Err(_) => (Action::Reject, None),
        };
        if let Some(record) = committed {
            self.commit_record(record);
        }
        let rendered = render_output(EgressMode::MultiResolution, committed);
        RuntimeStep {
            action,
            committed,
            rendered,
            proposal_slots_used: self.proposal_field.len(),
            flow_active_cells: self.flow.active_count(),
            ground_active_cells: self.ground.active_count(),
        }
    }

    pub fn process_atomic_proposals_preview(
        &mut self,
        policy: AtomicCommitPolicy,
        proposals: &[AtomicCommitProposal],
    ) -> AtomicRuntimeStep {
        self.proposal_field.clear_cycle();
        if proposals.len() > self.config.proposal_slots {
            let decision =
                AtomicCommitDecision::rejected(AtomicRejectReason::ProposalCapacityExceeded);
            return AtomicRuntimeStep {
                decision,
                committed: Vec::new(),
                proposal_slots_used: 0,
                flow_active_cells: self.flow.active_count(),
                ground_active_cells: self.ground.active_count(),
            };
        }

        let decision = agency_decide_atomic_batch(
            AgencyState {
                cycle_id: self.cycle_id,
            },
            policy,
            proposals,
        );
        let committed = if decision.committed() {
            decision.records.clone()
        } else {
            Vec::new()
        };
        for record in &committed {
            self.commit_record(*record);
        }
        AtomicRuntimeStep {
            decision,
            committed,
            proposal_slots_used: proposals.len(),
            flow_active_cells: self.flow.active_count(),
            ground_active_cells: self.ground.active_count(),
        }
    }

    pub fn process_atomic_overlay_canary(
        &self,
        config: AtomicOverlayCanaryConfig,
        policy: AtomicCommitPolicy,
        proposals: &[AtomicCommitProposal],
    ) -> AtomicOverlayCanaryStep {
        let default_flow_before = self.flow.active_count();
        let default_ground_before = self.ground.active_count();
        let mut overlay = self.clone();
        let overlay_step = if config.canary_overlay_active {
            overlay.process_atomic_proposals_preview(policy, proposals)
        } else {
            AtomicRuntimeStep {
                decision: AtomicCommitDecision::rejected(AtomicRejectReason::NoValidProposal),
                committed: Vec::new(),
                proposal_slots_used: 0,
                flow_active_cells: overlay.flow.active_count(),
                ground_active_cells: overlay.ground.active_count(),
            }
        };
        let default_flow_after = self.flow.active_count();
        let default_ground_after = self.ground.active_count();
        AtomicOverlayCanaryStep {
            canary_overlay_active: config.canary_overlay_active,
            production_apply_allowed_now: config.production_apply_allowed_now,
            rollback_snapshot_taken: config.rollback_snapshot_required
                && config.canary_overlay_active,
            default_route_unchanged: default_flow_before == default_flow_after
                && default_ground_before == default_ground_after,
            default_flow_active_cells_before: default_flow_before,
            default_flow_active_cells_after: default_flow_after,
            default_ground_active_cells_before: default_ground_before,
            default_ground_active_cells_after: default_ground_after,
            overlay_step,
        }
    }

    pub fn process_atomic_default_route_switch_canary(
        &mut self,
        config: AtomicDefaultRouteSwitchCanaryConfig,
        policy: AtomicCommitPolicy,
        proposals: &[AtomicCommitProposal],
    ) -> AtomicDefaultRouteSwitchCanaryStep {
        let snapshot = self.clone();
        let default_flow_before = self.flow.active_count();
        let default_ground_before = self.ground.active_count();
        let rollback_snapshot_taken =
            config.rollback_snapshot_required && config.switch_canary_active;

        let mut preview_runtime = self.clone();
        let preview_step = if config.switch_canary_active {
            preview_runtime.process_atomic_proposals_preview(policy, proposals)
        } else {
            AtomicRuntimeStep {
                decision: AtomicCommitDecision::rejected(AtomicRejectReason::NoValidProposal),
                committed: Vec::new(),
                proposal_slots_used: 0,
                flow_active_cells: self.flow.active_count(),
                ground_active_cells: self.ground.active_count(),
            }
        };

        let held_ids: Vec<u32> = proposals
            .iter()
            .filter(|proposal| {
                matches!(
                    proposal.role,
                    crate::agency::AtomicProposalRole::HeldChallenger
                        | crate::agency::AtomicProposalRole::HeldLineage
                )
            })
            .map(|proposal| proposal.source_pocket_id)
            .collect();
        let held_variant_promoted = preview_step
            .decision
            .records
            .iter()
            .any(|record| held_ids.contains(&record.source_pocket_id));
        let preview_guard_passed = config.switch_canary_active
            && config.default_route_apply_allowed
            && !config.production_apply_allowed_now
            && rollback_snapshot_taken
            && preview_step.decision.committed()
            && preview_step.committed.len() == preview_step.decision.records.len()
            && preview_step.decision.records.len() <= policy.max_multi_write
            && preview_step.decision.order_independent
            && preview_step.decision.runtime_direct_write_count == 0
            && preview_step.decision.oracle_plan_feature_use_count == 0
            && !held_variant_promoted;

        let mut applied_step = None;
        let mut preview_match = !preview_guard_passed;
        let mut default_route_applied = false;
        let mut default_route_rolled_back = false;

        if preview_guard_passed {
            let applied = self.process_atomic_proposals_preview(policy, proposals);
            preview_match = applied.decision == preview_step.decision
                && applied.committed == preview_step.committed
                && applied.flow_active_cells == preview_step.flow_active_cells
                && applied.ground_active_cells == preview_step.ground_active_cells;
            if config.require_preview_match && !preview_match {
                *self = snapshot.clone();
                default_route_rolled_back = true;
            } else {
                default_route_applied = true;
            }
            applied_step = Some(applied);
        }

        let unchanged_after_block = !default_route_applied
            && self.flow == snapshot.flow
            && self.ground == snapshot.ground
            && self.cycle_id == snapshot.cycle_id;
        let default_flow_after = self.flow.active_count();
        let default_ground_after = self.ground.active_count();

        AtomicDefaultRouteSwitchCanaryStep {
            switch_canary_active: config.switch_canary_active,
            default_route_apply_allowed: config.default_route_apply_allowed,
            production_apply_allowed_now: config.production_apply_allowed_now,
            rollback_snapshot_taken,
            rollback_ready: rollback_snapshot_taken,
            preview_checked: config.switch_canary_active,
            preview_guard_passed,
            preview_match,
            default_route_applied,
            default_route_rolled_back,
            default_route_unchanged_on_block: unchanged_after_block,
            held_variant_promoted,
            default_flow_active_cells_before: default_flow_before,
            default_flow_active_cells_after: default_flow_after,
            default_ground_active_cells_before: default_ground_before,
            default_ground_active_cells_after: default_ground_after,
            preview_step,
            applied_step,
        }
    }

    fn commit_record(&mut self, record: CommitRecord) {
        let flow_index = flow_cell_for(record.feature_id, self.config);
        let ground_index = ground_cell_for(record.feature_id, self.config);
        self.flow.write_committed(flow_index, record.value);
        self.ground.write_committed(ground_index, record.value);
    }
}

pub fn flow_cell_for(feature_id: u8, config: BodyConfig) -> usize {
    ((feature_id as usize) * 17 + 3) % config.flow_cells()
}

pub fn ground_cell_for(feature_id: u8, config: BodyConfig) -> usize {
    ((feature_id as usize) * 31 + 11) % config.ground_cells()
}
