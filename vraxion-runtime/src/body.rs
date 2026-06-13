//! Locked VRAXION body v1.
//!
//! This module consolidates the current runtime body locks from E64 into the
//! Rust kernel. It is intentionally mechanical: fields are capacity-bounded
//! state containers, pockets still emit proposals, and only Agency commits can
//! change Flow/Ground state.

use crate::agency::{agency_decide, Action, AgencyState, CommitRecord};
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
