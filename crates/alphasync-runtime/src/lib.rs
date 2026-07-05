//! Minimal executable `α-Sync` runtime loop.
//!
//! This crate is the first Rust vertical slice above `alphasync-core`. It does
//! not load datasets, mutate operators, write public claims, or perform
//! semantic truth work. It only wires the already-baked core primitives into a
//! single deterministic cycle:
//!
//! ```text
//! ObservationMatrix
//!   -> PrismionRule evaluation
//!   -> PrismionProposal collection
//!   -> Agency-owned Flow commit
//!   -> Completion/continuation gate
//!   -> safe aggregate cycle report
//! ```
//!
//! The intent is to prove that the Rust core is not just disconnected data
//! types. It is now runnable as a tiny machine loop that future registry,
//! durable writer, and training/evolution layers can call.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(missing_debug_implementations)]
#![deny(rust_2018_idioms)]
#![deny(unused_must_use)]
#![cfg_attr(
    not(test),
    deny(
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::unwrap_used
    )
)]

use core::fmt;

use std::collections::BTreeSet;
use std::sync::atomic::AtomicU64;

use alphasync_core::fabric::{
    CellAddress, CellSample, ConsensusDecision, ConsensusPolicy, ConsensusRecommendation,
    ConsensusVoteKind, FabricError, MAX_EVIDENCE_CELLS_PER_CYCLE, MAX_PROPOSAL_PATCHES_PER_FIELD,
    MAX_PROPOSALS_PER_FIELD, MatrixShape, ObservationMatrix, PrismionProposal, PrismionRule,
    ProposalField,
};
use alphasync_core::ids::{IdParseError, PrismionId, ProposalId};
use alphasync_core::progress::{
    CompletionDecision, CompletionGate, ContinuationDecision, ContinuationGate, EvidenceSet,
    EvidenceSlot, ProgressError, RunProgressCursor, WriteoutCadence, WriteoutDecision,
};

mod artifact_json;
mod artifact_writer;

use artifact_json::push_bounded_progress_report;
#[cfg(test)]
pub(crate) use artifact_json::{runtime_bundle_json, runtime_frame_json};
pub use artifact_writer::{
    ArtifactChecksum, RuntimeArtifactGeneration, RuntimeArtifactSnapshot, RuntimeArtifactWriter,
};

#[cfg(test)]
pub(crate) use artifact_writer::{
    DirectoryCleanupGuard, GenerationWriteLock, is_runtime_temp_file_name,
};

pub mod logic_iq;
pub mod synthetic;

/// Maximum progress rows embedded in `runtime_bundle.json`.
///
/// Public v0.1 artifacts intentionally avoid an unbounded append-only progress
/// log. Long runs refresh atomic snapshots/status files, while the GUI bundle
/// carries only a recent tail so local monitoring stays bounded.
pub const MAX_RUNTIME_BUNDLE_PROGRESS_ROWS: usize = 256;

/// Maximum committed generation directories retained after a successful
/// pointer commit.
///
/// The current generation is always protected. Older immutable generations are
/// local monitoring convenience, not an unbounded audit log, so v0.1 keeps a
/// short crash/debug window and prunes the rest while holding the writer lock.
pub const MAX_RUNTIME_GENERATION_RETAINED_DIRS: usize = 3;

/// Maximum bytes accepted when validating one runtime JSON artifact.
pub const MAX_RUNTIME_ARTIFACT_BYTES: u64 = 16 * 1024 * 1024;

const RUNTIME_ROOT_MARKER: &str = ".alphasync-runtime-root";
const RUNTIME_ROOT_MARKER_BODY: &str = "alphasync.runtime_root.v1\n";
const GENERATION_TMP_ROOT: &str = ".generation_tmp";
const GENERATION_PENDING_DIR: &str = "pending";
const GENERATION_LOCK_FILE: &str = ".generation_write.lock";

static DURABLE_WRITE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Runtime wiring error.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RuntimeError {
    /// The caller supplied a proposal-ID list whose length did not match the
    /// Prismion rule list.
    ProposalIdCountMismatch,
    /// Evaluation/proposal/matrix boundary failed inside `alphasync-core`.
    Fabric {
        /// Stable core fabric error.
        error: FabricError,
    },
    /// Completion/writeout/continuation boundary failed inside
    /// `alphasync-core`.
    Progress {
        /// Stable core progress error.
        error: ProgressError,
    },
    /// Typed identifier parsing failed inside runtime setup.
    IdParse {
        /// Stable identifier parse error.
        error: IdParseError,
    },
    /// Runtime policy used the same evidence slot for two different meanings.
    DuplicateEvidenceSlot,
    /// One cycle received more rules than the bounded proposal envelope allows.
    TooManyRulesPerCycle,
    /// One cycle repeated a Prismion source identity.
    DuplicatePrismionSource,
    /// One cycle repeated a caller proposal identity.
    DuplicateProposalId,
    /// A collection-only runtime cycle received a consensus policy that only
    /// the Agency path may consume.
    ConsensusPolicyRequiresAgencyPath,
    /// Runtime artifact path did not have a file name.
    InvalidArtifactPath,
    /// A published runtime generation is missing files, has extra files, or
    /// does not match its content-addressed checksum contract.
    CorruptArtifactGeneration,
    /// Another writer already owns the runtime generation commit lock.
    ArtifactWriterBusy,
    /// Runtime output directory was not empty and did not carry the `AlphaSync`
    /// ownership marker.
    UnownedArtifactDirectory,
    /// Runtime session configuration is outside the supported safe envelope.
    InvalidSessionConfig,
    /// Runtime artifact I/O failed.
    Io,
    /// Runtime artifact JSON serialization failed.
    ArtifactSerialization,
    /// A scoped scoring worker panicked before returning its safe aggregate.
    WorkerThreadPanicked,
}

impl RuntimeError {
    /// Returns the stable machine-readable runtime error code.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ProposalIdCountMismatch => "proposal_id_count_mismatch",
            Self::Fabric { .. } => "fabric_error",
            Self::Progress { .. } => "progress_error",
            Self::IdParse { .. } => "id_parse_error",
            Self::DuplicateEvidenceSlot => "duplicate_evidence_slot",
            Self::TooManyRulesPerCycle => "too_many_rules_per_cycle",
            Self::DuplicatePrismionSource => "duplicate_prismion_source",
            Self::DuplicateProposalId => "duplicate_proposal_id",
            Self::ConsensusPolicyRequiresAgencyPath => "consensus_policy_requires_agency_path",
            Self::InvalidArtifactPath => "invalid_artifact_path",
            Self::CorruptArtifactGeneration => "corrupt_artifact_generation",
            Self::ArtifactWriterBusy => "artifact_writer_busy",
            Self::UnownedArtifactDirectory => "unowned_artifact_directory",
            Self::InvalidSessionConfig => "invalid_session_config",
            Self::Io => "io_error",
            Self::ArtifactSerialization => "artifact_serialization",
            Self::WorkerThreadPanicked => "worker_thread_panicked",
        }
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fabric { error } => write!(formatter, "{}:{error}", self.as_str()),
            Self::Progress { error } => write!(formatter, "{}:{error}", self.as_str()),
            Self::IdParse { error } => write!(formatter, "{}:{error}", self.as_str()),
            _ => formatter.write_str(self.as_str()),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<FabricError> for RuntimeError {
    fn from(error: FabricError) -> Self {
        Self::Fabric { error }
    }
}

impl From<ProgressError> for RuntimeError {
    fn from(error: ProgressError) -> Self {
        Self::Progress { error }
    }
}

impl From<IdParseError> for RuntimeError {
    fn from(error: IdParseError) -> Self {
        Self::IdParse { error }
    }
}

impl From<std::io::Error> for RuntimeError {
    fn from(_error: std::io::Error) -> Self {
        Self::Io
    }
}

/// Synthetic runtime session kind.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RuntimeSessionKind {
    /// Repeated deterministic synthetic smoke iterations.
    SyntheticSmoke,
    /// Repeated deterministic synthetic scene cycles.
    SyntheticScene,
    /// Deterministic Logic-IQ-0 curriculum cycles.
    LogicIqZero,
    /// Deterministic Logic-IQ-0 multi-proposal consensus scene cycles.
    LogicIqConsensusScene,
}

impl RuntimeSessionKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::SyntheticSmoke => "synthetic_smoke",
            Self::SyntheticScene => "synthetic_scene",
            Self::LogicIqZero => "logic_iq_zero",
            Self::LogicIqConsensusScene => "logic_iq_consensus_scene",
        }
    }
}

/// Safe metadata for one repeated runtime session.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimeSessionManifest {
    session_kind: RuntimeSessionKind,
    cycles_requested: usize,
    cycles_completed: usize,
    writeout_count: usize,
}

impl RuntimeSessionManifest {
    /// Creates a bounded session manifest.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::InvalidSessionConfig`] when any count is zero or
    /// completed cycles exceed requested cycles.
    pub const fn new(
        session_kind: RuntimeSessionKind,
        cycles_requested: usize,
        cycles_completed: usize,
        writeout_count: usize,
    ) -> Result<Self, RuntimeError> {
        if cycles_requested == 0
            || cycles_completed == 0
            || writeout_count == 0
            || cycles_completed > cycles_requested
        {
            return Err(RuntimeError::InvalidSessionConfig);
        }

        Ok(Self {
            session_kind,
            cycles_requested,
            cycles_completed,
            writeout_count,
        })
    }

    /// Returns the fixed session kind.
    #[must_use]
    pub const fn session_kind(self) -> RuntimeSessionKind {
        self.session_kind
    }

    /// Returns requested cycle count.
    #[must_use]
    pub const fn cycles_requested(self) -> usize {
        self.cycles_requested
    }

    /// Returns completed cycle count.
    #[must_use]
    pub const fn cycles_completed(self) -> usize {
        self.cycles_completed
    }

    /// Returns durable writeout count.
    #[must_use]
    pub const fn writeout_count(self) -> usize {
        self.writeout_count
    }
}

/// Minimal runtime policy for one deterministic cycle.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimePolicy {
    proposal_shape: MatrixShape,
    flow_shape: MatrixShape,
    consensus_policy: Option<ConsensusPolicy>,
    writeout_cadence: WriteoutCadence,
    evaluation_slot: EvidenceSlot,
    proposal_slot: EvidenceSlot,
}

impl RuntimePolicy {
    /// Creates a runtime policy for the vertical slice.
    ///
    /// `evaluation_slot` means the rule scan completed. `proposal_slot` means
    /// at least one Prismion emitted a proposal. Keeping them distinct lets the
    /// completion gate distinguish "we ran but found nothing" from "we ran and
    /// produced a proposal".
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::DuplicateEvidenceSlot`] when both slots are the
    /// same.
    pub const fn new(
        proposal_shape: MatrixShape,
        writeout_cadence: WriteoutCadence,
        evaluation_slot: EvidenceSlot,
        proposal_slot: EvidenceSlot,
    ) -> Result<Self, RuntimeError> {
        Self::new_with_flow_shape(
            proposal_shape,
            proposal_shape,
            writeout_cadence,
            evaluation_slot,
            proposal_slot,
        )
    }

    /// Creates a runtime policy with distinct Proposal and Flow shapes.
    ///
    /// The Proposal Field is the temporary write surface for Prismion output.
    /// The Flow Field is the Agency-committed active state surface. They may
    /// have different sizes: the proposal surface can stay smaller while Flow
    /// keeps a larger persistent workspace.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::DuplicateEvidenceSlot`] when both slots are the
    /// same.
    pub const fn new_with_flow_shape(
        proposal_shape: MatrixShape,
        flow_shape: MatrixShape,
        writeout_cadence: WriteoutCadence,
        evaluation_slot: EvidenceSlot,
        proposal_slot: EvidenceSlot,
    ) -> Result<Self, RuntimeError> {
        if evaluation_slot.index() == proposal_slot.index() {
            return Err(RuntimeError::DuplicateEvidenceSlot);
        }

        Ok(Self {
            proposal_shape,
            flow_shape,
            consensus_policy: None,
            writeout_cadence,
            evaluation_slot,
            proposal_slot,
        })
    }

    /// Returns this policy with an Agency-sandwiched consensus gate enabled.
    ///
    /// This is crate-internal because the public proposal-collection runtime
    /// cannot execute consensus authority. Single-proposal public examples
    /// remain collection-only; consensus is used only by crate-owned Agency
    /// runners that attach Flow commits after arbitration.
    #[must_use]
    pub(crate) const fn with_consensus_policy(mut self, consensus_policy: ConsensusPolicy) -> Self {
        self.consensus_policy = Some(consensus_policy);
        self
    }

    /// Returns the active proposal-field shape.
    #[must_use]
    pub const fn proposal_shape(self) -> MatrixShape {
        self.proposal_shape
    }

    /// Returns the active Flow-field shape used by Agency admission.
    #[must_use]
    pub const fn flow_shape(self) -> MatrixShape {
        self.flow_shape
    }

    /// Returns the optional passive consensus policy.
    #[must_use]
    pub(crate) const fn consensus_policy(self) -> Option<ConsensusPolicy> {
        self.consensus_policy
    }

    /// Returns the runtime writeout cadence.
    #[must_use]
    pub const fn writeout_cadence(self) -> WriteoutCadence {
        self.writeout_cadence
    }

    /// Returns the evidence slot satisfied when rule evaluation completes.
    #[must_use]
    pub const fn evaluation_slot(self) -> EvidenceSlot {
        self.evaluation_slot
    }

    /// Returns the evidence slot satisfied when at least one proposal is
    /// emitted.
    #[must_use]
    pub const fn proposal_slot(self) -> EvidenceSlot {
        self.proposal_slot
    }

    fn required_evidence(self) -> EvidenceSet {
        EvidenceSet::single(self.evaluation_slot).with(self.proposal_slot)
    }
}

/// Aggregate report from one runtime cycle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeCycleReport {
    rules_evaluated: usize,
    rule_outcomes: Vec<RuleEvaluationOutcome>,
    proposals: Vec<PrismionProposal>,
    proposal_field: ProposalField,
    patch_count: usize,
    evidence_cell_count: usize,
    consensus_report: Option<RuntimeConsensusReport>,
    flow_commit: Option<RuntimeFlowCommit>,
    completion_decision: CompletionDecision,
    writeout_decision: WriteoutDecision,
    continuation_decision: ContinuationDecision,
}

impl RuntimeCycleReport {
    /// Returns the number of Prismion rules evaluated.
    #[must_use]
    pub const fn rules_evaluated(&self) -> usize {
        self.rules_evaluated
    }

    /// Returns per-rule activation outcomes in evaluation order.
    #[must_use]
    pub fn rule_outcomes(&self) -> &[RuleEvaluationOutcome] {
        &self.rule_outcomes
    }

    /// Returns emitted Prismion proposals.
    #[must_use]
    pub fn proposals(&self) -> &[PrismionProposal] {
        &self.proposals
    }

    /// Returns the materialized temporary Proposal Field for this cycle.
    ///
    /// The Proposal Field is a standard matrix view of the proposal patches,
    /// plus occupancy/collision counters. It is still not Flow/Ground truth.
    #[must_use]
    pub const fn proposal_field(&self) -> &ProposalField {
        &self.proposal_field
    }

    /// Returns the number of emitted proposals.
    #[must_use]
    pub fn proposal_count(&self) -> usize {
        self.proposals.len()
    }

    /// Returns the total emitted proposal patches.
    #[must_use]
    pub const fn patch_count(&self) -> usize {
        self.patch_count
    }

    /// Returns the total unique evidence-cell references inside emitted
    /// proposals.
    #[must_use]
    pub const fn evidence_cell_count(&self) -> usize {
        self.evidence_cell_count
    }

    /// Returns the optional passive consensus trace for this cycle.
    ///
    /// This is populated only when a caller runs the multi-proposal consensus
    /// path. It is a read-only decision trace, not a state update.
    #[must_use]
    pub const fn consensus_report(&self) -> Option<&RuntimeConsensusReport> {
        self.consensus_report.as_ref()
    }

    /// Returns the optional Agency-gated Flow commit record for this cycle.
    ///
    /// A Prismion proposal is not a state update. Reports without a committed
    /// Flow record remain blocked even when proposals exist; completion and
    /// quit authority are assigned only after Agency accepts a Flow commit.
    #[must_use]
    pub const fn flow_commit(&self) -> Option<RuntimeFlowCommit> {
        self.flow_commit
    }

    /// Returns this report with consensus and Flow Agency traces attached.
    ///
    /// This is crate-internal on purpose. Public consumers may inspect the
    /// resulting traces, but they cannot forge Agency state by attaching
    /// consensus/Flow records to a cycle report outside the runtime boundary.
    #[must_use]
    pub(crate) fn with_agency_reports(
        mut self,
        consensus_report: Option<RuntimeConsensusReport>,
        flow_commit: RuntimeFlowCommit,
    ) -> Self {
        self.completion_decision = if flow_commit.decision() == RuntimeFlowDecision::Commit {
            CompletionDecision::Complete
        } else {
            CompletionDecision::Blocked { blocker_count: 1 }
        };
        self.continuation_decision =
            ContinuationGate::quit_decision(self.completion_decision, self.writeout_decision, 0);
        self.consensus_report = consensus_report;
        self.flow_commit = Some(flow_commit);
        self
    }

    /// Returns the completion-gate decision for this cycle.
    #[must_use]
    pub const fn completion_decision(&self) -> CompletionDecision {
        self.completion_decision
    }

    /// Returns the writeout-cadence decision for this cycle.
    #[must_use]
    pub const fn writeout_decision(&self) -> WriteoutDecision {
        self.writeout_decision
    }

    /// Returns the anti-premature-quit decision for this cycle.
    #[must_use]
    pub const fn continuation_decision(&self) -> ContinuationDecision {
        self.continuation_decision
    }
}

/// Passive consensus trace from one multi-proposal runtime cycle.
///
/// The record stores only mechanical vote counts, ratio scores, and the
/// candidate patch selected for arbitration. It intentionally does not store
/// raw observations, semantic labels, or free-form explanations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeConsensusReport {
    action: ConsensusRecommendation,
    score_ppm: u32,
    conflict_ppm: u32,
    stale_ppm: u32,
    allowed_votes: u16,
    blocked_votes: u16,
    allowed_source_kinds: u8,
    allowed_independent_sources: u16,
    winner_independent_sources: u16,
    vote_count: usize,
    candidate_target: Option<CellAddress>,
    candidate_value: Option<u8>,
    vote_reports: Vec<RuntimeConsensusVoteReport>,
}

/// Source lane class for one consensus vote.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RuntimeConsensusSourceKind {
    /// Materialized Proposal Field candidate lane.
    ProposalField,
    /// Trace/evidence compatibility context lane.
    TraceContext,
    /// Ground, shape, target, and value compatibility context lane.
    GroundContext,
    /// Individual Prismion proposal lane.
    PrismionProposal,
}

/// Trace-safe diagnostic row for one consensus vote.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimeConsensusVoteReport {
    source_slot: u64,
    source_kind: RuntimeConsensusSourceKind,
    vote_kind: ConsensusVoteKind,
    admitted: bool,
    reputation_ppm: u32,
    confidence_ppm: u32,
    evidence_quality_ppm: u32,
    age_ticks: u16,
}

/// Mechanical channel in the runtime consensus-field matrix.
///
/// The consensus field is the standard matrix view of already Agency-ingressed
/// consensus votes. Rows are vote lanes; columns are fixed mechanical channels.
/// This keeps the field anonymous and numeric while still preserving enough
/// structure for downstream fabric nodes and the GUI to inspect the vote state.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RuntimeConsensusFieldChannel {
    /// Runtime source lane class as a compact fixed code.
    SourceKind,
    /// Support/reject vote direction as a compact fixed code.
    VoteKind,
    /// Agency ingress result for this vote.
    Admitted,
    /// Source reputation scaled from parts-per-million to `u8`.
    ReputationPpmU8,
    /// Source confidence scaled from parts-per-million to `u8`.
    ConfidencePpmU8,
    /// Evidence quality scaled from parts-per-million to `u8`.
    EvidenceQualityPpmU8,
    /// Vote age in ticks, saturated to `u8`.
    AgeTicks,
}

impl RuntimeConsensusFieldChannel {
    const CHANNELS: [Self; 7] = [
        Self::SourceKind,
        Self::VoteKind,
        Self::Admitted,
        Self::ReputationPpmU8,
        Self::ConfidencePpmU8,
        Self::EvidenceQualityPpmU8,
        Self::AgeTicks,
    ];

    const fn as_str(self) -> &'static str {
        match self {
            Self::SourceKind => "source_kind",
            Self::VoteKind => "vote_kind",
            Self::Admitted => "admitted",
            Self::ReputationPpmU8 => "reputation_ppm_u8",
            Self::ConfidencePpmU8 => "confidence_ppm_u8",
            Self::EvidenceQualityPpmU8 => "evidence_quality_ppm_u8",
            Self::AgeTicks => "age_ticks",
        }
    }

    const fn column(self) -> u16 {
        match self {
            Self::SourceKind => 0,
            Self::VoteKind => 1,
            Self::Admitted => 2,
            Self::ReputationPpmU8 => 3,
            Self::ConfidencePpmU8 => 4,
            Self::EvidenceQualityPpmU8 => 5,
            Self::AgeTicks => 6,
        }
    }
}

/// Materialized runtime consensus field matrix.
///
/// This is not a decision by itself. It is the passive fabric view of consensus
/// input lanes after Agency ingress checks. Flow/Ground state still changes
/// only through the final commit boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeConsensusField {
    shape: MatrixShape,
    non_zero_cells: Vec<CellSample>,
}

impl RuntimeConsensusField {
    /// Materializes a consensus-field matrix from one consensus report.
    ///
    /// Rows are vote lanes. Columns are [`RuntimeConsensusFieldChannel`]
    /// values. Zero channels are sparse-omitted in the same canonical style as
    /// other anonymous fabric matrices.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] if the vote count cannot be represented by the
    /// bounded matrix shape.
    pub fn from_report(report: &RuntimeConsensusReport) -> Result<Self, RuntimeError> {
        let rows = u16::try_from(report.vote_reports().len().max(1))
            .map_err(|_error| RuntimeError::InvalidSessionConfig)?;
        let columns = u16::try_from(RuntimeConsensusFieldChannel::CHANNELS.len())
            .map_err(|_error| RuntimeError::InvalidSessionConfig)?;
        let shape = MatrixShape::new(1, rows, columns)?;
        let mut non_zero_cells = Vec::new();

        for (row, vote) in report.vote_reports().iter().copied().enumerate() {
            let row = u16::try_from(row).map_err(|_error| RuntimeError::InvalidSessionConfig)?;
            push_consensus_sample(
                &mut non_zero_cells,
                row,
                RuntimeConsensusFieldChannel::SourceKind,
                consensus_source_kind_value(vote.source_kind()),
            )?;
            push_consensus_sample(
                &mut non_zero_cells,
                row,
                RuntimeConsensusFieldChannel::VoteKind,
                consensus_vote_kind_value(vote.vote_kind()),
            )?;
            push_consensus_sample(
                &mut non_zero_cells,
                row,
                RuntimeConsensusFieldChannel::Admitted,
                if vote.admitted() { 1 } else { 2 },
            )?;
            push_consensus_sample(
                &mut non_zero_cells,
                row,
                RuntimeConsensusFieldChannel::ReputationPpmU8,
                ppm_to_u8(vote.reputation_ppm()),
            )?;
            push_consensus_sample(
                &mut non_zero_cells,
                row,
                RuntimeConsensusFieldChannel::ConfidencePpmU8,
                ppm_to_u8(vote.confidence_ppm()),
            )?;
            push_consensus_sample(
                &mut non_zero_cells,
                row,
                RuntimeConsensusFieldChannel::EvidenceQualityPpmU8,
                ppm_to_u8(vote.evidence_quality_ppm()),
            )?;
            push_consensus_sample(
                &mut non_zero_cells,
                row,
                RuntimeConsensusFieldChannel::AgeTicks,
                u8::try_from(vote.age_ticks()).map_or(u8::MAX, |value| value),
            )?;
        }

        Ok(Self {
            shape,
            non_zero_cells,
        })
    }

    /// Returns the fixed consensus-field channel schema.
    #[must_use]
    pub const fn channel_schema() -> [RuntimeConsensusFieldChannel; 7] {
        RuntimeConsensusFieldChannel::CHANNELS
    }

    /// Returns the matrix shape.
    #[must_use]
    pub const fn shape(&self) -> MatrixShape {
        self.shape
    }

    /// Returns sparse non-zero consensus-field cells.
    #[must_use]
    pub fn non_zero_cells(&self) -> &[CellSample] {
        &self.non_zero_cells
    }
}

fn push_consensus_sample(
    samples: &mut Vec<CellSample>,
    row: u16,
    channel: RuntimeConsensusFieldChannel,
    value: u8,
) -> Result<(), RuntimeError> {
    if value == 0 {
        return Ok(());
    }

    samples.push(CellSample::new(
        CellAddress::new(0, row, channel.column()),
        value,
    )?);
    Ok(())
}

impl RuntimeConsensusVoteReport {
    /// Creates one trace-safe consensus vote report inside the runtime crate.
    ///
    /// Public consumers can inspect vote diagnostics but cannot forge vote
    /// provenance records for committed runtime artifacts.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub(crate) const fn new(
        source_slot: u64,
        source_kind: RuntimeConsensusSourceKind,
        vote_kind: ConsensusVoteKind,
        admitted: bool,
        reputation_ppm: u32,
        confidence_ppm: u32,
        evidence_quality_ppm: u32,
        age_ticks: u16,
    ) -> Self {
        Self {
            source_slot,
            source_kind,
            vote_kind,
            admitted,
            reputation_ppm,
            confidence_ppm,
            evidence_quality_ppm,
            age_ticks,
        }
    }

    /// Returns the consensus source slot.
    #[must_use]
    pub const fn source_slot(self) -> u64 {
        self.source_slot
    }

    /// Returns the source lane class.
    #[must_use]
    pub const fn source_kind(self) -> RuntimeConsensusSourceKind {
        self.source_kind
    }

    /// Returns the vote direction.
    #[must_use]
    pub const fn vote_kind(self) -> ConsensusVoteKind {
        self.vote_kind
    }

    /// Returns whether Agency ingress admitted this vote.
    #[must_use]
    pub const fn admitted(self) -> bool {
        self.admitted
    }

    /// Returns source reputation in parts per million.
    #[must_use]
    pub const fn reputation_ppm(self) -> u32 {
        self.reputation_ppm
    }

    /// Returns source confidence in parts per million.
    #[must_use]
    pub const fn confidence_ppm(self) -> u32 {
        self.confidence_ppm
    }

    /// Returns evidence quality in parts per million.
    #[must_use]
    pub const fn evidence_quality_ppm(self) -> u32 {
        self.evidence_quality_ppm
    }

    /// Returns vote age in consensus ticks.
    #[must_use]
    pub const fn age_ticks(self) -> u16 {
        self.age_ticks
    }
}

impl RuntimeConsensusReport {
    /// Creates a consensus trace from a core consensus decision.
    ///
    /// This constructor is crate-internal so public callers cannot fabricate a
    /// consensus trace that looks runtime-authored.
    #[must_use]
    pub(crate) fn new(
        decision: ConsensusDecision,
        vote_count: usize,
        candidate_target: Option<CellAddress>,
        candidate_value: Option<u8>,
        vote_reports: Vec<RuntimeConsensusVoteReport>,
    ) -> Self {
        Self {
            action: decision.action(),
            score_ppm: decision.score_ppm(),
            conflict_ppm: decision.conflict_ppm(),
            stale_ppm: decision.stale_ppm(),
            allowed_votes: decision.allowed_votes(),
            blocked_votes: decision.blocked_votes(),
            allowed_source_kinds: decision.allowed_source_kinds(),
            allowed_independent_sources: decision.allowed_independent_sources(),
            winner_independent_sources: decision.winner_independent_sources(),
            vote_count,
            candidate_target,
            candidate_value,
            vote_reports,
        }
    }

    /// Returns the Agency action emitted by consensus egress.
    #[must_use]
    pub const fn action(&self) -> ConsensusRecommendation {
        self.action
    }

    /// Returns absolute consensus score in parts per million.
    #[must_use]
    pub const fn score_ppm(&self) -> u32 {
        self.score_ppm
    }

    /// Returns conflict ratio in parts per million.
    #[must_use]
    pub const fn conflict_ppm(&self) -> u32 {
        self.conflict_ppm
    }

    /// Returns stale ratio in parts per million.
    #[must_use]
    pub const fn stale_ppm(&self) -> u32 {
        self.stale_ppm
    }

    /// Returns votes admitted by consensus ingress.
    #[must_use]
    pub const fn allowed_votes(&self) -> u16 {
        self.allowed_votes
    }

    /// Returns votes blocked by consensus ingress.
    #[must_use]
    pub const fn blocked_votes(&self) -> u16 {
        self.blocked_votes
    }

    /// Returns independent source families admitted by consensus ingress.
    #[must_use]
    pub const fn allowed_source_kinds(&self) -> u8 {
        self.allowed_source_kinds
    }

    /// Returns quorum-counted independent source lanes admitted by consensus.
    #[must_use]
    pub const fn allowed_independent_sources(&self) -> u16 {
        self.allowed_independent_sources
    }

    /// Returns independent source lanes on the winning consensus side.
    ///
    /// This is the quorum-authority value for a commit/reject decision. The
    /// allowed-independent count remains a diagnostic total across both sides.
    #[must_use]
    pub const fn winner_independent_sources(&self) -> u16 {
        self.winner_independent_sources
    }

    /// Returns total votes offered to consensus.
    #[must_use]
    pub const fn vote_count(&self) -> usize {
        self.vote_count
    }

    /// Returns the candidate target chosen for multi-proposal arbitration.
    #[must_use]
    pub const fn candidate_target(&self) -> Option<CellAddress> {
        self.candidate_target
    }

    /// Returns the candidate value chosen for multi-proposal arbitration.
    #[must_use]
    pub const fn candidate_value(&self) -> Option<u8> {
        self.candidate_value
    }

    /// Returns per-source consensus vote diagnostics.
    #[must_use]
    pub fn vote_reports(&self) -> &[RuntimeConsensusVoteReport] {
        &self.vote_reports
    }
}

/// Agency decision made after reading the temporary Proposal Field.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RuntimeFlowDecision {
    /// One proposal patch was admitted into the Flow view.
    Commit,
    /// No stable commit was available, but this is not a rejection.
    Defer,
    /// A proposal was unsafe or mechanically incompatible with the Flow view.
    Reject,
}

/// Stable reason code for an Agency Flow decision.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RuntimeFlowReason {
    /// The proposal passed the local mechanical admission checks.
    Accepted,
    /// No proposal was emitted in the cycle.
    NoProposal,
    /// More than one proposal tried to drive the same single-output cycle.
    ProposalCollision,
    /// A proposal emitted zero or multiple patches where one was required.
    PatchCollision,
    /// A patch target was outside the Flow matrix shape.
    PatchOutOfBounds,
    /// Proposal confidence was below the current runtime floor.
    LowConfidence,
    /// Proposal had no evidence cells attached.
    MissingEvidence,
    /// Patch targeted a cell that this task does not admit as output.
    InvalidTarget,
    /// Consensus rejected the candidate before Flow admission.
    ConsensusRejected,
    /// Consensus deferred the candidate before Flow admission.
    ConsensusDeferred,
    /// Consensus did not have enough independent Prismion source quorum.
    InsufficientConsensusSources,
    /// Consensus conflict pressure exceeded the policy cap.
    ConsensusConflict,
    /// Consensus stale pressure exceeded the policy cap.
    ConsensusStale,
}

/// Trace-safe Flow commit/admission record for one runtime cycle.
///
/// The record is deliberately mechanical. It stores only matrix coordinates,
/// counts, a stable reason code, and the admitted byte if a commit happened.
/// It does not store raw input text, semantic labels, oracle answers, or human
/// explanations. This makes it suitable for GUI inspection and restart
/// writeout without turning proposal traces into training labels.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimeFlowCommit {
    sequence: u64,
    flow_shape: MatrixShape,
    decision: RuntimeFlowDecision,
    reason: RuntimeFlowReason,
    target: Option<CellAddress>,
    value: Option<u8>,
    proposal_count: usize,
    patch_count: usize,
    evidence_cell_count: usize,
}

impl RuntimeFlowCommit {
    /// Creates an admitted Flow commit inside the runtime crate.
    ///
    /// Public callers may read Flow commit records from runtime artifacts, but
    /// cannot fabricate one through the release API.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when `target` is outside `flow_shape`.
    pub(crate) fn commit(
        sequence: u64,
        flow_shape: MatrixShape,
        target: CellAddress,
        value: u8,
        proposal_count: usize,
        patch_count: usize,
        evidence_cell_count: usize,
    ) -> Result<Self, RuntimeError> {
        if !flow_shape.contains(target) {
            return Err(RuntimeError::Fabric {
                error: FabricError::CellOutOfBounds,
            });
        }

        Ok(Self {
            sequence,
            flow_shape,
            decision: RuntimeFlowDecision::Commit,
            reason: RuntimeFlowReason::Accepted,
            target: Some(target),
            value: Some(value),
            proposal_count,
            patch_count,
            evidence_cell_count,
        })
    }

    /// Creates a non-committing Agency decision record inside the runtime crate.
    #[must_use]
    pub(crate) const fn no_commit(
        sequence: u64,
        flow_shape: MatrixShape,
        decision: RuntimeFlowDecision,
        reason: RuntimeFlowReason,
        proposal_count: usize,
        patch_count: usize,
        evidence_cell_count: usize,
    ) -> Self {
        Self {
            sequence,
            flow_shape,
            decision,
            reason,
            target: None,
            value: None,
            proposal_count,
            patch_count,
            evidence_cell_count,
        }
    }

    /// Returns the monotonic caller-supplied cycle sequence.
    #[must_use]
    pub const fn sequence(self) -> u64 {
        self.sequence
    }

    /// Returns the Flow matrix shape the decision was checked against.
    #[must_use]
    pub const fn flow_shape(self) -> MatrixShape {
        self.flow_shape
    }

    /// Returns the Agency decision.
    #[must_use]
    pub const fn decision(self) -> RuntimeFlowDecision {
        self.decision
    }

    /// Returns the stable decision reason.
    #[must_use]
    pub const fn reason(self) -> RuntimeFlowReason {
        self.reason
    }

    /// Returns the committed target cell, if a commit happened.
    #[must_use]
    pub const fn target(self) -> Option<CellAddress> {
        self.target
    }

    /// Returns the committed byte value, if a commit happened.
    #[must_use]
    pub const fn value(self) -> Option<u8> {
        self.value
    }

    /// Returns the proposal count seen by the Agency boundary.
    #[must_use]
    pub const fn proposal_count(self) -> usize {
        self.proposal_count
    }

    /// Returns the patch count seen by the Agency boundary.
    #[must_use]
    pub const fn patch_count(self) -> usize {
        self.patch_count
    }

    /// Returns the evidence-cell count seen by the Agency boundary.
    #[must_use]
    pub const fn evidence_cell_count(self) -> usize {
        self.evidence_cell_count
    }
}

/// Per-Prismion evaluation result from one runtime cycle.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuleEvaluationOutcome {
    source: PrismionId,
    activated: bool,
}

impl RuleEvaluationOutcome {
    /// Creates one per-rule evaluation outcome.
    #[must_use]
    pub fn new(source: PrismionId, activated: bool) -> Self {
        Self { source, activated }
    }

    /// Returns the evaluated Prismion source ID.
    #[must_use]
    pub fn source(&self) -> &PrismionId {
        &self.source
    }

    /// Returns whether this rule emitted a proposal in the cycle.
    #[must_use]
    pub const fn activated(&self) -> bool {
        self.activated
    }
}

/// Stateless minimal runtime executor.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AlphaSyncRuntime;

impl AlphaSyncRuntime {
    /// Runs one deterministic vertical-slice cycle.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the proposal-ID list does not match the
    /// rule list, a Prismion references an invalid cell/patch boundary, or the
    /// progress writeout cursor moves backward.
    pub fn run_cycle(
        policy: RuntimePolicy,
        matrix: &ObservationMatrix,
        rules: &[PrismionRule],
        proposal_ids: &[ProposalId],
        last_writeout: Option<RunProgressCursor>,
        current_progress: RunProgressCursor,
    ) -> Result<RuntimeCycleReport, RuntimeError> {
        if policy.consensus_policy().is_some() {
            return Err(RuntimeError::ConsensusPolicyRequiresAgencyPath);
        }
        Self::collect_proposals(
            policy,
            matrix,
            rules,
            proposal_ids,
            last_writeout,
            current_progress,
        )
    }

    fn collect_proposals(
        policy: RuntimePolicy,
        matrix: &ObservationMatrix,
        rules: &[PrismionRule],
        proposal_ids: &[ProposalId],
        last_writeout: Option<RunProgressCursor>,
        current_progress: RunProgressCursor,
    ) -> Result<RuntimeCycleReport, RuntimeError> {
        if rules.len() != proposal_ids.len() {
            return Err(RuntimeError::ProposalIdCountMismatch);
        }
        preflight_cycle_inputs(rules, proposal_ids)?;

        let mut completion_gate = CompletionGate::new(policy.required_evidence())?;
        let mut rule_outcomes = Vec::with_capacity(rules.len());
        let mut proposals = Vec::new();
        let mut patch_count = 0usize;
        let mut evidence_cell_count = 0usize;

        for (rule, proposal_id) in rules.iter().zip(proposal_ids) {
            let proposal = rule.evaluate(proposal_id.clone(), matrix, policy.proposal_shape())?;
            let activated = proposal.is_some();
            rule_outcomes.push(RuleEvaluationOutcome::new(rule.source().clone(), activated));
            if let Some(proposal) = proposal {
                patch_count = patch_count
                    .checked_add(proposal.patches().len())
                    .ok_or(FabricError::TooManyProposalPatches)?;
                if patch_count > MAX_PROPOSAL_PATCHES_PER_FIELD {
                    return Err(FabricError::TooManyProposalPatches.into());
                }
                evidence_cell_count = evidence_cell_count
                    .checked_add(proposal.evidence_cells().len())
                    .ok_or(FabricError::TooManyEvidenceCells)?;
                if evidence_cell_count > MAX_EVIDENCE_CELLS_PER_CYCLE {
                    return Err(FabricError::TooManyEvidenceCells.into());
                }
                proposals.push(proposal);
            }
        }

        completion_gate.satisfy(policy.evaluation_slot());
        if !proposals.is_empty() {
            completion_gate.satisfy(policy.proposal_slot());
        }
        let proposal_field = ProposalField::from_proposals(policy.proposal_shape(), &proposals)?;

        let _proposal_collection_ready =
            completion_gate.completion_decision(completion_gate.revision());
        let writeout_decision = policy
            .writeout_cadence()
            .writeout_decision(last_writeout, current_progress)?;
        let completion_decision = CompletionDecision::Blocked { blocker_count: 1 };
        let continuation_decision =
            ContinuationGate::quit_decision(completion_decision, writeout_decision, 0);

        Ok(RuntimeCycleReport {
            rules_evaluated: rules.len(),
            rule_outcomes,
            proposals,
            proposal_field,
            patch_count,
            evidence_cell_count,
            consensus_report: None,
            flow_commit: None,
            completion_decision,
            writeout_decision,
            continuation_decision,
        })
    }
}

fn preflight_cycle_inputs(
    rules: &[PrismionRule],
    proposal_ids: &[ProposalId],
) -> Result<(), RuntimeError> {
    if rules.len() > MAX_PROPOSALS_PER_FIELD {
        return Err(RuntimeError::TooManyRulesPerCycle);
    }

    let mut sources = BTreeSet::new();
    for rule in rules {
        if !sources.insert(rule.source().clone()) {
            return Err(RuntimeError::DuplicatePrismionSource);
        }
    }

    let mut proposal_id_set = BTreeSet::new();
    for proposal_id in proposal_ids {
        if !proposal_id_set.insert(proposal_id.clone()) {
            return Err(RuntimeError::DuplicateProposalId);
        }
    }

    Ok(())
}

fn completion_decision_code(decision: CompletionDecision) -> &'static str {
    match decision {
        CompletionDecision::Complete => "complete",
        CompletionDecision::Incomplete { .. } => "incomplete",
        CompletionDecision::Blocked { .. } => "blocked",
        CompletionDecision::StaleRevision { .. } => "stale",
    }
}

fn writeout_decision_code(decision: WriteoutDecision) -> &'static str {
    match decision {
        WriteoutDecision::Write { .. } => "write",
        WriteoutDecision::Wait => "wait",
    }
}

fn continuation_decision_code(decision: ContinuationDecision) -> &'static str {
    match decision {
        ContinuationDecision::Continue { .. } => "continue",
        ContinuationDecision::QuitAllowed => "quit_allowed",
    }
}

fn flow_decision_code(decision: RuntimeFlowDecision) -> &'static str {
    match decision {
        RuntimeFlowDecision::Commit => "commit",
        RuntimeFlowDecision::Defer => "defer",
        RuntimeFlowDecision::Reject => "reject",
    }
}

fn flow_reason_code(reason: RuntimeFlowReason) -> &'static str {
    match reason {
        RuntimeFlowReason::Accepted => "accepted",
        RuntimeFlowReason::NoProposal => "no_proposal",
        RuntimeFlowReason::ProposalCollision => "proposal_collision",
        RuntimeFlowReason::PatchCollision => "patch_collision",
        RuntimeFlowReason::PatchOutOfBounds => "patch_out_of_bounds",
        RuntimeFlowReason::LowConfidence => "low_confidence",
        RuntimeFlowReason::MissingEvidence => "missing_evidence",
        RuntimeFlowReason::InvalidTarget => "invalid_target",
        RuntimeFlowReason::ConsensusRejected => "consensus_rejected",
        RuntimeFlowReason::ConsensusDeferred => "consensus_deferred",
        RuntimeFlowReason::InsufficientConsensusSources => "insufficient_consensus_sources",
        RuntimeFlowReason::ConsensusConflict => "consensus_conflict",
        RuntimeFlowReason::ConsensusStale => "consensus_stale",
    }
}

fn agency_action_code(action: ConsensusRecommendation) -> &'static str {
    match action {
        ConsensusRecommendation::RecommendCommit => "commit",
        ConsensusRecommendation::RecommendReject => "reject",
        ConsensusRecommendation::RecommendDefer => "defer",
    }
}

fn consensus_source_kind_code(kind: RuntimeConsensusSourceKind) -> &'static str {
    match kind {
        RuntimeConsensusSourceKind::ProposalField => "proposal_field",
        RuntimeConsensusSourceKind::TraceContext => "trace_context",
        RuntimeConsensusSourceKind::GroundContext => "ground_context",
        RuntimeConsensusSourceKind::PrismionProposal => "prismion_proposal",
    }
}

fn consensus_vote_kind_code(kind: ConsensusVoteKind) -> &'static str {
    match kind {
        ConsensusVoteKind::Support => "support",
        ConsensusVoteKind::Reject => "reject",
    }
}

const fn consensus_source_kind_value(kind: RuntimeConsensusSourceKind) -> u8 {
    match kind {
        RuntimeConsensusSourceKind::ProposalField => 1,
        RuntimeConsensusSourceKind::TraceContext => 2,
        RuntimeConsensusSourceKind::GroundContext => 3,
        RuntimeConsensusSourceKind::PrismionProposal => 4,
    }
}

const fn consensus_vote_kind_value(kind: ConsensusVoteKind) -> u8 {
    match kind {
        ConsensusVoteKind::Support => 1,
        ConsensusVoteKind::Reject => 2,
    }
}

fn ppm_to_u8(value: u32) -> u8 {
    let scaled = (u128::from(value).saturating_mul(u128::from(u8::MAX)) + 500_000) / 1_000_000;
    u8::try_from(scaled).map_or(u8::MAX, |value| value)
}

pub(crate) const fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut index = 0usize;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        index += 1;
    }
    hash
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use alphasync_core::fabric::{
        CellAddress, CellComparison, CellPredicate, CellSample, ConfidencePpm, ConsensusPolicy,
        ConsensusSourceKind, ConsensusVote, ConsensusVoteKind, MatrixShape, ObservationMatrix,
        PrismionRule, ProposalPatch,
    };
    use alphasync_core::ids::{PrismionId, ProposalId};
    use alphasync_core::progress::{
        CompletionDecision, ContinuationDecision, ContinuationReason, EvidenceSlot,
        RunProgressCursor, WriteoutCadence, WriteoutDecision, WriteoutReason,
    };
    use sha2::{Digest, Sha256};

    use super::{
        AlphaSyncRuntime, DirectoryCleanupGuard, GenerationWriteLock,
        MAX_RUNTIME_BUNDLE_PROGRESS_ROWS, RuntimeArtifactSnapshot, RuntimeArtifactWriter,
        RuntimeConsensusField, RuntimeConsensusReport, RuntimeConsensusSourceKind,
        RuntimeConsensusVoteReport, RuntimeCycleReport, RuntimeError, RuntimeFlowCommit,
        RuntimePolicy, RuntimeSessionKind, RuntimeSessionManifest, runtime_bundle_json,
        runtime_frame_json,
    };

    static CURRENT_DIR_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    const PRISMION_HEX: &str = "0123456789abcdef0123456789abcdef";
    const PROPOSAL_HEX: &str = "fedcba9876543210fedcba9876543210";

    fn prismion_id() -> PrismionId {
        PrismionId::parse(PRISMION_HEX).expect("valid Prismion ID")
    }

    fn proposal_id() -> ProposalId {
        ProposalId::parse(PROPOSAL_HEX).expect("valid proposal ID")
    }

    fn legacy_progress_name() -> String {
        ["progress", concat!("json", "l")].join(".")
    }

    fn sha256_hex(payload: &str) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";

        let mut sha256 = Sha256::new();
        sha256.update(payload.as_bytes());
        let digest = sha256.finalize();
        let bytes: &[u8] = digest.as_ref();
        let mut encoded = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            encoded.push(char::from(HEX[usize::from(*byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(*byte & 0x0f)]));
        }
        encoded
    }

    fn checksum_record(name: &str, payload: &str) -> serde_json::Value {
        serde_json::json!({
            "bytes": payload.len(),
            "fnv1a64": format!("{:016x}", super::fnv1a64(payload.as_bytes())),
            "name": name,
            "sha256": sha256_hex(payload),
        })
    }

    fn copy_directory_tree(source: &std::path::Path, target: &std::path::Path) {
        std::fs::create_dir_all(target).expect("target directory created");
        for entry in std::fs::read_dir(source).expect("source directory readable") {
            let entry = entry.expect("source entry readable");
            let source_path = entry.path();
            let target_path = target.join(entry.file_name());
            let file_type = entry.file_type().expect("source file type readable");
            if file_type.is_dir() {
                copy_directory_tree(&source_path, &target_path);
            } else {
                std::fs::copy(&source_path, &target_path).expect("source file copied");
            }
        }
    }

    const fn runtime_consensus_source_kind(
        source_kind: ConsensusSourceKind,
    ) -> RuntimeConsensusSourceKind {
        match source_kind {
            ConsensusSourceKind::ProposalField => RuntimeConsensusSourceKind::ProposalField,
            ConsensusSourceKind::TraceContext => RuntimeConsensusSourceKind::TraceContext,
            ConsensusSourceKind::GroundContext => RuntimeConsensusSourceKind::GroundContext,
            ConsensusSourceKind::PrismionProposal => RuntimeConsensusSourceKind::PrismionProposal,
        }
    }

    fn address(plane: u16, row: u16, column: u16) -> CellAddress {
        CellAddress::new(plane, row, column)
    }

    fn policy() -> RuntimePolicy {
        RuntimePolicy::new(
            MatrixShape::new(1, 2, 2).expect("valid proposal shape"),
            WriteoutCadence::new(20_000, 10, 10).expect("valid cadence"),
            EvidenceSlot::new(0).expect("valid evidence slot"),
            EvidenceSlot::new(1).expect("valid evidence slot"),
        )
        .expect("valid runtime policy")
    }

    fn current_generation_dir(run_dir: &std::path::Path) -> std::path::PathBuf {
        let pointer = std::fs::read_to_string(run_dir.join("current_generation.json"))
            .expect("current pointer readable");
        let marker = "\"generation_dir\": \"";
        let start = pointer.find(marker).expect("generation dir exists") + marker.len();
        let end = pointer[start..]
            .find('"')
            .expect("generation dir value ends")
            + start;
        run_dir.join(&pointer[start..end])
    }

    struct CurrentDirGuard {
        previous: PathBuf,
    }

    impl CurrentDirGuard {
        fn enter(path: &std::path::Path) -> Self {
            let previous = std::env::current_dir().expect("current dir readable");
            std::env::set_current_dir(path).expect("current dir changed");
            Self { previous }
        }
    }

    impl Drop for CurrentDirGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.previous);
        }
    }

    #[cfg(windows)]
    fn create_test_dir_symlink(
        target: &std::path::Path,
        link: &std::path::Path,
    ) -> std::io::Result<()> {
        std::os::windows::fs::symlink_dir(target, link)
    }

    #[cfg(unix)]
    fn create_test_dir_symlink(
        target: &std::path::Path,
        link: &std::path::Path,
    ) -> std::io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    fn artifact_fixture() -> (
        ObservationMatrix,
        Vec<PrismionRule>,
        RuntimeCycleReport,
        Vec<RuntimeCycleReport>,
        RuntimeSessionManifest,
    ) {
        let shape = MatrixShape::new(1, 2, 2).expect("valid matrix shape");
        let matrix = ObservationMatrix::from_sparse(
            shape,
            vec![CellSample::new(address(0, 0, 0), 7).expect("non-zero sample")],
        )
        .expect("valid sparse matrix");
        let rule = PrismionRule::new(
            prismion_id(),
            vec![CellPredicate::new(address(0, 0, 0), CellComparison::Gte, 5)],
            vec![ProposalPatch::new(address(0, 1, 1), 1)],
            ConfidencePpm::new(900_000).expect("valid confidence"),
        )
        .expect("valid Prismion rule");
        let rules = vec![rule];
        let report = AlphaSyncRuntime::run_cycle(
            policy(),
            &matrix,
            &rules,
            &[proposal_id()],
            Some(RunProgressCursor::new(0, 0, 0)),
            RunProgressCursor::new(1, 1, 1),
        )
        .expect("runtime cycle succeeds");
        let progress_reports = vec![report.clone()];
        let session = RuntimeSessionManifest::new(RuntimeSessionKind::SyntheticScene, 1, 1, 1)
            .expect("valid session manifest");
        (matrix, rules, report, progress_reports, session)
    }

    #[test]
    fn runtime_vertical_slice_emits_proposal_and_blocks_quit_until_writeout() {
        let shape = MatrixShape::new(1, 2, 2).expect("valid matrix shape");
        let matrix = ObservationMatrix::from_sparse(
            shape,
            vec![CellSample::new(address(0, 0, 0), 7).expect("non-zero sample")],
        )
        .expect("valid sparse matrix");
        let rule = PrismionRule::new(
            prismion_id(),
            vec![CellPredicate::new(address(0, 0, 0), CellComparison::Gte, 5)],
            vec![ProposalPatch::new(address(0, 1, 1), 1)],
            ConfidencePpm::new(900_000).expect("valid confidence"),
        )
        .expect("valid Prismion rule");

        let report = AlphaSyncRuntime::run_cycle(
            policy(),
            &matrix,
            &[rule],
            &[proposal_id()],
            None,
            RunProgressCursor::new(0, 1, 1),
        )
        .expect("runtime cycle succeeds");

        assert_eq!(report.rules_evaluated(), 1);
        assert_eq!(report.proposal_count(), 1);
        assert_eq!(report.patch_count(), 1);
        assert_eq!(report.evidence_cell_count(), 1);
        assert_eq!(report.proposal_field().patch_count(), 1);
        assert_eq!(report.proposal_field().occupied_cell_count(), 1);
        assert_eq!(report.proposal_field().collision_cell_count(), 0);
        assert_eq!(
            report
                .proposal_field()
                .get(address(0, 1, 1))
                .expect("proposal field address is in bounds"),
            Some(1)
        );
        assert_eq!(
            report.completion_decision(),
            CompletionDecision::Blocked { blocker_count: 1 }
        );
        assert_eq!(
            report.writeout_decision(),
            WriteoutDecision::Write {
                reason: WriteoutReason::Initial
            }
        );
        assert_eq!(
            report.continuation_decision(),
            ContinuationDecision::Continue {
                reason: ContinuationReason::PartialWriteoutDue {
                    reason: WriteoutReason::Initial
                }
            }
        );
    }

    #[test]
    fn runtime_reports_incomplete_when_scan_finds_no_proposal() {
        let shape = MatrixShape::new(1, 2, 2).expect("valid matrix shape");
        let matrix = ObservationMatrix::zeroed(shape);
        let rule = PrismionRule::new(
            prismion_id(),
            vec![CellPredicate::new(address(0, 0, 0), CellComparison::Gte, 5)],
            vec![ProposalPatch::new(address(0, 1, 1), 1)],
            ConfidencePpm::new(900_000).expect("valid confidence"),
        )
        .expect("valid Prismion rule");

        let report = AlphaSyncRuntime::run_cycle(
            policy(),
            &matrix,
            &[rule],
            &[proposal_id()],
            Some(RunProgressCursor::new(0, 0, 0)),
            RunProgressCursor::new(1, 1, 1),
        )
        .expect("runtime cycle succeeds");

        assert_eq!(report.rules_evaluated(), 1);
        assert_eq!(report.proposal_count(), 0);
        assert_eq!(
            report.completion_decision(),
            CompletionDecision::Blocked { blocker_count: 1 }
        );
        assert_eq!(report.writeout_decision(), WriteoutDecision::Wait);
        assert_eq!(
            report.continuation_decision(),
            ContinuationDecision::Continue {
                reason: ContinuationReason::CompletionBlocked { blocker_count: 1 }
            }
        );
    }

    #[test]
    fn runtime_rejects_mismatched_proposal_ids() {
        let shape = MatrixShape::new(1, 1, 1).expect("valid matrix shape");
        let matrix = ObservationMatrix::zeroed(shape);

        assert_eq!(
            AlphaSyncRuntime::run_cycle(
                policy(),
                &matrix,
                &[],
                &[proposal_id()],
                None,
                RunProgressCursor::new(0, 0, 0),
            )
            .expect_err("mismatched proposal IDs rejected"),
            RuntimeError::ProposalIdCountMismatch
        );
    }

    #[test]
    fn runtime_rejects_duplicate_prismion_sources() {
        let shape = MatrixShape::new(1, 1, 1).expect("valid matrix shape");
        let matrix = ObservationMatrix::zeroed(shape);
        let rule = PrismionRule::new(
            prismion_id(),
            vec![CellPredicate::new(address(0, 0, 0), CellComparison::Eq, 0)],
            vec![ProposalPatch::new(address(0, 0, 0), 1)],
            ConfidencePpm::new(900_000).expect("valid confidence"),
        )
        .expect("valid Prismion rule");

        assert_eq!(
            AlphaSyncRuntime::run_cycle(
                policy(),
                &matrix,
                &[rule.clone(), rule],
                &[
                    ProposalId::parse("00000000000000000000000000000001")
                        .expect("valid proposal ID"),
                    ProposalId::parse("00000000000000000000000000000002")
                        .expect("valid proposal ID"),
                ],
                None,
                RunProgressCursor::new(0, 0, 0),
            )
            .expect_err("duplicate Prismion sources rejected"),
            RuntimeError::DuplicatePrismionSource
        );
    }

    #[test]
    fn runtime_rejects_duplicate_proposal_ids() {
        let shape = MatrixShape::new(1, 1, 1).expect("valid matrix shape");
        let matrix = ObservationMatrix::zeroed(shape);
        let rule_a = PrismionRule::new(
            prismion_id(),
            vec![CellPredicate::new(address(0, 0, 0), CellComparison::Eq, 0)],
            vec![ProposalPatch::new(address(0, 0, 0), 1)],
            ConfidencePpm::new(900_000).expect("valid confidence"),
        )
        .expect("valid Prismion rule");
        let rule_b = PrismionRule::new(
            PrismionId::parse("00000000000000000000000000000002").expect("valid Prismion ID"),
            vec![CellPredicate::new(address(0, 0, 0), CellComparison::Eq, 0)],
            vec![ProposalPatch::new(address(0, 0, 1), 1)],
            ConfidencePpm::new(900_000).expect("valid confidence"),
        )
        .expect("valid Prismion rule before proposal-shape binding");

        assert_eq!(
            AlphaSyncRuntime::run_cycle(
                policy(),
                &matrix,
                &[rule_a, rule_b],
                &[proposal_id(), proposal_id()],
                None,
                RunProgressCursor::new(0, 0, 0),
            )
            .expect_err("duplicate proposal IDs rejected"),
            RuntimeError::DuplicateProposalId
        );
    }

    #[test]
    fn runtime_policy_rejects_duplicate_evidence_slots() {
        assert_eq!(
            RuntimePolicy::new(
                MatrixShape::new(1, 1, 1).expect("valid proposal shape"),
                WriteoutCadence::frontier_default(),
                EvidenceSlot::new(0).expect("valid evidence slot"),
                EvidenceSlot::new(0).expect("valid evidence slot"),
            )
            .expect_err("duplicate evidence slots rejected"),
            RuntimeError::DuplicateEvidenceSlot
        );
    }

    #[test]
    fn collection_runtime_rejects_ignored_consensus_policy() {
        let shape = MatrixShape::new(1, 2, 2).expect("valid matrix shape");
        let matrix = ObservationMatrix::zeroed(shape);
        let policy = policy().with_consensus_policy(
            ConsensusPolicy::e150_best_safe().expect("valid consensus policy"),
        );

        assert_eq!(
            AlphaSyncRuntime::run_cycle(
                policy,
                &matrix,
                &[],
                &[],
                None,
                RunProgressCursor::new(0, 0, 0),
            )
            .expect_err("collection-only path rejects ignored consensus policy"),
            RuntimeError::ConsensusPolicyRequiresAgencyPath
        );
    }

    #[test]
    fn runtime_consensus_report_materializes_standard_sparse_field() {
        let policy = ConsensusPolicy::e150_best_safe().expect("valid consensus policy");
        let reputation = ConfidencePpm::new(900_000).expect("valid reputation");
        let confidence = ConfidencePpm::new(920_000).expect("valid confidence");
        let quality = ConfidencePpm::new(940_000).expect("valid quality");
        let votes = [
            ConsensusVote::proposal_field(
                ConsensusVoteKind::Support,
                reputation,
                confidence,
                quality,
                0,
            ),
            ConsensusVote::trace_context(
                ConsensusVoteKind::Support,
                reputation,
                confidence,
                quality,
                0,
            ),
        ];
        let decision = policy.decide(&votes).expect("votes are valid");
        let reports = votes
            .iter()
            .copied()
            .map(|vote| {
                RuntimeConsensusVoteReport::new(
                    vote.source_slot(),
                    runtime_consensus_source_kind(vote.source_kind()),
                    vote.kind(),
                    policy.admits_vote(vote),
                    vote.reputation().as_ppm(),
                    vote.confidence().as_ppm(),
                    vote.evidence_quality().as_ppm(),
                    vote.age_ticks(),
                )
            })
            .collect();
        let consensus = RuntimeConsensusReport::new(
            decision,
            votes.len(),
            Some(address(0, 0, 0)),
            Some(1),
            reports,
        );

        let field = RuntimeConsensusField::from_report(&consensus).expect("field materializes");

        assert_eq!(
            field.shape(),
            MatrixShape::new(1, 2, 7).expect("valid shape")
        );
        assert_eq!(field.non_zero_cells().len(), 12);
        assert!(
            field
                .non_zero_cells()
                .contains(&CellSample::new(address(0, 0, 0), 1).expect("source kind cell"))
        );
        assert!(
            field
                .non_zero_cells()
                .contains(&CellSample::new(address(0, 0, 1), 1).expect("vote kind cell"))
        );
        assert!(
            field
                .non_zero_cells()
                .contains(&CellSample::new(address(0, 0, 2), 1).expect("admitted cell"))
        );
        assert!(
            field
                .non_zero_cells()
                .contains(&CellSample::new(address(0, 1, 0), 2).expect("source kind cell"))
        );
    }

    #[test]
    fn consensus_runtime_artifacts_are_valid_json() {
        let shape = MatrixShape::new(1, 2, 2).expect("valid matrix shape");
        let matrix = ObservationMatrix::from_sparse(
            shape,
            vec![CellSample::new(address(0, 0, 0), 7).expect("non-zero sample")],
        )
        .expect("valid sparse matrix");
        let rule = PrismionRule::new(
            prismion_id(),
            vec![CellPredicate::new(address(0, 0, 0), CellComparison::Gte, 5)],
            vec![ProposalPatch::new(address(0, 1, 1), 1)],
            ConfidencePpm::new(900_000).expect("valid confidence"),
        )
        .expect("valid Prismion rule");
        let report = AlphaSyncRuntime::collect_proposals(
            policy(),
            &matrix,
            core::slice::from_ref(&rule),
            &[proposal_id()],
            None,
            RunProgressCursor::new(0, 1, 1),
        )
        .expect("proposal collection succeeds");
        let consensus_policy = ConsensusPolicy::e150_best_safe().expect("valid policy");
        let votes = [
            ConsensusVote::from_prismion_proposal(
                report.proposals().first().expect("proposal exists"),
                ConsensusVoteKind::Support,
                ConfidencePpm::new(900_000).expect("valid reputation"),
                ConfidencePpm::new(900_000).expect("valid quality"),
                0,
            ),
            ConsensusVote::ground_context(
                ConsensusVoteKind::Support,
                ConfidencePpm::new(900_000).expect("valid reputation"),
                ConfidencePpm::new(900_000).expect("valid confidence"),
                ConfidencePpm::new(900_000).expect("valid quality"),
                0,
            ),
        ];
        let decision = consensus_policy
            .decide(&votes)
            .expect("consensus decision succeeds");
        let vote_reports = votes
            .iter()
            .copied()
            .map(|vote| {
                RuntimeConsensusVoteReport::new(
                    vote.source_slot(),
                    runtime_consensus_source_kind(vote.source_kind()),
                    vote.kind(),
                    consensus_policy.admits_vote(vote),
                    vote.reputation().as_ppm(),
                    vote.confidence().as_ppm(),
                    vote.evidence_quality().as_ppm(),
                    vote.age_ticks(),
                )
            })
            .collect();
        let consensus = RuntimeConsensusReport::new(
            decision,
            votes.len(),
            Some(address(0, 1, 1)),
            Some(1),
            vote_reports,
        );
        let flow_commit = RuntimeFlowCommit::commit(
            1,
            shape,
            address(0, 1, 1),
            1,
            report.proposal_count(),
            report.patch_count(),
            report.evidence_cell_count(),
        )
        .expect("flow commit shape is valid");
        let report = report.with_agency_reports(Some(consensus), flow_commit);

        let frame = runtime_frame_json(&matrix, core::slice::from_ref(&rule), &report)
            .expect("frame writes");
        serde_json::from_str::<serde_json::Value>(&frame).expect("runtime frame is valid JSON");

        let bundle = runtime_bundle_json(&report, &[], &[], None).expect("bundle writes");
        serde_json::from_str::<serde_json::Value>(&bundle).expect("runtime bundle is valid JSON");
    }

    #[test]
    fn directory_cleanup_guard_drop_does_not_delete_staging_directory() {
        let run_dir = std::env::temp_dir().join(format!(
            "alphasync_drop_cleanup_guard_{}",
            std::process::id()
        ));
        if run_dir.exists() {
            std::fs::remove_dir_all(&run_dir).expect("old test directory removed");
        }
        let stage_dir = run_dir.join(".generation_tmp");
        std::fs::create_dir_all(&stage_dir).expect("stage directory created");
        std::fs::write(stage_dir.join("runtime_bundle.json"), "{}").expect("stage payload written");

        drop(DirectoryCleanupGuard::new(stage_dir.clone()));

        assert!(stage_dir.is_dir());
        assert!(stage_dir.join("runtime_bundle.json").is_file());
        std::fs::remove_dir_all(run_dir).expect("test directory removed");
    }

    #[test]
    fn runtime_writer_anchors_relative_root_after_current_dir_change() {
        let _lock = CURRENT_DIR_TEST_LOCK.lock().expect("current-dir test lock");
        let parent = std::env::temp_dir().join(format!(
            "alphasync_relative_root_parent_{}",
            std::process::id()
        ));
        let other = std::env::temp_dir().join(format!(
            "alphasync_relative_root_other_{}",
            std::process::id()
        ));
        if parent.exists() {
            std::fs::remove_dir_all(&parent).expect("old parent removed");
        }
        if other.exists() {
            std::fs::remove_dir_all(&other).expect("old other removed");
        }
        std::fs::create_dir_all(&parent).expect("parent created");
        std::fs::create_dir_all(&other).expect("other created");

        let guard = CurrentDirGuard::enter(&parent);
        let writer =
            RuntimeArtifactWriter::open_or_create("relative_runtime").expect("writer opened");
        std::env::set_current_dir(&other).expect("current dir moved after writer open");

        let (matrix, rules, report, progress_reports, session) = artifact_fixture();
        let snapshot = RuntimeArtifactSnapshot::new(
            &matrix,
            &rules,
            &report,
            &progress_reports,
            Some(&session),
        )
        .expect("snapshot valid");
        writer
            .write_runtime_generation(&snapshot)
            .expect("generation written through anchored writer");

        assert!(
            parent
                .join("relative_runtime")
                .join("current_generation.json")
                .is_file()
        );
        assert!(!other.join("relative_runtime").exists());
        drop(guard);
        std::fs::remove_dir_all(parent).expect("parent removed");
        std::fs::remove_dir_all(other).expect("other removed");
    }

    #[test]
    fn runtime_writer_rejects_symlink_ancestor_before_creating_child() {
        let base = std::env::temp_dir().join(format!(
            "alphasync_symlink_ancestor_test_{}",
            std::process::id()
        ));
        if base.exists() {
            std::fs::remove_dir_all(&base).expect("old test directory removed");
        }
        let target = base.join("target");
        let link = base.join("link");
        std::fs::create_dir_all(&target).expect("target directory created");

        if create_test_dir_symlink(&target, &link).is_err() {
            std::fs::remove_dir_all(base).expect("test directory removed");
            return;
        }

        let child = link.join("child_runtime");
        let error = RuntimeArtifactWriter::open_or_create(&child)
            .expect_err("symlink ancestor must be rejected");
        assert!(matches!(error, RuntimeError::InvalidArtifactPath));
        assert!(!target.join("child_runtime").exists());

        std::fs::remove_dir(&link)
            .or_else(|_| std::fs::remove_file(&link))
            .expect("symlink removed");
        std::fs::remove_dir_all(base).expect("test directory removed");
    }

    #[test]
    fn runtime_temp_file_match_is_exact_and_case_sensitive() {
        assert!(super::is_runtime_temp_file_name(
            "runtime_bundle.json.123.456.tmp"
        ));
        assert!(!super::is_runtime_temp_file_name("runtime_bundle.json.tmp"));
        assert!(!super::is_runtime_temp_file_name(
            "runtime_bundle.json.bad.tmp"
        ));
        assert!(!super::is_runtime_temp_file_name(
            "runtime_bundle.json.123.456.TMP"
        ));
        assert!(!super::is_runtime_temp_file_name(
            "runtime_bundle.json.123.456.tmp.extra"
        ));
    }

    #[test]
    fn corrupt_current_pointer_rejects_next_generation_without_false_success() {
        let run_dir = std::env::temp_dir().join(format!(
            "alphasync_corrupt_pointer_test_{}",
            std::process::id()
        ));
        if run_dir.exists() {
            std::fs::remove_dir_all(&run_dir).expect("old test directory removed");
        }
        let (matrix, rules, report, progress_reports, session) = artifact_fixture();
        let writer =
            RuntimeArtifactWriter::open_or_create(&run_dir).expect("runtime writer opened");
        let snapshot = RuntimeArtifactSnapshot::new(
            &matrix,
            &rules,
            &report,
            &progress_reports,
            Some(&session),
        )
        .expect("snapshot valid");
        writer
            .write_runtime_generation(&snapshot)
            .expect("initial generation written");
        std::fs::write(run_dir.join("current_generation.json"), "{}")
            .expect("corrupt pointer written");

        let error = writer
            .write_runtime_generation(&snapshot)
            .expect_err("corrupt pointer rejected");
        assert!(matches!(error, RuntimeError::CorruptArtifactGeneration));
        std::fs::remove_dir_all(run_dir).expect("test directory removed");
    }

    #[test]
    fn fake_hash_generation_dir_blocks_retention_and_is_not_deleted() {
        let run_dir = std::env::temp_dir().join(format!(
            "alphasync_fake_generation_test_{}",
            std::process::id()
        ));
        if run_dir.exists() {
            std::fs::remove_dir_all(&run_dir).expect("old test directory removed");
        }
        let (matrix, rules, report, progress_reports, session) = artifact_fixture();
        let writer =
            RuntimeArtifactWriter::open_or_create(&run_dir).expect("runtime writer opened");
        let snapshot = RuntimeArtifactSnapshot::new(
            &matrix,
            &rules,
            &report,
            &progress_reports,
            Some(&session),
        )
        .expect("snapshot valid");
        writer
            .write_runtime_generation(&snapshot)
            .expect("initial generation written");

        let fake_name = format!("generation_{}", "a".repeat(64));
        let fake_dir = run_dir.join("generations").join(&fake_name);
        std::fs::create_dir_all(&fake_dir).expect("fake generation created");
        std::fs::write(fake_dir.join("runtime_generation.json"), "{}")
            .expect("fake generation marker written");

        let error = writer
            .write_runtime_generation(&snapshot)
            .expect_err("fake generation rejected");
        assert!(matches!(error, RuntimeError::CorruptArtifactGeneration));
        assert!(fake_dir.is_dir());
        std::fs::remove_dir_all(run_dir).expect("test directory removed");
    }

    #[test]
    fn semantically_invalid_generation_is_rejected_even_when_manifest_hash_matches() {
        let run_dir = std::env::temp_dir().join(format!(
            "alphasync_semantic_generation_test_{}",
            std::process::id()
        ));
        if run_dir.exists() {
            std::fs::remove_dir_all(&run_dir).expect("old test directory removed");
        }
        let (matrix, rules, report, progress_reports, session) = artifact_fixture();
        let writer =
            RuntimeArtifactWriter::open_or_create(&run_dir).expect("runtime writer opened");
        let snapshot = RuntimeArtifactSnapshot::new(
            &matrix,
            &rules,
            &report,
            &progress_reports,
            Some(&session),
        )
        .expect("snapshot valid");
        writer
            .write_runtime_generation(&snapshot)
            .expect("initial generation written");

        let bad_payload = "not-json\n";
        let artifact_names = [
            "runtime_summary.json",
            "progress_snapshot.json",
            "runtime_frame.json",
            "artifact_checksums.json",
            "runtime_bundle.json",
        ];
        let artifacts = artifact_names
            .iter()
            .map(|name| checksum_record(name, bad_payload))
            .collect::<Vec<_>>();
        let runtime_bundle_sha = sha256_hex(bad_payload);
        let manifest = serde_json::json!({
            "artifacts": artifacts,
            "commit_protocol": "immutable_generation_directory_v1",
            "generation_id": runtime_bundle_sha,
            "schema_version": "alphasync.runtime_generation.v1",
        });
        let manifest_payload = format!(
            "{}\n",
            serde_json::to_string_pretty(&manifest).expect("manifest serialized")
        );
        let fake_name = format!("generation_{}", sha256_hex(&manifest_payload));
        let fake_dir = run_dir.join("generations").join(&fake_name);
        std::fs::create_dir_all(&fake_dir).expect("fake generation created");
        for name in artifact_names {
            std::fs::write(fake_dir.join(name), bad_payload).expect("bad artifact written");
        }
        std::fs::write(fake_dir.join("runtime_generation.json"), manifest_payload)
            .expect("manifest written");

        let error = writer
            .write_runtime_generation(&snapshot)
            .expect_err("semantic generation rejected");
        assert!(matches!(error, RuntimeError::CorruptArtifactGeneration));
        assert!(fake_dir.is_dir());
        std::fs::remove_dir_all(run_dir).expect("test directory removed");
    }

    #[test]
    fn current_pointer_bundle_hash_must_match_generation_manifest() {
        let run_dir = std::env::temp_dir().join(format!(
            "alphasync_pointer_crosslink_test_{}",
            std::process::id()
        ));
        if run_dir.exists() {
            std::fs::remove_dir_all(&run_dir).expect("old test directory removed");
        }
        let (matrix, rules, report, progress_reports, session) = artifact_fixture();
        let writer =
            RuntimeArtifactWriter::open_or_create(&run_dir).expect("runtime writer opened");
        let snapshot = RuntimeArtifactSnapshot::new(
            &matrix,
            &rules,
            &report,
            &progress_reports,
            Some(&session),
        )
        .expect("snapshot valid");
        writer
            .write_runtime_generation(&snapshot)
            .expect("initial generation written");
        let pointer_path = run_dir.join("current_generation.json");
        let pointer = std::fs::read_to_string(&pointer_path).expect("pointer readable");
        let mut pointer_value =
            serde_json::from_str::<serde_json::Value>(&pointer).expect("pointer JSON parsed");
        pointer_value["bundle_sha256"] = serde_json::Value::String("b".repeat(64));
        let corrupted_pointer = format!(
            "{}\n",
            serde_json::to_string_pretty(&pointer_value).expect("pointer serialized")
        );
        std::fs::write(&pointer_path, corrupted_pointer).expect("corrupt pointer written");

        let error = writer
            .write_runtime_generation(&snapshot)
            .expect_err("pointer cross-link rejected");
        assert!(matches!(error, RuntimeError::CorruptArtifactGeneration));
        std::fs::remove_dir_all(run_dir).expect("test directory removed");
    }

    #[test]
    fn valid_unpublished_generations_are_bounded_before_next_staging() {
        let run_dir = std::env::temp_dir().join(format!(
            "alphasync_orphan_generation_cap_test_{}",
            std::process::id()
        ));
        if run_dir.exists() {
            std::fs::remove_dir_all(&run_dir).expect("old test directory removed");
        }
        let (matrix, rules, report, progress_reports, session) = artifact_fixture();
        let writer =
            RuntimeArtifactWriter::open_or_create(&run_dir).expect("runtime writer opened");
        let snapshot = RuntimeArtifactSnapshot::new(
            &matrix,
            &rules,
            &report,
            &progress_reports,
            Some(&session),
        )
        .expect("snapshot valid");
        writer
            .write_runtime_generation(&snapshot)
            .expect("initial generation written");
        let protected_current = current_generation_dir(&run_dir);
        let generation_root = run_dir.join("generations");

        for index in 0..6 {
            let source_dir = std::env::temp_dir().join(format!(
                "alphasync_orphan_source_{}_{}",
                std::process::id(),
                index
            ));
            if source_dir.exists() {
                std::fs::remove_dir_all(&source_dir).expect("old source removed");
            }
            let source_writer =
                RuntimeArtifactWriter::open_or_create(&source_dir).expect("source writer opened");
            let source_session = RuntimeSessionManifest::new(
                RuntimeSessionKind::SyntheticScene,
                index + 2,
                index + 2,
                index + 2,
            )
            .expect("source session valid");
            let source_snapshot = RuntimeArtifactSnapshot::new(
                &matrix,
                &rules,
                &report,
                &progress_reports,
                Some(&source_session),
            )
            .expect("source snapshot valid");
            source_writer
                .write_runtime_generation(&source_snapshot)
                .expect("source generation written");
            let source_current = current_generation_dir(&source_dir);
            let target = generation_root.join(
                source_current
                    .file_name()
                    .expect("source generation has file name"),
            );
            if !target.exists() {
                copy_directory_tree(&source_current, &target);
            }
            std::fs::remove_dir_all(source_dir).expect("source removed");
        }

        let generation_count_before = std::fs::read_dir(&generation_root)
            .expect("generation root readable")
            .count();
        assert!(generation_count_before > super::MAX_RUNTIME_GENERATION_RETAINED_DIRS);

        writer
            .write_runtime_generation(&snapshot)
            .expect("orphan inventory pruned before staging");
        let generation_count_after = std::fs::read_dir(&generation_root)
            .expect("generation root readable")
            .count();
        assert!(generation_count_after <= super::MAX_RUNTIME_GENERATION_RETAINED_DIRS);
        assert!(protected_current.is_dir());
        std::fs::remove_dir_all(run_dir).expect("test directory removed");
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn runtime_artifact_writer_emits_safe_summary_snapshot_and_checksums() {
        let run_dir = std::env::temp_dir().join(format!(
            "alphasync_runtime_writer_test_{}",
            std::process::id()
        ));
        if run_dir.exists() {
            std::fs::remove_dir_all(&run_dir).expect("old test directory removed");
        }

        let shape = MatrixShape::new(1, 2, 2).expect("valid matrix shape");
        let matrix = ObservationMatrix::from_sparse(
            shape,
            vec![CellSample::new(address(0, 0, 0), 7).expect("non-zero sample")],
        )
        .expect("valid sparse matrix");
        let rule = PrismionRule::new(
            prismion_id(),
            vec![CellPredicate::new(address(0, 0, 0), CellComparison::Gte, 5)],
            vec![ProposalPatch::new(address(0, 1, 1), 1)],
            ConfidencePpm::new(900_000).expect("valid confidence"),
        )
        .expect("valid Prismion rule");
        let rules = [rule];
        let report = AlphaSyncRuntime::run_cycle(
            policy(),
            &matrix,
            &rules,
            &[proposal_id()],
            Some(RunProgressCursor::new(0, 0, 0)),
            RunProgressCursor::new(1, 1, 1),
        )
        .expect("runtime cycle succeeds");
        let writer =
            RuntimeArtifactWriter::open_or_create(&run_dir).expect("runtime writer opened");
        let session = RuntimeSessionManifest::new(RuntimeSessionKind::SyntheticScene, 2, 2, 2)
            .expect("valid session manifest");
        let progress_reports = [report.clone(), report.clone()];
        let snapshot = RuntimeArtifactSnapshot::new(
            &matrix,
            &rules,
            &report,
            &progress_reports,
            Some(&session),
        )
        .expect("artifact snapshot is valid");
        let generation = writer
            .write_runtime_generation(&snapshot)
            .expect("runtime generation written");

        let current_dir = current_generation_dir(&run_dir);
        let summary_text = std::fs::read_to_string(current_dir.join("runtime_summary.json"))
            .expect("summary readable");
        let snapshot_text = std::fs::read_to_string(current_dir.join("progress_snapshot.json"))
            .expect("snapshot readable");
        let frame_text = std::fs::read_to_string(current_dir.join("runtime_frame.json"))
            .expect("frame readable");
        let checksum_text = std::fs::read_to_string(current_dir.join("artifact_checksums.json"))
            .expect("checksums readable");
        let session_text = std::fs::read_to_string(current_dir.join("runtime_session.json"))
            .expect("session readable");
        let bundle_text = std::fs::read_to_string(current_dir.join("runtime_bundle.json"))
            .expect("bundle readable");
        let generation_text = std::fs::read_to_string(current_dir.join("runtime_generation.json"))
            .expect("generation readable");
        let pointer_text = std::fs::read_to_string(run_dir.join("current_generation.json"))
            .expect("current pointer readable");
        for (artifact_name, artifact_text) in [
            ("runtime_summary.json", &summary_text),
            ("progress_snapshot.json", &snapshot_text),
            ("runtime_frame.json", &frame_text),
            ("artifact_checksums.json", &checksum_text),
            ("runtime_session.json", &session_text),
            ("runtime_bundle.json", &bundle_text),
            ("runtime_generation.json", &generation_text),
            ("current_generation.json", &pointer_text),
        ] {
            serde_json::from_str::<serde_json::Value>(artifact_text)
                .unwrap_or_else(|error| panic!("{artifact_name} is valid JSON: {error}"));
        }

        assert!(summary_text.contains("\"schema_version\": \"alphasync.runtime_summary.v1\""));
        assert!(summary_text.contains("\"consensus_allowed_independent_sources\": 0"));
        assert!(summary_text.contains("\"consensus_allowed_source_kinds\": 0"));
        assert!(summary_text.contains("\"proposal_count\": 1"));
        assert!(snapshot_text.contains("\"stage\": \"runtime_cycle\""));
        assert!(!run_dir.join(legacy_progress_name()).exists());
        assert!(frame_text.contains("\"schema_version\": \"alphasync.runtime_frame.v2\""));
        assert!(frame_text.starts_with("{\n"));
        assert!(frame_text.contains("  \"matrix\": {\n"));
        assert!(!frame_text.contains("{{"));
        assert!(!frame_text.contains("}}"));
        assert!(frame_text.contains("\"non_zero_cell_count\": 1"));
        assert!(frame_text.contains("\"public_redaction\": \"observation_values_omitted\""));
        assert!(!frame_text.contains("\"non_zero_cells\""));
        assert!(frame_text.contains("\"condition_count\": 1"));
        assert!(frame_text.contains("\"evidence_cell_count\": 1"));
        assert!(frame_text.contains("\"patch_count\": 1"));
        assert!(frame_text.contains("\"proposal_field\""));
        assert!(frame_text.contains("\"occupied_cell_count\": 1"));
        assert!(frame_text.contains("\"public_redaction\": \"raw_text_omitted\""));
        assert!(frame_text.contains("\"proposals\""));
        assert!(!frame_text.contains("\"conditions\""));
        assert!(!frame_text.contains("\"evidence_cells\""));
        assert!(!frame_text.contains("\"patches\""));
        assert!(!frame_text.contains("\"proposal_id\""));
        assert!(!frame_text.contains("\"source\""));
        assert!(!frame_text.contains("\"matrix\": {\n    \"non_zero_cells\""));
        assert!(checksum_text.contains("runtime_summary.json"));
        assert!(checksum_text.contains("progress_snapshot.json"));
        assert!(checksum_text.contains("runtime_frame.json"));
        assert!(checksum_text.contains("runtime_session.json"));
        assert_eq!(
            generation.artifact_checksum().name(),
            "artifact_checksums.json"
        );
        assert!(generation.artifact_checksum().bytes() > 0);
        assert!(session_text.contains("\"schema_version\": \"alphasync.runtime_session.v1\""));
        assert!(session_text.contains("\"cycles_completed\": 2"));
        assert!(bundle_text.contains("\"schema_version\": \"alphasync.runtime_bundle.v1\""));
        assert!(bundle_text.contains("\"artifact_files\""));
        assert!(bundle_text.contains("\"runtime_summary.json\""));
        assert!(bundle_text.contains("\"runtime_frame.json\""));
        assert!(!bundle_text.contains("\"runtime_frame\""));
        assert!(bundle_text.contains("\"runtime_session\""));
        assert!(bundle_text.contains("\"progress_snapshot.json\""));
        assert!(bundle_text.contains("\"artifact_checksums\""));
        let bundle_value =
            serde_json::from_str::<serde_json::Value>(&bundle_text).expect("bundle JSON parsed");
        let progress = bundle_value
            .get("progress")
            .and_then(serde_json::Value::as_array)
            .expect("progress array exists");
        assert_eq!(progress.len(), 2);
        assert!(progress.iter().all(|row| {
            row.get("schema_version")
                .and_then(serde_json::Value::as_str)
                == Some("alphasync.progress.v1")
        }));
        assert_eq!(generation.bundle_checksum().name(), "runtime_bundle.json");
        assert!(generation.bundle_checksum().bytes() > 0);
        assert_eq!(
            generation.generation_checksum().name(),
            "runtime_generation.json"
        );
        assert!(generation.generation_checksum().bytes() > 0);
        assert!(
            generation_text.contains("\"schema_version\": \"alphasync.runtime_generation.v1\"")
        );
        assert!(
            generation_text.contains("\"commit_protocol\": \"immutable_generation_directory_v1\"")
        );
        assert!(generation_text.contains("\"runtime_summary.json\""));
        assert!(generation_text.contains("\"runtime_bundle.json\""));
        assert!(generation_text.contains("\"artifact_checksums.json\""));
        assert!(pointer_text.contains("\"schema_version\": \"alphasync.current_generation.v1\""));
        assert!(pointer_text.contains("\"generation_dir\": \"generations/generation_"));
        let current_dir_before_retry = current_dir.clone();
        let retry_snapshot = RuntimeArtifactSnapshot::new(
            &matrix,
            &rules,
            &report,
            &progress_reports,
            Some(&session),
        )
        .expect("retry snapshot is valid");
        let retry_generation = writer
            .write_runtime_generation(&retry_snapshot)
            .expect("same runtime generation retry is idempotent");
        assert_eq!(
            retry_generation.generation_checksum(),
            generation.generation_checksum()
        );
        assert!(current_dir_before_retry.is_dir());
        assert!(
            current_dir_before_retry
                .join("runtime_bundle.json")
                .is_file()
        );
        assert_eq!(
            std::fs::read_to_string(run_dir.join("current_generation.json"))
                .expect("retry pointer readable"),
            pointer_text
        );

        let interrupted_tmp = run_dir
            .join(".generation_tmp")
            .join("generation_interrupted");
        std::fs::create_dir_all(&interrupted_tmp).expect("interrupted tmp created");
        std::fs::write(interrupted_tmp.join("runtime_bundle.json"), "{}")
            .expect("interrupted tmp payload written");
        let unchanged_pointer = std::fs::read_to_string(run_dir.join("current_generation.json"))
            .expect("current pointer still readable");
        assert_eq!(pointer_text, unchanged_pointer);

        let oversized_reports = vec![report.clone(); MAX_RUNTIME_BUNDLE_PROGRESS_ROWS + 1];
        assert!(matches!(
            RuntimeArtifactSnapshot::new(
                &matrix,
                &rules,
                &report,
                &oversized_reports,
                Some(&session)
            ),
            Err(RuntimeError::InvalidSessionConfig)
        ));
        let capped_reports = vec![report.clone(); MAX_RUNTIME_BUNDLE_PROGRESS_ROWS];
        let capped_snapshot =
            RuntimeArtifactSnapshot::new(&matrix, &rules, &report, &capped_reports, Some(&session))
                .expect("capped snapshot is valid");
        writer
            .write_runtime_generation(&capped_snapshot)
            .expect("capped runtime generation written");
        assert!(!run_dir.join(".generation_tmp").exists());
        let capped_current_dir = current_generation_dir(&run_dir);
        let capped_bundle_text =
            std::fs::read_to_string(capped_current_dir.join("runtime_bundle.json"))
                .expect("bundle readable");
        let capped_bundle_value = serde_json::from_str::<serde_json::Value>(&capped_bundle_text)
            .expect("capped bundle JSON parsed");
        let capped_progress = capped_bundle_value
            .get("progress")
            .and_then(serde_json::Value::as_array)
            .expect("capped progress array exists");
        assert_eq!(capped_progress.len(), MAX_RUNTIME_BUNDLE_PROGRESS_ROWS);
        assert!(capped_progress.iter().all(|row| {
            row.get("schema_version")
                .and_then(serde_json::Value::as_str)
                == Some("alphasync.progress.v1")
        }));

        let lock_path = run_dir.join(".generation_write.lock");
        let generation_lock =
            GenerationWriteLock::acquire(&run_dir).expect("synthetic lock acquired");
        assert!(lock_path.is_file());
        assert_eq!(
            std::fs::metadata(&lock_path)
                .expect("lock metadata readable")
                .len(),
            0
        );
        drop(generation_lock);
        assert!(lock_path.is_file());

        std::fs::write(current_dir_before_retry.join("runtime_bundle.json"), "{}")
            .expect("committed generation corrupted");
        let corrupt_snapshot = RuntimeArtifactSnapshot::new(
            &matrix,
            &rules,
            &report,
            &progress_reports,
            Some(&session),
        )
        .expect("corrupt retry snapshot is valid");
        let corrupt_retry = writer.write_runtime_generation(&corrupt_snapshot);
        assert!(matches!(
            corrupt_retry,
            Err(RuntimeError::CorruptArtifactGeneration)
        ));

        std::fs::remove_dir_all(run_dir).expect("test directory removed");
    }

    #[test]
    fn runtime_writer_rejects_unowned_nonempty_directory_without_cleanup() {
        let run_dir = std::env::temp_dir().join(format!(
            "alphasync_runtime_unowned_writer_test_{}",
            std::process::id()
        ));
        if run_dir.exists() {
            std::fs::remove_dir_all(&run_dir).expect("old test directory removed");
        }
        std::fs::create_dir_all(&run_dir).expect("test directory created");
        let legacy_progress = run_dir.join(legacy_progress_name());
        std::fs::write(&legacy_progress, "keep\n").expect("legacy progress written");

        let error = RuntimeArtifactWriter::open_or_create(&run_dir)
            .expect_err("nonempty unowned directory rejected");
        assert!(matches!(error, RuntimeError::UnownedArtifactDirectory));
        assert_eq!(
            std::fs::read_to_string(&legacy_progress).expect("legacy progress still present"),
            "keep\n"
        );

        std::fs::remove_dir_all(run_dir).expect("test directory removed");
    }
}
