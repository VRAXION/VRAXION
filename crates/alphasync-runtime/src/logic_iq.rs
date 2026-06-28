//! Deterministic Logic-IQ-0 curriculum runner.
//!
//! This module is the first compiled logic-curriculum canary. It does not read
//! text, call an LLM, or claim general reasoning. It generates tiny anonymous
//! matrix worlds with known oracle outcomes, runs bounded Prismion rules over
//! them, and records whether the emitted proposal patch matches the oracle.
//!
//! The public `logic-iq-zero` path is a fixed-rule control. Training/search
//! code is intentionally outside this public-candidate runtime surface.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use alphasync_core::fabric::{
    CellAddress, CellComparison, CellPredicate, CellSample, ConfidencePpm, ConsensusPolicy,
    ConsensusRecommendation, ConsensusSourceKind, ConsensusVote, ConsensusVoteKind,
    FabricShapeProfile, FabricShapeProfileKind, MatrixShape, ObservationMatrix, PrismionProposal,
    PrismionRule, ProposalPatch,
};
use alphasync_core::ids::{PrismionId, ProposalId};
use alphasync_core::progress::{
    EvidenceSlot, RunProgressCursor, WriteoutCadence, WriteoutDecision,
};

use crate::{
    AlphaSyncRuntime, ArtifactChecksum, RuntimeArtifactSnapshot, RuntimeArtifactWriter,
    RuntimeConsensusReport, RuntimeConsensusSourceKind, RuntimeConsensusVoteReport,
    RuntimeCycleReport, RuntimeError, RuntimeFlowCommit, RuntimeFlowDecision, RuntimeFlowReason,
    RuntimePolicy, RuntimeSessionKind, RuntimeSessionManifest, push_bounded_progress_report,
};

/// Default Logic-IQ-0 cycle count.
pub const DEFAULT_LOGIC_IQ_CYCLES: usize = 10_000;

/// Default Logic-IQ-0 writeout interval in cycles.
pub const DEFAULT_LOGIC_IQ_WRITE_EVERY: usize = 100;

/// Default Logic-IQ consensus scene cycle count.
pub const DEFAULT_LOGIC_IQ_CONSENSUS_SCENE_CYCLES: usize = 1;

/// Default Logic-IQ consensus scene writeout interval in cycles.
pub const DEFAULT_LOGIC_IQ_CONSENSUS_SCENE_WRITE_EVERY: usize = 1;

/// Default self-contained output directory for Logic-IQ runtime artifacts.
///
/// The private GUI may monitor this folder, but the public runner must not have
/// a default path that points into the private monitor tree.
pub const DEFAULT_LOGIC_IQ_RUN_DIR: &str = "runtime_runs/logic_iq_current";

const TASK_CELL: CellAddress = CellAddress::new(0, 0, 0);
const A_CELL: CellAddress = CellAddress::new(0, 0, 1);
const B_CELL: CellAddress = CellAddress::new(0, 0, 2);
const NOT_A_CELL: CellAddress = CellAddress::new(0, 0, 3);
const RESULT_TARGET: CellAddress = CellAddress::new(0, 0, 0);

const TASK_IDENTITY: u8 = 1;
const TASK_NEGATION: u8 = 2;
const TASK_AND: u8 = 3;
const TASK_OR: u8 = 4;
const TASK_IMPLICATION: u8 = 5;
const TASK_CONTRADICTION: u8 = 6;
const TASK_MISSING_EVIDENCE: u8 = 7;

const VALUE_TRUE: u8 = 1;
const VALUE_FALSE: u8 = 2;
const VALUE_UNKNOWN: u8 = 3;

const RESULT_TRUE: u8 = 1;
const RESULT_FALSE: u8 = 2;
const RESULT_CONFLICT: u8 = 3;
const RESULT_UNKNOWN: u8 = 4;

const FLOW_COMMIT_CONFIDENCE_FLOOR_PPM: u32 = 600_000;
/// Configuration for one Logic-IQ-0 run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicIqRunConfig {
    cycles: usize,
    write_every: usize,
    profile_kind: FabricShapeProfileKind,
    out_dir: PathBuf,
}

impl LogicIqRunConfig {
    /// Creates a Logic-IQ-0 run configuration.
    #[must_use]
    pub fn new(cycles: usize, write_every: usize, out_dir: PathBuf) -> Self {
        Self::new_with_profile(
            cycles,
            write_every,
            FabricShapeProfileKind::LogicIqCanary,
            out_dir,
        )
    }

    /// Creates a Logic-IQ-0 run configuration with an explicit shape profile.
    #[must_use]
    pub fn new_with_profile(
        cycles: usize,
        write_every: usize,
        profile_kind: FabricShapeProfileKind,
        out_dir: PathBuf,
    ) -> Self {
        Self {
            cycles,
            write_every,
            profile_kind,
            out_dir,
        }
    }

    /// Returns requested cycle count.
    #[must_use]
    pub const fn cycles(&self) -> usize {
        self.cycles
    }

    /// Returns writeout interval in cycles.
    #[must_use]
    pub const fn write_every(&self) -> usize {
        self.write_every
    }

    /// Returns the matrix-shape profile kind.
    #[must_use]
    pub const fn profile_kind(&self) -> FabricShapeProfileKind {
        self.profile_kind
    }

    /// Returns output directory.
    #[must_use]
    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }
}

/// Safe aggregate result from a Logic-IQ-0 run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicIqRunResult {
    cycles: usize,
    writeout_count: usize,
    exact_count: usize,
    wrong_count: usize,
    false_commit_count: usize,
    conflict_or_unknown_count: usize,
    proposal_total: usize,
    patch_total: usize,
    elapsed: Duration,
    cycles_per_second: u128,
    out_dir: PathBuf,
    artifact_checksum: ArtifactChecksum,
    bundle_checksum: ArtifactChecksum,
}

impl LogicIqRunResult {
    /// Returns completed cycle count.
    #[must_use]
    pub const fn cycles(&self) -> usize {
        self.cycles
    }

    /// Returns durable writeout count.
    #[must_use]
    pub const fn writeout_count(&self) -> usize {
        self.writeout_count
    }

    /// Returns exact oracle match count.
    #[must_use]
    pub const fn exact_count(&self) -> usize {
        self.exact_count
    }

    /// Returns non-exact count.
    #[must_use]
    pub const fn wrong_count(&self) -> usize {
        self.wrong_count
    }

    /// Returns cases where an uncertain/conflict oracle received a commit-like
    /// true/false proposal.
    #[must_use]
    pub const fn false_commit_count(&self) -> usize {
        self.false_commit_count
    }

    /// Returns oracle cases whose correct answer is conflict or unknown.
    #[must_use]
    pub const fn conflict_or_unknown_count(&self) -> usize {
        self.conflict_or_unknown_count
    }

    /// Returns total emitted proposals.
    #[must_use]
    pub const fn proposal_total(&self) -> usize {
        self.proposal_total
    }

    /// Returns total emitted patches.
    #[must_use]
    pub const fn patch_total(&self) -> usize {
        self.patch_total
    }

    /// Returns measured elapsed runtime.
    #[must_use]
    pub const fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// Returns cycle throughput derived from elapsed runtime.
    #[must_use]
    pub const fn cycles_per_second(&self) -> u128 {
        self.cycles_per_second
    }

    /// Returns output directory.
    #[must_use]
    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }

    /// Returns checksum manifest artifact record.
    #[must_use]
    pub const fn artifact_checksum(&self) -> &ArtifactChecksum {
        &self.artifact_checksum
    }

    /// Returns the one-file GUI bundle checksum artifact record.
    #[must_use]
    pub const fn bundle_checksum(&self) -> &ArtifactChecksum {
        &self.bundle_checksum
    }
}

/// Runs the deterministic Logic-IQ-0 curriculum canary.
///
/// # Errors
///
/// Returns [`RuntimeError`] when configuration, matrix generation, runtime
/// evaluation, or artifact writing fails.
pub fn run_logic_iq_zero(config: &LogicIqRunConfig) -> Result<LogicIqRunResult, RuntimeError> {
    if config.cycles() == 0 || config.write_every() == 0 {
        return Err(RuntimeError::InvalidSessionConfig);
    }

    let (matrix_shape, policy) = logic_iq_matrix_and_policy(config.profile_kind())?;
    let rules = logic_rules()?;
    let proposal_ids = proposal_ids(rules.len())?;
    let writer = RuntimeArtifactWriter::open_or_create(config.out_dir())?;
    let mut last_writeout = None;
    let mut progress_reports = Vec::new();
    let mut exact_count = 0usize;
    let mut wrong_count = 0usize;
    let mut false_commit_count = 0usize;
    let mut conflict_or_unknown_count = 0usize;
    let mut proposal_total = 0usize;
    let mut patch_total = 0usize;
    let mut writeout_count = 0usize;
    let mut latest_artifact_checksum = None;
    let mut latest_bundle_checksum = None;
    let start = Instant::now();

    for cycle in 1..=config.cycles() {
        let case = logic_case(cycle);
        let matrix = case.matrix(matrix_shape)?;
        let current_progress = progress_cursor_since(start, cycle)?;
        let report = run_cycle_with_agency_commit(
            policy,
            &matrix,
            &rules,
            &proposal_ids,
            last_writeout,
            current_progress,
            cycle,
        )?;
        proposal_total += report.proposal_count();
        patch_total += report.patch_count();
        let expected = case.expected_result();
        let actual = committed_result(&report);
        if expected == RESULT_CONFLICT || expected == RESULT_UNKNOWN {
            conflict_or_unknown_count += 1;
            if matches!(actual, Some(RESULT_TRUE | RESULT_FALSE)) {
                false_commit_count += 1;
            }
        }
        if actual == Some(expected) {
            exact_count += 1;
        } else {
            wrong_count += 1;
        }

        if cycle % config.write_every() == 0
            || cycle == config.cycles()
            || writeout_due(report.writeout_decision())
        {
            writeout_count += 1;
            let (artifact_checksum, bundle_checksum) = write_logic_iq_artifacts(
                &writer,
                &matrix,
                &rules,
                &report,
                &mut progress_reports,
                ArtifactWriteContext {
                    session_kind: RuntimeSessionKind::LogicIqZero,
                    cycles_requested: config.cycles(),
                    cycle,
                    writeout_count,
                },
            )?;
            latest_artifact_checksum = Some(artifact_checksum);
            latest_bundle_checksum = Some(bundle_checksum);
            last_writeout = Some(current_progress);
        }
    }

    let elapsed = start.elapsed();
    let elapsed_ns = elapsed.as_nanos().max(1);
    let cycles_per_second = (config.cycles() as u128 * 1_000_000_000u128) / elapsed_ns;
    let Some(artifact_checksum) = latest_artifact_checksum else {
        return Err(RuntimeError::InvalidSessionConfig);
    };
    let Some(bundle_checksum) = latest_bundle_checksum else {
        return Err(RuntimeError::InvalidSessionConfig);
    };

    Ok(LogicIqRunResult {
        cycles: config.cycles(),
        writeout_count,
        exact_count,
        wrong_count,
        false_commit_count,
        conflict_or_unknown_count,
        proposal_total,
        patch_total,
        elapsed,
        cycles_per_second,
        out_dir: config.out_dir().to_path_buf(),
        artifact_checksum,
        bundle_checksum,
    })
}

/// Writes a deterministic multi-proposal Logic-IQ consensus scene.
///
/// This is a GUI/runtime inspection fixture, not a training path. It uses one
/// observation matrix with multiple matching Prismions so that the
/// `ProposalField`, `TraceContext`, `GroundContext`, and individual Prismion vote
/// lanes are all visible in the emitted runtime artifacts.
///
/// # Errors
///
/// Returns [`RuntimeError`] when configuration, matrix generation, runtime
/// evaluation, or artifact writing fails.
pub fn run_logic_iq_consensus_scene(
    config: &LogicIqRunConfig,
) -> Result<LogicIqRunResult, RuntimeError> {
    if config.cycles() == 0 || config.write_every() == 0 {
        return Err(RuntimeError::InvalidSessionConfig);
    }

    let (matrix_shape, policy) = logic_iq_matrix_and_policy(config.profile_kind())?;
    let case = logic_consensus_scene_case();
    let rules = logic_consensus_scene_rules()?;
    let proposal_ids = proposal_ids(rules.len())?;
    let writer = RuntimeArtifactWriter::open_or_create(config.out_dir())?;
    let mut last_writeout = None;
    let mut progress_reports = Vec::new();
    let mut exact_count = 0usize;
    let mut wrong_count = 0usize;
    let mut false_commit_count = 0usize;
    let mut proposal_total = 0usize;
    let mut patch_total = 0usize;
    let mut writeout_count = 0usize;
    let mut latest_artifact_checksum = None;
    let mut latest_bundle_checksum = None;
    let start = Instant::now();

    for cycle in 1..=config.cycles() {
        let matrix = case.matrix(matrix_shape)?;
        let current_progress = progress_cursor_since(start, cycle)?;
        let report = run_cycle_with_agency_commit(
            policy,
            &matrix,
            &rules,
            &proposal_ids,
            last_writeout,
            current_progress,
            cycle,
        )?;
        proposal_total += report.proposal_count();
        patch_total += report.patch_count();
        let expected = case.expected_result();
        let actual = committed_result(&report);
        if actual == Some(expected) {
            exact_count += 1;
        } else {
            wrong_count += 1;
            if matches!(actual, Some(RESULT_TRUE | RESULT_FALSE)) {
                false_commit_count += 1;
            }
        }

        if cycle % config.write_every() == 0
            || cycle == config.cycles()
            || writeout_due(report.writeout_decision())
        {
            writeout_count += 1;
            let (artifact_checksum, bundle_checksum) = write_logic_iq_artifacts(
                &writer,
                &matrix,
                &rules,
                &report,
                &mut progress_reports,
                ArtifactWriteContext {
                    session_kind: RuntimeSessionKind::LogicIqConsensusScene,
                    cycles_requested: config.cycles(),
                    cycle,
                    writeout_count,
                },
            )?;
            latest_artifact_checksum = Some(artifact_checksum);
            latest_bundle_checksum = Some(bundle_checksum);
            last_writeout = Some(current_progress);
        }
    }

    let elapsed = start.elapsed();
    let elapsed_ns = elapsed.as_nanos().max(1);
    let cycles_per_second = (config.cycles() as u128 * 1_000_000_000u128) / elapsed_ns;
    let Some(artifact_checksum) = latest_artifact_checksum else {
        return Err(RuntimeError::InvalidSessionConfig);
    };
    let Some(bundle_checksum) = latest_bundle_checksum else {
        return Err(RuntimeError::InvalidSessionConfig);
    };

    Ok(LogicIqRunResult {
        cycles: config.cycles(),
        writeout_count,
        exact_count,
        wrong_count,
        false_commit_count,
        conflict_or_unknown_count: 0,
        proposal_total,
        patch_total,
        elapsed,
        cycles_per_second,
        out_dir: config.out_dir().to_path_buf(),
        artifact_checksum,
        bundle_checksum,
    })
}

fn logic_iq_matrix_and_policy(
    profile_kind: FabricShapeProfileKind,
) -> Result<(MatrixShape, RuntimePolicy), RuntimeError> {
    let profile = FabricShapeProfile::from_kind(profile_kind)?;
    let policy = RuntimePolicy::new_with_flow_shape(
        profile.proposal(),
        profile.flow(),
        WriteoutCadence::frontier_default(),
        EvidenceSlot::new(0)?,
        EvidenceSlot::new(1)?,
    )?
    .with_consensus_policy(ConsensusPolicy::e150_best_safe()?);
    Ok((profile.observation(), policy))
}

fn progress_cursor_since(start: Instant, cycle: usize) -> Result<RunProgressCursor, RuntimeError> {
    let elapsed_ms = u64::try_from(start.elapsed().as_millis())
        .map_err(|_error| RuntimeError::InvalidSessionConfig)?;
    let cursor = u64::try_from(cycle).map_err(|_error| RuntimeError::InvalidSessionConfig)?;
    Ok(RunProgressCursor::new(elapsed_ms, cursor, cursor))
}

const fn writeout_due(decision: WriteoutDecision) -> bool {
    matches!(decision, WriteoutDecision::Write { .. })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LogicCase {
    task: u8,
    a: u8,
    b: u8,
    not_a: u8,
}

impl LogicCase {
    fn matrix(self, shape: MatrixShape) -> Result<ObservationMatrix, RuntimeError> {
        let mut samples = vec![
            CellSample::new(TASK_CELL, self.task)?,
            CellSample::new(A_CELL, self.a)?,
        ];
        if uses_b(self.task) {
            samples.push(CellSample::new(B_CELL, self.b)?);
        }
        if self.task == TASK_CONTRADICTION && self.not_a != 0 {
            samples.push(CellSample::new(NOT_A_CELL, self.not_a)?);
        }
        Ok(ObservationMatrix::from_sparse(shape, samples)?)
    }

    const fn expected_result(self) -> u8 {
        match self.task {
            TASK_IDENTITY => truth_result(self.a),
            TASK_NEGATION => negation_result(self.a),
            TASK_AND => and_result(self.a, self.b),
            TASK_OR => or_result(self.a, self.b),
            TASK_IMPLICATION => implication_result(self.a, self.b),
            TASK_CONTRADICTION => contradiction_result(self.a, self.not_a),
            _ => RESULT_UNKNOWN,
        }
    }
}

fn write_logic_iq_artifacts(
    writer: &RuntimeArtifactWriter,
    matrix: &ObservationMatrix,
    rules: &[PrismionRule],
    report: &RuntimeCycleReport,
    progress_reports: &mut Vec<RuntimeCycleReport>,
    context: ArtifactWriteContext,
) -> Result<(ArtifactChecksum, ArtifactChecksum), RuntimeError> {
    push_bounded_progress_report(progress_reports, report.clone());
    let session = RuntimeSessionManifest::new(
        context.session_kind,
        context.cycles_requested,
        context.cycle,
        context.writeout_count,
    )?;
    let snapshot =
        RuntimeArtifactSnapshot::new(matrix, rules, report, progress_reports, Some(&session))?;
    let generation = writer.write_runtime_generation(&snapshot)?;
    Ok((
        generation.artifact_checksum().clone(),
        generation.bundle_checksum().clone(),
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ArtifactWriteContext {
    session_kind: RuntimeSessionKind,
    cycles_requested: usize,
    cycle: usize,
    writeout_count: usize,
}

fn case_sequence(count: usize) -> Vec<LogicCase> {
    (1..=count).map(logic_case).collect()
}

fn logic_universe_cases() -> Vec<LogicCase> {
    case_sequence(7 * 3 * 3 * 2)
}

fn logic_case(cycle: usize) -> LogicCase {
    let index = cycle - 1;
    let task = u8::try_from(index % 7).map_or(TASK_IDENTITY, |value| value + 1);
    let a = truth_state((index / 7) % 3);
    let b = truth_state((index / 21) % 3);
    let not_a = u8::try_from((index / 63) % 2).unwrap_or(0);
    LogicCase { task, a, b, not_a }
}

const fn logic_consensus_scene_case() -> LogicCase {
    LogicCase {
        task: TASK_IDENTITY,
        a: VALUE_TRUE,
        b: VALUE_UNKNOWN,
        not_a: 0,
    }
}

fn logic_consensus_scene_rules() -> Result<Vec<PrismionRule>, RuntimeError> {
    Ok(vec![
        logic_consensus_scene_rule(1, RESULT_TARGET, RESULT_TRUE)?,
        logic_consensus_scene_rule(2, RESULT_TARGET, RESULT_TRUE)?,
        logic_consensus_scene_rule(3, A_CELL, RESULT_TRUE)?,
    ])
}

fn logic_consensus_scene_rule(
    ordinal: usize,
    target: CellAddress,
    result: u8,
) -> Result<PrismionRule, RuntimeError> {
    Ok(PrismionRule::new(
        prismion_id(ordinal)?,
        vec![
            CellPredicate::new(TASK_CELL, CellComparison::Eq, TASK_IDENTITY),
            CellPredicate::new(A_CELL, CellComparison::Eq, VALUE_TRUE),
        ],
        vec![ProposalPatch::new(target, result)],
        ConfidencePpm::new(900_000)?,
    )?)
}

fn logic_rules() -> Result<Vec<PrismionRule>, RuntimeError> {
    let mut rules = Vec::new();
    let mut ordinal = 0usize;

    for a in truth_states() {
        push_rule(&mut rules, &mut ordinal, TASK_IDENTITY, a, None, None)?;
        push_rule(&mut rules, &mut ordinal, TASK_NEGATION, a, None, None)?;
        push_rule(
            &mut rules,
            &mut ordinal,
            TASK_MISSING_EVIDENCE,
            a,
            None,
            None,
        )?;
        for not_a in [0, 1] {
            push_rule(
                &mut rules,
                &mut ordinal,
                TASK_CONTRADICTION,
                a,
                None,
                Some(not_a),
            )?;
        }
        for b in truth_states() {
            push_rule(&mut rules, &mut ordinal, TASK_AND, a, Some(b), None)?;
            push_rule(&mut rules, &mut ordinal, TASK_OR, a, Some(b), None)?;
            push_rule(&mut rules, &mut ordinal, TASK_IMPLICATION, a, Some(b), None)?;
        }
    }

    Ok(rules)
}

fn push_rule(
    rules: &mut Vec<PrismionRule>,
    ordinal: &mut usize,
    task: u8,
    a: u8,
    b: Option<u8>,
    not_a: Option<u8>,
) -> Result<(), RuntimeError> {
    let case = LogicCase {
        task,
        a,
        b: b.unwrap_or(VALUE_UNKNOWN),
        not_a: not_a.unwrap_or(0),
    };
    let mut conditions = vec![
        CellPredicate::new(TASK_CELL, CellComparison::Eq, task),
        CellPredicate::new(A_CELL, CellComparison::Eq, a),
    ];
    if let Some(value) = b {
        conditions.push(CellPredicate::new(B_CELL, CellComparison::Eq, value));
    }
    if let Some(value) = not_a {
        conditions.push(CellPredicate::new(NOT_A_CELL, CellComparison::Eq, value));
    }
    let patches = vec![ProposalPatch::new(RESULT_TARGET, case.expected_result())];
    *ordinal += 1;
    rules.push(PrismionRule::new(
        prismion_id(*ordinal)?,
        conditions,
        patches,
        ConfidencePpm::new(900_000)?,
    )?);
    Ok(())
}

fn run_cycle_with_agency_commit(
    policy: RuntimePolicy,
    matrix: &ObservationMatrix,
    rules: &[PrismionRule],
    proposal_ids: &[ProposalId],
    last_writeout: Option<RunProgressCursor>,
    current_progress: RunProgressCursor,
    sequence: usize,
) -> Result<RuntimeCycleReport, RuntimeError> {
    let report = AlphaSyncRuntime::collect_proposals(
        policy,
        matrix,
        rules,
        proposal_ids,
        last_writeout,
        current_progress,
    )?;
    let (consensus_report, flow_commit) = agency_commit_report(
        &report,
        policy.flow_shape(),
        policy.consensus_policy(),
        sequence,
    )?;
    Ok(report.with_agency_reports(consensus_report, flow_commit))
}

fn agency_commit_report(
    report: &RuntimeCycleReport,
    flow_shape: MatrixShape,
    consensus_policy: Option<ConsensusPolicy>,
    sequence: usize,
) -> Result<(Option<RuntimeConsensusReport>, RuntimeFlowCommit), RuntimeError> {
    let sequence = u64::try_from(sequence).map_err(|_error| RuntimeError::InvalidSessionConfig)?;

    let no_commit = |decision, reason| {
        RuntimeFlowCommit::no_commit(
            sequence,
            flow_shape,
            decision,
            reason,
            report.proposal_count(),
            report.patch_count(),
            report.evidence_cell_count(),
        )
    };

    if report.proposal_count() == 0 {
        return Ok((
            None,
            no_commit(RuntimeFlowDecision::Defer, RuntimeFlowReason::NoProposal),
        ));
    }

    if report.proposal_count() == 1 {
        return Ok((
            None,
            commit_single_proposal(report, flow_shape, sequence, no_commit)?,
        ));
    }

    let Some(consensus_policy) = consensus_policy else {
        return Ok((
            None,
            no_commit(
                RuntimeFlowDecision::Reject,
                RuntimeFlowReason::ProposalCollision,
            ),
        ));
    };

    commit_consensus_proposals(report, flow_shape, consensus_policy, sequence, no_commit)
}

fn commit_single_proposal(
    report: &RuntimeCycleReport,
    flow_shape: MatrixShape,
    sequence: u64,
    no_commit: impl Fn(RuntimeFlowDecision, RuntimeFlowReason) -> RuntimeFlowCommit,
) -> Result<RuntimeFlowCommit, RuntimeError> {
    let proposal = &report.proposals()[0];
    if proposal.confidence().as_ppm() < FLOW_COMMIT_CONFIDENCE_FLOOR_PPM {
        return Ok(no_commit(
            RuntimeFlowDecision::Reject,
            RuntimeFlowReason::LowConfidence,
        ));
    }
    if proposal.evidence_cells().is_empty() {
        return Ok(no_commit(
            RuntimeFlowDecision::Reject,
            RuntimeFlowReason::MissingEvidence,
        ));
    }

    if proposal.patches().len() != 1 {
        return Ok(no_commit(
            RuntimeFlowDecision::Reject,
            RuntimeFlowReason::PatchCollision,
        ));
    }

    let patch = proposal.patches()[0];
    if !flow_shape.contains(patch.target()) {
        return Ok(no_commit(
            RuntimeFlowDecision::Reject,
            RuntimeFlowReason::PatchOutOfBounds,
        ));
    }
    if patch.target() != RESULT_TARGET {
        return Ok(no_commit(
            RuntimeFlowDecision::Reject,
            RuntimeFlowReason::InvalidTarget,
        ));
    }
    if report.proposal_field().get(patch.target())? != Some(patch.value()) {
        return Ok(no_commit(
            RuntimeFlowDecision::Reject,
            RuntimeFlowReason::ProposalCollision,
        ));
    }

    RuntimeFlowCommit::commit(
        sequence,
        flow_shape,
        patch.target(),
        patch.value(),
        report.proposal_count(),
        report.patch_count(),
        report.evidence_cell_count(),
    )
}

fn commit_consensus_proposals(
    report: &RuntimeCycleReport,
    flow_shape: MatrixShape,
    consensus_policy: ConsensusPolicy,
    sequence: u64,
    no_commit: impl Fn(RuntimeFlowDecision, RuntimeFlowReason) -> RuntimeFlowCommit,
) -> Result<(Option<RuntimeConsensusReport>, RuntimeFlowCommit), RuntimeError> {
    for proposal in report.proposals() {
        if proposal.patches().len() != 1 {
            return Ok((
                None,
                no_commit(
                    RuntimeFlowDecision::Reject,
                    RuntimeFlowReason::PatchCollision,
                ),
            ));
        }
        let patch = proposal.patches()[0];
        if !flow_shape.contains(patch.target()) {
            return Ok((
                None,
                no_commit(
                    RuntimeFlowDecision::Reject,
                    RuntimeFlowReason::PatchOutOfBounds,
                ),
            ));
        }
    }

    let candidate_target = RESULT_TARGET;
    let Some(candidate_value) = report.proposal_field().get(candidate_target)? else {
        return Ok((
            None,
            no_commit(
                RuntimeFlowDecision::Reject,
                RuntimeFlowReason::ProposalCollision,
            ),
        ));
    };

    let votes = consensus_votes(
        report,
        report.proposals(),
        candidate_target,
        candidate_value,
        flow_shape,
    )?;
    let decision = consensus_policy.decide(&votes)?;
    let vote_reports = consensus_vote_reports(&votes, consensus_policy);
    let consensus_report = RuntimeConsensusReport::new(
        decision,
        votes.len(),
        Some(candidate_target),
        Some(candidate_value),
        vote_reports,
    );

    if decision.action() != ConsensusRecommendation::RecommendCommit {
        let decision_kind = match decision.action() {
            ConsensusRecommendation::RecommendCommit => RuntimeFlowDecision::Commit,
            ConsensusRecommendation::RecommendReject => RuntimeFlowDecision::Reject,
            ConsensusRecommendation::RecommendDefer => RuntimeFlowDecision::Defer,
        };
        return Ok((
            Some(consensus_report),
            no_commit(
                decision_kind,
                consensus_no_commit_reason(consensus_policy, decision),
            ),
        ));
    }

    let commit = RuntimeFlowCommit::commit(
        sequence,
        flow_shape,
        candidate_target,
        candidate_value,
        report.proposal_count(),
        report.patch_count(),
        report.evidence_cell_count(),
    )?;
    Ok((Some(consensus_report), commit))
}

fn consensus_no_commit_reason(
    policy: ConsensusPolicy,
    decision: alphasync_core::fabric::ConsensusDecision,
) -> RuntimeFlowReason {
    if decision.action() == ConsensusRecommendation::RecommendReject {
        return RuntimeFlowReason::ConsensusRejected;
    }
    if decision.conflict_ppm() > policy.max_conflict().as_ppm() {
        return RuntimeFlowReason::ConsensusConflict;
    }
    if decision.stale_ppm() > policy.max_stale().as_ppm() {
        return RuntimeFlowReason::ConsensusStale;
    }
    if decision.winner_independent_sources() < policy.min_distinct_proposal_sources() {
        return RuntimeFlowReason::InsufficientConsensusSources;
    }
    RuntimeFlowReason::ConsensusDeferred
}

fn consensus_vote_reports(
    votes: &[ConsensusVote],
    policy: ConsensusPolicy,
) -> Vec<RuntimeConsensusVoteReport> {
    votes
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
        .collect()
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

fn consensus_votes(
    report: &RuntimeCycleReport,
    proposals: &[PrismionProposal],
    candidate_target: CellAddress,
    candidate_value: u8,
    flow_shape: MatrixShape,
) -> Result<Vec<ConsensusVote>, RuntimeError> {
    let mut votes = Vec::with_capacity(proposals.len().saturating_add(3));
    votes.push(consensus_proposal_field_vote(
        report,
        candidate_target,
        candidate_value,
    )?);
    votes.push(consensus_trace_context_vote(
        proposals,
        candidate_target,
        candidate_value,
    )?);
    votes.push(consensus_ground_context_vote(
        candidate_target,
        candidate_value,
        flow_shape,
    )?);
    for proposal in proposals {
        let patch = proposal.patches()[0];
        let kind = if patch.target() == candidate_target && patch.value() == candidate_value {
            ConsensusVoteKind::Support
        } else {
            ConsensusVoteKind::Reject
        };
        let evidence_quality = consensus_evidence_quality(proposal)?;
        votes.push(ConsensusVote::from_prismion_proposal(
            proposal,
            kind,
            ConfidencePpm::new(900_000)?,
            evidence_quality,
            0,
        ));
    }
    Ok(votes)
}

fn consensus_proposal_field_vote(
    report: &RuntimeCycleReport,
    candidate_target: CellAddress,
    candidate_value: u8,
) -> Result<ConsensusVote, RuntimeError> {
    let kind = if report.proposal_field().get(candidate_target)? == Some(candidate_value) {
        ConsensusVoteKind::Support
    } else {
        ConsensusVoteKind::Reject
    };
    let evidence_quality = proposal_field_evidence_quality(report)?;
    Ok(ConsensusVote::proposal_field(
        kind,
        ConfidencePpm::new(920_000)?,
        ConfidencePpm::new(900_000)?,
        evidence_quality,
        0,
    ))
}

fn consensus_trace_context_vote(
    proposals: &[PrismionProposal],
    candidate_target: CellAddress,
    candidate_value: u8,
) -> Result<ConsensusVote, RuntimeError> {
    let mut supporting = 0usize;
    let mut conflicting = 0usize;
    let mut evidence_supported = 0usize;
    for proposal in proposals {
        let patch = proposal.patches()[0];
        if patch.target() != candidate_target {
            continue;
        }
        if patch.value() == candidate_value {
            supporting = supporting.saturating_add(1);
            if !proposal.evidence_cells().is_empty() {
                evidence_supported = evidence_supported.saturating_add(1);
            }
        } else {
            conflicting = conflicting.saturating_add(1);
        }
    }

    let kind = if evidence_supported > 0 && supporting > conflicting {
        ConsensusVoteKind::Support
    } else {
        ConsensusVoteKind::Reject
    };
    let quality = 620_000u32
        .saturating_add(u32::try_from(core::cmp::min(evidence_supported, 4)).unwrap_or(0) * 80_000)
        .saturating_sub(u32::try_from(core::cmp::min(conflicting, 4)).unwrap_or(4) * 90_000);
    Ok(ConsensusVote::trace_context(
        kind,
        ConfidencePpm::new(940_000)?,
        ConfidencePpm::new(880_000)?,
        ConfidencePpm::new(core::cmp::min(quality, 1_000_000))?,
        0,
    ))
}

fn consensus_ground_context_vote(
    candidate_target: CellAddress,
    candidate_value: u8,
    flow_shape: MatrixShape,
) -> Result<ConsensusVote, RuntimeError> {
    let structurally_valid = flow_shape.contains(candidate_target)
        && candidate_target == RESULT_TARGET
        && matches!(
            candidate_value,
            RESULT_TRUE | RESULT_FALSE | RESULT_CONFLICT | RESULT_UNKNOWN
        );
    Ok(ConsensusVote::ground_context(
        if structurally_valid {
            ConsensusVoteKind::Support
        } else {
            ConsensusVoteKind::Reject
        },
        ConfidencePpm::new(900_000)?,
        ConfidencePpm::new(820_000)?,
        ConfidencePpm::new(if structurally_valid { 900_000 } else { 250_000 })?,
        0,
    ))
}

fn proposal_field_evidence_quality(
    report: &RuntimeCycleReport,
) -> Result<ConfidencePpm, RuntimeError> {
    let occupied = core::cmp::min(report.proposal_field().occupied_cell_count(), 4);
    let collision_penalty = u32::try_from(core::cmp::min(
        report.proposal_field().conflicting_patch_count(),
        4,
    ))
    .unwrap_or(4)
    .saturating_mul(80_000);
    let quality = 620_000u32
        .saturating_add(u32::try_from(occupied).unwrap_or(0).saturating_mul(90_000))
        .saturating_sub(collision_penalty);
    ConfidencePpm::new(core::cmp::min(quality, 1_000_000)).map_err(RuntimeError::from)
}

fn consensus_evidence_quality(proposal: &PrismionProposal) -> Result<ConfidencePpm, RuntimeError> {
    if proposal.evidence_cells().is_empty() {
        return ConfidencePpm::new(0).map_err(RuntimeError::from);
    }
    let evidence_count = core::cmp::min(proposal.evidence_cells().len(), 4);
    let quality = 600_000u32.saturating_add(u32::try_from(evidence_count).unwrap_or(0) * 100_000);
    ConfidencePpm::new(core::cmp::min(quality, 1_000_000)).map_err(RuntimeError::from)
}

fn committed_result(report: &RuntimeCycleReport) -> Option<u8> {
    let commit = report.flow_commit()?;
    if commit.decision() != RuntimeFlowDecision::Commit || commit.target() != Some(RESULT_TARGET) {
        return None;
    }
    commit.value()
}

fn proposal_ids(count: usize) -> Result<Vec<ProposalId>, RuntimeError> {
    (0..count)
        .map(|index| {
            let id_text = format!("{:032x}", index + 1);
            Ok(ProposalId::parse(&id_text)?)
        })
        .collect()
}

fn prismion_id(ordinal: usize) -> Result<PrismionId, RuntimeError> {
    let id_text = format!("{:032x}", 0x1000usize + ordinal);
    Ok(PrismionId::parse(&id_text)?)
}

fn progress_cursor(cycle: usize) -> Result<RunProgressCursor, RuntimeError> {
    let cursor = u64::try_from(cycle).map_err(|_error| RuntimeError::InvalidSessionConfig)?;
    Ok(RunProgressCursor::new(cursor, cursor, cursor))
}

const fn truth_result(value: u8) -> u8 {
    match value {
        VALUE_TRUE => RESULT_TRUE,
        VALUE_FALSE => RESULT_FALSE,
        _ => RESULT_UNKNOWN,
    }
}

const fn negation_result(value: u8) -> u8 {
    match value {
        VALUE_TRUE => RESULT_FALSE,
        VALUE_FALSE => RESULT_TRUE,
        _ => RESULT_UNKNOWN,
    }
}

const fn and_result(left: u8, right: u8) -> u8 {
    if left == VALUE_FALSE || right == VALUE_FALSE {
        RESULT_FALSE
    } else if left == VALUE_TRUE && right == VALUE_TRUE {
        RESULT_TRUE
    } else {
        RESULT_UNKNOWN
    }
}

const fn or_result(left: u8, right: u8) -> u8 {
    if left == VALUE_TRUE || right == VALUE_TRUE {
        RESULT_TRUE
    } else if left == VALUE_FALSE && right == VALUE_FALSE {
        RESULT_FALSE
    } else {
        RESULT_UNKNOWN
    }
}

const fn implication_result(left: u8, right: u8) -> u8 {
    match (left, right) {
        (VALUE_FALSE, _) | (_, VALUE_TRUE) => RESULT_TRUE,
        (VALUE_TRUE, VALUE_FALSE) => RESULT_FALSE,
        _ => RESULT_UNKNOWN,
    }
}

const fn contradiction_result(a: u8, not_a: u8) -> u8 {
    if a == VALUE_TRUE && not_a == 1 {
        RESULT_CONFLICT
    } else {
        RESULT_UNKNOWN
    }
}

const fn truth_state(index: usize) -> u8 {
    match index {
        0 => VALUE_TRUE,
        1 => VALUE_FALSE,
        _ => VALUE_UNKNOWN,
    }
}

const fn truth_states() -> [u8; 3] {
    [VALUE_TRUE, VALUE_FALSE, VALUE_UNKNOWN]
}

const fn uses_b(task: u8) -> bool {
    matches!(task, TASK_AND | TASK_OR | TASK_IMPLICATION)
}
