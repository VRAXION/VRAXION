//! Deterministic synthetic runtime entrypoints.
//!
//! These helpers are the canonical local smoke/scene producers used by the
//! CLI and examples. They do not read datasets, infer semantic truth, or call
//! external services. Each function writes the same safe artifact set expected
//! by local operator tooling.

use std::path::{Path, PathBuf};
use std::time::Duration;
use std::time::Instant;

use alphasync_core::fabric::{
    CellAddress, CellComparison, CellPredicate, CellSample, ConfidencePpm, MatrixShape,
    ObservationMatrix, PrismionRule, ProposalPatch,
};
use alphasync_core::ids::{PrismionId, ProposalId};
use alphasync_core::progress::{
    EvidenceSlot, RunProgressCursor, WriteoutCadence, WriteoutDecision,
};

use crate::{
    AlphaSyncRuntime, ArtifactChecksum, RuntimeArtifactSnapshot, RuntimeArtifactWriter,
    RuntimeError, RuntimePolicy, RuntimeSessionKind, RuntimeSessionManifest,
    push_bounded_progress_report,
};

/// Default smoke iteration count.
pub const DEFAULT_SMOKE_ITERATIONS: usize = 1_000_000;

/// Default smoke wall-clock writeout interval in milliseconds.
pub const DEFAULT_SMOKE_WRITEOUT_INTERVAL_MS: u64 = 20_000;

/// Default smoke output directory, relative to the current working directory.
pub const DEFAULT_SMOKE_RUN_DIR: &str = "runtime_runs/e204_smoke_001";

/// Default scene output directory, relative to the current working directory.
pub const DEFAULT_SCENE_RUN_DIR: &str = "runtime_runs/e204_scene_001";

/// Default repeated-session cycle count.
pub const DEFAULT_SESSION_CYCLES: usize = 1_000;

/// Default repeated-session writeout interval, in cycles.
pub const DEFAULT_SESSION_WRITE_EVERY: usize = 100;

/// Default repeated-session output directory, relative to current working directory.
pub const DEFAULT_SESSION_RUN_DIR: &str = "runtime_runs/e206_session_001";

const SMOKE_PRISMION_HEX: &str = "0123456789abcdef0123456789abcdef";

/// Configuration for a synthetic smoke run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SmokeRunConfig {
    iterations: usize,
    out_dir: PathBuf,
}

impl SmokeRunConfig {
    /// Creates a synthetic smoke run configuration.
    #[must_use]
    pub fn new(iterations: usize, out_dir: PathBuf) -> Self {
        Self {
            iterations,
            out_dir,
        }
    }

    /// Returns the configured iteration count.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns the configured output directory.
    #[must_use]
    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }
}

/// Configuration for a repeated synthetic runtime session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionRunConfig {
    cycles: usize,
    write_every: usize,
    out_dir: PathBuf,
}

impl SessionRunConfig {
    /// Creates a repeated synthetic runtime session configuration.
    #[must_use]
    pub fn new(cycles: usize, write_every: usize, out_dir: PathBuf) -> Self {
        Self {
            cycles,
            write_every,
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

    /// Returns output directory.
    #[must_use]
    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }
}

/// Safe aggregate result from a synthetic smoke run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SmokeRunResult {
    iterations: usize,
    proposal_total: usize,
    patch_total: usize,
    elapsed: Duration,
    cycles_per_second: u128,
    out_dir: PathBuf,
    artifact_checksum: ArtifactChecksum,
    bundle_checksum: ArtifactChecksum,
}

impl SmokeRunResult {
    /// Returns the completed iteration count.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns the total proposal count accumulated across the loop.
    #[must_use]
    pub const fn proposal_total(&self) -> usize {
        self.proposal_total
    }

    /// Returns the total proposal patch count accumulated across the loop.
    #[must_use]
    pub const fn patch_total(&self) -> usize {
        self.patch_total
    }

    /// Returns the measured elapsed duration.
    #[must_use]
    pub const fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// Returns deterministic display throughput derived from elapsed time.
    #[must_use]
    pub const fn cycles_per_second(&self) -> u128 {
        self.cycles_per_second
    }

    /// Returns the output directory used by the run.
    #[must_use]
    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }

    /// Returns the checksum artifact record.
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

/// Safe aggregate result from a repeated synthetic runtime session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionRunResult {
    cycles: usize,
    writeout_count: usize,
    proposal_total: usize,
    patch_total: usize,
    elapsed: Duration,
    cycles_per_second: u128,
    proposals_per_second: u128,
    patches_per_second: u128,
    out_dir: PathBuf,
    artifact_checksum: ArtifactChecksum,
    bundle_checksum: ArtifactChecksum,
}

impl SessionRunResult {
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

    /// Returns total proposal count accumulated across the session.
    #[must_use]
    pub const fn proposal_total(&self) -> usize {
        self.proposal_total
    }

    /// Returns total patch count accumulated across the session.
    #[must_use]
    pub const fn patch_total(&self) -> usize {
        self.patch_total
    }

    /// Returns the measured elapsed session duration.
    #[must_use]
    pub const fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// Returns cycle throughput derived from elapsed session duration.
    #[must_use]
    pub const fn cycles_per_second(&self) -> u128 {
        self.cycles_per_second
    }

    /// Returns proposal throughput derived from elapsed session duration.
    #[must_use]
    pub const fn proposals_per_second(&self) -> u128 {
        self.proposals_per_second
    }

    /// Returns patch throughput derived from elapsed session duration.
    #[must_use]
    pub const fn patches_per_second(&self) -> u128 {
        self.patches_per_second
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

/// Safe aggregate result from a synthetic scene run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SceneRunResult {
    rules: usize,
    activated_rules: usize,
    proposals: usize,
    patches: usize,
    evidence_cells: usize,
    out_dir: PathBuf,
    artifact_checksum: ArtifactChecksum,
    bundle_checksum: ArtifactChecksum,
}

impl SceneRunResult {
    /// Returns the number of Prismion rules evaluated.
    #[must_use]
    pub const fn rules(&self) -> usize {
        self.rules
    }

    /// Returns how many Prismion rules emitted a proposal.
    #[must_use]
    pub const fn activated_rules(&self) -> usize {
        self.activated_rules
    }

    /// Returns the proposal count.
    #[must_use]
    pub const fn proposals(&self) -> usize {
        self.proposals
    }

    /// Returns the proposal patch count.
    #[must_use]
    pub const fn patches(&self) -> usize {
        self.patches
    }

    /// Returns the evidence-cell count.
    #[must_use]
    pub const fn evidence_cells(&self) -> usize {
        self.evidence_cells
    }

    /// Returns the output directory used by the run.
    #[must_use]
    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }

    /// Returns the checksum artifact record.
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

/// Runs the canonical deterministic synthetic smoke.
///
/// # Errors
///
/// Returns [`RuntimeError`] when core runtime evaluation, policy construction,
/// artifact writing, or synthetic matrix/rule construction fails.
#[allow(clippy::too_many_lines)]
pub fn run_smoke(config: &SmokeRunConfig) -> Result<SmokeRunResult, RuntimeError> {
    let shape = MatrixShape::new(1, 4, 4)?;
    let matrix = smoke_matrix(shape)?;
    let policy = default_policy(shape)?;
    let rules = [smoke_rule()?];
    let proposal_ids = [proposal_id(0)?];
    let mut last_writeout = Some(RunProgressCursor::new(0, 0, 0));
    let initial_progress = RunProgressCursor::new(0, 0, 0);

    let mut report = AlphaSyncRuntime::run_cycle(
        policy,
        &matrix,
        &rules,
        &proposal_ids,
        last_writeout,
        initial_progress,
    )?;
    let writer = RuntimeArtifactWriter::open_or_create(config.out_dir())?;
    let mut progress_reports = Vec::new();
    push_bounded_progress_report(&mut progress_reports, report.clone());
    let snapshot = RuntimeArtifactSnapshot::new(&matrix, &rules, &report, &progress_reports, None)?;
    let generation = writer.write_runtime_generation(&snapshot)?;
    let mut latest_artifact_checksum = generation.artifact_checksum().clone();
    let mut latest_bundle_checksum = generation.bundle_checksum().clone();
    let mut writeout_count = 1usize;
    last_writeout = Some(initial_progress);

    let start = Instant::now();
    let mut proposal_total = 0usize;
    let mut patch_total = 0usize;
    for iteration in 1..=config.iterations() {
        let current_progress = progress_cursor_since(start, iteration)?;
        report = AlphaSyncRuntime::run_cycle(
            policy,
            &matrix,
            &rules,
            &proposal_ids,
            last_writeout,
            current_progress,
        )?;
        proposal_total = proposal_total
            .checked_add(report.proposal_count())
            .ok_or(RuntimeError::InvalidSessionConfig)?;
        patch_total = patch_total
            .checked_add(report.patch_count())
            .ok_or(RuntimeError::InvalidSessionConfig)?;

        if writeout_due(report.writeout_decision()) {
            writeout_count = writeout_count
                .checked_add(1)
                .ok_or(RuntimeError::InvalidSessionConfig)?;
            push_bounded_progress_report(&mut progress_reports, report.clone());
            let session = RuntimeSessionManifest::new(
                RuntimeSessionKind::SyntheticSmoke,
                config.iterations(),
                iteration,
                writeout_count,
            )?;
            let snapshot = RuntimeArtifactSnapshot::new(
                &matrix,
                &rules,
                &report,
                &progress_reports,
                Some(&session),
            )?;
            let generation = writer.write_runtime_generation(&snapshot)?;
            latest_artifact_checksum = generation.artifact_checksum().clone();
            latest_bundle_checksum = generation.bundle_checksum().clone();
            last_writeout = Some(current_progress);
        }
    }
    let elapsed = start.elapsed();
    let elapsed_ns = elapsed.as_nanos().max(1);
    let cycles_per_second = (config.iterations() as u128 * 1_000_000_000u128) / elapsed_ns;

    if config.iterations() > 0 {
        writeout_count = writeout_count
            .checked_add(1)
            .ok_or(RuntimeError::InvalidSessionConfig)?;
        push_bounded_progress_report(&mut progress_reports, report.clone());
        let session = RuntimeSessionManifest::new(
            RuntimeSessionKind::SyntheticSmoke,
            config.iterations(),
            config.iterations(),
            writeout_count,
        )?;
        let snapshot = RuntimeArtifactSnapshot::new(
            &matrix,
            &rules,
            &report,
            &progress_reports,
            Some(&session),
        )?;
        let generation = writer.write_runtime_generation(&snapshot)?;
        latest_artifact_checksum = generation.artifact_checksum().clone();
        latest_bundle_checksum = generation.bundle_checksum().clone();
    }

    Ok(SmokeRunResult {
        iterations: config.iterations(),
        proposal_total,
        patch_total,
        elapsed,
        cycles_per_second,
        out_dir: config.out_dir().to_path_buf(),
        artifact_checksum: latest_artifact_checksum,
        bundle_checksum: latest_bundle_checksum,
    })
}

/// Runs the canonical deterministic multi-rule synthetic scene.
///
/// # Errors
///
/// Returns [`RuntimeError`] when core runtime evaluation, policy construction,
/// artifact writing, or synthetic matrix/rule construction fails.
pub fn run_scene(out_dir: &Path) -> Result<SceneRunResult, RuntimeError> {
    let shape = MatrixShape::new(1, 16, 16)?;
    let matrix = ObservationMatrix::from_sparse(shape, scene_cells()?)?;
    let rules = scene_rules()?;
    let proposal_ids = proposal_ids(rules.len())?;
    let report = AlphaSyncRuntime::run_cycle(
        default_policy(shape)?,
        &matrix,
        &rules,
        &proposal_ids,
        Some(RunProgressCursor::new(0, 0, 0)),
        RunProgressCursor::new(1, 1, 1),
    )?;
    let writer = RuntimeArtifactWriter::open_or_create(out_dir)?;
    let progress_reports = [report.clone()];
    let snapshot = RuntimeArtifactSnapshot::new(&matrix, &rules, &report, &progress_reports, None)?;
    let generation = writer.write_runtime_generation(&snapshot)?;
    let activated_rules = report
        .rule_outcomes()
        .iter()
        .filter(|outcome| outcome.activated())
        .count();

    Ok(SceneRunResult {
        rules: rules.len(),
        activated_rules,
        proposals: report.proposal_count(),
        patches: report.patch_count(),
        evidence_cells: report.evidence_cell_count(),
        out_dir: out_dir.to_path_buf(),
        artifact_checksum: generation.artifact_checksum().clone(),
        bundle_checksum: generation.bundle_checksum().clone(),
    })
}

/// Runs a repeated deterministic synthetic scene session with periodic writes.
///
/// # Errors
///
/// Returns [`RuntimeError`] when the session configuration is invalid, runtime
/// evaluation fails, or durable artifact writing fails.
pub fn run_session(config: &SessionRunConfig) -> Result<SessionRunResult, RuntimeError> {
    if config.cycles() == 0 || config.write_every() == 0 {
        return Err(RuntimeError::InvalidSessionConfig);
    }

    let shape = MatrixShape::new(1, 16, 16)?;
    let matrix = ObservationMatrix::from_sparse(shape, scene_cells()?)?;
    let rules = scene_rules()?;
    let proposal_ids = proposal_ids(rules.len())?;
    let policy = session_policy(shape)?;
    let writer = RuntimeArtifactWriter::open_or_create(config.out_dir())?;
    let mut last_writeout = None;
    let mut progress_reports = Vec::new();
    let mut proposal_total = 0usize;
    let mut patch_total = 0usize;
    let mut writeout_count = 0usize;
    let mut latest_artifact_checksum = None;
    let mut latest_bundle_checksum = None;
    let start = Instant::now();

    for cycle in 1..=config.cycles() {
        let current_progress = progress_cursor_since(start, cycle)?;
        let report = AlphaSyncRuntime::run_cycle(
            policy,
            &matrix,
            &rules,
            &proposal_ids,
            last_writeout,
            current_progress,
        )?;
        proposal_total += report.proposal_count();
        patch_total += report.patch_count();

        if cycle % config.write_every() == 0
            || cycle == config.cycles()
            || writeout_due(report.writeout_decision())
        {
            writeout_count += 1;
            push_bounded_progress_report(&mut progress_reports, report.clone());

            let session = RuntimeSessionManifest::new(
                RuntimeSessionKind::SyntheticScene,
                config.cycles(),
                cycle,
                writeout_count,
            )?;
            let snapshot = RuntimeArtifactSnapshot::new(
                &matrix,
                &rules,
                &report,
                &progress_reports,
                Some(&session),
            )?;
            let generation = writer.write_runtime_generation(&snapshot)?;
            latest_artifact_checksum = Some(generation.artifact_checksum().clone());
            latest_bundle_checksum = Some(generation.bundle_checksum().clone());
            last_writeout = Some(current_progress);
        }
    }

    let elapsed = start.elapsed();
    let elapsed_ns = elapsed.as_nanos().max(1);
    let cycles_per_second = (config.cycles() as u128 * 1_000_000_000u128) / elapsed_ns;
    let proposals_per_second = (proposal_total as u128 * 1_000_000_000u128) / elapsed_ns;
    let patches_per_second = (patch_total as u128 * 1_000_000_000u128) / elapsed_ns;

    let Some(artifact_checksum) = latest_artifact_checksum else {
        return Err(RuntimeError::InvalidSessionConfig);
    };
    let Some(bundle_checksum) = latest_bundle_checksum else {
        return Err(RuntimeError::InvalidSessionConfig);
    };

    Ok(SessionRunResult {
        cycles: config.cycles(),
        writeout_count,
        proposal_total,
        patch_total,
        elapsed,
        cycles_per_second,
        proposals_per_second,
        patches_per_second,
        out_dir: config.out_dir().to_path_buf(),
        artifact_checksum,
        bundle_checksum,
    })
}

fn default_policy(shape: MatrixShape) -> Result<RuntimePolicy, RuntimeError> {
    RuntimePolicy::new(
        shape,
        WriteoutCadence::new(DEFAULT_SMOKE_WRITEOUT_INTERVAL_MS, u64::MAX, u64::MAX)?,
        EvidenceSlot::new(0)?,
        EvidenceSlot::new(1)?,
    )
}

fn session_policy(shape: MatrixShape) -> Result<RuntimePolicy, RuntimeError> {
    RuntimePolicy::new(
        shape,
        WriteoutCadence::frontier_default(),
        EvidenceSlot::new(0)?,
        EvidenceSlot::new(1)?,
    )
}

fn smoke_matrix(shape: MatrixShape) -> Result<ObservationMatrix, RuntimeError> {
    Ok(ObservationMatrix::from_sparse(
        shape,
        vec![
            CellSample::new(address(0, 0, 0), 7)?,
            CellSample::new(address(0, 0, 1), 3)?,
        ],
    )?)
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

fn smoke_rule() -> Result<PrismionRule, RuntimeError> {
    Ok(PrismionRule::new(
        PrismionId::parse(SMOKE_PRISMION_HEX)?,
        vec![
            CellPredicate::new(address(0, 0, 0), CellComparison::Gte, 5),
            CellPredicate::new(address(0, 0, 1), CellComparison::Eq, 3),
        ],
        vec![ProposalPatch::new(address(0, 1, 1), 1)],
        ConfidencePpm::new(900_000)?,
    )?)
}

fn scene_cells() -> Result<Vec<CellSample>, RuntimeError> {
    let cells = [
        (0, 0, 0, 9),
        (0, 0, 1, 4),
        (0, 0, 5, 11),
        (0, 1, 2, 8),
        (0, 1, 8, 6),
        (0, 2, 3, 7),
        (0, 2, 4, 12),
        (0, 3, 6, 5),
        (0, 3, 7, 14),
        (0, 4, 4, 3),
        (0, 4, 9, 10),
        (0, 5, 5, 13),
        (0, 5, 12, 2),
        (0, 6, 1, 15),
        (0, 6, 10, 9),
        (0, 7, 7, 4),
        (0, 8, 2, 6),
        (0, 8, 11, 8),
        (0, 9, 3, 5),
        (0, 9, 13, 12),
        (0, 10, 0, 7),
        (0, 10, 14, 3),
        (0, 11, 6, 10),
        (0, 12, 8, 11),
        (0, 13, 9, 4),
        (0, 14, 10, 6),
        (0, 15, 15, 16),
    ];

    cells
        .iter()
        .map(|&(plane, row, column, value)| {
            Ok(CellSample::new(address(plane, row, column), value)?)
        })
        .collect()
}

fn scene_rules() -> Result<Vec<PrismionRule>, RuntimeError> {
    let mut rules = active_scene_rules()?;
    rules.extend(inactive_scene_rules()?);
    Ok(rules)
}

fn active_scene_rules() -> Result<Vec<PrismionRule>, RuntimeError> {
    Ok(vec![
        scene_rule(
            1,
            &[
                (0, 0, 0, CellComparison::Gte, 8),
                (0, 0, 1, CellComparison::Eq, 4),
            ],
            &[(0, 1, 1, 21), (0, 1, 2, 22)],
            930_000,
        )?,
        scene_rule(
            2,
            &[
                (0, 2, 4, CellComparison::Gt, 10),
                (0, 3, 7, CellComparison::Gte, 14),
            ],
            &[(0, 2, 2, 31), (0, 2, 3, 32)],
            910_000,
        )?,
        scene_rule(
            3,
            &[
                (0, 5, 5, CellComparison::Eq, 13),
                (0, 6, 1, CellComparison::Gte, 12),
            ],
            &[(0, 3, 3, 41), (0, 3, 4, 42)],
            880_000,
        )?,
        scene_rule(
            4,
            &[
                (0, 8, 11, CellComparison::Eq, 8),
                (0, 9, 13, CellComparison::Gte, 12),
            ],
            &[(0, 4, 4, 51)],
            860_000,
        )?,
        scene_rule(
            5,
            &[
                (0, 10, 0, CellComparison::Eq, 7),
                (0, 15, 15, CellComparison::Gte, 16),
            ],
            &[(0, 5, 5, 61), (0, 5, 6, 62)],
            840_000,
        )?,
    ])
}

fn inactive_scene_rules() -> Result<Vec<PrismionRule>, RuntimeError> {
    Ok(vec![
        scene_rule(
            6,
            &[
                (0, 0, 5, CellComparison::Lt, 4),
                (0, 1, 8, CellComparison::Eq, 6),
            ],
            &[(0, 6, 6, 71)],
            700_000,
        )?,
        scene_rule(
            7,
            &[
                (0, 4, 4, CellComparison::Gt, 8),
                (0, 4, 9, CellComparison::Eq, 10),
            ],
            &[(0, 7, 7, 81)],
            720_000,
        )?,
        scene_rule(
            8,
            &[
                (0, 12, 8, CellComparison::Eq, 10),
                (0, 13, 9, CellComparison::Eq, 4),
            ],
            &[(0, 8, 8, 91)],
            730_000,
        )?,
        scene_rule(
            9,
            &[
                (0, 14, 10, CellComparison::Gte, 7),
                (0, 15, 15, CellComparison::Lt, 8),
            ],
            &[(0, 9, 9, 101)],
            740_000,
        )?,
        scene_rule(
            10,
            &[
                (0, 1, 2, CellComparison::Eq, 8),
                (0, 11, 6, CellComparison::Lt, 4),
            ],
            &[(0, 10, 10, 111)],
            750_000,
        )?,
    ])
}

fn scene_rule(
    id: u128,
    conditions: &[(u16, u16, u16, CellComparison, u8)],
    patches: &[(u16, u16, u16, u8)],
    confidence: u32,
) -> Result<PrismionRule, RuntimeError> {
    Ok(PrismionRule::new(
        prismion_id(id)?,
        conditions
            .iter()
            .map(|&(plane, row, column, comparison, value)| {
                CellPredicate::new(address(plane, row, column), comparison, value)
            })
            .collect(),
        patches
            .iter()
            .map(|&(plane, row, column, value)| {
                ProposalPatch::new(address(plane, row, column), value)
            })
            .collect(),
        ConfidencePpm::new(confidence)?,
    )?)
}

fn proposal_ids(count: usize) -> Result<Vec<ProposalId>, RuntimeError> {
    (0..count)
        .map(|index| Ok(ProposalId::parse(&format!("{index:032x}"))?))
        .collect()
}

fn proposal_id(index: u128) -> Result<ProposalId, RuntimeError> {
    Ok(ProposalId::parse(&format!("{index:032x}"))?)
}

fn prismion_id(index: u128) -> Result<PrismionId, RuntimeError> {
    Ok(PrismionId::parse(&format!("{index:032x}"))?)
}

const fn address(plane: u16, row: u16, column: u16) -> CellAddress {
    CellAddress::new(plane, row, column)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{SessionRunConfig, SmokeRunConfig, run_session, run_smoke};

    fn test_run_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("alphasync_synthetic_{name}_{}", std::process::id()))
    }

    fn legacy_progress_name() -> String {
        ["progress", concat!("json", "l")].join(".")
    }

    #[test]
    fn session_writes_initial_and_final_even_when_cycle_interval_is_large() {
        let out_dir = test_run_dir("initial_final_writeout");
        let _ = fs::remove_dir_all(&out_dir);

        let result =
            run_session(&SessionRunConfig::new(3, 999, out_dir.clone())).expect("session succeeds");

        assert_eq!(result.writeout_count(), 2);
        assert!(out_dir.join("current_generation.json").is_file());

        assert!(!out_dir.join(legacy_progress_name()).exists());
    }

    #[test]
    fn smoke_writes_initial_and_final_artifacts() {
        let out_dir = test_run_dir("smoke_initial_final_writeout");
        let _ = fs::remove_dir_all(&out_dir);

        let result = run_smoke(&SmokeRunConfig::new(3, out_dir.clone())).expect("smoke succeeds");

        assert_eq!(result.iterations(), 3);
        assert!(out_dir.join("current_generation.json").is_file());

        assert!(!out_dir.join(legacy_progress_name()).exists());
    }
}
