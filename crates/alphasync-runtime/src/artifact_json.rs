//! Runtime artifact JSON schema emitters.
//!
//! This module owns JSON materialization for runtime artifacts so the crate root
//! can keep authority and runtime-cycle logic separate from schema formatting.

use serde::{Deserialize, Serialize};

use alphasync_core::fabric::{
    MatrixShape, ObservationMatrix, PrismionProposal, PrismionRule, ProposalField,
};

use crate::{
    ArtifactChecksum, MAX_RUNTIME_ARTIFACT_BYTES, MAX_RUNTIME_BUNDLE_PROGRESS_ROWS,
    RuleEvaluationOutcome, RuntimeConsensusField, RuntimeConsensusReport, RuntimeCycleReport,
    RuntimeError, RuntimeFlowCommit, RuntimeSessionManifest, agency_action_code,
    completion_decision_code, consensus_source_kind_code, consensus_vote_kind_code,
    continuation_decision_code, flow_decision_code, flow_reason_code, writeout_decision_code,
};
pub(crate) fn serialize_json<T: Serialize>(value: &T) -> Result<String, RuntimeError> {
    let mut payload = serde_json::to_string_pretty(value)
        .map_err(|_error| RuntimeError::ArtifactSerialization)?;
    payload.push('\n');
    Ok(payload)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArtifactChecksumWire {
    pub(crate) name: String,
    pub(crate) bytes: u64,
    pub(crate) fnv1a64: String,
    pub(crate) sha256: String,
}

impl ArtifactChecksumWire {
    pub(crate) fn from_checksum(checksum: &ArtifactChecksum) -> Self {
        Self {
            name: checksum.name().to_owned(),
            bytes: checksum.bytes() as u64,
            fnv1a64: checksum.fnv1a64_hex().to_owned(),
            sha256: checksum.sha256_hex().to_owned(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArtifactChecksumsWire {
    pub(crate) artifacts: Vec<ArtifactChecksumWire>,
    pub(crate) schema_version: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeGenerationWire {
    pub(crate) artifacts: Vec<ArtifactChecksumWire>,
    pub(crate) commit_protocol: String,
    pub(crate) generation_id: String,
    pub(crate) schema_version: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CurrentGenerationWire {
    pub(crate) bundle_fnv1a64: String,
    pub(crate) bundle_sha256: String,
    pub(crate) generation_dir: String,
    pub(crate) generation_fnv1a64: String,
    pub(crate) generation_sha256: String,
    pub(crate) schema_version: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeSummaryWire {
    completion_decision: String,
    consensus_action: String,
    consensus_allowed_independent_sources: u64,
    consensus_allowed_source_kinds: u64,
    consensus_allowed_votes: u64,
    consensus_conflict_ppm: u64,
    consensus_score_ppm: u64,
    consensus_vote_count: u64,
    consensus_winner_independent_sources: u64,
    continuation_decision: String,
    evidence_cell_count: u64,
    flow_decision: String,
    flow_reason: String,
    patch_count: u64,
    proposal_count: u64,
    rules_evaluated: u64,
    schema_version: String,
    writeout_decision: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProgressLineWire {
    consensus_action: String,
    flow_decision: String,
    flow_reason: String,
    proposal_count: u64,
    schema_version: String,
    stage: String,
    writeout_decision: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeFrameWire {
    consensus: serde_json::Value,
    consensus_field: serde_json::Value,
    flow: serde_json::Value,
    matrix: serde_json::Value,
    progress: serde_json::Value,
    proposal_field: serde_json::Value,
    proposals: Vec<serde_json::Value>,
    rules: Vec<serde_json::Value>,
    schema_version: String,
    trace: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeBundleWire {
    pub(crate) artifact_checksums: Vec<ArtifactChecksumWire>,
    pub(crate) artifact_files: Vec<String>,
    pub(crate) progress: Vec<serde_json::Value>,
    pub(crate) latest_report: serde_json::Value,
    #[serde(default)]
    pub(crate) runtime_session: Option<serde_json::Value>,
    pub(crate) schema_version: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeSessionWire {
    cycles_completed: u64,
    cycles_requested: u64,
    schema_version: String,
    session_kind: String,
    writeout_count: u64,
}

impl RuntimeSummaryWire {
    fn validate(self) -> Result<(), RuntimeError> {
        require_schema(&self.schema_version, "alphasync.runtime_summary.v1")?;
        require_nonempty_strings([
            &self.completion_decision,
            &self.consensus_action,
            &self.continuation_decision,
            &self.flow_decision,
            &self.flow_reason,
            &self.writeout_decision,
        ])?;
        let _ = self
            .consensus_allowed_independent_sources
            .checked_add(self.consensus_allowed_source_kinds)
            .and_then(|value| value.checked_add(self.consensus_allowed_votes))
            .and_then(|value| value.checked_add(self.consensus_conflict_ppm))
            .and_then(|value| value.checked_add(self.consensus_score_ppm))
            .and_then(|value| value.checked_add(self.consensus_vote_count))
            .and_then(|value| value.checked_add(self.consensus_winner_independent_sources))
            .and_then(|value| value.checked_add(self.evidence_cell_count))
            .and_then(|value| value.checked_add(self.patch_count))
            .and_then(|value| value.checked_add(self.proposal_count))
            .and_then(|value| value.checked_add(self.rules_evaluated))
            .ok_or(RuntimeError::CorruptArtifactGeneration)?;
        Ok(())
    }
}

impl ProgressLineWire {
    fn validate(self, expected_schema: &str) -> Result<(), RuntimeError> {
        require_schema(&self.schema_version, expected_schema)?;
        require_nonempty_strings([
            &self.consensus_action,
            &self.flow_decision,
            &self.flow_reason,
            &self.stage,
            &self.writeout_decision,
        ])?;
        if self.stage != "runtime_cycle" || self.proposal_count > u64::from(u32::MAX) {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
        Ok(())
    }
}

impl RuntimeFrameWire {
    fn validate(self) -> Result<(), RuntimeError> {
        require_schema(&self.schema_version, "alphasync.runtime_frame.v2")?;
        let Self {
            consensus,
            consensus_field,
            flow,
            matrix,
            progress,
            proposal_field,
            proposals,
            rules,
            schema_version: _,
            trace,
        } = self;
        for value in [
            consensus,
            consensus_field,
            flow,
            matrix,
            progress,
            proposal_field,
            trace,
        ] {
            if matches!(value, serde_json::Value::String(_)) {
                return Err(RuntimeError::CorruptArtifactGeneration);
            }
        }
        if proposals.len() > u16::MAX as usize || rules.len() > u16::MAX as usize {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
        Ok(())
    }
}

impl RuntimeSessionWire {
    fn validate(self) -> Result<(), RuntimeError> {
        require_schema(&self.schema_version, "alphasync.runtime_session.v1")?;
        if self.cycles_requested == 0
            || self.cycles_completed == 0
            || self.writeout_count == 0
            || self.cycles_completed > self.cycles_requested
            || !matches!(
                self.session_kind.as_str(),
                "synthetic_smoke"
                    | "synthetic_scene"
                    | "logic_iq_zero"
                    | "logic_iq_consensus_scene"
            )
        {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
        Ok(())
    }
}

pub(crate) fn parse_artifact_checksums_wire(
    payload: &str,
) -> Result<ArtifactChecksumsWire, RuntimeError> {
    let wire = parse_json::<ArtifactChecksumsWire>(payload)?;
    require_schema(&wire.schema_version, "alphasync.artifact_checksums.v1")?;
    Ok(wire)
}

pub(crate) fn parse_runtime_generation_wire(
    payload: &str,
) -> Result<RuntimeGenerationWire, RuntimeError> {
    let wire = parse_json::<RuntimeGenerationWire>(payload)?;
    require_schema(&wire.schema_version, "alphasync.runtime_generation.v1")?;
    if wire.commit_protocol != "immutable_generation_directory_v1" {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    Ok(wire)
}

pub(crate) fn parse_current_generation_wire(
    payload: &str,
) -> Result<CurrentGenerationWire, RuntimeError> {
    let wire = parse_json::<CurrentGenerationWire>(payload)?;
    require_schema(&wire.schema_version, "alphasync.current_generation.v1")?;
    Ok(wire)
}

pub(crate) fn parse_runtime_bundle_wire(payload: &str) -> Result<RuntimeBundleWire, RuntimeError> {
    let wire = parse_json::<RuntimeBundleWire>(payload)?;
    require_schema(&wire.schema_version, "alphasync.runtime_bundle.v1")?;
    Ok(wire)
}

pub(crate) fn validate_runtime_artifact_schema(
    name: &str,
    payload: &str,
) -> Result<(), RuntimeError> {
    if payload.len() as u64 > MAX_RUNTIME_ARTIFACT_BYTES {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    match name {
        "artifact_checksums.json" => {
            parse_artifact_checksums_wire(payload)?;
        }
        "progress_snapshot.json" => {
            parse_json::<ProgressLineWire>(payload)?.validate("alphasync.progress_snapshot.v1")?;
        }
        "runtime_bundle.json" => {
            parse_runtime_bundle_wire(payload)?;
        }
        "runtime_frame.json" => {
            parse_json::<RuntimeFrameWire>(payload)?.validate()?;
        }
        "runtime_generation.json" => {
            parse_runtime_generation_wire(payload)?;
        }
        "runtime_session.json" => {
            parse_json::<RuntimeSessionWire>(payload)?.validate()?;
        }
        "runtime_summary.json" => {
            parse_json::<RuntimeSummaryWire>(payload)?.validate()?;
        }
        _ => return Err(RuntimeError::CorruptArtifactGeneration),
    }
    Ok(())
}

fn parse_json<T>(payload: &str) -> Result<T, RuntimeError>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_str(payload).map_err(|_| RuntimeError::CorruptArtifactGeneration)
}

fn require_schema(actual: &str, expected: &str) -> Result<(), RuntimeError> {
    if actual == expected {
        Ok(())
    } else {
        Err(RuntimeError::CorruptArtifactGeneration)
    }
}

fn require_nonempty_strings<const N: usize>(values: [&str; N]) -> Result<(), RuntimeError> {
    if values.iter().any(|value| value.is_empty()) {
        Err(RuntimeError::CorruptArtifactGeneration)
    } else {
        Ok(())
    }
}

pub(crate) fn runtime_summary_json(report: &RuntimeCycleReport) -> Result<String, RuntimeError> {
    serialize_json(&serde_json::json!({
        "completion_decision": completion_decision_code(report.completion_decision()),
        "consensus_action": report
            .consensus_report()
            .map_or("none", |consensus| agency_action_code(consensus.action())),
        "consensus_allowed_independent_sources": report
            .consensus_report()
            .map_or(0, RuntimeConsensusReport::allowed_independent_sources),
        "consensus_allowed_source_kinds": report
            .consensus_report()
            .map_or(0, RuntimeConsensusReport::allowed_source_kinds),
        "consensus_allowed_votes": report
            .consensus_report()
            .map_or(0, RuntimeConsensusReport::allowed_votes),
        "consensus_conflict_ppm": report
            .consensus_report()
            .map_or(0, RuntimeConsensusReport::conflict_ppm),
        "consensus_score_ppm": report
            .consensus_report()
            .map_or(0, RuntimeConsensusReport::score_ppm),
        "consensus_vote_count": report
            .consensus_report()
            .map_or(0, RuntimeConsensusReport::vote_count),
        "consensus_winner_independent_sources": report
            .consensus_report()
            .map_or(0, RuntimeConsensusReport::winner_independent_sources),
        "continuation_decision": continuation_decision_code(report.continuation_decision()),
        "evidence_cell_count": report.evidence_cell_count(),
        "flow_decision": report
            .flow_commit()
            .map_or("none", |commit| flow_decision_code(commit.decision())),
        "flow_reason": report
            .flow_commit()
            .map_or("none", |commit| flow_reason_code(commit.reason())),
        "patch_count": report.patch_count(),
        "proposal_count": report.proposal_count(),
        "rules_evaluated": report.rules_evaluated(),
        "schema_version": "alphasync.runtime_summary.v1",
        "writeout_decision": writeout_decision_code(report.writeout_decision()),
    }))
}

pub(crate) fn progress_snapshot_json(report: &RuntimeCycleReport) -> Result<String, RuntimeError> {
    serialize_json(&progress_line_value(
        report,
        "alphasync.progress_snapshot.v1",
    ))
}

fn progress_line_value(report: &RuntimeCycleReport, schema_version: &str) -> serde_json::Value {
    serde_json::json!({
        "consensus_action": report
            .consensus_report()
            .map_or("none", |consensus| agency_action_code(consensus.action())),
        "flow_decision": report
            .flow_commit()
            .map_or("none", |commit| flow_decision_code(commit.decision())),
        "flow_reason": report
            .flow_commit()
            .map_or("none", |commit| flow_reason_code(commit.reason())),
        "proposal_count": report.proposal_count(),
        "schema_version": schema_version,
        "stage": "runtime_cycle",
        "writeout_decision": writeout_decision_code(report.writeout_decision()),
    })
}

pub(crate) fn runtime_frame_json(
    matrix: &ObservationMatrix,
    rules: &[PrismionRule],
    report: &RuntimeCycleReport,
) -> Result<String, RuntimeError> {
    let shape = matrix.shape();
    let non_zero_cell_count = matrix
        .dense_cells()
        .iter()
        .filter(|value| **value != 0)
        .count();

    serialize_json(&serde_json::json!({
        "matrix": {
            "non_zero_cell_count": non_zero_cell_count,
            "public_redaction": "observation_values_omitted",
            "shape": shape_value(shape),
        },
        "progress": {
            "completion_decision": completion_decision_code(report.completion_decision()),
            "continuation_decision": continuation_decision_code(report.continuation_decision()),
            "writeout_decision": writeout_decision_code(report.writeout_decision()),
        },
        "consensus": consensus_value(report.consensus_report()),
        "consensus_field": consensus_field_value(report.consensus_report())?,
        "flow": flow_value(report.flow_commit()),
        "trace": trace_value(report.flow_commit()),
        "proposal_field": proposal_field_value(report.proposal_field()),
        "proposals": proposals_value(report.proposals()),
        "rules": rules_value(rules, report.rule_outcomes()),
        "schema_version": "alphasync.runtime_frame.v2",
    }))
}

fn proposal_field_value(field: &ProposalField) -> serde_json::Value {
    serde_json::json!({
        "collision_cell_count": field.collision_cell_count(),
        "conflicting_patch_count": field.conflicting_patch_count(),
        "occupied_cell_count": field.occupied_cell_count(),
        "patch_count": field.patch_count(),
        "shape": shape_value(field.shape()),
        "occupied_cells": [],
        "public_redaction": "raw_text_omitted",
    })
}

fn consensus_value(consensus_report: Option<&RuntimeConsensusReport>) -> serde_json::Value {
    let Some(consensus) = consensus_report else {
        return serde_json::Value::Null;
    };
    let candidate_patch_present =
        consensus.candidate_target().is_some() && consensus.candidate_value().is_some();
    let votes = consensus
        .vote_reports()
        .iter()
        .map(|vote| {
            serde_json::json!({
                "admitted": vote.admitted(),
                "age_ticks": vote.age_ticks(),
                "confidence_ppm": vote.confidence_ppm(),
                "evidence_quality_ppm": vote.evidence_quality_ppm(),
                "reputation_ppm": vote.reputation_ppm(),
                "source_kind": consensus_source_kind_code(vote.source_kind()),
                "source_slot": vote.source_slot(),
                "vote_kind": consensus_vote_kind_code(vote.vote_kind()),
            })
        })
        .collect::<Vec<_>>();

    serde_json::json!({
        "action": agency_action_code(consensus.action()),
        "allowed_votes": consensus.allowed_votes(),
        "allowed_independent_sources": consensus.allowed_independent_sources(),
        "allowed_source_kinds": consensus.allowed_source_kinds(),
        "blocked_votes": consensus.blocked_votes(),
        "candidate_patch_present": candidate_patch_present,
        "conflict_ppm": consensus.conflict_ppm(),
        "score_ppm": consensus.score_ppm(),
        "stale_ppm": consensus.stale_ppm(),
        "vote_count": consensus.vote_count(),
        "winner_independent_sources": consensus.winner_independent_sources(),
        "votes": votes,
    })
}

fn consensus_field_value(
    consensus_report: Option<&RuntimeConsensusReport>,
) -> Result<serde_json::Value, RuntimeError> {
    let Some(consensus) = consensus_report else {
        return Ok(serde_json::Value::Null);
    };

    let field = RuntimeConsensusField::from_report(consensus)?;
    let shape = field.shape();
    let channel_schema = RuntimeConsensusField::channel_schema()
        .iter()
        .map(|channel| channel.as_str())
        .collect::<Vec<_>>();

    Ok(serde_json::json!({
        "channel_schema": channel_schema,
        "non_zero_cell_count": field.non_zero_cells().len(),
        "public_redaction": "consensus_cell_values_omitted",
        "shape": shape_value(shape),
    }))
}

fn flow_value(flow_commit: Option<RuntimeFlowCommit>) -> serde_json::Value {
    let Some(commit) = flow_commit else {
        return serde_json::Value::Null;
    };
    let committed_cell_present = commit.target().is_some() && commit.value().is_some();
    serde_json::json!({
        "committed_cell_present": committed_cell_present,
        "decision": flow_decision_code(commit.decision()),
        "reason": flow_reason_code(commit.reason()),
        "shape": shape_value(commit.flow_shape()),
    })
}

fn trace_value(flow_commit: Option<RuntimeFlowCommit>) -> serde_json::Value {
    let Some(commit) = flow_commit else {
        return serde_json::Value::Null;
    };
    serde_json::json!({
        "decision": flow_decision_code(commit.decision()),
        "evidence_cell_count": commit.evidence_cell_count(),
        "patch_count": commit.patch_count(),
        "proposal_count": commit.proposal_count(),
        "reason": flow_reason_code(commit.reason()),
        "schema_version": "alphasync.trace.v1",
        "sequence": commit.sequence(),
    })
}

pub(crate) fn runtime_bundle_json(
    report: &RuntimeCycleReport,
    checksums: &[ArtifactChecksum],
    progress_reports: &[RuntimeCycleReport],
    session: Option<&RuntimeSessionManifest>,
) -> Result<String, RuntimeError> {
    let artifact_files = checksums
        .iter()
        .map(ArtifactChecksum::name)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let progress = progress_bundle_tail(progress_reports)
        .iter()
        .map(|report| progress_line_value(report, "alphasync.progress.v1"))
        .collect::<Vec<_>>();

    let latest_report = serde_json::json!({
            "completion_decision": completion_decision_code(report.completion_decision()),
            "proposal_count": report.proposal_count(),
            "writeout_decision": writeout_decision_code(report.writeout_decision()),
    });

    serialize_json(&RuntimeBundleWire {
        artifact_checksums: artifact_checksums_wire(checksums),
        artifact_files,
        progress,
        latest_report,
        runtime_session: session.map(runtime_session_value),
        schema_version: "alphasync.runtime_bundle.v1".to_owned(),
    })
}

fn progress_bundle_tail(reports: &[RuntimeCycleReport]) -> &[RuntimeCycleReport] {
    let start = reports
        .len()
        .saturating_sub(MAX_RUNTIME_BUNDLE_PROGRESS_ROWS);
    &reports[start..]
}

pub(crate) fn push_bounded_progress_report(
    reports: &mut Vec<RuntimeCycleReport>,
    report: RuntimeCycleReport,
) {
    reports.push(report);
    let overflow = reports
        .len()
        .saturating_sub(MAX_RUNTIME_BUNDLE_PROGRESS_ROWS);
    if overflow > 0 {
        reports.drain(0..overflow);
    }
}

pub(crate) fn runtime_session_json(
    session: &RuntimeSessionManifest,
) -> Result<String, RuntimeError> {
    serialize_json(&runtime_session_value(session))
}

fn runtime_session_value(session: &RuntimeSessionManifest) -> serde_json::Value {
    serde_json::json!({
        "cycles_completed": session.cycles_completed(),
        "cycles_requested": session.cycles_requested(),
        "schema_version": "alphasync.runtime_session.v1",
        "session_kind": session.session_kind().as_str(),
        "writeout_count": session.writeout_count(),
    })
}

pub(crate) fn artifact_checksums_json(
    checksums: &[ArtifactChecksum],
) -> Result<String, RuntimeError> {
    serialize_json(&ArtifactChecksumsWire {
        artifacts: artifact_checksums_wire(checksums),
        schema_version: "alphasync.artifact_checksums.v1".to_owned(),
    })
}

fn artifact_checksums_wire(checksums: &[ArtifactChecksum]) -> Vec<ArtifactChecksumWire> {
    checksums
        .iter()
        .map(ArtifactChecksumWire::from_checksum)
        .collect()
}

pub(crate) fn runtime_generation_json(
    generation_source: &ArtifactChecksum,
    checksums: &[ArtifactChecksum],
) -> Result<String, RuntimeError> {
    serialize_json(&RuntimeGenerationWire {
        artifacts: artifact_checksums_wire(checksums),
        commit_protocol: "immutable_generation_directory_v1".to_owned(),
        generation_id: generation_source.sha256_hex().to_owned(),
        schema_version: "alphasync.runtime_generation.v1".to_owned(),
    })
}

pub(crate) fn current_generation_json(
    generation_dir_name: &str,
    generation_checksum: &ArtifactChecksum,
    bundle_checksum: &ArtifactChecksum,
) -> Result<String, RuntimeError> {
    serialize_json(&CurrentGenerationWire {
        bundle_fnv1a64: bundle_checksum.fnv1a64_hex().to_owned(),
        bundle_sha256: bundle_checksum.sha256_hex().to_owned(),
        generation_dir: format!("generations/{generation_dir_name}"),
        generation_fnv1a64: generation_checksum.fnv1a64_hex().to_owned(),
        generation_sha256: generation_checksum.sha256_hex().to_owned(),
        schema_version: "alphasync.current_generation.v1".to_owned(),
    })
}

fn shape_value(shape: MatrixShape) -> serde_json::Value {
    serde_json::json!({
        "cell_count": shape.cell_count(),
        "columns": shape.columns(),
        "planes": shape.planes(),
        "rows": shape.rows(),
    })
}

fn rules_value(rules: &[PrismionRule], outcomes: &[RuleEvaluationOutcome]) -> serde_json::Value {
    let rules = rules
        .iter()
        .enumerate()
        .map(|(index, rule)| {
            let activated = outcomes
                .get(index)
                .is_some_and(RuleEvaluationOutcome::activated);
            serde_json::json!({
                "activated": activated,
                "condition_count": rule.conditions().len(),
                "confidence_ppm": rule.confidence().as_ppm(),
                "patch_count": rule.patches().len(),
            })
        })
        .collect::<Vec<_>>();
    serde_json::Value::Array(rules)
}

fn proposals_value(proposals: &[PrismionProposal]) -> serde_json::Value {
    let proposals = proposals
        .iter()
        .map(|proposal| {
            serde_json::json!({
                "confidence_ppm": proposal.confidence().as_ppm(),
                "evidence_cell_count": proposal.evidence_cells().len(),
                "patch_count": proposal.patches().len(),
            })
        })
        .collect::<Vec<_>>();
    serde_json::Value::Array(proposals)
}
