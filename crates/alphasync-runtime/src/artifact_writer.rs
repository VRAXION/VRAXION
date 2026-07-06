//! Runtime artifact writer, generation validation, and checksum authority.
//!
//! This module owns filesystem authority and artifact transaction mechanics.
//! The crate root owns runtime-cycle semantics; `artifact_json` owns schema
//! materialization.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

#[cfg(windows)]
use std::os::windows::fs::MetadataExt;

use fs2::FileExt;
use sha2::{Digest, Sha256};

use alphasync_core::fabric::{
    MAX_EVIDENCE_CELLS_PER_CYCLE, MAX_PROPOSAL_PATCHES_PER_FIELD, MAX_PROPOSALS_PER_FIELD,
    ObservationMatrix, PrismionRule,
};

use crate::artifact_json::{
    ArtifactChecksumWire, artifact_checksums_json, current_generation_json,
    parse_artifact_checksums_wire, parse_current_generation_wire, parse_runtime_bundle_wire,
    parse_runtime_generation_wire, progress_snapshot_json, runtime_bundle_json, runtime_frame_json,
    runtime_generation_json, runtime_session_json, runtime_summary_json,
    validate_runtime_artifact_schema,
};
use crate::{
    DURABLE_WRITE_COUNTER, GENERATION_LOCK_FILE, GENERATION_PENDING_DIR, GENERATION_TMP_ROOT,
    MAX_RUNTIME_ARTIFACT_BYTES, MAX_RUNTIME_BUNDLE_PROGRESS_ROWS,
    MAX_RUNTIME_GENERATION_RETAINED_DIRS, RUNTIME_ROOT_MARKER, RUNTIME_ROOT_MARKER_BODY,
    RuntimeCycleReport, RuntimeError, RuntimeSessionManifest, fnv1a64,
};
/// Checksum record for one runtime artifact.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ArtifactChecksum {
    name: String,
    bytes: usize,
    fnv1a64_hex: String,
    sha256_hex: String,
}

impl ArtifactChecksum {
    /// Returns the artifact file name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns artifact byte length.
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    /// Returns lowercase hex FNV-1a 64-bit digest.
    ///
    /// This is a local runtime integrity checksum, not a cryptographic registry
    /// identity. Registry/manifest identity should use a separately frozen
    /// SHA-256 path.
    #[must_use]
    pub fn fnv1a64_hex(&self) -> &str {
        &self.fnv1a64_hex
    }

    /// Returns lowercase hex SHA-256 digest.
    ///
    /// This is the content-addressed identity used for immutable generation
    /// directories and public release-boundary checks. FNV remains only a
    /// compact diagnostic checksum for older local GUI summaries.
    #[must_use]
    pub fn sha256_hex(&self) -> &str {
        &self.sha256_hex
    }
}

/// Result of one transactionally committed runtime artifact generation.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimeArtifactGeneration {
    checksum_index: ArtifactChecksum,
    bundle_record: ArtifactChecksum,
    commit_marker: ArtifactChecksum,
}

impl RuntimeArtifactGeneration {
    /// Returns the checksum manifest artifact record.
    #[must_use]
    pub const fn artifact_checksum(&self) -> &ArtifactChecksum {
        &self.checksum_index
    }

    /// Returns the one-file GUI/runtime bundle artifact record.
    #[must_use]
    pub const fn bundle_checksum(&self) -> &ArtifactChecksum {
        &self.bundle_record
    }

    /// Returns the generation commit manifest artifact record.
    #[must_use]
    pub const fn generation_checksum(&self) -> &ArtifactChecksum {
        &self.commit_marker
    }
}

/// Opaque, bounded input for one runtime artifact generation.
///
/// This type deliberately has no public constructor. Public callers can run the
/// runtime demos/sessions, but they cannot synthesize an arbitrary
/// matrix/rule/report combination and ask the writer to publish it as an
/// Agency-owned generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeArtifactSnapshot {
    matrix: ObservationMatrix,
    rules: Vec<PrismionRule>,
    report: RuntimeCycleReport,
    progress_reports: Vec<RuntimeCycleReport>,
    session: Option<RuntimeSessionManifest>,
}

impl RuntimeArtifactSnapshot {
    pub(crate) fn new(
        matrix: &ObservationMatrix,
        rules: &[PrismionRule],
        report: &RuntimeCycleReport,
        progress_reports: &[RuntimeCycleReport],
        session: Option<&RuntimeSessionManifest>,
    ) -> Result<Self, RuntimeError> {
        preflight_artifact_snapshot(matrix, rules, report, progress_reports)?;
        Ok(Self {
            matrix: matrix.clone(),
            rules: rules.to_vec(),
            report: report.clone(),
            progress_reports: progress_reports.to_vec(),
            session: session.copied(),
        })
    }

    fn matrix(&self) -> &ObservationMatrix {
        &self.matrix
    }

    fn rules(&self) -> &[PrismionRule] {
        &self.rules
    }

    fn report(&self) -> &RuntimeCycleReport {
        &self.report
    }

    fn progress_reports(&self) -> &[RuntimeCycleReport] {
        &self.progress_reports
    }

    fn session(&self) -> Option<&RuntimeSessionManifest> {
        self.session.as_ref()
    }
}

/// Atomic process-crash runtime artifact writer.
///
/// This writer is deliberately small. It follows the crate's public artifact
/// rule: write a full temp file, flush it, replace the target, and expose a
/// checksum record. The v0.1 guarantee is coherent visibility after
/// ordinary process crashes; sudden OS or power loss directory-entry durability
/// is explicitly outside the public claim on platforms that do not expose a
/// portable fallible directory sync through stable Rust.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimeArtifactWriter {
    run_dir: PathBuf,
}

impl RuntimeArtifactWriter {
    /// Opens or claims a runtime output directory.
    ///
    /// A directory without the `AlphaSync` ownership marker must be empty. This
    /// prevents public callers from pointing the writer at an arbitrary folder
    /// and letting retention or migration cleanup delete unrelated files.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::UnownedArtifactDirectory`] when the directory is
    /// non-empty and not already marked as an `AlphaSync` runtime root.
    pub fn open_or_create(run_dir: impl Into<PathBuf>) -> Result<Self, RuntimeError> {
        let run_dir = run_dir.into();
        let run_dir = claim_or_validate_runtime_root(&run_dir)?;
        Ok(Self { run_dir })
    }

    /// Returns the run directory.
    #[must_use]
    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    /// Writes one coherent runtime artifact generation and commits it last.
    ///
    /// All replaceable JSON artifacts are generated from the same report in
    /// memory, written into a hidden staging directory, validated, retained, and
    /// only then atomically published through `current_generation.json`. Readers
    /// must use that pointer as the authority; a crash can leave an ignored
    /// staging directory, but cannot publish a mixed current generation.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when payload construction, generation
    /// validation, retention, or any durable artifact write fails.
    #[allow(clippy::too_many_lines)]
    pub fn write_runtime_generation(
        &self,
        snapshot: &RuntimeArtifactSnapshot,
    ) -> Result<RuntimeArtifactGeneration, RuntimeError> {
        self.validate_owned_root()?;
        let _generation_lock = GenerationWriteLock::acquire(&self.run_dir)?;
        self.reclaim_runtime_transients()?;
        self.prune_generation_inventory(None, None)?;

        let matrix = snapshot.matrix();
        let rules = snapshot.rules();
        let report = snapshot.report();
        let progress_reports = snapshot.progress_reports();
        let session = snapshot.session();

        let summary_payload = runtime_summary_json(report)?;
        let snapshot_payload = progress_snapshot_json(report)?;
        let frame_payload = runtime_frame_json(matrix, rules, report)?;
        let summary = artifact_checksum("runtime_summary.json", &summary_payload);
        let snapshot = artifact_checksum("progress_snapshot.json", &snapshot_payload);
        let frame = artifact_checksum("runtime_frame.json", &frame_payload);

        let session_payload = session.map(runtime_session_json).transpose()?;
        let session_checksum = session_payload
            .as_ref()
            .map(|payload| artifact_checksum("runtime_session.json", payload));

        let mut primary_checksums = vec![summary.clone(), snapshot.clone(), frame.clone()];
        if let Some(checksum) = session_checksum.clone() {
            primary_checksums.push(checksum);
        }

        let checksums_payload = artifact_checksums_json(&primary_checksums)?;
        let checksum_manifest = artifact_checksum("artifact_checksums.json", &checksums_payload);
        let bundle_payload =
            runtime_bundle_json(report, &primary_checksums, progress_reports, session)?;
        let bundle_checksum = artifact_checksum("runtime_bundle.json", &bundle_payload);

        let mut committed_checksums = primary_checksums;
        committed_checksums.push(checksum_manifest.clone());
        committed_checksums.push(bundle_checksum.clone());
        let generation_payload = runtime_generation_json(&bundle_checksum, &committed_checksums)?;
        let generation_checksum = artifact_checksum("runtime_generation.json", &generation_payload);
        let generation_dir_name = format!("generation_{}", generation_checksum.sha256_hex());
        let committed_dir = self.run_dir.join("generations").join(&generation_dir_name);
        let expected_artifacts = [
            ExpectedArtifact::new("runtime_summary.json", &summary_payload, &summary),
            ExpectedArtifact::new("progress_snapshot.json", &snapshot_payload, &snapshot),
            ExpectedArtifact::new("runtime_frame.json", &frame_payload, &frame),
            ExpectedArtifact::new(
                "artifact_checksums.json",
                &checksums_payload,
                &checksum_manifest,
            ),
            ExpectedArtifact::new("runtime_bundle.json", &bundle_payload, &bundle_checksum),
            ExpectedArtifact::new(
                "runtime_generation.json",
                &generation_payload,
                &generation_checksum,
            ),
        ];
        if let Some(metadata) = optional_symlink_metadata(&committed_dir)? {
            if metadata.file_type().is_symlink()
                || metadata_is_reparse_point(&metadata)
                || !metadata.is_dir()
            {
                return Err(RuntimeError::InvalidArtifactPath);
            }
            validate_generation_dir(
                &committed_dir,
                &expected_artifacts,
                session_payload.as_deref(),
                session_checksum.as_ref(),
            )?;
            let previous_current = self.current_generation_pointer_name()?;
            self.prune_generation_inventory(
                Some(&generation_dir_name),
                previous_current.as_deref(),
            )?;
            let pointer_payload = current_generation_json(
                &generation_dir_name,
                &generation_checksum,
                &bundle_checksum,
            )?;
            self.write_text_artifact_payload("current_generation.json", &pointer_payload)?;
            return Ok(RuntimeArtifactGeneration {
                checksum_index: checksum_manifest,
                bundle_record: bundle_checksum,
                commit_marker: generation_checksum,
            });
        }

        let tmp_root = self.run_dir.join(GENERATION_TMP_ROOT);
        let tmp_dir = tmp_root.join(GENERATION_PENDING_DIR);
        self.prepare_generation_stage(&tmp_root, &tmp_dir)?;
        let mut stage_cleanup = DirectoryCleanupGuard::new(tmp_root.clone());

        write_text_payload_at(&tmp_dir.join("runtime_summary.json"), &summary_payload)?;
        write_text_payload_at(&tmp_dir.join("progress_snapshot.json"), &snapshot_payload)?;
        write_text_payload_at(&tmp_dir.join("runtime_frame.json"), &frame_payload)?;
        if let Some(payload) = session_payload.as_ref() {
            write_text_payload_at(&tmp_dir.join("runtime_session.json"), payload)?;
        }
        write_text_payload_at(&tmp_dir.join("artifact_checksums.json"), &checksums_payload)?;
        write_text_payload_at(&tmp_dir.join("runtime_bundle.json"), &bundle_payload)?;
        write_text_payload_at(
            &tmp_dir.join("runtime_generation.json"),
            &generation_payload,
        )?;
        sync_directory(&tmp_dir)?;

        let generations_dir = self.run_dir.join("generations");
        ensure_artifact_parent_dir(&generations_dir)?;
        let created_committed_dir = match fs::rename(&tmp_dir, &committed_dir) {
            Ok(()) => true,
            Err(error) if error.kind() == ErrorKind::AlreadyExists && committed_dir.is_dir() => {
                stage_cleanup.cleanup_now()?;
                false
            }
            Err(error) => return Err(error.into()),
        };
        let publish_result = (|| {
            sync_directory(&generations_dir)?;
            stage_cleanup.cleanup_now()?;
            validate_generation_dir(
                &committed_dir,
                &expected_artifacts,
                session_payload.as_deref(),
                session_checksum.as_ref(),
            )?;
            let previous_current = self.current_generation_pointer_name()?;
            self.prune_generation_inventory(
                Some(&generation_dir_name),
                previous_current.as_deref(),
            )?;
            let pointer_payload = current_generation_json(
                &generation_dir_name,
                &generation_checksum,
                &bundle_checksum,
            )?;
            self.write_text_artifact_payload("current_generation.json", &pointer_payload)
        })();
        if let Err(error) = publish_result {
            if created_committed_dir && optional_symlink_metadata(&committed_dir)?.is_some() {
                reject_symlink_or_non_directory(&committed_dir)?;
                fs::remove_dir_all(&committed_dir)?;
                sync_directory(&generations_dir)?;
            }
            return Err(error);
        }

        Ok(RuntimeArtifactGeneration {
            checksum_index: checksum_manifest,
            bundle_record: bundle_checksum,
            commit_marker: generation_checksum,
        })
    }

    #[allow(dead_code)]
    fn write_text_artifact(
        &self,
        name: &str,
        payload: &str,
    ) -> Result<ArtifactChecksum, RuntimeError> {
        ensure_artifact_parent_dir(&self.run_dir)?;
        let target = self.run_dir.join(name);
        durable_write_text(&target, payload)?;
        Ok(artifact_checksum(name, payload))
    }

    fn write_text_artifact_payload(&self, name: &str, payload: &str) -> Result<(), RuntimeError> {
        ensure_artifact_parent_dir(&self.run_dir)?;
        let target = self.run_dir.join(name);
        durable_write_text(&target, payload)
    }

    fn prepare_generation_stage(
        &self,
        tmp_root: &Path,
        tmp_dir: &Path,
    ) -> Result<(), RuntimeError> {
        self.validate_owned_root()?;
        if optional_symlink_metadata(tmp_root)?.is_some() {
            reject_symlink_or_non_directory(tmp_root)?;
            fs::remove_dir_all(tmp_root)?;
            sync_directory(&self.run_dir)?;
        }
        fs::create_dir(tmp_root)?;
        fs::create_dir(tmp_dir)?;
        sync_directory(&self.run_dir)?;
        sync_directory(tmp_root)?;
        Ok(())
    }

    fn current_generation_pointer_name(&self) -> Result<Option<String>, RuntimeError> {
        let path = self.run_dir.join("current_generation.json");
        let Some(metadata) = optional_symlink_metadata(&path)? else {
            return Ok(None);
        };
        reject_file_metadata(&metadata)?;
        let text = read_text_file_bounded(&path, MAX_RUNTIME_ARTIFACT_BYTES)?;
        let pointer = parse_current_generation_pointer(&text)?;
        let target = self
            .run_dir
            .join("generations")
            .join(&pointer.generation_dir_name);
        validate_current_pointer_target(&target, &pointer)?;
        Ok(Some(pointer.generation_dir_name))
    }

    fn prune_generation_inventory(
        &self,
        current_generation: Option<&str>,
        previous_current_generation: Option<&str>,
    ) -> Result<(), RuntimeError> {
        self.validate_owned_root()?;
        if let Some(current_generation) = current_generation
            && !is_generation_dir_name(current_generation)
        {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
        if let Some(previous) = previous_current_generation
            && !is_generation_dir_name(previous)
        {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
        let pointer_current_generation =
            if current_generation.is_none() && previous_current_generation.is_none() {
                self.current_generation_pointer_name()?
            } else {
                None
            };

        let generations_dir = self.run_dir.join("generations");
        if optional_symlink_metadata(&generations_dir)?.is_none() {
            return Ok(());
        }
        reject_symlink_or_non_directory(&generations_dir)?;

        let mut generation_dirs = Vec::new();
        for entry in fs::read_dir(&generations_dir)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_symlink() || !file_type.is_dir() {
                return Err(RuntimeError::CorruptArtifactGeneration);
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                return Err(RuntimeError::CorruptArtifactGeneration);
            };
            if !is_generation_dir_name(name) {
                return Err(RuntimeError::CorruptArtifactGeneration);
            }
            let modified = entry.metadata()?.modified().ok();
            generation_dirs.push((name.to_owned(), entry.path(), modified));
        }

        for (name, path, _) in &generation_dirs {
            validate_retention_generation_dir(path, name)?;
        }

        generation_dirs
            .sort_by(|left, right| right.2.cmp(&left.2).then_with(|| left.0.cmp(&right.0)));

        let mut protected = BTreeSet::new();
        if let Some(current_generation) = current_generation {
            protected.insert(current_generation.to_owned());
        }
        if let Some(previous) = previous_current_generation {
            protected.insert(previous.to_owned());
        }
        if let Some(pointer_current) = pointer_current_generation {
            protected.insert(pointer_current);
        }
        if generation_dirs.len() <= MAX_RUNTIME_GENERATION_RETAINED_DIRS {
            return Ok(());
        }
        let remaining_slots = MAX_RUNTIME_GENERATION_RETAINED_DIRS.saturating_sub(protected.len());
        let retained_names = generation_dirs
            .iter()
            .filter(|(name, _, _)| current_generation != Some(name.as_str()))
            .filter(|(name, _, _)| previous_current_generation != Some(name.as_str()))
            .filter(|(name, _, _)| !protected.contains(name))
            .take(remaining_slots)
            .map(|(name, _, _)| name.clone())
            .collect::<Vec<_>>();
        for name in retained_names {
            protected.insert(name);
        }

        for (name, path, _) in generation_dirs {
            if protected.contains(&name) {
                continue;
            }
            reject_symlink_or_non_directory(&path)?;
            validate_retention_generation_dir(&path, &name)?;
            fs::remove_dir_all(path)?;
        }

        sync_directory(&generations_dir)?;
        Ok(())
    }

    fn validate_owned_root(&self) -> Result<(), RuntimeError> {
        validate_runtime_root(&self.run_dir)
    }

    fn reclaim_runtime_transients(&self) -> Result<(), RuntimeError> {
        let tmp_root = self.run_dir.join(GENERATION_TMP_ROOT);
        if optional_symlink_metadata(&tmp_root)?.is_some() {
            reject_symlink_or_non_directory(&tmp_root)?;
            fs::remove_dir_all(&tmp_root)?;
            sync_directory(&self.run_dir)?;
        }
        remove_stale_runtime_tmp_files(&self.run_dir)
    }
}

fn claim_or_validate_runtime_root(run_dir: &Path) -> Result<PathBuf, RuntimeError> {
    let absolute = lexical_absolute_path(run_dir)?;
    reject_reparse_ancestors(&absolute)?;
    fs::create_dir_all(&absolute)?;
    let run_dir = fs::canonicalize(&absolute)?;
    reject_symlink_or_non_directory(&run_dir)?;

    let marker = run_dir.join(RUNTIME_ROOT_MARKER);
    if optional_symlink_metadata(&marker)?.is_some() {
        validate_runtime_root(&run_dir)?;
    } else {
        if fs::read_dir(&run_dir)?.next().transpose()?.is_some() {
            return Err(RuntimeError::UnownedArtifactDirectory);
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&marker)?;
        file.write_all(RUNTIME_ROOT_MARKER_BODY.as_bytes())?;
        file.flush()?;
        file.sync_all()?;
        drop(file);
        sync_directory(&run_dir)?;
        validate_runtime_root(&run_dir)?;
    }
    Ok(run_dir)
}

fn validate_runtime_root(run_dir: &Path) -> Result<(), RuntimeError> {
    reject_symlink_or_non_directory(run_dir)?;
    let marker = run_dir.join(RUNTIME_ROOT_MARKER);
    reject_symlink_or_non_file(&marker).map_err(|_| RuntimeError::UnownedArtifactDirectory)?;
    let body = read_text_file_bounded(&marker, 1024)?;
    if body != RUNTIME_ROOT_MARKER_BODY {
        return Err(RuntimeError::UnownedArtifactDirectory);
    }
    Ok(())
}

fn lexical_absolute_path(path: &Path) -> Result<PathBuf, RuntimeError> {
    let input = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in input.components() {
        match component {
            std::path::Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            std::path::Component::RootDir => normalized.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    return Err(RuntimeError::InvalidArtifactPath);
                }
            }
            std::path::Component::Normal(part) => normalized.push(part),
        }
    }
    Ok(normalized)
}

fn reject_reparse_ancestors(path: &Path) -> Result<(), RuntimeError> {
    let mut ancestors: Vec<&Path> = path.ancestors().collect();
    ancestors.reverse();
    for ancestor in ancestors {
        if ancestor.as_os_str().is_empty() {
            continue;
        }
        match fs::symlink_metadata(ancestor) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || metadata_is_reparse_point(&metadata) {
                    return Err(RuntimeError::InvalidArtifactPath);
                }
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn remove_stale_runtime_tmp_files(run_dir: &Path) -> Result<(), RuntimeError> {
    let mut removed = false;
    for entry in fs::read_dir(run_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !is_runtime_temp_file_name(name) {
            continue;
        }
        let path = entry.path();
        reject_symlink_or_non_file(&path)?;
        fs::remove_file(path)?;
        removed = true;
    }
    if removed {
        sync_directory(run_dir)?;
    }
    Ok(())
}

pub(crate) fn is_runtime_temp_file_name(name: &str) -> bool {
    [
        "artifact_checksums.json",
        "current_generation.json",
        "progress_snapshot.json",
        "runtime_bundle.json",
        "runtime_frame.json",
        "runtime_generation.json",
        "runtime_session.json",
        "runtime_summary.json",
    ]
    .iter()
    .any(|base| is_runtime_temp_file_for_base(name, base))
}

fn is_runtime_temp_file_for_base(name: &str, base: &str) -> bool {
    let Some(rest) = name.strip_prefix(base) else {
        return false;
    };
    let Some(rest) = rest.strip_prefix('.') else {
        return false;
    };
    let mut parts = rest.split('.');
    let Some(pid) = parts.next() else {
        return false;
    };
    let Some(counter) = parts.next() else {
        return false;
    };
    let Some(extension) = parts.next() else {
        return false;
    };
    parts.next().is_none()
        && extension == "tmp"
        && !pid.is_empty()
        && !counter.is_empty()
        && pid.as_bytes().iter().all(u8::is_ascii_digit)
        && counter.as_bytes().iter().all(u8::is_ascii_digit)
}

#[derive(Debug)]
pub(crate) struct DirectoryCleanupGuard {
    path: PathBuf,
    armed: bool,
}

impl DirectoryCleanupGuard {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    pub(crate) fn cleanup_now(&mut self) -> Result<(), RuntimeError> {
        if !self.armed {
            return Ok(());
        }
        if optional_symlink_metadata(&self.path)?.is_some() {
            reject_symlink_or_non_directory(&self.path)?;
            fs::remove_dir_all(&self.path)?;
            if let Some(parent) = self.path.parent() {
                sync_directory(parent)?;
            }
        }
        self.armed = false;
        Ok(())
    }
}

impl Drop for DirectoryCleanupGuard {
    fn drop(&mut self) {
        // Drop cannot report validation or deletion failures. Runtime cleanup is
        // therefore explicit-only; interrupted staging is reclaimed under the
        // writer lock by `reclaim_runtime_transients` on the next write.
    }
}

#[derive(Debug)]
pub(crate) struct GenerationWriteLock {
    file: File,
}

impl GenerationWriteLock {
    pub(crate) fn acquire(run_dir: &Path) -> Result<Self, RuntimeError> {
        validate_runtime_root(run_dir)?;
        let path = run_dir.join(GENERATION_LOCK_FILE);
        if optional_symlink_metadata(&path)?.is_some() {
            reject_symlink_or_non_file(&path)?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        reject_symlink_or_non_file(&path)?;
        match file.try_lock_exclusive() {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                return Err(RuntimeError::ArtifactWriterBusy);
            }
            Err(error) => return Err(error.into()),
        }
        Ok(Self { file })
    }
}

impl Drop for GenerationWriteLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

fn preflight_artifact_snapshot(
    matrix: &ObservationMatrix,
    rules: &[PrismionRule],
    report: &RuntimeCycleReport,
    progress_reports: &[RuntimeCycleReport],
) -> Result<(), RuntimeError> {
    if matrix.dense_cells().len() != matrix.shape().cell_count() {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    if rules.len() > MAX_PROPOSALS_PER_FIELD
        || report.rules_evaluated != rules.len()
        || report.rule_outcomes.len() != rules.len()
        || progress_reports.len() > MAX_RUNTIME_BUNDLE_PROGRESS_ROWS
    {
        return Err(RuntimeError::InvalidSessionConfig);
    }

    preflight_report_bounds(report)?;
    for progress_report in progress_reports {
        preflight_report_bounds(progress_report)?;
    }

    Ok(())
}

fn preflight_report_bounds(report: &RuntimeCycleReport) -> Result<(), RuntimeError> {
    if report.proposal_count() > MAX_PROPOSALS_PER_FIELD
        || report.patch_count() > MAX_PROPOSAL_PATCHES_PER_FIELD
        || report.evidence_cell_count() > MAX_EVIDENCE_CELLS_PER_CYCLE
        || report.proposal_field().patch_count() > MAX_PROPOSAL_PATCHES_PER_FIELD
    {
        return Err(RuntimeError::InvalidSessionConfig);
    }
    Ok(())
}

fn durable_write_text(path: &Path, payload: &str) -> Result<(), RuntimeError> {
    let Some(file_name) = path.file_name() else {
        return Err(RuntimeError::InvalidArtifactPath);
    };
    let Some(parent) = path.parent() else {
        return Err(RuntimeError::InvalidArtifactPath);
    };
    ensure_artifact_parent_dir(parent)?;
    if optional_symlink_metadata(path)?.is_some() {
        reject_symlink_or_non_file(path)?;
    }
    let mut tmp_name = file_name.to_os_string();
    let tmp_id = DURABLE_WRITE_COUNTER.fetch_add(1, Ordering::Relaxed);
    tmp_name.push(format!(".{}.{}.tmp", std::process::id(), tmp_id));
    let tmp_path = path.with_file_name(tmp_name);
    let mut cleanup = TempFileGuard::new(tmp_path.clone());

    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)?;
        cleanup.arm();
        file.write_all(payload.as_bytes())?;
        file.flush()?;
        file.sync_all()?;
    }

    for attempt in 0..25 {
        match fs::rename(&tmp_path, path) {
            Ok(()) => {
                cleanup.disarm();
                let _ = sync_directory(parent);
                return Ok(());
            }
            Err(error) if attempt < 24 && error.kind() == std::io::ErrorKind::PermissionDenied => {
                std::thread::sleep(std::time::Duration::from_millis(40));
            }
            Err(error) => {
                return Err(RuntimeError::from(error));
            }
        }
    }

    Err(RuntimeError::Io)
}

#[derive(Debug)]
struct TempFileGuard {
    path: PathBuf,
    armed: bool,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: false }
    }

    fn arm(&mut self) {
        self.armed = true;
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn ensure_artifact_parent_dir(path: &Path) -> Result<(), RuntimeError> {
    let absolute = lexical_absolute_path(path)?;
    reject_reparse_ancestors(&absolute)?;
    fs::create_dir_all(&absolute)?;
    let canonical = fs::canonicalize(&absolute)?;
    reject_symlink_or_non_directory(&canonical)
}

fn optional_symlink_metadata(path: &Path) -> Result<Option<fs::Metadata>, RuntimeError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[derive(Clone, Copy, Debug)]
struct ExpectedArtifact<'a> {
    name: &'static str,
    payload: &'a str,
    checksum: &'a ArtifactChecksum,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RuntimeGenerationManifest {
    generation_id: String,
    artifacts: BTreeMap<String, ArtifactChecksum>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ValidatedGeneration {
    generation_checksum: ArtifactChecksum,
    bundle_checksum: ArtifactChecksum,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CurrentGenerationPointer {
    generation_dir_name: String,
    generation_sha256: String,
    generation_fnv1a64: String,
    bundle_sha256: String,
    bundle_fnv1a64: String,
}

impl<'a> ExpectedArtifact<'a> {
    const fn new(name: &'static str, payload: &'a str, checksum: &'a ArtifactChecksum) -> Self {
        Self {
            name,
            payload,
            checksum,
        }
    }
}

fn validate_generation_dir(
    dir: &Path,
    expected_artifacts: &[ExpectedArtifact<'_>],
    session_payload: Option<&str>,
    session_checksum: Option<&ArtifactChecksum>,
) -> Result<(), RuntimeError> {
    reject_symlink_or_non_directory(dir)?;
    let mut expected_names = BTreeSet::new();
    for expected in expected_artifacts {
        expected_names.insert(expected.name);
        validate_expected_artifact(dir, *expected)?;
    }

    match (session_payload, session_checksum) {
        (Some(payload), Some(checksum)) => {
            expected_names.insert("runtime_session.json");
            validate_expected_artifact(
                dir,
                ExpectedArtifact::new("runtime_session.json", payload, checksum),
            )?;
        }
        (None, None) => {}
        _ => return Err(RuntimeError::CorruptArtifactGeneration),
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() || !file_type.is_file() {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            return Err(RuntimeError::CorruptArtifactGeneration);
        };
        if !expected_names.contains(name) {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
    }

    Ok(())
}

fn reject_symlink_or_non_directory(path: &Path) -> Result<(), RuntimeError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink()
        || metadata_is_reparse_point(&metadata)
        || !metadata.is_dir()
    {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    Ok(())
}

fn reject_symlink_or_non_file(path: &Path) -> Result<(), RuntimeError> {
    let metadata = fs::symlink_metadata(path)?;
    reject_file_metadata(&metadata)
}

fn reject_file_metadata(metadata: &fs::Metadata) -> Result<(), RuntimeError> {
    if metadata.file_type().is_symlink()
        || metadata_is_reparse_point(metadata)
        || !metadata.is_file()
    {
        return Err(RuntimeError::InvalidArtifactPath);
    }
    Ok(())
}

#[cfg(windows)]
fn sync_directory(path: &Path) -> Result<(), RuntimeError> {
    reject_symlink_or_non_directory(path)?;
    Ok(())
}

#[cfg(not(windows))]
fn sync_directory(path: &Path) -> Result<(), RuntimeError> {
    reject_symlink_or_non_directory(path)?;
    let file = File::open(path)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(windows)]
fn metadata_is_reparse_point(metadata: &fs::Metadata) -> bool {
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;
    metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn metadata_is_reparse_point(_metadata: &fs::Metadata) -> bool {
    false
}

fn validate_expected_artifact(
    dir: &Path,
    expected: ExpectedArtifact<'_>,
) -> Result<(), RuntimeError> {
    let path = dir.join(expected.name);
    reject_symlink_or_non_file(&path)?;
    let metadata = fs::symlink_metadata(&path)?;
    if metadata.len() != expected.payload.len() as u64
        || metadata.len() > MAX_RUNTIME_ARTIFACT_BYTES
    {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    let payload = read_text_file_bounded(&path, MAX_RUNTIME_ARTIFACT_BYTES)?;
    if payload != expected.payload {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    if artifact_checksum(expected.name, &payload) != *expected.checksum {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    validate_runtime_artifact_schema(expected.name, &payload)?;
    Ok(())
}

fn validate_retention_generation_dir(
    dir: &Path,
    name: &str,
) -> Result<ValidatedGeneration, RuntimeError> {
    reject_symlink_or_non_directory(dir)?;
    let Some(expected_hash) = name.strip_prefix("generation_") else {
        return Err(RuntimeError::CorruptArtifactGeneration);
    };
    if !is_sha256_hex(expected_hash) {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }

    let generation_path = dir.join("runtime_generation.json");
    reject_symlink_or_non_file(&generation_path)?;
    let payload = read_text_file_bounded(&generation_path, MAX_RUNTIME_ARTIFACT_BYTES)?;
    let generation_checksum = artifact_checksum("runtime_generation.json", &payload);
    if generation_checksum.sha256_hex() != expected_hash {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    validate_runtime_artifact_schema("runtime_generation.json", &payload)?;
    let manifest = parse_runtime_generation_manifest(&payload)?;
    let manifest_artifacts = &manifest.artifacts;
    let bundle_checksum = manifest_artifacts
        .get("runtime_bundle.json")
        .ok_or(RuntimeError::CorruptArtifactGeneration)?;
    if manifest.generation_id != bundle_checksum.sha256_hex() {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    let mut expected_names = BTreeSet::new();
    expected_names.insert("runtime_generation.json".to_owned());
    for checksum in manifest_artifacts.values() {
        validate_manifest_artifact(dir, checksum)?;
        expected_names.insert(checksum.name().to_owned());
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() || !file_type.is_file() {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            return Err(RuntimeError::CorruptArtifactGeneration);
        };
        if !expected_names.contains(name) {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
        if entry.metadata()?.len() > MAX_RUNTIME_ARTIFACT_BYTES {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
    }
    validate_generation_cross_links(dir, &manifest)?;

    Ok(ValidatedGeneration {
        generation_checksum,
        bundle_checksum: bundle_checksum.clone(),
    })
}

fn parse_runtime_generation_manifest(
    payload: &str,
) -> Result<RuntimeGenerationManifest, RuntimeError> {
    let wire = parse_runtime_generation_wire(payload)?;
    if !is_sha256_hex(&wire.generation_id) {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    let mut manifest_artifacts = BTreeMap::new();
    for artifact in &wire.artifacts {
        let checksum = artifact_checksum_from_wire(artifact)?;
        if checksum.name() == "runtime_generation.json"
            || !is_runtime_artifact_file_name(checksum.name())
        {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
        if manifest_artifacts
            .insert(checksum.name().to_owned(), checksum)
            .is_some()
        {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
    }
    for required in [
        "artifact_checksums.json",
        "progress_snapshot.json",
        "runtime_bundle.json",
        "runtime_frame.json",
        "runtime_summary.json",
    ] {
        if !manifest_artifacts.contains_key(required) {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
    }
    Ok(RuntimeGenerationManifest {
        generation_id: wire.generation_id,
        artifacts: manifest_artifacts,
    })
}

fn validate_manifest_artifact(dir: &Path, expected: &ArtifactChecksum) -> Result<(), RuntimeError> {
    let path = dir.join(expected.name());
    reject_symlink_or_non_file(&path)?;
    let metadata = fs::symlink_metadata(&path)?;
    if metadata.len() != expected.bytes() as u64 || metadata.len() > MAX_RUNTIME_ARTIFACT_BYTES {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    let payload = read_text_file_bounded(&path, MAX_RUNTIME_ARTIFACT_BYTES)?;
    if artifact_checksum(expected.name(), &payload) != *expected {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    validate_runtime_artifact_schema(expected.name(), &payload)?;
    Ok(())
}

fn validate_generation_cross_links(
    dir: &Path,
    manifest: &RuntimeGenerationManifest,
) -> Result<(), RuntimeError> {
    let primary_checksums = manifest_primary_checksums(manifest)?;

    let checksum_index_payload = read_text_file_bounded(
        &dir.join("artifact_checksums.json"),
        MAX_RUNTIME_ARTIFACT_BYTES,
    )?;
    let checksum_index = parse_artifact_checksums_wire(&checksum_index_payload)?;
    let checksum_index_map = checksum_map_from_wire(&checksum_index.artifacts)?;
    if checksum_index_map != primary_checksums {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }

    let bundle_payload =
        read_text_file_bounded(&dir.join("runtime_bundle.json"), MAX_RUNTIME_ARTIFACT_BYTES)?;
    let bundle = parse_runtime_bundle_wire(&bundle_payload)?;
    let bundle_checksum_map = checksum_map_from_wire(&bundle.artifact_checksums)?;
    if bundle_checksum_map != primary_checksums {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    let bundle_file_set = string_set(&bundle.artifact_files)?;
    if bundle_file_set != primary_checksums.keys().cloned().collect::<BTreeSet<_>>() {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    let session_in_manifest = primary_checksums.contains_key("runtime_session.json");
    if session_in_manifest != bundle.runtime_session.is_some() {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    Ok(())
}

fn manifest_primary_checksums(
    manifest: &RuntimeGenerationManifest,
) -> Result<BTreeMap<String, ArtifactChecksum>, RuntimeError> {
    let mut primary = BTreeMap::new();
    for (name, checksum) in &manifest.artifacts {
        match name.as_str() {
            "artifact_checksums.json" | "runtime_bundle.json" => {}
            "progress_snapshot.json"
            | "runtime_frame.json"
            | "runtime_session.json"
            | "runtime_summary.json" => {
                primary.insert(name.clone(), checksum.clone());
            }
            _ => return Err(RuntimeError::CorruptArtifactGeneration),
        }
    }
    for required in [
        "progress_snapshot.json",
        "runtime_frame.json",
        "runtime_summary.json",
    ] {
        if !primary.contains_key(required) {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
    }
    Ok(primary)
}

fn checksum_map_from_wire(
    artifacts: &[ArtifactChecksumWire],
) -> Result<BTreeMap<String, ArtifactChecksum>, RuntimeError> {
    let mut checksums = BTreeMap::new();
    for artifact in artifacts {
        let checksum = artifact_checksum_from_wire(artifact)?;
        if !matches!(
            checksum.name(),
            "progress_snapshot.json"
                | "runtime_frame.json"
                | "runtime_session.json"
                | "runtime_summary.json"
        ) {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
        if checksums
            .insert(checksum.name().to_owned(), checksum)
            .is_some()
        {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
    }
    Ok(checksums)
}

fn string_set(values: &[String]) -> Result<BTreeSet<String>, RuntimeError> {
    let mut set = BTreeSet::new();
    for value in values {
        if !matches!(
            value.as_str(),
            "progress_snapshot.json"
                | "runtime_frame.json"
                | "runtime_session.json"
                | "runtime_summary.json"
        ) || !set.insert(value.clone())
        {
            return Err(RuntimeError::CorruptArtifactGeneration);
        }
    }
    Ok(set)
}

fn artifact_checksum_from_wire(
    wire: &ArtifactChecksumWire,
) -> Result<ArtifactChecksum, RuntimeError> {
    let name = wire.name.as_str();
    let bytes = wire.bytes;
    let fnv1a64_hex = wire.fnv1a64.as_str();
    let sha256_hex = wire.sha256.as_str();
    if !is_runtime_artifact_file_name(name)
        || bytes > MAX_RUNTIME_ARTIFACT_BYTES
        || !is_fnv1a64_hex(fnv1a64_hex)
        || !is_sha256_hex(sha256_hex)
    {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    let bytes = usize::try_from(bytes).map_err(|_| RuntimeError::CorruptArtifactGeneration)?;
    Ok(ArtifactChecksum {
        name: name.to_owned(),
        bytes,
        fnv1a64_hex: fnv1a64_hex.to_owned(),
        sha256_hex: sha256_hex.to_owned(),
    })
}

fn validate_current_pointer_target(
    target: &Path,
    pointer: &CurrentGenerationPointer,
) -> Result<(), RuntimeError> {
    let generation = validate_retention_generation_dir(target, &pointer.generation_dir_name)?;
    if pointer.generation_sha256 != generation.generation_checksum.sha256_hex()
        || pointer.generation_fnv1a64 != generation.generation_checksum.fnv1a64_hex()
        || pointer.bundle_sha256 != generation.bundle_checksum.sha256_hex()
        || pointer.bundle_fnv1a64 != generation.bundle_checksum.fnv1a64_hex()
    {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    Ok(())
}

fn is_runtime_artifact_file_name(name: &str) -> bool {
    matches!(
        name,
        "artifact_checksums.json"
            | "progress_snapshot.json"
            | "runtime_bundle.json"
            | "runtime_frame.json"
            | "runtime_generation.json"
            | "runtime_session.json"
            | "runtime_summary.json"
    )
}

fn read_text_file_bounded(path: &Path, max_bytes: u64) -> Result<String, RuntimeError> {
    reject_symlink_or_non_file(path)?;
    let metadata = fs::symlink_metadata(path)?;
    if metadata.len() > max_bytes {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    let file = File::open(path)?;
    let mut payload = String::new();
    file.take(max_bytes.saturating_add(1))
        .read_to_string(&mut payload)?;
    if payload.len() as u64 > max_bytes {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    Ok(payload)
}

fn write_text_payload_at(path: &Path, payload: &str) -> Result<(), RuntimeError> {
    let Some(parent) = path.parent() else {
        return Err(RuntimeError::InvalidArtifactPath);
    };
    ensure_artifact_parent_dir(parent)?;
    durable_write_text(path, payload)
}

fn parse_current_generation_pointer(
    pointer_json: &str,
) -> Result<CurrentGenerationPointer, RuntimeError> {
    let wire = parse_current_generation_wire(pointer_json)?;
    if !is_sha256_hex(&wire.bundle_sha256)
        || !is_fnv1a64_hex(&wire.bundle_fnv1a64)
        || !is_sha256_hex(&wire.generation_sha256)
        || !is_fnv1a64_hex(&wire.generation_fnv1a64)
    {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    let Some(name) = wire.generation_dir.strip_prefix("generations/") else {
        return Err(RuntimeError::CorruptArtifactGeneration);
    };
    if name.contains('/') || name.contains('\\') || !is_generation_dir_name(name) {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    let expected_digest = name
        .strip_prefix("generation_")
        .ok_or(RuntimeError::CorruptArtifactGeneration)?;
    if wire.generation_sha256 != expected_digest {
        return Err(RuntimeError::CorruptArtifactGeneration);
    }
    Ok(CurrentGenerationPointer {
        generation_dir_name: name.to_owned(),
        generation_sha256: wire.generation_sha256,
        generation_fnv1a64: wire.generation_fnv1a64,
        bundle_sha256: wire.bundle_sha256,
        bundle_fnv1a64: wire.bundle_fnv1a64,
    })
}

fn is_generation_dir_name(name: &str) -> bool {
    const PREFIX: &str = "generation_";
    let Some(digest) = name.strip_prefix(PREFIX) else {
        return false;
    };
    is_sha256_hex(digest)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

fn is_fnv1a64_hex(value: &str) -> bool {
    value.len() == 16
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

fn lower_hex(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let bytes = bytes.as_ref();
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(*byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(*byte & 0x0f)]));
    }
    encoded
}

fn artifact_checksum(name: &str, payload: &str) -> ArtifactChecksum {
    let mut sha256 = Sha256::new();
    sha256.update(payload.as_bytes());
    let sha256_hex = lower_hex(sha256.finalize());
    ArtifactChecksum {
        name: name.to_owned(),
        bytes: payload.len(),
        fnv1a64_hex: format!("{:016x}", fnv1a64(payload.as_bytes())),
        sha256_hex,
    }
}
