//! Mutual Inhibition experiment: create TWO competing clusters in the output zone.
//!
//! The hypothesis: the Brain collapses to a single attractor because there's no
//! "hill" separating different stable states. Mutual inhibition creates this hill:
//!
//!   Cluster A ←(inhibit)→ Cluster B
//!
//!   Input activates A → A suppresses B → stable state: A alive, B dead
//!   Input activates B → B suppresses A → stable state: B alive, A dead
//!
//! This is winner-take-all dynamics — the "domb a völgyek közt" (hill between valleys).

use instnct_core::{
    build_network, cosine_similarity, evolution_step_jackpot,
    evolution_step_jackpot_traced_with_policy_and_operator_weights,
    mutation_operator_baseline_probability, mutation_operator_index, save_checkpoint, softmax,
    AcceptancePolicy, CandidateTraceRecord, CheckpointMeta, InitConfig, Int8Projection, Network,
    StepOutcome, VcbpTable, MUTATION_OPERATORS,
};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::env;
use std::fs::{create_dir_all, write, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAX_CHARGE: i32 = 7;
const DEFAULT_STEPS: usize = 20_000;
const DEFAULT_EVAL_LEN: usize = 100;
const DEFAULT_FULL_LEN: usize = 1_000;
const PROGRESS_INTERVAL: usize = 2_000;
const PANEL_PROBE_COUNT: usize = 32;

// ── Reuse helpers from evolve_bytepair_proj ──

#[derive(Serialize)]
struct RunMeta {
    fixture: &'static str,
    phase: String,
    arm: String,
    run_id: String,
    seed: u64,
    seed_list: Option<String>,
    #[serde(rename = "H")]
    h: usize,
    steps: usize,
    horizon_steps: usize,
    jackpot: usize,
    ticks: usize,
    chain_count: usize,
    accept_ties: bool,
    accept_policy: String,
    neutral_p: Option<f64>,
    accept_epsilon: Option<f64>,
    input_scatter: bool,
    corpus: String,
    packed: String,
    checkpoint: String,
    candidate_log: Option<String>,
    panel_window_size: Option<usize>,
    panel_timeseries: Option<String>,
    operator_policy: String,
    operator_prior: Option<String>,
    operator_epsilon_random: f64,
    operator_weight_floor: f64,
    operator_weight_cap: f64,
    operator_ewma_alpha: f64,
    operator_ewma_reward: String,
    operator_policy_log: Option<String>,
    instrumentation_schema_version: Option<String>,
    d8_state_log: Option<String>,
    archive_parent_policy: String,
    archive_parent_log: Option<String>,
    archive_max_size: usize,
    archive_switch_interval_panels: usize,
    archive_min_cell_confidence: f64,
    archive_p2_model: Option<String>,
    embedding_anchored_highways: usize,
    diversity_guard_lambda: f64,
}

struct PanelMetrics {
    panel_probe_acc: f64,
    unique_predictions: usize,
    collision_rate: f64,
    f_active: f64,
    h_output_mean: f64,
    h_output_var: f64,
    stable_rank: f64,
    kernel_rank: usize,
    separation_sp: f64,
}

struct PanelTimeseriesWriter {
    writer: BufWriter<File>,
    path: PathBuf,
    window_size: usize,
    last_step: usize,
    last_accepted: u32,
    last_rejected: u32,
}

impl PanelTimeseriesWriter {
    fn new(path: PathBuf, window_size: usize) -> Self {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent).expect("failed to create panel timeseries directory");
            }
        }
        let mut writer =
            BufWriter::new(File::create(&path).expect("failed to create panel timeseries"));
        writeln!(
            writer,
            "step,panel_probe_acc,main_peak_acc,accept_rate_window,accepted_window,rejected_window,edges,unique_predictions,collision_rate,f_active,h_output_mean,h_output_var,stable_rank,kernel_rank,separation_sp"
        )
        .expect("failed to write panel timeseries header");
        Self {
            writer,
            path,
            window_size,
            last_step: 0,
            last_accepted: 0,
            last_rejected: 0,
        }
    }

    fn write_row(
        &mut self,
        step: usize,
        accepted: u32,
        rejected: u32,
        edges: usize,
        main_peak_acc: f64,
        metrics: &PanelMetrics,
    ) {
        let accepted_window = accepted.saturating_sub(self.last_accepted);
        let rejected_window = rejected.saturating_sub(self.last_rejected);
        let candidate_steps = (accepted_window + rejected_window).max(1);
        let accept_rate_window = accepted_window as f64 / candidate_steps as f64;
        writeln!(
            self.writer,
            "{},{:.17},{:.17},{:.17},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{},{:.17}",
            step,
            metrics.panel_probe_acc,
            main_peak_acc,
            accept_rate_window,
            accepted_window,
            rejected_window,
            edges,
            metrics.unique_predictions,
            metrics.collision_rate,
            metrics.f_active,
            metrics.h_output_mean,
            metrics.h_output_var,
            metrics.stable_rank,
            metrics.kernel_rank,
            metrics.separation_sp
        )
        .expect("failed to write panel timeseries row");
        self.last_step = step;
        self.last_accepted = accepted;
        self.last_rejected = rejected;
    }

    fn flush(&mut self) {
        self.writer
            .flush()
            .expect("failed to flush panel timeseries");
    }
}

struct CandidateLogWriter {
    writer: BufWriter<File>,
    run_id: String,
    arm: String,
    seed: u64,
    h: usize,
}

impl CandidateLogWriter {
    fn new(path: &Path, run_id: &str, arm: &str, seed: u64, h: usize) -> Self {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent).expect("failed to create candidate log directory");
            }
        }
        let mut writer =
            BufWriter::new(File::create(path).expect("failed to create candidate log"));
        writeln!(
            writer,
            "run_id,arm,seed,H,step,candidate_id,operator_id,mutated,evaluated,before_U,after_U,delta_U,within_cap,selected,accepted,candidate_eval_ms,step_wall_ms"
        )
        .expect("failed to write candidate log header");
        Self {
            writer,
            run_id: run_id.to_string(),
            arm: arm.to_string(),
            seed,
            h,
        }
    }

    fn write_record(&mut self, record: &CandidateTraceRecord) {
        writeln!(
            self.writer,
            "{},{},{},{},{},{},{},{},{},{:.17},{:.17},{:.17},{},{},{},{:.6},{:.6}",
            self.run_id,
            self.arm,
            self.seed,
            self.h,
            record.step,
            record.candidate_id,
            record.operator_id,
            record.mutated,
            record.evaluated,
            record.before_u,
            record.after_u,
            record.delta_u,
            record.within_cap,
            record.selected,
            record.accepted,
            record.candidate_eval_ms,
            record.step_wall_ms
        )
        .expect("failed to write candidate log record");
    }

    fn flush(&mut self) {
        self.writer.flush().expect("failed to flush candidate log");
    }
}

struct D8StateLogWriter {
    writer: BufWriter<File>,
    path: PathBuf,
    run_id: String,
    phase: String,
    arm: String,
    seed: u64,
    h: usize,
    checkpoint_ref: String,
}

impl D8StateLogWriter {
    const SCHEMA_VERSION: &'static str = "d8_state_log_v1";

    fn new(
        path: PathBuf,
        run_id: &str,
        phase: &str,
        arm: &str,
        seed: u64,
        h: usize,
        checkpoint_ref: String,
    ) -> Self {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent).expect("failed to create D8 state log directory");
            }
        }
        let mut writer =
            BufWriter::new(File::create(&path).expect("failed to create D8 state log"));
        writeln!(
            writer,
            "schema_version,state_id,parent_id,family_id,root_family_id,run_id,phase,arm,seed,H,panel_index,step,time_pct,accepted_total,rejected_total,current_peak,panel_probe_acc,accept_rate_window,accepted_window,rejected_window,edges,unique_predictions,collision_rate,f_active,stable_rank,kernel_rank,separation_sp,checkpoint_ref,archive_cell_id,psi_pred,cell_confidence"
        )
        .expect("failed to write D8 state log header");
        Self {
            writer,
            path,
            run_id: run_id.to_string(),
            phase: phase.to_string(),
            arm: arm.to_string(),
            seed,
            h,
            checkpoint_ref,
        }
    }

    fn write_panel_state(
        &mut self,
        panel_index: usize,
        step: usize,
        total_steps: usize,
        parent_override: Option<&str>,
        accepted_total: u32,
        rejected_total: u32,
        accepted_window: u32,
        rejected_window: u32,
        edges: usize,
        current_peak: f64,
        metrics: &PanelMetrics,
        archive_cell_id: Option<usize>,
        psi_pred: Option<f64>,
        cell_confidence: Option<f64>,
    ) {
        let state_id = format!("{}::{}", self.run_id, panel_index);
        let parent_id = if let Some(parent_id) = parent_override {
            parent_id.to_string()
        } else if panel_index > 0 {
            format!("{}::{}", self.run_id, panel_index - 1)
        } else {
            String::new()
        };
        let candidate_steps = (accepted_window + rejected_window).max(1);
        let accept_rate_window = accepted_window as f64 / candidate_steps as f64;
        let time_pct = step as f64 / total_steps.max(1) as f64;
        let archive_cell_id_s = archive_cell_id.map(|v| v.to_string()).unwrap_or_default();
        let psi_pred_s = psi_pred.map(|v| format!("{v:.17}")).unwrap_or_default();
        let cell_confidence_s = cell_confidence
            .map(|v| format!("{v:.17}"))
            .unwrap_or_default();
        writeln!(
            self.writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{:.17},{},{},{:.17},{:.17},{:.17},{},{},{},{},{:.17},{:.17},{:.17},{},{:.17},{},{},{},{}",
            Self::SCHEMA_VERSION,
            state_id,
            parent_id,
            self.run_id,
            self.run_id,
            self.run_id,
            self.phase,
            self.arm,
            self.seed,
            self.h,
            panel_index,
            step,
            time_pct,
            accepted_total,
            rejected_total,
            current_peak,
            metrics.panel_probe_acc,
            accept_rate_window,
            accepted_window,
            rejected_window,
            edges,
            metrics.unique_predictions,
            metrics.collision_rate,
            metrics.f_active,
            metrics.stable_rank,
            metrics.kernel_rank,
            metrics.separation_sp,
            self.checkpoint_ref,
            archive_cell_id_s,
            psi_pred_s,
            cell_confidence_s
        )
        .expect("failed to write D8 state log row");
    }

    fn flush(&mut self) {
        self.writer.flush().expect("failed to flush D8 state log");
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArchiveParentPolicyMode {
    CurrentBest,
    RandomArchive,
    ScoreArchive,
    P2PsiConf,
}

impl ArchiveParentPolicyMode {
    fn parse(value: &str) -> Self {
        match value {
            "current-best" | "current_best" => Self::CurrentBest,
            "random-archive" | "random_archive" => Self::RandomArchive,
            "score-archive" | "score_archive" => Self::ScoreArchive,
            "p2-psi-conf" | "p2_psi_conf" => Self::P2PsiConf,
            other => panic!(
                "--archive-parent-policy expects current-best|random-archive|score-archive|p2-psi-conf, got {other}"
            ),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::CurrentBest => "current-best",
            Self::RandomArchive => "random-archive",
            Self::ScoreArchive => "score-archive",
            Self::P2PsiConf => "p2-psi-conf",
        }
    }
}

#[derive(Deserialize)]
struct D8P2ExportModel {
    schema_version: String,
    psi_features: Vec<String>,
    sphere_features: Vec<String>,
    per_h: std::collections::BTreeMap<String, D8P2HModel>,
}

#[derive(Deserialize)]
struct D8P2HModel {
    knee_n: usize,
    psi: D8P2PsiModel,
    sphere: D8P2SphereModel,
}

#[derive(Deserialize)]
struct D8P2PsiModel {
    intercept: f64,
    beta: Vec<f64>,
    median: Vec<f64>,
    mean: Vec<f64>,
    std: Vec<f64>,
}

#[derive(Deserialize)]
struct D8P2SphereModel {
    median: Vec<f64>,
    iqr: Vec<f64>,
    anchors: Vec<Vec<f64>>,
}

impl D8P2ExportModel {
    fn load(path: &Path) -> Self {
        let text = std::fs::read_to_string(path).expect("failed to read D8 P2 model");
        let model: Self = serde_json::from_str(&text).expect("failed to parse D8 P2 model");
        assert_eq!(
            model.schema_version, "d8_p2_model_v1",
            "unsupported D8 P2 model schema"
        );
        model
    }

    fn h_model(&self, h: usize) -> Option<&D8P2HModel> {
        self.per_h.get(&h.to_string())
    }

    fn score_state(
        &self,
        h: usize,
        jackpot: usize,
        step: usize,
        total_steps: usize,
        current_peak: f64,
        accepted_window: u32,
        rejected_window: u32,
        edges: usize,
        metrics: &PanelMetrics,
    ) -> Option<(f64, usize)> {
        let h_model = self.h_model(h)?;
        let psi_values: Vec<f64> = self
            .psi_features
            .iter()
            .map(|name| {
                d8_feature_value(
                    name,
                    h,
                    jackpot,
                    step,
                    total_steps,
                    current_peak,
                    accepted_window,
                    rejected_window,
                    edges,
                    metrics,
                )
            })
            .collect();
        let psi = h_model.psi.predict(&psi_values);
        let sphere_values: Vec<f64> = self
            .sphere_features
            .iter()
            .map(|name| {
                d8_feature_value(
                    name,
                    h,
                    jackpot,
                    step,
                    total_steps,
                    current_peak,
                    accepted_window,
                    rejected_window,
                    edges,
                    metrics,
                )
            })
            .collect();
        let cell_id = h_model.sphere.assign_cell(&sphere_values);
        Some((psi, cell_id))
    }

    fn confidence(&self, h: usize, cell_count: usize) -> f64 {
        let knee = self.h_model(h).map(|m| m.knee_n).unwrap_or(1).max(1);
        (cell_count as f64 / knee as f64).min(1.0)
    }
}

impl D8P2PsiModel {
    fn predict(&self, values: &[f64]) -> f64 {
        let mut out = self.intercept;
        for i in 0..self.beta.len().min(values.len()) {
            let raw = if values[i].is_finite() {
                values[i]
            } else {
                self.median[i]
            };
            let std = self.std[i].abs().max(1e-12);
            out += self.beta[i] * ((raw - self.mean[i]) / std);
        }
        out
    }
}

impl D8P2SphereModel {
    fn assign_cell(&self, values: &[f64]) -> usize {
        if self.anchors.is_empty() {
            return 0;
        }
        let dim = self.median.len().min(values.len());
        if dim == 0 {
            return 0;
        }
        let mut coord = vec![0.0; dim];
        for i in 0..dim {
            let raw = if values[i].is_finite() {
                values[i]
            } else {
                self.median[i]
            };
            coord[i] = (raw - self.median[i]) / self.iqr[i].abs().max(1e-12);
        }
        let norm = coord.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
        for v in &mut coord {
            *v /= norm;
        }
        let mut best_idx = 0usize;
        let mut best_dot = f64::NEG_INFINITY;
        for (idx, anchor) in self.anchors.iter().enumerate() {
            let dot = coord
                .iter()
                .zip(anchor.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
            if dot > best_dot {
                best_dot = dot;
                best_idx = idx;
            }
        }
        best_idx
    }
}

fn d8_feature_value(
    name: &str,
    h: usize,
    jackpot: usize,
    step: usize,
    total_steps: usize,
    current_peak: f64,
    accepted_window: u32,
    rejected_window: u32,
    edges: usize,
    metrics: &PanelMetrics,
) -> f64 {
    match name {
        "edges" => edges as f64,
        "unique_predictions" => metrics.unique_predictions as f64,
        "collision_rate" => metrics.collision_rate,
        "f_active" => metrics.f_active,
        "stable_rank" => metrics.stable_rank,
        "kernel_rank" => metrics.kernel_rank as f64,
        "separation_sp" => metrics.separation_sp,
        "accept_rate_window" => {
            let total = (accepted_window + rejected_window).max(1);
            accepted_window as f64 / total as f64
        }
        "main_peak_acc" | "current_peak" => current_peak,
        "panel_probe_acc" => metrics.panel_probe_acc,
        "H" => h as f64,
        "jackpot" => jackpot as f64,
        "time_pct" => step as f64 / total_steps.max(1) as f64,
        _ => 0.0,
    }
}

#[derive(Clone)]
struct ArchiveEntry {
    state_id: String,
    family_id: String,
    archive_cell_id: Option<usize>,
    panel_index: usize,
    step: usize,
    current_peak: f64,
    panel_probe_acc: f64,
    psi_pred: f64,
    net: Network,
    proj: Int8Projection,
}

struct ArchiveParentLogWriter {
    writer: BufWriter<File>,
    path: PathBuf,
}

impl ArchiveParentLogWriter {
    fn new(path: PathBuf) -> Self {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent).expect("failed to create archive parent log directory");
            }
        }
        let mut writer =
            BufWriter::new(File::create(&path).expect("failed to create archive parent log"));
        writeln!(
            writer,
            "step,panel_index,policy,archive_size,selected_parent_state_id,selected_parent_family_id,selected_archive_cell_id,selected_parent_panel_index,selected_parent_step,selected_parent_score,selected_parent_current_peak,selected_parent_cell_confidence,restored"
        )
        .expect("failed to write archive parent log header");
        Self { writer, path }
    }

    fn write_choice(
        &mut self,
        step: usize,
        panel_index: usize,
        policy: ArchiveParentPolicyMode,
        archive_size: usize,
        selected: Option<&ArchiveEntry>,
        selected_score: f64,
        selected_confidence: f64,
        restored: bool,
    ) {
        if let Some(entry) = selected {
            let cell_id = entry
                .archive_cell_id
                .map(|v| v.to_string())
                .unwrap_or_default();
            writeln!(
                self.writer,
                "{},{},{},{},{},{},{},{},{},{:.17},{:.17},{:.17},{}",
                step,
                panel_index,
                policy.as_str(),
                archive_size,
                entry.state_id,
                entry.family_id,
                cell_id,
                entry.panel_index,
                entry.step,
                selected_score,
                entry.current_peak,
                selected_confidence,
                restored
            )
            .expect("failed to write archive parent log row");
        } else {
            writeln!(
                self.writer,
                "{},{},{},{},,,,,,,,,{}",
                step,
                panel_index,
                policy.as_str(),
                archive_size,
                restored
            )
            .expect("failed to write archive parent log row");
        }
    }

    fn flush(&mut self) {
        self.writer
            .flush()
            .expect("failed to flush archive parent log");
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OperatorPolicyMode {
    Baseline,
    StaticPrior,
    PriorEwma,
}

impl OperatorPolicyMode {
    fn parse(value: &str) -> Self {
        match value {
            "baseline" => Self::Baseline,
            "static-prior" | "static_prior" => Self::StaticPrior,
            "prior-ewma" | "prior_ewma" => Self::PriorEwma,
            other => {
                panic!("--operator-policy expects baseline|static-prior|prior-ewma, got {other}")
            }
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::StaticPrior => "static-prior",
            Self::PriorEwma => "prior-ewma",
        }
    }

    fn uses_weights(self) -> bool {
        !matches!(self, Self::Baseline)
    }
}

#[derive(Clone, Default)]
struct OperatorWindowStats {
    attempts: u64,
    selected: u64,
    accepted: u64,
    positive_delta: u64,
}

struct OperatorPolicyLogWriter {
    writer: BufWriter<File>,
    path: PathBuf,
}

impl OperatorPolicyLogWriter {
    fn new(path: PathBuf) -> Self {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent).expect("failed to create operator policy log directory");
            }
        }
        let mut writer =
            BufWriter::new(File::create(&path).expect("failed to create operator policy log"));
        writeln!(
            writer,
            "step,operator_id,probability,normalized_entropy,attempts_window,selected_window,accepted_window,positive_delta_window,positive_delta_rate_window,accepted_rate_window"
        )
        .expect("failed to write operator policy log header");
        Self { writer, path }
    }

    fn flush(&mut self) {
        self.writer
            .flush()
            .expect("failed to flush operator policy log");
    }
}

struct OperatorPolicyController {
    mode: OperatorPolicyMode,
    static_probabilities: Vec<f64>,
    probabilities: Vec<f64>,
    ewma_usefulness: Vec<f64>,
    epsilon_random: f64,
    weight_floor: f64,
    weight_cap: f64,
    ewma_alpha: f64,
    window_stats: Vec<OperatorWindowStats>,
    log_writer: Option<OperatorPolicyLogWriter>,
}

impl OperatorPolicyController {
    fn new(
        mode: OperatorPolicyMode,
        prior_path: Option<&Path>,
        h: usize,
        epsilon_random: f64,
        weight_floor: f64,
        weight_cap: f64,
        ewma_alpha: f64,
        log_path: Option<PathBuf>,
    ) -> Self {
        let baseline = baseline_probabilities();
        let static_probabilities = if mode.uses_weights() {
            let path = prior_path.expect("--operator-prior is required for weighted policies");
            load_operator_prior(path, h)
        } else {
            baseline.clone()
        };
        let probabilities = static_probabilities.clone();
        let log_writer = log_path.map(OperatorPolicyLogWriter::new);
        Self {
            mode,
            static_probabilities,
            probabilities,
            ewma_usefulness: vec![0.0; MUTATION_OPERATORS.len()],
            epsilon_random,
            weight_floor,
            weight_cap,
            ewma_alpha,
            window_stats: vec![OperatorWindowStats::default(); MUTATION_OPERATORS.len()],
            log_writer,
        }
    }

    fn weights_for_step(&self) -> Option<Vec<f64>> {
        if self.mode.uses_weights() {
            Some(self.probabilities.clone())
        } else {
            None
        }
    }

    fn observe(&mut self, record: &CandidateTraceRecord) {
        let idx = mutation_operator_index(record.operator_id)
            .unwrap_or_else(|| panic!("unknown candidate operator_id: {}", record.operator_id));
        let stats = &mut self.window_stats[idx];
        stats.attempts += 1;
        if record.selected {
            stats.selected += 1;
        }
        if record.accepted {
            stats.accepted += 1;
        }
        if record.evaluated && record.delta_u > 0.0 {
            stats.positive_delta += 1;
        }
        if matches!(self.mode, OperatorPolicyMode::PriorEwma) && record.evaluated {
            let usefulness = record.delta_u.max(0.0);
            self.ewma_usefulness[idx] =
                (1.0 - self.ewma_alpha) * self.ewma_usefulness[idx] + self.ewma_alpha * usefulness;
        }
    }

    fn end_step(&mut self) {
        if !matches!(self.mode, OperatorPolicyMode::PriorEwma) {
            return;
        }
        let online = normalize_or_baseline(&self.ewma_usefulness);
        let mixed: Vec<f64> = self
            .static_probabilities
            .iter()
            .zip(online.iter())
            .map(|(prior, live)| 0.70 * prior + 0.30 * live)
            .collect();
        self.probabilities = cap_floor_and_explore(
            &mixed,
            self.epsilon_random,
            self.weight_floor,
            self.weight_cap,
        );
    }

    fn write_window(&mut self, step: usize) {
        let Some(writer) = self.log_writer.as_mut() else {
            self.window_stats = vec![OperatorWindowStats::default(); MUTATION_OPERATORS.len()];
            return;
        };
        let entropy = normalized_entropy(&self.probabilities);
        for (idx, op) in MUTATION_OPERATORS.iter().enumerate() {
            let stats = &self.window_stats[idx];
            let positive_rate = if stats.attempts > 0 {
                stats.positive_delta as f64 / stats.attempts as f64
            } else {
                0.0
            };
            let accepted_rate = if stats.attempts > 0 {
                stats.accepted as f64 / stats.attempts as f64
            } else {
                0.0
            };
            writeln!(
                writer.writer,
                "{},{},{:.17},{:.17},{},{},{},{},{:.17},{:.17}",
                step,
                op.id,
                self.probabilities[idx],
                entropy,
                stats.attempts,
                stats.selected,
                stats.accepted,
                stats.positive_delta,
                positive_rate,
                accepted_rate
            )
            .expect("failed to write operator policy log row");
        }
        self.window_stats = vec![OperatorWindowStats::default(); MUTATION_OPERATORS.len()];
    }

    fn flush(&mut self) {
        if let Some(writer) = self.log_writer.as_mut() {
            writer.flush();
        }
    }
}

fn baseline_probabilities() -> Vec<f64> {
    (0..MUTATION_OPERATORS.len())
        .map(mutation_operator_baseline_probability)
        .collect()
}

fn normalize_or_baseline(values: &[f64]) -> Vec<f64> {
    let sum: f64 = values.iter().copied().filter(|v| *v > 0.0).sum();
    if sum.is_finite() && sum > 0.0 {
        values.iter().map(|v| v.max(0.0) / sum).collect()
    } else {
        baseline_probabilities()
    }
}

fn cap_floor_and_explore(raw: &[f64], epsilon_random: f64, floor: f64, cap: f64) -> Vec<f64> {
    let baseline = baseline_probabilities();
    let mut weighted = Vec::with_capacity(raw.len());
    for (idx, raw_p) in raw.iter().copied().enumerate() {
        let base = baseline[idx].max(1e-12);
        let multiplier = (raw_p.max(0.0) / base).clamp(floor, cap);
        weighted.push(base * multiplier);
    }
    let sum: f64 = weighted.iter().sum();
    let normalized: Vec<f64> = weighted.iter().map(|v| v / sum).collect();
    normalized
        .iter()
        .zip(baseline.iter())
        .map(|(p, b)| (1.0 - epsilon_random) * p + epsilon_random * b)
        .collect()
}

fn normalized_entropy(probabilities: &[f64]) -> f64 {
    let entropy: f64 = probabilities
        .iter()
        .copied()
        .filter(|p| *p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    entropy / (probabilities.len() as f64).ln()
}

fn load_operator_prior(path: &Path, h: usize) -> Vec<f64> {
    let text = std::fs::read_to_string(path).expect("failed to read --operator-prior");
    let mut probabilities = vec![None; MUTATION_OPERATORS.len()];
    let mut header: Vec<&str> = Vec::new();
    for (line_idx, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if line_idx == 0 {
            header = cols;
            continue;
        }
        let col = |name: &str| -> &str {
            let idx = header
                .iter()
                .position(|h| *h == name)
                .unwrap_or_else(|| panic!("operator prior missing column {name}"));
            cols.get(idx)
                .copied()
                .unwrap_or_else(|| panic!("operator prior row missing column {name}"))
        };
        let row_h: usize = col("H").parse().expect("invalid H in operator prior");
        if row_h != h {
            continue;
        }
        let operator_id = col("operator_id");
        let idx = mutation_operator_index(operator_id)
            .unwrap_or_else(|| panic!("unknown operator_id in prior: {operator_id}"));
        let probability: f64 = col("final_probability")
            .parse()
            .expect("invalid final_probability in operator prior");
        assert!(
            probability > 0.0,
            "operator prior probabilities must be positive"
        );
        probabilities[idx] = Some(probability);
    }
    let out: Vec<f64> = probabilities
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            value.unwrap_or_else(|| {
                panic!(
                    "operator prior missing H={} operator={}",
                    h, MUTATION_OPERATORS[idx].id
                )
            })
        })
        .collect();
    let sum: f64 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "operator prior probabilities must sum to 1"
    );
    out
}

fn quantize_embedding_to_input(
    table: &VcbpTable,
    embedding: &[f32],
    input: &mut [i32],
    input_end: usize,
    input_scatter: bool,
) {
    if !input_scatter {
        table.quantize_to_input(embedding, &mut input[..table.e], MAX_CHARGE);
        return;
    }

    let mut base = vec![0i32; table.e];
    table.quantize_to_input(embedding, &mut base, MAX_CHARGE);
    for dst in input.iter_mut().take(input_end) {
        *dst = 0;
    }
    for idx in 0..input_end.min(input.len()) {
        input[idx] = base[idx % table.e];
    }
}

fn build_corpus_pairs(
    corpus: &[u8],
    table: &VcbpTable,
    max_classes: usize,
) -> (Vec<u16>, Vec<usize>, Vec<u16>, usize) {
    let n_pairs = corpus.len() / 2;
    let mut pair_ids = Vec::with_capacity(n_pairs);
    let mut freq = vec![0u32; 65536];
    for i in 0..n_pairs {
        let pid = VcbpTable::pair_id(corpus[i * 2], corpus[i * 2 + 1]);
        pair_ids.push(pid);
        freq[pid as usize] += 1;
    }
    let mut hot_freq: Vec<(u16, u32)> = (0..65536u32)
        .filter(|&v| table.is_hot(v as u16) && freq[v as usize] > 0)
        .map(|v| (v as u16, freq[v as usize]))
        .collect();
    hot_freq.sort_by(|a, b| b.1.cmp(&a.1));
    let n_classes = hot_freq.len().min(max_classes);
    hot_freq.truncate(n_classes);
    let top_ids: Vec<u16> = hot_freq.iter().map(|&(id, _)| id).collect();
    let mut top_to_idx = vec![usize::MAX; 65536];
    for (i, &id) in top_ids.iter().enumerate() {
        top_to_idx[id as usize] = i;
    }
    (pair_ids, top_to_idx, top_ids, n_classes)
}

fn build_pair_bigram(pair_ids: &[u16], hot_to_idx: &[usize], n_hot: usize) -> Vec<Vec<f64>> {
    let mut counts = vec![vec![0u32; n_hot]; n_hot];
    for i in 0..pair_ids.len().saturating_sub(1) {
        let ci = hot_to_idx[pair_ids[i] as usize];
        let ni = hot_to_idx[pair_ids[i + 1] as usize];
        if ci != usize::MAX && ni != usize::MAX {
            counts[ci][ni] += 1;
        }
    }
    counts
        .iter()
        .map(|row| {
            let total: f64 = row.iter().map(|&c| c as f64).sum();
            if total < 1.0 {
                vec![1.0 / n_hot as f64; n_hot]
            } else {
                row.iter().map(|&c| c as f64 / total).collect()
            }
        })
        .collect()
}

fn eval_accuracy_proj(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    len: usize,
    rng: &mut StdRng,
    propagation: &instnct_core::PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    input_end: usize,
    input_scatter: bool,
) -> f64 {
    let n = pair_ids.len();
    if n <= len + 1 {
        return 0.0;
    }
    let off = rng.gen_range(1..=n - len - 1);
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let _prev_id = if i > 0 { pair_ids[off + i - 1] } else { cur_id };
        let tgt_id = pair_ids[off + i + 1];
        let tgt_idx = hot_to_idx[tgt_id as usize];
        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        quantize_embedding_to_input(table, emb, &mut input, input_end, input_scatter);
        net.propagate(&input, propagation).unwrap();
        let predicted_idx = proj.predict(&net.charge_vec(output_start..neuron_count));
        if tgt_idx != usize::MAX && predicted_idx == tgt_idx {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

fn eval_smooth_proj(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    bigram: &[Vec<f64>],
    len: usize,
    rng: &mut StdRng,
    propagation: &instnct_core::PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    input_end: usize,
    input_scatter: bool,
) -> f64 {
    let n = pair_ids.len();
    if n <= len + 1 {
        return 0.0;
    }
    let off = rng.gen_range(1..=n - len - 1);
    net.reset();
    let mut total_cos = 0.0f64;
    let mut counted = 0usize;
    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let cur_idx = hot_to_idx[cur_id as usize];
        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        quantize_embedding_to_input(table, emb, &mut input, input_end, input_scatter);
        net.propagate(&input, propagation).unwrap();
        if cur_idx == usize::MAX {
            continue;
        }
        let scores = proj.raw_scores(&net.charge_vec(output_start..neuron_count));
        let probs = softmax(&scores);
        let target = &bigram[cur_idx];
        total_cos += cosine_similarity(&probs, target);
        counted += 1;
    }
    if counted == 0 {
        0.0
    } else {
        total_cos / counted as f64
    }
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    squared_distance(a, b).sqrt()
}

fn f_active(matrix: &[Vec<f64>]) -> f64 {
    if matrix.is_empty() || matrix[0].is_empty() {
        return 0.0;
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let active = (0..cols)
        .filter(|&col| {
            let first = matrix[0][col];
            (1..rows).any(|row| matrix[row][col] != first)
        })
        .count();
    active as f64 / cols as f64
}

fn output_entropy_stats(entropies: &[f64]) -> (f64, f64) {
    if entropies.is_empty() {
        return (0.0, 0.0);
    }
    let mean = entropies.iter().sum::<f64>() / entropies.len() as f64;
    let var = entropies
        .iter()
        .map(|value| {
            let d = value - mean;
            d * d
        })
        .sum::<f64>()
        / entropies.len() as f64;
    (mean, var)
}

fn stable_rank(matrix: &[Vec<f64>]) -> f64 {
    if matrix.is_empty() || matrix[0].is_empty() {
        return 0.0;
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let fro2 = matrix.iter().flatten().map(|v| v * v).sum::<f64>();
    if fro2 <= 0.0 {
        return 0.0;
    }

    let mut v = vec![1.0 / (cols as f64).sqrt(); cols];
    for _ in 0..50 {
        let mut yv = vec![0.0; rows];
        for (row_idx, row) in matrix.iter().enumerate() {
            yv[row_idx] = row.iter().zip(&v).map(|(a, b)| a * b).sum();
        }
        let mut z = vec![0.0; cols];
        for (row, y) in matrix.iter().zip(&yv) {
            for col in 0..cols {
                z[col] += row[col] * y;
            }
        }
        let norm = z.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= 1e-12 {
            return 0.0;
        }
        for col in 0..cols {
            v[col] = z[col] / norm;
        }
    }

    let lambda = matrix
        .iter()
        .map(|row| row.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>())
        .map(|y| y * y)
        .sum::<f64>();
    if lambda <= 1e-12 {
        0.0
    } else {
        fro2 / lambda
    }
}

fn numerical_rank(matrix: &[Vec<f64>], tol: f64) -> usize {
    if matrix.is_empty() || matrix[0].is_empty() {
        return 0;
    }
    let mut mat = matrix.to_vec();
    let rows = mat.len();
    let cols = mat[0].len();
    let mut rank = 0usize;
    for col in 0..cols {
        let mut pivot = rank;
        for row in rank..rows {
            if mat[row][col].abs() > mat[pivot][col].abs() {
                pivot = row;
            }
        }
        if mat[pivot][col].abs() <= tol {
            continue;
        }
        mat.swap(rank, pivot);
        let pivot_val = mat[rank][col];
        for c in col..cols {
            mat[rank][c] /= pivot_val;
        }
        for row in 0..rows {
            if row == rank {
                continue;
            }
            let factor = mat[row][col];
            if factor.abs() <= tol {
                continue;
            }
            for c in col..cols {
                mat[row][c] -= factor * mat[rank][c];
            }
        }
        rank += 1;
        if rank == rows {
            break;
        }
    }
    rank
}

fn separation_sp(inputs: &[Vec<f64>], outputs: &[Vec<f64>]) -> f64 {
    let n = inputs.len().min(outputs.len());
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let in_dist = euclidean_distance(&inputs[i], &inputs[j]);
            let out_dist = euclidean_distance(&outputs[i], &outputs[j]);
            total += out_dist / (in_dist + 1e-12);
            count += 1;
        }
    }
    total / count as f64
}

fn entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

fn compute_panel_metrics(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    propagation: &instnct_core::PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    input_end: usize,
    input_scatter: bool,
) -> PanelMetrics {
    let snapshot = net.save_state();
    let usable_pairs = pair_ids.len().saturating_sub(1);
    let probe_count = PANEL_PROBE_COUNT.min(usable_pairs.max(1));
    let mut input_matrix = Vec::with_capacity(probe_count);
    let mut output_matrix = Vec::with_capacity(probe_count);
    let mut predictions = Vec::with_capacity(probe_count);
    let mut entropy_values = Vec::with_capacity(probe_count);
    let mut unique_inputs = HashSet::new();
    let mut correct = 0usize;

    if usable_pairs == 0 {
        net.restore_state(&snapshot);
        return PanelMetrics {
            panel_probe_acc: 0.0,
            unique_predictions: 0,
            collision_rate: 0.0,
            f_active: 0.0,
            h_output_mean: 0.0,
            h_output_var: 0.0,
            stable_rank: 0.0,
            kernel_rank: 0,
            separation_sp: 0.0,
        };
    }

    for k in 0..probe_count {
        let idx = (k * usable_pairs) / probe_count;
        let cur_id = pair_ids[idx];
        let tgt_id = pair_ids[idx + 1];
        let tgt_idx = hot_to_idx[tgt_id as usize];
        unique_inputs.insert(cur_id);

        net.reset();
        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        quantize_embedding_to_input(table, emb, &mut input, input_end, input_scatter);
        net.propagate(&input, propagation).unwrap();

        let charges_u8 = net.charge_vec(output_start..neuron_count);
        let charges: Vec<f64> = charges_u8.iter().map(|&v| v as f64).collect();
        let scores = proj.raw_scores(&charges_u8);
        let probs = softmax(&scores);
        let pred = proj.predict(&charges_u8);
        if tgt_idx != usize::MAX && pred == tgt_idx {
            correct += 1;
        }

        entropy_values.push(entropy(&probs));
        predictions.push(pred);
        input_matrix.push(emb.iter().map(|&v| v as f64).collect::<Vec<_>>());
        output_matrix.push(charges);
    }

    net.restore_state(&snapshot);

    let unique_predictions = predictions.iter().copied().collect::<HashSet<_>>().len();
    let collision_rate = if unique_inputs.is_empty() {
        0.0
    } else {
        unique_predictions as f64 / unique_inputs.len() as f64
    };
    let (h_output_mean, h_output_var) = output_entropy_stats(&entropy_values);

    PanelMetrics {
        panel_probe_acc: correct as f64 / probe_count as f64,
        unique_predictions,
        collision_rate,
        f_active: f_active(&output_matrix),
        h_output_mean,
        h_output_var,
        stable_rank: stable_rank(&output_matrix),
        kernel_rank: numerical_rank(&output_matrix, 1e-6),
        separation_sp: separation_sp(&input_matrix, &output_matrix),
    }
}

fn quick_output_diversity_score(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    propagation: &instnct_core::PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    input_end: usize,
    input_scatter: bool,
) -> f64 {
    let snapshot = net.save_state();
    let mut probes = vec![
        VcbpTable::pair_id(b't', b'h'),
        VcbpTable::pair_id(b'e', b' '),
        VcbpTable::pair_id(b' ', b't'),
        VcbpTable::pair_id(b'a', b'l'),
    ];
    if !pair_ids.is_empty() {
        let count = 4usize.min(pair_ids.len());
        for k in 0..count {
            probes.push(pair_ids[(k * pair_ids.len()) / count]);
        }
    }
    probes.sort_unstable();
    probes.dedup();

    let mut charges_list: Vec<Vec<u8>> = Vec::with_capacity(probes.len());
    let mut predictions = Vec::with_capacity(probes.len());
    for pair_id in probes.iter().copied() {
        net.reset();
        let emb = table.embed_id(pair_id);
        let mut input = vec![0i32; neuron_count];
        quantize_embedding_to_input(table, emb, &mut input, input_end, input_scatter);
        net.propagate(&input, propagation).unwrap();
        let charges = net.charge_vec(output_start..neuron_count);
        predictions.push(proj.predict(&charges));
        charges_list.push(charges);
    }
    net.restore_state(&snapshot);

    if charges_list.len() < 2 {
        return 0.0;
    }
    let output_len = neuron_count.saturating_sub(output_start).max(1);
    let mut diff_dims = 0usize;
    let mut pair_count = 0usize;
    for i in 0..charges_list.len() {
        for j in (i + 1)..charges_list.len() {
            diff_dims += charges_list[i]
                .iter()
                .zip(charges_list[j].iter())
                .filter(|(a, b)| a != b)
                .count();
            pair_count += 1;
        }
    }
    let charge_diversity = diff_dims as f64 / (pair_count.max(1) * output_len) as f64;
    let unique_predictions = predictions.iter().copied().collect::<HashSet<_>>().len();
    let prediction_diversity = unique_predictions as f64 / charges_list.len() as f64;
    0.5 * charge_diversity + 0.5 * prediction_diversity
}

/// Seed mutual inhibition: two clusters in the output zone that suppress each other.
fn seed_mutual_inhibition(net: &mut Network, output_start: usize, h: usize, rng: &mut impl Rng) {
    let mid = output_start + (h - output_start) / 2;

    // Cluster A: output_start..mid
    // Cluster B: mid..h
    // Cross-inhibition: random edges from A→B and B→A, with inhibitory polarity

    let n_cross = 20; // number of inhibitory cross-connections per direction

    for _ in 0..n_cross {
        // A → B (A inhibits B)
        let src = rng.gen_range(output_start..mid) as u16;
        let tgt = rng.gen_range(mid..h) as u16;
        net.graph_mut().add_edge(src, tgt);
        // Make source neuron inhibitory (polarity = -1)
        // But we can't set per-edge polarity, only per-neuron.
        // So we set the SOURCE neuron to inhibitory.
        // This means it inhibits ALL its targets, not just cross-cluster.
        // Compromise: pick dedicated "inhibitor" neurons at cluster boundary.
    }

    for _ in 0..n_cross {
        // B → A (B inhibits A)
        let src = rng.gen_range(mid..h) as u16;
        let tgt = rng.gen_range(output_start..mid) as u16;
        net.graph_mut().add_edge(src, tgt);
    }

    // Set boundary neurons as inhibitory (last 5 of each cluster)
    let a_boundary_start = mid - 5;
    let b_boundary_start = h - 5;
    for n in a_boundary_start..mid {
        if n < h {
            net.polarity_mut()[n] = -1;
            // Lower threshold so they fire easily and suppress the other cluster
            net.spike_data_mut()[n].threshold = 1;
        }
    }
    for n in b_boundary_start..h {
        net.polarity_mut()[n] = -1;
        net.spike_data_mut()[n].threshold = 1;
    }

    println!(
        "  Mutual inhibition: {} cross-edges per direction, boundary neurons inhibitory",
        n_cross
    );
    println!("  Cluster A: neurons {}..{}", output_start, mid);
    println!("  Cluster B: neurons {}..{}", mid, h);
}

// Rooted pathways (same as other examples)
fn bfs_forward(net: &Network, starts: &[usize], max_hops: usize) -> Vec<bool> {
    let h = net.neuron_count();
    let mut reached = vec![false; h];
    let mut queue = VecDeque::new();
    for &s in starts {
        reached[s] = true;
        queue.push_back((s, 0usize));
    }
    let mut adj: Vec<Vec<u16>> = vec![Vec::new(); h];
    for edge in net.graph().iter_edges() {
        adj[edge.source as usize].push(edge.target);
    }
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_hops {
            continue;
        }
        for &tgt in &adj[node] {
            if !reached[tgt as usize] {
                reached[tgt as usize] = true;
                queue.push_back((tgt as usize, depth + 1));
            }
        }
    }
    reached
}

fn bfs_reverse(net: &Network, ends: &[usize], max_hops: usize) -> Vec<bool> {
    let h = net.neuron_count();
    let mut reached = vec![false; h];
    let mut queue = VecDeque::new();
    for &e in ends {
        reached[e] = true;
        queue.push_back((e, 0usize));
    }
    let mut rev_adj: Vec<Vec<u16>> = vec![Vec::new(); h];
    for edge in net.graph().iter_edges() {
        rev_adj[edge.target as usize].push(edge.source);
    }
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_hops {
            continue;
        }
        for &src in &rev_adj[node] {
            if !reached[src as usize] {
                reached[src as usize] = true;
                queue.push_back((src as usize, depth + 1));
            }
        }
    }
    reached
}

fn seed_rooted_pathways(
    net: &mut Network,
    input_end: usize,
    output_start: usize,
    n_pathways: usize,
    rng: &mut impl Rng,
) -> usize {
    let h = net.neuron_count();
    let from_input = bfs_forward(net, &(0..input_end).collect::<Vec<_>>(), 4);
    let to_output = bfs_reverse(net, &(output_start..h).collect::<Vec<_>>(), 4);
    let input_anchors: Vec<usize> = (0..h)
        .filter(|&n| from_input[n] && n < output_start)
        .collect();
    let output_anchors: Vec<usize> = (0..h).filter(|&n| to_output[n] && n >= input_end).collect();
    if input_anchors.is_empty() || output_anchors.is_empty() {
        for _ in 0..n_pathways.min(5) {
            net.graph_mut().add_edge(
                rng.gen_range(0..input_end) as u16,
                rng.gen_range(output_start..h) as u16,
            );
        }
        return 0;
    }
    let mut built = 0;
    for _ in 0..n_pathways {
        let ai = input_anchors[rng.gen_range(0..input_anchors.len())];
        let ao = output_anchors[rng.gen_range(0..output_anchors.len())];
        let avail: Vec<usize> = (0..h).filter(|n| *n != ai && *n != ao).collect();
        if avail.len() < 2 {
            continue;
        }
        let nm = rng.gen_range(2..=3.min(avail.len()));
        let mut mids = Vec::new();
        let mut pool = avail;
        for _ in 0..nm {
            let idx = rng.gen_range(0..pool.len());
            mids.push(pool.swap_remove(idx));
        }
        let mut chain = vec![ai];
        chain.extend(&mids);
        chain.push(ao);
        let mut added = false;
        for w in chain.windows(2) {
            if net.graph_mut().add_edge(w[0] as u16, w[1] as u16) {
                added = true;
            }
        }
        if net.graph_mut().add_edge(ao as u16, ai as u16) {
            added = true;
        }
        if added {
            for &n in &chain {
                let sd = &mut net.spike_data_mut()[n];
                if sd.threshold > 1 {
                    sd.threshold -= 1;
                }
            }
            built += 1;
        }
    }
    built
}

fn prime_highway_neuron(net: &mut Network, idx: usize) {
    let sd = &mut net.spike_data_mut()[idx];
    sd.threshold = 0;
    sd.channel = 1;
    net.polarity_mut()[idx] = 1;
}

fn prime_readout_neuron(net: &mut Network, idx: usize) {
    let sd = &mut net.spike_data_mut()[idx];
    sd.threshold = 15;
    // Channel 8 keeps max-threshold readout anchors from firing during a 6-tick
    // token, so direct input charge is preserved instead of reset to a binary spike.
    sd.channel = 8;
    net.polarity_mut()[idx] = 1;
}

fn seed_embedding_anchored_highways(
    net: &mut Network,
    embedding_dim: usize,
    input_end: usize,
    output_start: usize,
    per_dim: usize,
    rng: &mut impl Rng,
) -> usize {
    let h = net.neuron_count();
    if per_dim == 0 || output_start >= input_end || input_end >= h {
        return 0;
    }
    let mut built = 0usize;
    let output_only_len = h - input_end;
    let mut direct_edges: Vec<(u16, u16)> = Vec::new();
    for src in 0..embedding_dim.min(input_end).min(h) {
        for path_idx in 0..per_dim {
            let hub_low = rng.gen_range(output_start..input_end);
            let hub_high = rng.gen_range(output_start..input_end);
            let out = rng.gen_range(input_end..h);
            let direct_out = input_end + ((src * per_dim + path_idx) % output_only_len);
            direct_edges.push((src as u16, direct_out as u16));
            let mut added = false;
            if net.graph_mut().add_edge(src as u16, direct_out as u16) {
                added = true;
            }
            if net.graph_mut().add_edge(src as u16, hub_low as u16) {
                added = true;
            }
            if net.graph_mut().add_edge(hub_low as u16, hub_high as u16) {
                added = true;
            }
            if net.graph_mut().add_edge(hub_high as u16, out as u16) {
                added = true;
            }
            prime_highway_neuron(net, src);
            prime_highway_neuron(net, hub_low);
            prime_highway_neuron(net, hub_high);
            prime_readout_neuron(net, direct_out);
            prime_readout_neuron(net, out);
            if added {
                built += 1;
            }
        }
    }
    let direct_targets: HashSet<u16> = direct_edges.iter().map(|&(_, target)| target).collect();
    let allowed_direct_edges: HashSet<(u16, u16)> = direct_edges.iter().copied().collect();
    let spurious_incoming: Vec<(u16, u16)> = net
        .graph()
        .iter_edges()
        .filter(|edge| {
            direct_targets.contains(&edge.target)
                && !allowed_direct_edges.contains(&(edge.source, edge.target))
        })
        .map(|edge| (edge.source, edge.target))
        .collect();
    for (source, target) in spurious_incoming {
        net.graph_mut().remove_edge(source, target);
    }
    built
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("Usage: evolve_mutual_inhibition <corpus.txt> <packed.bin> [--steps N]");
        std::process::exit(1);
    }

    let corpus_path = &args[0];
    let packed_path = &args[1];
    let mut steps = DEFAULT_STEPS;
    let mut cli_seed: u64 = 42;
    let mut cli_h: usize = 256;
    let mut jackpot: usize = 9;
    let mut cli_ticks: Option<usize> = None;
    let mut cli_chain_count: Option<usize> = None;
    let mut embedding_anchored_highways: usize = 0;
    let mut diversity_guard_lambda: f64 = 0.0;
    let mut cli_accept_ties: Option<bool> = None;
    let mut cli_accept_policy: Option<String> = None;
    let mut cli_neutral_p: f64 = 1.0;
    let mut cli_accept_epsilon: f64 = 0.0;
    let mut input_scatter = false;
    let mut candidate_log_path: Option<PathBuf> = None;
    let mut checkpoint_at_end: Option<PathBuf> = None;
    let mut panel_interval: Option<usize> = None;
    let mut panel_log_path: Option<PathBuf> = None;
    let mut operator_policy_name = String::from("baseline");
    let mut operator_prior_path: Option<PathBuf> = None;
    let mut operator_epsilon_random = 0.15f64;
    let mut operator_weight_floor = 0.25f64;
    let mut operator_weight_cap = 4.0f64;
    let mut operator_ewma_alpha = 0.05f64;
    let mut operator_policy_log_path: Option<PathBuf> = None;
    let mut d8_state_log_path: Option<PathBuf> = None;
    let mut archive_parent_policy_name = String::from("current-best");
    let mut archive_parent_log_path: Option<PathBuf> = None;
    let mut archive_max_size: usize = 64;
    let mut archive_switch_interval_panels: usize = 1;
    let mut archive_min_cell_confidence = 0.0f64;
    let mut archive_p2_model_path: Option<PathBuf> = None;
    let mut phase = String::from("default");
    let mut arm = String::from("default");
    let mut run_id = String::from("default");
    let mut seed_list: Option<String> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                i += 1;
                steps = args[i].parse().unwrap();
            }
            "--seed" => {
                i += 1;
                cli_seed = args[i].parse().unwrap();
            }
            "--H" => {
                i += 1;
                cli_h = args[i].parse().unwrap();
            }
            "--jackpot" => {
                i += 1;
                jackpot = args[i].parse().unwrap();
            }
            "--ticks" => {
                i += 1;
                cli_ticks = Some(args[i].parse().unwrap());
            }
            "--chain-count" => {
                i += 1;
                cli_chain_count = Some(args[i].parse().unwrap());
            }
            "--embedding-anchored-highways" => {
                i += 1;
                embedding_anchored_highways = args[i].parse().unwrap();
            }
            "--diversity-guard-lambda" => {
                i += 1;
                diversity_guard_lambda = args[i].parse().unwrap();
                assert!(
                    diversity_guard_lambda >= 0.0,
                    "--diversity-guard-lambda must be >= 0"
                );
            }
            "--accept-ties" => {
                i += 1;
                cli_accept_ties = Some(match args[i].as_str() {
                    "true" | "1" | "yes" | "on" => true,
                    "false" | "0" | "no" | "off" => false,
                    value => panic!("--accept-ties expects true|false, got {value}"),
                });
            }
            "--accept-policy" => {
                i += 1;
                cli_accept_policy = Some(args[i].clone());
            }
            "--neutral-p" => {
                i += 1;
                cli_neutral_p = args[i].parse().unwrap();
                assert!(
                    (0.0..=1.0).contains(&cli_neutral_p),
                    "--neutral-p must be in [0, 1]"
                );
            }
            "--accept-epsilon" => {
                i += 1;
                cli_accept_epsilon = args[i].parse().unwrap();
                assert!(cli_accept_epsilon >= 0.0, "--accept-epsilon must be >= 0");
            }
            "--input-scatter" => {
                input_scatter = true;
            }
            "--candidate-log" => {
                i += 1;
                candidate_log_path = Some(PathBuf::from(&args[i]));
            }
            "--checkpoint-at-end" => {
                i += 1;
                checkpoint_at_end = Some(PathBuf::from(&args[i]));
            }
            "--panel-interval" => {
                i += 1;
                let interval: usize = args[i].parse().unwrap();
                assert!(interval > 0, "--panel-interval must be > 0");
                panel_interval = Some(interval);
            }
            "--panel-log" => {
                i += 1;
                panel_log_path = Some(PathBuf::from(&args[i]));
            }
            "--operator-policy" => {
                i += 1;
                operator_policy_name = args[i].clone();
            }
            "--operator-prior" => {
                i += 1;
                operator_prior_path = Some(PathBuf::from(&args[i]));
            }
            "--operator-epsilon-random" => {
                i += 1;
                operator_epsilon_random = args[i].parse().unwrap();
                assert!(
                    (0.0..=1.0).contains(&operator_epsilon_random),
                    "--operator-epsilon-random must be in [0, 1]"
                );
            }
            "--operator-weight-floor" => {
                i += 1;
                operator_weight_floor = args[i].parse().unwrap();
                assert!(
                    operator_weight_floor > 0.0,
                    "--operator-weight-floor must be > 0"
                );
            }
            "--operator-weight-cap" => {
                i += 1;
                operator_weight_cap = args[i].parse().unwrap();
                assert!(
                    operator_weight_cap >= operator_weight_floor,
                    "--operator-weight-cap must be >= --operator-weight-floor"
                );
            }
            "--operator-ewma-alpha" => {
                i += 1;
                operator_ewma_alpha = args[i].parse().unwrap();
                assert!(
                    (0.0..=1.0).contains(&operator_ewma_alpha),
                    "--operator-ewma-alpha must be in [0, 1]"
                );
            }
            "--operator-policy-log" => {
                i += 1;
                operator_policy_log_path = Some(PathBuf::from(&args[i]));
            }
            "--d8-state-log" => {
                i += 1;
                d8_state_log_path = Some(PathBuf::from(&args[i]));
            }
            "--archive-parent-policy" => {
                i += 1;
                archive_parent_policy_name = args[i].clone();
            }
            "--archive-parent-log" => {
                i += 1;
                archive_parent_log_path = Some(PathBuf::from(&args[i]));
            }
            "--archive-max-size" => {
                i += 1;
                archive_max_size = args[i].parse().unwrap();
                assert!(archive_max_size > 0, "--archive-max-size must be > 0");
            }
            "--archive-switch-interval-panels" => {
                i += 1;
                archive_switch_interval_panels = args[i].parse().unwrap();
                assert!(
                    archive_switch_interval_panels > 0,
                    "--archive-switch-interval-panels must be > 0"
                );
            }
            "--archive-min-cell-confidence" => {
                i += 1;
                archive_min_cell_confidence = args[i].parse().unwrap();
                assert!(
                    (0.0..=1.0).contains(&archive_min_cell_confidence),
                    "--archive-min-cell-confidence must be in [0, 1]"
                );
            }
            "--archive-p2-model" => {
                i += 1;
                archive_p2_model_path = Some(PathBuf::from(&args[i]));
            }
            "--phase" => {
                i += 1;
                phase = args[i].clone();
            }
            "--arm" => {
                i += 1;
                arm = args[i].clone();
            }
            "--run-id" => {
                i += 1;
                run_id = args[i].clone();
            }
            "--seed-list" => {
                i += 1;
                seed_list = Some(args[i].clone());
            }
            other => panic!("unknown flag: {other}"),
        }
        i += 1;
    }

    println!("Loading...");
    let table = VcbpTable::from_packed(Path::new(packed_path)).unwrap();
    let corpus = std::fs::read(corpus_path).unwrap();
    let max_classes = 397;
    let (pair_ids, hot_to_idx, _hot_ids, n_classes) =
        build_corpus_pairs(&corpus, &table, max_classes);
    let bigram = build_pair_bigram(&pair_ids, &hot_to_idx, n_classes);

    let h = cli_h;
    let mut init = InitConfig::phi(h);
    if let Some(ticks) = cli_ticks {
        init.propagation.ticks_per_token = ticks;
    }
    if let Some(chain_count) = cli_chain_count {
        init.chain_count = chain_count;
    }
    if let Some(accept_ties) = cli_accept_ties {
        init.accept_ties = accept_ties;
    }
    let accept_policy_name = cli_accept_policy.unwrap_or_else(|| {
        if init.accept_ties {
            String::from("ties")
        } else {
            String::from("strict")
        }
    });
    let acceptance_policy = match accept_policy_name.as_str() {
        "strict" => AcceptancePolicy::Strict,
        "ties" => AcceptancePolicy::Ties,
        "zero-p" | "zero_p" => AcceptancePolicy::ZeroP {
            probability: cli_neutral_p,
            zero_tol: 1e-12,
        },
        "epsilon" => AcceptancePolicy::Epsilon {
            epsilon: cli_accept_epsilon,
        },
        other => panic!("--accept-policy expects strict|ties|zero-p|epsilon, got {other}"),
    };
    let operator_policy_mode = OperatorPolicyMode::parse(&operator_policy_name);
    let archive_parent_policy_mode = ArchiveParentPolicyMode::parse(&archive_parent_policy_name);
    if matches!(archive_parent_policy_mode, ArchiveParentPolicyMode::P2PsiConf)
        && archive_p2_model_path.is_none()
    {
        panic!("--archive-parent-policy p2-psi-conf requires --archive-p2-model");
    }
    init.accept_ties = matches!(acceptance_policy, AcceptancePolicy::Ties);
    let evo_config = init.evolution_config();

    println!("\n=== MUTUAL INHIBITION EXPERIMENT ===");
    println!(
        "  H={}, {} steps, {} classes, seed={}, jackpot={}, ticks={}, chain_count={}, embedding_anchored_highways={}, diversity_guard_lambda={:.3}, accept_ties={}, accept_policy={}, neutral_p={:.3}, accept_epsilon={:.6}, input_scatter={}, operator_policy={}, archive_parent_policy={}, phase={}, arm={}, run_id={}\n",
        h,
        steps,
        n_classes,
        cli_seed,
        jackpot,
        init.propagation.ticks_per_token,
        init.chain_count,
        embedding_anchored_highways,
        diversity_guard_lambda,
        init.accept_ties,
        accept_policy_name,
        cli_neutral_p,
        cli_accept_epsilon,
        input_scatter,
        operator_policy_mode.as_str(),
        archive_parent_policy_mode.as_str(),
        phase,
        arm,
        run_id
    );

    let t_start = Instant::now();
    let seed = cli_seed;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(&init, &mut rng);

    // Seed rooted pathways
    let n_paths = seed_rooted_pathways(
        &mut net,
        init.input_end(),
        init.output_start(),
        30,
        &mut rng,
    );
    println!("  Rooted pathways: {n_paths}, edges={}", net.edge_count());
    let anchored_paths = seed_embedding_anchored_highways(
        &mut net,
        table.e,
        init.input_end(),
        init.output_start(),
        embedding_anchored_highways,
        &mut rng,
    );
    if embedding_anchored_highways > 0 {
        println!(
            "  Embedding-anchored highways: {anchored_paths} (per_dim={embedding_anchored_highways}), edges={}",
            net.edge_count()
        );
    }

    // *** THE KEY: seed mutual inhibition ***
    seed_mutual_inhibition(&mut net, init.output_start(), h, &mut rng);
    println!("  After mutual inhibition: edges={}", net.edge_count());

    let mut proj = Int8Projection::new(
        init.phi_dim,
        n_classes,
        &mut StdRng::seed_from_u64(seed + 200),
    );
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let mut accepted = 0u32;
    let mut rejected = 0u32;
    let mut peak = 0.0f64;
    let mut candidate_log = candidate_log_path
        .as_deref()
        .map(|path| CandidateLogWriter::new(path, &run_id, &arm, seed, h));
    let inferred_panel_log_path = panel_log_path.clone().or_else(|| {
        checkpoint_at_end.as_ref().map(|path| {
            path.parent()
                .unwrap_or_else(|| Path::new("."))
                .join("panel_timeseries.csv")
        })
    });
    let fallback_panel_log_path = PathBuf::from("output")
        .join("phase_b_panels")
        .join(&run_id)
        .join("panel_timeseries.csv");
    let resolved_panel_log_path = panel_interval.map(|_| {
        inferred_panel_log_path
            .clone()
            .unwrap_or_else(|| fallback_panel_log_path.clone())
    });
    let mut panel_writer = match (panel_interval, resolved_panel_log_path.clone()) {
        (Some(interval), Some(path)) => Some(PanelTimeseriesWriter::new(path, interval)),
        _ => None,
    };
    if d8_state_log_path.is_some() && panel_writer.is_none() {
        panic!("--d8-state-log requires --panel-interval so panel states are well-defined");
    }
    if archive_parent_log_path.is_some() && panel_writer.is_none() {
        panic!(
            "--archive-parent-log requires --panel-interval so parent choices are panel-defined"
        );
    }
    if let Some(panel) = panel_writer.as_ref() {
        println!(
            "  Panel timeseries: {} (interval={} steps)",
            panel.path.display(),
            panel.window_size
        );
    }
    let inferred_operator_policy_log_path = operator_policy_log_path.clone().or_else(|| {
        checkpoint_at_end.as_ref().map(|path| {
            path.parent()
                .unwrap_or_else(|| Path::new("."))
                .join("operator_policy_timeseries.csv")
        })
    });
    let mut operator_policy = OperatorPolicyController::new(
        operator_policy_mode,
        operator_prior_path.as_deref(),
        h,
        operator_epsilon_random,
        operator_weight_floor,
        operator_weight_cap,
        operator_ewma_alpha,
        inferred_operator_policy_log_path.clone(),
    );
    if let Some(writer) = operator_policy.log_writer.as_ref() {
        println!("  Operator policy log: {}", writer.path.display());
    }
    let operator_policy_window = panel_interval.unwrap_or(PROGRESS_INTERVAL);
    let checkpoint_ref = checkpoint_at_end
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let mut d8_state_log = d8_state_log_path.clone().map(|path| {
        D8StateLogWriter::new(path, &run_id, &phase, &arm, seed, h, checkpoint_ref.clone())
    });
    if let Some(writer) = d8_state_log.as_ref() {
        println!(
            "  D8 state log: {} (schema={})",
            writer.path.display(),
            D8StateLogWriter::SCHEMA_VERSION
        );
    }
    let mut archive_parent_log = archive_parent_log_path
        .clone()
        .map(ArchiveParentLogWriter::new);
    if let Some(writer) = archive_parent_log.as_ref() {
        println!(
            "  Archive parent log: {} (policy={}, max_size={}, switch_interval_panels={})",
            writer.path.display(),
            archive_parent_policy_mode.as_str(),
            archive_max_size,
            archive_switch_interval_panels
        );
    }
    let mut archive_entries: VecDeque<ArchiveEntry> = VecDeque::new();
    let mut archive_rng = StdRng::seed_from_u64(seed + 8128);
    let d8_p2_model = archive_p2_model_path
        .as_deref()
        .map(D8P2ExportModel::load);
    let mut d8_next_parent_override: Option<String> = None;
    let mut d8_panel_index = 0usize;

    // Evolve with smooth cosine (champion fitness)
    for step in 0..steps {
        let use_traced = candidate_log.is_some()
            || !matches!(
                acceptance_policy,
                AcceptancePolicy::Strict | AcceptancePolicy::Ties
            )
            || operator_policy_mode.uses_weights()
            || operator_policy.log_writer.is_some();
        let step_operator_weights = operator_policy.weights_for_step();
        let outcome = if use_traced {
            evolution_step_jackpot_traced_with_policy_and_operator_weights(
                &mut net,
                &mut proj,
                &mut rng,
                &mut eval_rng,
                |n, p, er| {
                    let cos = eval_smooth_proj(
                        n,
                        p,
                        &table,
                        &pair_ids,
                        &hot_to_idx,
                        &bigram,
                        DEFAULT_EVAL_LEN,
                        er,
                        &init.propagation,
                        init.output_start(),
                        h,
                        init.input_end(),
                        input_scatter,
                    );
                    // Anti-monopoly: reward more alive output neurons
                    let charges = n.charge_vec(init.output_start()..h);
                    let alive = charges.iter().filter(|&&c| c > 0).count();
                    let alive_frac = alive as f64 / (h - init.output_start()) as f64;
                    let diversity = if diversity_guard_lambda > 0.0 {
                        quick_output_diversity_score(
                            n,
                            p,
                            &table,
                            &pair_ids,
                            &init.propagation,
                            init.output_start(),
                            h,
                            init.input_end(),
                            input_scatter,
                        )
                    } else {
                        0.0
                    };
                    cos * (1.0 + 0.1 * alive_frac) + diversity_guard_lambda * diversity
                },
                &evo_config,
                acceptance_policy,
                jackpot,
                step,
                step_operator_weights.as_deref(),
                |record| {
                    if let Some(log) = candidate_log.as_mut() {
                        log.write_record(record);
                    }
                    operator_policy.observe(record);
                },
            )
        } else {
            evolution_step_jackpot(
                &mut net,
                &mut proj,
                &mut rng,
                &mut eval_rng,
                |n, p, er| {
                    let cos = eval_smooth_proj(
                        n,
                        p,
                        &table,
                        &pair_ids,
                        &hot_to_idx,
                        &bigram,
                        DEFAULT_EVAL_LEN,
                        er,
                        &init.propagation,
                        init.output_start(),
                        h,
                        init.input_end(),
                        input_scatter,
                    );
                    // Anti-monopoly: reward more alive output neurons
                    let charges = n.charge_vec(init.output_start()..h);
                    let alive = charges.iter().filter(|&&c| c > 0).count();
                    let alive_frac = alive as f64 / (h - init.output_start()) as f64;
                    let diversity = if diversity_guard_lambda > 0.0 {
                        quick_output_diversity_score(
                            n,
                            p,
                            &table,
                            &pair_ids,
                            &init.propagation,
                            init.output_start(),
                            h,
                            init.input_end(),
                            input_scatter,
                        )
                    } else {
                        0.0
                    };
                    cos * (1.0 + 0.1 * alive_frac) + diversity_guard_lambda * diversity
                },
                &evo_config,
                jackpot,
            )
        };
        match outcome {
            StepOutcome::Accepted => accepted += 1,
            StepOutcome::Rejected => rejected += 1,
            StepOutcome::Skipped => {}
        }
        operator_policy.end_step();
        if (step + 1) % PROGRESS_INTERVAL == 0 {
            let acc = eval_accuracy_proj(
                &mut net,
                &proj,
                &table,
                &pair_ids,
                &hot_to_idx,
                DEFAULT_FULL_LEN,
                &mut eval_rng,
                &init.propagation,
                init.output_start(),
                h,
                init.input_end(),
                input_scatter,
            );
            peak = peak.max(acc);
            let rate = accepted as f64 / (accepted + rejected).max(1) as f64 * 100.0;
            println!(
                "  step {:>5}: acc={:.2}%  peak={:.2}%  accept={:.0}%  edges={}",
                step + 1,
                acc * 100.0,
                peak * 100.0,
                rate,
                net.edge_count()
            );
        }
        if let Some(panel) = panel_writer.as_mut() {
            if (step + 1) % panel.window_size == 0 {
                let current_panel_index = d8_panel_index;
                let accepted_window = accepted.saturating_sub(panel.last_accepted);
                let rejected_window = rejected.saturating_sub(panel.last_rejected);
                let metrics = compute_panel_metrics(
                    &mut net,
                    &proj,
                    &table,
                    &pair_ids,
                    &hot_to_idx,
                    &init.propagation,
                    init.output_start(),
                    h,
                    init.input_end(),
                    input_scatter,
                );
                let state_id = format!("{}::{}", run_id, current_panel_index);
                let (psi_pred, archive_cell_id) = d8_p2_model
                    .as_ref()
                    .and_then(|model| {
                        model.score_state(
                            h,
                            jackpot,
                            step + 1,
                            steps,
                            peak,
                            accepted_window,
                            rejected_window,
                            net.edge_count(),
                            &metrics,
                        )
                    })
                    .map(|(psi, cell)| (psi, Some(cell)))
                    .unwrap_or((0.0, None));
                let archive_cell_confidence = d8_p2_model
                    .as_ref()
                    .and_then(|model| {
                        archive_cell_id.map(|cell_id| {
                            let prior_count = archive_entries
                                .iter()
                                .filter(|entry| entry.archive_cell_id == Some(cell_id))
                                .count();
                            model.confidence(h, prior_count + 1)
                        })
                    });
                let parent_override = d8_next_parent_override.take();
                if let Some(log) = d8_state_log.as_mut() {
                    log.write_panel_state(
                        current_panel_index,
                        step + 1,
                        steps,
                        parent_override.as_deref(),
                        accepted,
                        rejected,
                        accepted_window,
                        rejected_window,
                        net.edge_count(),
                        peak,
                        &metrics,
                        archive_cell_id,
                        d8_p2_model.as_ref().map(|_| psi_pred),
                        archive_cell_confidence,
                    );
                }
                panel.write_row(
                    step + 1,
                    accepted,
                    rejected,
                    net.edge_count(),
                    peak,
                    &metrics,
                );
                archive_entries.push_back(ArchiveEntry {
                    state_id: state_id.clone(),
                    family_id: run_id.clone(),
                    archive_cell_id,
                    panel_index: current_panel_index,
                    step: step + 1,
                    current_peak: peak,
                    panel_probe_acc: metrics.panel_probe_acc,
                    psi_pred,
                    net: net.clone(),
                    proj: proj.clone(),
                });
                while archive_entries.len() > archive_max_size {
                    archive_entries.pop_front();
                }

                let mut selected_entry: Option<ArchiveEntry> = None;
                let mut restored_archive_parent = false;
                let mut selected_score = 0.0f64;
                let mut selected_confidence = 0.0f64;
                let should_switch =
                    !matches!(
                        archive_parent_policy_mode,
                        ArchiveParentPolicyMode::CurrentBest
                    ) && ((current_panel_index + 1) % archive_switch_interval_panels == 0);
                if should_switch && archive_entries.len() > 1 {
                    let eligible_len = archive_entries.len() - 1; // Exclude the just-written current state.
                    let selected_idx = match archive_parent_policy_mode {
                        ArchiveParentPolicyMode::CurrentBest => None,
                        ArchiveParentPolicyMode::RandomArchive => {
                            Some(archive_rng.gen_range(0..eligible_len))
                        }
                        ArchiveParentPolicyMode::ScoreArchive => {
                            let mut best_idx = 0usize;
                            for idx in 1..eligible_len {
                                let candidate = archive_entries.get(idx).expect("archive index");
                                let best = archive_entries.get(best_idx).expect("archive index");
                                let ordering =
                                    candidate.panel_probe_acc.total_cmp(&best.panel_probe_acc);
                                if ordering.is_gt()
                                    || (ordering.is_eq()
                                        && candidate.panel_index < best.panel_index)
                                {
                                    best_idx = idx;
                                }
                            }
                            Some(best_idx)
                        }
                        ArchiveParentPolicyMode::P2PsiConf => {
                            let model = d8_p2_model
                                .as_ref()
                                .expect("P2 archive policy requires D8 P2 model");
                            let mut counts = std::collections::BTreeMap::<usize, usize>::new();
                            for entry in archive_entries.iter() {
                                if let Some(cell_id) = entry.archive_cell_id {
                                    *counts.entry(cell_id).or_insert(0) += 1;
                                }
                            }
                            let mut best: Option<(usize, f64, f64)> = None;
                            for idx in 0..eligible_len {
                                let entry = archive_entries.get(idx).expect("archive index");
                                let Some(cell_id) = entry.archive_cell_id else {
                                    continue;
                                };
                                let confidence =
                                    model.confidence(h, *counts.get(&cell_id).unwrap_or(&0));
                                if confidence < archive_min_cell_confidence {
                                    continue;
                                }
                                let score = entry.psi_pred * confidence;
                                match best {
                                    None => best = Some((idx, score, confidence)),
                                    Some((best_idx, best_score, _)) => {
                                        let best_entry =
                                            archive_entries.get(best_idx).expect("archive index");
                                        if score > best_score
                                            || ((score - best_score).abs() <= 1e-12
                                                && entry.panel_index < best_entry.panel_index)
                                        {
                                            best = Some((idx, score, confidence));
                                        }
                                    }
                                }
                            }
                            if let Some((idx, score, confidence)) = best {
                                selected_score = score;
                                selected_confidence = confidence;
                                Some(idx)
                            } else {
                                None
                            }
                        }
                    };
                    if let Some(idx) = selected_idx {
                        let entry = archive_entries.get(idx).expect("archive index").clone();
                        if !matches!(archive_parent_policy_mode, ArchiveParentPolicyMode::P2PsiConf)
                        {
                            selected_score = entry.panel_probe_acc;
                        }
                        net = entry.net.clone();
                        proj = entry.proj.clone();
                        d8_next_parent_override = Some(entry.state_id.clone());
                        restored_archive_parent = true;
                        selected_entry = Some(entry);
                    }
                }
                if let Some(log) = archive_parent_log.as_mut() {
                    log.write_choice(
                        step + 1,
                        current_panel_index,
                        archive_parent_policy_mode,
                        archive_entries.len(),
                        selected_entry.as_ref(),
                        selected_score,
                        selected_confidence,
                        restored_archive_parent,
                    );
                }
                d8_panel_index += 1;
            }
        }
        if (step + 1) % operator_policy_window == 0 {
            operator_policy.write_window(step + 1);
        }
    }

    let final_acc = eval_accuracy_proj(
        &mut net,
        &proj,
        &table,
        &pair_ids,
        &hot_to_idx,
        DEFAULT_FULL_LEN.min(pair_ids.len() / 2),
        &mut eval_rng,
        &init.propagation,
        init.output_start(),
        h,
        init.input_end(),
        input_scatter,
    );
    peak = peak.max(final_acc);
    let final_accept_rate_pct = accepted as f64 / (accepted + rejected).max(1) as f64 * 100.0;
    println!(
        "\n  FINAL: {:.2}%  peak={:.2}%  edges={}  accept={:.0}%",
        final_acc * 100.0,
        peak * 100.0,
        net.edge_count(),
        final_accept_rate_pct
    );

    if let Some(checkpoint_path) = checkpoint_at_end.as_deref() {
        save_checkpoint(
            checkpoint_path,
            &net,
            &proj,
            CheckpointMeta {
                step: steps,
                accuracy: final_acc,
                label: format!("phase_b fixture=mutual_inhibition arm={arm} run_id={run_id}"),
            },
        )
        .expect("failed to save final checkpoint");

        let meta = RunMeta {
            fixture: "mutual_inhibition",
            phase: phase.clone(),
            arm: arm.clone(),
            run_id: run_id.clone(),
            seed,
            seed_list: seed_list.clone(),
            h,
            steps,
            horizon_steps: steps,
            jackpot,
            ticks: init.propagation.ticks_per_token,
            chain_count: init.chain_count,
            accept_ties: init.accept_ties,
            accept_policy: accept_policy_name.clone(),
            neutral_p: matches!(acceptance_policy, AcceptancePolicy::ZeroP { .. })
                .then_some(cli_neutral_p),
            accept_epsilon: matches!(acceptance_policy, AcceptancePolicy::Epsilon { .. })
                .then_some(cli_accept_epsilon),
            input_scatter,
            corpus: corpus_path.to_string(),
            packed: packed_path.to_string(),
            checkpoint: checkpoint_path.display().to_string(),
            candidate_log: candidate_log_path.as_ref().map(|p| p.display().to_string()),
            panel_window_size: panel_interval,
            panel_timeseries: resolved_panel_log_path
                .as_ref()
                .map(|p| p.display().to_string()),
            operator_policy: operator_policy_mode.as_str().to_string(),
            operator_prior: operator_prior_path
                .as_ref()
                .map(|p| p.display().to_string()),
            operator_epsilon_random,
            operator_weight_floor,
            operator_weight_cap,
            operator_ewma_alpha,
            operator_ewma_reward: String::from(
                "per-candidate max(delta_U, 0) from live local candidate outcomes only",
            ),
            operator_policy_log: inferred_operator_policy_log_path
                .as_ref()
                .map(|p| p.display().to_string()),
            instrumentation_schema_version: d8_state_log_path
                .as_ref()
                .map(|_| D8StateLogWriter::SCHEMA_VERSION.to_string()),
            d8_state_log: d8_state_log_path.as_ref().map(|p| p.display().to_string()),
            archive_parent_policy: archive_parent_policy_mode.as_str().to_string(),
            archive_parent_log: archive_parent_log_path
                .as_ref()
                .map(|p| p.display().to_string()),
            archive_max_size,
            archive_switch_interval_panels,
            archive_min_cell_confidence,
            archive_p2_model: archive_p2_model_path
                .as_ref()
                .map(|p| p.display().to_string()),
            embedding_anchored_highways,
            diversity_guard_lambda,
        };
        let meta_json = serde_json::to_string_pretty(&meta).expect("failed to serialize run meta");
        let meta_path = checkpoint_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("run_meta.json");
        write(&meta_path, meta_json).expect("failed to write run_meta.json");
        println!("  Checkpoint: {}", checkpoint_path.display());
        println!("  Run meta:   {}", meta_path.display());
    }

    // Alive-frac mean over 20 deterministically-spaced corpus pairs (for SUMMARY)
    let alive_samples = 20usize.min(pair_ids.len().max(1));
    let mut alive_frac_sum = 0.0f64;
    let out_zone = h - init.output_start();
    for k in 0..alive_samples {
        let pid = pair_ids[(k * pair_ids.len()) / alive_samples];
        net.reset();
        let emb = table.embed_id(pid);
        let mut input_buf = vec![0i32; h];
        quantize_embedding_to_input(&table, emb, &mut input_buf, init.input_end(), input_scatter);
        net.propagate(&input_buf, &init.propagation).unwrap();
        let charges = net.charge_vec(init.output_start()..h);
        let alive = charges.iter().filter(|&&c| c > 0).count();
        alive_frac_sum += alive as f64 / out_zone.max(1) as f64;
    }
    let alive_frac_mean = alive_frac_sum / alive_samples as f64;

    // Quick adversarial check: do different inputs produce different outputs?
    println!("\n  --- ADVERSARIAL DIVERSITY CHECK ---");
    let test_pairs = [
        VcbpTable::pair_id(b't', b'h'),
        VcbpTable::pair_id(b'e', b' '),
        VcbpTable::pair_id(b' ', b't'),
        VcbpTable::pair_id(b'a', b'l'),
    ];
    let mut charges_list: Vec<Vec<u8>> = Vec::new();
    let mut preds: Vec<usize> = Vec::new();
    for &pid in &test_pairs {
        net.reset();
        let emb = table.embed_id(pid);
        let mut input = vec![0i32; h];
        quantize_embedding_to_input(&table, emb, &mut input, init.input_end(), input_scatter);
        net.propagate(&input, &init.propagation).unwrap();
        let charges = net.charge_vec(init.output_start()..h);
        let alive = charges.iter().filter(|&&c| c > 0).count();
        let pred = proj.predict(&charges);
        preds.push(pred);
        charges_list.push(charges);
        let (hi, lo) = VcbpTable::pair_bytes(pid);
        println!(
            "    '{}{}' → pred={}, alive={}/158",
            hi as char, lo as char, pred, alive
        );
    }
    let unique: std::collections::HashSet<usize> = preds.iter().cloned().collect();
    let mut diffs = 0usize;
    let mut pairs = 0usize;
    for i in 0..charges_list.len() {
        for j in (i + 1)..charges_list.len() {
            diffs += charges_list[i]
                .iter()
                .zip(charges_list[j].iter())
                .filter(|(a, b)| a != b)
                .count();
            pairs += 1;
        }
    }
    println!(
        "    Unique predictions: {}/{}",
        unique.len(),
        test_pairs.len()
    );
    println!(
        "    Avg charge diff: {:.1}/158 ({:.1}%)",
        diffs as f64 / pairs as f64,
        diffs as f64 / pairs as f64 / 158.0 * 100.0
    );
    if unique.len() > 1 {
        println!("    ✓ DIVERSE OUTPUT — mutual inhibition creating multi-attractor!");
    } else {
        println!("    ⚠ Still constant — mutual inhibition not enough alone");
    }

    if let Some(log) = candidate_log.as_mut() {
        log.flush();
    }
    if let Some(panel) = panel_writer.as_mut() {
        panel.flush();
    }
    if let Some(log) = d8_state_log.as_mut() {
        log.flush();
    }
    if let Some(log) = archive_parent_log.as_mut() {
        log.flush();
    }
    operator_policy.flush();

    let wall_clock_s = t_start.elapsed().as_secs_f64();
    println!("\nRuntime: {:.1}s", wall_clock_s);

    // Machine-readable summary line for multi-seed drivers.
    println!(
        "SUMMARY {{\"fixture\":\"mutual_inhibition\",\"phase\":\"{}\",\"arm\":\"{}\",\"run_id\":\"{}\",\"seed\":{},\"H\":{},\"phi_dim\":{},\"horizon_steps\":{},\"accept_ties\":{},\"accept_policy\":\"{}\",\"neutral_p\":{:.6},\"accept_epsilon\":{:.12},\"operator_policy\":\"{}\",\"peak_acc\":{:.6},\"final_acc\":{:.6},\"accept_rate_pct\":{:.4},\"alive_frac_mean\":{:.6},\"edges\":{},\"unique_preds\":{},\"wall_clock_s\":{:.3}}}",
        phase, arm, run_id, seed, h, init.phi_dim, steps, init.accept_ties, accept_policy_name, cli_neutral_p, cli_accept_epsilon, operator_policy_mode.as_str(), peak, final_acc, final_accept_rate_pct, alive_frac_mean,
        net.edge_count(), unique.len(), wall_clock_s
    );
}
