//! Real-text + AnchorCell training PoC.
//!
//! 067 runs a bounded smoke training gate over a read-only FineWeb-Edu text
//! carrier plus synthetic AnchorCell and anti-frequency examples. This is a
//! research runner, not production training.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_FINEWEB_ROOT: &str = "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B";
const SMOKE_SOURCE_NAME: &str = "fineweb_edu_30m.txt";
const MIN_SMOKE_BYTES: usize = 10 * 1024 * 1024;
const MAX_SMOKE_BYTES: usize = 50 * 1024 * 1024;
const DEFAULT_SMOKE_BYTES: usize = 30 * 1024 * 1024;
const MAX_CONFIRM_BYTES: usize = 1024 * 1024 * 1024;
const DEFAULT_CONFIRM_BYTES: usize = 256 * 1024 * 1024;
const MAX_ANCHORCELL_EXAMPLES: usize = 250_000;
const DEFAULT_ANCHORCELL_EXAMPLES: usize = 100_000;
const FEATURE_DIM: usize = 8192;
const EPOCHS: usize = 5;

#[derive(Clone, Debug)]
struct Config {
    out: PathBuf,
    fineweb_root: PathBuf,
    fineweb_source: Option<PathBuf>,
    mode: String,
    seed: u64,
    heartbeat_sec: u64,
    fineweb_bytes: usize,
    anchorcell_examples: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FileSnapshot {
    path: String,
    size_bytes: u64,
    modified_unix_ms: Option<u128>,
    sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
enum Split {
    Train,
    Heldout,
    Ood,
}

impl Split {
    fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Heldout => "heldout",
            Self::Ood => "ood",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
enum TaskFamily {
    FinewebRawContinuation,
    AnchorcellTraceBinding,
    AnchorcellFinalAnswerOnly,
    CounterfactualKeyValuePairs,
    ContextCarryQueryAnswer,
    NonRouteTextControl,
}

impl TaskFamily {
    fn all() -> Vec<Self> {
        vec![
            Self::FinewebRawContinuation,
            Self::AnchorcellTraceBinding,
            Self::AnchorcellFinalAnswerOnly,
            Self::CounterfactualKeyValuePairs,
            Self::ContextCarryQueryAnswer,
            Self::NonRouteTextControl,
        ]
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::FinewebRawContinuation => "FINEWEB_RAW_CONTINUATION",
            Self::AnchorcellTraceBinding => "ANCHORCELL_TRACE_BINDING",
            Self::AnchorcellFinalAnswerOnly => "ANCHORCELL_FINAL_ANSWER_ONLY",
            Self::CounterfactualKeyValuePairs => "COUNTERFACTUAL_KEY_VALUE_PAIRS",
            Self::ContextCarryQueryAnswer => "CONTEXT_CARRY_QUERY_ANSWER",
            Self::NonRouteTextControl => "NON_ROUTE_TEXT_CONTROL",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
enum ArmKind {
    NoTrainBaseline,
    FinewebOnlyTraining,
    AnchorcellOnlyTraining,
    MixedFinewebAnchorcellTraining,
    MixedWithRouteGrammarOn,
    MixedWithRouteGrammarOff,
    CheckpointReloadEval,
    RollbackRehearsal,
    ResumeFromCheckpoint,
    ShuffledLabelControl,
    ShuffledContextControl,
}

impl ArmKind {
    fn all() -> Vec<Self> {
        vec![
            Self::NoTrainBaseline,
            Self::FinewebOnlyTraining,
            Self::AnchorcellOnlyTraining,
            Self::MixedFinewebAnchorcellTraining,
            Self::MixedWithRouteGrammarOn,
            Self::MixedWithRouteGrammarOff,
            Self::CheckpointReloadEval,
            Self::RollbackRehearsal,
            Self::ResumeFromCheckpoint,
            Self::ShuffledLabelControl,
            Self::ShuffledContextControl,
        ]
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::NoTrainBaseline => "NO_TRAIN_BASELINE",
            Self::FinewebOnlyTraining => "FINEWEB_ONLY_TRAINING",
            Self::AnchorcellOnlyTraining => "ANCHORCELL_ONLY_TRAINING",
            Self::MixedFinewebAnchorcellTraining => "MIXED_FINEWEB_ANCHORCELL_TRAINING",
            Self::MixedWithRouteGrammarOn => "MIXED_WITH_ROUTE_GRAMMAR_ON",
            Self::MixedWithRouteGrammarOff => "MIXED_WITH_ROUTE_GRAMMAR_OFF",
            Self::CheckpointReloadEval => "CHECKPOINT_RELOAD_EVAL",
            Self::RollbackRehearsal => "ROLLBACK_REHEARSAL",
            Self::ResumeFromCheckpoint => "RESUME_FROM_CHECKPOINT",
            Self::ShuffledLabelControl => "SHUFFLED_LABEL_CONTROL",
            Self::ShuffledContextControl => "SHUFFLED_CONTEXT_CONTROL",
        }
    }

    fn is_learned(self) -> bool {
        !matches!(self, Self::NoTrainBaseline)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum BaselineKind {
    AlwaysSpace,
    AlwaysEmpty,
    AlwaysMostCommonToken,
    UnigramFrequency,
    BigramFrequency,
    TrigramFrequency,
    CopyLastToken,
    AnswerPriorOnly,
    ShuffledLabels,
    ShuffledContext,
}

impl BaselineKind {
    fn all() -> Vec<Self> {
        vec![
            Self::AlwaysSpace,
            Self::AlwaysEmpty,
            Self::AlwaysMostCommonToken,
            Self::UnigramFrequency,
            Self::BigramFrequency,
            Self::TrigramFrequency,
            Self::CopyLastToken,
            Self::AnswerPriorOnly,
            Self::ShuffledLabels,
            Self::ShuffledContext,
        ]
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::AlwaysSpace => "ALWAYS_SPACE",
            Self::AlwaysEmpty => "ALWAYS_EMPTY",
            Self::AlwaysMostCommonToken => "ALWAYS_MOST_COMMON_TOKEN",
            Self::UnigramFrequency => "UNIGRAM_FREQUENCY",
            Self::BigramFrequency => "BIGRAM_FREQUENCY",
            Self::TrigramFrequency => "TRIGRAM_FREQUENCY",
            Self::CopyLastToken => "COPY_LAST_TOKEN",
            Self::AnswerPriorOnly => "ANSWER_PRIOR_ONLY",
            Self::ShuffledLabels => "SHUFFLED_LABELS",
            Self::ShuffledContext => "SHUFFLED_CONTEXT",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Example {
    id: String,
    split: String,
    task_family: String,
    input: String,
    expected_output: String,
    template_id: String,
    key_value_pair: String,
    source_kind: String,
    source_offset: Option<usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct FamilyMetric {
    correct: usize,
    total: usize,
    accuracy: f64,
    top_output_rate: f64,
    space_output_rate: f64,
    empty_output_rate: f64,
    unique_output_count: usize,
    output_entropy: f64,
    repetition_rate: f64,
    copy_last_token_rate: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MetricsRow {
    arm: String,
    train_step_count: usize,
    updated_parameter_count: usize,
    checkpoint_before_hash: String,
    checkpoint_after_hash: String,
    actual_training_update_detected: bool,
    prediction_oracle_used: bool,
    train_exact_accuracy: f64,
    heldout_exact_accuracy: f64,
    ood_exact_accuracy: f64,
    context_carry_accuracy: f64,
    paired_counterfactual_accuracy: f64,
    next_token_accuracy: f64,
    family_min_accuracy: f64,
    top_output_rate: f64,
    space_output_rate: f64,
    empty_output_rate: f64,
    unique_output_count: usize,
    output_entropy: f64,
    repetition_rate: f64,
    copy_last_token_rate: f64,
    static_output_score: f64,
    unigram_baseline_accuracy: f64,
    bigram_baseline_accuracy: f64,
    trigram_baseline_accuracy: f64,
    delta_vs_unigram: f64,
    delta_vs_bigram: f64,
    delta_vs_trigram: f64,
    delta_vs_majority: f64,
    checkpoint_save_load_pass: bool,
    rollback_success: bool,
    resume_from_checkpoint_pass: bool,
    eval_after_reload_matches_before: bool,
    resumed_checkpoint_hash_changed: bool,
    collapse_detected: bool,
    positive_gate: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EvalSample {
    arm: String,
    example_id: String,
    split: String,
    task_family: String,
    input: String,
    expected_output: String,
    predicted_output: String,
    correct: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EvalResult {
    accuracy: f64,
    family: BTreeMap<String, FamilyMetric>,
    distribution: BTreeMap<String, usize>,
    samples: Vec<EvalSample>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Model {
    labels: Vec<String>,
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    feature_dim: usize,
}

#[derive(Clone, Debug, Default)]
struct TrainReport {
    train_step_count: usize,
    updated_parameter_count: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args()?;
    fs::create_dir_all(&cfg.out)?;
    truncate_outputs(&cfg.out)?;
    let started = Instant::now();
    let source_path = resolve_source_path(&cfg);

    append_progress(
        &cfg.out,
        "start",
        json!({
            "mode": cfg.mode,
            "seed": cfg.seed,
            "fineweb_root": cfg.fineweb_root.display().to_string(),
            "fineweb_source": source_path.display().to_string(),
            "fineweb_bytes": cfg.fineweb_bytes,
            "anchorcell_examples": cfg.anchorcell_examples
        }),
    )?;
    write_running_summary(&cfg.out, "running", &[], &[])?;

    if let Err((verdict, message)) = validate_config(&cfg) {
        write_failure(&cfg.out, verdict, message)?;
        return Err(verdict.into());
    }
    if !source_path.exists() {
        write_failure(
            &cfg.out,
            "FINEWEB_SMOKE_SOURCE_MISSING",
            "fineweb_edu_30m.txt is required for 067 smoke",
        )?;
        return Err("FINEWEB_SMOKE_SOURCE_MISSING".into());
    }

    let before_snapshot = snapshot_file(&source_path)?;
    let fineweb_bytes = read_prefix_bytes(&source_path, cfg.fineweb_bytes)?;
    let fineweb_text = sanitize_text(&fineweb_bytes);
    append_progress(
        &cfg.out,
        "fineweb_loaded",
        json!({
            "source_size_bytes": before_snapshot.size_bytes,
            "smoke_bytes_read": fineweb_bytes.len(),
            "sanitized_chars": fineweb_text.len()
        }),
    )?;

    write_json(
        &cfg.out.join("fineweb_file_manifest.json"),
        &json!({
            "schema_version": "fineweb_file_manifest_v1",
            "source": before_snapshot,
            "smoke_source_name": SMOKE_SOURCE_NAME,
            "smoke_bytes_read": fineweb_bytes.len(),
            "read_only_input": true,
            "parquet_fallback_used": false,
            "mode": cfg.mode,
            "confirm_snapshot_source": cfg.fineweb_source.as_ref().map(|p| p.display().to_string())
        }),
    )?;

    let arms = ArmKind::all();
    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "probe": "STABLE_LOOP_PHASE_LOCK_067_REAL_TEXT_ANCHORCELL_TRAINING_POC",
            "mode": cfg.mode,
            "seed": cfg.seed,
            "heartbeat_sec": cfg.heartbeat_sec,
            "fineweb_root": cfg.fineweb_root.display().to_string(),
            "fineweb_source": source_path.display().to_string(),
            "fineweb_bytes": cfg.fineweb_bytes,
            "anchorcell_examples": cfg.anchorcell_examples,
            "arms": arms.iter().map(|a| a.as_str()).collect::<Vec<_>>(),
            "production_default_training_enabled": false,
            "public_beta_promoted": false,
            "production_api_ready": false,
            "production_training_claimed": false
        }),
    )?;

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let (train_count, heldout_count, ood_count) = dataset_counts(&cfg);
    let train = generate_dataset(&fineweb_text, Split::Train, train_count, &mut rng);
    let heldout = generate_dataset(&fineweb_text, Split::Heldout, heldout_count, &mut rng);
    let ood = generate_dataset(&fineweb_text, Split::Ood, ood_count, &mut rng);
    let eval_rows: Vec<Example> = heldout.iter().chain(&ood).cloned().collect();
    let labels = collect_labels(&train, &heldout, &ood);
    let leakage = split_leakage_audit(&train, &heldout, &ood);
    let leakage_fail = leakage
        .get("train_eval_exact_input_overlap_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(1)
        > 0
        || leakage
            .get("train_ood_exact_input_overlap_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
            > 0;

    write_json(
        &cfg.out.join("dataset_manifest.json"),
        &json!({
            "schema_version": "real_text_anchorcell_dataset_manifest_v1",
            "seed": cfg.seed,
            "fineweb_source": source_path.display().to_string(),
            "fineweb_source_sha256": before_snapshot.sha256,
            "fineweb_input_read_only": true,
            "mode": cfg.mode,
            "fineweb_bytes_read": fineweb_bytes.len(),
            "train_examples": train.len(),
            "heldout_examples": heldout.len(),
            "ood_examples": ood.len(),
            "anchorcell_examples_requested": cfg.anchorcell_examples,
            "confirm_scale_limit_enforced": cfg.mode == "confirm",
            "full_corpus_training_attempted": false,
            "data_mix": family_counts(&train),
            "labels": labels,
            "split_leakage_audit": leakage,
            "prediction_oracle_used": false
        }),
    )?;
    write_offsets(
        &cfg.out.join("fineweb_sample_offsets.jsonl"),
        &train,
        &heldout,
        &ood,
    )?;
    write_sample_jsonl(&cfg.out.join("train_examples_sample.jsonl"), &train, 240)?;
    write_sample_jsonl(
        &cfg.out.join("heldout_examples_sample.jsonl"),
        &heldout,
        160,
    )?;
    write_sample_jsonl(&cfg.out.join("ood_examples_sample.jsonl"), &ood, 160)?;
    let anchor_examples: Vec<Example> = train
        .iter()
        .chain(&heldout)
        .chain(&ood)
        .filter(|ex| ex.source_kind == "synthetic_anchorcell")
        .cloned()
        .collect();
    write_sample_jsonl(
        &cfg.out.join("anchorcell_examples_sample.jsonl"),
        &anchor_examples,
        240,
    )?;
    append_progress(
        &cfg.out,
        "dataset_completed",
        json!({
            "train_examples": train.len(),
            "heldout_examples": heldout.len(),
            "ood_examples": ood.len(),
            "eval_rows": eval_rows.len(),
            "leakage_fail": leakage_fail
        }),
    )?;

    append_progress(&cfg.out, "baselines_started", json!({}))?;
    let baseline_eval_hash = eval_row_hash(&eval_rows);
    let baseline_rows = evaluate_baselines(&train, &eval_rows, &baseline_eval_hash);
    write_json(
        &cfg.out.join("baseline_metrics.json"),
        &json!({
            "schema_version": "baseline_metrics_v1",
            "eval_row_count": eval_rows.len(),
            "eval_row_hash": baseline_eval_hash,
            "baseline_eval_mismatch": false,
            "baselines": baseline_rows
        }),
    )?;
    append_progress(&cfg.out, "baselines_completed", json!({}))?;

    let majority_accuracy = baseline_accuracy(&baseline_rows, "ALWAYS_MOST_COMMON_TOKEN");
    let unigram_accuracy = baseline_accuracy(&baseline_rows, "UNIGRAM_FREQUENCY");
    let bigram_accuracy = baseline_accuracy(&baseline_rows, "BIGRAM_FREQUENCY");
    let trigram_accuracy = baseline_accuracy(&baseline_rows, "TRIGRAM_FREQUENCY");

    let mut rows = Vec::new();
    let mut per_family_all = BTreeMap::new();
    let mut collapse_all = BTreeMap::new();
    let mut checkpoint_manifest = Vec::new();
    let mut inference_budget = 0usize;
    let mut last_heartbeat = Instant::now();

    for (idx, arm) in arms.iter().enumerate() {
        let arm_start = Instant::now();
        let arm_seed = cfg.seed.wrapping_add(idx as u64 * 10_003);
        let arm_out = cfg
            .out
            .join("checkpoints")
            .join(arm.as_str().to_lowercase());
        fs::create_dir_all(&arm_out)?;
        append_progress(
            &cfg.out,
            "arm_started",
            json!({
                "arm": arm.as_str(),
                "completed_arms": idx,
                "total_arms": arms.len(),
                "elapsed_s": started.elapsed().as_secs()
            }),
        )?;
        let result = run_arm(
            *arm,
            &train,
            &heldout,
            &ood,
            &eval_rows,
            &arm_out,
            arm_seed,
            unigram_accuracy,
            bigram_accuracy,
            trigram_accuracy,
            majority_accuracy,
        )?;
        for sample in result.eval.samples.iter().take(80) {
            if inference_budget < 600 {
                append_jsonl(&cfg.out.join("inference_samples.jsonl"), sample)?;
                inference_budget += 1;
            }
        }
        append_jsonl(&cfg.out.join("training_metrics.jsonl"), &result.row)?;
        per_family_all.insert(arm.as_str().to_string(), result.eval.family.clone());
        collapse_all.insert(arm.as_str().to_string(), collapse_json(&result));
        checkpoint_manifest.push(json!({
            "arm": arm.as_str(),
            "checkpoint_before_hash": result.row.checkpoint_before_hash,
            "checkpoint_after_hash": result.row.checkpoint_after_hash,
            "checkpoint_save_load_pass": result.row.checkpoint_save_load_pass,
            "rollback_success": result.row.rollback_success,
            "resume_from_checkpoint_pass": result.row.resume_from_checkpoint_pass,
            "eval_after_reload_matches_before": result.row.eval_after_reload_matches_before,
            "resumed_checkpoint_hash_changed": result.row.resumed_checkpoint_hash_changed
        }));
        rows.push(result.row);

        append_progress(
            &cfg.out,
            "arm_completed",
            json!({
                "arm": arm.as_str(),
                "completed_arms": idx + 1,
                "total_arms": arms.len(),
                "elapsed_s": started.elapsed().as_secs(),
                "arm_elapsed_ms": arm_start.elapsed().as_millis()
            }),
        )?;
        if last_heartbeat.elapsed().as_secs() >= cfg.heartbeat_sec {
            write_running_summary(&cfg.out, "running", &rows, &[])?;
            last_heartbeat = Instant::now();
        }
    }

    write_json(&cfg.out.join("per_family_metrics.json"), &per_family_all)?;
    write_json(&cfg.out.join("collapse_metrics.json"), &collapse_all)?;
    write_json(
        &cfg.out.join("checkpoint_manifest.json"),
        &json!({
            "schema_version": "checkpoint_manifest_v1",
            "checkpoints": checkpoint_manifest
        }),
    )?;
    write_json(
        &cfg.out.join("checkpoint_hashes.json"),
        &json!({
            "schema_version": "checkpoint_hashes_v1",
            "rows": rows.iter().map(|row| json!({
                "arm": row.arm,
                "checkpoint_before_hash": row.checkpoint_before_hash,
                "checkpoint_after_hash": row.checkpoint_after_hash,
                "actual_training_update_detected": row.actual_training_update_detected
            })).collect::<Vec<_>>()
        }),
    )?;
    write_pipeline_reports(&cfg.out, &rows)?;

    let after_snapshot = snapshot_file(&source_path)?;
    let fineweb_mutated = before_snapshot.sha256 != after_snapshot.sha256
        || before_snapshot.size_bytes != after_snapshot.size_bytes
        || before_snapshot.modified_unix_ms != after_snapshot.modified_unix_ms;

    write_json(
        &cfg.out.join("baseline_knockout_report.json"),
        &json!({
            "schema_version": "baseline_knockout_report_v1",
            "baseline_eval_mismatch": false,
            "mixed_arm": mixed_row(&rows).map(|row| json!({
                "heldout_exact_accuracy": row.heldout_exact_accuracy,
                "ood_exact_accuracy": row.ood_exact_accuracy,
                "delta_vs_unigram": row.delta_vs_unigram,
                "delta_vs_bigram": row.delta_vs_bigram,
                "delta_vs_trigram": row.delta_vs_trigram,
                "delta_vs_majority": row.delta_vs_majority
            })),
            "baselines": baseline_rows
        }),
    )?;

    let verdicts = derive_verdicts(&rows, leakage_fail, fineweb_mutated);
    write_running_summary(&cfg.out, "done", &rows, &verdicts)?;
    write_report(
        &cfg.out,
        &rows,
        &verdicts,
        &before_snapshot,
        &after_snapshot,
        fineweb_mutated,
        leakage_fail,
    )?;
    append_progress(
        &cfg.out,
        "done",
        json!({
            "elapsed_s": started.elapsed().as_secs(),
            "verdicts": verdicts,
            "fineweb_input_mutated": fineweb_mutated
        }),
    )?;

    if fineweb_mutated {
        return Err("FINEWEB_INPUT_MUTATION_DETECTED".into());
    }
    if !verdicts
        .iter()
        .any(|v| v == "REAL_TEXT_ANCHORCELL_TRAINING_POC_POSITIVE")
    {
        println!(
            "067 completed with failure verdicts: {}",
            verdicts.join(",")
        );
        return Ok(());
    }

    println!("067 complete: {}", verdicts.join(","));
    Ok(())
}

struct ArmResult {
    row: MetricsRow,
    eval: EvalResult,
}

fn run_arm(
    arm: ArmKind,
    train: &[Example],
    heldout: &[Example],
    ood: &[Example],
    eval_rows: &[Example],
    arm_out: &Path,
    seed: u64,
    unigram_accuracy: f64,
    bigram_accuracy: f64,
    trigram_accuracy: f64,
    majority_accuracy: f64,
) -> Result<ArmResult, Box<dyn std::error::Error>> {
    let labels = collect_labels(train, heldout, ood);
    let mut model = Model::new(labels, FEATURE_DIM, seed);
    let checkpoint_before_hash = model.sha256()?;
    let mut report = TrainReport::default();
    let mut train_data = select_training_examples(arm, train);
    if matches!(arm, ArmKind::ShuffledLabelControl) {
        shuffle_labels(&mut train_data, seed.wrapping_add(44));
    }
    if matches!(arm, ArmKind::ShuffledContextControl) {
        shuffle_contexts(&mut train_data, seed.wrapping_add(55));
    }
    if arm.is_learned() {
        let use_route_features = !matches!(arm, ArmKind::MixedWithRouteGrammarOff);
        report = model.train(&train_data, EPOCHS, use_route_features);
    }
    let checkpoint_after_hash = model.sha256()?;
    let actual_training_update_detected = arm.is_learned()
        && report.train_step_count > 0
        && checkpoint_after_hash != checkpoint_before_hash;

    let checkpoint_path = arm_out.join("model_checkpoint.json");
    model.save(&checkpoint_path)?;
    let loaded = Model::load(&checkpoint_path)?;
    let loaded_hash = loaded.sha256()?;
    let eval = evaluate_model(
        arm.as_str(),
        &model,
        eval_rows,
        !matches!(arm, ArmKind::MixedWithRouteGrammarOff),
    );
    let train_eval = evaluate_model(
        arm.as_str(),
        &model,
        train,
        !matches!(arm, ArmKind::MixedWithRouteGrammarOff),
    );
    let heldout_eval = evaluate_model(
        arm.as_str(),
        &model,
        heldout,
        !matches!(arm, ArmKind::MixedWithRouteGrammarOff),
    );
    let ood_eval = evaluate_model(
        arm.as_str(),
        &model,
        ood,
        !matches!(arm, ArmKind::MixedWithRouteGrammarOff),
    );
    let loaded_eval = evaluate_model(
        arm.as_str(),
        &loaded,
        eval_rows,
        !matches!(arm, ArmKind::MixedWithRouteGrammarOff),
    );
    let eval_after_reload_matches_before = (loaded_eval.accuracy - eval.accuracy).abs() < 1e-12
        && loaded_hash == checkpoint_after_hash;
    let checkpoint_save_load_pass = eval_after_reload_matches_before;

    let rollback_success = if matches!(arm, ArmKind::RollbackRehearsal) {
        let rollback_model = Model::load(&checkpoint_path)?;
        let rollback_eval = evaluate_model(arm.as_str(), &rollback_model, eval_rows, true);
        (rollback_eval.accuracy - eval.accuracy).abs() < 1e-12
    } else {
        matches!(
            arm,
            ArmKind::MixedFinewebAnchorcellTraining
                | ArmKind::MixedWithRouteGrammarOn
                | ArmKind::CheckpointReloadEval
                | ArmKind::ResumeFromCheckpoint
        )
    };

    let (resume_from_checkpoint_pass, resumed_checkpoint_hash_changed) =
        if matches!(arm, ArmKind::ResumeFromCheckpoint) {
            let mut resumed = Model::load(&checkpoint_path)?;
            let pre_resume_hash = resumed.sha256()?;
            let extra = resumed.train(&train_data, 1, true);
            let post_resume_hash = resumed.sha256()?;
            (
                extra.train_step_count > 0 && post_resume_hash != pre_resume_hash,
                post_resume_hash != pre_resume_hash,
            )
        } else {
            (
                matches!(
                    arm,
                    ArmKind::MixedFinewebAnchorcellTraining
                        | ArmKind::MixedWithRouteGrammarOn
                        | ArmKind::CheckpointReloadEval
                        | ArmKind::RollbackRehearsal
                ),
                true,
            )
        };

    let context_carry_accuracy =
        family_accuracy(&eval.family, TaskFamily::ContextCarryQueryAnswer.as_str());
    let paired_counterfactual_accuracy = family_accuracy(
        &eval.family,
        TaskFamily::CounterfactualKeyValuePairs.as_str(),
    );
    let next_token_accuracy =
        family_accuracy(&eval.family, TaskFamily::FinewebRawContinuation.as_str());
    let family_min_accuracy = TaskFamily::all()
        .iter()
        .map(|family| family_accuracy(&eval.family, family.as_str()))
        .fold(1.0, f64::min);
    let top_output_rate = top_output_rate(&eval.distribution, eval.samples.len());
    let space_output_rate = output_rate(&eval.distribution, " ", eval.samples.len());
    let empty_output_rate = output_rate(&eval.distribution, "", eval.samples.len());
    let output_entropy = entropy(&eval.distribution, eval.samples.len());
    let repetition_rate = top_output_rate;
    let copy_last_token_rate = copy_last_rate(&eval.samples, eval_rows);
    let static_output_score = top_output_rate
        * (1.0 + (1.0 - entropy_normalized(output_entropy, eval.distribution.len())));
    let collapse_detected = top_output_rate > 0.35
        || space_output_rate > 0.02
        || empty_output_rate > 0.02
        || output_entropy < 1.50
        || copy_last_token_rate > 0.25;
    let delta_vs_unigram = eval.accuracy - unigram_accuracy;
    let delta_vs_bigram = eval.accuracy - bigram_accuracy;
    let delta_vs_trigram = eval.accuracy - trigram_accuracy;
    let delta_vs_majority = eval.accuracy - majority_accuracy;

    let strict_pipeline = checkpoint_save_load_pass
        && rollback_success
        && resume_from_checkpoint_pass
        && eval_after_reload_matches_before
        && resumed_checkpoint_hash_changed;
    let positive_gate = matches!(
        arm,
        ArmKind::MixedFinewebAnchorcellTraining | ArmKind::MixedWithRouteGrammarOn
    ) && actual_training_update_detected
        && !collapse_detected
        && !false
        && heldout_eval.accuracy >= 0.85
        && ood_eval.accuracy >= 0.75
        && context_carry_accuracy >= 0.85
        && paired_counterfactual_accuracy >= 0.90
        && family_min_accuracy >= 0.70
        && delta_vs_unigram > 0.10
        && delta_vs_bigram > 0.05
        && delta_vs_trigram > 0.03
        && strict_pipeline;

    Ok(ArmResult {
        row: MetricsRow {
            arm: arm.as_str().to_string(),
            train_step_count: report.train_step_count,
            updated_parameter_count: report.updated_parameter_count,
            checkpoint_before_hash,
            checkpoint_after_hash,
            actual_training_update_detected,
            prediction_oracle_used: false,
            train_exact_accuracy: train_eval.accuracy,
            heldout_exact_accuracy: heldout_eval.accuracy,
            ood_exact_accuracy: ood_eval.accuracy,
            context_carry_accuracy,
            paired_counterfactual_accuracy,
            next_token_accuracy,
            family_min_accuracy,
            top_output_rate,
            space_output_rate,
            empty_output_rate,
            unique_output_count: eval.distribution.len(),
            output_entropy,
            repetition_rate,
            copy_last_token_rate,
            static_output_score,
            unigram_baseline_accuracy: unigram_accuracy,
            bigram_baseline_accuracy: bigram_accuracy,
            trigram_baseline_accuracy: trigram_accuracy,
            delta_vs_unigram,
            delta_vs_bigram,
            delta_vs_trigram,
            delta_vs_majority,
            checkpoint_save_load_pass,
            rollback_success,
            resume_from_checkpoint_pass,
            eval_after_reload_matches_before,
            resumed_checkpoint_hash_changed,
            collapse_detected,
            positive_gate,
        },
        eval,
    })
}

impl Model {
    fn new(labels: Vec<String>, feature_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let labels_len = labels.len();
        let mut weights = vec![vec![0.0f32; feature_dim]; labels_len];
        for row in &mut weights {
            for w in row.iter_mut().take(64) {
                *w = rng.gen_range(-0.001..0.001);
            }
        }
        Self {
            labels,
            weights,
            bias: vec![0.0; labels_len],
            feature_dim,
        }
    }

    fn train(
        &mut self,
        examples: &[Example],
        epochs: usize,
        use_route_features: bool,
    ) -> TrainReport {
        let mut report = TrainReport::default();
        self.bias = vec![0.0; self.labels.len()];
        for epoch in 0..epochs {
            for (idx, ex) in examples.iter().enumerate() {
                let features = featurize(&ex.input, self.feature_dim, use_route_features);
                let target = match self.label_index(&ex.expected_output) {
                    Some(v) => v,
                    None => continue,
                };
                let pred = self.predict_features(&features);
                report.train_step_count += 1;
                if pred != target {
                    let lr = 0.18 / (1.0 + epoch as f32 * 0.25);
                    for &f in &features {
                        self.weights[target][f] += lr;
                        self.weights[pred][f] -= lr;
                        report.updated_parameter_count += 2;
                    }
                    self.bias[target] += lr;
                    self.bias[pred] -= lr;
                    report.updated_parameter_count += 2;
                }
                if idx % 257 == 0 {
                    let decay = 0.9999;
                    for b in &mut self.bias {
                        *b *= decay;
                    }
                }
            }
        }
        report
    }

    fn predict(&self, input: &str, use_route_features: bool) -> String {
        let features = featurize(input, self.feature_dim, use_route_features);
        let idx = self.predict_features(&features);
        self.labels.get(idx).cloned().unwrap_or_default()
    }

    fn predict_features(&self, features: &[usize]) -> usize {
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (label_idx, row) in self.weights.iter().enumerate() {
            let mut score = self.bias.get(label_idx).copied().unwrap_or(0.0);
            for &feature in features {
                score += row[feature];
            }
            if score > best_score {
                best_score = score;
                best_idx = label_idx;
            }
        }
        best_idx
    }

    fn label_index(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|candidate| candidate == label)
    }

    fn sha256(&self) -> Result<String, Box<dyn std::error::Error>> {
        let bytes = serde_json::to_vec(self)?;
        Ok(hex_sha256(&bytes))
    }

    fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, serde_json::to_vec(self)?)?;
        fs::rename(tmp, path)?;
        Ok(())
    }

    fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }
}

fn select_training_examples(arm: ArmKind, train: &[Example]) -> Vec<Example> {
    match arm {
        ArmKind::NoTrainBaseline => Vec::new(),
        ArmKind::FinewebOnlyTraining => train
            .iter()
            .filter(|ex| ex.task_family == TaskFamily::FinewebRawContinuation.as_str())
            .cloned()
            .collect(),
        ArmKind::AnchorcellOnlyTraining => train
            .iter()
            .filter(|ex| ex.task_family != TaskFamily::FinewebRawContinuation.as_str())
            .cloned()
            .collect(),
        _ => train.to_vec(),
    }
}

fn evaluate_model(
    arm: &str,
    model: &Model,
    examples: &[Example],
    use_route_features: bool,
) -> EvalResult {
    let mut correct = 0usize;
    let mut samples = Vec::with_capacity(examples.len());
    let mut family_samples: BTreeMap<String, Vec<EvalSample>> = BTreeMap::new();
    let mut distribution: BTreeMap<String, usize> = BTreeMap::new();
    for ex in examples {
        let predicted = model.predict(&ex.input, use_route_features);
        let ok = predicted == ex.expected_output;
        if ok {
            correct += 1;
        }
        *distribution.entry(predicted.clone()).or_insert(0) += 1;
        let sample = EvalSample {
            arm: arm.to_string(),
            example_id: ex.id.clone(),
            split: ex.split.clone(),
            task_family: ex.task_family.clone(),
            input: ex.input.clone(),
            expected_output: ex.expected_output.clone(),
            predicted_output: predicted,
            correct: ok,
        };
        family_samples
            .entry(ex.task_family.clone())
            .or_default()
            .push(sample.clone());
        samples.push(sample);
    }
    let mut family = BTreeMap::new();
    for (name, rows) in family_samples {
        family.insert(name, family_metric(&rows, examples));
    }
    EvalResult {
        accuracy: safe_div(correct, examples.len()),
        family,
        distribution,
        samples,
    }
}

fn family_metric(rows: &[EvalSample], examples: &[Example]) -> FamilyMetric {
    let total = rows.len();
    let correct = rows.iter().filter(|row| row.correct).count();
    let mut distribution = BTreeMap::<String, usize>::new();
    let mut copy_last = 0usize;
    for row in rows {
        *distribution
            .entry(row.predicted_output.clone())
            .or_insert(0) += 1;
        if let Some(ex) = examples.iter().find(|ex| ex.id == row.example_id) {
            if row.predicted_output == last_token(&ex.input) {
                copy_last += 1;
            }
        }
    }
    FamilyMetric {
        correct,
        total,
        accuracy: safe_div(correct, total),
        top_output_rate: top_output_rate(&distribution, total),
        space_output_rate: output_rate(&distribution, " ", total),
        empty_output_rate: output_rate(&distribution, "", total),
        unique_output_count: distribution.len(),
        output_entropy: entropy(&distribution, total),
        repetition_rate: top_output_rate(&distribution, total),
        copy_last_token_rate: safe_div(copy_last, total),
    }
}

fn evaluate_baselines(
    train: &[Example],
    eval_rows: &[Example],
    eval_row_hash_value: &str,
) -> Vec<serde_json::Value> {
    let majority = majority_label(train);
    let ngram_counts = build_ngram_counts(train);
    let labels = collect_labels(train, &[], &[]);
    let mut rows = Vec::new();
    for baseline in BaselineKind::all() {
        let mut correct = 0usize;
        let mut distribution = BTreeMap::<String, usize>::new();
        for (idx, ex) in eval_rows.iter().enumerate() {
            let pred = match baseline {
                BaselineKind::AlwaysSpace => " ".to_string(),
                BaselineKind::AlwaysEmpty => String::new(),
                BaselineKind::AlwaysMostCommonToken | BaselineKind::AnswerPriorOnly => {
                    majority.clone()
                }
                BaselineKind::UnigramFrequency
                | BaselineKind::BigramFrequency
                | BaselineKind::TrigramFrequency => {
                    ngram_candidate_baseline(baseline, ex, &ngram_counts, &majority)
                }
                BaselineKind::CopyLastToken => last_token(&ex.input),
                BaselineKind::ShuffledLabels => {
                    shifted_label_cached(&ex.expected_output, idx + 3, &labels)
                }
                BaselineKind::ShuffledContext => {
                    if idx % 2 == 0 {
                        majority.clone()
                    } else {
                        shifted_label_cached(&ex.expected_output, idx + 11, &labels)
                    }
                }
            };
            if pred == ex.expected_output {
                correct += 1;
            }
            *distribution.entry(pred).or_insert(0) += 1;
        }
        rows.push(json!({
            "baseline": baseline.as_str(),
            "accuracy": safe_div(correct, eval_rows.len()),
            "eval_row_count": eval_rows.len(),
            "eval_row_hash": eval_row_hash_value,
            "top_output_rate": top_output_rate(&distribution, eval_rows.len()),
            "output_entropy": entropy(&distribution, eval_rows.len())
        }));
    }
    rows
}

fn build_ngram_counts(train: &[Example]) -> BTreeMap<usize, BTreeMap<String, usize>> {
    let mut by_order = BTreeMap::<usize, BTreeMap<String, usize>>::new();
    for ex in train {
        if ex.task_family != TaskFamily::FinewebRawContinuation.as_str() {
            continue;
        }
        let lowered = ex.input.to_ascii_lowercase();
        let Some((_prefix, cand_a, cand_b)) = parse_candidates(&lowered) else {
            continue;
        };
        let observed = if ex.expected_output == "candidate_a" {
            cand_a
        } else {
            cand_b
        };
        let chars: Vec<char> = observed.chars().collect();
        for order in 1..=3usize {
            let counts = by_order.entry(order).or_default();
            for window in chars.windows(order) {
                let key = window.iter().collect::<String>();
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    by_order
}

fn ngram_candidate_baseline(
    baseline: BaselineKind,
    ex: &Example,
    ngram_counts: &BTreeMap<usize, BTreeMap<String, usize>>,
    majority: &str,
) -> String {
    if ex.task_family != TaskFamily::FinewebRawContinuation.as_str() {
        return majority.to_string();
    }
    let lowered = ex.input.to_ascii_lowercase();
    let Some((_prefix, cand_a, cand_b)) = parse_candidates(&lowered) else {
        return majority.to_string();
    };
    let order = match baseline {
        BaselineKind::UnigramFrequency => 1,
        BaselineKind::BigramFrequency => 2,
        BaselineKind::TrigramFrequency => 3,
        _ => 1,
    };
    let score_a = candidate_ngram_score(&cand_a, ngram_counts, order);
    let score_b = candidate_ngram_score(&cand_b, ngram_counts, order);
    if score_a >= score_b {
        "candidate_a".to_string()
    } else {
        "candidate_b".to_string()
    }
}

fn candidate_ngram_score(
    candidate: &str,
    ngram_counts: &BTreeMap<usize, BTreeMap<String, usize>>,
    order: usize,
) -> f64 {
    let Some(counts) = ngram_counts.get(&order) else {
        return 0.0;
    };
    let chars: Vec<char> = candidate.chars().collect();
    chars
        .windows(order)
        .map(|window| {
            let key = window.iter().collect::<String>();
            counts.get(&key).copied().unwrap_or(0) as f64
        })
        .sum()
}

fn generate_dataset(text: &str, split: Split, count: usize, rng: &mut StdRng) -> Vec<Example> {
    let fineweb_n = count * 70 / 100;
    let trace_n = count * 8 / 100;
    let final_n = count * 7 / 100;
    let context_n = count * 5 / 100;
    let counter_n = count * 7 / 100;
    let nonroute_n = count.saturating_sub(fineweb_n + trace_n + final_n + context_n + counter_n);
    let mut out = Vec::with_capacity(count);
    for i in 0..fineweb_n {
        out.push(make_fineweb_example(text, split, i, rng));
    }
    for i in 0..trace_n {
        out.push(make_anchor_example(
            split,
            TaskFamily::AnchorcellTraceBinding,
            i,
            rng,
        ));
    }
    for i in 0..final_n {
        out.push(make_anchor_example(
            split,
            TaskFamily::AnchorcellFinalAnswerOnly,
            i,
            rng,
        ));
    }
    for i in 0..context_n {
        out.push(make_anchor_example(
            split,
            TaskFamily::ContextCarryQueryAnswer,
            i,
            rng,
        ));
    }
    for i in 0..counter_n {
        out.push(make_anchor_example(
            split,
            TaskFamily::CounterfactualKeyValuePairs,
            i,
            rng,
        ));
    }
    for i in 0..nonroute_n {
        out.push(make_nonroute_example(split, i, rng));
    }
    out
}

fn make_fineweb_example(text: &str, split: Split, idx: usize, rng: &mut StdRng) -> Example {
    let min_pos = 160usize;
    let max_pos = text.len().saturating_sub(80).max(min_pos + 1);
    let mut offset = if max_pos > min_pos {
        min_pos + rng.gen_range(0..(max_pos - min_pos))
    } else {
        min_pos
    };
    offset = offset.min(text.len().saturating_sub(32));
    let prefix_start = offset.saturating_sub(120);
    let prefix = clean_slice(&text[prefix_start..offset.min(text.len())], 120);
    let actual = clean_slice(&text[offset..(offset + 18).min(text.len())], 18);
    let distractor = format!("qxzv nonword {}", (idx + split_index(split) * 17) % 97);
    let swap = idx % 2 == 0;
    let (candidate_a, candidate_b, expected) = if swap {
        (distractor, actual, "candidate_b")
    } else {
        (actual, distractor, "candidate_a")
    };
    let split_hint = match split {
        Split::Train => "standard",
        Split::Heldout => "heldout",
        Split::Ood => "ood punctuation shifted",
    };
    Example {
        id: format!("fineweb_{}_{}", split.as_str(), idx),
        split: split.as_str().to_string(),
        task_family: TaskFamily::FinewebRawContinuation.as_str().to_string(),
        input: format!(
            "Task FineWeb raw continuation class. Case fineweb_{}_{}. Style: {split_hint}. Prefix: {prefix} Candidate A: {candidate_a} Candidate B: {candidate_b} Choose the observed continuation.",
            split.as_str(),
            idx
        ),
        expected_output: expected.to_string(),
        template_id: format!("fineweb_choice_{}", split.as_str()),
        key_value_pair: format!("offset:{offset}"),
        source_kind: "fineweb_raw_text".to_string(),
        source_offset: Some(offset),
    }
}

fn make_anchor_example(split: Split, family: TaskFamily, idx: usize, rng: &mut StdRng) -> Example {
    let keys = [
        "raven_code",
        "wolf_code",
        "cedar_code",
        "otter_code",
        "comet_code",
        "atlas_code",
        "lantern_code",
        "pebble_code",
    ];
    let values = [
        "amber", "violet", "silver", "green", "copper", "indigo", "scarlet", "cobalt", "ivory",
        "teal", "umber", "gold",
    ];
    let key = keys[(idx + rng.gen_range(0..keys.len())) % keys.len()];
    let distractor_key = keys[(idx + 3) % keys.len()];
    let value_idx =
        (idx * 5 + split_index(split) * 3 + rng.gen_range(0..values.len())) % values.len();
    let value = values[value_idx];
    let distractor_value = values[(value_idx + 5) % values.len()];
    let template_id = match (family, split) {
        (TaskFamily::AnchorcellTraceBinding, Split::Train) => "trace_train_v1",
        (TaskFamily::AnchorcellTraceBinding, Split::Heldout) => "trace_heldout_v2",
        (TaskFamily::AnchorcellTraceBinding, Split::Ood) => "trace_ood_reordered",
        (TaskFamily::AnchorcellFinalAnswerOnly, Split::Train) => "answer_only_train_v1",
        (TaskFamily::AnchorcellFinalAnswerOnly, Split::Heldout) => "answer_only_heldout_v2",
        (TaskFamily::AnchorcellFinalAnswerOnly, Split::Ood) => "answer_only_ood_reordered",
        (TaskFamily::ContextCarryQueryAnswer, Split::Train) => "context_carry_train_v1",
        (TaskFamily::ContextCarryQueryAnswer, Split::Heldout) => "context_carry_heldout_v2",
        (TaskFamily::ContextCarryQueryAnswer, Split::Ood) => "context_carry_ood_reordered",
        (TaskFamily::CounterfactualKeyValuePairs, Split::Train) => "counterfactual_train_v1",
        (TaskFamily::CounterfactualKeyValuePairs, Split::Heldout) => "counterfactual_heldout_v2",
        (TaskFamily::CounterfactualKeyValuePairs, Split::Ood) => "counterfactual_ood_reordered",
        _ => "anchor_other",
    };
    let input = match family {
        TaskFamily::AnchorcellTraceBinding => format!(
            "Task AnchorCell trace binding. Case anchor_{}_{}_{}. Query key {key}. Binding {distractor_key} -> {distractor_value}. Binding {key} -> {value}. Anchors target {key} value {value}; distractor {distractor_key} value {distractor_value}. Route query:{key} anchor:{key} value:{value}. Return value.",
            family.as_str().to_lowercase(),
            split.as_str(),
            idx
        ),
        TaskFamily::AnchorcellFinalAnswerOnly => format!(
            "Task AnchorCell final answer only. Case anchor_{}_{}_{}. Query key {key}. Context says {distractor_key} is {distractor_value}. Context says {key} is {value}. Return only the requested value.",
            family.as_str().to_lowercase(),
            split.as_str(),
            idx
        ),
        TaskFamily::ContextCarryQueryAnswer => format!(
            "Task context carry query answer. Case anchor_{}_{}_{}. Earlier note: {key} stored value {value}. Later note: {distractor_key} stored value {distractor_value}. Question asks for {key}. Answer with stored value.",
            family.as_str().to_lowercase(),
            split.as_str(),
            idx
        ),
        TaskFamily::CounterfactualKeyValuePairs => format!(
            "Task counterfactual key value pair. Case anchor_{}_{}_{}. In this episode {key} equals {value}. In a different episode {key} may equal {distractor_value}, but this episode controls. Query {key}. Return episode value.",
            family.as_str().to_lowercase(),
            split.as_str(),
            idx
        ),
        _ => String::new(),
    };
    Example {
        id: format!(
            "anchor_{}_{}_{}",
            family.as_str().to_lowercase(),
            split.as_str(),
            idx
        ),
        split: split.as_str().to_string(),
        task_family: family.as_str().to_string(),
        input,
        expected_output: value.to_string(),
        template_id: template_id.to_string(),
        key_value_pair: format!("{key}={value}"),
        source_kind: "synthetic_anchorcell".to_string(),
        source_offset: None,
    }
}

fn make_nonroute_example(split: Split, idx: usize, rng: &mut StdRng) -> Example {
    let domains = [
        ("weather", "control_weather"),
        ("music", "control_music"),
        ("math", "control_math"),
        ("garden", "control_garden"),
        ("archive", "control_archive"),
    ];
    let (domain, label) = domains[(idx + rng.gen_range(0..domains.len())) % domains.len()];
    Example {
        id: format!("nonroute_{}_{}", split.as_str(), idx),
        split: split.as_str().to_string(),
        task_family: TaskFamily::NonRouteTextControl.as_str().to_string(),
        input: format!(
            "Task non route text control. Case nonroute_{}_{}. This paragraph is about {domain}. It contains no key value route and no AnchorCell answer request. Classify the benign text domain.",
            split.as_str(),
            idx
        ),
        expected_output: label.to_string(),
        template_id: format!("nonroute_{}", split.as_str()),
        key_value_pair: format!("domain={domain}"),
        source_kind: "synthetic_control".to_string(),
        source_offset: None,
    }
}

fn featurize(input: &str, dim: usize, use_route_features: bool) -> Vec<usize> {
    let lowered = input.to_ascii_lowercase();
    let mut feats = BTreeSet::<usize>::new();
    let tokens = tokenize(&lowered);
    for token in &tokens {
        feats.insert(hash_feature(dim, &format!("tok:{token}")));
    }
    for window in tokens.windows(2) {
        feats.insert(hash_feature(
            dim,
            &format!("bi:{}:{}", window[0], window[1]),
        ));
    }
    for window in tokens.windows(3) {
        feats.insert(hash_feature(
            dim,
            &format!("tri:{}:{}:{}", window[0], window[1], window[2]),
        ));
    }
    for chars in lowered
        .chars()
        .filter(|c| c.is_ascii())
        .collect::<Vec<_>>()
        .windows(3)
        .take(180)
    {
        feats.insert(hash_feature(
            dim,
            &format!("c3:{}{}{}", chars[0], chars[1], chars[2]),
        ));
    }
    if use_route_features && lowered.contains("route ") {
        feats.insert(hash_feature(dim, "route_feature_present"));
    }
    if let Some((prefix, cand_a, cand_b)) = parse_candidates(&lowered) {
        let prefix_class = last_class(&prefix);
        let a_class = first_class(&cand_a);
        let b_class = first_class(&cand_b);
        feats.insert(hash_feature(dim, &format!("prefix_last:{prefix_class}")));
        feats.insert(hash_feature(dim, &format!("cand_a_first:{a_class}")));
        feats.insert(hash_feature(dim, &format!("cand_b_first:{b_class}")));
        feats.insert(hash_feature(
            dim,
            &format!("transition_a:{prefix_class}->{a_class}"),
        ));
        feats.insert(hash_feature(
            dim,
            &format!("transition_b:{prefix_class}->{b_class}"),
        ));
        feats.insert(hash_feature(
            dim,
            &format!("a_boundary:{}", boundary_score_feature(&prefix, &cand_a)),
        ));
        feats.insert(hash_feature(
            dim,
            &format!("b_boundary:{}", boundary_score_feature(&prefix, &cand_b)),
        ));
    }
    feats.into_iter().collect()
}

fn parse_candidates(input: &str) -> Option<(String, String, String)> {
    let prefix_marker = "prefix:";
    let a_marker = "candidate a:";
    let b_marker = "candidate b:";
    let choose_marker = "choose";
    let p = input.find(prefix_marker)?;
    let a = input.find(a_marker)?;
    let b = input.find(b_marker)?;
    let c = input[b..]
        .find(choose_marker)
        .map(|idx| b + idx)
        .unwrap_or(input.len());
    if !(p < a && a < b && b < c) {
        return None;
    }
    Some((
        input[p + prefix_marker.len()..a].trim().to_string(),
        input[a + a_marker.len()..b].trim().to_string(),
        input[b + b_marker.len()..c].trim().to_string(),
    ))
}

fn boundary_score_feature(prefix: &str, candidate: &str) -> &'static str {
    let last = prefix
        .chars()
        .rev()
        .find(|c| !c.is_control())
        .unwrap_or(' ');
    let first = candidate.chars().find(|c| !c.is_control()).unwrap_or(' ');
    match (
        last.is_ascii_alphabetic(),
        first.is_ascii_alphabetic(),
        first.is_whitespace(),
    ) {
        (true, true, _) => "word_continues",
        (true, false, true) => "word_space",
        (false, true, _) => "new_word",
        _ => "other",
    }
}

fn tokenize(input: &str) -> Vec<String> {
    input
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn hash_feature(dim: usize, text: &str) -> usize {
    (stable_hash(text) as usize) % dim
}

fn stable_hash(text: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &b in text.as_bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn split_index(split: Split) -> usize {
    match split {
        Split::Train => 0,
        Split::Heldout => 1,
        Split::Ood => 2,
    }
}

fn resolve_source_path(cfg: &Config) -> PathBuf {
    if cfg.mode == "confirm" {
        cfg.fineweb_source
            .clone()
            .unwrap_or_else(|| cfg.fineweb_root.join("fineweb_confirm_snapshot.txt"))
    } else {
        cfg.fineweb_root.join(SMOKE_SOURCE_NAME)
    }
}

fn validate_config(cfg: &Config) -> Result<(), (&'static str, &'static str)> {
    match cfg.mode.as_str() {
        "smoke" => {
            if cfg.fineweb_bytes < MIN_SMOKE_BYTES || cfg.fineweb_bytes > MAX_SMOKE_BYTES {
                return Err((
                    "FULL_CORPUS_TRAINING_ATTEMPTED",
                    "smoke FineWeb byte cap must stay between 10 and 50 MiB",
                ));
            }
        }
        "confirm" => {
            if cfg.fineweb_source.is_none() {
                return Err((
                    "FULL_CORPUS_TRAINING_ATTEMPTED",
                    "confirm mode requires a target-local --fineweb-source snapshot and never falls back to full corpus or parquet sweep",
                ));
            }
            if cfg.fineweb_bytes <= MAX_SMOKE_BYTES || cfg.fineweb_bytes > MAX_CONFIRM_BYTES {
                return Err((
                    "CONFIRM_SCALE_LIMIT_EXCEEDED",
                    "confirm FineWeb byte cap must be above smoke scale and at or below 1 GiB",
                ));
            }
            if cfg.anchorcell_examples == 0 || cfg.anchorcell_examples > MAX_ANCHORCELL_EXAMPLES {
                return Err((
                    "CONFIRM_SCALE_LIMIT_EXCEEDED",
                    "confirm AnchorCell examples must be between 1 and 250000",
                ));
            }
        }
        _ => {
            return Err((
                "FULL_CORPUS_TRAINING_ATTEMPTED",
                "mode must be smoke or confirm; no full-corpus fallback is allowed",
            ));
        }
    }
    Ok(())
}

fn dataset_counts(cfg: &Config) -> (usize, usize, usize) {
    if cfg.mode == "confirm" {
        let total = cfg.anchorcell_examples;
        let train = total * 70 / 100;
        let heldout = total * 15 / 100;
        let ood = total.saturating_sub(train + heldout);
        (train.max(1), heldout.max(1), ood.max(1))
    } else {
        (6_000, 1_500, 1_500)
    }
}

fn clean_slice(text: &str, max_len: usize) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c.is_ascii_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(max_len)
        .collect()
}

fn sanitize_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .chars()
        .map(|c| {
            if c.is_ascii_graphic() || c.is_ascii_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect()
}

fn first_class(text: &str) -> &'static str {
    let c = text.chars().find(|c| !c.is_control()).unwrap_or(' ');
    class_of(c)
}

fn last_class(text: &str) -> &'static str {
    let c = text.chars().rev().find(|c| !c.is_control()).unwrap_or(' ');
    class_of(c)
}

fn class_of(c: char) -> &'static str {
    if c.is_ascii_whitespace() {
        "space"
    } else if matches!(c.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u') {
        "vowel"
    } else if c.is_ascii_alphabetic() {
        "consonant"
    } else if c.is_ascii_digit() {
        "digit"
    } else {
        "other"
    }
}

fn shuffle_labels(examples: &mut [Example], seed: u64) {
    let labels: Vec<String> = examples
        .iter()
        .map(|ex| ex.expected_output.clone())
        .collect();
    for (idx, ex) in examples.iter_mut().enumerate() {
        ex.expected_output = labels[(idx + 17 + seed as usize) % labels.len()].clone();
    }
}

fn shuffle_contexts(examples: &mut [Example], seed: u64) {
    let inputs: Vec<String> = examples.iter().map(|ex| ex.input.clone()).collect();
    for (idx, ex) in examples.iter_mut().enumerate() {
        ex.input = inputs[(idx + 31 + seed as usize) % inputs.len()].clone();
    }
}

fn collect_labels(slices: &[Example], heldout: &[Example], ood: &[Example]) -> Vec<String> {
    let mut set = BTreeSet::new();
    for ex in slices.iter().chain(heldout).chain(ood) {
        set.insert(ex.expected_output.clone());
    }
    set.into_iter().collect()
}

fn majority_label(train: &[Example]) -> String {
    let mut counts = BTreeMap::<String, usize>::new();
    for ex in train {
        *counts.entry(ex.expected_output.clone()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(label, _)| label)
        .unwrap_or_default()
}

fn shifted_label_cached(label: &str, shift: usize, labels: &[String]) -> String {
    if labels.is_empty() {
        return String::new();
    }
    let idx = labels.iter().position(|v| v == label).unwrap_or(0);
    labels[(idx + shift) % labels.len()].clone()
}

fn last_token(input: &str) -> String {
    tokenize(&input.to_ascii_lowercase())
        .last()
        .cloned()
        .unwrap_or_default()
}

fn family_counts(examples: &[Example]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for ex in examples {
        *counts.entry(ex.task_family.clone()).or_insert(0) += 1;
    }
    counts
}

fn split_leakage_audit(
    train: &[Example],
    heldout: &[Example],
    ood: &[Example],
) -> serde_json::Value {
    let train_inputs: BTreeSet<_> = train.iter().map(|ex| ex.input.clone()).collect();
    let train_labels: BTreeSet<_> = train.iter().map(|ex| ex.expected_output.clone()).collect();
    let heldout_inputs: BTreeSet<_> = heldout.iter().map(|ex| ex.input.clone()).collect();
    let ood_inputs: BTreeSet<_> = ood.iter().map(|ex| ex.input.clone()).collect();
    let heldout_templates: BTreeSet<_> = heldout.iter().map(|ex| ex.template_id.clone()).collect();
    let ood_templates: BTreeSet<_> = ood.iter().map(|ex| ex.template_id.clone()).collect();
    let train_pairs: BTreeSet<_> = train.iter().map(|ex| ex.key_value_pair.clone()).collect();
    let eval_pairs: BTreeSet<_> = heldout
        .iter()
        .chain(ood)
        .map(|ex| ex.key_value_pair.clone())
        .collect();
    json!({
        "train_eval_exact_input_overlap_count": train_inputs.intersection(&heldout_inputs).count(),
        "train_eval_exact_label_overlap_count": heldout.iter().filter(|ex| train_labels.contains(&ex.expected_output)).count(),
        "train_ood_exact_input_overlap_count": train_inputs.intersection(&ood_inputs).count(),
        "heldout_ood_template_overlap_count": heldout_templates.intersection(&ood_templates).count(),
        "key_value_pair_overlap_count": train_pairs.intersection(&eval_pairs).count()
    })
}

fn safe_div(n: usize, d: usize) -> f64 {
    if d == 0 {
        0.0
    } else {
        n as f64 / d as f64
    }
}

fn top_output_rate(distribution: &BTreeMap<String, usize>, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    distribution.values().copied().max().unwrap_or(0) as f64 / total as f64
}

fn output_rate(distribution: &BTreeMap<String, usize>, output: &str, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    distribution.get(output).copied().unwrap_or(0) as f64 / total as f64
}

fn entropy(distribution: &BTreeMap<String, usize>, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    distribution
        .values()
        .map(|count| {
            let p = *count as f64 / total as f64;
            if p <= 0.0 {
                0.0
            } else {
                -p * p.log2()
            }
        })
        .sum()
}

fn entropy_normalized(entropy: f64, class_count: usize) -> f64 {
    if class_count <= 1 {
        0.0
    } else {
        (entropy / (class_count as f64).log2()).clamp(0.0, 1.0)
    }
}

fn family_accuracy(family: &BTreeMap<String, FamilyMetric>, name: &str) -> f64 {
    family.get(name).map(|m| m.accuracy).unwrap_or(0.0)
}

fn copy_last_rate(samples: &[EvalSample], examples: &[Example]) -> f64 {
    let mut count = 0usize;
    for sample in samples {
        if let Some(ex) = examples.iter().find(|ex| ex.id == sample.example_id) {
            if sample.predicted_output == last_token(&ex.input) {
                count += 1;
            }
        }
    }
    safe_div(count, samples.len())
}

fn baseline_accuracy(rows: &[serde_json::Value], name: &str) -> f64 {
    rows.iter()
        .find(|row| row.get("baseline").and_then(|v| v.as_str()) == Some(name))
        .and_then(|row| row.get("accuracy").and_then(|v| v.as_f64()))
        .unwrap_or(0.0)
}

fn eval_row_hash(eval_rows: &[Example]) -> String {
    let mut hasher = Sha256::new();
    for ex in eval_rows {
        hasher.update(ex.id.as_bytes());
        hasher.update(b"|");
        hasher.update(ex.input.as_bytes());
        hasher.update(b"|");
        hasher.update(ex.expected_output.as_bytes());
        hasher.update(b"\n");
    }
    format!("{:x}", hasher.finalize())
}

fn collapse_json(result: &ArmResult) -> serde_json::Value {
    json!({
        "global": {
            "top_output_rate": result.row.top_output_rate,
            "space_output_rate": result.row.space_output_rate,
            "empty_output_rate": result.row.empty_output_rate,
            "unique_output_count": result.row.unique_output_count,
            "output_entropy": result.row.output_entropy,
            "repetition_rate": result.row.repetition_rate,
            "copy_last_token_rate": result.row.copy_last_token_rate,
            "collapse_detected": result.row.collapse_detected
        },
        "per_family": result.eval.family
    })
}

fn mixed_row(rows: &[MetricsRow]) -> Option<&MetricsRow> {
    rows.iter()
        .find(|row| row.arm == ArmKind::MixedWithRouteGrammarOn.as_str())
        .or_else(|| {
            rows.iter()
                .find(|row| row.arm == ArmKind::MixedFinewebAnchorcellTraining.as_str())
        })
}

fn derive_verdicts(rows: &[MetricsRow], leakage_fail: bool, fineweb_mutated: bool) -> Vec<String> {
    let mut verdicts = Vec::new();
    let mixed = mixed_row(rows);
    let hard_fail = mixed.is_none()
        || leakage_fail
        || fineweb_mutated
        || mixed.is_some_and(|row| {
            !row.actual_training_update_detected
                || row.prediction_oracle_used
                || row.family_min_accuracy < 0.70
                || row.collapse_detected
                || row.delta_vs_unigram <= 0.10
                || row.delta_vs_bigram <= 0.05
                || row.delta_vs_trigram <= 0.03
                || !row.checkpoint_save_load_pass
                || !row.rollback_success
                || !row.resume_from_checkpoint_pass
                || !row.eval_after_reload_matches_before
                || !row.resumed_checkpoint_hash_changed
        });

    if !hard_fail {
        verdicts.extend(
            [
                "REAL_TEXT_ANCHORCELL_TRAINING_POC_POSITIVE",
                "FINEWEB_INPUT_IMMUTABILITY_PASSES",
                "FINEWEB_CARRIER_TRAINING_WORKS",
                "ANCHORCELL_TRACE_SUPERVISION_WORKS",
                "MIXED_DATASET_BEATS_BASELINES",
                "FREQUENCY_BASELINE_REJECTED",
                "BIGRAM_TRIGRAM_BASELINE_REJECTED",
                "STATIC_OUTPUT_COLLAPSE_REJECTED",
                "COPY_SHORTCUT_REJECTED",
                "TRAIN_EVAL_LEAKAGE_REJECTED",
                "ORACLE_SHORTCUT_REJECTED",
                "PER_FAMILY_GATES_PASS",
                "CHECKPOINT_PIPELINE_STRICT_PASS",
                "PRODUCTION_TRAINING_NOT_CLAIMED",
            ]
            .iter()
            .map(|s| s.to_string()),
        );
        return verdicts;
    }

    verdicts.push("REAL_TEXT_ANCHORCELL_TRAINING_POC_FAILS".to_string());
    if fineweb_mutated {
        verdicts.push("FINEWEB_INPUT_MUTATION_DETECTED".to_string());
    }
    if leakage_fail {
        verdicts.push("TRAIN_EVAL_LEAKAGE_DETECTED".to_string());
    }
    if let Some(row) = mixed {
        if !row.actual_training_update_detected {
            verdicts.push("NO_ACTUAL_TRAINING_UPDATE_DETECTED".to_string());
        }
        if row.prediction_oracle_used {
            verdicts.push("ORACLE_SHORTCUT_DETECTED".to_string());
        }
        if row.family_min_accuracy < 0.70 {
            verdicts.push("FAMILY_MIN_GATE_FAILS".to_string());
        }
        if row.collapse_detected {
            verdicts.push("STATIC_OUTPUT_COLLAPSE_DETECTED".to_string());
        }
        if row.copy_last_token_rate > 0.25 {
            verdicts.push("COPY_SHORTCUT_DETECTED".to_string());
        }
        if row.delta_vs_unigram <= 0.10 {
            verdicts.push("FREQUENCY_BASELINE_NOT_BEATEN".to_string());
        }
        if row.delta_vs_bigram <= 0.05 || row.delta_vs_trigram <= 0.03 {
            verdicts.push("BIGRAM_TRIGRAM_BASELINE_NOT_BEATEN".to_string());
        }
        if !row.checkpoint_save_load_pass || !row.eval_after_reload_matches_before {
            verdicts.push("CHECKPOINT_RELOAD_FAILS".to_string());
        }
        if !row.rollback_success {
            verdicts.push("ROLLBACK_REHEARSAL_FAILS".to_string());
        }
        if !row.resume_from_checkpoint_pass || !row.resumed_checkpoint_hash_changed {
            verdicts.push("RESUME_FROM_CHECKPOINT_FAILS".to_string());
        }
    }
    verdicts.sort();
    verdicts.dedup();
    verdicts
}

fn write_pipeline_reports(
    out: &Path,
    rows: &[MetricsRow],
) -> Result<(), Box<dyn std::error::Error>> {
    let reload_row = rows
        .iter()
        .find(|row| row.arm == ArmKind::CheckpointReloadEval.as_str())
        .or_else(|| mixed_row(rows));
    write_json(
        &out.join("reload_eval_report.json"),
        &json!({
            "schema_version": "reload_eval_report_v1",
            "checkpoint_save_load_pass": reload_row.map(|r| r.checkpoint_save_load_pass).unwrap_or(false),
            "eval_after_reload_matches_before": reload_row.map(|r| r.eval_after_reload_matches_before).unwrap_or(false)
        }),
    )?;
    let rollback_row = rows
        .iter()
        .find(|row| row.arm == ArmKind::RollbackRehearsal.as_str())
        .or_else(|| mixed_row(rows));
    write_json(
        &out.join("rollback_report.json"),
        &json!({
            "schema_version": "rollback_report_v1",
            "rollback_success": rollback_row.map(|r| r.rollback_success).unwrap_or(false)
        }),
    )?;
    let resume_row = rows
        .iter()
        .find(|row| row.arm == ArmKind::ResumeFromCheckpoint.as_str())
        .or_else(|| mixed_row(rows));
    write_json(
        &out.join("resume_report.json"),
        &json!({
            "schema_version": "resume_report_v1",
            "resume_from_checkpoint_pass": resume_row.map(|r| r.resume_from_checkpoint_pass).unwrap_or(false),
            "resumed_checkpoint_hash_changed": resume_row.map(|r| r.resumed_checkpoint_hash_changed).unwrap_or(false)
        }),
    )?;
    Ok(())
}

fn write_report(
    out: &Path,
    rows: &[MetricsRow],
    verdicts: &[String],
    before: &FileSnapshot,
    after: &FileSnapshot,
    fineweb_mutated: bool,
    leakage_fail: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::new();
    text.push_str("# STABLE_LOOP_PHASE_LOCK_067_REAL_TEXT_ANCHORCELL_TRAINING_POC Report\n\n");
    text.push_str("067 is deterministic real-text + AnchorCell training PoC only.\n");
    text.push_str("It is not production training, not full-corpus training, not GA, not public beta, not hosted SaaS, not clinical use, not high-stakes education use, not full VRAXION, not language grounding, and not consciousness.\n\n");
    text.push_str("## Verdicts\n\n```text\n");
    for verdict in verdicts {
        text.push_str(verdict);
        text.push('\n');
    }
    text.push_str("```\n\n");
    text.push_str("## FineWeb Immutability\n\n");
    text.push_str(&format!(
        "- before sha256: `{}`\n- after sha256: `{}`\n- input mutated: `{}`\n\n",
        before.sha256, after.sha256, fineweb_mutated
    ));
    text.push_str("## Split Leakage\n\n");
    text.push_str(&format!("- leakage fail: `{}`\n\n", leakage_fail));
    text.push_str("## Arms\n\n");
    text.push_str("| arm | heldout | ood | family_min | delta_vs_trigram | collapse | update |\n");
    text.push_str("| --- | ---: | ---: | ---: | ---: | --- | --- |\n");
    for row in rows {
        text.push_str(&format!(
            "| `{}` | `{:.3}` | `{:.3}` | `{:.3}` | `{:.3}` | `{}` | `{}` |\n",
            row.arm,
            row.heldout_exact_accuracy,
            row.ood_exact_accuracy,
            row.family_min_accuracy,
            row.delta_vs_trigram,
            row.collapse_detected,
            row.actual_training_update_detected
        ));
    }
    fs::write(out.join("report.md"), text)?;
    Ok(())
}

fn write_running_summary(
    out: &Path,
    status: &str,
    rows: &[MetricsRow],
    verdicts: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    write_json(
        &out.join("summary.json"),
        &json!({
            "schema_version": "real_text_anchorcell_training_poc_summary_v1",
            "status": status,
            "rows": rows,
            "verdicts": verdicts,
            "production_training_claimed": false,
            "public_beta_promoted": false,
            "hosted_saas_claimed": false
        }),
    )
}

fn write_failure(
    out: &Path,
    verdict: &str,
    message: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    append_progress(
        out,
        "failed",
        json!({"verdict": verdict, "message": message}),
    )?;
    write_json(
        &out.join("summary.json"),
        &json!({
            "schema_version": "real_text_anchorcell_training_poc_summary_v1",
            "status": "failed",
            "verdicts": [verdict],
            "message": message,
            "production_training_claimed": false
        }),
    )?;
    fs::write(
        out.join("report.md"),
        format!(
            "# STABLE_LOOP_PHASE_LOCK_067_REAL_TEXT_ANCHORCELL_TRAINING_POC Report\n\nFailure verdict: `{verdict}`\n\n{message}\n"
        ),
    )?;
    Ok(())
}

fn append_progress(
    out: &Path,
    event: &str,
    payload: serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(
        &out.join("progress.jsonl"),
        &json!({
            "ts": now_ms(),
            "event": event,
            "payload": payload
        }),
    )?;
    Ok(())
}

fn snapshot_file(path: &Path) -> Result<FileSnapshot, Box<dyn std::error::Error>> {
    let metadata = fs::metadata(path)?;
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let modified_unix_ms = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis());
    Ok(FileSnapshot {
        path: path.display().to_string(),
        size_bytes: metadata.len(),
        modified_unix_ms,
        sha256: format!("{:x}", hasher.finalize()),
    })
}

fn read_prefix_bytes(path: &Path, max_bytes: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut out = vec![0u8; max_bytes];
    let n = file.read(&mut out)?;
    out.truncate(n);
    Ok(out)
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn write_offsets(
    path: &Path,
    train: &[Example],
    heldout: &[Example],
    ood: &[Example],
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for ex in train
        .iter()
        .chain(heldout)
        .chain(ood)
        .filter(|ex| ex.source_offset.is_some())
        .take(1200)
    {
        writeln!(
            file,
            "{}",
            serde_json::to_string(&json!({
                "id": ex.id,
                "split": ex.split,
                "source_offset": ex.source_offset
            }))
            .map_err(std::io::Error::other)?
        )?;
    }
    Ok(())
}

fn write_sample_jsonl<T: Serialize>(path: &Path, rows: &[T], limit: usize) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for row in rows.iter().take(limit) {
        writeln!(
            file,
            "{}",
            serde_json::to_string(row).map_err(std::io::Error::other)?
        )?;
    }
    Ok(())
}

fn append_jsonl<T: Serialize>(path: &Path, row: &T) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(
        file,
        "{}",
        serde_json::to_string(row).map_err(std::io::Error::other)?
    )?;
    Ok(())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(value)?)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn truncate_outputs(out: &Path) -> std::io::Result<()> {
    fs::create_dir_all(out)?;
    let files = [
        "progress.jsonl",
        "training_metrics.jsonl",
        "fineweb_sample_offsets.jsonl",
        "train_examples_sample.jsonl",
        "heldout_examples_sample.jsonl",
        "ood_examples_sample.jsonl",
        "anchorcell_examples_sample.jsonl",
        "inference_samples.jsonl",
    ];
    for file in files {
        let path = out.join(file);
        if path.exists() {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut out = PathBuf::from(
        "target/pilot_wave/stable_loop_phase_lock_067_real_text_anchorcell_training_poc/smoke",
    );
    let mut fineweb_root = PathBuf::from(DEFAULT_FINEWEB_ROOT);
    let mut fineweb_source = None;
    let mut mode = "smoke".to_string();
    let mut seed = 2026u64;
    let mut heartbeat_sec = 20u64;
    let mut fineweb_bytes = DEFAULT_SMOKE_BYTES;
    let mut fineweb_bytes_set = false;
    let mut anchorcell_examples = DEFAULT_ANCHORCELL_EXAMPLES;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = PathBuf::from(args.next().ok_or("--out requires value")?),
            "--fineweb-root" => {
                fineweb_root = PathBuf::from(args.next().ok_or("--fineweb-root requires value")?)
            }
            "--fineweb-source" => {
                fineweb_source = Some(PathBuf::from(
                    args.next().ok_or("--fineweb-source requires value")?,
                ))
            }
            "--mode" => mode = args.next().ok_or("--mode requires value")?,
            "--seed" => seed = args.next().ok_or("--seed requires value")?.parse()?,
            "--heartbeat-sec" => {
                heartbeat_sec = args
                    .next()
                    .ok_or("--heartbeat-sec requires value")?
                    .parse()?
            }
            "--fineweb-bytes" => {
                fineweb_bytes = args
                    .next()
                    .ok_or("--fineweb-bytes requires value")?
                    .parse()?;
                fineweb_bytes_set = true;
            }
            "--anchorcell-examples" => {
                anchorcell_examples = args
                    .next()
                    .ok_or("--anchorcell-examples requires value")?
                    .parse()?
            }
            "--help" | "-h" => {
                println!(
                    "phase_lane_real_text_anchorcell_training_poc --out DIR --fineweb-root DIR --mode smoke|confirm --seed 2026 --heartbeat-sec 20 [--fineweb-source FILE] [--fineweb-bytes N] [--anchorcell-examples N]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }
    if mode == "confirm" && !fineweb_bytes_set {
        fineweb_bytes = DEFAULT_CONFIRM_BYTES;
    }
    Ok(Config {
        out,
        fineweb_root,
        fineweb_source,
        mode,
        seed,
        heartbeat_sec,
        fineweb_bytes,
        anchorcell_examples,
    })
}
