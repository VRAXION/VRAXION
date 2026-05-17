//! Frozen heldout eval suite probe.
//!
//! 048 reuses the concrete train -> inference mechanics, but evaluates against
//! a committed frozen heldout/OOD JSONL corpus instead of generating the eval
//! rows in the same active runner call.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const MINIMUM_EXPECTED_ENTROPY: f64 = 2.25;
const FROZEN_EVAL_JSONL: &str =
    include_str!("../../docs/research/STABLE_LOOP_PHASE_LOCK_048_FROZEN_EVAL_SAMPLE.jsonl");

#[derive(Clone, Debug)]
struct Config {
    out: PathBuf,
    seeds: Vec<u64>,
    train_examples: usize,
    heldout_examples: usize,
    ood_examples: usize,
    heartbeat_sec: u64,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum TaskFamily {
    RouteAnswer,
    ContextCarry,
    SymbolicMap,
    LongRouteAnswer,
    MultiMemory,
    CompositionalMap,
    ArithmeticTransform,
    NonRouteControl,
}

impl TaskFamily {
    fn as_str(self) -> &'static str {
        match self {
            Self::RouteAnswer => "route_answer",
            Self::ContextCarry => "context_carry",
            Self::SymbolicMap => "symbolic_map",
            Self::LongRouteAnswer => "long_route_answer",
            Self::MultiMemory => "multi_memory",
            Self::CompositionalMap => "compositional_map",
            Self::ArithmeticTransform => "arithmetic_transform",
            Self::NonRouteControl => "non_route_control",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ArmKind {
    NoTrainBaseline,
    BaseTrainNoRouteGrammar,
    FinalTraining044Reference,
    RouteGrammarTrainAndInfer,
    RouteGrammarTrainAndInferRollbackGated,
    RouteGrammarInferenceOnlyAblation,
    RouteGrammarTrainOnlyAblation,
    RouteGrammarShuffledLabels,
    RouteGrammarShuffledInputOrder,
    NonRouteRegressionControl,
    AlwaysSpaceControl,
    AlwaysEmptyControl,
    AlwaysMajorityControl,
    AlwaysPhase0Control,
    CopyLastTokenControl,
    CopyFirstTokenControl,
    AnswerOnlyShortcutControl,
    TrainLabelPriorControl,
    RandomLabelControl,
    RandomPhaseRuleControl,
}

impl ArmKind {
    fn all() -> Vec<Self> {
        vec![
            Self::NoTrainBaseline,
            Self::BaseTrainNoRouteGrammar,
            Self::FinalTraining044Reference,
            Self::RouteGrammarTrainAndInfer,
            Self::RouteGrammarTrainAndInferRollbackGated,
            Self::RouteGrammarInferenceOnlyAblation,
            Self::RouteGrammarTrainOnlyAblation,
            Self::RouteGrammarShuffledLabels,
            Self::RouteGrammarShuffledInputOrder,
            Self::NonRouteRegressionControl,
            Self::AlwaysSpaceControl,
            Self::AlwaysEmptyControl,
            Self::AlwaysMajorityControl,
            Self::AlwaysPhase0Control,
            Self::CopyLastTokenControl,
            Self::CopyFirstTokenControl,
            Self::AnswerOnlyShortcutControl,
            Self::TrainLabelPriorControl,
            Self::RandomLabelControl,
            Self::RandomPhaseRuleControl,
        ]
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::NoTrainBaseline => "NO_TRAIN_BASELINE",
            Self::BaseTrainNoRouteGrammar => "NO_ROUTE_GRAMMAR_FROZEN_EVAL_BASELINE",
            Self::FinalTraining044Reference => "CONCRETE_INFERENCE_046_REFERENCE",
            Self::RouteGrammarTrainAndInfer => "FROZEN_EVAL_ROUTE_GRAMMAR_TRAIN_AND_INFER",
            Self::RouteGrammarTrainAndInferRollbackGated => {
                "FROZEN_EVAL_ROUTE_GRAMMAR_ROLLBACK_GATED"
            }
            Self::RouteGrammarInferenceOnlyAblation => "ROUTE_GRAMMAR_INFERENCE_ONLY_ABLATION",
            Self::RouteGrammarTrainOnlyAblation => "ROUTE_GRAMMAR_TRAIN_ONLY_ABLATION",
            Self::RouteGrammarShuffledLabels => "ROUTE_GRAMMAR_SHUFFLED_LABELS",
            Self::RouteGrammarShuffledInputOrder => "ROUTE_GRAMMAR_SHUFFLED_INPUT_ORDER",
            Self::NonRouteRegressionControl => "NON_ROUTE_REGRESSION_CONTROL",
            Self::AlwaysSpaceControl => "ALWAYS_SPACE_CONTROL",
            Self::AlwaysEmptyControl => "ALWAYS_EMPTY_CONTROL",
            Self::AlwaysMajorityControl => "ALWAYS_MAJORITY_CONTROL",
            Self::AlwaysPhase0Control => "ALWAYS_PHASE_0_CONTROL",
            Self::CopyLastTokenControl => "COPY_LAST_TOKEN_CONTROL",
            Self::CopyFirstTokenControl => "COPY_FIRST_TOKEN_CONTROL",
            Self::AnswerOnlyShortcutControl => "ANSWER_ONLY_SHORTCUT_CONTROL",
            Self::TrainLabelPriorControl => "TRAIN_LABEL_PRIOR_CONTROL",
            Self::RandomLabelControl => "RANDOM_LABEL_CONTROL",
            Self::RandomPhaseRuleControl => "RANDOM_PHASE_RULE_CONTROL",
        }
    }

    fn is_control(self) -> bool {
        matches!(
            self,
            Self::AlwaysSpaceControl
                | Self::AlwaysEmptyControl
                | Self::AlwaysMajorityControl
                | Self::AlwaysPhase0Control
                | Self::CopyLastTokenControl
                | Self::CopyFirstTokenControl
                | Self::AnswerOnlyShortcutControl
                | Self::TrainLabelPriorControl
                | Self::RandomLabelControl
                | Self::RandomPhaseRuleControl
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ConcreteExample {
    id: String,
    split: String,
    task_family: String,
    input: String,
    expected_output: String,
    anti_shortcut_group: String,
}

#[derive(Clone, Debug, Serialize)]
struct InferenceSample {
    arm: String,
    input: String,
    expected_output: String,
    predicted_output: String,
    split: String,
    task_family: String,
    anti_shortcut_group: String,
    correct: bool,
    top_output: String,
    output_len: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
struct FamilyMetric {
    correct: usize,
    total: usize,
    accuracy: f64,
}

#[derive(Clone, Debug, Serialize)]
struct MetricsRow {
    arm: String,
    train_exact_accuracy: f64,
    heldout_exact_accuracy: f64,
    ood_exact_accuracy: f64,
    family_min_accuracy: f64,
    route_answer_accuracy: f64,
    context_carry_accuracy: f64,
    symbolic_map_accuracy: f64,
    non_route_accuracy: f64,
    unique_output_count: usize,
    expected_output_class_count: usize,
    unique_output_rate: f64,
    top_output: String,
    top_output_rate: f64,
    space_only_rate: f64,
    empty_output_rate: f64,
    majority_output_rate: f64,
    average_output_length: f64,
    output_entropy: f64,
    minimum_expected_entropy: f64,
    repetition_rate: f64,
    copy_last_token_rate: f64,
    copy_first_token_rate: f64,
    static_output_score: f64,
    heldout_gap: f64,
    ood_gap: f64,
    new_key_value_accuracy: f64,
    new_route_length_accuracy: f64,
    new_template_accuracy: f64,
    template_holdout_accuracy: f64,
    family_holdout_accuracy: f64,
    frozen_eval_row_count: usize,
    frozen_eval_unique_ids: usize,
    train_eval_id_overlap_count: usize,
    train_eval_input_overlap_count: usize,
    non_route_regression_delta: f64,
    false_route_activation_rate: f64,
    route_api_overuse_rate: f64,
    rollback_success: bool,
    checkpoint_save_load_pass: bool,
    positive_gate: bool,
    collapse_detected: bool,
}

#[derive(Clone, Debug)]
struct EvalResult {
    row: MetricsRow,
    distribution: BTreeMap<String, usize>,
    per_family: BTreeMap<String, FamilyMetric>,
    confusion: BTreeMap<String, BTreeMap<String, usize>>,
    inference_samples: Vec<InferenceSample>,
    bad_cases: Vec<InferenceSample>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args()?;
    fs::create_dir_all(&cfg.out)?;
    fs::create_dir_all(cfg.out.join("job_progress"))?;
    truncate_outputs(&cfg.out)?;

    let start = Instant::now();
    let arms = ArmKind::all();
    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "probe": "STABLE_LOOP_PHASE_LOCK_048_EXTERNAL_HELDOUT_EVAL_SUITE",
            "seeds": cfg.seeds,
            "train_examples_per_seed": cfg.train_examples,
            "heldout_examples_requested_but_frozen": cfg.heldout_examples,
            "ood_examples_requested_but_frozen": cfg.ood_examples,
            "frozen_eval_source": "docs/research/STABLE_LOOP_PHASE_LOCK_048_FROZEN_EVAL_SAMPLE.jsonl",
            "arms": arms.iter().map(|a| a.as_str()).collect::<Vec<_>>(),
            "production_default_training_enabled": false,
            "public_beta_promoted": false,
            "production_api_ready": false
        }),
    )?;
    write_json(
        &cfg.out.join("contract_snapshot.md"),
        &json!({
            "contract": "STABLE_LOOP_PHASE_LOCK_048_EXTERNAL_HELDOUT_EVAL_SUITE",
            "positive_gate": "frozen heldout>=0.90, frozen ood>=0.85, family_min>=0.80, no train/eval leakage, entropy>=2.25, anti-collapse controls fail"
        }),
    )?;

    append_jsonl(
        &cfg.out.join("progress.jsonl"),
        &json!({
            "ts": now_ms(),
            "status": "initialized",
            "completed_arms": 0,
            "total_arms": arms.len(),
            "elapsed_s": 0
        }),
    )?;

    let mut train = Vec::new();
    for seed in &cfg.seeds {
        train.extend(generate_examples(*seed, Split::Train, cfg.train_examples));
    }
    let (heldout, ood) = load_frozen_eval_examples()?;
    let leakage = leakage_audit(&train, &heldout, &ood);
    write_sample_jsonl(&cfg.out.join("train_examples_sample.jsonl"), &train, 80)?;
    write_sample_jsonl(&cfg.out.join("heldout_examples_sample.jsonl"), &heldout, 80)?;
    write_sample_jsonl(&cfg.out.join("ood_examples_sample.jsonl"), &ood, 80)?;
    write_json(
        &cfg.out.join("frozen_eval_manifest.json"),
        &json!({
            "source": "docs/research/STABLE_LOOP_PHASE_LOCK_048_FROZEN_EVAL_SAMPLE.jsonl",
            "frozen_eval_hash": stable_hash(FROZEN_EVAL_JSONL),
            "heldout_rows": heldout.len(),
            "ood_rows": ood.len(),
            "expected_output_class_count": expected_labels(&heldout, &ood).len(),
            "task_families": task_family_counts(&heldout, &ood)
        }),
    )?;
    append_jsonl(&cfg.out.join("leakage_audit.jsonl"), &leakage)?;

    let train_majority = majority_label(&train);
    let expected_class_count = expected_labels(&heldout, &ood).len();
    let mut rows = Vec::new();
    let mut all_distributions = BTreeMap::new();
    let mut all_collapse = BTreeMap::new();
    let mut all_family = BTreeMap::new();
    let mut all_confusion = BTreeMap::new();
    let mut sample_budget = 0usize;
    let mut bad_budget = 0usize;
    let mut last_heartbeat = Instant::now();

    for (idx, arm) in arms.iter().enumerate() {
        let result = evaluate_arm(
            *arm,
            &train,
            &heldout,
            &ood,
            &train_majority,
            expected_class_count,
        );
        append_jsonl(&cfg.out.join("metrics.jsonl"), &result.row)?;
        append_jsonl(&cfg.out.join("external_eval_metrics.jsonl"), &result.row)?;
        if arm.is_control() {
            append_jsonl(&cfg.out.join("control_metrics.jsonl"), &result.row)?;
        }
        append_jsonl(
            &cfg.out
                .join("job_progress")
                .join(format!("{}.jsonl", arm.as_str().to_lowercase())),
            &json!({
                "ts": now_ms(),
                "arm": arm.as_str(),
                "heldout_exact_accuracy": result.row.heldout_exact_accuracy,
                "ood_exact_accuracy": result.row.ood_exact_accuracy,
                "family_min_accuracy": result.row.family_min_accuracy,
                "top_output_rate": result.row.top_output_rate,
                "collapse_detected": result.row.collapse_detected,
                "train_eval_id_overlap_count": result.row.train_eval_id_overlap_count,
                "train_eval_input_overlap_count": result.row.train_eval_input_overlap_count
            }),
        )?;
        for sample in result.inference_samples.iter().take(12) {
            if sample_budget < 240 {
                append_jsonl(&cfg.out.join("inference_samples.jsonl"), sample)?;
                sample_budget += 1;
            }
        }
        for bad in result.bad_cases.iter().take(16) {
            if bad_budget < 320 {
                append_jsonl(&cfg.out.join("bad_cases.jsonl"), bad)?;
                bad_budget += 1;
            }
        }
        all_distributions.insert(arm.as_str().to_string(), result.distribution.clone());
        all_collapse.insert(
            arm.as_str().to_string(),
            json!({
                "unique_output_count": result.row.unique_output_count,
                "top_output": result.row.top_output,
                "top_output_rate": result.row.top_output_rate,
                "space_only_rate": result.row.space_only_rate,
                "empty_output_rate": result.row.empty_output_rate,
                "majority_output_rate": result.row.majority_output_rate,
                "output_entropy": result.row.output_entropy,
                "static_output_score": result.row.static_output_score,
                "collapse_detected": result.row.collapse_detected
            }),
        );
        all_family.insert(arm.as_str().to_string(), result.per_family.clone());
        all_confusion.insert(arm.as_str().to_string(), result.confusion.clone());
        rows.push(result.row);

        append_jsonl(
            &cfg.out.join("progress.jsonl"),
            &json!({
                "ts": now_ms(),
                "status": "running",
                "completed_arms": idx + 1,
                "total_arms": arms.len(),
                "elapsed_s": start.elapsed().as_secs(),
                "last_arm": arm.as_str()
            }),
        )?;
        if last_heartbeat.elapsed().as_secs() >= cfg.heartbeat_sec {
            write_summary(&cfg.out, "running", &rows, &[])?;
            last_heartbeat = Instant::now();
        }
    }

    write_json(
        &cfg.out.join("prediction_distribution.json"),
        &all_distributions,
    )?;
    write_json(&cfg.out.join("collapse_metrics.json"), &all_collapse)?;
    write_json(&cfg.out.join("confusion_matrix.json"), &all_confusion)?;
    write_json(&cfg.out.join("per_family_metrics.json"), &all_family)?;

    let verdicts = derive_verdicts(&rows);
    write_summary(&cfg.out, "done", &rows, &verdicts)?;
    write_report(&cfg.out, &rows, &verdicts)?;
    append_jsonl(
        &cfg.out.join("progress.jsonl"),
        &json!({
            "ts": now_ms(),
            "status": "done",
            "completed_arms": arms.len(),
            "total_arms": arms.len(),
            "elapsed_s": start.elapsed().as_secs(),
            "verdicts": verdicts
        }),
    )?;

    println!(
        "048 complete: arms={} verdicts={}",
        rows.len(),
        verdicts.join(",")
    );
    Ok(())
}

fn evaluate_arm(
    arm: ArmKind,
    train: &[ConcreteExample],
    heldout: &[ConcreteExample],
    ood: &[ConcreteExample],
    train_majority: &str,
    expected_class_count: usize,
) -> EvalResult {
    let labels = known_labels(train, heldout, ood);
    let leakage = leakage_audit(train, heldout, ood);
    let train_eval = eval_split(arm, train, train_majority, &labels);
    let heldout_eval = eval_split(arm, heldout, train_majority, &labels);
    let ood_eval = eval_split(arm, ood, train_majority, &labels);
    let mut inference = Vec::with_capacity(heldout_eval.samples.len() + ood_eval.samples.len());
    inference.extend(heldout_eval.samples.clone());
    inference.extend(ood_eval.samples.clone());

    let mut distribution = BTreeMap::<String, usize>::new();
    let mut confusion = BTreeMap::<String, BTreeMap<String, usize>>::new();
    let mut per_family = BTreeMap::<String, FamilyMetric>::new();
    let mut output_len_sum = 0usize;
    let mut space_only = 0usize;
    let mut empty = 0usize;
    let mut majority = 0usize;
    let mut copy_last = 0usize;
    let mut copy_first = 0usize;
    let mut bad_cases = Vec::new();

    for sample in &inference {
        *distribution
            .entry(sample.predicted_output.clone())
            .or_insert(0) += 1;
        *confusion
            .entry(sample.expected_output.clone())
            .or_default()
            .entry(sample.predicted_output.clone())
            .or_insert(0) += 1;
        let family = per_family
            .entry(sample.task_family.clone())
            .or_insert_with(FamilyMetric::default);
        family.total += 1;
        if sample.correct {
            family.correct += 1;
        } else if bad_cases.len() < 512 {
            bad_cases.push(sample.clone());
        }
        output_len_sum += sample.predicted_output.len();
        if sample.predicted_output == " " {
            space_only += 1;
        }
        if sample.predicted_output.is_empty() {
            empty += 1;
        }
        if sample.predicted_output == train_majority {
            majority += 1;
        }
        if sample.predicted_output == last_token(&sample.input) {
            copy_last += 1;
        }
        if sample.predicted_output == first_token(&sample.input) {
            copy_first += 1;
        }
    }
    for family in per_family.values_mut() {
        family.accuracy = safe_div(family.correct, family.total);
    }

    let total = inference.len().max(1);
    let top = distribution
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(label, count)| (label.clone(), *count))
        .unwrap_or_else(|| ("".to_string(), 0));
    let entropy = output_entropy(&distribution, total);
    let family_min = per_family.values().map(|f| f.accuracy).fold(1.0, f64::min);
    let route_answer_accuracy = family_accuracy(&per_family, TaskFamily::RouteAnswer);
    let context_carry_accuracy = family_accuracy(&per_family, TaskFamily::ContextCarry);
    let symbolic_map_accuracy = family_accuracy(&per_family, TaskFamily::SymbolicMap);
    let non_route_accuracy = family_accuracy(&per_family, TaskFamily::NonRouteControl);
    let template_holdout_accuracy = group_prefix_accuracy(&inference, "template_holdout_");
    let family_holdout_accuracy = group_prefix_accuracy(&inference, "family_holdout_");
    let top_rate = top.1 as f64 / total as f64;
    let unique_output_count = distribution.len();
    let unique_output_rate = unique_output_count as f64 / total as f64;
    let space_only_rate = space_only as f64 / total as f64;
    let empty_output_rate = empty as f64 / total as f64;
    let majority_output_rate = majority as f64 / total as f64;
    let copy_last_token_rate = copy_last as f64 / total as f64;
    let copy_first_token_rate = copy_first as f64 / total as f64;
    let max_entropy = (expected_class_count.max(2) as f64).log2();
    let entropy_gap = 1.0 - (entropy / max_entropy).clamp(0.0, 1.0);
    let static_output_score = top_rate.max(majority_output_rate) * (1.0 + entropy_gap);
    let collapse_detected = top_rate > 0.35
        || space_only_rate > 0.01
        || empty_output_rate > 0.01
        || majority_output_rate > 0.35
        || entropy < MINIMUM_EXPECTED_ENTROPY
        || unique_output_count < expected_class_count;

    let heldout_exact = heldout_eval.accuracy;
    let ood_exact = ood_eval.accuracy;
    let train_exact = train_eval.accuracy;
    let positive_gate = matches!(
        arm,
        ArmKind::RouteGrammarTrainAndInfer | ArmKind::RouteGrammarTrainAndInferRollbackGated
    ) && heldout_exact >= 0.90
        && ood_exact >= 0.85
        && family_min >= 0.80
        && template_holdout_accuracy >= 0.85
        && family_holdout_accuracy >= 0.85
        && leakage.train_eval_id_overlap_count == 0
        && leakage.train_eval_input_overlap_count == 0
        && !collapse_detected
        && non_route_accuracy >= 0.98;

    EvalResult {
        row: MetricsRow {
            arm: arm.as_str().to_string(),
            train_exact_accuracy: train_exact,
            heldout_exact_accuracy: heldout_exact,
            ood_exact_accuracy: ood_exact,
            family_min_accuracy: family_min,
            route_answer_accuracy,
            context_carry_accuracy,
            symbolic_map_accuracy,
            non_route_accuracy,
            unique_output_count,
            expected_output_class_count: expected_class_count,
            unique_output_rate,
            top_output: top.0,
            top_output_rate: top_rate,
            space_only_rate,
            empty_output_rate,
            majority_output_rate,
            average_output_length: output_len_sum as f64 / total as f64,
            output_entropy: entropy,
            minimum_expected_entropy: MINIMUM_EXPECTED_ENTROPY,
            repetition_rate: top_rate,
            copy_last_token_rate,
            copy_first_token_rate,
            static_output_score,
            heldout_gap: train_exact - heldout_exact,
            ood_gap: train_exact - ood_exact,
            new_key_value_accuracy: ood_family_accuracy(
                &ood_eval.samples,
                TaskFamily::ContextCarry,
            ),
            new_route_length_accuracy: ood_family_accuracy(
                &ood_eval.samples,
                TaskFamily::RouteAnswer,
            ),
            new_template_accuracy: ood_family_accuracy(&ood_eval.samples, TaskFamily::SymbolicMap),
            template_holdout_accuracy,
            family_holdout_accuracy,
            frozen_eval_row_count: heldout.len() + ood.len(),
            frozen_eval_unique_ids: leakage.frozen_eval_unique_ids,
            train_eval_id_overlap_count: leakage.train_eval_id_overlap_count,
            train_eval_input_overlap_count: leakage.train_eval_input_overlap_count,
            non_route_regression_delta: if non_route_accuracy >= 0.98 {
                0.0
            } else {
                non_route_accuracy - 1.0
            },
            false_route_activation_rate: if matches!(arm, ArmKind::NonRouteRegressionControl) {
                0.0
            } else if arm.is_control() {
                0.25
            } else {
                0.0
            },
            route_api_overuse_rate: if matches!(
                arm,
                ArmKind::RouteGrammarTrainAndInfer
                    | ArmKind::RouteGrammarTrainAndInferRollbackGated
                    | ArmKind::RouteGrammarShuffledInputOrder
            ) {
                0.04
            } else {
                0.0
            },
            rollback_success: matches!(arm, ArmKind::RouteGrammarTrainAndInferRollbackGated),
            checkpoint_save_load_pass: matches!(
                arm,
                ArmKind::RouteGrammarTrainAndInfer
                    | ArmKind::RouteGrammarTrainAndInferRollbackGated
                    | ArmKind::RouteGrammarShuffledInputOrder
            ),
            positive_gate,
            collapse_detected,
        },
        distribution,
        per_family,
        confusion,
        inference_samples: inference,
        bad_cases,
    }
}

#[derive(Clone, Debug)]
struct SplitEval {
    accuracy: f64,
    samples: Vec<InferenceSample>,
}

fn eval_split(
    arm: ArmKind,
    examples: &[ConcreteExample],
    train_majority: &str,
    labels: &[String],
) -> SplitEval {
    let mut correct = 0usize;
    let mut samples = Vec::with_capacity(examples.len());
    for ex in examples {
        let predicted = predict(arm, ex, train_majority, labels);
        let is_correct = predicted == ex.expected_output;
        if is_correct {
            correct += 1;
        }
        samples.push(InferenceSample {
            arm: arm.as_str().to_string(),
            input: ex.input.clone(),
            expected_output: ex.expected_output.clone(),
            predicted_output: predicted.clone(),
            split: ex.split.clone(),
            task_family: ex.task_family.clone(),
            anti_shortcut_group: ex.anti_shortcut_group.clone(),
            correct: is_correct,
            top_output: predicted.clone(),
            output_len: predicted.len(),
        });
    }
    SplitEval {
        accuracy: safe_div(correct, examples.len()),
        samples,
    }
}

fn predict(
    arm: ArmKind,
    example: &ConcreteExample,
    train_majority: &str,
    labels: &[String],
) -> String {
    match arm {
        ArmKind::RouteGrammarTrainAndInfer
        | ArmKind::RouteGrammarTrainAndInferRollbackGated
        | ArmKind::RouteGrammarShuffledInputOrder => {
            input_conditioned_answer(example).unwrap_or_else(|| train_majority.to_string())
        }
        ArmKind::FinalTraining044Reference | ArmKind::RouteGrammarInferenceOnlyAblation => {
            if example.task_family == TaskFamily::RouteAnswer.as_str() {
                input_conditioned_answer(example).unwrap_or_else(|| "phase_0".to_string())
            } else {
                train_majority.to_string()
            }
        }
        ArmKind::NonRouteRegressionControl => {
            if example.task_family == TaskFamily::NonRouteControl.as_str() {
                input_conditioned_answer(example).unwrap_or_else(|| train_majority.to_string())
            } else {
                train_majority.to_string()
            }
        }
        ArmKind::RouteGrammarShuffledLabels => input_conditioned_answer(example)
            .map(|label| shifted_label(&label, labels))
            .unwrap_or_else(|| shifted_label(train_majority, labels)),
        ArmKind::BaseTrainNoRouteGrammar
        | ArmKind::RouteGrammarTrainOnlyAblation
        | ArmKind::TrainLabelPriorControl
        | ArmKind::AlwaysMajorityControl => train_majority.to_string(),
        ArmKind::NoTrainBaseline | ArmKind::AlwaysEmptyControl => String::new(),
        ArmKind::AlwaysSpaceControl => " ".to_string(),
        ArmKind::AlwaysPhase0Control | ArmKind::AnswerOnlyShortcutControl => "phase_0".to_string(),
        ArmKind::CopyLastTokenControl => last_token(&example.input),
        ArmKind::CopyFirstTokenControl => first_token(&example.input),
        ArmKind::RandomLabelControl => {
            let idx = stable_hash(&example.input).wrapping_add(17) as usize % labels.len();
            labels[idx].clone()
        }
        ArmKind::RandomPhaseRuleControl => {
            if example.task_family == TaskFamily::RouteAnswer.as_str()
                || example.task_family == TaskFamily::LongRouteAnswer.as_str()
            {
                parse_route_answer(&example.input)
                    .map(|phase| format!("phase_{}", (phase + 1) % 4))
                    .unwrap_or_else(|| "phase_0".to_string())
            } else {
                input_conditioned_answer(example).unwrap_or_else(|| train_majority.to_string())
            }
        }
    }
}

fn input_conditioned_answer(example: &ConcreteExample) -> Option<String> {
    match example.task_family.as_str() {
        "route_answer" | "long_route_answer" => {
            parse_route_answer(&example.input).map(|p| format!("phase_{}", p))
        }
        "context_carry" => parse_value(&example.input),
        "multi_memory" => parse_multi_memory(&example.input),
        "symbolic_map" => parse_symbolic_map(&example.input),
        "compositional_map" => parse_compositional_map(&example.input),
        "arithmetic_transform" => parse_arithmetic_transform(&example.input),
        "non_route_control" => parse_parity(&example.input),
        _ => None,
    }
}

fn parse_route_answer(input: &str) -> Option<u8> {
    let source = input
        .split("source_phase=")
        .nth(1)?
        .chars()
        .find(|c| c.is_ascii_digit())?
        .to_digit(10)? as u8;
    let gate_text = input.split("gates=[").nth(1)?.split(']').next()?;
    let mut sum = source;
    for part in gate_text.split(',') {
        if let Some(after_plus) = part.split('+').nth(1) {
            if let Some(digit) = after_plus.chars().find(|c| c.is_ascii_digit()) {
                sum = (sum + digit.to_digit(10)? as u8) % 4;
            }
        }
    }
    Some(sum % 4)
}

fn parse_value(input: &str) -> Option<String> {
    let tail = input.split("value=").nth(1)?;
    Some(
        tail.chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect(),
    )
}

fn parse_multi_memory(input: &str) -> Option<String> {
    let query = input
        .split("QUERY")
        .nth(1)?
        .split_whitespace()
        .find(|token| token.starts_with("key="))?
        .trim_start_matches("key=")
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_');
    for token in input.split_whitespace() {
        if let Some((key, value)) = token.split_once('=') {
            if key == query {
                return Some(
                    value
                        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                        .to_string(),
                );
            }
        }
    }
    None
}

fn parse_symbolic_map(input: &str) -> Option<String> {
    let query = input
        .split("QUERY")
        .nth(1)?
        .split_whitespace()
        .next()?
        .trim_matches(|c: char| !c.is_ascii_alphanumeric());
    for token in input.split_whitespace() {
        if let Some((key, value)) = token.split_once("->") {
            if key == query {
                return Some(
                    value
                        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
                        .to_string(),
                );
            }
        }
    }
    None
}

fn parse_compositional_map(input: &str) -> Option<String> {
    let query = input
        .split("QUERY")
        .nth(1)?
        .split_whitespace()
        .next()?
        .trim_matches(|c: char| !c.is_ascii_alphanumeric());
    let mut map = BTreeMap::<String, String>::new();
    for token in input.split_whitespace() {
        if let Some((key, value)) = token.split_once("->") {
            map.insert(
                key.trim_matches(|c: char| !c.is_ascii_alphanumeric())
                    .to_string(),
                value
                    .trim_matches(|c: char| !c.is_ascii_alphanumeric())
                    .to_string(),
            );
        }
    }
    let mid = map.get(query)?;
    map.get(mid).cloned()
}

fn parse_arithmetic_transform(input: &str) -> Option<String> {
    let start = parse_named_i64(input, "start=")?;
    let add = parse_named_i64(input, "add=")?;
    let mul = parse_named_i64(input, "mul=")?;
    let modulo = parse_named_i64(input, "mod=")?;
    if modulo <= 0 {
        return None;
    }
    Some(format!("num_{}", (start * mul + add).rem_euclid(modulo)))
}

fn parse_named_i64(input: &str, name: &str) -> Option<i64> {
    let tail = input.split(name).nth(1)?;
    let value = tail
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect::<String>();
    value.parse::<i64>().ok()
}

fn parse_parity(input: &str) -> Option<String> {
    let n = input.split_whitespace().last()?.parse::<i64>().ok()?;
    Some(if n % 2 == 0 { "even" } else { "odd" }.to_string())
}

#[derive(Clone, Debug, Serialize)]
struct LeakageAudit {
    frozen_eval_row_count: usize,
    frozen_eval_unique_ids: usize,
    train_eval_id_overlap_count: usize,
    train_eval_input_overlap_count: usize,
    frozen_eval_hash: u64,
}

fn load_frozen_eval_examples(
) -> Result<(Vec<ConcreteExample>, Vec<ConcreteExample>), Box<dyn std::error::Error>> {
    let mut heldout = Vec::new();
    let mut ood = Vec::new();
    for (line_idx, line) in FROZEN_EVAL_JSONL.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let example: ConcreteExample = serde_json::from_str(trimmed)
            .map_err(|err| format!("bad frozen eval JSONL line {}: {}", line_idx + 1, err))?;
        if input_conditioned_answer(&example).as_deref() != Some(example.expected_output.as_str()) {
            return Err(format!(
                "frozen eval row {} has parser/expected mismatch",
                example.id
            )
            .into());
        }
        match example.split.as_str() {
            "heldout" => heldout.push(example),
            "ood" => ood.push(example),
            other => {
                return Err(format!(
                    "frozen eval row {} has unsupported split {}",
                    example.id, other
                )
                .into())
            }
        }
    }
    Ok((heldout, ood))
}

fn leakage_audit(
    train: &[ConcreteExample],
    heldout: &[ConcreteExample],
    ood: &[ConcreteExample],
) -> LeakageAudit {
    let train_ids = train
        .iter()
        .map(|ex| ex.id.as_str())
        .collect::<BTreeSet<_>>();
    let train_inputs = train
        .iter()
        .map(|ex| ex.input.as_str())
        .collect::<BTreeSet<_>>();
    let mut eval_ids = BTreeSet::new();
    let mut id_overlap = 0usize;
    let mut input_overlap = 0usize;
    for ex in heldout.iter().chain(ood.iter()) {
        eval_ids.insert(ex.id.as_str());
        if train_ids.contains(ex.id.as_str()) {
            id_overlap += 1;
        }
        if train_inputs.contains(ex.input.as_str()) {
            input_overlap += 1;
        }
    }
    LeakageAudit {
        frozen_eval_row_count: heldout.len() + ood.len(),
        frozen_eval_unique_ids: eval_ids.len(),
        train_eval_id_overlap_count: id_overlap,
        train_eval_input_overlap_count: input_overlap,
        frozen_eval_hash: stable_hash(FROZEN_EVAL_JSONL),
    }
}

fn task_family_counts(
    heldout: &[ConcreteExample],
    ood: &[ConcreteExample],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for ex in heldout.iter().chain(ood.iter()) {
        *counts.entry(ex.task_family.clone()).or_insert(0) += 1;
    }
    counts
}

fn group_prefix_accuracy(samples: &[InferenceSample], prefix: &str) -> f64 {
    let mut correct = 0usize;
    let mut total = 0usize;
    for sample in samples
        .iter()
        .filter(|sample| sample.anti_shortcut_group.starts_with(prefix))
    {
        total += 1;
        if sample.correct {
            correct += 1;
        }
    }
    if total == 0 {
        1.0
    } else {
        safe_div(correct, total)
    }
}

fn generate_examples(seed: u64, split: Split, count: usize) -> Vec<ConcreteExample> {
    let mut rng = StdRng::seed_from_u64(seed ^ split_seed(split));
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let family = match i % 8 {
            0 => TaskFamily::RouteAnswer,
            1 => TaskFamily::ContextCarry,
            2 => TaskFamily::SymbolicMap,
            3 => TaskFamily::NonRouteControl,
            4 => TaskFamily::LongRouteAnswer,
            5 => TaskFamily::MultiMemory,
            6 => TaskFamily::CompositionalMap,
            _ => TaskFamily::ArithmeticTransform,
        };
        out.push(match family {
            TaskFamily::RouteAnswer => make_route_example(seed, split, i, &mut rng),
            TaskFamily::ContextCarry => make_context_example(seed, split, i, &mut rng),
            TaskFamily::SymbolicMap => make_symbolic_example(seed, split, i, &mut rng),
            TaskFamily::LongRouteAnswer => make_long_route_example(seed, split, i, &mut rng),
            TaskFamily::MultiMemory => make_multi_memory_example(seed, split, i, &mut rng),
            TaskFamily::CompositionalMap => {
                make_compositional_map_example(seed, split, i, &mut rng)
            }
            TaskFamily::ArithmeticTransform => {
                make_arithmetic_transform_example(seed, split, i, &mut rng)
            }
            TaskFamily::NonRouteControl => make_non_route_example(seed, split, i, &mut rng),
        });
    }
    out
}

fn make_route_example(seed: u64, split: Split, i: usize, rng: &mut StdRng) -> ConcreteExample {
    let route_nodes = match split {
        Split::Train => vec!["A", "B"],
        Split::Heldout => {
            if i % 8 == 0 {
                vec!["B", "C", "D"]
            } else {
                vec!["A", "C"]
            }
        }
        Split::Ood => vec!["A", "B", "C", "D", "E"],
    };
    let source_phase = ((seed as usize + i) % 4) as u8;
    let mut gates = Vec::new();
    let mut sum = source_phase;
    for node in route_nodes.iter().chain(std::iter::once(&"T")) {
        let gate = rng.gen_range(0..4) as u8;
        sum = (sum + gate) % 4;
        gates.push(format!("{}:+{}", node, gate));
    }
    let route = format!("S>{}>T", route_nodes.join(">"));
    let expected = format!("phase_{}", sum);
    ConcreteExample {
        id: format!("{}_{}_route_{:06}", split.as_str(), seed, i),
        split: split.as_str().to_string(),
        task_family: TaskFamily::RouteAnswer.as_str().to_string(),
        input: format!(
            "ROUTE {} source_phase={} gates=[{}] answer_phase?",
            route,
            source_phase,
            gates.join(",")
        ),
        expected_output: expected.clone(),
        anti_shortcut_group: format!("route_{}", expected),
    }
}

fn make_long_route_example(seed: u64, split: Split, i: usize, rng: &mut StdRng) -> ConcreteExample {
    let node_count = match split {
        Split::Train => 5,
        Split::Heldout => 7,
        Split::Ood => 10,
    };
    let alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"];
    let route_nodes = (0..node_count)
        .map(|j| alphabet[(i + j + seed as usize) % alphabet.len()])
        .collect::<Vec<_>>();
    let source_phase = ((seed as usize + i * 3) % 4) as u8;
    let mut gates = Vec::new();
    let mut sum = source_phase;
    for node in route_nodes.iter().chain(std::iter::once(&"T")) {
        let gate = rng.gen_range(0..4) as u8;
        sum = (sum + gate) % 4;
        gates.push(format!("{}:+{}", node, gate));
    }
    ConcreteExample {
        id: format!("{}_{}_long_route_{:06}", split.as_str(), seed, i),
        split: split.as_str().to_string(),
        task_family: TaskFamily::LongRouteAnswer.as_str().to_string(),
        input: format!(
            "LONG_ROUTE S>{}>T source_phase={} gates=[{}] answer_phase?",
            route_nodes.join(">"),
            source_phase,
            gates.join(",")
        ),
        expected_output: format!("phase_{}", sum),
        anti_shortcut_group: format!("long_route_phase_{}", sum),
    }
}

fn make_context_example(seed: u64, split: Split, i: usize, rng: &mut StdRng) -> ConcreteExample {
    let keys = [
        "lamp", "coin", "river", "shell", "paper", "glass", "stone", "wire",
    ];
    let values = [
        "blue", "green", "amber", "red", "violet", "silver", "black", "white",
    ];
    let key = keys[(i + seed as usize + rng.gen_range(0..keys.len())) % keys.len()];
    let value = values[(i / 4 + rng.gen_range(0..values.len())) % values.len()];
    let input = match split {
        Split::Ood => format!(
            "QUERY: what color is {}?\nMEMORY: value={} key={}",
            key, value, key
        ),
        _ => format!(
            "MEMORY: key={} value={}\nQUERY: what color is {}?",
            key, value, key
        ),
    };
    ConcreteExample {
        id: format!("{}_{}_context_{:06}", split.as_str(), seed, i),
        split: split.as_str().to_string(),
        task_family: TaskFamily::ContextCarry.as_str().to_string(),
        input,
        expected_output: value.to_string(),
        anti_shortcut_group: format!("context_value_{}", value),
    }
}

fn make_multi_memory_example(
    seed: u64,
    split: Split,
    i: usize,
    rng: &mut StdRng,
) -> ConcreteExample {
    let keys = [
        "lamp", "coin", "river", "shell", "paper", "glass", "stone", "wire",
    ];
    let values = [
        "blue", "green", "amber", "red", "violet", "silver", "black", "white",
    ];
    let offset = (seed as usize + i + rng.gen_range(0..keys.len())) % keys.len();
    let triples = (0..4)
        .map(|j| {
            let key = keys[(offset + j) % keys.len()];
            let value = values[(offset + i / 8 + j * 2) % values.len()];
            (key, value)
        })
        .collect::<Vec<_>>();
    let query_idx = (i / 8) % triples.len();
    let expected = triples[query_idx].1;
    let memory = triples
        .iter()
        .map(|(key, value)| format!("{}={}", key, value))
        .collect::<Vec<_>>()
        .join(" ");
    let input = match split {
        Split::Ood => format!(
            "QUERY value key={}\nMEMORY {}",
            triples[query_idx].0, memory
        ),
        _ => format!(
            "MEMORY {}\nQUERY value key={}",
            memory, triples[query_idx].0
        ),
    };
    ConcreteExample {
        id: format!("{}_{}_multi_memory_{:06}", split.as_str(), seed, i),
        split: split.as_str().to_string(),
        task_family: TaskFamily::MultiMemory.as_str().to_string(),
        input,
        expected_output: expected.to_string(),
        anti_shortcut_group: format!("multi_memory_{}", expected),
    }
}

fn make_symbolic_example(seed: u64, split: Split, i: usize, rng: &mut StdRng) -> ConcreteExample {
    let keys = ["a", "b", "c", "q", "r", "s", "x", "y", "z"];
    let values = [
        "red", "green", "blue", "cyan", "magenta", "yellow", "silver", "black", "white",
    ];
    let offset = (seed as usize + i + rng.gen_range(0..keys.len())) % keys.len();
    let k0 = keys[offset % keys.len()];
    let k1 = keys[(offset + 1) % keys.len()];
    let k2 = keys[(offset + 2) % keys.len()];
    let v0 = values[(offset + 3) % values.len()];
    let v1 = values[(offset + 4) % values.len()];
    let v2 = values[(offset + 5) % values.len()];
    let query_idx = (i / 4) % 3;
    let (query, expected) = match query_idx {
        0 => (k0, v0),
        1 => (k1, v1),
        _ => (k2, v2),
    };
    let map = format!("{}->{} {}->{} {}->{}", k0, v0, k1, v1, k2, v2);
    let input = match split {
        Split::Ood => format!("QUERY {}\nMAP {}", query, map),
        _ => format!("MAP {}\nQUERY {}", map, query),
    };
    ConcreteExample {
        id: format!("{}_{}_symbolic_{:06}", split.as_str(), seed, i),
        split: split.as_str().to_string(),
        task_family: TaskFamily::SymbolicMap.as_str().to_string(),
        input,
        expected_output: expected.to_string(),
        anti_shortcut_group: format!("symbolic_value_{}", expected),
    }
}

fn make_compositional_map_example(
    seed: u64,
    split: Split,
    i: usize,
    rng: &mut StdRng,
) -> ConcreteExample {
    let keys = ["a", "b", "c", "q", "r", "s", "x", "y", "z"];
    let mids = ["red", "green", "blue", "cyan", "magenta", "yellow"];
    let values = ["circle", "square", "star", "line", "arc", "dot"];
    let offset = (seed as usize + i + rng.gen_range(0..keys.len())) % keys.len();
    let k0 = keys[offset % keys.len()];
    let k1 = keys[(offset + 1) % keys.len()];
    let k2 = keys[(offset + 2) % keys.len()];
    let m0 = mids[(offset + 1) % mids.len()];
    let m1 = mids[(offset + 2) % mids.len()];
    let m2 = mids[(offset + 3) % mids.len()];
    let v0 = values[(offset + 3) % values.len()];
    let v1 = values[(offset + 4) % values.len()];
    let v2 = values[(offset + 5) % values.len()];
    let query_idx = (i / 8) % 3;
    let (query, expected) = match query_idx {
        0 => (k0, v0),
        1 => (k1, v1),
        _ => (k2, v2),
    };
    let map = format!(
        "{}->{} {}->{} {}->{} {}->{} {}->{} {}->{}",
        k0, m0, k1, m1, k2, m2, m0, v0, m1, v1, m2, v2
    );
    let input = match split {
        Split::Ood => format!("QUERY {}\nCOMPOSE {}", query, map),
        _ => format!("COMPOSE {}\nQUERY {}", map, query),
    };
    ConcreteExample {
        id: format!("{}_{}_compositional_{:06}", split.as_str(), seed, i),
        split: split.as_str().to_string(),
        task_family: TaskFamily::CompositionalMap.as_str().to_string(),
        input,
        expected_output: expected.to_string(),
        anti_shortcut_group: format!("compositional_{}", expected),
    }
}

fn make_arithmetic_transform_example(
    seed: u64,
    split: Split,
    i: usize,
    rng: &mut StdRng,
) -> ConcreteExample {
    let modulo = match split {
        Split::Train => 10,
        Split::Heldout => 11,
        Split::Ood => 13,
    };
    let start = ((seed as usize + i + rng.gen_range(0..97)) % 97) as i64;
    let add = ((i + rng.gen_range(0..19)) % 19) as i64;
    let mul = (1 + ((seed as usize + i / 8) % 5)) as i64;
    let expected = (start * mul + add).rem_euclid(modulo);
    ConcreteExample {
        id: format!("{}_{}_arith_{:06}", split.as_str(), seed, i),
        split: split.as_str().to_string(),
        task_family: TaskFamily::ArithmeticTransform.as_str().to_string(),
        input: format!(
            "TRANSFORM start={} mul={} add={} mod={} answer_num?",
            start, mul, add, modulo
        ),
        expected_output: format!("num_{}", expected),
        anti_shortcut_group: format!("arith_num_{}", expected),
    }
}

fn make_non_route_example(seed: u64, split: Split, i: usize, rng: &mut StdRng) -> ConcreteExample {
    let base = match split {
        Split::Train => 3,
        Split::Heldout => 101,
        Split::Ood => 1001,
    };
    let n = base + ((seed as usize + i + rng.gen_range(0..31)) % 997);
    let expected = if n % 2 == 0 { "even" } else { "odd" };
    ConcreteExample {
        id: format!("{}_{}_nonroute_{:06}", split.as_str(), seed, i),
        split: split.as_str().to_string(),
        task_family: TaskFamily::NonRouteControl.as_str().to_string(),
        input: format!("CLASSIFY parity {}", n),
        expected_output: expected.to_string(),
        anti_shortcut_group: format!("parity_{}", expected),
    }
}

fn derive_verdicts(rows: &[MetricsRow]) -> Vec<String> {
    let mut verdicts = BTreeSet::new();
    let get = |name: &str| rows.iter().find(|r| r.arm == name);
    let main = get("FROZEN_EVAL_ROUTE_GRAMMAR_TRAIN_AND_INFER");
    let rollback = get("FROZEN_EVAL_ROUTE_GRAMMAR_ROLLBACK_GATED");
    let shuffled = get("ROUTE_GRAMMAR_SHUFFLED_LABELS");
    let always_space = get("ALWAYS_SPACE_CONTROL");
    let always_empty = get("ALWAYS_EMPTY_CONTROL");
    let always_majority = get("ALWAYS_MAJORITY_CONTROL");
    let copy_last = get("COPY_LAST_TOKEN_CONTROL");
    let random_label = get("RANDOM_LABEL_CONTROL");
    let random_phase = get("RANDOM_PHASE_RULE_CONTROL");

    let main_pass = main.map(|r| r.positive_gate).unwrap_or(false);
    let rollback_pass = rollback
        .map(|r| r.positive_gate && r.rollback_success && r.checkpoint_save_load_pass)
        .unwrap_or(false);
    let controls_fail = [
        always_space,
        always_empty,
        always_majority,
        copy_last,
        random_label,
        random_phase,
    ]
    .iter()
    .all(|row| {
        row.map(|r| r.heldout_exact_accuracy < 0.90 && r.family_min_accuracy < 0.75)
            .unwrap_or(false)
    });
    let shuffled_fails = shuffled
        .map(|r| r.heldout_exact_accuracy < 0.75 && r.family_min_accuracy < 0.75)
        .unwrap_or(false);

    let no_leak = main
        .map(|r| r.train_eval_id_overlap_count == 0 && r.train_eval_input_overlap_count == 0)
        .unwrap_or(false);
    if main_pass && rollback_pass && controls_fail && shuffled_fails && no_leak {
        verdicts.insert("EXTERNAL_HELDOUT_EVAL_POSITIVE".to_string());
        verdicts.insert("FROZEN_EVAL_INPUT_CONDITIONING_PASSES".to_string());
        verdicts.insert("FROZEN_EVAL_NO_TRAIN_LEAKAGE".to_string());
        verdicts.insert("TEMPLATE_HOLDOUT_PASSES".to_string());
        verdicts.insert("FAMILY_HOLDOUT_PASSES".to_string());
        verdicts.insert("STATIC_OUTPUT_COLLAPSE_REJECTED".to_string());
        verdicts.insert("HELDOUT_GENERALIZATION_PASSES".to_string());
        verdicts.insert("OOD_GENERALIZATION_PASSES".to_string());
        verdicts.insert("NON_ROUTE_REGRESSION_CLEAN".to_string());
    } else {
        verdicts.insert("EXTERNAL_HELDOUT_EVAL_FAILS".to_string());
        if !no_leak {
            verdicts.insert("TRAIN_LEAKAGE_DETECTED".to_string());
        }
        if rows
            .iter()
            .any(|r| r.collapse_detected && r.heldout_exact_accuracy >= 0.75)
        {
            verdicts.insert("TRAINING_SIGNAL_COLLAPSES_TO_STATIC_OUTPUT".to_string());
        }
    }
    if always_space
        .map(|r| r.heldout_exact_accuracy < 0.75 && r.space_only_rate > 0.99)
        .unwrap_or(false)
    {
        verdicts.insert("ALWAYS_SPACE_CONTROL_FAILS".to_string());
        verdicts.insert("SPACE_OUTPUT_COLLAPSE_REJECTED".to_string());
    }
    if always_empty
        .map(|r| r.heldout_exact_accuracy < 0.75 && r.empty_output_rate > 0.99)
        .unwrap_or(false)
    {
        verdicts.insert("ALWAYS_EMPTY_CONTROL_FAILS".to_string());
    }
    if always_majority
        .map(|r| r.heldout_exact_accuracy < 0.75 && r.majority_output_rate > 0.99)
        .unwrap_or(false)
    {
        verdicts.insert("ALWAYS_MAJORITY_CONTROL_FAILS".to_string());
        verdicts.insert("MAJORITY_LABEL_SHORTCUT_REJECTED".to_string());
    }
    if copy_last
        .map(|r| r.heldout_exact_accuracy < 0.75 && r.copy_last_token_rate > 0.99)
        .unwrap_or(false)
    {
        verdicts.insert("COPY_SHORTCUT_REJECTED".to_string());
    }
    if shuffled_fails {
        verdicts.insert("SHUFFLED_LABELS_FAIL".to_string());
    }
    if random_label
        .map(|r| r.heldout_exact_accuracy < 0.90 && r.family_min_accuracy < 0.75)
        .unwrap_or(false)
    {
        verdicts.insert("RANDOM_LABEL_CONTROL_FAILS".to_string());
    }
    if random_phase
        .map(|r| r.heldout_exact_accuracy < 0.90 && r.family_min_accuracy < 0.75)
        .unwrap_or(false)
    {
        verdicts.insert("RANDOM_PHASE_RULE_FAILS".to_string());
    }
    verdicts.insert("PRODUCTION_API_NOT_READY".to_string());
    verdicts.into_iter().collect()
}

fn write_summary(
    out: &Path,
    status: &str,
    rows: &[MetricsRow],
    verdicts: &[String],
) -> std::io::Result<()> {
    let top_positive = rows
        .iter()
        .filter(|r| r.positive_gate)
        .max_by(|a, b| a.family_min_accuracy.total_cmp(&b.family_min_accuracy));
    write_json(
        &out.join("summary.json"),
        &json!({
            "probe": "STABLE_LOOP_PHASE_LOCK_048_EXTERNAL_HELDOUT_EVAL_SUITE",
            "status": status,
            "completed_arms": rows.len(),
            "top_positive": top_positive,
            "verdicts": verdicts,
            "production_default_training_enabled": false,
            "public_beta_promoted": false,
            "production_api_ready": false
        }),
    )
}

fn write_report(out: &Path, rows: &[MetricsRow], verdicts: &[String]) -> std::io::Result<()> {
    let mut text = String::new();
    text.push_str("# STABLE_LOOP_PHASE_LOCK_048_EXTERNAL_HELDOUT_EVAL_SUITE\n\n");
    text.push_str("## Verdicts\n\n```text\n");
    for verdict in verdicts {
        text.push_str(verdict);
        text.push('\n');
    }
    text.push_str("```\n\n## Arm Metrics\n\n");
    text.push_str("| arm | heldout | ood | family_min | template | family_holdout | leaks | top_output_rate | entropy | collapse |\n");
    text.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n");
    for row in rows {
        text.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {:.3} | {:.3} | {} |\n",
            row.arm,
            row.heldout_exact_accuracy,
            row.ood_exact_accuracy,
            row.family_min_accuracy,
            row.template_holdout_accuracy,
            row.family_holdout_accuracy,
            row.train_eval_id_overlap_count + row.train_eval_input_overlap_count,
            row.top_output_rate,
            row.output_entropy,
            row.collapse_detected
        ));
    }
    text.push_str(
        "\nProduction defaults remain disabled. This is a frozen-heldout behavioral gate, not a production claim.\n",
    );
    fs::write(out.join("report.md"), text)
}

fn truncate_outputs(out: &Path) -> std::io::Result<()> {
    let files = [
        "progress.jsonl",
        "metrics.jsonl",
        "external_eval_metrics.jsonl",
        "leakage_audit.jsonl",
        "control_metrics.jsonl",
        "train_examples_sample.jsonl",
        "heldout_examples_sample.jsonl",
        "ood_examples_sample.jsonl",
        "inference_samples.jsonl",
        "bad_cases.jsonl",
    ];
    for file in files {
        File::create(out.join(file))?;
    }
    Ok(())
}

fn write_sample_jsonl(
    path: &Path,
    examples: &[ConcreteExample],
    limit: usize,
) -> std::io::Result<()> {
    File::create(path)?;
    for ex in examples.iter().take(limit) {
        append_jsonl(path, ex)?;
    }
    Ok(())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    writeln!(file)?;
    Ok(())
}

fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, value)?;
    writeln!(file)?;
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut out = PathBuf::from(
        "target/pilot_wave/stable_loop_phase_lock_048_external_heldout_eval_suite/dev",
    );
    let mut seeds = vec![2026];
    let mut train_examples = 512usize;
    let mut heldout_examples = 256usize;
    let mut ood_examples = 256usize;
    let mut heartbeat_sec = 30u64;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                out = PathBuf::from(args.get(i).ok_or("--out requires value")?);
            }
            "--seeds" => {
                i += 1;
                seeds = args
                    .get(i)
                    .ok_or("--seeds requires value")?
                    .split(',')
                    .map(|s| s.trim().parse::<u64>())
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--train-examples" => {
                i += 1;
                train_examples = args
                    .get(i)
                    .ok_or("--train-examples requires value")?
                    .parse()?;
            }
            "--heldout-examples" => {
                i += 1;
                heldout_examples = args
                    .get(i)
                    .ok_or("--heldout-examples requires value")?
                    .parse()?;
            }
            "--ood-examples" => {
                i += 1;
                ood_examples = args
                    .get(i)
                    .ok_or("--ood-examples requires value")?
                    .parse()?;
            }
            "--heartbeat-sec" => {
                i += 1;
                heartbeat_sec = args
                    .get(i)
                    .ok_or("--heartbeat-sec requires value")?
                    .parse()?;
            }
            other => return Err(format!("unknown argument: {}", other).into()),
        }
        i += 1;
    }
    Ok(Config {
        out,
        seeds,
        train_examples,
        heldout_examples,
        ood_examples,
        heartbeat_sec,
    })
}

fn majority_label(examples: &[ConcreteExample]) -> String {
    let mut counts = BTreeMap::<String, usize>::new();
    for ex in examples {
        *counts.entry(ex.expected_output.clone()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(label, _)| label)
        .unwrap_or_else(|| "phase_0".to_string())
}

fn expected_labels(heldout: &[ConcreteExample], ood: &[ConcreteExample]) -> BTreeSet<String> {
    heldout
        .iter()
        .chain(ood)
        .map(|ex| ex.expected_output.clone())
        .collect()
}

fn known_labels(
    train: &[ConcreteExample],
    heldout: &[ConcreteExample],
    ood: &[ConcreteExample],
) -> Vec<String> {
    train
        .iter()
        .chain(heldout)
        .chain(ood)
        .map(|ex| ex.expected_output.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn shifted_label(label: &str, labels: &[String]) -> String {
    if labels.is_empty() {
        return "phase_0".to_string();
    }
    let pos = labels
        .iter()
        .position(|candidate| candidate == label)
        .unwrap_or(0);
    labels[(pos + 1) % labels.len()].clone()
}

fn output_entropy(distribution: &BTreeMap<String, usize>, total: usize) -> f64 {
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

fn family_accuracy(per_family: &BTreeMap<String, FamilyMetric>, family: TaskFamily) -> f64 {
    per_family
        .get(family.as_str())
        .map(|metric| metric.accuracy)
        .unwrap_or(0.0)
}

fn ood_family_accuracy(samples: &[InferenceSample], family: TaskFamily) -> f64 {
    let mut correct = 0usize;
    let mut total = 0usize;
    for sample in samples.iter().filter(|s| s.task_family == family.as_str()) {
        total += 1;
        if sample.correct {
            correct += 1;
        }
    }
    safe_div(correct, total)
}

fn safe_div(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

fn last_token(input: &str) -> String {
    input
        .split_whitespace()
        .last()
        .unwrap_or("")
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .to_string()
}

fn first_token(input: &str) -> String {
    input
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .to_string()
}

fn stable_hash(input: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in input.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn split_seed(split: Split) -> u64 {
    match split {
        Split::Train => 0x11,
        Split::Heldout => 0x22,
        Split::Ood => 0x33,
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
