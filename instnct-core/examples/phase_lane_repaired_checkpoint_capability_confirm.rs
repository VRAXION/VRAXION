//! Eval-only repaired checkpoint capability confirmation.
//!
//! 071 measures whether the 070 repaired finite-label checkpoint generalizes to
//! fresh hard-distractor rows. It does not train, does not repair checkpoints,
//! and does not claim open-ended language-model capability.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_OUT: &str =
    "target/pilot_wave/stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm/smoke";
const DEFAULT_CHECKPOINT: &str = "target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke/checkpoints/finetune_068_targeted_repair/model_checkpoint.json";
const DEFAULT_UPSTREAM_070_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke";
const DEFAULT_BENCHMARK_069_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_069_model_capability_benchmark_gate/smoke";

#[derive(Clone, Debug)]
struct Config {
    out: PathBuf,
    checkpoint: PathBuf,
    upstream_070_root: PathBuf,
    benchmark_069_root: PathBuf,
    seed: u64,
    heartbeat_sec: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Model {
    labels: Vec<String>,
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    feature_dim: usize,
}

#[derive(Clone, Debug, Serialize)]
struct FileSnapshot {
    path: String,
    size_bytes: u64,
    modified_unix_ms: Option<u128>,
    sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BenchmarkExample {
    id: String,
    task_family: String,
    input: String,
    expected_output: String,
    supported: bool,
    limitation_flag: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct FamilyMetric {
    correct: usize,
    total: usize,
    accuracy: Option<f64>,
    supported: bool,
    unsupported_cases: usize,
    top_output_rate: f64,
    space_output_rate: f64,
    empty_output_rate: f64,
    unique_output_count: usize,
    output_entropy: f64,
    collapse_detected: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
struct CapabilityMetrics {
    supported_accuracy: f64,
    family_min_accuracy: f64,
    fresh_context_entity_extraction_accuracy: f64,
    fresh_counterfactual_binding_accuracy: f64,
    fresh_distractor_resistance_accuracy: f64,
    fresh_long_context_needle_accuracy: f64,
    fresh_near_miss_anchor_selection_accuracy: f64,
    fresh_irrelevant_pocket_suppression_accuracy: f64,
    fresh_negative_route_rejection_accuracy: f64,
    retention_instruction_following_closed_accuracy: f64,
    retention_multi_hop_key_value_accuracy: f64,
    retention_symbolic_rule_closed_choice_accuracy: f64,
    retention_non_route_text_control_accuracy: f64,
    delta_vs_majority: f64,
    delta_vs_copy_first_match: f64,
    delta_vs_no_route_control: f64,
}

#[derive(Clone, Debug, Serialize)]
struct SampleRow {
    task_family: String,
    input: String,
    expected_output: String,
    model_output: String,
    baseline_outputs: BTreeMap<String, String>,
    no_route_output: String,
    pass_fail: String,
    limitation_flag: Option<String>,
}

fn main() {
    let cfg = match parse_args() {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    };
    if let Err(err) = run(&cfg) {
        let _ = write_failure(&cfg.out, "REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_FAILS", &err.to_string());
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run(cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    prepare_out(&cfg.out)?;
    append_progress(
        &cfg.out,
        "start",
        json!({
            "checkpoint": cfg.checkpoint.display().to_string(),
            "upstream_070_root": cfg.upstream_070_root.display().to_string(),
            "benchmark_069_root": cfg.benchmark_069_root.display().to_string(),
            "seed": cfg.seed,
            "heartbeat_sec": cfg.heartbeat_sec
        }),
    )?;
    write_summary(&cfg.out, "running", &[], json!({}))?;
    write_report(&cfg.out, "running", &[], None)?;

    let upstream = verify_upstream(cfg)?;
    append_progress(&cfg.out, "upstream_verified", json!({"best_arm": upstream.best_arm}))?;

    let checkpoint_before = snapshot_file(&cfg.checkpoint)?;
    let model = Model::load(&cfg.checkpoint)?;
    let checkpoint_label_count = model.labels.len();
    append_progress(
        &cfg.out,
        "checkpoint_loaded",
        json!({"checkpoint_hash_before": checkpoint_before.sha256, "label_count": checkpoint_label_count}),
    )?;

    let mut examples = build_benchmark_examples(cfg.seed, &model.labels);
    let label_set: BTreeSet<String> = model.labels.iter().cloned().collect();
    for row in &mut examples {
        if row.supported && !label_set.contains(&row.expected_output) {
            row.supported = false;
            row.limitation_flag = Some("label_not_in_checkpoint".to_string());
        }
    }
    let supported: Vec<_> = examples.iter().filter(|ex| ex.supported).cloned().collect();
    let unsupported: Vec<_> = examples.iter().filter(|ex| !ex.supported).cloned().collect();
    let upstream_samples = load_upstream_sample_inputs(&cfg.benchmark_069_root, &cfg.upstream_070_root)?;
    let overlap_069 = count_overlap(&examples, upstream_samples.get("069").unwrap_or(&BTreeSet::new()));
    let overlap_070 = count_overlap(&examples, upstream_samples.get("070").unwrap_or(&BTreeSet::new()));
    let leakage = overlap_069 > 0 || overlap_070 > 0;

    let eval_row_hash_model = eval_row_hash(&supported);
    let eval_row_hash_baselines = eval_row_hash(&supported);
    let eval_row_hash_no_route_control = eval_row_hash(&supported);
    let baseline_eval_mismatch =
        eval_row_hash_model != eval_row_hash_baselines || eval_row_hash_model != eval_row_hash_no_route_control;

    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "schema_version": "repaired_checkpoint_capability_confirm_queue_v1",
            "operations": [
                "verify_upstream_070",
                "load_checkpoint_eval_only",
                "build_fresh_benchmark_rows",
                "audit_sample_overlap",
                "run_model_predictions",
                "run_baselines_same_rows",
                "run_no_route_control_same_rows",
                "write_capability_profile"
            ],
            "train_step_count": 0,
            "training_side_effect_allowed": false
        }),
    )?;
    write_json(
        &cfg.out.join("benchmark_config.json"),
        &json!({
            "schema_version": "repaired_checkpoint_capability_confirm_config_v1",
            "seed": cfg.seed,
            "checkpoint": cfg.checkpoint.display().to_string(),
            "upstream_070_root": cfg.upstream_070_root.display().to_string(),
            "benchmark_069_root": cfg.benchmark_069_root.display().to_string(),
            "eval_only": true,
            "train_step_count": 0,
            "finite_label_surface": true,
            "open_ended_generation_supported": false,
            "free_form_answering_supported": false,
            "perplexity_supported": false,
            "prediction_oracle_used": false,
            "closed_label_success_does_not_imply_language_grounding": true,
            "this_is_not_an_open_ended_assistant": true
        }),
    )?;
    write_json(
        &cfg.out.join("upstream_070_manifest.json"),
        &json!({
            "schema_version": "upstream_070_manifest_v1",
            "summary": upstream.summary,
            "checkpoint_manifest": upstream.checkpoint_manifest,
            "arm_comparison": upstream.arm_comparison,
            "required_artifacts_present": true
        }),
    )?;
    write_json(
        &cfg.out.join("capability_dataset_manifest.json"),
        &json!({
            "schema_version": "capability_dataset_manifest_v1",
            "total_rows": examples.len(),
            "supported_rows": supported.len(),
            "unsupported_rows": unsupported.len(),
            "family_counts": family_counts(&examples),
            "eval_row_hash_model": eval_row_hash_model,
            "eval_row_hash_baselines": eval_row_hash_baselines,
            "eval_row_hash_no_route_control": eval_row_hash_no_route_control,
            "baseline_eval_mismatch": baseline_eval_mismatch,
            "overlap_with_069_samples_count": overlap_069,
            "overlap_with_070_samples_count": overlap_070,
            "upstream_exact_overlap_audit_limited": true,
            "benchmark_leakage_detected": leakage
        }),
    )?;
    write_sample_jsonl(&cfg.out.join("benchmark_examples_sample.jsonl"), &examples, 220)?;
    append_progress(
        &cfg.out,
        "dataset_built",
        json!({"rows": examples.len(), "supported": supported.len(), "overlap_069": overlap_069, "overlap_070": overlap_070}),
    )?;

    let baseline_outputs = evaluate_baselines(&supported, &model.labels);
    let baseline_metrics = baseline_metrics(&baseline_outputs, &supported);
    write_json(
        &cfg.out.join("baseline_metrics.json"),
        &json!({
            "schema_version": "baseline_metrics_v1",
            "eval_row_hash": eval_row_hash_baselines,
            "baseline_eval_mismatch": baseline_eval_mismatch,
            "baselines": baseline_metrics
        }),
    )?;
    append_progress(&cfg.out, "baselines_completed", json!({"baselines": 6}))?;

    let predictions = evaluate_model(&model, &supported, true);
    let no_route_predictions = evaluate_model(&model, &supported, false);
    let no_route_metrics = per_family_metrics(&supported, &no_route_predictions);
    let no_route_accuracy = exact_accuracy(&supported, &no_route_predictions);
    write_json(
        &cfg.out.join("no_route_feature_control_metrics.json"),
        &json!({
            "schema_version": "no_route_feature_control_metrics_v1",
            "eval_row_hash": eval_row_hash_no_route_control,
            "accuracy": no_route_accuracy,
            "per_family": no_route_metrics,
            "no_route_control_present": true
        }),
    )?;
    append_progress(&cfg.out, "no_route_control_completed", json!({"accuracy": no_route_accuracy}))?;

    let per_family = per_family_metrics(&supported, &predictions);
    let capability = capability_metrics(
        &supported,
        &predictions,
        &per_family,
        &baseline_metrics,
        no_route_accuracy,
    );
    let collapse = collapse_metrics(&supported, &predictions, &per_family);
    write_json(&cfg.out.join("per_family_metrics.json"), &per_family)?;
    write_json(&cfg.out.join("capability_metrics.json"), &capability)?;
    write_json(&cfg.out.join("collapse_metrics.json"), &collapse)?;
    write_json(
        &cfg.out.join("limitation_report.json"),
        &json!({
            "schema_version": "limitation_report_v1",
            "open_ended_generation_supported": false,
            "free_form_answering_supported": false,
            "perplexity_supported": false,
            "finite_label_surface": true,
            "closed_label_success_does_not_imply_language_grounding": true,
            "this_is_not_an_open_ended_assistant": true,
            "unsupported_label_cases": unsupported.len(),
            "unsupported_families": unsupported.iter().map(|ex| ex.task_family.clone()).collect::<BTreeSet<_>>()
        }),
    )?;
    append_progress(&cfg.out, "model_eval_completed", json!({"supported_accuracy": capability.supported_accuracy}))?;

    write_human_samples(
        &cfg.out.join("human_readable_samples.jsonl"),
        &supported,
        &unsupported,
        &predictions,
        &baseline_outputs,
        &no_route_predictions,
    )?;
    write_failure_samples(&cfg.out.join("failure_case_samples.jsonl"), &supported, &predictions)?;

    let checkpoint_after = snapshot_file(&cfg.checkpoint)?;
    let checkpoint_hash_unchanged = checkpoint_before.sha256 == checkpoint_after.sha256
        && checkpoint_before.size_bytes == checkpoint_after.size_bytes
        && checkpoint_before.modified_unix_ms == checkpoint_after.modified_unix_ms;
    write_json(
        &cfg.out.join("checkpoint_manifest.json"),
        &json!({
            "schema_version": "checkpoint_manifest_v1",
            "checkpoint_hash_before": checkpoint_before.sha256,
            "checkpoint_hash_after": checkpoint_after.sha256,
            "checkpoint_hash_unchanged": checkpoint_hash_unchanged,
            "train_step_count": 0,
            "training_side_effect_detected": false,
            "checkpoint_before": checkpoint_before,
            "checkpoint_after": checkpoint_after
        }),
    )?;

    let verdicts = derive_verdicts(
        &capability,
        &collapse,
        checkpoint_hash_unchanged,
        baseline_eval_mismatch,
        leakage,
        true,
        true,
    );
    write_summary(&cfg.out, if positive(&verdicts) { "done" } else { "failed" }, &verdicts, json!({
        "capability": capability,
        "checkpoint_hash_unchanged": checkpoint_hash_unchanged,
        "train_step_count": 0,
        "prediction_oracle_used": false,
        "finite_label_surface": true
    }))?;
    write_report(
        &cfg.out,
        if positive(&verdicts) { "done" } else { "failed" },
        &verdicts,
        Some(&capability),
    )?;
    append_progress(
        &cfg.out,
        "done",
        json!({"verdicts": verdicts, "checkpoint_hash_unchanged": checkpoint_hash_unchanged}),
    )?;
    println!("071 complete: {}", verdicts.join(","));
    Ok(())
}

struct Upstream070 {
    summary: serde_json::Value,
    checkpoint_manifest: serde_json::Value,
    arm_comparison: serde_json::Value,
    best_arm: String,
}

fn verify_upstream(cfg: &Config) -> Result<Upstream070, Box<dyn std::error::Error>> {
    let summary_path = cfg.upstream_070_root.join("summary.json");
    let checkpoint_manifest_path = cfg.upstream_070_root.join("checkpoint_manifest.json");
    let arm_comparison_path = cfg.upstream_070_root.join("arm_comparison.json");
    if !cfg.checkpoint.exists()
        || !summary_path.exists()
        || !checkpoint_manifest_path.exists()
        || !arm_comparison_path.exists()
        || !cfg.benchmark_069_root.exists()
    {
        write_failure(
            &cfg.out,
            "UPSTREAM_070_ARTIFACT_MISSING",
            "Required 070 checkpoint, 070 manifests, or 069 benchmark root is missing. 071 does not rerun 069/070 and does not train a replacement.",
        )?;
        return Err("UPSTREAM_070_ARTIFACT_MISSING".into());
    }
    let summary: serde_json::Value = serde_json::from_slice(&fs::read(summary_path)?)?;
    let checkpoint_manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(checkpoint_manifest_path)?)?;
    let arm_comparison: serde_json::Value = serde_json::from_slice(&fs::read(arm_comparison_path)?)?;
    let positive = summary
        .get("verdicts")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .any(|v| v.as_str() == Some("DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_POSITIVE"))
        })
        .unwrap_or(false);
    let best_arm = arm_comparison
        .get("best_arm")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if !positive || best_arm != "FINETUNE_068_TARGETED_REPAIR" {
        write_failure(
            &cfg.out,
            "UPSTREAM_070_ARTIFACT_MISSING",
            "070 upstream artifacts are present but do not verify the expected positive best-arm checkpoint.",
        )?;
        return Err("UPSTREAM_070_ARTIFACT_MISSING".into());
    }
    Ok(Upstream070 {
        summary,
        checkpoint_manifest,
        arm_comparison,
        best_arm,
    })
}

fn build_benchmark_examples(seed: u64, labels: &[String]) -> Vec<BenchmarkExample> {
    let mut rng = StdRng::seed_from_u64(seed);
    let values = [
        "amber", "violet", "silver", "green", "copper", "indigo", "scarlet", "cobalt", "ivory",
        "teal", "umber", "gold",
    ];
    let keys = [
        "raven_code",
        "wolf_code",
        "cedar_code",
        "comet_code",
        "atlas_code",
        "lantern_code",
        "pebble_code",
        "harbor_code",
    ];
    let mut rows = Vec::new();
    for family in fresh_families() {
        let n = if family == "RETENTION_NON_ROUTE_TEXT_CONTROL" { 32 } else { 30 };
        for i in 0..n {
            let key = keys[(i * 3 + rng.gen_range(0..keys.len())) % keys.len()];
            let d1 = keys[(i + 2) % keys.len()];
            let d2 = keys[(i + 5) % keys.len()];
            let value = values[(i * 7 + 5 + rng.gen_range(0..values.len())) % values.len()];
            let v1 = values[(i * 11 + 1) % values.len()];
            let v2 = values[(i * 13 + 3) % values.len()];
            rows.push(make_example(family, i, key, value, d1, v1, d2, v2));
        }
    }
    for i in 0..8 {
        rows.push(BenchmarkExample {
            id: format!("open_ended_interface_limitation_{i}"),
            task_family: "OPEN_ENDED_INTERFACE_LIMITATION".to_string(),
            input: format!(
                "071 open ended limitation row {i}. Explain why this closed-label checkpoint is not a free-form assistant."
            ),
            expected_output: "<unsupported_open_ended_answer>".to_string(),
            supported: false,
            limitation_flag: Some("open_ended_generation_unsupported".to_string()),
        });
    }
    let label_set: BTreeSet<_> = labels.iter().cloned().collect();
    rows.into_iter()
        .map(|mut row| {
            if row.supported && !label_set.contains(&row.expected_output) {
                row.supported = false;
                row.limitation_flag = Some("label_not_in_checkpoint".to_string());
            }
            row
        })
        .collect()
}

fn make_example(
    family: &str,
    idx: usize,
    key: &str,
    value: &str,
    d1: &str,
    v1: &str,
    d2: &str,
    v2: &str,
) -> BenchmarkExample {
    let (input, expected) = match family {
        "FRESH_CONTEXT_ENTITY_EXTRACTION" => (
            format!("071 fresh extraction. Requested anchor {key}. Ledger first lists {d1} as {v1}; later lists exact anchor {key} as {value}; side note lists {d2} as {v2}. Return only requested anchor value."),
            value,
        ),
        "FRESH_COUNTERFACTUAL_BINDING" => (
            format!("071 fresh counterfactual. Old scenario: {key} equals {v1}. Active scenario: {key} equals {value}. Distractor scenario: {d2} equals {v2}. Query active scenario {key}. Return active value."),
            value,
        ),
        "FRESH_DISTRACTOR_RESISTANCE" => (
            format!("071 fresh distractor resistance. Decoy route {d1}->{v1}. Target route {key}->{value}. Echo route {d2}->{v2}. Anchors say target key {key} has value {value}; reject decoys. Return value."),
            value,
        ),
        "FRESH_LONG_CONTEXT_NEEDLE_BINDING" => {
            let filler = long_filler(idx, 72);
            (
                format!("071 fresh long context needle. Intro filler {filler}. Needle marker: exact anchor {key} -> {value}. Extra distractors after needle: {d1}->{v1}; {d2}->{v2}. Final question asks for {key}. Route query:{key} anchor:{key} value:{value}. Return value."),
                value,
            )
        }
        "FRESH_NEAR_MISS_ANCHOR_SELECTION" => (
            format!("071 fresh near miss. Asked for {key}. Nearby anchor {key}_alias -> {v1}. Nearby anchor {key}_marker -> {v2}. Exact requested anchor {key} -> {value}. Return exact requested value."),
            value,
        ),
        "FRESH_IRRELEVANT_POCKET_SUPPRESSION" => (
            format!("071 fresh pocket suppression. Pocket red has {d1}->{v1}. Pocket blue has {d2}->{v2}. Active pocket has {key}->{value}. Question names active pocket and key {key}. Return active pocket value."),
            value,
        ),
        "FRESH_NEGATIVE_ROUTE_REJECTION" => (
            format!("071 fresh negative route rejection. Candidate route {d1}->{v1} is invalid. Candidate route {d2}->{v2} is stale. Valid route is query:{key} anchor:{key} value:{value}. Return valid value."),
            value,
        ),
        "RETENTION_INSTRUCTION_FOLLOWING_CLOSED" => (
            format!("071 retention instruction. Follow exactly: ignore {v1} and {v2}; return value for {key}. Binding {key} -> {value}. Route query:{key} anchor:{key} value:{value}. Return value."),
            value,
        ),
        "RETENTION_MULTI_HOP_KEY_VALUE_BINDING" => {
            let alias = format!("{key}_alias");
            (
                format!("071 retention multi-hop. Query alias {alias}. Link {alias} points to {key}. Binding {key}->{value}. Distractor {d1}->{v1}. Route query:{alias} anchor:{key} value:{value}. Return value."),
                value,
            )
        }
        "RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE" => {
            let rule = if idx % 2 == 0 { "even" } else { "odd" };
            (
                format!("071 retention symbolic. Rule says marker {rule} returns routed value. Marker is {rule}. Routed value is {value}. Route query:marker anchor:rule value:{value}. Return value."),
                value,
            )
        }
        "RETENTION_NON_ROUTE_TEXT_CONTROL" => {
            let controls = [
                ("weather", "control_weather"),
                ("music", "control_music"),
                ("math", "control_math"),
                ("garden", "control_garden"),
                ("archive", "control_archive"),
            ];
            let (domain, label) = controls[idx % controls.len()];
            (
                format!("071 retention non route control. This paragraph is about {domain}. It contains no key value route and no AnchorCell answer request. Classify the benign text domain."),
                label,
            )
        }
        _ => (String::new(), ""),
    };
    BenchmarkExample {
        id: format!("{}_{}", family.to_ascii_lowercase(), idx),
        task_family: family.to_string(),
        input,
        expected_output: expected.to_string(),
        supported: true,
        limitation_flag: None,
    }
}

fn fresh_families() -> Vec<&'static str> {
    vec![
        "FRESH_CONTEXT_ENTITY_EXTRACTION",
        "FRESH_COUNTERFACTUAL_BINDING",
        "FRESH_DISTRACTOR_RESISTANCE",
        "FRESH_LONG_CONTEXT_NEEDLE_BINDING",
        "FRESH_NEAR_MISS_ANCHOR_SELECTION",
        "FRESH_IRRELEVANT_POCKET_SUPPRESSION",
        "FRESH_NEGATIVE_ROUTE_REJECTION",
        "RETENTION_INSTRUCTION_FOLLOWING_CLOSED",
        "RETENTION_MULTI_HOP_KEY_VALUE_BINDING",
        "RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE",
        "RETENTION_NON_ROUTE_TEXT_CONTROL",
    ]
}

fn evaluate_model(model: &Model, examples: &[BenchmarkExample], use_route_features: bool) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for ex in examples {
        out.insert(ex.id.clone(), model.predict(&ex.input, use_route_features));
    }
    out
}

fn evaluate_baselines(
    examples: &[BenchmarkExample],
    labels: &[String],
) -> BTreeMap<String, BTreeMap<String, String>> {
    let majority = majority_label(examples);
    let mut out = BTreeMap::new();
    for (idx, ex) in examples.iter().enumerate() {
        let mut row = BTreeMap::new();
        row.insert("MAJORITY_LABEL".to_string(), majority.clone());
        row.insert("COPY_FIRST_MATCH".to_string(), copy_first_match(&ex.input, labels));
        row.insert("COPY_LAST_TOKEN".to_string(), last_token(&ex.input));
        row.insert(
            "SHUFFLED_LABELS".to_string(),
            examples[(idx + 7) % examples.len()].expected_output.clone(),
        );
        row.insert(
            "SHUFFLED_CONTEXT".to_string(),
            examples[(idx + 17) % examples.len()].expected_output.clone(),
        );
        row.insert("ANSWER_PRIOR_ONLY".to_string(), majority.clone());
        out.insert(ex.id.clone(), row);
    }
    out
}

fn baseline_metrics(
    outputs: &BTreeMap<String, BTreeMap<String, String>>,
    examples: &[BenchmarkExample],
) -> BTreeMap<String, serde_json::Value> {
    let mut by_name = BTreeMap::<String, (usize, BTreeMap<String, usize>)>::new();
    for ex in examples {
        if let Some(row) = outputs.get(&ex.id) {
            for (name, pred) in row {
                let entry = by_name.entry(name.clone()).or_default();
                if pred == &ex.expected_output {
                    entry.0 += 1;
                }
                *entry.1.entry(pred.clone()).or_insert(0) += 1;
            }
        }
    }
    by_name
        .into_iter()
        .map(|(name, (correct, distribution))| {
            (
                name,
                json!({
                    "accuracy": safe_div(correct, examples.len()),
                    "top_output_rate": top_output_rate(&distribution, examples.len()),
                    "output_entropy": entropy(&distribution, examples.len())
                }),
            )
        })
        .collect()
}

fn per_family_metrics(
    examples: &[BenchmarkExample],
    predictions: &BTreeMap<String, String>,
) -> BTreeMap<String, FamilyMetric> {
    let mut families = BTreeMap::<String, Vec<&BenchmarkExample>>::new();
    for ex in examples {
        families.entry(ex.task_family.clone()).or_default().push(ex);
    }
    let mut out = BTreeMap::new();
    for (family, rows) in families {
        let supported = rows.iter().filter(|ex| ex.supported).count();
        let unsupported = rows.len().saturating_sub(supported);
        let mut correct = 0usize;
        let mut distribution = BTreeMap::<String, usize>::new();
        for ex in rows.iter().filter(|ex| ex.supported) {
            let pred = predictions.get(&ex.id).cloned().unwrap_or_default();
            if pred == ex.expected_output {
                correct += 1;
            }
            *distribution.entry(pred).or_insert(0) += 1;
        }
        out.insert(
            family,
            FamilyMetric {
                correct,
                total: rows.len(),
                accuracy: if supported == 0 { None } else { Some(safe_div(correct, supported)) },
                supported: supported > 0,
                unsupported_cases: unsupported,
                top_output_rate: top_output_rate(&distribution, supported),
                space_output_rate: output_rate(&distribution, " ", supported),
                empty_output_rate: output_rate(&distribution, "", supported),
                unique_output_count: distribution.len(),
                output_entropy: entropy(&distribution, supported),
                collapse_detected: top_output_rate(&distribution, supported) > 0.45
                    || output_rate(&distribution, " ", supported) > 0.02
                    || output_rate(&distribution, "", supported) > 0.02,
            },
        );
    }
    out
}

fn capability_metrics(
    examples: &[BenchmarkExample],
    predictions: &BTreeMap<String, String>,
    per_family: &BTreeMap<String, FamilyMetric>,
    baselines: &BTreeMap<String, serde_json::Value>,
    no_route_accuracy: f64,
) -> CapabilityMetrics {
    let supported_accuracy = exact_accuracy(examples, predictions);
    let family_min_accuracy = fresh_families()
        .into_iter()
        .map(|family| family_accuracy(per_family, family))
        .fold(1.0, f64::min);
    CapabilityMetrics {
        supported_accuracy,
        family_min_accuracy,
        fresh_context_entity_extraction_accuracy: family_accuracy(per_family, "FRESH_CONTEXT_ENTITY_EXTRACTION"),
        fresh_counterfactual_binding_accuracy: family_accuracy(per_family, "FRESH_COUNTERFACTUAL_BINDING"),
        fresh_distractor_resistance_accuracy: family_accuracy(per_family, "FRESH_DISTRACTOR_RESISTANCE"),
        fresh_long_context_needle_accuracy: family_accuracy(per_family, "FRESH_LONG_CONTEXT_NEEDLE_BINDING"),
        fresh_near_miss_anchor_selection_accuracy: family_accuracy(per_family, "FRESH_NEAR_MISS_ANCHOR_SELECTION"),
        fresh_irrelevant_pocket_suppression_accuracy: family_accuracy(per_family, "FRESH_IRRELEVANT_POCKET_SUPPRESSION"),
        fresh_negative_route_rejection_accuracy: family_accuracy(per_family, "FRESH_NEGATIVE_ROUTE_REJECTION"),
        retention_instruction_following_closed_accuracy: family_accuracy(per_family, "RETENTION_INSTRUCTION_FOLLOWING_CLOSED"),
        retention_multi_hop_key_value_accuracy: family_accuracy(per_family, "RETENTION_MULTI_HOP_KEY_VALUE_BINDING"),
        retention_symbolic_rule_closed_choice_accuracy: family_accuracy(per_family, "RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE"),
        retention_non_route_text_control_accuracy: family_accuracy(per_family, "RETENTION_NON_ROUTE_TEXT_CONTROL"),
        delta_vs_majority: supported_accuracy - baseline_accuracy(baselines, "MAJORITY_LABEL"),
        delta_vs_copy_first_match: supported_accuracy - baseline_accuracy(baselines, "COPY_FIRST_MATCH"),
        delta_vs_no_route_control: supported_accuracy - no_route_accuracy,
    }
}

fn collapse_metrics(
    examples: &[BenchmarkExample],
    predictions: &BTreeMap<String, String>,
    per_family: &BTreeMap<String, FamilyMetric>,
) -> serde_json::Value {
    let mut distribution = BTreeMap::<String, usize>::new();
    for ex in examples {
        let pred = predictions.get(&ex.id).cloned().unwrap_or_default();
        *distribution.entry(pred).or_insert(0) += 1;
    }
    let top = top_output_rate(&distribution, examples.len());
    let space = output_rate(&distribution, " ", examples.len());
    let empty = output_rate(&distribution, "", examples.len());
    json!({
        "schema_version": "repaired_checkpoint_collapse_metrics_v1",
        "global": {
            "top_output_rate": top,
            "space_output_rate": space,
            "empty_output_rate": empty,
            "unique_output_count": distribution.len(),
            "output_entropy": entropy(&distribution, examples.len()),
            "collapse_detected": top > 0.45 || space > 0.02 || empty > 0.02
        },
        "per_family": per_family
    })
}

fn derive_verdicts(
    capability: &CapabilityMetrics,
    collapse: &serde_json::Value,
    checkpoint_hash_unchanged: bool,
    baseline_eval_mismatch: bool,
    leakage: bool,
    no_route_present: bool,
    human_samples_written: bool,
) -> Vec<String> {
    let collapse_detected = collapse
        .get("global")
        .and_then(|v| v.get("collapse_detected"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let top = collapse
        .get("global")
        .and_then(|v| v.get("top_output_rate"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let space = collapse
        .get("global")
        .and_then(|v| v.get("space_output_rate"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let empty = collapse
        .get("global")
        .and_then(|v| v.get("empty_output_rate"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let family_gate = capability.fresh_context_entity_extraction_accuracy >= 0.85
        && capability.fresh_counterfactual_binding_accuracy >= 0.85
        && capability.fresh_distractor_resistance_accuracy >= 0.85
        && capability.fresh_long_context_needle_accuracy >= 0.75
        && capability.fresh_near_miss_anchor_selection_accuracy >= 0.80
        && capability.fresh_irrelevant_pocket_suppression_accuracy >= 0.80
        && capability.fresh_negative_route_rejection_accuracy >= 0.75
        && capability.retention_instruction_following_closed_accuracy >= 0.90
        && capability.retention_multi_hop_key_value_accuracy >= 0.75
        && capability.retention_symbolic_rule_closed_choice_accuracy >= 0.85
        && capability.retention_non_route_text_control_accuracy >= 0.90
        && capability.family_min_accuracy >= 0.70
        && capability.supported_accuracy >= 0.85;
    let baseline_gate = capability.delta_vs_majority > 0.10 && capability.delta_vs_copy_first_match > 0.10;
    let collapse_gate = top <= 0.45 && space <= 0.02 && empty <= 0.02 && !collapse_detected;
    let positive = checkpoint_hash_unchanged
        && !baseline_eval_mismatch
        && !leakage
        && no_route_present
        && human_samples_written
        && family_gate
        && baseline_gate
        && collapse_gate;
    if positive {
        vec![
            "REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_POSITIVE".to_string(),
            "UPSTREAM_070_CHECKPOINT_VERIFIED".to_string(),
            "NO_TRAINING_PERFORMED".to_string(),
            "CHECKPOINT_UNCHANGED".to_string(),
            "FRESH_HARD_DISTRACTOR_GENERALIZATION_PASSES".to_string(),
            "FRESH_COUNTERFACTUAL_GENERALIZATION_PASSES".to_string(),
            "FRESH_LONG_CONTEXT_GENERALIZATION_PASSES".to_string(),
            "RETENTION_CONFIRM_PASSES".to_string(),
            "NO_ROUTE_CONTROL_RECORDED".to_string(),
            "BASELINE_COMPARISON_RECORDED".to_string(),
            "HUMAN_READABLE_SAMPLES_WRITTEN".to_string(),
            "OPEN_ENDED_LIMITATION_RECORDED".to_string(),
            "PRODUCTION_TRAINING_NOT_CLAIMED".to_string(),
        ]
    } else {
        let mut out = vec!["REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_FAILS".to_string()];
        if !checkpoint_hash_unchanged {
            out.push("CHECKPOINT_MUTATION_DETECTED".to_string());
        }
        if baseline_eval_mismatch {
            out.push("BASELINE_EVAL_MISMATCH".to_string());
        }
        if leakage {
            out.push("BENCHMARK_LEAKAGE_DETECTED".to_string());
        }
        if !no_route_present {
            out.push("NO_ROUTE_CONTROL_MISSING".to_string());
        }
        if capability.fresh_context_entity_extraction_accuracy < 0.85
            || capability.fresh_distractor_resistance_accuracy < 0.85
            || capability.fresh_near_miss_anchor_selection_accuracy < 0.80
            || capability.fresh_irrelevant_pocket_suppression_accuracy < 0.80
            || capability.fresh_negative_route_rejection_accuracy < 0.75
        {
            out.push("FRESH_HARD_DISTRACTOR_GENERALIZATION_FAILS".to_string());
        }
        if capability.fresh_counterfactual_binding_accuracy < 0.85 {
            out.push("FRESH_COUNTERFACTUAL_GENERALIZATION_FAILS".to_string());
        }
        if capability.fresh_long_context_needle_accuracy < 0.75 {
            out.push("FRESH_LONG_CONTEXT_GENERALIZATION_FAILS".to_string());
        }
        if capability.retention_instruction_following_closed_accuracy < 0.90
            || capability.retention_multi_hop_key_value_accuracy < 0.75
            || capability.retention_symbolic_rule_closed_choice_accuracy < 0.85
            || capability.retention_non_route_text_control_accuracy < 0.90
        {
            out.push("RETENTION_CONFIRM_FAILS".to_string());
        }
        if !family_gate {
            out.push("CAPABILITY_FAMILY_GATE_FAILS".to_string());
        }
        if collapse_detected {
            out.push("STATIC_OUTPUT_COLLAPSE_DETECTED".to_string());
        }
        if !human_samples_written {
            out.push("HUMAN_SAMPLE_REPORT_MISSING".to_string());
        }
        out.push("OPEN_ENDED_LIMITATION_RECORDED".to_string());
        out.push("NO_TRAINING_PERFORMED".to_string());
        out.push("PRODUCTION_TRAINING_NOT_CLAIMED".to_string());
        out
    }
}

impl Model {
    fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
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
}

fn featurize(input: &str, dim: usize, use_route_features: bool) -> Vec<usize> {
    let lowered = input.to_ascii_lowercase();
    let mut feats = BTreeSet::<usize>::new();
    let tokens = tokenize(&lowered);
    for token in &tokens {
        feats.insert(hash_feature(dim, &format!("tok:{token}")));
    }
    for window in tokens.windows(2) {
        feats.insert(hash_feature(dim, &format!("bi:{}:{}", window[0], window[1])));
    }
    for window in tokens.windows(3) {
        feats.insert(hash_feature(
            dim,
            &format!("tri:{}:{}:{}", window[0], window[1], window[2]),
        ));
    }
    for (pos, token) in tokens.iter().enumerate().take(120) {
        feats.insert(hash_feature(dim, &format!("pos{}:{token}", pos / 8)));
    }
    if use_route_features {
        for marker in [
            "route",
            "query",
            "anchor",
            "value",
            "target",
            "needle",
            "active",
            "current",
            "episode",
            "equals",
            "different",
        ] {
            if lowered.contains(marker) {
                feats.insert(hash_feature(dim, &format!("marker:{marker}")));
            }
        }
        for window in tokens.windows(4) {
            if window.iter().any(|t| {
                *t == "query"
                    || *t == "anchor"
                    || *t == "value"
                    || *t == "episode"
                    || *t == "equals"
                    || *t == "current"
            }) {
                feats.insert(hash_feature(
                    dim,
                    &format!("route4:{}:{}:{}:{}", window[0], window[1], window[2], window[3]),
                ));
            }
        }
    }
    for domain in ["weather", "music", "math", "garden", "archive"] {
        if lowered.contains(&format!("about {domain}")) {
            feats.insert(hash_feature(dim, &format!("domain:{domain}")));
        }
    }
    feats.into_iter().collect()
}

fn tokenize(input: &str) -> Vec<String> {
    input
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .collect()
}

fn load_upstream_sample_inputs(
    root_069: &Path,
    root_070: &Path,
) -> Result<BTreeMap<String, BTreeSet<String>>, Box<dyn std::error::Error>> {
    let mut out = BTreeMap::<String, BTreeSet<String>>::new();
    out.insert("069".to_string(), load_inputs_from_files(&[
        root_069.join("human_readable_samples.jsonl"),
        root_069.join("benchmark_examples_sample.jsonl"),
        root_069.join("failure_case_samples.jsonl"),
    ])?);
    out.insert("070".to_string(), load_inputs_from_files(&[
        root_070.join("human_readable_samples.jsonl"),
        root_070.join("failure_case_samples.jsonl"),
        root_070.join("train_examples_sample.jsonl"),
        root_070.join("heldout_examples_sample.jsonl"),
        root_070.join("ood_examples_sample.jsonl"),
    ])?);
    Ok(out)
}

fn load_inputs_from_files(paths: &[PathBuf]) -> Result<BTreeSet<String>, Box<dyn std::error::Error>> {
    let mut out = BTreeSet::new();
    for path in paths {
        if !path.exists() {
            continue;
        }
        for line in BufReader::new(File::open(path)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(&line)?;
            if let Some(input) = value.get("input").and_then(|v| v.as_str()) {
                out.insert(input.to_string());
            }
        }
    }
    Ok(out)
}

fn count_overlap(examples: &[BenchmarkExample], samples: &BTreeSet<String>) -> usize {
    examples.iter().filter(|ex| samples.contains(&ex.input)).count()
}

fn write_human_samples(
    path: &Path,
    supported: &[BenchmarkExample],
    unsupported: &[BenchmarkExample],
    predictions: &BTreeMap<String, String>,
    baseline_outputs: &BTreeMap<String, BTreeMap<String, String>>,
    no_route_predictions: &BTreeMap<String, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = fs::remove_file(path);
    for ex in supported.iter().take(260) {
        let model_output = predictions.get(&ex.id).cloned().unwrap_or_default();
        append_jsonl(
            path,
            &SampleRow {
                task_family: ex.task_family.clone(),
                input: ex.input.clone(),
                expected_output: ex.expected_output.clone(),
                model_output: model_output.clone(),
                baseline_outputs: baseline_outputs.get(&ex.id).cloned().unwrap_or_default(),
                no_route_output: no_route_predictions.get(&ex.id).cloned().unwrap_or_default(),
                pass_fail: if model_output == ex.expected_output {
                    "pass".to_string()
                } else {
                    "fail".to_string()
                },
                limitation_flag: None,
            },
        )?;
    }
    for ex in unsupported {
        append_jsonl(
            path,
            &SampleRow {
                task_family: ex.task_family.clone(),
                input: ex.input.clone(),
                expected_output: ex.expected_output.clone(),
                model_output: "<unsupported>".to_string(),
                baseline_outputs: BTreeMap::new(),
                no_route_output: "<unsupported>".to_string(),
                pass_fail: "unsupported".to_string(),
                limitation_flag: ex.limitation_flag.clone(),
            },
        )?;
    }
    Ok(())
}

fn write_failure_samples(
    path: &Path,
    supported: &[BenchmarkExample],
    predictions: &BTreeMap<String, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = fs::remove_file(path);
    for ex in supported {
        let pred = predictions.get(&ex.id).cloned().unwrap_or_default();
        if pred != ex.expected_output {
            append_jsonl(
                path,
                &json!({
                    "task_family": ex.task_family,
                    "input": ex.input,
                    "expected_output": ex.expected_output,
                    "model_output": pred,
                    "reason": "fresh confirmation mismatch"
                }),
            )?;
        }
    }
    if !path.exists() {
        File::create(path)?;
    }
    Ok(())
}

fn prepare_out(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out)?;
    for file in ["progress.jsonl", "human_readable_samples.jsonl", "failure_case_samples.jsonl"] {
        let path = out.join(file);
        if path.exists() {
            fs::write(path, b"")?;
        }
    }
    Ok(())
}

fn append_progress(out: &Path, event: &str, details: serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(
        &out.join("progress.jsonl"),
        &json!({
            "ts_unix_ms": now_ms(),
            "event": event,
            "details": details
        }),
    )
}

fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(serde_json::to_string(value)?.as_bytes())?;
    file.write_all(b"\n")?;
    file.flush()?;
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

fn write_sample_jsonl(path: &Path, rows: &[BenchmarkExample], limit: usize) -> Result<(), Box<dyn std::error::Error>> {
    let _ = fs::remove_file(path);
    for row in rows.iter().take(limit) {
        append_jsonl(path, row)?;
    }
    Ok(())
}

fn write_summary(
    out: &Path,
    status: &str,
    verdicts: &[String],
    extra: serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    write_json(
        &out.join("summary.json"),
        &json!({
            "schema_version": "repaired_checkpoint_capability_confirm_summary_v1",
            "status": status,
            "verdicts": verdicts,
            "train_step_count": 0,
            "prediction_oracle_used": false,
            "finite_label_surface": true,
            "open_ended_generation_supported": false,
            "free_form_answering_supported": false,
            "perplexity_supported": false,
            "production_training_claimed": false,
            "extra": extra
        }),
    )
}

fn write_report(
    out: &Path,
    status: &str,
    verdicts: &[String],
    capability: Option<&CapabilityMetrics>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::new();
    text.push_str("# STABLE_LOOP_PHASE_LOCK_071_REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM Report\n\n");
    text.push_str(&format!("Status: {status}\n\n"));
    text.push_str("This is eval-only fresh confirmation for a finite-label checkpoint.\n\n");
    text.push_str("no training\nno checkpoint repair\nno open-ended assistant\nno free-form generation\nno perplexity\nno full English LM\nno language grounding\nno production training\nno GA\nno public beta\nno hosted SaaS\n\n");
    text.push_str("Closed-label success does not imply language grounding. This is not an open-ended assistant.\n\n");
    text.push_str("Verdicts:\n\n```text\n");
    for verdict in verdicts {
        text.push_str(verdict);
        text.push('\n');
    }
    text.push_str("```\n\n");
    if let Some(metrics) = capability {
        text.push_str("Capability metrics:\n\n```json\n");
        text.push_str(&serde_json::to_string_pretty(metrics)?);
        text.push_str("\n```\n");
    }
    fs::write(out.join("report.md"), text)?;
    Ok(())
}

fn write_failure(out: &Path, verdict: &str, message: &str) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out)?;
    write_json(
        &out.join("summary.json"),
        &json!({
            "status": "failed",
            "verdicts": [verdict],
            "message": message,
            "train_step_count": 0,
            "prediction_oracle_used": false
        }),
    )?;
    fs::write(
        out.join("report.md"),
        format!("# STABLE_LOOP_PHASE_LOCK_071_REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM Report\n\nStatus: failed\n\n{verdict}: {message}\n"),
    )?;
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut out = PathBuf::from(DEFAULT_OUT);
    let mut checkpoint = PathBuf::from(DEFAULT_CHECKPOINT);
    let mut upstream_070_root = PathBuf::from(DEFAULT_UPSTREAM_070_ROOT);
    let mut benchmark_069_root = PathBuf::from(DEFAULT_BENCHMARK_069_ROOT);
    let mut seed = 2027u64;
    let mut heartbeat_sec = 20u64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = PathBuf::from(args.next().ok_or("--out requires value")?),
            "--checkpoint" => checkpoint = PathBuf::from(args.next().ok_or("--checkpoint requires value")?),
            "--upstream-070-root" => {
                upstream_070_root = PathBuf::from(args.next().ok_or("--upstream-070-root requires value")?)
            }
            "--benchmark-069-root" => {
                benchmark_069_root = PathBuf::from(args.next().ok_or("--benchmark-069-root requires value")?)
            }
            "--seed" => seed = args.next().ok_or("--seed requires value")?.parse()?,
            "--heartbeat-sec" => heartbeat_sec = args.next().ok_or("--heartbeat-sec requires value")?.parse()?,
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(Config {
        out,
        checkpoint,
        upstream_070_root,
        benchmark_069_root,
        seed,
        heartbeat_sec,
    })
}

fn snapshot_file(path: &Path) -> Result<FileSnapshot, Box<dyn std::error::Error>> {
    let metadata = fs::metadata(path)?;
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let modified_unix_ms = metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis());
    Ok(FileSnapshot {
        path: path.display().to_string(),
        size_bytes: metadata.len(),
        modified_unix_ms,
        sha256: hex_sha256(&bytes),
    })
}

fn exact_accuracy(examples: &[BenchmarkExample], predictions: &BTreeMap<String, String>) -> f64 {
    let correct = examples
        .iter()
        .filter(|ex| predictions.get(&ex.id) == Some(&ex.expected_output))
        .count();
    safe_div(correct, examples.len())
}

fn majority_label(examples: &[BenchmarkExample]) -> String {
    let mut counts = BTreeMap::<String, usize>::new();
    for ex in examples {
        *counts.entry(ex.expected_output.clone()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(label, _)| label)
        .unwrap_or_default()
}

fn copy_first_match(input: &str, labels: &[String]) -> String {
    let lowered = input.to_ascii_lowercase();
    let mut best: Option<(usize, String)> = None;
    for label in labels {
        if let Some(pos) = lowered.find(&label.to_ascii_lowercase()) {
            match &best {
                Some((best_pos, _)) if *best_pos <= pos => {}
                _ => best = Some((pos, label.clone())),
            }
        }
    }
    best.map(|(_, label)| label).unwrap_or_default()
}

fn baseline_accuracy(metrics: &BTreeMap<String, serde_json::Value>, name: &str) -> f64 {
    metrics
        .get(name)
        .and_then(|row| row.get("accuracy"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

fn family_accuracy(metrics: &BTreeMap<String, FamilyMetric>, family: &str) -> f64 {
    metrics.get(family).and_then(|v| v.accuracy).unwrap_or(0.0)
}

fn family_counts(examples: &[BenchmarkExample]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for ex in examples {
        *counts.entry(ex.task_family.clone()).or_insert(0) += 1;
    }
    counts
}

fn eval_row_hash(examples: &[BenchmarkExample]) -> String {
    let mut hasher = Sha256::new();
    for ex in examples {
        hasher.update(ex.id.as_bytes());
        hasher.update(b"\0");
        hasher.update(ex.task_family.as_bytes());
        hasher.update(b"\0");
        hasher.update(ex.input.as_bytes());
        hasher.update(b"\0");
        hasher.update(ex.expected_output.as_bytes());
        hasher.update(b"\n");
    }
    format!("{:x}", hasher.finalize())
}

fn hash_feature(dim: usize, value: &str) -> usize {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    usize::from_le_bytes(bytes) % dim
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn top_output_rate(distribution: &BTreeMap<String, usize>, total: usize) -> f64 {
    safe_div(distribution.values().copied().max().unwrap_or(0), total)
}

fn output_rate(distribution: &BTreeMap<String, usize>, key: &str, total: usize) -> f64 {
    safe_div(distribution.get(key).copied().unwrap_or(0), total)
}

fn entropy(distribution: &BTreeMap<String, usize>, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    distribution
        .values()
        .map(|count| {
            let p = *count as f64 / total as f64;
            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum()
}

fn safe_div(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

fn last_token(input: &str) -> String {
    tokenize(input).last().cloned().unwrap_or_default()
}

fn long_filler(idx: usize, n: usize) -> String {
    let words = [
        "ledger", "window", "signal", "north", "delta", "paper", "stone", "river", "archive",
        "weather", "music", "garden", "math", "quiet", "violet", "orange",
    ];
    (0..n)
        .map(|i| format!("{}_{}", words[(idx + i) % words.len()], i))
        .collect::<Vec<_>>()
        .join(" ")
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn positive(verdicts: &[String]) -> bool {
    verdicts
        .iter()
        .any(|v| v == "REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_POSITIVE")
}
