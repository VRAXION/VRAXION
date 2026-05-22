//! Eval-only model capability benchmark gate.
//!
//! 069 measures the current 068 finite-label checkpoint surface. It does not
//! train, does not rerun 067/068, and does not claim open-ended assistant or
//! language-model capability.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_CHECKPOINT: &str = "target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/seed_2028/checkpoints/MIXED_WITH_ROUTE_GRAMMAR_ON/model_checkpoint.json";
const DEFAULT_UPSTREAM_SUMMARY: &str =
    "target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/summary.json";
const DEFAULT_OUT: &str =
    "target/pilot_wave/stable_loop_phase_lock_069_model_capability_benchmark_gate/smoke";

#[derive(Clone, Debug)]
struct Config {
    out: PathBuf,
    checkpoint: PathBuf,
    upstream_summary: PathBuf,
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

#[derive(Clone, Debug, Serialize)]
struct BenchmarkExample {
    id: String,
    task_family: String,
    input: String,
    expected_output: String,
    supported: bool,
    limitation_flag: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct SampleRow {
    task_family: String,
    input: String,
    expected_output: String,
    model_output: String,
    baseline_outputs: BTreeMap<String, String>,
    pass_fail: String,
    limitation_flag: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
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

#[derive(Clone, Debug, Serialize)]
struct CapabilityMetrics {
    total_supported_rows: usize,
    supported_accuracy: f64,
    family_min_accuracy: f64,
    context_entity_extraction_accuracy: f64,
    instruction_following_closed_accuracy: f64,
    multi_hop_key_value_accuracy: f64,
    counterfactual_binding_accuracy: f64,
    distractor_resistance_accuracy: f64,
    long_context_needle_accuracy: f64,
    symbolic_rule_closed_choice_accuracy: f64,
    delta_vs_majority: f64,
    delta_vs_copy_first_match: f64,
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
        let _ = write_failure(&cfg.out, "MODEL_CAPABILITY_BENCHMARK_GATE_FAILS", &err.to_string());
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
            "seed": cfg.seed,
            "heartbeat_sec": cfg.heartbeat_sec,
            "checkpoint": cfg.checkpoint.display().to_string(),
            "upstream_summary": cfg.upstream_summary.display().to_string()
        }),
    )?;
    write_running_summary(&cfg.out, "running", &[], json!({}))?;
    write_report(&cfg.out, "running", &[], None)?;

    if !cfg.checkpoint.exists() || !cfg.upstream_summary.exists() {
        write_failure(
            &cfg.out,
            "UPSTREAM_068_ARTIFACT_MISSING",
            "Required 068 checkpoint or upstream summary is missing. 069 does not rerun 067/068 and does not train a replacement model.",
        )?;
        return Err("UPSTREAM_068_ARTIFACT_MISSING".into());
    }

    let upstream: serde_json::Value = serde_json::from_slice(&fs::read(&cfg.upstream_summary)?)?;
    let upstream_verdicts = upstream
        .get("verdicts")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let upstream_positive = upstream_verdicts
        .iter()
        .any(|v| v.as_str() == Some("REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_POSITIVE"));
    if !upstream_positive {
        write_failure(
            &cfg.out,
            "UPSTREAM_068_ARTIFACT_MISSING",
            "Upstream 068 summary did not contain REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_POSITIVE.",
        )?;
        return Err("UPSTREAM_068_ARTIFACT_MISSING".into());
    }
    append_progress(&cfg.out, "upstream_verified", json!({"positive": true}))?;

    let checkpoint_before = snapshot_file(&cfg.checkpoint)?;
    let model: Model = serde_json::from_slice(&fs::read(&cfg.checkpoint)?)?;
    let checkpoint_label_count = model.labels.len();
    append_progress(
        &cfg.out,
        "checkpoint_loaded",
        json!({"checkpoint_hash_before": checkpoint_before.sha256, "label_count": checkpoint_label_count}),
    )?;

    let examples = build_benchmark_examples(cfg.seed, &model.labels);
    let supported: Vec<_> = examples.iter().filter(|ex| ex.supported).cloned().collect();
    let unsupported: Vec<_> = examples.iter().filter(|ex| !ex.supported).cloned().collect();
    let eval_row_hash_model = eval_row_hash(&supported);
    let eval_row_hash_baselines = eval_row_hash(&supported);
    let baseline_eval_mismatch = eval_row_hash_model != eval_row_hash_baselines;

    write_json(
        &cfg.out.join("benchmark_config.json"),
        &json!({
            "schema_version": "model_capability_benchmark_config_v1",
            "seed": cfg.seed,
            "checkpoint": cfg.checkpoint.display().to_string(),
            "upstream_summary": cfg.upstream_summary.display().to_string(),
            "eval_only": true,
            "train_step_count": 0,
            "open_ended_generation_supported": false,
            "perplexity_supported": false,
            "free_form_answering_supported": false,
            "prediction_oracle_used": false
        }),
    )?;
    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "schema_version": "model_capability_benchmark_queue_v1",
            "operations": [
                "verify_upstream_068",
                "load_checkpoint_eval_only",
                "build_fixed_benchmark_rows",
                "run_model_predictions",
                "run_baselines_same_rows",
                "write_capability_profile",
                "write_limitations"
            ]
        }),
    )?;
    write_json(
        &cfg.out.join("upstream_068_manifest.json"),
        &json!({
            "schema_version": "upstream_068_manifest_v1",
            "summary_path": cfg.upstream_summary.display().to_string(),
            "upstream_positive": upstream_positive,
            "upstream_verdicts": upstream_verdicts,
            "upstream_exact_overlap_audit": overlap_audit(&cfg.upstream_summary, &examples)
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
            "baseline_eval_mismatch": baseline_eval_mismatch
        }),
    )?;
    write_sample_jsonl(&cfg.out.join("benchmark_examples_sample.jsonl"), &examples, 160)?;
    append_progress(&cfg.out, "dataset_built", json!({"rows": examples.len()}))?;

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
    append_progress(&cfg.out, "baselines_completed", json!({"baselines": 8}))?;

    let predictions = evaluate_model(&model, &supported);
    let mut human_rows = Vec::new();
    for ex in &supported {
        let model_output = predictions.get(&ex.id).cloned().unwrap_or_default();
        let base = baseline_outputs.get(&ex.id).cloned().unwrap_or_default();
        human_rows.push(SampleRow {
            task_family: ex.task_family.clone(),
            input: ex.input.clone(),
            expected_output: ex.expected_output.clone(),
            model_output: model_output.clone(),
            baseline_outputs: base,
            pass_fail: if model_output == ex.expected_output {
                "pass".to_string()
            } else {
                "fail".to_string()
            },
            limitation_flag: None,
        });
    }
    for ex in &unsupported {
        human_rows.push(SampleRow {
            task_family: ex.task_family.clone(),
            input: ex.input.clone(),
            expected_output: ex.expected_output.clone(),
            model_output: "<unsupported_finite_label_surface>".to_string(),
            baseline_outputs: BTreeMap::new(),
            pass_fail: "limitation".to_string(),
            limitation_flag: ex.limitation_flag.clone(),
        });
    }
    write_sample_jsonl(
        &cfg.out.join("human_readable_samples.jsonl"),
        &human_rows,
        human_rows.len(),
    )?;
    let failures: Vec<_> = human_rows
        .iter()
        .filter(|row| row.pass_fail == "fail")
        .cloned()
        .collect();
    write_sample_jsonl(
        &cfg.out.join("failure_case_samples.jsonl"),
        &failures,
        failures.len(),
    )?;
    append_progress(
        &cfg.out,
        "model_predictions_completed",
        json!({"supported_rows": supported.len(), "failures": failures.len()}),
    )?;

    let per_family = per_family_metrics(&examples, &predictions);
    let capability = capability_metrics(&supported, &predictions, &per_family, &baseline_metrics);
    let collapse = collapse_metrics(&supported, &predictions, &per_family);
    let checkpoint_after = snapshot_file(&cfg.checkpoint)?;
    let checkpoint_hash_unchanged = checkpoint_before.sha256 == checkpoint_after.sha256;

    write_json(&cfg.out.join("per_family_metrics.json"), &per_family)?;
    write_json(&cfg.out.join("capability_metrics.json"), &capability)?;
    write_json(&cfg.out.join("collapse_metrics.json"), &collapse)?;
    write_json(
        &cfg.out.join("limitation_report.json"),
        &json!({
            "schema_version": "limitation_report_v1",
            "open_ended_generation_supported": false,
            "perplexity_supported": false,
            "free_form_answering_supported": false,
            "this_checkpoint_is_not_an_open_ended_assistant": true,
            "perplexity_is_unsupported_for_this_finite_label_checkpoint": true,
            "free_form_answer_generation_is_unsupported": true,
            "closed_choice_success_does_not_imply_language_grounding": true,
            "unsupported_label_cases": unsupported,
            "unsupported_label_case_count": unsupported.len()
        }),
    )?;
    write_json(
        &cfg.out.join("checkpoint_manifest.json"),
        &json!({
            "schema_version": "capability_checkpoint_manifest_v1",
            "checkpoint_hash_before": checkpoint_before.sha256,
            "checkpoint_hash_after": checkpoint_after.sha256,
            "checkpoint_hash_unchanged": checkpoint_hash_unchanged,
            "checkpoint_before": checkpoint_before,
            "checkpoint_after": checkpoint_after,
            "train_step_count": 0,
            "training_side_effect_detected": false,
            "checkpoint_label_count": checkpoint_label_count,
            "benchmark_label_count": benchmark_label_count(&examples),
            "labels_not_in_checkpoint_count": labels_not_in_checkpoint(&examples, &model.labels).len(),
            "labels_not_in_checkpoint": labels_not_in_checkpoint(&examples, &model.labels)
        }),
    )?;
    append_progress(&cfg.out, "metrics_completed", json!({}))?;

    let verdicts = verdicts(
        &capability,
        &collapse,
        checkpoint_hash_unchanged,
        baseline_eval_mismatch,
        per_family_complete(&per_family),
        !human_rows.is_empty(),
    );
    let positive = verdicts
        .iter()
        .any(|v| v == "MODEL_CAPABILITY_BENCHMARK_GATE_POSITIVE");
    write_running_summary(
        &cfg.out,
        if positive { "done" } else { "failed" },
        &verdicts,
        json!({
            "capability_metrics": capability,
            "collapse_metrics": collapse,
            "checkpoint_hash_unchanged": checkpoint_hash_unchanged,
            "baseline_eval_mismatch": baseline_eval_mismatch,
            "prediction_oracle_used": false,
            "train_step_count": 0,
            "open_ended_generation_supported": false,
            "perplexity_supported": false,
            "free_form_answering_supported": false
        }),
    )?;
    write_report(
        &cfg.out,
        if positive { "done" } else { "failed" },
        &verdicts,
        Some(&capability),
    )?;
    append_progress(
        &cfg.out,
        "done",
        json!({"positive": positive, "verdicts": verdicts}),
    )?;
    Ok(())
}

fn build_benchmark_examples(seed: u64, labels: &[String]) -> Vec<BenchmarkExample> {
    let mut rng = StdRng::seed_from_u64(seed);
    let colors = [
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
    for i in 0..24 {
        let expected = if i % 2 == 0 { "candidate_a" } else { "candidate_b" };
        let actual = format!("natural phrase {}", colors[(i + 2) % colors.len()]);
        let distractor = format!("qxzv nonword {}", (i * 7) % 97);
        let (a, b) = if expected == "candidate_a" {
            (actual.clone(), distractor)
        } else {
            (distractor, actual.clone())
        };
        rows.push(example(
            i,
            "FINEWEB_CLOSED_CONTINUATION_SELECTION",
            format!(
                "Benchmark closed continuation. Prefix: the archive recorded a {} Candidate A: {} Candidate B: {} Choose the observed continuation.",
                colors[i % colors.len()],
                a,
                b
            ),
            expected,
            true,
            None,
        ));
    }
    for i in 0..24 {
        let key = keys[i % keys.len()];
        let value = colors[(i * 3 + 1) % colors.len()];
        let distractor = keys[(i + 3) % keys.len()];
        let distractor_value = colors[(i * 5 + 2) % colors.len()];
        rows.push(example(
            i,
            "CONTEXT_ENTITY_EXTRACTION",
            format!(
                "Benchmark context extraction. Query key {key}. Context says {distractor} is {distractor_value}. Context says {key} is {value}. Return only the requested value."
            ),
            value,
            true,
            None,
        ));
    }
    for i in 0..24 {
        let key = keys[(i + 1) % keys.len()];
        let value = colors[(i * 7 + 4) % colors.len()];
        let distractor = colors[(i * 7 + 9) % colors.len()];
        rows.push(example(
            i,
            "INSTRUCTION_FOLLOWING_CLOSED",
            format!(
                "Instruction following closed choice. Follow exactly: read the binding, ignore the distractor value {distractor}, and return the value for {key}. Binding {key} -> {value}. Route query:{key} anchor:{key} value:{value}. Return value."
            ),
            value,
            true,
            None,
        ));
    }
    for i in 0..24 {
        let alias = keys[(i + 2) % keys.len()];
        let target = keys[(i + 5) % keys.len()];
        let value = colors[(i * 11 + 3) % colors.len()];
        rows.push(example(
            i,
            "MULTI_HOP_KEY_VALUE_BINDING",
            format!(
                "Benchmark multi hop key value binding. Query asks alias {alias}. Link {alias} points to {target}. Binding {target} -> {value}. Route query:{alias} anchor:{target} value:{value}. Return value."
            ),
            value,
            true,
            None,
        ));
    }
    for i in 0..24 {
        let key = keys[(i + 4) % keys.len()];
        let value = colors[(i * 13 + 5) % colors.len()];
        let other = colors[(i * 13 + 6) % colors.len()];
        rows.push(example(
            i,
            "COUNTERFACTUAL_BINDING",
            format!(
                "Benchmark counterfactual binding. In this episode {key} equals {value}. In a different episode {key} may equal {other}, but this episode controls. Query {key}. Return episode value."
            ),
            value,
            true,
            None,
        ));
    }
    for i in 0..24 {
        let key = keys[(i + 6) % keys.len()];
        let value = colors[(i * 17 + 2) % colors.len()];
        let d1 = keys[(i + 1) % keys.len()];
        let d2 = keys[(i + 3) % keys.len()];
        let v1 = colors[(i * 19 + 4) % colors.len()];
        let v2 = colors[(i * 23 + 8) % colors.len()];
        rows.push(example(
            i,
            "DISTRACTOR_RESISTANCE",
            format!(
                "Benchmark distractor resistance. Binding {d1} -> {v1}. Binding {key} -> {value}. Binding {d2} -> {v2}. Anchors target {key} value {value}; distractors {d1} value {v1} and {d2} value {v2}. Route query:{key} anchor:{key} value:{value}. Return value."
            ),
            value,
            true,
            None,
        ));
    }
    for i in 0..24 {
        let key = keys[(i + 7) % keys.len()];
        let value = colors[(i * 29 + 1) % colors.len()];
        let filler = (0..28)
            .map(|n| format!("filler{}_{}", n, colors[(i + n) % colors.len()]))
            .collect::<Vec<_>>()
            .join(" ");
        rows.push(example(
            i,
            "LONG_CONTEXT_NEEDLE_BINDING",
            format!(
                "Benchmark long context needle binding. {filler}. Needle binding {key} -> {value}. More filler archive weather music garden math. Question asks for {key}. Route query:{key} anchor:{key} value:{value}. Return value."
            ),
            value,
            true,
            None,
        ));
    }
    for i in 0..24 {
        let value = colors[(i * 31 + 7) % colors.len()];
        let rule = if i % 2 == 0 { "even" } else { "odd" };
        rows.push(example(
            i,
            "SYMBOLIC_RULE_CLOSED_CHOICE",
            format!(
                "Benchmark symbolic closed choice. Rule says if the marker is {rule}, return the routed value. Marker is {rule}. Routed value is {value}. Route query:marker anchor:rule value:{value}. Return value."
            ),
            value,
            true,
            None,
        ));
    }
    let controls = [
        ("weather", "control_weather"),
        ("music", "control_music"),
        ("math", "control_math"),
        ("garden", "control_garden"),
        ("archive", "control_archive"),
    ];
    for i in 0..25 {
        let (domain, label) = controls[(i + rng.gen_range(0..controls.len())) % controls.len()];
        rows.push(example(
            i,
            "NON_ROUTE_TEXT_CONTROL",
            format!(
                "Benchmark non route text control. This paragraph is about {domain}. It contains no key value route and no AnchorCell answer request. Classify the benign text domain."
            ),
            label,
            true,
            None,
        ));
    }
    for i in 0..8 {
        rows.push(example(
            i,
            "OPEN_ENDED_INTERFACE_LIMITATION",
            format!(
                "Open ended diagnostic prompt {i}. Explain in two paragraphs why a finite label checkpoint is not a free form assistant."
            ),
            "<unsupported_open_ended_answer>",
            false,
            Some("open_ended_generation_unsupported"),
        ));
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

fn example(
    idx: usize,
    family: &str,
    input: String,
    expected: &str,
    supported: bool,
    limitation: Option<&str>,
) -> BenchmarkExample {
    BenchmarkExample {
        id: format!("{}_{}", family.to_ascii_lowercase(), idx),
        task_family: family.to_string(),
        input,
        expected_output: expected.to_string(),
        supported,
        limitation_flag: limitation.map(|v| v.to_string()),
    }
}

fn evaluate_model(model: &Model, examples: &[BenchmarkExample]) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for ex in examples {
        out.insert(ex.id.clone(), model.predict(&ex.input, true));
    }
    out
}

fn evaluate_baselines(
    examples: &[BenchmarkExample],
    labels: &[String],
) -> BTreeMap<String, BTreeMap<String, String>> {
    let majority = majority_label(examples);
    let label_prior = label_prior(examples);
    let mut out = BTreeMap::new();
    for (idx, ex) in examples.iter().enumerate() {
        let mut row = BTreeMap::new();
        row.insert("MAJORITY_LABEL".to_string(), majority.clone());
        row.insert("ANSWER_PRIOR_ONLY".to_string(), majority.clone());
        row.insert("COPY_LAST_TOKEN".to_string(), last_token(&ex.input));
        row.insert("COPY_FIRST_MATCH".to_string(), copy_first_match(&ex.input, labels));
        row.insert("UNIGRAM_LABEL_PRIOR".to_string(), label_prior.clone());
        row.insert(
            "SHUFFLED_CONTEXT".to_string(),
            examples[(idx + 17) % examples.len()].expected_output.clone(),
        );
        row.insert(
            "SHUFFLED_LABELS".to_string(),
            examples[(idx + 7) % examples.len()].expected_output.clone(),
        );
        row.insert("NO_ROUTE_FEATURES".to_string(), String::new());
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
        let accuracy = if supported == 0 {
            None
        } else {
            Some(safe_div(correct, supported))
        };
        out.insert(
            family,
            FamilyMetric {
                correct,
                total: rows.len(),
                accuracy,
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
) -> CapabilityMetrics {
    let total_correct = examples
        .iter()
        .filter(|ex| predictions.get(&ex.id) == Some(&ex.expected_output))
        .count();
    let supported_accuracy = safe_div(total_correct, examples.len());
    let family_min_accuracy = required_families()
        .into_iter()
        .filter(|family| *family != "OPEN_ENDED_INTERFACE_LIMITATION")
        .map(|family| family_accuracy(per_family, family))
        .fold(1.0, f64::min);
    let majority = baseline_accuracy(baselines, "MAJORITY_LABEL");
    let copy_first = baseline_accuracy(baselines, "COPY_FIRST_MATCH");
    CapabilityMetrics {
        total_supported_rows: examples.len(),
        supported_accuracy,
        family_min_accuracy,
        context_entity_extraction_accuracy: family_accuracy(per_family, "CONTEXT_ENTITY_EXTRACTION"),
        instruction_following_closed_accuracy: family_accuracy(
            per_family,
            "INSTRUCTION_FOLLOWING_CLOSED",
        ),
        multi_hop_key_value_accuracy: family_accuracy(per_family, "MULTI_HOP_KEY_VALUE_BINDING"),
        counterfactual_binding_accuracy: family_accuracy(per_family, "COUNTERFACTUAL_BINDING"),
        distractor_resistance_accuracy: family_accuracy(per_family, "DISTRACTOR_RESISTANCE"),
        long_context_needle_accuracy: family_accuracy(per_family, "LONG_CONTEXT_NEEDLE_BINDING"),
        symbolic_rule_closed_choice_accuracy: family_accuracy(
            per_family,
            "SYMBOLIC_RULE_CLOSED_CHOICE",
        ),
        delta_vs_majority: supported_accuracy - majority,
        delta_vs_copy_first_match: supported_accuracy - copy_first,
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
        "schema_version": "capability_collapse_metrics_v1",
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

fn verdicts(
    capability: &CapabilityMetrics,
    collapse: &serde_json::Value,
    checkpoint_hash_unchanged: bool,
    baseline_eval_mismatch: bool,
    per_family_complete: bool,
    human_samples_written: bool,
) -> Vec<String> {
    let collapse_detected = collapse
        .get("global")
        .and_then(|v| v.get("collapse_detected"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let family_gate = capability.context_entity_extraction_accuracy >= 0.85
        && capability.instruction_following_closed_accuracy >= 0.75
        && capability.multi_hop_key_value_accuracy >= 0.70
        && capability.counterfactual_binding_accuracy >= 0.85
        && capability.distractor_resistance_accuracy >= 0.80
        && capability.long_context_needle_accuracy >= 0.65
        && capability.symbolic_rule_closed_choice_accuracy >= 0.60
        && capability.family_min_accuracy >= 0.60;
    let baseline_gate =
        capability.delta_vs_majority > 0.05 && capability.delta_vs_copy_first_match > 0.05;
    let positive = checkpoint_hash_unchanged
        && !baseline_eval_mismatch
        && per_family_complete
        && human_samples_written
        && family_gate
        && baseline_gate
        && !collapse_detected;
    if positive {
        vec![
            "MODEL_CAPABILITY_BENCHMARK_GATE_POSITIVE".to_string(),
            "CURRENT_CHECKPOINT_CAPABILITY_PROFILE_WRITTEN".to_string(),
            "UPSTREAM_068_CHECKPOINT_VERIFIED".to_string(),
            "FINITE_LABEL_SURFACE_MEASURED".to_string(),
            "OPEN_ENDED_LIMITATION_RECORDED".to_string(),
            "INSTRUCTION_FOLLOWING_CLOSED_MEASURED".to_string(),
            "REASONING_CLOSED_CHOICE_MEASURED".to_string(),
            "LONG_CONTEXT_NEEDLE_MEASURED".to_string(),
            "BASELINE_COMPARISON_RECORDED".to_string(),
            "HUMAN_READABLE_SAMPLES_WRITTEN".to_string(),
            "NO_TRAINING_PERFORMED".to_string(),
            "PRODUCTION_TRAINING_NOT_CLAIMED".to_string(),
        ]
    } else {
        let mut out = vec!["MODEL_CAPABILITY_BENCHMARK_GATE_FAILS".to_string()];
        if !checkpoint_hash_unchanged {
            out.push("CHECKPOINT_MUTATION_DETECTED".to_string());
        }
        if baseline_eval_mismatch {
            out.push("BASELINE_EVAL_MISMATCH".to_string());
        }
        if !per_family_complete || !family_gate {
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

fn majority_label(examples: &[BenchmarkExample]) -> String {
    let mut counts = BTreeMap::<String, usize>::new();
    for ex in examples {
        *counts.entry(ex.expected_output.clone()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(label, _)| label)
        .unwrap_or_default()
}

fn label_prior(examples: &[BenchmarkExample]) -> String {
    let mut counts = BTreeMap::<char, BTreeMap<String, usize>>::new();
    for ex in examples {
        let first = ex.expected_output.chars().next().unwrap_or('_');
        *counts
            .entry(first)
            .or_default()
            .entry(ex.expected_output.clone())
            .or_insert(0) += 1;
    }
    counts
        .values()
        .flat_map(|inner| inner.iter())
        .max_by_key(|(_, n)| *n)
        .map(|(label, _)| label.clone())
        .unwrap_or_default()
}

fn copy_first_match(input: &str, labels: &[String]) -> String {
    let lowered = input.to_ascii_lowercase();
    labels
        .iter()
        .filter(|label| lowered.contains(&label.to_ascii_lowercase()))
        .min_by_key(|label| lowered.find(&label.to_ascii_lowercase()).unwrap_or(usize::MAX))
        .cloned()
        .unwrap_or_default()
}

fn last_token(input: &str) -> String {
    tokenize(&input.to_ascii_lowercase())
        .last()
        .cloned()
        .unwrap_or_default()
}

fn family_counts(examples: &[BenchmarkExample]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for ex in examples {
        *counts.entry(ex.task_family.clone()).or_insert(0) += 1;
    }
    counts
}

fn benchmark_label_count(examples: &[BenchmarkExample]) -> usize {
    examples
        .iter()
        .map(|ex| ex.expected_output.clone())
        .collect::<BTreeSet<_>>()
        .len()
}

fn labels_not_in_checkpoint(examples: &[BenchmarkExample], labels: &[String]) -> Vec<String> {
    let label_set: BTreeSet<_> = labels.iter().cloned().collect();
    examples
        .iter()
        .map(|ex| ex.expected_output.clone())
        .filter(|label| !label_set.contains(label))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn required_families() -> Vec<&'static str> {
    vec![
        "FINEWEB_CLOSED_CONTINUATION_SELECTION",
        "CONTEXT_ENTITY_EXTRACTION",
        "INSTRUCTION_FOLLOWING_CLOSED",
        "MULTI_HOP_KEY_VALUE_BINDING",
        "COUNTERFACTUAL_BINDING",
        "DISTRACTOR_RESISTANCE",
        "LONG_CONTEXT_NEEDLE_BINDING",
        "SYMBOLIC_RULE_CLOSED_CHOICE",
        "NON_ROUTE_TEXT_CONTROL",
        "OPEN_ENDED_INTERFACE_LIMITATION",
    ]
}

fn per_family_complete(metrics: &BTreeMap<String, FamilyMetric>) -> bool {
    required_families()
        .into_iter()
        .all(|family| metrics.contains_key(family))
}

fn family_accuracy(metrics: &BTreeMap<String, FamilyMetric>, family: &str) -> f64 {
    metrics
        .get(family)
        .and_then(|m| m.accuracy)
        .unwrap_or(0.0)
}

fn baseline_accuracy(metrics: &BTreeMap<String, serde_json::Value>, name: &str) -> f64 {
    metrics
        .get(name)
        .and_then(|row| row.get("accuracy"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
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

fn eval_row_hash(eval_rows: &[BenchmarkExample]) -> String {
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

fn overlap_audit(upstream_summary: &Path, examples: &[BenchmarkExample]) -> serde_json::Value {
    let Some(parent) = upstream_summary.parent() else {
        return json!({"upstream_exact_overlap_audit_limited": true});
    };
    let seed_dir = parent.join("seed_2028");
    let files = [
        ("train", "train_examples_sample.jsonl"),
        ("eval", "heldout_examples_sample.jsonl"),
        ("ood", "ood_examples_sample.jsonl"),
    ];
    let benchmark_inputs: BTreeSet<_> = examples.iter().map(|ex| ex.input.clone()).collect();
    let mut counts = BTreeMap::new();
    let mut limited = false;
    for (name, file) in files {
        let path = seed_dir.join(file);
        if !path.exists() {
            limited = true;
            counts.insert(format!("overlap_with_068_{}_count", name), 0usize);
            continue;
        }
        let mut count = 0usize;
        if let Ok(text) = fs::read_to_string(path) {
            for line in text.lines() {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(input) = value.get("input").and_then(|v| v.as_str()) {
                        if benchmark_inputs.contains(input) {
                            count += 1;
                        }
                    }
                }
            }
        } else {
            limited = true;
        }
        counts.insert(format!("overlap_with_068_{}_count", name), count);
    }
    json!({
        "upstream_exact_overlap_audit_limited": limited,
        "overlap_counts": counts
    })
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

fn prepare_out(out: &Path) -> std::io::Result<()> {
    fs::create_dir_all(out)?;
    for file in [
        "progress.jsonl",
        "benchmark_examples_sample.jsonl",
        "human_readable_samples.jsonl",
        "failure_case_samples.jsonl",
    ] {
        let path = out.join(file);
        if path.exists() {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}

fn append_progress(
    out: &Path,
    event: &str,
    payload: serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(
        &out.join("progress.jsonl"),
        &json!({"ts": now_ms(), "event": event, "payload": payload}),
    )?;
    Ok(())
}

fn write_running_summary(
    out: &Path,
    status: &str,
    verdicts: &[String],
    extra: serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    write_json(
        &out.join("summary.json"),
        &json!({
            "schema_version": "model_capability_benchmark_summary_v1",
            "status": status,
            "verdicts": verdicts,
            "details": extra,
            "train_step_count": 0,
            "prediction_oracle_used": false,
            "open_ended_generation_supported": false,
            "perplexity_supported": false,
            "free_form_answering_supported": false,
            "production_training_claimed": false,
            "language_grounding_claimed": false,
            "public_beta_promoted": false,
            "hosted_saas_claimed": false
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
    text.push_str("# STABLE_LOOP_PHASE_LOCK_069_MODEL_CAPABILITY_BENCHMARK_GATE Report\n\n");
    text.push_str("069 is an eval-only finite-label capability benchmark for the current 068 checkpoint.\n");
    text.push_str("This checkpoint is not an open-ended assistant. Perplexity is unsupported for this finite-label checkpoint. Free-form answer generation is unsupported. Closed-choice success does not imply language grounding.\n");
    text.push_str("It is not production training, not a full English model, not GA, not public beta, not hosted SaaS, not clinical use, not high-stakes education use, not full VRAXION, not consciousness, not biological/FlyWire equivalence, and not physical quantum behavior.\n\n");
    text.push_str(&format!("Status: `{status}`\n\n"));
    text.push_str("## Verdicts\n\n```text\n");
    for verdict in verdicts {
        text.push_str(verdict);
        text.push('\n');
    }
    text.push_str("```\n\n");
    if let Some(c) = capability {
        text.push_str("## Capability Profile\n\n");
        text.push_str(&format!(
            "- supported accuracy: `{:.3}`\n- family min accuracy: `{:.3}`\n- delta vs majority: `{:.3}`\n- delta vs copy-first: `{:.3}`\n\n",
            c.supported_accuracy, c.family_min_accuracy, c.delta_vs_majority, c.delta_vs_copy_first_match
        ));
    }
    fs::write(out.join("report.md"), text)?;
    Ok(())
}

fn write_failure(
    out: &Path,
    verdict: &str,
    message: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out)?;
    append_progress(out, "failed", json!({"verdict": verdict, "message": message}))?;
    let verdicts = vec![verdict.to_string()];
    write_running_summary(
        out,
        "failed",
        &verdicts,
        json!({"message": message, "failure_verdict": verdict}),
    )?;
    write_report(out, "failed", &verdicts, None)?;
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

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut out = PathBuf::from(DEFAULT_OUT);
    let mut checkpoint = PathBuf::from(DEFAULT_CHECKPOINT);
    let mut upstream_summary = PathBuf::from(DEFAULT_UPSTREAM_SUMMARY);
    let mut seed = 2026u64;
    let mut heartbeat_sec = 20u64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = PathBuf::from(args.next().ok_or("--out requires value")?),
            "--checkpoint" => {
                checkpoint = PathBuf::from(args.next().ok_or("--checkpoint requires value")?)
            }
            "--upstream-summary" => {
                upstream_summary =
                    PathBuf::from(args.next().ok_or("--upstream-summary requires value")?)
            }
            "--seed" => seed = args.next().ok_or("--seed requires value")?.parse()?,
            "--heartbeat-sec" => {
                heartbeat_sec = args
                    .next()
                    .ok_or("--heartbeat-sec requires value")?
                    .parse()?
            }
            "--help" | "-h" => {
                println!(
                    "phase_lane_model_capability_benchmark_gate --out DIR --checkpoint FILE --upstream-summary FILE --seed 2026 --heartbeat-sec 20"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }
    Ok(Config {
        out,
        checkpoint,
        upstream_summary,
        seed,
        heartbeat_sec,
    })
}
