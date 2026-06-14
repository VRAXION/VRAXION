use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::final_training::training_candidates_for_rounds;

const REQUIRED_SPLITS: [&str; 3] = ["train", "validation", "adversarial"];
const REQUIRED_FAMILIES: [&str; 8] = [
    "frame_integrity",
    "requested_feature_match",
    "trace_commit",
    "library_reload",
    "bitslip_recovery",
    "text_mode_selection",
    "agency_commit",
    "library_governance",
];
const REQUIRED_CAPABILITY_SIGNATURES: [&str; 8] = [
    "binary_training_frame_guard",
    "binary_training_request_guard",
    "binary_training_trace_guard",
    "binary_training_reload_guard",
    "binary_training_bitslip_guard",
    "text_training_mode_guard",
    "agency_training_commit_guard",
    "library_training_reload_guard",
];
const MIN_ROUNDS_FOR_FULL_CANDIDATE_ROTATION: u64 = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingDataReadinessConfig {
    pub lanes: usize,
    pub rounds_per_lane: u64,
    pub out: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingDataReadinessSummary {
    pub passed: bool,
    pub lanes: usize,
    pub rounds_per_lane: u64,
    pub lesson_count: usize,
    pub required_lesson_count: usize,
    pub split_count: usize,
    pub family_count: usize,
    pub capability_count: usize,
    pub candidate_unique_count: usize,
    pub duplicate_lesson_id_count: usize,
    pub missing_family_split_count: usize,
    pub missing_candidate_capability_count: usize,
    pub invalid_row_count: usize,
    pub score_contract_complete: bool,
    pub inference_contract_complete: bool,
    pub curriculum_digest: u64,
    pub seconds: f64,
    pub out: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingLessonSpec {
    pub lesson_id: String,
    pub split: String,
    pub family: String,
    pub requested_feature: u8,
    pub expected_value: u8,
    pub source_pocket_id: u32,
    pub evidence_digest: String,
    pub capability_signature: String,
    pub scoring_policy: String,
    pub inference_target: String,
}

#[derive(Debug, Clone, PartialEq)]
struct ReadinessEvaluation {
    lesson_count: usize,
    required_lesson_count: usize,
    split_count: usize,
    family_count: usize,
    capability_count: usize,
    candidate_unique_count: usize,
    duplicate_lesson_id_count: usize,
    missing_family_split_count: usize,
    missing_candidate_capability_count: usize,
    invalid_row_count: usize,
    score_contract_complete: bool,
    inference_contract_complete: bool,
    curriculum_digest: u64,
}

impl TrainingDataReadinessConfig {
    pub fn new(lanes: usize, rounds_per_lane: u64, out: PathBuf) -> Self {
        Self {
            lanes,
            rounds_per_lane,
            out,
        }
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_millis()
}

fn append_jsonl(path: &PathBuf, line: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create training data progress parent");
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open training data progress file");
    writeln!(file, "{line}").expect("write training data progress line");
}

fn json_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

fn json_string_array(items: &[&str]) -> String {
    items
        .iter()
        .map(|item| format!("\"{}\"", json_escape(item)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn canonical_training_lessons() -> Vec<TrainingLessonSpec> {
    let mut lessons = Vec::with_capacity(REQUIRED_FAMILIES.len() * REQUIRED_SPLITS.len());
    for (family_index, family) in REQUIRED_FAMILIES.iter().enumerate() {
        for (split_index, split) in REQUIRED_SPLITS.iter().enumerate() {
            let capability = REQUIRED_CAPABILITY_SIGNATURES[family_index];
            lessons.push(TrainingLessonSpec {
                lesson_id: format!("e79_{family}_{split}"),
                split: (*split).to_string(),
                family: (*family).to_string(),
                requested_feature: ((family_index * 3 + split_index + 1) % 23 + 1) as u8,
                expected_value: ((family_index + split_index) % 2) as u8,
                source_pocket_id: 79,
                evidence_digest: format!("digest_e79_{family}_{split}"),
                capability_signature: capability.to_string(),
                scoring_policy: "quality_delta_positive_and_zero_bad_commit".to_string(),
                inference_target: "agency_commit_to_egress_render".to_string(),
            });
        }
    }
    lessons
}

fn update_digest(acc: &mut u64, text: &str) {
    for byte in text.bytes() {
        *acc ^= byte as u64;
        *acc = acc.wrapping_mul(1_099_511_628_211);
    }
    *acc ^= 0xFF;
    *acc = acc.wrapping_mul(1_099_511_628_211);
}

fn curriculum_digest(lessons: &[TrainingLessonSpec]) -> u64 {
    let mut acc = 14_695_981_039_346_656_037u64;
    for lesson in lessons {
        update_digest(&mut acc, &lesson.lesson_id);
        update_digest(&mut acc, &lesson.split);
        update_digest(&mut acc, &lesson.family);
        update_digest(&mut acc, &lesson.requested_feature.to_string());
        update_digest(&mut acc, &lesson.expected_value.to_string());
        update_digest(&mut acc, &lesson.source_pocket_id.to_string());
        update_digest(&mut acc, &lesson.evidence_digest);
        update_digest(&mut acc, &lesson.capability_signature);
        update_digest(&mut acc, &lesson.scoring_policy);
        update_digest(&mut acc, &lesson.inference_target);
    }
    acc
}

fn evaluate_readiness(
    lanes: usize,
    rounds_per_lane: u64,
    lessons: &[TrainingLessonSpec],
) -> ReadinessEvaluation {
    let mut lesson_ids = BTreeSet::new();
    let mut splits = BTreeSet::new();
    let mut families = BTreeSet::new();
    let mut family_split_cells = BTreeSet::new();
    let mut lesson_capabilities = BTreeSet::new();
    let mut duplicate_lesson_id_count = 0usize;
    let mut invalid_row_count = 0usize;
    let mut inference_contract_complete = true;

    for lesson in lessons {
        if !lesson_ids.insert(lesson.lesson_id.as_str()) {
            duplicate_lesson_id_count += 1;
        }
        splits.insert(lesson.split.as_str());
        families.insert(lesson.family.as_str());
        family_split_cells.insert((lesson.family.as_str(), lesson.split.as_str()));
        lesson_capabilities.insert(lesson.capability_signature.as_str());

        let valid_split = REQUIRED_SPLITS.contains(&lesson.split.as_str());
        let valid_family = REQUIRED_FAMILIES.contains(&lesson.family.as_str());
        let valid_capability =
            REQUIRED_CAPABILITY_SIGNATURES.contains(&lesson.capability_signature.as_str());
        let valid_feature = (1..=23).contains(&lesson.requested_feature);
        let valid_value = lesson.expected_value <= 1;
        let valid_evidence = lesson.evidence_digest.starts_with("digest_e79_");
        let valid_score_policy =
            lesson.scoring_policy == "quality_delta_positive_and_zero_bad_commit";
        let valid_inference_target = lesson.inference_target == "agency_commit_to_egress_render";
        if !valid_inference_target {
            inference_contract_complete = false;
        }
        if !(valid_split
            && valid_family
            && valid_capability
            && valid_feature
            && valid_value
            && valid_evidence
            && valid_score_policy
            && valid_inference_target)
        {
            invalid_row_count += 1;
        }
    }

    let missing_family_split_count = REQUIRED_FAMILIES
        .iter()
        .flat_map(|family| REQUIRED_SPLITS.iter().map(move |split| (*family, *split)))
        .filter(|cell| !family_split_cells.contains(cell))
        .count();

    let candidates = training_candidates_for_rounds(rounds_per_lane);
    let candidate_capabilities: BTreeSet<&str> = candidates
        .iter()
        .map(|candidate| candidate.capability_signature)
        .collect();
    let missing_candidate_capability_count = REQUIRED_CAPABILITY_SIGNATURES
        .iter()
        .filter(|signature| !candidate_capabilities.contains(**signature))
        .count();
    let score_contract_complete = lanes > 0
        && rounds_per_lane >= MIN_ROUNDS_FOR_FULL_CANDIDATE_ROTATION
        && candidates
            .iter()
            .all(|candidate| candidate.quality_delta > 0.0)
        && missing_candidate_capability_count == 0
        && lesson_capabilities.len() == REQUIRED_CAPABILITY_SIGNATURES.len();

    ReadinessEvaluation {
        lesson_count: lessons.len(),
        required_lesson_count: REQUIRED_FAMILIES.len() * REQUIRED_SPLITS.len(),
        split_count: splits.len(),
        family_count: families.len(),
        capability_count: lesson_capabilities.len(),
        candidate_unique_count: candidates.len(),
        duplicate_lesson_id_count,
        missing_family_split_count,
        missing_candidate_capability_count,
        invalid_row_count,
        score_contract_complete,
        inference_contract_complete,
        curriculum_digest: curriculum_digest(lessons),
    }
}

fn write_curriculum_manifest(out: &Path, lessons: &[TrainingLessonSpec]) {
    let path = out.join("training_curriculum_manifest.jsonl");
    let _ = fs::remove_file(&path);
    for lesson in lessons {
        append_jsonl(
            &path,
            &format!(
                "{{\"lesson_id\":\"{}\",\"split\":\"{}\",\"family\":\"{}\",\"requested_feature\":{},\"expected_value\":{},\"source_pocket_id\":{},\"evidence_digest\":\"{}\",\"capability_signature\":\"{}\",\"scoring_policy\":\"{}\",\"inference_target\":\"{}\"}}",
                json_escape(&lesson.lesson_id),
                json_escape(&lesson.split),
                json_escape(&lesson.family),
                lesson.requested_feature,
                lesson.expected_value,
                lesson.source_pocket_id,
                json_escape(&lesson.evidence_digest),
                json_escape(&lesson.capability_signature),
                json_escape(&lesson.scoring_policy),
                json_escape(&lesson.inference_target)
            ),
        );
    }
}

fn write_manifest(out: &Path, config: &TrainingDataReadinessConfig) {
    fs::write(
        out.join("training_data_readiness_manifest.json"),
        format!(
            concat!(
                "{{\n",
                "  \"artifact_contract\": \"E79_TRAINING_DATA_CURRICULUM_READINESS\",\n",
                "  \"runtime_surface\": \"vraxion-runtime\",\n",
                "  \"lanes\": {},\n",
                "  \"rounds_per_lane\": {},\n",
                "  \"required_splits\": [{}],\n",
                "  \"required_families\": [{}],\n",
                "  \"required_capability_signatures\": [{}],\n",
                "  \"minimum_rounds_for_full_candidate_rotation\": {},\n",
                "  \"required_artifacts\": [\n",
                "    \"training_data_readiness_results.json\",\n",
                "    \"training_data_readiness_manifest.json\",\n",
                "    \"training_data_readiness_progress.jsonl\",\n",
                "    \"training_curriculum_manifest.jsonl\",\n",
                "    \"training_data_readiness_report.md\"\n",
                "  ],\n",
                "  \"boundary\": \"dataset/curriculum contract readiness gate; not final production data or trained weights\"\n",
                "}}\n"
            ),
            config.lanes,
            config.rounds_per_lane,
            json_string_array(&REQUIRED_SPLITS),
            json_string_array(&REQUIRED_FAMILIES),
            json_string_array(&REQUIRED_CAPABILITY_SIGNATURES),
            MIN_ROUNDS_FOR_FULL_CANDIDATE_ROTATION
        ),
    )
    .expect("write training data readiness manifest");
}

fn write_results(out: &Path, summary: &TrainingDataReadinessSummary) {
    fs::write(
        out.join("training_data_readiness_results.json"),
        format!(
            concat!(
                "{{\n",
                "  \"passed\": {},\n",
                "  \"lanes\": {},\n",
                "  \"rounds_per_lane\": {},\n",
                "  \"lesson_count\": {},\n",
                "  \"required_lesson_count\": {},\n",
                "  \"split_count\": {},\n",
                "  \"family_count\": {},\n",
                "  \"capability_count\": {},\n",
                "  \"candidate_unique_count\": {},\n",
                "  \"duplicate_lesson_id_count\": {},\n",
                "  \"missing_family_split_count\": {},\n",
                "  \"missing_candidate_capability_count\": {},\n",
                "  \"invalid_row_count\": {},\n",
                "  \"score_contract_complete\": {},\n",
                "  \"inference_contract_complete\": {},\n",
                "  \"curriculum_digest\": {},\n",
                "  \"seconds\": {:.9}\n",
                "}}\n"
            ),
            summary.passed,
            summary.lanes,
            summary.rounds_per_lane,
            summary.lesson_count,
            summary.required_lesson_count,
            summary.split_count,
            summary.family_count,
            summary.capability_count,
            summary.candidate_unique_count,
            summary.duplicate_lesson_id_count,
            summary.missing_family_split_count,
            summary.missing_candidate_capability_count,
            summary.invalid_row_count,
            summary.score_contract_complete,
            summary.inference_contract_complete,
            summary.curriculum_digest,
            summary.seconds
        ),
    )
    .expect("write training data readiness results");
}

fn write_report(out: &Path, summary: &TrainingDataReadinessSummary) {
    fs::write(
        out.join("training_data_readiness_report.md"),
        format!(
            "# E79 Training Data Curriculum Readiness\n\n\
             ```text\n\
             passed = {}\n\
             lanes = {}\n\
             rounds_per_lane = {}\n\
             lesson_count = {}\n\
             required_lesson_count = {}\n\
             split_count = {}\n\
             family_count = {}\n\
             capability_count = {}\n\
             candidate_unique_count = {}\n\
             duplicate_lesson_id_count = {}\n\
             missing_family_split_count = {}\n\
             missing_candidate_capability_count = {}\n\
             invalid_row_count = {}\n\
             score_contract_complete = {}\n\
             inference_contract_complete = {}\n\
             curriculum_digest = {}\n\
             ```\n",
            summary.passed,
            summary.lanes,
            summary.rounds_per_lane,
            summary.lesson_count,
            summary.required_lesson_count,
            summary.split_count,
            summary.family_count,
            summary.capability_count,
            summary.candidate_unique_count,
            summary.duplicate_lesson_id_count,
            summary.missing_family_split_count,
            summary.missing_candidate_capability_count,
            summary.invalid_row_count,
            summary.score_contract_complete,
            summary.inference_contract_complete,
            summary.curriculum_digest
        ),
    )
    .expect("write training data readiness report");
}

pub fn run_training_data_readiness_preflight(
    config: TrainingDataReadinessConfig,
) -> TrainingDataReadinessSummary {
    fs::create_dir_all(&config.out).expect("create training data readiness output directory");
    let progress = config.out.join("training_data_readiness_progress.jsonl");
    let _ = fs::remove_file(&progress);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"start\",\"lanes\":{},\"rounds_per_lane\":{},\"contract\":\"E79 training data curriculum readiness\"}}",
            now_millis(),
            config.lanes,
            config.rounds_per_lane
        ),
    );

    let started = Instant::now();
    let lessons = canonical_training_lessons();
    write_curriculum_manifest(&config.out, &lessons);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"curriculum_manifest_written\",\"lesson_count\":{}}}",
            now_millis(),
            lessons.len()
        ),
    );

    let evaluation = evaluate_readiness(config.lanes, config.rounds_per_lane, &lessons);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"contract_evaluated\",\"missing_family_split_count\":{},\"missing_candidate_capability_count\":{},\"invalid_row_count\":{}}}",
            now_millis(),
            evaluation.missing_family_split_count,
            evaluation.missing_candidate_capability_count,
            evaluation.invalid_row_count
        ),
    );

    let passed = config.lanes > 0
        && config.rounds_per_lane >= MIN_ROUNDS_FOR_FULL_CANDIDATE_ROTATION
        && evaluation.lesson_count == evaluation.required_lesson_count
        && evaluation.split_count == REQUIRED_SPLITS.len()
        && evaluation.family_count == REQUIRED_FAMILIES.len()
        && evaluation.capability_count == REQUIRED_CAPABILITY_SIGNATURES.len()
        && evaluation.duplicate_lesson_id_count == 0
        && evaluation.missing_family_split_count == 0
        && evaluation.missing_candidate_capability_count == 0
        && evaluation.invalid_row_count == 0
        && evaluation.score_contract_complete
        && evaluation.inference_contract_complete;
    let summary = TrainingDataReadinessSummary {
        passed,
        lanes: config.lanes,
        rounds_per_lane: config.rounds_per_lane,
        lesson_count: evaluation.lesson_count,
        required_lesson_count: evaluation.required_lesson_count,
        split_count: evaluation.split_count,
        family_count: evaluation.family_count,
        capability_count: evaluation.capability_count,
        candidate_unique_count: evaluation.candidate_unique_count,
        duplicate_lesson_id_count: evaluation.duplicate_lesson_id_count,
        missing_family_split_count: evaluation.missing_family_split_count,
        missing_candidate_capability_count: evaluation.missing_candidate_capability_count,
        invalid_row_count: evaluation.invalid_row_count,
        score_contract_complete: evaluation.score_contract_complete,
        inference_contract_complete: evaluation.inference_contract_complete,
        curriculum_digest: evaluation.curriculum_digest,
        seconds: started.elapsed().as_secs_f64(),
        out: config.out.clone(),
    };

    write_manifest(&summary.out, &config);
    write_results(&summary.out, &summary);
    write_report(&summary.out, &summary);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"complete\",\"passed\":{},\"curriculum_digest\":{},\"seconds\":{:.9}}}",
            now_millis(),
            summary.passed,
            summary.curriculum_digest,
            summary.seconds
        ),
    );
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_data_readiness_contract_passes_for_full_rotation() {
        let out = std::env::temp_dir().join(format!(
            "vraxion_e79_training_data_readiness_test_{}",
            now_millis()
        ));
        let summary = run_training_data_readiness_preflight(TrainingDataReadinessConfig::new(
            3,
            8,
            out.clone(),
        ));

        assert!(summary.passed);
        assert_eq!(summary.lesson_count, 24);
        assert_eq!(summary.required_lesson_count, 24);
        assert_eq!(summary.split_count, 3);
        assert_eq!(summary.family_count, 8);
        assert_eq!(summary.capability_count, 8);
        assert_eq!(summary.candidate_unique_count, 16);
        assert_eq!(summary.missing_family_split_count, 0);
        assert_eq!(summary.missing_candidate_capability_count, 0);
        assert!(summary.score_contract_complete);
        assert!(summary.inference_contract_complete);
        assert!(out.join("training_data_readiness_results.json").exists());
        assert!(out.join("training_data_readiness_manifest.json").exists());
        assert!(out.join("training_data_readiness_progress.jsonl").exists());
        assert!(out.join("training_curriculum_manifest.jsonl").exists());

        let _ = fs::remove_dir_all(out);
    }

    #[test]
    fn training_data_readiness_blocks_missing_split_and_short_rotation() {
        let mut lessons = canonical_training_lessons();
        lessons.retain(|lesson| {
            !(lesson.family == "frame_integrity" && lesson.split == "adversarial")
        });
        let evaluation = evaluate_readiness(2, 2, &lessons);

        assert_eq!(evaluation.missing_family_split_count, 1);
        assert!(evaluation.missing_candidate_capability_count > 0);
        assert!(!evaluation.score_contract_complete);
    }
}
