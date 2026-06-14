use std::env;
use std::path::PathBuf;

use vraxion_runtime::{run_training_data_readiness_preflight, TrainingDataReadinessConfig};

fn json_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

fn main() {
    let lanes = env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(8);
    let rounds_per_lane = env::args()
        .nth(2)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(100_000);
    let out = env::args()
        .nth(3)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/pilot_wave/e79_training_data_readiness"));

    let summary = run_training_data_readiness_preflight(TrainingDataReadinessConfig::new(
        lanes,
        rounds_per_lane,
        out,
    ));

    println!(
        "{{\"passed\":{},\"lanes\":{},\"rounds_per_lane\":{},\"lesson_count\":{},\"required_lesson_count\":{},\"split_count\":{},\"family_count\":{},\"capability_count\":{},\"candidate_unique_count\":{},\"missing_family_split_count\":{},\"missing_candidate_capability_count\":{},\"invalid_row_count\":{},\"score_contract_complete\":{},\"inference_contract_complete\":{},\"curriculum_digest\":{},\"seconds\":{:.9},\"out\":\"{}\"}}",
        summary.passed,
        summary.lanes,
        summary.rounds_per_lane,
        summary.lesson_count,
        summary.required_lesson_count,
        summary.split_count,
        summary.family_count,
        summary.capability_count,
        summary.candidate_unique_count,
        summary.missing_family_split_count,
        summary.missing_candidate_capability_count,
        summary.invalid_row_count,
        summary.score_contract_complete,
        summary.inference_contract_complete,
        summary.curriculum_digest,
        summary.seconds,
        json_escape(&summary.out.display().to_string())
    );
}
