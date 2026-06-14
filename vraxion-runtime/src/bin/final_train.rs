use std::env;
use std::path::PathBuf;

use vraxion_runtime::{run_final_train, FinalTrainConfig};

fn json_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

fn arg_after(flag: &str) -> Option<String> {
    env::args()
        .position(|arg| arg == flag)
        .and_then(|idx| env::args().nth(idx + 1))
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
        .unwrap_or_else(|| PathBuf::from("target/pilot_wave/e78_final_train_campaign_entrypoint"));
    let preflight_rounds = arg_after("--preflight-rounds")
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1_000);
    let checkpoint_interval = arg_after("--checkpoint-interval")
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1_000);

    let mut config = FinalTrainConfig::new(lanes, rounds_per_lane, out);
    config.preflight_rounds = preflight_rounds;
    config.checkpoint_interval = checkpoint_interval;

    let summary = run_final_train(config);
    println!(
        "{{\"passed\":{},\"lanes\":{},\"rounds_per_lane\":{},\"total_rounds\":{},\"training_data_readiness_passed\":{},\"training_data_lesson_count\":{},\"training_data_capability_count\":{},\"training_data_curriculum_digest\":{},\"global_generated_pocket_count\":{},\"promoted_to_global\":{},\"duplicate_candidates_blocked\":{},\"failed_promotions\":{},\"redundant_clone_block_rate\":{:.6},\"bad_commit_rate\":{:.6},\"unsafe_promotion_rate\":{:.6},\"seconds\":{:.9},\"out\":\"{}\",\"training_data_readiness_out\":\"{}\",\"global_supervisor_out\":\"{}\"}}",
        summary.passed,
        summary.lanes,
        summary.rounds_per_lane,
        summary.total_rounds,
        summary.training_data_readiness_passed,
        summary.training_data_lesson_count,
        summary.training_data_capability_count,
        summary.training_data_curriculum_digest,
        summary.global_generated_pocket_count,
        summary.promoted_to_global,
        summary.duplicate_candidates_blocked,
        summary.failed_promotions,
        summary.redundant_clone_block_rate,
        summary.bad_commit_rate,
        summary.unsafe_promotion_rate,
        summary.seconds,
        json_escape(&summary.out.display().to_string()),
        json_escape(&summary.training_data_readiness_out.display().to_string()),
        json_escape(&summary.global_supervisor_out.display().to_string())
    );
}
