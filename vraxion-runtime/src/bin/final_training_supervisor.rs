use std::env;
use std::path::PathBuf;

use vraxion_runtime::{run_final_training_supervisor, FinalTrainingSupervisorConfig};

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
    let out = env::args().nth(3).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("target/pilot_wave/e76_rust_final_training_multilane_supervisor")
    });
    let preflight_rounds = arg_after("--preflight-rounds")
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1_000);
    let checkpoint_interval = arg_after("--checkpoint-interval")
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1_000);

    let mut config = FinalTrainingSupervisorConfig::new(lanes, rounds_per_lane, out);
    config.preflight_rounds = preflight_rounds;
    config.checkpoint_interval = checkpoint_interval;

    let summary = run_final_training_supervisor(config);
    println!(
        "{{\"passed\":{},\"lanes\":{},\"rounds_per_lane\":{},\"total_rounds\":{},\"completed_queues\":{},\"completed_lessons\":{},\"promoted_count\":{},\"lane_generated_pocket_count_sum\":{},\"bad_commit_rate\":{:.6},\"unsafe_promotion_rate\":{:.6},\"seconds\":{:.9},\"out\":\"{}\"}}",
        summary.passed,
        summary.lanes,
        summary.rounds_per_lane,
        summary.total_rounds,
        summary.completed_queues,
        summary.completed_lessons,
        summary.promoted_count,
        summary.lane_generated_pocket_count_sum,
        summary.bad_commit_rate,
        summary.unsafe_promotion_rate,
        summary.seconds,
        summary.out.display()
    );
}
