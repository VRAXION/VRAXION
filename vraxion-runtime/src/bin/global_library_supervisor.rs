use std::env;
use std::path::PathBuf;

use vraxion_runtime::{run_global_library_supervisor, GlobalLibrarySupervisorConfig};

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
        PathBuf::from("target/pilot_wave/e77_global_pocket_library_merge_supervisor")
    });
    let preflight_rounds = arg_after("--preflight-rounds")
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1_000);
    let checkpoint_interval = arg_after("--checkpoint-interval")
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1_000);

    let mut config = GlobalLibrarySupervisorConfig::new(lanes, rounds_per_lane, out);
    config.preflight_rounds = preflight_rounds;
    config.checkpoint_interval = checkpoint_interval;

    let summary = run_global_library_supervisor(config);
    println!(
        "{{\"passed\":{},\"lanes\":{},\"rounds_per_lane\":{},\"total_local_candidates\":{},\"unique_candidates\":{},\"promoted_to_global\":{},\"duplicate_candidates_blocked\":{},\"failed_promotions\":{},\"global_generated_pocket_count\":{},\"redundant_clone_block_rate\":{:.6},\"bad_commit_rate\":{:.6},\"unsafe_promotion_rate\":{:.6},\"seconds\":{:.9},\"out\":\"{}\"}}",
        summary.passed,
        summary.lanes,
        summary.rounds_per_lane,
        summary.total_local_candidates,
        summary.unique_candidates,
        summary.promoted_to_global,
        summary.duplicate_candidates_blocked,
        summary.failed_promotions,
        summary.global_generated_pocket_count,
        summary.redundant_clone_block_rate,
        summary.bad_commit_rate,
        summary.unsafe_promotion_rate,
        summary.seconds,
        summary.out.display()
    );
}
