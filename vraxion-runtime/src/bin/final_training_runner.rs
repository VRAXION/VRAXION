use std::env;
use std::path::PathBuf;

use vraxion_runtime::{run_final_curriculum_pocket_generation, FinalTrainingConfig};

fn main() {
    let rounds = env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(100_000);
    let out = env::args().nth(2).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("target/pilot_wave/e75_rust_final_curriculum_pocket_generation_runner")
    });
    let resume = env::args().any(|arg| arg == "--resume");
    let preflight_rounds = env::args()
        .position(|arg| arg == "--preflight-rounds")
        .and_then(|idx| env::args().nth(idx + 1))
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(10_000);
    let checkpoint_interval = env::args()
        .position(|arg| arg == "--checkpoint-interval")
        .and_then(|idx| env::args().nth(idx + 1))
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1_000);

    let mut config = FinalTrainingConfig::new(rounds, out);
    config.resume_from_checkpoint = resume;
    config.preflight_rounds = preflight_rounds;
    config.checkpoint_interval = checkpoint_interval;

    let summary = run_final_curriculum_pocket_generation(config);
    println!(
        "{{\"passed\":{},\"rounds\":{},\"completed_queues\":{},\"completed_lessons\":{},\"promoted_count\":{},\"generated_pocket_count\":{},\"bad_commit_rate\":{:.6},\"unsafe_promotion_rate\":{:.6},\"preflight_gate_passed\":{},\"checkpoint_written\":{},\"seconds\":{:.9},\"out\":\"{}\"}}",
        summary.passed,
        summary.rounds,
        summary.completed_queues,
        summary.completed_lessons,
        summary.promoted_count,
        summary.generated_pocket_count,
        summary.bad_commit_rate,
        summary.unsafe_promotion_rate,
        summary.preflight_gate_passed,
        summary.checkpoint_written,
        summary.seconds,
        summary.out.display()
    );
}
