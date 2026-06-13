use std::env;
use std::path::PathBuf;

use vraxion_runtime::run_final_bake_preflight;

fn main() {
    let rounds = env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(100_000);
    let out = env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/pilot_wave/e73_rust_final_bake_preflight"));

    let summary = run_final_bake_preflight(rounds, out);
    println!(
        "{{\"passed\":{},\"rounds\":{},\"resume_passed\":{},\"final_checksum_match\":{},\"bad_commit_rate\":{:.6},\"unsafe_promotion_rate\":{:.6},\"seconds\":{:.9},\"out\":\"{}\"}}",
        summary.passed,
        summary.rounds,
        summary.resume_passed,
        summary.final_checksum_match,
        summary.bad_commit_rate,
        summary.unsafe_promotion_rate,
        summary.seconds,
        summary.out.display()
    );
}
