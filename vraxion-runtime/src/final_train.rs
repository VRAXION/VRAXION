use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::{
    run_global_library_supervisor, GlobalLibrarySupervisorConfig, GlobalLibrarySupervisorSummary,
};

#[derive(Debug, Clone, PartialEq)]
pub struct FinalTrainConfig {
    pub lanes: usize,
    pub rounds_per_lane: u64,
    pub out: PathBuf,
    pub preflight_rounds: u64,
    pub checkpoint_interval: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FinalTrainSummary {
    pub passed: bool,
    pub lanes: usize,
    pub rounds_per_lane: u64,
    pub total_rounds: u64,
    pub global_generated_pocket_count: usize,
    pub duplicate_candidates_blocked: u64,
    pub failed_promotions: u64,
    pub redundant_clone_block_rate: f64,
    pub bad_commit_rate: f64,
    pub unsafe_promotion_rate: f64,
    pub seconds: f64,
    pub out: PathBuf,
    pub global_supervisor_out: PathBuf,
}

impl FinalTrainConfig {
    pub fn new(lanes: usize, rounds_per_lane: u64, out: PathBuf) -> Self {
        Self {
            lanes,
            rounds_per_lane,
            out,
            preflight_rounds: 1_000,
            checkpoint_interval: 1_000,
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
        fs::create_dir_all(parent).expect("create final train progress parent");
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open final train progress file");
    writeln!(file, "{line}").expect("write final train progress line");
}

fn json_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

fn write_final_results(
    out: &Path,
    summary: &FinalTrainSummary,
    global: &GlobalLibrarySupervisorSummary,
) {
    fs::write(
        out.join("final_train_results.json"),
        format!(
            concat!(
                "{{\n",
                "  \"passed\": {},\n",
                "  \"lanes\": {},\n",
                "  \"rounds_per_lane\": {},\n",
                "  \"total_rounds\": {},\n",
                "  \"global_generated_pocket_count\": {},\n",
                "  \"promoted_to_global\": {},\n",
                "  \"duplicate_candidates_blocked\": {},\n",
                "  \"failed_promotions\": {},\n",
                "  \"redundant_clone_block_rate\": {:.6},\n",
                "  \"bad_commit_rate\": {:.6},\n",
                "  \"unsafe_promotion_rate\": {:.6},\n",
                "  \"seconds\": {:.9},\n",
                "  \"global_supervisor_out\": \"{}\"\n",
                "}}\n"
            ),
            summary.passed,
            summary.lanes,
            summary.rounds_per_lane,
            summary.total_rounds,
            summary.global_generated_pocket_count,
            global.promoted_to_global,
            summary.duplicate_candidates_blocked,
            summary.failed_promotions,
            summary.redundant_clone_block_rate,
            summary.bad_commit_rate,
            summary.unsafe_promotion_rate,
            summary.seconds,
            json_escape(&summary.global_supervisor_out.display().to_string())
        ),
    )
    .expect("write final train results");
}

fn write_manifest(out: &Path, summary: &FinalTrainSummary) {
    fs::write(
        out.join("final_train_manifest.json"),
        format!(
            concat!(
                "{{\n",
                "  \"canonical_entrypoint\": \"final_train\",\n",
                "  \"runtime_surface\": \"vraxion-runtime\",\n",
                "  \"artifact_contract\": \"E78_FINAL_TRAIN_CAMPAIGN_ENTRYPOINT\",\n",
                "  \"lanes\": {},\n",
                "  \"rounds_per_lane\": {},\n",
                "  \"global_supervisor_out\": \"{}\",\n",
                "  \"required_artifacts\": [\n",
                "    \"final_train_results.json\",\n",
                "    \"final_train_manifest.json\",\n",
                "    \"final_train_progress.jsonl\",\n",
                "    \"final_train_report.md\",\n",
                "    \"global_supervisor/global_merge_results.json\",\n",
                "    \"global_supervisor/global_library_summary.json\",\n",
                "    \"global_supervisor/lane_supervisor/supervisor_results.json\"\n",
                "  ],\n",
                "  \"boundary\": \"controlled Rust runtime final-training campaign entrypoint; no AGI/model-scale claim\"\n",
                "}}\n"
            ),
            summary.lanes,
            summary.rounds_per_lane,
            json_escape(&summary.global_supervisor_out.display().to_string())
        ),
    )
    .expect("write final train manifest");
}

fn write_report(out: &Path, summary: &FinalTrainSummary) {
    fs::write(
        out.join("final_train_report.md"),
        format!(
            "# E78 Final Train Campaign Entrypoint\n\n\
             ```text\n\
             passed = {}\n\
             lanes = {}\n\
             rounds_per_lane = {}\n\
             total_rounds = {}\n\
             global_generated_pocket_count = {}\n\
             duplicate_candidates_blocked = {}\n\
             failed_promotions = {}\n\
             redundant_clone_block_rate = {:.6}\n\
             bad_commit_rate = {:.6}\n\
             unsafe_promotion_rate = {:.6}\n\
             global_supervisor_out = {}\n\
             ```\n",
            summary.passed,
            summary.lanes,
            summary.rounds_per_lane,
            summary.total_rounds,
            summary.global_generated_pocket_count,
            summary.duplicate_candidates_blocked,
            summary.failed_promotions,
            summary.redundant_clone_block_rate,
            summary.bad_commit_rate,
            summary.unsafe_promotion_rate,
            summary.global_supervisor_out.display()
        ),
    )
    .expect("write final train report");
}

pub fn run_final_train(config: FinalTrainConfig) -> FinalTrainSummary {
    fs::create_dir_all(&config.out).expect("create final train output directory");
    let progress = config.out.join("final_train_progress.jsonl");
    let _ = fs::remove_file(&progress);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"start\",\"lanes\":{},\"rounds_per_lane\":{},\"preflight_rounds\":{},\"checkpoint_interval\":{}}}",
            now_millis(),
            config.lanes,
            config.rounds_per_lane,
            config.preflight_rounds,
            config.checkpoint_interval
        ),
    );

    let started = Instant::now();
    let global_supervisor_out = config.out.join("global_supervisor");
    let mut global_config = GlobalLibrarySupervisorConfig::new(
        config.lanes,
        config.rounds_per_lane,
        global_supervisor_out.clone(),
    );
    global_config.preflight_rounds = config.preflight_rounds;
    global_config.checkpoint_interval = config.checkpoint_interval;
    let global = run_global_library_supervisor(global_config);

    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"global_supervisor_complete\",\"passed\":{},\"global_generated_pocket_count\":{},\"duplicate_candidates_blocked\":{},\"failed_promotions\":{}}}",
            now_millis(),
            global.passed,
            global.global_generated_pocket_count,
            global.duplicate_candidates_blocked,
            global.failed_promotions
        ),
    );

    let total_rounds = global.lanes as u64 * global.rounds_per_lane;
    let seconds = started.elapsed().as_secs_f64();
    let passed = global.passed
        && total_rounds > 0
        && global.global_generated_pocket_count > 0
        && global.failed_promotions == 0
        && global.redundant_clone_block_rate == 1.0
        && global.bad_commit_rate == 0.0
        && global.unsafe_promotion_rate == 0.0;

    let summary = FinalTrainSummary {
        passed,
        lanes: global.lanes,
        rounds_per_lane: global.rounds_per_lane,
        total_rounds,
        global_generated_pocket_count: global.global_generated_pocket_count,
        duplicate_candidates_blocked: global.duplicate_candidates_blocked,
        failed_promotions: global.failed_promotions,
        redundant_clone_block_rate: global.redundant_clone_block_rate,
        bad_commit_rate: global.bad_commit_rate,
        unsafe_promotion_rate: global.unsafe_promotion_rate,
        seconds,
        out: config.out,
        global_supervisor_out,
    };

    write_final_results(&summary.out, &summary, &global);
    write_manifest(&summary.out, &summary);
    write_report(&summary.out, &summary);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"complete\",\"passed\":{},\"total_rounds\":{},\"seconds\":{:.9}}}",
            now_millis(),
            summary.passed,
            summary.total_rounds,
            summary.seconds
        ),
    );
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_train_entrypoint_runs_global_supervisor_and_manifest() {
        let out =
            std::env::temp_dir().join(format!("vraxion_e78_final_train_test_{}", now_millis()));
        let mut config = FinalTrainConfig::new(3, 4, out.clone());
        config.preflight_rounds = 4;
        config.checkpoint_interval = 2;

        let summary = run_final_train(config);

        assert!(summary.passed);
        assert_eq!(summary.lanes, 3);
        assert_eq!(summary.total_rounds, 12);
        assert_eq!(summary.global_generated_pocket_count, 16);
        assert_eq!(summary.duplicate_candidates_blocked, 32);
        assert!(out.join("final_train_results.json").exists());
        assert!(out.join("final_train_manifest.json").exists());
        assert!(out.join("final_train_progress.jsonl").exists());
        assert!(out
            .join("global_supervisor")
            .join("global_merge_results.json")
            .exists());

        let _ = fs::remove_dir_all(out);
    }
}
