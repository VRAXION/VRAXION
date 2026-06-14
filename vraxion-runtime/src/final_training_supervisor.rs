use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::{run_final_curriculum_pocket_generation, FinalTrainingConfig, FinalTrainingSummary};

#[derive(Debug, Clone, PartialEq)]
pub struct FinalTrainingSupervisorConfig {
    pub lanes: usize,
    pub rounds_per_lane: u64,
    pub out: PathBuf,
    pub preflight_rounds: u64,
    pub checkpoint_interval: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FinalTrainingSupervisorSummary {
    pub passed: bool,
    pub lanes: usize,
    pub rounds_per_lane: u64,
    pub total_rounds: u64,
    pub completed_queues: u64,
    pub completed_lessons: u64,
    pub promoted_count: u64,
    pub failed_count: u64,
    pub lane_generated_pocket_count_sum: usize,
    pub bad_commit_rate: f64,
    pub unsafe_promotion_rate: f64,
    pub seconds: f64,
    pub out: PathBuf,
}

impl FinalTrainingSupervisorConfig {
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
        fs::create_dir_all(parent).expect("create supervisor progress parent");
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open supervisor progress file");
    writeln!(file, "{line}").expect("write supervisor progress line");
}

fn write_partial_aggregate(
    out: &Path,
    lane_count: usize,
    completed_lanes: usize,
    summaries: &[FinalTrainingSummary],
    seconds: f64,
) {
    let completed_queues: u64 = summaries
        .iter()
        .map(|summary| summary.completed_queues)
        .sum();
    let completed_lessons: u64 = summaries
        .iter()
        .map(|summary| summary.completed_lessons)
        .sum();
    let promoted_count: u64 = summaries.iter().map(|summary| summary.promoted_count).sum();
    let failed_count: u64 = summaries.iter().map(|summary| summary.failed_count).sum();
    let generated_sum: usize = summaries
        .iter()
        .map(|summary| summary.generated_pocket_count)
        .sum();
    fs::write(
        out.join("partial_aggregate_snapshot.json"),
        format!(
            concat!(
                "{{\n",
                "  \"lane_count\": {},\n",
                "  \"completed_lanes\": {},\n",
                "  \"completed_queues\": {},\n",
                "  \"completed_lessons\": {},\n",
                "  \"promoted_count\": {},\n",
                "  \"failed_count\": {},\n",
                "  \"lane_generated_pocket_count_sum\": {},\n",
                "  \"seconds\": {:.9}\n",
                "}}\n"
            ),
            lane_count,
            completed_lanes,
            completed_queues,
            completed_lessons,
            promoted_count,
            failed_count,
            generated_sum,
            seconds
        ),
    )
    .expect("write partial aggregate");
}

pub fn run_final_training_supervisor(
    config: FinalTrainingSupervisorConfig,
) -> FinalTrainingSupervisorSummary {
    fs::create_dir_all(&config.out).expect("create supervisor output directory");
    let progress = config.out.join("progress.jsonl");
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
    let lanes = config.lanes.max(1);
    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::with_capacity(lanes);

    for lane_index in 0..lanes {
        let tx = tx.clone();
        let lane_out = config.out.join(format!("lane_{lane_index:02}"));
        let mut lane_config = FinalTrainingConfig::new(config.rounds_per_lane, lane_out);
        lane_config.preflight_rounds = config.preflight_rounds;
        lane_config.checkpoint_interval = config.checkpoint_interval;
        handles.push(thread::spawn(move || {
            let summary = run_final_curriculum_pocket_generation(lane_config);
            tx.send((lane_index, summary)).expect("send lane summary");
        }));
    }
    drop(tx);

    let mut summaries: Vec<Option<FinalTrainingSummary>> = vec![None; lanes];
    let mut completed_lanes = 0usize;
    for (lane_index, summary) in rx {
        append_jsonl(
            &progress,
            &format!(
                "{{\"timestamp_ms\":{},\"event\":\"lane_complete\",\"lane\":{},\"passed\":{},\"completed_queues\":{},\"completed_lessons\":{},\"generated_pocket_count\":{},\"seconds\":{:.9}}}",
                now_millis(),
                lane_index,
                summary.passed,
                summary.completed_queues,
                summary.completed_lessons,
                summary.generated_pocket_count,
                summary.seconds
            ),
        );
        summaries[lane_index] = Some(summary);
        completed_lanes += 1;
        let complete: Vec<FinalTrainingSummary> =
            summaries.iter().filter_map(Clone::clone).collect();
        write_partial_aggregate(
            &config.out,
            lanes,
            completed_lanes,
            &complete,
            started.elapsed().as_secs_f64(),
        );
    }

    for handle in handles {
        handle.join().expect("join final training lane");
    }

    let complete: Vec<FinalTrainingSummary> = summaries
        .into_iter()
        .map(|summary| summary.expect("lane summary present"))
        .collect();
    let completed_queues: u64 = complete
        .iter()
        .map(|summary| summary.completed_queues)
        .sum();
    let completed_lessons: u64 = complete
        .iter()
        .map(|summary| summary.completed_lessons)
        .sum();
    let promoted_count: u64 = complete.iter().map(|summary| summary.promoted_count).sum();
    let failed_count: u64 = complete.iter().map(|summary| summary.failed_count).sum();
    let lane_generated_pocket_count_sum: usize = complete
        .iter()
        .map(|summary| summary.generated_pocket_count)
        .sum();
    let bad_commit_count = complete
        .iter()
        .filter(|summary| summary.bad_commit_rate > 0.0)
        .count() as u64;
    let unsafe_promotion_count = complete
        .iter()
        .filter(|summary| summary.unsafe_promotion_rate > 0.0)
        .count() as u64;
    let bad_commit_rate = bad_commit_count as f64 / lanes as f64;
    let unsafe_promotion_rate = unsafe_promotion_count as f64 / lanes as f64;
    let seconds = started.elapsed().as_secs_f64();
    let total_rounds = config.rounds_per_lane * lanes as u64;
    let passed = complete.iter().all(|summary| summary.passed)
        && completed_queues == total_rounds
        && failed_count == 0
        && bad_commit_rate == 0.0
        && unsafe_promotion_rate == 0.0;

    fs::write(
        config.out.join("supervisor_results.json"),
        format!(
            concat!(
                "{{\n",
                "  \"passed\": {},\n",
                "  \"lanes\": {},\n",
                "  \"rounds_per_lane\": {},\n",
                "  \"total_rounds\": {},\n",
                "  \"completed_queues\": {},\n",
                "  \"completed_lessons\": {},\n",
                "  \"promoted_count\": {},\n",
                "  \"failed_count\": {},\n",
                "  \"lane_generated_pocket_count_sum\": {},\n",
                "  \"bad_commit_rate\": {:.6},\n",
                "  \"unsafe_promotion_rate\": {:.6},\n",
                "  \"seconds\": {:.9},\n",
                "  \"queues_per_sec\": {:.3},\n",
                "  \"lessons_per_sec\": {:.3}\n",
                "}}\n"
            ),
            passed,
            lanes,
            config.rounds_per_lane,
            total_rounds,
            completed_queues,
            completed_lessons,
            promoted_count,
            failed_count,
            lane_generated_pocket_count_sum,
            bad_commit_rate,
            unsafe_promotion_rate,
            seconds,
            completed_queues as f64 / seconds.max(0.000_001),
            completed_lessons as f64 / seconds.max(0.000_001)
        ),
    )
    .expect("write supervisor results");

    fs::write(
        config.out.join("report.md"),
        format!(
            "# E76 Rust Final Training Multi-Lane Supervisor\n\n\
             ```text\n\
             passed = {}\n\
             lanes = {}\n\
             rounds_per_lane = {}\n\
             completed_queues = {}\n\
             completed_lessons = {}\n\
             promoted_count = {}\n\
             lane_generated_pocket_count_sum = {}\n\
             bad_commit_rate = {:.6}\n\
             unsafe_promotion_rate = {:.6}\n\
             ```\n",
            passed,
            lanes,
            config.rounds_per_lane,
            completed_queues,
            completed_lessons,
            promoted_count,
            lane_generated_pocket_count_sum,
            bad_commit_rate,
            unsafe_promotion_rate
        ),
    )
    .expect("write supervisor report");

    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"complete\",\"passed\":{},\"lanes\":{},\"completed_queues\":{},\"completed_lessons\":{}}}",
            now_millis(),
            passed,
            lanes,
            completed_queues,
            completed_lessons
        ),
    );

    FinalTrainingSupervisorSummary {
        passed,
        lanes,
        rounds_per_lane: config.rounds_per_lane,
        total_rounds,
        completed_queues,
        completed_lessons,
        promoted_count,
        failed_count,
        lane_generated_pocket_count_sum,
        bad_commit_rate,
        unsafe_promotion_rate,
        seconds,
        out: config.out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supervisor_runs_parallel_lanes_and_writes_aggregate() {
        let out =
            std::env::temp_dir().join(format!("vraxion_e76_supervisor_test_{}", now_millis()));
        let mut config = FinalTrainingSupervisorConfig::new(3, 8, out.clone());
        config.preflight_rounds = 4;
        config.checkpoint_interval = 4;

        let summary = run_final_training_supervisor(config);

        assert!(summary.passed);
        assert_eq!(summary.lanes, 3);
        assert_eq!(summary.completed_queues, 24);
        assert_eq!(summary.completed_lessons, 96);
        assert_eq!(summary.failed_count, 0);
        assert!(summary.lane_generated_pocket_count_sum > 0);
        assert!(out.join("supervisor_results.json").exists());
        assert!(out.join("partial_aggregate_snapshot.json").exists());
        assert!(out.join("lane_00").join("checkpoint_latest.json").exists());

        let _ = fs::remove_dir_all(out);
    }
}
