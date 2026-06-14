use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::{
    final_bake::{append_jsonl, base_store, lifecycle, now_millis, promotion},
    CurriculumCheckpoint, CurriculumLesson, CurriculumQueueLesson, PocketLibraryStore,
    RustCurriculumRunner, StorePromotionCandidate,
};

#[derive(Debug, Clone, PartialEq)]
pub struct FinalTrainingConfig {
    pub rounds: u64,
    pub out: PathBuf,
    pub preflight_rounds: u64,
    pub checkpoint_interval: u64,
    pub heartbeat_seconds: u64,
    pub resume_from_checkpoint: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FinalTrainingSummary {
    pub passed: bool,
    pub rounds: u64,
    pub completed_queues: u64,
    pub completed_lessons: u64,
    pub promoted_count: u64,
    pub failed_count: u64,
    pub generated_pocket_count: usize,
    pub bad_commit_rate: f64,
    pub unsafe_promotion_rate: f64,
    pub preflight_gate_passed: bool,
    pub checkpoint_written: bool,
    pub seconds: f64,
    pub out: PathBuf,
}

const TRAINING_CANDIDATES: [StorePromotionCandidate; 16] = [
    StorePromotionCandidate {
        pocket_uid: "train_frame_guard_a",
        human_alias: "train_frame_guard_a",
        content_digest: "digest_train_frame_guard_a",
        token_hash: "tok_train_frame_guard_a",
        capability_signature: "binary_training_frame_guard",
        quality_delta: 0.021,
    },
    StorePromotionCandidate {
        pocket_uid: "train_frame_guard_b",
        human_alias: "train_frame_guard_b",
        content_digest: "digest_train_frame_guard_b",
        token_hash: "tok_train_frame_guard_b",
        capability_signature: "binary_training_frame_guard",
        quality_delta: 0.022,
    },
    StorePromotionCandidate {
        pocket_uid: "train_request_guard_a",
        human_alias: "train_request_guard_a",
        content_digest: "digest_train_request_guard_a",
        token_hash: "tok_train_request_guard_a",
        capability_signature: "binary_training_request_guard",
        quality_delta: 0.023,
    },
    StorePromotionCandidate {
        pocket_uid: "train_request_guard_b",
        human_alias: "train_request_guard_b",
        content_digest: "digest_train_request_guard_b",
        token_hash: "tok_train_request_guard_b",
        capability_signature: "binary_training_request_guard",
        quality_delta: 0.024,
    },
    StorePromotionCandidate {
        pocket_uid: "train_trace_guard_a",
        human_alias: "train_trace_guard_a",
        content_digest: "digest_train_trace_guard_a",
        token_hash: "tok_train_trace_guard_a",
        capability_signature: "binary_training_trace_guard",
        quality_delta: 0.025,
    },
    StorePromotionCandidate {
        pocket_uid: "train_trace_guard_b",
        human_alias: "train_trace_guard_b",
        content_digest: "digest_train_trace_guard_b",
        token_hash: "tok_train_trace_guard_b",
        capability_signature: "binary_training_trace_guard",
        quality_delta: 0.026,
    },
    StorePromotionCandidate {
        pocket_uid: "train_reload_guard_a",
        human_alias: "train_reload_guard_a",
        content_digest: "digest_train_reload_guard_a",
        token_hash: "tok_train_reload_guard_a",
        capability_signature: "binary_training_reload_guard",
        quality_delta: 0.027,
    },
    StorePromotionCandidate {
        pocket_uid: "train_reload_guard_b",
        human_alias: "train_reload_guard_b",
        content_digest: "digest_train_reload_guard_b",
        token_hash: "tok_train_reload_guard_b",
        capability_signature: "binary_training_reload_guard",
        quality_delta: 0.028,
    },
    StorePromotionCandidate {
        pocket_uid: "train_bitslip_guard_a",
        human_alias: "train_bitslip_guard_a",
        content_digest: "digest_train_bitslip_guard_a",
        token_hash: "tok_train_bitslip_guard_a",
        capability_signature: "binary_training_bitslip_guard",
        quality_delta: 0.029,
    },
    StorePromotionCandidate {
        pocket_uid: "train_bitslip_guard_b",
        human_alias: "train_bitslip_guard_b",
        content_digest: "digest_train_bitslip_guard_b",
        token_hash: "tok_train_bitslip_guard_b",
        capability_signature: "binary_training_bitslip_guard",
        quality_delta: 0.030,
    },
    StorePromotionCandidate {
        pocket_uid: "train_text_mode_guard_a",
        human_alias: "train_text_mode_guard_a",
        content_digest: "digest_train_text_mode_guard_a",
        token_hash: "tok_train_text_mode_guard_a",
        capability_signature: "text_training_mode_guard",
        quality_delta: 0.031,
    },
    StorePromotionCandidate {
        pocket_uid: "train_text_mode_guard_b",
        human_alias: "train_text_mode_guard_b",
        content_digest: "digest_train_text_mode_guard_b",
        token_hash: "tok_train_text_mode_guard_b",
        capability_signature: "text_training_mode_guard",
        quality_delta: 0.032,
    },
    StorePromotionCandidate {
        pocket_uid: "train_agency_guard_a",
        human_alias: "train_agency_guard_a",
        content_digest: "digest_train_agency_guard_a",
        token_hash: "tok_train_agency_guard_a",
        capability_signature: "agency_training_commit_guard",
        quality_delta: 0.033,
    },
    StorePromotionCandidate {
        pocket_uid: "train_agency_guard_b",
        human_alias: "train_agency_guard_b",
        content_digest: "digest_train_agency_guard_b",
        token_hash: "tok_train_agency_guard_b",
        capability_signature: "agency_training_commit_guard",
        quality_delta: 0.034,
    },
    StorePromotionCandidate {
        pocket_uid: "train_library_guard_a",
        human_alias: "train_library_guard_a",
        content_digest: "digest_train_library_guard_a",
        token_hash: "tok_train_library_guard_a",
        capability_signature: "library_training_reload_guard",
        quality_delta: 0.035,
    },
    StorePromotionCandidate {
        pocket_uid: "train_library_guard_b",
        human_alias: "train_library_guard_b",
        content_digest: "digest_train_library_guard_b",
        token_hash: "tok_train_library_guard_b",
        capability_signature: "library_training_reload_guard",
        quality_delta: 0.036,
    },
];

pub(crate) fn training_candidates_for_rounds(rounds: u64) -> Vec<StorePromotionCandidate> {
    let mut candidates = Vec::new();
    for round in 0..rounds {
        for idx in 0..4 {
            let candidate_index = ((round as usize * 4) + idx) % TRAINING_CANDIDATES.len();
            let candidate = TRAINING_CANDIDATES[candidate_index];
            if !candidates.iter().any(|existing: &StorePromotionCandidate| {
                existing.pocket_uid == candidate.pocket_uid
            }) {
                candidates.push(candidate);
            }
        }
    }
    candidates
}

impl FinalTrainingConfig {
    pub fn new(rounds: u64, out: PathBuf) -> Self {
        Self {
            rounds,
            out,
            preflight_rounds: 10_000,
            checkpoint_interval: 1_000,
            heartbeat_seconds: 20,
            resume_from_checkpoint: false,
        }
    }
}

fn training_queue_lessons(round: u64) -> [CurriculumQueueLesson; 4] {
    let families = [
        "frame_integrity",
        "requested_feature_match",
        "trace_commit",
        "library_reload",
    ];
    std::array::from_fn(|idx| {
        let candidate_index = ((round as usize * families.len()) + idx) % TRAINING_CANDIDATES.len();
        CurriculumQueueLesson {
            family: families[idx],
            lesson: CurriculumLesson {
                requested_feature: (((round * 7 + idx as u64 * 5) % 23) + 1) as u8,
                value: ((round + idx as u64) % 2) as u8,
                source_pocket_id: 75,
                nonce: (((round + 11) * (idx as u64 + 3)) % 31) as u8,
            },
            candidate: TRAINING_CANDIDATES[candidate_index],
            lifecycle_evidence: lifecycle(),
            promotion_evidence: promotion(false),
        }
    })
}

fn parse_json_u64(text: &str, key: &str) -> Option<u64> {
    let marker = format!("\"{key}\":");
    let start = text.find(&marker)? + marker.len();
    let digits: String = text[start..]
        .chars()
        .skip_while(|ch| ch.is_whitespace())
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

fn write_checkpoint(path: &PathBuf, checkpoint: CurriculumCheckpoint) {
    fs::write(
        path,
        format!(
            concat!(
                "{{\n",
                "  \"run_id\": {},\n",
                "  \"completed_queues\": {},\n",
                "  \"completed_lessons\": {},\n",
                "  \"promoted_count\": {},\n",
                "  \"failed_count\": {},\n",
                "  \"bad_commit_count\": {},\n",
                "  \"unsafe_promotion_count\": {},\n",
                "  \"store_generation\": {},\n",
                "  \"quality_delta\": {:.6},\n",
                "  \"checksum\": {}\n",
                "}}\n"
            ),
            checkpoint.run_id,
            checkpoint.completed_queues,
            checkpoint.completed_lessons,
            checkpoint.promoted_count,
            checkpoint.failed_count,
            checkpoint.bad_commit_count,
            checkpoint.unsafe_promotion_count,
            checkpoint.store_generation,
            checkpoint.quality_delta,
            checkpoint.checksum
        ),
    )
    .expect("write checkpoint");
}

fn write_partial_summary(
    out: &Path,
    checkpoint: CurriculumCheckpoint,
    store: &PocketLibraryStore,
    preflight_gate_passed: bool,
    seconds: f64,
) {
    let snapshot = store.snapshot();
    fs::write(
        out.join("partial_summary.json"),
        format!(
            concat!(
                "{{\n",
                "  \"preflight_gate_passed\": {},\n",
                "  \"completed_queues\": {},\n",
                "  \"completed_lessons\": {},\n",
                "  \"promoted_count\": {},\n",
                "  \"failed_count\": {},\n",
                "  \"registry_entry_count\": {},\n",
                "  \"token_count\": {},\n",
                "  \"artifact_count\": {},\n",
                "  \"store_generation\": {},\n",
                "  \"bad_commit_rate\": {:.6},\n",
                "  \"unsafe_promotion_rate\": {:.6},\n",
                "  \"seconds\": {:.9}\n",
                "}}\n"
            ),
            preflight_gate_passed,
            checkpoint.completed_queues,
            checkpoint.completed_lessons,
            checkpoint.promoted_count,
            checkpoint.failed_count,
            snapshot.registry_entry_count,
            snapshot.token_count,
            snapshot.artifact_count,
            snapshot.generation,
            checkpoint.bad_commit_count as f64 / checkpoint.completed_queues.max(1) as f64,
            checkpoint.unsafe_promotion_count as f64 / checkpoint.completed_lessons.max(1) as f64,
            seconds
        ),
    )
    .expect("write partial summary");
}

fn write_library_summary(out: &Path, store: &PocketLibraryStore) {
    let snapshot = store.snapshot();
    fs::write(
        out.join("library_summary.json"),
        format!(
            concat!(
                "{{\n",
                "  \"registry_entry_count\": {},\n",
                "  \"token_count\": {},\n",
                "  \"artifact_count\": {},\n",
                "  \"generation\": {},\n",
                "  \"ledger_complete\": {},\n",
                "  \"quality_delta\": {:.6}\n",
                "}}\n"
            ),
            snapshot.registry_entry_count,
            snapshot.token_count,
            snapshot.artifact_count,
            snapshot.generation,
            snapshot.ledger_complete,
            store.quality_delta()
        ),
    )
    .expect("write library summary");
}

fn checkpoint_resume_point(out: &Path) -> Option<u64> {
    let text = fs::read_to_string(out.join("checkpoint_latest.json")).ok()?;
    parse_json_u64(&text, "completed_queues")
}

struct TrainingRangeContext<'a> {
    config: &'a FinalTrainingConfig,
    preflight_gate_passed: bool,
    phase: &'a str,
    started: Instant,
}

fn run_training_range(
    runner: &mut RustCurriculumRunner,
    checkpoint: &mut CurriculumCheckpoint,
    start_queue: u64,
    end_queue: u64,
    context: TrainingRangeContext<'_>,
) {
    let config = context.config;
    let progress = config.out.join("progress.jsonl");
    let checkpoint_path = config.out.join("checkpoint_latest.json");
    let checkpoint_interval = config.checkpoint_interval.max(1);
    let heartbeat = Duration::from_secs(config.heartbeat_seconds.max(1));
    let mut last_heartbeat = Instant::now();

    for queue_index in start_queue..end_queue {
        let lessons = training_queue_lessons(queue_index);
        let report = runner.run_queue(&lessons);
        checkpoint.record_queue(queue_index, report, runner.store.generation);

        if checkpoint
            .completed_queues
            .is_multiple_of(checkpoint_interval)
            || checkpoint.completed_queues == end_queue
        {
            write_checkpoint(&checkpoint_path, *checkpoint);
            write_partial_summary(
                &config.out,
                *checkpoint,
                &runner.store,
                context.preflight_gate_passed,
                context.started.elapsed().as_secs_f64(),
            );
            append_jsonl(
                &progress,
                &format!(
                    "{{\"timestamp_ms\":{},\"event\":\"checkpoint\",\"phase\":\"{}\",\"completed_queues\":{},\"completed_lessons\":{},\"promoted_count\":{},\"store_generation\":{},\"checksum\":{}}}",
                    now_millis(),
                    context.phase,
                    checkpoint.completed_queues,
                    checkpoint.completed_lessons,
                    checkpoint.promoted_count,
                    checkpoint.store_generation,
                    checkpoint.checksum
                ),
            );
        }

        if last_heartbeat.elapsed() >= heartbeat {
            write_partial_summary(
                &config.out,
                *checkpoint,
                &runner.store,
                context.preflight_gate_passed,
                context.started.elapsed().as_secs_f64(),
            );
            append_jsonl(
                &progress,
                &format!(
                    "{{\"timestamp_ms\":{},\"event\":\"heartbeat\",\"phase\":\"{}\",\"queue_index\":{},\"target_rounds\":{},\"completed_queues\":{},\"completed_lessons\":{},\"failed_count\":{},\"bad_commit_count\":{},\"unsafe_promotion_count\":{},\"checksum\":{}}}",
                    now_millis(),
                    context.phase,
                    queue_index + 1,
                    config.rounds,
                    checkpoint.completed_queues,
                    checkpoint.completed_lessons,
                    checkpoint.failed_count,
                    checkpoint.bad_commit_count,
                    checkpoint.unsafe_promotion_count,
                    checkpoint.checksum
                ),
            );
            last_heartbeat = Instant::now();
        }
    }
}

pub fn run_final_curriculum_pocket_generation(config: FinalTrainingConfig) -> FinalTrainingSummary {
    fs::create_dir_all(&config.out).expect("create output directory");
    let progress = config.out.join("progress.jsonl");
    if !config.resume_from_checkpoint {
        let _ = fs::remove_file(&progress);
    }
    let started = Instant::now();
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"start\",\"rounds\":{},\"preflight_rounds\":{},\"checkpoint_interval\":{},\"resume\":{}}}",
            now_millis(),
            config.rounds,
            config.preflight_rounds,
            config.checkpoint_interval,
            config.resume_from_checkpoint
        ),
    );

    let preflight_rounds = config.preflight_rounds.max(1);
    let preflight =
        crate::run_final_bake_preflight(preflight_rounds, config.out.join("preflight_gate"));
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"preflight_complete\",\"passed\":{},\"rounds\":{},\"resume_passed\":{},\"checksum_match\":{}}}",
            now_millis(),
            preflight.passed,
            preflight.rounds,
            preflight.resume_passed,
            preflight.final_checksum_match
        ),
    );

    let mut runner = RustCurriculumRunner::default_body(base_store());
    let mut checkpoint = CurriculumCheckpoint::new(75);
    let resume_to = if config.resume_from_checkpoint {
        checkpoint_resume_point(&config.out)
            .unwrap_or(0)
            .min(config.rounds)
    } else {
        0
    };

    if resume_to > 0 {
        append_jsonl(
            &progress,
            &format!(
                "{{\"timestamp_ms\":{},\"event\":\"resume_replay_start\",\"queues\":{}}}",
                now_millis(),
                resume_to
            ),
        );
        run_training_range(
            &mut runner,
            &mut checkpoint,
            0,
            resume_to,
            TrainingRangeContext {
                config: &config,
                preflight_gate_passed: preflight.passed,
                phase: "resume_replay",
                started,
            },
        );
        append_jsonl(
            &progress,
            &format!(
                "{{\"timestamp_ms\":{},\"event\":\"resume_replay_complete\",\"completed_queues\":{},\"checksum\":{}}}",
                now_millis(),
                checkpoint.completed_queues,
                checkpoint.checksum
            ),
        );
    }

    run_training_range(
        &mut runner,
        &mut checkpoint,
        resume_to,
        config.rounds,
        TrainingRangeContext {
            config: &config,
            preflight_gate_passed: preflight.passed,
            phase: "final_training",
            started,
        },
    );

    let checkpoint_path = config.out.join("checkpoint_latest.json");
    write_checkpoint(&checkpoint_path, checkpoint);
    write_partial_summary(
        &config.out,
        checkpoint,
        &runner.store,
        preflight.passed,
        started.elapsed().as_secs_f64(),
    );
    write_library_summary(&config.out, &runner.store);

    let snapshot = runner.store.snapshot();
    let seconds = started.elapsed().as_secs_f64();
    let generated_pocket_count = snapshot.registry_entry_count.saturating_sub(1);
    let bad_commit_rate =
        checkpoint.bad_commit_count as f64 / checkpoint.completed_queues.max(1) as f64;
    let unsafe_promotion_rate =
        checkpoint.unsafe_promotion_count as f64 / checkpoint.completed_lessons.max(1) as f64;
    let passed = preflight.passed
        && checkpoint.completed_queues == config.rounds
        && checkpoint.failed_count == 0
        && checkpoint.bad_commit_count == 0
        && checkpoint.unsafe_promotion_count == 0
        && generated_pocket_count > 0
        && snapshot.ledger_complete;

    fs::write(
        config.out.join("final_training_results.json"),
        format!(
            concat!(
                "{{\n",
                "  \"passed\": {},\n",
                "  \"rounds\": {},\n",
                "  \"completed_queues\": {},\n",
                "  \"completed_lessons\": {},\n",
                "  \"promoted_count\": {},\n",
                "  \"failed_count\": {},\n",
                "  \"generated_pocket_count\": {},\n",
                "  \"registry_entry_count\": {},\n",
                "  \"bad_commit_rate\": {:.6},\n",
                "  \"unsafe_promotion_rate\": {:.6},\n",
                "  \"preflight_gate_passed\": {},\n",
                "  \"checkpoint_written\": {},\n",
                "  \"seconds\": {:.9},\n",
                "  \"queues_per_sec\": {:.3},\n",
                "  \"lessons_per_sec\": {:.3}\n",
                "}}\n"
            ),
            passed,
            config.rounds,
            checkpoint.completed_queues,
            checkpoint.completed_lessons,
            checkpoint.promoted_count,
            checkpoint.failed_count,
            generated_pocket_count,
            snapshot.registry_entry_count,
            bad_commit_rate,
            unsafe_promotion_rate,
            preflight.passed,
            checkpoint_path.exists(),
            seconds,
            checkpoint.completed_queues as f64 / seconds.max(0.000_001),
            checkpoint.completed_lessons as f64 / seconds.max(0.000_001)
        ),
    )
    .expect("write final training results");

    fs::write(
        config.out.join("report.md"),
        format!(
            "# E75 Rust Final Curriculum Pocket Generation Runner\n\n\
             ```text\n\
             passed = {}\n\
             rounds = {}\n\
             completed_queues = {}\n\
             completed_lessons = {}\n\
             promoted_count = {}\n\
             generated_pocket_count = {}\n\
             bad_commit_rate = {:.6}\n\
             unsafe_promotion_rate = {:.6}\n\
             preflight_gate_passed = {}\n\
             checkpoint_written = {}\n\
             ```\n",
            passed,
            config.rounds,
            checkpoint.completed_queues,
            checkpoint.completed_lessons,
            checkpoint.promoted_count,
            generated_pocket_count,
            bad_commit_rate,
            unsafe_promotion_rate,
            preflight.passed,
            checkpoint_path.exists(),
        ),
    )
    .expect("write report");

    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"complete\",\"passed\":{},\"completed_queues\":{},\"completed_lessons\":{},\"generated_pocket_count\":{},\"checkpoint_written\":{}}}",
            now_millis(),
            passed,
            checkpoint.completed_queues,
            checkpoint.completed_lessons,
            generated_pocket_count,
            checkpoint_path.exists()
        ),
    );

    FinalTrainingSummary {
        passed,
        rounds: config.rounds,
        completed_queues: checkpoint.completed_queues,
        completed_lessons: checkpoint.completed_lessons,
        promoted_count: checkpoint.promoted_count,
        failed_count: checkpoint.failed_count,
        generated_pocket_count,
        bad_commit_rate,
        unsafe_promotion_rate,
        preflight_gate_passed: preflight.passed,
        checkpoint_written: checkpoint_path.exists(),
        seconds,
        out: config.out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_training_runner_writes_checkpoint_and_grows_library() {
        let out =
            std::env::temp_dir().join(format!("vraxion_e75_final_training_test_{}", now_millis()));
        let mut config = FinalTrainingConfig::new(12, out.clone());
        config.preflight_rounds = 4;
        config.checkpoint_interval = 3;

        let summary = run_final_curriculum_pocket_generation(config);

        assert!(summary.passed);
        assert_eq!(summary.completed_queues, 12);
        assert_eq!(summary.completed_lessons, 48);
        assert_eq!(summary.failed_count, 0);
        assert!(summary.generated_pocket_count > 0);
        assert!(summary.preflight_gate_passed);
        assert!(summary.checkpoint_written);
        assert!(out.join("progress.jsonl").exists());
        assert!(out.join("checkpoint_latest.json").exists());
        assert!(out.join("final_training_results.json").exists());
        assert!(out.join("library_summary.json").exists());

        let _ = fs::remove_dir_all(out);
    }
}
