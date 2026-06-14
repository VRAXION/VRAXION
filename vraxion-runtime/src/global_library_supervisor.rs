use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::{
    final_bake::{base_store, lifecycle, promotion},
    final_training::training_candidates_for_rounds,
    run_final_training_supervisor, FinalTrainingSupervisorConfig, PocketLibraryStore,
    StorePromotionCandidate,
};

#[derive(Debug, Clone, PartialEq)]
pub struct GlobalLibrarySupervisorConfig {
    pub lanes: usize,
    pub rounds_per_lane: u64,
    pub out: PathBuf,
    pub preflight_rounds: u64,
    pub checkpoint_interval: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlobalLibrarySupervisorSummary {
    pub passed: bool,
    pub lanes: usize,
    pub rounds_per_lane: u64,
    pub total_local_candidates: u64,
    pub unique_candidates: u64,
    pub promoted_to_global: u64,
    pub duplicate_candidates_blocked: u64,
    pub failed_promotions: u64,
    pub lane_artifact_pass_count: u64,
    pub global_registry_entry_count: usize,
    pub global_generated_pocket_count: usize,
    pub redundant_clone_block_rate: f64,
    pub bad_commit_rate: f64,
    pub unsafe_promotion_rate: f64,
    pub seconds: f64,
    pub out: PathBuf,
}

impl GlobalLibrarySupervisorConfig {
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
        fs::create_dir_all(parent).expect("create global supervisor progress parent");
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open global supervisor progress file");
    writeln!(file, "{line}").expect("write global supervisor progress line");
}

fn json_bool(text: &str, key: &str) -> Option<bool> {
    let marker = format!("\"{key}\":");
    let start = text.find(&marker)? + marker.len();
    let value: String = text[start..]
        .chars()
        .skip_while(|ch| ch.is_whitespace())
        .take_while(|ch| ch.is_ascii_alphabetic())
        .collect();
    match value.as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn lane_artifact_passed(lane_supervisor_out: &Path, lane_index: usize) -> bool {
    let path = lane_supervisor_out
        .join(format!("lane_{lane_index:02}"))
        .join("final_training_results.json");
    let Ok(text) = fs::read_to_string(path) else {
        return false;
    };
    json_bool(&text, "passed") == Some(true)
}

fn same_candidate(a: StorePromotionCandidate, b: StorePromotionCandidate) -> bool {
    a.pocket_uid == b.pocket_uid
        || a.content_digest == b.content_digest
        || a.token_hash == b.token_hash
}

fn candidate_seen(
    candidates: &[StorePromotionCandidate],
    candidate: StorePromotionCandidate,
) -> bool {
    candidates
        .iter()
        .any(|existing| same_candidate(*existing, candidate))
}

fn write_partial_global_snapshot(
    out: &Path,
    store: &PocketLibraryStore,
    total_local_candidates: u64,
    promoted_to_global: u64,
    duplicate_candidates_blocked: u64,
    failed_promotions: u64,
    seconds: f64,
) {
    let snapshot = store.snapshot();
    fs::write(
        out.join("partial_global_library_snapshot.json"),
        format!(
            concat!(
                "{{\n",
                "  \"total_local_candidates\": {},\n",
                "  \"promoted_to_global\": {},\n",
                "  \"duplicate_candidates_blocked\": {},\n",
                "  \"failed_promotions\": {},\n",
                "  \"global_registry_entry_count\": {},\n",
                "  \"global_token_count\": {},\n",
                "  \"global_artifact_count\": {},\n",
                "  \"global_store_generation\": {},\n",
                "  \"global_quality_delta\": {:.6},\n",
                "  \"seconds\": {:.9}\n",
                "}}\n"
            ),
            total_local_candidates,
            promoted_to_global,
            duplicate_candidates_blocked,
            failed_promotions,
            snapshot.registry_entry_count,
            snapshot.token_count,
            snapshot.artifact_count,
            snapshot.generation,
            store.quality_delta(),
            seconds
        ),
    )
    .expect("write partial global snapshot");
}

fn write_global_registry_jsonl(out: &Path, store: &PocketLibraryStore) {
    let path = out.join("global_registry.jsonl");
    let _ = fs::remove_file(&path);
    for (entry, artifact) in store.registry.iter().zip(store.artifacts.iter()) {
        append_jsonl(
            &path,
            &format!(
                "{{\"pocket_uid\":\"{}\",\"human_alias\":\"{}\",\"content_digest\":\"{}\",\"token_hash\":\"{}\",\"capability_signature\":\"{}\",\"quality_delta\":{:.6},\"generation\":{}}}",
                entry.pocket_uid,
                entry.human_alias,
                artifact.content_digest,
                entry.token_hash,
                entry.capability_signature,
                artifact.quality_delta,
                artifact.generation
            ),
        );
    }
}

pub fn run_global_library_supervisor(
    config: GlobalLibrarySupervisorConfig,
) -> GlobalLibrarySupervisorSummary {
    fs::create_dir_all(&config.out).expect("create global supervisor output directory");
    let progress = config.out.join("progress.jsonl");
    let _ = fs::remove_file(&progress);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"start\",\"lanes\":{},\"rounds_per_lane\":{},\"policy\":\"E77 global merge/dedupe/challenger\"}}",
            now_millis(),
            config.lanes,
            config.rounds_per_lane
        ),
    );

    let started = Instant::now();
    let lane_supervisor_out = config.out.join("lane_supervisor");
    let mut lane_config = FinalTrainingSupervisorConfig::new(
        config.lanes,
        config.rounds_per_lane,
        lane_supervisor_out.clone(),
    );
    lane_config.preflight_rounds = config.preflight_rounds;
    lane_config.checkpoint_interval = config.checkpoint_interval;
    let lane_summary = run_final_training_supervisor(lane_config);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"lane_supervisor_complete\",\"passed\":{},\"completed_queues\":{},\"completed_lessons\":{},\"bad_commit_rate\":{:.6},\"unsafe_promotion_rate\":{:.6}}}",
            now_millis(),
            lane_summary.passed,
            lane_summary.completed_queues,
            lane_summary.completed_lessons,
            lane_summary.bad_commit_rate,
            lane_summary.unsafe_promotion_rate
        ),
    );

    let candidates = training_candidates_for_rounds(config.rounds_per_lane);
    let mut store = base_store();
    let mut seen = Vec::new();
    let mut total_local_candidates = 0u64;
    let mut promoted_to_global = 0u64;
    let mut duplicate_candidates_blocked = 0u64;
    let mut failed_promotions = 0u64;
    let mut lane_artifact_pass_count = 0u64;
    let heartbeat = Duration::from_secs(20);
    let mut last_heartbeat = Instant::now();

    for lane_index in 0..lane_summary.lanes {
        let lane_passed = lane_artifact_passed(&lane_supervisor_out, lane_index);
        lane_artifact_pass_count += lane_passed as u64;
        append_jsonl(
            &progress,
            &format!(
                "{{\"timestamp_ms\":{},\"event\":\"global_merge_lane_start\",\"lane\":{},\"lane_artifact_passed\":{},\"candidate_count\":{}}}",
                now_millis(),
                lane_index,
                lane_passed,
                candidates.len()
            ),
        );
        if !lane_passed {
            continue;
        }
        for candidate in candidates.iter().copied() {
            total_local_candidates += 1;
            if candidate_seen(&seen, candidate) {
                duplicate_candidates_blocked += 1;
                append_jsonl(
                    &progress,
                    &format!(
                        "{{\"timestamp_ms\":{},\"event\":\"redundant_clone_blocked\",\"lane\":{},\"pocket_uid\":\"{}\",\"reason\":\"uid_or_digest_or_token_seen\"}}",
                        now_millis(),
                        lane_index,
                        candidate.pocket_uid
                    ),
                );
            } else {
                let decision = store.promote_candidate(candidate, lifecycle(), promotion(false));
                if decision.allowed {
                    let reload_allowed = store
                        .tokens
                        .iter()
                        .copied()
                        .find(|token| token.pocket_uid == candidate.pocket_uid)
                        .is_some_and(|token| store.guarded_load(token).allowed);
                    if reload_allowed {
                        seen.push(candidate);
                        promoted_to_global += 1;
                        append_jsonl(
                            &progress,
                            &format!(
                                "{{\"timestamp_ms\":{},\"event\":\"global_candidate_promoted\",\"lane\":{},\"pocket_uid\":\"{}\",\"global_generation\":{},\"guarded_reload\":true}}",
                                now_millis(),
                                lane_index,
                                candidate.pocket_uid,
                                store.generation
                            ),
                        );
                    } else {
                        failed_promotions += 1;
                        append_jsonl(
                            &progress,
                            &format!(
                                "{{\"timestamp_ms\":{},\"event\":\"global_candidate_reload_failed\",\"lane\":{},\"pocket_uid\":\"{}\"}}",
                                now_millis(),
                                lane_index,
                                candidate.pocket_uid
                            ),
                        );
                    }
                } else {
                    failed_promotions += 1;
                    append_jsonl(
                        &progress,
                        &format!(
                            "{{\"timestamp_ms\":{},\"event\":\"global_candidate_rejected\",\"lane\":{},\"pocket_uid\":\"{}\",\"reason\":\"{:?}\"}}",
                            now_millis(),
                            lane_index,
                            candidate.pocket_uid,
                            decision.reason
                        ),
                    );
                }
            }
            if last_heartbeat.elapsed() >= heartbeat {
                write_partial_global_snapshot(
                    &config.out,
                    &store,
                    total_local_candidates,
                    promoted_to_global,
                    duplicate_candidates_blocked,
                    failed_promotions,
                    started.elapsed().as_secs_f64(),
                );
                append_jsonl(
                    &progress,
                    &format!(
                        "{{\"timestamp_ms\":{},\"event\":\"heartbeat\",\"total_local_candidates\":{},\"promoted_to_global\":{},\"duplicate_candidates_blocked\":{},\"failed_promotions\":{}}}",
                        now_millis(),
                        total_local_candidates,
                        promoted_to_global,
                        duplicate_candidates_blocked,
                        failed_promotions
                    ),
                );
                last_heartbeat = Instant::now();
            }
        }
        write_partial_global_snapshot(
            &config.out,
            &store,
            total_local_candidates,
            promoted_to_global,
            duplicate_candidates_blocked,
            failed_promotions,
            started.elapsed().as_secs_f64(),
        );
    }

    write_global_registry_jsonl(&config.out, &store);

    let duplicate_denominator = total_local_candidates
        .saturating_sub(promoted_to_global)
        .max(1);
    let redundant_clone_block_rate =
        duplicate_candidates_blocked as f64 / duplicate_denominator as f64;
    let snapshot = store.snapshot();
    let global_generated_pocket_count = snapshot.registry_entry_count.saturating_sub(1);
    let seconds = started.elapsed().as_secs_f64();
    let passed = lane_summary.passed
        && lane_artifact_pass_count == lane_summary.lanes as u64
        && promoted_to_global == candidates.len() as u64
        && duplicate_candidates_blocked
            == ((lane_summary.lanes as u64).saturating_sub(1) * candidates.len() as u64)
        && failed_promotions == 0
        && redundant_clone_block_rate == 1.0
        && global_generated_pocket_count == candidates.len()
        && snapshot.ledger_complete
        && lane_summary.bad_commit_rate == 0.0
        && lane_summary.unsafe_promotion_rate == 0.0;

    fs::write(
        config.out.join("global_merge_results.json"),
        format!(
            concat!(
                "{{\n",
                "  \"passed\": {},\n",
                "  \"lanes\": {},\n",
                "  \"rounds_per_lane\": {},\n",
                "  \"total_local_candidates\": {},\n",
                "  \"unique_candidates\": {},\n",
                "  \"promoted_to_global\": {},\n",
                "  \"duplicate_candidates_blocked\": {},\n",
                "  \"failed_promotions\": {},\n",
                "  \"lane_artifact_pass_count\": {},\n",
                "  \"global_registry_entry_count\": {},\n",
                "  \"global_generated_pocket_count\": {},\n",
                "  \"redundant_clone_block_rate\": {:.6},\n",
                "  \"bad_commit_rate\": {:.6},\n",
                "  \"unsafe_promotion_rate\": {:.6},\n",
                "  \"seconds\": {:.9}\n",
                "}}\n"
            ),
            passed,
            lane_summary.lanes,
            config.rounds_per_lane,
            total_local_candidates,
            candidates.len(),
            promoted_to_global,
            duplicate_candidates_blocked,
            failed_promotions,
            lane_artifact_pass_count,
            snapshot.registry_entry_count,
            global_generated_pocket_count,
            redundant_clone_block_rate,
            lane_summary.bad_commit_rate,
            lane_summary.unsafe_promotion_rate,
            seconds
        ),
    )
    .expect("write global merge results");

    fs::write(
        config.out.join("global_library_summary.json"),
        format!(
            concat!(
                "{{\n",
                "  \"registry_entry_count\": {},\n",
                "  \"token_count\": {},\n",
                "  \"artifact_count\": {},\n",
                "  \"generation\": {},\n",
                "  \"ledger_complete\": {},\n",
                "  \"quality_delta\": {:.6},\n",
                "  \"global_generated_pocket_count\": {}\n",
                "}}\n"
            ),
            snapshot.registry_entry_count,
            snapshot.token_count,
            snapshot.artifact_count,
            snapshot.generation,
            snapshot.ledger_complete,
            store.quality_delta(),
            global_generated_pocket_count
        ),
    )
    .expect("write global library summary");

    fs::write(
        config.out.join("report.md"),
        format!(
            "# E77 Global Pocket Library Merge Supervisor\n\n\
             ```text\n\
             passed = {}\n\
             lanes = {}\n\
             rounds_per_lane = {}\n\
             total_local_candidates = {}\n\
             unique_candidates = {}\n\
             promoted_to_global = {}\n\
             duplicate_candidates_blocked = {}\n\
             failed_promotions = {}\n\
             redundant_clone_block_rate = {:.6}\n\
             global_generated_pocket_count = {}\n\
             bad_commit_rate = {:.6}\n\
             unsafe_promotion_rate = {:.6}\n\
             ```\n",
            passed,
            lane_summary.lanes,
            config.rounds_per_lane,
            total_local_candidates,
            candidates.len(),
            promoted_to_global,
            duplicate_candidates_blocked,
            failed_promotions,
            redundant_clone_block_rate,
            global_generated_pocket_count,
            lane_summary.bad_commit_rate,
            lane_summary.unsafe_promotion_rate
        ),
    )
    .expect("write global merge report");

    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"complete\",\"passed\":{},\"promoted_to_global\":{},\"duplicate_candidates_blocked\":{},\"global_generated_pocket_count\":{}}}",
            now_millis(),
            passed,
            promoted_to_global,
            duplicate_candidates_blocked,
            global_generated_pocket_count
        ),
    );

    GlobalLibrarySupervisorSummary {
        passed,
        lanes: lane_summary.lanes,
        rounds_per_lane: config.rounds_per_lane,
        total_local_candidates,
        unique_candidates: candidates.len() as u64,
        promoted_to_global,
        duplicate_candidates_blocked,
        failed_promotions,
        lane_artifact_pass_count,
        global_registry_entry_count: snapshot.registry_entry_count,
        global_generated_pocket_count,
        redundant_clone_block_rate,
        bad_commit_rate: lane_summary.bad_commit_rate,
        unsafe_promotion_rate: lane_summary.unsafe_promotion_rate,
        seconds,
        out: config.out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_supervisor_merges_lanes_and_blocks_redundant_clones() {
        let out = std::env::temp_dir().join(format!(
            "vraxion_e77_global_supervisor_test_{}",
            now_millis()
        ));
        let mut config = GlobalLibrarySupervisorConfig::new(3, 4, out.clone());
        config.preflight_rounds = 4;
        config.checkpoint_interval = 2;

        let summary = run_global_library_supervisor(config);

        assert!(summary.passed);
        assert_eq!(summary.lanes, 3);
        assert_eq!(summary.unique_candidates, 16);
        assert_eq!(summary.promoted_to_global, 16);
        assert_eq!(summary.duplicate_candidates_blocked, 32);
        assert_eq!(summary.failed_promotions, 0);
        assert_eq!(summary.global_generated_pocket_count, 16);
        assert_eq!(summary.redundant_clone_block_rate, 1.0);
        assert!(out.join("global_merge_results.json").exists());
        assert!(out.join("global_library_summary.json").exists());
        assert!(out.join("global_registry.jsonl").exists());
        assert!(out
            .join("lane_supervisor")
            .join("lane_00")
            .join("checkpoint_latest.json")
            .exists());

        let _ = fs::remove_dir_all(out);
    }
}
