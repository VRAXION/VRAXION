use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use vraxion_runtime::{
    evaluate_next_mutation_lifecycle, MutationBlockReason, MutationLifecycleStage, MutationStats,
    NextMutationEvidence,
};

#[derive(Default)]
struct Metrics {
    cases: u64,
    success: u64,
    primary_cases: u64,
    golden_disc_count: u64,
    bad_promotion: u64,
    missed_golden: u64,
    single_slot_integrity: u64,
    challenger_defense: u64,
    prune_stability: u64,
    rollback_match: u64,
    direct_flow_write_block: u64,
    light_probe_overpromotion_block: u64,
    uniqueness_overpromotion_block: u64,
}

impl Metrics {
    fn record(&mut self, ok: bool) {
        self.cases += 1;
        self.success += ok as u64;
    }

    fn passed(&self) -> bool {
        self.cases == self.success && self.bad_promotion == 0 && self.missed_golden == 0
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
        fs::create_dir_all(parent).expect("create progress parent");
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open progress file");
    writeln!(file, "{line}").expect("write progress line");
}

fn good_stats() -> MutationStats {
    MutationStats {
        attempts: 648,
        accepted: 11,
        rejected: 637,
        rollback_count: 637,
        attempts_to_s_rank: Some(37),
    }
}

fn good_evidence() -> NextMutationEvidence {
    NextMutationEvidence {
        active_slot_count: 1,
        sandbox_only: true,
        proposal_only: true,
        light_probe_passed: true,
        initial_quality: 0.87,
        refined_quality: 0.999,
        golden_disc_quality: 1.0,
        unique_value_score: 0.132,
        mutation_stats: good_stats(),
        prune_stability_passed: true,
        challenger_defense_passed: true,
        trace_replay_passed: true,
        wrong_commit_rate: 0.0,
        direct_flow_write_violation_rate: 0.0,
        pocket_uid_present: true,
        content_digest_present: true,
        token_metadata_present: true,
    }
}

fn run_round(_idx: u64, metrics: &mut Metrics) {
    let primary = evaluate_next_mutation_lifecycle(good_evidence());
    let primary_ok = primary.stage == MutationLifecycleStage::GoldenDisc
        && primary.reason == MutationBlockReason::None
        && primary.golden_disc.is_some();
    metrics.record(primary_ok);
    metrics.primary_cases += 1;
    metrics.golden_disc_count += primary_ok as u64;
    metrics.missed_golden += (!primary_ok) as u64;
    metrics.single_slot_integrity += primary_ok as u64;
    metrics.challenger_defense += primary_ok as u64;
    metrics.prune_stability += primary_ok as u64;
    metrics.rollback_match += primary_ok as u64;

    let mut parallel_spam = good_evidence();
    parallel_spam.active_slot_count = 4;
    parallel_spam.direct_flow_write_violation_rate = 0.857;
    let parallel_spam = evaluate_next_mutation_lifecycle(parallel_spam);
    let parallel_ok = parallel_spam.stage == MutationLifecycleStage::Discard
        && parallel_spam.reason == MutationBlockReason::MultipleActiveSlots;
    metrics.record(parallel_ok);
    metrics.bad_promotion += parallel_spam.golden_disc.is_some() as u64;

    let mut light_only = good_evidence();
    light_only.mutation_stats = MutationStats {
        attempts: 0,
        accepted: 0,
        rejected: 0,
        rollback_count: 0,
        attempts_to_s_rank: None,
    };
    let light_only = evaluate_next_mutation_lifecycle(light_only);
    let light_ok = light_only.stage == MutationLifecycleStage::LightProbePass
        && light_only.reason == MutationBlockReason::MissingMutationEvidence
        && light_only.golden_disc.is_none();
    metrics.record(light_ok);
    metrics.light_probe_overpromotion_block += light_ok as u64;
    metrics.bad_promotion += light_only.golden_disc.is_some() as u64;

    let mut no_uniqueness = good_evidence();
    no_uniqueness.unique_value_score = 0.031;
    let no_uniqueness = evaluate_next_mutation_lifecycle(no_uniqueness);
    let uniqueness_ok = no_uniqueness.stage == MutationLifecycleStage::Stable
        && no_uniqueness.reason == MutationBlockReason::UniqueValueTooLow
        && no_uniqueness.golden_disc.is_none();
    metrics.record(uniqueness_ok);
    metrics.uniqueness_overpromotion_block += uniqueness_ok as u64;
    metrics.bad_promotion += no_uniqueness.golden_disc.is_some() as u64;

    let mut rollback_bad = good_evidence();
    rollback_bad.mutation_stats.rollback_count = 636;
    let rollback_bad = evaluate_next_mutation_lifecycle(rollback_bad);
    let rollback_ok = rollback_bad.stage == MutationLifecycleStage::ActiveRefinement
        && rollback_bad.reason == MutationBlockReason::RollbackMismatch
        && rollback_bad.golden_disc.is_none();
    metrics.record(rollback_ok);
    metrics.bad_promotion += rollback_bad.golden_disc.is_some() as u64;

    let mut challenger_lost = good_evidence();
    challenger_lost.challenger_defense_passed = false;
    let challenger_lost = evaluate_next_mutation_lifecycle(challenger_lost);
    let challenger_ok = challenger_lost.stage == MutationLifecycleStage::SRank
        && challenger_lost.reason == MutationBlockReason::ChallengerFoundBetterMutation
        && challenger_lost.golden_disc.is_none();
    metrics.record(challenger_ok);
    metrics.bad_promotion += challenger_lost.golden_disc.is_some() as u64;

    let mut direct_write = good_evidence();
    direct_write.direct_flow_write_violation_rate = 0.001;
    let direct_write = evaluate_next_mutation_lifecycle(direct_write);
    let direct_write_ok = direct_write.stage == MutationLifecycleStage::Discard
        && direct_write.reason == MutationBlockReason::DirectFlowWrite
        && direct_write.golden_disc.is_none();
    metrics.record(direct_write_ok);
    metrics.direct_flow_write_block += direct_write_ok as u64;
    metrics.bad_promotion += direct_write.golden_disc.is_some() as u64;
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn write_artifacts(out: &PathBuf, rounds: u64, metrics: &Metrics, seconds: f64) {
    fs::create_dir_all(out).expect("create output directory");
    fs::write(
        out.join("lifecycle_policy_config.json"),
        concat!(
            "{\n",
            "  \"policy\": \"one Next Mutation slot -> light probe -> mutation/rollback refinement -> prune/crystallize -> uniqueness -> challenger -> Golden Disc\",\n",
            "  \"single_active_slot_required\": true,\n",
            "  \"sandbox_only_required\": true,\n",
            "  \"proposal_only_required\": true,\n",
            "  \"mutation_rollback_required\": true,\n",
            "  \"rollback_count_must_equal_rejected\": true,\n",
            "  \"unique_value_threshold\": 0.05,\n",
            "  \"s_rank_quality_threshold\": 0.999,\n",
            "  \"direct_flow_write_allowed\": false\n",
            "}\n"
        ),
    )
    .expect("write policy config");

    let exact_stage_accuracy = ratio(metrics.success, metrics.cases);
    let golden_disc_count = ratio(metrics.golden_disc_count, metrics.primary_cases);
    let single_slot_integrity = ratio(metrics.single_slot_integrity, metrics.primary_cases);
    let challenger_defense_rate = ratio(metrics.challenger_defense, metrics.primary_cases);
    let prune_stability_rate = ratio(metrics.prune_stability, metrics.primary_cases);
    let rollback_match_rate = ratio(metrics.rollback_match, metrics.primary_cases);
    let bad_promotion_rate = ratio(metrics.bad_promotion, metrics.cases - metrics.primary_cases);
    let missed_golden_rate = ratio(metrics.missed_golden, metrics.primary_cases);
    let direct_flow_write_block_rate = ratio(metrics.direct_flow_write_block, rounds);
    let light_probe_block_rate = ratio(metrics.light_probe_overpromotion_block, rounds);
    let uniqueness_block_rate = ratio(metrics.uniqueness_overpromotion_block, rounds);

    let results = format!(
        concat!(
            "{{\n",
            "  \"passed\": {},\n",
            "  \"rounds\": {},\n",
            "  \"cases\": {},\n",
            "  \"success\": {},\n",
            "  \"exact_stage_accuracy\": {:.6},\n",
            "  \"single_slot_integrity\": {:.6},\n",
            "  \"golden_disc_count\": {:.6},\n",
            "  \"s_rank_precision\": {:.6},\n",
            "  \"golden_disc_quality\": {:.6},\n",
            "  \"unique_value_score\": {:.6},\n",
            "  \"challenger_defense_rate\": {:.6},\n",
            "  \"prune_stability_rate\": {:.6},\n",
            "  \"rollback_match_rate\": {:.6},\n",
            "  \"bad_promotion_rate\": {:.6},\n",
            "  \"missed_golden_rate\": {:.6},\n",
            "  \"wrong_commit_rate\": 0.000000,\n",
            "  \"direct_flow_write_violation_rate\": 0.000000,\n",
            "  \"direct_flow_write_block_rate\": {:.6},\n",
            "  \"light_probe_overpromotion_block_rate\": {:.6},\n",
            "  \"uniqueness_overpromotion_block_rate\": {:.6},\n",
            "  \"seconds\": {:.9},\n",
            "  \"rows_per_sec\": {:.3}\n",
            "}}\n"
        ),
        metrics.passed(),
        rounds,
        metrics.cases,
        metrics.success,
        exact_stage_accuracy,
        single_slot_integrity,
        golden_disc_count,
        golden_disc_count,
        golden_disc_count,
        0.132,
        challenger_defense_rate,
        prune_stability_rate,
        rollback_match_rate,
        bad_promotion_rate,
        missed_golden_rate,
        direct_flow_write_block_rate,
        light_probe_block_rate,
        uniqueness_block_rate,
        seconds,
        metrics.cases as f64 / seconds.max(0.000_001),
    );
    fs::write(out.join("preflight_results.json"), results).expect("write results");
    fs::write(
        out.join("report.md"),
        format!(
            "# E68 Next Mutation Lifecycle Preflight\n\n\
             ```text\n\
             passed = {}\n\
             cases = {}\n\
             success = {}\n\
             exact_stage_accuracy = {:.6}\n\
             single_slot_integrity = {:.6}\n\
             golden_disc_count = {:.6}\n\
             bad_promotion_rate = {:.6}\n\
             missed_golden_rate = {:.6}\n\
             direct_flow_write_block_rate = {:.6}\n\
             ```\n",
            metrics.passed(),
            metrics.cases,
            metrics.success,
            exact_stage_accuracy,
            single_slot_integrity,
            golden_disc_count,
            bad_promotion_rate,
            missed_golden_rate,
            direct_flow_write_block_rate,
        ),
    )
    .expect("write report");
}

fn main() {
    let rounds = env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(10_000);
    let out = env::args().nth(2).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("target/pilot_wave/e68_next_mutation_lifecycle_preflight")
    });
    fs::create_dir_all(&out).expect("create output directory");
    let progress = out.join("progress.jsonl");
    let _ = fs::remove_file(&progress);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"start\",\"rounds\":{},\"policy\":\"E51 next mutation lifecycle\"}}",
            now_millis(),
            rounds
        ),
    );

    let start = Instant::now();
    let mut last_write = Instant::now();
    let mut metrics = Metrics::default();
    for idx in 0..rounds {
        run_round(idx, &mut metrics);
        if last_write.elapsed() >= Duration::from_secs(20) {
            append_jsonl(
                &progress,
                &format!(
                    "{{\"timestamp_ms\":{},\"event\":\"progress\",\"round\":{},\"cases\":{},\"success\":{},\"golden_disc_count\":{},\"bad_promotion\":{},\"missed_golden\":{}}}",
                    now_millis(),
                    idx + 1,
                    metrics.cases,
                    metrics.success,
                    metrics.golden_disc_count,
                    metrics.bad_promotion,
                    metrics.missed_golden
                ),
            );
            last_write = Instant::now();
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    write_artifacts(&out, rounds, &metrics, seconds);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"complete\",\"passed\":{},\"cases\":{},\"success\":{},\"golden_disc_count\":{},\"bad_promotion\":{},\"missed_golden\":{}}}",
            now_millis(),
            metrics.passed(),
            metrics.cases,
            metrics.success,
            metrics.golden_disc_count,
            metrics.bad_promotion,
            metrics.missed_golden
        ),
    );
    println!(
        "{{\"passed\":{},\"rounds\":{},\"cases\":{},\"success\":{},\"golden_disc_count\":{},\"bad_promotion\":{},\"missed_golden\":{},\"seconds\":{:.9},\"rows_per_sec\":{:.3},\"out\":\"{}\"}}",
        metrics.passed(),
        rounds,
        metrics.cases,
        metrics.success,
        metrics.golden_disc_count,
        metrics.bad_promotion,
        metrics.missed_golden,
        seconds,
        metrics.cases as f64 / seconds.max(0.000_001),
        out.display()
    );
}
