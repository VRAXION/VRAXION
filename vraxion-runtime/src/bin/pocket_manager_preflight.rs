use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use vraxion_runtime::{
    evaluate_promotion, ChallengerEvidence, PromotionBlockReason, PromotionEvidence,
    PromotionLevel, SafetyGate, ScoreVector,
};

#[derive(Default)]
struct Metrics {
    cases: u64,
    success: u64,
    expected_core: u64,
    core_promoted: u64,
    bad_core_promotion: u64,
    missed_core: u64,
    rare_critical_cases: u64,
    rare_critical_preserved: u64,
    unsafe_high_utility_block: u64,
    credit_hijack_block: u64,
    delayed_poison_detection: u64,
    negative_transfer_detection: u64,
    redundant_clone_rejection: u64,
    scope_violation_block: u64,
    reload_transfer_success: u64,
    long_horizon_no_harm: u64,
    challenger_required_block: u64,
}

impl Metrics {
    fn record_case(
        &mut self,
        ok: bool,
        expected_core: bool,
        actual_level: PromotionLevel,
        adversarial: bool,
    ) {
        self.cases += 1;
        self.success += ok as u64;
        self.expected_core += expected_core as u64;
        self.core_promoted += actual_level.is_core_or_above() as u64;
        self.missed_core += (expected_core && !actual_level.is_core_or_above()) as u64;
        self.bad_core_promotion += (adversarial && actual_level.is_core_or_above()) as u64;
    }

    fn passed(&self) -> bool {
        self.cases == self.success && self.bad_core_promotion == 0 && self.missed_core == 0
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

fn strong_score() -> ScoreVector {
    ScoreVector {
        utility: 0.96,
        safety: 0.99,
        eligible_activation: 0.90,
        generality: 0.96,
        uniqueness: 0.95,
        transfer: 0.96,
        robustness: 0.97,
        cost: 0.08,
        stability: 0.98,
        scope_clarity: 0.96,
    }
}

fn safe_gate() -> SafetyGate {
    SafetyGate {
        trace_safe: true,
        no_direct_flow_write: true,
        no_credit_hijack: true,
        no_delayed_poison: true,
        no_negative_transfer: true,
        no_unsafe_high_utility: true,
        no_scope_violation: true,
    }
}

fn challenger() -> ChallengerEvidence {
    ChallengerEvidence {
        challenger_sweep_passed: true,
        counterfactual_unique: true,
        reload_shadow_import_passed: true,
        long_horizon_no_harm: true,
        redundant_clone_rejected: true,
        rare_critical_preserved: true,
    }
}

fn base_evidence() -> PromotionEvidence {
    PromotionEvidence {
        score: strong_score(),
        safety: safe_gate(),
        challenger: challenger(),
        rare_critical: false,
        global_scope_allowed: false,
    }
}

fn is_preserved_rare_critical(level: PromotionLevel) -> bool {
    matches!(
        level,
        PromotionLevel::SemiPerma | PromotionLevel::Core | PromotionLevel::TrueGoldenDisc
    )
}

fn run_round(idx: u64, metrics: &mut Metrics) {
    let core = evaluate_promotion(base_evidence());
    let core_ok = core.level == PromotionLevel::Core && core.reason == PromotionBlockReason::None;
    metrics.record_case(core_ok, true, core.level, false);
    metrics.reload_transfer_success += core_ok as u64;
    metrics.long_horizon_no_harm += core_ok as u64;

    let mut true_golden = base_evidence();
    true_golden.global_scope_allowed = true;
    true_golden.score = ScoreVector {
        utility: 1.0,
        safety: 1.0,
        eligible_activation: 1.0,
        generality: 1.0,
        uniqueness: 1.0,
        transfer: 1.0,
        robustness: 1.0,
        cost: 0.0,
        stability: 1.0,
        scope_clarity: 1.0,
    };
    let true_golden = evaluate_promotion(true_golden);
    let true_ok = true_golden.level == PromotionLevel::TrueGoldenDisc;
    metrics.record_case(true_ok, true, true_golden.level, false);

    let mut rare = base_evidence();
    rare.rare_critical = true;
    rare.score.eligible_activation = 0.03 + ((idx & 1) as f32 * 0.02);
    let rare = evaluate_promotion(rare);
    let rare_ok = is_preserved_rare_critical(rare.level);
    metrics.record_case(rare_ok, true, rare.level, false);
    metrics.rare_critical_cases += 1;
    metrics.rare_critical_preserved += rare_ok as u64;

    let mut unsafe_utility = base_evidence();
    unsafe_utility.safety.no_unsafe_high_utility = false;
    unsafe_utility.score.utility = 1.0;
    let unsafe_utility = evaluate_promotion(unsafe_utility);
    let unsafe_ok = unsafe_utility.level == PromotionLevel::Quarantine
        && unsafe_utility.reason == PromotionBlockReason::UnsafeHighUtility;
    metrics.record_case(unsafe_ok, false, unsafe_utility.level, true);
    metrics.unsafe_high_utility_block += unsafe_ok as u64;

    let mut hijack = base_evidence();
    hijack.safety.no_credit_hijack = false;
    let hijack = evaluate_promotion(hijack);
    let hijack_ok = hijack.level == PromotionLevel::Quarantine
        && hijack.reason == PromotionBlockReason::CreditHijackRisk;
    metrics.record_case(hijack_ok, false, hijack.level, true);
    metrics.credit_hijack_block += hijack_ok as u64;

    let mut poison = base_evidence();
    poison.safety.no_delayed_poison = false;
    let poison = evaluate_promotion(poison);
    let poison_ok = poison.level == PromotionLevel::Quarantine
        && poison.reason == PromotionBlockReason::DelayedPoisonRisk;
    metrics.record_case(poison_ok, false, poison.level, true);
    metrics.delayed_poison_detection += poison_ok as u64;

    let mut transfer = base_evidence();
    transfer.safety.no_negative_transfer = false;
    let transfer = evaluate_promotion(transfer);
    let transfer_ok = transfer.level == PromotionLevel::Quarantine
        && transfer.reason == PromotionBlockReason::NegativeTransfer;
    metrics.record_case(transfer_ok, false, transfer.level, true);
    metrics.negative_transfer_detection += transfer_ok as u64;

    let mut scope = base_evidence();
    scope.safety.no_scope_violation = false;
    let scope = evaluate_promotion(scope);
    let scope_ok = scope.level == PromotionLevel::Quarantine
        && scope.reason == PromotionBlockReason::ScopeViolation;
    metrics.record_case(scope_ok, false, scope.level, true);
    metrics.scope_violation_block += scope_ok as u64;

    let mut clone = base_evidence();
    clone.challenger.counterfactual_unique = false;
    let clone = evaluate_promotion(clone);
    let clone_ok =
        !clone.level.is_core_or_above() && clone.reason == PromotionBlockReason::RedundantClone;
    metrics.record_case(clone_ok, false, clone.level, true);
    metrics.redundant_clone_rejection += clone_ok as u64;

    let mut no_challenger = base_evidence();
    no_challenger.challenger.challenger_sweep_passed = false;
    let no_challenger = evaluate_promotion(no_challenger);
    let no_challenger_ok = !no_challenger.level.is_core_or_above()
        && no_challenger.reason == PromotionBlockReason::ChallengerRequired;
    metrics.record_case(no_challenger_ok, false, no_challenger.level, true);
    metrics.challenger_required_block += no_challenger_ok as u64;

    let mut reload_fail = base_evidence();
    reload_fail.challenger.reload_shadow_import_passed = false;
    let reload_fail = evaluate_promotion(reload_fail);
    let reload_ok = !reload_fail.level.is_core_or_above()
        && reload_fail.reason == PromotionBlockReason::ReloadOrShadowImportFailed;
    metrics.record_case(reload_ok, false, reload_fail.level, true);

    let mut horizon_harm = base_evidence();
    horizon_harm.challenger.long_horizon_no_harm = false;
    let horizon_harm = evaluate_promotion(horizon_harm);
    let horizon_ok = !horizon_harm.level.is_core_or_above()
        && horizon_harm.reason == PromotionBlockReason::LongHorizonHarm;
    metrics.record_case(horizon_ok, false, horizon_harm.level, true);
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
        out.join("promotion_policy_config.json"),
        concat!(
            "{\n",
            "  \"policy\": \"hard safety gate -> vector score -> challenger sweep -> reload/shadow import -> scope-limited promotion\",\n",
            "  \"score_dimensions\": [\"utility\", \"safety\", \"eligible_activation\", \"generality\", \"uniqueness\", \"transfer\", \"robustness\", \"cost\", \"stability\", \"scope_clarity\"],\n",
            "  \"core_requires_challenger\": true,\n",
            "  \"rare_critical_activation_is_eligibility_conditioned\": true,\n",
            "  \"final_answer_only_allowed\": false,\n",
            "  \"popularity_only_allowed\": false,\n",
            "  \"scalar_average_only_allowed\": false\n",
            "}\n"
        ),
    )
    .expect("write policy config");

    let promotion_accuracy = ratio(metrics.success, metrics.cases);
    let bad_core_rate = ratio(metrics.bad_core_promotion, metrics.cases);
    let missed_core_rate = ratio(metrics.missed_core, metrics.expected_core);
    let rare_preservation = ratio(metrics.rare_critical_preserved, metrics.rare_critical_cases);
    let results = format!(
        concat!(
            "{{\n",
            "  \"passed\": {},\n",
            "  \"rounds\": {},\n",
            "  \"cases\": {},\n",
            "  \"success\": {},\n",
            "  \"promotion_accuracy\": {:.6},\n",
            "  \"bad_core_promotion_rate\": {:.6},\n",
            "  \"missed_core_rate\": {:.6},\n",
            "  \"rare_critical_preservation\": {:.6},\n",
            "  \"unsafe_high_utility_block_rate\": {:.6},\n",
            "  \"credit_hijack_block_rate\": {:.6},\n",
            "  \"delayed_poison_detection\": {:.6},\n",
            "  \"negative_transfer_detection\": {:.6},\n",
            "  \"redundant_clone_rejection\": {:.6},\n",
            "  \"scope_violation_block_rate\": {:.6},\n",
            "  \"challenger_required_block_rate\": {:.6},\n",
            "  \"reload_transfer_success\": {:.6},\n",
            "  \"long_horizon_no_harm\": {:.6},\n",
            "  \"seconds\": {:.9},\n",
            "  \"rows_per_sec\": {:.3}\n",
            "}}\n"
        ),
        metrics.passed(),
        rounds,
        metrics.cases,
        metrics.success,
        promotion_accuracy,
        bad_core_rate,
        missed_core_rate,
        rare_preservation,
        ratio(metrics.unsafe_high_utility_block, rounds),
        ratio(metrics.credit_hijack_block, rounds),
        ratio(metrics.delayed_poison_detection, rounds),
        ratio(metrics.negative_transfer_detection, rounds),
        ratio(metrics.redundant_clone_rejection, rounds),
        ratio(metrics.scope_violation_block, rounds),
        ratio(metrics.challenger_required_block, rounds),
        ratio(metrics.reload_transfer_success, rounds),
        ratio(metrics.long_horizon_no_harm, rounds),
        seconds,
        metrics.cases as f64 / seconds.max(0.000_001),
    );
    fs::write(out.join("preflight_results.json"), results).expect("write results");
    fs::write(
        out.join("report.md"),
        format!(
            "# E67 Pocket Manager Promotion Policy Preflight\n\n\
             ```text\n\
             passed = {}\n\
             cases = {}\n\
             success = {}\n\
             promotion_accuracy = {:.6}\n\
             bad_core_promotion_rate = {:.6}\n\
             missed_core_rate = {:.6}\n\
             rare_critical_preservation = {:.6}\n\
             unsafe_high_utility_block_rate = {:.6}\n\
             challenger_required_block_rate = {:.6}\n\
             ```\n",
            metrics.passed(),
            metrics.cases,
            metrics.success,
            promotion_accuracy,
            bad_core_rate,
            missed_core_rate,
            rare_preservation,
            ratio(metrics.unsafe_high_utility_block, rounds),
            ratio(metrics.challenger_required_block, rounds),
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
        PathBuf::from("target/pilot_wave/e67_pocket_manager_promotion_policy_preflight")
    });
    fs::create_dir_all(&out).expect("create output directory");
    let progress = out.join("progress.jsonl");
    let _ = fs::remove_file(&progress);
    append_jsonl(
        &progress,
        &format!(
            "{{\"timestamp_ms\":{},\"event\":\"start\",\"rounds\":{},\"policy\":\"E52 vector+challenger\"}}",
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
                    "{{\"timestamp_ms\":{},\"event\":\"progress\",\"round\":{},\"cases\":{},\"success\":{},\"bad_core_promotion\":{},\"missed_core\":{}}}",
                    now_millis(),
                    idx + 1,
                    metrics.cases,
                    metrics.success,
                    metrics.bad_core_promotion,
                    metrics.missed_core
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
            "{{\"timestamp_ms\":{},\"event\":\"complete\",\"passed\":{},\"cases\":{},\"success\":{},\"bad_core_promotion\":{},\"missed_core\":{}}}",
            now_millis(),
            metrics.passed(),
            metrics.cases,
            metrics.success,
            metrics.bad_core_promotion,
            metrics.missed_core
        ),
    );
    println!(
        "{{\"passed\":{},\"rounds\":{},\"cases\":{},\"success\":{},\"bad_core_promotion\":{},\"missed_core\":{},\"seconds\":{:.9},\"rows_per_sec\":{:.3},\"out\":\"{}\"}}",
        metrics.passed(),
        rounds,
        metrics.cases,
        metrics.success,
        metrics.bad_core_promotion,
        metrics.missed_core,
        seconds,
        metrics.cases as f64 / seconds.max(0.000_001),
        out.display()
    );
}
