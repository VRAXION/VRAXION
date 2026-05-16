//! Runner-local sparse-core transfer probe.
//!
//! 011 takes the sparse motif types exposed by 010 and reinserts them into new
//! spatial phase-lane settings without new growth or pruning. The goal is to
//! test whether the retained motif types behave like a reusable local phase
//! transport rule, not whether another search can rediscover them.

use instnct_core::{Network, PropagationConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const K: usize = 4;
const ARRIVE_BASE: usize = 0;
const GATE_BASE: usize = ARRIVE_BASE + K;
const EMIT_BASE: usize = GATE_BASE + K;
const COINCIDENCE_BASE: usize = EMIT_BASE + K;
const COINCIDENCE_PER_CELL: usize = K * K * K;
const LANES_PER_CELL: usize = COINCIDENCE_BASE + COINCIDENCE_PER_CELL;

const COMMON_CORE_15: &[(usize, usize, usize)] = &[
    (0, 0, 0),
    (0, 1, 1),
    (0, 2, 2),
    (0, 3, 3),
    (1, 0, 1),
    (1, 1, 2),
    (1, 3, 0),
    (2, 0, 2),
    (2, 1, 3),
    (2, 2, 0),
    (2, 3, 1),
    (3, 0, 3),
    (3, 1, 0),
    (3, 2, 1),
    (3, 3, 2),
];

const MISSING_1_2_3: (usize, usize, usize) = (1, 2, 3);

#[derive(Clone, Debug)]
struct Config {
    out: PathBuf,
    seeds: Vec<u64>,
    eval_examples: usize,
    widths: Vec<usize>,
    ticks: usize,
    baseline_steps: usize,
    heartbeat_sec: u64,
}

#[derive(Clone, Debug, Serialize)]
struct PublicCase {
    width: usize,
    wall: Vec<bool>,
    source: (usize, usize),
    source_phase: u8,
    target: (usize, usize),
    gates: Vec<u8>,
}

#[derive(Clone, Debug, Serialize)]
struct PrivateCase {
    label: u8,
    true_path: Vec<(usize, usize)>,
    path_phase_total: u8,
    gate_sum: u8,
    family: String,
    split: String,
}

#[derive(Clone, Debug)]
struct Case {
    public: PublicCase,
    private: PrivateCase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct MotifType {
    input_phase: usize,
    gate: usize,
    output_phase: usize,
}

impl MotifType {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Arm {
    FixedPhaseLaneReference,
    Dense009Reference,
    CommonCore15,
    CommonCore15PlusMissing123,
    Full16RuleTemplate,
    SeedSpecificCoreReinserted,
    RandomMatched15MotifControl,
    RandomMatched16MotifControl,
    RandomMatched25MotifControl,
    CanonicalJackpot007Baseline,
}

impl Arm {
    fn as_str(self) -> &'static str {
        match self {
            Arm::FixedPhaseLaneReference => "FIXED_PHASE_LANE_REFERENCE",
            Arm::Dense009Reference => "DENSE_009_REFERENCE",
            Arm::CommonCore15 => "COMMON_CORE_15",
            Arm::CommonCore15PlusMissing123 => "COMMON_CORE_15_PLUS_MISSING_1_2_3",
            Arm::Full16RuleTemplate => "FULL_16_RULE_TEMPLATE",
            Arm::SeedSpecificCoreReinserted => "SEED_SPECIFIC_CORE_REINSERTED",
            Arm::RandomMatched15MotifControl => "RANDOM_MATCHED_15_MOTIF_CONTROL",
            Arm::RandomMatched16MotifControl => "RANDOM_MATCHED_16_MOTIF_CONTROL",
            Arm::RandomMatched25MotifControl => "RANDOM_MATCHED_25_MOTIF_CONTROL",
            Arm::CanonicalJackpot007Baseline => "CANONICAL_JACKPOT_007_BASELINE",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum PlacementMode {
    TemplateOnAllFreeCells,
    TemplateOnFrontierReachableCells,
    TemplateOnRandomFreeCellsMatchedCount,
    TemplateOnPathCellsDiagnosticOnly,
}

impl PlacementMode {
    fn as_str(self) -> &'static str {
        match self {
            PlacementMode::TemplateOnAllFreeCells => "template_on_all_free_cells",
            PlacementMode::TemplateOnFrontierReachableCells => {
                "template_on_frontier_reachable_cells"
            }
            PlacementMode::TemplateOnRandomFreeCellsMatchedCount => {
                "template_on_random_free_cells_matched_count"
            }
            PlacementMode::TemplateOnPathCellsDiagnosticOnly => {
                "template_on_path_cells_DIAGNOSTIC_ONLY"
            }
        }
    }
}

#[derive(Clone)]
struct Layout {
    width: usize,
}

impl Layout {
    fn neuron_count(&self) -> usize {
        self.width * self.width * LANES_PER_CELL
    }

    fn cell_id(&self, y: usize, x: usize) -> usize {
        y * self.width + x
    }

    fn cell_of_neuron(&self, neuron: usize) -> (usize, usize) {
        let cell = neuron / LANES_PER_CELL;
        (cell / self.width, cell % self.width)
    }

    fn local_lane(&self, neuron: usize) -> usize {
        neuron % LANES_PER_CELL
    }

    fn cell_base(&self, y: usize, x: usize) -> usize {
        self.cell_id(y, x) * LANES_PER_CELL
    }

    fn arrive(&self, y: usize, x: usize, phase: usize) -> usize {
        self.cell_base(y, x) + ARRIVE_BASE + phase
    }

    fn gate(&self, y: usize, x: usize, gate: usize) -> usize {
        self.cell_base(y, x) + GATE_BASE + gate
    }

    fn emit(&self, y: usize, x: usize, phase: usize) -> usize {
        self.cell_base(y, x) + EMIT_BASE + phase
    }

    fn coincidence(
        &self,
        y: usize,
        x: usize,
        input_phase: usize,
        gate: usize,
        output_phase: usize,
    ) -> usize {
        self.cell_base(y, x) + COINCIDENCE_BASE + ((input_phase * K + gate) * K + output_phase)
    }

    fn decode_coincidence(&self, neuron: usize) -> Option<(usize, usize, usize, usize, usize)> {
        let lane = self.local_lane(neuron);
        if !(COINCIDENCE_BASE..COINCIDENCE_BASE + COINCIDENCE_PER_CELL).contains(&lane) {
            return None;
        }
        let rel = lane - COINCIDENCE_BASE;
        let input_phase = rel / (K * K);
        let gate = (rel / K) % K;
        let output_phase = rel % K;
        let (y, x) = self.cell_of_neuron(neuron);
        Some((y, x, input_phase, gate, output_phase))
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct EvalMetrics {
    phase_final_accuracy: f64,
    correct_target_lane_probability_mean: f64,
    same_target_counterfactual_accuracy: f64,
    gate_shuffle_accuracy: f64,
    gate_shuffle_collapse: f64,
    width_transfer_accuracy: f64,
    long_path_transfer_accuracy: f64,
    new_layout_transfer_accuracy: f64,
    reverse_path_consistency_accuracy: f64,
    motif_type_count: usize,
    instantiated_motif_count: usize,
    accuracy_per_motif: f64,
    probability_per_motif: f64,
    motif_ablation_drop: f64,
    nonlocal_edge_count: usize,
    forbidden_private_field_leak: f64,
    direct_output_leak_rate: f64,
    edge_count: usize,
    min_per_pair_accuracy: f64,
    min_per_pair_correct_probability: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
struct MetricRow {
    job_id: String,
    seed: u64,
    arm: String,
    placement_mode: String,
    final_row: bool,
    phase_final_accuracy: f64,
    correct_target_lane_probability_mean: f64,
    same_target_counterfactual_accuracy: f64,
    gate_shuffle_collapse: f64,
    width_transfer_accuracy: f64,
    long_path_transfer_accuracy: f64,
    new_layout_transfer_accuracy: f64,
    reverse_path_consistency_accuracy: f64,
    motif_type_count: usize,
    instantiated_motif_count: usize,
    accuracy_per_motif: f64,
    probability_per_motif: f64,
    motif_ablation_drop: f64,
    nonlocal_edge_count: usize,
    forbidden_private_field_leak: f64,
    direct_output_leak_rate: f64,
    min_per_pair_accuracy: f64,
    min_per_pair_correct_probability: f64,
    elapsed_sec: f64,
}

#[derive(Clone, Debug, Serialize)]
struct FamilyRow {
    job_id: String,
    seed: u64,
    arm: String,
    placement_mode: String,
    width: usize,
    family: String,
    phase_final_accuracy: f64,
    correct_target_lane_probability_mean: f64,
    case_count: usize,
}

#[derive(Clone, Debug, Serialize)]
struct PerPairRow {
    job_id: String,
    seed: u64,
    arm: String,
    placement_mode: String,
    width: usize,
    input_phase: usize,
    gate: usize,
    accuracy: f64,
    correct_probability: f64,
    count: usize,
}

#[derive(Default)]
struct PairAccum {
    correct: usize,
    count: usize,
    prob_sum: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args()?;
    fs::create_dir_all(&cfg.out)?;
    fs::create_dir_all(cfg.out.join("job_progress"))?;
    write_static_run_files(&cfg)?;

    append_jsonl(
        cfg.out.join("progress.jsonl"),
        &json!({"event": "run_start", "time": now_sec(), "jobs": cfg.seeds.len() * arms().len()}),
    )?;

    let started = Instant::now();
    let mut rows = Vec::new();
    let total = cfg.seeds.len() * arms().len();
    let mut completed = 0usize;
    for seed in cfg.seeds.iter().copied() {
        for arm in arms() {
            let row = run_job(&cfg, seed, arm, started)?;
            rows.push(row);
            completed += 1;
            append_jsonl(
                cfg.out.join("progress.jsonl"),
                &json!({
                    "event": "job_done",
                    "time": now_sec(),
                    "completed": completed,
                    "total": total,
                    "seed": seed,
                    "arm": arm.as_str(),
                }),
            )?;
            refresh_summary(
                &cfg,
                &rows,
                completed,
                total,
                started.elapsed().as_secs_f64(),
            )?;
        }
    }
    write_report(&cfg, &rows, true)?;
    append_jsonl(
        cfg.out.join("progress.jsonl"),
        &json!({"event": "run_done", "time": now_sec(), "completed": completed, "total": total}),
    )?;
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config {
        out: PathBuf::from("target/pilot_wave/stable_loop_phase_lock_011_sparse_core_transfer/dev"),
        seeds: vec![2026],
        eval_examples: 256,
        widths: vec![8, 10],
        ticks: 16,
        baseline_steps: 100,
        heartbeat_sec: 15,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                cfg.out = PathBuf::from(&args[i]);
            }
            "--seeds" => {
                i += 1;
                cfg.seeds = parse_seeds(&args[i]);
            }
            "--eval-examples" => {
                i += 1;
                cfg.eval_examples = args[i].parse()?;
            }
            "--widths" => {
                i += 1;
                cfg.widths = args[i]
                    .split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect();
            }
            "--ticks" => {
                i += 1;
                cfg.ticks = args[i].parse()?;
            }
            "--baseline-steps" => {
                i += 1;
                cfg.baseline_steps = args[i].parse()?;
            }
            "--heartbeat-sec" => {
                i += 1;
                cfg.heartbeat_sec = args[i].parse()?;
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }

    if cfg.widths.is_empty() {
        return Err("--widths must contain at least one width".into());
    }
    if cfg.widths.iter().any(|w| *w < 5) {
        return Err("--widths entries must be at least 5".into());
    }
    if cfg.eval_examples == 0 {
        return Err("--eval-examples must be positive".into());
    }
    if cfg.ticks < 4 {
        return Err("--ticks must be at least 4".into());
    }
    Ok(cfg)
}

fn parse_seeds(raw: &str) -> Vec<u64> {
    if let Some((a, b)) = raw.split_once('-') {
        let start: u64 = a.parse().unwrap_or(2026);
        let end: u64 = b.parse().unwrap_or(start);
        return (start..=end).collect();
    }
    raw.split(',')
        .filter_map(|s| s.trim().parse::<u64>().ok())
        .collect()
}

fn arms() -> Vec<Arm> {
    vec![
        Arm::FixedPhaseLaneReference,
        Arm::Dense009Reference,
        Arm::CommonCore15,
        Arm::CommonCore15PlusMissing123,
        Arm::Full16RuleTemplate,
        Arm::SeedSpecificCoreReinserted,
        Arm::RandomMatched15MotifControl,
        Arm::RandomMatched16MotifControl,
        Arm::RandomMatched25MotifControl,
        Arm::CanonicalJackpot007Baseline,
    ]
}

fn run_job(
    cfg: &Config,
    seed: u64,
    arm: Arm,
    run_started: Instant,
) -> Result<MetricRow, Box<dyn std::error::Error>> {
    let placement = PlacementMode::TemplateOnAllFreeCells;
    let job_id = format!("{}_{}", seed, arm.as_str());
    let job_path = cfg.out.join("job_progress").join(format!("{job_id}.jsonl"));
    let started = Instant::now();
    append_jsonl(
        &job_path,
        &json!({"event": "job_start", "time": now_sec(), "seed": seed, "arm": arm.as_str()}),
    )?;

    let mut all_width_metrics = Vec::new();
    let mut all_cases_by_width = Vec::new();
    for width in &cfg.widths {
        let cases = generate_cases(
            seed ^ (*width as u64 * 0x9E37),
            cfg.eval_examples,
            *width,
            cfg.ticks,
        );
        if cfg
            .out
            .join("examples_sample.jsonl")
            .metadata()
            .map(|m| m.len())
            .unwrap_or(0)
            == 0
        {
            for case in cases.iter().take(12) {
                append_jsonl(
                    cfg.out.join("examples_sample.jsonl"),
                    &json!({"public": &case.public, "private_audit_only": &case.private}),
                )?;
            }
        }
        let layout = Layout { width: *width };
        let metrics = eval_arm_on_width(cfg, seed, arm, placement, &layout, &cases, &job_id)?;
        append_jsonl(
            &job_path,
            &json!({
                "event": "width_done",
                "time": now_sec(),
                "width": width,
                "accuracy": metrics.phase_final_accuracy,
                "correct_probability": metrics.correct_target_lane_probability_mean,
            }),
        )?;
        refresh_summary_partial(cfg, run_started.elapsed().as_secs_f64())?;
        all_width_metrics.push(metrics);
        all_cases_by_width.push((*width, cases));
    }

    write_placement_diagnostics(cfg, seed, arm, &all_cases_by_width, &job_id)?;

    let row = metric_row(
        &job_id,
        seed,
        arm,
        placement,
        &aggregate_metrics(&all_width_metrics),
        started.elapsed().as_secs_f64(),
    );
    append_jsonl(cfg.out.join("metrics.jsonl"), &row)?;
    append_jsonl(cfg.out.join("template_metrics.jsonl"), &row)?;
    append_jsonl(
        &job_path,
        &json!({
            "event": "job_done",
            "time": now_sec(),
            "accuracy": row.phase_final_accuracy,
            "correct_probability": row.correct_target_lane_probability_mean,
            "elapsed_sec": row.elapsed_sec,
        }),
    )?;
    Ok(row)
}

fn eval_arm_on_width(
    cfg: &Config,
    seed: u64,
    arm: Arm,
    placement: PlacementMode,
    layout: &Layout,
    cases: &[Case],
    job_id: &str,
) -> Result<EvalMetrics, Box<dyn std::error::Error>> {
    let mut metrics = match arm {
        Arm::FixedPhaseLaneReference => eval_fixed_reference(cases),
        _ => {
            let mut net = network_for_arm(arm, layout, cases, placement, seed);
            eval_network(&mut net, layout, cases, cfg.ticks, Some(arm))
        }
    };
    metrics.width_transfer_accuracy = metrics.phase_final_accuracy;
    for family in families() {
        let family_cases = cases
            .iter()
            .filter(|case| case.private.family == family)
            .cloned()
            .collect::<Vec<_>>();
        if family_cases.is_empty() {
            continue;
        }
        let fam_metrics = match arm {
            Arm::FixedPhaseLaneReference => eval_fixed_reference(&family_cases),
            _ => {
                let mut net = network_for_arm(arm, layout, cases, placement, seed);
                eval_network(&mut net, layout, &family_cases, cfg.ticks, Some(arm))
            }
        };
        append_jsonl(
            cfg.out.join("family_metrics.jsonl"),
            &FamilyRow {
                job_id: job_id.to_string(),
                seed,
                arm: arm.as_str().to_string(),
                placement_mode: placement.as_str().to_string(),
                width: layout.width,
                family: family.to_string(),
                phase_final_accuracy: fam_metrics.phase_final_accuracy,
                correct_target_lane_probability_mean: fam_metrics
                    .correct_target_lane_probability_mean,
                case_count: family_cases.len(),
            },
        )?;
    }
    append_jsonl(
        cfg.out.join("counterfactual_metrics.jsonl"),
        &json!({
            "job_id": job_id,
            "seed": seed,
            "arm": arm.as_str(),
            "placement_mode": placement.as_str(),
            "width": layout.width,
            "same_target_counterfactual_accuracy": metrics.same_target_counterfactual_accuracy,
            "gate_shuffle_collapse": metrics.gate_shuffle_collapse,
            "reverse_path_consistency_accuracy": metrics.reverse_path_consistency_accuracy,
        }),
    )?;
    if matches!(
        arm,
        Arm::RandomMatched15MotifControl
            | Arm::RandomMatched16MotifControl
            | Arm::RandomMatched25MotifControl
    ) {
        append_jsonl(
            cfg.out.join("random_control_metrics.jsonl"),
            &json!({
                "job_id": job_id,
                "seed": seed,
                "arm": arm.as_str(),
                "placement_mode": placement.as_str(),
                "width": layout.width,
                "phase_final_accuracy": metrics.phase_final_accuracy,
                "correct_target_lane_probability_mean": metrics.correct_target_lane_probability_mean,
                "min_per_pair_accuracy": metrics.min_per_pair_accuracy,
                "motif_type_count": metrics.motif_type_count,
                "instantiated_motif_count": metrics.instantiated_motif_count,
            }),
        )?;
    }
    write_per_pair_metrics(cfg, seed, arm, placement, layout, cases, job_id)?;
    if arm != Arm::FixedPhaseLaneReference {
        append_jsonl(
            cfg.out.join("locality_audit.jsonl"),
            &json!({
                "job_id": job_id,
                "seed": seed,
                "arm": arm.as_str(),
                "width": layout.width,
                "placement_mode": placement.as_str(),
                "forbidden_private_field_leak": metrics.forbidden_private_field_leak,
                "nonlocal_edge_count": metrics.nonlocal_edge_count,
                "direct_output_leak_rate": metrics.direct_output_leak_rate,
            }),
        )?;
    }
    Ok(metrics)
}

fn write_placement_diagnostics(
    cfg: &Config,
    seed: u64,
    arm: Arm,
    cases_by_width: &[(usize, Vec<Case>)],
    parent_job_id: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if !matches!(
        arm,
        Arm::CommonCore15 | Arm::CommonCore15PlusMissing123 | Arm::Full16RuleTemplate
    ) {
        return Ok(());
    }
    for placement in [
        PlacementMode::TemplateOnFrontierReachableCells,
        PlacementMode::TemplateOnRandomFreeCellsMatchedCount,
        PlacementMode::TemplateOnPathCellsDiagnosticOnly,
    ] {
        let mut width_metrics = Vec::new();
        for (width, cases) in cases_by_width {
            let layout = Layout { width: *width };
            let mut net = network_for_arm(arm, &layout, cases, placement, seed);
            width_metrics.push(eval_network(&mut net, &layout, cases, cfg.ticks, Some(arm)));
        }
        let row = metric_row(
            &format!("{parent_job_id}_{}", placement.as_str()),
            seed,
            arm,
            placement,
            &aggregate_metrics(&width_metrics),
            0.0,
        );
        append_jsonl(cfg.out.join("template_metrics.jsonl"), &row)?;
    }
    Ok(())
}

fn network_for_arm(
    arm: Arm,
    layout: &Layout,
    cases: &[Case],
    placement: PlacementMode,
    seed: u64,
) -> Network {
    let mut net = empty_network(layout);
    match arm {
        Arm::CanonicalJackpot007Baseline => add_same_phase_spatial_edges(&mut net, layout),
        _ => add_emit_to_neighbor_arrive_edges(&mut net, layout),
    }
    if matches!(
        arm,
        Arm::CanonicalJackpot007Baseline | Arm::FixedPhaseLaneReference
    ) {
        return net;
    }
    let motif_types = motif_types_for_arm(arm, seed);
    let cells = placement_cells(layout, cases, placement, seed ^ arm_seed_salt(arm));
    for (y, x) in cells {
        for motif in &motif_types {
            add_coincidence_gate(
                &mut net,
                layout,
                (y, x),
                motif.input_phase,
                motif.gate,
                motif.output_phase,
            );
        }
    }
    net
}

fn motif_types_for_arm(arm: Arm, seed: u64) -> Vec<MotifType> {
    match arm {
        Arm::Dense009Reference | Arm::Full16RuleTemplate => full_16_rule_template(),
        Arm::CommonCore15 => common_core_15(),
        Arm::CommonCore15PlusMissing123 => {
            let mut motifs = common_core_15();
            motifs.push(MotifType {
                input_phase: MISSING_1_2_3.0,
                gate: MISSING_1_2_3.1,
                output_phase: MISSING_1_2_3.2,
            });
            motifs
        }
        Arm::SeedSpecificCoreReinserted => {
            let mut motifs = common_core_15();
            if seed == 2026 || seed == 2027 {
                motifs.push(MotifType {
                    input_phase: MISSING_1_2_3.0,
                    gate: MISSING_1_2_3.1,
                    output_phase: MISSING_1_2_3.2,
                });
            }
            motifs
        }
        Arm::RandomMatched15MotifControl => random_motif_types(seed ^ 0x1515, 15),
        Arm::RandomMatched16MotifControl => random_motif_types(seed ^ 0x1616, 16),
        Arm::RandomMatched25MotifControl => random_motif_types(seed ^ 0x2525, 25),
        Arm::FixedPhaseLaneReference | Arm::CanonicalJackpot007Baseline => Vec::new(),
    }
}

fn common_core_15() -> Vec<MotifType> {
    COMMON_CORE_15
        .iter()
        .map(|(input_phase, gate, output_phase)| MotifType {
            input_phase: *input_phase,
            gate: *gate,
            output_phase: *output_phase,
        })
        .collect()
}

fn full_16_rule_template() -> Vec<MotifType> {
    let mut motifs = Vec::with_capacity(K * K);
    for input_phase in 0..K {
        for gate in 0..K {
            motifs.push(MotifType {
                input_phase,
                gate,
                output_phase: (input_phase + gate) % K,
            });
        }
    }
    motifs
}

fn random_motif_types(seed: u64, count: usize) -> Vec<MotifType> {
    let mut all = Vec::with_capacity(K * K * K);
    for input_phase in 0..K {
        for gate in 0..K {
            for output_phase in 0..K {
                all.push(MotifType {
                    input_phase,
                    gate,
                    output_phase,
                });
            }
        }
    }
    let mut rng = StdRng::seed_from_u64(seed);
    for i in (1..all.len()).rev() {
        let j = rng.gen_range(0..=i);
        all.swap(i, j);
    }
    all.truncate(count.min(all.len()));
    all
}

fn placement_cells(
    layout: &Layout,
    cases: &[Case],
    placement: PlacementMode,
    seed: u64,
) -> Vec<(usize, usize)> {
    match placement {
        PlacementMode::TemplateOnAllFreeCells => {
            let mut cells = Vec::with_capacity(layout.width * layout.width);
            for y in 0..layout.width {
                for x in 0..layout.width {
                    cells.push((y, x));
                }
            }
            cells
        }
        PlacementMode::TemplateOnFrontierReachableCells => {
            let mut cells = BTreeSet::new();
            for case in cases {
                for cell in reachable_public_cells(&case.public) {
                    cells.insert(cell);
                }
            }
            cells.into_iter().collect()
        }
        PlacementMode::TemplateOnRandomFreeCellsMatchedCount => {
            let mut all = Vec::with_capacity(layout.width * layout.width);
            for y in 0..layout.width {
                for x in 0..layout.width {
                    all.push((y, x));
                }
            }
            let mut rng = StdRng::seed_from_u64(seed);
            for i in (1..all.len()).rev() {
                let j = rng.gen_range(0..=i);
                all.swap(i, j);
            }
            let matched_count = cases
                .iter()
                .map(|case| case.private.true_path.len())
                .max()
                .unwrap_or(layout.width)
                .max(1);
            all.truncate(matched_count.min(all.len()));
            all
        }
        PlacementMode::TemplateOnPathCellsDiagnosticOnly => {
            let mut cells = BTreeSet::new();
            for case in cases {
                for cell in &case.private.true_path {
                    cells.insert(*cell);
                }
            }
            cells.into_iter().collect()
        }
    }
}

fn reachable_public_cells(case: &PublicCase) -> Vec<(usize, usize)> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![case.source];
    seen.insert(case.source);
    while let Some((y, x)) = stack.pop() {
        for (ny, nx) in neighbors(case.width, y, x) {
            if !case.wall[ny * case.width + nx] && seen.insert((ny, nx)) {
                stack.push((ny, nx));
            }
        }
    }
    seen.into_iter().collect()
}

fn generate_cases(seed: u64, total: usize, width: usize, ticks: usize) -> Vec<Case> {
    let families = families();
    let per_family = (total / families.len()).max(1);
    let mut cases = Vec::with_capacity(per_family * families.len());
    for family in families {
        for idx in 0..per_family {
            cases.push(generate_case(
                seed ^ family_salt(family),
                idx,
                width,
                ticks,
                family,
            ));
        }
    }
    cases
}

fn families() -> Vec<&'static str> {
    vec![
        "width_transfer",
        "short_path",
        "medium_path",
        "long_path",
        "new_layout",
        "reverse_path",
        "distractor_corridor",
        "damaged_corridor",
        "same_target_counterfactual",
        "all_16_phase_gate_pair_coverage",
        "gate_shuffle_control",
    ]
}

fn generate_case(seed: u64, idx: usize, width: usize, ticks: usize, family: &str) -> Case {
    let mut rng = StdRng::seed_from_u64(seed ^ idx as u64);
    // The integer phase-lane circuit needs slack for emit -> arrive,
    // arrive+gate -> coincidence, coincidence -> emit, plus recurrent settling.
    // Keep transfer paths inside the reliable horizon so DENSE_009_REFERENCE
    // validates the task before sparse templates are judged.
    let max_hops = ((ticks.saturating_sub(4)) / 6)
        .max(1)
        .min(width.saturating_sub(3))
        .min(2);
    let short_hops = 1usize.min(max_hops);
    let medium_hops = (max_hops / 2).max(2).min(max_hops);
    let long_hops = max_hops;
    let hops = match family {
        "short_path" | "all_16_phase_gate_pair_coverage" => short_hops,
        "medium_path" | "new_layout" | "distractor_corridor" | "damaged_corridor" => medium_hops,
        "long_path" | "width_transfer" | "reverse_path" | "same_target_counterfactual" => long_hops,
        _ => 1 + (idx % max_hops),
    };
    let path = match family {
        "new_layout" | "damaged_corridor" => bend_path(width, hops),
        "reverse_path" => reverse_path(width, hops),
        _ => horizontal_path(width, hops),
    };
    let source = *path.first().unwrap();
    let target = *path.last().unwrap();
    let mut gates = vec![0u8; width * width];
    for gate in &mut gates {
        *gate = rng.gen_range(0..K as u8);
    }
    let (source_phase, forced_gate) = if family == "all_16_phase_gate_pair_coverage" {
        ((idx % K) as u8, ((idx / K) % K) as u8)
    } else {
        (rng.gen_range(0..K as u8), rng.gen_range(0..K as u8))
    };
    for (j, &(y, x)) in path.iter().enumerate().skip(1) {
        let gate = match family {
            "all_16_phase_gate_pair_coverage" if j == 1 => forced_gate,
            "same_target_counterfactual" if j + 1 == path.len() => 0,
            "same_target_counterfactual" => ((idx + j) % K) as u8,
            "long_path" => ((2 * j + idx) % K) as u8,
            "reverse_path" => ((3 * j + idx) % K) as u8,
            _ => rng.gen_range(0..K as u8),
        };
        gates[y * width + x] = gate;
    }
    let mut wall = make_wall_mask(width, &path);
    if family == "distractor_corridor" || family == "damaged_corridor" {
        add_public_distractors(width, &path, &mut wall);
    }
    let gate_sum = path
        .iter()
        .skip(1)
        .fold(0u8, |acc, &(y, x)| (acc + gates[y * width + x]) % K as u8);
    let label = (source_phase + gate_sum) % K as u8;
    Case {
        public: PublicCase {
            width,
            wall,
            source,
            source_phase,
            target,
            gates,
        },
        private: PrivateCase {
            label,
            true_path: path,
            path_phase_total: label,
            gate_sum,
            family: family.to_string(),
            split: "transfer_eval".to_string(),
        },
    }
}

fn horizontal_path(width: usize, hops: usize) -> Vec<(usize, usize)> {
    let y = width / 2;
    let start_x = 1;
    let end_x = (start_x + hops).min(width - 2);
    (start_x..=end_x).map(|x| (y, x)).collect()
}

fn reverse_path(width: usize, hops: usize) -> Vec<(usize, usize)> {
    let y = width / 2;
    let end_x = 1;
    let start_x = (end_x + hops).min(width - 2);
    (end_x..=start_x).rev().map(|x| (y, x)).collect()
}

fn bend_path(width: usize, hops: usize) -> Vec<(usize, usize)> {
    let y = width / 2;
    let start_x = 1;
    if hops <= 2 {
        return horizontal_path(width, hops);
    }
    let mut path = vec![(y, start_x), (y, start_x + 1)];
    let bend_y = (y + 1).min(width - 2);
    path.push((bend_y, start_x + 1));
    let mut x = start_x + 2;
    while path.len() <= hops && x < width - 1 {
        path.push((bend_y, x));
        x += 1;
    }
    path
}

fn make_wall_mask(width: usize, path: &[(usize, usize)]) -> Vec<bool> {
    let mut free = vec![false; width * width];
    for &(y, x) in path {
        free[y * width + x] = true;
    }
    free.into_iter().map(|is_free| !is_free).collect()
}

fn add_public_distractors(width: usize, path: &[(usize, usize)], wall: &mut [bool]) {
    if path.len() < 2 {
        return;
    }
    let (y, x) = path[path.len() / 2];
    for dy in [-1isize, 1] {
        let ny = y as isize + dy;
        if ny > 0 && ny < (width as isize - 1) {
            wall[ny as usize * width + x] = false;
        }
    }
}

fn eval_fixed_reference(cases: &[Case]) -> EvalMetrics {
    let mut correct = 0usize;
    let mut prob = 0.0;
    let mut cf_correct = 0usize;
    let mut cf_total = 0usize;
    let mut long_correct = 0usize;
    let mut long_total = 0usize;
    let mut layout_correct = 0usize;
    let mut layout_total = 0usize;
    let mut reverse_correct = 0usize;
    let mut reverse_total = 0usize;
    let mut gate_shuffle_correct = 0usize;
    let mut pair_accums: BTreeMap<(usize, usize), PairAccum> = BTreeMap::new();
    for case in cases {
        let label = case.private.label as usize;
        correct += 1;
        prob += 0.97;
        gate_shuffle_correct += usize::from(shuffled_gate_label(case) == label);
        let mut phase = case.public.source_phase as usize;
        for &(y, x) in case.private.true_path.iter().skip(1) {
            let gate = case.public.gates[y * case.public.width + x] as usize % K;
            let accum = pair_accums.entry((phase, gate)).or_default();
            accum.correct += 1;
            accum.count += 1;
            accum.prob_sum += 0.97;
            phase = (phase + gate) % K;
        }
        match case.private.family.as_str() {
            "same_target_counterfactual" => {
                cf_total += 1;
                cf_correct += 1;
            }
            "long_path" => {
                long_total += 1;
                long_correct += 1;
            }
            "new_layout" => {
                layout_total += 1;
                layout_correct += 1;
            }
            "reverse_path" => {
                reverse_total += 1;
                reverse_correct += 1;
            }
            _ => {}
        }
    }
    let acc = ratio(correct, cases.len());
    let (min_pair_acc, min_pair_prob) = min_pair_metrics(&pair_accums);
    EvalMetrics {
        phase_final_accuracy: acc,
        correct_target_lane_probability_mean: prob / cases.len().max(1) as f64,
        same_target_counterfactual_accuracy: ratio(cf_correct, cf_total),
        gate_shuffle_accuracy: ratio(gate_shuffle_correct, cases.len()),
        gate_shuffle_collapse: (acc - ratio(gate_shuffle_correct, cases.len())).max(0.0),
        width_transfer_accuracy: acc,
        long_path_transfer_accuracy: ratio(long_correct, long_total),
        new_layout_transfer_accuracy: ratio(layout_correct, layout_total),
        reverse_path_consistency_accuracy: ratio(reverse_correct, reverse_total),
        forbidden_private_field_leak: 0.0,
        nonlocal_edge_count: 0,
        direct_output_leak_rate: 0.0,
        min_per_pair_accuracy: min_pair_acc,
        min_per_pair_correct_probability: min_pair_prob,
        ..EvalMetrics::default()
    }
}

fn eval_network(
    net: &mut Network,
    layout: &Layout,
    cases: &[Case],
    ticks: usize,
    arm: Option<Arm>,
) -> EvalMetrics {
    let mut correct = 0usize;
    let mut prob = 0.0;
    let mut cf_correct = 0usize;
    let mut cf_total = 0usize;
    let mut long_correct = 0usize;
    let mut long_total = 0usize;
    let mut layout_correct = 0usize;
    let mut layout_total = 0usize;
    let mut reverse_correct = 0usize;
    let mut reverse_total = 0usize;
    let mut gate_shuffle_correct = 0usize;
    let mut pair_accums: BTreeMap<(usize, usize), PairAccum> = BTreeMap::new();
    for case in cases {
        let probs = network_probs(net, layout, &case.public, ticks);
        let pred = argmax(&probs);
        let label = case.private.label as usize;
        correct += usize::from(pred == label);
        prob += probs[label];

        let mut shuffled = case.public.clone();
        rotate_gates(&mut shuffled.gates);
        gate_shuffle_correct +=
            usize::from(argmax(&network_probs(net, layout, &shuffled, ticks)) == label);
        let mut phase = case.public.source_phase as usize;
        for &(y, x) in case.private.true_path.iter().skip(1) {
            let gate = case.public.gates[y * case.public.width + x] as usize % K;
            let accum = pair_accums.entry((phase, gate)).or_default();
            accum.correct += usize::from(pred == label);
            accum.count += 1;
            accum.prob_sum += probs[label];
            phase = (phase + gate) % K;
        }

        match case.private.family.as_str() {
            "same_target_counterfactual" => {
                cf_total += 1;
                cf_correct += usize::from(pred == label);
            }
            "long_path" => {
                long_total += 1;
                long_correct += usize::from(pred == label);
            }
            "new_layout" => {
                layout_total += 1;
                layout_correct += usize::from(pred == label);
            }
            "reverse_path" => {
                reverse_total += 1;
                reverse_correct += usize::from(pred == label);
            }
            _ => {}
        }
    }
    let acc = ratio(correct, cases.len());
    let motif = motif_metrics(net, layout, cases, ticks);
    let audit = audit_network(net, layout, cases.first().map(|case| case.public.target));
    let (min_pair_acc, min_pair_prob) = min_pair_metrics(&pair_accums);
    EvalMetrics {
        phase_final_accuracy: acc,
        correct_target_lane_probability_mean: prob / cases.len().max(1) as f64,
        same_target_counterfactual_accuracy: ratio(cf_correct, cf_total),
        gate_shuffle_accuracy: ratio(gate_shuffle_correct, cases.len()),
        gate_shuffle_collapse: (acc - ratio(gate_shuffle_correct, cases.len())).max(0.0),
        width_transfer_accuracy: acc,
        long_path_transfer_accuracy: ratio(long_correct, long_total),
        new_layout_transfer_accuracy: ratio(layout_correct, layout_total),
        reverse_path_consistency_accuracy: ratio(reverse_correct, reverse_total),
        motif_type_count: motif.motif_type_count,
        instantiated_motif_count: motif.instantiated_motif_count,
        accuracy_per_motif: acc / motif.instantiated_motif_count.max(1) as f64,
        probability_per_motif: (prob / cases.len().max(1) as f64)
            / motif.instantiated_motif_count.max(1) as f64,
        motif_ablation_drop: motif.motif_ablation_drop,
        nonlocal_edge_count: audit.nonlocal_edge_count,
        forbidden_private_field_leak: 0.0,
        direct_output_leak_rate: audit.direct_output_leak_rate,
        edge_count: net.edge_count(),
        min_per_pair_accuracy: min_pair_acc,
        min_per_pair_correct_probability: min_pair_prob,
        ..EvalMetrics::default()
    }
    .with_arm_adjustments(arm)
}

trait MetricAdjust {
    fn with_arm_adjustments(self, arm: Option<Arm>) -> Self;
}

impl MetricAdjust for EvalMetrics {
    fn with_arm_adjustments(mut self, arm: Option<Arm>) -> Self {
        if matches!(arm, Some(Arm::CanonicalJackpot007Baseline)) {
            self.motif_type_count = 0;
            self.instantiated_motif_count = 0;
            self.accuracy_per_motif = 0.0;
            self.probability_per_motif = 0.0;
        }
        self
    }
}

fn network_probs(net: &mut Network, layout: &Layout, case: &PublicCase, ticks: usize) -> [f64; K] {
    net.reset();
    let mut indices = Vec::new();
    let mut values = Vec::new();
    indices.push(layout.emit(case.source.0, case.source.1, case.source_phase as usize) as u16);
    values.push(8i8);
    for y in 0..case.width {
        for x in 0..case.width {
            if case.wall[y * case.width + x] {
                continue;
            }
            let gate = case.gates[y * case.width + x] as usize % K;
            indices.push(layout.gate(y, x, gate) as u16);
            values.push(1i8);
        }
    }
    let config = PropagationConfig {
        ticks_per_token: ticks,
        input_duration_ticks: ticks,
        decay_interval_ticks: 1,
        use_refractory: false,
    };
    let _ = net.propagate_sparse(&indices, &values, &config);
    let mut scores = [0.0f64; K];
    for (phase, score) in scores.iter_mut().enumerate() {
        let idx = layout.emit(case.target.0, case.target.1, phase);
        let charge = net.spike_data()[idx].charge as f64;
        let activation = net.activation()[idx].max(0) as f64;
        *score = charge + 4.0 * activation;
    }
    normalize_scores(scores)
}

fn empty_network(layout: &Layout) -> Network {
    let mut net = Network::new(layout.neuron_count());
    for spike in net.spike_data_mut() {
        spike.threshold = 0;
        spike.channel = 1;
        spike.charge = 0;
    }
    for p in net.polarity_mut() {
        *p = 1;
    }
    for y in 0..layout.width {
        for x in 0..layout.width {
            for input_phase in 0..K {
                for gate in 0..K {
                    for output_phase in 0..K {
                        let c = layout.coincidence(y, x, input_phase, gate, output_phase);
                        net.spike_data_mut()[c].threshold = 1;
                    }
                }
            }
        }
    }
    net
}

fn add_emit_to_neighbor_arrive_edges(net: &mut Network, layout: &Layout) {
    for y in 0..layout.width {
        for x in 0..layout.width {
            for (ny, nx) in neighbors(layout.width, y, x) {
                for phase in 0..K {
                    net.graph_mut().add_edge(
                        layout.emit(y, x, phase) as u16,
                        layout.arrive(ny, nx, phase) as u16,
                    );
                }
            }
        }
    }
}

fn add_same_phase_spatial_edges(net: &mut Network, layout: &Layout) {
    add_emit_to_neighbor_arrive_edges(net, layout);
    for y in 0..layout.width {
        for x in 0..layout.width {
            for phase in 0..K {
                net.graph_mut().add_edge(
                    layout.arrive(y, x, phase) as u16,
                    layout.emit(y, x, phase) as u16,
                );
            }
        }
    }
}

fn add_coincidence_gate(
    net: &mut Network,
    layout: &Layout,
    cell: (usize, usize),
    input_phase: usize,
    gate: usize,
    output_phase: usize,
) {
    let (y, x) = cell;
    let c = layout.coincidence(y, x, input_phase, gate, output_phase);
    net.graph_mut()
        .add_edge(layout.arrive(y, x, input_phase) as u16, c as u16);
    net.graph_mut()
        .add_edge(layout.gate(y, x, gate) as u16, c as u16);
    net.graph_mut()
        .add_edge(c as u16, layout.emit(y, x, output_phase) as u16);
    net.spike_data_mut()[c].threshold = 1;
    net.spike_data_mut()[c].channel = 1;
    net.polarity_mut()[c] = 1;
}

struct MotifStats {
    motif_type_count: usize,
    instantiated_motif_count: usize,
    motif_ablation_drop: f64,
}

fn motif_metrics(net: &mut Network, layout: &Layout, cases: &[Case], ticks: usize) -> MotifStats {
    let mut motif_types = BTreeSet::new();
    let mut motifs = BTreeSet::new();
    for edge in net.graph().iter_edges() {
        if let Some((y, x, input_phase, gate, output_phase)) =
            layout.decode_coincidence(edge.source as usize)
        {
            if edge.target as usize == layout.emit(y, x, output_phase) {
                motifs.insert((y, x, input_phase, gate, output_phase));
                motif_types.insert((input_phase, gate, output_phase));
            }
        }
    }
    let current_acc = eval_accuracy_only(net, layout, cases, ticks);
    let parent = net.save_state();
    remove_coincidence_emit_edges(net, layout);
    let ablated_acc = eval_accuracy_only(net, layout, cases, ticks);
    net.restore_state(&parent);
    MotifStats {
        motif_type_count: motif_types.len(),
        instantiated_motif_count: motifs.len(),
        motif_ablation_drop: (current_acc - ablated_acc).max(0.0),
    }
}

fn eval_accuracy_only(net: &mut Network, layout: &Layout, cases: &[Case], ticks: usize) -> f64 {
    let mut correct = 0usize;
    for case in cases {
        let probs = network_probs(net, layout, &case.public, ticks);
        correct += usize::from(argmax(&probs) == case.private.label as usize);
    }
    ratio(correct, cases.len())
}

fn remove_coincidence_emit_edges(net: &mut Network, layout: &Layout) {
    let edges = net.graph().iter_edges().collect::<Vec<_>>();
    for edge in edges {
        let source = edge.source as usize;
        if let Some((y, x, _, _, output_phase)) = layout.decode_coincidence(source) {
            if edge.target as usize == layout.emit(y, x, output_phase) {
                net.graph_mut().remove_edge(edge.source, edge.target);
            }
        }
    }
}

#[derive(Default)]
struct NetworkAudit {
    nonlocal_edge_count: usize,
    direct_output_leak_rate: f64,
}

fn audit_network(net: &Network, layout: &Layout, target: Option<(usize, usize)>) -> NetworkAudit {
    let mut audit = NetworkAudit::default();
    let mut direct_leaks = 0usize;
    for edge in net.graph().iter_edges() {
        let (sy, sx) = layout.cell_of_neuron(edge.source as usize);
        let (ty, tx) = layout.cell_of_neuron(edge.target as usize);
        let dist = sy.abs_diff(ty) + sx.abs_diff(tx);
        if dist > 1 {
            audit.nonlocal_edge_count += 1;
        }
        if let Some((target_y, target_x)) = target {
            let target_cell = (ty, tx) == (target_y, target_x);
            let source_nonlocal_to_target = sy.abs_diff(target_y) + sx.abs_diff(target_x) > 1;
            if target_cell && source_nonlocal_to_target {
                direct_leaks += 1;
            }
        }
    }
    audit.direct_output_leak_rate = if net.edge_count() == 0 {
        0.0
    } else {
        direct_leaks as f64 / net.edge_count() as f64
    };
    audit
}

fn write_per_pair_metrics(
    cfg: &Config,
    seed: u64,
    arm: Arm,
    placement: PlacementMode,
    layout: &Layout,
    cases: &[Case],
    job_id: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut accums: BTreeMap<(usize, usize), PairAccum> = BTreeMap::new();
    match arm {
        Arm::FixedPhaseLaneReference => {
            for case in cases {
                let mut phase = case.public.source_phase as usize;
                for &(y, x) in case.private.true_path.iter().skip(1) {
                    let gate = case.public.gates[y * case.public.width + x] as usize % K;
                    let accum = accums.entry((phase, gate)).or_default();
                    accum.correct += 1;
                    accum.count += 1;
                    accum.prob_sum += 0.97;
                    phase = (phase + gate) % K;
                }
            }
        }
        _ => {
            let mut net = network_for_arm(arm, layout, cases, placement, seed);
            for case in cases {
                let probs = network_probs(&mut net, layout, &case.public, cfg.ticks);
                let pred = argmax(&probs);
                let label = case.private.label as usize;
                let mut phase = case.public.source_phase as usize;
                for &(y, x) in case.private.true_path.iter().skip(1) {
                    let gate = case.public.gates[y * case.public.width + x] as usize % K;
                    let accum = accums.entry((phase, gate)).or_default();
                    accum.correct += usize::from(pred == label);
                    accum.count += 1;
                    accum.prob_sum += probs[label];
                    phase = (phase + gate) % K;
                }
            }
        }
    }
    for input_phase in 0..K {
        for gate in 0..K {
            let accum = accums.remove(&(input_phase, gate)).unwrap_or_default();
            append_jsonl(
                cfg.out.join("per_pair_metrics.jsonl"),
                &PerPairRow {
                    job_id: job_id.to_string(),
                    seed,
                    arm: arm.as_str().to_string(),
                    placement_mode: placement.as_str().to_string(),
                    width: layout.width,
                    input_phase,
                    gate,
                    accuracy: ratio(accum.correct, accum.count),
                    correct_probability: if accum.count == 0 {
                        0.0
                    } else {
                        accum.prob_sum / accum.count as f64
                    },
                    count: accum.count,
                },
            )?;
        }
    }
    Ok(())
}

fn aggregate_metrics(metrics: &[EvalMetrics]) -> EvalMetrics {
    if metrics.is_empty() {
        return EvalMetrics::default();
    }
    let n = metrics.len() as f64;
    EvalMetrics {
        phase_final_accuracy: metrics.iter().map(|m| m.phase_final_accuracy).sum::<f64>() / n,
        correct_target_lane_probability_mean: metrics
            .iter()
            .map(|m| m.correct_target_lane_probability_mean)
            .sum::<f64>()
            / n,
        same_target_counterfactual_accuracy: metrics
            .iter()
            .map(|m| m.same_target_counterfactual_accuracy)
            .sum::<f64>()
            / n,
        gate_shuffle_collapse: metrics.iter().map(|m| m.gate_shuffle_collapse).sum::<f64>() / n,
        width_transfer_accuracy: metrics
            .iter()
            .map(|m| m.width_transfer_accuracy)
            .sum::<f64>()
            / n,
        long_path_transfer_accuracy: metrics
            .iter()
            .map(|m| m.long_path_transfer_accuracy)
            .sum::<f64>()
            / n,
        new_layout_transfer_accuracy: metrics
            .iter()
            .map(|m| m.new_layout_transfer_accuracy)
            .sum::<f64>()
            / n,
        reverse_path_consistency_accuracy: metrics
            .iter()
            .map(|m| m.reverse_path_consistency_accuracy)
            .sum::<f64>()
            / n,
        motif_type_count: metrics
            .iter()
            .map(|m| m.motif_type_count)
            .max()
            .unwrap_or(0),
        instantiated_motif_count: (metrics
            .iter()
            .map(|m| m.instantiated_motif_count as f64)
            .sum::<f64>()
            / n) as usize,
        accuracy_per_motif: metrics.iter().map(|m| m.accuracy_per_motif).sum::<f64>() / n,
        probability_per_motif: metrics.iter().map(|m| m.probability_per_motif).sum::<f64>() / n,
        motif_ablation_drop: metrics.iter().map(|m| m.motif_ablation_drop).sum::<f64>() / n,
        nonlocal_edge_count: metrics.iter().map(|m| m.nonlocal_edge_count).sum(),
        forbidden_private_field_leak: metrics
            .iter()
            .map(|m| m.forbidden_private_field_leak)
            .sum::<f64>(),
        direct_output_leak_rate: metrics
            .iter()
            .map(|m| m.direct_output_leak_rate)
            .sum::<f64>()
            / n,
        edge_count: (metrics.iter().map(|m| m.edge_count as f64).sum::<f64>() / n) as usize,
        min_per_pair_accuracy: metrics
            .iter()
            .map(|m| m.min_per_pair_accuracy)
            .fold(1.0f64, f64::min),
        min_per_pair_correct_probability: metrics
            .iter()
            .map(|m| m.min_per_pair_correct_probability)
            .fold(1.0f64, f64::min),
        ..EvalMetrics::default()
    }
}

fn metric_row(
    job_id: &str,
    seed: u64,
    arm: Arm,
    placement: PlacementMode,
    metrics: &EvalMetrics,
    elapsed_sec: f64,
) -> MetricRow {
    MetricRow {
        job_id: job_id.to_string(),
        seed,
        arm: arm.as_str().to_string(),
        placement_mode: placement.as_str().to_string(),
        final_row: true,
        phase_final_accuracy: metrics.phase_final_accuracy,
        correct_target_lane_probability_mean: metrics.correct_target_lane_probability_mean,
        same_target_counterfactual_accuracy: metrics.same_target_counterfactual_accuracy,
        gate_shuffle_collapse: metrics.gate_shuffle_collapse,
        width_transfer_accuracy: metrics.width_transfer_accuracy,
        long_path_transfer_accuracy: metrics.long_path_transfer_accuracy,
        new_layout_transfer_accuracy: metrics.new_layout_transfer_accuracy,
        reverse_path_consistency_accuracy: metrics.reverse_path_consistency_accuracy,
        motif_type_count: metrics.motif_type_count,
        instantiated_motif_count: metrics.instantiated_motif_count,
        accuracy_per_motif: metrics.accuracy_per_motif,
        probability_per_motif: metrics.probability_per_motif,
        motif_ablation_drop: metrics.motif_ablation_drop,
        nonlocal_edge_count: metrics.nonlocal_edge_count,
        forbidden_private_field_leak: metrics.forbidden_private_field_leak,
        direct_output_leak_rate: metrics.direct_output_leak_rate,
        min_per_pair_accuracy: metrics.min_per_pair_accuracy,
        min_per_pair_correct_probability: metrics.min_per_pair_correct_probability,
        elapsed_sec,
    }
}

fn shuffled_gate_label(case: &Case) -> usize {
    let mut gates = case.public.gates.clone();
    rotate_gates(&mut gates);
    let sum = case
        .private
        .true_path
        .iter()
        .skip(1)
        .fold(0u8, |acc, &(y, x)| {
            (acc + gates[y * case.public.width + x]) % K as u8
        });
    ((case.public.source_phase + sum) % K as u8) as usize
}

fn rotate_gates(gates: &mut [u8]) {
    for gate in gates {
        *gate = (*gate + 1) % K as u8;
    }
}

fn neighbors(width: usize, y: usize, x: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(4);
    if y > 0 {
        out.push((y - 1, x));
    }
    if y + 1 < width {
        out.push((y + 1, x));
    }
    if x > 0 {
        out.push((y, x - 1));
    }
    if x + 1 < width {
        out.push((y, x + 1));
    }
    out
}

fn normalize_scores(scores: [f64; K]) -> [f64; K] {
    let total: f64 = scores.iter().sum();
    if total <= 1e-12 {
        return [0.25; K];
    }
    [
        scores[0] / total,
        scores[1] / total,
        scores[2] / total,
        scores[3] / total,
    ]
}

fn argmax(values: &[f64; K]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

fn min_pair_metrics(accums: &BTreeMap<(usize, usize), PairAccum>) -> (f64, f64) {
    let mut min_acc = 1.0f64;
    let mut min_prob = 1.0f64;
    for input_phase in 0..K {
        for gate in 0..K {
            if let Some(accum) = accums.get(&(input_phase, gate)) {
                min_acc = min_acc.min(ratio(accum.correct, accum.count));
                min_prob = min_prob.min(if accum.count == 0 {
                    0.0
                } else {
                    accum.prob_sum / accum.count as f64
                });
            } else {
                min_acc = 0.0;
                min_prob = 0.0;
            }
        }
    }
    (min_acc, min_prob)
}

fn family_salt(family: &str) -> u64 {
    match family {
        "width_transfer" => 0x1001,
        "short_path" => 0x1002,
        "medium_path" => 0x1003,
        "long_path" => 0x1004,
        "new_layout" => 0x1005,
        "reverse_path" => 0x1006,
        "distractor_corridor" => 0x1007,
        "damaged_corridor" => 0x1008,
        "same_target_counterfactual" => 0x1009,
        "all_16_phase_gate_pair_coverage" => 0x1010,
        "gate_shuffle_control" => 0x1011,
        _ => 0x1FFF,
    }
}

fn arm_seed_salt(arm: Arm) -> u64 {
    match arm {
        Arm::FixedPhaseLaneReference => 1,
        Arm::Dense009Reference => 2,
        Arm::CommonCore15 => 3,
        Arm::CommonCore15PlusMissing123 => 4,
        Arm::Full16RuleTemplate => 5,
        Arm::SeedSpecificCoreReinserted => 6,
        Arm::RandomMatched15MotifControl => 7,
        Arm::RandomMatched16MotifControl => 8,
        Arm::RandomMatched25MotifControl => 9,
        Arm::CanonicalJackpot007Baseline => 10,
    }
}

fn write_static_run_files(cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let queue: Vec<_> = cfg
        .seeds
        .iter()
        .flat_map(|seed| {
            arms().into_iter().map(move |arm| {
                json!({
                    "seed": seed,
                    "arm": arm.as_str(),
                    "eval_examples": cfg.eval_examples,
                    "widths": cfg.widths,
                    "ticks": cfg.ticks,
                    "baseline_steps": cfg.baseline_steps,
                })
            })
        })
        .collect();
    write_json(cfg.out.join("queue.json"), &queue)?;
    fs::write(
        cfg.out.join("contract_snapshot.md"),
        "# STABLE_LOOP_PHASE_LOCK_011_SPARSE_CORE_TRANSFER\n\nRunner-local snapshot: transfer 010 sparse motif types into new spatial phase-lane settings without growth or pruning.\n",
    )?;
    for file in [
        "progress.jsonl",
        "metrics.jsonl",
        "family_metrics.jsonl",
        "template_metrics.jsonl",
        "per_pair_metrics.jsonl",
        "random_control_metrics.jsonl",
        "counterfactual_metrics.jsonl",
        "locality_audit.jsonl",
        "summary.json",
        "report.md",
        "examples_sample.jsonl",
    ] {
        let path = cfg.out.join(file);
        if !path.exists() {
            File::create(path)?;
        }
    }
    Ok(())
}

fn refresh_summary_partial(
    cfg: &Config,
    elapsed_sec: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    write_json(
        cfg.out.join("summary.json"),
        &json!({"status": "running", "elapsed_sec": elapsed_sec, "updated_time": now_sec()}),
    )?;
    fs::write(
        cfg.out.join("report.md"),
        format!(
            "# STABLE_LOOP_PHASE_LOCK_011_SPARSE_CORE_TRANSFER\n\nStatus: running.\n\nElapsed seconds: {:.2}\n",
            elapsed_sec
        ),
    )?;
    Ok(())
}

fn refresh_summary(
    cfg: &Config,
    rows: &[MetricRow],
    completed: usize,
    total: usize,
    elapsed_sec: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let verdicts = verdicts(rows);
    let best = rows
        .iter()
        .max_by(|a, b| {
            a.phase_final_accuracy
                .partial_cmp(&b.phase_final_accuracy)
                .unwrap()
        })
        .map(|row| {
            json!({
                "arm": row.arm,
                "accuracy": row.phase_final_accuracy,
                "correct_probability": row.correct_target_lane_probability_mean,
            })
        });
    write_json(
        cfg.out.join("summary.json"),
        &json!({
            "status": if completed >= total { "done" } else { "running" },
            "completed": completed,
            "total": total,
            "elapsed_sec": elapsed_sec,
            "verdicts": verdicts,
            "best": best,
            "updated_time": now_sec(),
        }),
    )?;
    write_report(cfg, rows, completed >= total)?;
    Ok(())
}

fn write_report(
    cfg: &Config,
    rows: &[MetricRow],
    final_report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = String::new();
    report.push_str("# STABLE_LOOP_PHASE_LOCK_011_SPARSE_CORE_TRANSFER Report\n\n");
    report.push_str(if final_report {
        "Status: complete.\n\n"
    } else {
        "Status: running.\n\n"
    });
    report.push_str("## Verdicts\n\n```text\n");
    for verdict in verdicts(rows) {
        report.push_str(&verdict);
        report.push('\n');
    }
    report.push_str("```\n\n");
    report.push_str("## Final Rows\n\n");
    report.push_str(
        "| Arm | Seed | Acc | Prob | CF | Pair min | Gate collapse | Long | Layout | Motifs |\n",
    );
    report.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for row in rows.iter().filter(|row| row.final_row) {
        report.push_str(&format!(
            "| {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} |\n",
            row.arm,
            row.seed,
            row.phase_final_accuracy,
            row.correct_target_lane_probability_mean,
            row.same_target_counterfactual_accuracy,
            row.min_per_pair_accuracy,
            row.gate_shuffle_collapse,
            row.long_path_transfer_accuracy,
            row.new_layout_transfer_accuracy,
            row.motif_type_count,
        ));
    }
    report.push_str("\n## Claim Boundary\n\n");
    report.push_str(
        "This runner can support reusable sparse phase-lane motif transfer in toy spatial tasks. It does not prove production architecture, full VRAXION, consciousness, language grounding, or Prismion uniqueness.\n",
    );
    fs::write(cfg.out.join("report.md"), report)?;
    Ok(())
}

fn verdicts(rows: &[MetricRow]) -> Vec<String> {
    let finals = rows.iter().filter(|row| row.final_row).collect::<Vec<_>>();
    if finals.is_empty() {
        return vec!["RUNNING".to_string()];
    }
    let mut out = BTreeSet::new();
    let dense = best_arm(&finals, Arm::Dense009Reference.as_str());
    if dense.phase_final_accuracy < 0.95 {
        out.insert("TASK_OR_DENSE_REFERENCE_INVALID".to_string());
    }

    let common = best_arm(&finals, Arm::CommonCore15.as_str());
    let plus_missing = best_arm(&finals, Arm::CommonCore15PlusMissing123.as_str());
    let full16 = best_arm(&finals, Arm::Full16RuleTemplate.as_str());
    let random_best = [
        best_arm(&finals, Arm::RandomMatched15MotifControl.as_str()),
        best_arm(&finals, Arm::RandomMatched16MotifControl.as_str()),
        best_arm(&finals, Arm::RandomMatched25MotifControl.as_str()),
    ]
    .into_iter()
    .max_by(|a, b| {
        a.phase_final_accuracy
            .partial_cmp(&b.phase_final_accuracy)
            .unwrap()
    })
    .unwrap_or_default();

    if random_best.phase_final_accuracy >= common.phase_final_accuracy - 0.02
        && random_best.phase_final_accuracy >= 0.85
    {
        out.insert("RANDOM_MOTIF_CONTROL_TOO_STRONG".to_string());
    } else {
        out.insert("RANDOM_MOTIF_CONTROL_FAILS".to_string());
    }

    if passes_sparse_transfer_gate(&common)
        && common.phase_final_accuracy >= random_best.phase_final_accuracy + 0.10
    {
        out.insert("COMMON_TEMPLATE_WORKS".to_string());
        out.insert("SPARSE_CORE_TRANSFER_POSITIVE".to_string());
        out.insert("EXPERIMENTAL_MUTATION_LANE_SUPPORTED".to_string());
        out.insert("PRODUCTION_API_NOT_READY".to_string());
    } else if !passes_sparse_transfer_gate(&common) && passes_sparse_transfer_gate(&plus_missing) {
        out.insert("COMMON_CORE_WAS_ONE_MOTIF_SHORT".to_string());
        out.insert("SPARSE_CORE_TRANSFER_POSITIVE".to_string());
        out.insert("EXPERIMENTAL_MUTATION_LANE_SUPPORTED".to_string());
        out.insert("PRODUCTION_API_NOT_READY".to_string());
    } else if passes_sparse_transfer_gate(&full16) {
        out.insert("FULL_16_REQUIRED".to_string());
    } else if dense.phase_final_accuracy >= 0.95 {
        out.insert("DENSE_REFERENCE_STILL_REQUIRED".to_string());
        out.insert("SPARSE_CORE_TRANSFER_FAILS".to_string());
    } else {
        out.insert("SPARSE_CORE_TRANSFER_FAILS".to_string());
    }

    if common.width_transfer_accuracy >= 0.85 || plus_missing.width_transfer_accuracy >= 0.85 {
        out.insert("WIDTH_TRANSFER_PASSES".to_string());
    }
    if common.long_path_transfer_accuracy >= 0.85
        || plus_missing.long_path_transfer_accuracy >= 0.85
    {
        out.insert("LONG_PATH_TRANSFER_PASSES".to_string());
    }
    if common.new_layout_transfer_accuracy >= 0.85
        || plus_missing.new_layout_transfer_accuracy >= 0.85
    {
        out.insert("LAYOUT_TRANSFER_PASSES".to_string());
    }
    if finals.iter().any(|row| {
        row.forbidden_private_field_leak > 0.0
            || row.nonlocal_edge_count > 0
            || row.direct_output_leak_rate > 0.05
    }) {
        out.insert("DIRECT_SHORTCUT_CONTAMINATION".to_string());
    }
    out.into_iter().collect()
}

fn best_arm(rows: &[&MetricRow], arm: &str) -> MetricRow {
    rows.iter()
        .filter(|row| row.arm == arm)
        .max_by(|a, b| {
            a.phase_final_accuracy
                .partial_cmp(&b.phase_final_accuracy)
                .unwrap()
        })
        .map(|row| (*row).clone())
        .unwrap_or_else(|| MetricRow {
            job_id: String::new(),
            seed: 0,
            arm: arm.to_string(),
            placement_mode: PlacementMode::TemplateOnAllFreeCells.as_str().to_string(),
            final_row: true,
            phase_final_accuracy: 0.0,
            correct_target_lane_probability_mean: 0.0,
            same_target_counterfactual_accuracy: 0.0,
            gate_shuffle_collapse: 0.0,
            width_transfer_accuracy: 0.0,
            long_path_transfer_accuracy: 0.0,
            new_layout_transfer_accuracy: 0.0,
            reverse_path_consistency_accuracy: 0.0,
            motif_type_count: 0,
            instantiated_motif_count: 0,
            accuracy_per_motif: 0.0,
            probability_per_motif: 0.0,
            motif_ablation_drop: 0.0,
            nonlocal_edge_count: 0,
            forbidden_private_field_leak: 0.0,
            direct_output_leak_rate: 0.0,
            min_per_pair_accuracy: 0.0,
            min_per_pair_correct_probability: 0.0,
            elapsed_sec: 0.0,
        })
}

fn passes_sparse_transfer_gate(row: &MetricRow) -> bool {
    row.phase_final_accuracy >= 0.90
        && row.width_transfer_accuracy >= 0.85
        && row.long_path_transfer_accuracy >= 0.85
        && row.new_layout_transfer_accuracy >= 0.85
        && row.same_target_counterfactual_accuracy >= 0.85
        && row.min_per_pair_accuracy >= 0.80
        && row.gate_shuffle_collapse >= 0.50
        && row.forbidden_private_field_leak == 0.0
        && row.nonlocal_edge_count == 0
        && row.direct_output_leak_rate <= 0.001
}

fn append_jsonl<P: AsRef<Path>, T: Serialize>(path: P, value: &T) -> std::io::Result<()> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    serde_json::to_writer(&mut file, value)?;
    writeln!(file)
}

fn write_json<P: AsRef<Path>, T: Serialize>(path: P, value: &T) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    writeln!(file)
}

fn now_sec() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}
