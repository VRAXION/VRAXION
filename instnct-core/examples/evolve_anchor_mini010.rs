//! ANCHOR-MINI-010 serialization robustness probe.
//!
//! This audit-sized sparse mutation-selection carrier tests whether the
//! process-first route from MINI-009 still works when the task is serialized
//! through held-out byte formats.

use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::path::PathBuf;

const CANDIDATE_COUNT: usize = 4;
const CATEGORY_COUNT: usize = 4;
const SLOT_LETTERS: [u8; CANDIDATE_COUNT] = [b'A', b'B', b'C', b'D'];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carrier {
    ByteDirectAnswer,
    ByteAuxPlanDirectAnswer,
    BytePlanFirst,
    BytePlanFirstHybrid,
    ByteShuffledTeacher,
    ByteShortcutTeacher,
    ByteOraclePlanVisible,
}

impl Carrier {
    fn parse(raw: &str) -> Self {
        match raw {
            "BYTE_DIRECT_ANSWER" => Self::ByteDirectAnswer,
            "BYTE_AUX_PLAN_DIRECT_ANSWER" => Self::ByteAuxPlanDirectAnswer,
            "BYTE_PLAN_FIRST" => Self::BytePlanFirst,
            "BYTE_PLAN_FIRST_HYBRID" => Self::BytePlanFirstHybrid,
            "BYTE_SHUFFLED_TEACHER" => Self::ByteShuffledTeacher,
            "BYTE_SHORTCUT_TEACHER" => Self::ByteShortcutTeacher,
            "BYTE_ORACLE_PLAN_VISIBLE" => Self::ByteOraclePlanVisible,
            other => panic!("unknown carrier: {other}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::ByteDirectAnswer => "BYTE_DIRECT_ANSWER",
            Self::ByteAuxPlanDirectAnswer => "BYTE_AUX_PLAN_DIRECT_ANSWER",
            Self::BytePlanFirst => "BYTE_PLAN_FIRST",
            Self::BytePlanFirstHybrid => "BYTE_PLAN_FIRST_HYBRID",
            Self::ByteShuffledTeacher => "BYTE_SHUFFLED_TEACHER",
            Self::ByteShortcutTeacher => "BYTE_SHORTCUT_TEACHER",
            Self::ByteOraclePlanVisible => "BYTE_ORACLE_PLAN_VISIBLE",
        }
    }

    fn uses_direct_branch(self) -> bool {
        matches!(
            self,
            Self::ByteDirectAnswer | Self::ByteAuxPlanDirectAnswer | Self::BytePlanFirstHybrid
        )
    }

    fn uses_plan_branch(self) -> bool {
        !matches!(self, Self::ByteDirectAnswer)
    }

    fn final_uses_direct(self) -> bool {
        matches!(
            self,
            Self::ByteDirectAnswer | Self::ByteAuxPlanDirectAnswer | Self::BytePlanFirstHybrid
        )
    }

    fn final_uses_policy(self) -> bool {
        matches!(
            self,
            Self::BytePlanFirst
                | Self::BytePlanFirstHybrid
                | Self::ByteShuffledTeacher
                | Self::ByteShortcutTeacher
                | Self::ByteOraclePlanVisible
        )
    }

    fn is_oracle(self) -> bool {
        matches!(self, Self::ByteOraclePlanVisible)
    }

    fn uses_shuffled_teacher(self) -> bool {
        matches!(self, Self::ByteShuffledTeacher)
    }

    fn uses_shortcut_teacher(self) -> bool {
        matches!(self, Self::ByteShortcutTeacher)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecoderMode {
    SchemaAware,
    FixedTemplateControl,
}

impl DecoderMode {
    fn parse(raw: &str) -> Self {
        match raw {
            "schema_aware" => Self::SchemaAware,
            "fixed_template_control" => Self::FixedTemplateControl,
            other => panic!("unknown decoder mode: {other}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::SchemaAware => "schema_aware",
            Self::FixedTemplateControl => "fixed_template_control",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Branch {
    Direct,
    Shortcut,
    Policy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FeatureKind {
    ByteSurface,
    ByteGoalCat(usize),
    ByteEffectCat(usize),
    ByteGoalEffectConjunction(usize),
    ByteShiftedConjunction(usize),
    OraclePolicy,
    Bias,
}

impl FeatureKind {
    fn as_str(self) -> String {
        match self {
            Self::ByteSurface => "byte_surface_bucket".to_string(),
            Self::ByteGoalCat(category) => format!("byte_goal_cat_{category}"),
            Self::ByteEffectCat(category) => format!("byte_effect_cat_{category}"),
            Self::ByteGoalEffectConjunction(category) => {
                format!("byte_goal_and_effect_cat_{category}")
            }
            Self::ByteShiftedConjunction(category) => {
                format!("byte_goal_and_shifted_effect_cat_{category}")
            }
            Self::OraclePolicy => "oracle_policy".to_string(),
            Self::Bias => "bias".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
struct Edge {
    branch: Branch,
    feature: FeatureKind,
    weight: f64,
}

#[derive(Clone, Debug, Default)]
struct Genome {
    edges: Vec<Edge>,
}

impl Genome {
    fn initial(carrier: Carrier) -> Self {
        let mut genome = Self::default();
        if carrier.uses_plan_branch() {
            genome.edges.push(Edge {
                branch: Branch::Shortcut,
                feature: FeatureKind::ByteSurface,
                weight: 5.0,
            });
        }
        genome
    }
}

#[derive(Clone, Deserialize)]
struct Example {
    task_bytes: String,
    goal_category: usize,
    effect_categories: Vec<usize>,
    answer_label: usize,
    match_bits: Vec<u8>,
    surface_shortcut_label: usize,
    surface_shortcut_is_gold: bool,
    #[serde(skip)]
    decoded_goal_category: usize,
    #[serde(skip)]
    decoded_effect_categories: Vec<usize>,
    #[serde(skip)]
    decoded_surface_buckets: Vec<usize>,
    #[serde(skip)]
    decode_ok: bool,
}

#[derive(Deserialize)]
struct Dataset {
    train_examples: Vec<Example>,
    eval_examples: Vec<Example>,
}

#[derive(Clone)]
struct Config {
    dataset: PathBuf,
    out_dir: PathBuf,
    carrier: Carrier,
    decoder_mode: DecoderMode,
    seed: u64,
    max_steps: usize,
    proposals: usize,
    edge_cap: usize,
    aux_weight: f64,
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        value as f64 / ((1u64 << 53) as f64)
    }

    fn usize(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }

    fn signed_weight(&mut self) -> f64 {
        let sign = if self.f64() < 0.5 { -1.0 } else { 1.0 };
        sign * (0.5 + 2.5 * self.f64())
    }
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut dataset = None;
    let mut out_dir = None;
    let mut carrier = Carrier::BytePlanFirst;
    let mut decoder_mode = DecoderMode::SchemaAware;
    let mut seed = 2026u64;
    let mut max_steps = 1800usize;
    let mut proposals = 9usize;
    let mut edge_cap = 24usize;
    let mut aux_weight = 4.0f64;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dataset" => {
                i += 1;
                dataset = Some(PathBuf::from(&args[i]));
            }
            "--out-dir" => {
                i += 1;
                out_dir = Some(PathBuf::from(&args[i]));
            }
            "--carrier" => {
                i += 1;
                carrier = Carrier::parse(&args[i]);
            }
            "--decoder-mode" => {
                i += 1;
                decoder_mode = DecoderMode::parse(&args[i]);
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("invalid --seed");
            }
            "--max-steps" => {
                i += 1;
                max_steps = args[i].parse().expect("invalid --max-steps");
            }
            "--proposals" => {
                i += 1;
                proposals = args[i].parse().expect("invalid --proposals");
            }
            "--edge-cap" => {
                i += 1;
                edge_cap = args[i].parse().expect("invalid --edge-cap");
            }
            "--aux-weight" => {
                i += 1;
                aux_weight = args[i].parse().expect("invalid --aux-weight");
            }
            _ => {}
        }
        i += 1;
    }
    Config {
        dataset: dataset.expect("--dataset is required"),
        out_dir: out_dir.expect("--out-dir is required"),
        carrier,
        decoder_mode,
        seed,
        max_steps,
        proposals,
        edge_cap,
        aux_weight,
    }
}

fn byte_process_features() -> Vec<FeatureKind> {
    let mut features = Vec::new();
    for category in 0..CATEGORY_COUNT {
        features.push(FeatureKind::ByteGoalCat(category));
        features.push(FeatureKind::ByteEffectCat(category));
        features.push(FeatureKind::ByteGoalEffectConjunction(category));
        features.push(FeatureKind::ByteShiftedConjunction(category));
    }
    features.push(FeatureKind::Bias);
    features
}

fn allowed_features(carrier: Carrier, branch: Branch) -> Vec<FeatureKind> {
    match branch {
        Branch::Direct => {
            if carrier.uses_direct_branch() {
                vec![FeatureKind::ByteSurface, FeatureKind::Bias]
            } else {
                vec![]
            }
        }
        Branch::Shortcut => {
            if carrier.uses_plan_branch() {
                vec![FeatureKind::ByteSurface, FeatureKind::Bias]
            } else {
                vec![]
            }
        }
        Branch::Policy => {
            if !carrier.uses_plan_branch() {
                vec![]
            } else if carrier.is_oracle() {
                vec![FeatureKind::OraclePolicy, FeatureKind::Bias]
            } else if carrier.uses_shortcut_teacher() {
                vec![FeatureKind::ByteSurface, FeatureKind::Bias]
            } else {
                byte_process_features()
            }
        }
    }
}

fn shifted_category(goal_category: usize) -> usize {
    (goal_category + 1) % CATEGORY_COUNT
}

fn task_bytes(ex: &Example) -> &[u8] {
    ex.task_bytes.as_bytes()
}

fn digit_after_str(haystack: &str, marker: &str) -> Option<usize> {
    let idx = haystack.find(marker)?;
    let value = haystack.as_bytes().get(idx + marker.len())?;
    if value.is_ascii_digit() {
        Some((value - b'0') as usize)
    } else {
        None
    }
}

fn slot_segment<'a>(task: &'a str, candidate: usize) -> Option<&'a str> {
    let letter = SLOT_LETTERS[candidate] as char;
    let canonical = format!("{letter}=");
    let alias = format!("{letter}(");
    task.split([';', '|'])
        .find(|part| part.starts_with(&canonical) || part.starts_with(&alias))
}

fn fixed_segment<'a>(task: &'a str, candidate: usize) -> Option<&'a str> {
    let parts: Vec<&str> = task.split(';').collect();
    if parts.len() != CANDIDATE_COUNT + 1 || !parts.first()?.starts_with("G=") {
        return None;
    }
    let expected = format!("{}=E", SLOT_LETTERS[candidate] as char);
    let segment = parts.get(candidate + 1)?;
    if segment.starts_with(&expected) && segment.contains(":S") && !segment.contains(":X") {
        Some(segment)
    } else {
        None
    }
}

fn parse_goal_category(ex: &Example, decoder: DecoderMode) -> Option<usize> {
    let task = &ex.task_bytes;
    match decoder {
        DecoderMode::SchemaAware => {
            digit_after_str(task, "GOAL=").or_else(|| digit_after_str(task, "G="))
        }
        DecoderMode::FixedTemplateControl => {
            if task.split(';').count() == CANDIDATE_COUNT + 1 {
                digit_after_str(task, "G=")
            } else {
                None
            }
        }
    }
}

fn parse_effect_category(ex: &Example, decoder: DecoderMode, candidate: usize) -> Option<usize> {
    let task = &ex.task_bytes;
    match decoder {
        DecoderMode::SchemaAware => {
            let segment = slot_segment(task, candidate)?;
            digit_after_str(segment, "EFFECT=")
                .or_else(|| digit_after_str(segment, "=E"))
                .or_else(|| digit_after_str(segment, ":E"))
        }
        DecoderMode::FixedTemplateControl => {
            let segment = fixed_segment(task, candidate)?;
            digit_after_str(segment, "=E")
        }
    }
}

fn parse_surface_bucket(ex: &Example, decoder: DecoderMode, candidate: usize) -> Option<usize> {
    let task = &ex.task_bytes;
    match decoder {
        DecoderMode::SchemaAware => {
            let segment = slot_segment(task, candidate)?;
            digit_after_str(segment, "SURFACE=")
                .or_else(|| digit_after_str(segment, ":S"))
                .or_else(|| digit_after_str(segment, "=S"))
        }
        DecoderMode::FixedTemplateControl => {
            let segment = fixed_segment(task, candidate)?;
            digit_after_str(segment, ":S")
        }
    }
}

fn surface_shortcut_from_buckets(buckets: &[usize]) -> usize {
    let mut best = 0usize;
    let mut best_bucket = buckets.first().copied().unwrap_or(0);
    for candidate in 1..CANDIDATE_COUNT {
        let bucket = buckets.get(candidate).copied().unwrap_or(0);
        if bucket > best_bucket {
            best = candidate;
            best_bucket = bucket;
        }
    }
    best
}

fn prepare_example(ex: &mut Example, decoder: DecoderMode) {
    let decoded_goal = parse_goal_category(ex, decoder);
    let decoded_effects: Vec<Option<usize>> = (0..CANDIDATE_COUNT)
        .map(|candidate| parse_effect_category(ex, decoder, candidate))
        .collect();
    let decoded_buckets: Vec<Option<usize>> = (0..CANDIDATE_COUNT)
        .map(|candidate| parse_surface_bucket(ex, decoder, candidate))
        .collect();
    ex.decoded_goal_category = decoded_goal.unwrap_or(0);
    ex.decoded_effect_categories = decoded_effects
        .iter()
        .map(|value| value.unwrap_or(usize::MAX))
        .collect();
    ex.decoded_surface_buckets = decoded_buckets
        .iter()
        .map(|value| value.unwrap_or(0))
        .collect();
    ex.decode_ok = decoded_goal == Some(ex.goal_category)
        && decoded_effects
            .iter()
            .enumerate()
            .all(|(candidate, value)| *value == Some(ex.effect_categories[candidate]))
        && decoded_buckets.iter().all(Option::is_some)
        && surface_shortcut_from_buckets(&ex.decoded_surface_buckets) == ex.surface_shortcut_label;
}

fn prepare_dataset(dataset: &mut Dataset, decoder: DecoderMode) {
    for ex in dataset
        .train_examples
        .iter_mut()
        .chain(dataset.eval_examples.iter_mut())
    {
        prepare_example(ex, decoder);
    }
}

fn byte_goal_category(ex: &Example, _decoder: DecoderMode) -> usize {
    ex.decoded_goal_category
}

fn byte_effect_category(ex: &Example, _decoder: DecoderMode, candidate: usize) -> usize {
    ex.decoded_effect_categories
        .get(candidate)
        .copied()
        .unwrap_or(usize::MAX)
}

fn byte_surface_bucket(ex: &Example, _decoder: DecoderMode, candidate: usize) -> usize {
    ex.decoded_surface_buckets.get(candidate).copied().unwrap_or(0)
}

fn byte_surface_shortcut_label(ex: &Example, decoder: DecoderMode) -> usize {
    let mut best = 0usize;
    let mut best_bucket = byte_surface_bucket(ex, decoder, 0);
    for candidate in 1..CANDIDATE_COUNT {
        let bucket = byte_surface_bucket(ex, decoder, candidate);
        if bucket > best_bucket {
            best = candidate;
            best_bucket = bucket;
        }
    }
    best
}

fn task_bytes_forbidden_leak(ex: &Example) -> bool {
    let upper = ex.task_bytes.to_ascii_uppercase();
    ["ANS", "ANSWER", "MATCH", "POLICY", "CHOOSE", "GOLD"]
        .iter()
        .any(|needle| upper.contains(needle))
}

fn byte_input_integrity(ex: &Example) -> bool {
    !task_bytes_forbidden_leak(ex) && !task_bytes(ex).is_empty()
}

fn dataset_byte_input_integrity(dataset: &Dataset) -> bool {
    dataset
        .train_examples
        .iter()
        .chain(dataset.eval_examples.iter())
        .all(byte_input_integrity)
}

fn dataset_decode_integrity(dataset: &Dataset, decoder: DecoderMode) -> bool {
    dataset
        .train_examples
        .iter()
        .chain(dataset.eval_examples.iter())
        .all(|ex| {
            let _ = decoder;
            ex.decode_ok
        })
}

fn target_policy_bit(carrier: Carrier, decoder: DecoderMode, ex: &Example, candidate: usize) -> u8 {
    if carrier.uses_shuffled_teacher() {
        (byte_effect_category(ex, decoder, candidate) == shifted_category(byte_goal_category(ex, decoder))) as u8
    } else if carrier.uses_shortcut_teacher() {
        (candidate == byte_surface_shortcut_label(ex, decoder)) as u8
    } else {
        ex.match_bits[candidate]
    }
}

fn true_policy_bit(ex: &Example, candidate: usize) -> u8 {
    ex.match_bits[candidate]
}

fn teacher_answer_label(carrier: Carrier, decoder: DecoderMode, ex: &Example) -> usize {
    for candidate in 0..CANDIDATE_COUNT {
        if target_policy_bit(carrier, decoder, ex, candidate) == 1 {
            return candidate;
        }
    }
    ex.answer_label
}

fn feature_value(
    feature: FeatureKind,
    carrier: Carrier,
    decoder: DecoderMode,
    ex: &Example,
    candidate: usize,
) -> f64 {
    match feature {
        FeatureKind::ByteSurface => byte_surface_bucket(ex, decoder, candidate) as f64 / 9.0,
        FeatureKind::ByteGoalCat(category) => (byte_goal_category(ex, decoder) == category) as u8 as f64,
        FeatureKind::ByteEffectCat(category) => {
            (byte_effect_category(ex, decoder, candidate) == category) as u8 as f64
        }
        FeatureKind::ByteGoalEffectConjunction(category) => {
            (byte_goal_category(ex, decoder) == category
                && byte_effect_category(ex, decoder, candidate) == category)
                as u8 as f64
        }
        FeatureKind::ByteShiftedConjunction(category) => {
            let shifted = shifted_category(category);
            (byte_goal_category(ex, decoder) == category
                && byte_effect_category(ex, decoder, candidate) == shifted)
                as u8 as f64
        }
        FeatureKind::OraclePolicy => target_policy_bit(carrier, decoder, ex, candidate) as f64,
        FeatureKind::Bias => 1.0,
    }
}

fn branch_scores(
    genome: &Genome,
    branch: Branch,
    carrier: Carrier,
    decoder: DecoderMode,
    ex: &Example,
) -> [f64; CANDIDATE_COUNT] {
    let mut scores = [0.0; CANDIDATE_COUNT];
    for edge in &genome.edges {
        if edge.branch != branch {
            continue;
        }
        for candidate in 0..CANDIDATE_COUNT {
            scores[candidate] += edge.weight * feature_value(edge.feature, carrier, decoder, ex, candidate);
        }
    }
    scores
}

fn final_scores(
    genome: &Genome,
    carrier: Carrier,
    decoder: DecoderMode,
    ex: &Example,
) -> [f64; CANDIDATE_COUNT] {
    let mut scores = [0.0; CANDIDATE_COUNT];
    if carrier.final_uses_direct() {
        let direct = branch_scores(genome, Branch::Direct, carrier, decoder, ex);
        for i in 0..CANDIDATE_COUNT {
            scores[i] += direct[i];
        }
    }
    if carrier.final_uses_policy() {
        let policy = branch_scores(genome, Branch::Policy, carrier, decoder, ex);
        for i in 0..CANDIDATE_COUNT {
            scores[i] += policy[i];
        }
    }
    scores
}

fn softmax_gold(scores: [f64; CANDIDATE_COUNT], label: usize) -> f64 {
    let max_v = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    let mut gold = 0.0;
    for (idx, score) in scores.iter().enumerate() {
        let value = (*score - max_v).exp();
        sum += value;
        if idx == label {
            gold = value;
        }
    }
    gold / sum.max(1e-12)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn predict(scores: [f64; CANDIDATE_COUNT]) -> usize {
    let mut best = 0usize;
    let mut best_score = scores[0];
    for (idx, score) in scores.iter().enumerate().skip(1) {
        if *score > best_score {
            best = idx;
            best_score = *score;
        }
    }
    best
}

fn answer_score(genome: &Genome, carrier: Carrier, decoder: DecoderMode, rows: &[Example]) -> f64 {
    rows.iter()
        .map(|ex| softmax_gold(final_scores(genome, carrier, decoder, ex), ex.answer_label))
        .sum::<f64>()
        / rows.len() as f64
}

fn fitness_answer_score(genome: &Genome, carrier: Carrier, decoder: DecoderMode, rows: &[Example]) -> f64 {
    rows.iter()
        .map(|ex| softmax_gold(final_scores(genome, carrier, decoder, ex), teacher_answer_label(carrier, decoder, ex)))
        .sum::<f64>()
        / rows.len() as f64
}

fn policy_bit_score(genome: &Genome, carrier: Carrier, decoder: DecoderMode, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.5;
    }
    let mut pos_total = 0usize;
    let mut neg_total = 0usize;
    let mut pos_score = 0.0;
    let mut neg_score = 0.0;
    for ex in rows {
        let scores = branch_scores(genome, Branch::Policy, carrier, decoder, ex);
        for i in 0..CANDIDATE_COUNT {
            if target_policy_bit(carrier, decoder, ex, i) == 1 {
                pos_total += 1;
                pos_score += sigmoid(scores[i]);
            } else {
                neg_total += 1;
                neg_score += sigmoid(-scores[i]);
            }
        }
    }
    let pos = if pos_total == 0 {
        0.5
    } else {
        pos_score / pos_total as f64
    };
    let neg = if neg_total == 0 {
        0.5
    } else {
        neg_score / neg_total as f64
    };
    0.5 * (pos + neg)
}

fn shortcut_detection_score(genome: &Genome, carrier: Carrier, decoder: DecoderMode, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.5;
    }
    rows.iter()
        .map(|ex| {
            softmax_gold(
                branch_scores(genome, Branch::Shortcut, carrier, decoder, ex),
                ex.surface_shortcut_label,
            )
        })
        .sum::<f64>()
        / rows.len() as f64
}

fn shortcut_validity_score(genome: &Genome, carrier: Carrier, decoder: DecoderMode, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.5;
    }
    rows.iter()
        .map(|ex| {
            let policy = branch_scores(genome, Branch::Policy, carrier, decoder, ex);
            let observed = ex.surface_shortcut_label;
            if target_policy_bit(carrier, decoder, ex, observed) == 1 {
                sigmoid(policy[observed])
            } else {
                sigmoid(-policy[observed])
            }
        })
        .sum::<f64>()
        / rows.len() as f64
}

fn invalid_shortcut_rejection_score(
    genome: &Genome,
    carrier: Carrier,
    decoder: DecoderMode,
    rows: &[Example],
) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.5;
    }
    let mut total = 0usize;
    let mut score = 0.0;
    for ex in rows {
        let observed = ex.surface_shortcut_label;
        if target_policy_bit(carrier, decoder, ex, observed) == 0 {
            total += 1;
            let policy = branch_scores(genome, Branch::Policy, carrier, decoder, ex);
            score += sigmoid(-policy[observed]);
        }
    }
    if total == 0 {
        0.5
    } else {
        score / total as f64
    }
}

fn plan_score(genome: &Genome, carrier: Carrier, decoder: DecoderMode, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.5;
    }
    0.40 * shortcut_detection_score(genome, carrier, decoder, rows)
        + 0.15 * shortcut_validity_score(genome, carrier, decoder, rows)
        + 0.15 * invalid_shortcut_rejection_score(genome, carrier, decoder, rows)
        + 0.30 * policy_bit_score(genome, carrier, decoder, rows)
}

fn fitness(genome: &Genome, carrier: Carrier, decoder: DecoderMode, rows: &[Example], aux_weight: f64) -> f64 {
    let answer = fitness_answer_score(genome, carrier, decoder, rows);
    if carrier.uses_plan_branch() {
        answer + aux_weight * plan_score(genome, carrier, decoder, rows)
    } else {
        answer
    }
}

fn accuracy(genome: &Genome, carrier: Carrier, decoder: DecoderMode, rows: &[Example]) -> f64 {
    rows.iter()
        .filter(|ex| predict(final_scores(genome, carrier, decoder, ex)) == ex.answer_label)
        .count() as f64
        / rows.len() as f64
}

fn shortcut_trap_rate(genome: &Genome, carrier: Carrier, decoder: DecoderMode, rows: &[Example]) -> f64 {
    let mut opportunities = 0usize;
    let mut traps = 0usize;
    for ex in rows {
        if ex.surface_shortcut_label != ex.answer_label {
            opportunities += 1;
            if predict(final_scores(genome, carrier, decoder, ex)) == ex.surface_shortcut_label {
                traps += 1;
            }
        }
    }
    if opportunities == 0 {
        0.0
    } else {
        traps as f64 / opportunities as f64
    }
}

fn observed_shortcut_accuracy(genome: &Genome, carrier: Carrier, decoder: DecoderMode, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.0;
    }
    rows.iter()
        .filter(|ex| {
            predict(branch_scores(genome, Branch::Shortcut, carrier, decoder, ex)) == ex.surface_shortcut_label
        })
        .count() as f64
        / rows.len() as f64
}

fn shortcut_validity_accuracy(genome: &Genome, carrier: Carrier, decoder: DecoderMode, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.0;
    }
    rows.iter()
        .filter(|ex| {
            let policy = branch_scores(genome, Branch::Policy, carrier, decoder, ex);
            let observed = ex.surface_shortcut_label;
            let predicted_valid = policy[observed] > 0.0;
            let true_valid = true_policy_bit(ex, observed) == 1;
            predicted_valid == true_valid
        })
        .count() as f64
        / rows.len() as f64
}

fn invalid_shortcut_rejection_accuracy(
    genome: &Genome,
    carrier: Carrier,
    decoder: DecoderMode,
    rows: &[Example],
) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.0;
    }
    let mut total = 0usize;
    let mut correct = 0usize;
    for ex in rows {
        let observed = ex.surface_shortcut_label;
        if true_policy_bit(ex, observed) == 0 {
            total += 1;
            let policy = branch_scores(genome, Branch::Policy, carrier, decoder, ex);
            correct += (policy[observed] <= 0.0) as usize;
        }
    }
    if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    }
}

fn policy_bit_accuracy(genome: &Genome, carrier: Carrier, decoder: DecoderMode, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.0;
    }
    let mut total = 0usize;
    let mut correct = 0usize;
    for ex in rows {
        let scores = branch_scores(genome, Branch::Policy, carrier, decoder, ex);
        for i in 0..CANDIDATE_COUNT {
            total += 1;
            correct += ((scores[i] > 0.0) as u8 == true_policy_bit(ex, i)) as usize;
        }
    }
    correct as f64 / total as f64
}

fn plan_exact_row_accuracy(genome: &Genome, carrier: Carrier, decoder: DecoderMode, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.0;
    }
    rows.iter()
        .filter(|ex| {
            let shortcut_ok =
                predict(branch_scores(genome, Branch::Shortcut, carrier, decoder, ex)) == ex.surface_shortcut_label;
            let policy = branch_scores(genome, Branch::Policy, carrier, decoder, ex);
            let policy_ok =
                (0..CANDIDATE_COUNT).all(|i| (policy[i] > 0.0) as u8 == true_policy_bit(ex, i));
            shortcut_ok && policy_ok
        })
        .count() as f64
        / rows.len() as f64
}

fn answer_from_plan_consistency(genome: &Genome, carrier: Carrier, decoder: DecoderMode, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.0;
    }
    rows.iter()
        .filter(|ex| {
            predict(final_scores(genome, carrier, decoder, ex))
                == predict(branch_scores(genome, Branch::Policy, carrier, decoder, ex))
        })
        .count() as f64
        / rows.len() as f64
}

fn surface_alignment(rows: &[Example]) -> f64 {
    rows.iter()
        .filter(|ex| ex.surface_shortcut_is_gold)
        .count() as f64
        / rows.len() as f64
}

fn mutate(parent: &Genome, carrier: Carrier, cfg: &Config, rng: &mut Rng) -> Genome {
    let mut child = parent.clone();
    let mut branches = Vec::new();
    if carrier.uses_direct_branch() {
        branches.push(Branch::Direct);
    }
    if carrier.uses_plan_branch() {
        branches.push(Branch::Shortcut);
        branches.push(Branch::Policy);
    }
    if branches.is_empty() {
        return child;
    }
    let roll = rng.f64();
    if child.edges.is_empty() || (roll < 0.45 && child.edges.len() < cfg.edge_cap) {
        let branch = branches[rng.usize(branches.len())];
        let features = allowed_features(carrier, branch);
        if features.is_empty() {
            return child;
        }
        child.edges.push(Edge {
            branch,
            feature: features[rng.usize(features.len())],
            weight: rng.signed_weight(),
        });
    } else if roll < 0.80 {
        let idx = rng.usize(child.edges.len());
        child.edges[idx].weight += rng.signed_weight() * 0.35;
    } else if roll < 0.92 {
        let idx = rng.usize(child.edges.len());
        let branch = child.edges[idx].branch;
        let features = allowed_features(carrier, branch);
        if !features.is_empty() {
            child.edges[idx].feature = features[rng.usize(features.len())];
        }
    } else if child.edges.len() > 1 {
        let idx = rng.usize(child.edges.len());
        child.edges.swap_remove(idx);
    }
    child
}

fn evolve(dataset: &Dataset, cfg: &Config) -> (Genome, f64, usize) {
    let mut rng = Rng::new(cfg.seed ^ 0xA17C_010u64);
    let mut parent = Genome::initial(cfg.carrier);
    let mut parent_score = fitness(
        &parent,
        cfg.carrier,
        cfg.decoder_mode,
        &dataset.train_examples,
        cfg.aux_weight,
    );
    let mut accepted = 0usize;
    for _ in 0..cfg.max_steps {
        let mut best = parent.clone();
        let mut best_score = parent_score;
        for _ in 0..cfg.proposals {
            let candidate = mutate(&parent, cfg.carrier, cfg, &mut rng);
            let score = fitness(
                &candidate,
                cfg.carrier,
                cfg.decoder_mode,
                &dataset.train_examples,
                cfg.aux_weight,
            );
            if score > best_score + 1e-12 {
                best = candidate;
                best_score = score;
            }
        }
        if best_score > parent_score + 1e-12 {
            parent = best;
            parent_score = best_score;
            accepted += 1;
        }
    }
    (parent, parent_score, accepted)
}

fn edge_summary(genome: &Genome) -> Vec<serde_json::Value> {
    genome
        .edges
        .iter()
        .map(|edge| {
            json!({
                "branch": match edge.branch {
                    Branch::Direct => "direct",
                    Branch::Shortcut => "shortcut",
                    Branch::Policy => "policy",
                },
                "feature": edge.feature.as_str(),
                "weight": edge.weight,
            })
        })
        .collect()
}

fn main() {
    let cfg = parse_args();
    fs::create_dir_all(&cfg.out_dir).expect("failed to create out dir");
    let raw = fs::read_to_string(&cfg.dataset).expect("failed to read dataset");
    let mut dataset: Dataset = serde_json::from_str(&raw).expect("failed to parse dataset");
    prepare_dataset(&mut dataset, cfg.decoder_mode);
    let train_align = surface_alignment(&dataset.train_examples);
    let eval_flip = 1.0 - surface_alignment(&dataset.eval_examples);
    let byte_integrity = dataset_byte_input_integrity(&dataset);
    let decode_integrity = dataset_decode_integrity(&dataset, cfg.decoder_mode);
    let invalid_stress = train_align < 0.85 || eval_flip < 0.85 || !byte_integrity || !decode_integrity;
    let (genome, train_fitness, accepted_steps) = evolve(&dataset, &cfg);
    let payload = json!({
        "status": if invalid_stress {
            "ANCHOR_MINI_010_JOB_INVALID_STRESS"
        } else {
            "ANCHOR_MINI_010_JOB_COMPLETE"
        },
        "config": {
            "carrier": cfg.carrier.as_str(),
            "decoder_mode": cfg.decoder_mode.as_str(),
            "seed": cfg.seed,
            "max_steps": cfg.max_steps,
            "proposals": cfg.proposals,
            "edge_cap": cfg.edge_cap,
            "aux_weight": cfg.aux_weight,
            "dataset": cfg.dataset.display().to_string(),
        },
        "stress": {
            "surface_shortcut_train_alignment": train_align,
            "surface_shortcut_eval_flip_rate": eval_flip,
            "byte_input_integrity": byte_integrity,
            "decode_integrity": decode_integrity,
            "invalid_stress": invalid_stress,
        },
        "metrics": {
            "answer_train_accuracy": accuracy(&genome, cfg.carrier, cfg.decoder_mode, &dataset.train_examples),
            "answer_eval_accuracy": accuracy(&genome, cfg.carrier, cfg.decoder_mode, &dataset.eval_examples),
            "answer_train_score": answer_score(&genome, cfg.carrier, cfg.decoder_mode, &dataset.train_examples),
            "answer_eval_score": answer_score(&genome, cfg.carrier, cfg.decoder_mode, &dataset.eval_examples),
            "shortcut_trap_rate": shortcut_trap_rate(&genome, cfg.carrier, cfg.decoder_mode, &dataset.eval_examples),
            "shortcut_detection_score": shortcut_detection_score(&genome, cfg.carrier, cfg.decoder_mode, &dataset.eval_examples),
            "shortcut_validity_score": shortcut_validity_score(&genome, cfg.carrier, cfg.decoder_mode, &dataset.eval_examples),
            "invalid_shortcut_rejection_score": invalid_shortcut_rejection_score(&genome, cfg.carrier, cfg.decoder_mode, &dataset.eval_examples),
            "policy_bit_score": policy_bit_score(&genome, cfg.carrier, cfg.decoder_mode, &dataset.eval_examples),
            "plan_score": plan_score(&genome, cfg.carrier, cfg.decoder_mode, &dataset.eval_examples),
            "observed_shortcut_accuracy": observed_shortcut_accuracy(&genome, cfg.carrier, cfg.decoder_mode, &dataset.eval_examples),
            "shortcut_validity_accuracy": shortcut_validity_accuracy(&genome, cfg.carrier, cfg.decoder_mode, &dataset.eval_examples),
            "invalid_shortcut_rejection_accuracy": invalid_shortcut_rejection_accuracy(&genome, cfg.carrier, cfg.decoder_mode, &dataset.eval_examples),
            "policy_bit_accuracy": policy_bit_accuracy(&genome, cfg.carrier, cfg.decoder_mode, &dataset.eval_examples),
            "plan_exact_row_accuracy": plan_exact_row_accuracy(&genome, cfg.carrier, cfg.decoder_mode, &dataset.eval_examples),
            "answer_from_plan_consistency": answer_from_plan_consistency(&genome, cfg.carrier, cfg.decoder_mode, &dataset.eval_examples),
            "byte_input_integrity": if byte_integrity { 1.0 } else { 0.0 },
            "decode_integrity": if decode_integrity { 1.0 } else { 0.0 },
            "train_fitness": train_fitness,
            "accepted_steps": accepted_steps,
            "edge_count": genome.edges.len(),
        },
        "feature_visibility": {
            "oracle_policy_visible": cfg.carrier.is_oracle(),
            "final_uses_direct_surface_path": cfg.carrier.final_uses_direct(),
            "final_uses_policy_plan_path": cfg.carrier.final_uses_policy(),
            "learned_plan_uses_oracle_policy_as_input": cfg.carrier.is_oracle(),
            "allowed_non_oracle_features": [
                "byte_surface_bucket(candidate) from task_bytes",
                "byte_goal_category(k) from task_bytes",
                "byte_effect_category(candidate,k) from task_bytes",
                "byte_goal_effect_conjunction(candidate,k) derived from task_bytes",
                "byte_goal_shifted_effect_conjunction(candidate,k) derived from task_bytes",
                "bias"
            ],
        },
        "genome": {
            "edges": edge_summary(&genome),
        }
    });
    let out_file = cfg.out_dir.join("report.json");
    fs::write(
        &out_file,
        serde_json::to_string_pretty(&payload).unwrap() + "\n",
    )
    .expect("failed to write report");
    println!("{}", out_file.display());
}
