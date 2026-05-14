//! ANCHOR-MINI-006 symbolic process-format sweep.
//!
//! This audit carrier reuses the MINI-005 shortcut-flip task and varies only
//! the process-decomposition feature layout. It is a symbolic control sweep,
//! not a literal text serialization benchmark.

use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::path::PathBuf;

const CANDIDATE_COUNT: usize = 4;
const CATEGORY_COUNT: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum FormatArm {
    AnswerOnly,
    OracleMatchBits,
    RawSymbolicProcess,
    ProseProcess,
    InnerMonologueProcess,
    StrictJsonProcess,
    FlatKeyValueProcess,
    RelationalTriplesProcess,
    ActionOutcomeTableProcess,
    RelationPlusActionProcess,
    CompactHybridProcess,
    ShuffledCompactHybrid,
}

impl FormatArm {
    fn parse(raw: &str) -> Self {
        match raw {
            "ANSWER_ONLY" => Self::AnswerOnly,
            "ORACLE_MATCH_BITS" => Self::OracleMatchBits,
            "RAW_SYMBOLIC_PROCESS" => Self::RawSymbolicProcess,
            "PROSE_PROCESS" => Self::ProseProcess,
            "INNER_MONOLOGUE_PROCESS" => Self::InnerMonologueProcess,
            "STRICT_JSON_PROCESS" => Self::StrictJsonProcess,
            "FLAT_KEY_VALUE_PROCESS" => Self::FlatKeyValueProcess,
            "RELATIONAL_TRIPLES_PROCESS" => Self::RelationalTriplesProcess,
            "ACTION_OUTCOME_TABLE_PROCESS" => Self::ActionOutcomeTableProcess,
            "RELATION_PLUS_ACTION_PROCESS" => Self::RelationPlusActionProcess,
            "COMPACT_HYBRID_PROCESS" => Self::CompactHybridProcess,
            "SHUFFLED_COMPACT_HYBRID" => Self::ShuffledCompactHybrid,
            other => panic!("unknown format arm: {other}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::AnswerOnly => "ANSWER_ONLY",
            Self::OracleMatchBits => "ORACLE_MATCH_BITS",
            Self::RawSymbolicProcess => "RAW_SYMBOLIC_PROCESS",
            Self::ProseProcess => "PROSE_PROCESS",
            Self::InnerMonologueProcess => "INNER_MONOLOGUE_PROCESS",
            Self::StrictJsonProcess => "STRICT_JSON_PROCESS",
            Self::FlatKeyValueProcess => "FLAT_KEY_VALUE_PROCESS",
            Self::RelationalTriplesProcess => "RELATIONAL_TRIPLES_PROCESS",
            Self::ActionOutcomeTableProcess => "ACTION_OUTCOME_TABLE_PROCESS",
            Self::RelationPlusActionProcess => "RELATION_PLUS_ACTION_PROCESS",
            Self::CompactHybridProcess => "COMPACT_HYBRID_PROCESS",
            Self::ShuffledCompactHybrid => "SHUFFLED_COMPACT_HYBRID",
        }
    }

    fn uses_direct(self) -> bool {
        matches!(self, Self::AnswerOnly)
    }

    fn uses_process(self) -> bool {
        !matches!(self, Self::AnswerOnly)
    }

    fn is_oracle(self) -> bool {
        matches!(self, Self::OracleMatchBits)
    }

    fn is_shuffled(self) -> bool {
        matches!(self, Self::ShuffledCompactHybrid)
    }

    fn is_non_oracle_learned(self) -> bool {
        self.uses_process() && !self.is_oracle()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Branch {
    Direct,
    Process,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FeatureKind {
    Surface,
    GoalCat(usize),
    EffectCat(usize),
    CategoryConjunction(usize),
    ShiftedConjunction(usize),
    OracleMatch,
    Bias,
}

impl FeatureKind {
    fn as_str(self) -> String {
        match self {
            Self::Surface => "surface".to_string(),
            Self::GoalCat(category) => format!("goal_cat_{category}"),
            Self::EffectCat(category) => format!("effect_cat_{category}"),
            Self::CategoryConjunction(category) => format!("goal_and_effect_cat_{category}"),
            Self::ShiftedConjunction(category) => format!("goal_and_shifted_effect_cat_{category}"),
            Self::OracleMatch => "oracle_match".to_string(),
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

#[derive(Clone, Deserialize)]
struct Example {
    goal_category: usize,
    effect_categories: Vec<usize>,
    surface_priors: Vec<f64>,
    answer_label: usize,
    match_bits: Vec<u8>,
    surface_shortcut_label: usize,
    surface_shortcut_is_gold: bool,
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
    format_arm: FormatArm,
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
    let mut format_arm = FormatArm::CompactHybridProcess;
    let mut seed = 2026u64;
    let mut max_steps = 1600usize;
    let mut proposals = 9usize;
    let mut edge_cap = 20usize;
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
            "--format-arm" => {
                i += 1;
                format_arm = FormatArm::parse(&args[i]);
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
        format_arm,
        seed,
        max_steps,
        proposals,
        edge_cap,
        aux_weight,
    }
}

fn category_features(
    include_goal: bool,
    include_effect: bool,
    include_match_gate: bool,
    include_shifted_gate: bool,
    include_surface: bool,
) -> Vec<FeatureKind> {
    let mut features = Vec::new();
    if include_surface {
        features.push(FeatureKind::Surface);
    }
    for category in 0..CATEGORY_COUNT {
        if include_goal {
            features.push(FeatureKind::GoalCat(category));
        }
        if include_effect {
            features.push(FeatureKind::EffectCat(category));
        }
        if include_match_gate {
            features.push(FeatureKind::CategoryConjunction(category));
        }
        if include_shifted_gate {
            features.push(FeatureKind::ShiftedConjunction(category));
        }
    }
    features.push(FeatureKind::Bias);
    features
}

fn allowed_features(format_arm: FormatArm, branch: Branch) -> Vec<FeatureKind> {
    match branch {
        Branch::Direct => {
            if format_arm.uses_direct() {
                vec![FeatureKind::Surface, FeatureKind::Bias]
            } else {
                vec![]
            }
        }
        Branch::Process => match format_arm {
            FormatArm::AnswerOnly => vec![],
            FormatArm::OracleMatchBits => vec![FeatureKind::OracleMatch, FeatureKind::Bias],
            FormatArm::RawSymbolicProcess => category_features(false, false, true, false, false),
            FormatArm::ProseProcess => category_features(true, true, false, false, true),
            FormatArm::InnerMonologueProcess => category_features(true, true, false, true, true),
            FormatArm::StrictJsonProcess => category_features(true, true, true, false, false),
            FormatArm::FlatKeyValueProcess => category_features(true, true, false, false, false),
            FormatArm::RelationalTriplesProcess => category_features(true, true, true, false, false),
            FormatArm::ActionOutcomeTableProcess => category_features(false, false, true, false, true),
            FormatArm::RelationPlusActionProcess => category_features(true, true, true, false, true),
            FormatArm::CompactHybridProcess => category_features(false, false, true, false, true),
            FormatArm::ShuffledCompactHybrid => category_features(true, true, false, true, false),
        },
    }
}

fn feature_value(feature: FeatureKind, ex: &Example, candidate: usize) -> f64 {
    match feature {
        FeatureKind::Surface => ex.surface_priors[candidate],
        FeatureKind::GoalCat(category) => (ex.goal_category == category) as u8 as f64,
        FeatureKind::EffectCat(category) => (ex.effect_categories[candidate] == category) as u8 as f64,
        FeatureKind::CategoryConjunction(category) => {
            (ex.goal_category == category && ex.effect_categories[candidate] == category) as u8 as f64
        }
        FeatureKind::ShiftedConjunction(category) => {
            let shifted = (category + 1) % CATEGORY_COUNT;
            (ex.goal_category == category && ex.effect_categories[candidate] == shifted) as u8 as f64
        }
        FeatureKind::OracleMatch => ex.match_bits[candidate] as f64,
        FeatureKind::Bias => 1.0,
    }
}

fn target_bit(format_arm: FormatArm, ex: &Example, candidate: usize) -> u8 {
    if format_arm.is_shuffled() {
        (ex.effect_categories[candidate] == (ex.goal_category + 1) % CATEGORY_COUNT) as u8
    } else {
        ex.match_bits[candidate]
    }
}

fn branch_scores(genome: &Genome, branch: Branch, ex: &Example) -> [f64; CANDIDATE_COUNT] {
    let mut scores = [0.0; CANDIDATE_COUNT];
    for edge in &genome.edges {
        if edge.branch != branch {
            continue;
        }
        for candidate in 0..CANDIDATE_COUNT {
            scores[candidate] += edge.weight * feature_value(edge.feature, ex, candidate);
        }
    }
    scores
}

fn final_scores(genome: &Genome, format_arm: FormatArm, ex: &Example) -> [f64; CANDIDATE_COUNT] {
    if format_arm.uses_direct() {
        branch_scores(genome, Branch::Direct, ex)
    } else {
        branch_scores(genome, Branch::Process, ex)
    }
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

fn answer_score(genome: &Genome, format_arm: FormatArm, rows: &[Example]) -> f64 {
    rows.iter()
        .map(|ex| softmax_gold(final_scores(genome, format_arm, ex), ex.answer_label))
        .sum::<f64>()
        / rows.len() as f64
}

fn process_score(genome: &Genome, format_arm: FormatArm, rows: &[Example]) -> f64 {
    if !format_arm.uses_process() {
        return 0.5;
    }
    let mut pos_total = 0usize;
    let mut neg_total = 0usize;
    let mut pos_score = 0.0;
    let mut neg_score = 0.0;
    for ex in rows {
        let scores = branch_scores(genome, Branch::Process, ex);
        for i in 0..CANDIDATE_COUNT {
            if target_bit(format_arm, ex, i) == 1 {
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

fn fitness(genome: &Genome, format_arm: FormatArm, rows: &[Example], aux_weight: f64) -> f64 {
    let answer = answer_score(genome, format_arm, rows);
    if format_arm.uses_process() {
        answer + aux_weight * process_score(genome, format_arm, rows)
    } else {
        answer
    }
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

fn accuracy(genome: &Genome, format_arm: FormatArm, rows: &[Example]) -> f64 {
    rows.iter()
        .filter(|ex| predict(final_scores(genome, format_arm, ex)) == ex.answer_label)
        .count() as f64
        / rows.len() as f64
}

fn shortcut_trap_rate(genome: &Genome, format_arm: FormatArm, rows: &[Example]) -> f64 {
    let mut opportunities = 0usize;
    let mut traps = 0usize;
    for ex in rows {
        if ex.surface_shortcut_label != ex.answer_label {
            opportunities += 1;
            if predict(final_scores(genome, format_arm, ex)) == ex.surface_shortcut_label {
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

fn process_bit_accuracy(genome: &Genome, format_arm: FormatArm, rows: &[Example]) -> f64 {
    if !format_arm.uses_process() {
        return 0.0;
    }
    let mut total = 0usize;
    let mut correct = 0usize;
    for ex in rows {
        let scores = branch_scores(genome, Branch::Process, ex);
        for i in 0..CANDIDATE_COUNT {
            total += 1;
            correct += ((scores[i] > 0.0) as u8 == target_bit(format_arm, ex, i)) as usize;
        }
    }
    correct as f64 / total as f64
}

fn true_process_bit_accuracy(genome: &Genome, format_arm: FormatArm, rows: &[Example]) -> f64 {
    if !format_arm.uses_process() {
        return 0.0;
    }
    let mut total = 0usize;
    let mut correct = 0usize;
    for ex in rows {
        let scores = branch_scores(genome, Branch::Process, ex);
        for i in 0..CANDIDATE_COUNT {
            total += 1;
            correct += ((scores[i] > 0.0) as u8 == ex.match_bits[i]) as usize;
        }
    }
    correct as f64 / total as f64
}

fn process_exact_row_accuracy(genome: &Genome, format_arm: FormatArm, rows: &[Example]) -> f64 {
    if !format_arm.uses_process() {
        return 0.0;
    }
    rows.iter()
        .filter(|ex| {
            let scores = branch_scores(genome, Branch::Process, ex);
            (0..CANDIDATE_COUNT).all(|i| (scores[i] > 0.0) as u8 == target_bit(format_arm, ex, i))
        })
        .count() as f64
        / rows.len() as f64
}

fn true_process_exact_row_accuracy(genome: &Genome, format_arm: FormatArm, rows: &[Example]) -> f64 {
    if !format_arm.uses_process() {
        return 0.0;
    }
    rows.iter()
        .filter(|ex| {
            let scores = branch_scores(genome, Branch::Process, ex);
            (0..CANDIDATE_COUNT).all(|i| (scores[i] > 0.0) as u8 == ex.match_bits[i])
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

fn mutate(parent: &Genome, format_arm: FormatArm, cfg: &Config, rng: &mut Rng) -> Genome {
    let mut child = parent.clone();
    let branch = if format_arm.uses_direct() {
        Branch::Direct
    } else {
        Branch::Process
    };
    let roll = rng.f64();
    if child.edges.is_empty() || (roll < 0.45 && child.edges.len() < cfg.edge_cap) {
        let features = allowed_features(format_arm, branch);
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
        let features = allowed_features(format_arm, child.edges[idx].branch);
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
    let mut rng = Rng::new(cfg.seed ^ 0xA17C_006u64);
    let mut parent = if cfg.format_arm.is_oracle() {
        Genome {
            edges: vec![Edge {
                branch: Branch::Process,
                feature: FeatureKind::OracleMatch,
                weight: 8.0,
            }],
        }
    } else {
        Genome::default()
    };
    let mut parent_score = fitness(&parent, cfg.format_arm, &dataset.train_examples, cfg.aux_weight);
    let mut accepted = 0usize;
    for _ in 0..cfg.max_steps {
        let mut best = parent.clone();
        let mut best_score = parent_score;
        for _ in 0..cfg.proposals {
            let candidate = mutate(&parent, cfg.format_arm, cfg, &mut rng);
            let score = fitness(&candidate, cfg.format_arm, &dataset.train_examples, cfg.aux_weight);
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
                "branch": match edge.branch { Branch::Direct => "direct", Branch::Process => "process" },
                "feature": edge.feature.as_str(),
                "weight": edge.weight,
            })
        })
        .collect()
}

fn feature_names(format_arm: FormatArm) -> Vec<String> {
    let branch = if format_arm.uses_direct() {
        Branch::Direct
    } else {
        Branch::Process
    };
    allowed_features(format_arm, branch)
        .iter()
        .map(|feature| feature.as_str())
        .collect()
}

fn main() {
    let cfg = parse_args();
    fs::create_dir_all(&cfg.out_dir).expect("failed to create out dir");
    let raw = fs::read_to_string(&cfg.dataset).expect("failed to read dataset");
    let dataset: Dataset = serde_json::from_str(&raw).expect("failed to parse dataset");
    let train_align = surface_alignment(&dataset.train_examples);
    let eval_flip = 1.0 - surface_alignment(&dataset.eval_examples);
    let invalid_stress = train_align < 0.85 || eval_flip < 0.85;
    let (genome, train_fitness, accepted_steps) = evolve(&dataset, &cfg);
    let payload = json!({
        "status": if invalid_stress { "ANCHOR_MINI_006_JOB_INVALID_STRESS" } else { "ANCHOR_MINI_006_JOB_COMPLETE" },
        "config": {
            "format_arm": cfg.format_arm.as_str(),
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
            "invalid_stress": invalid_stress,
        },
        "metrics": {
            "answer_train_accuracy": accuracy(&genome, cfg.format_arm, &dataset.train_examples),
            "answer_eval_ood_accuracy": accuracy(&genome, cfg.format_arm, &dataset.eval_examples),
            "answer_train_score": answer_score(&genome, cfg.format_arm, &dataset.train_examples),
            "answer_eval_score": answer_score(&genome, cfg.format_arm, &dataset.eval_examples),
            "process_train_score": process_score(&genome, cfg.format_arm, &dataset.train_examples),
            "process_eval_score": process_score(&genome, cfg.format_arm, &dataset.eval_examples),
            "process_bit_accuracy": process_bit_accuracy(&genome, cfg.format_arm, &dataset.eval_examples),
            "process_exact_row_accuracy": process_exact_row_accuracy(&genome, cfg.format_arm, &dataset.eval_examples),
            "true_process_bit_accuracy": true_process_bit_accuracy(&genome, cfg.format_arm, &dataset.eval_examples),
            "true_process_exact_row_accuracy": true_process_exact_row_accuracy(&genome, cfg.format_arm, &dataset.eval_examples),
            "shortcut_trap_rate": shortcut_trap_rate(&genome, cfg.format_arm, &dataset.eval_examples),
            "train_fitness": train_fitness,
            "accepted_steps": accepted_steps,
            "edge_count": genome.edges.len(),
        },
        "feature_visibility": {
            "oracle_match_visible": cfg.format_arm.is_oracle(),
            "non_oracle_eval_uses_match_bits_as_input": false,
            "format_arm": cfg.format_arm.as_str(),
            "allowed_features": feature_names(cfg.format_arm),
            "target": if cfg.format_arm.is_shuffled() {
                "one_hot_goal_plus_one_effect_category"
            } else {
                "true_goal_effect_match"
            },
            "is_non_oracle_learned": cfg.format_arm.is_non_oracle_learned(),
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
