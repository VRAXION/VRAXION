//! ANCHOR-MINI-008 sparse process-first teaching probe.
//!
//! The carrier is an audit-sized sparse mutation-selection system. It tests
//! whether a scored PLAN route can close a flipped surface shortcut before the
//! final answer is emitted.

use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::path::PathBuf;

const CANDIDATE_COUNT: usize = 4;
const CATEGORY_COUNT: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carrier {
    SparseDirectAnswer,
    SparseAuxPlanDirectAnswer,
    SparsePlanFirst,
    SparsePlanFirstHybrid,
    SparseShuffledTeacher,
    SparseShortcutTeacher,
    SparseOraclePlanVisible,
}

impl Carrier {
    fn parse(raw: &str) -> Self {
        match raw {
            "SPARSE_DIRECT_ANSWER" => Self::SparseDirectAnswer,
            "SPARSE_AUX_PLAN_DIRECT_ANSWER" => Self::SparseAuxPlanDirectAnswer,
            "SPARSE_PLAN_FIRST" => Self::SparsePlanFirst,
            "SPARSE_PLAN_FIRST_HYBRID" => Self::SparsePlanFirstHybrid,
            "SPARSE_SHUFFLED_TEACHER" => Self::SparseShuffledTeacher,
            "SPARSE_SHORTCUT_TEACHER" => Self::SparseShortcutTeacher,
            "SPARSE_ORACLE_PLAN_VISIBLE" => Self::SparseOraclePlanVisible,
            other => panic!("unknown carrier: {other}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::SparseDirectAnswer => "SPARSE_DIRECT_ANSWER",
            Self::SparseAuxPlanDirectAnswer => "SPARSE_AUX_PLAN_DIRECT_ANSWER",
            Self::SparsePlanFirst => "SPARSE_PLAN_FIRST",
            Self::SparsePlanFirstHybrid => "SPARSE_PLAN_FIRST_HYBRID",
            Self::SparseShuffledTeacher => "SPARSE_SHUFFLED_TEACHER",
            Self::SparseShortcutTeacher => "SPARSE_SHORTCUT_TEACHER",
            Self::SparseOraclePlanVisible => "SPARSE_ORACLE_PLAN_VISIBLE",
        }
    }

    fn uses_direct_branch(self) -> bool {
        matches!(
            self,
            Self::SparseDirectAnswer
                | Self::SparseAuxPlanDirectAnswer
                | Self::SparsePlanFirstHybrid
        )
    }

    fn uses_plan_branch(self) -> bool {
        !matches!(self, Self::SparseDirectAnswer)
    }

    fn final_uses_direct(self) -> bool {
        matches!(
            self,
            Self::SparseDirectAnswer
                | Self::SparseAuxPlanDirectAnswer
                | Self::SparsePlanFirstHybrid
        )
    }

    fn final_uses_policy(self) -> bool {
        matches!(
            self,
            Self::SparsePlanFirst
                | Self::SparsePlanFirstHybrid
                | Self::SparseShuffledTeacher
                | Self::SparseShortcutTeacher
                | Self::SparseOraclePlanVisible
        )
    }

    fn is_oracle(self) -> bool {
        matches!(self, Self::SparseOraclePlanVisible)
    }

    fn uses_shuffled_teacher(self) -> bool {
        matches!(self, Self::SparseShuffledTeacher)
    }

    fn uses_shortcut_teacher(self) -> bool {
        matches!(self, Self::SparseShortcutTeacher)
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
    Surface,
    GoalCat(usize),
    EffectCat(usize),
    CategoryConjunction(usize),
    ShiftedConjunction(usize),
    OraclePolicy,
    Bias,
}

impl FeatureKind {
    fn as_str(self) -> String {
        match self {
            Self::Surface => "surface".to_string(),
            Self::GoalCat(category) => format!("goal_cat_{category}"),
            Self::EffectCat(category) => format!("effect_cat_{category}"),
            Self::CategoryConjunction(category) => format!("goal_and_effect_cat_{category}"),
            Self::ShiftedConjunction(category) => {
                format!("goal_and_shifted_effect_cat_{category}")
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
                feature: FeatureKind::Surface,
                weight: 5.0,
            });
        }
        genome
    }
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
    carrier: Carrier,
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
    let mut carrier = Carrier::SparsePlanFirst;
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
        seed,
        max_steps,
        proposals,
        edge_cap,
        aux_weight,
    }
}

fn category_features() -> Vec<FeatureKind> {
    let mut features = Vec::new();
    for category in 0..CATEGORY_COUNT {
        features.push(FeatureKind::GoalCat(category));
        features.push(FeatureKind::EffectCat(category));
        features.push(FeatureKind::CategoryConjunction(category));
        features.push(FeatureKind::ShiftedConjunction(category));
    }
    features.push(FeatureKind::Bias);
    features
}

fn allowed_features(carrier: Carrier, branch: Branch) -> Vec<FeatureKind> {
    match branch {
        Branch::Direct => {
            if carrier.uses_direct_branch() {
                vec![FeatureKind::Surface, FeatureKind::Bias]
            } else {
                vec![]
            }
        }
        Branch::Shortcut => {
            if carrier.uses_plan_branch() {
                vec![FeatureKind::Surface, FeatureKind::Bias]
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
                vec![FeatureKind::Surface, FeatureKind::Bias]
            } else {
                category_features()
            }
        }
    }
}

fn shifted_category(goal_category: usize) -> usize {
    (goal_category + 1) % CATEGORY_COUNT
}

fn target_policy_bit(carrier: Carrier, ex: &Example, candidate: usize) -> u8 {
    if carrier.uses_shuffled_teacher() {
        (ex.effect_categories[candidate] == shifted_category(ex.goal_category)) as u8
    } else if carrier.uses_shortcut_teacher() {
        (candidate == ex.surface_shortcut_label) as u8
    } else {
        ex.match_bits[candidate]
    }
}

fn true_policy_bit(ex: &Example, candidate: usize) -> u8 {
    ex.match_bits[candidate]
}

fn teacher_answer_label(carrier: Carrier, ex: &Example) -> usize {
    for candidate in 0..CANDIDATE_COUNT {
        if target_policy_bit(carrier, ex, candidate) == 1 {
            return candidate;
        }
    }
    ex.answer_label
}

fn feature_value(feature: FeatureKind, carrier: Carrier, ex: &Example, candidate: usize) -> f64 {
    match feature {
        FeatureKind::Surface => ex.surface_priors[candidate],
        FeatureKind::GoalCat(category) => (ex.goal_category == category) as u8 as f64,
        FeatureKind::EffectCat(category) => (ex.effect_categories[candidate] == category) as u8 as f64,
        FeatureKind::CategoryConjunction(category) => {
            (ex.goal_category == category && ex.effect_categories[candidate] == category) as u8 as f64
        }
        FeatureKind::ShiftedConjunction(category) => {
            let shifted = shifted_category(category);
            (ex.goal_category == category && ex.effect_categories[candidate] == shifted) as u8 as f64
        }
        FeatureKind::OraclePolicy => target_policy_bit(carrier, ex, candidate) as f64,
        FeatureKind::Bias => 1.0,
    }
}

fn branch_scores(
    genome: &Genome,
    branch: Branch,
    carrier: Carrier,
    ex: &Example,
) -> [f64; CANDIDATE_COUNT] {
    let mut scores = [0.0; CANDIDATE_COUNT];
    for edge in &genome.edges {
        if edge.branch != branch {
            continue;
        }
        for candidate in 0..CANDIDATE_COUNT {
            scores[candidate] += edge.weight * feature_value(edge.feature, carrier, ex, candidate);
        }
    }
    scores
}

fn final_scores(genome: &Genome, carrier: Carrier, ex: &Example) -> [f64; CANDIDATE_COUNT] {
    let mut scores = [0.0; CANDIDATE_COUNT];
    if carrier.final_uses_direct() {
        let direct = branch_scores(genome, Branch::Direct, carrier, ex);
        for i in 0..CANDIDATE_COUNT {
            scores[i] += direct[i];
        }
    }
    if carrier.final_uses_policy() {
        let policy = branch_scores(genome, Branch::Policy, carrier, ex);
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

fn answer_score(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    rows.iter()
        .map(|ex| softmax_gold(final_scores(genome, carrier, ex), ex.answer_label))
        .sum::<f64>()
        / rows.len() as f64
}

fn fitness_answer_score(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    rows.iter()
        .map(|ex| softmax_gold(final_scores(genome, carrier, ex), teacher_answer_label(carrier, ex)))
        .sum::<f64>()
        / rows.len() as f64
}

fn policy_bit_score(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.5;
    }
    let mut pos_total = 0usize;
    let mut neg_total = 0usize;
    let mut pos_score = 0.0;
    let mut neg_score = 0.0;
    for ex in rows {
        let scores = branch_scores(genome, Branch::Policy, carrier, ex);
        for i in 0..CANDIDATE_COUNT {
            if target_policy_bit(carrier, ex, i) == 1 {
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

fn shortcut_detection_score(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.5;
    }
    rows.iter()
        .map(|ex| {
            softmax_gold(
                branch_scores(genome, Branch::Shortcut, carrier, ex),
                ex.surface_shortcut_label,
            )
        })
        .sum::<f64>()
        / rows.len() as f64
}

fn shortcut_validity_score(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.5;
    }
    rows.iter()
        .map(|ex| {
            let policy = branch_scores(genome, Branch::Policy, carrier, ex);
            let observed = ex.surface_shortcut_label;
            if target_policy_bit(carrier, ex, observed) == 1 {
                sigmoid(policy[observed])
            } else {
                sigmoid(-policy[observed])
            }
        })
        .sum::<f64>()
        / rows.len() as f64
}

fn invalid_shortcut_rejection_score(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.5;
    }
    let mut total = 0usize;
    let mut score = 0.0;
    for ex in rows {
        let observed = ex.surface_shortcut_label;
        if target_policy_bit(carrier, ex, observed) == 0 {
            total += 1;
            let policy = branch_scores(genome, Branch::Policy, carrier, ex);
            score += sigmoid(-policy[observed]);
        }
    }
    if total == 0 {
        0.5
    } else {
        score / total as f64
    }
}

fn plan_score(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.5;
    }
    0.40 * shortcut_detection_score(genome, carrier, rows)
        + 0.15 * shortcut_validity_score(genome, carrier, rows)
        + 0.15 * invalid_shortcut_rejection_score(genome, carrier, rows)
        + 0.30 * policy_bit_score(genome, carrier, rows)
}

fn fitness(genome: &Genome, carrier: Carrier, rows: &[Example], aux_weight: f64) -> f64 {
    let answer = fitness_answer_score(genome, carrier, rows);
    if carrier.uses_plan_branch() {
        answer + aux_weight * plan_score(genome, carrier, rows)
    } else {
        answer
    }
}

fn accuracy(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    rows.iter()
        .filter(|ex| predict(final_scores(genome, carrier, ex)) == ex.answer_label)
        .count() as f64
        / rows.len() as f64
}

fn shortcut_trap_rate(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    let mut opportunities = 0usize;
    let mut traps = 0usize;
    for ex in rows {
        if ex.surface_shortcut_label != ex.answer_label {
            opportunities += 1;
            if predict(final_scores(genome, carrier, ex)) == ex.surface_shortcut_label {
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

fn observed_shortcut_accuracy(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.0;
    }
    rows.iter()
        .filter(|ex| {
            predict(branch_scores(genome, Branch::Shortcut, carrier, ex)) == ex.surface_shortcut_label
        })
        .count() as f64
        / rows.len() as f64
}

fn shortcut_validity_accuracy(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.0;
    }
    rows.iter()
        .filter(|ex| {
            let policy = branch_scores(genome, Branch::Policy, carrier, ex);
            let observed = ex.surface_shortcut_label;
            let predicted_valid = policy[observed] > 0.0;
            let true_valid = true_policy_bit(ex, observed) == 1;
            predicted_valid == true_valid
        })
        .count() as f64
        / rows.len() as f64
}

fn invalid_shortcut_rejection_accuracy(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.0;
    }
    let mut total = 0usize;
    let mut correct = 0usize;
    for ex in rows {
        let observed = ex.surface_shortcut_label;
        if true_policy_bit(ex, observed) == 0 {
            total += 1;
            let policy = branch_scores(genome, Branch::Policy, carrier, ex);
            correct += (policy[observed] <= 0.0) as usize;
        }
    }
    if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    }
}

fn policy_bit_accuracy(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.0;
    }
    let mut total = 0usize;
    let mut correct = 0usize;
    for ex in rows {
        let scores = branch_scores(genome, Branch::Policy, carrier, ex);
        for i in 0..CANDIDATE_COUNT {
            total += 1;
            correct += ((scores[i] > 0.0) as u8 == true_policy_bit(ex, i)) as usize;
        }
    }
    correct as f64 / total as f64
}

fn plan_exact_row_accuracy(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.0;
    }
    rows.iter()
        .filter(|ex| {
            let shortcut_ok =
                predict(branch_scores(genome, Branch::Shortcut, carrier, ex)) == ex.surface_shortcut_label;
            let policy = branch_scores(genome, Branch::Policy, carrier, ex);
            let policy_ok =
                (0..CANDIDATE_COUNT).all(|i| (policy[i] > 0.0) as u8 == true_policy_bit(ex, i));
            shortcut_ok && policy_ok
        })
        .count() as f64
        / rows.len() as f64
}

fn answer_from_plan_consistency(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    if !carrier.uses_plan_branch() {
        return 0.0;
    }
    rows.iter()
        .filter(|ex| {
            predict(final_scores(genome, carrier, ex))
                == predict(branch_scores(genome, Branch::Policy, carrier, ex))
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
    let mut rng = Rng::new(cfg.seed ^ 0xA17C_008u64);
    let mut parent = Genome::initial(cfg.carrier);
    let mut parent_score = fitness(&parent, cfg.carrier, &dataset.train_examples, cfg.aux_weight);
    let mut accepted = 0usize;
    for _ in 0..cfg.max_steps {
        let mut best = parent.clone();
        let mut best_score = parent_score;
        for _ in 0..cfg.proposals {
            let candidate = mutate(&parent, cfg.carrier, cfg, &mut rng);
            let score = fitness(&candidate, cfg.carrier, &dataset.train_examples, cfg.aux_weight);
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
    let dataset: Dataset = serde_json::from_str(&raw).expect("failed to parse dataset");
    let train_align = surface_alignment(&dataset.train_examples);
    let eval_flip = 1.0 - surface_alignment(&dataset.eval_examples);
    let invalid_stress = train_align < 0.85 || eval_flip < 0.85;
    let (genome, train_fitness, accepted_steps) = evolve(&dataset, &cfg);
    let payload = json!({
        "status": if invalid_stress {
            "ANCHOR_MINI_008_JOB_INVALID_STRESS"
        } else {
            "ANCHOR_MINI_008_JOB_COMPLETE"
        },
        "config": {
            "carrier": cfg.carrier.as_str(),
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
            "answer_train_accuracy": accuracy(&genome, cfg.carrier, &dataset.train_examples),
            "answer_eval_accuracy": accuracy(&genome, cfg.carrier, &dataset.eval_examples),
            "answer_train_score": answer_score(&genome, cfg.carrier, &dataset.train_examples),
            "answer_eval_score": answer_score(&genome, cfg.carrier, &dataset.eval_examples),
            "shortcut_trap_rate": shortcut_trap_rate(&genome, cfg.carrier, &dataset.eval_examples),
            "shortcut_detection_score": shortcut_detection_score(&genome, cfg.carrier, &dataset.eval_examples),
            "shortcut_validity_score": shortcut_validity_score(&genome, cfg.carrier, &dataset.eval_examples),
            "invalid_shortcut_rejection_score": invalid_shortcut_rejection_score(&genome, cfg.carrier, &dataset.eval_examples),
            "policy_bit_score": policy_bit_score(&genome, cfg.carrier, &dataset.eval_examples),
            "plan_score": plan_score(&genome, cfg.carrier, &dataset.eval_examples),
            "observed_shortcut_accuracy": observed_shortcut_accuracy(&genome, cfg.carrier, &dataset.eval_examples),
            "shortcut_validity_accuracy": shortcut_validity_accuracy(&genome, cfg.carrier, &dataset.eval_examples),
            "invalid_shortcut_rejection_accuracy": invalid_shortcut_rejection_accuracy(&genome, cfg.carrier, &dataset.eval_examples),
            "policy_bit_accuracy": policy_bit_accuracy(&genome, cfg.carrier, &dataset.eval_examples),
            "plan_exact_row_accuracy": plan_exact_row_accuracy(&genome, cfg.carrier, &dataset.eval_examples),
            "answer_from_plan_consistency": answer_from_plan_consistency(&genome, cfg.carrier, &dataset.eval_examples),
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
                "surface_prior(candidate) for direct/shortcut branches",
                "goal_category_indicator(k)",
                "effect_category_indicator(candidate,k)",
                "goal_and_effect_category_conjunction(candidate,k)",
                "goal_and_shifted_effect_category_conjunction(candidate,k)",
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
