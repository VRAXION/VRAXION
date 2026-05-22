//! ANCHOR-MINI-004 sparse mutation carrier for the MINI-003 shortcut-flip task.
//!
//! This is an audit-sized sparse mutation-selection carrier. It intentionally
//! keeps the carrier small so direct/routed/hybrid decision paths are easy to
//! inspect. It is not the full production grower.

use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::path::PathBuf;

const CANDIDATE_COUNT: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carrier {
    SparseDirect,
    SparseAuxDirect,
    SparseRouted,
    SparseHybrid,
    SparseShuffledRouted,
}

impl Carrier {
    fn parse(raw: &str) -> Self {
        match raw {
            "SPARSE_DIRECT" => Self::SparseDirect,
            "SPARSE_AUX_DIRECT" => Self::SparseAuxDirect,
            "SPARSE_ROUTED" => Self::SparseRouted,
            "SPARSE_HYBRID" => Self::SparseHybrid,
            "SPARSE_SHUFFLED_ROUTED" => Self::SparseShuffledRouted,
            other => panic!("unknown carrier: {other}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::SparseDirect => "SPARSE_DIRECT",
            Self::SparseAuxDirect => "SPARSE_AUX_DIRECT",
            Self::SparseRouted => "SPARSE_ROUTED",
            Self::SparseHybrid => "SPARSE_HYBRID",
            Self::SparseShuffledRouted => "SPARSE_SHUFFLED_ROUTED",
        }
    }

    fn uses_direct(self) -> bool {
        matches!(self, Self::SparseDirect | Self::SparseAuxDirect | Self::SparseHybrid)
    }

    fn uses_process(self) -> bool {
        matches!(
            self,
            Self::SparseAuxDirect | Self::SparseRouted | Self::SparseHybrid | Self::SparseShuffledRouted
        )
    }

    fn final_uses_direct(self) -> bool {
        matches!(self, Self::SparseDirect | Self::SparseAuxDirect | Self::SparseHybrid)
    }

    fn final_uses_process(self) -> bool {
        matches!(self, Self::SparseRouted | Self::SparseHybrid | Self::SparseShuffledRouted)
    }

    fn uses_shuffled_process(self) -> bool {
        matches!(self, Self::SparseShuffledRouted)
    }
}

#[derive(Clone, Copy, Debug)]
enum Branch {
    Direct,
    Process,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FeatureKind {
    Surface,
    Match,
    ShuffledMatch,
    Bias,
}

impl FeatureKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Surface => "surface",
            Self::Match => "match",
            Self::ShuffledMatch => "shuffled_match",
            Self::Bias => "bias",
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
    surface_priors: Vec<f64>,
    answer_label: usize,
    match_bits: Vec<u8>,
    shuffled_match_bits: Vec<u8>,
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
    let mut carrier = Carrier::SparseRouted;
    let mut seed = 2026u64;
    let mut max_steps = 1600usize;
    let mut proposals = 9usize;
    let mut edge_cap = 16usize;
    let mut aux_weight = 2.0f64;
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

fn allowed_features(carrier: Carrier, branch: Branch) -> Vec<FeatureKind> {
    match branch {
        Branch::Direct => {
            if carrier.uses_direct() {
                vec![FeatureKind::Surface, FeatureKind::Bias]
            } else {
                vec![]
            }
        }
        Branch::Process => {
            if !carrier.uses_process() {
                vec![]
            } else if carrier.uses_shuffled_process() {
                vec![FeatureKind::ShuffledMatch, FeatureKind::Bias]
            } else {
                vec![FeatureKind::Match, FeatureKind::Bias]
            }
        }
    }
}

fn feature_value(feature: FeatureKind, ex: &Example, candidate: usize) -> f64 {
    match feature {
        FeatureKind::Surface => ex.surface_priors[candidate],
        FeatureKind::Match => ex.match_bits[candidate] as f64,
        FeatureKind::ShuffledMatch => ex.shuffled_match_bits[candidate] as f64,
        FeatureKind::Bias => 1.0,
    }
}

fn branch_scores(genome: &Genome, branch: Branch, ex: &Example) -> [f64; CANDIDATE_COUNT] {
    let mut scores = [0.0; CANDIDATE_COUNT];
    for edge in &genome.edges {
        if std::mem::discriminant(&edge.branch) != std::mem::discriminant(&branch) {
            continue;
        }
        for candidate in 0..CANDIDATE_COUNT {
            scores[candidate] += edge.weight * feature_value(edge.feature, ex, candidate);
        }
    }
    scores
}

fn final_scores(genome: &Genome, carrier: Carrier, ex: &Example) -> [f64; CANDIDATE_COUNT] {
    let mut scores = [0.0; CANDIDATE_COUNT];
    if carrier.final_uses_direct() {
        let direct = branch_scores(genome, Branch::Direct, ex);
        for i in 0..CANDIDATE_COUNT {
            scores[i] += direct[i];
        }
    }
    if carrier.final_uses_process() {
        let process = branch_scores(genome, Branch::Process, ex);
        for i in 0..CANDIDATE_COUNT {
            scores[i] += process[i];
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

fn process_targets<'a>(carrier: Carrier, ex: &'a Example) -> &'a [u8] {
    if carrier.uses_shuffled_process() {
        &ex.shuffled_match_bits
    } else {
        &ex.match_bits
    }
}

fn answer_score(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    rows.iter()
        .map(|ex| softmax_gold(final_scores(genome, carrier, ex), ex.answer_label))
        .sum::<f64>()
        / rows.len() as f64
}

fn process_score(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    if !carrier.uses_process() {
        return 0.5;
    }
    let mut total = 0usize;
    let mut score = 0.0;
    for ex in rows {
        let targets = process_targets(carrier, ex);
        let scores = branch_scores(genome, Branch::Process, ex);
        for i in 0..CANDIDATE_COUNT {
            total += 1;
            score += if targets[i] == 1 {
                sigmoid(scores[i])
            } else {
                sigmoid(-scores[i])
            };
        }
    }
    score / total as f64
}

fn fitness(genome: &Genome, carrier: Carrier, rows: &[Example], aux_weight: f64) -> f64 {
    let answer = answer_score(genome, carrier, rows);
    if carrier.uses_process() {
        answer + aux_weight * process_score(genome, carrier, rows)
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

fn process_bit_accuracy(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    if !carrier.uses_process() {
        return 0.0;
    }
    let mut total = 0usize;
    let mut correct = 0usize;
    for ex in rows {
        let scores = branch_scores(genome, Branch::Process, ex);
        let targets = process_targets(carrier, ex);
        for i in 0..CANDIDATE_COUNT {
            total += 1;
            correct += ((scores[i] > 0.0) as u8 == targets[i]) as usize;
        }
    }
    correct as f64 / total as f64
}

fn process_exact_row_accuracy(genome: &Genome, carrier: Carrier, rows: &[Example]) -> f64 {
    if !carrier.uses_process() {
        return 0.0;
    }
    rows.iter()
        .filter(|ex| {
            let scores = branch_scores(genome, Branch::Process, ex);
            let targets = process_targets(carrier, ex);
            (0..CANDIDATE_COUNT).all(|i| (scores[i] > 0.0) as u8 == targets[i])
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
    if carrier.uses_direct() {
        branches.push(Branch::Direct);
    }
    if carrier.uses_process() {
        branches.push(Branch::Process);
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
    let mut rng = Rng::new(cfg.seed ^ 0xA17C_004u64);
    let mut parent = Genome::default();
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
                "branch": match edge.branch { Branch::Direct => "direct", Branch::Process => "process" },
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
        "status": if invalid_stress { "ANCHOR_MINI_004_JOB_INVALID_STRESS" } else { "ANCHOR_MINI_004_JOB_COMPLETE" },
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
            "answer_eval_ood_accuracy": accuracy(&genome, cfg.carrier, &dataset.eval_examples),
            "answer_train_score": answer_score(&genome, cfg.carrier, &dataset.train_examples),
            "answer_eval_score": answer_score(&genome, cfg.carrier, &dataset.eval_examples),
            "process_train_score": process_score(&genome, cfg.carrier, &dataset.train_examples),
            "process_eval_score": process_score(&genome, cfg.carrier, &dataset.eval_examples),
            "process_bit_accuracy": process_bit_accuracy(&genome, cfg.carrier, &dataset.eval_examples),
            "process_exact_row_accuracy": process_exact_row_accuracy(&genome, cfg.carrier, &dataset.eval_examples),
            "shortcut_trap_rate": shortcut_trap_rate(&genome, cfg.carrier, &dataset.eval_examples),
            "train_fitness": train_fitness,
            "accepted_steps": accepted_steps,
            "edge_count": genome.edges.len(),
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
