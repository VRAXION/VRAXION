//! Phase D9.0b: direct H=256 genome landscape probe.
//!
//! Offline-only diagnostic. Loads existing checkpoints, freezes the projection,
//! mutates only the persisted core genome (edges, threshold, channel, polarity),
//! and evaluates local neighborhoods around the base genome.

use instnct_core::{
    load_checkpoint, save_checkpoint, softmax, CheckpointMeta, InitConfig, Int8Projection, Network,
    VcbpTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::collections::HashSet;
use std::env;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

const MAX_CHARGE: i32 = 7;
const DEFAULT_EVAL_LEN: usize = 1_000;
const PANEL_PROBE_COUNT: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
struct DirectGenomeCoord {
    h: usize,
    edge_bits: Vec<u8>,
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MutationType {
    Edge,
    Threshold,
    Channel,
    Polarity,
    Projection,
    Mixed,
}

impl MutationType {
    fn parse(value: &str) -> Self {
        match value {
            "edge" => Self::Edge,
            "threshold" | "theta" => Self::Threshold,
            "channel" => Self::Channel,
            "polarity" => Self::Polarity,
            "projection" | "proj" | "readout" => Self::Projection,
            "mixed" => Self::Mixed,
            other => panic!("unknown mutation type: {other}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Edge => "edge",
            Self::Threshold => "threshold",
            Self::Channel => "channel",
            Self::Polarity => "polarity",
            Self::Projection => "projection",
            Self::Mixed => "mixed",
        }
    }
}

#[derive(Default, Clone, Debug)]
struct MutationCounts {
    edge: usize,
    threshold: usize,
    channel: usize,
    polarity: usize,
    projection: usize,
}

#[derive(Clone, Debug)]
struct PanelMetrics {
    panel_probe_acc: f64,
    unique_predictions: usize,
    collision_rate: f64,
    f_active: f64,
    h_output_mean: f64,
    h_output_var: f64,
    stable_rank: f64,
    kernel_rank: usize,
    separation_sp: f64,
}

#[derive(Clone, Debug)]
struct Cli {
    checkpoints: Vec<PathBuf>,
    h: usize,
    mode: String,
    score_mode: String,
    radii: Vec<usize>,
    mutation_types: Vec<MutationType>,
    samples_per_type: usize,
    samples_per_tile: usize,
    lat_bins: usize,
    lon_bins: usize,
    target_tiles: Option<Vec<(usize, usize)>>,
    sample_layer: String,
    climbers_per_tile: usize,
    climb_steps: usize,
    accept_epsilon: f64,
    export_top_endpoints: usize,
    endpoint_eval_len: usize,
    endpoint_eval_seeds: Vec<u64>,
    bridge_endpoints: Vec<PathBuf>,
    bridge_steps: usize,
    overlap_samples: usize,
    robustness_eval_lens: Vec<usize>,
    repair_start: Option<PathBuf>,
    repair_samples_per_bucket: usize,
    repair_eval_seeds: Vec<u64>,
    repair_export_top: usize,
    mo_climbers: usize,
    mo_steps: usize,
    mo_eval_seeds: Vec<u64>,
    mo_export_top: usize,
    out: PathBuf,
    seed: u64,
    eval_len: usize,
    packed: PathBuf,
    corpus: PathBuf,
    input_scatter: bool,
}

#[derive(Serialize)]
struct RunMeta {
    phase: &'static str,
    tool: &'static str,
    h: usize,
    mode: String,
    score_mode: String,
    seed: u64,
    eval_len: usize,
    samples_per_type: usize,
    samples_per_tile: usize,
    lat_bins: Option<usize>,
    lon_bins: Option<usize>,
    target_tiles: Option<Vec<String>>,
    sample_layer: String,
    climbers_per_tile: Option<usize>,
    climb_steps: Option<usize>,
    accept_epsilon: Option<f64>,
    export_top_endpoints: Option<usize>,
    endpoint_eval_len: Option<usize>,
    endpoint_eval_seeds: Option<Vec<u64>>,
    bridge_endpoints: Option<Vec<String>>,
    bridge_steps: Option<usize>,
    overlap_samples: Option<usize>,
    robustness_eval_lens: Option<Vec<usize>>,
    repair_start: Option<String>,
    repair_samples_per_bucket: Option<usize>,
    repair_eval_seeds: Option<Vec<u64>>,
    repair_export_top: Option<usize>,
    mo_climbers: Option<usize>,
    mo_steps: Option<usize>,
    mo_eval_seeds: Option<Vec<u64>>,
    mo_export_top: Option<usize>,
    radii: Vec<usize>,
    mutation_types: Vec<String>,
    checkpoints: Vec<String>,
    projection_policy: &'static str,
    core_coord_dims: usize,
    packed: String,
    corpus: String,
    input_scatter: bool,
}

#[derive(Clone)]
struct EndpointCandidate {
    base_index: usize,
    base_checkpoint: String,
    base_step: usize,
    base_accuracy: f64,
    tile_id: String,
    lat_bin: usize,
    lon_bin: usize,
    climber_id: usize,
    climber_seed: u64,
    eval_seed: u64,
    best_step: usize,
    best_score: f64,
    start_score: f64,
    final_score: f64,
    mutation_type_at_best: String,
    accepted_seed_chain: Vec<u64>,
    net: Network,
    proj: Int8Projection,
}

#[derive(Serialize)]
struct EndpointSidecar<'a> {
    endpoint_rank: usize,
    endpoint_checkpoint: &'a str,
    base_index: usize,
    base_checkpoint: &'a str,
    base_step: usize,
    base_accuracy: f64,
    tile_id: &'a str,
    lat_bin: usize,
    lon_bin: usize,
    climber_id: usize,
    climber_seed: u64,
    eval_seed: u64,
    best_step: usize,
    best_score: f64,
    start_score: f64,
    climb_best_delta: f64,
    final_delta: f64,
    mutation_type_at_best: &'a str,
    accepted_seed_chain: &'a [u64],
    endpoint_eval_len: usize,
    endpoint_eval_seeds: &'a [u64],
    baseline_reeval_mean: f64,
    endpoint_reeval_mean: f64,
    reeval_delta_4000: f64,
    retention_pct: f64,
    delta_vs_baseline: f64,
    pass_retention_70: bool,
    pass_positive_delta: bool,
}

#[derive(Clone)]
struct RepairCandidate {
    proposal_id: usize,
    proposal_seed: u64,
    radius: usize,
    mutation_type: MutationType,
    counts: MutationCounts,
    direct_distance: usize,
    edges: usize,
    smooth_score: f64,
    smooth_delta: f64,
    accuracy_score: f64,
    accuracy_delta: f64,
    echo_score: f64,
    echo_delta: f64,
    unigram_score: f64,
    unigram_delta: f64,
    smooth_retain: bool,
    accuracy_retain: bool,
    echo_safe: bool,
    repair_class: &'static str,
    rank_score: f64,
    net: Network,
    proj: Int8Projection,
}

#[derive(Clone, Copy, Debug)]
struct MultiMetricScores {
    smooth: f64,
    accuracy: f64,
    echo: f64,
    unigram: f64,
}

#[derive(Clone)]
struct MultiObjectiveCandidate {
    climber_id: usize,
    step_index: usize,
    proposal_seed: u64,
    radius: usize,
    mutation_type: MutationType,
    counts: MutationCounts,
    direct_distance: usize,
    edges: usize,
    scores: MultiMetricScores,
    deltas: MultiMetricScores,
    mo_score: f64,
    mo_class: &'static str,
    accepted: bool,
    net: Network,
    proj: Int8Projection,
}

fn parse_csv_usize(value: &str) -> Vec<usize> {
    value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse::<usize>().expect("usize csv value"))
        .collect()
}

fn parse_csv_types(value: &str) -> Vec<MutationType> {
    value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| MutationType::parse(part.trim()))
        .collect()
}

fn parse_args() -> Cli {
    let mut checkpoints: Vec<PathBuf> = Vec::new();
    let mut h = 256usize;
    let mut mode = String::from("fail-fast");
    let mut score_mode = String::from("accuracy");
    let mut radii: Option<Vec<usize>> = None;
    let mut mutation_types: Option<Vec<MutationType>> = None;
    let mut samples_per_type = 200usize;
    let mut samples_per_tile = 1usize;
    let mut lat_bins = 16usize;
    let mut lon_bins = 32usize;
    let mut target_tiles: Option<Vec<(usize, usize)>> = None;
    let mut sample_layer = String::from("scout");
    let mut climbers_per_tile = 16usize;
    let mut climb_steps = 50usize;
    let mut accept_epsilon = 0.001f64;
    let mut export_top_endpoints = 0usize;
    let mut endpoint_eval_len = 4_000usize;
    let mut endpoint_eval_seeds = vec![730_001u64, 730_002, 730_003, 730_004];
    let mut bridge_endpoints: Vec<PathBuf> = Vec::new();
    let mut bridge_steps = 10usize;
    let mut overlap_samples = 64usize;
    let mut robustness_eval_lens = vec![1_000usize, 4_000, 16_000];
    let mut repair_start: Option<PathBuf> = None;
    let mut repair_samples_per_bucket = 20usize;
    let mut repair_eval_seeds = vec![
        940_001u64, 940_002, 940_003, 940_004, 940_005, 940_006, 940_007, 940_008,
    ];
    let mut repair_export_top = 8usize;
    let mut mo_climbers = 12usize;
    let mut mo_steps = 40usize;
    let mut mo_eval_seeds = vec![960_001u64, 960_002, 960_003, 960_004];
    let mut mo_export_top = 8usize;
    let mut out = PathBuf::from("output/phase_d9_direct_genome_landscape_20260428");
    let mut seed = 90210u64;
    let mut eval_len = DEFAULT_EVAL_LEN;
    let mut packed = PathBuf::from("output/block_c_bytepair_champion/packed.bin");
    let mut corpus = PathBuf::from("instnct-core/tests/fixtures/alice_corpus.txt");
    let mut input_scatter = false;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--checkpoint" => {
                checkpoints.push(PathBuf::from(args.next().expect("--checkpoint path")))
            }
            "--checkpoints" => {
                let value = args.next().expect("--checkpoints list");
                checkpoints.extend(
                    value
                        .split(',')
                        .filter(|part| !part.trim().is_empty())
                        .map(|part| PathBuf::from(part.trim())),
                );
            }
            "--H" | "--h" => h = args.next().expect("--H value").parse().expect("H"),
            "--mode" => mode = args.next().expect("--mode value"),
            "--score-mode" => {
                score_mode = args.next().expect("--score-mode value");
                assert!(
                    matches!(
                        score_mode.as_str(),
                        "accuracy" | "smooth" | "echo" | "unigram"
                    ),
                    "--score-mode expects accuracy|smooth|echo|unigram"
                );
            }
            "--radii" => radii = Some(parse_csv_usize(&args.next().expect("--radii csv"))),
            "--mutation-types" => {
                mutation_types = Some(parse_csv_types(&args.next().expect("--mutation-types csv")))
            }
            "--samples-per-type" => {
                samples_per_type = args
                    .next()
                    .expect("--samples-per-type value")
                    .parse()
                    .expect("samples-per-type")
            }
            "--samples-per-tile" => {
                samples_per_tile = args
                    .next()
                    .expect("--samples-per-tile value")
                    .parse()
                    .expect("samples-per-tile")
            }
            "--lat-bins" => {
                lat_bins = args
                    .next()
                    .expect("--lat-bins value")
                    .parse()
                    .expect("lat-bins")
            }
            "--lon-bins" => {
                lon_bins = args
                    .next()
                    .expect("--lon-bins value")
                    .parse()
                    .expect("lon-bins")
            }
            "--tiles" => {
                let value = args.next().expect("--tiles csv");
                let mut parsed = Vec::new();
                for part in value.split(',').filter(|part| !part.trim().is_empty()) {
                    let (lat, lon) = part
                        .trim()
                        .split_once('_')
                        .unwrap_or_else(|| panic!("tile id must be lat_lon, got {part}"));
                    parsed.push((
                        lat.parse::<usize>().expect("tile lat"),
                        lon.parse::<usize>().expect("tile lon"),
                    ));
                }
                target_tiles = Some(parsed);
            }
            "--sample-layer" => {
                sample_layer = args.next().expect("--sample-layer value");
                assert!(
                    matches!(sample_layer.as_str(), "scout" | "confirmed"),
                    "--sample-layer expects scout|confirmed"
                );
            }
            "--climbers-per-tile" => {
                climbers_per_tile = args
                    .next()
                    .expect("--climbers-per-tile value")
                    .parse()
                    .expect("climbers-per-tile")
            }
            "--climb-steps" => {
                climb_steps = args
                    .next()
                    .expect("--climb-steps value")
                    .parse()
                    .expect("climb-steps")
            }
            "--accept-epsilon" => {
                accept_epsilon = args
                    .next()
                    .expect("--accept-epsilon value")
                    .parse()
                    .expect("accept-epsilon")
            }
            "--export-top-endpoints" => {
                export_top_endpoints = args
                    .next()
                    .expect("--export-top-endpoints value")
                    .parse()
                    .expect("export-top-endpoints")
            }
            "--endpoint-eval-len" => {
                endpoint_eval_len = args
                    .next()
                    .expect("--endpoint-eval-len value")
                    .parse()
                    .expect("endpoint-eval-len")
            }
            "--endpoint-eval-seeds" => {
                endpoint_eval_seeds = args
                    .next()
                    .expect("--endpoint-eval-seeds csv")
                    .split(',')
                    .filter(|part| !part.trim().is_empty())
                    .map(|part| part.trim().parse::<u64>().expect("endpoint eval seed"))
                    .collect();
                assert!(
                    !endpoint_eval_seeds.is_empty(),
                    "--endpoint-eval-seeds must not be empty"
                );
            }
            "--bridge-endpoints" => {
                let value = args.next().expect("--bridge-endpoints csv");
                bridge_endpoints.extend(
                    value
                        .split(',')
                        .filter(|part| !part.trim().is_empty())
                        .map(|part| PathBuf::from(part.trim())),
                );
            }
            "--bridge-steps" => {
                bridge_steps = args
                    .next()
                    .expect("--bridge-steps value")
                    .parse()
                    .expect("bridge-steps")
            }
            "--overlap-samples" => {
                overlap_samples = args
                    .next()
                    .expect("--overlap-samples value")
                    .parse()
                    .expect("overlap-samples")
            }
            "--robustness-eval-lens" => {
                robustness_eval_lens =
                    parse_csv_usize(&args.next().expect("--robustness-eval-lens csv"));
                assert!(
                    !robustness_eval_lens.is_empty(),
                    "--robustness-eval-lens must not be empty"
                );
            }
            "--repair-start" => {
                repair_start = Some(PathBuf::from(args.next().expect("--repair-start path")));
            }
            "--repair-samples-per-bucket" => {
                repair_samples_per_bucket = args
                    .next()
                    .expect("--repair-samples-per-bucket value")
                    .parse()
                    .expect("repair-samples-per-bucket")
            }
            "--repair-eval-seeds" => {
                repair_eval_seeds = args
                    .next()
                    .expect("--repair-eval-seeds csv")
                    .split(',')
                    .filter(|part| !part.trim().is_empty())
                    .map(|part| part.trim().parse::<u64>().expect("repair eval seed"))
                    .collect();
                assert!(
                    !repair_eval_seeds.is_empty(),
                    "--repair-eval-seeds must not be empty"
                );
            }
            "--repair-export-top" => {
                repair_export_top = args
                    .next()
                    .expect("--repair-export-top value")
                    .parse()
                    .expect("repair-export-top")
            }
            "--mo-climbers" => {
                mo_climbers = args
                    .next()
                    .expect("--mo-climbers value")
                    .parse()
                    .expect("mo-climbers")
            }
            "--mo-steps" => {
                mo_steps = args
                    .next()
                    .expect("--mo-steps value")
                    .parse()
                    .expect("mo-steps")
            }
            "--mo-eval-seeds" => {
                mo_eval_seeds = args
                    .next()
                    .expect("--mo-eval-seeds csv")
                    .split(',')
                    .filter(|part| !part.trim().is_empty())
                    .map(|part| part.trim().parse::<u64>().expect("mo eval seed"))
                    .collect();
                assert!(
                    !mo_eval_seeds.is_empty(),
                    "--mo-eval-seeds must not be empty"
                );
            }
            "--mo-export-top" => {
                mo_export_top = args
                    .next()
                    .expect("--mo-export-top value")
                    .parse()
                    .expect("mo-export-top")
            }
            "--out" => out = PathBuf::from(args.next().expect("--out path")),
            "--seed" => seed = args.next().expect("--seed value").parse().expect("seed"),
            "--eval-len" => {
                eval_len = args
                    .next()
                    .expect("--eval-len value")
                    .parse()
                    .expect("eval-len")
            }
            "--packed" => packed = PathBuf::from(args.next().expect("--packed path")),
            "--corpus" => corpus = PathBuf::from(args.next().expect("--corpus path")),
            "--input-scatter" => input_scatter = true,
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run -p instnct-core --example d9_direct_landscape -- \
                     --checkpoint PATH --H 256 --mode fail-fast|medium|full|staged|planet-scout|paratrooper-climb|endpoint-bridge \
                     --score-mode accuracy|smooth --samples-per-type N --out PATH"
                );
                std::process::exit(0);
            }
            other => panic!("unknown arg: {other}"),
        }
    }

    if checkpoints.is_empty() {
        checkpoints.push(PathBuf::from(
            "output/phase_d7_operator_bandit_20260427/H_256/D7_BASELINE/seed_42/final.ckpt",
        ));
    }

    let default_radii = match mode.as_str() {
        "fail-fast" => vec![1],
        "medium" => vec![1, 4, 16],
        "full" | "staged" => vec![1, 2, 4, 8, 16, 32, 64, 128],
        "planet-scout"
        | "homogeneous"
        | "paratrooper-climb"
        | "endpoint-bridge"
        | "endpoint-overlap"
        | "endpoint-robustness"
        | "repair-scan"
        | "multi-objective-climb" => vec![1],
        other => panic!("--mode expects fail-fast|medium|full|staged|planet-scout|paratrooper-climb|endpoint-bridge|endpoint-overlap|endpoint-robustness|repair-scan|multi-objective-climb, got {other}"),
    };
    let default_mutation_types = if matches!(
        mode.as_str(),
        "planet-scout"
            | "homogeneous"
            | "paratrooper-climb"
            | "endpoint-bridge"
            | "endpoint-overlap"
            | "endpoint-robustness"
            | "repair-scan"
            | "multi-objective-climb"
    ) {
        vec![MutationType::Edge, MutationType::Threshold]
    } else {
        vec![
            MutationType::Edge,
            MutationType::Threshold,
            MutationType::Channel,
            MutationType::Polarity,
        ]
    };

    Cli {
        checkpoints,
        h,
        mode,
        score_mode,
        radii: radii.unwrap_or(default_radii),
        mutation_types: mutation_types.unwrap_or(default_mutation_types),
        samples_per_type,
        samples_per_tile,
        lat_bins,
        lon_bins,
        target_tiles,
        sample_layer,
        climbers_per_tile,
        climb_steps,
        accept_epsilon,
        export_top_endpoints,
        endpoint_eval_len,
        endpoint_eval_seeds,
        bridge_endpoints,
        bridge_steps,
        overlap_samples,
        robustness_eval_lens,
        repair_start,
        repair_samples_per_bucket,
        repair_eval_seeds,
        repair_export_top,
        mo_climbers,
        mo_steps,
        mo_eval_seeds,
        mo_export_top,
        out,
        seed,
        eval_len,
        packed,
        corpus,
        input_scatter,
    }
}

fn edge_coord_index(h: usize, source: usize, target: usize) -> usize {
    debug_assert_ne!(source, target);
    source * (h - 1) + if target < source { target } else { target - 1 }
}

fn edge_from_coord_index(h: usize, index: usize) -> (usize, usize) {
    let source = index / (h - 1);
    let target_compact = index % (h - 1);
    let target = if target_compact < source {
        target_compact
    } else {
        target_compact + 1
    };
    (source, target)
}

fn encode_coord(net: &Network) -> DirectGenomeCoord {
    let h = net.neuron_count();
    let mut edge_bits = vec![0u8; h * (h - 1)];
    for edge in net.graph().iter_edges() {
        let idx = edge_coord_index(h, edge.source as usize, edge.target as usize);
        edge_bits[idx] = 1;
    }
    DirectGenomeCoord {
        h,
        edge_bits,
        threshold: net.spike_data().iter().map(|s| s.threshold).collect(),
        channel: net.spike_data().iter().map(|s| s.channel).collect(),
        polarity: net.polarity().to_vec(),
    }
}

fn decode_coord(coord: &DirectGenomeCoord) -> Network {
    let mut net = Network::new(coord.h);
    for (idx, &bit) in coord.edge_bits.iter().enumerate() {
        if bit != 0 {
            let (source, target) = edge_from_coord_index(coord.h, idx);
            assert!(net.graph_mut().add_edge(source as u16, target as u16));
        }
    }
    for (idx, spike) in net.spike_data_mut().iter_mut().enumerate() {
        spike.threshold = coord.threshold[idx];
        spike.channel = coord.channel[idx];
    }
    for (idx, pol) in net.polarity_mut().iter_mut().enumerate() {
        *pol = coord.polarity[idx];
    }
    net
}

fn coord_distance(a: &DirectGenomeCoord, b: &DirectGenomeCoord) -> usize {
    assert_eq!(a.h, b.h);
    let edge = a
        .edge_bits
        .iter()
        .zip(&b.edge_bits)
        .filter(|(x, y)| x != y)
        .count();
    let threshold = a
        .threshold
        .iter()
        .zip(&b.threshold)
        .map(|(&x, &y)| x.abs_diff(y) as usize)
        .sum::<usize>();
    let channel = a
        .channel
        .iter()
        .zip(&b.channel)
        .map(|(&x, &y)| x.abs_diff(y) as usize)
        .sum::<usize>();
    let polarity = a
        .polarity
        .iter()
        .zip(&b.polarity)
        .filter(|(x, y)| x != y)
        .count();
    edge + threshold + channel + polarity
}

fn compare_core(a: &Network, b: &Network) -> bool {
    encode_coord(a) == encode_coord(b)
}

fn interpolate_coord(
    a: &DirectGenomeCoord,
    b: &DirectGenomeCoord,
    step: usize,
    steps: usize,
) -> DirectGenomeCoord {
    assert_eq!(a.h, b.h);
    let steps = steps.max(1);
    if step == 0 {
        return a.clone();
    }
    if step >= steps {
        return b.clone();
    }
    let mut edge_bits = a.edge_bits.clone();
    let edge_diffs: Vec<usize> = a
        .edge_bits
        .iter()
        .zip(&b.edge_bits)
        .enumerate()
        .filter_map(|(idx, (&x, &y))| (x != y).then_some(idx))
        .collect();
    let edge_take = (edge_diffs.len() * step + steps / 2) / steps;
    for &idx in edge_diffs.iter().take(edge_take) {
        edge_bits[idx] = b.edge_bits[idx];
    }

    let mut threshold = a.threshold.clone();
    for (idx, value) in threshold.iter_mut().enumerate() {
        let av = a.threshold[idx] as i32;
        let bv = b.threshold[idx] as i32;
        *value =
            (av + ((bv - av) * step as i32 + (steps / 2) as i32) / steps as i32).clamp(0, 15) as u8;
    }

    let mut channel = a.channel.clone();
    for (idx, value) in channel.iter_mut().enumerate() {
        let av = a.channel[idx] as i32;
        let bv = b.channel[idx] as i32;
        *value =
            (av + ((bv - av) * step as i32 + (steps / 2) as i32) / steps as i32).clamp(1, 8) as u8;
    }

    let mut polarity = a.polarity.clone();
    let polarity_diffs: Vec<usize> = a
        .polarity
        .iter()
        .zip(&b.polarity)
        .enumerate()
        .filter_map(|(idx, (&x, &y))| (x != y).then_some(idx))
        .collect();
    let polarity_take = (polarity_diffs.len() * step + steps / 2) / steps;
    for &idx in polarity_diffs.iter().take(polarity_take) {
        polarity[idx] = b.polarity[idx];
    }

    DirectGenomeCoord {
        h: a.h,
        edge_bits,
        threshold,
        channel,
        polarity,
    }
}

fn apply_one_edge_flip(net: &mut Network, rng: &mut StdRng) -> bool {
    let h = net.neuron_count();
    if h < 2 {
        return false;
    }
    let idx = rng.gen_range(0..h * (h - 1));
    let (source, target) = edge_from_coord_index(h, idx);
    if net.graph().has_edge(source as u16, target as u16) {
        net.graph_mut().remove_edge(source as u16, target as u16)
    } else {
        net.graph_mut().add_edge(source as u16, target as u16)
    }
}

fn apply_one_threshold_step(net: &mut Network, rng: &mut StdRng) -> bool {
    let h = net.neuron_count();
    if h == 0 {
        return false;
    }
    let idx = rng.gen_range(0..h);
    let value = net.spike_data()[idx].threshold;
    let next = if value == 0 {
        1
    } else if value == 15 {
        14
    } else if rng.gen_bool(0.5) {
        value + 1
    } else {
        value - 1
    };
    net.spike_data_mut()[idx].threshold = next;
    true
}

fn apply_one_channel_step(net: &mut Network, rng: &mut StdRng) -> bool {
    let h = net.neuron_count();
    if h == 0 {
        return false;
    }
    let idx = rng.gen_range(0..h);
    let value = net.spike_data()[idx].channel;
    let next = if value <= 1 {
        2
    } else if value >= 8 {
        7
    } else if rng.gen_bool(0.5) {
        value + 1
    } else {
        value - 1
    };
    net.spike_data_mut()[idx].channel = next;
    true
}

fn apply_one_polarity_flip(net: &mut Network, rng: &mut StdRng) -> bool {
    let h = net.neuron_count();
    if h == 0 {
        return false;
    }
    let idx = rng.gen_range(0..h);
    net.polarity_mut()[idx] = -net.polarity()[idx];
    true
}

fn apply_radius_mutation(
    net: &mut Network,
    radius: usize,
    mutation_type: MutationType,
    rng: &mut StdRng,
) -> MutationCounts {
    let mut counts = MutationCounts::default();
    for _ in 0..radius {
        let selected = if mutation_type == MutationType::Mixed {
            match rng.gen_range(0..4) {
                0 => MutationType::Edge,
                1 => MutationType::Threshold,
                2 => MutationType::Channel,
                _ => MutationType::Polarity,
            }
        } else {
            mutation_type
        };
        let ok = match selected {
            MutationType::Edge => {
                let ok = apply_one_edge_flip(net, rng);
                if ok {
                    counts.edge += 1;
                }
                ok
            }
            MutationType::Threshold => {
                let ok = apply_one_threshold_step(net, rng);
                if ok {
                    counts.threshold += 1;
                }
                ok
            }
            MutationType::Channel => {
                let ok = apply_one_channel_step(net, rng);
                if ok {
                    counts.channel += 1;
                }
                ok
            }
            MutationType::Polarity => {
                let ok = apply_one_polarity_flip(net, rng);
                if ok {
                    counts.polarity += 1;
                }
                ok
            }
            MutationType::Projection => {
                panic!("projection mutation is supported only by paratrooper-climb")
            }
            MutationType::Mixed => unreachable!(),
        };
        assert!(ok, "direct mutation should be valid for H>=2");
    }
    counts
}

fn apply_radius_projection_mutation(
    proj: &mut Int8Projection,
    radius: usize,
    rng: &mut StdRng,
) -> MutationCounts {
    let mut counts = MutationCounts::default();
    for _ in 0..radius {
        let _ = proj.mutate_one(rng);
        counts.projection += 1;
    }
    counts
}

fn quantize_embedding_to_input(
    table: &VcbpTable,
    embedding: &[f32],
    input: &mut [i32],
    input_end: usize,
    input_scatter: bool,
) {
    if !input_scatter {
        table.quantize_to_input(embedding, &mut input[..table.e], MAX_CHARGE);
        return;
    }
    let mut base = vec![0i32; table.e];
    table.quantize_to_input(embedding, &mut base, MAX_CHARGE);
    for dst in input.iter_mut().take(input_end) {
        *dst = 0;
    }
    for idx in 0..input_end.min(input.len()) {
        input[idx] = base[idx % table.e];
    }
}

fn build_corpus_pairs(
    corpus: &[u8],
    table: &VcbpTable,
    max_classes: usize,
) -> (Vec<u16>, Vec<usize>, usize) {
    let n_pairs = corpus.len() / 2;
    let mut pair_ids = Vec::with_capacity(n_pairs);
    let mut freq = vec![0u32; 65536];
    for i in 0..n_pairs {
        let pid = VcbpTable::pair_id(corpus[i * 2], corpus[i * 2 + 1]);
        pair_ids.push(pid);
        freq[pid as usize] += 1;
    }
    let mut hot_freq: Vec<(u16, u32)> = (0..65536u32)
        .filter(|&v| table.is_hot(v as u16) && freq[v as usize] > 0)
        .map(|v| (v as u16, freq[v as usize]))
        .collect();
    hot_freq.sort_by(|a, b| b.1.cmp(&a.1));
    let n_classes = hot_freq.len().min(max_classes);
    hot_freq.truncate(n_classes);
    let mut top_to_idx = vec![usize::MAX; 65536];
    for (i, &(id, _)) in hot_freq.iter().enumerate() {
        top_to_idx[id as usize] = i;
    }
    (pair_ids, top_to_idx, n_classes)
}

fn build_pair_bigram(pair_ids: &[u16], hot_to_idx: &[usize], n_hot: usize) -> Vec<Vec<f64>> {
    let mut counts = vec![vec![0u32; n_hot]; n_hot];
    for i in 0..pair_ids.len().saturating_sub(1) {
        let cur_idx = hot_to_idx[pair_ids[i] as usize];
        let next_idx = hot_to_idx[pair_ids[i + 1] as usize];
        if cur_idx != usize::MAX && next_idx != usize::MAX {
            counts[cur_idx][next_idx] += 1;
        }
    }
    counts
        .iter()
        .map(|row| {
            let total: f64 = row.iter().map(|&c| c as f64).sum();
            if total < 1.0 {
                vec![1.0 / n_hot as f64; n_hot]
            } else {
                row.iter().map(|&c| c as f64 / total).collect()
            }
        })
        .collect()
}

fn build_pair_unigram(pair_ids: &[u16], hot_to_idx: &[usize], n_hot: usize) -> Vec<f64> {
    let mut counts = vec![0u32; n_hot];
    for &pid in pair_ids {
        let idx = hot_to_idx[pid as usize];
        if idx != usize::MAX {
            counts[idx] += 1;
        }
    }
    let total: f64 = counts.iter().map(|&c| c as f64).sum();
    if total < 1.0 {
        vec![1.0 / n_hot as f64; n_hot]
    } else {
        counts.iter().map(|&c| c as f64 / total).collect()
    }
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 1e-12 || nb <= 1e-12 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn eval_accuracy_proj(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    len: usize,
    rng: &mut StdRng,
    propagation: &instnct_core::PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    input_end: usize,
    input_scatter: bool,
) -> f64 {
    let n = pair_ids.len();
    if n <= len + 1 {
        return 0.0;
    }
    let off = rng.gen_range(1..=n - len - 1);
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let tgt_id = pair_ids[off + i + 1];
        let tgt_idx = hot_to_idx[tgt_id as usize];
        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        quantize_embedding_to_input(table, emb, &mut input, input_end, input_scatter);
        net.propagate(&input, propagation).unwrap();
        let predicted_idx = proj.predict(&net.charge_vec(output_start..neuron_count));
        if tgt_idx != usize::MAX && predicted_idx == tgt_idx {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

fn one_hot_cosine(probs: &[f64], target_idx: usize) -> f64 {
    let norm = probs.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= 1e-12 {
        0.0
    } else {
        probs[target_idx] / norm
    }
}

fn eval_echo_proj(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    len: usize,
    rng: &mut StdRng,
    propagation: &instnct_core::PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    input_end: usize,
    input_scatter: bool,
) -> f64 {
    let n = pair_ids.len();
    if n <= len + 1 {
        return 0.0;
    }
    let off = rng.gen_range(1..=n - len - 1);
    net.reset();
    let mut total_cos = 0.0f64;
    let mut counted = 0usize;
    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let cur_idx = hot_to_idx[cur_id as usize];
        if cur_idx == usize::MAX {
            continue;
        }
        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        quantize_embedding_to_input(table, emb, &mut input, input_end, input_scatter);
        net.propagate(&input, propagation).unwrap();
        let scores = proj.raw_scores(&net.charge_vec(output_start..neuron_count));
        let probs = softmax(&scores);
        total_cos += one_hot_cosine(&probs, cur_idx);
        counted += 1;
    }
    if counted == 0 {
        0.0
    } else {
        total_cos / counted as f64
    }
}

fn eval_smooth_proj(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    bigram: &[Vec<f64>],
    len: usize,
    rng: &mut StdRng,
    propagation: &instnct_core::PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    input_end: usize,
    input_scatter: bool,
) -> f64 {
    let n = pair_ids.len();
    if n <= len + 1 {
        return 0.0;
    }
    let off = rng.gen_range(1..=n - len - 1);
    net.reset();
    let mut total_cos = 0.0f64;
    let mut counted = 0usize;
    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let cur_idx = hot_to_idx[cur_id as usize];
        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        quantize_embedding_to_input(table, emb, &mut input, input_end, input_scatter);
        net.propagate(&input, propagation).unwrap();
        if cur_idx == usize::MAX {
            continue;
        }
        let scores = proj.raw_scores(&net.charge_vec(output_start..neuron_count));
        let probs = softmax(&scores);
        total_cos += cosine_similarity(&probs, &bigram[cur_idx]);
        counted += 1;
    }
    if counted == 0 {
        0.0
    } else {
        total_cos / counted as f64
    }
}

fn eval_unigram_proj(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    unigram: &[f64],
    len: usize,
    rng: &mut StdRng,
    propagation: &instnct_core::PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    input_end: usize,
    input_scatter: bool,
) -> f64 {
    let n = pair_ids.len();
    if n <= len + 1 {
        return 0.0;
    }
    let off = rng.gen_range(1..=n - len - 1);
    net.reset();
    let mut total_cos = 0.0f64;
    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        quantize_embedding_to_input(table, emb, &mut input, input_end, input_scatter);
        net.propagate(&input, propagation).unwrap();
        let scores = proj.raw_scores(&net.charge_vec(output_start..neuron_count));
        let probs = softmax(&scores);
        total_cos += cosine_similarity(&probs, unigram);
    }
    total_cos / len as f64
}

fn eval_score(
    score_mode: &str,
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    bigram: &[Vec<f64>],
    unigram: &[f64],
    len: usize,
    rng: &mut StdRng,
    propagation: &instnct_core::PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    input_end: usize,
    input_scatter: bool,
) -> f64 {
    match score_mode {
        "accuracy" => eval_accuracy_proj(
            net,
            proj,
            table,
            pair_ids,
            hot_to_idx,
            len,
            rng,
            propagation,
            output_start,
            neuron_count,
            input_end,
            input_scatter,
        ),
        "echo" => eval_echo_proj(
            net,
            proj,
            table,
            pair_ids,
            hot_to_idx,
            len,
            rng,
            propagation,
            output_start,
            neuron_count,
            input_end,
            input_scatter,
        ),
        "smooth" => eval_smooth_proj(
            net,
            proj,
            table,
            pair_ids,
            hot_to_idx,
            bigram,
            len,
            rng,
            propagation,
            output_start,
            neuron_count,
            input_end,
            input_scatter,
        ),
        "unigram" => eval_unigram_proj(
            net,
            proj,
            table,
            pair_ids,
            unigram,
            len,
            rng,
            propagation,
            output_start,
            neuron_count,
            input_end,
            input_scatter,
        ),
        other => panic!("unknown score_mode: {other}"),
    }
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    squared_distance(a, b).sqrt()
}

fn f_active(matrix: &[Vec<f64>]) -> f64 {
    if matrix.is_empty() || matrix[0].is_empty() {
        return 0.0;
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let active = (0..cols)
        .filter(|&col| {
            let first = matrix[0][col];
            (1..rows).any(|row| matrix[row][col] != first)
        })
        .count();
    active as f64 / cols as f64
}

fn output_entropy_stats(entropies: &[f64]) -> (f64, f64) {
    if entropies.is_empty() {
        return (0.0, 0.0);
    }
    let mean = entropies.iter().sum::<f64>() / entropies.len() as f64;
    let var = entropies
        .iter()
        .map(|value| {
            let d = value - mean;
            d * d
        })
        .sum::<f64>()
        / entropies.len() as f64;
    (mean, var)
}

fn stable_rank(matrix: &[Vec<f64>]) -> f64 {
    if matrix.is_empty() || matrix[0].is_empty() {
        return 0.0;
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let fro2 = matrix.iter().flatten().map(|v| v * v).sum::<f64>();
    if fro2 <= 0.0 {
        return 0.0;
    }
    let mut v = vec![1.0 / (cols as f64).sqrt(); cols];
    for _ in 0..50 {
        let mut yv = vec![0.0; rows];
        for (row_idx, row) in matrix.iter().enumerate() {
            yv[row_idx] = row.iter().zip(&v).map(|(a, b)| a * b).sum();
        }
        let mut z = vec![0.0; cols];
        for (row, y) in matrix.iter().zip(&yv) {
            for col in 0..cols {
                z[col] += row[col] * y;
            }
        }
        let norm = z.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= 1e-12 {
            return 0.0;
        }
        for col in 0..cols {
            v[col] = z[col] / norm;
        }
    }
    let lambda = matrix
        .iter()
        .map(|row| row.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>())
        .map(|y| y * y)
        .sum::<f64>();
    if lambda <= 1e-12 {
        0.0
    } else {
        fro2 / lambda
    }
}

fn numerical_rank(matrix: &[Vec<f64>], tol: f64) -> usize {
    if matrix.is_empty() || matrix[0].is_empty() {
        return 0;
    }
    let mut mat = matrix.to_vec();
    let rows = mat.len();
    let cols = mat[0].len();
    let mut rank = 0usize;
    for col in 0..cols {
        let mut pivot = rank;
        for row in rank..rows {
            if mat[row][col].abs() > mat[pivot][col].abs() {
                pivot = row;
            }
        }
        if mat[pivot][col].abs() <= tol {
            continue;
        }
        mat.swap(rank, pivot);
        let pivot_val = mat[rank][col];
        for c in col..cols {
            mat[rank][c] /= pivot_val;
        }
        for row in 0..rows {
            if row == rank {
                continue;
            }
            let factor = mat[row][col];
            if factor.abs() <= tol {
                continue;
            }
            for c in col..cols {
                mat[row][c] -= factor * mat[rank][c];
            }
        }
        rank += 1;
        if rank == rows {
            break;
        }
    }
    rank
}

fn separation_sp(inputs: &[Vec<f64>], outputs: &[Vec<f64>]) -> f64 {
    let n = inputs.len().min(outputs.len());
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let in_dist = euclidean_distance(&inputs[i], &inputs[j]);
            let out_dist = euclidean_distance(&outputs[i], &outputs[j]);
            total += out_dist / (in_dist + 1e-12);
            count += 1;
        }
    }
    total / count as f64
}

fn entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

fn compute_panel_metrics(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    propagation: &instnct_core::PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    input_end: usize,
    input_scatter: bool,
) -> PanelMetrics {
    let snapshot = net.save_state();
    let usable_pairs = pair_ids.len().saturating_sub(1);
    let probe_count = PANEL_PROBE_COUNT.min(usable_pairs.max(1));
    let mut input_matrix = Vec::with_capacity(probe_count);
    let mut output_matrix = Vec::with_capacity(probe_count);
    let mut predictions = Vec::with_capacity(probe_count);
    let mut entropy_values = Vec::with_capacity(probe_count);
    let mut unique_inputs = HashSet::new();
    let mut correct = 0usize;

    if usable_pairs == 0 {
        net.restore_state(&snapshot);
        return PanelMetrics {
            panel_probe_acc: 0.0,
            unique_predictions: 0,
            collision_rate: 0.0,
            f_active: 0.0,
            h_output_mean: 0.0,
            h_output_var: 0.0,
            stable_rank: 0.0,
            kernel_rank: 0,
            separation_sp: 0.0,
        };
    }

    for k in 0..probe_count {
        let idx = (k * usable_pairs) / probe_count;
        let cur_id = pair_ids[idx];
        let tgt_id = pair_ids[idx + 1];
        let tgt_idx = hot_to_idx[tgt_id as usize];
        unique_inputs.insert(cur_id);
        net.reset();
        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        quantize_embedding_to_input(table, emb, &mut input, input_end, input_scatter);
        net.propagate(&input, propagation).unwrap();
        let charges_u8 = net.charge_vec(output_start..neuron_count);
        let charges: Vec<f64> = charges_u8.iter().map(|&v| v as f64).collect();
        let scores = proj.raw_scores(&charges_u8);
        let probs = softmax(&scores);
        let pred = proj.predict(&charges_u8);
        if tgt_idx != usize::MAX && pred == tgt_idx {
            correct += 1;
        }
        entropy_values.push(entropy(&probs));
        predictions.push(pred);
        input_matrix.push(emb.iter().map(|&v| v as f64).collect::<Vec<_>>());
        output_matrix.push(charges);
    }

    net.restore_state(&snapshot);
    let unique_predictions = predictions.iter().copied().collect::<HashSet<_>>().len();
    let collision_rate = if unique_inputs.is_empty() {
        0.0
    } else {
        unique_predictions as f64 / unique_inputs.len() as f64
    };
    let (h_output_mean, h_output_var) = output_entropy_stats(&entropy_values);
    PanelMetrics {
        panel_probe_acc: correct as f64 / probe_count as f64,
        unique_predictions,
        collision_rate,
        f_active: f_active(&output_matrix),
        h_output_mean,
        h_output_var,
        stable_rank: stable_rank(&output_matrix),
        kernel_rank: numerical_rank(&output_matrix, 1e-6),
        separation_sp: separation_sp(&input_matrix, &output_matrix),
    }
}

fn behavior_distance(base: &PanelMetrics, sample: &PanelMetrics, h: usize) -> f64 {
    let base_sep_scale = base.separation_sp.abs().max(1.0);
    let values = [
        sample.panel_probe_acc - base.panel_probe_acc,
        (sample.unique_predictions as f64 - base.unique_predictions as f64) / 397.0,
        sample.collision_rate - base.collision_rate,
        sample.f_active - base.f_active,
        (sample.stable_rank - base.stable_rank) / base.stable_rank.abs().max(1.0),
        (sample.kernel_rank as f64 - base.kernel_rank as f64) / h.max(1) as f64,
        (sample.separation_sp - base.separation_sp) / base_sep_scale,
    ];
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn write_header(writer: &mut BufWriter<File>) {
    writeln!(
        writer,
        "base_index,base_checkpoint,base_step,base_accuracy,sample_id,seed,H,mode,mutation_type,requested_radius,direct_genome_distance,edge_edits,threshold_edits,channel_edits,polarity_edits,base_score,score,delta_score,behavior_distance,edges,panel_probe_acc,unique_predictions,collision_rate,f_active,h_output_mean,h_output_var,stable_rank,kernel_rank,separation_sp,eval_ms,score_mode,tile_id,lat_bin,lon_bin,scout_layer"
    )
    .expect("failed to write samples header");
}

fn write_climb_header(writer: &mut BufWriter<File>) {
    writeln!(
        writer,
        "base_index,base_checkpoint,base_step,base_accuracy,climber_id,tile_id,lat_bin,lon_bin,step_index,proposal_seed,mutation_type,accepted,accept_reason,current_score_before,candidate_score,accepted_score,best_score,delta_from_start,best_delta_from_start,step_delta,base_score,requested_radius,direct_genome_distance,edge_edits,threshold_edits,channel_edits,polarity_edits,projection_edits,current_edges,candidate_edges,behavior_distance,eval_ms,score_mode,accept_epsilon"
    )
    .expect("failed to write paratrooper_paths header");
}

fn mean_endpoint_score(
    cli: &Cli,
    net: &Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    bigram: &[Vec<f64>],
    unigram: &[f64],
    init: &InitConfig,
) -> f64 {
    let mut total = 0.0;
    for &seed in &cli.endpoint_eval_seeds {
        let mut eval_net = net.clone();
        total += eval_score(
            &cli.score_mode,
            &mut eval_net,
            proj,
            table,
            pair_ids,
            hot_to_idx,
            bigram,
            unigram,
            cli.endpoint_eval_len,
            &mut StdRng::seed_from_u64(seed),
            &init.propagation,
            init.output_start(),
            cli.h,
            init.input_end(),
            cli.input_scatter,
        );
    }
    total / cli.endpoint_eval_seeds.len() as f64
}

fn export_endpoint_candidates(
    cli: &Cli,
    endpoint_candidates: &[EndpointCandidate],
    base_net: &Network,
    base_proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    bigram: &[Vec<f64>],
    unigram: &[f64],
    init: &InitConfig,
) {
    if endpoint_candidates.is_empty() {
        return;
    }
    let endpoints_dir = cli.out.join("endpoints");
    create_dir_all(&endpoints_dir).expect("failed to create endpoints directory");
    let baseline_reeval_mean = mean_endpoint_score(
        cli, base_net, base_proj, table, pair_ids, hot_to_idx, bigram, unigram, init,
    );
    let summary_path = endpoints_dir.join("endpoint_reeval_summary.csv");
    let mut writer =
        BufWriter::new(File::create(&summary_path).expect("failed to create endpoint summary"));
    writeln!(
        writer,
        "endpoint_rank,endpoint_checkpoint,sidecar_json,base_index,base_checkpoint,tile_id,climber_id,best_step,climb_best_delta,final_delta,mutation_type_at_best,accepted_seed_count,baseline_reeval_mean,endpoint_reeval_mean,reeval_delta_4000,retention_pct,delta_vs_baseline,pass_retention_70,pass_positive_delta,endpoint_eval_len,endpoint_eval_seeds"
    )
    .expect("failed to write endpoint summary header");

    for (rank_idx, endpoint) in endpoint_candidates.iter().enumerate() {
        let endpoint_rank = rank_idx + 1;
        let endpoint_reeval_mean = mean_endpoint_score(
            cli,
            &endpoint.net,
            &endpoint.proj,
            table,
            pair_ids,
            hot_to_idx,
            bigram,
            unigram,
            init,
        );
        let climb_best_delta = endpoint.best_score - endpoint.start_score;
        let final_delta = endpoint.final_score - endpoint.start_score;
        let reeval_delta = endpoint_reeval_mean - baseline_reeval_mean;
        let retention_pct = if climb_best_delta.abs() > 1e-12 {
            reeval_delta / climb_best_delta
        } else {
            0.0
        };
        let pass_retention_70 = retention_pct >= 0.70;
        let pass_positive_delta = reeval_delta > 0.0;
        let ckpt_path = endpoints_dir.join(format!("endpoint_{endpoint_rank:02}.ckpt"));
        let sidecar_path = endpoints_dir.join(format!("endpoint_{endpoint_rank:02}.json"));
        save_checkpoint(
            &ckpt_path,
            &endpoint.net,
            &endpoint.proj,
            CheckpointMeta {
                step: endpoint.base_step + endpoint.best_step,
                accuracy: endpoint_reeval_mean,
                label: format!(
                    "D9.0r endpoint rank={} tile={} delta={:.6}",
                    endpoint_rank, endpoint.tile_id, reeval_delta
                ),
            },
        )
        .expect("failed to save endpoint checkpoint");

        let ckpt_str = ckpt_path.display().to_string();
        let sidecar = EndpointSidecar {
            endpoint_rank,
            endpoint_checkpoint: &ckpt_str,
            base_index: endpoint.base_index,
            base_checkpoint: &endpoint.base_checkpoint,
            base_step: endpoint.base_step,
            base_accuracy: endpoint.base_accuracy,
            tile_id: &endpoint.tile_id,
            lat_bin: endpoint.lat_bin,
            lon_bin: endpoint.lon_bin,
            climber_id: endpoint.climber_id,
            climber_seed: endpoint.climber_seed,
            eval_seed: endpoint.eval_seed,
            best_step: endpoint.best_step,
            best_score: endpoint.best_score,
            start_score: endpoint.start_score,
            climb_best_delta,
            final_delta,
            mutation_type_at_best: &endpoint.mutation_type_at_best,
            accepted_seed_chain: &endpoint.accepted_seed_chain,
            endpoint_eval_len: cli.endpoint_eval_len,
            endpoint_eval_seeds: &cli.endpoint_eval_seeds,
            baseline_reeval_mean,
            endpoint_reeval_mean,
            reeval_delta_4000: reeval_delta,
            retention_pct,
            delta_vs_baseline: reeval_delta,
            pass_retention_70,
            pass_positive_delta,
        };
        std::fs::write(
            &sidecar_path,
            serde_json::to_string_pretty(&sidecar).expect("endpoint sidecar serialize"),
        )
        .expect("failed to write endpoint sidecar");
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{:.17},{:.17},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{},{},{},{}",
            endpoint_rank,
            ckpt_path.display(),
            sidecar_path.display(),
            endpoint.base_index,
            endpoint.base_checkpoint,
            endpoint.tile_id,
            endpoint.climber_id,
            endpoint.best_step,
            climb_best_delta,
            final_delta,
            endpoint.mutation_type_at_best,
            endpoint.accepted_seed_chain.len(),
            baseline_reeval_mean,
            endpoint_reeval_mean,
            reeval_delta,
            retention_pct,
            reeval_delta,
            pass_retention_70,
            pass_positive_delta,
            cli.endpoint_eval_len,
            cli.endpoint_eval_seeds
                .iter()
                .map(|seed| seed.to_string())
                .collect::<Vec<_>>()
                .join("|"),
        )
        .expect("failed to write endpoint summary row");
    }
    writer.flush().expect("failed to flush endpoint summary");
}

fn run_endpoint_bridge(
    cli: &Cli,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    bigram: &[Vec<f64>],
    unigram: &[f64],
    init: &InitConfig,
) {
    assert!(
        cli.bridge_endpoints.len() >= 2,
        "--mode endpoint-bridge requires at least two --bridge-endpoints"
    );

    let baseline_path = cli
        .checkpoints
        .first()
        .expect("baseline checkpoint required");
    let (baseline_net, baseline_proj, _) =
        load_checkpoint(baseline_path).expect("failed to load baseline checkpoint");
    let baseline_score = mean_endpoint_score(
        cli,
        &baseline_net,
        &baseline_proj,
        table,
        pair_ids,
        hot_to_idx,
        bigram,
        unigram,
        init,
    );

    let mut endpoints = Vec::new();
    for path in &cli.bridge_endpoints {
        let (net, proj, _meta) = load_checkpoint(path).expect("failed to load bridge endpoint");
        assert_eq!(net.neuron_count(), cli.h, "bridge endpoint H mismatch");
        endpoints.push((path.display().to_string(), net, proj));
    }

    let path = cli.out.join("bridge_samples.csv");
    let mut writer = BufWriter::new(File::create(&path).expect("failed to create bridge csv"));
    writeln!(
        writer,
        "pair_id,endpoint_a,endpoint_b,step,steps,fraction,score,delta_vs_baseline,baseline_score,coord_distance_from_a,coord_distance_to_b,pass_positive"
    )
    .expect("failed to write bridge header");

    for i in 0..endpoints.len() {
        for j in (i + 1)..endpoints.len() {
            let (name_a, net_a, proj_a) = &endpoints[i];
            let (name_b, net_b, _proj_b) = &endpoints[j];
            let coord_a = encode_coord(net_a);
            let coord_b = encode_coord(net_b);
            for step in 0..=cli.bridge_steps {
                let coord = interpolate_coord(&coord_a, &coord_b, step, cli.bridge_steps);
                let waypoint = decode_coord(&coord);
                let score = mean_endpoint_score(
                    cli, &waypoint, proj_a, table, pair_ids, hot_to_idx, bigram, unigram, init,
                );
                let dist_a = coord_distance(&coord_a, &coord);
                let dist_b = coord_distance(&coord, &coord_b);
                let delta = score - baseline_score;
                writeln!(
                    writer,
                    "{},\"{}\",\"{}\",{},{},{:.6},{:.12},{:.12},{:.12},{},{},{}",
                    format!("{}__{}", i + 1, j + 1),
                    name_a,
                    name_b,
                    step,
                    cli.bridge_steps,
                    step as f64 / cli.bridge_steps.max(1) as f64,
                    score,
                    delta,
                    baseline_score,
                    dist_a,
                    dist_b,
                    delta > 0.0
                )
                .expect("failed to write bridge row");
            }
        }
    }
    writer.flush().expect("failed to flush bridge csv");
}

fn run_endpoint_overlap(
    cli: &Cli,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    bigram: &[Vec<f64>],
    unigram: &[f64],
    init: &InitConfig,
) {
    assert!(
        !cli.bridge_endpoints.is_empty(),
        "--mode endpoint-overlap requires --bridge-endpoints"
    );
    let baseline_path = cli
        .checkpoints
        .first()
        .expect("baseline checkpoint required");
    let (baseline_net, baseline_proj, _) =
        load_checkpoint(baseline_path).expect("failed to load baseline checkpoint");
    let baseline_score = mean_endpoint_score(
        cli,
        &baseline_net,
        &baseline_proj,
        table,
        pair_ids,
        hot_to_idx,
        bigram,
        unigram,
        init,
    );

    let mut endpoints = Vec::new();
    for path in &cli.bridge_endpoints {
        let (net, proj, _meta) = load_checkpoint(path).expect("failed to load overlap endpoint");
        endpoints.push((path.display().to_string(), net, proj));
    }
    let endpoint_coords: Vec<_> = endpoints
        .iter()
        .map(|(_, net, _)| encode_coord(net))
        .collect();

    let path = cli.out.join("overlap_samples.csv");
    let mut writer = BufWriter::new(File::create(&path).expect("failed to create overlap csv"));
    writeln!(
        writer,
        "endpoint_index,endpoint_path,radius,sample_id,mutation_type,score,delta_vs_baseline,baseline_score,dist_to_source,nearest_other_endpoint,dist_to_nearest_other,positive"
    )
    .expect("failed to write overlap header");

    for (endpoint_idx, (endpoint_path, endpoint_net, endpoint_proj)) in endpoints.iter().enumerate()
    {
        let source_coord = &endpoint_coords[endpoint_idx];
        for &radius in &cli.radii {
            for sample_id in 0..cli.overlap_samples {
                let mutation_type = cli.mutation_types[sample_id % cli.mutation_types.len()];
                let seed = cli.seed
                    ^ ((endpoint_idx as u64) << 48)
                    ^ ((radius as u64) << 32)
                    ^ ((sample_id as u64) << 8)
                    ^ 0xD90D_0AAAu64;
                let mut sample_net = endpoint_net.clone();
                let mut rng = StdRng::seed_from_u64(seed);
                let _counts =
                    apply_radius_mutation(&mut sample_net, radius, mutation_type, &mut rng);
                let sample_coord = encode_coord(&sample_net);
                let dist_to_source = coord_distance(source_coord, &sample_coord);
                let mut nearest_other_endpoint = String::from("none");
                let mut dist_to_nearest_other = usize::MAX;
                for (other_idx, other_coord) in endpoint_coords.iter().enumerate() {
                    if other_idx == endpoint_idx {
                        continue;
                    }
                    let dist = coord_distance(&sample_coord, other_coord);
                    if dist < dist_to_nearest_other {
                        dist_to_nearest_other = dist;
                        nearest_other_endpoint = (other_idx + 1).to_string();
                    }
                }
                let score = mean_endpoint_score(
                    cli,
                    &sample_net,
                    endpoint_proj,
                    table,
                    pair_ids,
                    hot_to_idx,
                    bigram,
                    unigram,
                    init,
                );
                let delta = score - baseline_score;
                writeln!(
                    writer,
                    "{},\"{}\",{},{},{},{:.12},{:.12},{:.12},{},{},{},{}",
                    endpoint_idx + 1,
                    endpoint_path,
                    radius,
                    sample_id,
                    mutation_type.as_str(),
                    score,
                    delta,
                    baseline_score,
                    dist_to_source,
                    nearest_other_endpoint,
                    dist_to_nearest_other,
                    delta > 0.0
                )
                .expect("failed to write overlap row");
            }
        }
    }
    writer.flush().expect("failed to flush overlap csv");
}

fn run_endpoint_robustness(
    cli: &Cli,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    bigram: &[Vec<f64>],
    unigram: &[f64],
    init: &InitConfig,
) {
    assert!(
        !cli.bridge_endpoints.is_empty(),
        "--mode endpoint-robustness requires --bridge-endpoints"
    );
    let baseline_path = cli
        .checkpoints
        .first()
        .expect("baseline checkpoint required");
    let (baseline_net, baseline_proj, _) =
        load_checkpoint(baseline_path).expect("failed to load baseline checkpoint");

    let mut endpoints = Vec::new();
    for path in &cli.bridge_endpoints {
        let (net, proj, _meta) = load_checkpoint(path).expect("failed to load robustness endpoint");
        assert_eq!(net.neuron_count(), cli.h, "robustness endpoint H mismatch");
        endpoints.push((path.display().to_string(), net, proj));
    }

    let path = cli.out.join("robustness_samples.csv");
    let mut writer = BufWriter::new(File::create(&path).expect("failed to create robustness csv"));
    writeln!(
        writer,
        "endpoint_index,endpoint_path,eval_len,eval_seed,baseline_score,endpoint_score,delta_vs_baseline,positive"
    )
    .expect("failed to write robustness header");

    for (endpoint_idx, (endpoint_path, endpoint_net, endpoint_proj)) in endpoints.iter().enumerate()
    {
        for &eval_len in &cli.robustness_eval_lens {
            for &seed in &cli.endpoint_eval_seeds {
                let mut base_eval_net = baseline_net.clone();
                let baseline_score = eval_score(
                    &cli.score_mode,
                    &mut base_eval_net,
                    &baseline_proj,
                    table,
                    pair_ids,
                    hot_to_idx,
                    bigram,
                    unigram,
                    eval_len,
                    &mut StdRng::seed_from_u64(seed),
                    &init.propagation,
                    init.output_start(),
                    cli.h,
                    init.input_end(),
                    cli.input_scatter,
                );
                let mut endpoint_eval_net = endpoint_net.clone();
                let endpoint_score = eval_score(
                    &cli.score_mode,
                    &mut endpoint_eval_net,
                    endpoint_proj,
                    table,
                    pair_ids,
                    hot_to_idx,
                    bigram,
                    unigram,
                    eval_len,
                    &mut StdRng::seed_from_u64(seed),
                    &init.propagation,
                    init.output_start(),
                    cli.h,
                    init.input_end(),
                    cli.input_scatter,
                );
                let delta = endpoint_score - baseline_score;
                writeln!(
                    writer,
                    "{},\"{}\",{},{},{:.12},{:.12},{:.12},{}",
                    endpoint_idx + 1,
                    endpoint_path,
                    eval_len,
                    seed,
                    baseline_score,
                    endpoint_score,
                    delta,
                    delta > 0.0
                )
                .expect("failed to write robustness row");
            }
        }
    }
    writer.flush().expect("failed to flush robustness csv");
}

fn mean_score_for_mode(
    score_mode: &str,
    net: &Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    bigram: &[Vec<f64>],
    unigram: &[f64],
    eval_len: usize,
    eval_seeds: &[u64],
    init: &InitConfig,
    h: usize,
    input_scatter: bool,
) -> f64 {
    let mut total = 0.0;
    for &seed in eval_seeds {
        let mut eval_net = net.clone();
        total += eval_score(
            score_mode,
            &mut eval_net,
            proj,
            table,
            pair_ids,
            hot_to_idx,
            bigram,
            unigram,
            eval_len,
            &mut StdRng::seed_from_u64(seed),
            &init.propagation,
            init.output_start(),
            h,
            init.input_end(),
            input_scatter,
        );
    }
    total / eval_seeds.len() as f64
}

fn repair_class_rank(class_name: &str) -> i32 {
    match class_name {
        "FULL_REPAIR" => 4,
        "STRONG_REPAIR" => 3,
        "WEAK_REPAIR" => 2,
        "NO_REPAIR" => 1,
        _ => 0,
    }
}

fn run_repair_scan(
    cli: &Cli,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    bigram: &[Vec<f64>],
    unigram: &[f64],
    init: &InitConfig,
) {
    let repair_start = cli
        .repair_start
        .as_ref()
        .expect("--mode repair-scan requires --repair-start");
    let baseline_path = cli
        .checkpoints
        .first()
        .expect("baseline checkpoint required");
    let (baseline_net, baseline_proj, _) =
        load_checkpoint(baseline_path).expect("failed to load baseline checkpoint");
    let (repair_net, repair_proj, _) =
        load_checkpoint(repair_start).expect("failed to load repair start checkpoint");
    assert_eq!(baseline_net.neuron_count(), cli.h, "baseline H mismatch");
    assert_eq!(repair_net.neuron_count(), cli.h, "repair-start H mismatch");

    let repair_coord = encode_coord(&repair_net);
    let samples_path = cli.out.join("repair_samples.csv");
    let mut writer =
        BufWriter::new(File::create(&samples_path).expect("failed to create repair_samples.csv"));
    writeln!(
        writer,
        "proposal_id,proposal_seed,radius,mutation_type,edge_edits,threshold_edits,channel_edits,polarity_edits,direct_genome_distance,edges,smooth_baseline,smooth_score,smooth_delta,accuracy_baseline,accuracy_score,accuracy_delta,echo_baseline,echo_score,echo_delta,unigram_baseline,unigram_score,unigram_delta,smooth_retain,accuracy_retain,echo_safe,repair_class,rank_score"
    )
    .expect("failed to write repair_samples header");

    let metric_names = ["smooth", "accuracy", "echo", "unigram"];
    let mut baseline_scores = Vec::new();
    for metric in metric_names {
        baseline_scores.push(mean_score_for_mode(
            metric,
            &baseline_net,
            &baseline_proj,
            table,
            pair_ids,
            hot_to_idx,
            bigram,
            unigram,
            cli.eval_len,
            &cli.repair_eval_seeds,
            init,
            cli.h,
            cli.input_scatter,
        ));
    }

    let mut candidates = Vec::new();
    let mut proposal_id = 0usize;
    for &radius in &cli.radii {
        for &mutation_type in &cli.mutation_types {
            assert!(
                matches!(mutation_type, MutationType::Edge | MutationType::Threshold),
                "repair-scan supports only edge,threshold mutation types"
            );
            for sample_id in 0..cli.repair_samples_per_bucket {
                proposal_id += 1;
                let proposal_seed = cli.seed
                    ^ ((radius as u64) << 40)
                    ^ ((sample_id as u64) << 16)
                    ^ match mutation_type {
                        MutationType::Edge => 0xD91A_E000u64,
                        MutationType::Threshold => 0xD91A_7000u64,
                        _ => unreachable!(),
                    };
                let mut candidate_net = repair_net.clone();
                let candidate_proj = repair_proj.clone();
                let mut rng = StdRng::seed_from_u64(proposal_seed);
                let counts =
                    apply_radius_mutation(&mut candidate_net, radius, mutation_type, &mut rng);
                let candidate_coord = encode_coord(&candidate_net);
                let direct_distance = coord_distance(&repair_coord, &candidate_coord);

                let mut scores = Vec::new();
                for metric in metric_names {
                    scores.push(mean_score_for_mode(
                        metric,
                        &candidate_net,
                        &candidate_proj,
                        table,
                        pair_ids,
                        hot_to_idx,
                        bigram,
                        unigram,
                        cli.eval_len,
                        &cli.repair_eval_seeds,
                        init,
                        cli.h,
                        cli.input_scatter,
                    ));
                }
                let smooth_delta = scores[0] - baseline_scores[0];
                let accuracy_delta = scores[1] - baseline_scores[1];
                let echo_delta = scores[2] - baseline_scores[2];
                let unigram_delta = scores[3] - baseline_scores[3];
                let smooth_retain = smooth_delta >= 0.0120;
                let accuracy_retain = accuracy_delta >= 0.0020;
                let echo_safe = echo_delta.abs() <= 0.0010;
                let repair_class = if smooth_retain && accuracy_retain && echo_safe {
                    if unigram_delta >= 0.0 {
                        "FULL_REPAIR"
                    } else if unigram_delta >= -0.0044 {
                        "STRONG_REPAIR"
                    } else if unigram_delta > -0.008823296 {
                        "WEAK_REPAIR"
                    } else {
                        "NO_REPAIR"
                    }
                } else {
                    "FAIL_RETAIN"
                };
                let rank_score = repair_class_rank(repair_class) as f64 * 1_000.0
                    + unigram_delta * 100.0
                    + smooth_delta * 10.0
                    + accuracy_delta;
                writeln!(
                    writer,
                    "{},{},{},{},{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{},{},{},{},{:.12}",
                    proposal_id,
                    proposal_seed,
                    radius,
                    mutation_type.as_str(),
                    counts.edge,
                    counts.threshold,
                    counts.channel,
                    counts.polarity,
                    direct_distance,
                    candidate_net.edge_count(),
                    baseline_scores[0],
                    scores[0],
                    smooth_delta,
                    baseline_scores[1],
                    scores[1],
                    accuracy_delta,
                    baseline_scores[2],
                    scores[2],
                    echo_delta,
                    baseline_scores[3],
                    scores[3],
                    unigram_delta,
                    smooth_retain,
                    accuracy_retain,
                    echo_safe,
                    repair_class,
                    rank_score
                )
                .expect("failed to write repair row");
                candidates.push(RepairCandidate {
                    proposal_id,
                    proposal_seed,
                    radius,
                    mutation_type,
                    counts,
                    direct_distance,
                    edges: candidate_net.edge_count(),
                    smooth_score: scores[0],
                    smooth_delta,
                    accuracy_score: scores[1],
                    accuracy_delta,
                    echo_score: scores[2],
                    echo_delta,
                    unigram_score: scores[3],
                    unigram_delta,
                    smooth_retain,
                    accuracy_retain,
                    echo_safe,
                    repair_class,
                    rank_score,
                    net: candidate_net,
                    proj: candidate_proj,
                });
            }
        }
    }
    writer.flush().expect("failed to flush repair_samples");

    candidates.sort_by(|a, b| {
        b.rank_score
            .partial_cmp(&a.rank_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.unigram_delta.partial_cmp(&a.unigram_delta).unwrap())
            .then_with(|| b.smooth_delta.partial_cmp(&a.smooth_delta).unwrap())
    });
    let candidates_dir = cli.out.join("candidates");
    create_dir_all(&candidates_dir).expect("failed to create repair candidates dir");
    let mut cand_writer = BufWriter::new(
        File::create(cli.out.join("repair_candidates.csv"))
            .expect("failed to create repair_candidates.csv"),
    );
    writeln!(
        cand_writer,
        "rank,checkpoint,proposal_id,proposal_seed,radius,mutation_type,edge_edits,threshold_edits,direct_genome_distance,edges,smooth_score,smooth_delta,accuracy_score,accuracy_delta,echo_score,echo_delta,unigram_score,unigram_delta,smooth_retain,accuracy_retain,echo_safe,repair_class,rank_score"
    )
    .expect("failed to write repair_candidates header");
    for (rank_idx, candidate) in candidates.iter().take(cli.repair_export_top).enumerate() {
        let rank = rank_idx + 1;
        let ckpt_path = candidates_dir.join(format!("top_{rank:02}.ckpt"));
        save_checkpoint(
            &ckpt_path,
            &candidate.net,
            &candidate.proj,
            CheckpointMeta {
                step: rank,
                accuracy: candidate.accuracy_score,
                label: format!(
                    "D9.1a repair rank={} class={} smooth_delta={:.6} unigram_delta={:.6}",
                    rank, candidate.repair_class, candidate.smooth_delta, candidate.unigram_delta
                ),
            },
        )
        .expect("failed to save repair candidate checkpoint");
        writeln!(
            cand_writer,
            "{},{},{},{},{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{},{},{},{},{:.12}",
            rank,
            ckpt_path.display(),
            candidate.proposal_id,
            candidate.proposal_seed,
            candidate.radius,
            candidate.mutation_type.as_str(),
            candidate.counts.edge,
            candidate.counts.threshold,
            candidate.direct_distance,
            candidate.edges,
            candidate.smooth_score,
            candidate.smooth_delta,
            candidate.accuracy_score,
            candidate.accuracy_delta,
            candidate.echo_score,
            candidate.echo_delta,
            candidate.unigram_score,
            candidate.unigram_delta,
            candidate.smooth_retain,
            candidate.accuracy_retain,
            candidate.echo_safe,
            candidate.repair_class,
            candidate.rank_score,
        )
        .expect("failed to write repair candidate row");
    }
    cand_writer
        .flush()
        .expect("failed to flush repair_candidates");
}

fn evaluate_multi_metrics(
    net: &Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    bigram: &[Vec<f64>],
    unigram: &[f64],
    eval_len: usize,
    eval_seeds: &[u64],
    init: &InitConfig,
    h: usize,
    input_scatter: bool,
) -> MultiMetricScores {
    MultiMetricScores {
        smooth: mean_score_for_mode(
            "smooth",
            net,
            proj,
            table,
            pair_ids,
            hot_to_idx,
            bigram,
            unigram,
            eval_len,
            eval_seeds,
            init,
            h,
            input_scatter,
        ),
        accuracy: mean_score_for_mode(
            "accuracy",
            net,
            proj,
            table,
            pair_ids,
            hot_to_idx,
            bigram,
            unigram,
            eval_len,
            eval_seeds,
            init,
            h,
            input_scatter,
        ),
        echo: mean_score_for_mode(
            "echo",
            net,
            proj,
            table,
            pair_ids,
            hot_to_idx,
            bigram,
            unigram,
            eval_len,
            eval_seeds,
            init,
            h,
            input_scatter,
        ),
        unigram: mean_score_for_mode(
            "unigram",
            net,
            proj,
            table,
            pair_ids,
            hot_to_idx,
            bigram,
            unigram,
            eval_len,
            eval_seeds,
            init,
            h,
            input_scatter,
        ),
    }
}

fn metric_deltas(candidate: MultiMetricScores, baseline: MultiMetricScores) -> MultiMetricScores {
    MultiMetricScores {
        smooth: candidate.smooth - baseline.smooth,
        accuracy: candidate.accuracy - baseline.accuracy,
        echo: candidate.echo - baseline.echo,
        unigram: candidate.unigram - baseline.unigram,
    }
}

fn mo_constraints_pass(deltas: MultiMetricScores) -> bool {
    deltas.smooth >= 0.0120 && deltas.accuracy >= 0.0020 && deltas.echo.abs() <= 0.0010
}

fn mo_score(deltas: MultiMetricScores) -> f64 {
    deltas.smooth + 0.50 * deltas.accuracy + 1.50 * deltas.unigram.max(-0.0120)
        - 0.25 * deltas.echo.abs()
}

fn mo_class(deltas: MultiMetricScores) -> &'static str {
    if !mo_constraints_pass(deltas) {
        "FAIL_RETAIN"
    } else if deltas.unigram >= 0.0 {
        "FULL_GENERALIST"
    } else if deltas.unigram >= -0.0044 {
        "MULTI_OBJECTIVE_SUCCESS"
    } else if deltas.unigram > -0.008735006 {
        "WEAK_SIGNAL"
    } else {
        "RETAINED_SPECIALIST"
    }
}

fn mo_class_rank(class_name: &str) -> i32 {
    match class_name {
        "FULL_GENERALIST" => 5,
        "MULTI_OBJECTIVE_SUCCESS" => 4,
        "WEAK_SIGNAL" => 3,
        "RETAINED_SPECIALIST" => 2,
        "FAIL_RETAIN" => 1,
        _ => 0,
    }
}

fn run_multi_objective_climb(
    cli: &Cli,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    bigram: &[Vec<f64>],
    unigram: &[f64],
    init: &InitConfig,
) {
    let repair_start = cli
        .repair_start
        .as_ref()
        .expect("--mode multi-objective-climb requires --repair-start");
    let baseline_path = cli
        .checkpoints
        .first()
        .expect("baseline checkpoint required");
    let (baseline_net, baseline_proj, _) =
        load_checkpoint(baseline_path).expect("failed to load baseline checkpoint");
    let (start_net, start_proj, _) =
        load_checkpoint(repair_start).expect("failed to load repair-start checkpoint");
    assert_eq!(baseline_net.neuron_count(), cli.h, "baseline H mismatch");
    assert_eq!(start_net.neuron_count(), cli.h, "repair-start H mismatch");
    assert!(
        cli.mutation_types
            .iter()
            .all(|t| matches!(t, MutationType::Edge | MutationType::Threshold)),
        "D9.2a multi-objective-climb only supports edge,threshold mutation types"
    );
    assert!(
        !cli.mo_eval_seeds.is_empty(),
        "--mo-eval-seeds must not be empty"
    );

    println!(
        "D9.2a multi-objective climb: baseline={} start={} climbers={} steps={} eval_len={} seeds={} radii={:?}",
        baseline_path.display(),
        repair_start.display(),
        cli.mo_climbers,
        cli.mo_steps,
        cli.eval_len,
        cli.mo_eval_seeds.len(),
        cli.radii
    );

    let start_coord = encode_coord(&start_net);
    let baseline_scores = evaluate_multi_metrics(
        &baseline_net,
        &baseline_proj,
        table,
        pair_ids,
        hot_to_idx,
        bigram,
        unigram,
        cli.eval_len,
        &cli.mo_eval_seeds,
        init,
        cli.h,
        cli.input_scatter,
    );
    let start_scores = evaluate_multi_metrics(
        &start_net,
        &start_proj,
        table,
        pair_ids,
        hot_to_idx,
        bigram,
        unigram,
        cli.eval_len,
        &cli.mo_eval_seeds,
        init,
        cli.h,
        cli.input_scatter,
    );
    let start_deltas = metric_deltas(start_scores, baseline_scores);
    let start_mo = mo_score(start_deltas);
    println!(
        "  start deltas: smooth={:.6} accuracy={:.6} echo={:.6} unigram={:.6} mo={:.6} class={}",
        start_deltas.smooth,
        start_deltas.accuracy,
        start_deltas.echo,
        start_deltas.unigram,
        start_mo,
        mo_class(start_deltas)
    );

    let path = cli.out.join("multi_objective_paths.csv");
    let mut writer =
        BufWriter::new(File::create(&path).expect("failed to create multi_objective_paths.csv"));
    writeln!(
        writer,
        "climber_id,step_index,proposal_seed,radius,mutation_type,edge_edits,threshold_edits,direct_genome_distance,edges,smooth_score,smooth_delta,accuracy_score,accuracy_delta,echo_score,echo_delta,unigram_score,unigram_delta,mo_score,mo_class,constraints_pass,accepted,accept_reason"
    )
    .expect("failed to write multi_objective_paths header");

    let mut candidates: Vec<MultiObjectiveCandidate> = Vec::new();
    for climber_id in 0..cli.mo_climbers {
        let climber_seed = cli.seed ^ ((climber_id as u64) << 32) ^ 0xD92A_0001u64;
        let mut proposal_rng = StdRng::seed_from_u64(climber_seed);
        let mut current = start_net.clone();
        let current_proj = start_proj.clone();
        let mut current_mo = start_mo;
        for step_index in 0..cli.mo_steps {
            let mutation_type =
                cli.mutation_types[proposal_rng.gen_range(0..cli.mutation_types.len())];
            let radius = cli.radii[proposal_rng.gen_range(0..cli.radii.len())];
            let proposal_seed = proposal_rng.gen::<u64>();
            let mut mutation_rng = StdRng::seed_from_u64(proposal_seed);
            let mut candidate_net = current.clone();
            let candidate_proj = current_proj.clone();
            let counts =
                apply_radius_mutation(&mut candidate_net, radius, mutation_type, &mut mutation_rng);
            let candidate_coord = encode_coord(&candidate_net);
            let direct_distance = coord_distance(&start_coord, &candidate_coord);
            let eval_start = Instant::now();
            let scores = evaluate_multi_metrics(
                &candidate_net,
                &candidate_proj,
                table,
                pair_ids,
                hot_to_idx,
                bigram,
                unigram,
                cli.eval_len,
                &cli.mo_eval_seeds,
                init,
                cli.h,
                cli.input_scatter,
            );
            let deltas = metric_deltas(scores, baseline_scores);
            let score = mo_score(deltas);
            let constraints_pass = mo_constraints_pass(deltas);
            let accepted = constraints_pass && score >= current_mo - 0.00025;
            let accept_reason = if !constraints_pass {
                "constraint_fail"
            } else if score > current_mo {
                "improve"
            } else if accepted {
                "epsilon_keep"
            } else {
                "score_drop"
            };
            let class_name = mo_class(deltas);
            if accepted {
                current = candidate_net.clone();
                current_mo = score;
            }
            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{},{},{},{}",
                climber_id,
                step_index,
                proposal_seed,
                radius,
                mutation_type.as_str(),
                counts.edge,
                counts.threshold,
                direct_distance,
                candidate_net.edge_count(),
                scores.smooth,
                deltas.smooth,
                scores.accuracy,
                deltas.accuracy,
                scores.echo,
                deltas.echo,
                scores.unigram,
                deltas.unigram,
                score,
                class_name,
                constraints_pass,
                accepted,
                accept_reason
            )
            .expect("failed to write multi-objective path row");
            candidates.push(MultiObjectiveCandidate {
                climber_id,
                step_index,
                proposal_seed,
                radius,
                mutation_type,
                counts,
                direct_distance,
                edges: candidate_net.edge_count(),
                scores,
                deltas,
                mo_score: score,
                mo_class: class_name,
                accepted,
                net: candidate_net,
                proj: candidate_proj,
            });
            println!(
                "  climber={} step={} class={} accepted={} mo={:.6} d=[{:.5},{:.5},{:.5},{:.5}] eval_ms={:.1}",
                climber_id,
                step_index,
                class_name,
                accepted,
                score,
                deltas.smooth,
                deltas.accuracy,
                deltas.echo,
                deltas.unigram,
                eval_start.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
    writer
        .flush()
        .expect("failed to flush multi_objective_paths");

    candidates.sort_by(|a, b| {
        mo_class_rank(b.mo_class)
            .cmp(&mo_class_rank(a.mo_class))
            .then_with(|| b.deltas.unigram.partial_cmp(&a.deltas.unigram).unwrap())
            .then_with(|| b.mo_score.partial_cmp(&a.mo_score).unwrap())
            .then_with(|| b.deltas.smooth.partial_cmp(&a.deltas.smooth).unwrap())
    });
    let candidates_dir = cli.out.join("candidates");
    create_dir_all(&candidates_dir).expect("failed to create multi-objective candidates dir");
    let mut cand_writer = BufWriter::new(
        File::create(cli.out.join("multi_objective_candidates.csv"))
            .expect("failed to create multi_objective_candidates.csv"),
    );
    writeln!(
        cand_writer,
        "rank,checkpoint,climber_id,step_index,proposal_seed,radius,mutation_type,edge_edits,threshold_edits,direct_genome_distance,edges,smooth_score,smooth_delta,accuracy_score,accuracy_delta,echo_score,echo_delta,unigram_score,unigram_delta,mo_score,mo_class,accepted"
    )
    .expect("failed to write multi_objective_candidates header");
    for (rank_idx, candidate) in candidates.iter().take(cli.mo_export_top).enumerate() {
        let rank = rank_idx + 1;
        let ckpt_path = candidates_dir.join(format!("top_{rank:02}.ckpt"));
        save_checkpoint(
            &ckpt_path,
            &candidate.net,
            &candidate.proj,
            CheckpointMeta {
                step: candidate.step_index,
                accuracy: candidate.scores.accuracy,
                label: format!(
                    "D9.2a mo rank={} class={} smooth_delta={:.6} unigram_delta={:.6}",
                    rank, candidate.mo_class, candidate.deltas.smooth, candidate.deltas.unigram
                ),
            },
        )
        .expect("failed to save multi-objective candidate checkpoint");
        writeln!(
            cand_writer,
            "{},{},{},{},{},{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{},{}",
            rank,
            ckpt_path.display(),
            candidate.climber_id,
            candidate.step_index,
            candidate.proposal_seed,
            candidate.radius,
            candidate.mutation_type.as_str(),
            candidate.counts.edge,
            candidate.counts.threshold,
            candidate.direct_distance,
            candidate.edges,
            candidate.scores.smooth,
            candidate.deltas.smooth,
            candidate.scores.accuracy,
            candidate.deltas.accuracy,
            candidate.scores.echo,
            candidate.deltas.echo,
            candidate.scores.unigram,
            candidate.deltas.unigram,
            candidate.mo_score,
            candidate.mo_class,
            candidate.accepted
        )
        .expect("failed to write multi-objective candidate row");
    }
    cand_writer
        .flush()
        .expect("failed to flush multi_objective_candidates");
}

fn main() {
    let cli = parse_args();
    create_dir_all(&cli.out).expect("failed to create output directory");

    let table = VcbpTable::from_packed(&cli.packed).expect("failed to load packed VCBP table");
    let corpus = std::fs::read(&cli.corpus).expect("failed to read corpus");
    let (pair_ids, hot_to_idx, n_classes) = build_corpus_pairs(&corpus, &table, 397);
    let bigram = build_pair_bigram(&pair_ids, &hot_to_idx, n_classes);
    let unigram = build_pair_unigram(&pair_ids, &hot_to_idx, n_classes);
    let init = InitConfig::phi(cli.h);
    let samples_path = cli.out.join("samples.csv");
    let roundtrip_path = cli.out.join("roundtrip.json");
    let meta_path = cli.out.join("run_meta.json");
    let mut writer =
        BufWriter::new(File::create(&samples_path).expect("failed to create samples.csv"));
    write_header(&mut writer);

    let mut roundtrip_results = Vec::new();
    let mut global_sample_id = 0usize;
    let total_start = Instant::now();

    for (base_index, checkpoint) in cli.checkpoints.iter().enumerate() {
        println!(
            "Loading checkpoint {}: {}",
            base_index,
            checkpoint.display()
        );
        let (base_net, proj, meta) =
            load_checkpoint(checkpoint).expect("failed to load checkpoint");
        assert_eq!(
            base_net.neuron_count(),
            cli.h,
            "checkpoint H mismatch: expected {}, got {}",
            cli.h,
            base_net.neuron_count()
        );

        let base_coord = encode_coord(&base_net);
        let decoded = decode_coord(&base_coord);
        let roundtrip_ok = compare_core(&base_net, &decoded);
        roundtrip_results.push(serde_json::json!({
            "base_index": base_index,
            "checkpoint": checkpoint.display().to_string(),
            "roundtrip_ok": roundtrip_ok,
            "edge_count": base_net.edge_count(),
            "coord_dims": cli.h * (cli.h - 1) + cli.h * 3,
            "meta_step": meta.step,
            "meta_accuracy": meta.accuracy,
        }));
        if !roundtrip_ok {
            panic!(
                "direct genome coord roundtrip failed for {}",
                checkpoint.display()
            );
        }

        let mut base_for_metrics = base_net.clone();
        let base_metrics = compute_panel_metrics(
            &mut base_for_metrics,
            &proj,
            &table,
            &pair_ids,
            &hot_to_idx,
            &init.propagation,
            init.output_start(),
            cli.h,
            init.input_end(),
            cli.input_scatter,
        );

        if cli.mode == "endpoint-bridge" {
            run_endpoint_bridge(
                &cli,
                &table,
                &pair_ids,
                &hot_to_idx,
                &bigram,
                &unigram,
                &init,
            );
            break;
        }

        if cli.mode == "endpoint-overlap" {
            run_endpoint_overlap(
                &cli,
                &table,
                &pair_ids,
                &hot_to_idx,
                &bigram,
                &unigram,
                &init,
            );
            break;
        }

        if cli.mode == "endpoint-robustness" {
            run_endpoint_robustness(
                &cli,
                &table,
                &pair_ids,
                &hot_to_idx,
                &bigram,
                &unigram,
                &init,
            );
            break;
        }

        if cli.mode == "repair-scan" {
            run_repair_scan(
                &cli,
                &table,
                &pair_ids,
                &hot_to_idx,
                &bigram,
                &unigram,
                &init,
            );
            break;
        }

        if cli.mode == "multi-objective-climb" {
            run_multi_objective_climb(
                &cli,
                &table,
                &pair_ids,
                &hot_to_idx,
                &bigram,
                &unigram,
                &init,
            );
            break;
        }

        if cli.mode == "paratrooper-climb" {
            let climb_path = cli.out.join("paratrooper_paths.csv");
            let mut climb_writer = BufWriter::new(
                File::create(&climb_path).expect("failed to create paratrooper_paths.csv"),
            );
            write_climb_header(&mut climb_writer);
            let mut endpoint_candidates: Vec<EndpointCandidate> = Vec::new();
            let tile_list: Vec<(usize, usize)> = cli.target_tiles.clone().unwrap_or_else(|| {
                let mut tiles = Vec::with_capacity(cli.lat_bins * cli.lon_bins);
                for lat_bin in 0..cli.lat_bins {
                    for lon_bin in 0..cli.lon_bins {
                        tiles.push((lat_bin, lon_bin));
                    }
                }
                tiles
            });
            let radius = cli.radii.first().copied().unwrap_or(1);
            let mut global_climber_id = 0usize;
            for (tile_idx, &(lat_bin, lon_bin)) in tile_list.iter().enumerate() {
                let tile_id = format!("{}_{}", lat_bin, lon_bin);
                for climber_idx in 0..cli.climbers_per_tile {
                    let climber_seed = cli.seed
                        ^ ((base_index as u64) << 56)
                        ^ ((lat_bin as u64) << 44)
                        ^ ((lon_bin as u64) << 32)
                        ^ ((climber_idx as u64) << 16)
                        ^ 0xD90D_0001u64;
                    let eval_seed = climber_seed.wrapping_add(1_000_000);
                    let mut proposal_rng = StdRng::seed_from_u64(climber_seed);
                    let mut current = base_net.clone();
                    let mut current_proj = proj.clone();
                    let mut current_eval_net = current.clone();
                    let start_score = eval_score(
                        &cli.score_mode,
                        &mut current_eval_net,
                        &current_proj,
                        &table,
                        &pair_ids,
                        &hot_to_idx,
                        &bigram,
                        &unigram,
                        cli.eval_len,
                        &mut StdRng::seed_from_u64(eval_seed),
                        &init.propagation,
                        init.output_start(),
                        cli.h,
                        init.input_end(),
                        cli.input_scatter,
                    );
                    let mut current_score = start_score;
                    let mut best_score = start_score;
                    let mut best_net = current.clone();
                    let mut best_proj = current_proj.clone();
                    let mut best_step = 0usize;
                    let mut best_mutation_type_at_best = String::from("start");
                    let mut accepted_seed_chain: Vec<u64> = Vec::new();
                    let mut best_seed_chain: Vec<u64> = Vec::new();
                    for step_idx in 0..cli.climb_steps {
                        let mutation_type =
                            cli.mutation_types[proposal_rng.gen_range(0..cli.mutation_types.len())];
                        let proposal_seed = proposal_rng.gen::<u64>();
                        let mut mutation_rng = StdRng::seed_from_u64(proposal_seed);
                        let mut candidate = current.clone();
                        let mut candidate_proj = current_proj.clone();
                        let counts = if mutation_type == MutationType::Projection {
                            apply_radius_projection_mutation(
                                &mut candidate_proj,
                                radius,
                                &mut mutation_rng,
                            )
                        } else {
                            apply_radius_mutation(
                                &mut candidate,
                                radius,
                                mutation_type,
                                &mut mutation_rng,
                            )
                        };
                        let candidate_coord = encode_coord(&candidate);
                        let direct_distance = coord_distance(&base_coord, &candidate_coord);
                        let eval_start = Instant::now();
                        let mut cand_eval_net = candidate.clone();
                        let candidate_score = eval_score(
                            &cli.score_mode,
                            &mut cand_eval_net,
                            &candidate_proj,
                            &table,
                            &pair_ids,
                            &hot_to_idx,
                            &bigram,
                            &unigram,
                            cli.eval_len,
                            &mut StdRng::seed_from_u64(eval_seed),
                            &init.propagation,
                            init.output_start(),
                            cli.h,
                            init.input_end(),
                            cli.input_scatter,
                        );
                        let current_score_before = current_score;
                        let step_delta = candidate_score - current_score;
                        let accepted = step_delta >= -cli.accept_epsilon;
                        let accept_reason = if accepted && step_delta > 0.0 {
                            "improve"
                        } else if accepted {
                            "neutral"
                        } else {
                            "reject"
                        };
                        let mut accepted_score = current_score;
                        if accepted {
                            let mut next_seed_chain = accepted_seed_chain.clone();
                            next_seed_chain.push(proposal_seed);
                            if candidate_score > best_score {
                                best_score = candidate_score;
                                best_net = candidate.clone();
                                best_proj = candidate_proj.clone();
                                best_step = step_idx;
                                best_mutation_type_at_best = mutation_type.as_str().to_string();
                                best_seed_chain = next_seed_chain.clone();
                            }
                            current = candidate.clone();
                            current_proj = candidate_proj.clone();
                            current_score = candidate_score;
                            accepted_seed_chain = next_seed_chain;
                            accepted_score = current_score;
                        }
                        let mut cand_metrics_net = candidate.clone();
                        let metrics = compute_panel_metrics(
                            &mut cand_metrics_net,
                            &candidate_proj,
                            &table,
                            &pair_ids,
                            &hot_to_idx,
                            &init.propagation,
                            init.output_start(),
                            cli.h,
                            init.input_end(),
                            cli.input_scatter,
                        );
                        let eval_ms = eval_start.elapsed().as_secs_f64() * 1000.0;
                        let bdist = behavior_distance(&base_metrics, &metrics, cli.h);
                        let candidate_edges = candidate.edge_count();
                        let current_edges = current.edge_count();

                        writeln!(
                            climb_writer,
                            "{},{},{},{:.17},{},{},{},{},{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{},{},{},{},{},{},{},{},{},{:.17},{:.6},{},{:.17}",
                            base_index,
                            checkpoint.display(),
                            meta.step,
                            meta.accuracy,
                            global_climber_id,
                            tile_id,
                            lat_bin,
                            lon_bin,
                            step_idx,
                            proposal_seed,
                            mutation_type.as_str(),
                            accepted,
                            accept_reason,
                            current_score_before,
                            candidate_score,
                            accepted_score,
                            best_score,
                            accepted_score - start_score,
                            best_score - start_score,
                            step_delta,
                            start_score,
                            radius,
                            direct_distance,
                            counts.edge,
                            counts.threshold,
                            counts.channel,
                            counts.polarity,
                            counts.projection,
                            current_edges,
                            candidate_edges,
                            bdist,
                            eval_ms,
                            cli.score_mode,
                            cli.accept_epsilon,
                        )
                        .expect("failed to write climb row");
                        global_sample_id += 1;
                    }
                    if cli.export_top_endpoints > 0 && best_score > start_score {
                        endpoint_candidates.push(EndpointCandidate {
                            base_index,
                            base_checkpoint: checkpoint.display().to_string(),
                            base_step: meta.step,
                            base_accuracy: meta.accuracy,
                            tile_id: tile_id.clone(),
                            lat_bin,
                            lon_bin,
                            climber_id: global_climber_id,
                            climber_seed,
                            eval_seed,
                            best_step,
                            best_score,
                            start_score,
                            final_score: current_score,
                            mutation_type_at_best: best_mutation_type_at_best.clone(),
                            accepted_seed_chain: best_seed_chain.clone(),
                            net: best_net,
                            proj: best_proj,
                        });
                        endpoint_candidates.sort_by(|a, b| {
                            (b.best_score - b.start_score)
                                .partial_cmp(&(a.best_score - a.start_score))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        endpoint_candidates.truncate(cli.export_top_endpoints);
                    }
                    global_climber_id += 1;
                }
                climb_writer
                    .flush()
                    .expect("failed to flush paratrooper_paths.csv");
                println!(
                    "  base={} paratrooper tile={}/{} climbers={} total_rows={}",
                    base_index,
                    tile_idx + 1,
                    tile_list.len(),
                    cli.climbers_per_tile,
                    global_sample_id
                );
            }
            if cli.export_top_endpoints > 0 {
                export_endpoint_candidates(
                    &cli,
                    &endpoint_candidates,
                    &base_net,
                    &proj,
                    &table,
                    &pair_ids,
                    &hot_to_idx,
                    &bigram,
                    &unigram,
                    &init,
                );
            }
        } else if matches!(cli.mode.as_str(), "planet-scout" | "homogeneous") {
            let tile_list: Vec<(usize, usize)> = cli.target_tiles.clone().unwrap_or_else(|| {
                let mut tiles = Vec::with_capacity(cli.lat_bins * cli.lon_bins);
                for lat_bin in 0..cli.lat_bins {
                    for lon_bin in 0..cli.lon_bins {
                        tiles.push((lat_bin, lon_bin));
                    }
                }
                tiles
            });
            for (tile_idx, &(lat_bin, lon_bin)) in tile_list.iter().enumerate() {
                let tile_id = format!("{}_{}", lat_bin, lon_bin);
                for &radius in &cli.radii {
                    for &mutation_type in &cli.mutation_types {
                        for sample_idx in 0..cli.samples_per_tile {
                            let sample_seed = cli.seed
                                ^ ((base_index as u64) << 56)
                                ^ ((lat_bin as u64) << 44)
                                ^ ((lon_bin as u64) << 32)
                                ^ ((radius as u64) << 20)
                                ^ ((sample_idx as u64) << 8)
                                ^ (mutation_type.as_str().as_bytes()[0] as u64)
                                ^ ((cli.sample_layer.as_bytes()[0] as u64) << 4);
                            let mut mutation_rng = StdRng::seed_from_u64(sample_seed);
                            let mut candidate = base_net.clone();
                            let counts = apply_radius_mutation(
                                &mut candidate,
                                radius,
                                mutation_type,
                                &mut mutation_rng,
                            );
                            let candidate_coord = encode_coord(&candidate);
                            let direct_distance = coord_distance(&base_coord, &candidate_coord);

                            let eval_seed = cli
                                .seed
                                .wrapping_add(1_000_000)
                                .wrapping_add(global_sample_id as u64);
                            let mut base_eval_net = base_net.clone();
                            let mut cand_eval_net = candidate.clone();
                            let eval_start = Instant::now();
                            let base_score = eval_score(
                                &cli.score_mode,
                                &mut base_eval_net,
                                &proj,
                                &table,
                                &pair_ids,
                                &hot_to_idx,
                                &bigram,
                                &unigram,
                                cli.eval_len,
                                &mut StdRng::seed_from_u64(eval_seed),
                                &init.propagation,
                                init.output_start(),
                                cli.h,
                                init.input_end(),
                                cli.input_scatter,
                            );
                            let score = eval_score(
                                &cli.score_mode,
                                &mut cand_eval_net,
                                &proj,
                                &table,
                                &pair_ids,
                                &hot_to_idx,
                                &bigram,
                                &unigram,
                                cli.eval_len,
                                &mut StdRng::seed_from_u64(eval_seed),
                                &init.propagation,
                                init.output_start(),
                                cli.h,
                                init.input_end(),
                                cli.input_scatter,
                            );
                            let mut cand_metrics_net = candidate.clone();
                            let metrics = compute_panel_metrics(
                                &mut cand_metrics_net,
                                &proj,
                                &table,
                                &pair_ids,
                                &hot_to_idx,
                                &init.propagation,
                                init.output_start(),
                                cli.h,
                                init.input_end(),
                                cli.input_scatter,
                            );
                            let eval_ms = eval_start.elapsed().as_secs_f64() * 1000.0;
                            let bdist = behavior_distance(&base_metrics, &metrics, cli.h);

                            writeln!(
                                    writer,
                                    "{},{},{},{:.17},{},{},{},{},{},{},{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{},{:.17},{},{:.17},{:.17},{:.17},{:.17},{:.17},{},{:.17},{:.6},{},{},{},{},{}",
                                    base_index,
                                    checkpoint.display(),
                                    meta.step,
                                    meta.accuracy,
                                    global_sample_id,
                                    sample_seed,
                                    cli.h,
                                    cli.mode,
                                    mutation_type.as_str(),
                                    radius,
                                    direct_distance,
                                    counts.edge,
                                    counts.threshold,
                                    counts.channel,
                                    counts.polarity,
                                    base_score,
                                    score,
                                    score - base_score,
                                    bdist,
                                    candidate.edge_count(),
                                    metrics.panel_probe_acc,
                                    metrics.unique_predictions,
                                    metrics.collision_rate,
                                    metrics.f_active,
                                    metrics.h_output_mean,
                                    metrics.h_output_var,
                                    metrics.stable_rank,
                                    metrics.kernel_rank,
                                    metrics.separation_sp,
                                    eval_ms,
                                    cli.score_mode,
                                    tile_id,
                                    lat_bin,
                                    lon_bin,
                                    cli.sample_layer
                                )
                                .expect("failed to write planet scout sample row");
                            global_sample_id += 1;
                        }
                    }
                }
                writer.flush().expect("failed to flush samples.csv");
                println!(
                    "  base={} planet_scout tile={}/{} total_rows={}",
                    base_index,
                    tile_idx + 1,
                    tile_list.len(),
                    global_sample_id
                );
            }
        } else {
            for &radius in &cli.radii {
                for &mutation_type in &cli.mutation_types {
                    for sample_idx in 0..cli.samples_per_type {
                        let sample_seed = cli.seed
                            ^ ((base_index as u64) << 48)
                            ^ ((radius as u64) << 32)
                            ^ ((sample_idx as u64) << 8)
                            ^ (mutation_type.as_str().as_bytes()[0] as u64);
                        let mut mutation_rng = StdRng::seed_from_u64(sample_seed);
                        let mut candidate = base_net.clone();
                        let counts = apply_radius_mutation(
                            &mut candidate,
                            radius,
                            mutation_type,
                            &mut mutation_rng,
                        );
                        let candidate_coord = encode_coord(&candidate);
                        let direct_distance = coord_distance(&base_coord, &candidate_coord);

                        let eval_seed = cli
                            .seed
                            .wrapping_add(1_000_000)
                            .wrapping_add(global_sample_id as u64);
                        let mut base_eval_net = base_net.clone();
                        let mut cand_eval_net = candidate.clone();
                        let eval_start = Instant::now();
                        let base_score = eval_score(
                            &cli.score_mode,
                            &mut base_eval_net,
                            &proj,
                            &table,
                            &pair_ids,
                            &hot_to_idx,
                            &bigram,
                            &unigram,
                            cli.eval_len,
                            &mut StdRng::seed_from_u64(eval_seed),
                            &init.propagation,
                            init.output_start(),
                            cli.h,
                            init.input_end(),
                            cli.input_scatter,
                        );
                        let score = eval_score(
                            &cli.score_mode,
                            &mut cand_eval_net,
                            &proj,
                            &table,
                            &pair_ids,
                            &hot_to_idx,
                            &bigram,
                            &unigram,
                            cli.eval_len,
                            &mut StdRng::seed_from_u64(eval_seed),
                            &init.propagation,
                            init.output_start(),
                            cli.h,
                            init.input_end(),
                            cli.input_scatter,
                        );
                        let mut cand_metrics_net = candidate.clone();
                        let metrics = compute_panel_metrics(
                            &mut cand_metrics_net,
                            &proj,
                            &table,
                            &pair_ids,
                            &hot_to_idx,
                            &init.propagation,
                            init.output_start(),
                            cli.h,
                            init.input_end(),
                            cli.input_scatter,
                        );
                        let eval_ms = eval_start.elapsed().as_secs_f64() * 1000.0;
                        let bdist = behavior_distance(&base_metrics, &metrics, cli.h);

                        writeln!(
                        writer,
                        "{},{},{},{:.17},{},{},{},{},{},{},{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{},{:.17},{},{:.17},{:.17},{:.17},{:.17},{:.17},{},{:.17},{:.6},{},,,,direct_random",
                        base_index,
                        checkpoint.display(),
                        meta.step,
                        meta.accuracy,
                        global_sample_id,
                        sample_seed,
                        cli.h,
                        cli.mode,
                        mutation_type.as_str(),
                        radius,
                        direct_distance,
                        counts.edge,
                        counts.threshold,
                        counts.channel,
                        counts.polarity,
                        base_score,
                        score,
                        score - base_score,
                        bdist,
                        candidate.edge_count(),
                        metrics.panel_probe_acc,
                        metrics.unique_predictions,
                        metrics.collision_rate,
                        metrics.f_active,
                        metrics.h_output_mean,
                        metrics.h_output_var,
                        metrics.stable_rank,
                        metrics.kernel_rank,
                        metrics.separation_sp,
                        eval_ms,
                        cli.score_mode
                    )
                    .expect("failed to write sample row");
                        global_sample_id += 1;
                    }
                    writer.flush().expect("failed to flush samples.csv");
                    println!(
                        "  base={} radius={} type={} samples={} total_rows={}",
                        base_index,
                        radius,
                        mutation_type.as_str(),
                        cli.samples_per_type,
                        global_sample_id
                    );
                }
            }
        }
    }

    writer.flush().expect("failed to flush samples.csv");
    std::fs::write(
        &roundtrip_path,
        serde_json::to_string_pretty(&roundtrip_results).expect("roundtrip json"),
    )
    .expect("failed to write roundtrip.json");

    let meta = RunMeta {
        phase: if cli.mode == "paratrooper-climb" {
            "D9.0i"
        } else if cli.mode == "endpoint-bridge" {
            "D9.0v"
        } else if cli.mode == "endpoint-overlap" {
            "D9.0w"
        } else if cli.mode == "endpoint-robustness" {
            "D9.0x"
        } else if cli.mode == "repair-scan" {
            "D9.1a"
        } else if cli.mode == "multi-objective-climb" {
            "D9.2a"
        } else if matches!(cli.mode.as_str(), "planet-scout" | "homogeneous") {
            "D9.0e"
        } else {
            "D9.0b"
        },
        tool: "d9_direct_landscape",
        h: cli.h,
        mode: cli.mode.clone(),
        score_mode: cli.score_mode.clone(),
        seed: cli.seed,
        eval_len: cli.eval_len,
        samples_per_type: cli.samples_per_type,
        samples_per_tile: cli.samples_per_tile,
        lat_bins: matches!(
            cli.mode.as_str(),
            "planet-scout" | "homogeneous" | "paratrooper-climb"
        )
        .then_some(cli.lat_bins),
        lon_bins: matches!(
            cli.mode.as_str(),
            "planet-scout" | "homogeneous" | "paratrooper-climb"
        )
        .then_some(cli.lon_bins),
        target_tiles: cli.target_tiles.as_ref().map(|tiles| {
            tiles
                .iter()
                .map(|(lat, lon)| format!("{}_{}", lat, lon))
                .collect()
        }),
        sample_layer: cli.sample_layer.clone(),
        climbers_per_tile: (cli.mode == "paratrooper-climb").then_some(cli.climbers_per_tile),
        climb_steps: (cli.mode == "paratrooper-climb").then_some(cli.climb_steps),
        accept_epsilon: (cli.mode == "paratrooper-climb").then_some(cli.accept_epsilon),
        export_top_endpoints: (cli.mode == "paratrooper-climb").then_some(cli.export_top_endpoints),
        endpoint_eval_len: (cli.mode == "paratrooper-climb").then_some(cli.endpoint_eval_len),
        endpoint_eval_seeds: (cli.mode == "paratrooper-climb")
            .then_some(cli.endpoint_eval_seeds.clone()),
        bridge_endpoints: (cli.mode == "endpoint-bridge").then_some(
            cli.bridge_endpoints
                .iter()
                .map(|path| path.display().to_string())
                .collect(),
        ),
        bridge_steps: (cli.mode == "endpoint-bridge").then_some(cli.bridge_steps),
        overlap_samples: (cli.mode == "endpoint-overlap").then_some(cli.overlap_samples),
        robustness_eval_lens: (cli.mode == "endpoint-robustness")
            .then_some(cli.robustness_eval_lens.clone()),
        repair_start: (cli.mode == "repair-scan").then(|| {
            cli.repair_start
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default()
        }),
        repair_samples_per_bucket: (cli.mode == "repair-scan")
            .then_some(cli.repair_samples_per_bucket),
        repair_eval_seeds: (cli.mode == "repair-scan").then_some(cli.repair_eval_seeds.clone()),
        repair_export_top: (cli.mode == "repair-scan").then_some(cli.repair_export_top),
        mo_climbers: (cli.mode == "multi-objective-climb").then_some(cli.mo_climbers),
        mo_steps: (cli.mode == "multi-objective-climb").then_some(cli.mo_steps),
        mo_eval_seeds: (cli.mode == "multi-objective-climb").then_some(cli.mo_eval_seeds.clone()),
        mo_export_top: (cli.mode == "multi-objective-climb").then_some(cli.mo_export_top),
        radii: cli.radii.clone(),
        mutation_types: cli
            .mutation_types
            .iter()
            .map(|t| t.as_str().to_string())
            .collect(),
        checkpoints: cli
            .checkpoints
            .iter()
            .map(|p| p.display().to_string())
            .collect(),
        projection_policy: "fixed_from_checkpoint_not_in_direct_coord",
        core_coord_dims: cli.h * (cli.h - 1) + cli.h * 3,
        packed: cli.packed.display().to_string(),
        corpus: cli.corpus.display().to_string(),
        input_scatter: cli.input_scatter,
    };
    std::fs::write(
        &meta_path,
        serde_json::to_string_pretty(&meta).expect("meta json"),
    )
    .expect("failed to write run_meta.json");

    println!(
        "Wrote {} rows to {} in {:.1}s (classes={})",
        global_sample_id,
        samples_path.display(),
        total_start.elapsed().as_secs_f64(),
        n_classes
    );
}
