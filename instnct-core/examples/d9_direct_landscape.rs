//! Phase D9.0b: direct H=256 genome landscape probe.
//!
//! Offline-only diagnostic. Loads existing checkpoints, freezes the projection,
//! mutates only the persisted core genome (edges, threshold, channel, polarity),
//! and evaluates local neighborhoods around the base genome.

use instnct_core::{load_checkpoint, softmax, InitConfig, Int8Projection, Network, VcbpTable};
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
    Mixed,
}

impl MutationType {
    fn parse(value: &str) -> Self {
        match value {
            "edge" => Self::Edge,
            "threshold" | "theta" => Self::Threshold,
            "channel" => Self::Channel,
            "polarity" => Self::Polarity,
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
    radii: Vec<usize>,
    mutation_types: Vec<String>,
    checkpoints: Vec<String>,
    projection_policy: &'static str,
    core_coord_dims: usize,
    packed: String,
    corpus: String,
    input_scatter: bool,
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
                    matches!(score_mode.as_str(), "accuracy" | "smooth"),
                    "--score-mode expects accuracy|smooth"
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
                     --checkpoint PATH --H 256 --mode fail-fast|medium|full|staged|planet-scout|paratrooper-climb \
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
        "planet-scout" | "homogeneous" | "paratrooper-climb" => vec![1],
        other => panic!("--mode expects fail-fast|medium|full|staged|planet-scout, got {other}"),
    };
    let default_mutation_types = if matches!(
        mode.as_str(),
        "planet-scout" | "homogeneous" | "paratrooper-climb"
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
            MutationType::Mixed => unreachable!(),
        };
        assert!(ok, "direct mutation should be valid for H>=2");
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

fn eval_score(
    score_mode: &str,
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
        "base_index,base_checkpoint,base_step,base_accuracy,climber_id,tile_id,lat_bin,lon_bin,step_index,proposal_seed,mutation_type,accepted,accept_reason,current_score_before,candidate_score,accepted_score,best_score,delta_from_start,best_delta_from_start,step_delta,base_score,requested_radius,direct_genome_distance,edge_edits,threshold_edits,channel_edits,polarity_edits,current_edges,candidate_edges,behavior_distance,eval_ms,score_mode,accept_epsilon"
    )
    .expect("failed to write paratrooper_paths header");
}

fn main() {
    let cli = parse_args();
    create_dir_all(&cli.out).expect("failed to create output directory");

    let table = VcbpTable::from_packed(&cli.packed).expect("failed to load packed VCBP table");
    let corpus = std::fs::read(&cli.corpus).expect("failed to read corpus");
    let (pair_ids, hot_to_idx, n_classes) = build_corpus_pairs(&corpus, &table, 397);
    let bigram = build_pair_bigram(&pair_ids, &hot_to_idx, n_classes);
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

        if cli.mode == "paratrooper-climb" {
            let climb_path = cli.out.join("paratrooper_paths.csv");
            let mut climb_writer = BufWriter::new(
                File::create(&climb_path).expect("failed to create paratrooper_paths.csv"),
            );
            write_climb_header(&mut climb_writer);
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
                    let mut current_eval_net = current.clone();
                    let start_score = eval_score(
                        &cli.score_mode,
                        &mut current_eval_net,
                        &proj,
                        &table,
                        &pair_ids,
                        &hot_to_idx,
                        &bigram,
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
                    for step_idx in 0..cli.climb_steps {
                        let mutation_type =
                            cli.mutation_types[proposal_rng.gen_range(0..cli.mutation_types.len())];
                        let proposal_seed = proposal_rng.gen::<u64>();
                        let mut mutation_rng = StdRng::seed_from_u64(proposal_seed);
                        let mut candidate = current.clone();
                        let counts = apply_radius_mutation(
                            &mut candidate,
                            radius,
                            mutation_type,
                            &mut mutation_rng,
                        );
                        let candidate_coord = encode_coord(&candidate);
                        let direct_distance = coord_distance(&base_coord, &candidate_coord);
                        let eval_start = Instant::now();
                        let mut cand_eval_net = candidate.clone();
                        let candidate_score = eval_score(
                            &cli.score_mode,
                            &mut cand_eval_net,
                            &proj,
                            &table,
                            &pair_ids,
                            &hot_to_idx,
                            &bigram,
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
                        if candidate_score > best_score {
                            best_score = candidate_score;
                        }
                        let candidate_edges = candidate.edge_count();
                        let accepted_score = if accepted {
                            current = candidate;
                            current_score = candidate_score;
                            current_score
                        } else {
                            current_score
                        };
                        let current_edges = current.edge_count();

                        writeln!(
                            climb_writer,
                            "{},{},{},{:.17},{},{},{},{},{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{},{},{},{},{},{},{},{},{:.17},{:.6},{},{:.17}",
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
