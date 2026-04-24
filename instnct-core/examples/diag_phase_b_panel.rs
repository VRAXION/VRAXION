//! Post-hoc Phase B checkpoint panel.
//!
//! Loads a run directory containing `final.ckpt` and `run_meta.json`, replays a
//! deterministic probe set, and writes `panel_summary.json`.

use instnct_core::{
    load_checkpoint, softmax, InitConfig, Int8Projection, Network, PropagationConfig, VcbpTable,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_CHARGE: i32 = 7;
const PROBE_COUNT: usize = 32;

#[derive(Deserialize)]
struct RunMeta {
    fixture: String,
    arm: String,
    run_id: String,
    seed: u64,
    #[serde(rename = "H")]
    h: usize,
    steps: usize,
    jackpot: usize,
    ticks: usize,
    input_scatter: bool,
    corpus: String,
    packed: String,
    checkpoint: String,
}

#[derive(Serialize)]
struct PanelSummary {
    fixture: String,
    arm: String,
    run_id: String,
    seed: u64,
    #[serde(rename = "H")]
    h: usize,
    steps: usize,
    jackpot: usize,
    ticks: usize,
    input_scatter: bool,
    checkpoint_step: usize,
    checkpoint_accuracy: f64,
    probe_count: usize,
    output_dim: usize,
    unique_predictions: usize,
    collision_rate: f64,
    f_active: f64,
    h_output_mean: f64,
    h_output_var: f64,
    stable_rank: f64,
    kernel_rank: usize,
    separation_sp: f64,
    dcor_io: f64,
    cka_linear: f64,
}

fn resolve_path(run_dir: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() || path.exists() {
        path
    } else {
        run_dir.join(path)
    }
}

fn build_pair_ids(corpus: &[u8]) -> Vec<u16> {
    let n_pairs = corpus.len() / 2;
    let mut pair_ids = Vec::with_capacity(n_pairs);
    for i in 0..n_pairs {
        pair_ids.push(VcbpTable::pair_id(corpus[i * 2], corpus[i * 2 + 1]));
    }
    pair_ids
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

fn centered_gram(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut gram = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            gram[i][j] = matrix[i].iter().zip(&matrix[j]).map(|(a, b)| a * b).sum();
        }
    }
    let row_means: Vec<f64> = gram
        .iter()
        .map(|row| row.iter().sum::<f64>() / n as f64)
        .collect();
    let col_means: Vec<f64> = (0..n)
        .map(|col| gram.iter().map(|row| row[col]).sum::<f64>() / n as f64)
        .collect();
    let grand = row_means.iter().sum::<f64>() / n as f64;
    for i in 0..n {
        for (j, col_mean) in col_means.iter().enumerate().take(n) {
            gram[i][j] = gram[i][j] - row_means[i] - col_mean + grand;
        }
    }
    gram
}

fn linear_cka(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }
    let k = centered_gram(&x[..n]);
    let l = centered_gram(&y[..n]);
    let mut numerator = 0.0;
    let mut k2 = 0.0;
    let mut l2 = 0.0;
    for i in 0..n {
        for j in 0..n {
            numerator += k[i][j] * l[i][j];
            k2 += k[i][j] * k[i][j];
            l2 += l[i][j] * l[i][j];
        }
    }
    let denom = (k2 * l2).sqrt();
    if denom <= 1e-12 {
        0.0
    } else {
        numerator / denom
    }
}

fn centered_distance_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            dist[i][j] = euclidean_distance(&matrix[i], &matrix[j]);
        }
    }
    let row_means: Vec<f64> = dist
        .iter()
        .map(|row| row.iter().sum::<f64>() / n as f64)
        .collect();
    let col_means: Vec<f64> = (0..n)
        .map(|col| dist.iter().map(|row| row[col]).sum::<f64>() / n as f64)
        .collect();
    let grand = row_means.iter().sum::<f64>() / n as f64;
    for i in 0..n {
        for (j, col_mean) in col_means.iter().enumerate().take(n) {
            dist[i][j] = dist[i][j] - row_means[i] - col_mean + grand;
        }
    }
    dist
}

fn distance_correlation(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let a = centered_distance_matrix(&x[..n]);
    let b = centered_distance_matrix(&y[..n]);
    let mut dcov2 = 0.0;
    let mut dvarx = 0.0;
    let mut dvary = 0.0;
    for i in 0..n {
        for j in 0..n {
            dcov2 += a[i][j] * b[i][j];
            dvarx += a[i][j] * a[i][j];
            dvary += b[i][j] * b[i][j];
        }
    }
    let norm = (n * n) as f64;
    dcov2 /= norm;
    dvarx /= norm;
    dvary /= norm;
    if dvarx <= 1e-12 || dvary <= 1e-12 {
        0.0
    } else {
        (dcov2.max(0.0) / (dvarx * dvary).sqrt()).sqrt()
    }
}

fn entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

fn run_probe(
    mut net: Network,
    proj: Int8Projection,
    meta: RunMeta,
    checkpoint_step: usize,
    checkpoint_accuracy: f64,
    table: &VcbpTable,
    pair_ids: &[u16],
    propagation: &PropagationConfig,
) -> PanelSummary {
    let output_start = meta.h - (meta.h as f64 / 1.618_033_988_749_895).round() as usize;
    let input_end = (meta.h as f64 / 1.618_033_988_749_895).round() as usize;
    let probe_count = PROBE_COUNT.min(pair_ids.len().max(1));
    let mut input_matrix = Vec::with_capacity(probe_count);
    let mut output_matrix = Vec::with_capacity(probe_count);
    let mut predictions = Vec::with_capacity(probe_count);
    let mut entropy_values = Vec::with_capacity(probe_count);
    let mut unique_inputs = HashSet::new();

    for k in 0..probe_count {
        let pid = pair_ids[(k * pair_ids.len()) / probe_count];
        unique_inputs.insert(pid);
        net.reset();
        let emb = table.embed_id(pid);
        let mut input = vec![0i32; meta.h];
        quantize_embedding_to_input(table, emb, &mut input, input_end, meta.input_scatter);
        net.propagate(&input, propagation).unwrap();
        let charges_u8 = net.charge_vec(output_start..meta.h);
        let charges: Vec<f64> = charges_u8.iter().map(|&v| v as f64).collect();
        let scores = proj.raw_scores(&charges_u8);
        let probs = softmax(&scores);
        entropy_values.push(entropy(&probs));
        predictions.push(proj.predict(&charges_u8));
        input_matrix.push(emb.iter().map(|&v| v as f64).collect::<Vec<_>>());
        output_matrix.push(charges);
    }

    let unique_predictions = predictions.iter().copied().collect::<HashSet<_>>().len();
    let collision_rate = if unique_inputs.is_empty() {
        0.0
    } else {
        unique_predictions as f64 / unique_inputs.len() as f64
    };
    let (h_output_mean, h_output_var) = output_entropy_stats(&entropy_values);

    PanelSummary {
        fixture: meta.fixture,
        arm: meta.arm,
        run_id: meta.run_id,
        seed: meta.seed,
        h: meta.h,
        steps: meta.steps,
        jackpot: meta.jackpot,
        ticks: meta.ticks,
        input_scatter: meta.input_scatter,
        checkpoint_step,
        checkpoint_accuracy,
        probe_count,
        output_dim: output_matrix.first().map_or(0, Vec::len),
        unique_predictions,
        collision_rate,
        f_active: f_active(&output_matrix),
        h_output_mean,
        h_output_var,
        stable_rank: stable_rank(&output_matrix),
        kernel_rank: numerical_rank(&output_matrix, 1e-6),
        separation_sp: separation_sp(&input_matrix, &output_matrix),
        dcor_io: distance_correlation(&input_matrix, &output_matrix),
        cka_linear: linear_cka(&input_matrix, &output_matrix),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: diag_phase_b_panel <run_dir>");
        std::process::exit(1);
    }
    let run_dir = PathBuf::from(&args[1]);
    let meta_path = run_dir.join("run_meta.json");
    let meta: RunMeta =
        serde_json::from_slice(&fs::read(&meta_path).expect("failed to read run_meta.json"))
            .expect("failed to parse run_meta.json");
    let checkpoint_path = resolve_path(&run_dir, &meta.checkpoint);
    let corpus_path = resolve_path(&run_dir, &meta.corpus);
    let packed_path = resolve_path(&run_dir, &meta.packed);

    let (net, proj, checkpoint_meta) =
        load_checkpoint(&checkpoint_path).expect("failed to load checkpoint");
    let mut init = InitConfig::phi(meta.h);
    init.propagation.ticks_per_token = meta.ticks;
    let table = VcbpTable::from_packed(&packed_path).expect("failed to load packed table");
    let corpus = fs::read(&corpus_path).expect("failed to read corpus");
    let pair_ids = build_pair_ids(&corpus);
    let summary = run_probe(
        net,
        proj,
        meta,
        checkpoint_meta.step,
        checkpoint_meta.accuracy,
        &table,
        &pair_ids,
        &init.propagation,
    );

    let out_path = run_dir.join("panel_summary.json");
    fs::write(
        &out_path,
        serde_json::to_string_pretty(&summary).expect("failed to serialize panel summary"),
    )
    .expect("failed to write panel_summary.json");
    println!(
        "PANEL {{\"run_id\":\"{}\",\"probe_count\":{},\"unique_predictions\":{},\"panel\":\"{}\"}}",
        summary.run_id,
        summary.probe_count,
        summary.unique_predictions,
        out_path.display()
    );
}
