//! grid3_prune_analyze — leave-one-out pruning analysis for a trained
//! `grid3_curriculum` grower state.
//!
//! Loads a `state.tsv` produced by `grid3_curriculum`, replays the exact
//! validation set it was trained against (same task / seed / noise / n_per),
//! and measures how much each neuron contributes to the ensemble vote.
//!
//! Pruning model:
//! Removal is implemented by *zeroing the neuron's alpha* in the AdaBoost
//! weighted-sign predictor. This kills the neuron's vote without reindexing
//! downstream parents (which would be required for a structural remove and
//! would break any hidden neuron whose output is consumed by later neurons).
//! For neurons that are also used as parents by later neurons, the table
//! flags `deps=N` in verbose mode so the reader knows the output still feeds
//! downstream features even though its vote was removed.
//!
//! Rule:
//!   delta = baseline_val_acc - pruned_val_acc
//!     delta > 1.0            -> keep       (meaningful contribution)
//!     0 < delta <= 1.0       -> marginal
//!     delta <= 0             -> redundant  (safe to drop, or improves val)
//!
//! Iterative greedy prune: repeatedly drop the single most-redundant neuron
//! (largest negative delta, with largest down-vote count as tiebreak) and
//! re-measure until every surviving neuron has delta > 0.
//!
//! Usage:
//!     cargo run --release --example grid3_prune_analyze -p instnct-core -- \
//!         --state target/grid3_smoke_parity/state.tsv
//!     cargo run --release --example grid3_prune_analyze -p instnct-core -- \
//!         --state target/grid3_smoke/state.tsv --verbose --json out.json
//!
//! Read-only: never writes back to state.tsv. Zero new dependencies, stdlib
//! only, minimal code duplicated verbatim from grid3_curriculum.rs.

use std::io::Write as IoWrite;

// ══════════════════════════════════════════════════════
// CLI
// ══════════════════════════════════════════════════════
struct Config {
    state: String,
    verbose: bool,
    json: Option<String>,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut state: Option<String> = None;
    let mut verbose = false;
    let mut json: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--state" => { i += 1; state = Some(args[i].clone()); }
            "--verbose" => { verbose = true; }
            "--json" => { i += 1; json = Some(args[i].clone()); }
            "--help" | "-h" => {
                eprintln!("usage: grid3_prune_analyze --state <path> [--verbose] [--json <path>]");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    let state = match state {
        Some(s) => s,
        None => {
            eprintln!("ERROR: --state <path> is required");
            std::process::exit(2);
        }
    };

    Config { state, verbose, json }
}

// ══════════════════════════════════════════════════════
// PRNG  (duplicated verbatim from grid3_curriculum.rs)
// ══════════════════════════════════════════════════════
struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn bool_p(&mut self, p: f32) -> bool { self.f32() < p }
}

// ══════════════════════════════════════════════════════
// FONT + DATA  (Font9 only — duplicated verbatim)
// ══════════════════════════════════════════════════════
const FONT: [[u8; 9]; 10] = [
    [1,1,1, 1,0,1, 1,1,1], [0,1,0, 0,1,0, 0,1,0],
    [1,1,0, 0,1,0, 0,1,1], [1,1,0, 0,1,0, 1,1,0],
    [1,0,1, 1,1,1, 0,0,1], [0,1,1, 0,1,0, 1,1,0],
    [1,0,0, 1,1,0, 1,1,0], [1,1,1, 0,0,1, 0,0,1],
    [1,1,1, 1,1,1, 1,1,1], [1,1,1, 1,1,1, 0,1,1],
];

struct Data { train: Vec<(Vec<u8>, u8)>, val: Vec<(Vec<u8>, u8)>, test: Vec<(Vec<u8>, u8)> }

fn gen_data(
    label_fn: &dyn Fn(usize, &[u8]) -> Option<u8>,
    noise: f32,
    n_per: usize,
    seed: u64,
) -> Data {
    let mut rng = Rng::new(seed);
    let (mut tr, mut va, mut te) = (Vec::new(), Vec::new(), Vec::new());
    for d in 0..10 { for i in 0..n_per {
        let mut px = FONT[d].to_vec();
        for p in px.iter_mut() { if rng.bool_p(noise) { *p = 1 - *p; } }
        if let Some(label) = label_fn(d, &px) {
            match i % 5 { 0 => va.push((px, label)), 1 => te.push((px, label)), _ => tr.push((px, label)) }
        }
    }}
    Data { train: tr, val: va, test: te }
}

// ══════════════════════════════════════════════════════
// LABEL FN DISPATCH  (10 grid3_* tasks — duplicated verbatim)
// ══════════════════════════════════════════════════════
fn task_n_in(task: &str) -> usize {
    if task.starts_with("grid3_") { 9 }
    else {
        eprintln!("ERROR: unknown task '{}' (expected grid3_*)", task);
        std::process::exit(2);
    }
}

fn label_fn_for(task: &str) -> Box<dyn Fn(usize, &[u8]) -> Option<u8>> {
    match task {
        "grid3_horizontal_line" => Box::new(|_, px| {
            let r0 = px[0] == 1 && px[1] == 1 && px[2] == 1;
            let r1 = px[3] == 1 && px[4] == 1 && px[5] == 1;
            let r2 = px[6] == 1 && px[7] == 1 && px[8] == 1;
            Some(if r0 || r1 || r2 { 1 } else { 0 })
        }),
        "grid3_vertical_line" => Box::new(|_, px| {
            let c0 = px[0] == 1 && px[3] == 1 && px[6] == 1;
            let c1 = px[1] == 1 && px[4] == 1 && px[7] == 1;
            let c2 = px[2] == 1 && px[5] == 1 && px[8] == 1;
            Some(if c0 || c1 || c2 { 1 } else { 0 })
        }),
        "grid3_diagonal" => Box::new(|_, px| {
            Some(if px[0] == 1 && px[4] == 1 && px[8] == 1 { 1 } else { 0 })
        }),
        "grid3_center" => Box::new(|_, px| Some(px[4])),
        "grid3_corner" => Box::new(|_, px| {
            let c = px[0] == 1 || px[2] == 1 || px[6] == 1 || px[8] == 1;
            Some(if c { 1 } else { 0 })
        }),
        "grid3_diag_xor" => Box::new(|_, px| {
            let parity = (px[0] ^ px[4] ^ px[8]) & 1;
            Some(parity)
        }),
        "grid3_full_parity" => Box::new(|_, px| {
            let parity =
                (px[0] ^ px[1] ^ px[2] ^
                 px[3] ^ px[4] ^ px[5] ^
                 px[6] ^ px[7] ^ px[8]) & 1;
            Some(parity)
        }),
        "grid3_majority" => Box::new(|_, px| {
            let s: usize = px.iter().map(|&v| v as usize).sum();
            Some(if s >= 5 { 1 } else { 0 })
        }),
        "grid3_symmetry_h" => Box::new(|_, px| {
            let sym = px[0] == px[2] && px[3] == px[5] && px[6] == px[8];
            Some(if sym { 1 } else { 0 })
        }),
        "grid3_top_heavy" => Box::new(|_, px| {
            let top: usize = (px[0] as usize) + (px[1] as usize) + (px[2] as usize);
            let bot: usize = (px[6] as usize) + (px[7] as usize) + (px[8] as usize);
            Some(if top > bot { 1 } else { 0 })
        }),
        other => {
            eprintln!("ERROR: unknown task '{}' (expected one of the 10 grid3_*)", other);
            std::process::exit(2);
        }
    }
}

// ══════════════════════════════════════════════════════
// NEURON + NET  (duplicated verbatim)
// ══════════════════════════════════════════════════════
#[derive(Clone)]
struct Neuron {
    id: usize, parents: Vec<usize>,
    #[allow(dead_code)] tick: u32,
    weights: Vec<i8>, threshold: i32,
    alpha: f32, train_acc: f32, val_acc: f32,
}

impl Neuron {
    fn eval(&self, sigs: &[u8]) -> u8 {
        let mut d = 0i32;
        for (&w, &p) in self.weights.iter().zip(&self.parents) { d += (w as i32) * (sigs[p] as i32); }
        if d >= self.threshold { 1 } else { 0 }
    }
}

#[derive(Clone)]
struct Net { neurons: Vec<Neuron>, n_in: usize }
impl Net {
    fn new(n: usize) -> Self { Net { neurons: Vec::new(), n_in: n } }
    fn eval_all(&self, inp: &[u8]) -> Vec<u8> {
        let mut s: Vec<u8> = inp.to_vec();
        for n in &self.neurons { s.push(n.eval(&s)); }
        s
    }
    fn predict(&self, inp: &[u8]) -> u8 {
        if self.neurons.is_empty() { return 0; }
        let s = self.eval_all(inp);
        let score: f32 = self.neurons.iter().enumerate()
            .map(|(i, n)| n.alpha * if s[self.n_in + i] == 1 { 1.0 } else { -1.0 }).sum();
        if score >= 0.0 { 1 } else { 0 }
    }
    fn accuracy(&self, data: &[(Vec<u8>, u8)]) -> f32 {
        if self.neurons.is_empty() { return 50.0; }
        data.iter().filter(|(x, y)| self.predict(x) == *y).count() as f32 / data.len() as f32 * 100.0
    }
    fn add(&mut self, n: Neuron) { self.neurons.push(n); }
}

// ══════════════════════════════════════════════════════
// STATE LOADER  (duplicated verbatim, but returns a read-only Net)
// ══════════════════════════════════════════════════════
struct StateHead { task: String, data_seed: u64, noise: f32, n_per: usize, n_in: usize }

fn load_state(path: &str) -> Result<(StateHead, Net), String> {
    let s = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read state {}: {}", path, e))?;
    let mut lines = s.lines();
    let head_line = lines.next().ok_or_else(|| format!("empty state file: {}", path))?;
    let h: Vec<&str> = head_line.split('\t').collect();
    if h.len() < 6 || h[0] != "HEAD" {
        return Err(format!("invalid state header in {}", path));
    }
    let head = StateHead {
        task: h[1].to_string(),
        data_seed: h[2].parse().map_err(|_| format!("invalid data_seed in {}", path))?,
        noise: h[3].parse().map_err(|_| format!("invalid noise in {}", path))?,
        n_per: h[4].parse().map_err(|_| format!("invalid n_per in {}", path))?,
        n_in: h[5].parse().map_err(|_| format!("invalid n_in in {}", path))?,
    };
    let mut net = Net::new(head.n_in);
    for line in lines {
        if line.is_empty() { continue; }
        let c: Vec<&str> = line.split('\t').collect();
        if c[0] != "N" { continue; }
        if c.len() != 9 {
            return Err(format!(
                "incompatible state schema in {} (expected bias-free 9-column neuron rows, got {})",
                path, c.len()
            ));
        }
        let parents: Vec<usize> = if c[2].is_empty() { Vec::new() }
            else { c[2].split(',').map(|v| v.parse()).collect::<Result<Vec<_>, _>>()
                .map_err(|_| format!("invalid parent list in {}", path))? };
        let weights: Vec<i8> = if c[4].is_empty() { Vec::new() }
            else { c[4].split(',').map(|v| v.parse()).collect::<Result<Vec<_>, _>>()
                .map_err(|_| format!("invalid weight list in {}", path))? };
        let n = Neuron {
            id: c[1].parse().map_err(|_| format!("invalid neuron id in {}", path))?,
            parents,
            tick: c[3].parse().map_err(|_| format!("invalid tick in {}", path))?,
            weights,
            threshold: c[5].parse().map_err(|_| format!("invalid threshold in {}", path))?,
            alpha: c[6].parse().map_err(|_| format!("invalid alpha in {}", path))?,
            train_acc: c[7].parse().map_err(|_| format!("invalid train_acc in {}", path))?,
            val_acc: c[8].parse().map_err(|_| format!("invalid val_acc in {}", path))?,
        };
        net.add(n);
    }
    Ok((head, net))
}

// ══════════════════════════════════════════════════════
// PRUNE CORE
// ══════════════════════════════════════════════════════
/// Evaluate ensemble accuracy with a subset of neurons' alphas zeroed out.
/// The network graph is *not* restructured — hidden neurons downstream can
/// still consume the "removed" neuron's output as a parent feature.
fn accuracy_with_mask(net: &Net, val: &[(Vec<u8>, u8)], mask: &[bool]) -> f32 {
    if mask.iter().all(|&m| m) { return 50.0; } // all-pruned -> empty vote
    let mut correct: usize = 0;
    for (x, y) in val {
        let s = net.eval_all(x);
        let mut any_alive = false;
        let score: f32 = net.neurons.iter().enumerate()
            .map(|(i, n)| {
                if mask[i] { 0.0 }
                else {
                    any_alive = true;
                    n.alpha * if s[net.n_in + i] == 1 { 1.0 } else { -1.0 }
                }
            }).sum();
        let pred: u8 = if !any_alive { 0 } else if score >= 0.0 { 1 } else { 0 };
        if pred == *y { correct += 1; }
    }
    correct as f32 / val.len() as f32 * 100.0
}

/// Count how many later neurons list `ni` as a parent. If count > 0, removing
/// neuron `ni` structurally would break their graph; alpha-zeroing is the
/// only safe way to simulate its absence from the ensemble vote.
fn downstream_dependents(net: &Net, ni: usize) -> usize {
    let sig_idx = net.n_in + ni;
    let mut count = 0;
    for later in net.neurons.iter().skip(ni + 1) {
        if later.parents.iter().any(|&p| p == sig_idx) { count += 1; }
    }
    count
}

#[derive(Clone)]
struct PruneRow {
    idx: usize,
    alpha: f32,
    parents: Vec<usize>,
    weights: Vec<i8>,
    threshold: i32,
    train_acc: f32,
    val_acc_standalone: f32,
    val_if_removed: f32,
    delta: f32,
    verdict: &'static str,
    deps: usize,
}

fn verdict_for(delta: f32) -> &'static str {
    if delta > 1.0 { "keep" }
    else if delta > 0.0 { "marginal" }
    else { "redundant" }
}

fn sig_name(idx: usize, n_in: usize) -> String {
    if idx < n_in { format!("x{}", idx) } else { format!("N{}", idx - n_in) }
}

fn format_parents(parents: &[usize], n_in: usize) -> String {
    let names: Vec<String> = parents.iter().map(|&p| sig_name(p, n_in)).collect();
    format!("[{}]", names.join(","))
}

fn format_weights(weights: &[i8]) -> String {
    let parts: Vec<String> = weights.iter().map(|v| v.to_string()).collect();
    format!("[{}]", parts.join(","))
}

fn format_weights_compact(weights: &[i8]) -> String {
    weights.iter().map(|&v| match v { 1 => '+', -1 => '-', _ => '0' }).collect::<String>()
}

// ══════════════════════════════════════════════════════
// OUTPUT
// ══════════════════════════════════════════════════════
fn print_markdown_table(rows: &[PruneRow], baseline: f32, n_in: usize, verbose: bool) {
    println!();
    println!("## Leave-one-out pruning (vote-removal via alpha=0)");
    println!();
    println!("baseline_val_acc = {:.2}%", baseline);
    println!();
    if verbose {
        println!("| N   | alpha    | parents                        | weights                  | thr | train% | standalone_val% | val_if_removed | delta   | deps | verdict    |");
        println!("|-----|----------|--------------------------------|--------------------------|----:|-------:|----------------:|---------------:|--------:|-----:|------------|");
        for r in rows {
            println!(
                "| N{:<3}| {:.4}  | {:<30}| {:<24}| {:>3} | {:>6.1}| {:>15.1}| {:>14.2}| {:>+7.2}| {:>4} | {:<10} |",
                r.idx, r.alpha,
                format_parents(&r.parents, n_in),
                format_weights(&r.weights),
                r.threshold,
                r.train_acc,
                r.val_acc_standalone,
                r.val_if_removed,
                r.delta,
                r.deps,
                r.verdict,
            );
        }
    } else {
        println!("| N   | alpha    | parents                        | weights  | thr | val_if_removed | delta   | verdict    |");
        println!("|-----|----------|--------------------------------|----------|----:|---------------:|--------:|------------|");
        for r in rows {
            println!(
                "| N{:<3}| {:.4}  | {:<30}| {:<8} | {:>3} | {:>14.2}| {:>+7.2}| {:<10} |",
                r.idx, r.alpha,
                format_parents(&r.parents, n_in),
                format_weights_compact(&r.weights),
                r.threshold,
                r.val_if_removed,
                r.delta,
                r.verdict,
            );
        }
    }
    println!();
}

fn print_summary(rows: &[PruneRow], baseline: f32) {
    let redundant = rows.iter().filter(|r| r.verdict == "redundant").count();
    let marginal = rows.iter().filter(|r| r.verdict == "marginal").count();
    let keep = rows.iter().filter(|r| r.verdict == "keep").count();
    println!(
        "SUMMARY: baseline_val_acc={:.2}%, redundant_count={}, marginal_count={}, keep_count={}",
        baseline, redundant, marginal, keep
    );
}

fn write_json(
    path: &str,
    task: &str,
    baseline: f32,
    rows: &[PruneRow],
    greedy_kept: &[usize],
    greedy_val: f32,
) -> std::io::Result<()> {
    if let Some(p) = std::path::Path::new(path).parent() { std::fs::create_dir_all(p)?; }
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "{{")?;
    writeln!(f, "  \"task\": \"{}\",", task)?;
    writeln!(f, "  \"baseline_val_acc\": {:.4},", baseline)?;
    writeln!(f, "  \"n_neurons\": {},", rows.len())?;
    writeln!(f, "  \"rows\": [")?;
    for (i, r) in rows.iter().enumerate() {
        let ps: Vec<String> = r.parents.iter().map(|v| v.to_string()).collect();
        let ws: Vec<String> = r.weights.iter().map(|v| v.to_string()).collect();
        let comma = if i + 1 < rows.len() { "," } else { "" };
        writeln!(f,
            "    {{\"id\":{},\"alpha\":{:.6},\"parents\":[{}],\"weights\":[{}],\"threshold\":{},\"train_acc\":{:.2},\"val_acc_standalone\":{:.2},\"val_if_removed\":{:.4},\"delta\":{:.4},\"deps\":{},\"verdict\":\"{}\"}}{}",
            r.idx, r.alpha, ps.join(","), ws.join(","), r.threshold,
            r.train_acc, r.val_acc_standalone, r.val_if_removed, r.delta, r.deps, r.verdict, comma
        )?;
    }
    writeln!(f, "  ],")?;
    let kept_str: Vec<String> = greedy_kept.iter().map(|v| v.to_string()).collect();
    writeln!(f, "  \"greedy_kept\": [{}],", kept_str.join(","))?;
    writeln!(f, "  \"greedy_kept_count\": {},", greedy_kept.len())?;
    writeln!(f, "  \"greedy_final_val_acc\": {:.4}", greedy_val)?;
    writeln!(f, "}}")?;
    Ok(())
}

// ══════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════
fn main() {
    let cfg = parse_args();

    // Load the trained state
    let (head, net) = match load_state(&cfg.state) {
        Ok((h, n)) => (h, n),
        Err(e) => {
            eprintln!("ERROR: {}", e);
            std::process::exit(2);
        }
    };

    // Sanity: task must be a grid3_* task
    let n_in_expected = task_n_in(&head.task);
    if n_in_expected != head.n_in {
        eprintln!("ERROR: state n_in={} != task {} expected n_in={}", head.n_in, head.task, n_in_expected);
        std::process::exit(2);
    }

    println!("===========================================================");
    println!("  grid3_prune_analyze — {}", cfg.state);
    println!("  task={} data_seed={} noise={} n_per={} n_in={}",
        head.task, head.data_seed, head.noise, head.n_per, head.n_in);
    println!("  loaded {} neurons", net.neurons.len());
    println!("===========================================================");

    // Regenerate the validation set deterministically (same split as gen_data)
    let label_fn = label_fn_for(&head.task);
    let data = gen_data(label_fn.as_ref(), head.noise, head.n_per, head.data_seed);
    println!("  Data: {} train / {} val / {} test", data.train.len(), data.val.len(), data.test.len());

    // Baseline val_acc via AdaBoost predict
    let baseline_val = net.accuracy(&data.val);
    println!("  Baseline val_acc via AdaBoost predict: {:.2}%", baseline_val);
    println!();

    if net.neurons.is_empty() {
        println!("SUMMARY: baseline_val_acc={:.2}%, redundant_count=0, marginal_count=0, keep_count=0 (no neurons)", baseline_val);
        return;
    }

    // Leave-one-out: for each neuron, mask its alpha and re-measure
    let nn = net.neurons.len();
    let mut rows: Vec<PruneRow> = Vec::with_capacity(nn);
    for i in 0..nn {
        let mut mask = vec![false; nn];
        mask[i] = true;
        let val_if_removed = accuracy_with_mask(&net, &data.val, &mask);
        let delta = baseline_val - val_if_removed;
        let verdict = verdict_for(delta);
        let deps = downstream_dependents(&net, i);
        let n = &net.neurons[i];
        rows.push(PruneRow {
            idx: n.id,
            alpha: n.alpha,
            parents: n.parents.clone(),
            weights: n.weights.clone(),
            threshold: n.threshold,
            train_acc: n.train_acc,
            val_acc_standalone: n.val_acc,
            val_if_removed,
            delta,
            verdict,
            deps,
        });
    }

    // Sort by id (keep native order for the main table; greedy prune works
    // on a separate indexing below)
    print_markdown_table(&rows, baseline_val, net.n_in, cfg.verbose);
    print_summary(&rows, baseline_val);

    // Also sort-by-delta printout (most redundant first) for readability
    let mut sorted = rows.clone();
    sorted.sort_by(|a, b| a.delta.partial_cmp(&b.delta).unwrap());
    println!();
    println!("## Ranked by delta (most redundant first)");
    println!();
    println!("| rank | N   | alpha    | delta   | verdict    | deps |");
    println!("|-----:|-----|---------:|--------:|------------|-----:|");
    for (rank, r) in sorted.iter().enumerate() {
        println!("| {:>4} | N{:<3}| {:.4}  | {:>+7.2}| {:<10} | {:>4} |",
            rank + 1, r.idx, r.alpha, r.delta, r.verdict, r.deps);
    }
    println!();

    // Iterative greedy prune: repeatedly drop the single worst neuron until
    // every survivor has delta > 0. At each iteration, the current subset's
    // baseline shifts, so we recompute.
    println!("## Iterative greedy prune");
    println!();
    let mut mask = vec![false; nn];
    let mut iter = 0usize;
    let mut current_val = baseline_val;
    loop {
        // Measure each still-alive neuron's delta under the current mask
        let alive: Vec<usize> = (0..nn).filter(|i| !mask[*i]).collect();
        if alive.is_empty() { break; }
        let mut worst: Option<(usize, f32, f32)> = None; // (idx, delta, val_if_removed)
        for &i in &alive {
            let mut probe = mask.clone();
            probe[i] = true;
            let v = accuracy_with_mask(&net, &data.val, &probe);
            let delta = current_val - v;
            match worst {
                None => { worst = Some((i, delta, v)); }
                Some((_, wd, _)) if delta < wd => { worst = Some((i, delta, v)); }
                _ => {}
            }
        }
        let (wi, wd, wv) = worst.unwrap();
        if wd > 0.0 {
            // every remaining neuron is contributing — stop
            println!("  iter {}: all {} survivors have delta > 0 (min delta = +{:.2}), stopping.", iter, alive.len(), wd);
            break;
        }
        // Drop it
        mask[wi] = true;
        let kept = alive.len() - 1;
        println!("  iter {}: drop N{} (delta={:+.2}, alpha={:.4}) -> val={:.2}% kept={}",
            iter, wi, wd, net.neurons[wi].alpha, wv, kept);
        current_val = wv;
        iter += 1;
        if kept == 0 { break; }
    }
    let greedy_kept: Vec<usize> = (0..nn).filter(|i| !mask[*i]).collect();
    let greedy_val = if greedy_kept.is_empty() { 50.0 } else {
        accuracy_with_mask(&net, &data.val, &mask)
    };
    let kept_names: Vec<String> = greedy_kept.iter().map(|i| format!("N{}", i)).collect();
    println!();
    println!("  GREEDY RESULT: kept {} of {} neurons — {{ {} }}",
        greedy_kept.len(), nn, kept_names.join(", "));
    println!("  GREEDY final val_acc = {:.2}% (baseline was {:.2}%, delta_full = {:+.2})",
        greedy_val, baseline_val, greedy_val - baseline_val);
    println!();

    // Optional JSON output
    if let Some(path) = cfg.json.as_ref() {
        match write_json(path, &head.task, baseline_val, &rows, &greedy_kept, greedy_val) {
            Ok(_) => println!("  wrote {}", path),
            Err(e) => { eprintln!("ERROR: failed to write JSON to {}: {}", path, e); std::process::exit(1); }
        }
    }
}
