//! Neuron Infer — load a frozen grower state.tsv and run inference.
//!
//! Pairs with `neuron_grower.rs`. Read-only: duplicates only the minimal
//! eval surface (Neuron, Net, StateHead, load_state, task_n_in) plus
//! one new helper — `predict_with_score`, which returns the raw AdaBoost
//! ensemble sum alongside the label.
//!
//! Usage:
//!   cargo run --release --example neuron_infer -p instnct-core -- \
//!     --state <path-to-state.tsv> --input "1 0 0 1 0 1 0 1 0" \
//!     [--input ...] [--task <name>] [--scores] [--format tsv|human]
//!
//! Exit codes: 0 ok, 1 bad CLI usage, 2 runtime/data mismatch.

use std::process::ExitCode;

// ─── Task shape (neuron_grower.rs:108-110) ─────────────
fn task_n_in(task: &str) -> usize {
    if task.starts_with("four_") { 4 } else { 9 }
}

// ─── Neuron + Net (neuron_grower.rs:153-189) ───────────
// + new: predict_with_score
#[derive(Clone)]
#[allow(dead_code)] // id, train_acc, val_acc are schema fields we parse but don't use at eval time
struct Neuron {
    id: usize,
    parents: Vec<usize>,
    tick: u32,
    weights: Vec<i8>,
    threshold: i32, // effective threshold: dot >= threshold
    alpha: f32,
    train_acc: f32,
    val_acc: f32,
}

impl Neuron {
    fn eval(&self, sigs: &[u8]) -> u8 {
        let mut d = 0i32;
        for (&w, &p) in self.weights.iter().zip(&self.parents) {
            d += (w as i32) * (sigs[p] as i32);
        }
        if d >= self.threshold { 1 } else { 0 }
    }
}

struct Net { neurons: Vec<Neuron>, n_in: usize, sig_ticks: Vec<u32> }

impl Net {
    fn new(n: usize) -> Self { Net { neurons: Vec::new(), n_in: n, sig_ticks: vec![0; n] } }
    fn eval_all(&self, inp: &[u8]) -> Vec<u8> {
        let mut s: Vec<u8> = inp.to_vec();
        for n in &self.neurons { s.push(n.eval(&s)); }
        s
    }
    fn add(&mut self, n: Neuron) { self.sig_ticks.push(n.tick); self.neurons.push(n); }

    /// NEW (inference-only): return (label, raw AdaBoost ensemble sum).
    /// Mirrors `Net::predict` from the grower, but exposes the pre-sign
    /// score so callers can inspect margin / confidence.
    fn predict_with_score(&self, inp: &[u8]) -> (u8, f32) {
        if self.neurons.is_empty() { return (0, 0.0); }
        let s = self.eval_all(inp);
        let score: f32 = self.neurons.iter().enumerate()
            .map(|(i, n)| n.alpha * if s[self.n_in + i] == 1 { 1.0 } else { -1.0 })
            .sum();
        let label = if score >= 0.0 { 1 } else { 0 };
        (label, score)
    }
}

// ─── State TSV loader (neuron_grower.rs:428, 447-496) ──
struct StateHead { task: String, data_seed: u64, noise: f32, n_per: usize, n_in: usize }

fn load_state(path: &str) -> Result<Option<(StateHead, Net)>, String> {
    let s = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("failed to read state {}: {}", path, e)),
    };
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
    Ok(Some((head, net)))
}

// ─── CLI ───────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq)]
enum OutFormat { Human, Tsv }

struct CliConfig {
    state: Option<String>,
    inputs: Vec<String>,
    task: Option<String>,
    scores: bool,
    format: OutFormat,
}

fn parse_args() -> Result<CliConfig, String> {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = CliConfig {
        state: None, inputs: Vec::new(), task: None, scores: false, format: OutFormat::Human,
    };
    let need = |i: usize, name: &str| -> Result<(), String> {
        if i >= args.len() { Err(format!("ERROR: {} requires a value", name)) } else { Ok(()) }
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--state"  => { i += 1; need(i, "--state <path>")?; cfg.state = Some(args[i].clone()); }
            "--input"  => { i += 1; need(i, "--input <bits>")?; cfg.inputs.push(args[i].clone()); }
            "--task"   => { i += 1; need(i, "--task <name>")?; cfg.task = Some(args[i].clone()); }
            "--scores" => { cfg.scores = true; }
            "--format" => {
                i += 1; need(i, "--format tsv|human")?;
                cfg.format = match args[i].as_str() {
                    "human" => OutFormat::Human,
                    "tsv"   => OutFormat::Tsv,
                    other   => return Err(format!("ERROR: --format must be 'tsv' or 'human', got '{}'", other)),
                };
            }
            "--help" | "-h" => return Err(
                "USAGE: neuron_infer --state <path> --input \"1 0 1 ...\" [--input ...] [--task <name>] [--scores] [--format tsv|human]".into()
            ),
            other => return Err(format!("ERROR: unknown argument '{}'", other)),
        }
        i += 1;
    }
    Ok(cfg)
}

fn parse_bits(raw: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for tok in raw.split(|c: char| c.is_whitespace() || c == ',') {
        if tok.is_empty() { continue; }
        match tok {
            "0" => out.push(0u8),
            "1" => out.push(1u8),
            x => return Err(format!("ERROR: input token '{}' is not 0 or 1", x)),
        }
    }
    Ok(out)
}

fn format_score(s: f32) -> String {
    if s >= 0.0 { format!("+{:.3}", s) } else { format!("{:.3}", s) }
}

// ─── Main ──────────────────────────────────────────────
fn main() -> ExitCode {
    let cfg = match parse_args() {
        Ok(c) => c,
        Err(msg) => { eprintln!("{}", msg); return ExitCode::from(1); }
    };

    let state_path = match cfg.state.as_deref() {
        Some(p) => p,
        None => { eprintln!("ERROR: --state <path> is required"); return ExitCode::from(1); }
    };
    if cfg.inputs.is_empty() {
        eprintln!("ERROR: at least one --input is required");
        return ExitCode::from(1);
    }

    // Parse bits up front so CLI errors (exit 1) beat state errors (exit 2).
    let mut parsed_inputs: Vec<Vec<u8>> = Vec::with_capacity(cfg.inputs.len());
    for raw in &cfg.inputs {
        match parse_bits(raw) {
            Ok(bits) => parsed_inputs.push(bits),
            Err(msg) => { eprintln!("{}", msg); return ExitCode::from(1); }
        }
    }

    // Load state (exit 2 on any missing/malformed case).
    let (head, net) = match load_state(state_path) {
        Ok(Some(pair)) => pair,
        Ok(None) => { eprintln!("ERROR: no state at {}", state_path); return ExitCode::from(2); }
        Err(msg) => { eprintln!("ERROR: {}", msg); return ExitCode::from(2); }
    };

    // Optional task guard (exit 2 on mismatch).
    if let Some(expected) = cfg.task.as_deref() {
        if expected != head.task {
            eprintln!("ERROR: state task={} != expected task={}", head.task, expected);
            return ExitCode::from(2);
        }
    }

    // Informational only: head.n_in vs task_n_in.
    let task_expected_n_in = task_n_in(&head.task);
    if head.n_in != task_expected_n_in {
        eprintln!(
            "WARN: state n_in={} does not match task_n_in({})={} — trusting HEAD",
            head.n_in, head.task, task_expected_n_in
        );
    }

    // Shape guard per input (exit 2 on mismatch).
    for (idx, bits) in parsed_inputs.iter().enumerate() {
        if bits.len() != head.n_in {
            eprintln!(
                "ERROR: input #{} has {} bits, task {} expects {}",
                idx, bits.len(), head.task, head.n_in
            );
            return ExitCode::from(2);
        }
    }

    if net.neurons.is_empty() {
        eprintln!("WARN: state at {} contains zero neurons — predictions will all be 0", state_path);
    }

    // Header.
    match cfg.format {
        OutFormat::Human => {
            println!("# neuron_infer");
            println!("#   state:   {}", state_path);
            println!("#   task:    {}", head.task);
            println!("#   neurons: {}", net.neurons.len());
            println!("#   n_in:    {}", head.n_in);
            println!("#   inputs:  {}", parsed_inputs.len());
            if cfg.scores {
                println!("#   data_seed: {}", head.data_seed);
                println!("#   noise:     {}", head.noise);
                println!("#   n_per:     {}", head.n_per);
            }
            println!();
        }
        OutFormat::Tsv => {
            let mut cols = vec!["idx", "input", "label"];
            if cfg.scores { cols.push("score"); }
            println!("{}", cols.join("\t"));
        }
    }

    // Per-input inference.
    for (idx, bits) in parsed_inputs.iter().enumerate() {
        let (label, score) = net.predict_with_score(bits);
        let bits_vec: Vec<String> = bits.iter().map(|b| b.to_string()).collect();
        match cfg.format {
            OutFormat::Human => if cfg.scores {
                println!("input=[{}] -> label={} score={}", bits_vec.join(" "), label, format_score(score));
            } else {
                println!("input=[{}] -> label={}", bits_vec.join(" "), label);
            },
            OutFormat::Tsv => if cfg.scores {
                println!("{}\t{}\t{}\t{:.6}", idx, bits_vec.join(","), label, score);
            } else {
                println!("{}\t{}\t{}", idx, bits_vec.join(","), label);
            },
        }
    }

    ExitCode::from(0)
}
