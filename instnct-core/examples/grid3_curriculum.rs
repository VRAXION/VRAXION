//! grid3_curriculum — 3x3 pattern recognition curriculum builder
//! (duplicates the neuron_grower.rs pipeline per the grid3 design spec; do
//!  not edit neuron_grower.rs — this example is self-contained)
//!
//! Builds a single bias-free ternary threshold network for one task in the
//! 10-task `grid3_*` curriculum on the Font9 domain (9-bit row-major 3x3).
//!
//! Run: cargo run --release --example grid3_curriculum -p instnct-core -- \
//!        --task grid3_center --search-seed 1 --out-dir target/grid3/center_1
//!
//! In addition to the grower's `state.tsv` / `checkpoints/*.json`, this binary
//! writes a `trace.json` summarising every accepted neuron — consumed by the
//! Wave 5 harness and the brain_replay viewer.

use std::collections::HashSet;
use std::io::Write as IoWrite;
use std::time::Instant;

// ══════════════════════════════════════════════════════
// CLI
// ══════════════════════════════════════════════════════
struct Config {
    task: String,
    search_seed: u64,
    out_dir: String,
    data_seed: u64,
    max_neurons: usize,
    max_fan: usize,
    n_proposals: usize,
    stall_limit: usize,
    noise: f32,
    n_per: usize,
    scout_top: usize,
    pair_top: usize,
    probe_epochs: usize,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut task: Option<String> = None;
    let mut search_seed: Option<u64> = None;
    let mut out_dir: Option<String> = None;
    let mut data_seed: u64 = 42;
    let mut max_neurons: usize = 32;
    let mut max_fan: usize = 6;
    let mut n_proposals: usize = 20;
    let mut stall_limit: usize = 16;
    let mut noise: f32 = 0.10;
    let mut n_per: usize = 200;
    let mut scout_top: usize = 12;
    let mut pair_top: usize = 8;
    let mut probe_epochs: usize = 400;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--task" => { i += 1; task = Some(args[i].clone()); }
            "--search-seed" => { i += 1; search_seed = Some(args[i].parse().unwrap_or_else(|_| { eprintln!("ERROR: --search-seed must be a non-negative integer"); std::process::exit(1); })); }
            "--out-dir" => { i += 1; out_dir = Some(args[i].clone()); }
            "--data-seed" => { i += 1; data_seed = args[i].parse().unwrap_or(42); }
            "--max-neurons" => { i += 1; max_neurons = args[i].parse().unwrap_or(32); }
            "--max-fan" => { i += 1; max_fan = args[i].parse().unwrap_or(6); }
            "--proposals" => { i += 1; n_proposals = args[i].parse().unwrap_or(20); }
            "--stall" => { i += 1; stall_limit = args[i].parse().unwrap_or(16); }
            "--noise" => { i += 1; noise = args[i].parse().unwrap_or(0.10); }
            "--n-per" => { i += 1; n_per = args[i].parse().unwrap_or(200); }
            "--scout-top" => { i += 1; scout_top = args[i].parse().unwrap_or(12); }
            "--pair-top" => { i += 1; pair_top = args[i].parse().unwrap_or(8); }
            "--probe-epochs" => { i += 1; probe_epochs = args[i].parse().unwrap_or(400); }
            _ => {}
        }
        i += 1;
    }

    let task = match task {
        Some(t) => t,
        None => { eprintln!("ERROR: --task is required"); std::process::exit(2); }
    };
    let search_seed = match search_seed {
        Some(s) => s,
        None => { eprintln!("ERROR: --search-seed is required"); std::process::exit(2); }
    };
    let out_dir = match out_dir {
        Some(o) => o,
        None => { eprintln!("ERROR: --out-dir is required"); std::process::exit(2); }
    };

    Config {
        task,
        search_seed,
        out_dir,
        data_seed,
        max_neurons,
        max_fan,
        n_proposals,
        stall_limit,
        noise,
        n_per,
        scout_top,
        pair_top,
        probe_epochs,
    }
}

// ══════════════════════════════════════════════════════
// PRNG + SIGMOID  (duplicated verbatim from neuron_grower.rs)
// ══════════════════════════════════════════════════════
struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }
    fn bool_p(&mut self, p: f32) -> bool { self.f32() < p }
    fn pick(&mut self, n: usize) -> usize { self.next() as usize % n }
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ══════════════════════════════════════════════════════
// FONT + DATA  (Font9 only — grid3_* does not use Bits4)
// ══════════════════════════════════════════════════════
const FONT: [[u8; 9]; 10] = [
    [1,1,1, 1,0,1, 1,1,1], [0,1,0, 0,1,0, 0,1,0],
    [1,1,0, 0,1,0, 0,1,1], [1,1,0, 0,1,0, 1,1,0],
    [1,0,1, 1,1,1, 0,0,1], [0,1,1, 0,1,0, 1,1,0],
    [1,0,0, 1,1,0, 1,1,0], [1,1,1, 0,0,1, 0,0,1],
    [1,1,1, 1,1,1, 1,1,1], [1,1,1, 1,1,1, 0,1,1],
];

fn task_n_in(task: &str) -> usize {
    if task.starts_with("grid3_") {
        9
    } else {
        eprintln!("ERROR: unknown task '{}' (expected grid3_*)", task);
        std::process::exit(2);
    }
}

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
// NEURON + NET  (duplicated verbatim from neuron_grower.rs)
// ══════════════════════════════════════════════════════
#[derive(Clone)]
struct Neuron {
    id: usize, parents: Vec<usize>, tick: u32,
    weights: Vec<i8>, threshold: i32, // effective threshold: dot >= threshold
    alpha: f32, train_acc: f32, val_acc: f32,
}

impl Neuron {
    fn eval(&self, sigs: &[u8]) -> u8 {
        let mut d = 0i32;
        for (&w, &p) in self.weights.iter().zip(&self.parents) { d += (w as i32) * (sigs[p] as i32); }
        if d >= self.threshold { 1 } else { 0 }
    }
}

struct Net { neurons: Vec<Neuron>, n_in: usize, sig_ticks: Vec<u32> }
impl Net {
    fn new(n: usize) -> Self { Net { neurons: Vec::new(), n_in: n, sig_ticks: vec![0; n] } }
    fn n_sig(&self) -> usize { self.n_in + self.neurons.len() }
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
    fn add(&mut self, n: Neuron) { self.sig_ticks.push(n.tick); self.neurons.push(n); }
}

fn output_match_rate(a: &[u8], all_sigs: &[Vec<u8>], sig: usize) -> f32 {
    a.iter().enumerate().filter(|(i, v)| all_sigs[*i][sig] == **v).count() as f32 / a.len() as f32
}

#[derive(Clone)]
struct SignalScout {
    idx: usize,
    single_score: f32,
    single_sign: i8,
    probe_w: f32,
    rank_sum: usize,
}

#[derive(Clone)]
struct PairScout {
    a: usize,
    b: usize,
    score: f32,
    gain: f32,
}

fn sig_name(idx: usize, n_in: usize) -> String {
    if idx < n_in { format!("x{}", idx) } else { format!("N{}", idx - n_in) }
}

fn weighted_score(outputs: &[u8], labels: &[(Vec<u8>, u8)], sw: &[f32]) -> f32 {
    outputs.iter().zip(labels).zip(sw)
        .map(|((&pred, (_, y)), &wt)| if pred == *y { wt } else { 0.0 })
        .sum()
}

fn best_single_signal_scores(all_sigs: &[Vec<u8>], data: &Data, sw: &[f32], n_sig: usize) -> Vec<SignalScout> {
    let mut out = Vec::with_capacity(n_sig);
    for sig in 0..n_sig {
        let pos: f32 = data.train.iter().enumerate().zip(sw)
            .map(|((pi, (_, y)), &wt)| if all_sigs[pi][sig] == *y { wt } else { 0.0 })
            .sum();
        let neg: f32 = data.train.iter().enumerate().zip(sw)
            .map(|((pi, (_, y)), &wt)| if (1 - all_sigs[pi][sig]) == *y { wt } else { 0.0 })
            .sum();
        let (single_score, single_sign) = if pos >= neg { (pos, 1) } else { (neg, -1) };
        out.push(SignalScout { idx: sig, single_score, single_sign, probe_w: 0.0, rank_sum: 0 });
    }
    out
}

fn backprop_probe_all(all_sigs: &[Vec<u8>], data: &Data, sw: &[f32], n_sig: usize, epochs: usize) -> (Vec<f32>, f32) {
    let mut w = vec![0.0f32; n_sig];
    let mut b = 0.0f32;
    if n_sig == 0 { return (w, b); }
    for _ in 0..epochs {
        for (pi, (_, y)) in data.train.iter().enumerate() {
            let z = b + (0..n_sig).map(|i| w[i] * all_sigs[pi][i] as f32).sum::<f32>();
            let a = sigmoid(z);
            let g = (a - *y as f32) * sw[pi] * data.train.len() as f32;
            for i in 0..n_sig { w[i] -= 0.15 * g * all_sigs[pi][i] as f32; }
            b -= 0.15 * g;
        }
    }
    (w, b)
}

fn merge_signal_ranks(mut scouts: Vec<SignalScout>, probe_w: &[f32]) -> Vec<SignalScout> {
    let mut by_single: Vec<usize> = (0..scouts.len()).collect();
    by_single.sort_by(|&a, &b| scouts[b].single_score.partial_cmp(&scouts[a].single_score).unwrap());
    let mut single_rank = vec![0usize; scouts.len()];
    for (rank, idx) in by_single.iter().enumerate() { single_rank[*idx] = rank; }

    let mut by_probe: Vec<usize> = (0..probe_w.len()).collect();
    by_probe.sort_by(|&a, &b| probe_w[b].abs().partial_cmp(&probe_w[a].abs()).unwrap());
    let mut probe_rank = vec![0usize; probe_w.len()];
    for (rank, idx) in by_probe.iter().enumerate() { probe_rank[*idx] = rank; }

    for s in &mut scouts {
        s.probe_w = probe_w[s.idx];
        s.rank_sum = single_rank[s.idx] + probe_rank[s.idx];
    }
    scouts.sort_by(|a, b| {
        a.rank_sum.cmp(&b.rank_sum)
            .then_with(|| b.single_score.partial_cmp(&a.single_score).unwrap())
            .then_with(|| b.probe_w.abs().partial_cmp(&a.probe_w.abs()).unwrap())
    });
    scouts
}

fn best_small_ternary_score(parents: &[usize], all_sigs: &[Vec<u8>], data: &Data, sw: &[f32]) -> f32 {
    let ni = parents.len();
    if ni == 0 { return -1.0; }
    let np = data.train.len();
    let total = 3u64.pow(ni as u32);
    let mut best = -1.0f32;
    for combo in 0..total {
        let mut r = combo;
        let mut w = vec![0i8; ni];
        for wi in &mut w { *wi = (r % 3) as i8 - 1; r /= 3; }
        let dots: Vec<i32> = (0..np).map(|pi| {
            let mut d = -1i32;
            for (j, &pidx) in parents.iter().enumerate() { d += (w[j] as i32) * (all_sigs[pi][pidx] as i32); }
            d
        }).collect();
        let mn = dots.iter().copied().min().unwrap_or(0);
        let mx = dots.iter().copied().max().unwrap_or(0);
        for thresh in (mn - 1)..=(mx + 1) {
            let outs: Vec<u8> = dots.iter().map(|&d| if d >= thresh { 1 } else { 0 }).collect();
            let sc = weighted_score(&outs, &data.train, sw);
            if sc > best { best = sc; }
        }
    }
    best
}

fn pair_lifts_from_ranked(ranked: &[SignalScout], all_sigs: &[Vec<u8>], data: &Data, sw: &[f32], pair_top: usize) -> Vec<PairScout> {
    let top_n = ranked.len().min(pair_top.max(2));
    let mut out = Vec::new();
    for i in 0..top_n {
        for j in (i + 1)..top_n {
            let a = ranked[i].idx;
            let b = ranked[j].idx;
            let score = best_small_ternary_score(&[a, b], all_sigs, data, sw);
            let gain = score - ranked[i].single_score.max(ranked[j].single_score);
            out.push(PairScout { a, b, score, gain });
        }
    }
    out.sort_by(|a, b| b.gain.partial_cmp(&a.gain).unwrap().then_with(|| b.score.partial_cmp(&a.score).unwrap()));
    out
}

fn push_candidate(sets: &mut Vec<Vec<usize>>, seen: &mut HashSet<Vec<usize>>, parents: Vec<usize>) {
    if parents.len() < 2 { return; }
    let mut p = parents;
    p.sort_unstable();
    p.dedup();
    if p.len() < 2 { return; }
    if seen.insert(p.clone()) { sets.push(p); }
}

fn build_candidate_sets(
    ranked: &[SignalScout], pairs: &[PairScout], n_sig: usize, n_in: usize, cfg: &Config, step: usize,
) -> Vec<Vec<usize>> {
    let max_fan = cfg.max_fan.min(n_sig).max(2);
    let pool_n = ranked.len().min(cfg.scout_top.max(max_fan));
    let pool: Vec<usize> = ranked.iter().take(pool_n).map(|s| s.idx).collect();
    let hidden: Vec<usize> = ranked.iter().filter(|s| s.idx >= n_in).take(pool_n).map(|s| s.idx).collect();
    let raw: Vec<usize> = ranked.iter().filter(|s| s.idx < n_in).take(pool_n).map(|s| s.idx).collect();

    let mut sets = Vec::new();
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    for &sz in &[2usize, 3, 4, 6, 8, 10, 12] {
        let sz = sz.min(max_fan).min(pool.len());
        if sz >= 2 { push_candidate(&mut sets, &mut seen, pool.iter().copied().take(sz).collect()); }
    }

    if !hidden.is_empty() {
        let mut mix = Vec::new();
        let h_take = hidden.len().min(max_fan / 2 + 1);
        mix.extend(hidden.iter().copied().take(h_take));
        for &p in &pool {
            if mix.len() >= max_fan { break; }
            if !mix.contains(&p) { mix.push(p); }
        }
        push_candidate(&mut sets, &mut seen, mix);
    }

    if !raw.is_empty() {
        push_candidate(&mut sets, &mut seen, raw.iter().copied().take(raw.len().min(max_fan)).collect());
    }

    for pair in pairs.iter().take(4) {
        let mut seeded = vec![pair.a, pair.b];
        for &p in &pool {
            if seeded.len() >= max_fan.min(6) { break; }
            if !seeded.contains(&p) { seeded.push(p); }
        }
        push_candidate(&mut sets, &mut seen, seeded);
    }

    let mut rng = Rng::new(cfg.search_seed ^ (step as u64 * 6361 + n_sig as u64 * 17 + 99));
    let mut stuck = 0;
    while sets.len() < cfg.n_proposals && !pool.is_empty() {
        let before = sets.len();
        let target = 2 + rng.pick(max_fan - 1);
        let mut cand = Vec::new();
        for _ in 0..pool.len() * 3 {
            if cand.len() >= target.min(pool.len()) { break; }
            let p = pool[rng.pick(pool.len())];
            if !cand.contains(&p) { cand.push(p); }
        }
        push_candidate(&mut sets, &mut seen, cand);
        if sets.len() == before { stuck += 1; if stuck > 32 { break; } } else { stuck = 0; }
    }

    let mut stuck = 0;
    while sets.len() < cfg.n_proposals && n_sig >= 2 {
        let before = sets.len();
        let target = 2 + rng.pick(max_fan - 1);
        let mut cand = Vec::new();
        for _ in 0..n_sig * 3 {
            if cand.len() >= target.min(n_sig) { break; }
            let p = rng.pick(n_sig);
            if !cand.contains(&p) { cand.push(p); }
        }
        push_candidate(&mut sets, &mut seen, cand);
        if sets.len() == before { stuck += 1; if stuck > 32 { break; } } else { stuck = 0; }
    }
    sets
}

fn print_scout(step: usize, ranked: &[SignalScout], pairs: &[PairScout], n_in: usize, proposal_sets: &[Vec<usize>]) {
    let top_sig: Vec<String> = ranked.iter().take(5).map(|s| {
        format!("{} s={:.3} sign={} |w|={:.2}", sig_name(s.idx, n_in), s.single_score, s.single_sign, s.probe_w.abs())
    }).collect();
    println!("    scout top: {}", top_sig.join(" | "));
    if !pairs.is_empty() {
        let top_pairs: Vec<String> = pairs.iter().take(3).map(|p| {
            format!("({},{}) +{:.3}", sig_name(p.a, n_in), sig_name(p.b, n_in), p.gain)
        }).collect();
        println!("    scout pairs: {}", top_pairs.join(" | "));
    }
    let cand_preview: Vec<String> = proposal_sets.iter().take(5).map(|set| {
        let names: Vec<String> = set.iter().map(|&p| sig_name(p, n_in)).collect();
        format!("[{}]", names.join(","))
    }).collect();
    println!("    scout cands: {}", cand_preview.join(" "));
    if step == usize::MAX { println!(); }
}

// ══════════════════════════════════════════════════════
// PERSISTENT STATE (TSV) — load on start, save after each neuron
// ══════════════════════════════════════════════════════
struct StateHead { task: String, data_seed: u64, noise: f32, n_per: usize, n_in: usize }

fn save_state(net: &Net, path: &str, head: &StateHead) {
    if let Some(p) = std::path::Path::new(path).parent() { std::fs::create_dir_all(p).ok(); }
    let tmp = format!("{}.tmp", path);
    let mut f = std::fs::File::create(&tmp).unwrap();
    writeln!(f, "HEAD\t{}\t{}\t{}\t{}\t{}", head.task, head.data_seed, head.noise, head.n_per, head.n_in).unwrap();
    for n in &net.neurons {
        let ps: Vec<String> = n.parents.iter().map(|v| v.to_string()).collect();
        let ws: Vec<String> = n.weights.iter().map(|v| v.to_string()).collect();
        writeln!(f, "N\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.2}\t{:.2}",
            n.id, ps.join(","), n.tick, ws.join(","), n.threshold,
            n.alpha, n.train_acc, n.val_acc).unwrap();
    }
    drop(f);
    std::fs::rename(&tmp, path).unwrap();
}

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

fn replay_sw(net: &Net, data: &Data) -> Vec<f32> {
    let mut sw = vec![1.0 / data.train.len() as f32; data.train.len()];
    if net.neurons.is_empty() { return sw; }
    let all_sigs: Vec<Vec<u8>> = data.train.iter().map(|(x, _)| net.eval_all(x)).collect();
    for (ni, n) in net.neurons.iter().enumerate() {
        let sig_idx = net.n_in + ni;
        let alpha = n.alpha;
        let mut norm = 0.0f32;
        for (pi, ((_, y), wt)) in data.train.iter().zip(sw.iter_mut()).enumerate() {
            let pred = all_sigs[pi][sig_idx];
            let ys = if *y == 1 { 1.0 } else { -1.0 };
            let hs = if pred == 1 { 1.0 } else { -1.0 };
            *wt *= (-alpha * ys * hs).exp();
            norm += *wt;
        }
        if norm > 0.0 { for w in &mut sw { *w /= norm; } }
    }
    sw
}

// ══════════════════════════════════════════════════════
// CHECKPOINT (per-step JSON snapshot — human readable, not used for load)
// ══════════════════════════════════════════════════════
fn save_checkpoint(net: &Net, path: &str, task: &str, step: usize, data: &Data) {
    if let Some(p) = std::path::Path::new(path).parent() { std::fs::create_dir_all(p).ok(); }
    let et = net.accuracy(&data.train); let ev = net.accuracy(&data.val); let ete = net.accuracy(&data.test);
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "{{").unwrap();
    writeln!(f, "\"task\":\"{}\",\"step\":{},\"n_inputs\":{},", task, step, net.n_in).unwrap();
    writeln!(f, "\"ensemble_train\":{:.2},\"ensemble_val\":{:.2},\"ensemble_test\":{:.2},", et, ev, ete).unwrap();
    writeln!(f, "\"neurons\":[").unwrap();
    for (i, n) in net.neurons.iter().enumerate() {
        let wj: Vec<String> = n.weights.iter().map(|v| v.to_string()).collect();
        let pj: Vec<String> = n.parents.iter().map(|v| v.to_string()).collect();
        writeln!(f, "{{\"id\":{},\"parents\":[{}],\"tick\":{},\"weights\":[{}],\"threshold\":{},\"alpha\":{:.6},\"train_acc\":{:.2},\"val_acc\":{:.2}}}{}",
            n.id, pj.join(","), n.tick, wj.join(","), n.threshold, n.alpha, n.train_acc, n.val_acc,
            if i < net.neurons.len()-1 { "," } else { "" }).unwrap();
    }
    writeln!(f, "]}}").unwrap();
}

// ══════════════════════════════════════════════════════
// NEW: TRAINING TRACE RECORDER — one JSON per (task, seed) run
// ══════════════════════════════════════════════════════
struct TraceEvent {
    event: &'static str,
    tick: u32,
    id: usize,
    parents: Vec<usize>,
    weights: Vec<i8>,
    threshold: i32,
    alpha: f32,
    train_acc: f32,
    val_acc: f32,
}

#[allow(clippy::too_many_arguments)]
fn write_trace_json(
    out_dir: &str,
    cfg: &Config,
    n_in: usize,
    events: &[TraceEvent],
    final_val: f32,
    final_test: f32,
    final_neurons: usize,
    max_depth: u32,
    stall: usize,
) -> std::io::Result<()> {
    let path = format!("{}/trace.json", out_dir);
    if let Some(p) = std::path::Path::new(&path).parent() { std::fs::create_dir_all(p)?; }
    let mut f = std::fs::File::create(&path)?;
    writeln!(f, "{{")?;
    writeln!(f, "  \"task\": \"{}\",", cfg.task)?;
    writeln!(f, "  \"data_seed\": {},", cfg.data_seed)?;
    writeln!(f, "  \"search_seed\": {},", cfg.search_seed)?;
    writeln!(f, "  \"n_in\": {},", n_in)?;
    writeln!(f, "  \"n_per\": {},", cfg.n_per)?;
    writeln!(f, "  \"noise\": {},", cfg.noise)?;
    writeln!(f, "  \"events\": [")?;
    for (i, e) in events.iter().enumerate() {
        let ps: Vec<String> = e.parents.iter().map(|v| v.to_string()).collect();
        let ws: Vec<String> = e.weights.iter().map(|v| v.to_string()).collect();
        let comma = if i + 1 < events.len() { "," } else { "" };
        writeln!(f,
            "    {{\"event\":\"{}\",\"tick\":{},\"id\":{},\"parents\":[{}],\"weights\":[{}],\"threshold\":{},\"alpha\":{:.6},\"train_acc\":{:.1},\"val_acc\":{:.1}}}{}",
            e.event, e.tick, e.id, ps.join(","), ws.join(","),
            e.threshold, e.alpha, e.train_acc, e.val_acc, comma
        )?;
    }
    writeln!(f, "  ],")?;
    writeln!(f, "  \"final\": {{")?;
    writeln!(f, "    \"best_val_acc\": {:.1},", final_val)?;
    writeln!(f, "    \"best_test_acc\": {:.1},", final_test)?;
    writeln!(f, "    \"total_neurons\": {},", final_neurons)?;
    writeln!(f, "    \"max_depth\": {},", max_depth)?;
    writeln!(f, "    \"stall_count\": {}", stall)?;
    writeln!(f, "  }}")?;
    writeln!(f, "}}")?;
    Ok(())
}

// ══════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════
fn main() {
    let cfg = parse_args();
    let t0 = Instant::now();

    let n_in = task_n_in(&cfg.task);

    let label_fn: Box<dyn Fn(usize, &[u8]) -> Option<u8>> = match cfg.task.as_str() {

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
            // 3-bit XOR on main diagonal: true iff popcount of {b0,b4,b8} is odd
            let parity = (px[0] ^ px[4] ^ px[8]) & 1;
            Some(parity)
        }),

        "grid3_full_parity" => Box::new(|_, px| {
            // 9-bit XOR over the whole grid: true iff popcount of {b0..b8} is odd
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

        "grid3_copy_bit_0" => Box::new(|_, px| Some(px[0])),
        "grid3_copy_bit_1" => Box::new(|_, px| Some(px[1])),
        "grid3_copy_bit_2" => Box::new(|_, px| Some(px[2])),
        "grid3_copy_bit_3" => Box::new(|_, px| Some(px[3])),
        "grid3_copy_bit_4" => Box::new(|_, px| Some(px[4])),
        "grid3_copy_bit_5" => Box::new(|_, px| Some(px[5])),
        "grid3_copy_bit_6" => Box::new(|_, px| Some(px[6])),
        "grid3_copy_bit_7" => Box::new(|_, px| Some(px[7])),
        "grid3_copy_bit_8" => Box::new(|_, px| Some(px[8])),

        other => {
            eprintln!("ERROR: unknown task '{}' (expected one of the grid3_* tasks)", other);
            std::process::exit(2);
        }
    };

    // Resolve data params from CLI, then potentially override from loaded state
    let mut noise = cfg.noise;
    let mut n_per = cfg.n_per;
    let mut data_seed = cfg.data_seed;

    // Per-run out dir (task × search_seed); each run owns its own state.tsv + trace.json
    std::fs::create_dir_all(&cfg.out_dir).unwrap_or_else(|e| {
        eprintln!("ERROR: failed to create out-dir '{}': {}", cfg.out_dir, e);
        std::process::exit(1);
    });
    let state_path = format!("{}/state.tsv", cfg.out_dir);

    // Try to load existing state — enforces that task + n_in match
    let (mut net, loaded) = match load_state(&state_path) {
        Ok(Some((head, net))) => {
            if head.task != cfg.task {
                eprintln!("ERROR: state task={} != cli task={}", head.task, cfg.task);
                std::process::exit(2);
            }
            if head.n_in != n_in {
                eprintln!("ERROR: state n_in={} != task {} expected n_in={}", head.n_in, cfg.task, n_in);
                std::process::exit(2);
            }
            // State wins over CLI — AdaBoost sw is only valid on the data it was built on
            if data_seed != head.data_seed || (noise - head.noise).abs() > 1e-6 || n_per != head.n_per {
                println!("  [load] state data params override CLI: seed {}→{} noise {}→{} n_per {}→{}",
                    data_seed, head.data_seed, noise, head.noise, n_per, head.n_per);
            }
            data_seed = head.data_seed;
            noise = head.noise;
            n_per = head.n_per;
            println!("  [load] {} neurons from {}", net.neurons.len(), state_path);
            (net, true)
        }
        Ok(None) => {
            println!("  [load] no prior state at {} — starting fresh", state_path);
            (Net::new(n_in), false)
        }
        Err(e) => {
            eprintln!("ERROR: {}", e);
            std::process::exit(2);
        }
    };

    let data = gen_data(label_fn.as_ref(), noise, n_per, data_seed);

    // Replay AdaBoost to reconstruct sample weights consistent with the loaded network
    let mut sw = if loaded { replay_sw(&net, &data) }
        else { vec![1.0 / data.train.len() as f32; data.train.len()] };

    let head = StateHead { task: cfg.task.clone(), data_seed, noise, n_per, n_in: net.n_in };
    let mut stall: usize = 0;
    let mut best_val = if loaded { net.accuracy(&data.val) } else { 50.0f32 };

    // Training trace — every accepted neuron appends here; written at end of main()
    let mut trace_events: Vec<TraceEvent> = Vec::new();

    println!("===========================================================");
    println!("  grid3_curriculum — {} (data_seed={} search_seed={})", cfg.task, cfg.data_seed, cfg.search_seed);
    println!("  {} proposals/step, max_fan={}, stall={}, max_neurons={}",
        cfg.n_proposals, cfg.max_fan, cfg.stall_limit, cfg.max_neurons);
    println!("  scout_top={} pair_top={} probe_epochs={}",
        cfg.scout_top, cfg.pair_top, cfg.probe_epochs);
    println!("  Data: {} train / {} val / {} test", data.train.len(), data.val.len(), data.test.len());
    println!("===========================================================");

    for step in 0..cfg.max_neurons {
        let t_step = Instant::now();
        let ens_val = net.accuracy(&data.val);
        let ens_test = net.accuracy(&data.test);
        println!("\n  Step {:3} | {} neurons | val={:.1}% test={:.1}% | {:.0}s elapsed",
            step, net.neurons.len(), ens_val, ens_test, t0.elapsed().as_secs_f64());

        if ens_val >= 99.0 { println!("  >> Target reached!"); break; }

        // Precompute signals
        let all_sigs: Vec<Vec<u8>> = data.train.iter().map(|(x, _)| net.eval_all(x)).collect();
        let all_val_sigs: Vec<Vec<u8>> = data.val.iter().map(|(x, _)| net.eval_all(x)).collect();
        let n_sig = net.n_sig();

        // STEP A: Scout promising parents quickly
        let single_scores = best_single_signal_scores(&all_sigs, &data, &sw, n_sig);
        let (probe_w, probe_b) = backprop_probe_all(&all_sigs, &data, &sw, n_sig, cfg.probe_epochs);
        let ranked = merge_signal_ranks(single_scores, &probe_w);
        let pairs = pair_lifts_from_ranked(&ranked, &all_sigs, &data, &sw, cfg.pair_top);
        let proposal_sets = build_candidate_sets(&ranked, &pairs, n_sig, net.n_in, &cfg, step);
        print_scout(step, &ranked, &pairs, net.n_in, &proposal_sets);

        // STEP B: Generate proposals from ranked candidate sets
        let mut proposals: Vec<(Vec<usize>, Vec<f32>, f32, f32)> = Vec::new();
        for (seed, parents) in proposal_sets.iter().enumerate() {
            let ni = parents.len();
                let mut rng = Rng::new(cfg.search_seed ^ (seed as u64 * 7919 + 31 + step as u64 * 104729));
            let w: Vec<f32> = parents.iter().map(|&p| {
                let base = probe_w[p];
                if base.abs() > 0.05 { base + rng.range(-0.15, 0.15) }
                else {
                    let s = ranked.iter().find(|r| r.idx == p).map(|r| r.single_sign as f32).unwrap_or(1.0);
                    s * rng.range(0.1, 0.8)
                }
            }).collect();
            let b = probe_b + rng.range(-0.2, 0.2);
            let score: f32 = data.train.iter().enumerate().map(|(pi, (_, y))| {
                let z: f32 = b + (0..ni).map(|i| w[i] * all_sigs[pi][parents[i]] as f32).sum::<f32>();
                if (if sigmoid(z) > 0.5 { 1u8 } else { 0 }) == *y { sw[pi] } else { 0.0 }
            }).sum();
            proposals.push((parents.clone(), w, b, score));
        }
        proposals.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

        if proposals.is_empty() { println!("    X No proposals"); break; }

        // STEP C: Backprop top-5
        let top_k = proposals.len().min(5);
        struct Trained { parents: Vec<usize>, val_acc: f32, consensus: Vec<i8> }
        let mut trained: Vec<Trained> = Vec::new();

        for pi in 0..top_k {
            let (ref parents, ref init_w, init_b, _) = proposals[pi];
            let ni = parents.len();
            let mut all_converged: Vec<(Vec<f32>, f32)> = Vec::new();

            for restart in 0..5u64 {
                let mut rng = Rng::new(cfg.search_seed ^ (restart * 1000 + 77));
                let mut w: Vec<f32> = if restart == 0 { init_w.clone() }
                    else { init_w.iter().map(|&v| v + rng.range(-0.5, 0.5)).collect() };
                let mut b = if restart == 0 { init_b } else { init_b + rng.range(-0.3, 0.3) };

                for _ in 0..2000 {
                    for (pii, (_, y)) in data.train.iter().enumerate() {
                        let z: f32 = b + (0..ni).map(|i| w[i] * all_sigs[pii][parents[i]] as f32).sum::<f32>();
                        let a = sigmoid(z);
                        let g = (a - *y as f32) * a * (1.0 - a) * sw[pii] * data.train.len() as f32;
                        for i in 0..ni { w[i] -= 0.5 * g * all_sigs[pii][parents[i]] as f32; }
                        b -= 0.5 * g;
                    }
                }
                all_converged.push((w, b));
            }

            let consensus: Vec<i8> = (0..ni).map(|i| {
                let pos = all_converged.iter().filter(|(w, _)| w[i] > 0.3).count();
                let neg = all_converged.iter().filter(|(w, _)| w[i] < -0.3).count();
                if pos * 10 / 5 >= 7 { 1 } else if neg * 10 / 5 >= 7 { -1 } else { 2 }
            }).collect();

            let best_w = &all_converged[0].0; let best_b = all_converged[0].1;
            let val_acc = {
                let c = data.val.iter().enumerate().filter(|(vi, (_, y))| {
                    let z: f32 = best_b + (0..ni).map(|i| best_w[i] * all_val_sigs[*vi][parents[i]] as f32).sum::<f32>();
                    (if sigmoid(z) > 0.5 { 1u8 } else { 0 }) == *y
                }).count();
                c as f32 / data.val.len() as f32 * 100.0
            };

            trained.push(Trained { parents: parents.clone(), val_acc, consensus });
        }
        trained.sort_by(|a, b| b.val_acc.partial_cmp(&a.val_acc).unwrap());

        // STEP D+E: Ternary + accept gate — first that improves
        let mut accepted = false;

        for tp in &trained {
            let ni = tp.parents.len();
            let np = data.train.len();

            // Ternary guided + blind
            let locked: Vec<Option<i8>> = tp.consensus.iter().map(|&s| {
                if s == 1 { Some(1) } else if s == -1 { Some(-1) } else { None }
            }).collect();
            let free_pos: Vec<usize> = (0..ni).filter(|&i| locked[i].is_none()).collect();

            let mut bw = vec![0i8; ni]; let mut bt: i32 = 0;
            let mut bs = -1.0f32; let mut bo = vec![0u8; np];

            // Guided — search directly in the bias-free threshold form: dot >= threshold
            let total_free = 3u64.pow(free_pos.len() as u32);
            for combo in 0..total_free {
                let mut w = vec![0i8; ni];
                for i in 0..ni { w[i] = locked[i].unwrap_or(0); }
                let mut r = combo;
                for &fp in &free_pos { w[fp] = (r % 3) as i8 - 1; r /= 3; }
                let _ = r;
                let dots: Vec<i32> = (0..np).map(|pi| {
                    let mut d = 0i32;
                    for (j, &pidx) in tp.parents.iter().enumerate() { d += (w[j] as i32) * (all_sigs[pi][pidx] as i32); }
                    d
                }).collect();
                let mn = dots.iter().copied().min().unwrap_or(0);
                let mx = dots.iter().copied().max().unwrap_or(0);
                for threshold in (mn-1)..=(mx+1) {
                    let outs: Vec<u8> = dots.iter().map(|&d| if d >= threshold { 1 } else { 0 }).collect();
                    let sc: f32 = outs.iter().zip(&data.train).zip(&sw)
                        .map(|((&pred, (_, y)), &wt)| if pred == *y { wt } else { 0.0 }).sum();
                    if sc > bs { bs=sc; bw=w.clone(); bt=threshold; bo=outs; }
                }
            }

            // Blind — same bias-free threshold form, just without consensus locking
            let total_blind = 3u64.pow(ni as u32);
            if total_blind <= 500_000 {
                for combo in 0..total_blind {
                    let mut w = vec![0i8; ni]; let mut r = combo;
                    for wi in w.iter_mut() { *wi = (r % 3) as i8 - 1; r /= 3; }
                    let _ = r;
                    let dots: Vec<i32> = (0..np).map(|pi| {
                        let mut d = 0i32;
                        for (j, &pidx) in tp.parents.iter().enumerate() { d += (w[j] as i32) * (all_sigs[pi][pidx] as i32); }
                        d
                    }).collect();
                    let mn = dots.iter().copied().min().unwrap_or(0);
                    let mx = dots.iter().copied().max().unwrap_or(0);
                    for threshold in (mn-1)..=(mx+1) {
                        let outs: Vec<u8> = dots.iter().map(|&d| if d >= threshold { 1 } else { 0 }).collect();
                        let sc: f32 = outs.iter().zip(&data.train).zip(&sw)
                            .map(|((&pred, (_, y)), &wt)| if pred == *y { wt } else { 0.0 }).sum();
                        if sc > bs { bs=sc; bw=w.clone(); bt=threshold; bo=outs; }
                    }
                }
            }

            // Accept gate
            let is_dup = (net.n_in..n_sig).any(|e| output_match_rate(&bo, &all_sigs, e) >= 0.999);
            if is_dup { continue; }

            let tick = tp.parents.iter().map(|&p| net.sig_ticks[p]).max().unwrap_or(0) + 1;
            let werr: f32 = bo.iter().zip(&data.train).zip(&sw)
                .map(|((&pred, (_, y)), &w)| if pred == *y { 0.0 } else { w }).sum();
            if werr >= 0.499 { continue; }
            let alpha = 0.5 * ((1.0 - werr).max(1e-6) / werr.max(1e-6)).ln();

            let qr_train = {
                let c = data.train.iter().enumerate().filter(|(ti, (_, y))| {
                    let mut d = 0i32;
                    for (&w, &pidx) in bw.iter().zip(&tp.parents) { d += (w as i32) * (all_sigs[*ti][pidx] as i32); }
                    (if d >= bt { 1u8 } else { 0 }) == *y
                }).count();
                c as f32 / data.train.len() as f32 * 100.0
            };

            let qr_val = {
                let c = data.val.iter().enumerate().filter(|(vi, (_, y))| {
                    let mut d = 0i32;
                    for (&w, &pidx) in bw.iter().zip(&tp.parents) { d += (w as i32) * (all_val_sigs[*vi][pidx] as i32); }
                    (if d >= bt { 1u8 } else { 0 }) == *y
                }).count();
                c as f32 / data.val.len() as f32 * 100.0
            };

            let neuron = Neuron {
                id: net.neurons.len(), parents: tp.parents.clone(), tick,
                weights: bw.clone(), threshold: bt,
                alpha, train_acc: qr_train, val_acc: qr_val,
            };
            net.add(neuron);
            let new_val = net.accuracy(&data.val);

            if new_val < ens_val {
                net.sig_ticks.pop();
                net.neurons.pop();
                continue;
            }

            // ACCEPTED — record trace event before persistence so the writer
            // always has the most recent neuron even if save_state panics.
            let accepted_neuron = net.neurons.last().unwrap();
            trace_events.push(TraceEvent {
                event: "neuron_added",
                tick: accepted_neuron.tick,
                id: accepted_neuron.id,
                parents: accepted_neuron.parents.clone(),
                weights: accepted_neuron.weights.clone(),
                threshold: accepted_neuron.threshold,
                alpha: accepted_neuron.alpha,
                train_acc: accepted_neuron.train_acc,
                val_acc: accepted_neuron.val_acc,
            });

            let has_hidden = tp.parents.iter().any(|&p| p >= net.n_in);
            let pnames: Vec<String> = tp.parents.iter().map(|&p| {
                sig_name(p, net.n_in)
            }).collect();
            let wstr: String = bw.iter().map(|&v| match v { 1=>"+", -1=>"-", _=>"0" }).collect::<Vec<_>>().join("");

            println!("    >> N{}: [{}] thr={} tick={} parents=[{}] val={:.1}->{:.1}% hidden={} ({:.0}ms)",
                net.neurons.len()-1, wstr, bt, tick, pnames.join(","),
                ens_val, new_val, has_hidden, t_step.elapsed().as_millis());

            // AdaBoost reweight
            let mut norm = 0.0f32;
            for ((pred, (_, y)), wt) in bo.iter().zip(&data.train).zip(sw.iter_mut()) {
                let ys = if *y == 1 { 1.0 } else { -1.0 };
                let hs = if *pred == 1 { 1.0 } else { -1.0 };
                *wt *= (-alpha * ys * hs).exp();
                norm += *wt;
            }
            if norm > 0.0 { for w in &mut sw { *w /= norm; } }

            // Persistent state — THE authoritative file for future resumes
            save_state(&net, &state_path, &head);

            // Per-step JSON snapshot (human-readable history)
            let ckpt = format!("{}/checkpoints/n{:03}.json", cfg.out_dir, net.neurons.len());
            save_checkpoint(&net, &ckpt, &cfg.task, step, &data);

            if new_val > best_val + 0.5 { best_val = new_val; stall = 0; }
            else { stall += 1; }

            accepted = true;
            break;
        }

        if !accepted {
            println!("    X No improvement ({:.0}ms)", t_step.elapsed().as_millis());
            stall += 1;
        }

        if stall >= cfg.stall_limit {
            println!("\n  Stalled {} steps.", cfg.stall_limit);
            break;
        }
    }

    let ft = net.accuracy(&data.train);
    let fv = net.accuracy(&data.val);
    let fte = net.accuracy(&data.test);
    let mt = net.neurons.iter().map(|n| n.tick).max().unwrap_or(0);
    let hid = net.neurons.iter().any(|n| n.parents.iter().any(|&p| p >= net.n_in));

    println!("\n  FINAL: {} neurons, depth={}, hidden={}", net.neurons.len(), mt, hid);
    println!("  train={:.1}% val={:.1}% test={:.1}%", ft, fv, fte);
    println!("  Time: {:.1}s", t0.elapsed().as_secs_f64());

    let ckpt = format!("{}/final.json", cfg.out_dir);
    save_checkpoint(&net, &ckpt, &cfg.task, net.neurons.len(), &data);
    // Final state save — in case last neuron was accepted mid-step but loop broke before save
    save_state(&net, &state_path, &head);

    // NEW: write training trace for brain_replay + harness
    if let Err(e) = write_trace_json(
        &cfg.out_dir,
        &cfg,
        net.n_in,
        &trace_events,
        fv,
        fte,
        net.neurons.len(),
        mt,
        stall,
    ) {
        eprintln!("ERROR: failed to write trace.json: {}", e);
        std::process::exit(1);
    }
}
