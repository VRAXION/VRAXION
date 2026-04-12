//! Neuron Grower — single network on one task, grows forever
//!
//! Same logic as neuron_canonical but:
//! - Single task (not 6)
//! - High neuron limit (100)
//! - High stall limit (30)
//! - Checkpoints after every accepted neuron
//!
//! This IS the canonical builder focused on one task for overnight growth.
//!
//! Run: cargo run --example neuron_grower --release
//! Args: cargo run --example neuron_grower --release -- --data-seed 42 --task digit_parity

use std::io::Write as IoWrite;
use std::time::Instant;
use std::collections::HashSet;

// ══════════════════════════════════════════════════════
// CLI
// ══════════════════════════════════════════════════════
struct Config {
    data_seed: u64,
    task: String,
    max_neurons: usize,
    max_fan: usize,
    n_proposals: usize,
    stall_limit: usize,
    noise: f32,
    n_per: usize,
    scout_top: usize,
    pair_top: usize,
    probe_epochs: usize,
    scout_only: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config {
        data_seed: 42, task: "digit_parity".into(),
        max_neurons: 100, max_fan: 10, n_proposals: 20, stall_limit: 30,
        noise: 0.0, n_per: 0,
        scout_top: 12, pair_top: 8, probe_epochs: 400, scout_only: false,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data-seed" => { i += 1; cfg.data_seed = args[i].parse().unwrap_or(42); }
            "--task" => { i += 1; cfg.task = args[i].clone(); }
            "--max-neurons" => { i += 1; cfg.max_neurons = args[i].parse().unwrap_or(100); }
            "--max-fan" => { i += 1; cfg.max_fan = args[i].parse().unwrap_or(10); }
            "--proposals" => { i += 1; cfg.n_proposals = args[i].parse().unwrap_or(20); }
            "--stall" => { i += 1; cfg.stall_limit = args[i].parse().unwrap_or(30); }
            "--noise" => { i += 1; cfg.noise = args[i].parse().unwrap_or(0.0); }
            "--n-per" => { i += 1; cfg.n_per = args[i].parse().unwrap_or(0); }
            "--scout-top" => { i += 1; cfg.scout_top = args[i].parse().unwrap_or(12); }
            "--pair-top" => { i += 1; cfg.pair_top = args[i].parse().unwrap_or(8); }
            "--probe-epochs" => { i += 1; cfg.probe_epochs = args[i].parse().unwrap_or(400); }
            "--scout-only" => { cfg.scout_only = true; }
            _ => {}
        }
        i += 1;
    }
    cfg
}

// ══════════════════════════════════════════════════════
// PRNG + SIGMOID
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
// FONT + DATA
// ══════════════════════════════════════════════════════
const FONT: [[u8; 9]; 10] = [
    [1,1,1, 1,0,1, 1,1,1], [0,1,0, 0,1,0, 0,1,0],
    [1,1,0, 0,1,0, 0,1,1], [1,1,0, 0,1,0, 1,1,0],
    [1,0,1, 1,1,1, 0,0,1], [0,1,1, 0,1,0, 1,1,0],
    [1,0,0, 1,1,0, 1,1,0], [1,1,1, 0,0,1, 0,0,1],
    [1,1,1, 1,1,1, 1,1,1], [1,1,1, 1,1,1, 0,1,1],
];

struct Data { train: Vec<(Vec<u8>, u8)>, val: Vec<(Vec<u8>, u8)>, test: Vec<(Vec<u8>, u8)> }

fn gen_data(label_fn: &dyn Fn(usize, &[u8]) -> Option<u8>, noise: f32, n_per: usize, seed: u64) -> Data {
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
// NEURON + NET
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

    let mut rng = Rng::new(step as u64 * 6361 + n_sig as u64 * 17 + 99);
    while sets.len() < cfg.n_proposals && !pool.is_empty() {
        let target = 2 + rng.pick(max_fan - 1);
        let mut cand = Vec::new();
        for _ in 0..pool.len() * 3 {
            if cand.len() >= target.min(pool.len()) { break; }
            let p = pool[rng.pick(pool.len())];
            if !cand.contains(&p) { cand.push(p); }
        }
        push_candidate(&mut sets, &mut seen, cand);
    }

    while sets.len() < cfg.n_proposals && n_sig >= 2 {
        let target = 2 + rng.pick(max_fan - 1);
        let mut cand = Vec::new();
        for _ in 0..n_sig * 3 {
            if cand.len() >= target.min(n_sig) { break; }
            let p = rng.pick(n_sig);
            if !cand.contains(&p) { cand.push(p); }
        }
        push_candidate(&mut sets, &mut seen, cand);
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
// Format (tab-separated):
//   HEAD\ttask\tdata_seed\tnoise\tn_per\tn_in
//   N\tid\tparents(csv)\ttick\tweights(csv)\tthreshold\talpha\ttrain_acc\tval_acc
//   N\t...
//
// Load enforces task/data_seed/noise/n_per match — mismatched state is refused,
// because AdaBoost sample weights replayed on different data would be wrong.

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
    // Atomic replace — survives crashes mid-write
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

// Replay AdaBoost on the loaded network → exact sample weights as if grown in one pass.
// Uses each neuron's STORED alpha (not a recomputed one) so the replay matches the
// trajectory that was actually taken at save time.
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
// MAIN — canonical pipeline, single task, forever
// ══════════════════════════════════════════════════════
fn main() {
    let cfg = parse_args();
    let t0 = Instant::now();

    let label_fn: Box<dyn Fn(usize, &[u8]) -> Option<u8>> = match cfg.task.as_str() {
        "digit_parity" => Box::new(|_, px| { let p: usize = px.iter().map(|&v| v as usize).sum(); Some((p % 2) as u8) }),
        "is_symmetric" => Box::new(|_, px| { let s = px[0]==px[2] && px[3]==px[5] && px[6]==px[8]; Some(if s {1} else {0}) }),
        "digit_2_vs_3" => Box::new(|d, _| if d == 2 { Some(0) } else if d == 3 { Some(1) } else { None }),
        "is_digit_0" => Box::new(|d, _| Some(if d == 0 { 1 } else { 0 })),
        "is_digit_even" => Box::new(|d, _| Some(if d % 2 == 0 { 1 } else { 0 })),
        "digit_mod3" => Box::new(|d, _| Some(if d % 3 == 0 { 1 } else { 0 })),
        "top_row_sum" => Box::new(|_, px| Some(if px[0]+px[1]+px[2] >= 2 { 1 } else { 0 })),
        "diagonal_xor" => Box::new(|_, px| Some(((px[0] ^ px[4] ^ px[8]) & 1) as u8)),
        "popcount_gt4" => Box::new(|_, px| Some(if px.iter().map(|&v| v as usize).sum::<usize>() > 4 { 1 } else { 0 })),
        "corners_xor" => Box::new(|_, px| Some(((px[0] ^ px[2] ^ px[6] ^ px[8]) & 1) as u8)),
        "full_parity_4" => Box::new(|_, px| { let s: usize = (0..4).map(|i| px[i] as usize).sum(); Some((s % 2) as u8) }),
        "has_center" => Box::new(|_, px| Some(px[4])),
        "digit_is_prime" => Box::new(|d, _| Some(if d == 2 || d == 3 || d == 5 || d == 7 { 1 } else { 0 })),
        "center_plus_corners" => Box::new(|_, px| Some(if (px[0]+px[2]+px[4]+px[6]+px[8]) as usize >= 3 { 1 } else { 0 })),
        _ => Box::new(|_, px| { let p: usize = px.iter().map(|&v| v as usize).sum(); Some((p % 2) as u8) }),
    };

    // Resolve data params from CLI, then potentially override from loaded state
    let mut noise = if cfg.noise > 0.0 { cfg.noise }
        else if cfg.task == "digit_parity" { 0.10 } else { 0.15 };
    let mut n_per = if cfg.n_per > 0 { cfg.n_per }
        else if cfg.task == "digit_2_vs_3" { 200 } else { 100 };
    let mut data_seed = cfg.data_seed;

    // Persistent state dir: ONE growing network per task, seed-independent path
    let out_dir = format!("results/neuron_grower_persistent/{}", cfg.task);
    std::fs::create_dir_all(&out_dir).unwrap();
    let state_path = format!("{}/state.tsv", out_dir);

    // Try to load existing state — enforces that data params match what was saved
    let (mut net, loaded) = match load_state(&state_path) {
        Ok(Some((head, net))) => {
            if head.task != cfg.task {
                eprintln!("ERROR: state task={} != cli task={}", head.task, cfg.task);
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
            (Net::new(9), false)
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
    let mut stall = 0;
    let mut best_val = if loaded { net.accuracy(&data.val) } else { 50.0f32 };

    println!("===========================================================");
    println!("  Neuron Grower — {} (data_seed={})", cfg.task, cfg.data_seed);
    println!("  {} proposals/step, max_fan={}, stall={}, max_neurons={}",
        cfg.n_proposals, cfg.max_fan, cfg.stall_limit, cfg.max_neurons);
    println!("  scout_top={} pair_top={} probe_epochs={} scout_only={}",
        cfg.scout_top, cfg.pair_top, cfg.probe_epochs, cfg.scout_only);
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

        if cfg.scout_only {
            println!("  >> scout-only mode: exiting after parent ranking");
            break;
        }

        // STEP B: Generate proposals from ranked candidate sets
        let mut proposals: Vec<(Vec<usize>, Vec<f32>, f32, f32)> = Vec::new();
        for (seed, parents) in proposal_sets.iter().enumerate() {
            let ni = parents.len();
            let mut rng = Rng::new(seed as u64 * 7919 + 31 + step as u64 * 104729);
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
                let mut rng = Rng::new(restart * 1000 + 77);
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
                alpha, train_acc: qr_val, val_acc: qr_val,
            };
            net.add(neuron);
            let new_val = net.accuracy(&data.val);

            if new_val <= ens_val {
                net.sig_ticks.pop();
                net.neurons.pop();
                continue;
            }

            // ACCEPTED
            let has_hidden = tp.parents.iter().any(|&p| p >= net.n_in);
            let pnames: Vec<String> = tp.parents.iter().map(|&p| {
                if p < 9 { format!("x{}", p) } else { format!("N{}", p - 9) }
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
            let ckpt = format!("{}/checkpoints/n{:03}.json", out_dir, net.neurons.len());
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

    let ckpt = format!("{}/final.json", out_dir);
    save_checkpoint(&net, &ckpt, &cfg.task, net.neurons.len(), &data);
    // Final state save — in case last neuron was accepted mid-step but loop broke before save
    save_state(&net, &state_path, &head);
}
