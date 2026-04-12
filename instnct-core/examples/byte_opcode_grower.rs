//! Grower-coupled byte + opcode v1 benchmark.
//!
//! Uses the current bias-free scout-based grower logic to train 8 separate
//! binary heads on the exact domain:
//!   input  = 8 data bits + 4 opcode one-hot
//!   output = 1 target bit of the result byte
//!
//! The combined 8 heads are then evaluated as a full-byte predictor.
//!
//! Run:
//!   cargo run --release --example byte_opcode_grower

use std::collections::{HashMap, HashSet};

const DATA_BITS: usize = 8;
const OPCODES: usize = 4;
const INPUT_DIM: usize = DATA_BITS + OPCODES;
const ADV_CASES: [u8; 10] = [0x00, 0x01, 0x0F, 0x10, 0x7F, 0x80, 0xFE, 0xFF, 0x55, 0xAA];

struct Config {
    search_seed: u64,
    max_neurons: usize,
    max_fan: usize,
    proposals: usize,
    stall_limit: usize,
    scout_top: usize,
    pair_top: usize,
    probe_epochs: usize,
    translator_hidden: usize,
    translator_epochs: usize,
    translator_lr: f32,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config {
        search_seed: 42,
        max_neurons: 8,
        max_fan: 10,
        proposals: 16,
        stall_limit: 6,
        scout_top: 12,
        pair_top: 8,
        probe_epochs: 200,
        translator_hidden: 24,
        translator_epochs: 800,
        translator_lr: 0.03,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--search-seed" => { i += 1; cfg.search_seed = args[i].parse().unwrap_or(42); }
            "--max-neurons" => { i += 1; cfg.max_neurons = args[i].parse().unwrap_or(8); }
            "--max-fan" => { i += 1; cfg.max_fan = args[i].parse().unwrap_or(10); }
            "--proposals" => { i += 1; cfg.proposals = args[i].parse().unwrap_or(16); }
            "--stall" => { i += 1; cfg.stall_limit = args[i].parse().unwrap_or(6); }
            "--scout-top" => { i += 1; cfg.scout_top = args[i].parse().unwrap_or(12); }
            "--pair-top" => { i += 1; cfg.pair_top = args[i].parse().unwrap_or(8); }
            "--probe-epochs" => { i += 1; cfg.probe_epochs = args[i].parse().unwrap_or(200); }
            "--translator-hidden" => { i += 1; cfg.translator_hidden = args[i].parse().unwrap_or(24); }
            "--translator-epochs" => { i += 1; cfg.translator_epochs = args[i].parse().unwrap_or(800); }
            "--translator-lr" => { i += 1; cfg.translator_lr = args[i].parse().unwrap_or(0.03); }
            _ => {}
        }
        i += 1;
    }
    cfg
}

#[derive(Clone, Copy)]
enum Opcode { Copy, Not, Inc, Dec }
impl Opcode {
    fn all() -> [Self; 4] { [Self::Copy, Self::Not, Self::Inc, Self::Dec] }
    fn idx(self) -> usize {
        match self {
            Self::Copy => 0,
            Self::Not => 1,
            Self::Inc => 2,
            Self::Dec => 3,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Copy => "COPY",
            Self::Not => "NOT",
            Self::Inc => "INC",
            Self::Dec => "DEC",
        }
    }
    fn apply(self, x: u8) -> u8 {
        match self {
            Self::Copy => x,
            Self::Not => !x,
            Self::Inc => x.wrapping_add(1),
            Self::Dec => x.wrapping_sub(1),
        }
    }
}

#[derive(Clone)]
struct Sample {
    input: Vec<u8>,
    opcode: Opcode,
    target_byte: u8,
}

fn bits8(x: u8) -> [u8; DATA_BITS] {
    let mut out = [0u8; DATA_BITS];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (x >> i) & 1;
    }
    out
}

fn dataset() -> Vec<Sample> {
    let mut out = Vec::with_capacity(256 * OPCODES);
    for x in 0u16..=255 {
        let xb = x as u8;
        let bits = bits8(xb);
        for op in Opcode::all() {
            let mut input = Vec::with_capacity(INPUT_DIM);
            input.extend(bits);
            for oi in 0..OPCODES {
                input.push(if oi == op.idx() { 1 } else { 0 });
            }
            out.push(Sample {
                input,
                opcode: op,
                target_byte: op.apply(xb),
            });
        }
    }
    out
}

struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Self { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 {
        self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.s
    }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }
    fn pick(&mut self, n: usize) -> usize { self.next() as usize % n }
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

#[derive(Clone)]
struct Neuron {
    parents: Vec<usize>,
    tick: u32,
    weights: Vec<i8>,
    threshold: i32,
    alpha: f32,
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

#[derive(Clone)]
struct Net {
    n_in: usize,
    neurons: Vec<Neuron>,
    sig_ticks: Vec<u32>,
}

impl Net {
    fn new(n_in: usize) -> Self { Self { n_in, neurons: Vec::new(), sig_ticks: vec![0; n_in] } }
    fn n_sig(&self) -> usize { self.n_in + self.neurons.len() }
    fn eval_all(&self, inp: &[u8]) -> Vec<u8> {
        let mut s = inp.to_vec();
        for n in &self.neurons { s.push(n.eval(&s)); }
        s
    }
    fn score_from_sigs(&self, sigs: &[u8]) -> f32 {
        self.neurons.iter().enumerate()
            .map(|(i, n)| n.alpha * if sigs[self.n_in + i] == 1 { 1.0 } else { -1.0 })
            .sum()
    }
    fn score(&self, inp: &[u8]) -> f32 {
        if self.neurons.is_empty() { return 0.0; }
        let sigs = self.eval_all(inp);
        self.score_from_sigs(&sigs)
    }
    fn predict(&self, inp: &[u8]) -> u8 {
        if self.neurons.is_empty() { return 0; }
        let sigs = self.eval_all(inp);
        let score = self.score_from_sigs(&sigs);
        if score >= 0.0 { 1 } else { 0 }
    }
    fn accuracy(&self, data: &[(Vec<u8>, u8)]) -> f32 {
        if self.neurons.is_empty() { return 50.0; }
        data.iter().filter(|(x, y)| self.predict(x) == *y).count() as f32 / data.len() as f32 * 100.0
    }
    fn add(&mut self, n: Neuron) {
        self.sig_ticks.push(n.tick);
        self.neurons.push(n);
    }
}

fn relu(x: f32) -> f32 { x.max(0.0) }

#[derive(Clone)]
struct Mlp {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
}

impl Mlp {
    fn new(input: usize, hidden: usize, output: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15);
        let mut w1 = vec![vec![0.0; input]; hidden];
        let mut w2 = vec![vec![0.0; hidden]; output];
        for row in &mut w1 {
            for v in row {
                *v = rng.range(-0.20, 0.20);
            }
        }
        for row in &mut w2 {
            for v in row {
                *v = rng.range(-0.20, 0.20);
            }
        }
        Self { w1, b1: vec![0.0; hidden], w2, b2: vec![0.0; output] }
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut hidden_raw = vec![0.0; self.w1.len()];
        let mut hidden = vec![0.0; self.w1.len()];
        for i in 0..self.w1.len() {
            let mut s = self.b1[i];
            for (j, &xj) in x.iter().enumerate() {
                s += self.w1[i][j] * xj;
            }
            hidden_raw[i] = s;
            hidden[i] = relu(s);
        }
        let mut out = vec![0.0; self.w2.len()];
        for i in 0..self.w2.len() {
            let mut s = self.b2[i];
            for (j, &hj) in hidden.iter().enumerate() {
                s += self.w2[i][j] * hj;
            }
            out[i] = s;
        }
        (hidden_raw, hidden, out)
    }
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

fn weighted_score(outputs: &[u8], labels: &[(Vec<u8>, u8)], sw: &[f32]) -> f32 {
    outputs.iter().zip(labels).zip(sw)
        .map(|((&pred, (_, y)), &wt)| if pred == *y { wt } else { 0.0 })
        .sum()
}

fn best_single_signal_scores(all_sigs: &[Vec<u8>], data: &[(Vec<u8>, u8)], sw: &[f32], n_sig: usize) -> Vec<SignalScout> {
    let mut out = Vec::with_capacity(n_sig);
    for sig in 0..n_sig {
        let pos: f32 = data.iter().enumerate().zip(sw)
            .map(|((pi, (_, y)), &wt)| if all_sigs[pi][sig] == *y { wt } else { 0.0 })
            .sum();
        let neg: f32 = data.iter().enumerate().zip(sw)
            .map(|((pi, (_, y)), &wt)| if (1 - all_sigs[pi][sig]) == *y { wt } else { 0.0 })
            .sum();
        let (single_score, single_sign) = if pos >= neg { (pos, 1) } else { (neg, -1) };
        out.push(SignalScout { idx: sig, single_score, single_sign, probe_w: 0.0, rank_sum: 0 });
    }
    out
}

fn backprop_probe_all(all_sigs: &[Vec<u8>], data: &[(Vec<u8>, u8)], sw: &[f32], n_sig: usize, epochs: usize) -> (Vec<f32>, f32) {
    let mut w = vec![0.0f32; n_sig];
    let mut b = 0.0f32;
    for _ in 0..epochs {
        for (pi, (_, y)) in data.iter().enumerate() {
            let z = b + (0..n_sig).map(|i| w[i] * all_sigs[pi][i] as f32).sum::<f32>();
            let a = sigmoid(z);
            let g = (a - *y as f32) * sw[pi] * data.len() as f32;
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

fn best_small_ternary_score(parents: &[usize], all_sigs: &[Vec<u8>], data: &[(Vec<u8>, u8)], sw: &[f32]) -> f32 {
    let ni = parents.len();
    let total = 3u64.pow(ni as u32);
    let mut best = -1.0f32;
    for combo in 0..total {
        let mut r = combo;
        let mut w = vec![0i8; ni];
        for wi in &mut w { *wi = (r % 3) as i8 - 1; r /= 3; }
        let dots: Vec<i32> = (0..data.len()).map(|pi| {
            let mut d = 0i32;
            for (j, &pidx) in parents.iter().enumerate() {
                d += (w[j] as i32) * (all_sigs[pi][pidx] as i32);
            }
            d
        }).collect();
        let mn = dots.iter().copied().min().unwrap_or(0);
        let mx = dots.iter().copied().max().unwrap_or(0);
        for threshold in (mn - 1)..=(mx + 1) {
            let outs: Vec<u8> = dots.iter().map(|&d| if d >= threshold { 1 } else { 0 }).collect();
            let sc = weighted_score(&outs, data, sw);
            if sc > best { best = sc; }
        }
    }
    best
}

fn pair_lifts_from_ranked(ranked: &[SignalScout], all_sigs: &[Vec<u8>], data: &[(Vec<u8>, u8)], sw: &[f32], pair_top: usize) -> Vec<PairScout> {
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
    let mut p = parents;
    p.sort_unstable();
    p.dedup();
    if p.len() < 2 { return; }
    if seen.insert(p.clone()) { sets.push(p); }
}

fn build_candidate_sets(ranked: &[SignalScout], pairs: &[PairScout], n_sig: usize, cfg: &Config, step: usize) -> Vec<Vec<usize>> {
    let max_fan = cfg.max_fan.min(n_sig).max(2);
    let pool_n = ranked.len().min(cfg.scout_top.max(max_fan));
    let pool: Vec<usize> = ranked.iter().take(pool_n).map(|s| s.idx).collect();

    let mut sets = Vec::new();
    let mut seen = HashSet::new();
    for &sz in &[2usize, 3, 4, 6, 8, 10] {
        let sz = sz.min(max_fan).min(pool.len());
        if sz >= 2 {
            push_candidate(&mut sets, &mut seen, pool.iter().copied().take(sz).collect());
        }
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
    while sets.len() < cfg.proposals && !pool.is_empty() {
        let target = 2 + rng.pick(max_fan - 1);
        let mut cand = Vec::new();
        for _ in 0..pool.len() * 3 {
            if cand.len() >= target.min(pool.len()) { break; }
            let p = pool[rng.pick(pool.len())];
            if !cand.contains(&p) { cand.push(p); }
        }
        push_candidate(&mut sets, &mut seen, cand);
        if sets.len() >= cfg.proposals { break; }
    }
    sets
}

fn train_bit_head(bit: usize, samples: &[Sample], cfg: &Config) -> Net {
    let data: Vec<(Vec<u8>, u8)> = samples.iter()
        .map(|s| (s.input.clone(), (s.target_byte >> bit) & 1))
        .collect();

    let mut net = Net::new(INPUT_DIM);
    let mut sw = vec![1.0 / data.len() as f32; data.len()];
    let mut stall = 0usize;

    for step in 0..cfg.max_neurons {
        let ens_acc = net.accuracy(&data);
        let all_sigs: Vec<Vec<u8>> = data.iter().map(|(x, _)| net.eval_all(x)).collect();
        let n_sig = net.n_sig();
        let ranked = merge_signal_ranks(
            best_single_signal_scores(&all_sigs, &data, &sw, n_sig),
            &backprop_probe_all(&all_sigs, &data, &sw, n_sig, cfg.probe_epochs).0,
        );
        let pairs = pair_lifts_from_ranked(&ranked, &all_sigs, &data, &sw, cfg.pair_top);
        let proposal_sets = build_candidate_sets(&ranked, &pairs, n_sig, cfg, step + bit * 1000);

        let mut proposals: Vec<(Vec<usize>, Vec<f32>, f32, f32)> = Vec::new();
        let (_, probe_b) = backprop_probe_all(&all_sigs, &data, &sw, n_sig, cfg.probe_epochs);
        for (seed, parents) in proposal_sets.iter().enumerate() {
            let ni = parents.len();
            let mut rng = Rng::new(cfg.search_seed ^ (bit as u64 * 100_003 + step as u64 * 7919 + seed as u64));
            let w: Vec<f32> = parents.iter().map(|&p| {
                let base = ranked.iter().find(|r| r.idx == p).map(|r| r.probe_w).unwrap_or(0.0);
                if base.abs() > 0.05 { base + rng.range(-0.15, 0.15) }
                else {
                    let s = ranked.iter().find(|r| r.idx == p).map(|r| r.single_sign as f32).unwrap_or(1.0);
                    s * rng.range(0.1, 0.8)
                }
            }).collect();
            let b = probe_b + rng.range(-0.2, 0.2);
            let score: f32 = data.iter().enumerate().map(|(pi, (_, y))| {
                let z: f32 = b + (0..ni).map(|i| w[i] * all_sigs[pi][parents[i]] as f32).sum::<f32>();
                if (if sigmoid(z) > 0.5 { 1u8 } else { 0 }) == *y { sw[pi] } else { 0.0 }
            }).sum();
            proposals.push((parents.clone(), w, b, score));
        }
        proposals.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        if proposals.is_empty() { break; }

        struct Trained { parents: Vec<usize>, val_acc: f32, consensus: Vec<i8> }
        let mut trained = Vec::new();
        for pi in 0..proposals.len().min(5) {
            let (ref parents, ref init_w, init_b, _) = proposals[pi];
            let ni = parents.len();
            let mut all_converged = Vec::new();
            for restart in 0..4u64 {
                let mut rng = Rng::new(cfg.search_seed ^ (bit as u64 * 13 + step as u64 * 31 + restart));
                let mut w = if restart == 0 { init_w.clone() } else { init_w.iter().map(|&v| v + rng.range(-0.5, 0.5)).collect() };
                let mut b = if restart == 0 { init_b } else { init_b + rng.range(-0.3, 0.3) };
                for _ in 0..1200 {
                    for (pii, (_, y)) in data.iter().enumerate() {
                        let z: f32 = b + (0..ni).map(|i| w[i] * all_sigs[pii][parents[i]] as f32).sum::<f32>();
                        let a = sigmoid(z);
                        let g = (a - *y as f32) * a * (1.0 - a) * sw[pii] * data.len() as f32;
                        for i in 0..ni { w[i] -= 0.5 * g * all_sigs[pii][parents[i]] as f32; }
                        b -= 0.5 * g;
                    }
                }
                all_converged.push((w, b));
            }
            let consensus: Vec<i8> = (0..ni).map(|i| {
                let pos = all_converged.iter().filter(|(w, _)| w[i] > 0.3).count();
                let neg = all_converged.iter().filter(|(w, _)| w[i] < -0.3).count();
                if pos * 10 / 4 >= 7 { 1 } else if neg * 10 / 4 >= 7 { -1 } else { 2 }
            }).collect();
            let best_w = &all_converged[0].0;
            let best_b = all_converged[0].1;
            let val_acc = data.iter().enumerate().filter(|(vi, (_, y))| {
                let z: f32 = best_b + (0..ni).map(|i| best_w[i] * all_sigs[*vi][parents[i]] as f32).sum::<f32>();
                (if sigmoid(z) > 0.5 { 1u8 } else { 0 }) == *y
            }).count() as f32 / data.len() as f32 * 100.0;
            trained.push(Trained { parents: parents.clone(), val_acc, consensus });
        }
        trained.sort_by(|a, b| b.val_acc.partial_cmp(&a.val_acc).unwrap());

        let mut accepted = false;
        for tp in &trained {
            let ni = tp.parents.len();
            let locked: Vec<Option<i8>> = tp.consensus.iter().map(|&s| if s == 1 { Some(1) } else if s == -1 { Some(-1) } else { None }).collect();
            let free_pos: Vec<usize> = (0..ni).filter(|&i| locked[i].is_none()).collect();
            let mut best_w = vec![0i8; ni];
            let mut best_t = 0i32;
            let mut best_s = -1.0f32;
            let mut best_out = vec![0u8; data.len()];

            let total_free = 3u64.pow(free_pos.len() as u32);
            for combo in 0..total_free {
                let mut w = vec![0i8; ni];
                for i in 0..ni { w[i] = locked[i].unwrap_or(0); }
                let mut r = combo;
                for &fp in &free_pos { w[fp] = (r % 3) as i8 - 1; r /= 3; }
                let dots: Vec<i32> = (0..data.len()).map(|pi| {
                    let mut d = 0i32;
                    for (j, &pidx) in tp.parents.iter().enumerate() {
                        d += (w[j] as i32) * (all_sigs[pi][pidx] as i32);
                    }
                    d
                }).collect();
                let mn = dots.iter().copied().min().unwrap_or(0);
                let mx = dots.iter().copied().max().unwrap_or(0);
                for threshold in (mn - 1)..=(mx + 1) {
                    let outs: Vec<u8> = dots.iter().map(|&d| if d >= threshold { 1 } else { 0 }).collect();
                    let sc = weighted_score(&outs, &data, &sw);
                    if sc > best_s { best_s = sc; best_w = w.clone(); best_t = threshold; best_out = outs; }
                }
            }

            let is_dup = (net.n_in..n_sig).any(|e| {
                best_out.iter().enumerate().all(|(i, &v)| all_sigs[i][e] == v)
            });
            if is_dup { continue; }

            let tick = tp.parents.iter().map(|&p| net.sig_ticks[p]).max().unwrap_or(0) + 1;
            let werr: f32 = best_out.iter().zip(&data).zip(&sw)
                .map(|((&pred, (_, y)), &w)| if pred == *y { 0.0 } else { w }).sum();
            if werr >= 0.499 { continue; }
            let alpha = 0.5 * ((1.0 - werr).max(1e-6) / werr.max(1e-6)).ln();

            let neuron = Neuron {
                parents: tp.parents.clone(),
                tick,
                weights: best_w.clone(),
                threshold: best_t,
                alpha,
            };
            net.add(neuron);
            let new_acc = net.accuracy(&data);
            if new_acc < ens_acc {
                net.sig_ticks.pop();
                net.neurons.pop();
                continue;
            }

            let mut norm = 0.0f32;
            for ((pred, (_, y)), wt) in best_out.iter().zip(&data).zip(sw.iter_mut()) {
                let ys = if *y == 1 { 1.0 } else { -1.0 };
                let hs = if *pred == 1 { 1.0 } else { -1.0 };
                *wt *= (-alpha * ys * hs).exp();
                norm += *wt;
            }
            if norm > 0.0 {
                for w in &mut sw { *w /= norm; }
            }

            accepted = true;
            stall = 0;
            break;
        }

        if !accepted {
            stall += 1;
            if stall >= cfg.stall_limit { break; }
        }
    }

    net
}

fn predict_byte(heads: &[Net], input: &[u8]) -> u8 {
    let mut out = 0u8;
    for (bit, net) in heads.iter().enumerate() {
        if net.predict(input) == 1 { out |= 1 << bit; }
    }
    out
}

fn charge_features(heads: &[Net], input: &[u8]) -> Vec<f32> {
    heads.iter().map(|net| net.score(input)).collect()
}

fn full_latent_features(heads: &[Net], input: &[u8]) -> Vec<f32> {
    let mut out = Vec::new();
    for net in heads {
        let sigs = net.eval_all(input);
        out.push(net.score_from_sigs(&sigs));
        out.extend(sigs[net.n_in..].iter().map(|&v| v as f32));
    }
    out
}

fn binary_latent_signature(heads: &[Net], input: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for net in heads {
        let sigs = net.eval_all(input);
        out.extend_from_slice(&sigs[net.n_in..]);
    }
    out
}

fn target_bits(byte: u8) -> [u8; 8] {
    let mut out = [0u8; 8];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (byte >> i) & 1;
    }
    out
}

fn train_translator(features: &[Vec<f32>], samples: &[Sample], hidden: usize, epochs: usize, lr: f32, seed: u64) -> Mlp {
    let input_dim = features.first().map(|f| f.len()).unwrap_or(0);
    let mut model = Mlp::new(input_dim, hidden, 8, seed);
    let mut order: Vec<usize> = (0..features.len()).collect();
    let mut rng = Rng::new(seed ^ 0xD1B5_4A32);
    for _ in 0..epochs {
        for i in (1..order.len()).rev() {
            let j = rng.pick(i + 1);
            order.swap(i, j);
        }
        for &idx in &order {
            let x = &features[idx];
            let target = target_bits(samples[idx].target_byte);
            let (hidden_raw, hidden_act, out_raw) = model.forward(x);
            let out_act: Vec<f32> = out_raw.iter().map(|&v| sigmoid(v)).collect();
            let mut grad_out = vec![0.0f32; 8];
            for i in 0..8 {
                let err = out_act[i] - target[i] as f32;
                grad_out[i] = err;
            }
            let mut grad_hidden = vec![0.0f32; hidden];
            for h in 0..hidden {
                let mut s = 0.0;
                for o in 0..8 {
                    s += grad_out[o] * model.w2[o][h];
                }
                grad_hidden[h] = if hidden_raw[h] > 0.0 { s } else { 0.0 };
            }
            for o in 0..8 {
                for h in 0..hidden {
                    model.w2[o][h] -= lr * grad_out[o] * hidden_act[h];
                }
                model.b2[o] -= lr * grad_out[o];
            }
            for h in 0..hidden {
                for (j, &xj) in x.iter().enumerate() {
                    model.w1[h][j] -= lr * grad_hidden[h] * xj;
                }
                model.b1[h] -= lr * grad_hidden[h];
            }
        }
    }
    model
}

fn translator_predict(model: &Mlp, features: &[f32]) -> u8 {
    let (_, _, out_raw) = model.forward(features);
    let mut out = 0u8;
    for (bit, &logit) in out_raw.iter().enumerate().take(8) {
        if sigmoid(logit) >= 0.5 { out |= 1 << bit; }
    }
    out
}

struct TranslatorReport {
    exact_acc: f32,
    per_op: [f32; 4],
    misses: Vec<(Opcode, u8, u8, u8)>,
}

fn eval_translator(model: &Mlp, features: &[Vec<f32>], samples: &[Sample]) -> TranslatorReport {
    let mut exact = 0usize;
    let mut per_op_ok = [0usize; 4];
    let mut per_op_total = [0usize; 4];
    let mut misses = Vec::new();
    for (idx, sample) in samples.iter().enumerate() {
        let pred = translator_predict(model, &features[idx]);
        if pred == sample.target_byte { exact += 1; }
        else if misses.len() < 12 {
            let input_byte = sample.input[..DATA_BITS].iter().enumerate()
                .fold(0u8, |acc, (i, &b)| acc | ((b & 1) << i));
            misses.push((sample.opcode, input_byte, pred, sample.target_byte));
        }
        per_op_ok[sample.opcode.idx()] += usize::from(pred == sample.target_byte);
        per_op_total[sample.opcode.idx()] += 1;
    }
    let mut per_op = [0.0f32; 4];
    for op in Opcode::all() {
        per_op[op.idx()] = per_op_ok[op.idx()] as f32 / per_op_total[op.idx()] as f32 * 100.0;
    }
    TranslatorReport {
        exact_acc: exact as f32 / samples.len() as f32 * 100.0,
        per_op,
        misses,
    }
}

struct LutReport {
    exact_acc: f32,
    per_op: [f32; 4],
    distinct_keys: usize,
    conflicting_keys: usize,
}

fn eval_lut_translator(heads: &[Net], samples: &[Sample]) -> LutReport {
    let mut counts: HashMap<Vec<u8>, Vec<u16>> = HashMap::new();
    for sample in samples {
        let sig = binary_latent_signature(heads, &sample.input);
        let bucket = counts.entry(sig).or_insert_with(|| vec![0u16; 256]);
        bucket[sample.target_byte as usize] += 1;
    }

    let mut lut: HashMap<Vec<u8>, u8> = HashMap::new();
    let mut conflicting_keys = 0usize;
    for (sig, bucket) in &counts {
        let mut best_idx = 0usize;
        let mut best_count = 0u16;
        let mut nonzero = 0usize;
        for (i, &count) in bucket.iter().enumerate() {
            if count > 0 {
                nonzero += 1;
                if count > best_count {
                    best_count = count;
                    best_idx = i;
                }
            }
        }
        if nonzero > 1 {
            conflicting_keys += 1;
        }
        lut.insert(sig.clone(), best_idx as u8);
    }

    let mut exact = 0usize;
    let mut per_op_ok = [0usize; 4];
    let mut per_op_total = [0usize; 4];
    for sample in samples {
        let sig = binary_latent_signature(heads, &sample.input);
        let pred = *lut.get(&sig).unwrap_or(&0);
        if pred == sample.target_byte { exact += 1; }
        per_op_ok[sample.opcode.idx()] += usize::from(pred == sample.target_byte);
        per_op_total[sample.opcode.idx()] += 1;
    }

    let mut per_op = [0.0f32; 4];
    for op in Opcode::all() {
        per_op[op.idx()] = per_op_ok[op.idx()] as f32 / per_op_total[op.idx()] as f32 * 100.0;
    }
    LutReport {
        exact_acc: exact as f32 / samples.len() as f32 * 100.0,
        per_op,
        distinct_keys: counts.len(),
        conflicting_keys,
    }
}

fn main() {
    let cfg = parse_args();
    let samples = dataset();
    let mut heads = Vec::with_capacity(8);

    println!("=== BYTE + OPCODE GROWER ===");
    println!("Domain: 1 byte + 4 opcode -> 1 byte (exact 1024 samples)");
    println!(
        "Config: max_neurons={} max_fan={} proposals={} stall={} scout_top={} pair_top={} probe_epochs={} translator_hidden={} translator_epochs={} translator_lr={} search_seed={}",
        cfg.max_neurons, cfg.max_fan, cfg.proposals, cfg.stall_limit, cfg.scout_top, cfg.pair_top, cfg.probe_epochs,
        cfg.translator_hidden, cfg.translator_epochs, cfg.translator_lr, cfg.search_seed
    );

    for bit in 0..8 {
        let net = train_bit_head(bit, &samples, &cfg);
        let data: Vec<(Vec<u8>, u8)> = samples.iter().map(|s| (s.input.clone(), (s.target_byte >> bit) & 1)).collect();
        let acc = net.accuracy(&data);
        let depth = net.neurons.iter().map(|n| n.tick).max().unwrap_or(0);
        println!(
            "  bit{}: acc={:.1}% neurons={} depth={} hidden={}",
            bit,
            acc,
            net.neurons.len(),
            depth,
            net.neurons.iter().any(|n| n.parents.iter().any(|&p| p >= INPUT_DIM))
        );
        heads.push(net);
    }

    let mut exact = 0usize;
    let mut per_op_ok = [0usize; 4];
    let mut per_op_total = [0usize; 4];
    for sample in &samples {
        let pred = predict_byte(&heads, &sample.input);
        if pred == sample.target_byte { exact += 1; }
        per_op_ok[sample.opcode.idx()] += usize::from(pred == sample.target_byte);
        per_op_total[sample.opcode.idx()] += 1;
    }

    println!("\n=== COMBINED BYTE RESULT ===");
    println!("  exact byte acc: {:.1}%", exact as f32 / samples.len() as f32 * 100.0);
    for op in Opcode::all() {
        println!(
            "  {}: {:.1}%",
            op.name(),
            per_op_ok[op.idx()] as f32 / per_op_total[op.idx()] as f32 * 100.0
        );
    }

    println!("  adversarial:");
    for &x in &ADV_CASES {
        for op in Opcode::all() {
            let mut input = Vec::with_capacity(INPUT_DIM);
            input.extend(bits8(x));
            for oi in 0..OPCODES {
                input.push(if oi == op.idx() { 1 } else { 0 });
            }
            let pred = predict_byte(&heads, &input);
            let target = op.apply(x);
            println!(
                "    {}({:#04X}) -> pred={:#04X} target={:#04X} {}",
                op.name(),
                x,
                pred,
                target,
                if pred == target { "OK" } else { "MISS" }
            );
        }
    }

    let max_depth = heads.iter().flat_map(|n| n.neurons.iter().map(|nn| nn.tick)).max().unwrap_or(0);
    let total_neurons: usize = heads.iter().map(|n| n.neurons.len()).sum();
    println!("\n  total neurons across heads: {}", total_neurons);
    println!("  max head depth: {}", max_depth);

    let charge_latents: Vec<Vec<f32>> = samples.iter().map(|s| charge_features(&heads, &s.input)).collect();
    let full_latents: Vec<Vec<f32>> = samples.iter().map(|s| full_latent_features(&heads, &s.input)).collect();

    let charge_model = train_translator(
        &charge_latents,
        &samples,
        cfg.translator_hidden,
        cfg.translator_epochs,
        cfg.translator_lr,
        cfg.search_seed ^ 0xA11CE001,
    );
    let full_model = train_translator(
        &full_latents,
        &samples,
        cfg.translator_hidden,
        cfg.translator_epochs,
        cfg.translator_lr,
        cfg.search_seed ^ 0xA11CE002,
    );
    let charge_report = eval_translator(&charge_model, &charge_latents, &samples);
    let full_report = eval_translator(&full_model, &full_latents, &samples);
    let lut_report = eval_lut_translator(&heads, &samples);

    println!("\n=== FROZEN TRANSLATOR HEADS ===");
    println!("  charge-only translator: {:.1}%", charge_report.exact_acc);
    for op in Opcode::all() {
        println!("    {}: {:.1}%", op.name(), charge_report.per_op[op.idx()]);
    }
    println!("  full-latent translator: {:.1}%", full_report.exact_acc);
    for op in Opcode::all() {
        println!("    {}: {:.1}%", op.name(), full_report.per_op[op.idx()]);
    }
    println!(
        "  latent-LUT translator: {:.1}% (distinct_keys={} conflicting_keys={})",
        lut_report.exact_acc,
        lut_report.distinct_keys,
        lut_report.conflicting_keys
    );
    for op in Opcode::all() {
        println!("    {}: {:.1}%", op.name(), lut_report.per_op[op.idx()]);
    }

    if !full_report.misses.is_empty() {
        println!("  full-latent misses (first {}):", full_report.misses.len());
        for (op, x, pred, target) in &full_report.misses {
            println!(
                "    {}({:#04X}) -> pred={:#04X} target={:#04X}",
                op.name(),
                x,
                pred,
                target
            );
        }
    }
}
