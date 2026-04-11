//! ONE network, growing neuron by neuron.
//!
//! Per neuron:
//!   Step 0: EXHAUSTIVE parent search — try MANY connection sets, find best
//!   Step 1: Lock parent set → backprop from multiple starts → landscape
//!   Step 2: Guided + blind ternary on locked parents
//!   Step 3: Accept gate → freeze → checkpoint
//!
//! cargo run --example neuron_one --release

use std::io::Write as IoWrite;
use std::time::Instant;

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

const FONT: [[u8; 9]; 10] = [
    [1,1,1,1,0,1,1,1,1],[0,1,0,0,1,0,0,1,0],[1,1,0,0,1,0,0,1,1],
    [1,1,0,0,1,0,1,1,0],[1,0,1,1,1,1,0,0,1],[0,1,1,0,1,0,1,1,0],
    [1,0,0,1,1,0,1,1,0],[1,1,1,0,0,1,0,0,1],[1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,0,1,1],
];

struct Data { train: Vec<(Vec<u8>, u8)>, val: Vec<(Vec<u8>, u8)>, test: Vec<(Vec<u8>, u8)> }

fn gen_data(noise: f32, n_per: usize, seed: u64) -> Data {
    let mut r = Rng::new(seed);
    let (mut tr, mut va, mut te) = (vec![], vec![], vec![]);
    for d in 0..10 { for i in 0..n_per {
        let mut px = FONT[d].to_vec();
        for p in px.iter_mut() { if r.bool_p(noise) { *p = 1 - *p; } }
        let pop: usize = px.iter().map(|&v| v as usize).sum();
        let label = (pop % 2) as u8;
        match i % 5 { 0 => va.push((px, label)), 1 => te.push((px, label)), _ => tr.push((px, label)) }
    }}
    Data { train: tr, val: va, test: te }
}

#[derive(Clone)]
struct Neuron { id: usize, parents: Vec<usize>, tick: u32, weights: Vec<i8>, bias: i8, threshold: i32, alpha: f32 }

impl Neuron {
    fn eval(&self, sigs: &[u8]) -> u8 {
        let mut d = self.bias as i32;
        for (&w, &p) in self.weights.iter().zip(&self.parents) { d += (w as i32) * (sigs[p] as i32); }
        if d >= self.threshold { 1 } else { 0 }
    }
}

struct Net { neurons: Vec<Neuron>, n_in: usize, ticks: Vec<u32> }
impl Net {
    fn new(n: usize) -> Self { Net { neurons: vec![], n_in: n, ticks: vec![0; n] } }
    fn n_sig(&self) -> usize { self.n_in + self.neurons.len() }
    fn eval_all(&self, inp: &[u8]) -> Vec<u8> {
        let mut s = inp.to_vec();
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
    fn add(&mut self, n: Neuron) { self.ticks.push(n.tick); self.neurons.push(n); }
}

// ══════════════════════════════════════════════════════
// STEP 0: EXHAUSTIVE PARENT SEARCH
// Try as many connection sets as possible, score each
// with a quick ternary search, return ranked parent sets
// ══════════════════════════════════════════════════════

fn exhaustive_parent_search(
    train_sigs: &[Vec<u8>], data: &[(Vec<u8>, u8)], sw: &[f32],
    n_sig: usize, max_fan: usize, n_tries: usize,
) -> Vec<(Vec<usize>, f32)> {
    let mut results: Vec<(Vec<usize>, f32)> = Vec::new();

    for seed in 0..n_tries as u64 {
        let mut rng = Rng::new(seed * 7919 + 31);

        // Random parent set: 2..max_fan parents
        let np = 2 + rng.pick(max_fan.min(n_sig) - 1);
        let np = np.min(n_sig);
        let mut parents = Vec::new();
        for _ in 0..np * 3 {
            if parents.len() >= np { break; }
            let p = rng.pick(n_sig);
            if !parents.contains(&p) { parents.push(p); }
        }
        if parents.len() < 2 { continue; }

        // Score this parent set with quick ternary search
        let score = quick_ternary_score(train_sigs, data, &parents, sw);
        results.push((parents, score));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.dedup_by(|a, b| {
        let mut sa = a.0.clone(); sa.sort();
        let mut sb = b.0.clone(); sb.sort();
        sa == sb
    });
    results
}

fn quick_ternary_score(
    sigs: &[Vec<u8>], data: &[(Vec<u8>, u8)], parents: &[usize], sw: &[f32],
) -> f32 {
    let ni = parents.len();
    let np = data.len();
    let total = 3u64.pow((ni + 1) as u32);
    let mut best = 0.0f32;

    for combo in 0..total {
        let mut w = vec![0i8; ni]; let mut r = combo;
        for wi in w.iter_mut() { *wi = (r % 3) as i8 - 1; r /= 3; }
        let b = (r % 3) as i8 - 1;
        let dots: Vec<i32> = (0..np).map(|pi| {
            let mut d = b as i32;
            for (wi, &p) in w.iter().zip(parents) { d += (*wi as i32) * (sigs[pi][p] as i32); }
            d
        }).collect();
        let (mn, mx) = (dots.iter().copied().min().unwrap_or(0), dots.iter().copied().max().unwrap_or(0));
        for t in (mn - 1)..=(mx + 1) {
            let sc: f32 = dots.iter().zip(data).zip(sw).map(|((&d, (_, y)), &wt)| {
                if (if d >= t { 1u8 } else { 0 }) == *y { wt } else { 0.0 }
            }).sum();
            if sc > best { best = sc; }
        }
    }
    best
}

// ══════════════════════════════════════════════════════
// STEP 1: BACKPROP on locked parent set → landscape
// ══════════════════════════════════════════════════════

fn backprop_landscape(
    train_sigs: &[Vec<u8>], data: &[(Vec<u8>, u8)], parents: &[usize], sw: &[f32],
    n_restarts: usize,
) -> Vec<i8> {
    let ni = parents.len();
    let mut sign_counts = vec![[0usize; 3]; ni];

    for restart in 0..n_restarts {
        let mut rng = Rng::new(restart as u64 * 1000 + 77);
        let mut w: Vec<f32> = (0..ni).map(|_| rng.range(-2.0, 2.0)).collect();
        let mut b = rng.range(-1.0, 1.0);

        for _ in 0..2000 {
            for (pi, (_, y)) in data.iter().enumerate() {
                let z: f32 = b + (0..ni).map(|i| w[i] * train_sigs[pi][parents[i]] as f32).sum::<f32>();
                let a = sigmoid(z);
                let g = (a - *y as f32) * a * (1.0 - a) * sw[pi] * data.len() as f32;
                for i in 0..ni { w[i] -= 0.5 * g * train_sigs[pi][parents[i]] as f32; }
                b -= 0.5 * g;
            }
        }

        for i in 0..ni {
            if w[i] < -0.3 { sign_counts[i][0] += 1; }
            else if w[i] > 0.3 { sign_counts[i][2] += 1; }
            else { sign_counts[i][1] += 1; }
        }
    }

    // Consensus
    sign_counts.iter().map(|s| {
        let tot = (s[0] + s[1] + s[2]).max(1);
        if s[2] * 10 / tot >= 7 { 1i8 }
        else if s[0] * 10 / tot >= 7 { -1 }
        else { 2 } // no consensus = FREE
    }).collect()
}

// ══════════════════════════════════════════════════════
// STEP 2: TERNARY (guided + blind)
// ══════════════════════════════════════════════════════

fn full_ternary(
    sigs: &[Vec<u8>], data: &[(Vec<u8>, u8)], parents: &[usize], sw: &[f32],
) -> (Vec<i8>, i8, i32, Vec<u8>) {
    let ni = parents.len();
    let np = data.len();
    let total = 3u64.pow((ni + 1) as u32);
    let (mut bw, mut bb, mut bt, mut bo) = (vec![0i8; ni], 0i8, 0i32, vec![0u8; np]);
    let mut bs = -1.0f32;

    for combo in 0..total {
        let mut w = vec![0i8; ni]; let mut r = combo;
        for wi in w.iter_mut() { *wi = (r % 3) as i8 - 1; r /= 3; }
        let b = (r % 3) as i8 - 1;
        let dots: Vec<i32> = (0..np).map(|pi| {
            let mut d = b as i32;
            for (wi, &p) in w.iter().zip(parents) { d += (*wi as i32) * (sigs[pi][p] as i32); }
            d
        }).collect();
        let (mn, mx) = (dots.iter().copied().min().unwrap_or(0), dots.iter().copied().max().unwrap_or(0));
        for t in (mn - 1)..=(mx + 1) {
            let outs: Vec<u8> = dots.iter().map(|&d| if d >= t { 1 } else { 0 }).collect();
            let sc: f32 = outs.iter().zip(data).zip(sw)
                .map(|((&p, (_, y)), &wt)| if p == *y { wt } else { 0.0 }).sum();
            if sc > bs { bs = sc; bw = w.clone(); bb = b; bt = t; bo = outs; }
        }
    }
    (bw, bb, bt, bo)
}

// ══════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════

fn save(net: &Net, data: &Data, path: &str) {
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "{{\"ensemble_train\":{:.2},\"ensemble_val\":{:.2},\"ensemble_test\":{:.2},\"neurons\":[",
        net.accuracy(&data.train), net.accuracy(&data.val), net.accuracy(&data.test)).unwrap();
    for (i, n) in net.neurons.iter().enumerate() {
        let wj: Vec<String> = n.weights.iter().map(|v| v.to_string()).collect();
        let pj: Vec<String> = n.parents.iter().map(|v| v.to_string()).collect();
        writeln!(f, "{{\"id\":{},\"parents\":[{}],\"tick\":{},\"weights\":[{}],\"bias\":{},\"threshold\":{},\"alpha\":{:.6}}}{}",
            n.id, pj.join(","), n.tick, wj.join(","), n.bias, n.threshold, n.alpha,
            if i < net.neurons.len() - 1 { "," } else { "" }).unwrap();
    }
    writeln!(f, "]}}").unwrap();
}

fn main() {
    let out = "results/neuron_one";
    std::fs::create_dir_all(out).unwrap();
    let log_path = format!("{}/growth.log", out);
    // Clear old log
    std::fs::write(&log_path, "").ok();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  ONE NETWORK — continuous growth                            ║");
    println!("║  Step 0: exhaustive parent search (connection discovery)     ║");
    println!("║  Step 1: backprop landscape on locked parents               ║");
    println!("║  Step 2: ternary quantization (blind exhaustive)            ║");
    println!("║  Step 3: accept gate → freeze → checkpoint                  ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let data = gen_data(0.15, 100, 42);
    let mut net = Net::new(9);
    let mut sw = vec![1.0 / data.train.len() as f32; data.train.len()];
    let max_neurons = 200;
    let max_fan = 10;
    let n_parent_tries = 200;  // how many random parent sets to try
    let mut stall = 0;

    println!("  Data: {} train / {} val / {} test", data.train.len(), data.val.len(), data.test.len());
    println!("  Parent search: {} random connection sets per neuron", n_parent_tries);
    println!("  Max fan-in: {}\n", max_fan);

    let t_total = Instant::now();

    for step in 0..max_neurons {
        let t0 = Instant::now();
        let ev = net.accuracy(&data.val);
        let et = net.accuracy(&data.test);

        let status = format!("N{:03} │ val={:.1}% test={:.1}% │ depth={} │ {:.0}s elapsed",
            net.neurons.len(), ev, et,
            net.neurons.iter().map(|n| n.tick).max().unwrap_or(0),
            t_total.elapsed().as_secs_f64());
        println!("  {}", status);
        if let Ok(mut lf) = std::fs::OpenOptions::new().create(true).append(true).open(&log_path) {
            writeln!(lf, "{}", status).ok();
        }

        if ev >= 99.0 { println!("  ✓ TARGET REACHED"); break; }

        // Precompute all signals
        let train_sigs: Vec<Vec<u8>> = data.train.iter().map(|(x, _)| net.eval_all(x)).collect();
        let ns = net.n_sig();
        let ensemble_val = ev;

        // ── STEP 0: Exhaustive parent search ──
        let t_ps = Instant::now();
        let parent_candidates = exhaustive_parent_search(
            &train_sigs, &data.train, &sw, ns, max_fan, n_parent_tries,
        );
        let ps_ms = t_ps.elapsed().as_millis();

        if parent_candidates.is_empty() { println!("    ✗ no parent sets found"); break; }

        println!("    Step 0: {} parent sets scored in {}ms (top score={:.4})",
            parent_candidates.len().min(n_parent_tries), ps_ms,
            parent_candidates[0].1);

        // ── Try top parent sets until one is accepted ──
        let mut accepted = false;

        for (rank, (parents, _pscore)) in parent_candidates.iter().take(20).enumerate() {
            // ── STEP 1: Backprop landscape ──
            let _consensus = backprop_landscape(&train_sigs, &data.train, parents, &sw, 5);

            // ── STEP 2: Full ternary (blind, exhaustive on this parent set) ──
            let (tw, tb, tt, outputs) = full_ternary(&train_sigs, &data.train, parents, &sw);

            // Duplicate check
            let is_dup = (net.n_in..ns).any(|e| {
                outputs.iter().enumerate().filter(|(i, v)| train_sigs[*i][e] == **v).count() as f32
                    / outputs.len() as f32 >= 0.999
            });
            if is_dup { continue; }

            // Weighted error + alpha
            let werr: f32 = outputs.iter().zip(&data.train).zip(&sw)
                .map(|((&p, (_, y)), &w)| if p == *y { 0.0 } else { w }).sum();
            if werr >= 0.499 { continue; }
            let alpha = 0.5 * ((1.0 - werr).max(1e-6) / werr.max(1e-6)).ln();

            // ── STEP 3: Accept gate ──
            let tick = parents.iter().map(|&p| net.ticks[p]).max().unwrap_or(0) + 1;
            let neuron = Neuron {
                id: net.neurons.len(), parents: parents.clone(), tick,
                weights: tw.clone(), bias: tb, threshold: tt, alpha,
            };

            // Temp add, check ensemble val
            net.add(neuron);
            let new_val = net.accuracy(&data.val);

            if new_val <= ensemble_val {
                net.ticks.pop(); net.neurons.pop();
                continue;
            }

            // ── ACCEPTED ──
            let has_hidden = parents.iter().any(|&p| p >= net.n_in);
            let pnames: Vec<String> = parents.iter().map(|&p| {
                if p < 9 { format!("x{}", p) } else { format!("N{}", p - 9) }
            }).collect();
            let wstr: String = tw.iter().map(|&v| match v { 1=>"+", -1=>"-", _=>"0" }).collect::<Vec<_>>().join("");

            println!("    ✓ [{}] tick={} parents=[{}] hidden={} val={:.1}→{:.1}% (rank={}, {:.0}ms)",
                wstr, tick, pnames.join(","), has_hidden, ensemble_val, new_val, rank, t0.elapsed().as_millis());

            // Reweight (AdaBoost)
            let mut norm = 0.0f32;
            for ((pred, (_, y)), wt) in outputs.iter().zip(&data.train).zip(sw.iter_mut()) {
                let ys = if *y == 1 { 1.0 } else { -1.0 };
                let hs = if *pred == 1 { 1.0 } else { -1.0 };
                *wt *= (-alpha * ys * hs).exp();
                norm += *wt;
            }
            if norm > 0.0 { for w in &mut sw { *w /= norm; } }

            // Checkpoint
            save(&net, &data, &format!("{}/net_{:03}.json", out, net.neurons.len()));
            save(&net, &data, &format!("{}/latest.json", out));

            stall = 0;
            accepted = true;
            break;
        }

        if !accepted {
            println!("    ✗ no improvement ({:.0}ms)", t0.elapsed().as_millis());
            stall += 1;
            if stall >= 30 { println!("  ✗ Stalled 30 steps, stopping."); break; }
        }
    }

    save(&net, &data, &format!("{}/final.json", out));
    println!("\n  DONE: {} neurons, depth={}, val={:.1}%, test={:.1}%, {:.0}s",
        net.neurons.len(),
        net.neurons.iter().map(|n| n.tick).max().unwrap_or(0),
        net.accuracy(&data.val), net.accuracy(&data.test),
        t_total.elapsed().as_secs_f64());
}
