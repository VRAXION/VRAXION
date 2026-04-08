//! Reward-modulated Hebbian learning on addition.
//! NO try-keep-revert. Each synapse learns LOCALLY.
//!
//! RUNNING: hebbian_addition
//!
//! Run: cargo run --example hebbian_addition --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIGITS: usize = 5;
const SUMS: usize = 9;
const H: usize = 64;
const TICKS: usize = 50;
const INPUT_TICKS: usize = 20;
const INPUT_END: usize = 16;
const OUTPUT_START: usize = 32;
const EPOCHS: usize = 200;  // how many passes through all examples

#[derive(Clone)]
struct HebbNet {
    edges: Vec<(u16, u16, i16)>,  // (src, tgt, weight)
    threshold: Vec<i32>,
    polarity: Vec<i8>,
    // LIF state
    g: Vec<i32>,
    v: Vec<i32>,
    refractory: Vec<u8>,
    firing: Vec<bool>,
    spike_count: Vec<u32>,
    // Per-edge activity tracking for Hebbian update
    pre_active: Vec<bool>,   // did source fire this example?
    post_active: Vec<bool>,  // did target fire this example?
    h: usize,
}

impl HebbNet {
    fn new(h: usize, rng: &mut impl Rng) -> Self {
        let mut threshold = vec![0i32; h]; let mut polarity = vec![1i8; h];
        for i in 0..h {
            threshold[i] = rng.gen_range(5..=10);
            if rng.gen_ratio(3, 10) { polarity[i] = -1; }
        }
        HebbNet {
            edges: Vec::new(), threshold, polarity,
            g: vec![0; h], v: vec![0; h], refractory: vec![0; h],
            firing: vec![false; h], spike_count: vec![0; h],
            pre_active: Vec::new(), post_active: Vec::new(), h,
        }
    }

    fn reset_state(&mut self) {
        // Reset runtime state but NOT weights/topology (no revert!)
        self.g.iter_mut().for_each(|x| *x = 0);
        self.v.iter_mut().for_each(|x| *x = 0);
        self.refractory.iter_mut().for_each(|x| *x = 0);
        self.firing.iter_mut().for_each(|x| *x = false);
        self.spike_count.iter_mut().for_each(|x| *x = 0);
    }

    fn step(&mut self) {
        let h = self.h;
        let mut g_in = vec![0i32; h];
        for &(src, tgt, weight) in &self.edges {
            let s = src as usize; let t = tgt as usize;
            if s >= h || t >= h { continue; }
            if self.firing[s] {
                g_in[t] += weight as i32 * self.polarity[s] as i32;
            }
        }
        for i in 0..h {
            if self.refractory[i] > 0 { self.refractory[i] -= 1; self.firing[i] = false; continue; }
            self.g[i] += g_in[i];
            self.g[i] -= self.g[i] / 5;
            self.v[i] += (-self.v[i] + self.g[i]) / 20;
            if self.v[i] >= self.threshold[i] {
                self.firing[i] = true; self.v[i] = 0; self.g[i] = 0;
                self.refractory[i] = 2; self.spike_count[i] += 1;
            } else { self.firing[i] = false; }
        }
    }

    fn inject_thermo(&mut self, a: usize, b: usize) {
        for i in 0..a.min(8) { self.g[i] += 5; }
        for i in 0..b.min(8) { self.g[8 + i] += 5; }
    }

    fn readout(&self) -> usize {
        let zone_len = self.h - OUTPUT_START;
        let mut scores = vec![0u32; SUMS];
        for i in 0..zone_len {
            let class = i * SUMS / zone_len;
            scores[class] += self.spike_count[OUTPUT_START + i];
        }
        scores.iter().enumerate().max_by_key(|&(_, v)| *v).map(|(i, _)| i).unwrap_or(0)
    }

    /// Run one example, return (prediction, was_correct)
    fn run_example(&mut self, a: usize, b: usize, target: usize) -> (usize, bool) {
        self.reset_state();
        for tick in 0..TICKS {
            if tick < INPUT_TICKS { self.inject_thermo(a, b); }
            self.step();
        }
        let pred = self.readout();
        (pred, pred == target)
    }

    /// Reward-modulated Hebbian update: modify weights based on co-activation + reward
    fn hebbian_update(&mut self, reward: i16) {
        // Track which neurons were active this example
        let active: Vec<bool> = self.spike_count.iter().map(|&s| s > 0).collect();

        for edge in &mut self.edges {
            let src = edge.0 as usize;
            let tgt = edge.1 as usize;
            if src >= self.h || tgt >= self.h { continue; }

            // Hebbian rule: if BOTH pre and post were active
            if active[src] && active[tgt] {
                // Reward modulation: strengthen if reward, weaken if punishment
                edge.2 = (edge.2 + reward).max(-64).min(64);
            }
            // Anti-Hebbian: if pre active but post NOT active, slight weakening
            // (decorrelation — prevents everything from strengthening)
            if active[src] && !active[tgt] && reward < 0 {
                edge.2 = (edge.2 - 1).max(-64);
            }
        }
    }

    fn add_edge(&mut self, s: u16, t: u16, w: i16) -> bool {
        if s == t || s as usize >= self.h || t as usize >= self.h { return false; }
        if self.edges.iter().any(|&(es,et,_)| es == s && et == t) { return false; }
        self.edges.push((s, t, w)); true
    }
}

fn eval_all(net: &mut HebbNet, examples: &[(usize,usize,usize)]) -> (f64, Vec<bool>) {
    let mut results = Vec::new();
    for &(a, b, target) in examples {
        let (_, correct) = net.run_example(a, b, target);
        results.push(correct);
    }
    let acc = results.iter().filter(|&&c| c).count() as f64 / results.len() as f64;
    (acc, results)
}

fn main() {
    let all: Vec<_> = (0..DIGITS).flat_map(|a| (0..DIGITS).map(move |b| (a, b, a+b))).collect();
    let train: Vec<_> = all.iter().filter(|&&(_,_,s)| s != 4).cloned().collect();
    let test: Vec<_> = all.iter().filter(|&&(_,_,s)| s == 4).cloned().collect();

    println!("=== REWARD-MODULATED HEBBIAN LEARNING ===");
    println!("RUNNING: hebbian_addition");
    println!("NO try-keep-revert. Each synapse learns LOCALLY.");
    println!("Rule: if pre+post both fire → weight += reward");
    println!("H={}, ticks={}, {} epochs over {} train examples\n", H, TICKS, EPOCHS, train.len());

    for &seed in &[42u64, 1042, 2042] {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net = HebbNet::new(H, &mut rng);

        // Init: dense bridges from input → hidden → output
        for _ in 0..50 {
            let s = rng.gen_range(0..INPUT_END) as u16;
            let t = rng.gen_range(INPUT_END..OUTPUT_START) as u16;
            net.add_edge(s, t, rng.gen_range(1..=3));
        }
        for _ in 0..50 {
            let s = rng.gen_range(INPUT_END..OUTPUT_START) as u16;
            let t = rng.gen_range(OUTPUT_START..H) as u16;
            net.add_edge(s, t, rng.gen_range(1..=3));
        }
        // Some direct input→output
        for _ in 0..20 {
            let s = rng.gen_range(0..INPUT_END) as u16;
            let t = rng.gen_range(OUTPUT_START..H) as u16;
            net.add_edge(s, t, rng.gen_range(1..=2));
        }

        println!("--- seed {} ({} edges) ---", seed, net.edges.len());

        // Train: present examples, apply Hebbian update with reward
        for epoch in 0..EPOCHS {
            // Shuffle training examples each epoch
            let mut order: Vec<usize> = (0..train.len()).collect();
            for i in (1..order.len()).rev() { let j = rng.gen_range(0..=i); order.swap(i, j); }

            let mut epoch_correct = 0;
            for &idx in &order {
                let (a, b, target) = train[idx];
                let (pred, correct) = net.run_example(a, b, target);

                // Reward signal
                let reward: i16 = if correct { 1 } else { -1 };

                // Hebbian update — LOCAL, no global eval
                net.hebbian_update(reward);

                if correct { epoch_correct += 1; }
            }

            // Report every 20 epochs
            if epoch % 20 == 0 || epoch == EPOCHS - 1 {
                let (train_acc, _) = eval_all(&mut net, &train);
                let (test_acc, _) = eval_all(&mut net, &test);
                let (all_acc, _) = eval_all(&mut net, &all);
                let mean_w: f64 = net.edges.iter().map(|e| e.2 as f64).sum::<f64>() / net.edges.len() as f64;
                println!("  epoch {:>3}: train={:.0}% test={:.0}% all={:.0}% | epoch_hit={}/{} mean_w={:.1}",
                    epoch, train_acc*100.0, test_acc*100.0, all_acc*100.0, epoch_correct, train.len(), mean_w);
            }
        }
        println!();
    }
}
