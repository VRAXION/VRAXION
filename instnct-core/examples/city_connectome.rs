//! City Connectome: hierarchical vs flat neuromorphic architecture
//!
//! Hierarchy: Highway (main connectome) → Cities → Inner/On-ramp neurons
//! Inner neurons: read input + local connectome, write to local connectome
//! On-ramp neurons: read input + local + highway, write to highway
//! Greedy exhaustive: binary ±1 weights × C sweep per neuron
//!
//! Key question: Does hierarchical organization help vs flat?
//! Same neuron budget, different structures, compare accuracy.
//!
//! Run: cargo run --example city_connectome --release

use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const LOCAL_CAP: usize = 3;
const RHO: f32 = 4.0;

fn c19(x: f32, c: f32) -> f32 {
    let c = c.max(0.01);
    let l = 6.0 * c;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let scaled = x / c;
    let n = scaled.floor();
    let t = scaled - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + RHO * h * h)
}

fn thermo_2(a: usize, b: usize) -> [f32; 8] {
    let mut v = [0.0f32; 8];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

// ============================================================
// Structs
// ============================================================

#[derive(Clone)]
struct Worker {
    weights: Vec<i8>,
    c_val: f32,
    is_onramp: bool,
}

#[derive(Clone)]
struct City {
    workers: Vec<Worker>,
    n_inner: usize,
    n_onramp: usize,
    act_offset: usize,
}

#[derive(Clone)]
struct CityNet {
    nc: usize,
    nc_local: usize,
    cities: Vec<City>,
}

fn worker_n_params(is_onramp: bool, wi: usize, nc: usize, nc_local: usize) -> usize {
    let nl = LOCAL_CAP.min(wi);
    // [input:8] [neighbors:nl] [local_conn:nc_local] [highway:nc if onramp] [write:1] [bias:1]
    INPUT_DIM + nl + nc_local + if is_onramp { nc } else { 0 } + 2
}

// ============================================================
// CityNet implementation
// ============================================================

impl CityNet {
    fn new(nc: usize, nc_local: usize, n_cities: usize) -> Self {
        let cities = (0..n_cities).map(|_| City {
            workers: Vec::new(), n_inner: 0, n_onramp: 0, act_offset: 0,
        }).collect();
        CityNet { nc, nc_local, cities }
    }

    fn total_workers(&self) -> usize {
        self.cities.iter().map(|c| c.workers.len()).sum()
    }

    fn ticks(&self) -> usize {
        if self.cities.iter().any(|c| c.n_inner > 0) { 3 } else { 2 }
    }

    fn recompute_offsets(&mut self) {
        let mut offset = 0;
        for city in &mut self.cities {
            city.act_offset = offset;
            offset += city.workers.len();
        }
    }

    fn forward(&self, a: usize, b: usize) -> f32 {
        let input = thermo_2(a, b);
        let total = self.total_workers();
        if total == 0 { return 0.0; }
        let ticks = self.ticks();
        let nc = self.nc;
        let nc_local = self.nc_local;
        let n_cities = self.cities.len();

        let mut act = vec![0.0f32; total];
        let mut highway = vec![0.0f32; nc];
        let mut locals: Vec<Vec<f32>> = if nc_local > 0 {
            (0..n_cities).map(|_| vec![0.0f32; nc_local]).collect()
        } else {
            Vec::new()
        };

        for _t in 0..ticks {
            // Phase 1: WRITE to connectomes
            let mut new_highway = vec![0.0f32; nc];
            let mut new_locals: Vec<Vec<f32>> = if nc_local > 0 {
                (0..n_cities).map(|_| vec![0.0f32; nc_local]).collect()
            } else {
                Vec::new()
            };

            for (ci, city) in self.cities.iter().enumerate() {
                for (wi, worker) in city.workers.iter().enumerate() {
                    let act_val = act[city.act_offset + wi];
                    let nl = LOCAL_CAP.min(wi);
                    let write_idx = INPUT_DIM + nl + nc_local
                        + if worker.is_onramp { nc } else { 0 };
                    let write_w = worker.weights[write_idx] as f32;

                    if worker.is_onramp {
                        let onramp_idx = wi.saturating_sub(city.n_inner);
                        let slot = (ci + onramp_idx) % nc;
                        new_highway[slot] += act_val * write_w;
                    } else if nc_local > 0 {
                        let slot = wi % nc_local;
                        new_locals[ci][slot] += act_val * write_w;
                    }
                }
            }

            highway = new_highway;
            if nc_local > 0 { locals = new_locals; }

            // Phase 2: READ + COMPUTE
            let old_act = act.clone();

            for (ci, city) in self.cities.iter().enumerate() {
                for (wi, worker) in city.workers.iter().enumerate() {
                    let nl = LOCAL_CAP.min(wi);
                    let bias_idx = INPUT_DIM + nl + nc_local
                        + if worker.is_onramp { nc } else { 0 } + 1;
                    let mut s = worker.weights[bias_idx] as f32;

                    // a) Global input
                    for j in 0..INPUT_DIM {
                        s += input[j] * worker.weights[j] as f32;
                    }

                    // b) Local neighbors (within same city, previous workers)
                    let ls = wi.saturating_sub(nl);
                    for (k, prev) in (ls..wi).enumerate() {
                        s += old_act[city.act_offset + prev]
                            * worker.weights[INPUT_DIM + k] as f32;
                    }

                    // c) Local connectome read
                    if nc_local > 0 {
                        for k in 0..nc_local {
                            s += locals[ci][k]
                                * worker.weights[INPUT_DIM + nl + k] as f32;
                        }
                    }

                    // d) Highway read (on-ramp only)
                    if worker.is_onramp {
                        for k in 0..nc {
                            s += highway[k]
                                * worker.weights[INPUT_DIM + nl + nc_local + k] as f32;
                        }
                    }

                    act[city.act_offset + wi] = c19(s, worker.c_val);
                }
            }
        }

        act.iter().sum()
    }

    fn accuracy(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut correct = 0;
        for a in 0..DIGITS { for b in 0..DIGITS {
            if (self.forward(a, b).round() as i32) == (op(a, b) as i32) { correct += 1; }
        }}
        correct as f64 / 25.0
    }

    fn mse(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut l = 0.0f64;
        for a in 0..DIGITS { for b in 0..DIGITS {
            let d = self.forward(a, b) as f64 - op(a, b) as f64;
            l += d * d;
        }}
        l / 25.0
    }

    /// Exhaustive binary×C search for one worker
    fn add_best_worker(
        &mut self, city_idx: usize, is_onramp: bool,
        op: fn(usize, usize) -> usize, c_step: f32, c_max: f32,
    ) -> (f64, f32) {
        let wi = self.cities[city_idx].workers.len();
        let np = worker_n_params(is_onramp, wi, self.nc, self.nc_local);
        let total_binary: u32 = 1 << np;

        let c_steps: Vec<f32> = {
            let mut v = Vec::new();
            let mut c = c_step;
            while c <= c_max { v.push(c); c += c_step; }
            v
        };

        let base_net = self.clone();
        let city_n_inner = self.cities[city_idx].n_inner;
        let city_n_onramp = self.cities[city_idx].n_onramp;

        let results: Vec<(f64, f64, f32, Vec<i8>)> = c_steps.par_iter().map(|&c_val| {
            let mut best_acc = 0.0f64;
            let mut best_mse = f64::MAX;
            let mut best_weights: Vec<i8> = vec![-1; np];

            for config in 0..total_binary {
                let weights: Vec<i8> = (0..np).map(|bit| {
                    if (config >> bit) & 1 == 1 { 1i8 } else { -1i8 }
                }).collect();

                let mut test_net = base_net.clone();
                test_net.cities[city_idx].workers.push(Worker {
                    weights: weights.clone(), c_val, is_onramp,
                });
                if is_onramp {
                    test_net.cities[city_idx].n_onramp = city_n_onramp + 1;
                } else {
                    test_net.cities[city_idx].n_inner = city_n_inner + 1;
                }
                test_net.recompute_offsets();

                let acc = test_net.accuracy(op);
                let mse = test_net.mse(op);

                if acc > best_acc || (acc == best_acc && mse < best_mse) {
                    best_acc = acc; best_mse = mse; best_weights = weights;
                }
            }
            (best_acc, best_mse, c_val, best_weights)
        }).collect();

        let mut best = &results[0];
        for r in &results {
            if r.0 > best.0 || (r.0 == best.0 && r.1 < best.1) { best = r; }
        }

        let best_c = best.2;
        let best_weights = best.3.clone();
        let best_acc = best.0;

        self.cities[city_idx].workers.push(Worker {
            weights: best_weights, c_val: best_c, is_onramp,
        });
        if is_onramp { self.cities[city_idx].n_onramp += 1; }
        else { self.cities[city_idx].n_inner += 1; }
        self.recompute_offsets();
        (best_acc, best_c)
    }
}

// ============================================================
// Build helper
// ============================================================

fn build_net(
    nc: usize, nc_local: usize,
    n_cities: usize, n_inner: usize, n_onramp: usize,
    op: fn(usize, usize) -> usize, c_step: f32, c_max: f32,
    verbose: bool,
) -> CityNet {
    let mut net = CityNet::new(nc, nc_local, n_cities);
    for ci in 0..n_cities {
        for i in 0..n_inner {
            if verbose {
                let wi = net.cities[ci].workers.len();
                let np = worker_n_params(false, wi, nc, nc_local);
                print!("    C{} inner[{}]: 2^{} × C ... ", ci, i, np);
                std::io::stdout().flush().unwrap();
            }
            let t0 = Instant::now();
            let (acc, c_val) = net.add_best_worker(ci, false, op, c_step, c_max);
            if verbose {
                println!("C={:.2} acc={:>5.1}% ({:.1}s)",
                    c_val, acc * 100.0, t0.elapsed().as_secs_f64());
            }
        }
        for i in 0..n_onramp {
            if verbose {
                let wi = net.cities[ci].workers.len();
                let np = worker_n_params(true, wi, nc, nc_local);
                print!("    C{} onramp[{}]: 2^{} × C ... ", ci, i, np);
                std::io::stdout().flush().unwrap();
            }
            let t0 = Instant::now();
            let (acc, c_val) = net.add_best_worker(ci, true, op, c_step, c_max);
            if verbose {
                println!("C={:.2} acc={:>5.1}% ({:.1}s)",
                    c_val, acc * 100.0, t0.elapsed().as_secs_f64());
            }
        }
    }
    net
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

// ============================================================
// Main: experiments
// ============================================================

fn main() {
    println!("=== CITY CONNECTOME: flat vs hierarchical ===\n");
    println!("Inner: read input + local conn, write local conn");
    println!("On-ramp: read input + local + highway, write highway");
    println!("Greedy exhaustive: binary ±1 × C sweep\n");

    let nc = 3;
    let nc_local = 2;
    let c_step = 0.1;
    let c_max = 5.0;
    let c_count = (c_max / c_step) as u32;

    let ops: Vec<(&str, fn(usize, usize) -> usize)> = vec![
        ("ADD",   op_add),
        ("MAX",   op_max),
        ("MIN",   op_min),
        ("|a-b|", op_sub_abs),
        ("MUL",   op_mul),
    ];

    // =========================================================
    // EXP 1: Flat baseline — greedy build demo (ADD only)
    // =========================================================
    println!("--- EXP 1: Flat baseline (nc={}, nc_local=0) ---\n", nc);
    println!("  ADD — greedy flat build:");
    {
        let mut net = CityNet::new(nc, 0, 1);
        for w in 0..8 {
            let np = worker_n_params(true, w, nc, 0);
            print!("    W{}: 2^{} × {} C = {} configs ... ",
                w, np, c_count, (1u32 << np) as u64 * c_count as u64);
            std::io::stdout().flush().unwrap();
            let t0 = Instant::now();
            let (acc, c_val) = net.add_best_worker(0, true, op_add, c_step, c_max);
            println!("C={:.2} acc={:>5.1}% mse={:.4} ({:.1}s)",
                c_val, acc * 100.0, net.mse(op_add), t0.elapsed().as_secs_f64());
            if acc >= 1.0 {
                println!("    SOLVED with {} workers!\n", w + 1);
                break;
            }
        }
    }

    // Quick flat summary for all tasks
    println!("  Flat summary (max 6 workers):");
    println!("  {:>6} {:>8} {:>6} {:>8}", "task", "workers", "acc%", "mse");
    println!("  {}", "=".repeat(32));
    for &(name, op) in &ops {
        let t0 = Instant::now();
        let net = build_net(nc, 0, 1, 0, 6, op, c_step, c_max, false);
        let acc = net.accuracy(op);
        println!("  {:>6} {:>8} {:>5.1}% {:>8.4}  ({:.0}s)",
            name, net.total_workers(), acc * 100.0, net.mse(op),
            t0.elapsed().as_secs_f64());
    }

    // =========================================================
    // EXP 2: Flat vs Hierarchical — 6 neurons, main comparison
    // =========================================================
    println!("\n--- EXP 2: Flat vs Hierarchical (6 neurons) ---\n");
    println!("  FLAT:  1 city, 0 inner, 6 on-ramp, ticks=2");
    println!("  HIER:  2 cities × (2 inner + 1 on-ramp), ticks=3\n");

    println!("  {:>6} {:>6} {:>6} {:>8} {:>8} {:>6}",
        "task", "config", "ticks", "acc%", "mse", "time");
    println!("  {}", "=".repeat(46));

    for &(name, op) in &ops {
        // Flat
        let t0 = Instant::now();
        let flat = build_net(nc, 0, 1, 0, 6, op, c_step, c_max, false);
        let ft = t0.elapsed().as_secs_f64();

        // Hierarchical
        let t0 = Instant::now();
        let hier = build_net(nc, nc_local, 2, 2, 1, op, c_step, c_max, false);
        let ht = t0.elapsed().as_secs_f64();

        println!("  {:>6} {:>6} {:>6} {:>7.1}% {:>8.4} {:>5.0}s",
            name, "flat", flat.ticks(), flat.accuracy(op) * 100.0, flat.mse(op), ft);
        println!("  {:>6} {:>6} {:>6} {:>7.1}% {:>8.4} {:>5.0}s",
            "", "hier", hier.ticks(), hier.accuracy(op) * 100.0, hier.mse(op), ht);
    }

    // =========================================================
    // EXP 3: Width sweep — 6 neurons, different city counts
    // =========================================================
    println!("\n--- EXP 3: Width sweep (6 neurons, depth=1) ---\n");
    println!("  {:>6} {:>6} {:>10} {:>12} {:>5} {:>7} {:>8}",
        "task", "cities", "inner/city", "onramp/city", "ticks", "acc%", "mse");
    println!("  {}", "=".repeat(60));

    // Configs: total 6 neurons
    let width_configs: Vec<(usize, usize, usize, usize)> = vec![
        // (n_cities, n_inner_per_city, n_onramp_per_city, nc_local_override)
        (1, 4, 2, nc_local),  // narrow+deep: 1 city, 4 inner + 2 on-ramp
        (2, 2, 1, nc_local),  // balanced: 2 cities, 2 inner + 1 on-ramp each
        (3, 1, 1, nc_local),  // wide: 3 cities, 1 inner + 1 on-ramp each
        (6, 0, 1, 0),         // flattest: 6 cities, 0 inner + 1 on-ramp each
    ];

    let sweep_ops: Vec<(&str, fn(usize, usize) -> usize)> = vec![
        ("ADD", op_add), ("MIN", op_min), ("|a-b|", op_sub_abs),
    ];

    for &(name, op) in &sweep_ops {
        for &(n_cities, n_inner, n_onramp, ncl) in &width_configs {
            let t0 = Instant::now();
            let net = build_net(nc, ncl, n_cities, n_inner, n_onramp, op, c_step, c_max, false);
            let tt = t0.elapsed().as_secs_f64();
            println!("  {:>6} {:>6} {:>10} {:>12} {:>5} {:>6.1}% {:>8.4}  ({:.0}s)",
                name, n_cities, n_inner, n_onramp, net.ticks(),
                net.accuracy(op) * 100.0, net.mse(op), tt);
        }
        println!();
    }

    // =========================================================
    // EXP 4: Scaling — 9 and 12 neurons (ADD only)
    // =========================================================
    println!("--- EXP 4: Scaling (ADD, flat vs hier) ---\n");
    println!("  {:>6} {:>8} {:>6} {:>10} {:>12} {:>5} {:>7} {:>8}",
        "budget", "config", "cities", "inner/city", "onramp/city", "ticks", "acc%", "mse");
    println!("  {}", "=".repeat(65));

    // 9 neurons
    for &(label, n_cities, n_inner, n_onramp, ncl) in &[
        ("flat",     1usize, 0usize, 9usize, 0usize),
        ("hier-3",   3, 2, 1, nc_local),  // 3 cities × (2+1)
    ] {
        let t0 = Instant::now();
        let net = build_net(nc, ncl, n_cities, n_inner, n_onramp, op_add, c_step, c_max, false);
        println!("  {:>6} {:>8} {:>6} {:>10} {:>12} {:>5} {:>6.1}% {:>8.4}  ({:.0}s)",
            9, label, n_cities, n_inner, n_onramp, net.ticks(),
            net.accuracy(op_add) * 100.0, net.mse(op_add), t0.elapsed().as_secs_f64());
    }
    println!();

    // 12 neurons
    for &(label, n_cities, n_inner, n_onramp, ncl) in &[
        ("flat",     1usize, 0usize, 12usize, 0usize),
        ("hier-3",   3, 2, 2, nc_local),  // 3 cities × (2+2)
        ("hier-4",   4, 2, 1, nc_local),  // 4 cities × (2+1)
    ] {
        let t0 = Instant::now();
        let net = build_net(nc, ncl, n_cities, n_inner, n_onramp, op_add, c_step, c_max, false);
        println!("  {:>6} {:>8} {:>6} {:>10} {:>12} {:>5} {:>6.1}% {:>8.4}  ({:.0}s)",
            12, label, n_cities, n_inner, n_onramp, net.ticks(),
            net.accuracy(op_add) * 100.0, net.mse(op_add), t0.elapsed().as_secs_f64());
    }

    // =========================================================
    // EXP 5: Detailed hier build (ADD) — show per-neuron progress
    // =========================================================
    println!("\n--- EXP 5: Detailed hierarchical build (ADD, 2 cities) ---\n");
    let _net = build_net(nc, nc_local, 2, 2, 1, op_add, c_step, c_max, true);
    println!("  Final: acc={:.1}% mse={:.4}", _net.accuracy(op_add) * 100.0, _net.mse(op_add));

    println!("\n=== DONE ===");
}
