//! Ternary Exhaustive Search Benchmark — MULTI-CORE
//! How fast can we search all ternary weight combos for N-input neurons?
//! Weights ∈ {-1, 0, +1}, inputs ∈ {0, 1}
//! Max 10 minutes per N value.
//!
//! Run: cargo run --example ternary_search_bench --release

use std::time::Instant;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

/// Evaluate a ternary neuron on all 2^N binary inputs, return packed bitvec
fn eval_ternary_neuron(weights: &[i8], bias: i8, n: usize) -> Vec<u64> {
    let n_patterns = 1usize << n;
    let n_words = (n_patterns + 63) / 64;
    let mut result = vec![0u64; n_words];
    for pattern in 0..n_patterns {
        let mut sum = bias as i32;
        for i in 0..n {
            if pattern & (1 << i) != 0 {
                sum += weights[i] as i32;
            }
        }
        if c19(sum as f32, 8.0) > 0.5 {
            result[pattern / 64] |= 1u64 << (pattern % 64);
        }
    }
    result
}

/// Count agreement between two bitvectors
fn agreement(a: &[u64], b: &[u64], total_bits: usize) -> usize {
    let mut agree = 0usize;
    for (wa, wb) in a.iter().zip(b.iter()) {
        agree += (!(*wa ^ *wb)).count_ones() as usize;
    }
    let extra = a.len() * 64 - total_bits;
    agree - extra
}

/// Decode combo index to ternary weights
fn decode_combo(mut idx: u64, n: usize) -> (Vec<i8>, i8) {
    let mut weights = vec![0i8; n];
    for i in 0..n {
        weights[i] = (idx % 3) as i8 - 1;
        idx /= 3;
    }
    let bias = (idx % 3) as i8 - 1;
    (weights, bias)
}

/// Multi-threaded exhaustive search with 10-minute timeout
fn exhaustive_search_parallel(
    n: usize, target: &[u64], total_patterns: usize, n_threads: usize,
) -> (u64, usize, f64, bool) {
    let total_combos = 3u64.pow((n + 1) as u32);
    let best_score = Arc::new(AtomicUsize::new(0));
    let total_searched = Arc::new(AtomicU64::new(0));
    let timeout_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let start = Instant::now();
    let timeout_secs = 600.0; // 10 minutes

    let chunk_size = (total_combos / n_threads as u64).max(1);
    let target = Arc::new(target.to_vec());

    let mut handles = vec![];
    for thread_id in 0..n_threads {
        let best = Arc::clone(&best_score);
        let searched = Arc::clone(&total_searched);
        let timeout = Arc::clone(&timeout_flag);
        let tgt = Arc::clone(&target);

        let from = thread_id as u64 * chunk_size;
        let to = if thread_id == n_threads - 1 { total_combos } else { from + chunk_size };

        let handle = std::thread::spawn(move || {
            let mut local_best = 0usize;
            let mut local_count = 0u64;
            let check_interval = 10000u64;

            for combo_idx in from..to {
                if local_count % check_interval == 0 {
                    if timeout.load(Ordering::Relaxed) { break; }
                    if start.elapsed().as_secs_f64() > timeout_secs {
                        timeout.store(true, Ordering::Relaxed);
                        break;
                    }
                }

                let (weights, bias) = decode_combo(combo_idx, n);
                let output = eval_ternary_neuron(&weights, bias, n);
                let score = agreement(&output, &tgt, 1 << n);
                if score > local_best { local_best = score; }
                local_count += 1;
            }

            best.fetch_max(local_best, Ordering::Relaxed);
            searched.fetch_add(local_count, Ordering::Relaxed);
        });
        handles.push(handle);
    }

    for h in handles { h.join().unwrap(); }

    let elapsed = start.elapsed().as_secs_f64();
    let searched = total_searched.load(Ordering::Relaxed);
    let timed_out = timeout_flag.load(Ordering::Relaxed);
    let best = best_score.load(Ordering::Relaxed);

    (searched, best, elapsed, timed_out)
}

fn main() {
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get()).unwrap_or(4);

    println!("================================================================");
    println!("  TERNARY EXHAUSTIVE SEARCH — MULTI-CORE BENCHMARK");
    println!("  Weights ∈ {{-1, 0, +1}}, Inputs ∈ {{0, 1}}");
    println!("  Threads: {}, Timeout: 10 min per N", n_threads);
    println!("================================================================\n");

    let test_ns: Vec<usize> = vec![8, 10, 12, 14, 15, 16, 17, 18, 20];

    println!("  {:>4} {:>12} {:>12} {:>12} {:>10} {:>10} {:>8}",
        "N", "Patterns", "TotalCombos", "Searched", "Time", "Status", "Rate");
    println!("  {}", "─".repeat(78));

    let mut max_feasible = 0;

    for &n in &test_ns {
        let n_patterns = 1usize << n;
        let total_combos = 3u64.checked_pow((n + 1) as u32).unwrap_or(u64::MAX);

        // Create target: popcount > n/2
        let n_words = (n_patterns + 63) / 64;
        let mut target = vec![0u64; n_words];
        for pattern in 0..n_patterns {
            if (pattern as u32).count_ones() > (n as u32) / 2 {
                target[pattern / 64] |= 1u64 << (pattern % 64);
            }
        }

        let (searched, best, elapsed, timed_out) =
            exhaustive_search_parallel(n, &target, n_patterns, n_threads);

        let rate = if elapsed > 0.0 { searched as f64 / elapsed } else { 0.0 };

        let status = if timed_out {
            let pct = searched as f64 / total_combos as f64 * 100.0;
            format!("{:.1}% done", pct)
        } else {
            max_feasible = n;
            "COMPLETE".to_string()
        };

        let full_est = if timed_out && rate > 0.0 {
            let est = total_combos as f64 / rate;
            if est < 60.0 { format!("(~{:.0}s full)", est) }
            else if est < 3600.0 { format!("(~{:.1}m full)", est / 60.0) }
            else if est < 86400.0 { format!("(~{:.1}h full)", est / 3600.0) }
            else { format!("(~{:.1}d full)", est / 86400.0) }
        } else { String::new() };

        println!("  {:>4} {:>12} {:>12} {:>12} {:>8.1}s {:>10} {:>7.0}K/s {}",
            n, n_patterns, total_combos, searched, elapsed, status, rate / 1000.0, full_est);
    }

    println!();
    println!("  ┌──────────────────────────────────────────────────┐");
    println!("  │ RESULT ({} threads):                       │", n_threads);
    println!("  │                                                  │");
    println!("  │ Max N fully searchable in <10 min: N = {:>2}       │", max_feasible);
    println!("  │                                                  │");
    println!("  │ → Layer width = {} neurons                       │", max_feasible);
    println!("  │ → Each neuron exhaustive verified                │");
    println!("  │ → Stack unlimited layers for depth               │");
    println!("  └──────────────────────────────────────────────────┘");
    println!("================================================================");
}
