use std::time::Instant;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::sync::Arc;

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

fn main() {
    let n = 13usize;
    let n_patterns = 1usize << n; // 8192
    let total_combos = 3u64.pow((n + 1) as u32); // 3^14 = 4,782,969
    let n_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
    
    println!("N={}, patterns={}, combos={}, threads={}", n, n_patterns, total_combos, n_threads);
    
    // Target: popcount > n/2
    let n_words = (n_patterns + 63) / 64;
    let target: Vec<u64> = {
        let mut t = vec![0u64; n_words];
        for p in 0..n_patterns {
            if (p as u32).count_ones() > (n as u32) / 2 {
                t[p / 64] |= 1u64 << (p % 64);
            }
        }
        t
    };
    let target = Arc::new(target);
    
    let best_score = Arc::new(AtomicUsize::new(0));
    let total_searched = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let chunk = (total_combos / n_threads as u64).max(1);
    
    let mut handles = vec![];
    for tid in 0..n_threads {
        let tgt = Arc::clone(&target);
        let best = Arc::clone(&best_score);
        let searched = Arc::clone(&total_searched);
        let from = tid as u64 * chunk;
        let to = if tid == n_threads - 1 { total_combos } else { from + chunk };
        
        handles.push(std::thread::spawn(move || {
            let mut local_best = 0usize;
            let mut cnt = 0u64;
            let mut weights = vec![0i8; n];
            for combo in from..to {
                let mut idx = combo;
                for i in 0..n { weights[i] = (idx % 3) as i8 - 1; idx /= 3; }
                let bias = (idx % 3) as i8 - 1;
                
                // Eval all 2^13 = 8192 patterns
                let mut agree = 0usize;
                for p in 0..n_patterns {
                    let mut sum = bias as i32;
                    for i in 0..n { if p & (1 << i) != 0 { sum += weights[i] as i32; } }
                    let out = if c19(sum as f32, 8.0) > 0.5 { 1u64 } else { 0u64 };
                    let expected = (tgt[p / 64] >> (p % 64)) & 1;
                    if out == expected { agree += 1; }
                }
                if agree > local_best { local_best = agree; }
                cnt += 1;
            }
            best.fetch_max(local_best, Ordering::Relaxed);
            searched.fetch_add(cnt, Ordering::Relaxed);
        }));
    }
    for h in handles { h.join().unwrap(); }
    
    let elapsed = start.elapsed().as_secs_f64();
    let searched = total_searched.load(Ordering::Relaxed);
    let best = best_score.load(Ordering::Relaxed);
    
    println!("DONE: {}/{} combos in {:.1}s ({:.0}K/s)", searched, total_combos, elapsed, searched as f64 / elapsed / 1000.0);
    println!("Best score: {}/{} = {:.1}%", best, n_patterns, best as f64 / n_patterns as f64 * 100.0);
}
