//! Holographic Memory Diagnostics — Interference & Capacity Analysis
//!
//! 5 targeted tests to understand WHY HoloLayer fails at scale:
//!   T1: Interference curve — accuracy vs pattern count
//!   T2: Dimension scaling — dim × n_patterns matrix
//!   T3: Encoding quality — cosine similarity of encoded vectors
//!   T4: Store order sensitivity — permutation variance
//!   T5: Forgetting curve — early vs late pattern accuracy
//!
//! Run: cargo run --example holo_diagnostic --release

use std::io::Write;

// ============================================================
// RNG (same as c19_arch_sweep.rs)
// ============================================================

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range(&mut self, max: u64) -> u64 { self.next() % max }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.range((i + 1) as u64) as usize;
            v.swap(i, j);
        }
    }
}

// ============================================================
// HoloLayer (copied from c19_arch_sweep.rs)
// ============================================================

struct HoloLayer {
    dim: usize,
    matrix: Vec<f32>,      // dim × dim
    proj: Vec<Vec<f32>>,   // n_proj × dim
    n_proj: usize,
}

impl HoloLayer {
    fn new(dim: usize, n_proj: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let proj = (0..n_proj).map(|_| (0..dim).map(|_| rng.f32() * 2.0 - 1.0).collect()).collect();
        HoloLayer { dim, matrix: vec![0.0; dim * dim], proj, n_proj }
    }

    fn encode_input(&self, bits: &[u8]) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 && i < self.n_proj {
                for d in 0..self.dim { v[d] += self.proj[i][d]; }
            }
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 { for x in &mut v { *x /= norm; } }
        v
    }

    fn store(&mut self, bits: &[u8], target: usize) {
        let inp = self.encode_input(bits);
        for i in 0..self.dim {
            let out_i = if i == target { 1.0 } else { 0.0 };
            for j in 0..self.dim {
                self.matrix[i * self.dim + j] += inp[j] * out_i;
            }
        }
    }

    fn predict(&self, bits: &[u8]) -> usize {
        let scores = self.predict_scores(bits);
        scores.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0)
    }

    fn predict_scores(&self, bits: &[u8]) -> Vec<f32> {
        let inp = self.encode_input(bits);
        let mut output = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                output[i] += self.matrix[i * self.dim + j] * inp[j];
            }
        }
        output
    }

    fn clear(&mut self) {
        self.matrix.iter_mut().for_each(|x| *x = 0.0);
    }
}

// ============================================================
// C19 activation (init-time only, for LutGate baking)
// ============================================================

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

// ============================================================
// LutGate + ParallelAlu (from c19_arch_sweep.rs)
// ============================================================

#[derive(Clone)]
struct LutGate {
    w_int: Vec<i32>,
    bias_int: i32,
    lut: Vec<u8>,
    min_sum: i32,
}

impl LutGate {
    fn new(w: &[f32], bias: f32, rho: f32, thr: f32) -> Self {
        let mut all = w.to_vec(); all.push(bias);
        let mut denom = 1;
        for d in 1..=100 {
            if all.iter().all(|&v| ((v * d as f32).round() - v * d as f32).abs() < 1e-6) {
                denom = d; break;
            }
        }
        let w_int: Vec<i32> = w.iter().map(|&v| (v * denom as f32).round() as i32).collect();
        let bias_int = (bias * denom as f32).round() as i32;
        let mut min_s = bias_int; let mut max_s = bias_int;
        for &wi in &w_int { if wi > 0 { max_s += wi; } else { min_s += wi; } }
        let mut lut = vec![0u8; (max_s - min_s + 1) as usize];
        for s in min_s..=max_s {
            lut[(s - min_s) as usize] = if c19(s as f32 / denom as f32, rho) > thr { 1 } else { 0 };
        }
        LutGate { w_int, bias_int, lut, min_sum: min_s }
    }
    fn eval(&self, inputs: &[u8]) -> u8 {
        let s: i32 = inputs.iter().zip(&self.w_int).map(|(&i, &w)| i as i32 * w).sum::<i32>() + self.bias_int;
        let idx = (s - self.min_sum) as usize;
        if idx < self.lut.len() { self.lut[idx] } else { 0 }
    }
}

struct ParallelAlu {
    xor3: LutGate, maj: LutGate, not_g: LutGate,
    and_g: LutGate, or_g: LutGate, xor_g: LutGate,
}

impl ParallelAlu {
    fn new() -> Self {
        ParallelAlu {
            xor3: LutGate::new(&[1.5,1.5,1.5], 3.0, 16.0, 0.6),
            maj: LutGate::new(&[8.5,8.5,8.5], -2.75, 0.0, 4.0),
            not_g: LutGate::new(&[-9.75], -5.5, 16.0, -4.0),
            and_g: LutGate::new(&[10.0,10.0], -4.5, 0.0, 4.0),
            or_g: LutGate::new(&[8.75,8.75], 5.5, 0.0, 4.0),
            xor_g: LutGate::new(&[0.5,0.5], 0.0, 16.0, 0.6),
        }
    }

    fn add4(&self, a: u8, b: u8) -> u8 {
        let mut c = 0u8; let mut r = 0u8;
        for bit in 0..4 {
            let ab = (a>>bit)&1; let bb = (b>>bit)&1;
            r |= self.xor3.eval(&[ab,bb,c]) << bit;
            c = self.maj.eval(&[ab,bb,c]);
        }
        r & 0xF
    }
    fn sub4(&self, a: u8, b: u8) -> u8 {
        let mut bn = 0u8;
        for bit in 0..4 { bn |= self.not_g.eval(&[(b>>bit)&1]) << bit; }
        let mut c = 1u8; let mut r = 0u8;
        for bit in 0..4 {
            let ab = (a>>bit)&1; let bb = (bn>>bit)&1;
            r |= self.xor3.eval(&[ab,bb,c]) << bit;
            c = self.maj.eval(&[ab,bb,c]);
        }
        r & 0xF
    }
    fn and4(&self, a: u8, b: u8) -> u8 {
        let mut r = 0u8;
        for bit in 0..4 { r |= self.and_g.eval(&[(a>>bit)&1,(b>>bit)&1]) << bit; }
        r
    }
    fn or4(&self, a: u8, b: u8) -> u8 {
        let mut r = 0u8;
        for bit in 0..4 { r |= self.or_g.eval(&[(a>>bit)&1,(b>>bit)&1]) << bit; }
        r
    }
    fn xor4(&self, a: u8, b: u8) -> u8 {
        let mut r = 0u8;
        for bit in 0..4 { r |= self.xor_g.eval(&[(a>>bit)&1,(b>>bit)&1]) << bit; }
        r
    }

    fn execute_all(&self, a: u8, b: u8) -> [u8; 5] {
        [self.add4(a,b), self.sub4(a,b), self.and4(a,b), self.or4(a,b), self.xor4(a,b)]
    }
}

fn expected_op(a: u8, b: u8, op: usize) -> u8 {
    match op {
        0 => (a.wrapping_add(b)) & 0xF,
        1 => (a.wrapping_sub(b)) & 0xF,
        2 => a & b, 3 => a | b, 4 => a ^ b,
        _ => 0,
    }
}

// ============================================================
// Helpers
// ============================================================

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 { return 0.0; }
    dot / (na * nb)
}

fn compute_snr_margin(scores: &[f32], target: usize) -> (f32, f32, usize) {
    let target_score = scores[target];
    let mut sorted: Vec<(usize, f32)> = scores.iter().enumerate()
        .map(|(i, &s)| (i, s)).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // rank of correct answer
    let rank = sorted.iter().position(|&(i, _)| i == target).unwrap_or(scores.len()) + 1;

    // second best score (best that isn't target)
    let second_best = sorted.iter()
        .find(|&&(i, _)| i != target)
        .map(|&(_, s)| s)
        .unwrap_or(f32::NEG_INFINITY);

    let margin = target_score - second_best;
    let snr = if second_best.abs() < 1e-12 { f32::INFINITY } else { target_score / second_best };

    (snr, margin, rank)
}

fn generate_patterns(n: usize, n_bits: usize, n_classes: usize, rng: &mut Rng) -> Vec<(Vec<u8>, usize)> {
    (0..n).map(|_| {
        let bits: Vec<u8> = (0..n_bits).map(|_| rng.range(2) as u8).collect();
        let target = rng.range(n_classes as u64) as usize;
        (bits, target)
    }).collect()
}

struct LogWriter {
    file: std::fs::File,
}

impl LogWriter {
    fn new(path: &str) -> Self {
        LogWriter { file: std::fs::File::create(path).unwrap() }
    }
    fn log(&mut self, msg: &str) {
        let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
        let s = d.as_secs(); let h = (s/3600)%24; let m = (s/60)%60; let sec = s%60;
        let line = format!("[{:02}:{:02}:{:02}] {}\n", h, m, sec, msg);
        print!("{}", line);
        self.file.write_all(line.as_bytes()).ok();
        self.file.flush().ok();
    }
}

// ============================================================
// TEST 1: Interference Curve
// ============================================================

fn test1_interference_curve(log: &mut LogWriter, tsv: &mut std::fs::File) {
    log.log("========================================");
    log.log("=== TEST 1: Interference Curve (dim=64) ===");
    log.log("========================================");

    let dim = 64;
    let n_bits = 8;
    let n_classes = 16;
    let pattern_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

    // Generate a big pool of patterns, then use subsets
    let mut rng = Rng::new(42);
    let all_patterns = generate_patterns(512, n_bits, n_classes, &mut rng);

    for &n in &pattern_counts {
        let patterns = &all_patterns[..n];
        let mut holo = HoloLayer::new(dim, n_bits, 42);

        // Store all
        for (bits, target) in patterns {
            holo.store(bits, *target);
        }

        // Retrieve all and measure
        let mut correct = 0;
        let mut total_snr = 0.0f64;
        let mut min_margin = f32::INFINITY;
        let mut total_rank = 0usize;
        let mut snr_count = 0;

        for (bits, target) in patterns {
            let scores = holo.predict_scores(bits);
            let pred = scores.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i,_)| i).unwrap_or(0);
            if pred == *target { correct += 1; }

            let (snr, margin, rank) = compute_snr_margin(&scores, *target);
            if snr.is_finite() { total_snr += snr as f64; snr_count += 1; }
            if margin < min_margin { min_margin = margin; }
            total_rank += rank;
        }

        let accuracy = correct as f64 / n as f64;
        let avg_snr = if snr_count > 0 { total_snr / snr_count as f64 } else { f64::INFINITY };
        let avg_rank = total_rank as f64 / n as f64;

        let marker = if accuracy < 0.5 && n > 1 { " ← BREAKDOWN" } else { "" };
        log.log(&format!("  n={:>4}: acc={:>5.1}%  snr={:>7.2}  margin={:>+7.3}  avg_rank={:.1}{}",
            n, accuracy * 100.0, avg_snr, min_margin, avg_rank, marker));

        writeln!(tsv, "interference\t{}\t{}\t{}\t{}\t42\t{:.6}\t{:.4}\t{:.4}\t-\t-",
            dim, n_bits, n, n_classes, accuracy, avg_snr, min_margin).ok();
    }
}

// ============================================================
// TEST 2: Dimension Scaling
// ============================================================

fn test2_dim_scaling(log: &mut LogWriter, tsv: &mut std::fs::File) {
    log.log("");
    log.log("========================================");
    log.log("=== TEST 2: Dimension Scaling ===");
    log.log("========================================");

    let dims = [8, 16, 32, 64, 128, 256, 512, 1024];
    let pattern_counts = [4, 8, 16, 32, 64, 128, 256, 512];
    let n_bits = 8;
    let n_classes = 16;

    // Header
    log.log(&format!("  {:>6} | {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "dim\\n", 4, 8, 16, 32, 64, 128, 256, 512));
    log.log(&format!("  {}+{}", "-".repeat(6), "-".repeat(56)));

    let mut rng = Rng::new(42);
    let all_patterns = generate_patterns(512, n_bits, n_classes, &mut rng);

    for &dim in &dims {
        let mut row = format!("  {:>6} |", dim);

        for &n in &pattern_counts {
            let patterns = &all_patterns[..n];
            let mut holo = HoloLayer::new(dim, n_bits, 42);

            for (bits, target) in patterns {
                holo.store(bits, *target);
            }

            let mut correct = 0;
            for (bits, target) in patterns {
                if holo.predict(bits) == *target { correct += 1; }
            }

            let accuracy = correct as f64 / n as f64;
            row.push_str(&format!(" {:>5.1}%", accuracy * 100.0));

            writeln!(tsv, "dim_scaling\t{}\t{}\t{}\t{}\t42\t{:.6}\t-\t-\t-\t-",
                dim, n_bits, n, n_classes, accuracy).ok();
        }

        log.log(&row);
    }
}

// ============================================================
// TEST 3: Encoding Quality
// ============================================================

fn test3_encoding_quality(log: &mut LogWriter, tsv: &mut std::fs::File) {
    log.log("");
    log.log("========================================");
    log.log("=== TEST 3: Encoding Quality ===");
    log.log("========================================");

    let dim = 64;
    let n_proj_values = [4, 8, 11, 16, 32];
    let n_patterns = 256;

    // Generate 256 unique 8-bit patterns (all possible for 8 bits)
    let patterns: Vec<Vec<u8>> = (0..n_patterns).map(|i| {
        (0..8).map(|b| ((i >> b) & 1) as u8).collect()
    }).collect();

    for &n_proj in &n_proj_values {
        let n_bits = n_proj.max(8); // need at least 8 bits for 256 patterns
        let holo = HoloLayer::new(dim, n_bits, 42);

        // Encode all patterns
        let encoded: Vec<Vec<f32>> = patterns.iter().map(|p| {
            let mut bits = p.clone();
            while bits.len() < n_bits { bits.push(0); }
            holo.encode_input(&bits)
        }).collect();

        // Compute pairwise cosine similarity
        let mut total_cos = 0.0f64;
        let mut max_cos = 0.0f32;
        let mut n_pairs = 0u64;

        for i in 0..encoded.len() {
            for j in (i+1)..encoded.len() {
                let cs = cosine_similarity(&encoded[i], &encoded[j]).abs();
                total_cos += cs as f64;
                if cs > max_cos { max_cos = cs; }
                n_pairs += 1;
            }
        }

        let mean_cos = total_cos / n_pairs as f64;

        // Effective rank via Gram matrix eigenvalue approximation
        // (simplified: ratio of Frobenius norm to spectral norm)
        let gram_trace = encoded.iter().map(|v| {
            v.iter().map(|x| x * x).sum::<f32>()
        }).sum::<f32>();

        // Approximate effective rank: trace(G) / ||G||_2
        // Since all vectors are normalized, trace = n_patterns
        // We estimate ||G||_2 via power iteration
        let eff_rank = estimate_effective_rank(&encoded);

        log.log(&format!("  n_proj={:>2}: mean_cos={:.4}  max_cos={:.4}  eff_rank={:.1}/{}",
            n_proj, mean_cos, max_cos, eff_rank, n_patterns));

        writeln!(tsv, "encoding\t{}\t{}\t{}\t-\t42\t-\t-\t-\t{:.6}\t{:.6}",
            dim, n_proj, n_patterns, mean_cos, max_cos).ok();
    }
}

fn estimate_effective_rank(vecs: &[Vec<f32>]) -> f32 {
    let n = vecs.len();
    if n == 0 || vecs[0].is_empty() { return 0.0; }
    let dim = vecs[0].len();

    // Compute G = V * V^T where V is n × dim
    // Then effective rank ≈ trace(G)² / ||G||_F²
    // trace(G) = sum of ||v_i||² = n (since normalized)
    let trace = n as f32;

    // ||G||_F² = sum_{i,j} (v_i · v_j)²
    let mut frob_sq = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let dot: f32 = (0..dim).map(|k| vecs[i][k] * vecs[j][k]).sum();
            frob_sq += (dot * dot) as f64;
        }
        // For large n, this is O(n²×dim) — fine for n=256, dim≤64
    }

    (trace * trace) / frob_sq as f32
}

// ============================================================
// TEST 4: Store Order Sensitivity
// ============================================================

fn test4_order_sensitivity(log: &mut LogWriter, tsv: &mut std::fs::File) {
    log.log("");
    log.log("========================================");
    log.log("=== TEST 4: Store Order Sensitivity ===");
    log.log("========================================");

    let dim = 64;
    let n_bits = 8;
    let n_classes = 16;
    let n_patterns = 64;
    let n_perms = 10;

    let mut rng = Rng::new(42);
    let patterns = generate_patterns(n_patterns, n_bits, n_classes, &mut rng);

    let mut accuracies = Vec::new();

    for perm in 0..n_perms {
        let mut order: Vec<usize> = (0..n_patterns).collect();
        let mut perm_rng = Rng::new(100 + perm);
        perm_rng.shuffle(&mut order);

        let mut holo = HoloLayer::new(dim, n_bits, 42); // same projections!

        for &idx in &order {
            holo.store(&patterns[idx].0, patterns[idx].1);
        }

        let mut correct = 0;
        for (bits, target) in &patterns {
            if holo.predict(bits) == *target { correct += 1; }
        }

        let accuracy = correct as f64 / n_patterns as f64;
        accuracies.push(accuracy);

        log.log(&format!("  perm {:>2}: acc={:.1}%", perm, accuracy * 100.0));
    }

    let mean: f64 = accuracies.iter().sum::<f64>() / n_perms as f64;
    let variance: f64 = accuracies.iter().map(|&a| (a - mean) * (a - mean)).sum::<f64>() / n_perms as f64;
    let stddev = variance.sqrt();

    log.log(&format!("  → mean={:.1}%  stddev={:.2}%  ({})",
        mean * 100.0, stddev * 100.0,
        if stddev < 0.01 { "ORDER INVARIANT — outer product is commutative!" } else { "ORDER MATTERS" }));

    writeln!(tsv, "order_sensitivity\t{}\t{}\t{}\t{}\t-\t{:.6}\t{:.6}\t-\t-\t-",
        dim, n_bits, n_patterns, n_classes, mean, stddev).ok();
}

// ============================================================
// TEST 5: Forgetting Curve
// ============================================================

fn test5_forgetting_curve(log: &mut LogWriter, tsv: &mut std::fs::File) {
    log.log("");
    log.log("========================================");
    log.log("=== TEST 5: Forgetting Curve ===");
    log.log("========================================");

    let dim = 64;
    let n_bits = 8;
    let n_classes = 16;
    let n_patterns = 128;
    let checkpoint_every = 8;

    let mut rng = Rng::new(42);
    let patterns = generate_patterns(n_patterns, n_bits, n_classes, &mut rng);

    let mut holo = HoloLayer::new(dim, n_bits, 42);

    log.log(&format!("  {:>6} | {:>8} {:>8} {:>8} {:>8}",
        "stored", "overall", "early", "mid", "late"));
    log.log(&format!("  {}+{}", "-".repeat(6), "-".repeat(38)));

    for i in 0..n_patterns {
        holo.store(&patterns[i].0, patterns[i].1);

        if (i + 1) % checkpoint_every == 0 || i == n_patterns - 1 {
            let stored = i + 1;

            // Measure accuracy in cohorts
            let (mut early_correct, mut early_total) = (0, 0);
            let (mut mid_correct, mut mid_total) = (0, 0);
            let (mut late_correct, mut late_total) = (0, 0);
            let mut overall_correct = 0;

            for j in 0..stored {
                let pred = holo.predict(&patterns[j].0);
                let ok = pred == patterns[j].1;
                if ok { overall_correct += 1; }

                if j < 32 {
                    early_total += 1;
                    if ok { early_correct += 1; }
                } else if j < 96 {
                    mid_total += 1;
                    if ok { mid_correct += 1; }
                } else {
                    late_total += 1;
                    if ok { late_correct += 1; }
                }
            }

            let overall = overall_correct as f64 / stored as f64;
            let early = if early_total > 0 { early_correct as f64 / early_total as f64 } else { -1.0 };
            let mid = if mid_total > 0 { mid_correct as f64 / mid_total as f64 } else { -1.0 };
            let late = if late_total > 0 { late_correct as f64 / late_total as f64 } else { -1.0 };

            let fmt = |v: f64| -> String {
                if v < 0.0 { "    -   ".to_string() } else { format!("{:>7.1}%", v * 100.0) }
            };

            log.log(&format!("  {:>6} | {} {} {} {}",
                stored, fmt(overall), fmt(early), fmt(mid), fmt(late)));

            writeln!(tsv, "forgetting\t{}\t{}\t{}\t{}\t42\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t-",
                dim, n_bits, stored, n_classes, overall, early, mid, late).ok();
        }
    }
}

// ============================================================
// TEST 6: ALU → Holo Pipeline (does holo degrade ALU's 100%?)
// ============================================================

fn test6_alu_then_holo(log: &mut LogWriter, tsv: &mut std::fs::File) {
    log.log("");
    log.log("========================================");
    log.log("=== TEST 6: ALU → Holo Pipeline ===");
    log.log("========================================");
    log.log("  Question: ALU alone = 100%. Does Holo after ALU degrade it?");

    let alu = ParallelAlu::new();
    let op_names = ["ADD", "SUB", "AND", "OR", "XOR"];

    // Sanity: verify ALU is 100%
    log.log("");
    log.log("  --- Sanity: ALU alone ---");
    for (op_idx, op_name) in op_names.iter().enumerate() {
        let mut correct = 0;
        for a in 0..16u8 {
            for b in 0..16u8 {
                let result = alu.execute_all(a, b)[op_idx];
                let expected = expected_op(a, b, op_idx);
                if result == expected { correct += 1; }
            }
        }
        log.log(&format!("    {} : {}/256 = {:.1}%", op_name, correct, correct as f64 / 256.0 * 100.0));
    }

    // Now: ALU output → encode as bits → HoloLayer → predict op index
    // Pipeline: given (a, b, op), ALU computes result, we encode
    // [a_bits(4), b_bits(4), result_bits(4), all_5_results_bits(20)] = 28 bits
    // Target: op index (0-4)
    log.log("");
    log.log("  --- ALU → Holo: can Holo learn which op was requested? ---");
    log.log("  Encoding: [a(4bit), b(4bit), result(4bit), all_5_alu_outputs(20bit)] = 28 bits");
    log.log("  Target: op index (0-4)");

    let dims = [32, 64, 128, 256, 512];

    for &dim in &dims {
        // Generate all (a, b, op) patterns = 16 × 16 × 5 = 1280 patterns
        let mut patterns: Vec<(Vec<u8>, usize)> = Vec::new();

        for a in 0..16u8 {
            for b in 0..16u8 {
                let all_results = alu.execute_all(a, b);
                for op in 0..5usize {
                    let result = all_results[op];
                    // Encode: a bits + b bits + result bits + all 5 results bits
                    let mut bits = Vec::with_capacity(28);
                    for bit in 0..4 { bits.push((a >> bit) & 1); }
                    for bit in 0..4 { bits.push((b >> bit) & 1); }
                    for bit in 0..4 { bits.push((result >> bit) & 1); }
                    for r in &all_results {
                        for bit in 0..4 { bits.push((r >> bit) & 1); }
                    }
                    patterns.push((bits, op));
                }
            }
        }

        let n_proj = 28; // 28-bit encoding
        let mut holo = HoloLayer::new(dim, n_proj, 42);

        // Store all patterns
        for (bits, target) in &patterns {
            holo.store(bits, *target);
        }

        // Retrieve and measure
        let mut correct = 0;
        let mut total_snr = 0.0f64;
        let mut snr_count = 0;

        for (bits, target) in &patterns {
            let scores = holo.predict_scores(bits);
            let pred = scores.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i,_)| i).unwrap_or(0);
            if pred == *target { correct += 1; }

            let (snr, _, _) = compute_snr_margin(&scores, *target);
            if snr.is_finite() { total_snr += snr as f64; snr_count += 1; }
        }

        let accuracy = correct as f64 / patterns.len() as f64;
        let avg_snr = if snr_count > 0 { total_snr / snr_count as f64 } else { f64::INFINITY };

        let verdict = if accuracy > 0.99 { "PRESERVES" }
                     else if accuracy > 0.5 { "DEGRADES" }
                     else { "DESTROYS" };

        log.log(&format!("    dim={:>4}: acc={:>5.1}%  snr={:>7.2}  ({} patterns)  → {}",
            dim, accuracy * 100.0, avg_snr, patterns.len(), verdict));

        writeln!(tsv, "alu_then_holo\t{}\t{}\t{}\t5\t42\t{:.6}\t{:.4}\t-\t-\t-",
            dim, n_proj, patterns.len(), accuracy, avg_snr).ok();
    }

    // Also test: simpler encoding — just the result + op bits
    log.log("");
    log.log("  --- Simpler encoding: [result(4bit), op(3bit)] = 7 bits ---");

    for &dim in &dims {
        let mut patterns: Vec<(Vec<u8>, usize)> = Vec::new();

        for a in 0..16u8 {
            for b in 0..16u8 {
                let all_results = alu.execute_all(a, b);
                for op in 0..5usize {
                    let result = all_results[op];
                    let mut bits = Vec::with_capacity(7);
                    for bit in 0..4 { bits.push((result >> bit) & 1); }
                    for bit in 0..3 { bits.push(((op as u8) >> bit) & 1); }
                    patterns.push((bits, result as usize));
                }
            }
        }

        let n_proj = 7;
        let n_classes = 16; // result is 0-15
        let mut holo = HoloLayer::new(dim, n_proj, 42);

        for (bits, target) in &patterns {
            holo.store(bits, *target);
        }

        let mut correct = 0;
        for (bits, target) in &patterns {
            if holo.predict(bits) == *target { correct += 1; }
        }

        let accuracy = correct as f64 / patterns.len() as f64;
        log.log(&format!("    dim={:>4}: acc={:>5.1}%  ({} patterns, 16 classes)",
            dim, accuracy * 100.0, patterns.len()));

        writeln!(tsv, "alu_holo_simple\t{}\t{}\t{}\t{}\t42\t{:.6}\t-\t-\t-\t-",
            dim, n_proj, patterns.len(), n_classes, accuracy).ok();
    }
}

// ============================================================
// TEST 7: Raw HoloLayer output inspection
// ============================================================

fn test7_raw_output(log: &mut LogWriter, _tsv: &mut std::fs::File) {
    log.log("");
    log.log("========================================");
    log.log("=== TEST 7: Raw HoloLayer Output ===");
    log.log("========================================");
    log.log("  What does the score vector actually look like?");

    let dim = 64;
    let n_bits = 8;
    let n_classes = 16;

    // Test A: Few patterns (should work)
    log.log("");
    log.log("  --- A) 2 patterns stored (dim=64, 16 classes) ---");
    {
        let mut rng = Rng::new(42);
        let patterns = generate_patterns(2, n_bits, n_classes, &mut rng);
        let mut holo = HoloLayer::new(dim, n_bits, 42);

        for (bits, target) in &patterns {
            holo.store(bits, *target);
        }

        for (idx, (bits, target)) in patterns.iter().enumerate() {
            let scores = holo.predict_scores(bits);
            let pred = scores.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i,_)| i).unwrap_or(0);

            // Show top-5 scores and the target score
            let mut indexed: Vec<(usize, f32)> = scores.iter().enumerate()
                .map(|(i, &s)| (i, s)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            log.log(&format!("    Pattern {}: target={}, pred={} {}",
                idx, target, pred, if pred == *target { "OK" } else { "FAIL" }));
            log.log(&format!("      Top-5 scores:"));
            for &(i, s) in indexed.iter().take(5) {
                let marker = if i == *target { " ← TARGET" } else { "" };
                log.log(&format!("        class {:>2}: {:>+8.4}{}", i, s, marker));
            }
            // Show full score vector (first 16 classes only)
            let first16: Vec<String> = scores.iter().take(16)
                .map(|s| format!("{:>+6.3}", s)).collect();
            log.log(&format!("      Full scores [0..15]: [{}]", first16.join(", ")));
        }
    }

    // Test B: 16 patterns
    log.log("");
    log.log("  --- B) 16 patterns stored ---");
    {
        let mut rng = Rng::new(42);
        let patterns = generate_patterns(16, n_bits, n_classes, &mut rng);
        let mut holo = HoloLayer::new(dim, n_bits, 42);

        for (bits, target) in &patterns {
            holo.store(bits, *target);
        }

        for (idx, (bits, target)) in patterns.iter().enumerate().take(4) {
            let scores = holo.predict_scores(bits);
            let pred = scores.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i,_)| i).unwrap_or(0);

            let mut indexed: Vec<(usize, f32)> = scores.iter().enumerate()
                .map(|(i, &s)| (i, s)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            log.log(&format!("    Pattern {}: target={}, pred={} {}",
                idx, target, pred, if pred == *target { "OK" } else { "FAIL" }));
            let first16: Vec<String> = scores.iter().take(16)
                .map(|s| format!("{:>+6.3}", s)).collect();
            log.log(&format!("      Scores [0..15]: [{}]", first16.join(", ")));
        }
    }

    // Test C: 64 patterns — show the score "landscape"
    log.log("");
    log.log("  --- C) 64 patterns stored — score statistics ---");
    {
        let mut rng = Rng::new(42);
        let patterns = generate_patterns(64, n_bits, n_classes, &mut rng);
        let mut holo = HoloLayer::new(dim, n_bits, 42);

        for (bits, target) in &patterns {
            holo.store(bits, *target);
        }

        let mut target_scores: Vec<f32> = Vec::new();
        let mut non_target_scores: Vec<f32> = Vec::new();
        let mut all_maxes: Vec<f32> = Vec::new();

        for (bits, target) in &patterns {
            let scores = holo.predict_scores(bits);
            target_scores.push(scores[*target]);
            all_maxes.push(scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
            for (i, &s) in scores.iter().enumerate().take(n_classes) {
                if i != *target { non_target_scores.push(s); }
            }
        }

        let ts_mean: f32 = target_scores.iter().sum::<f32>() / target_scores.len() as f32;
        let ts_min: f32 = target_scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let ts_max: f32 = target_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let ns_mean: f32 = non_target_scores.iter().sum::<f32>() / non_target_scores.len() as f32;
        let ns_min: f32 = non_target_scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let ns_max: f32 = non_target_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        log.log(&format!("    Target scores:     mean={:>+7.4}  min={:>+7.4}  max={:>+7.4}", ts_mean, ts_min, ts_max));
        log.log(&format!("    Non-target scores: mean={:>+7.4}  min={:>+7.4}  max={:>+7.4}", ns_mean, ns_min, ns_max));
        log.log(&format!("    Separation gap:    {:.4} (target_mean - nontarget_mean)", ts_mean - ns_mean));
        log.log(&format!("    Overlap?           target_min={:>+.4} vs nontarget_max={:>+.4} → {}",
            ts_min, ns_max,
            if ts_min > ns_max { "NO OVERLAP — should be separable!" } else { "OVERLAP — scores bleed into each other" }));

        // Show a few raw examples
        log.log("");
        log.log("    First 3 patterns detail:");
        for (idx, (bits, target)) in patterns.iter().enumerate().take(3) {
            let scores = holo.predict_scores(bits);
            let first16: Vec<String> = scores.iter().take(16)
                .map(|s| format!("{:>+6.3}", s)).collect();
            log.log(&format!("      pat{}: target={} scores=[{}]", idx, target, first16.join(",")));
        }
    }

    // Test D: ALU output scores
    log.log("");
    log.log("  --- D) ALU → Holo: raw score vectors (dim=64) ---");
    {
        let alu = ParallelAlu::new();
        let mut holo = HoloLayer::new(64, 28, 42);

        // Store a subset: a=0..3, b=0..3, all 5 ops = 80 patterns
        let mut patterns: Vec<(Vec<u8>, usize)> = Vec::new();
        for a in 0..4u8 {
            for b in 0..4u8 {
                let all_results = alu.execute_all(a, b);
                for op in 0..5usize {
                    let result = all_results[op];
                    let mut bits = Vec::with_capacity(28);
                    for bit in 0..4 { bits.push((a >> bit) & 1); }
                    for bit in 0..4 { bits.push((b >> bit) & 1); }
                    for bit in 0..4 { bits.push((result >> bit) & 1); }
                    for r in &all_results {
                        for bit in 0..4 { bits.push((r >> bit) & 1); }
                    }
                    patterns.push((bits, op));
                }
            }
        }

        for (bits, target) in &patterns {
            holo.store(bits, *target);
        }

        let op_names = ["ADD", "SUB", "AND", "OR", "XOR"];
        log.log(&format!("    {} patterns stored (a,b=0..3, 5 ops)", patterns.len()));

        // Show first 10 patterns
        for (idx, (bits, target)) in patterns.iter().enumerate().take(10) {
            let scores = holo.predict_scores(bits);
            let top5: Vec<String> = scores.iter().take(5)
                .enumerate()
                .map(|(i, s)| {
                    let mark = if i == *target { "*" } else { " " };
                    format!("{}{}={:>+6.3}", op_names[i], mark, s)
                }).collect();
            log.log(&format!("      pat{:>2} (target={}): [{}]",
                idx, op_names[*target], top5.join(", ")));
        }
    }
}

// ============================================================
// MAIN
// ============================================================

fn main() {
    let log_path = "instnct-core/holo_diagnostic_log.txt";
    let tsv_path = "instnct-core/holo_diagnostic_results.tsv";

    let mut log = LogWriter::new(log_path);
    let mut tsv = std::fs::File::create(tsv_path).unwrap();

    writeln!(tsv, "test\tdim\tn_proj\tn_patterns\tn_classes\tseed\taccuracy\tsnr_or_stddev\tmargin_or_mid\tmean_cos_or_late\tmax_cos").ok();

    log.log("╔══════════════════════════════════════════════════╗");
    log.log("║  HOLOGRAPHIC MEMORY DIAGNOSTICS                 ║");
    log.log("║  Why does HoloLayer fail at 256 patterns?       ║");
    log.log("╚══════════════════════════════════════════════════╝");

    let t0 = std::time::Instant::now();

    test1_interference_curve(&mut log, &mut tsv);
    test2_dim_scaling(&mut log, &mut tsv);
    test3_encoding_quality(&mut log, &mut tsv);
    test4_order_sensitivity(&mut log, &mut tsv);
    test5_forgetting_curve(&mut log, &mut tsv);
    test6_alu_then_holo(&mut log, &mut tsv);
    test7_raw_output(&mut log, &mut tsv);

    // Summary
    log.log("");
    log.log("========================================");
    log.log("=== SUMMARY ===");
    log.log("========================================");
    log.log(&format!("  Total time: {:.2}s", t0.elapsed().as_secs_f64()));
    log.log(&format!("  Results: {}", tsv_path));
    log.log(&format!("  Log: {}", log_path));
    log.log("=== DONE ===");
}
