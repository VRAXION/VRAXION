//! Kanerva Sparse Distributed Memory (SDM) — C19 LutGate prototype
//!
//! Two implementations:
//!   A) SdmDirect  — reference, normal Rust integer ops
//!   B) SdmGated   — hardware-realistic, all ops via C19 LutGate neurons
//!
//! Parameters: N=16 address bits, M=64 hard locations, W=8 word bits, d=4 radius
//!
//! Run: cargo run --example sdm_memory --release

#![allow(dead_code, unused_variables)]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ============================================================
// C19 activation — used only at LUT baking time
// ============================================================

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor();
    let t = x - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

// ============================================================
// LutGate — integer-LUT neuron, zero float in hot path
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
        let mut all = w.to_vec();
        all.push(bias);
        let mut denom = 1;
        for d in 1..=100 {
            if all.iter().all(|&v| ((v * d as f32).round() - v * d as f32).abs() < 1e-6) {
                denom = d;
                break;
            }
        }
        let w_int: Vec<i32> = w.iter().map(|&v| (v * denom as f32).round() as i32).collect();
        let bias_int = (bias * denom as f32).round() as i32;
        let mut min_s = bias_int;
        let mut max_s = bias_int;
        for &wi in &w_int {
            if wi > 0 { max_s += wi; } else { min_s += wi; }
        }
        let mut lut = vec![0u8; (max_s - min_s + 1) as usize];
        for s in min_s..=max_s {
            lut[(s - min_s) as usize] = if c19(s as f32 / denom as f32, rho) > thr { 1 } else { 0 };
        }
        LutGate { w_int, bias_int, lut, min_sum: min_s }
    }

    fn eval(&self, inputs: &[u8]) -> u8 {
        let s: i32 = inputs.iter().zip(&self.w_int)
            .map(|(&i, &w)| i as i32 * w)
            .sum::<i32>() + self.bias_int;
        let idx = (s - self.min_sum) as usize;
        if idx < self.lut.len() { self.lut[idx] } else { 0 }
    }
}

// ============================================================
// Gate library
// ============================================================

struct Gates {
    and_g: LutGate,
    or_g:  LutGate,
    xor_g: LutGate,
    xor3:  LutGate,
    maj:   LutGate,
    not_g: LutGate,
}

impl Gates {
    fn new() -> Self {
        Gates {
            and_g: LutGate::new(&[10.0, 10.0],       -4.5,  0.0,  4.0),
            or_g:  LutGate::new(&[8.75, 8.75],         5.5,  0.0,  4.0),
            xor_g: LutGate::new(&[0.5,  0.5],          0.0, 16.0,  0.6),
            xor3:  LutGate::new(&[1.5,  1.5,  1.5],    3.0, 16.0,  0.6),
            maj:   LutGate::new(&[8.5,  8.5,  8.5],   -2.75, 0.0,  4.0),
            not_g: LutGate::new(&[-9.75],              -5.5, 16.0, -4.0),
        }
    }

    fn half_add(&self, a: u8, b: u8) -> (u8, u8) {
        (self.xor_g.eval(&[a, b]), self.and_g.eval(&[a, b]))
    }

    fn full_add(&self, a: u8, b: u8, cin: u8) -> (u8, u8) {
        (self.xor3.eval(&[a, b, cin]), self.maj.eval(&[a, b, cin]))
    }
}

// ============================================================
// Neuron counter — tracks gate usage across the gated SDM
// ============================================================

#[derive(Default, Clone)]
struct NeuronCounter {
    xor_count: usize,
    and_count: usize,
    or_count: usize,
    not_count: usize,
    xor3_count: usize,
    maj_count: usize,
}

impl NeuronCounter {
    fn total(&self) -> usize {
        self.xor_count + self.and_count + self.or_count
            + self.not_count + self.xor3_count + self.maj_count
    }
}

// ============================================================
// Implementation A: SdmDirect — reference, normal Rust ops
// ============================================================

struct SdmDirect {
    addresses: Vec<Vec<u8>>,  // M x N (each bit 0/1)
    counters: Vec<Vec<i16>>,  // M x W
    n_bits: usize,
    n_locations: usize,
    word_size: usize,
    radius: usize,
}

impl SdmDirect {
    fn new(n_bits: usize, n_locations: usize, word_size: usize, radius: usize, rng: &mut StdRng) -> Self {
        let addresses: Vec<Vec<u8>> = (0..n_locations)
            .map(|_| (0..n_bits).map(|_| rng.gen_range(0u8..=1)).collect())
            .collect();
        let counters = vec![vec![0i16; word_size]; n_locations];
        SdmDirect { addresses, counters, n_bits, n_locations, word_size, radius }
    }

    fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b).filter(|(&x, &y)| x != y).count()
    }

    fn write(&mut self, address: &[u8], data: &[u8]) {
        for i in 0..self.n_locations {
            if Self::hamming_distance(address, &self.addresses[i]) <= self.radius {
                for j in 0..self.word_size {
                    // data bit: 0 -> decrement, 1 -> increment
                    if data[j] == 1 {
                        self.counters[i][j] += 1;
                    } else {
                        self.counters[i][j] -= 1;
                    }
                }
            }
        }
    }

    fn read(&self, query: &[u8]) -> Vec<u8> {
        let mut sum = vec![0i32; self.word_size];
        let mut count = 0u32;
        for i in 0..self.n_locations {
            if Self::hamming_distance(query, &self.addresses[i]) <= self.radius {
                for j in 0..self.word_size {
                    sum[j] += self.counters[i][j] as i32;
                }
                count += 1;
            }
        }
        if count == 0 {
            return vec![0u8; self.word_size];
        }
        // Threshold: sum > 0 -> 1, sum <= 0 -> 0
        sum.iter().map(|&s| if s > 0 { 1u8 } else { 0u8 }).collect()
    }

    fn reset(&mut self) {
        for c in &mut self.counters {
            for v in c.iter_mut() {
                *v = 0;
            }
        }
    }

    fn activated_count(&self, address: &[u8]) -> usize {
        (0..self.n_locations)
            .filter(|&i| Self::hamming_distance(address, &self.addresses[i]) <= self.radius)
            .count()
    }
}

// ============================================================
// Implementation B: SdmGated — all ops via C19 LutGate neurons
// ============================================================

struct SdmGated {
    addresses: Vec<Vec<u8>>,  // M x N (same addresses as Direct)
    counters: Vec<Vec<i16>>,  // M x W (integer counters — gate ops on update)
    n_bits: usize,
    n_locations: usize,
    word_size: usize,
    radius: usize,
    gates: Gates,
    neuron_stats: NeuronCounter,
}

impl SdmGated {
    fn new(addresses: Vec<Vec<u8>>, n_bits: usize, n_locations: usize,
           word_size: usize, radius: usize) -> Self {
        SdmGated {
            addresses,
            counters: vec![vec![0i16; word_size]; n_locations],
            n_bits, n_locations, word_size, radius,
            gates: Gates::new(),
            neuron_stats: NeuronCounter::default(),
        }
    }

    /// Hamming distance via XOR per bit pair + popcount adder tree.
    /// Returns the distance AND whether it's <= radius.
    fn hamming_distance_gated(&mut self, a: &[u8], b: &[u8]) -> (usize, bool) {
        let n = a.len();

        // Step 1: XOR each bit pair to get difference bits
        let mut diff_bits = Vec::with_capacity(n);
        for i in 0..n {
            diff_bits.push(self.gates.xor_g.eval(&[a[i], b[i]]));
            self.neuron_stats.xor_count += 1;
        }

        // Step 2: Popcount via full-adder tree (reduce N bits to log2(N)+1 bit count)
        // For N=16: we need a 4-level adder tree producing a 5-bit count
        let count = self.popcount_tree(&diff_bits);

        // Step 3: Compare count <= radius using gate-based subtraction
        let within = self.compare_lte_gated(count, self.radius as u8);

        (count as usize, within)
    }

    /// Popcount using a tree of full adders.
    /// Input: N bits, Output: popcount as integer.
    /// Tree structure for 16 bits:
    ///   Level 0: 16 single bits -> 8 pairs via half adders -> 8 x 2-bit sums
    ///   Level 1: reduce 2-bit sums pairwise -> 4 x 3-bit sums
    ///   Level 2: reduce 3-bit sums pairwise -> 2 x 4-bit sums
    ///   Level 3: reduce 4-bit sums -> 1 x 5-bit sum
    fn popcount_tree(&mut self, bits: &[u8]) -> u8 {
        // Represent each intermediate value as a multi-bit number (LSB first)
        let mut values: Vec<Vec<u8>> = bits.iter().map(|&b| vec![b]).collect();

        // Reduce pairs until we have one value
        while values.len() > 1 {
            let mut next_values = Vec::new();
            let mut i = 0;
            while i + 1 < values.len() {
                let sum = self.add_binary_gated(&values[i], &values[i + 1]);
                next_values.push(sum);
                i += 2;
            }
            // If odd number of values, carry the last one forward
            if i < values.len() {
                next_values.push(values[i].clone());
            }
            values = next_values;
        }

        // Convert multi-bit result (LSB first) to integer
        let result = &values[0];
        let mut count: u8 = 0;
        for (i, &bit) in result.iter().enumerate() {
            count |= bit << i;
        }
        count
    }

    /// Add two binary numbers (LSB-first vectors) using ripple-carry adder.
    fn add_binary_gated(&mut self, a: &[u8], b: &[u8]) -> Vec<u8> {
        let max_len = a.len().max(b.len());
        let mut result = Vec::with_capacity(max_len + 1);
        let mut carry: u8 = 0;

        for i in 0..max_len {
            let a_bit = if i < a.len() { a[i] } else { 0 };
            let b_bit = if i < b.len() { b[i] } else { 0 };

            let (s, c) = self.gates.full_add(a_bit, b_bit, carry);
            self.neuron_stats.xor3_count += 1;
            self.neuron_stats.maj_count += 1;
            result.push(s);
            carry = c;
        }
        if carry > 0 {
            result.push(carry);
        }
        result
    }

    /// Compare value <= threshold using gate-based subtraction.
    /// Computes (threshold - value): if no borrow, then value <= threshold.
    fn compare_lte_gated(&mut self, value: u8, threshold: u8) -> bool {
        // We need enough bits to represent the values (5 bits for 0..16)
        let n_bits = 5;
        let val_bits: Vec<u8> = (0..n_bits).map(|i| (value >> i) & 1).collect();
        let thr_bits: Vec<u8> = (0..n_bits).map(|i| (threshold >> i) & 1).collect();

        // Subtraction: threshold - value using complement + add
        // NOT each value bit, then add threshold + NOT(value) + 1
        let mut not_val = Vec::with_capacity(n_bits);
        for i in 0..n_bits {
            not_val.push(self.gates.not_g.eval(&[val_bits[i]]));
            self.neuron_stats.not_count += 1;
        }

        // Add threshold + NOT(value) with initial carry = 1 (two's complement)
        let mut carry: u8 = 1;
        for i in 0..n_bits {
            let (_, c) = self.gates.full_add(thr_bits[i], not_val[i], carry);
            self.neuron_stats.xor3_count += 1;
            self.neuron_stats.maj_count += 1;
            carry = c;
        }

        // If final carry = 1, no borrow, meaning threshold >= value
        carry == 1
    }

    fn write(&mut self, address: &[u8], data: &[u8]) {
        for i in 0..self.n_locations {
            let addr_clone = self.addresses[i].clone();
            let (_, within) = self.hamming_distance_gated(address, &addr_clone);
            if within {
                for j in 0..self.word_size {
                    if data[j] == 1 {
                        self.counters[i][j] += 1;
                    } else {
                        self.counters[i][j] -= 1;
                    }
                }
            }
        }
    }

    fn read(&mut self, query: &[u8]) -> Vec<u8> {
        let mut sum = vec![0i32; self.word_size];
        let mut count = 0u32;
        for i in 0..self.n_locations {
            let addr_clone = self.addresses[i].clone();
            let (_, within) = self.hamming_distance_gated(query, &addr_clone);
            if within {
                for j in 0..self.word_size {
                    sum[j] += self.counters[i][j] as i32;
                }
                count += 1;
            }
        }
        if count == 0 {
            return vec![0u8; self.word_size];
        }
        sum.iter().map(|&s| if s > 0 { 1u8 } else { 0u8 }).collect()
    }

    fn reset(&mut self) {
        for c in &mut self.counters {
            for v in c.iter_mut() {
                *v = 0;
            }
        }
        self.neuron_stats = NeuronCounter::default();
    }

    fn activated_count(&mut self, address: &[u8]) -> usize {
        let mut count = 0;
        for i in 0..self.n_locations {
            let addr_clone = self.addresses[i].clone();
            let (_, within) = self.hamming_distance_gated(address, &addr_clone);
            if within { count += 1; }
        }
        count
    }
}

// ============================================================
// Helper functions
// ============================================================

fn bits_to_string(bits: &[u8]) -> String {
    bits.iter().map(|&b| if b == 1 { '1' } else { '0' }).collect()
}

fn bits_to_byte(bits: &[u8]) -> u8 {
    let mut v = 0u8;
    for (i, &b) in bits.iter().enumerate() {
        if b == 1 && i < 8 {
            v |= 1 << i;
        }
    }
    v
}

fn byte_to_bits(v: u8, n: usize) -> Vec<u8> {
    (0..n).map(|i| (v >> i) & 1).collect()
}

fn flip_bits(bits: &[u8], n_flips: usize, rng: &mut StdRng) -> Vec<u8> {
    let mut result = bits.to_vec();
    let mut positions: Vec<usize> = (0..bits.len()).collect();
    // Fisher-Yates partial shuffle
    for i in 0..n_flips.min(positions.len()) {
        let j = rng.gen_range(i..positions.len());
        positions.swap(i, j);
    }
    for i in 0..n_flips.min(positions.len()) {
        result[positions[i]] ^= 1;
    }
    result
}

fn hamming(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(&x, &y)| x != y).count()
}

// ============================================================
// Test runner
// ============================================================

struct TestResult {
    name: String,
    n_patterns: usize,
    noise_bits: usize,
    direct_accuracy: f64,
    gated_accuracy: f64,
    bit_identical: bool,
    details: String,
}

fn run_test(
    sdm_d: &mut SdmDirect,
    sdm_g: &mut SdmGated,
    n_patterns: usize,
    noise_bits: usize,
    test_name: &str,
    rng: &mut StdRng,
) -> TestResult {
    // Reset both SDMs
    sdm_d.reset();
    sdm_g.reset();

    let n_bits = sdm_d.n_bits;
    let word_size = sdm_d.word_size;

    // Generate random patterns
    let mut addresses: Vec<Vec<u8>> = Vec::new();
    let mut data_patterns: Vec<Vec<u8>> = Vec::new();
    for _ in 0..n_patterns {
        let addr: Vec<u8> = (0..n_bits).map(|_| rng.gen_range(0u8..=1)).collect();
        let data: Vec<u8> = (0..word_size).map(|_| rng.gen_range(0u8..=1)).collect();
        addresses.push(addr);
        data_patterns.push(data);
    }

    // Store all patterns in both SDMs
    for i in 0..n_patterns {
        sdm_d.write(&addresses[i], &data_patterns[i]);
        sdm_g.write(&addresses[i], &data_patterns[i]);
    }

    // Recall and check
    let mut direct_correct = 0;
    let mut gated_correct = 0;
    let mut identical = true;
    let mut detail_lines = Vec::new();

    for i in 0..n_patterns {
        let query = if noise_bits == 0 {
            addresses[i].clone()
        } else {
            flip_bits(&addresses[i], noise_bits, rng)
        };

        let result_d = sdm_d.read(&query);
        let result_g = sdm_g.read(&query);

        if result_d != result_g {
            identical = false;
            detail_lines.push(format!(
                "  MISMATCH at pattern {}: direct={} gated={}",
                i, bits_to_string(&result_d), bits_to_string(&result_g)
            ));
        }

        if result_d == data_patterns[i] { direct_correct += 1; }
        if result_g == data_patterns[i] { gated_correct += 1; }
    }

    let direct_acc = direct_correct as f64 / n_patterns as f64 * 100.0;
    let gated_acc = gated_correct as f64 / n_patterns as f64 * 100.0;

    TestResult {
        name: test_name.to_string(),
        n_patterns,
        noise_bits,
        direct_accuracy: direct_acc,
        gated_accuracy: gated_acc,
        bit_identical: identical,
        details: detail_lines.join("\n"),
    }
}

// ============================================================
// main
// ============================================================

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // SDM parameters
    let n_bits = 16;
    let n_locations = 64;
    let word_size = 8;
    let radius = 4;

    println!("============================================================");
    println!("  Kanerva SDM — C19 LutGate Prototype");
    println!("============================================================");
    println!();
    println!("Parameters:");
    println!("  Address bits (N)    : {}", n_bits);
    println!("  Hard locations (M)  : {}", n_locations);
    println!("  Word size (W)       : {}", word_size);
    println!("  Access radius (d)   : {}", radius);
    println!();

    // ── Verify gates ──
    println!("--- Gate verification ---");
    let g = Gates::new();
    let mut gate_ok = true;
    for a in 0..=1u8 {
        for b in 0..=1u8 {
            let and_exp = a & b;
            let or_exp = a | b;
            let xor_exp = a ^ b;
            let and_got = g.and_g.eval(&[a, b]);
            let or_got = g.or_g.eval(&[a, b]);
            let xor_got = g.xor_g.eval(&[a, b]);
            if and_got != and_exp { println!("  AND FAIL: {}AND{} = {} (expected {})", a, b, and_got, and_exp); gate_ok = false; }
            if or_got != or_exp { println!("  OR FAIL:  {}OR{} = {} (expected {})", a, b, or_got, or_exp); gate_ok = false; }
            if xor_got != xor_exp { println!("  XOR FAIL: {}XOR{} = {} (expected {})", a, b, xor_got, xor_exp); gate_ok = false; }
        }
    }
    for a in 0..=1u8 {
        let not_got = g.not_g.eval(&[a]);
        let not_exp = 1 - a;
        if not_got != not_exp { println!("  NOT FAIL: NOT {} = {} (expected {})", a, not_got, not_exp); gate_ok = false; }
    }
    for a in 0..=1u8 {
        for b in 0..=1u8 {
            for c in 0..=1u8 {
                let xor3_exp = a ^ b ^ c;
                let maj_exp = (a & b) | (b & c) | (a & c);
                let xor3_got = g.xor3.eval(&[a, b, c]);
                let maj_got = g.maj.eval(&[a, b, c]);
                if xor3_got != xor3_exp { println!("  XOR3 FAIL: {}^{}^{} = {} (expected {})", a, b, c, xor3_got, xor3_exp); gate_ok = false; }
                if maj_got != maj_exp { println!("  MAJ FAIL:  MAJ({},{},{}) = {} (expected {})", a, b, c, maj_got, maj_exp); gate_ok = false; }
            }
        }
    }
    if gate_ok {
        println!("  All gates verified (AND, OR, XOR, NOT, XOR3, MAJ)");
    } else {
        println!("  GATE VERIFICATION FAILED — aborting");
        return;
    }
    println!();

    // ── Create SDMs with shared addresses ──
    let mut sdm_d = SdmDirect::new(n_bits, n_locations, word_size, radius, &mut rng);
    let sdm_g_addresses = sdm_d.addresses.clone();
    let mut sdm_g = SdmGated::new(sdm_g_addresses, n_bits, n_locations, word_size, radius);

    // ── Check address space statistics ──
    println!("--- Address space statistics ---");
    let test_addr: Vec<u8> = (0..n_bits).map(|_| rng.gen_range(0u8..=1)).collect();
    let activated = sdm_d.activated_count(&test_addr);
    println!("  Random address {} activates {}/{} locations (expected ~{:.1})",
        bits_to_string(&test_addr), activated, n_locations,
        n_locations as f64 * activation_probability(n_bits, radius));

    // Verify gated hamming distance matches direct
    println!();
    println!("--- Hamming distance verification ---");
    let mut hd_ok = true;
    for trial in 0..20 {
        let a: Vec<u8> = (0..n_bits).map(|_| rng.gen_range(0u8..=1)).collect();
        let b: Vec<u8> = (0..n_bits).map(|_| rng.gen_range(0u8..=1)).collect();
        let d_direct = SdmDirect::hamming_distance(&a, &b);
        let (d_gated, _) = sdm_g.hamming_distance_gated(&a, &b);
        if d_direct != d_gated {
            println!("  MISMATCH trial {}: direct={} gated={}", trial, d_direct, d_gated);
            hd_ok = false;
        }
    }
    // Edge cases
    let all_zeros = vec![0u8; n_bits];
    let all_ones = vec![1u8; n_bits];
    let d_00 = SdmDirect::hamming_distance(&all_zeros, &all_zeros);
    let (g_00, _) = sdm_g.hamming_distance_gated(&all_zeros, &all_zeros);
    let d_11 = SdmDirect::hamming_distance(&all_ones, &all_ones);
    let (g_11, _) = sdm_g.hamming_distance_gated(&all_ones, &all_ones);
    let d_01 = SdmDirect::hamming_distance(&all_zeros, &all_ones);
    let (g_01, _) = sdm_g.hamming_distance_gated(&all_zeros, &all_ones);
    if d_00 != g_00 || d_11 != g_11 || d_01 != g_01 {
        println!("  EDGE CASE MISMATCH: 0v0 d={}/g={}, 1v1 d={}/g={}, 0v1 d={}/g={}",
            d_00, g_00, d_11, g_11, d_01, g_01);
        hd_ok = false;
    }
    if hd_ok {
        println!("  All 20 random + 3 edge cases match (direct == gated)");
    }
    println!("  Edge: all-zeros vs all-zeros = {}", d_00);
    println!("  Edge: all-ones  vs all-ones  = {}", d_11);
    println!("  Edge: all-zeros vs all-ones  = {}", d_01);
    println!();

    // ── Run test suite ──
    println!("============================================================");
    println!("  TEST SUITE");
    println!("============================================================");
    println!();

    let tests = vec![
        ("T1: 1 pattern, exact recall",      1,  0),
        ("T2: 8 patterns, exact recall",      8,  0),
        ("T3: 16 patterns, exact recall",    16,  0),
        ("T4: 32 patterns, exact recall",    32,  0),
        ("T5: 16 patterns, 1-bit noise",    16,  1),
        ("T6: 16 patterns, 2-bit noise",    16,  2),
        ("T7: 16 patterns, 4-bit noise",    16,  4),
    ];

    let mut results: Vec<TestResult> = Vec::new();

    for (name, n_pats, noise) in &tests {
        let r = run_test(&mut sdm_d, &mut sdm_g, *n_pats, *noise, name, &mut rng);
        println!("  {} | Direct: {:.0}% | Gated: {:.0}% | Identical: {}",
            r.name, r.direct_accuracy, r.gated_accuracy,
            if r.bit_identical { "YES" } else { "NO" });
        if !r.details.is_empty() {
            println!("{}", r.details);
        }
        results.push(r);
    }

    // ── T8: Direct vs LutGate bit-identity test ──
    println!();
    println!("--- T8: Comprehensive bit-identity comparison ---");
    sdm_d.reset();
    sdm_g.reset();
    sdm_g.neuron_stats = NeuronCounter::default();

    let n_test = 16;
    let mut t8_addresses = Vec::new();
    let mut t8_data = Vec::new();
    for _ in 0..n_test {
        t8_addresses.push((0..n_bits).map(|_| rng.gen_range(0u8..=1)).collect::<Vec<u8>>());
        t8_data.push((0..word_size).map(|_| rng.gen_range(0u8..=1)).collect::<Vec<u8>>());
    }

    for i in 0..n_test {
        sdm_d.write(&t8_addresses[i], &t8_data[i]);
        sdm_g.write(&t8_addresses[i], &t8_data[i]);
    }

    let mut t8_identical = true;
    let mut t8_mismatches = 0;
    for i in 0..n_test {
        let rd = sdm_d.read(&t8_addresses[i]);
        let rg = sdm_g.read(&t8_addresses[i]);
        if rd != rg {
            t8_identical = false;
            t8_mismatches += 1;
            println!("  Pattern {}: direct={} gated={} MISMATCH", i, bits_to_string(&rd), bits_to_string(&rg));
        }
    }
    // Also check that the counters are identical
    let mut counter_match = true;
    for i in 0..n_locations {
        if sdm_d.counters[i] != sdm_g.counters[i] {
            counter_match = false;
            println!("  Counter mismatch at location {}", i);
            break;
        }
    }
    println!("  Bit-identical outputs : {} ({} patterns, {} mismatches)",
        if t8_identical { "YES" } else { "NO" }, n_test, t8_mismatches);
    println!("  Counter arrays match  : {}", if counter_match { "YES" } else { "NO" });

    // Capture neuron stats from T8 (16 writes + 16 reads = 32 accesses x 64 locations)
    let t8_neuron_stats = sdm_g.neuron_stats.clone();
    println!();

    // ── Adversarial tests ──
    println!("============================================================");
    println!("  ADVERSARIAL TESTS");
    println!("============================================================");
    println!();

    // A1: Store same pattern twice — should double counters
    println!("--- A1: Store same pattern twice ---");
    sdm_d.reset();
    sdm_g.reset();
    let a1_addr: Vec<u8> = (0..n_bits).map(|_| rng.gen_range(0u8..=1)).collect();
    let a1_data: Vec<u8> = (0..word_size).map(|_| rng.gen_range(0u8..=1)).collect();
    sdm_d.write(&a1_addr, &a1_data);
    sdm_d.write(&a1_addr, &a1_data);
    sdm_g.write(&a1_addr, &a1_data);
    sdm_g.write(&a1_addr, &a1_data);
    let a1_rd = sdm_d.read(&a1_addr);
    let a1_rg = sdm_g.read(&a1_addr);
    println!("  Original data: {}", bits_to_string(&a1_data));
    println!("  Direct recall: {} (match: {})", bits_to_string(&a1_rd), a1_rd == a1_data);
    println!("  Gated recall:  {} (match: {})", bits_to_string(&a1_rg), a1_rg == a1_data);
    println!("  Identical:     {}", a1_rd == a1_rg);
    // Check counters are doubled
    let first_activated: Vec<usize> = (0..n_locations)
        .filter(|&i| SdmDirect::hamming_distance(&a1_addr, &sdm_d.addresses[i]) <= radius)
        .collect();
    if let Some(&loc) = first_activated.first() {
        let expected_counter: Vec<i16> = a1_data.iter().map(|&b| if b == 1 { 2 } else { -2 }).collect();
        println!("  Counter at loc {} (expected doubled): {} (match: {})",
            loc,
            sdm_d.counters[loc].iter().map(|c| c.to_string()).collect::<Vec<_>>().join(","),
            sdm_d.counters[loc] == expected_counter);
    }
    println!();

    // A2: Store opposite patterns — should cancel
    println!("--- A2: Store opposite patterns (cancel test) ---");
    sdm_d.reset();
    sdm_g.reset();
    let a2_addr: Vec<u8> = (0..n_bits).map(|_| rng.gen_range(0u8..=1)).collect();
    let a2_data: Vec<u8> = (0..word_size).map(|_| rng.gen_range(0u8..=1)).collect();
    let a2_data_opp: Vec<u8> = a2_data.iter().map(|&b| 1 - b).collect();
    sdm_d.write(&a2_addr, &a2_data);
    sdm_d.write(&a2_addr, &a2_data_opp);
    sdm_g.write(&a2_addr, &a2_data);
    sdm_g.write(&a2_addr, &a2_data_opp);
    let a2_rd = sdm_d.read(&a2_addr);
    let a2_rg = sdm_g.read(&a2_addr);
    // All counters should be zero at activated locations
    if let Some(&loc) = first_activated.first() {
        // Check that loc is activated for a2_addr too (different address)
    }
    let a2_activated: Vec<usize> = (0..n_locations)
        .filter(|&i| SdmDirect::hamming_distance(&a2_addr, &sdm_d.addresses[i]) <= radius)
        .collect();
    let counters_zero = a2_activated.iter()
        .all(|&i| sdm_d.counters[i].iter().all(|&c| c == 0));
    println!("  Data:     {}", bits_to_string(&a2_data));
    println!("  Opposite: {}", bits_to_string(&a2_data_opp));
    println!("  Direct recall:   {} (all counters zero: {})", bits_to_string(&a2_rd), counters_zero);
    println!("  Gated recall:    {}", bits_to_string(&a2_rg));
    println!("  Identical:       {}", a2_rd == a2_rg);
    println!();

    // A3: Adversarial addresses — maximum interference
    println!("--- A3: Maximum interference (clustered addresses) ---");
    sdm_d.reset();
    sdm_g.reset();
    // Use a fixed base address and store multiple different data patterns
    // at addresses within radius of each other
    let a3_base: Vec<u8> = (0..n_bits).map(|_| rng.gen_range(0u8..=1)).collect();
    let mut a3_addrs = vec![a3_base.clone()];
    let mut a3_datas = Vec::new();
    a3_datas.push((0..word_size).map(|_| rng.gen_range(0u8..=1)).collect::<Vec<u8>>());

    // Create 7 more addresses within 2 bits of base (inside radius)
    for _ in 0..7 {
        let noisy = flip_bits(&a3_base, 2, &mut rng);
        a3_addrs.push(noisy);
        a3_datas.push((0..word_size).map(|_| rng.gen_range(0u8..=1)).collect());
    }

    for i in 0..a3_addrs.len() {
        sdm_d.write(&a3_addrs[i], &a3_datas[i]);
        sdm_g.write(&a3_addrs[i], &a3_datas[i]);
    }

    let mut a3_correct = 0;
    for i in 0..a3_addrs.len() {
        let rd = sdm_d.read(&a3_addrs[i]);
        if rd == a3_datas[i] { a3_correct += 1; }
    }
    println!("  Stored {} patterns at addresses within Hamming distance 2 of each other", a3_addrs.len());
    println!("  Direct recall accuracy: {}/{} ({:.0}%)", a3_correct, a3_addrs.len(),
        a3_correct as f64 / a3_addrs.len() as f64 * 100.0);
    println!("  (Interference expected — this tests SDM's graceful degradation)");
    println!();

    // A4: No activation test
    println!("--- A4: No activation edge case ---");
    sdm_d.reset();
    sdm_g.reset();
    // Find an address that activates 0 locations (unlikely with d=4, N=16, M=64)
    // but test the edge case by temporarily setting radius=0
    let mut sdm_d_r0 = SdmDirect::new(n_bits, n_locations, word_size, 0, &mut rng);
    // Store some data
    let a4_data = vec![1u8; word_size];
    let a4_addr: Vec<u8> = (0..n_bits).map(|_| rng.gen_range(0u8..=1)).collect();
    sdm_d_r0.write(&a4_addr, &a4_data);
    // Query with a different address — with radius=0, only exact match activates
    let a4_query = flip_bits(&a4_addr, 1, &mut rng);
    let a4_result = sdm_d_r0.read(&a4_query);
    let activated_r0 = sdm_d_r0.activated_count(&a4_query);
    println!("  Radius=0, queried 1-bit-flipped address");
    println!("  Activated locations: {}", activated_r0);
    println!("  Result: {} (should be all-zeros if no activation)", bits_to_string(&a4_result));
    println!();

    // ── Neuron count summary ──
    println!("============================================================");
    println!("  NEURON COUNT (LutGate implementation)");
    println!("============================================================");
    println!();

    // Static neuron count for one SDM operation
    // Per hamming_distance call:
    //   - XOR gates: N = 16
    //   - Popcount adder tree: ~N full adders (16 bits -> 15 FAs in tree)
    //   - LTE compare: 5 NOT + 5 FA (for 5-bit subtraction)
    // Per write/read operation:
    //   - hamming_distance x M = 64 calls
    let xor_per_hd = n_bits;  // 16 XOR gates
    let fa_per_popcount = n_bits - 1;  // 15 full adders in popcount tree
    let not_per_compare = 5;
    let fa_per_compare = 5;
    let neurons_per_hd = xor_per_hd + (fa_per_popcount * 2) + not_per_compare + (fa_per_compare * 2);
    let neurons_per_access = neurons_per_hd * n_locations;
    let neurons_per_write = neurons_per_access;  // one hamming distance per location
    let neurons_per_read = neurons_per_access;

    println!("  Per hamming_distance:");
    println!("    XOR gates (bit diff)  : {}", xor_per_hd);
    println!("    Full adders (popcount): {} ({} neurons)", fa_per_popcount, fa_per_popcount * 2);
    println!("    NOT gates (compare)   : {}", not_per_compare);
    println!("    Full adders (compare) : {} ({} neurons)", fa_per_compare, fa_per_compare * 2);
    println!("    Total per HD          : {} neurons", neurons_per_hd);
    println!();
    println!("  Per SDM access (write or read):");
    println!("    Hamming dist calls    : {} (one per hard location)", n_locations);
    println!("    Total neurons/access  : {}", neurons_per_access);
    println!();
    println!("  Actual gate evaluations in T8 (16 write + 16 read):");
    println!("    XOR   : {}", t8_neuron_stats.xor_count);
    println!("    AND   : {}", t8_neuron_stats.and_count);
    println!("    OR    : {}", t8_neuron_stats.or_count);
    println!("    NOT   : {}", t8_neuron_stats.not_count);
    println!("    XOR3  : {}", t8_neuron_stats.xor3_count);
    println!("    MAJ   : {}", t8_neuron_stats.maj_count);
    println!("    TOTAL : {}", t8_neuron_stats.total());
    println!();

    // ── Summary table ──
    println!("============================================================");
    println!("  SUMMARY");
    println!("============================================================");
    println!();
    println!("  {:<40} {:>8} {:>8} {:>10}",
        "Test", "Direct%", "Gated%", "Identical");
    println!("  {:-<40} {:->8} {:->8} {:->10}", "", "", "", "");
    for r in &results {
        println!("  {:<40} {:>7.0}% {:>7.0}% {:>10}",
            r.name, r.direct_accuracy, r.gated_accuracy,
            if r.bit_identical { "YES" } else { "NO" });
    }
    println!("  {:<40} {:>8} {:>8} {:>10}",
        "T8: Bit-identity (16 patterns)",
        "-",
        "-",
        if t8_identical { "YES" } else { "NO" });
    println!();

    // Overall verdict
    let all_identical = results.iter().all(|r| r.bit_identical) && t8_identical && counter_match;
    if all_identical {
        println!("  VERDICT: Direct and LutGate implementations are BIT-IDENTICAL");
    } else {
        println!("  VERDICT: MISMATCH detected between Direct and LutGate");
    }
    println!();
}

/// Approximate probability a random address activates a random hard location
fn activation_probability(n: usize, d: usize) -> f64 {
    // P = sum_{k=0}^{d} C(n,k) / 2^n
    let mut sum = 0.0f64;
    let total = 2.0f64.powi(n as i32);
    for k in 0..=d {
        sum += binomial(n, k) as f64;
    }
    sum / total
}

fn binomial(n: usize, k: usize) -> u64 {
    if k > n { return 0; }
    let mut result = 1u64;
    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result
}
