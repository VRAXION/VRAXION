//! Mirrored Autoencoder PoC — tied-weight byte encoder with round-trip validation
//!
//! Tests whether threshold neurons with transposed (mirrored) weights can
//! losslessly encode ASCII bytes through progressively smaller bottlenecks.
//!
//! Architecture per level:
//!   Input (N bits) → Encoder neurons (M < N, ternary weights W, thresholds)
//!                  → Decoder neurons (N bits, weights = Wᵀ, own thresholds)
//!   Fitness = round-trip accuracy: decode(encode(byte)) == byte
//!
//! The decoder is TEMPORARY — only used to validate the encoding.
//! If round-trip works, the encoder alone is the useful output.
//!
//! Run: cargo run --example mirror_autoencoder --release

use std::time::Instant;

// ══════════════════════════════════════════════════════
// PRNG
// ══════════════════════════════════════════════════════
struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 {
        self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.s
    }
    fn pick(&mut self, n: usize) -> usize { self.next() as usize % n }
    fn ternary(&mut self) -> i8 { (self.next() % 3) as i8 - 1 } // -1, 0, +1
    fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        if lo >= hi { return lo; }
        lo + (self.next() as i32).abs() % (hi - lo + 1)
    }
}

// ══════════════════════════════════════════════════════
// DATA — extract unique bytes from Alice corpus
// ══════════════════════════════════════════════════════
fn load_unique_bytes(path: &str) -> Vec<u8> {
    let text = std::fs::read(path).expect("failed to read corpus");
    let mut seen = [false; 256];
    for &b in &text { seen[b as usize] = true; }
    let mut unique: Vec<u8> = (0..=255u8).filter(|&b| seen[b as usize]).collect();
    unique.sort();
    unique
}

fn byte_to_bits(b: u8) -> [u8; 8] {
    let mut bits = [0u8; 8];
    for i in 0..8 { bits[i] = (b >> i) & 1; }
    bits
}

fn bits_to_byte(bits: &[u8]) -> u8 {
    let mut b = 0u8;
    for (i, &v) in bits.iter().enumerate().take(8) {
        if v > 0 { b |= 1 << i; }
    }
    b
}

// ══════════════════════════════════════════════════════
// MIRRORED AUTOENCODER
// ══════════════════════════════════════════════════════
// Encoder: M neurons, each with N ternary weights + 1 threshold
// Decoder: N neurons, using TRANSPOSED weights + own thresholds
//
// Encoder neuron h_k:  fires if Σ_j W[k][j] * input[j] >= enc_thresh[k]
// Decoder neuron o_j:  fires if Σ_k W[k][j] * hidden[k] >= dec_thresh[j]
//                       ^^^^^ same W, but indexed by column = transposed access

#[derive(Clone)]
struct MirrorAutoencoder {
    n_in: usize,           // input dimension (e.g., 8 for bytes)
    n_hidden: usize,       // bottleneck size (e.g., 7)
    w: Vec<Vec<i8>>,       // W[hidden][input] — ternary weights
    enc_thresh: Vec<i32>,  // one per hidden neuron
    dec_thresh: Vec<i32>,  // one per output neuron (= input dimension)
}

impl MirrorAutoencoder {
    fn new_random(n_in: usize, n_hidden: usize, rng: &mut Rng) -> Self {
        let w: Vec<Vec<i8>> = (0..n_hidden)
            .map(|_| (0..n_in).map(|_| rng.ternary()).collect())
            .collect();
        let enc_thresh: Vec<i32> = (0..n_hidden).map(|_| rng.range_i32(-2, 3)).collect();
        let dec_thresh: Vec<i32> = (0..n_in).map(|_| rng.range_i32(-2, 3)).collect();
        MirrorAutoencoder { n_in, n_hidden, w, enc_thresh, dec_thresh }
    }

    fn encode(&self, input: &[u8]) -> Vec<u8> {
        let mut hidden = vec![0u8; self.n_hidden];
        for k in 0..self.n_hidden {
            let mut dot = 0i32;
            for j in 0..self.n_in {
                dot += self.w[k][j] as i32 * input[j] as i32;
            }
            hidden[k] = if dot >= self.enc_thresh[k] { 1 } else { 0 };
        }
        hidden
    }

    fn decode(&self, hidden: &[u8]) -> Vec<u8> {
        let mut output = vec![0u8; self.n_in];
        for j in 0..self.n_in {
            let mut dot = 0i32;
            for k in 0..self.n_hidden {
                // Transposed access: W[k][j] used as weight from h_k to o_j
                dot += self.w[k][j] as i32 * hidden[k] as i32;
            }
            output[j] = if dot >= self.dec_thresh[j] { 1 } else { 0 };
        }
        output
    }

    fn round_trip(&self, input: &[u8]) -> Vec<u8> {
        let h = self.encode(input);
        self.decode(&h)
    }

    /// Count how many bytes survive round-trip perfectly
    fn eval_accuracy(&self, unique_bytes: &[u8]) -> (usize, usize, Vec<u8>) {
        let mut correct = 0usize;
        let mut failed = Vec::new();
        for &b in unique_bytes {
            let bits_in = byte_to_bits(b);
            let bits_out = self.round_trip(&bits_in);
            if bits_in == bits_out.as_slice() {
                correct += 1;
            } else {
                failed.push(b);
            }
        }
        (correct, unique_bytes.len(), failed)
    }

    /// Count unique hidden codes produced
    fn unique_codes(&self, unique_bytes: &[u8]) -> usize {
        let mut codes: Vec<Vec<u8>> = unique_bytes.iter()
            .map(|&b| self.encode(&byte_to_bits(b)))
            .collect();
        codes.sort();
        codes.dedup();
        codes.len()
    }

    /// Total parameter count
    fn n_params(&self) -> usize {
        self.n_hidden * self.n_in + self.n_hidden + self.n_in
    }

    /// Mutate one random parameter, return undo closure info
    fn mutate(&mut self, rng: &mut Rng) -> MutateUndo {
        let total = self.n_hidden * self.n_in + self.n_hidden + self.n_in;
        let idx = rng.pick(total);

        if idx < self.n_hidden * self.n_in {
            // Weight mutation
            let k = idx / self.n_in;
            let j = idx % self.n_in;
            let old = self.w[k][j];
            self.w[k][j] = rng.ternary();
            MutateUndo::Weight(k, j, old)
        } else if idx < self.n_hidden * self.n_in + self.n_hidden {
            // Encoder threshold mutation
            let k = idx - self.n_hidden * self.n_in;
            let old = self.enc_thresh[k];
            self.enc_thresh[k] = rng.range_i32(-3, 4);
            MutateUndo::EncThresh(k, old)
        } else {
            // Decoder threshold mutation
            let j = idx - self.n_hidden * self.n_in - self.n_hidden;
            let old = self.dec_thresh[j];
            self.dec_thresh[j] = rng.range_i32(-3, 4);
            MutateUndo::DecThresh(j, old)
        }
    }

    fn undo(&mut self, u: MutateUndo) {
        match u {
            MutateUndo::Weight(k, j, old) => self.w[k][j] = old,
            MutateUndo::EncThresh(k, old) => self.enc_thresh[k] = old,
            MutateUndo::DecThresh(j, old) => self.dec_thresh[j] = old,
        }
    }
}

enum MutateUndo {
    Weight(usize, usize, i8),
    EncThresh(usize, i32),
    DecThresh(usize, i32),
}

// ══════════════════════════════════════════════════════
// SEARCH — random restart + perturbation (try-keep-revert)
// ══════════════════════════════════════════════════════
fn search_mirror(
    n_in: usize,
    n_hidden: usize,
    unique_bytes: &[u8],
    seed: u64,
    random_inits: usize,
    perturb_steps: usize,
) -> MirrorAutoencoder {
    let mut rng = Rng::new(seed);
    let target = unique_bytes.len();

    // Phase 1: random search for good starting point
    let mut best = MirrorAutoencoder::new_random(n_in, n_hidden, &mut rng);
    let (mut best_score, _, _) = best.eval_accuracy(unique_bytes);

    for i in 0..random_inits {
        let cand = MirrorAutoencoder::new_random(n_in, n_hidden, &mut rng);
        let (score, _, _) = cand.eval_accuracy(unique_bytes);
        if score > best_score {
            best_score = score;
            best = cand;
            if i < 1000 || score > best_score.max(1) - 1 {
                println!("    random [{:>7}] {}/{} ({:.1}%)",
                    i + 1, score, target, score as f64 / target as f64 * 100.0);
            }
        }
        if best_score == target { break; }
    }
    println!("    random phase done: {}/{} ({:.1}%)",
        best_score, target, best_score as f64 / target as f64 * 100.0);

    if best_score == target { return best; }

    // Phase 2: perturbation refinement
    let mut current = best.clone();
    let mut current_score = best_score;
    let mut accepts = 0u64;
    let report = perturb_steps / 20;

    for step in 0..perturb_steps {
        let undo = current.mutate(&mut rng);
        let (new_score, _, _) = current.eval_accuracy(unique_bytes);

        if new_score >= current_score {
            current_score = new_score;
            if new_score > best_score {
                best_score = new_score;
                best = current.clone();
                println!("    perturb [{:>7}] NEW BEST: {}/{} ({:.1}%)",
                    step + 1, new_score, target, new_score as f64 / target as f64 * 100.0);
            }
            accepts += 1;
        } else {
            current.undo(undo);
        }

        if best_score == target {
            println!("    perturb [{:>7}] *** PERFECT ROUND-TRIP! ***", step + 1);
            break;
        }

        if report > 0 && (step + 1) % report == 0 {
            println!("    perturb [{:>7}] best={}/{} accepts={}",
                step + 1, best_score, target, accepts);
        }
    }

    best
}

// ══════════════════════════════════════════════════════
// DISPLAY
// ══════════════════════════════════════════════════════
fn display_encoding(ae: &MirrorAutoencoder, unique_bytes: &[u8]) {
    println!("\n  Encoding table (byte → hidden code → reconstructed byte):");
    println!("  {:>6} {:>4}  {:>code_w$}  {:>6} {:>5}",
        "byte", "char", "hidden_code", "recon", "ok?",
        code_w = ae.n_hidden);

    let mut ok_count = 0;
    for &b in unique_bytes {
        let bits_in = byte_to_bits(b);
        let hidden = ae.encode(&bits_in);
        let bits_out = ae.decode(&hidden);
        let recon_b = bits_to_byte(&bits_out);
        let ok = b == recon_b;
        if ok { ok_count += 1; }

        let ch = if b >= 32 && b < 127 {
            format!("'{}'", b as char)
        } else {
            format!("x{:02x}", b)
        };
        let code: String = hidden.iter().map(|&v| if v == 1 { '1' } else { '0' }).collect();

        // Only show failures and a sample of successes
        if !ok || b == b' ' || b == b'e' || b == b't' || b == b'a' || b == b'z' || b == b'A' || b == b'\n' {
            println!("  {:>6} {:>4}  {}  {:>6} {:>5}",
                b, ch, code, recon_b, if ok { "✓" } else { "✗ FAIL" });
        }
    }
    println!("  ... ({}/{} shown above, {}/{} total correct)",
        7.min(unique_bytes.len()), unique_bytes.len(), ok_count, unique_bytes.len());
}

fn display_weights(ae: &MirrorAutoencoder) {
    println!("\n  Encoder weights W[hidden][input] (ternary):");
    println!("  {:>4}  {:>width$}  {:>6}", "h_k", "weights [b0..b7]", "thresh",
        width = ae.n_in * 3);
    for k in 0..ae.n_hidden {
        let ws: String = ae.w[k].iter().map(|&w| match w {
            -1 => " -1", 0 => "  0", 1 => " +1", _ => "  ?"
        }).collect::<Vec<_>>().join("");
        println!("  h_{:<2} [{}]  {:>4}", k, ws, ae.enc_thresh[k]);
    }
    println!("\n  Decoder thresholds (weights = Wᵀ, auto-derived):");
    let ts: Vec<String> = ae.dec_thresh.iter().map(|t| format!("{:>3}", t)).collect();
    println!("  [{}]", ts.join(", "));
}

fn display_code_collisions(ae: &MirrorAutoencoder, unique_bytes: &[u8]) {
    // Group bytes by their hidden code
    let mut code_map: std::collections::HashMap<Vec<u8>, Vec<u8>> = std::collections::HashMap::new();
    for &b in unique_bytes {
        let code = ae.encode(&byte_to_bits(b));
        code_map.entry(code).or_default().push(b);
    }

    let n_unique_codes = code_map.len();
    let collisions: Vec<_> = code_map.iter()
        .filter(|(_, bytes)| bytes.len() > 1)
        .collect();

    println!("\n  Code statistics:");
    println!("    Unique codes: {}/{} (capacity: {})", n_unique_codes, unique_bytes.len(), 1 << ae.n_hidden);
    println!("    Collisions: {} groups", collisions.len());

    if !collisions.is_empty() {
        println!("    Collision details:");
        for (code, bytes) in &collisions {
            let code_str: String = code.iter().map(|&v| if v == 1 { '1' } else { '0' }).collect();
            let byte_strs: Vec<String> = bytes.iter().map(|&b| {
                if b >= 32 && b < 127 { format!("'{}'", b as char) }
                else { format!("x{:02x}", b) }
            }).collect();
            println!("      {} → [{}]", code_str, byte_strs.join(", "));
        }
    }
}

// ══════════════════════════════════════════════════════
// CASCADED COMPRESSION
// ══════════════════════════════════════════════════════
fn cascade_level(
    input_codes: &[Vec<u8>],  // one code per unique byte
    unique_bytes: &[u8],       // original bytes for reference
    n_in: usize,
    n_hidden: usize,
    seed: u64,
) -> (MirrorAutoencoder, Vec<Vec<u8>>, usize) {
    // Build byte-equivalents: map from input code to original byte
    // We evaluate: can the hidden codes reconstruct the INPUT codes?
    let ae = search_mirror_on_codes(
        n_in, n_hidden, input_codes, seed, 200_000, 500_000,
    );

    let mut correct = 0;
    let mut output_codes = Vec::new();
    for code_in in input_codes {
        let hidden = ae.encode(code_in);
        let reconstructed = ae.decode(&hidden);
        if code_in.as_slice() == reconstructed.as_slice() {
            correct += 1;
        }
        output_codes.push(hidden);
    }

    (ae, output_codes, correct)
}

fn search_mirror_on_codes(
    n_in: usize,
    n_hidden: usize,
    input_codes: &[Vec<u8>],
    seed: u64,
    random_inits: usize,
    perturb_steps: usize,
) -> MirrorAutoencoder {
    let mut rng = Rng::new(seed);
    let target = input_codes.len();

    let eval = |ae: &MirrorAutoencoder| -> usize {
        input_codes.iter()
            .filter(|code| {
                let h = ae.encode(code);
                let r = ae.decode(&h);
                code.as_slice() == r.as_slice()
            })
            .count()
    };

    // Phase 1: random
    let mut best = MirrorAutoencoder::new_random(n_in, n_hidden, &mut rng);
    let mut best_score = eval(&best);

    for _ in 0..random_inits {
        let cand = MirrorAutoencoder::new_random(n_in, n_hidden, &mut rng);
        let score = eval(&cand);
        if score > best_score {
            best_score = score;
            best = cand;
        }
        if best_score == target { break; }
    }
    println!("      random: {}/{}", best_score, target);

    if best_score == target { return best; }

    // Phase 2: perturb
    let mut current = best.clone();
    let mut current_score = best_score;
    let report = perturb_steps / 10;

    for step in 0..perturb_steps {
        let undo = current.mutate(&mut rng);
        let new_score = eval(&current);
        if new_score >= current_score {
            current_score = new_score;
            if new_score > best_score {
                best_score = new_score;
                best = current.clone();
            }
        } else {
            current.undo(undo);
        }
        if best_score == target {
            println!("      perturb [{}]: PERFECT", step + 1);
            break;
        }
        if report > 0 && (step + 1) % report == 0 {
            println!("      perturb [{:>7}]: best={}/{}", step + 1, best_score, target);
        }
    }

    best
}

// ══════════════════════════════════════════════════════
// UNTIED AUTOENCODER (ablation: independent decoder weights)
// ══════════════════════════════════════════════════════
#[derive(Clone)]
struct UntiedAutoencoder {
    n_in: usize,
    n_hidden: usize,
    enc_w: Vec<Vec<i8>>,    // encoder W[hidden][input]
    enc_thresh: Vec<i32>,
    dec_w: Vec<Vec<i8>>,    // decoder W[output][hidden] — INDEPENDENT
    dec_thresh: Vec<i32>,
}

impl UntiedAutoencoder {
    fn new_random(n_in: usize, n_hidden: usize, rng: &mut Rng) -> Self {
        let enc_w: Vec<Vec<i8>> = (0..n_hidden).map(|_| (0..n_in).map(|_| rng.ternary()).collect()).collect();
        let enc_thresh: Vec<i32> = (0..n_hidden).map(|_| rng.range_i32(-2, 3)).collect();
        let dec_w: Vec<Vec<i8>> = (0..n_in).map(|_| (0..n_hidden).map(|_| rng.ternary()).collect()).collect();
        let dec_thresh: Vec<i32> = (0..n_in).map(|_| rng.range_i32(-2, 3)).collect();
        UntiedAutoencoder { n_in, n_hidden, enc_w, enc_thresh, dec_w, dec_thresh }
    }

    fn round_trip(&self, input: &[u8]) -> Vec<u8> {
        // Encode
        let mut hidden = vec![0u8; self.n_hidden];
        for k in 0..self.n_hidden {
            let mut dot = 0i32;
            for j in 0..self.n_in { dot += self.enc_w[k][j] as i32 * input[j] as i32; }
            hidden[k] = if dot >= self.enc_thresh[k] { 1 } else { 0 };
        }
        // Decode (independent weights)
        let mut output = vec![0u8; self.n_in];
        for j in 0..self.n_in {
            let mut dot = 0i32;
            for k in 0..self.n_hidden { dot += self.dec_w[j][k] as i32 * hidden[k] as i32; }
            output[j] = if dot >= self.dec_thresh[j] { 1 } else { 0 };
        }
        output
    }

    fn eval_accuracy_untied(&self, unique_bytes: &[u8]) -> (usize, usize, Vec<u8>) {
        let mut correct = 0;
        let mut failed = Vec::new();
        for &b in unique_bytes {
            let bits_in = byte_to_bits(b);
            let bits_out = self.round_trip(&bits_in);
            if bits_in == bits_out.as_slice() { correct += 1; } else { failed.push(b); }
        }
        (correct, unique_bytes.len(), failed)
    }

    fn mutate(&mut self, rng: &mut Rng) -> UntiedUndo {
        let enc_params = self.n_hidden * self.n_in + self.n_hidden;
        let dec_params = self.n_in * self.n_hidden + self.n_in;
        let total = enc_params + dec_params;
        let idx = rng.pick(total);

        if idx < self.n_hidden * self.n_in {
            let k = idx / self.n_in; let j = idx % self.n_in;
            let old = self.enc_w[k][j]; self.enc_w[k][j] = rng.ternary();
            UntiedUndo::EncW(k, j, old)
        } else if idx < enc_params {
            let k = idx - self.n_hidden * self.n_in;
            let old = self.enc_thresh[k]; self.enc_thresh[k] = rng.range_i32(-3, 4);
            UntiedUndo::EncT(k, old)
        } else if idx < enc_params + self.n_in * self.n_hidden {
            let local = idx - enc_params;
            let j = local / self.n_hidden; let k = local % self.n_hidden;
            let old = self.dec_w[j][k]; self.dec_w[j][k] = rng.ternary();
            UntiedUndo::DecW(j, k, old)
        } else {
            let j = idx - enc_params - self.n_in * self.n_hidden;
            let old = self.dec_thresh[j]; self.dec_thresh[j] = rng.range_i32(-3, 4);
            UntiedUndo::DecT(j, old)
        }
    }

    fn undo(&mut self, u: UntiedUndo) {
        match u {
            UntiedUndo::EncW(k, j, v) => self.enc_w[k][j] = v,
            UntiedUndo::EncT(k, v) => self.enc_thresh[k] = v,
            UntiedUndo::DecW(j, k, v) => self.dec_w[j][k] = v,
            UntiedUndo::DecT(j, v) => self.dec_thresh[j] = v,
        }
    }
}

enum UntiedUndo {
    EncW(usize, usize, i8),
    EncT(usize, i32),
    DecW(usize, usize, i8),
    DecT(usize, i32),
}

fn search_untied(
    n_in: usize, n_hidden: usize, unique_bytes: &[u8], seed: u64,
    random_inits: usize, perturb_steps: usize,
) -> UntiedAutoencoder {
    let mut rng = Rng::new(seed);
    let target = unique_bytes.len();

    let eval = |ae: &UntiedAutoencoder| -> usize {
        unique_bytes.iter().filter(|&&b| {
            let bits = byte_to_bits(b);
            let out = ae.round_trip(&bits);
            bits == out.as_slice()
        }).count()
    };

    let mut best = UntiedAutoencoder::new_random(n_in, n_hidden, &mut rng);
    let mut best_score = eval(&best);
    for _ in 0..random_inits {
        let cand = UntiedAutoencoder::new_random(n_in, n_hidden, &mut rng);
        let score = eval(&cand);
        if score > best_score { best_score = score; best = cand; }
        if best_score == target { break; }
    }
    println!("    untied random: {}/{}", best_score, target);

    if best_score == target { return best; }

    let mut current = best.clone();
    let mut current_score = best_score;
    let report = perturb_steps / 10;
    for step in 0..perturb_steps {
        let undo = current.mutate(&mut rng);
        let new_score = eval(&current);
        if new_score >= current_score {
            current_score = new_score;
            if new_score > best_score {
                best_score = new_score;
                best = current.clone();
                println!("    untied perturb [{:>7}] NEW BEST: {}/{}", step+1, new_score, target);
            }
        } else { current.undo(undo); }
        if best_score == target {
            println!("    untied perturb [{:>7}] *** PERFECT! ***", step+1);
            break;
        }
        if report > 0 && (step + 1) % report == 0 {
            println!("    untied perturb [{:>7}]: best={}/{}", step+1, best_score, target);
        }
    }
    best
}

// ══════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════
fn main() {
    let t0 = Instant::now();

    // Load corpus and find unique bytes
    let corpus_path = "instnct-core/tests/fixtures/alice_corpus.txt";
    let unique = load_unique_bytes(corpus_path);
    println!("=== MIRRORED AUTOENCODER — Tied-Weight Byte Encoder ===");
    println!("Corpus: {} ({} unique bytes out of 256)", corpus_path, unique.len());
    println!("Architecture: Input(N) → Encoder(M, W, thresh) → Decoder(N, Wᵀ, thresh)");
    println!("Constraint: Decoder weights = Encoder weights TRANSPOSED (tied)");
    println!("Fitness: round-trip accuracy (byte → encode → decode → same byte?)");
    println!();

    // Show byte distribution summary
    let printable = unique.iter().filter(|&&b| b >= 32 && b < 127).count();
    let control = unique.iter().filter(|&&b| b < 32).count();
    let extended = unique.iter().filter(|&&b| b >= 128).count();
    println!("Byte classes: {} printable, {} control, {} extended (>127)", printable, control, extended);
    println!();

    // ─── Level 1: 8 → 7 ───
    println!("━━━ Level 1: 8 → 7 (128 codes for {} bytes) ━━━", unique.len());
    let ae1 = search_mirror(8, 7, &unique, 42, 500_000, 1_000_000);
    let (c1, t1, failed1) = ae1.eval_accuracy(&unique);
    println!("\n  RESULT: {}/{} round-trip ({:.1}%)", c1, t1, c1 as f64 / t1 as f64 * 100.0);
    if !failed1.is_empty() {
        let fs: Vec<String> = failed1.iter().take(10).map(|&b| {
            if b >= 32 && b < 127 { format!("'{}'", b as char) } else { format!("x{:02x}", b) }
        }).collect();
        println!("  Failed: [{}]", fs.join(", "));
    }
    display_weights(&ae1);
    display_encoding(&ae1, &unique);
    display_code_collisions(&ae1, &unique);
    println!("  Time: {:.1}s\n", t0.elapsed().as_secs_f64());

    // Build codes for cascade
    let codes_7: Vec<Vec<u8>> = unique.iter()
        .map(|&b| ae1.encode(&byte_to_bits(b)))
        .collect();

    // ─── Level 2: 7 → 6 ───
    println!("━━━ Level 2: 7 → 6 (64 codes for {} inputs) ━━━", unique.len());
    let (ae2, codes_6, c2) = cascade_level(&codes_7, &unique, 7, 6, 1042);
    println!("  RESULT: {}/{} round-trip ({:.1}%)", c2, unique.len(),
        c2 as f64 / unique.len() as f64 * 100.0);
    println!("  Unique codes at 6-bit: {}", ae2.unique_codes(
        &(0..codes_7.len()).map(|i| i as u8).collect::<Vec<_>>()  // dummy
    ));
    // Actually count unique codes properly
    let mut c6_uniq: Vec<Vec<u8>> = codes_6.clone();
    c6_uniq.sort(); c6_uniq.dedup();
    println!("  Unique 6-bit codes: {}/{}", c6_uniq.len(), unique.len());
    println!("  Time: {:.1}s\n", t0.elapsed().as_secs_f64());

    // ─── Level 3: 6 → 5 ───
    println!("━━━ Level 3: 6 → 5 (32 codes for {} inputs) ━━━", unique.len());
    let (_, codes_5, c3) = cascade_level(&codes_6, &unique, 6, 5, 2042);
    println!("  RESULT: {}/{} round-trip ({:.1}%)", c3, unique.len(),
        c3 as f64 / unique.len() as f64 * 100.0);
    let mut c5_uniq: Vec<Vec<u8>> = codes_5.clone();
    c5_uniq.sort(); c5_uniq.dedup();
    println!("  Unique 5-bit codes: {}/{}", c5_uniq.len(), unique.len());
    println!("  Time: {:.1}s\n", t0.elapsed().as_secs_f64());

    // ─── Level 4: 5 → 4 ───
    println!("━━━ Level 4: 5 → 4 (16 codes for {} inputs) ━━━", unique.len());
    let (_, codes_4, c4) = cascade_level(&codes_5, &unique, 5, 4, 3042);
    println!("  RESULT: {}/{} round-trip ({:.1}%)", c4, unique.len(),
        c4 as f64 / unique.len() as f64 * 100.0);
    let mut c4_uniq: Vec<Vec<u8>> = codes_4.clone();
    c4_uniq.sort(); c4_uniq.dedup();
    println!("  Unique 4-bit codes: {}/{}", c4_uniq.len(), unique.len());
    println!("  Time: {:.1}s\n", t0.elapsed().as_secs_f64());

    // ─── Level 5: 4 → 3 ───
    println!("━━━ Level 5: 4 → 3 (8 codes for {} inputs) ━━━", unique.len());
    let (_, codes_3, c5) = cascade_level(&codes_4, &unique, 4, 3, 4042);
    println!("  RESULT: {}/{} round-trip ({:.1}%)", c5, unique.len(),
        c5 as f64 / unique.len() as f64 * 100.0);
    let mut c3_uniq: Vec<Vec<u8>> = codes_3.clone();
    c3_uniq.sort(); c3_uniq.dedup();
    println!("  Unique 3-bit codes: {}/{}", c3_uniq.len(), unique.len());
    println!("  Time: {:.1}s\n", t0.elapsed().as_secs_f64());

    // ─── Summary ───
    println!("━━━ CASCADE SUMMARY ━━━");
    println!("  {:>10} {:>10} {:>12} {:>12} {:>12}", "Level", "Bits", "Round-trip", "Accuracy", "Unique codes");
    println!("  {:>10} {:>10} {:>12} {:>12} {:>12}", "raw", "8", format!("{}/{}", t1, t1), "100.0%", format!("{}", unique.len()));
    println!("  {:>10} {:>10} {:>12} {:>12} {:>12}", "8→7", "7", format!("{}/{}", c1, t1), format!("{:.1}%", c1 as f64/t1 as f64*100.0), format!("{}", ae1.unique_codes(&unique)));

    let c6_count = { let mut c = codes_6.clone(); c.sort(); c.dedup(); c.len() };
    let c5_count = { let mut c = codes_5.clone(); c.sort(); c.dedup(); c.len() };
    let c4_count = { let mut c = codes_4.clone(); c.sort(); c.dedup(); c.len() };
    let c3_count = { let mut c = codes_3.clone(); c.sort(); c.dedup(); c.len() };

    println!("  {:>10} {:>10} {:>12} {:>12} {:>12}", "7→6", "6", format!("{}/{}", c2, unique.len()), format!("{:.1}%", c2 as f64/unique.len() as f64*100.0), format!("{}", c6_count));
    println!("  {:>10} {:>10} {:>12} {:>12} {:>12}", "6→5", "5", format!("{}/{}", c3, unique.len()), format!("{:.1}%", c3 as f64/unique.len() as f64*100.0), format!("{}", c5_count));
    println!("  {:>10} {:>10} {:>12} {:>12} {:>12}", "5→4", "4", format!("{}/{}", c4, unique.len()), format!("{:.1}%", c4 as f64/unique.len() as f64*100.0), format!("{}", c4_count));
    println!("  {:>10} {:>10} {:>12} {:>12} {:>12}", "4→3", "3", format!("{}/{}", c5, unique.len()), format!("{:.1}%", c5 as f64/unique.len() as f64*100.0), format!("{}", c3_count));

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());

    // ─── Key question ───
    println!("\n━━━ KEY FINDING ━━━");
    if c1 == t1 {
        println!("  8→7 LOSSLESS: Mirrored threshold autoencoder achieves PERFECT round-trip!");
        println!("  The encoder (7 neurons, tied weights) faithfully encodes all {} bytes.", unique.len());
        println!("  Decoder validated and can be DISCARDED — encoder alone is the preprocessor.");
    } else {
        println!("  8→7: {}/{} — {} bytes lost in round-trip.", c1, t1, t1 - c1);
        println!("  Threshold non-linearity may limit tied-weight symmetry.");
    }
    println!("  First lossy level: {} (where round-trip < 100%)",
        if c1 < t1 { "8→7" } else if c2 < unique.len() { "7→6" }
        else if c3 < unique.len() { "6→5" } else if c4 < unique.len() { "5→4" }
        else { "4→3 or below" });

    // ─── ABLATION: untied weights (independent decoder) ───
    println!("\n━━━ ABLATION: Untied weights (independent decoder) ━━━");
    println!("  Same bottleneck (8→7), but decoder has its OWN weights.");
    println!("  If untied=100% but tied=72% → the tied constraint is the bottleneck.\n");

    let ae_untied = search_untied(8, 7, &unique, 9999, 500_000, 1_000_000);
    let (cu, tu, failed_u) = ae_untied.eval_accuracy_untied(&unique);
    println!("\n  UNTIED RESULT: {}/{} round-trip ({:.1}%)", cu, tu, cu as f64 / tu as f64 * 100.0);
    if !failed_u.is_empty() {
        let fs: Vec<String> = failed_u.iter().take(10).map(|&b| {
            if b >= 32 && b < 127 { format!("'{}'", b as char) } else { format!("x{:02x}", b) }
        }).collect();
        println!("  Failed: [{}]", fs.join(", "));
    }

    println!("\n━━━ TIED vs UNTIED ━━━");
    println!("  Tied weights (Wᵀ):      {}/{} ({:.1}%)", c1, t1, c1 as f64/t1 as f64*100.0);
    println!("  Untied weights (free W): {}/{} ({:.1}%)", cu, tu, cu as f64/tu as f64*100.0);
    if cu > c1 {
        println!("  → Tied constraint costs {} bytes. Threshold breaks symmetry.", cu - c1);
        println!("  → For production: use UNTIED (free decoder for validation only).");
    } else {
        println!("  → Tied constraint is NOT the bottleneck!");
    }
    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
