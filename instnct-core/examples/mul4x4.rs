//! 4×4 Bit Multiplier — built from verified C19 LutGate neurons
//!
//! Architecture: array multiplier
//!   - 16 AND gates  → 16 partial products pp[i][j] = A[j] & B[i]
//!   - 3 half adders (XOR + AND) for bit positions that only combine 2 terms
//!   - 8 full adders  (XOR3 + MAJ) for all remaining carry-propagation stages
//!
//! Exhaustive test: all 256 pairs (a,b) in 0..=15
//!
//! Run: cargo run --example mul4x4 --release

// ============================================================
// C19 activation — used only at gate construction (LUT baking)
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
// Gate library — verified parameters
// ============================================================

struct Gates {
    and_g: LutGate,  // 2-input AND
    xor_g: LutGate,  // 2-input XOR  (half-adder sum)
    xor3:  LutGate,  // 3-input XOR  (full-adder sum)
    maj:   LutGate,  // 3-input MAJ  (full-adder carry)
}

impl Gates {
    fn new() -> Self {
        Gates {
            and_g: LutGate::new(&[10.0, 10.0],             -4.5, 0.0,  4.0),
            xor_g: LutGate::new(&[0.5,  0.5],               0.0, 16.0, 0.6),
            xor3:  LutGate::new(&[1.5,  1.5,  1.5],         3.0, 16.0, 0.6),
            maj:   LutGate::new(&[8.5,  8.5,  8.5],        -2.75, 0.0, 4.0),
        }
    }

    /// Half adder: returns (sum, carry)
    fn half_add(&self, a: u8, b: u8) -> (u8, u8) {
        (self.xor_g.eval(&[a, b]), self.and_g.eval(&[a, b]))
    }

    /// Full adder: returns (sum, carry)
    fn full_add(&self, a: u8, b: u8, cin: u8) -> (u8, u8) {
        (self.xor3.eval(&[a, b, cin]), self.maj.eval(&[a, b, cin]))
    }
}

// ============================================================
// 4×4 Array Multiplier
//
// Partial products layout (rows are indexed by B bit, cols by A bit):
//
//          A3  A2  A1  A0
//       ×  B3  B2  B1  B0
//       --------------------
// Row 0:  p03 p02 p01 p00       (× 2^0)
// Row 1:  p13 p12 p11 p10   0   (× 2^1)
// Row 2:  p23 p22 p21 p20  0 0  (× 2^2)
// Row 3:  p33 p32 p31 p30 0 0 0 (× 2^3)
//
// Column sums (before carry propagation), bit position index:
//  P0 = p00
//  P1 = p01 + p10                            (2 terms → half adder)
//  P2 = p02 + p11 + p20                      (3 terms → full adder)
//  P3 = p03 + p12 + p21 + p30               (4 terms → FA chain)
//  P4 =       p13 + p22 + p31               (3 terms + carries)
//  P5 =             p23 + p32               (2 terms + carries)
//  P6 =                   p33               (1 term  + carries)
//  P7 =                                      (carries only)
//
// Implementation: classic ripple array, left-to-right carry propagation.
// ============================================================

struct Mul4x4 {
    gates: Gates,
    // Neuron accounting
    and_count:  usize,
    ha_count:   usize,
    fa_count:   usize,
}

impl Mul4x4 {
    fn new() -> Self {
        Mul4x4 {
            gates:     Gates::new(),
            and_count: 16,  // 4×4 partial products
            ha_count:  3,   // bits 1, 5(top), 6(top)
            fa_count:  8,   // bits 2, 3(×2), 4(×2), 5, 6, 7(carry only treated as FA with 0)
        }
    }

    /// Multiply two 4-bit values, return 8-bit result.
    /// Also returns the number of gate evaluations performed.
    fn mul(&self, a: u8, b: u8) -> (u8, usize) {
        let g = &self.gates;
        let mut evals: usize = 0;

        // ── Step 1: 16 partial products ──────────────────────
        // pp[row][col] = B[row] & A[col]
        let mut pp = [[0u8; 4]; 4];
        for row in 0..4 {
            for col in 0..4 {
                pp[row][col] = g.and_g.eval(&[(a >> col) & 1, (b >> row) & 1]);
                evals += 1;
            }
        }

        // ── Step 2: Add partial product columns ──────────────
        //
        // We track a running sum array s[0..8] and carry array c[0..8].
        // At each bit position we reduce however many terms land there.
        //
        // Bit 0: just pp[0][0]
        let p0 = pp[0][0];

        // Bit 1: pp[0][1] + pp[1][0]  → half adder
        let (s1, c1_to_2) = g.half_add(pp[0][1], pp[1][0]);
        evals += 2; // XOR + AND

        // Bit 2: pp[0][2] + pp[1][1] + pp[2][0] + carry-in from bit1
        //   First FA: pp[0][2] + pp[1][1] + pp[2][0] → sum2a, carry2a
        let (s2a, c2a) = g.full_add(pp[0][2], pp[1][1], pp[2][0]);
        evals += 2; // XOR3 + MAJ
        //   Then add carry from bit1 via half adder
        let (s2, c2_to_3) = g.half_add(s2a, c1_to_2);
        evals += 2;
        let c2a_to_3 = c2a; // carry out of FA propagates to bit 3

        // Bit 3: pp[0][3] + pp[1][2] + pp[2][1] + pp[3][0] + carries from bit2
        //   FA1: pp[0][3] + pp[1][2] + pp[2][1]
        let (s3a, c3a) = g.full_add(pp[0][3], pp[1][2], pp[2][1]);
        evals += 2;
        //   FA2: s3a + pp[3][0] + c2_to_3
        let (s3b, c3b) = g.full_add(s3a, pp[3][0], c2_to_3);
        evals += 2;
        //   HA: s3b + c2a_to_3
        let (s3, c3_to_4) = g.half_add(s3b, c2a_to_3);
        evals += 2;
        let c3a_to_4 = c3a;
        let c3b_to_4 = c3b;

        // Bit 4: pp[1][3] + pp[2][2] + pp[3][1] + carries from bit3
        //   FA1: pp[1][3] + pp[2][2] + pp[3][1]
        let (s4a, c4a) = g.full_add(pp[1][3], pp[2][2], pp[3][1]);
        evals += 2;
        //   FA2: s4a + c3_to_4 + c3a_to_4
        let (s4b, c4b) = g.full_add(s4a, c3_to_4, c3a_to_4);
        evals += 2;
        //   HA: s4b + c3b_to_4
        let (s4, c4_to_5) = g.half_add(s4b, c3b_to_4);
        evals += 2;
        let c4a_to_5 = c4a;
        let c4b_to_5 = c4b;

        // Bit 5: pp[2][3] + pp[3][2] + carries from bit4
        //   FA: pp[2][3] + pp[3][2] + c4_to_5
        let (s5a, c5a) = g.full_add(pp[2][3], pp[3][2], c4_to_5);
        evals += 2;
        //   FA: s5a + c4a_to_5 + c4b_to_5
        let (s5, c5_to_6) = g.full_add(s5a, c4a_to_5, c4b_to_5);
        evals += 2;
        let c5a_to_6 = c5a;

        // Bit 6: pp[3][3] + carries from bit5
        //   FA: pp[3][3] + c5_to_6 + c5a_to_6
        let (s6, c6_to_7) = g.full_add(pp[3][3], c5_to_6, c5a_to_6);
        evals += 2;

        // Bit 7: final carry
        let p7 = c6_to_7;

        let result = p0
            | (s1 << 1)
            | (s2 << 2)
            | (s3 << 3)
            | (s4 << 4)
            | (s5 << 5)
            | (s6 << 6)
            | (p7 << 7);

        (result, evals)
    }

    fn neuron_count(&self) -> usize {
        // AND gates for partial products
        let and_neurons = self.and_count;
        // Each half adder = 1 XOR + 1 AND = 2 gates
        let ha_neurons = self.ha_count * 2;
        // Each full adder = 1 XOR3 + 1 MAJ = 2 gates
        let fa_neurons = self.fa_count * 2;
        and_neurons + ha_neurons + fa_neurons
    }
}

// ============================================================
// main — exhaustive verification
// ============================================================

fn main() {
    let mul = Mul4x4::new();

    println!("=== 4×4 C19 LutGate Multiplier ===");
    println!();
    println!("Gate inventory:");
    println!("  AND gates (partial products) : {}", mul.and_count);
    println!("  Half adders (XOR + AND)      : {}  ({} neurons)", mul.ha_count, mul.ha_count * 2);
    println!("  Full adders (XOR3 + MAJ)     : {}  ({} neurons)", mul.fa_count, mul.fa_count * 2);
    println!("  Total neurons                : {}", mul.neuron_count());
    println!();

    // Dry-run one call to get evals count
    let (_, sample_evals) = mul.mul(7, 11);
    println!("Gate evaluations per multiply  : {}", sample_evals);
    println!();

    // Exhaustive test
    println!("Exhaustive verification (all 256 pairs):");
    println!("{:-<60}", "");

    let mut passes = 0usize;
    let mut failures = 0usize;
    let mut fail_details: Vec<(u8, u8, u8, u8)> = Vec::new(); // (a, b, got, expected)

    for a in 0u8..=15 {
        for b in 0u8..=15 {
            let expected = a * b;
            let (got, _) = mul.mul(a, b);
            if got == expected {
                passes += 1;
            } else {
                failures += 1;
                fail_details.push((a, b, got, expected));
            }
        }
    }

    // Print result grid (16×16)
    println!("Result grid (a=rows 0-15, b=cols 0-15) — showing 8-bit product:");
    print!("    ");
    for b in 0u8..=15 { print!("{:4}", b); }
    println!();
    print!("    ");
    for _ in 0u8..=15 { print!("----"); }
    println!();
    for a in 0u8..=15 {
        print!("{:3}|", a);
        for b in 0u8..=15 {
            let (got, _) = mul.mul(a, b);
            let expected = a * b;
            if got == expected {
                print!("{:4}", got);
            } else {
                print!(" !! ");
            }
        }
        println!();
    }
    println!();

    // Summary
    println!("{:-<60}", "");
    println!("RESULTS: {}/256 PASS", passes);
    if failures == 0 {
        println!("ALL TESTS PASSED — 100% correct");
    } else {
        println!("FAILURES: {}", failures);
        println!();
        println!("Failure details:");
        for (a, b, got, exp) in &fail_details {
            println!("  {:2} × {:2}  got={:3}  expected={:3}  (binary got={:08b} exp={:08b})",
                a, b, got, exp, got, exp);
        }
    }
}
