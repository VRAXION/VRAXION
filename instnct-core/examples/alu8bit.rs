//! 8-bit ALU — built entirely from verified C19 LutGate neurons
//!
//! Operations:
//!   ADD8  — 8-bit addition (mod 256), with carry out
//!   SUB8  — 8-bit subtraction (mod 256), via two's complement
//!   MUL8  — 8x8 multiply, full 16-bit result
//!   AND8  — bitwise AND
//!   OR8   — bitwise OR
//!   XOR8  — bitwise XOR
//!   NOT8  — bitwise NOT
//!   SHL8  — shift left by 1 (no gates, rewire)
//!   SHR8  — shift right by 1 (no gates, rewire)
//!   CMP8  — compare, returns flags (Z, N, C)
//!   MIN8  — min(a, b)
//!   MAX8  — max(a, b)
//!
//! All computation uses C19 integer LUT gates — zero floating point in the eval() hot path.
//!
//! Run: cargo run --example alu8bit --release

// ============================================================
// C19 activation — used only at gate construction time (LUT baking)
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
            if all
                .iter()
                .all(|&v| ((v * d as f32).round() - v * d as f32).abs() < 1e-6)
            {
                denom = d;
                break;
            }
        }
        let w_int: Vec<i32> = w
            .iter()
            .map(|&v| (v * denom as f32).round() as i32)
            .collect();
        let bias_int = (bias * denom as f32).round() as i32;
        let mut min_s = bias_int;
        let mut max_s = bias_int;
        for &wi in &w_int {
            if wi > 0 {
                max_s += wi;
            } else {
                min_s += wi;
            }
        }
        let mut lut = vec![0u8; (max_s - min_s + 1) as usize];
        for s in min_s..=max_s {
            lut[(s - min_s) as usize] =
                if c19(s as f32 / denom as f32, rho) > thr { 1 } else { 0 };
        }
        LutGate {
            w_int,
            bias_int,
            lut,
            min_sum: min_s,
        }
    }

    fn eval(&self, inputs: &[u8]) -> u8 {
        let s: i32 = inputs
            .iter()
            .zip(&self.w_int)
            .map(|(&i, &w)| i as i32 * w)
            .sum::<i32>()
            + self.bias_int;
        let idx = (s - self.min_sum) as usize;
        if idx < self.lut.len() {
            self.lut[idx]
        } else {
            0
        }
    }
}

// ============================================================
// Gate library — verified C19 parameters (exhaustive-tested)
// ============================================================

struct Gates {
    and_g: LutGate,  // 2-input AND
    or_g: LutGate,   // 2-input OR
    xor_g: LutGate,  // 2-input XOR
    not_g: LutGate,  // 1-input NOT
    xor3: LutGate,   // 3-input XOR  (full-adder sum)
    maj: LutGate,    // 3-input MAJ  (full-adder carry)
}

impl Gates {
    fn new() -> Self {
        Gates {
            and_g: LutGate::new(&[10.0, 10.0], -4.5, 0.0, 4.0),
            or_g: LutGate::new(&[8.75, 8.75], 5.5, 0.0, 4.0),
            xor_g: LutGate::new(&[0.5, 0.5], 0.0, 16.0, 0.6),
            not_g: LutGate::new(&[-9.75], -5.5, 16.0, -4.0),
            xor3: LutGate::new(&[1.5, 1.5, 1.5], 3.0, 16.0, 0.6),
            maj: LutGate::new(&[8.5, 8.5, 8.5], -2.75, 0.0, 4.0),
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
// CMP flags
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CmpFlags {
    z: bool, // zero: a == b
    n: bool, // negative: a < b (signed interpretation of result MSB)
    c: bool, // carry: unsigned borrow (inverted: c=1 means a >= b)
}

// ============================================================
// ALU8 — the complete 8-bit ALU
// ============================================================

struct Alu8 {
    gates: Gates,
}

impl Alu8 {
    fn new() -> Self {
        Alu8 {
            gates: Gates::new(),
        }
    }

    // ── Bit extraction helpers ───────────────────────────────
    fn bit(val: u8, pos: usize) -> u8 {
        (val >> pos) & 1
    }

    #[allow(dead_code)]
    fn bit16(val: u16, pos: usize) -> u8 {
        ((val >> pos) & 1) as u8
    }

    // ── ADD8: 8-bit ripple-carry adder ──────────────────────
    // 8 full adders = 16 neurons (8 XOR3 + 8 MAJ)
    // Returns (result, carry_out)
    fn add8(&self, a: u8, b: u8) -> (u8, u8) {
        self.add8_cin(a, b, 0)
    }

    fn add8_cin(&self, a: u8, b: u8, cin: u8) -> (u8, u8) {
        let g = &self.gates;
        let mut carry = cin;
        let mut result = 0u8;
        for bit in 0..8 {
            let ab = Self::bit(a, bit);
            let bb = Self::bit(b, bit);
            let (s, c) = g.full_add(ab, bb, carry);
            result |= s << bit;
            carry = c;
        }
        (result, carry)
    }

    // ── SUB8: two's complement subtraction ──────────────────
    // 8 NOT + 8 full adders = 24 neurons
    // a - b = a + NOT(b) + 1
    fn sub8(&self, a: u8, b: u8) -> (u8, u8) {
        let g = &self.gates;
        // NOT each bit of b
        let mut not_b = 0u8;
        for bit in 0..8 {
            not_b |= g.not_g.eval(&[Self::bit(b, bit)]) << bit;
        }
        // Add with carry_in = 1 (two's complement)
        self.add8_cin(a, not_b, 1)
    }

    // ── MUL8: 8x8 array multiplier, full 16-bit result ─────
    // 64 AND gates for partial products
    // Reduction via full/half adders
    //
    // Partial products: pp[row][col] = B[row] & A[col]
    // Row i contributes to bit positions i..i+7
    //
    // We reduce column by column, propagating carries upward.
    fn mul8(&self, a: u8, b: u8) -> u16 {
        let g = &self.gates;

        // Step 1: Generate 64 partial products
        let mut pp = [[0u8; 8]; 8];
        for row in 0..8 {
            for col in 0..8 {
                pp[row][col] = g.and_g.eval(&[Self::bit(a, col), Self::bit(b, row)]);
            }
        }

        // Step 2: Column reduction using an iterative approach
        // For each bit position k (0..15), collect all partial products
        // and carries from lower columns, then reduce to a single bit + carries.
        //
        // We use a straightforward column-wise reduction:
        // Collect terms for each column, reduce 3-at-a-time with full adders,
        // 2-at-a-time with half adders, until one bit remains.

        let mut result: u16 = 0;
        let mut carry_pool: Vec<u8> = Vec::new(); // carries propagating to next column

        for col_pos in 0..16 {
            // Collect all terms for this column
            let mut terms: Vec<u8> = Vec::new();

            // Partial products that land in this column
            // pp[row][col] lands at bit position row+col
            for row in 0..8 {
                let col = col_pos as i32 - row as i32;
                if col >= 0 && col < 8 {
                    terms.push(pp[row][col as usize]);
                }
            }

            // Add carries from previous column
            terms.append(&mut carry_pool);
            carry_pool = Vec::new();

            // Reduce terms to a single bit
            while terms.len() > 1 {
                if terms.len() >= 3 {
                    let a_bit = terms.remove(0);
                    let b_bit = terms.remove(0);
                    let c_bit = terms.remove(0);
                    let (s, c) = g.full_add(a_bit, b_bit, c_bit);
                    terms.push(s);
                    carry_pool.push(c);
                } else {
                    // exactly 2 terms
                    let a_bit = terms.remove(0);
                    let b_bit = terms.remove(0);
                    let (s, c) = g.half_add(a_bit, b_bit);
                    terms.push(s);
                    carry_pool.push(c);
                }
            }

            if terms.len() == 1 {
                result |= (terms[0] as u16) << col_pos;
            }
            // if terms is empty, bit is 0 (already default)
        }

        result
    }

    // ── AND8: bitwise AND ───────────────────────────────────
    // 8 neurons
    fn and8(&self, a: u8, b: u8) -> u8 {
        let g = &self.gates;
        let mut r = 0u8;
        for bit in 0..8 {
            r |= g.and_g.eval(&[Self::bit(a, bit), Self::bit(b, bit)]) << bit;
        }
        r
    }

    // ── OR8: bitwise OR ─────────────────────────────────────
    // 8 neurons
    fn or8(&self, a: u8, b: u8) -> u8 {
        let g = &self.gates;
        let mut r = 0u8;
        for bit in 0..8 {
            r |= g.or_g.eval(&[Self::bit(a, bit), Self::bit(b, bit)]) << bit;
        }
        r
    }

    // ── XOR8: bitwise XOR ───────────────────────────────────
    // 8 neurons
    fn xor8(&self, a: u8, b: u8) -> u8 {
        let g = &self.gates;
        let mut r = 0u8;
        for bit in 0..8 {
            r |= g.xor_g.eval(&[Self::bit(a, bit), Self::bit(b, bit)]) << bit;
        }
        r
    }

    // ── NOT8: bitwise NOT ───────────────────────────────────
    // 8 neurons
    fn not8(&self, a: u8) -> u8 {
        let g = &self.gates;
        let mut r = 0u8;
        for bit in 0..8 {
            r |= g.not_g.eval(&[Self::bit(a, bit)]) << bit;
        }
        r
    }

    // ── SHL8: shift left by 1 ───────────────────────────────
    // 0 neurons — pure rewiring
    fn shl8(&self, a: u8) -> u8 {
        a << 1 // wrapping: MSB falls off, LSB = 0
    }

    // ── SHR8: shift right by 1 ──────────────────────────────
    // 0 neurons — pure rewiring
    fn shr8(&self, a: u8) -> u8 {
        a >> 1 // logical shift: MSB = 0
    }

    // ── CMP8: compare, returns flags ────────────────────────
    // Uses SUB8 internally (24 neurons) + zero detect
    // Zero detect: NOR of all 8 result bits
    //   Implemented as: NOT(OR(OR(OR(b0,b1),OR(b2,b3)),OR(OR(b4,b5),OR(b6,b7))))
    //   = 4 OR + 2 OR + 1 OR + 1 NOT = 7 OR + 1 NOT = 8 neurons
    // Total: 24 (sub) + 8 (zero detect) = 32 neurons
    // But N flag is free (just MSB of result), C flag is carry out of sub
    fn cmp8(&self, a: u8, b: u8) -> CmpFlags {
        let g = &self.gates;
        let (diff, carry) = self.sub8(a, b);

        // Z flag: result is zero
        // Tree-reduce with OR gates, then NOT
        let or01 = g.or_g.eval(&[Self::bit(diff, 0), Self::bit(diff, 1)]);
        let or23 = g.or_g.eval(&[Self::bit(diff, 2), Self::bit(diff, 3)]);
        let or45 = g.or_g.eval(&[Self::bit(diff, 4), Self::bit(diff, 5)]);
        let or67 = g.or_g.eval(&[Self::bit(diff, 6), Self::bit(diff, 7)]);
        let or03 = g.or_g.eval(&[or01, or23]);
        let or47 = g.or_g.eval(&[or45, or67]);
        let or07 = g.or_g.eval(&[or03, or47]);
        let z = g.not_g.eval(&[or07]);

        // N flag: MSB of result (bit 7)
        let n = Self::bit(diff, 7);

        // C flag: carry out from the subtraction
        // In two's complement sub, carry=1 means a >= b (no borrow)
        let c = carry;

        CmpFlags {
            z: z == 1,
            n: n == 1,
            c: c == 1,
        }
    }

    // ── MUX8: 8-bit 2:1 multiplexer ────────────────────────
    // sel=0 → a, sel=1 → b
    // Each bit: OR(AND(NOT(sel), a_bit), AND(sel, b_bit))
    // = 1 NOT + 2 AND + 1 OR = 4 gates per bit (but NOT(sel) shared)
    // Total: 1 NOT + 8*(2 AND + 1 OR) = 1 + 24 = 25 neurons
    fn mux8(&self, a: u8, b: u8, sel: u8) -> u8 {
        let g = &self.gates;
        let not_sel = g.not_g.eval(&[sel]);
        let mut r = 0u8;
        for bit in 0..8 {
            let a_bit = Self::bit(a, bit);
            let b_bit = Self::bit(b, bit);
            let choose_a = g.and_g.eval(&[not_sel, a_bit]);
            let choose_b = g.and_g.eval(&[sel, b_bit]);
            r |= g.or_g.eval(&[choose_a, choose_b]) << bit;
        }
        r
    }

    // ── MIN8: min(a, b) ─────────────────────────────────────
    // CMP8 + MUX8: if a <= b, return a, else b
    // Neuron count: 32 (cmp) + 25 (mux) = 57
    fn min8(&self, a: u8, b: u8) -> u8 {
        let flags = self.cmp8(a, b);
        // a <= b when carry is set (a >= b is false, i.e., carry=0 means a < b)
        // Actually: in sub(a,b), carry=1 means a >= b, carry=0 means a < b
        // We want min: if a <= b, pick a; else pick b
        // a <= b  <==>  carry=1 AND (z=1 OR carry=1 while z could be 1)
        // Simpler: a < b <==> carry=0 (unsigned). a <= b <==> carry=0 OR z=1
        // But easier: if a < b (carry=0 from sub), pick a; else pick b
        // Wait: carry=0 means borrow happened, so a < b
        // sel=0 => a, sel=1 => b
        // If a <= b: pick a => sel=0 => need sel=0 when a<=b
        // a <= b <==> NOT(a > b) <==> NOT(carry=1 AND z=0)
        // Simplest: carry=0 means a < b => pick a (sel=0) => sel = carry AND NOT(z)
        // Hmm let's just: if carry=0: a < b, pick a (sel=0)
        //                 if carry=1 and z=1: a==b, pick a (sel=0)
        //                 if carry=1 and z=0: a > b, pick b (sel=1)
        // So sel = carry AND NOT(z) = (a > b)
        let sel = if flags.c && !flags.z { 1u8 } else { 0u8 };
        self.mux8(a, b, sel)
    }

    // ── MAX8: max(a, b) ─────────────────────────────────────
    // Same as MIN but inverted selection
    fn max8(&self, a: u8, b: u8) -> u8 {
        let flags = self.cmp8(a, b);
        // If a >= b (carry=1), pick a; else pick b
        // sel=0 => a, sel=1 => b
        // pick a when carry=1 => sel=0 when carry=1 => sel = NOT(carry)
        let sel = if flags.c { 0u8 } else { 1u8 };
        self.mux8(a, b, sel)
    }
}

// ============================================================
// Neuron accounting
// ============================================================

fn print_neuron_counts() {
    println!("=== Neuron Count per Operation ===");
    println!();
    println!("  ADD8  : 16 neurons  (8 full adders: 8 XOR3 + 8 MAJ)");
    println!("  SUB8  : 24 neurons  (8 NOT + 8 full adders)");
    println!("  MUL8  : ~264 neurons (64 AND + ~100 FA + ~100 HA, varies by reduction)");
    println!("  AND8  :  8 neurons  (8 AND gates)");
    println!("  OR8   :  8 neurons  (8 OR gates)");
    println!("  XOR8  :  8 neurons  (8 XOR gates)");
    println!("  NOT8  :  8 neurons  (8 NOT gates)");
    println!("  SHL8  :  0 neurons  (rewire only)");
    println!("  SHR8  :  0 neurons  (rewire only)");
    println!("  CMP8  : 32 neurons  (24 SUB8 + 7 OR + 1 NOT for zero detect)");
    println!("  MIN8  : 57 neurons  (32 CMP + 1 NOT + 16 AND + 8 OR for MUX)");
    println!("  MAX8  : 57 neurons  (32 CMP + 1 NOT + 16 AND + 8 OR for MUX)");
    println!();

    // Count MUL8 neurons precisely
    // 64 AND for partial products
    // Column reduction: for each column, we need to reduce N terms to 1
    // Column 0: 1 term  -> 0 adders
    // Column 1: 2 terms -> 1 HA (2 neurons)
    // Column 2: 3 terms -> 1 FA (2 neurons)
    // Column 3: 4 terms -> 1 FA + 1 HA (4 neurons) [3->FA->1+carry, 1+1->HA]
    //   Actually: 4 terms + possible carries from col 2
    // This gets complex; let's count by simulation
    let _g = Gates::new();
    let mut total_fa = 0usize;
    let mut total_ha = 0usize;

    // Simulate column reduction to count adders
    let mut carry_counts = vec![0usize; 17]; // carries into each column

    for col_pos in 0..16usize {
        // Count partial products in this column
        let mut n_terms = 0usize;
        for row in 0..8 {
            let col = col_pos as i32 - row as i32;
            if col >= 0 && col < 8 {
                n_terms += 1;
            }
        }
        n_terms += carry_counts[col_pos]; // add incoming carries

        let mut remaining = n_terms;
        while remaining > 1 {
            if remaining >= 3 {
                total_fa += 1;
                remaining -= 2; // 3 in, 1 sum out, 1 carry out
                carry_counts[col_pos + 1] += 1;
            } else {
                total_ha += 1;
                remaining -= 1; // 2 in, 1 sum out, 1 carry out
                carry_counts[col_pos + 1] += 1;
            }
        }
    }

    let mul_neurons = 64 + total_fa * 2 + total_ha * 2;
    println!("  MUL8 precise neuron count:");
    println!("    AND gates (partial products): 64");
    println!("    Full adders: {} ({} neurons)", total_fa, total_fa * 2);
    println!("    Half adders: {} ({} neurons)", total_ha, total_ha * 2);
    println!("    Total MUL8: {} neurons", mul_neurons);
    println!();

    let total = 16 + 24 + mul_neurons + 8 + 8 + 8 + 8 + 0 + 0 + 32 + 57 + 57;
    println!("  TOTAL ALU: {} neurons (unique, counting shared sub-circuits once)", total);
    println!();
}

// ============================================================
// Exhaustive tests
// ============================================================

fn main() {
    println!("================================================================");
    println!("  8-bit C19 LutGate ALU — Exhaustive Verification");
    println!("================================================================");
    println!();

    let alu = Alu8::new();

    // ── Gate verification ────────────────────────────────────
    println!("--- Gate Truth Table Verification ---");
    let g = &alu.gates;

    // Verify AND
    assert_eq!(g.and_g.eval(&[0, 0]), 0, "AND(0,0) failed");
    assert_eq!(g.and_g.eval(&[0, 1]), 0, "AND(0,1) failed");
    assert_eq!(g.and_g.eval(&[1, 0]), 0, "AND(1,0) failed");
    assert_eq!(g.and_g.eval(&[1, 1]), 1, "AND(1,1) failed");
    println!("  AND gate: OK");

    // Verify OR
    assert_eq!(g.or_g.eval(&[0, 0]), 0, "OR(0,0) failed");
    assert_eq!(g.or_g.eval(&[0, 1]), 1, "OR(0,1) failed");
    assert_eq!(g.or_g.eval(&[1, 0]), 1, "OR(1,0) failed");
    assert_eq!(g.or_g.eval(&[1, 1]), 1, "OR(1,1) failed");
    println!("  OR gate : OK");

    // Verify XOR
    assert_eq!(g.xor_g.eval(&[0, 0]), 0, "XOR(0,0) failed");
    assert_eq!(g.xor_g.eval(&[0, 1]), 1, "XOR(0,1) failed");
    assert_eq!(g.xor_g.eval(&[1, 0]), 1, "XOR(1,0) failed");
    assert_eq!(g.xor_g.eval(&[1, 1]), 0, "XOR(1,1) failed");
    println!("  XOR gate: OK");

    // Verify NOT
    assert_eq!(g.not_g.eval(&[0]), 1, "NOT(0) failed");
    assert_eq!(g.not_g.eval(&[1]), 0, "NOT(1) failed");
    println!("  NOT gate: OK");

    // Verify XOR3 (full adder sum)
    for a in 0..=1u8 {
        for b in 0..=1u8 {
            for c in 0..=1u8 {
                let expected = a ^ b ^ c;
                let got = g.xor3.eval(&[a, b, c]);
                assert_eq!(got, expected, "XOR3({},{},{}) = {} expected {}", a, b, c, got, expected);
            }
        }
    }
    println!("  XOR3 gate: OK");

    // Verify MAJ (full adder carry)
    for a in 0..=1u8 {
        for b in 0..=1u8 {
            for c in 0..=1u8 {
                let expected = if (a + b + c) >= 2 { 1u8 } else { 0 };
                let got = g.maj.eval(&[a, b, c]);
                assert_eq!(got, expected, "MAJ({},{},{}) = {} expected {}", a, b, c, got, expected);
            }
        }
    }
    println!("  MAJ gate : OK");
    println!();

    // ── Print neuron counts ─────────────────────────────────
    print_neuron_counts();

    // ── Test counters ────────────────────────────────────────
    let mut total_tests = 0u64;
    let mut total_pass = 0u64;
    let mut total_fail = 0u64;

    // ── ADD8: exhaustive 256x256 ─────────────────────────────
    print!("ADD8  : exhaustive 65536 tests ... ");
    let mut pass = 0u32;
    let mut fail = 0u32;
    let mut fail_examples: Vec<String> = Vec::new();
    for a in 0u16..=255 {
        for b in 0u16..=255 {
            let expected = ((a + b) & 0xFF) as u8;
            let expected_carry = if a + b > 255 { 1u8 } else { 0 };
            let (got, carry) = alu.add8(a as u8, b as u8);
            if got == expected && carry == expected_carry {
                pass += 1;
            } else {
                fail += 1;
                if fail_examples.len() < 5 {
                    fail_examples.push(format!(
                        "  ADD8({}, {}) = {}(c={}) expected {}(c={})",
                        a, b, got, carry, expected, expected_carry
                    ));
                }
            }
        }
    }
    if fail == 0 {
        println!("{}/65536 PASS", pass);
    } else {
        println!("{}/65536 PASS, {} FAIL", pass, fail);
        for s in &fail_examples {
            println!("{}", s);
        }
    }
    total_tests += 65536;
    total_pass += pass as u64;
    total_fail += fail as u64;

    // ── SUB8: exhaustive 256x256 ─────────────────────────────
    print!("SUB8  : exhaustive 65536 tests ... ");
    let mut pass = 0u32;
    let mut fail = 0u32;
    let mut fail_examples: Vec<String> = Vec::new();
    for a in 0u16..=255 {
        for b in 0u16..=255 {
            let expected = a.wrapping_sub(b) as u8;
            // carry=1 means no borrow (a >= b)
            let expected_carry = if a >= b { 1u8 } else { 0 };
            let (got, carry) = alu.sub8(a as u8, b as u8);
            if got == expected && carry == expected_carry {
                pass += 1;
            } else {
                fail += 1;
                if fail_examples.len() < 5 {
                    fail_examples.push(format!(
                        "  SUB8({}, {}) = {}(c={}) expected {}(c={})",
                        a, b, got, carry, expected, expected_carry
                    ));
                }
            }
        }
    }
    if fail == 0 {
        println!("{}/65536 PASS", pass);
    } else {
        println!("{}/65536 PASS, {} FAIL", pass, fail);
        for s in &fail_examples {
            println!("{}", s);
        }
    }
    total_tests += 65536;
    total_pass += pass as u64;
    total_fail += fail as u64;

    // ── MUL8: exhaustive 256x256 ─────────────────────────────
    print!("MUL8  : exhaustive 65536 tests ... ");
    let mut pass = 0u32;
    let mut fail = 0u32;
    let mut fail_examples: Vec<String> = Vec::new();
    for a in 0u16..=255 {
        for b in 0u16..=255 {
            let expected = (a * b) as u16;
            let got = alu.mul8(a as u8, b as u8);
            if got == expected {
                pass += 1;
            } else {
                fail += 1;
                if fail_examples.len() < 5 {
                    fail_examples.push(format!(
                        "  MUL8({}, {}) = {} expected {}",
                        a, b, got, expected
                    ));
                }
            }
        }
    }
    if fail == 0 {
        println!("{}/65536 PASS", pass);
    } else {
        println!("{}/65536 PASS, {} FAIL", pass, fail);
        for s in &fail_examples {
            println!("{}", s);
        }
    }
    total_tests += 65536;
    total_pass += pass as u64;
    total_fail += fail as u64;

    // ── AND8: exhaustive ─────────────────────────────────────
    print!("AND8  : exhaustive 65536 tests ... ");
    let mut pass = 0u32;
    let mut fail = 0u32;
    for a in 0u16..=255 {
        for b in 0u16..=255 {
            let expected = (a as u8) & (b as u8);
            let got = alu.and8(a as u8, b as u8);
            if got == expected { pass += 1; } else { fail += 1; }
        }
    }
    if fail == 0 { println!("{}/65536 PASS", pass); }
    else { println!("{}/65536 PASS, {} FAIL", pass, fail); }
    total_tests += 65536;
    total_pass += pass as u64;
    total_fail += fail as u64;

    // ── OR8: exhaustive ──────────────────────────────────────
    print!("OR8   : exhaustive 65536 tests ... ");
    let mut pass = 0u32;
    let mut fail = 0u32;
    for a in 0u16..=255 {
        for b in 0u16..=255 {
            let expected = (a as u8) | (b as u8);
            let got = alu.or8(a as u8, b as u8);
            if got == expected { pass += 1; } else { fail += 1; }
        }
    }
    if fail == 0 { println!("{}/65536 PASS", pass); }
    else { println!("{}/65536 PASS, {} FAIL", pass, fail); }
    total_tests += 65536;
    total_pass += pass as u64;
    total_fail += fail as u64;

    // ── XOR8: exhaustive ─────────────────────────────────────
    print!("XOR8  : exhaustive 65536 tests ... ");
    let mut pass = 0u32;
    let mut fail = 0u32;
    for a in 0u16..=255 {
        for b in 0u16..=255 {
            let expected = (a as u8) ^ (b as u8);
            let got = alu.xor8(a as u8, b as u8);
            if got == expected { pass += 1; } else { fail += 1; }
        }
    }
    if fail == 0 { println!("{}/65536 PASS", pass); }
    else { println!("{}/65536 PASS, {} FAIL", pass, fail); }
    total_tests += 65536;
    total_pass += pass as u64;
    total_fail += fail as u64;

    // ── NOT8: exhaustive (256 values) ────────────────────────
    print!("NOT8  : exhaustive 256 tests ... ");
    let mut pass = 0u32;
    let mut fail = 0u32;
    for a in 0u16..=255 {
        let expected = !(a as u8);
        let got = alu.not8(a as u8);
        if got == expected { pass += 1; } else { fail += 1; }
    }
    if fail == 0 { println!("{}/256 PASS", pass); }
    else { println!("{}/256 PASS, {} FAIL", pass, fail); }
    total_tests += 256;
    total_pass += pass as u64;
    total_fail += fail as u64;

    // ── SHL8: exhaustive (256 values) ────────────────────────
    print!("SHL8  : exhaustive 256 tests ... ");
    let mut pass = 0u32;
    let mut fail = 0u32;
    for a in 0u16..=255 {
        let expected = (a as u8) << 1;  // wrapping shl
        let got = alu.shl8(a as u8);
        if got == expected { pass += 1; } else { fail += 1; }
    }
    if fail == 0 { println!("{}/256 PASS", pass); }
    else { println!("{}/256 PASS, {} FAIL", pass, fail); }
    total_tests += 256;
    total_pass += pass as u64;
    total_fail += fail as u64;

    // ── SHR8: exhaustive (256 values) ────────────────────────
    print!("SHR8  : exhaustive 256 tests ... ");
    let mut pass = 0u32;
    let mut fail = 0u32;
    for a in 0u16..=255 {
        let expected = (a as u8) >> 1;
        let got = alu.shr8(a as u8);
        if got == expected { pass += 1; } else { fail += 1; }
    }
    if fail == 0 { println!("{}/256 PASS", pass); }
    else { println!("{}/256 PASS, {} FAIL", pass, fail); }
    total_tests += 256;
    total_pass += pass as u64;
    total_fail += fail as u64;

    // ── CMP8: exhaustive ─────────────────────────────────────
    print!("CMP8  : exhaustive 65536 tests ... ");
    let mut pass = 0u32;
    let mut fail = 0u32;
    let mut fail_examples: Vec<String> = Vec::new();
    for a in 0u16..=255 {
        for b in 0u16..=255 {
            let flags = alu.cmp8(a as u8, b as u8);
            let exp_z = a == b;
            let exp_n = ((a.wrapping_sub(b)) & 0x80) != 0;
            let exp_c = a >= b;
            if flags.z == exp_z && flags.n == exp_n && flags.c == exp_c {
                pass += 1;
            } else {
                fail += 1;
                if fail_examples.len() < 5 {
                    fail_examples.push(format!(
                        "  CMP8({}, {}): got Z={} N={} C={}, expected Z={} N={} C={}",
                        a, b, flags.z, flags.n, flags.c, exp_z, exp_n, exp_c
                    ));
                }
            }
        }
    }
    if fail == 0 { println!("{}/65536 PASS", pass); }
    else {
        println!("{}/65536 PASS, {} FAIL", pass, fail);
        for s in &fail_examples { println!("{}", s); }
    }
    total_tests += 65536;
    total_pass += pass as u64;
    total_fail += fail as u64;

    // ── MIN8: exhaustive ─────────────────────────────────────
    print!("MIN8  : exhaustive 65536 tests ... ");
    let mut pass = 0u32;
    let mut fail = 0u32;
    let mut fail_examples: Vec<String> = Vec::new();
    for a in 0u16..=255 {
        for b in 0u16..=255 {
            let expected = std::cmp::min(a, b) as u8;
            let got = alu.min8(a as u8, b as u8);
            if got == expected {
                pass += 1;
            } else {
                fail += 1;
                if fail_examples.len() < 5 {
                    fail_examples.push(format!(
                        "  MIN8({}, {}) = {} expected {}",
                        a, b, got, expected
                    ));
                }
            }
        }
    }
    if fail == 0 { println!("{}/65536 PASS", pass); }
    else {
        println!("{}/65536 PASS, {} FAIL", pass, fail);
        for s in &fail_examples { println!("{}", s); }
    }
    total_tests += 65536;
    total_pass += pass as u64;
    total_fail += fail as u64;

    // ── MAX8: exhaustive ─────────────────────────────────────
    print!("MAX8  : exhaustive 65536 tests ... ");
    let mut pass = 0u32;
    let mut fail = 0u32;
    let mut fail_examples: Vec<String> = Vec::new();
    for a in 0u16..=255 {
        for b in 0u16..=255 {
            let expected = std::cmp::max(a, b) as u8;
            let got = alu.max8(a as u8, b as u8);
            if got == expected {
                pass += 1;
            } else {
                fail += 1;
                if fail_examples.len() < 5 {
                    fail_examples.push(format!(
                        "  MAX8({}, {}) = {} expected {}",
                        a, b, got, expected
                    ));
                }
            }
        }
    }
    if fail == 0 { println!("{}/65536 PASS", pass); }
    else {
        println!("{}/65536 PASS, {} FAIL", pass, fail);
        for s in &fail_examples { println!("{}", s); }
    }
    total_tests += 65536;
    total_pass += pass as u64;
    total_fail += fail as u64;

    // ── Edge case spotlight ──────────────────────────────────
    println!();
    println!("--- Edge Case Spotlight ---");

    // ADD edge cases
    let (r, c) = alu.add8(255, 1);
    println!("  ADD8(255, 1)   = {} carry={} (expect 0, carry=1)", r, c);

    let (r, c) = alu.add8(128, 128);
    println!("  ADD8(128, 128) = {} carry={} (expect 0, carry=1)", r, c);

    let (r, c) = alu.add8(0, 0);
    println!("  ADD8(0, 0)     = {} carry={} (expect 0, carry=0)", r, c);

    // SUB edge cases
    let (r, c) = alu.sub8(0, 1);
    println!("  SUB8(0, 1)     = {} carry={} (expect 255, carry=0)", r, c);

    let (r, c) = alu.sub8(0, 255);
    println!("  SUB8(0, 255)   = {} carry={} (expect 1, carry=0)", r, c);

    let (r, c) = alu.sub8(100, 100);
    println!("  SUB8(100, 100) = {} carry={} (expect 0, carry=1)", r, c);

    // MUL edge cases
    let r = alu.mul8(255, 255);
    println!("  MUL8(255, 255) = {} (expect 65025)", r);

    let r = alu.mul8(0, 123);
    println!("  MUL8(0, 123)   = {} (expect 0)", r);

    let r = alu.mul8(1, 200);
    println!("  MUL8(1, 200)   = {} (expect 200)", r);

    let r = alu.mul8(16, 16);
    println!("  MUL8(16, 16)   = {} (expect 256)", r);

    // CMP edge cases
    let f = alu.cmp8(0, 0);
    println!("  CMP8(0, 0)     = Z={} N={} C={} (expect Z=true N=false C=true)", f.z, f.n, f.c);

    let f = alu.cmp8(0, 1);
    println!("  CMP8(0, 1)     = Z={} N={} C={} (expect Z=false N=true C=false)", f.z, f.n, f.c);

    let f = alu.cmp8(255, 0);
    println!("  CMP8(255, 0)   = Z={} N={} C={} (expect Z=false N=true C=true)", f.z, f.n, f.c);

    // MIN/MAX
    println!("  MIN8(42, 99)   = {} (expect 42)", alu.min8(42, 99));
    println!("  MAX8(42, 99)   = {} (expect 99)", alu.max8(42, 99));
    println!("  MIN8(7, 7)     = {} (expect 7)", alu.min8(7, 7));
    println!("  MAX8(7, 7)     = {} (expect 7)", alu.max8(7, 7));

    // ── Final Summary ────────────────────────────────────────
    println!();
    println!("================================================================");
    println!("  FINAL SUMMARY");
    println!("================================================================");
    println!("  Total tests : {}", total_tests);
    println!("  Passed      : {}", total_pass);
    println!("  Failed      : {}", total_fail);
    println!();
    if total_fail == 0 {
        println!("  ALL {} TESTS PASSED — 8-bit ALU 100% CORRECT", total_tests);
    } else {
        println!("  FAILURES DETECTED: {}", total_fail);
    }
    println!("================================================================");
}
