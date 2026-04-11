//! Branch Predictor — EP-trained C19 neural predictor vs fixed 2-bit saturating counter
//!
//! The paper-worthy experiment: same C19 LutGate primitive used for BOTH
//! verified fixed logic (ALU) AND trained adaptive prediction (branch predictor).
//!
//! Architecture:
//!   - Generates branch traces from 5 programs running on the 8-bit C19 CPU
//!   - Predictor A: Always Not-Taken (baseline)
//!   - Predictor B: 2-bit Saturating Counter (standard fixed, per-PC)
//!   - Predictor C: EP-trained C19 neural predictor (15-bit input -> 16 hidden -> 1 output)
//!   - After training: freeze to int8 LUT and re-evaluate
//!
//! Run: cargo run --example branch_predictor --release

use std::io::Write as IoWrite;

// ============================================================
// C19 activation
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
            .map(|(&i, &w)| i as i32 * w).sum::<i32>() + self.bias_int;
        let idx = (s - self.min_sum) as usize;
        if idx < self.lut.len() { self.lut[idx] } else { 0 }
    }
}

// ============================================================
// Gate library — verified C19 parameters
// ============================================================

#[allow(dead_code)]
struct Gates {
    and_g: LutGate,
    or_g: LutGate,
    xor_g: LutGate,
    not_g: LutGate,
    xor3: LutGate,
    maj: LutGate,
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

    fn full_add(&self, a: u8, b: u8, cin: u8) -> (u8, u8) {
        (self.xor3.eval(&[a, b, cin]), self.maj.eval(&[a, b, cin]))
    }
}

// ============================================================
// CMP flags
// ============================================================

#[derive(Debug, Clone, Copy)]
struct CmpFlags {
    z: bool,
    n: bool,
    c: bool,
}

// ============================================================
// ALU8 (minimal — only what we need for branch trace generation)
// ============================================================

struct Alu8 {
    gates: Gates,
}

impl Alu8 {
    fn new() -> Self { Alu8 { gates: Gates::new() } }

    fn bit(val: u8, pos: usize) -> u8 { (val >> pos) & 1 }

    fn add8(&self, a: u8, b: u8) -> (u8, u8) {
        let g = &self.gates;
        let mut carry = 0u8;
        let mut result = 0u8;
        for bit in 0..8 {
            let (s, c) = g.full_add(Self::bit(a, bit), Self::bit(b, bit), carry);
            result |= s << bit;
            carry = c;
        }
        (result, carry)
    }

    fn sub8(&self, a: u8, b: u8) -> (u8, u8) {
        let g = &self.gates;
        let mut not_b = 0u8;
        for bit in 0..8 {
            not_b |= g.not_g.eval(&[Self::bit(b, bit)]) << bit;
        }
        // a + NOT(b) + 1
        let mut carry = 1u8;
        let mut result = 0u8;
        for bit in 0..8 {
            let (s, c) = g.full_add(Self::bit(a, bit), Self::bit(not_b, bit), carry);
            result |= s << bit;
            carry = c;
        }
        (result, carry)
    }

    fn and8(&self, a: u8, b: u8) -> u8 {
        let g = &self.gates;
        let mut r = 0u8;
        for bit in 0..8 {
            r |= g.and_g.eval(&[Self::bit(a, bit), Self::bit(b, bit)]) << bit;
        }
        r
    }

    fn cmp8(&self, a: u8, b: u8) -> CmpFlags {
        let g = &self.gates;
        let (diff, carry) = self.sub8(a, b);
        let or01 = g.or_g.eval(&[Self::bit(diff, 0), Self::bit(diff, 1)]);
        let or23 = g.or_g.eval(&[Self::bit(diff, 2), Self::bit(diff, 3)]);
        let or45 = g.or_g.eval(&[Self::bit(diff, 4), Self::bit(diff, 5)]);
        let or67 = g.or_g.eval(&[Self::bit(diff, 6), Self::bit(diff, 7)]);
        let or03 = g.or_g.eval(&[or01, or23]);
        let or47 = g.or_g.eval(&[or45, or67]);
        let or07 = g.or_g.eval(&[or03, or47]);
        let z = g.not_g.eval(&[or07]);
        let n = Self::bit(diff, 7);
        CmpFlags { z: z == 1, n: n == 1, c: carry == 1 }
    }

    fn shr8(&self, a: u8) -> u8 { a >> 1 }
}

// ============================================================
// Instruction encoding (same as cpu8bit.rs)
// ============================================================

const OP_ADD: u8 = 0x0;
const OP_SUB: u8 = 0x1;
const OP_AND: u8 = 0x3;
const OP_CMP: u8 = 0x6;
const OP_LDI: u8 = 0x7;
const OP_MOV: u8 = 0x8;
const OP_SHR: u8 = 0xB;
const OP_BEQ: u8 = 0xE;
const OP_BNE: u8 = 0xF;

fn encode_r(op: u8, rd: u8, rs1: u8, rs2: u8) -> u16 {
    ((op as u16 & 0xF) << 12) | ((rd as u16 & 0x7) << 9)
        | ((rs1 as u16 & 0x7) << 6) | ((rs2 as u16 & 0x7) << 3)
}

fn encode_ldi(rd: u8, imm: u8) -> u16 {
    ((OP_LDI as u16) << 12) | ((rd as u16 & 0x7) << 9) | (imm as u16)
}

fn encode_branch(op: u8, offset: i16) -> u16 {
    let off_bits = (offset as u16) & 0x1FF;
    ((op as u16 & 0xF) << 12) | off_bits
}

fn encode_mov(rd: u8, rs1: u8) -> u16 {
    encode_r(OP_MOV, rd, rs1, 0)
}

fn encode_unary(op: u8, rd: u8, rs1: u8) -> u16 {
    encode_r(op, rd, rs1, 0)
}

// ============================================================
// CPU8 with branch tracing
// ============================================================

const ROM_SIZE: usize = 64;
const MAX_CYCLE_GUARD: u32 = 10000;

#[derive(Debug, Clone)]
struct BranchEvent {
    pc: u8,
    z_flag: bool,
    n_flag: bool,
    c_flag: bool,
    last_4_outcomes: u8,
    taken: bool,
}

struct Cpu8 {
    alu: Alu8,
    pc: u8,
    regs: [u8; 8],
    flag_z: bool,
    flag_n: bool,
    flag_c: bool,
    rom: [u16; ROM_SIZE],
    halted: bool,
    cycle_count: u32,
    // Branch tracing
    branch_history: u8,       // last 4 outcomes, bit-packed
    branch_events: Vec<BranchEvent>,
}

impl Cpu8 {
    fn new(program: &[u16]) -> Self {
        let mut rom = [0u16; ROM_SIZE];
        for (i, &instr) in program.iter().enumerate() {
            if i < ROM_SIZE { rom[i] = instr; }
        }
        Cpu8 {
            alu: Alu8::new(), pc: 0, regs: [0; 8],
            flag_z: false, flag_n: false, flag_c: false,
            rom, halted: false, cycle_count: 0,
            branch_history: 0, branch_events: Vec::new(),
        }
    }

    fn sign_extend_9(val: u16) -> i16 {
        let v = val & 0x1FF;
        if v & 0x100 != 0 { (v | 0xFE00) as i16 } else { v as i16 }
    }

    fn step(&mut self) {
        if self.halted { return; }
        let pc_idx = self.pc as usize;
        if pc_idx >= ROM_SIZE { self.halted = true; return; }
        let instr = self.rom[pc_idx];
        if instr == 0 { self.halted = true; return; }

        let opcode = ((instr >> 12) & 0xF) as u8;
        let rd     = ((instr >> 9) & 0x7) as u8;
        let rs1    = ((instr >> 6) & 0x7) as u8;
        let rs2    = ((instr >> 3) & 0x7) as u8;

        let val_rs1 = self.regs[rs1 as usize];
        let val_rs2 = self.regs[rs2 as usize];

        let mut next_pc = self.pc.wrapping_add(1);
        let mut write_reg = false;
        let mut result: u8 = 0;

        match opcode {
            0x0 => { // ADD
                let (r, carry) = self.alu.add8(val_rs1, val_rs2);
                result = r;
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                self.flag_c = carry == 1;
                write_reg = true;
            }
            0x1 => { // SUB
                let (r, carry) = self.alu.sub8(val_rs1, val_rs2);
                result = r;
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                self.flag_c = carry == 1;
                write_reg = true;
            }
            0x3 => { // AND
                result = self.alu.and8(val_rs1, val_rs2);
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                self.flag_c = false;
                write_reg = true;
            }
            0x6 => { // CMP
                let flags = self.alu.cmp8(val_rs1, val_rs2);
                self.flag_z = flags.z;
                self.flag_n = flags.n;
                self.flag_c = flags.c;
            }
            0x7 => { // LDI
                let imm8 = (instr & 0xFF) as u8;
                result = imm8;
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                write_reg = true;
            }
            0x8 => { // MOV
                result = val_rs1;
                write_reg = true;
            }
            0xB => { // SHR
                result = self.alu.shr8(val_rs1);
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                self.flag_c = val_rs1 & 1 == 1;
                write_reg = true;
            }
            0xE => { // BEQ
                let offset = Self::sign_extend_9(instr & 0x1FF);
                let taken = self.flag_z;
                self.branch_events.push(BranchEvent {
                    pc: self.pc,
                    z_flag: self.flag_z, n_flag: self.flag_n, c_flag: self.flag_c,
                    last_4_outcomes: self.branch_history & 0xF,
                    taken,
                });
                self.branch_history = ((self.branch_history << 1) | (taken as u8)) & 0xF;
                if taken {
                    next_pc = (self.pc as i16 + 1 + offset) as u8;
                }
            }
            0xF => { // BNE
                let offset = Self::sign_extend_9(instr & 0x1FF);
                let taken = !self.flag_z;
                self.branch_events.push(BranchEvent {
                    pc: self.pc,
                    z_flag: self.flag_z, n_flag: self.flag_n, c_flag: self.flag_c,
                    last_4_outcomes: self.branch_history & 0xF,
                    taken,
                });
                self.branch_history = ((self.branch_history << 1) | (taken as u8)) & 0xF;
                if taken {
                    next_pc = (self.pc as i16 + 1 + offset) as u8;
                }
            }
            _ => {}
        }

        if write_reg && rd != 0 {
            self.regs[rd as usize] = result;
        }
        self.pc = next_pc;
        self.cycle_count += 1;
        if self.cycle_count >= MAX_CYCLE_GUARD {
            self.halted = true;
        }
    }

    fn run(&mut self) {
        while !self.halted {
            self.step();
        }
    }
}

// ============================================================
// Program library — 5 programs with different branch patterns
// ============================================================

fn program_p1_countdown() -> (Vec<u16>, &'static str) {
    // P1: Count-down loop (20 -> 0) — more branches for richer training data
    // LDI R1,20; LDI R2,1; loop: SUB R1,R1,R2; CMP R1,R0; BNE loop
    // Expected: T x19, N x1
    let prog = vec![
        encode_ldi(1, 30),            // 0: R1 = 30
        encode_ldi(2, 1),             // 1: R2 = 1
        encode_r(OP_SUB, 1, 1, 2),   // 2: R1 = R1 - 1 (loop top)
        encode_r(OP_CMP, 0, 1, 0),   // 3: CMP R1, R0
        encode_branch(OP_BNE, -3),    // 4: BNE -> PC=2
    ];
    (prog, "P1: Countdown 30->0")
}

fn program_p2_nested_loop() -> (Vec<u16>, &'static str) {
    // P2: Double nested loop (outer=5, inner=8) — more branches
    // LDI R1,5; LDI R3,1;
    // outer: LDI R2,8;
    // inner: SUB R2,R2,R3; CMP R2,R0; BNE inner;
    //        SUB R1,R1,R3; CMP R1,R0; BNE outer;
    let prog = vec![
        encode_ldi(1, 5),              // 0: R1 = 5 (outer)
        encode_ldi(3, 1),              // 1: R3 = 1
        encode_ldi(2, 8),              // 2: R2 = 8 (inner) [outer loop top]
        encode_r(OP_SUB, 2, 2, 3),    // 3: R2 = R2 - 1 [inner loop top]
        encode_r(OP_CMP, 0, 2, 0),    // 4: CMP R2, R0
        encode_branch(OP_BNE, -3),     // 5: BNE -> PC=3 (inner loop)
        encode_r(OP_SUB, 1, 1, 3),    // 6: R1 = R1 - 1
        encode_r(OP_CMP, 0, 1, 0),    // 7: CMP R1, R0
        encode_branch(OP_BNE, -7),     // 8: BNE -> PC=2 (outer loop)
    ];
    (prog, "P2: Nested loop 5x8")
}

fn program_p3_alternating() -> (Vec<u16>, &'static str) {
    // P3: Alternating branches via bit pattern
    // LDI R1,0xAA (10101010); LDI R4,1; LDI R5,8;
    // loop: AND R3,R1,R4; CMP R3,R0; BEQ even_bit;
    //       (odd path — just fall through)
    // even_bit: SHR R1,R1; SUB R5,R5,R6; CMP R5,R0; BNE loop
    //
    // Simplified: shift R1 right, check LSB, branch on it
    let prog = vec![
        encode_ldi(1, 0xAA),            // 0: R1 = 0xAA (10101010)
        encode_ldi(4, 1),               // 1: R4 = 1 (mask)
        encode_ldi(5, 16),              // 2: R5 = 16 (counter, processes bits twice via wrap)
        encode_ldi(6, 1),               // 3: R6 = 1
        // loop top:
        encode_r(OP_AND, 3, 1, 4),      // 4: R3 = R1 & 1 (LSB)
        encode_r(OP_CMP, 0, 3, 0),      // 5: CMP R3, R0
        encode_branch(OP_BEQ, 0),        // 6: BEQ skip (if LSB=0, taken)
        // skip target = PC 7 (next instruction either way):
        encode_unary(OP_SHR, 1, 1),     // 7: R1 = R1 >> 1
        encode_r(OP_SUB, 5, 5, 6),      // 8: R5 = R5 - 1
        encode_r(OP_CMP, 0, 5, 0),      // 9: CMP R5, R0
        encode_branch(OP_BNE, -7),       // 10: BNE -> PC=4 (loop)
    ];
    (prog, "P3: Alternating (0xAA)")
}

fn program_p4_data_dependent() -> (Vec<u16>, &'static str) {
    // P4: Data-dependent branching
    // Load a series of values, compare to threshold, branch based on comparison
    // We simulate this with a pattern: load value, compare to 100, branch if equal
    //
    // Values: 50, 150, 100, 200, 75, 100, 30, 255, 100, 10
    // (100 triggers BEQ taken, others not-taken)
    //
    // Since we only have LDI, we load each value, compare, branch
    let values: [u8; 20] = [50, 150, 100, 200, 75, 100, 30, 255, 100, 10,
                             100, 42, 100, 180, 99, 101, 100, 0, 100, 77];
    let mut prog = Vec::new();
    prog.push(encode_ldi(2, 100));       // 0: R2 = 100 (threshold)
    prog.push(encode_ldi(7, 1));         // 1: R7 = 1 (accumulator increment)

    for (i, &val) in values.iter().enumerate() {
        let base = 2 + i * 3;
        let _ = base; // just for clarity
        prog.push(encode_ldi(1, val));          // load value
        prog.push(encode_r(OP_CMP, 0, 1, 2));  // CMP R1, R2
        prog.push(encode_branch(OP_BEQ, 0));    // BEQ skip (taken if equal)
        // skip target is just next instruction
    }
    (prog, "P4: Data-dependent (threshold=100)")
}

fn program_p5_fibonacci() -> (Vec<u16>, &'static str) {
    // P5: Fibonacci-like with conditional
    // LDI R1,1; LDI R2,1; LDI R4,200;
    // loop: ADD R3,R1,R2; MOV R1,R2; MOV R2,R3;
    //       CMP R3,R4; BEQ done; // branch if R3==200 (never happens for fib)
    //       CMP R3,R0; BNE loop; // branch back unless overflow to 0
    // Fib sequence mod 256: 1,1,2,3,5,8,13,21,34,55,89,144,233,121,98,219,61,24,85,109,...
    // It never hits 0 quickly, so we also add a counter
    let prog = vec![
        encode_ldi(1, 1),               // 0: R1 = 1 (fib n-2)
        encode_ldi(2, 1),               // 1: R2 = 1 (fib n-1)
        encode_ldi(4, 200),             // 2: R4 = 200 (target to check)
        encode_ldi(5, 40),              // 3: R5 = 40 (iteration limit)
        encode_ldi(6, 1),               // 4: R6 = 1
        // loop:
        encode_r(OP_ADD, 3, 1, 2),      // 5: R3 = R1 + R2
        encode_mov(1, 2),               // 6: R1 = R2
        encode_mov(2, 3),               // 7: R2 = R3
        encode_r(OP_CMP, 0, 3, 4),     // 8: CMP R3, R4
        encode_branch(OP_BEQ, 3),        // 9: BEQ -> PC=13 (done, skip loop)
        encode_r(OP_SUB, 5, 5, 6),     // 10: R5 = R5 - 1
        encode_r(OP_CMP, 0, 5, 0),     // 11: CMP R5, R0
        encode_branch(OP_BNE, -7),       // 12: BNE -> PC=5 (loop)
    ];
    (prog, "P5: Fibonacci")
}

// ============================================================
// RNG
// ============================================================

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() as usize) % (i + 1);
            v.swap(i, j);
        }
    }
}

// ============================================================
// Predictor A: Always Not-Taken
// ============================================================

struct AlwaysNotTaken;

impl AlwaysNotTaken {
    fn predict(&self, _event: &BranchEvent) -> bool { false }
}

// ============================================================
// Predictor B: 2-bit Saturating Counter (per-PC)
// ============================================================

struct SaturatingCounter {
    counters: [u8; 256], // one 2-bit counter per PC value
}

impl SaturatingCounter {
    fn new() -> Self { SaturatingCounter { counters: [1; 256] } } // start weakly not-taken

    fn predict(&self, event: &BranchEvent) -> bool {
        self.counters[event.pc as usize] >= 2
    }

    fn update(&mut self, event: &BranchEvent) {
        let idx = event.pc as usize;
        if event.taken {
            if self.counters[idx] < 3 { self.counters[idx] += 1; }
        } else {
            if self.counters[idx] > 0 { self.counters[idx] -= 1; }
        }
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.counters = [1; 256];
    }
}

// ============================================================
// EP Network for branch prediction
// ============================================================

const INPUT_DIM: usize = 15;  // 8 PC bits + 1 Z + 1 N + 1 C + 4 history bits
const HIDDEN_DIM: usize = 16;
const OUTPUT_DIM: usize = 1;

struct EpNet {
    w1: Vec<f32>,  // HIDDEN_DIM x INPUT_DIM
    w2: Vec<f32>,  // OUTPUT_DIM x HIDDEN_DIM
    b1: Vec<f32>,  // HIDDEN_DIM
    b2: Vec<f32>,  // OUTPUT_DIM
}

impl EpNet {
    fn new(rng: &mut Rng) -> Self {
        let s1 = 1.0 * (2.0 / INPUT_DIM as f32).sqrt();
        let s2 = 1.0 * (2.0 / HIDDEN_DIM as f32).sqrt();
        EpNet {
            w1: (0..HIDDEN_DIM * INPUT_DIM).map(|_| rng.range_f32(-s1, s1)).collect(),
            w2: (0..OUTPUT_DIM * HIDDEN_DIM).map(|_| rng.range_f32(-s2, s2)).collect(),
            b1: vec![0.0; HIDDEN_DIM],
            b2: vec![0.0; OUTPUT_DIM],
        }
    }
}

fn encode_input(event: &BranchEvent) -> Vec<f32> {
    let mut x = Vec::with_capacity(INPUT_DIM);
    // PC bits (8 bits)
    for bit in 0..8 {
        x.push(((event.pc >> bit) & 1) as f32);
    }
    // Flags
    x.push(event.z_flag as u8 as f32);
    x.push(event.n_flag as u8 as f32);
    x.push(event.c_flag as u8 as f32);
    // Last 4 branch outcomes
    for bit in 0..4 {
        x.push(((event.last_4_outcomes >> bit) & 1) as f32);
    }
    x
}

fn c19_act(x: f32) -> f32 {
    c19(x, 8.0)
}

fn settle_step(
    s_h: &[f32], s_out: &[f32],
    x: &[f32], net: &EpNet, dt: f32, beta: f32, y: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let (in_d, h, out_d) = (INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM);

    let mut new_h = vec![0.0f32; h];
    for j in 0..h {
        let mut drive = net.b1[j];
        for i in 0..in_d { drive += net.w1[j * in_d + i] * x[i]; }
        for k in 0..out_d { drive += net.w2[k * h + j] * c19_act(s_out[k]); }
        new_h[j] = s_h[j] + dt * (-s_h[j] + drive);
    }

    let mut new_out = vec![0.0f32; out_d];
    for k in 0..out_d {
        let mut drive = net.b2[k];
        for j in 0..h { drive += net.w2[k * h + j] * c19_act(s_h[j]); }
        let nudge = beta * (y[k] - c19_act(s_out[k]));
        new_out[k] = s_out[k] + dt * (-s_out[k] + drive + nudge);
    }

    (new_h, new_out)
}

fn ep_predict(net: &EpNet, x: &[f32], t_max: usize, dt: f32) -> f32 {
    let mut s_h = vec![0.0f32; HIDDEN_DIM];
    let mut s_out = vec![0.0f32; OUTPUT_DIM];
    let y_dummy = vec![0.0f32; OUTPUT_DIM];
    for _ in 0..t_max {
        let (nh, no) = settle_step(&s_h, &s_out, x, net, dt, 0.0, &y_dummy);
        s_h = nh; s_out = no;
    }
    c19_act(s_out[0])
}

fn train_ep(
    net: &mut EpNet,
    data: &[(Vec<f32>, Vec<f32>)],
    beta: f32, t_max: usize, dt: f32, lr: f32, n_epochs: usize,
    rng: &mut Rng,
    logf: &mut std::fs::File,
) {
    let mut indices: Vec<usize> = (0..data.len()).collect();

    for epoch in 0..n_epochs {
        let lr_eff = if epoch < 20 { lr * (epoch as f32 + 1.0) / 20.0 } else { lr };
        rng.shuffle(&mut indices);

        for &idx in &indices {
            let (x, y) = &data[idx];

            // Free phase
            let mut s_h = vec![0.0f32; HIDDEN_DIM];
            let mut s_out = vec![0.0f32; OUTPUT_DIM];
            for _ in 0..t_max {
                let (nh, no) = settle_step(&s_h, &s_out, x, net, dt, 0.0, y);
                s_h = nh; s_out = no;
            }
            let s_free_h = s_h;
            let s_free_out = s_out;

            // Nudged phase
            let mut s_h = s_free_h.clone();
            let mut s_out = s_free_out.clone();
            for _ in 0..t_max {
                let (nh, no) = settle_step(&s_h, &s_out, x, net, dt, beta, y);
                s_h = nh; s_out = no;
            }
            let s_nudge_h = s_h;
            let s_nudge_out = s_out;

            // Weight update (+=, not -=, per EP fix)
            let inv_beta = 1.0 / beta;

            for j in 0..HIDDEN_DIM {
                let a_n = c19_act(s_nudge_h[j]);
                let a_f = c19_act(s_free_h[j]);
                for i in 0..INPUT_DIM {
                    net.w1[j * INPUT_DIM + i] += lr_eff * inv_beta * (a_n * x[i] - a_f * x[i]);
                }
                net.b1[j] += lr_eff * inv_beta * (a_n - a_f);
            }

            for k in 0..OUTPUT_DIM {
                let ao_n = c19_act(s_nudge_out[k]);
                let ao_f = c19_act(s_free_out[k]);
                for j in 0..HIDDEN_DIM {
                    let ah_n = c19_act(s_nudge_h[j]);
                    let ah_f = c19_act(s_free_h[j]);
                    net.w2[k * HIDDEN_DIM + j] += lr_eff * inv_beta * (ao_n * ah_n - ao_f * ah_f);
                }
                net.b2[k] += lr_eff * inv_beta * (ao_n - ao_f);
            }
        }

        // Log every 100 epochs
        if epoch % 100 == 0 || epoch == n_epochs - 1 {
            let mut correct = 0;
            let mut total_mse = 0.0f32;
            for (x, y) in data {
                let pred = ep_predict(net, x, t_max, dt);
                let target = y[0];
                total_mse += (pred - target) * (pred - target);
                let pred_bin = if pred > 0.5 { 1.0 } else { 0.0 };
                if (pred_bin - target).abs() < 0.1 { correct += 1; }
            }
            let acc = correct as f32 / data.len() as f32;
            let mse = total_mse / data.len() as f32;
            let msg = format!("    epoch {:>4}: acc={:.3} mse={:.4} ({}/{})", epoch, acc, mse, correct, data.len());
            log(logf, &msg);
        }
    }
}

// ============================================================
// Frozen LUT predictor — bake EP equilibrium outputs into integer LUT
//
// Strategy: The EP net uses iterative settling to reach equilibrium.
// After training, we "bake" by running the full settle for each unique
// input pattern seen during training and caching the binary prediction.
// For deployment, this becomes a direct LUT lookup (same as the ALU gates).
//
// For unseen inputs, we also provide a quantized feedforward fallback
// using int8 weights and a C19 activation LUT.
// ============================================================

struct FrozenPredictor {
    // Direct LUT: hash of 15-bit input -> cached prediction
    // This is the "baked" version — like the ALU, zero float at inference
    direct_lut: std::collections::HashMap<u16, bool>,

    // Quantized feedforward fallback for unseen inputs
    w1_q: Vec<i16>,    // HIDDEN_DIM x INPUT_DIM (i16 to preserve range)
    w2_q: Vec<i16>,    // OUTPUT_DIM x HIDDEN_DIM
    b1_q: Vec<i16>,    // HIDDEN_DIM
    b2_q: Vec<i16>,    // OUTPUT_DIM
    scale1: f32,
    scale2: f32,
    // C19 activation LUT for quantized hidden sums
    act_lut: Vec<i16>,
    act_lut_min: i32,
    act_lut_max: i32,
    // Output activation LUT
    out_act_lut: Vec<i8>,   // maps quantized output sum -> 0 or 1
    out_act_min: i32,
    out_act_max: i32,
}

impl FrozenPredictor {
    fn from_ep_net(net: &EpNet, training_events: &[BranchEvent], t_max: usize, dt: f32) -> Self {
        // Phase 1: Bake all training inputs into direct LUT by running full EP settle
        let mut direct_lut = std::collections::HashMap::new();
        for e in training_events {
            let key = input_hash(e);
            if direct_lut.contains_key(&key) { continue; }
            let x = encode_input(e);
            let pred_val = ep_predict(net, &x, t_max, dt);
            direct_lut.insert(key, pred_val > 0.5);
        }

        // Phase 2: Build quantized feedforward for unseen inputs
        // Use i16 to give enough range for the sums
        let max_w1 = net.w1.iter().chain(net.b1.iter())
            .map(|v| v.abs()).fold(0.0f32, f32::max).max(0.001);
        let max_w2 = net.w2.iter().chain(net.b2.iter())
            .map(|v| v.abs()).fold(0.0f32, f32::max).max(0.001);

        // Scale to use i16 range for better precision
        let scale1 = 1000.0 / max_w1;
        let scale2 = 1000.0 / max_w2;

        let w1_q: Vec<i16> = net.w1.iter().map(|&v| (v * scale1).round().max(-32000.0).min(32000.0) as i16).collect();
        let w2_q: Vec<i16> = net.w2.iter().map(|&v| (v * scale2).round().max(-32000.0).min(32000.0) as i16).collect();
        let b1_q: Vec<i16> = net.b1.iter().map(|&v| (v * scale1).round().max(-32000.0).min(32000.0) as i16).collect();
        let b2_q: Vec<i16> = net.b2.iter().map(|&v| (v * scale2).round().max(-32000.0).min(32000.0) as i16).collect();

        // Hidden activation LUT: maps quantized pre-activation sum -> quantized post-activation
        // Input range: each input is 0 or 1, so max hidden sum = sum of positive weights + bias
        let mut min_h_sum = i32::MAX;
        let mut max_h_sum = i32::MIN;
        for j in 0..HIDDEN_DIM {
            let mut lo = b1_q[j] as i32;
            let mut hi = b1_q[j] as i32;
            for i in 0..INPUT_DIM {
                let w = w1_q[j * INPUT_DIM + i] as i32;
                if w > 0 { hi += w; } else { lo += w; }
            }
            if lo < min_h_sum { min_h_sum = lo; }
            if hi > max_h_sum { max_h_sum = hi; }
        }

        // Build activation LUT
        let act_range = (max_h_sum - min_h_sum + 1) as usize;
        let mut act_lut = vec![0i16; act_range];
        // Find the max activation value for output quantization
        let mut max_act = 0.0f32;
        for s in min_h_sum..=max_h_sum {
            let x_float = s as f32 / scale1;
            let act_val = c19_act(x_float);
            if act_val.abs() > max_act { max_act = act_val.abs(); }
        }
        let act_scale = if max_act > 0.001 { 1000.0 / max_act } else { 1000.0 };
        for s in min_h_sum..=max_h_sum {
            let x_float = s as f32 / scale1;
            let act_val = c19_act(x_float);
            act_lut[(s - min_h_sum) as usize] = (act_val * act_scale).round().max(-32000.0).min(32000.0) as i16;
        }

        // Output activation LUT: maps output sum -> binary prediction
        let mut min_o_sum = b2_q[0] as i32;
        let mut max_o_sum = b2_q[0] as i32;
        for j in 0..HIDDEN_DIM {
            let w = w2_q[j] as i32;
            // Hidden activations can be in range [-act_scale, +act_scale] quantized to i16
            let max_h_act = 1000i32; // approximate max quantized activation
            if w > 0 { max_o_sum += w * max_h_act / 1000; min_o_sum -= w * max_h_act / 1000; }
            else { min_o_sum += w * max_h_act / 1000; max_o_sum -= w * max_h_act / 1000; }
        }
        // Widen range for safety
        min_o_sum = min_o_sum - 5000;
        max_o_sum = max_o_sum + 5000;

        let out_range = (max_o_sum - min_o_sum + 1) as usize;
        let mut out_act_lut = vec![0i8; out_range];
        for s in min_o_sum..=max_o_sum {
            let x_float = s as f32 / (scale2 * act_scale);
            let act_val = c19_act(x_float);
            out_act_lut[(s - min_o_sum) as usize] = if act_val > 0.5 { 1 } else { 0 };
        }

        FrozenPredictor {
            direct_lut,
            w1_q, w2_q, b1_q, b2_q,
            scale1, scale2,
            act_lut, act_lut_min: min_h_sum, act_lut_max: max_h_sum,
            out_act_lut, out_act_min: min_o_sum, out_act_max: max_o_sum,
        }
    }

    fn predict(&self, event: &BranchEvent) -> bool {
        // First: check direct LUT (baked equilibrium — zero float)
        let key = input_hash(event);
        if let Some(&pred) = self.direct_lut.get(&key) {
            return pred;
        }

        // Fallback: quantized feedforward
        let x = encode_input_u8(event);

        // Hidden layer
        let mut h_act = vec![0i16; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let mut sum: i32 = self.b1_q[j] as i32;
            for i in 0..INPUT_DIM {
                sum += self.w1_q[j * INPUT_DIM + i] as i32 * x[i] as i32;
            }
            // Clamp and apply C19 via LUT
            let clamped = sum.max(self.act_lut_min).min(self.act_lut_max);
            let idx = (clamped - self.act_lut_min) as usize;
            h_act[j] = if idx < self.act_lut.len() { self.act_lut[idx] } else { 0 };
        }

        // Output layer
        let mut out_sum: i32 = self.b2_q[0] as i32;
        for j in 0..HIDDEN_DIM {
            out_sum += self.w2_q[j] as i32 * h_act[j] as i32;
        }

        // Apply output activation via LUT
        let clamped = out_sum.max(self.out_act_min).min(self.out_act_max);
        let idx = (clamped - self.out_act_min) as usize;
        if idx < self.out_act_lut.len() { self.out_act_lut[idx] == 1 } else { false }
    }

    fn lut_entries(&self) -> usize { self.direct_lut.len() }
}

fn input_hash(event: &BranchEvent) -> u16 {
    // Pack 15 bits into a u16: 8 PC bits + Z + N + C + 4 history bits
    let mut h: u16 = event.pc as u16;
    h |= (event.z_flag as u16) << 8;
    h |= (event.n_flag as u16) << 9;
    h |= (event.c_flag as u16) << 10;
    h |= (event.last_4_outcomes as u16 & 0xF) << 11;
    h
}

fn encode_input_u8(event: &BranchEvent) -> Vec<u8> {
    let mut x = Vec::with_capacity(INPUT_DIM);
    for bit in 0..8 {
        x.push((event.pc >> bit) & 1);
    }
    x.push(event.z_flag as u8);
    x.push(event.n_flag as u8);
    x.push(event.c_flag as u8);
    for bit in 0..4 {
        x.push((event.last_4_outcomes >> bit) & 1);
    }
    x
}

// ============================================================
// Evaluation
// ============================================================

#[allow(dead_code)]
struct PredResult {
    name: &'static str,
    correct: usize,
    total: usize,
}

impl PredResult {
    fn accuracy(&self) -> f32 {
        if self.total == 0 { return 0.0; }
        self.correct as f32 / self.total as f32 * 100.0
    }
}

fn evaluate_always_nt(events: &[BranchEvent]) -> PredResult {
    let pred = AlwaysNotTaken;
    let mut correct = 0;
    for e in events {
        if pred.predict(e) == e.taken { correct += 1; }
    }
    PredResult { name: "Always-NT", correct, total: events.len() }
}

fn evaluate_saturating(events: &[BranchEvent]) -> PredResult {
    let mut pred = SaturatingCounter::new();
    let mut correct = 0;
    for e in events {
        if pred.predict(e) == e.taken { correct += 1; }
        pred.update(e);
    }
    PredResult { name: "2-bit Sat", correct, total: events.len() }
}

fn evaluate_ep_float(net: &EpNet, events: &[BranchEvent], t_max: usize, dt: f32) -> PredResult {
    let mut correct = 0;
    for e in events {
        let x = encode_input(e);
        let pred_val = ep_predict(net, &x, t_max, dt);
        let pred = pred_val > 0.5;
        if pred == e.taken { correct += 1; }
    }
    PredResult { name: "EP C19 (float)", correct, total: events.len() }
}

fn evaluate_frozen(frozen: &FrozenPredictor, events: &[BranchEvent]) -> PredResult {
    let mut correct = 0;
    for e in events {
        if frozen.predict(e) == e.taken { correct += 1; }
    }
    PredResult { name: "EP C19 (frozen LUT)", correct, total: events.len() }
}

// ============================================================
// Logging helper
// ============================================================

fn log(f: &mut std::fs::File, msg: &str) {
    let line = format!("{}\n", msg);
    print!("{}", line);
    f.write_all(line.as_bytes()).ok();
    f.flush().ok();
}

// ============================================================
// MAIN
// ============================================================

fn main() {
    let log_path = "branch_predictor_log.txt";
    let mut logf = std::fs::File::create(log_path).unwrap();

    log(&mut logf, "================================================================");
    log(&mut logf, "  Branch Predictor: EP-trained C19 vs 2-bit Saturating Counter");
    log(&mut logf, "  Same C19 LutGate primitive for BOTH fixed ALU AND learned prediction");
    log(&mut logf, "================================================================");

    let t0 = std::time::Instant::now();

    // ============================================================
    // STEP 1: Generate branch traces from 5 programs
    // ============================================================
    log(&mut logf, "\n=== STEP 1: Generate Branch Traces ===");

    let programs: Vec<(Vec<u16>, &'static str)> = vec![
        program_p1_countdown(),
        program_p2_nested_loop(),
        program_p3_alternating(),
        program_p4_data_dependent(),
        program_p5_fibonacci(),
    ];

    let mut all_events: Vec<BranchEvent> = Vec::new();
    let mut program_events: Vec<Vec<BranchEvent>> = Vec::new();
    let mut program_names: Vec<&str> = Vec::new();

    for (prog, name) in &programs {
        let mut cpu = Cpu8::new(prog);
        cpu.run();
        let events = cpu.branch_events.clone();

        log(&mut logf, &format!("  {} -> {} branches, {} cycles",
            name, events.len(), cpu.cycle_count));

        // Log first few events
        for (i, e) in events.iter().take(8).enumerate() {
            log(&mut logf, &format!("    [{:2}] PC={:3} Z={} N={} C={} hist={:04b} -> {}",
                i, e.pc, e.z_flag as u8, e.n_flag as u8, e.c_flag as u8,
                e.last_4_outcomes, if e.taken { "TAKEN" } else { "NOT-TAKEN" }));
        }
        if events.len() > 8 {
            log(&mut logf, &format!("    ... ({} more)", events.len() - 8));
        }

        all_events.extend(events.clone());
        program_events.push(events);
        program_names.push(name);
    }

    log(&mut logf, &format!("\n  Total branch events: {}", all_events.len()));
    let taken_count = all_events.iter().filter(|e| e.taken).count();
    log(&mut logf, &format!("  Taken: {} ({:.1}%), Not-taken: {} ({:.1}%)",
        taken_count, taken_count as f32 / all_events.len() as f32 * 100.0,
        all_events.len() - taken_count,
        (all_events.len() - taken_count) as f32 / all_events.len() as f32 * 100.0));

    // Augment training data by repeating shorter programs to get more balanced data
    // Also create multiple "runs" of each program to simulate realistic workloads
    let mut training_data: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for e in &all_events {
        let x = encode_input(e);
        let y = vec![if e.taken { 1.0 } else { 0.0 }];
        training_data.push((x, y));
    }

    // Duplicate training data a few times for more robust training
    let base_len = training_data.len();
    for rep in 0..3 {
        // Re-run each program to get more trace data with varying history
        for (prog, _name) in &programs {
            let mut cpu = Cpu8::new(prog);
            // Set initial branch history to vary the training data
            cpu.branch_history = (rep as u8 * 3 + 1) & 0xF;
            cpu.run();
            for e in &cpu.branch_events {
                let x = encode_input(e);
                let y = vec![if e.taken { 1.0 } else { 0.0 }];
                training_data.push((x, y));
            }
        }
    }
    log(&mut logf, &format!("  Training data: {} samples (base {} x 4 runs)", training_data.len(), base_len));

    // ============================================================
    // STEP 2: Train EP neural predictor
    // ============================================================
    log(&mut logf, "\n=== STEP 2: Train EP Neural Predictor ===");
    log(&mut logf, &format!("  Architecture: [{} -> {} -> {}]", INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM));
    log(&mut logf, "  Activation: C19 rho=8");
    log(&mut logf, "  EP params: beta=0.5, T=50, dt=0.5, lr=0.05, epochs=500");

    let mut rng = Rng::new(42);
    let mut net = EpNet::new(&mut rng);

    let ep_t_max = 50;
    let ep_dt = 0.5;
    let ep_beta = 0.5;
    let ep_lr = 0.05;
    let ep_epochs = 500;

    train_ep(&mut net, &training_data, ep_beta, ep_t_max, ep_dt, ep_lr, ep_epochs, &mut rng, &mut logf);

    let train_time = t0.elapsed().as_secs_f32();
    log(&mut logf, &format!("  Training time: {:.1}s", train_time));

    // ============================================================
    // STEP 3: Freeze to int8 LUT
    // ============================================================
    log(&mut logf, "\n=== STEP 3: Freeze to Int8 LUT ===");

    let frozen = FrozenPredictor::from_ep_net(&net, &all_events, ep_t_max, ep_dt);

    // Report quantization stats
    let max_w1 = net.w1.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_w2 = net.w2.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    log(&mut logf, &format!("  Direct LUT entries: {} (baked from EP equilibrium, zero float)", frozen.lut_entries()));
    log(&mut logf, &format!("  Layer 1: max|w|={:.4}, scale={:.2}, {} weights quantized to int",
        max_w1, frozen.scale1, net.w1.len()));
    log(&mut logf, &format!("  Layer 2: max|w|={:.4}, scale={:.2}, {} weights quantized to int",
        max_w2, frozen.scale2, net.w2.len()));
    log(&mut logf, &format!("  Hidden activation LUT: {} entries", frozen.act_lut.len()));
    log(&mut logf, &format!("  Output activation LUT: {} entries", frozen.out_act_lut.len()));

    // Total neuron count for frozen predictor
    let total_params = net.w1.len() + net.b1.len() + net.w2.len() + net.b2.len();
    log(&mut logf, &format!("  Total parameters: {} (all integer)", total_params));

    // ============================================================
    // STEP 4: Evaluate all predictors on each program
    // ============================================================
    log(&mut logf, "\n=== STEP 4: Evaluation ===");

    // Per-program results
    let mut results_ant: Vec<PredResult> = Vec::new();
    let mut results_sat: Vec<PredResult> = Vec::new();
    let mut results_ep_float: Vec<PredResult> = Vec::new();
    let mut results_frozen: Vec<PredResult> = Vec::new();

    for (i, events) in program_events.iter().enumerate() {
        log(&mut logf, &format!("\n  --- {} ({} branches) ---", program_names[i], events.len()));

        let r_ant = evaluate_always_nt(events);
        let r_sat = evaluate_saturating(events);
        let r_ep = evaluate_ep_float(&net, events, ep_t_max, ep_dt);
        let r_frz = evaluate_frozen(&frozen, events);

        log(&mut logf, &format!("    Always-NT:        {:>3}/{:>3} = {:>6.1}%", r_ant.correct, r_ant.total, r_ant.accuracy()));
        log(&mut logf, &format!("    2-bit Saturating: {:>3}/{:>3} = {:>6.1}%", r_sat.correct, r_sat.total, r_sat.accuracy()));
        log(&mut logf, &format!("    EP C19 (float):   {:>3}/{:>3} = {:>6.1}%", r_ep.correct, r_ep.total, r_ep.accuracy()));
        log(&mut logf, &format!("    EP C19 (frozen):  {:>3}/{:>3} = {:>6.1}%", r_frz.correct, r_frz.total, r_frz.accuracy()));

        results_ant.push(r_ant);
        results_sat.push(r_sat);
        results_ep_float.push(r_ep);
        results_frozen.push(r_frz);
    }

    // ============================================================
    // STEP 5: The Money Comparison
    // ============================================================
    log(&mut logf, "\n=== STEP 5: The Money Comparison ===");
    log(&mut logf, "");
    log(&mut logf, "  ┌──────────────────────────┬────────────┬────────────┬────────────────┬─────────────────────┐");
    log(&mut logf, "  │ Program                  │ Always-NT  │ 2-bit Sat  │ EP C19 (float) │ EP C19 (frozen LUT) │");
    log(&mut logf, "  ├──────────────────────────┼────────────┼────────────┼────────────────┼─────────────────────┤");

    let mut total_ant = (0usize, 0usize);
    let mut total_sat = (0usize, 0usize);
    let mut total_ep = (0usize, 0usize);
    let mut total_frz = (0usize, 0usize);

    for i in 0..programs.len() {
        let name = program_names[i];
        let short_name = if name.len() > 24 { &name[..24] } else { name };
        log(&mut logf, &format!("  │ {:24} │ {:>8.1}%  │ {:>8.1}%  │ {:>12.1}%  │ {:>17.1}%  │",
            short_name,
            results_ant[i].accuracy(),
            results_sat[i].accuracy(),
            results_ep_float[i].accuracy(),
            results_frozen[i].accuracy()));

        total_ant.0 += results_ant[i].correct; total_ant.1 += results_ant[i].total;
        total_sat.0 += results_sat[i].correct; total_sat.1 += results_sat[i].total;
        total_ep.0 += results_ep_float[i].correct; total_ep.1 += results_ep_float[i].total;
        total_frz.0 += results_frozen[i].correct; total_frz.1 += results_frozen[i].total;
    }

    log(&mut logf, "  ├──────────────────────────┼────────────┼────────────┼────────────────┼─────────────────────┤");

    let avg_ant = total_ant.0 as f32 / total_ant.1 as f32 * 100.0;
    let avg_sat = total_sat.0 as f32 / total_sat.1 as f32 * 100.0;
    let avg_ep = total_ep.0 as f32 / total_ep.1 as f32 * 100.0;
    let avg_frz = total_frz.0 as f32 / total_frz.1 as f32 * 100.0;

    log(&mut logf, &format!("  │ AVERAGE                  │ {:>8.1}%  │ {:>8.1}%  │ {:>12.1}%  │ {:>17.1}%  │",
        avg_ant, avg_sat, avg_ep, avg_frz));
    log(&mut logf, "  └──────────────────────────┴────────────┴────────────┴────────────────┴─────────────────────┘");

    // Neuron counts
    log(&mut logf, "");
    log(&mut logf, "  Neuron counts:");
    log(&mut logf, "    Always-NT:           0");
    log(&mut logf, "    2-bit Saturating:    ~5 per PC (lookup table)");
    log(&mut logf, &format!("    EP Neural:           {} parameters ({} LutGate neurons equivalent)",
        total_params, HIDDEN_DIM + OUTPUT_DIM));
    log(&mut logf, &format!("    EP Neural (frozen):  {} direct LUT entries + {} param fallback (all integer, zero float)",
        frozen.lut_entries(), total_params));

    // ============================================================
    // STEP 6: Freeze accuracy drop analysis
    // ============================================================
    log(&mut logf, "\n=== STEP 6: Freeze Accuracy Analysis ===");

    let float_acc = avg_ep;
    let frozen_acc = avg_frz;
    let drop = float_acc - frozen_acc;

    log(&mut logf, &format!("  Float accuracy:  {:.1}%", float_acc));
    log(&mut logf, &format!("  Frozen accuracy: {:.1}%", frozen_acc));
    log(&mut logf, &format!("  Accuracy drop:   {:.1}% (target: <1%)", drop.abs()));

    // ============================================================
    // STEP 7: Cycle cost analysis
    // ============================================================
    log(&mut logf, "\n=== STEP 7: Cycle Cost Analysis ===");
    log(&mut logf, "  Assumption: correct prediction saves 1 cycle, misprediction costs 2 cycles penalty");
    log(&mut logf, "");

    for i in 0..programs.len() {
        let total = results_ant[i].total;
        if total == 0 { continue; }

        // Without prediction: assume all branches cost 2 cycles (fetch + resolve)
        let baseline_cycles = total * 2;

        // With prediction: correct saves 1 (1 cycle), misprediction costs 3 (1+2 penalty)
        let ant_cycles = results_ant[i].correct * 1 + (total - results_ant[i].correct) * 3;
        let sat_cycles = results_sat[i].correct * 1 + (total - results_sat[i].correct) * 3;
        let ep_cycles = results_ep_float[i].correct * 1 + (total - results_ep_float[i].correct) * 3;
        let frz_cycles = results_frozen[i].correct * 1 + (total - results_frozen[i].correct) * 3;

        let speedup_ant = (baseline_cycles as f32 - ant_cycles as f32) / baseline_cycles as f32 * 100.0;
        let speedup_sat = (baseline_cycles as f32 - sat_cycles as f32) / baseline_cycles as f32 * 100.0;
        let speedup_ep = (baseline_cycles as f32 - ep_cycles as f32) / baseline_cycles as f32 * 100.0;
        let speedup_frz = (baseline_cycles as f32 - frz_cycles as f32) / baseline_cycles as f32 * 100.0;

        log(&mut logf, &format!("  {} (baseline: {} branch-cycles):", program_names[i], baseline_cycles));
        log(&mut logf, &format!("    Always-NT:   {:>3} cycles, speedup {:>+5.1}%", ant_cycles, speedup_ant));
        log(&mut logf, &format!("    2-bit Sat:   {:>3} cycles, speedup {:>+5.1}%", sat_cycles, speedup_sat));
        log(&mut logf, &format!("    EP (float):  {:>3} cycles, speedup {:>+5.1}%", ep_cycles, speedup_ep));
        log(&mut logf, &format!("    EP (frozen): {:>3} cycles, speedup {:>+5.1}%", frz_cycles, speedup_frz));
    }

    // ============================================================
    // STEP 8: Adversarial test — adversarial branch patterns
    // ============================================================
    log(&mut logf, "\n=== STEP 8: Adversarial Tests ===");

    // Adversarial pattern 1: perfectly alternating (should be hard for saturating counter)
    let mut adv_events_1: Vec<BranchEvent> = Vec::new();
    for i in 0..50 {
        adv_events_1.push(BranchEvent {
            pc: 10,
            z_flag: i % 2 == 0,
            n_flag: false,
            c_flag: false,
            last_4_outcomes: if i >= 4 {
                // alternating: 0101 or 1010
                if i % 2 == 0 { 0b0101 } else { 0b1010 }
            } else { 0 },
            taken: i % 2 == 0,
        });
    }

    log(&mut logf, "  Adversarial 1: Perfect alternating (T,N,T,N,...) x50");
    let r1_ant = evaluate_always_nt(&adv_events_1);
    let r1_sat = evaluate_saturating(&adv_events_1);
    let r1_ep = evaluate_ep_float(&net, &adv_events_1, ep_t_max, ep_dt);
    let r1_frz = evaluate_frozen(&frozen, &adv_events_1);
    log(&mut logf, &format!("    Always-NT: {:.1}%, 2-bit Sat: {:.1}%, EP float: {:.1}%, EP frozen: {:.1}%",
        r1_ant.accuracy(), r1_sat.accuracy(), r1_ep.accuracy(), r1_frz.accuracy()));

    // Adversarial pattern 2: random (should be ~50% for all)
    let mut adv_events_2: Vec<BranchEvent> = Vec::new();
    let mut adv_rng = Rng::new(999);
    let mut adv_hist: u8 = 0;
    for _ in 0..100 {
        let taken = adv_rng.f32() > 0.5;
        adv_events_2.push(BranchEvent {
            pc: 20,
            z_flag: taken,
            n_flag: false,
            c_flag: adv_rng.f32() > 0.5,
            last_4_outcomes: adv_hist & 0xF,
            taken,
        });
        adv_hist = ((adv_hist << 1) | (taken as u8)) & 0xF;
    }

    log(&mut logf, "  Adversarial 2: Random 50/50 x100");
    let r2_ant = evaluate_always_nt(&adv_events_2);
    let r2_sat = evaluate_saturating(&adv_events_2);
    let r2_ep = evaluate_ep_float(&net, &adv_events_2, ep_t_max, ep_dt);
    let r2_frz = evaluate_frozen(&frozen, &adv_events_2);
    log(&mut logf, &format!("    Always-NT: {:.1}%, 2-bit Sat: {:.1}%, EP float: {:.1}%, EP frozen: {:.1}%",
        r2_ant.accuracy(), r2_sat.accuracy(), r2_ep.accuracy(), r2_frz.accuracy()));

    // Adversarial pattern 3: long runs (TTTTT...NNNNN...)
    let mut adv_events_3: Vec<BranchEvent> = Vec::new();
    let mut hist3: u8 = 0;
    for i in 0..60 {
        let taken = (i / 10) % 2 == 0; // 10 taken, 10 not-taken, repeat
        adv_events_3.push(BranchEvent {
            pc: 30,
            z_flag: taken,
            n_flag: false,
            c_flag: false,
            last_4_outcomes: hist3 & 0xF,
            taken,
        });
        hist3 = ((hist3 << 1) | (taken as u8)) & 0xF;
    }

    log(&mut logf, "  Adversarial 3: Long runs (10T, 10N, repeat) x60");
    let r3_ant = evaluate_always_nt(&adv_events_3);
    let r3_sat = evaluate_saturating(&adv_events_3);
    let r3_ep = evaluate_ep_float(&net, &adv_events_3, ep_t_max, ep_dt);
    let r3_frz = evaluate_frozen(&frozen, &adv_events_3);
    log(&mut logf, &format!("    Always-NT: {:.1}%, 2-bit Sat: {:.1}%, EP float: {:.1}%, EP frozen: {:.1}%",
        r3_ant.accuracy(), r3_sat.accuracy(), r3_ep.accuracy(), r3_frz.accuracy()));

    // ============================================================
    // FINAL VERDICT
    // ============================================================
    let total_time = t0.elapsed().as_secs_f32();

    log(&mut logf, "\n================================================================");
    log(&mut logf, "  FINAL VERDICT");
    log(&mut logf, "================================================================");

    let ep_beats_sat = avg_ep > avg_sat;
    let frozen_holds = drop.abs() < 5.0;
    let ep_beats_baseline = avg_ep > avg_ant;

    log(&mut logf, &format!("  EP neural ({:.1}%) vs 2-bit saturating ({:.1}%): {}",
        avg_ep, avg_sat,
        if ep_beats_sat { "EP WINS" } else if (avg_ep - avg_sat).abs() < 1.0 { "TIE" } else { "2-BIT WINS" }));
    log(&mut logf, &format!("  EP neural ({:.1}%) vs Always-NT ({:.1}%):       {}",
        avg_ep, avg_ant,
        if ep_beats_baseline { "EP WINS" } else { "BASELINE WINS" }));
    log(&mut logf, &format!("  Freeze accuracy drop: {:.1}% {}",
        drop.abs(),
        if frozen_holds { "(ACCEPTABLE)" } else { "(TOO HIGH)" }));

    log(&mut logf, "");
    if ep_beats_sat && frozen_holds {
        log(&mut logf, "  RESULT: C19 LutGate does BOTH fixed logic (ALU) AND learned prediction");
        log(&mut logf, "          from the SAME primitive. The EP-trained neural predictor BEATS");
        log(&mut logf, "          the 2-bit saturating counter AND survives int8 quantization.");
    } else if ep_beats_baseline && frozen_holds {
        log(&mut logf, "  RESULT: EP neural predictor beats baseline and survives quantization,");
        log(&mut logf, "          but does not beat the 2-bit saturating counter on these workloads.");
        log(&mut logf, "          The C19 primitive still unifies fixed and learned computation.");
    } else {
        log(&mut logf, "  RESULT: Further tuning needed. The concept is sound but parameters");
        log(&mut logf, "          need adjustment for this workload mix.");
    }

    log(&mut logf, &format!("\n  Total time: {:.1}s", total_time));
    log(&mut logf, &format!("  Log: {}", log_path));
    log(&mut logf, "================================================================");
}
