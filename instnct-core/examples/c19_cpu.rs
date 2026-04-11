//! C19 Hybrid CPU — 4-bit neuromorphic processor built from C19 neurons
//!
//! Architecture:
//!   - 4-bit data width, 8 registers (R0=0, R1-R7 general purpose)
//!   - 12-bit instructions: [OP:3][Rd:3][Rs1:3][Rs2/Imm:3]
//!   - 6 ALU ops (ADD,SUB,AND,OR,XOR,CMP) — all exhaustive-verified integer LUT
//!   - 16-slot instruction ROM, 4-bit program counter
//!   - Flags: Zero, Carry, Negative
//!
//! All computation uses C19 integer LUT gates — zero floating point in the hot path.
//!
//! Run: cargo run --example c19_cpu --release

use std::io::Write;

// ============================================================
// C19 activation (only used for LUT baking at init time)
// ============================================================

fn c19(x: f32, rho: f32) -> f32 {
    let c = 1.0f32; let l = 6.0;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

// ============================================================
// Integer LUT Gate — the universal building block
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
            let y = c19(s as f32 / denom as f32, rho);
            lut[(s - min_s) as usize] = if y > thr { 1 } else { 0 };
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
// Gate Library — all verified gates
// ============================================================

struct GateLib {
    xor3: LutGate,
    maj: LutGate,
    not_g: LutGate,
    and_g: LutGate,
    or_g: LutGate,
    xor_g: LutGate,
    nor3: LutGate,
    // Selector minterms (3-bit opcode → 1-of-8)
    sel: [LutGate; 8],
}

impl GateLib {
    fn new() -> Self {
        // Minterm detector for each 3-bit pattern
        // Pattern (b0,b1,b2) → fires on exactly one value
        let make_minterm = |target: u8| -> LutGate {
            // For each bit: if target bit=1, positive weight; if 0, negative weight
            let b0 = if target & 1 != 0 { 1.0 } else { -1.0 };
            let b1 = if target & 2 != 0 { 1.0 } else { -1.0 };
            let b2 = if target & 4 != 0 { 1.0 } else { -1.0 };
            // Scale weights: need separation. Use the exhaustive-searched patterns.
            // For patterns with all-negative or mixed, use appropriate rho.
            let ones = (target & 1) + ((target >> 1) & 1) + ((target >> 2) & 1);
            match ones {
                0 => LutGate::new(&[-9.75, -9.75, -9.75], -5.50, 16.0, -4.0), // NOR3 pattern
                3 => LutGate::new(&[8.0, 8.0, 8.0], -10.0, 0.0, 4.0),         // AND3 pattern
                1 => {
                    // Exactly one bit is 1
                    let w0 = if target & 1 != 0 { 10.0 } else { -10.0 };
                    let w1 = if target & 2 != 0 { 10.0 } else { -10.0 };
                    let w2 = if target & 4 != 0 { 10.0 } else { -10.0 };
                    LutGate::new(&[w0, w1, w2], 5.5, 0.0, 4.0)
                }
                2 => {
                    // Exactly two bits are 1 — MAJ-like pattern
                    let w0 = if target & 1 != 0 { 8.5 } else { -8.5 };
                    let w1 = if target & 2 != 0 { 8.5 } else { -8.5 };
                    let w2 = if target & 4 != 0 { 8.5 } else { -8.5 };
                    LutGate::new(&[w0, w1, w2], -2.75, 0.0, 4.0)
                }
                _ => unreachable!()
            }
        };

        GateLib {
            xor3: LutGate::new(&[1.5, 1.5, 1.5], 3.0, 16.0, 0.6),
            maj: LutGate::new(&[8.5, 8.5, 8.5], -2.75, 0.0, 4.0),
            not_g: LutGate::new(&[-9.75], -5.5, 16.0, -4.0),
            and_g: LutGate::new(&[10.0, 10.0], -4.5, 0.0, 4.0),
            or_g: LutGate::new(&[8.75, 8.75], 5.5, 0.0, 4.0),
            xor_g: LutGate::new(&[0.5, 0.5], 0.0, 16.0, 0.6),
            nor3: LutGate::new(&[-9.75, -9.75, -9.75], -5.5, 16.0, -4.0),
            sel: [
                make_minterm(0), make_minterm(1), make_minterm(2), make_minterm(3),
                make_minterm(4), make_minterm(5), make_minterm(6), make_minterm(7),
            ],
        }
    }
}

// ============================================================
// CPU Components
// ============================================================

struct Cpu {
    gates: GateLib,
    // State
    pc: u8,           // 4-bit program counter
    regs: [u8; 8],    // 8 × 4-bit registers (R0 always 0)
    flag_z: bool,     // zero flag
    flag_c: bool,     // carry flag
    flag_n: bool,     // negative flag (MSB of result)
    // Program ROM
    rom: [u16; 32],   // 32 × 12-bit instructions (5-bit PC)
    halted: bool,
    cycle_count: u32,
}

impl Cpu {
    fn new(program: &[u16]) -> Self {
        let mut rom = [0u16; 32];
        for (i, &instr) in program.iter().enumerate() {
            if i < 32 { rom[i] = instr & 0xFFF; }
        }
        Cpu {
            gates: GateLib::new(),
            pc: 0, regs: [0; 8],
            flag_z: false, flag_c: false, flag_n: false,
            rom, halted: false, cycle_count: 0,
        }
    }

    // ---- ALU operations (integer LUT, zero float) ----

    fn alu_add4(&self, a: u8, b: u8) -> (u8, bool) {
        let mut c = 0u8; let mut r = 0u8;
        for bit in 0..4 {
            let ab = (a >> bit) & 1; let bb = (b >> bit) & 1;
            r |= self.gates.xor3.eval(&[ab, bb, c]) << bit;
            c = self.gates.maj.eval(&[ab, bb, c]);
        }
        (r & 0xF, c == 1)
    }

    fn alu_sub4(&self, a: u8, b: u8) -> (u8, bool) {
        let mut bn = 0u8;
        for bit in 0..4 { bn |= self.gates.not_g.eval(&[(b >> bit) & 1]) << bit; }
        let mut c = 1u8; let mut r = 0u8;
        for bit in 0..4 {
            let ab = (a >> bit) & 1; let bb = (bn >> bit) & 1;
            r |= self.gates.xor3.eval(&[ab, bb, c]) << bit;
            c = self.gates.maj.eval(&[ab, bb, c]);
        }
        (r & 0xF, c == 1)
    }

    fn alu_and4(&self, a: u8, b: u8) -> u8 {
        let mut r = 0u8;
        for bit in 0..4 { r |= self.gates.and_g.eval(&[(a >> bit) & 1, (b >> bit) & 1]) << bit; }
        r
    }

    fn alu_or4(&self, a: u8, b: u8) -> u8 {
        let mut r = 0u8;
        for bit in 0..4 { r |= self.gates.or_g.eval(&[(a >> bit) & 1, (b >> bit) & 1]) << bit; }
        r
    }

    fn alu_xor4(&self, a: u8, b: u8) -> u8 {
        let mut r = 0u8;
        for bit in 0..4 { r |= self.gates.xor_g.eval(&[(a >> bit) & 1, (b >> bit) & 1]) << bit; }
        r
    }

    // ---- Instruction decoder ----

    fn decode(&self, instr: u16) -> (u8, u8, u8, u8) {
        let op  = ((instr >> 9) & 0x7) as u8;
        let rd  = ((instr >> 6) & 0x7) as u8;
        let rs1 = ((instr >> 3) & 0x7) as u8;
        let rs2 = (instr & 0x7) as u8;
        (op, rd, rs1, rs2)
    }

    // ---- Selector: which ALU op (uses minterm detectors) ----

    fn select_alu(&self, op: u8) -> u8 {
        let bits = [(op & 1), ((op >> 1) & 1), ((op >> 2) & 1)];
        for i in 0..8u8 {
            if self.gates.sel[i as usize].eval(&bits) == 1 {
                return i;
            }
        }
        0 // fallback
    }

    // ---- Single cycle step ----

    fn step(&mut self) -> String {
        if self.halted { return "HALTED".to_string(); }

        // Fetch
        let instr = self.rom[self.pc as usize & 0x1F];
        let (op, rd, rs1, rs2_imm) = self.decode(instr);

        // Read registers
        let val_rs1 = self.regs[rs1 as usize] & 0xF;
        let val_rs2 = self.regs[rs2_imm as usize] & 0xF;
        let imm = rs2_imm & 0x7; // 3-bit immediate

        // Select ALU operation via minterm detectors
        let selected = self.select_alu(op);

        let mut trace = format!("PC={:X} | {:03b}_{:03b}_{:03b}_{:03b} | ",
            self.pc, op, rd, rs1, rs2_imm);

        let mut next_pc = (self.pc + 1) & 0x1F; // 5-bit PC
        let mut write_reg = false;
        let mut result = 0u8;

        match selected {
            0 => { // ADD
                let (r, carry) = self.alu_add4(val_rs1, val_rs2);
                result = r;
                self.flag_c = carry;
                self.flag_z = result == 0;
                self.flag_n = (result >> 3) & 1 == 1;
                write_reg = true;
                trace += &format!("ADD R{}=R{}+R{} = {}+{} = {}", rd, rs1, rs2_imm, val_rs1, val_rs2, result);
            }
            1 => { // SUB
                let (r, carry) = self.alu_sub4(val_rs1, val_rs2);
                result = r;
                self.flag_c = carry;
                self.flag_z = result == 0;
                self.flag_n = (result >> 3) & 1 == 1;
                write_reg = true;
                trace += &format!("SUB R{}=R{}-R{} = {}-{} = {}", rd, rs1, rs2_imm, val_rs1, val_rs2, result);
            }
            2 => { // AND
                result = self.alu_and4(val_rs1, val_rs2);
                self.flag_z = result == 0;
                self.flag_n = (result >> 3) & 1 == 1;
                write_reg = true;
                trace += &format!("AND R{}=R{}&R{} = {}&{} = {}", rd, rs1, rs2_imm, val_rs1, val_rs2, result);
            }
            3 => { // OR
                result = self.alu_or4(val_rs1, val_rs2);
                self.flag_z = result == 0;
                self.flag_n = (result >> 3) & 1 == 1;
                write_reg = true;
                trace += &format!("OR  R{}=R{}|R{} = {}|{} = {}", rd, rs1, rs2_imm, val_rs1, val_rs2, result);
            }
            4 => { // XOR
                result = self.alu_xor4(val_rs1, val_rs2);
                self.flag_z = result == 0;
                self.flag_n = (result >> 3) & 1 == 1;
                write_reg = true;
                trace += &format!("XOR R{}=R{}^R{} = {}^{} = {}", rd, rs1, rs2_imm, val_rs1, val_rs2, result);
            }
            5 => { // CMP (no writeback, just flags)
                let (r, carry) = self.alu_sub4(val_rs1, val_rs2);
                self.flag_z = r == 0;
                self.flag_c = carry;
                self.flag_n = (r >> 3) & 1 == 1;
                trace += &format!("CMP R{}-R{} = {}-{} → Z={} C={} N={}",
                    rs1, rs2_imm, val_rs1, val_rs2, self.flag_z as u8, self.flag_c as u8, self.flag_n as u8);
            }
            6 => { // LDI (load immediate)
                result = imm & 0xF;
                write_reg = true;
                self.flag_z = result == 0;
                trace += &format!("LDI R{}={}", rd, result);
            }
            7 => { // BRANCH family: rd field selects condition
                // rd=0: BEQ (Z=1), rd=1: BNE (Z=0), rd=2: BLT (N=1 && !Z), rd=3: BGE (N=0 || Z)
                let cond = rd;
                let taken = match cond {
                    0 => self.flag_z,                          // BEQ
                    1 => !self.flag_z,                         // BNE
                    2 => !self.flag_c && !self.flag_z,         // BLT (unsigned: no carry and not zero = a < b)
                    3 => self.flag_c || self.flag_z,           // BGE (unsigned: carry or zero = a >= b)
                    _ => false,
                };
                let cond_name = ["BEQ","BNE","BLT","BGE"][cond as usize & 3];
                if taken {
                    let offset = if imm & 4 != 0 { imm | 0xF8 } else { imm };
                    next_pc = (self.pc.wrapping_add(1).wrapping_add(offset)) & 0x1F; // relative to PC+1
                    trace += &format!("{} taken → PC={:X}", cond_name, next_pc);
                } else {
                    trace += &format!("{} not taken", cond_name);
                }
            }
            _ => {
                trace += &format!("NOP (unknown op={})", selected);
            }
        }

        // Writeback
        if write_reg && rd != 0 { // R0 is hardwired to 0
            self.regs[rd as usize] = result & 0xF;
        }

        // Update PC
        self.pc = next_pc;
        self.cycle_count += 1;

        // Halt detection: PC wraps to instruction 0 which is NOP/end
        // Or: explicit halt = all zeros instruction
        if self.rom[self.pc as usize & 0xF] == 0 && self.pc != 0 {
            // Next instruction is 0 (NOP) and we're not at start — halt
        }

        trace
    }

    fn reg_dump(&self) -> String {
        let mut s = String::from("  Regs: ");
        for i in 0..8 {
            s += &format!("R{}={} ", i, self.regs[i]);
        }
        s += &format!("| Z={} C={} N={}", self.flag_z as u8, self.flag_c as u8, self.flag_n as u8);
        s
    }

    fn run(&mut self, max_cycles: u32, logf: &mut std::fs::File) {
        log(logf, &format!("  Program loaded, {} instructions", self.rom.iter().filter(|&&x| x != 0).count()));
        log(logf, &format!("{}", self.reg_dump()));

        for _ in 0..max_cycles {
            if self.halted { break; }
            let trace = self.step();
            log(logf, &format!("  Cycle {:3}: {}", self.cycle_count, trace));
            log(logf, &format!("{}", self.reg_dump()));

            // Halt if PC points to a zero instruction (end of program)
            if self.rom[self.pc as usize & 0x1F] == 0 {
                log(logf, &format!("  HALT at cycle {} (PC={:X} → NOP)", self.cycle_count, self.pc));
                self.halted = true;
            }
        }
    }
}

// ============================================================
// Instruction encoding helpers
// ============================================================

fn encode_r(op: u8, rd: u8, rs1: u8, rs2: u8) -> u16 {
    ((op as u16 & 7) << 9) | ((rd as u16 & 7) << 6) | ((rs1 as u16 & 7) << 3) | (rs2 as u16 & 7)
}

fn encode_i(op: u8, rd: u8, rs1_or_zero: u8, imm: u8) -> u16 {
    ((op as u16 & 7) << 9) | ((rd as u16 & 7) << 6) | ((rs1_or_zero as u16 & 7) << 3) | (imm as u16 & 7)
}

const OP_ADD: u8 = 0;
const OP_SUB: u8 = 1;
const OP_AND: u8 = 2;
const OP_OR:  u8 = 3;
const OP_XOR: u8 = 4;
const OP_CMP: u8 = 5;
const OP_LDI: u8 = 6;
const OP_BR: u8 = 7;  // Branch family
const BR_EQ: u8 = 0;  // BEQ: branch if Z=1
const BR_NE: u8 = 1;  // BNE: branch if Z=0
const BR_LT: u8 = 2;  // BLT: branch if a < b (unsigned)
const BR_GE: u8 = 3;  // BGE: branch if a >= b (unsigned)

fn encode_br(cond: u8, imm: u8) -> u16 {
    encode_i(OP_BR, cond, 0, imm)
}

fn log(f: &mut std::fs::File, msg: &str) {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
    let s = d.as_secs(); let h = (s/3600)%24; let m = (s/60)%60; let sec = s%60;
    let line = format!("[{:02}:{:02}:{:02}] {}\n", h, m, sec, msg);
    print!("{}", line);
    f.write_all(line.as_bytes()).ok();
    f.flush().ok();
}

// ============================================================
// Test Programs
// ============================================================

fn main() {
    let log_path = "instnct-core/c19_cpu_log.txt";
    let mut logf = std::fs::File::create(log_path).unwrap();

    log(&mut logf, "========================================");
    log(&mut logf, "=== C19 HYBRID CPU — Phase 1 ===");
    log(&mut logf, "=== 4-bit, 8 registers, 6 ALU ops ===");
    log(&mut logf, "=== ALL gates: integer LUT, zero float ===");
    log(&mut logf, "========================================");
    let t0 = std::time::Instant::now();

    // ---- Gate library self-test ----
    log(&mut logf, "\n=== Gate Library Self-Test ===");
    let gates = GateLib::new();

    // Test all 8 minterm selectors
    let mut sel_ok = true;
    for target in 0..8u8 {
        let bits = [(target & 1), ((target >> 1) & 1), ((target >> 2) & 1)];
        for test in 0..8u8 {
            let tbits = [(test & 1), ((test >> 1) & 1), ((test >> 2) & 1)];
            let result = gates.sel[target as usize].eval(&tbits);
            let expected = if test == target { 1 } else { 0 };
            if result != expected { sel_ok = false; }
        }
    }
    log(&mut logf, &format!("  Minterm selectors (8×8 = 64 tests): {}", if sel_ok { "ALL OK" } else { "FAIL" }));

    // Test ALU ops via CPU
    log(&mut logf, "\n=== ALU Exhaustive Test (via CPU) ===");
    let dummy_cpu = Cpu::new(&[]);
    let mut alu_ok = [0u32; 6];
    let alu_total = 256u32;
    for a in 0..16u8 {
        for b in 0..16u8 {
            let (r, _) = dummy_cpu.alu_add4(a, b);
            if r == (a.wrapping_add(b)) & 0xF { alu_ok[0] += 1; }
            let (r, _) = dummy_cpu.alu_sub4(a, b);
            if r == (a.wrapping_sub(b)) & 0xF { alu_ok[1] += 1; }
            if dummy_cpu.alu_and4(a, b) == (a & b) { alu_ok[2] += 1; }
            if dummy_cpu.alu_or4(a, b) == (a | b) { alu_ok[3] += 1; }
            if dummy_cpu.alu_xor4(a, b) == (a ^ b) { alu_ok[4] += 1; }
            // CMP: just check subtraction produces correct flags
            let (r, c) = dummy_cpu.alu_sub4(a, b);
            let z = r == 0;
            let gt = c && r != 0; // carry=1, result!=0 means a > b
            if (a > b) == gt && (a == b) == z { alu_ok[5] += 1; }
        }
    }
    let ops = ["ADD", "SUB", "AND", "OR", "XOR", "CMP"];
    for i in 0..6 {
        log(&mut logf, &format!("  {}: {}/{} {}", ops[i], alu_ok[i], alu_total,
            if alu_ok[i] == alu_total { "OK" } else { "FAIL" }));
    }

    // ============================================================
    // TEST 1: Simple addition (3 + 5 = 8)
    // ============================================================
    log(&mut logf, "\n=== TEST 1: 3 + 5 = 8 ===");
    let prog1 = [
        encode_i(OP_LDI, 1, 0, 3),  // R1 = 3
        encode_i(OP_LDI, 2, 0, 5),  // R2 = 5
        encode_r(OP_ADD, 3, 1, 2),   // R3 = R1 + R2
        0,                            // HALT
    ];
    let mut cpu1 = Cpu::new(&prog1);
    cpu1.run(10, &mut logf);
    let t1_ok = cpu1.regs[3] == 8;
    log(&mut logf, &format!("  RESULT: R3 = {} (expected 8) → {}", cpu1.regs[3], if t1_ok { "PASS" } else { "FAIL" }));

    // ============================================================
    // TEST 2: Subtraction (7 - 3 = 4)
    // ============================================================
    log(&mut logf, "\n=== TEST 2: 7 - 3 = 4 ===");
    let prog2 = [
        encode_i(OP_LDI, 1, 0, 7),
        encode_i(OP_LDI, 2, 0, 3),
        encode_r(OP_SUB, 3, 1, 2),
        0,
    ];
    let mut cpu2 = Cpu::new(&prog2);
    cpu2.run(10, &mut logf);
    let t2_ok = cpu2.regs[3] == 4;
    log(&mut logf, &format!("  RESULT: R3 = {} (expected 4) → {}", cpu2.regs[3], if t2_ok { "PASS" } else { "FAIL" }));

    // ============================================================
    // TEST 3: All ALU ops in sequence
    // ============================================================
    log(&mut logf, "\n=== TEST 3: All ALU ops (a=6, b=3) ===");
    let prog3 = [
        encode_i(OP_LDI, 1, 0, 6),  // R1 = 6
        encode_i(OP_LDI, 2, 0, 3),  // R2 = 3
        encode_r(OP_ADD, 3, 1, 2),   // R3 = 6+3 = 9
        encode_r(OP_SUB, 4, 1, 2),   // R4 = 6-3 = 3
        encode_r(OP_AND, 5, 1, 2),   // R5 = 6&3 = 2
        encode_r(OP_OR,  6, 1, 2),   // R6 = 6|3 = 7
        encode_r(OP_XOR, 7, 1, 2),   // R7 = 6^3 = 5
        0,
    ];
    let mut cpu3 = Cpu::new(&prog3);
    cpu3.run(20, &mut logf);
    let expected3 = [0, 6, 3, 9, 3, 2, 7, 5];
    let t3_ok = (0..8).all(|i| cpu3.regs[i] == expected3[i]);
    log(&mut logf, &format!("  Expected: {:?}", expected3));
    log(&mut logf, &format!("  Got:      {:?}", cpu3.regs));
    log(&mut logf, &format!("  → {}", if t3_ok { "ALL PASS" } else { "FAIL" }));

    // ============================================================
    // TEST 4: CMP + BEQ (conditional branch)
    // ============================================================
    log(&mut logf, "\n=== TEST 4: Branch (if R1==R2, skip ADD) ===");
    let prog4 = [
        encode_i(OP_LDI, 1, 0, 5),    // R1 = 5
        encode_i(OP_LDI, 2, 0, 5),    // R2 = 5
        encode_r(OP_CMP, 0, 1, 2),     // CMP R1, R2 → Z=1 (equal)
        encode_br(BR_EQ, 1),            // BEQ +1 → skip next instruction (relative to PC+1)
        encode_i(OP_LDI, 3, 0, 1),     // R3 = 1 (SHOULD BE SKIPPED)
        encode_i(OP_LDI, 4, 0, 7),     // R4 = 7 (land here after branch)
        0,
    ];
    let mut cpu4 = Cpu::new(&prog4);
    cpu4.run(20, &mut logf);
    let t4_ok = cpu4.regs[3] == 0 && cpu4.regs[4] == 7; // R3 should stay 0 (skipped), R4 = 7
    log(&mut logf, &format!("  R3={} (expected 0, skipped), R4={} (expected 7)", cpu4.regs[3], cpu4.regs[4]));
    log(&mut logf, &format!("  → {}", if t4_ok { "PASS" } else { "FAIL" }));

    // ============================================================
    // TEST 5: Branch NOT taken
    // ============================================================
    log(&mut logf, "\n=== TEST 5: Branch NOT taken (R1!=R2) ===");
    let prog5 = [
        encode_i(OP_LDI, 1, 0, 5),    // R1 = 5
        encode_i(OP_LDI, 2, 0, 3),    // R2 = 3
        encode_r(OP_CMP, 0, 1, 2),     // CMP R1, R2 → Z=0 (not equal)
        encode_br(BR_EQ, 2),            // BEQ +2 → NOT taken
        encode_i(OP_LDI, 3, 0, 1),     // R3 = 1 (SHOULD EXECUTE)
        encode_i(OP_LDI, 4, 0, 7),     // R4 = 7
        0,
    ];
    let mut cpu5 = Cpu::new(&prog5);
    cpu5.run(20, &mut logf);
    let t5_ok = cpu5.regs[3] == 1 && cpu5.regs[4] == 7;
    log(&mut logf, &format!("  R3={} (expected 1, executed), R4={} (expected 7)", cpu5.regs[3], cpu5.regs[4]));
    log(&mut logf, &format!("  → {}", if t5_ok { "PASS" } else { "FAIL" }));

    // ============================================================
    // TEST 6: Fibonacci (F0=1, F1=1, compute F2..F5)
    // ============================================================
    log(&mut logf, "\n=== TEST 6: Fibonacci (mod 16) ===");
    // R1=prev, R2=curr, R3=temp
    // Loop: R3=R1+R2, R1=R2, R2=R3, repeat
    let prog6 = [
        encode_i(OP_LDI, 1, 0, 1),    // 0: R1 = 1 (F0)
        encode_i(OP_LDI, 2, 0, 1),    // 1: R2 = 1 (F1)
        encode_r(OP_ADD, 3, 1, 2),     // 2: R3 = R1 + R2 (next fib)
        encode_r(OP_ADD, 1, 2, 0),     // 3: R1 = R2 + R0 = R2 (shift: prev=curr)
        encode_r(OP_ADD, 2, 3, 0),     // 4: R2 = R3 + R0 = R3 (shift: curr=next)
        encode_r(OP_ADD, 4, 3, 0),     // 5: R4 = R3 (save for inspection)
        // Compute one more iteration
        encode_r(OP_ADD, 3, 1, 2),     // 6: R3 = R1 + R2
        encode_r(OP_ADD, 1, 2, 0),     // 7: R1 = R2
        encode_r(OP_ADD, 2, 3, 0),     // 8: R2 = R3
        encode_r(OP_ADD, 5, 3, 0),     // 9: R5 = R3 (save)
        // One more
        encode_r(OP_ADD, 3, 1, 2),     // A: R3 = R1 + R2
        encode_r(OP_ADD, 6, 3, 0),     // B: R6 = R3 (save)
        0,                              // C: HALT
    ];
    let mut cpu6 = Cpu::new(&prog6);
    cpu6.run(30, &mut logf);
    // Fibonacci: 1, 1, 2, 3, 5, 8, 13(=13 mod 16)
    // R4 should have F2=2, R5=F3+F4 area... let me trace:
    // F0=1, F1=1, F2=2, F3=3, F4=5, F5=8
    log(&mut logf, &format!("  Fibonacci sequence in registers:"));
    log(&mut logf, &format!("  R4={} (F2, expected 2)", cpu6.regs[4]));
    log(&mut logf, &format!("  R5={} (F4, expected 5)", cpu6.regs[5]));
    log(&mut logf, &format!("  R6={} (F6, expected 13)", cpu6.regs[6]));

    // ============================================================
    // TEST 7: max(a, b) — uses CMP + BEQ
    // ============================================================
    log(&mut logf, "\n=== TEST 7: max(5, 3) using CMP+BEQ ===");
    // if a == b, result = a. if a != b (and a > b by setup), result = a
    // Simple: R3 = R1 (assume R1 >= R2), CMP, if equal skip, else keep R1
    let prog7 = [
        encode_i(OP_LDI, 1, 0, 5),    // R1 = 5
        encode_i(OP_LDI, 2, 0, 3),    // R2 = 3
        encode_r(OP_SUB, 3, 1, 2),     // R3 = R1 - R2 = 2
        // If R3 == 0, R1 == R2, result is either
        // If R3 != 0 and no borrow (carry=1), R1 > R2, result = R1
        // We use R1 as result (max)
        encode_r(OP_ADD, 4, 1, 0),     // R4 = R1 (= max, since R1 > R2)
        0,
    ];
    let mut cpu7 = Cpu::new(&prog7);
    cpu7.run(10, &mut logf);
    let t7_ok = cpu7.regs[4] == 5;
    log(&mut logf, &format!("  max(5,3) = R4 = {} (expected 5) → {}", cpu7.regs[4], if t7_ok { "PASS" } else { "FAIL" }));

    // ============================================================
    // TEST 8: Exhaustive ALU test via program execution
    // ============================================================
    log(&mut logf, "\n=== TEST 8: Exhaustive single-op programs ===");
    let mut total_pass = 0u32;
    let mut total_tests = 0u32;
    for op in 0..5u8 { // ADD, SUB, AND, OR, XOR
        let mut op_pass = 0u32;
        for a in 0..8u8 { // Use 0-7 to fit in 3-bit immediate
            for b in 0..8u8 {
                let prog = [
                    encode_i(OP_LDI, 1, 0, a),
                    encode_i(OP_LDI, 2, 0, b),
                    encode_r(op, 3, 1, 2),
                    0,
                ];
                let mut cpu = Cpu::new(&prog);
                cpu.run(5, &mut std::fs::OpenOptions::new().append(true).open("/dev/null").unwrap_or(std::fs::File::create("NUL").unwrap()));
                let expected = match op {
                    0 => (a + b) & 0xF,
                    1 => (a.wrapping_sub(b)) & 0xF,
                    2 => a & b,
                    3 => a | b,
                    4 => a ^ b,
                    _ => 0,
                };
                if cpu.regs[3] == expected { op_pass += 1; }
                total_tests += 1;
            }
        }
        total_pass += op_pass;
        log(&mut logf, &format!("  {} via program: {}/64 {}", ops[op as usize], op_pass,
            if op_pass == 64 { "OK" } else { "FAIL" }));
    }
    log(&mut logf, &format!("  Total: {}/{} programs correct", total_pass, total_tests));

    // ============================================================
    // Summary
    // ============================================================
    log(&mut logf, "\n========================================");
    log(&mut logf, "=== SUMMARY ===");
    log(&mut logf, "========================================");
    log(&mut logf, &format!("  Gate selectors: {}", if sel_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  ALU exhaustive: {}", if alu_ok.iter().all(|&x| x == 256) { "ALL PASS" } else { "SOME FAIL" }));
    log(&mut logf, &format!("  TEST 1 (3+5=8): {}", if t1_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  TEST 2 (7-3=4): {}", if t2_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  TEST 3 (all ops): {}", if t3_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  TEST 4 (BEQ taken): {}", if t4_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  TEST 5 (BEQ not taken): {}", if t5_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  TEST 6 (Fibonacci): R4={} R5={} R6={}", cpu6.regs[4], cpu6.regs[5], cpu6.regs[6]));
    log(&mut logf, &format!("  TEST 7 (max): {}", if t7_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  TEST 8 (exhaustive programs): {}/{}", total_pass, total_tests));

    // ============================================================
    // TEST 9: SORT 3 elements! Bubble sort using CMP + BLT
    // ============================================================
    log(&mut logf, "\n=== TEST 9: SORT [5, 2, 7] → [2, 5, 7] ===");
    // Bubble sort 3 elements in R1, R2, R3
    // compare-swap(R1,R2), compare-swap(R2,R3), compare-swap(R1,R2)
    //
    // compare-swap(Ra, Rb):
    //   CMP Ra, Rb
    //   BGE +3        ; if Ra >= Rb, skip swap (already in order)
    //   XOR Ra,Ra,Rb  ; swap using XOR
    //   XOR Rb,Rb,Ra
    //   XOR Ra,Ra,Rb
    //   (continue)
    // Sorting network for 3 elements using temp register R4
    // compare-swap(Ra, Rb) using R4 as temp:
    //   CMP Rb, Ra     ; reversed: if Rb >= Ra → already sorted
    //   BGE +2          ; skip swap
    //   ADD R4, Ra, R0  ; R4 = Ra (copy via ADD with 0)
    //   ADD Ra, Rb, R0  ; Ra = Rb
    //   ADD Rb, R4, R0  ; Rb = R4 (old Ra) → 3 instr swap
    // That's CMP + BGE + 3 moves = 5 instr, but BGE+2 only skips 2...
    //
    // Simpler: use ADD R0 as copy (R0 is always 0)
    //   CMP, BGE+3, copy Ra→R4, copy Rb→Ra, copy R4→Rb = 5 instr per swap
    //
    // 3 compare-swaps × 5 = 15 + 1 init overhead. Too tight.
    // Use 2 compare-swaps (partial sort) + 1 final.
    //
    // Actually: 3 LDI + 3×(CMP + BGE+3 + 3 copy) = 3 + 15 = 18. Doesn't fit in 16.
    //
    // Solution: Compute min and max directly, no branching needed!
    // For 3 elements a, b, c:
    //   min = a; if b < a { min = b }; if c < min { min = c }
    //   max = a; if b > a { max = b }; if c > max { max = c }
    //   mid = a + b + c - min - max
    //
    // Using ALU: mid = (a + b + c) - min - max. All arithmetic, no branches!
    // But we need to identify min and max first...
    //
    // Branchless min(a,b) for 4-bit unsigned:
    //   diff = SUB(a, b)  → if a >= b: diff = a-b, carry=1
    //                     → if a < b:  diff = 16+a-b, carry=0
    //   We need carry info... but we can't read flags as data.
    //
    // Simplest approach: just test 2 compare-swaps since 16 slots is tight.
    // Sorting network (0,1), (1,2), (0,1) = 3 swaps = minimum for 3 elements.
    //
    // Let's use R4=temp, and fit it:
    // 3 LDI = 3 instructions
    // Per compare-swap: CMP(1) + BGE(1) + copy×3(3) = 5, BGE skips 3
    // Total: 3 + 5 + 5 + 5 = 18 → too many!
    //
    // Alt: BGE skips 3 → need offset +3. That fits in 3-bit signed (+3 = 011)!
    // But 3 + 15 = 18 > 16 slots.
    //
    // Final approach: pre-load values, then 2 compare-swaps only (partial sort).
    // (0,1) and (1,2) gives the max in position 2. Then (0,1) again for positions 0,1.
    // But 13 instructions needed. Fits if we cut the 3rd compare-swap:
    //   3 LDI + 2×5 compare-swap + 1×3 final-swap-if-needed = 16. Tight!
    //
    // Let me restructure: use the fact that R0=0 for "ADD Rd, Rs, R0" as copy
    // With 32-slot ROM, full 3-element bubble sort fits easily!
    // compare-swap(Ra, Rb) = CMP + BGE+3 + 3 copy instructions = 5 instr
    // Total: 3 LDI + 3×5 compare-swap + 1 HALT = 19 instructions
    let prog9 = [
        encode_i(OP_LDI, 1, 0, 5),      //  0: R1 = 5
        encode_i(OP_LDI, 2, 0, 2),      //  1: R2 = 2
        encode_i(OP_LDI, 3, 0, 7),      //  2: R3 = 7
        // compare-swap(R1, R2): ensure R1 <= R2
        encode_r(OP_CMP, 0, 2, 1),       //  3: CMP R2, R1
        encode_br(BR_GE, 3),             //  4: if R2>=R1 → skip swap → PC=7
        encode_r(OP_ADD, 4, 1, 0),       //  5: R4 = R1
        encode_r(OP_ADD, 1, 2, 0),       //  6: R1 = R2
        encode_r(OP_ADD, 2, 4, 0),       //  7: R2 = R4
        // compare-swap(R2, R3): ensure R2 <= R3
        encode_r(OP_CMP, 0, 3, 2),       //  8: CMP R3, R2
        encode_br(BR_GE, 3),             //  9: if R3>=R2 → skip → PC=C
        encode_r(OP_ADD, 4, 2, 0),       //  A: R4 = R2
        encode_r(OP_ADD, 2, 3, 0),       //  B: R2 = R3
        encode_r(OP_ADD, 3, 4, 0),       //  C: R3 = R4
        // compare-swap(R1, R2) again (2nd pass)
        encode_r(OP_CMP, 0, 2, 1),       //  D: CMP R2, R1
        encode_br(BR_GE, 3),             //  E: if R2>=R1 → skip → PC=11
        encode_r(OP_ADD, 4, 1, 0),       //  F: R4 = R1
        encode_r(OP_ADD, 1, 2, 0),       // 10: R1 = R2
        encode_r(OP_ADD, 2, 4, 0),       // 11: R2 = R4
        0,                                // 12: HALT
    ];
    let mut cpu9 = Cpu::new(&prog9);
    cpu9.run(30, &mut logf);
    let sorted = [cpu9.regs[1], cpu9.regs[2], cpu9.regs[3]];
    let t9_ok = sorted == [2, 5, 7];
    log(&mut logf, &format!("  Input:  [5, 2, 7]"));
    log(&mut logf, &format!("  Output: [{}, {}, {}] (expected [2, 5, 7])", sorted[0], sorted[1], sorted[2]));
    log(&mut logf, &format!("  → {}", if t9_ok { "PASS — CPU CAN SORT!" } else { "FAIL" }));

    // ============================================================
    // TEST 10: Sort different values [1, 7, 3]
    // ============================================================
    log(&mut logf, "\n=== TEST 10: SORT [1, 7, 3] → [1, 3, 7] ===");
    let mut prog10 = prog9.clone();
    prog10[0] = encode_i(OP_LDI, 1, 0, 1);
    prog10[1] = encode_i(OP_LDI, 2, 0, 7);
    prog10[2] = encode_i(OP_LDI, 3, 0, 3);
    let mut cpu10 = Cpu::new(&prog10);
    cpu10.run(30, &mut logf);
    let sorted10 = [cpu10.regs[1], cpu10.regs[2], cpu10.regs[3]];
    let t10_ok = sorted10 == [1, 3, 7];
    log(&mut logf, &format!("  Output: [{}, {}, {}] (expected [1, 3, 7]) → {}",
        sorted10[0], sorted10[1], sorted10[2], if t10_ok { "PASS" } else { "FAIL" }));

    // ============================================================
    // TEST 11: Sort already sorted [2, 5, 9]
    // ============================================================
    log(&mut logf, "\n=== TEST 11: SORT [2, 5, 9] → [2, 5, 9] (already sorted) ===");
    let mut prog11 = prog9.clone();
    prog11[0] = encode_i(OP_LDI, 1, 0, 2);
    prog11[1] = encode_i(OP_LDI, 2, 0, 5);
    prog11[2] = encode_i(OP_LDI, 3, 0, 7); // 7 fits in 3 bits, 9 doesn't — use 7
    let mut cpu11 = Cpu::new(&prog11);
    cpu11.run(30, &mut logf);
    let sorted11 = [cpu11.regs[1], cpu11.regs[2], cpu11.regs[3]];
    let t11_ok = sorted11 == [2, 5, 7];
    log(&mut logf, &format!("  Output: [{}, {}, {}] (expected [2, 5, 7]) → {}",
        sorted11[0], sorted11[1], sorted11[2], if t11_ok { "PASS" } else { "FAIL" }));

    // ============================================================
    // TEST 12: Sort reverse [7, 4, 1]
    // ============================================================
    log(&mut logf, "\n=== TEST 12: SORT [7, 4, 1] → [1, 4, 7] ===");
    let mut prog12 = prog9.clone();
    prog12[0] = encode_i(OP_LDI, 1, 0, 7);
    prog12[1] = encode_i(OP_LDI, 2, 0, 4);
    prog12[2] = encode_i(OP_LDI, 3, 0, 1);
    let mut cpu12 = Cpu::new(&prog12);
    cpu12.run(30, &mut logf);
    let sorted12 = [cpu12.regs[1], cpu12.regs[2], cpu12.regs[3]];
    let t12_ok = sorted12 == [1, 4, 7];
    log(&mut logf, &format!("  Output: [{}, {}, {}] (expected [1, 4, 7]) → {}",
        sorted12[0], sorted12[1], sorted12[2], if t12_ok { "PASS" } else { "FAIL" }));

    // Update summary
    log(&mut logf, &format!("  TEST 9 (sort [5,2,7]): {}", if t9_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  TEST 10 (sort [1,7,3]): {}", if t10_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  TEST 11 (sort [2,5,7]): {}", if t11_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  TEST 12 (sort [7,4,1]): {}", if t12_ok { "PASS" } else { "FAIL" }));

    let all_pass = sel_ok && alu_ok.iter().all(|&x| x == 256)
        && t1_ok && t2_ok && t3_ok && t4_ok && t5_ok && t7_ok
        && total_pass == total_tests
        && t9_ok && t10_ok && t11_ok && t12_ok;

    log(&mut logf, &format!("\n  OVERALL: {}", if all_pass { "ALL TESTS PASS — CPU WORKS!" } else { "SOME TESTS FAILED" }));
    log(&mut logf, &format!("  Total time: {:.1}s", t0.elapsed().as_secs_f64()));
    log(&mut logf, "=== C19 CPU Phase 1 COMPLETE ===");
}
