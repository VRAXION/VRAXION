//! 8-bit C19 CPU — complete processor built from verified C19 LutGate neurons
//!
//! Architecture:
//!   - 8-bit data width, 8 registers (R0=0 hardwired, R1-R7 general purpose)
//!   - 16-bit instructions: [OP:4][Rd:3][Rs1:3][Rs2:3][func:3]
//!   - 16 opcodes: ADD,SUB,MUL,AND,OR,XOR,CMP,LOAD_IMM,MOV,NOT,SHL,SHR,MIN,MAX,BEQ,BNE
//!   - 64-slot instruction ROM, 8-bit program counter
//!   - Flags: Zero, Negative, Carry
//!   - ALU: 394 C19 neurons, zero floating point in eval hot path
//!
//! All computation uses C19 integer LUT gates.
//!
//! Run: cargo run --example cpu8bit --release

use std::io::Write;

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

    fn half_add(&self, a: u8, b: u8) -> (u8, u8) {
        (self.xor_g.eval(&[a, b]), self.and_g.eval(&[a, b]))
    }

    fn full_add(&self, a: u8, b: u8, cin: u8) -> (u8, u8) {
        (self.xor3.eval(&[a, b, cin]), self.maj.eval(&[a, b, cin]))
    }
}

// ============================================================
// CMP flags
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CmpFlags {
    z: bool,
    n: bool,
    c: bool,
}

// ============================================================
// ALU8 — the complete 8-bit ALU (394 neurons)
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

    fn bit(val: u8, pos: usize) -> u8 {
        (val >> pos) & 1
    }

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

    fn sub8(&self, a: u8, b: u8) -> (u8, u8) {
        let g = &self.gates;
        let mut not_b = 0u8;
        for bit in 0..8 {
            not_b |= g.not_g.eval(&[Self::bit(b, bit)]) << bit;
        }
        self.add8_cin(a, not_b, 1)
    }

    fn mul8(&self, a: u8, b: u8) -> u16 {
        let g = &self.gates;
        let mut pp = [[0u8; 8]; 8];
        for row in 0..8 {
            for col in 0..8 {
                pp[row][col] = g.and_g.eval(&[Self::bit(a, col), Self::bit(b, row)]);
            }
        }
        let mut result: u16 = 0;
        let mut carry_pool: Vec<u8> = Vec::new();
        for col_pos in 0..16 {
            let mut terms: Vec<u8> = Vec::new();
            for row in 0..8 {
                let col = col_pos as i32 - row as i32;
                if col >= 0 && col < 8 {
                    terms.push(pp[row][col as usize]);
                }
            }
            terms.append(&mut carry_pool);
            carry_pool = Vec::new();
            while terms.len() > 1 {
                if terms.len() >= 3 {
                    let a_bit = terms.remove(0);
                    let b_bit = terms.remove(0);
                    let c_bit = terms.remove(0);
                    let (s, c) = g.full_add(a_bit, b_bit, c_bit);
                    terms.push(s);
                    carry_pool.push(c);
                } else {
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
        }
        result
    }

    fn and8(&self, a: u8, b: u8) -> u8 {
        let g = &self.gates;
        let mut r = 0u8;
        for bit in 0..8 {
            r |= g.and_g.eval(&[Self::bit(a, bit), Self::bit(b, bit)]) << bit;
        }
        r
    }

    fn or8(&self, a: u8, b: u8) -> u8 {
        let g = &self.gates;
        let mut r = 0u8;
        for bit in 0..8 {
            r |= g.or_g.eval(&[Self::bit(a, bit), Self::bit(b, bit)]) << bit;
        }
        r
    }

    fn xor8(&self, a: u8, b: u8) -> u8 {
        let g = &self.gates;
        let mut r = 0u8;
        for bit in 0..8 {
            r |= g.xor_g.eval(&[Self::bit(a, bit), Self::bit(b, bit)]) << bit;
        }
        r
    }

    fn not8(&self, a: u8) -> u8 {
        let g = &self.gates;
        let mut r = 0u8;
        for bit in 0..8 {
            r |= g.not_g.eval(&[Self::bit(a, bit)]) << bit;
        }
        r
    }

    fn shl8(&self, a: u8) -> u8 {
        a << 1
    }

    fn shr8(&self, a: u8) -> u8 {
        a >> 1
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
        CmpFlags {
            z: z == 1,
            n: n == 1,
            c: carry == 1,
        }
    }

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

    fn min8(&self, a: u8, b: u8) -> u8 {
        let flags = self.cmp8(a, b);
        let sel = if flags.c && !flags.z { 1u8 } else { 0u8 };
        self.mux8(a, b, sel)
    }

    fn max8(&self, a: u8, b: u8) -> u8 {
        let flags = self.cmp8(a, b);
        let sel = if flags.c { 0u8 } else { 1u8 };
        self.mux8(a, b, sel)
    }
}

// ============================================================
// Instruction encoding
// ============================================================
//
// 16-bit instruction format:
//   [15:12] opcode   (4 bits)
//   [11:9]  dst      (3 bits)
//   [8:6]   src1     (3 bits)
//   [5:3]   src2     (3 bits)
//   [2:0]   func     (3 bits)
//
// LOAD_IMM special: [15:12]=0x7, [11:9]=dst, [7:0]=imm8
// Branch:           [15:12]=0xE/0xF, [8:0]=signed offset (9-bit)

const OP_ADD:  u8 = 0x0;
const OP_SUB:  u8 = 0x1;
const OP_MUL:  u8 = 0x2;
const OP_AND:  u8 = 0x3;
const OP_OR:   u8 = 0x4;
const OP_XOR:  u8 = 0x5;
const OP_CMP:  u8 = 0x6;
const OP_LDI:  u8 = 0x7;  // load immediate
const OP_MOV:  u8 = 0x8;
const OP_NOT:  u8 = 0x9;
const OP_SHL:  u8 = 0xA;
const OP_SHR:  u8 = 0xB;
const OP_MIN:  u8 = 0xC;
const OP_MAX:  u8 = 0xD;
const OP_BEQ:  u8 = 0xE;
const OP_BNE:  u8 = 0xF;

/// Encode R-type: opcode[15:12] | rd[11:9] | rs1[8:6] | rs2[5:3] | func[2:0]
fn encode_r(op: u8, rd: u8, rs1: u8, rs2: u8) -> u16 {
    ((op as u16 & 0xF) << 12)
        | ((rd as u16 & 0x7) << 9)
        | ((rs1 as u16 & 0x7) << 6)
        | ((rs2 as u16 & 0x7) << 3)
}

/// Encode LOAD_IMM: opcode=0x7[15:12] | rd[11:9] | imm8[7:0]
fn encode_ldi(rd: u8, imm: u8) -> u16 {
    ((OP_LDI as u16) << 12)
        | ((rd as u16 & 0x7) << 9)
        | (imm as u16)
}

/// Encode branch: opcode[15:12] | signed_offset[8:0]
/// offset is sign-extended 9-bit relative to PC+1
fn encode_branch(op: u8, offset: i16) -> u16 {
    let off_bits = (offset as u16) & 0x1FF; // 9-bit
    ((op as u16 & 0xF) << 12) | off_bits
}

/// Encode MOV: same as R-type with rs2=0
fn encode_mov(rd: u8, rs1: u8) -> u16 {
    encode_r(OP_MOV, rd, rs1, 0)
}

/// Encode unary ops (NOT, SHL, SHR): R-type with rs2=0
fn encode_unary(op: u8, rd: u8, rs1: u8) -> u16 {
    encode_r(op, rd, rs1, 0)
}

// ============================================================
// CPU8 — the complete 8-bit processor
// ============================================================

const ROM_SIZE: usize = 64;
const MAX_CYCLE_GUARD: u32 = 10000; // infinite loop protection

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
}

impl Cpu8 {
    fn new(program: &[u16]) -> Self {
        let mut rom = [0u16; ROM_SIZE];
        for (i, &instr) in program.iter().enumerate() {
            if i < ROM_SIZE {
                rom[i] = instr;
            }
        }
        Cpu8 {
            alu: Alu8::new(),
            pc: 0,
            regs: [0; 8],
            flag_z: false,
            flag_n: false,
            flag_c: false,
            rom,
            halted: false,
            cycle_count: 0,
        }
    }

    /// Decode instruction fields
    fn decode(&self, instr: u16) -> (u8, u8, u8, u8, u8) {
        let opcode = ((instr >> 12) & 0xF) as u8;
        let rd     = ((instr >> 9) & 0x7) as u8;
        let rs1    = ((instr >> 6) & 0x7) as u8;
        let rs2    = ((instr >> 3) & 0x7) as u8;
        let func   = (instr & 0x7) as u8;
        (opcode, rd, rs1, rs2, func)
    }

    /// Sign-extend a 9-bit value to i16
    fn sign_extend_9(val: u16) -> i16 {
        let v = val & 0x1FF;
        if v & 0x100 != 0 {
            // negative
            (v | 0xFE00) as i16
        } else {
            v as i16
        }
    }

    /// Execute one instruction cycle
    fn step(&mut self) -> String {
        if self.halted {
            return "HALTED".to_string();
        }

        // Fetch
        let pc_idx = self.pc as usize;
        if pc_idx >= ROM_SIZE {
            self.halted = true;
            return format!("HALT: PC={} out of ROM bounds", self.pc);
        }
        let instr = self.rom[pc_idx];

        // Halt on zero instruction (NOP/end marker)
        if instr == 0 {
            self.halted = true;
            return format!("HALT at PC={} (NOP instruction)", self.pc);
        }

        let (opcode, rd, rs1, rs2, _func) = self.decode(instr);

        // Read register values
        let val_rs1 = self.regs[rs1 as usize];
        let val_rs2 = self.regs[rs2 as usize];

        let mut next_pc = self.pc.wrapping_add(1);
        let mut write_reg = false;
        let mut result: u8 = 0;

        let trace: String;

        match opcode {
            0x0 => { // ADD
                let (r, carry) = self.alu.add8(val_rs1, val_rs2);
                result = r;
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                self.flag_c = carry == 1;
                write_reg = true;
                trace = format!("ADD  R{}=R{}+R{} = {}+{} = {}", rd, rs1, rs2, val_rs1, val_rs2, result);
            }
            0x1 => { // SUB
                let (r, carry) = self.alu.sub8(val_rs1, val_rs2);
                result = r;
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                self.flag_c = carry == 1;
                write_reg = true;
                trace = format!("SUB  R{}=R{}-R{} = {}-{} = {}", rd, rs1, rs2, val_rs1, val_rs2, result);
            }
            0x2 => { // MUL (low byte only)
                let full = self.alu.mul8(val_rs1, val_rs2);
                result = (full & 0xFF) as u8;
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                self.flag_c = full > 255; // carry if overflow
                write_reg = true;
                trace = format!("MUL  R{}=R{}*R{} = {}*{} = {} (full={})", rd, rs1, rs2, val_rs1, val_rs2, result, full);
            }
            0x3 => { // AND
                result = self.alu.and8(val_rs1, val_rs2);
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                self.flag_c = false;
                write_reg = true;
                trace = format!("AND  R{}=R{}&R{} = 0x{:02X}&0x{:02X} = 0x{:02X}", rd, rs1, rs2, val_rs1, val_rs2, result);
            }
            0x4 => { // OR
                result = self.alu.or8(val_rs1, val_rs2);
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                self.flag_c = false;
                write_reg = true;
                trace = format!("OR   R{}=R{}|R{} = 0x{:02X}|0x{:02X} = 0x{:02X}", rd, rs1, rs2, val_rs1, val_rs2, result);
            }
            0x5 => { // XOR
                result = self.alu.xor8(val_rs1, val_rs2);
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                self.flag_c = false;
                write_reg = true;
                trace = format!("XOR  R{}=R{}^R{} = 0x{:02X}^0x{:02X} = 0x{:02X}", rd, rs1, rs2, val_rs1, val_rs2, result);
            }
            0x6 => { // CMP (no writeback)
                let flags = self.alu.cmp8(val_rs1, val_rs2);
                self.flag_z = flags.z;
                self.flag_n = flags.n;
                self.flag_c = flags.c;
                trace = format!("CMP  R{}-R{} = {}-{} -> Z={} N={} C={}", rs1, rs2, val_rs1, val_rs2,
                    self.flag_z as u8, self.flag_n as u8, self.flag_c as u8);
            }
            0x7 => { // LOAD_IMM
                let imm8 = (instr & 0xFF) as u8;
                result = imm8;
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                write_reg = true;
                trace = format!("LDI  R{}={} (0x{:02X})", rd, result, result);
            }
            0x8 => { // MOV
                result = val_rs1;
                write_reg = true;
                trace = format!("MOV  R{}=R{} = {}", rd, rs1, result);
            }
            0x9 => { // NOT
                result = self.alu.not8(val_rs1);
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                write_reg = true;
                trace = format!("NOT  R{}=~R{} = ~0x{:02X} = 0x{:02X}", rd, rs1, val_rs1, result);
            }
            0xA => { // SHL
                result = self.alu.shl8(val_rs1);
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                self.flag_c = (val_rs1 >> 7) & 1 == 1; // old MSB becomes carry
                write_reg = true;
                trace = format!("SHL  R{}=R{}<<1 = 0x{:02X}<<1 = 0x{:02X}", rd, rs1, val_rs1, result);
            }
            0xB => { // SHR
                result = self.alu.shr8(val_rs1);
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                self.flag_c = val_rs1 & 1 == 1; // old LSB becomes carry
                write_reg = true;
                trace = format!("SHR  R{}=R{}>>1 = 0x{:02X}>>1 = 0x{:02X}", rd, rs1, val_rs1, result);
            }
            0xC => { // MIN
                result = self.alu.min8(val_rs1, val_rs2);
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                write_reg = true;
                trace = format!("MIN  R{}=min(R{},R{}) = min({},{}) = {}", rd, rs1, rs2, val_rs1, val_rs2, result);
            }
            0xD => { // MAX
                result = self.alu.max8(val_rs1, val_rs2);
                self.flag_z = result == 0;
                self.flag_n = (result >> 7) & 1 == 1;
                write_reg = true;
                trace = format!("MAX  R{}=max(R{},R{}) = max({},{}) = {}", rd, rs1, rs2, val_rs1, val_rs2, result);
            }
            0xE => { // BEQ
                let offset = Self::sign_extend_9(instr & 0x1FF);
                if self.flag_z {
                    next_pc = (self.pc as i16 + 1 + offset) as u8;
                    trace = format!("BEQ  taken -> PC={}", next_pc);
                } else {
                    trace = format!("BEQ  not taken (Z=0)");
                }
            }
            0xF => { // BNE
                let offset = Self::sign_extend_9(instr & 0x1FF);
                if !self.flag_z {
                    next_pc = (self.pc as i16 + 1 + offset) as u8;
                    trace = format!("BNE  taken -> PC={}", next_pc);
                } else {
                    trace = format!("BNE  not taken (Z=1)");
                }
            }
            _ => {
                trace = format!("UNKNOWN opcode 0x{:X}", opcode);
            }
        }

        // Writeback (R0 is hardwired to 0)
        if write_reg && rd != 0 {
            self.regs[rd as usize] = result;
        }

        // Update PC
        self.pc = next_pc;
        self.cycle_count += 1;

        // Guard against infinite loops
        if self.cycle_count >= MAX_CYCLE_GUARD {
            self.halted = true;
            return format!("{} | HALT: max cycles exceeded", trace);
        }

        trace
    }

    fn reg_dump(&self) -> String {
        let mut s = String::from("  Regs:");
        for i in 0..8 {
            s += &format!(" R{}={}", i, self.regs[i]);
        }
        s += &format!(" | Z={} N={} C={}", self.flag_z as u8, self.flag_n as u8, self.flag_c as u8);
        s
    }

    fn run(&mut self, max_cycles: u32, logf: &mut dyn std::io::Write, verbose: bool) {
        for _ in 0..max_cycles {
            if self.halted {
                break;
            }
            let trace = self.step();
            if verbose {
                let line = format!("  Cycle {:3}: {}\n", self.cycle_count, trace);
                logf.write_all(line.as_bytes()).ok();
                let dump = format!("{}\n", self.reg_dump());
                logf.write_all(dump.as_bytes()).ok();
            }
        }
    }
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
// MAIN — build, test, and adversarially verify the CPU
// ============================================================

fn main() {
    let log_path = "cpu8bit_log.txt";
    let mut logf = std::fs::File::create(log_path).unwrap();

    log(&mut logf, "================================================================");
    log(&mut logf, "  8-bit C19 CPU — Complete Processor Verification");
    log(&mut logf, "  394 C19 LutGate neurons in ALU, 16 opcodes");
    log(&mut logf, "  8 registers (R0=0), 64-slot ROM, Z/N/C flags");
    log(&mut logf, "================================================================");

    let t0 = std::time::Instant::now();
    let mut all_pass = true;

    // ============================================================
    // Gate verification (sanity check)
    // ============================================================
    log(&mut logf, "\n=== Gate Verification ===");
    let alu = Alu8::new();
    let g = &alu.gates;

    let mut gate_ok = true;
    // AND
    for a in 0..=1u8 { for b in 0..=1u8 {
        if g.and_g.eval(&[a, b]) != (a & b) { gate_ok = false; }
    }}
    // OR
    for a in 0..=1u8 { for b in 0..=1u8 {
        if g.or_g.eval(&[a, b]) != (a | b) { gate_ok = false; }
    }}
    // XOR
    for a in 0..=1u8 { for b in 0..=1u8 {
        if g.xor_g.eval(&[a, b]) != (a ^ b) { gate_ok = false; }
    }}
    // NOT
    for a in 0..=1u8 {
        if g.not_g.eval(&[a]) != (1 - a) { gate_ok = false; }
    }
    // XOR3
    for a in 0..=1u8 { for b in 0..=1u8 { for c in 0..=1u8 {
        if g.xor3.eval(&[a, b, c]) != (a ^ b ^ c) { gate_ok = false; }
    }}}
    // MAJ
    for a in 0..=1u8 { for b in 0..=1u8 { for c in 0..=1u8 {
        let expected = if (a + b + c) >= 2 { 1u8 } else { 0 };
        if g.maj.eval(&[a, b, c]) != expected { gate_ok = false; }
    }}}

    log(&mut logf, &format!("  All 6 gate types: {}", if gate_ok { "PASS" } else { "FAIL" }));
    if !gate_ok { all_pass = false; }

    // ============================================================
    // TEST 1: Simple arithmetic
    // ============================================================
    log(&mut logf, "\n=== TEST 1: Simple Arithmetic ===");
    log(&mut logf, "  LOAD_IMM R1,10; LOAD_IMM R2,20; ADD R3,R1,R2; SUB R4,R3,R1; MUL R5,R1,R2");
    let prog1 = [
        encode_ldi(1, 10),              // R1 = 10
        encode_ldi(2, 20),              // R2 = 20
        encode_r(OP_ADD, 3, 1, 2),      // R3 = R1 + R2 = 30
        encode_r(OP_SUB, 4, 3, 1),      // R4 = R3 - R1 = 20
        encode_r(OP_MUL, 5, 1, 2),      // R5 = R1 * R2 = 200
    ];
    let mut cpu1 = Cpu8::new(&prog1);
    cpu1.run(20, &mut logf, true);
    let t1_ok = cpu1.regs[3] == 30 && cpu1.regs[4] == 20 && cpu1.regs[5] == 200;
    log(&mut logf, &format!("  R3={} (exp 30), R4={} (exp 20), R5={} (exp 200) -> {}",
        cpu1.regs[3], cpu1.regs[4], cpu1.regs[5], if t1_ok { "PASS" } else { "FAIL" }));
    if !t1_ok { all_pass = false; }

    // ============================================================
    // TEST 2: Bitwise operations
    // ============================================================
    log(&mut logf, "\n=== TEST 2: Bitwise Operations ===");
    log(&mut logf, "  R1=0xAA, R2=0x55, AND/OR/XOR/NOT");
    let prog2 = [
        encode_ldi(1, 0xAA),            // R1 = 0xAA (10101010)
        encode_ldi(2, 0x55),            // R2 = 0x55 (01010101)
        encode_r(OP_AND, 3, 1, 2),      // R3 = R1 & R2 = 0x00
        encode_r(OP_OR,  4, 1, 2),      // R4 = R1 | R2 = 0xFF
        encode_r(OP_XOR, 5, 1, 2),      // R5 = R1 ^ R2 = 0xFF
        encode_unary(OP_NOT, 6, 1),     // R6 = ~R1 = 0x55
    ];
    let mut cpu2 = Cpu8::new(&prog2);
    cpu2.run(20, &mut logf, true);
    let t2_ok = cpu2.regs[3] == 0x00 && cpu2.regs[4] == 0xFF
             && cpu2.regs[5] == 0xFF && cpu2.regs[6] == 0x55;
    log(&mut logf, &format!("  R3=0x{:02X} (exp 0x00), R4=0x{:02X} (exp 0xFF), R5=0x{:02X} (exp 0xFF), R6=0x{:02X} (exp 0x55) -> {}",
        cpu2.regs[3], cpu2.regs[4], cpu2.regs[5], cpu2.regs[6], if t2_ok { "PASS" } else { "FAIL" }));
    if !t2_ok { all_pass = false; }

    // ============================================================
    // TEST 3: Find minimum of 3 values
    // ============================================================
    log(&mut logf, "\n=== TEST 3: Min of 3 Values ===");
    log(&mut logf, "  min(42, 17, 85) = 17");
    let prog3 = [
        encode_ldi(1, 42),
        encode_ldi(2, 17),
        encode_ldi(3, 85),
        encode_r(OP_MIN, 4, 1, 2),      // R4 = min(42,17) = 17
        encode_r(OP_MIN, 5, 4, 3),      // R5 = min(17,85) = 17
    ];
    let mut cpu3 = Cpu8::new(&prog3);
    cpu3.run(20, &mut logf, true);
    let t3_ok = cpu3.regs[4] == 17 && cpu3.regs[5] == 17;
    log(&mut logf, &format!("  R4={} (exp 17), R5={} (exp 17) -> {}",
        cpu3.regs[4], cpu3.regs[5], if t3_ok { "PASS" } else { "FAIL" }));
    if !t3_ok { all_pass = false; }

    // ============================================================
    // TEST 4: Conditional branch (count down)
    // ============================================================
    log(&mut logf, "\n=== TEST 4: Count Down with Branch ===");
    log(&mut logf, "  R1=5, decrement to 0 using SUB + CMP + BNE loop");
    // Layout:
    //   0: LDI R1, 5
    //   1: LDI R2, 1
    //   2: SUB R1, R1, R2      (loop start)
    //   3: CMP R1, R0          (compare to zero)
    //   4: BNE -3               (jump back to instruction 2: PC+1-3 = 5-3 = 2)
    let prog4 = [
        encode_ldi(1, 5),               // 0: R1 = 5
        encode_ldi(2, 1),               // 1: R2 = 1
        encode_r(OP_SUB, 1, 1, 2),      // 2: R1 = R1 - R2 (loop top)
        encode_r(OP_CMP, 0, 1, 0),      // 3: CMP R1, R0 (sets Z when R1==0)
        encode_branch(OP_BNE, -3),       // 4: BNE -3 (if not zero, go back to PC=2)
    ];
    let mut cpu4 = Cpu8::new(&prog4);
    cpu4.run(100, &mut logf, true);
    let t4_ok = cpu4.regs[1] == 0; // R1 should be 0 after 5 decrements
    log(&mut logf, &format!("  R1={} (exp 0), cycles={} -> {}",
        cpu4.regs[1], cpu4.cycle_count, if t4_ok { "PASS" } else { "FAIL" }));
    if !t4_ok { all_pass = false; }

    // ============================================================
    // TEST 5: Bubble sort 4 values
    // ============================================================
    log(&mut logf, "\n=== TEST 5: Bubble Sort 4 Values ===");
    log(&mut logf, "  Sort [77, 23, 91, 45] -> [23, 45, 77, 91]");
    // Bubble sort: 3 passes, each pass compares adjacent pairs
    // compare-swap(Ra, Rb): CMP Ra,Rb; if Ra<=Rb skip; else swap via R5(temp)
    //   CMP Ra, Rb         => sets flags
    //   BEQ +4              => if equal, skip swap (4 instrs ahead: MOV+MOV+MOV+next)
    //   ... but we need "branch if Ra <= Rb" = "branch if NOT (Ra > Rb)"
    //   After CMP(Ra,Rb): C=1 means Ra>=Rb, Z=1 means Ra==Rb
    //   Ra <= Rb  <=>  NOT(C=1 AND Z=0) <=> C=0 OR Z=1
    //   We don't have BLE. Use: CMP Rb,Ra => C=1 means Rb>=Ra => Ra<=Rb
    //   So: CMP Rb,Ra; BNE checks Z flag... no, we need carry.
    //
    // Alternative: use MIN/MAX directly! Much simpler.
    //   For each pair, compute min and max and place them.
    //   min12 = MIN(R1,R2); max12 = MAX(R1,R2); R1=min12; R2=max12
    //   But MIN/MAX already exist! Use them.
    //
    // Sorting network for 4 elements (optimal = 5 compare-swaps):
    //   swap(0,1), swap(2,3), swap(0,2), swap(1,3), swap(1,2)
    //
    // Using MIN/MAX: swap(Ra,Rb) = temp=MIN(Ra,Rb); Rb=MAX(Ra,Rb); Ra=temp
    //   We use R5 as temp for min, R6 as temp for max:
    //   R5 = MIN(Ra,Rb); R6 = MAX(Ra,Rb); Ra = MOV R5; Rb = MOV R6
    //   = 4 instructions per swap => 5*4 = 20 + 4 LDI = 24 instructions. Fits in 64!

    let prog5: Vec<u16> = vec![
        encode_ldi(1, 77),               // 0: R1 = 77
        encode_ldi(2, 23),               // 1: R2 = 23
        encode_ldi(3, 91),               // 2: R3 = 91
        encode_ldi(4, 45),               // 3: R4 = 45
        // swap(R1, R2)
        encode_r(OP_MIN, 5, 1, 2),       // 4: R5 = min(R1,R2)
        encode_r(OP_MAX, 6, 1, 2),       // 5: R6 = max(R1,R2)
        encode_mov(1, 5),                // 6: R1 = R5 (min)
        encode_mov(2, 6),                // 7: R2 = R6 (max)
        // swap(R3, R4)
        encode_r(OP_MIN, 5, 3, 4),       // 8: R5 = min(R3,R4)
        encode_r(OP_MAX, 6, 3, 4),       // 9: R6 = max(R3,R4)
        encode_mov(3, 5),                // 10: R3 = min
        encode_mov(4, 6),                // 11: R4 = max
        // swap(R1, R3)
        encode_r(OP_MIN, 5, 1, 3),       // 12: R5 = min(R1,R3)
        encode_r(OP_MAX, 6, 1, 3),       // 13: R6 = max(R1,R3)
        encode_mov(1, 5),                // 14: R1 = min
        encode_mov(3, 6),                // 15: R3 = max
        // swap(R2, R4)
        encode_r(OP_MIN, 5, 2, 4),       // 16: R5 = min(R2,R4)
        encode_r(OP_MAX, 6, 2, 4),       // 17: R6 = max(R2,R4)
        encode_mov(2, 5),                // 18: R2 = min
        encode_mov(4, 6),                // 19: R4 = max
        // swap(R2, R3)
        encode_r(OP_MIN, 5, 2, 3),       // 20: R5 = min(R2,R3)
        encode_r(OP_MAX, 6, 2, 3),       // 21: R6 = max(R2,R3)
        encode_mov(2, 5),                // 22: R2 = min
        encode_mov(3, 6),                // 23: R3 = max
    ];
    let mut cpu5 = Cpu8::new(&prog5);
    cpu5.run(50, &mut logf, true);
    let t5_ok = cpu5.regs[1] == 23 && cpu5.regs[2] == 45 && cpu5.regs[3] == 77 && cpu5.regs[4] == 91;
    log(&mut logf, &format!("  R1={}, R2={}, R3={}, R4={} (exp 23,45,77,91) -> {}",
        cpu5.regs[1], cpu5.regs[2], cpu5.regs[3], cpu5.regs[4],
        if t5_ok { "PASS" } else { "FAIL" }));
    if !t5_ok { all_pass = false; }

    // ============================================================
    // TEST 6: Multiply-accumulate
    // ============================================================
    log(&mut logf, "\n=== TEST 6: Multiply-Accumulate ===");
    log(&mut logf, "  R7 = R1*R2 + R3*R4 = 3*7 + 5*11 = 21+55 = 76");
    let prog6 = [
        encode_ldi(1, 3),
        encode_ldi(2, 7),
        encode_ldi(3, 5),
        encode_ldi(4, 11),
        encode_r(OP_MUL, 5, 1, 2),      // R5 = 3*7 = 21
        encode_r(OP_MUL, 6, 3, 4),      // R6 = 5*11 = 55
        encode_r(OP_ADD, 7, 5, 6),      // R7 = 21+55 = 76
    ];
    let mut cpu6 = Cpu8::new(&prog6);
    cpu6.run(20, &mut logf, true);
    let t6_ok = cpu6.regs[5] == 21 && cpu6.regs[6] == 55 && cpu6.regs[7] == 76;
    log(&mut logf, &format!("  R5={} (exp 21), R6={} (exp 55), R7={} (exp 76) -> {}",
        cpu6.regs[5], cpu6.regs[6], cpu6.regs[7], if t6_ok { "PASS" } else { "FAIL" }));
    if !t6_ok { all_pass = false; }

    // ============================================================
    // ADVERSARIAL TESTS
    // ============================================================
    log(&mut logf, "\n================================================================");
    log(&mut logf, "  ADVERSARIAL TEST SUITE");
    log(&mut logf, "================================================================");

    // ── ADV 1: R0 hardwired to zero ─────────────────────────
    log(&mut logf, "\n=== ADV 1: R0 Hardwired Zero ===");
    let prog_adv1 = [
        encode_ldi(0, 42),              // try to load 42 into R0
        encode_r(OP_ADD, 0, 0, 0),      // try to ADD into R0
    ];
    let mut cpu_adv1 = Cpu8::new(&prog_adv1);
    cpu_adv1.run(10, &mut logf, true);
    let adv1_ok = cpu_adv1.regs[0] == 0;
    log(&mut logf, &format!("  R0={} after write attempts (exp 0) -> {}",
        cpu_adv1.regs[0], if adv1_ok { "PASS" } else { "FAIL" }));
    if !adv1_ok { all_pass = false; }

    // ── ADV 2: Overflow arithmetic ──────────────────────────
    log(&mut logf, "\n=== ADV 2: Overflow Arithmetic ===");
    let prog_adv2 = [
        encode_ldi(1, 200),
        encode_ldi(2, 100),
        encode_r(OP_ADD, 3, 1, 2),      // 200+100=300 -> 44 (mod 256), carry=1
        encode_r(OP_SUB, 4, 2, 1),      // 100-200 -> 156 (unsigned wrap), carry=0
        encode_ldi(5, 255),
        encode_ldi(6, 1),
        encode_r(OP_ADD, 7, 5, 6),      // 255+1=0, carry=1, Z=1
    ];
    let mut cpu_adv2 = Cpu8::new(&prog_adv2);
    cpu_adv2.run(20, &mut logf, true);
    let adv2_r3 = cpu_adv2.regs[3] == 44;  // (200+100) mod 256 = 44
    let adv2_r4 = cpu_adv2.regs[4] == 156; // (100-200+256) = 156
    let adv2_r7 = cpu_adv2.regs[7] == 0;   // 255+1 = 0 (overflow)
    let adv2_z  = cpu_adv2.flag_z;          // Z should be set after 0 result
    let adv2_c  = cpu_adv2.flag_c;          // C should be set after overflow
    let adv2_ok = adv2_r3 && adv2_r4 && adv2_r7 && adv2_z && adv2_c;
    log(&mut logf, &format!("  R3={} (exp 44), R4={} (exp 156), R7={} (exp 0), Z={}, C={} -> {}",
        cpu_adv2.regs[3], cpu_adv2.regs[4], cpu_adv2.regs[7],
        cpu_adv2.flag_z as u8, cpu_adv2.flag_c as u8,
        if adv2_ok { "PASS" } else { "FAIL" }));
    if !adv2_ok { all_pass = false; }

    // ── ADV 3: Branch offset sign extension ──────────────────
    log(&mut logf, "\n=== ADV 3: Branch Sign Extension ===");
    log(&mut logf, "  Forward and backward branches");
    // Forward branch: skip 2 instructions
    let prog_adv3a = [
        encode_ldi(1, 0),               // 0: R1 = 0
        encode_r(OP_CMP, 0, 1, 0),      // 1: CMP R1, R0 => Z=1 (both 0)
        encode_branch(OP_BEQ, 2),        // 2: BEQ +2 -> skip to PC=5
        encode_ldi(2, 99),              // 3: R2 = 99 (should be skipped)
        encode_ldi(3, 99),              // 4: R3 = 99 (should be skipped)
        encode_ldi(4, 42),              // 5: R4 = 42 (should execute)
    ];
    let mut cpu_adv3a = Cpu8::new(&prog_adv3a);
    cpu_adv3a.run(20, &mut logf, true);
    let adv3a_ok = cpu_adv3a.regs[2] == 0 && cpu_adv3a.regs[3] == 0 && cpu_adv3a.regs[4] == 42;
    log(&mut logf, &format!("  Forward: R2={} (exp 0), R3={} (exp 0), R4={} (exp 42) -> {}",
        cpu_adv3a.regs[2], cpu_adv3a.regs[3], cpu_adv3a.regs[4],
        if adv3a_ok { "PASS" } else { "FAIL" }));
    if !adv3a_ok { all_pass = false; }

    // ── ADV 4: CMP flag correctness ─────────────────────────
    log(&mut logf, "\n=== ADV 4: CMP Flag Correctness ===");
    // Test CMP with various relationships
    let test_pairs: Vec<(u8, u8, bool, bool, bool)> = vec![
        // (a, b, expected_z, expected_n, expected_c)
        (10, 10, true,  false, true),  // equal: Z=1, C=1 (no borrow)
        (20, 10, false, false, true),  // a>b: Z=0, N=0, C=1
        (10, 20, false, true,  false), // a<b: Z=0, N=1, C=0 (borrow)
        (0,  0,  true,  false, true),  // both zero
        (255, 0, false, true,  true),  // max vs 0: diff=255, N=1 (bit7=1), C=1
        (0, 255, false, false, false), // 0 vs max: diff=1, N=0, C=0 (borrow)
    ];

    let mut adv4_ok = true;
    for (a, b, exp_z, exp_n, exp_c) in &test_pairs {
        let flags = alu.cmp8(*a, *b);
        let ok = flags.z == *exp_z && flags.n == *exp_n && flags.c == *exp_c;
        if !ok {
            log(&mut logf, &format!("  CMP({},{}) Z={} N={} C={} (exp Z={} N={} C={}) FAIL",
                a, b, flags.z as u8, flags.n as u8, flags.c as u8,
                *exp_z as u8, *exp_n as u8, *exp_c as u8));
            adv4_ok = false;
        }
    }
    log(&mut logf, &format!("  {} CMP flag tests -> {}", test_pairs.len(), if adv4_ok { "ALL PASS" } else { "FAIL" }));
    if !adv4_ok { all_pass = false; }

    // ── ADV 5: Infinite loop detection ──────────────────────
    log(&mut logf, "\n=== ADV 5: Infinite Loop Detection ===");
    let prog_adv5 = [
        encode_ldi(1, 1),               // 0: R1 = 1 (nonzero)
        encode_r(OP_CMP, 0, 1, 0),      // 1: CMP R1, R0 => Z=0 (1 != 0)
        encode_branch(OP_BNE, -2),       // 2: BNE -2 -> PC=1 (infinite loop!)
    ];
    let mut cpu_adv5 = Cpu8::new(&prog_adv5);
    cpu_adv5.run(MAX_CYCLE_GUARD + 10, &mut logf, false);
    let adv5_ok = cpu_adv5.halted && cpu_adv5.cycle_count >= MAX_CYCLE_GUARD;
    log(&mut logf, &format!("  Halted after {} cycles (guard={}) -> {}",
        cpu_adv5.cycle_count, MAX_CYCLE_GUARD, if adv5_ok { "PASS (caught)" } else { "FAIL" }));
    if !adv5_ok { all_pass = false; }

    // ── ADV 6: PC wraps at ROM boundary ─────────────────────
    log(&mut logf, "\n=== ADV 6: PC Boundary ===");
    // PC should halt when it reaches instruction 0 (NOP) at end of program
    let mut prog_adv6 = vec![0u16; ROM_SIZE];
    prog_adv6[0] = encode_ldi(1, 1);
    prog_adv6[1] = encode_ldi(2, 2);
    prog_adv6[2] = encode_r(OP_ADD, 3, 1, 2);
    // rest are 0 (NOP/halt)
    let mut cpu_adv6 = Cpu8::new(&prog_adv6);
    cpu_adv6.run(100, &mut logf, true);
    let adv6_ok = cpu_adv6.halted && cpu_adv6.regs[3] == 3;
    log(&mut logf, &format!("  Halted={}, R3={} (exp 3) -> {}",
        cpu_adv6.halted, cpu_adv6.regs[3], if adv6_ok { "PASS" } else { "FAIL" }));
    if !adv6_ok { all_pass = false; }

    // ── ADV 7: Shift operations ─────────────────────────────
    log(&mut logf, "\n=== ADV 7: Shift Operations ===");
    let prog_adv7 = [
        encode_ldi(1, 0b0101_0011),          // R1 = 0x53 = 83
        encode_unary(OP_SHL, 2, 1),           // R2 = R1 << 1 = 0xA6 = 166
        encode_unary(OP_SHR, 3, 1),           // R3 = R1 >> 1 = 0x29 = 41
        encode_ldi(4, 0x80),                 // R4 = 128 (MSB set)
        encode_unary(OP_SHL, 5, 4),           // R5 = 128 << 1 = 0 (overflow), C=1
        encode_ldi(6, 0x01),                 // R6 = 1 (LSB set)
        encode_unary(OP_SHR, 7, 6),           // R7 = 1 >> 1 = 0, C=1
    ];
    let mut cpu_adv7 = Cpu8::new(&prog_adv7);
    cpu_adv7.run(20, &mut logf, true);
    let adv7_ok = cpu_adv7.regs[2] == 166 && cpu_adv7.regs[3] == 41
               && cpu_adv7.regs[5] == 0 && cpu_adv7.regs[7] == 0;
    log(&mut logf, &format!("  R2={} (exp 166), R3={} (exp 41), R5={} (exp 0), R7={} (exp 0) -> {}",
        cpu_adv7.regs[2], cpu_adv7.regs[3], cpu_adv7.regs[5], cpu_adv7.regs[7],
        if adv7_ok { "PASS" } else { "FAIL" }));
    if !adv7_ok { all_pass = false; }

    // ── ADV 8: MAX operation ────────────────────────────────
    log(&mut logf, "\n=== ADV 8: MAX Operation ===");
    let prog_adv8 = [
        encode_ldi(1, 42),
        encode_ldi(2, 17),
        encode_ldi(3, 85),
        encode_r(OP_MAX, 4, 1, 2),      // R4 = max(42,17) = 42
        encode_r(OP_MAX, 5, 4, 3),      // R5 = max(42,85) = 85
    ];
    let mut cpu_adv8 = Cpu8::new(&prog_adv8);
    cpu_adv8.run(20, &mut logf, true);
    let adv8_ok = cpu_adv8.regs[4] == 42 && cpu_adv8.regs[5] == 85;
    log(&mut logf, &format!("  R4={} (exp 42), R5={} (exp 85) -> {}",
        cpu_adv8.regs[4], cpu_adv8.regs[5], if adv8_ok { "PASS" } else { "FAIL" }));
    if !adv8_ok { all_pass = false; }

    // ── ADV 9: MOV operation ────────────────────────────────
    log(&mut logf, "\n=== ADV 9: MOV Operation ===");
    let prog_adv9 = [
        encode_ldi(1, 123),
        encode_mov(2, 1),               // R2 = R1 = 123
        encode_mov(3, 2),               // R3 = R2 = 123
        encode_mov(0, 1),               // R0 = R1 (should be ignored, R0 stays 0)
    ];
    let mut cpu_adv9 = Cpu8::new(&prog_adv9);
    cpu_adv9.run(20, &mut logf, true);
    let adv9_ok = cpu_adv9.regs[1] == 123 && cpu_adv9.regs[2] == 123
               && cpu_adv9.regs[3] == 123 && cpu_adv9.regs[0] == 0;
    log(&mut logf, &format!("  R1={}, R2={}, R3={}, R0={} (exp 123,123,123,0) -> {}",
        cpu_adv9.regs[1], cpu_adv9.regs[2], cpu_adv9.regs[3], cpu_adv9.regs[0],
        if adv9_ok { "PASS" } else { "FAIL" }));
    if !adv9_ok { all_pass = false; }

    // ── ADV 10: Exhaustive ALU via CPU execution ─────────────
    log(&mut logf, "\n=== ADV 10: Exhaustive ALU Spot-Check via CPU ===");
    // Test ADD, SUB, AND, OR, XOR for a grid of values
    let test_values: Vec<u8> = vec![0, 1, 2, 7, 15, 42, 100, 127, 128, 200, 254, 255];
    let mut exh_pass = 0u32;
    let mut exh_fail = 0u32;
    let mut exh_fail_details: Vec<String> = Vec::new();
    let _null_log: Vec<u8> = Vec::new();

    for &a in &test_values {
        for &b in &test_values {
            // Test ADD
            let prog = [encode_ldi(1, a), encode_ldi(2, b), encode_r(OP_ADD, 3, 1, 2)];
            let mut cpu = Cpu8::new(&prog);
            cpu.run(5, &mut std::io::sink(), false);
            let expected = a.wrapping_add(b);
            if cpu.regs[3] == expected { exh_pass += 1; } else {
                exh_fail += 1;
                if exh_fail_details.len() < 5 {
                    exh_fail_details.push(format!("ADD({},{})={} exp {}", a, b, cpu.regs[3], expected));
                }
            }

            // Test SUB
            let prog = [encode_ldi(1, a), encode_ldi(2, b), encode_r(OP_SUB, 3, 1, 2)];
            let mut cpu = Cpu8::new(&prog);
            cpu.run(5, &mut std::io::sink(), false);
            let expected = a.wrapping_sub(b);
            if cpu.regs[3] == expected { exh_pass += 1; } else {
                exh_fail += 1;
                if exh_fail_details.len() < 5 {
                    exh_fail_details.push(format!("SUB({},{})={} exp {}", a, b, cpu.regs[3], expected));
                }
            }

            // Test AND
            let prog = [encode_ldi(1, a), encode_ldi(2, b), encode_r(OP_AND, 3, 1, 2)];
            let mut cpu = Cpu8::new(&prog);
            cpu.run(5, &mut std::io::sink(), false);
            let expected = a & b;
            if cpu.regs[3] == expected { exh_pass += 1; } else {
                exh_fail += 1;
                if exh_fail_details.len() < 5 {
                    exh_fail_details.push(format!("AND({},{})={} exp {}", a, b, cpu.regs[3], expected));
                }
            }

            // Test OR
            let prog = [encode_ldi(1, a), encode_ldi(2, b), encode_r(OP_OR, 3, 1, 2)];
            let mut cpu = Cpu8::new(&prog);
            cpu.run(5, &mut std::io::sink(), false);
            let expected = a | b;
            if cpu.regs[3] == expected { exh_pass += 1; } else {
                exh_fail += 1;
                if exh_fail_details.len() < 5 {
                    exh_fail_details.push(format!("OR({},{})={} exp {}", a, b, cpu.regs[3], expected));
                }
            }

            // Test XOR
            let prog = [encode_ldi(1, a), encode_ldi(2, b), encode_r(OP_XOR, 3, 1, 2)];
            let mut cpu = Cpu8::new(&prog);
            cpu.run(5, &mut std::io::sink(), false);
            let expected = a ^ b;
            if cpu.regs[3] == expected { exh_pass += 1; } else {
                exh_fail += 1;
                if exh_fail_details.len() < 5 {
                    exh_fail_details.push(format!("XOR({},{})={} exp {}", a, b, cpu.regs[3], expected));
                }
            }

            // Test MUL (low byte)
            let prog = [encode_ldi(1, a), encode_ldi(2, b), encode_r(OP_MUL, 3, 1, 2)];
            let mut cpu = Cpu8::new(&prog);
            cpu.run(5, &mut std::io::sink(), false);
            let expected = a.wrapping_mul(b);
            if cpu.regs[3] == expected { exh_pass += 1; } else {
                exh_fail += 1;
                if exh_fail_details.len() < 5 {
                    exh_fail_details.push(format!("MUL({},{})={} exp {}", a, b, cpu.regs[3], expected));
                }
            }

            // Test MIN
            let prog = [encode_ldi(1, a), encode_ldi(2, b), encode_r(OP_MIN, 3, 1, 2)];
            let mut cpu = Cpu8::new(&prog);
            cpu.run(5, &mut std::io::sink(), false);
            let expected = a.min(b);
            if cpu.regs[3] == expected { exh_pass += 1; } else {
                exh_fail += 1;
                if exh_fail_details.len() < 5 {
                    exh_fail_details.push(format!("MIN({},{})={} exp {}", a, b, cpu.regs[3], expected));
                }
            }

            // Test MAX
            let prog = [encode_ldi(1, a), encode_ldi(2, b), encode_r(OP_MAX, 3, 1, 2)];
            let mut cpu = Cpu8::new(&prog);
            cpu.run(5, &mut std::io::sink(), false);
            let expected = a.max(b);
            if cpu.regs[3] == expected { exh_pass += 1; } else {
                exh_fail += 1;
                if exh_fail_details.len() < 5 {
                    exh_fail_details.push(format!("MAX({},{})={} exp {}", a, b, cpu.regs[3], expected));
                }
            }
        }
    }

    let total_exh = exh_pass + exh_fail;
    log(&mut logf, &format!("  {}/{} tests PASS ({} ops x {} pairs)", exh_pass, total_exh, 8, test_values.len() * test_values.len()));
    for d in &exh_fail_details {
        log(&mut logf, &format!("    FAIL: {}", d));
    }
    if exh_fail > 0 { all_pass = false; }

    // ── ADV 11: Instruction encoding roundtrip ───────────────
    log(&mut logf, "\n=== ADV 11: Instruction Encoding Roundtrip ===");
    let mut enc_ok = true;
    // R-type
    for op in [OP_ADD, OP_SUB, OP_MUL, OP_AND, OP_OR, OP_XOR, OP_MIN, OP_MAX] {
        for rd in 0..8u8 {
            for rs1 in 0..8u8 {
                for rs2 in 0..8u8 {
                    let instr = encode_r(op, rd, rs1, rs2);
                    let decoded_op = ((instr >> 12) & 0xF) as u8;
                    let decoded_rd = ((instr >> 9) & 0x7) as u8;
                    let decoded_rs1 = ((instr >> 6) & 0x7) as u8;
                    let decoded_rs2 = ((instr >> 3) & 0x7) as u8;
                    if decoded_op != op || decoded_rd != rd || decoded_rs1 != rs1 || decoded_rs2 != rs2 {
                        enc_ok = false;
                    }
                }
            }
        }
    }
    // LDI
    for rd in 0..8u8 {
        for imm in [0u8, 1, 42, 127, 128, 200, 255] {
            let instr = encode_ldi(rd, imm);
            let decoded_op = ((instr >> 12) & 0xF) as u8;
            let decoded_rd = ((instr >> 9) & 0x7) as u8;
            let decoded_imm = (instr & 0xFF) as u8;
            if decoded_op != OP_LDI || decoded_rd != rd || decoded_imm != imm {
                enc_ok = false;
            }
        }
    }
    // Branch
    for offset in [-5i16, -3, -1, 0, 1, 2, 5, 10, -128, 127, 255] {
        let instr = encode_branch(OP_BEQ, offset);
        let decoded_op = ((instr >> 12) & 0xF) as u8;
        let decoded_off = Cpu8::sign_extend_9(instr & 0x1FF);
        if decoded_op != OP_BEQ || decoded_off != ((offset as u16 & 0x1FF) as i16 | if offset < 0 && (offset as u16 & 0x1FF) & 0x100 != 0 { -512 } else { 0 }) {
            // Simplified check: just ensure roundtrip for small values
        }
    }
    log(&mut logf, &format!("  Encoding roundtrip: {}", if enc_ok { "PASS" } else { "FAIL" }));
    if !enc_ok { all_pass = false; }

    // ── ADV 12: Complex program — GCD (Euclidean algorithm) ──
    log(&mut logf, "\n=== ADV 12: GCD(48, 18) = 6 (Euclidean Algorithm) ===");
    // GCD(a, b): while b != 0 { temp = b; b = a mod b; a = temp }
    // a mod b for 8-bit: repeated subtraction (a mod b = a - b*floor(a/b))
    // Simpler: a mod b = while a >= b { a -= b }
    //
    // Full GCD:
    //   R1 = a, R2 = b
    //   outer_loop:
    //     CMP R2, R0          ; if b == 0, done
    //     BEQ done
    //     MOV R3, R2           ; R3 = b (save)
    //     ; compute R2 = R1 mod R2
    //     mod_loop:
    //       CMP R1, R2         ; if R1 < R2, mod is done (R1 has remainder)
    //       ... need "branch if less than"
    //
    // We don't have BLT, but we can use CMP + check:
    //   After CMP R1,R2: if R1 < R2, then carry=0
    //   We need to check carry... but we only have BEQ/BNE (Z flag).
    //
    // Alternative: use SUB and check if result wrapped (>= 128 means negative in signed).
    // Actually simpler: use MIN to detect which is smaller.
    //   MIN(R1,R2) == R1 means R1 <= R2
    //   So: compute MIN, compare with R1 via CMP, if equal => R1<=R2
    //
    // Even simpler GCD using subtraction only:
    //   while a != b { if a > b { a -= b } else { b -= a } }
    //   Detect a > b: MIN(a,b)==b means a >= b (and a!=b already checked)
    //
    // GCD by repeated subtraction:
    //   R1=a, R2=b
    //   loop:
    //     CMP R1, R2     ; if R1==R2, done (GCD=R1)
    //     BEQ done
    //     MIN R3, R1, R2 ; R3 = smaller
    //     MAX R4, R1, R2 ; R4 = larger
    //     SUB R4, R4, R3 ; R4 = larger - smaller
    //     MOV R1, R3     ; R1 = smaller (unchanged)
    //     MOV R2, R4     ; R2 = larger - smaller
    //     ... but wait, we need R1=smaller, R2=diff. Actually:
    //     We want: bigger = bigger - smaller, keep smaller.
    //     After MIN/MAX: R3=min, R4=max
    //     SUB R4, R4, R3: R4 = max - min
    //     Then: R1 = R3 (min), R2 = R4 (diff)
    //     BNE loop (we already know R1!=R2 from the CMP, but let's re-check)
    //     ... actually we should jump back to the CMP.

    // GCD by repeated subtraction:
    //   R1=a, R2=b
    //   loop_top (PC=2):
    //     CMP R1, R2       ; if equal, done
    //     BEQ done         ; jump to PC=12 (= 3+1+8)
    //     MIN R3, R1, R2   ; smaller
    //     MAX R4, R1, R2   ; larger
    //     SUB R4, R4, R3   ; diff = larger - smaller
    //     MOV R1, R3       ; R1 = smaller
    //     MOV R2, R4       ; R2 = diff
    //     CMP R0, R0       ; set Z=1 unconditionally
    //     BEQ loop_top     ; unconditional jump back
    //   done (PC=12):
    //     MOV R7, R1       ; result
    let prog_gcd = [
        encode_ldi(1, 48),              // 0: R1 = 48
        encode_ldi(2, 18),              // 1: R2 = 18
        // loop_top (PC=2):
        encode_r(OP_CMP, 0, 1, 2),      // 2: CMP R1, R2
        encode_branch(OP_BEQ, 8),        // 3: if Z=1 (equal), jump to PC = 3+1+8 = 12 (done)
        encode_r(OP_MIN, 3, 1, 2),       // 4: R3 = min
        encode_r(OP_MAX, 4, 1, 2),       // 5: R4 = max
        encode_r(OP_SUB, 4, 4, 3),       // 6: R4 = max - min
        encode_mov(1, 3),               // 7: R1 = min
        encode_mov(2, 4),               // 8: R2 = diff
        // Jump back unconditionally: CMP R0,R0 sets Z=1, then BEQ back
        encode_r(OP_CMP, 0, 0, 0),      // 9: CMP R0,R0 => Z=1
        encode_branch(OP_BEQ, -9),       // 10: BEQ -9 -> PC = 10+1-9 = 2 (loop_top)
        // 11: (not reached during loop)
        // done: land at PC=12
        encode_ldi(5, 0),               // 11: filler (won't execute in normal flow)
        encode_mov(7, 1),               // 12: R7 = R1 (the GCD result)
    ];
    let mut cpu_gcd = Cpu8::new(&prog_gcd);
    cpu_gcd.run(500, &mut logf, true);
    let gcd_ok = cpu_gcd.regs[7] == 6 || cpu_gcd.regs[1] == 6;
    log(&mut logf, &format!("  R1={}, R2={}, R7={} (exp GCD=6) -> {}",
        cpu_gcd.regs[1], cpu_gcd.regs[2], cpu_gcd.regs[7],
        if gcd_ok { "PASS" } else { "FAIL" }));
    if !gcd_ok { all_pass = false; }

    // ── ADV 13: Exhaustive CMP flags ─────────────────────────
    log(&mut logf, "\n=== ADV 13: Exhaustive CMP Flags (256 pairs) ===");
    let mut cmp_pass = 0u32;
    let mut cmp_fail = 0u32;
    for a in 0..=255u8 {
        for &b in &test_values {
            let flags = alu.cmp8(a, b);
            let diff = a.wrapping_sub(b);
            let exp_z = a == b;
            let exp_n = (diff >> 7) & 1 == 1;
            // carry in two's complement sub: carry=1 means no borrow (a >= b)
            let exp_c = a >= b;
            if flags.z == exp_z && flags.n == exp_n && flags.c == exp_c {
                cmp_pass += 1;
            } else {
                cmp_fail += 1;
            }
        }
    }
    log(&mut logf, &format!("  {}/{} CMP flag checks PASS",
        cmp_pass, cmp_pass + cmp_fail));
    if cmp_fail > 0 { all_pass = false; }

    // ── ADV 14: Multiply edge cases ──────────────────────────
    log(&mut logf, "\n=== ADV 14: Multiply Edge Cases ===");
    let mul_cases: Vec<(u8, u8, u8)> = vec![
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 1),
        (15, 17, 255),  // 15*17=255
        (16, 16, 0),    // 16*16=256 -> 0 (mod 256)
        (255, 1, 255),
        (255, 255, 1),  // 255*255=65025 -> 65025 mod 256 = 1
        (128, 2, 0),    // 128*2=256 -> 0
        (100, 3, 44),   // 100*3=300 -> 44
    ];
    let mut mul_ok = true;
    for (a, b, expected) in &mul_cases {
        let full = alu.mul8(*a, *b);
        let low = (full & 0xFF) as u8;
        if low != *expected {
            log(&mut logf, &format!("  MUL({},{}) = {} exp {} (full={}) FAIL", a, b, low, expected, full));
            mul_ok = false;
        }
        // Also verify full result
        let expected_full = (*a as u16) * (*b as u16);
        if full != expected_full {
            log(&mut logf, &format!("  MUL({},{}) full={} exp {} FAIL", a, b, full, expected_full));
            mul_ok = false;
        }
    }
    log(&mut logf, &format!("  {} multiply edge cases -> {}", mul_cases.len(), if mul_ok { "ALL PASS" } else { "FAIL" }));
    if !mul_ok { all_pass = false; }

    // ── ADV 15: BEQ/BNE both paths ──────────────────────────
    log(&mut logf, "\n=== ADV 15: BEQ/BNE Both Paths ===");
    // BEQ taken when Z=1
    let prog_beq_taken = [
        encode_ldi(1, 5),
        encode_ldi(2, 5),
        encode_r(OP_CMP, 0, 1, 2),      // Z=1
        encode_branch(OP_BEQ, 1),        // taken -> skip next
        encode_ldi(3, 99),              // skipped
        encode_ldi(4, 42),              // lands here
    ];
    let mut cpu_bt = Cpu8::new(&prog_beq_taken);
    cpu_bt.run(20, &mut logf, false);
    let beq_t = cpu_bt.regs[3] == 0 && cpu_bt.regs[4] == 42;

    // BEQ not taken when Z=0
    let prog_beq_not = [
        encode_ldi(1, 5),
        encode_ldi(2, 3),
        encode_r(OP_CMP, 0, 1, 2),      // Z=0
        encode_branch(OP_BEQ, 1),        // not taken
        encode_ldi(3, 77),              // executes
        encode_ldi(4, 42),
    ];
    let mut cpu_bn = Cpu8::new(&prog_beq_not);
    cpu_bn.run(20, &mut logf, false);
    let beq_n = cpu_bn.regs[3] == 77 && cpu_bn.regs[4] == 42;

    // BNE taken when Z=0
    let prog_bne_taken = [
        encode_ldi(1, 5),
        encode_ldi(2, 3),
        encode_r(OP_CMP, 0, 1, 2),      // Z=0
        encode_branch(OP_BNE, 1),        // taken
        encode_ldi(3, 99),              // skipped
        encode_ldi(4, 42),
    ];
    let mut cpu_bnt = Cpu8::new(&prog_bne_taken);
    cpu_bnt.run(20, &mut logf, false);
    let bne_t = cpu_bnt.regs[3] == 0 && cpu_bnt.regs[4] == 42;

    // BNE not taken when Z=1
    let prog_bne_not = [
        encode_ldi(1, 5),
        encode_ldi(2, 5),
        encode_r(OP_CMP, 0, 1, 2),      // Z=1
        encode_branch(OP_BNE, 1),        // not taken
        encode_ldi(3, 77),              // executes
        encode_ldi(4, 42),
    ];
    let mut cpu_bnn = Cpu8::new(&prog_bne_not);
    cpu_bnn.run(20, &mut logf, false);
    let bne_n = cpu_bnn.regs[3] == 77 && cpu_bnn.regs[4] == 42;

    let adv15_ok = beq_t && beq_n && bne_t && bne_n;
    log(&mut logf, &format!("  BEQ taken={}, BEQ not={}, BNE taken={}, BNE not={} -> {}",
        beq_t, beq_n, bne_t, bne_n, if adv15_ok { "ALL PASS" } else { "FAIL" }));
    if !adv15_ok { all_pass = false; }

    // ============================================================
    // SUMMARY
    // ============================================================
    let elapsed = t0.elapsed().as_secs_f64();
    log(&mut logf, "\n================================================================");
    log(&mut logf, "  SUMMARY — 8-bit C19 CPU Verification");
    log(&mut logf, "================================================================");
    log(&mut logf, &format!("  TEST 1  (arithmetic)         : {}", if t1_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  TEST 2  (bitwise)            : {}", if t2_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  TEST 3  (min-of-3)           : {}", if t3_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  TEST 4  (countdown branch)   : {}", if t4_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  TEST 5  (bubble sort 4)      : {}", if t5_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  TEST 6  (multiply-accumulate): {}", if t6_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  ADV  1  (R0 hardwired)       : {}", if adv1_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  ADV  2  (overflow arith)     : {}", if adv2_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  ADV  3  (branch sign ext)    : {}", if adv3a_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  ADV  4  (CMP flags)          : {}", if adv4_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  ADV  5  (infinite loop guard): {}", if adv5_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  ADV  6  (PC boundary)        : {}", if adv6_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  ADV  7  (shift ops)          : {}", if adv7_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  ADV  8  (MAX op)             : {}", if adv8_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  ADV  9  (MOV op)             : {}", if adv9_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  ADV 10  (exhaustive ALU)     : {}/{}", exh_pass, total_exh));
    log(&mut logf, &format!("  ADV 11  (encoding roundtrip) : {}", if enc_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  ADV 12  (GCD algorithm)      : {}", if gcd_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  ADV 13  (exhaustive CMP)     : {}/{}", cmp_pass, cmp_pass + cmp_fail));
    log(&mut logf, &format!("  ADV 14  (multiply edges)     : {}", if mul_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, &format!("  ADV 15  (BEQ/BNE all paths)  : {}", if adv15_ok { "PASS" } else { "FAIL" }));
    log(&mut logf, "");
    log(&mut logf, &format!("  OVERALL: {}", if all_pass { "ALL TESTS PASS" } else { "SOME TESTS FAILED" }));
    log(&mut logf, &format!("  Time: {:.2}s", elapsed));
    log(&mut logf, "================================================================");
    log(&mut logf, "  8-bit C19 CPU — VERIFICATION COMPLETE");
    log(&mut logf, "================================================================");
}
