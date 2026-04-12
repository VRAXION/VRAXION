//! Byte + opcode v1 benchmark scaffold.
//!
//! Goal:
//!   compare output-head families on the exact 1 byte + 4 opcode -> 1 byte domain
//!   before wiring the winner into the grower mainline.
//!
//! Domain:
//!   input  = 8 data bits + 4 opcode one-hot
//!   ops    = COPY / NOT / INC / DEC
//!   output = 1 byte
//!
//! Variants:
//!   A. direct-binary   : 8 sigmoid outputs -> thresholded bits
//!   B. scalar-bucket   : 1 scalar -> nearest byte bucket
//!   C. nibble-proto    : 2 x 8D latent head -> fixed 16-class prototype decode
//!
//! This is intentionally a B1 scaffold, not the final grower integration.
//!
//! Run:
//!   cargo run --release --example byte_opcode_v1

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const DATA_BITS: usize = 8;
const OPCODES: usize = 4;
const INPUT_DIM: usize = DATA_BITS + OPCODES;
const HIDDEN_DIM: usize = 32;
const PROTO_DIM: usize = 8;
const PROTO_OUT: usize = PROTO_DIM * 2;

const ADV_CASES: [u8; 10] = [0x00, 0x01, 0x0F, 0x10, 0x7F, 0x80, 0xFE, 0xFF, 0x55, 0xAA];

#[derive(Clone, Copy)]
enum Opcode {
    Copy,
    Not,
    Inc,
    Dec,
}

impl Opcode {
    fn all() -> [Self; 4] {
        [Self::Copy, Self::Not, Self::Inc, Self::Dec]
    }

    fn idx(self) -> usize {
        match self {
            Self::Copy => 0,
            Self::Not => 1,
            Self::Inc => 2,
            Self::Dec => 3,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Copy => "COPY",
            Self::Not => "NOT",
            Self::Inc => "INC",
            Self::Dec => "DEC",
        }
    }

    fn apply(self, x: u8) -> u8 {
        match self {
            Self::Copy => x,
            Self::Not => !x,
            Self::Inc => x.wrapping_add(1),
            Self::Dec => x.wrapping_sub(1),
        }
    }
}

#[derive(Clone)]
struct Sample {
    input: [f32; INPUT_DIM],
    opcode: Opcode,
    target_byte: u8,
}

fn bits8(x: u8) -> [f32; DATA_BITS] {
    let mut out = [0.0f32; DATA_BITS];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = if (x >> i) & 1 == 1 { 1.0 } else { 0.0 };
    }
    out
}

fn byte_from_bits(bits: &[u8]) -> u8 {
    let mut out = 0u8;
    for (i, &b) in bits.iter().enumerate().take(DATA_BITS) {
        if b != 0 { out |= 1 << i; }
    }
    out
}

fn dataset() -> Vec<Sample> {
    let mut out = Vec::with_capacity(256 * OPCODES);
    for x in 0u16..=255 {
        let xb = x as u8;
        let xb_bits = bits8(xb);
        for op in Opcode::all() {
            let mut input = [0.0f32; INPUT_DIM];
            input[..DATA_BITS].copy_from_slice(&xb_bits);
            input[DATA_BITS + op.idx()] = 1.0;
            out.push(Sample {
                input,
                opcode: op,
                target_byte: op.apply(xb),
            });
        }
    }
    out
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn relu(x: f32) -> f32 { x.max(0.0) }

#[derive(Clone)]
struct Mlp {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
}

impl Mlp {
    fn new(input: usize, hidden: usize, output: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut w1 = vec![vec![0.0; input]; hidden];
        let mut w2 = vec![vec![0.0; hidden]; output];
        for row in &mut w1 {
            for v in row {
                *v = rng.gen_range(-0.20..0.20);
            }
        }
        for row in &mut w2 {
            for v in row {
                *v = rng.gen_range(-0.20..0.20);
            }
        }
        Self { w1, b1: vec![0.0; hidden], w2, b2: vec![0.0; output] }
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut hidden_raw = vec![0.0; self.w1.len()];
        let mut hidden = vec![0.0; self.w1.len()];
        for i in 0..self.w1.len() {
            let mut s = self.b1[i];
            for (j, &xj) in x.iter().enumerate() {
                s += self.w1[i][j] * xj;
            }
            hidden_raw[i] = s;
            hidden[i] = relu(s);
        }
        let mut out = vec![0.0; self.w2.len()];
        for i in 0..self.w2.len() {
            let mut s = self.b2[i];
            for (j, &hj) in hidden.iter().enumerate() {
                s += self.w2[i][j] * hj;
            }
            out[i] = s;
        }
        (hidden_raw, hidden, out)
    }
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v {
            *x /= norm;
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (&ai, &bi) in a.iter().zip(b) {
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    if na <= 1e-8 || nb <= 1e-8 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}

fn build_prototypes(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut protos: Vec<Vec<f32>> = Vec::with_capacity(16);
    for _class in 0..16 {
        let mut best = vec![0.0; PROTO_DIM];
        let mut best_score = f32::INFINITY;
        for _ in 0..64 {
            let mut cand: Vec<f32> = (0..PROTO_DIM).map(|_| rng.gen_range(-1.0..1.0)).collect();
            normalize(&mut cand);
            let max_sim = protos.iter().map(|p| cosine(&cand, p).abs()).fold(0.0f32, f32::max);
            if max_sim < best_score {
                best_score = max_sim;
                best = cand;
            }
        }
        protos.push(best);
    }
    protos
}

#[derive(Clone, Copy)]
enum HeadKind {
    DirectBinary,
    ScalarBucket,
    NibblePrototype,
}

impl HeadKind {
    fn name(self) -> &'static str {
        match self {
            Self::DirectBinary => "direct-binary",
            Self::ScalarBucket => "scalar-bucket",
            Self::NibblePrototype => "nibble-prototype",
        }
    }

    fn output_dim(self) -> usize {
        match self {
            Self::DirectBinary => 8,
            Self::ScalarBucket => 1,
            Self::NibblePrototype => PROTO_OUT,
        }
    }
}

struct TrainConfig {
    epochs: usize,
    lr: f32,
    hidden: usize,
}

fn target_bits(byte: u8) -> [u8; 8] {
    let mut out = [0u8; 8];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (byte >> i) & 1;
    }
    out
}

fn target_proto(byte: u8, protos: &[Vec<f32>]) -> Vec<f32> {
    let low = (byte & 0x0F) as usize;
    let high = ((byte >> 4) & 0x0F) as usize;
    let mut out = Vec::with_capacity(PROTO_OUT);
    out.extend_from_slice(&protos[low]);
    out.extend_from_slice(&protos[high]);
    out
}

fn train_head(kind: HeadKind, data: &[Sample], cfg: &TrainConfig, protos: &[Vec<f32>], seed: u64) -> Mlp {
    let mut model = Mlp::new(INPUT_DIM, cfg.hidden, kind.output_dim(), seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0xBAD5EED);
    let mut order: Vec<usize> = (0..data.len()).collect();

    for _epoch in 0..cfg.epochs {
        order.shuffle(&mut rng);
        for &idx in &order {
            let sample = &data[idx];
            let (hidden_raw, hidden, out) = model.forward(&sample.input);
            let mut grad_out = vec![0.0; out.len()];

            match kind {
                HeadKind::DirectBinary => {
                    let bits = target_bits(sample.target_byte);
                    for i in 0..8 {
                        let y = sigmoid(out[i]);
                        grad_out[i] = y - bits[i] as f32;
                    }
                }
                HeadKind::ScalarBucket => {
                    let target = sample.target_byte as f32 / 255.0;
                    grad_out[0] = 2.0 * (out[0] - target);
                }
                HeadKind::NibblePrototype => {
                    let target = target_proto(sample.target_byte, protos);
                    for i in 0..PROTO_OUT {
                        grad_out[i] = 2.0 * (out[i] - target[i]) / PROTO_OUT as f32;
                    }
                }
            }

            let mut grad_hidden = vec![0.0; hidden.len()];
            for (i, go) in grad_out.iter().enumerate() {
                for (j, &hj) in hidden.iter().enumerate() {
                    grad_hidden[j] += *go * model.w2[i][j];
                    model.w2[i][j] -= cfg.lr * *go * hj;
                }
                model.b2[i] -= cfg.lr * *go;
            }

            for j in 0..hidden.len() {
                let gh = if hidden_raw[j] > 0.0 { grad_hidden[j] } else { 0.0 };
                for (k, &xk) in sample.input.iter().enumerate() {
                    model.w1[j][k] -= cfg.lr * gh * xk;
                }
                model.b1[j] -= cfg.lr * gh;
            }
        }
    }

    model
}

fn decode_direct(out: &[f32]) -> u8 {
    let mut bits = [0u8; 8];
    for i in 0..8 {
        bits[i] = if sigmoid(out[i]) >= 0.5 { 1 } else { 0 };
    }
    byte_from_bits(&bits)
}

fn decode_scalar(out: &[f32]) -> u8 {
    (out[0].mul_add(255.0, 0.0).round() as i32).clamp(0, 255) as u8
}

fn decode_proto(out: &[f32], protos: &[Vec<f32>]) -> (u8, f32, f32) {
    let low = &out[..PROTO_DIM];
    let high = &out[PROTO_DIM..PROTO_OUT];
    let (low_idx, low_margin) = nearest_proto(low, protos);
    let (high_idx, high_margin) = nearest_proto(high, protos);
    (((high_idx as u8) << 4) | low_idx as u8, low_margin, high_margin)
}

fn nearest_proto(v: &[f32], protos: &[Vec<f32>]) -> (usize, f32) {
    let mut scored: Vec<(usize, f32)> = protos.iter().enumerate().map(|(i, p)| (i, cosine(v, p))).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let margin = if scored.len() >= 2 { scored[0].1 - scored[1].1 } else { 0.0 };
    (scored[0].0, margin)
}

struct EvalResult {
    exact_byte_acc: f32,
    low_nibble_acc: f32,
    high_nibble_acc: f32,
    per_op: Vec<(Opcode, f32)>,
    avg_low_margin: f32,
    avg_high_margin: f32,
}

fn eval_head(kind: HeadKind, model: &Mlp, data: &[Sample], protos: &[Vec<f32>]) -> EvalResult {
    let mut exact = 0usize;
    let mut low_ok = 0usize;
    let mut high_ok = 0usize;
    let mut per_op_ok = [0usize; 4];
    let mut per_op_total = [0usize; 4];
    let mut low_margin_sum = 0.0;
    let mut high_margin_sum = 0.0;

    for sample in data {
        let (_, _, out) = model.forward(&sample.input);
        let (pred, low_margin, high_margin) = match kind {
            HeadKind::DirectBinary => (decode_direct(&out), 0.0, 0.0),
            HeadKind::ScalarBucket => (decode_scalar(&out), 0.0, 0.0),
            HeadKind::NibblePrototype => decode_proto(&out, protos),
        };

        if pred == sample.target_byte { exact += 1; }
        if (pred & 0x0F) == (sample.target_byte & 0x0F) { low_ok += 1; }
        if (pred >> 4) == (sample.target_byte >> 4) { high_ok += 1; }
        per_op_ok[sample.opcode.idx()] += usize::from(pred == sample.target_byte);
        per_op_total[sample.opcode.idx()] += 1;
        low_margin_sum += low_margin;
        high_margin_sum += high_margin;
    }

    EvalResult {
        exact_byte_acc: exact as f32 / data.len() as f32 * 100.0,
        low_nibble_acc: low_ok as f32 / data.len() as f32 * 100.0,
        high_nibble_acc: high_ok as f32 / data.len() as f32 * 100.0,
        per_op: Opcode::all().into_iter().map(|op| {
            let total = per_op_total[op.idx()].max(1);
            (op, per_op_ok[op.idx()] as f32 / total as f32 * 100.0)
        }).collect(),
        avg_low_margin: low_margin_sum / data.len() as f32,
        avg_high_margin: high_margin_sum / data.len() as f32,
    }
}

fn print_adversarial(kind: HeadKind, model: &Mlp, protos: &[Vec<f32>]) {
    println!("  adversarial:");
    for &x in &ADV_CASES {
        for op in Opcode::all() {
            let mut input = [0.0f32; INPUT_DIM];
            input[..DATA_BITS].copy_from_slice(&bits8(x));
            input[DATA_BITS + op.idx()] = 1.0;
            let (_, _, out) = model.forward(&input);
            let pred = match kind {
                HeadKind::DirectBinary => decode_direct(&out),
                HeadKind::ScalarBucket => decode_scalar(&out),
                HeadKind::NibblePrototype => decode_proto(&out, protos).0,
            };
            let target = op.apply(x);
            println!(
                "    {}({:#04X}) -> pred={:#04X} target={:#04X} {}",
                op.name(),
                x,
                pred,
                target,
                if pred == target { "OK" } else { "MISS" }
            );
        }
    }
}

fn main() {
    let data = dataset();
    let protos = build_prototypes(0xC0DEC0DE);
    let cfg = TrainConfig { epochs: 700, lr: 0.01, hidden: HIDDEN_DIM };
    let variants = [
        HeadKind::DirectBinary,
        HeadKind::ScalarBucket,
        HeadKind::NibblePrototype,
    ];
    let seeds = [42u64, 142, 47636];

    println!("=== BYTE + OPCODE V1 HEAD SWEEP ===");
    println!("Domain: 1 byte + 4 opcode -> 1 byte (COPY, NOT, INC, DEC)");
    println!("Samples: {}", data.len());
    println!("Hidden: {}  Epochs: {}  LR: {:.4}\n", cfg.hidden, cfg.epochs, cfg.lr);

    let mut summary: Vec<(&'static str, f32, u64)> = Vec::new();

    for kind in variants {
        println!("--- {} ---", kind.name());
        let mut best = None::<(u64, EvalResult, Mlp)>;
        for &seed in &seeds {
            let model = train_head(kind, &data, &cfg, &protos, seed);
            let eval = eval_head(kind, &model, &data, &protos);
            println!(
                "  seed {:>5}: byte={:>5.1}% low={:>5.1}% high={:>5.1}%{}",
                seed,
                eval.exact_byte_acc,
                eval.low_nibble_acc,
                eval.high_nibble_acc,
                if matches!(kind, HeadKind::NibblePrototype) {
                    format!(" margin=({:.3},{:.3})", eval.avg_low_margin, eval.avg_high_margin)
                } else {
                    String::new()
                }
            );
            match &best {
                Some((_, best_eval, _)) if best_eval.exact_byte_acc >= eval.exact_byte_acc => {}
                _ => best = Some((seed, eval, model)),
            }
        }

        let (best_seed, best_eval, best_model) = best.expect("at least one seed");
        println!("  best seed: {}", best_seed);
        for (op, acc) in &best_eval.per_op {
            println!("    {}: {:.1}%", op.name(), acc);
        }
        print_adversarial(kind, &best_model, &protos);
        println!();
        summary.push((kind.name(), best_eval.exact_byte_acc, best_seed));
    }

    summary.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("=== SUMMARY ===");
    for (name, acc, seed) in summary {
        println!("  {:<16} best={:>5.1}% seed={}", name, acc, seed);
    }
}
