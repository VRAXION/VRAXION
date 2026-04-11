//! Heterogeneous Multi-Unit Processor with EP-trained Router
//!
//! Architecture: 5 SPECIALIZED compute units (not identical ALUs!)
//!   Unit 0: ARITH_A  (ADD, SUB)            — 40 neurons, 1 tick
//!   Unit 1: ARITH_B  (ADD, SUB)            — 40 neurons, 1 tick (redundant copy)
//!   Unit 2: MUL      (MUL only)            — 176 neurons, 3 ticks
//!   Unit 3: LOGIC    (AND, OR, XOR, NOT)   — 32 neurons, 1 tick
//!   Unit 4: CMP      (CMP, MIN, MAX)       — 80 neurons, 2 ticks
//!   Total: 368 neurons (vs 1,576 for 4x full ALU = 4.3x cheaper)
//!
//! Three routers compared:
//!   A. Static Dispatch — type-match + first-free among compatible units
//!   B. Priority Queue  — type-match + out-of-order reordering from queue
//!   C. EP Neural       — trained to match optimal hindsight scheduling
//!   + Frozen variant of C
//!
//! Five workloads:
//!   W1: Balanced mix
//!   W2: Arithmetic heavy (stress dual ARITH units)
//!   W3: MUL bottleneck (50% MUL, 3-tick latency)
//!   W4: Logic burst then arithmetic burst (workload shift)
//!   W5: Adversarial alternating heavy ops (MUL, CMP, MUL, CMP...)
//!
//! Run: cargo run --example hetero_alu --release

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

fn c19_act(x: f32) -> f32 { c19(x, 8.0) }

// ============================================================
// RNG
// ============================================================

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed.wrapping_mul(6364136223846793005).wrapping_add(1) }
    }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }
    #[allow(dead_code)]
    fn usize(&mut self, n: usize) -> usize { (self.next() as usize) % n }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() as usize) % (i + 1);
            v.swap(i, j);
        }
    }
}

// ============================================================
// Operations and their properties
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpType {
    Add,    // ARITH: 1 tick
    Sub,    // ARITH: 1 tick
    Mul,    // MUL: 3 ticks
    And,    // LOGIC: 1 tick
    Or,     // LOGIC: 1 tick
    Xor,    // LOGIC: 1 tick
    Not,    // LOGIC: 1 tick
    Cmp,    // CMP: 2 ticks
    Min,    // CMP: 2 ticks
    #[allow(dead_code)]
    Max,    // CMP: 2 ticks
}

/// Which specialized unit category an op belongs to
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnitCategory {
    Arith,  // Units 0, 1
    Mul,    // Unit 2
    Logic,  // Unit 3
    Cmp,    // Unit 4
}

impl OpType {
    fn ticks(&self) -> usize {
        match self {
            OpType::Add | OpType::Sub => 1,
            OpType::And | OpType::Or | OpType::Xor | OpType::Not => 1,
            OpType::Cmp | OpType::Min | OpType::Max => 2,
            OpType::Mul => 3,
        }
    }

    fn category(&self) -> UnitCategory {
        match self {
            OpType::Add | OpType::Sub => UnitCategory::Arith,
            OpType::Mul => UnitCategory::Mul,
            OpType::And | OpType::Or | OpType::Xor | OpType::Not => UnitCategory::Logic,
            OpType::Cmp | OpType::Min | OpType::Max => UnitCategory::Cmp,
        }
    }

    /// Compatible unit indices for this operation
    fn compatible_units(&self) -> &'static [usize] {
        match self.category() {
            UnitCategory::Arith => &[0, 1],  // Two ARITH units
            UnitCategory::Mul   => &[2],      // Single MUL unit
            UnitCategory::Logic => &[3],      // Single LOGIC unit
            UnitCategory::Cmp   => &[4],      // Single CMP unit
        }
    }

    fn type_id(&self) -> u8 {
        match self {
            OpType::Add => 0,
            OpType::Sub => 1,
            OpType::Mul => 2,
            OpType::And => 3,
            OpType::Or  => 4,
            OpType::Xor => 5,
            OpType::Not => 6,
            OpType::Cmp => 7,
            OpType::Min => 8,
            OpType::Max => 9,
        }
    }

    fn execute(&self, a: u8, b: u8) -> u8 {
        match self {
            OpType::Add => a.wrapping_add(b),
            OpType::Sub => a.wrapping_sub(b),
            OpType::Mul => a.wrapping_mul(b),
            OpType::And => a & b,
            OpType::Or  => a | b,
            OpType::Xor => a ^ b,
            OpType::Not => !a,
            OpType::Cmp => if a >= b { 1 } else { 0 },
            OpType::Min => if a < b { a } else { b },
            OpType::Max => if a > b { a } else { b },
        }
    }

    #[allow(dead_code)]
    fn name(&self) -> &'static str {
        match self {
            OpType::Add => "ADD", OpType::Sub => "SUB", OpType::Mul => "MUL",
            OpType::And => "AND", OpType::Or => "OR", OpType::Xor => "XOR",
            OpType::Not => "NOT", OpType::Cmp => "CMP", OpType::Min => "MIN",
            OpType::Max => "MAX",
        }
    }
}

const NUM_UNITS: usize = 5;

// Neuron counts per unit
#[allow(dead_code)]
const UNIT_NEURONS: [usize; NUM_UNITS] = [40, 40, 176, 32, 80];
const TOTAL_COMPUTE_NEURONS: usize = 40 + 40 + 176 + 32 + 80; // = 368

/// Which unit category each unit index belongs to
fn unit_category(unit_idx: usize) -> UnitCategory {
    match unit_idx {
        0 | 1 => UnitCategory::Arith,
        2     => UnitCategory::Mul,
        3     => UnitCategory::Logic,
        4     => UnitCategory::Cmp,
        _     => panic!("invalid unit index"),
    }
}

#[allow(dead_code)]
fn unit_name(unit_idx: usize) -> &'static str {
    match unit_idx {
        0 => "ARITH_A",
        1 => "ARITH_B",
        2 => "MUL",
        3 => "LOGIC",
        4 => "CMP",
        _ => "?",
    }
}

/// Check if a task can run on a unit
fn is_compatible(op: &OpType, unit_idx: usize) -> bool {
    op.category() == unit_category(unit_idx)
}

// ============================================================
// Task
// ============================================================

#[derive(Clone, Debug)]
struct Task {
    id: usize,
    op: OpType,
    a: u8,
    b: u8,
}

// ============================================================
// Unit state
// ============================================================

#[derive(Clone)]
struct UnitState {
    busy_ticks_remaining: usize,
    current_task: Option<Task>,
    tasks_completed: usize,
    ticks_busy: usize,
}

impl UnitState {
    fn new() -> Self {
        UnitState {
            busy_ticks_remaining: 0,
            current_task: None,
            tasks_completed: 0,
            ticks_busy: 0,
        }
    }

    fn is_free(&self) -> bool { self.busy_ticks_remaining == 0 }

    fn assign(&mut self, task: Task) {
        self.busy_ticks_remaining = task.op.ticks();
        self.current_task = Some(task);
    }

    fn tick(&mut self) -> Option<(usize, u8)> {
        if self.busy_ticks_remaining > 0 {
            self.ticks_busy += 1;
            self.busy_ticks_remaining -= 1;
            if self.busy_ticks_remaining == 0 {
                self.tasks_completed += 1;
                if let Some(ref t) = self.current_task {
                    let result = t.op.execute(t.a, t.b);
                    let id = t.id;
                    self.current_task = None;
                    return Some((id, result));
                }
            }
        }
        None
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        *self = UnitState::new();
    }
}

// ============================================================
// Simulation metrics
// ============================================================

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct SimMetrics {
    total_tasks: usize,
    total_ticks: usize,
    tasks_completed: usize,
    stall_ticks: usize,
    unit_busy_ticks: [usize; NUM_UNITS],
    tasks_per_unit: [usize; NUM_UNITS],
    task_latencies: Vec<usize>,
    task_arrival_tick: Vec<usize>,
    reorder_count: usize,
    wrong_unit_errors: usize,
}

impl SimMetrics {
    fn new(n_tasks: usize) -> Self {
        SimMetrics {
            total_tasks: n_tasks,
            total_ticks: 0,
            tasks_completed: 0,
            stall_ticks: 0,
            unit_busy_ticks: [0; NUM_UNITS],
            tasks_per_unit: [0; NUM_UNITS],
            task_latencies: Vec::with_capacity(n_tasks),
            task_arrival_tick: vec![0; n_tasks],
            reorder_count: 0,
            wrong_unit_errors: 0,
        }
    }

    fn throughput(&self) -> f32 {
        if self.total_ticks == 0 { return 0.0; }
        self.tasks_completed as f32 / self.total_ticks as f32
    }

    fn utilization(&self) -> f32 {
        let total = NUM_UNITS * self.total_ticks;
        if total == 0 { return 0.0; }
        self.unit_busy_ticks.iter().sum::<usize>() as f32 / total as f32
    }

    fn per_unit_utilization(&self) -> [f32; NUM_UNITS] {
        let mut u = [0.0f32; NUM_UNITS];
        if self.total_ticks == 0 { return u; }
        for i in 0..NUM_UNITS {
            u[i] = self.unit_busy_ticks[i] as f32 / self.total_ticks as f32;
        }
        u
    }

    fn stall_rate(&self) -> f32 {
        if self.total_ticks == 0 { return 0.0; }
        self.stall_ticks as f32 / self.total_ticks as f32
    }

    fn avg_latency(&self) -> f32 {
        if self.task_latencies.is_empty() { return 0.0; }
        self.task_latencies.iter().sum::<usize>() as f32 / self.task_latencies.len() as f32
    }
}

// ============================================================
// Router trait — dispatch decision
// ============================================================

/// Router returns (unit_index, was_reordered)
/// If no unit is available, returns None (stall).
/// was_reordered = true if the task was pulled from deeper in the queue.

enum DispatchResult {
    /// Dispatch the front-of-queue task to this unit
    Front(usize),
    /// Skip the front task, dispatch task at queue_index to this unit
    Reorder { unit: usize, queue_index: usize },
    /// All compatible units busy, stall
    Stall,
}

// ============================================================
// Router A: Static Dispatch
// ============================================================

fn static_dispatch(task: &Task, units: &[UnitState; NUM_UNITS]) -> Option<usize> {
    let compat = task.op.compatible_units();
    for &u in compat {
        if units[u].is_free() {
            return Some(u);
        }
    }
    None
}

// ============================================================
// Router B: Priority Queue with out-of-order reordering
// ============================================================

fn priority_queue_dispatch(
    queue: &[Task],
    units: &[UnitState; NUM_UNITS],
) -> DispatchResult {
    if queue.is_empty() { return DispatchResult::Stall; }

    // Try front of queue first
    let front = &queue[0];
    if let Some(u) = static_dispatch(front, units) {
        return DispatchResult::Front(u);
    }

    // Front is blocked — scan deeper in queue for a task that CAN run now
    for qi in 1..queue.len().min(16) {
        let task = &queue[qi];
        if let Some(u) = static_dispatch(task, units) {
            return DispatchResult::Reorder { unit: u, queue_index: qi };
        }
    }

    DispatchResult::Stall
}

// ============================================================
// EP Neural Router
// ============================================================

const EP_IN_DIM: usize = 26; // see encode_input
const EP_H_DIM: usize = 16;
const EP_OUT_DIM: usize = NUM_UNITS; // 5 outputs, one per unit

struct EpRouter {
    w1: Vec<f32>,  // h_dim x in_dim
    w2: Vec<f32>,  // out_dim x h_dim
    b1: Vec<f32>,  // h_dim
    b2: Vec<f32>,  // out_dim
}

impl EpRouter {
    fn new(rng: &mut Rng) -> Self {
        let s1 = 1.0 * (2.0 / EP_IN_DIM as f32).sqrt();
        let s2 = 1.0 * (2.0 / EP_H_DIM as f32).sqrt();

        EpRouter {
            w1: (0..EP_H_DIM * EP_IN_DIM).map(|_| rng.range_f32(-s1, s1)).collect(),
            w2: (0..EP_OUT_DIM * EP_H_DIM).map(|_| rng.range_f32(-s2, s2)).collect(),
            b1: vec![0.0; EP_H_DIM],
            b2: vec![0.0; EP_OUT_DIM],
        }
    }

    fn encode_input(task: &Task, units: &[UnitState; NUM_UNITS], queue_depth: usize, next_queue_cat: Option<UnitCategory>) -> Vec<f32> {
        let mut x = Vec::with_capacity(EP_IN_DIM);

        // task_type: 4 bits (encode type_id as 4 binary bits, 0..9 fits in 4 bits)
        let tid = task.op.type_id();
        x.push((tid & 1) as f32);
        x.push(((tid >> 1) & 1) as f32);
        x.push(((tid >> 2) & 1) as f32);
        x.push(((tid >> 3) & 1) as f32);

        // unit_busy: 5 bits (one per unit)
        for i in 0..NUM_UNITS {
            x.push(if units[i].is_free() { 0.0 } else { 1.0 });
        }

        // unit_remaining: 5 x 2 bits (ticks remaining, 0-3 encoded in 2 bits)
        for i in 0..NUM_UNITS {
            let rem = units[i].busy_ticks_remaining.min(3) as u8;
            x.push((rem & 1) as f32);
            x.push(((rem >> 1) & 1) as f32);
        }

        // queue_depth: 3 bits (0..7)
        let qd = queue_depth.min(7) as u8;
        x.push((qd & 1) as f32);
        x.push(((qd >> 1) & 1) as f32);
        x.push(((qd >> 2) & 1) as f32);

        // queue_next_types: 4 bits (category of next queued task, one-hot)
        // 0=arith, 1=mul, 2=logic, 3=cmp, all zero if queue empty
        let mut cat_bits = [0.0f32; 4];
        if let Some(cat) = next_queue_cat {
            match cat {
                UnitCategory::Arith => cat_bits[0] = 1.0,
                UnitCategory::Mul   => cat_bits[1] = 1.0,
                UnitCategory::Logic => cat_bits[2] = 1.0,
                UnitCategory::Cmp   => cat_bits[3] = 1.0,
            }
        }
        x.extend_from_slice(&cat_bits);

        assert_eq!(x.len(), EP_IN_DIM);
        x
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut h = vec![0.0f32; EP_H_DIM];
        for j in 0..EP_H_DIM {
            let mut sum = self.b1[j];
            for i in 0..EP_IN_DIM {
                sum += self.w1[j * EP_IN_DIM + i] * x[i];
            }
            h[j] = c19_act(sum);
        }

        let mut out = vec![0.0f32; EP_OUT_DIM];
        for k in 0..EP_OUT_DIM {
            let mut sum = self.b2[k];
            for j in 0..EP_H_DIM {
                sum += self.w2[k * EP_H_DIM + j] * h[j];
            }
            out[k] = c19_act(sum);
        }
        out
    }

    fn route(&self, task: &Task, units: &[UnitState; NUM_UNITS], queue_depth: usize, next_queue_cat: Option<UnitCategory>) -> Option<usize> {
        let x = Self::encode_input(task, units, queue_depth, next_queue_cat);
        let scores = self.forward(&x);

        // Pick highest-scoring FREE and COMPATIBLE unit
        let compat = task.op.compatible_units();
        let mut best_idx: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;
        for &u in compat {
            if units[u].is_free() && scores[u] > best_score {
                best_score = scores[u];
                best_idx = Some(u);
            }
        }
        best_idx
    }
}

// ============================================================
// Frozen Router — int8 quantized LUT version
// ============================================================

#[allow(dead_code)]
struct FrozenRouter {
    w1_q: Vec<i16>,
    w2_q: Vec<i16>,
    b1_q: Vec<i16>,
    b2_q: Vec<i16>,
    scale1: f32,
    act_lut: Vec<i16>,
    act_lut_min: i32,
}

impl FrozenRouter {
    fn from_ep(net: &EpRouter) -> Self {
        let max_w1 = net.w1.iter().chain(net.b1.iter())
            .map(|v| v.abs()).fold(0.0f32, f32::max).max(0.001);
        let max_w2 = net.w2.iter().chain(net.b2.iter())
            .map(|v| v.abs()).fold(0.0f32, f32::max).max(0.001);

        let scale1 = 1000.0 / max_w1;
        let scale2 = 1000.0 / max_w2;

        let w1_q: Vec<i16> = net.w1.iter().map(|&v| (v * scale1).round().clamp(-32000.0, 32000.0) as i16).collect();
        let w2_q: Vec<i16> = net.w2.iter().map(|&v| (v * scale2).round().clamp(-32000.0, 32000.0) as i16).collect();
        let b1_q: Vec<i16> = net.b1.iter().map(|&v| (v * scale1).round().clamp(-32000.0, 32000.0) as i16).collect();
        let b2_q: Vec<i16> = net.b2.iter().map(|&v| (v * scale2).round().clamp(-32000.0, 32000.0) as i16).collect();

        // Build activation LUT for hidden layer
        let mut min_h_sum = i32::MAX;
        let mut max_h_sum = i32::MIN;
        for j in 0..EP_H_DIM {
            let mut lo = b1_q[j] as i32;
            let mut hi = b1_q[j] as i32;
            for i in 0..EP_IN_DIM {
                let w = w1_q[j * EP_IN_DIM + i] as i32;
                if w > 0 { hi += w; } else { lo += w; }
            }
            if lo < min_h_sum { min_h_sum = lo; }
            if hi > max_h_sum { max_h_sum = hi; }
        }

        let act_range = (max_h_sum - min_h_sum + 1) as usize;
        let mut act_lut = vec![0i16; act_range];
        let mut max_act = 0.0f32;
        for s in min_h_sum..=max_h_sum {
            let act_val = c19_act(s as f32 / scale1);
            if act_val.abs() > max_act { max_act = act_val.abs(); }
        }
        let act_scale = if max_act > 0.001 { 1000.0 / max_act } else { 1000.0 };
        for s in min_h_sum..=max_h_sum {
            let act_val = c19_act(s as f32 / scale1);
            act_lut[(s - min_h_sum) as usize] = (act_val * act_scale).round().clamp(-32000.0, 32000.0) as i16;
        }

        FrozenRouter {
            w1_q, w2_q, b1_q, b2_q,
            scale1,
            act_lut,
            act_lut_min: min_h_sum,
        }
    }

    fn forward_int(&self, x: &[f32]) -> Vec<i32> {
        let x_u8: Vec<u8> = x.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).collect();

        let mut h_act = vec![0i16; EP_H_DIM];
        for j in 0..EP_H_DIM {
            let mut sum: i32 = self.b1_q[j] as i32;
            for i in 0..EP_IN_DIM {
                sum += self.w1_q[j * EP_IN_DIM + i] as i32 * x_u8[i] as i32;
            }
            let clamped = sum.clamp(self.act_lut_min, self.act_lut_min + self.act_lut.len() as i32 - 1);
            let idx = (clamped - self.act_lut_min) as usize;
            h_act[j] = if idx < self.act_lut.len() { self.act_lut[idx] } else { 0 };
        }

        let mut out = vec![0i32; EP_OUT_DIM];
        for k in 0..EP_OUT_DIM {
            let mut sum: i32 = self.b2_q[k] as i32;
            for j in 0..EP_H_DIM {
                sum += self.w2_q[k * EP_H_DIM + j] as i32 * h_act[j] as i32;
            }
            out[k] = sum;
        }
        out
    }

    fn route(&self, task: &Task, units: &[UnitState; NUM_UNITS], queue_depth: usize, next_queue_cat: Option<UnitCategory>, ep_net: &EpRouter) -> Option<usize> {
        let x = EpRouter::encode_input(task, units, queue_depth, next_queue_cat);
        let scores = self.forward_int(&x);

        let compat = task.op.compatible_units();
        let mut best_idx: Option<usize> = None;
        let mut best_score = i32::MIN;
        for &u in compat {
            if units[u].is_free() && scores[u] > best_score {
                best_score = scores[u];
                best_idx = Some(u);
            }
        }
        // Fallback: the frozen router uses the same encode as EP, so if EP chose
        // something, this should too. We pass ep_net for encode_input only.
        let _ = ep_net;
        best_idx
    }
}

// ============================================================
// EP Training
// ============================================================

struct TrainingSample {
    input: Vec<f32>,
    target: Vec<f32>, // one-hot: 1.0 for optimal unit, 0.0 for others
}

fn settle_router(
    s_h: &[f32], s_out: &[f32],
    x: &[f32], net: &EpRouter, dt: f32, beta: f32, y: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut new_h = vec![0.0f32; EP_H_DIM];
    for j in 0..EP_H_DIM {
        let mut drive = net.b1[j];
        for i in 0..EP_IN_DIM { drive += net.w1[j * EP_IN_DIM + i] * x[i]; }
        for k in 0..EP_OUT_DIM { drive += net.w2[k * EP_H_DIM + j] * c19_act(s_out[k]); }
        new_h[j] = s_h[j] + dt * (-s_h[j] + drive);
    }

    let mut new_out = vec![0.0f32; EP_OUT_DIM];
    for k in 0..EP_OUT_DIM {
        let mut drive = net.b2[k];
        for j in 0..EP_H_DIM { drive += net.w2[k * EP_H_DIM + j] * c19_act(s_h[j]); }
        let nudge = beta * (y[k] - c19_act(s_out[k]));
        new_out[k] = s_out[k] + dt * (-s_out[k] + drive + nudge);
    }

    (new_h, new_out)
}

/// Optimal hindsight scheduling:
/// For ARITH tasks (which have 2 compatible units), pick the unit with
/// fewer tasks completed so far (load balance). For single-unit ops,
/// just pick that unit if free.
/// If we also have lookahead into the queue, we can choose the unit
/// whose remaining ticks free up soonest.
fn optimal_unit_choice(task: &Task, units: &[UnitState; NUM_UNITS]) -> Option<usize> {
    let compat = task.op.compatible_units();
    let mut best: Option<usize> = None;
    let mut best_score = i32::MAX;
    for &u in compat {
        if units[u].is_free() {
            // Score = tasks_completed (fewer = more balanced)
            let score = units[u].tasks_completed as i32;
            if score < best_score {
                best_score = score;
                best = Some(u);
            }
        }
    }
    best
}

fn generate_training_data(
    tasks: &[Task],
    _net: &EpRouter,
) -> Vec<TrainingSample> {
    let mut samples = Vec::new();
    let mut units = [UnitState::new(), UnitState::new(), UnitState::new(), UnitState::new(), UnitState::new()];
    let mut queue: Vec<Task> = Vec::new();

    for task in tasks {
        queue.push(task.clone());

        // Try to dispatch from queue
        loop {
            if queue.is_empty() { break; }

            // Find a task in queue that can be dispatched
            let mut dispatched = false;
            for qi in 0..queue.len().min(8) {
                let t = &queue[qi];
                if let Some(opt_unit) = optimal_unit_choice(t, &units) {
                    let queue_depth = queue.len();
                    let next_cat = if qi + 1 < queue.len() { Some(queue[qi + 1].op.category()) } else { None };
                    let input = EpRouter::encode_input(t, &units, queue_depth, next_cat);
                    let mut target = vec![0.0f32; NUM_UNITS];
                    target[opt_unit] = 1.0;
                    samples.push(TrainingSample { input, target });

                    let t_removed = queue.remove(qi);
                    units[opt_unit].assign(t_removed);
                    dispatched = true;
                    break;
                }
            }
            if !dispatched { break; }
        }

        // Tick all units
        for u in units.iter_mut() { u.tick(); }
    }

    samples
}

fn train_ep_router(
    net: &mut EpRouter,
    samples: &[TrainingSample],
    beta: f32, t_max: usize, dt: f32, lr: f32, n_epochs: usize,
    rng: &mut Rng,
    logf: &mut std::fs::File,
) {
    // Subsample for speed: use at most 2000 samples per epoch
    let max_per_epoch = 2000;
    let use_n = samples.len().min(max_per_epoch);
    let mut indices: Vec<usize> = (0..samples.len()).collect();

    for epoch in 0..n_epochs {
        let lr_eff = if epoch < 20 { lr * (epoch as f32 + 1.0) / 20.0 } else { lr };
        rng.shuffle(&mut indices);

        for si in 0..use_n {
            let idx = indices[si];
            let x = &samples[idx].input;
            let y = &samples[idx].target;

            // Free phase
            let mut s_h = vec![0.0f32; EP_H_DIM];
            let mut s_out = vec![0.0f32; EP_OUT_DIM];
            for _ in 0..t_max {
                let (nh, no) = settle_router(&s_h, &s_out, x, net, dt, 0.0, y);
                s_h = nh; s_out = no;
            }
            let s_free_h = s_h;
            let s_free_out = s_out;

            // Nudged phase
            let mut s_h = s_free_h.clone();
            let mut s_out = s_free_out.clone();
            for _ in 0..t_max {
                let (nh, no) = settle_router(&s_h, &s_out, x, net, dt, beta, y);
                s_h = nh; s_out = no;
            }
            let s_nudge_h = s_h;
            let s_nudge_out = s_out;

            // Weight update (EP: += not -=)
            let inv_beta = 1.0 / beta;

            for j in 0..EP_H_DIM {
                let a_n = c19_act(s_nudge_h[j]);
                let a_f = c19_act(s_free_h[j]);
                if (a_n - a_f).abs() < 1e-8 { continue; } // skip dead neurons
                for i in 0..EP_IN_DIM {
                    let delta = lr_eff * inv_beta * (a_n * x[i] - a_f * x[i]);
                    if delta.is_finite() {
                        net.w1[j * EP_IN_DIM + i] += delta;
                    }
                }
                let delta_b = lr_eff * inv_beta * (a_n - a_f);
                if delta_b.is_finite() {
                    net.b1[j] += delta_b;
                }
            }

            for k in 0..EP_OUT_DIM {
                let ao_n = c19_act(s_nudge_out[k]);
                let ao_f = c19_act(s_free_out[k]);
                if (ao_n - ao_f).abs() < 1e-8 { continue; }
                for j in 0..EP_H_DIM {
                    let ah_n = c19_act(s_nudge_h[j]);
                    let ah_f = c19_act(s_free_h[j]);
                    let delta = lr_eff * inv_beta * (ao_n * ah_n - ao_f * ah_f);
                    if delta.is_finite() {
                        net.w2[k * EP_H_DIM + j] += delta;
                    }
                }
                let delta_b = lr_eff * inv_beta * (ao_n - ao_f);
                if delta_b.is_finite() {
                    net.b2[k] += delta_b;
                }
            }
        }

        // Clamp weights to prevent divergence (per epoch, not per sample)
        let wclamp = 10.0;
        for w in net.w1.iter_mut() { *w = w.clamp(-wclamp, wclamp); }
        for w in net.w2.iter_mut() { *w = w.clamp(-wclamp, wclamp); }
        for b in net.b1.iter_mut() { *b = b.clamp(-wclamp, wclamp); }
        for b in net.b2.iter_mut() { *b = b.clamp(-wclamp, wclamp); }

        // Log progress
        if epoch % 50 == 0 || epoch == n_epochs - 1 {
            let mut correct = 0;
            let mut correct_compat = 0;
            let eval_n = samples.len().min(1000);
            for s in &samples[..eval_n] {
                let scores = net.forward(&s.input);
                let target_unit = s.target.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap_or(0);
                let pred_unit = scores.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap_or(0);
                if pred_unit == target_unit { correct += 1; }
                if unit_category(pred_unit) == unit_category(target_unit) { correct_compat += 1; }
            }
            let acc = correct as f32 / eval_n as f32 * 100.0;
            let compat_acc = correct_compat as f32 / eval_n as f32 * 100.0;
            let msg = format!("    epoch {:>4}: exact={:.1}%  compat={:.1}%  ({}/{})",
                epoch, acc, compat_acc, correct, eval_n);
            log(logf, &msg);
        }
    }
}

// ============================================================
// Simulation engine
// ============================================================

enum RouterType<'a> {
    Static,
    PriorityQueue,
    EpFloat(&'a EpRouter),
    EpFrozen(&'a FrozenRouter, &'a EpRouter),
}

fn simulate(
    tasks: &[Task],
    router: &RouterType,
    arrival_rate: usize,
) -> SimMetrics {
    let mut units = [UnitState::new(), UnitState::new(), UnitState::new(), UnitState::new(), UnitState::new()];
    let mut metrics = SimMetrics::new(tasks.len());
    let mut completed_results: Vec<Option<u8>> = vec![None; tasks.len()];
    let mut queue: Vec<Task> = Vec::new();
    let mut next_task_idx = 0;
    let mut tick: usize = 0;

    // Pre-fill arrival ticks
    for i in 0..tasks.len() {
        metrics.task_arrival_tick[i] = i / arrival_rate;
    }

    loop {
        // Inject tasks arriving this tick
        for _ in 0..arrival_rate {
            if next_task_idx < tasks.len() {
                queue.push(tasks[next_task_idx].clone());
                next_task_idx += 1;
            }
        }

        // Dispatch loop — try to dispatch as many as possible
        let mut dispatched_this_tick = false;
        loop {
            if queue.is_empty() { break; }

            match router {
                RouterType::Static => {
                    let front = &queue[0];
                    if let Some(u) = static_dispatch(front, &units) {
                        // Verify compatibility (safety check)
                        if !is_compatible(&queue[0].op, u) {
                            metrics.wrong_unit_errors += 1;
                        }
                        let task = queue.remove(0);
                        metrics.tasks_per_unit[u] += 1;
                        units[u].assign(task);
                        dispatched_this_tick = true;
                        continue;
                    }
                    break;
                },
                RouterType::PriorityQueue => {
                    match priority_queue_dispatch(&queue, &units) {
                        DispatchResult::Front(u) => {
                            if !is_compatible(&queue[0].op, u) {
                                metrics.wrong_unit_errors += 1;
                            }
                            let task = queue.remove(0);
                            metrics.tasks_per_unit[u] += 1;
                            units[u].assign(task);
                            dispatched_this_tick = true;
                            continue;
                        },
                        DispatchResult::Reorder { unit, queue_index } => {
                            if !is_compatible(&queue[queue_index].op, unit) {
                                metrics.wrong_unit_errors += 1;
                            }
                            let task = queue.remove(queue_index);
                            metrics.tasks_per_unit[unit] += 1;
                            units[unit].assign(task);
                            metrics.reorder_count += 1;
                            dispatched_this_tick = true;
                            continue;
                        },
                        DispatchResult::Stall => break,
                    }
                },
                RouterType::EpFloat(ep_net) => {
                    // EP router can also reorder: try front first, then scan
                    let mut found = false;
                    for qi in 0..queue.len().min(8) {
                        let task = &queue[qi];
                        let queue_depth = queue.len();
                        let next_cat = if qi + 1 < queue.len() { Some(queue[qi + 1].op.category()) } else { None };
                        if let Some(u) = ep_net.route(task, &units, queue_depth, next_cat) {
                            if !is_compatible(&queue[qi].op, u) {
                                metrics.wrong_unit_errors += 1;
                            }
                            if qi > 0 { metrics.reorder_count += 1; }
                            let task = queue.remove(qi);
                            metrics.tasks_per_unit[u] += 1;
                            units[u].assign(task);
                            dispatched_this_tick = true;
                            found = true;
                            break;
                        }
                    }
                    if !found { break; }
                },
                RouterType::EpFrozen(frozen, ep_net) => {
                    let mut found = false;
                    for qi in 0..queue.len().min(8) {
                        let task = &queue[qi];
                        let queue_depth = queue.len();
                        let next_cat = if qi + 1 < queue.len() { Some(queue[qi + 1].op.category()) } else { None };
                        if let Some(u) = frozen.route(task, &units, queue_depth, next_cat, ep_net) {
                            if !is_compatible(&queue[qi].op, u) {
                                metrics.wrong_unit_errors += 1;
                            }
                            if qi > 0 { metrics.reorder_count += 1; }
                            let task = queue.remove(qi);
                            metrics.tasks_per_unit[u] += 1;
                            units[u].assign(task);
                            dispatched_this_tick = true;
                            found = true;
                            break;
                        }
                    }
                    if !found { break; }
                },
            }
        }

        if !dispatched_this_tick && !queue.is_empty() {
            metrics.stall_ticks += 1;
        }

        // Record busy state before ticking
        for i in 0..NUM_UNITS {
            if !units[i].is_free() {
                metrics.unit_busy_ticks[i] += 1;
            }
        }

        // Tick all units
        for i in 0..NUM_UNITS {
            if let Some((task_id, result)) = units[i].tick() {
                metrics.tasks_completed += 1;
                completed_results[task_id] = Some(result);
                if task_id < metrics.task_arrival_tick.len() {
                    let lat = tick + 1 - metrics.task_arrival_tick[task_id];
                    metrics.task_latencies.push(lat);
                }
            }
        }

        tick += 1;
        metrics.total_ticks = tick;

        if next_task_idx >= tasks.len() && queue.is_empty()
            && units.iter().all(|u| u.is_free())
        {
            break;
        }

        if tick > tasks.len() * 20 + 200 {
            break;
        }
    }

    // Verify all task results
    for task in tasks {
        let expected = task.op.execute(task.a, task.b);
        if let Some(got) = completed_results[task.id] {
            if got != expected {
                metrics.wrong_unit_errors += 100; // flag as severe
            }
        }
    }

    metrics
}

// ============================================================
// Simulate a 1x full ALU baseline (sequential)
// All ops on one unit, ticks = sum of all task ticks
// ============================================================

fn simulate_sequential_alu(tasks: &[Task]) -> SimMetrics {
    let mut metrics = SimMetrics::new(tasks.len());
    let mut tick = 0usize;
    for (i, task) in tasks.iter().enumerate() {
        metrics.task_arrival_tick[i] = i; // one task per tick arrival
        let lat = task.op.ticks();
        tick += lat;
        metrics.tasks_completed += 1;
        metrics.task_latencies.push(lat + i); // queue wait + execution
    }
    metrics.total_ticks = tick;
    // Single unit utilization = always 100%
    metrics.unit_busy_ticks[0] = tick;
    metrics
}

// ============================================================
// Workload generators
// ============================================================

fn workload_w1_balanced(n: usize, rng: &mut Rng) -> Vec<Task> {
    // 20% ADD, 10% SUB, 15% MUL, 15% AND, 10% OR, 10% XOR, 5% NOT, 10% CMP, 5% MIN
    (0..n).map(|id| {
        let r = rng.f32();
        let op = if r < 0.20      { OpType::Add }
            else if r < 0.30      { OpType::Sub }
            else if r < 0.45      { OpType::Mul }
            else if r < 0.60      { OpType::And }
            else if r < 0.70      { OpType::Or }
            else if r < 0.80      { OpType::Xor }
            else if r < 0.85      { OpType::Not }
            else if r < 0.95      { OpType::Cmp }
            else                  { OpType::Min };
        Task { id, op, a: (rng.next() & 0xFF) as u8, b: (rng.next() & 0xFF) as u8 }
    }).collect()
}

fn workload_w2_arith_heavy(n: usize, rng: &mut Rng) -> Vec<Task> {
    // 35% ADD, 25% SUB, 20% MUL, 5% AND, 5% OR, 5% XOR, 5% CMP
    (0..n).map(|id| {
        let r = rng.f32();
        let op = if r < 0.35      { OpType::Add }
            else if r < 0.60      { OpType::Sub }
            else if r < 0.80      { OpType::Mul }
            else if r < 0.85      { OpType::And }
            else if r < 0.90      { OpType::Or }
            else if r < 0.95      { OpType::Xor }
            else                  { OpType::Cmp };
        Task { id, op, a: (rng.next() & 0xFF) as u8, b: (rng.next() & 0xFF) as u8 }
    }).collect()
}

fn workload_w3_mul_bottleneck(n: usize, rng: &mut Rng) -> Vec<Task> {
    // 10% ADD, 10% SUB, 50% MUL, 10% AND, 10% OR, 5% CMP, 5% MIN
    (0..n).map(|id| {
        let r = rng.f32();
        let op = if r < 0.10      { OpType::Add }
            else if r < 0.20      { OpType::Sub }
            else if r < 0.70      { OpType::Mul }
            else if r < 0.80      { OpType::And }
            else if r < 0.90      { OpType::Or }
            else if r < 0.95      { OpType::Cmp }
            else                  { OpType::Min };
        Task { id, op, a: (rng.next() & 0xFF) as u8, b: (rng.next() & 0xFF) as u8 }
    }).collect()
}

fn workload_w4_burst_shift(n: usize, rng: &mut Rng) -> Vec<Task> {
    // First half: 80% logic, 20% other
    // Second half: 80% arithmetic, 20% other
    let half = n / 2;
    (0..n).map(|id| {
        let r = rng.f32();
        let op = if id < half {
            // Logic burst
            if r < 0.30      { OpType::And }
            else if r < 0.55 { OpType::Or }
            else if r < 0.80 { OpType::Xor }
            else if r < 0.90 { OpType::Add }
            else              { OpType::Mul }
        } else {
            // Arithmetic burst
            if r < 0.35      { OpType::Add }
            else if r < 0.60 { OpType::Sub }
            else if r < 0.80 { OpType::Mul }
            else if r < 0.90 { OpType::And }
            else              { OpType::Cmp }
        };
        Task { id, op, a: (rng.next() & 0xFF) as u8, b: (rng.next() & 0xFF) as u8 }
    }).collect()
}

fn workload_w5_adversarial(n: usize, _rng: &mut Rng) -> Vec<Task> {
    // Alternating: MUL, CMP, MUL, CMP, ...
    // Both slow units constantly busy, fast units idle
    (0..n).map(|id| {
        let op = if id % 2 == 0 { OpType::Mul } else { OpType::Cmp };
        Task { id, op, a: (id * 7 + 1) as u8, b: (id * 3 + 2) as u8 }
    }).collect()
}

// ============================================================
// Logging
// ============================================================

fn log(f: &mut std::fs::File, msg: &str) {
    let line = format!("{}\n", msg);
    print!("{}", line);
    f.write_all(line.as_bytes()).ok();
    f.flush().ok();
}

fn fmt_row(name: &str, m: &SimMetrics) -> String {
    format!("  {:<22} tput={:.3}  util={:.1}%  stall={:.1}%  lat={:.1}  reorder={}  errs={}  ({} tasks, {} ticks)",
        name,
        m.throughput(),
        m.utilization() * 100.0,
        m.stall_rate() * 100.0,
        m.avg_latency(),
        m.reorder_count,
        m.wrong_unit_errors,
        m.tasks_completed,
        m.total_ticks,
    )
}

// ============================================================
// MAIN
// ============================================================

fn main() {
    let log_path = "S:/Git/VRAXION/.claude/research/swarm_log.txt";
    let mut logf = std::fs::File::create(log_path).unwrap();

    let t0 = std::time::Instant::now();

    log(&mut logf, "================================================================");
    log(&mut logf, "  Heterogeneous Multi-Unit Processor");
    log(&mut logf, "  5 specialized units + EP-trained neural router");
    log(&mut logf, "================================================================");
    log(&mut logf, "");
    log(&mut logf, "  Architecture:");
    log(&mut logf, "    Unit 0: ARITH_A  (ADD, SUB)          40 neurons, 1 tick");
    log(&mut logf, "    Unit 1: ARITH_B  (ADD, SUB)          40 neurons, 1 tick");
    log(&mut logf, "    Unit 2: MUL      (MUL only)         176 neurons, 3 ticks");
    log(&mut logf, "    Unit 3: LOGIC    (AND, OR, XOR, NOT)  32 neurons, 1 tick");
    log(&mut logf, "    Unit 4: CMP      (CMP, MIN, MAX)     80 neurons, 2 ticks");
    log(&mut logf, &format!("    Total compute neurons: {}", TOTAL_COMPUTE_NEURONS));
    log(&mut logf, "");

    let mut rng = Rng::new(42);
    let n_tasks = 100;
    let arrival_rate = 3;

    // ================================================================
    // Phase 1: Generate training data with optimal hindsight scheduling
    // ================================================================
    log(&mut logf, "--- PHASE 1: Generate training data ---");
    log(&mut logf, &format!("  Generating diverse training data: 10 seq x 5 workload types = 50 sequences"));

    let mut all_training_tasks: Vec<Vec<Task>> = Vec::new();
    // Use all 5 workload types for training diversity
    for seq in 0..10 {
        let s = 1000 + seq as u64;
        all_training_tasks.push(workload_w1_balanced(n_tasks, &mut Rng::new(s)));
        all_training_tasks.push(workload_w2_arith_heavy(n_tasks, &mut Rng::new(s + 100)));
        all_training_tasks.push(workload_w3_mul_bottleneck(n_tasks, &mut Rng::new(s + 200)));
        all_training_tasks.push(workload_w4_burst_shift(n_tasks, &mut Rng::new(s + 300)));
        all_training_tasks.push(workload_w5_adversarial(n_tasks, &mut Rng::new(s + 400)));
    }

    let mut ep_net = EpRouter::new(&mut rng);
    log(&mut logf, &format!("  EP Router: input={}, hidden={}, output={}", EP_IN_DIM, EP_H_DIM, EP_OUT_DIM));
    let ep_router_neurons = EP_IN_DIM + EP_H_DIM; // ~42 virtual neurons (26 input + 16 hidden)

    // Generate training samples
    let mut all_samples: Vec<TrainingSample> = Vec::new();
    for seq_tasks in &all_training_tasks {
        let samples = generate_training_data(seq_tasks, &ep_net);
        all_samples.extend(samples);
    }
    log(&mut logf, &format!("  Total training samples: {}", all_samples.len()));

    // Count target distribution
    let mut target_dist = [0usize; NUM_UNITS];
    for s in &all_samples {
        let target_unit = s.target.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
        target_dist[target_unit] += 1;
    }
    log(&mut logf, &format!("  Target distribution: ARITH_A={} ARITH_B={} MUL={} LOGIC={} CMP={}",
        target_dist[0], target_dist[1], target_dist[2], target_dist[3], target_dist[4]));
    log(&mut logf, "");

    // ================================================================
    // Phase 2: Train EP router
    // ================================================================
    log(&mut logf, "--- PHASE 2: Train EP router ---");
    log(&mut logf, "  beta=0.5, T=20, dt=0.5, lr=0.01, epochs=200");

    train_ep_router(
        &mut ep_net,
        &all_samples,
        0.5,   // beta
        20,    // t_max (reduced for speed + stability)
        0.5,   // dt
        0.01,  // lr (conservative for 10k samples)
        200,   // epochs
        &mut rng,
        &mut logf,
    );

    let train_time = t0.elapsed().as_secs_f32();
    log(&mut logf, &format!("  Training complete in {:.1}s", train_time));
    log(&mut logf, "");

    // ================================================================
    // Phase 3: Freeze to int8 LUT
    // ================================================================
    log(&mut logf, "--- PHASE 3: Freeze to int8 LUT ---");
    let frozen = FrozenRouter::from_ep(&ep_net);
    log(&mut logf, &format!("  W1: {} params, W2: {} params", frozen.w1_q.len(), frozen.w2_q.len()));
    log(&mut logf, &format!("  Activation LUT: {} entries", frozen.act_lut.len()));

    // Verify frozen matches float
    let mut match_count = 0;
    let check_n = all_samples.len().min(500);
    for s in &all_samples[..check_n] {
        let float_scores = ep_net.forward(&s.input);
        let frozen_scores = frozen.forward_int(&s.input);
        let float_best = float_scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| i).unwrap_or(0);
        let frozen_best = frozen_scores.iter().enumerate()
            .max_by(|a, b| a.1.cmp(b.1)).map(|(i, _)| i).unwrap();
        if float_best == frozen_best { match_count += 1; }
    }
    let freeze_match = match_count as f32 / check_n as f32 * 100.0;
    log(&mut logf, &format!("  Float vs Frozen routing agreement: {:.1}% ({}/{})",
        freeze_match, match_count, check_n));
    log(&mut logf, "");

    // ================================================================
    // Phase 4: Evaluate on 5 workloads
    // ================================================================
    log(&mut logf, "--- PHASE 4: Evaluate on 5 workloads ---");
    log(&mut logf, &format!("  {} tasks per workload, arrival_rate = {}", n_tasks, arrival_rate));
    log(&mut logf, "");

    let workload_names = ["W1: Balanced", "W2: Arith Heavy", "W3: MUL Bottlnk", "W4: Burst Shift", "W5: Adversarial"];

    struct WorkloadResults {
        name: String,
        static_m: SimMetrics,
        pq_m: SimMetrics,
        ep_m: SimMetrics,
        frozen_m: SimMetrics,
    }

    let mut all_results: Vec<WorkloadResults> = Vec::new();

    for (wi, wname) in workload_names.iter().enumerate() {
        let tasks = match wi {
            0 => workload_w1_balanced(n_tasks, &mut Rng::new(777)),
            1 => workload_w2_arith_heavy(n_tasks, &mut Rng::new(777)),
            2 => workload_w3_mul_bottleneck(n_tasks, &mut Rng::new(777)),
            3 => workload_w4_burst_shift(n_tasks, &mut Rng::new(777)),
            4 => workload_w5_adversarial(n_tasks, &mut Rng::new(777)),
            _ => unreachable!(),
        };

        // Count ops per category
        let mut cat_counts = [0usize; 4];
        for t in &tasks {
            match t.op.category() {
                UnitCategory::Arith => cat_counts[0] += 1,
                UnitCategory::Mul   => cat_counts[1] += 1,
                UnitCategory::Logic => cat_counts[2] += 1,
                UnitCategory::Cmp   => cat_counts[3] += 1,
            }
        }
        log(&mut logf, &format!("  === {} === (arith={} mul={} logic={} cmp={})",
            wname, cat_counts[0], cat_counts[1], cat_counts[2], cat_counts[3]));

        // Router A: Static Dispatch
        let m_static = simulate(&tasks, &RouterType::Static, arrival_rate);
        log(&mut logf, &fmt_row("Static", &m_static));

        // Router B: Priority Queue
        let m_pq = simulate(&tasks, &RouterType::PriorityQueue, arrival_rate);
        log(&mut logf, &fmt_row("Priority-Q", &m_pq));

        // Router C: EP float
        let m_ep = simulate(&tasks, &RouterType::EpFloat(&ep_net), arrival_rate);
        log(&mut logf, &fmt_row("EP Neural", &m_ep));

        // Router D: EP frozen
        let m_frozen = simulate(&tasks, &RouterType::EpFrozen(&frozen, &ep_net), arrival_rate);
        log(&mut logf, &fmt_row("EP Frozen", &m_frozen));

        // Per-unit utilization for EP
        let util = m_ep.per_unit_utilization();
        log(&mut logf, &format!("    EP unit util: A={:.0}% B={:.0}% MUL={:.0}% LOG={:.0}% CMP={:.0}%",
            util[0]*100.0, util[1]*100.0, util[2]*100.0, util[3]*100.0, util[4]*100.0));

        log(&mut logf, "");

        all_results.push(WorkloadResults {
            name: wname.to_string(),
            static_m: m_static,
            pq_m: m_pq,
            ep_m: m_ep,
            frozen_m: m_frozen,
        });
    }

    // ================================================================
    // Phase 5: Adversarial safety checks
    // ================================================================
    log(&mut logf, "--- PHASE 5: Adversarial safety checks ---");

    let mut total_errors = 0;
    for wr in &all_results {
        total_errors += wr.static_m.wrong_unit_errors;
        total_errors += wr.pq_m.wrong_unit_errors;
        total_errors += wr.ep_m.wrong_unit_errors;
        total_errors += wr.frozen_m.wrong_unit_errors;
    }
    log(&mut logf, &format!("  Wrong-unit dispatch errors: {}", total_errors));

    // Verify all tasks completed with correct results
    let mut completion_ok = true;
    for wr in &all_results {
        if wr.static_m.tasks_completed != n_tasks { completion_ok = false; }
        if wr.pq_m.tasks_completed != n_tasks { completion_ok = false; }
        if wr.ep_m.tasks_completed != n_tasks { completion_ok = false; }
        if wr.frozen_m.tasks_completed != n_tasks { completion_ok = false; }
    }
    log(&mut logf, &format!("  All tasks completed: {}", if completion_ok { "YES" } else { "NO -- TASKS DROPPED" }));

    // Check freeze preserves routing quality
    let mut freeze_ok = true;
    for wr in &all_results {
        let ep_tput = wr.ep_m.throughput();
        let fr_tput = wr.frozen_m.throughput();
        let diff = (ep_tput - fr_tput).abs() / ep_tput.max(0.001);
        if diff > 0.15 { freeze_ok = false; } // allow 15% degradation
    }
    log(&mut logf, &format!("  Freeze preserves quality (<15% throughput loss): {}", if freeze_ok { "YES" } else { "NO" }));
    log(&mut logf, "");

    // ================================================================
    // Phase 6: Comparison table
    // ================================================================
    log(&mut logf, "================================================================");
    log(&mut logf, "  COMPARISON TABLE");
    log(&mut logf, "================================================================");
    log(&mut logf, "");
    log(&mut logf, "                         Static    Priority-Q   EP Neural    EP Frozen");
    log(&mut logf, "  -------------------------------------------------------------------------");

    let mut avg_tput = [0.0f32; 4];
    let mut avg_stall = [0.0f32; 4];
    let mut avg_util = [0.0f32; 4];
    let mut avg_lat = [0.0f32; 4];
    let mut avg_reorder = [0.0f32; 4];

    for wr in &all_results {
        let ms = [&wr.static_m, &wr.pq_m, &wr.ep_m, &wr.frozen_m];
        log(&mut logf, &format!("  {:<20} tput: {:>6.3}     {:>6.3}       {:>6.3}       {:>6.3}",
            wr.name, ms[0].throughput(), ms[1].throughput(), ms[2].throughput(), ms[3].throughput()));
        for r in 0..4 {
            avg_tput[r] += ms[r].throughput();
            avg_stall[r] += ms[r].stall_rate();
            avg_util[r] += ms[r].utilization();
            avg_lat[r] += ms[r].avg_latency();
            avg_reorder[r] += ms[r].reorder_count as f32;
        }
    }
    let nw = all_results.len() as f32;
    log(&mut logf, "  -------------------------------------------------------------------------");
    log(&mut logf, &format!("  {:<20} tput: {:>6.3}     {:>6.3}       {:>6.3}       {:>6.3}",
        "AVERAGE", avg_tput[0]/nw, avg_tput[1]/nw, avg_tput[2]/nw, avg_tput[3]/nw));
    log(&mut logf, &format!("  {:<20} util: {:>5.1}%     {:>5.1}%       {:>5.1}%       {:>5.1}%",
        "", avg_util[0]/nw*100.0, avg_util[1]/nw*100.0, avg_util[2]/nw*100.0, avg_util[3]/nw*100.0));
    log(&mut logf, &format!("  {:<20} stall:{:>5.1}%     {:>5.1}%       {:>5.1}%       {:>5.1}%",
        "", avg_stall[0]/nw*100.0, avg_stall[1]/nw*100.0, avg_stall[2]/nw*100.0, avg_stall[3]/nw*100.0));
    log(&mut logf, &format!("  {:<20} lat:  {:>5.1}      {:>5.1}        {:>5.1}        {:>5.1}",
        "", avg_lat[0]/nw, avg_lat[1]/nw, avg_lat[2]/nw, avg_lat[3]/nw));
    log(&mut logf, &format!("  {:<20} reord:{:>5.0}      {:>5.0}        {:>5.0}        {:>5.0}",
        "", avg_reorder[0]/nw, avg_reorder[1]/nw, avg_reorder[2]/nw, avg_reorder[3]/nw));
    log(&mut logf, "");

    // ================================================================
    // Phase 7: Cost efficiency comparison
    // ================================================================
    log(&mut logf, "================================================================");
    log(&mut logf, "  COST EFFICIENCY ANALYSIS");
    log(&mut logf, "================================================================");
    log(&mut logf, "");

    // Simulate 1x full ALU baseline on each workload
    let full_alu_neurons = 394; // from alu8bit.rs
    let four_alu_neurons = full_alu_neurons * 4; // 1576

    let mut seq_avg_tput = 0.0f32;
    log(&mut logf, "  1x Full ALU (sequential):");
    for (wi, wname) in workload_names.iter().enumerate() {
        let tasks = match wi {
            0 => workload_w1_balanced(n_tasks, &mut Rng::new(777)),
            1 => workload_w2_arith_heavy(n_tasks, &mut Rng::new(777)),
            2 => workload_w3_mul_bottleneck(n_tasks, &mut Rng::new(777)),
            3 => workload_w4_burst_shift(n_tasks, &mut Rng::new(777)),
            4 => workload_w5_adversarial(n_tasks, &mut Rng::new(777)),
            _ => unreachable!(),
        };
        let m = simulate_sequential_alu(&tasks);
        log(&mut logf, &format!("    {}: tput={:.3} ({} tasks, {} ticks)",
            wname, m.throughput(), m.tasks_completed, m.total_ticks));
        seq_avg_tput += m.throughput();
    }
    seq_avg_tput /= nw;
    log(&mut logf, &format!("    AVERAGE throughput: {:.3}", seq_avg_tput));
    log(&mut logf, "");

    // For 4x full ALU (homogeneous), simulate with round-robin on 4 identical units
    // that all accept every op. We approximate by using the parallel_alu-style sim.
    log(&mut logf, "  4x Full ALU (homogeneous round-robin, simulated):");
    let mut homo_avg_tput = 0.0f32;
    for (wi, wname) in workload_names.iter().enumerate() {
        let tasks = match wi {
            0 => workload_w1_balanced(n_tasks, &mut Rng::new(777)),
            1 => workload_w2_arith_heavy(n_tasks, &mut Rng::new(777)),
            2 => workload_w3_mul_bottleneck(n_tasks, &mut Rng::new(777)),
            3 => workload_w4_burst_shift(n_tasks, &mut Rng::new(777)),
            4 => workload_w5_adversarial(n_tasks, &mut Rng::new(777)),
            _ => unreachable!(),
        };
        // Simulate 4 homogeneous units: each accepts any op
        let m = simulate_homogeneous_4x(&tasks, arrival_rate);
        log(&mut logf, &format!("    {}: tput={:.3} ({} tasks, {} ticks)",
            wname, m.throughput(), m.tasks_completed, m.total_ticks));
        homo_avg_tput += m.throughput();
    }
    homo_avg_tput /= nw;
    log(&mut logf, &format!("    AVERAGE throughput: {:.3}", homo_avg_tput));
    log(&mut logf, "");

    // Cost efficiency table
    let hetero_static_neurons = TOTAL_COMPUTE_NEURONS; // 368, router = 0
    let hetero_pq_neurons = TOTAL_COMPUTE_NEURONS + 30; // ~30 for reorder buffer
    let hetero_ep_neurons = TOTAL_COMPUTE_NEURONS + ep_router_neurons; // 368 + ~42

    let avg_static_tput = avg_tput[0] / nw;
    let avg_pq_tput = avg_tput[1] / nw;
    let avg_ep_tput = avg_tput[2] / nw;
    let avg_frozen_tput = avg_tput[3] / nw;

    log(&mut logf, "  COST EFFICIENCY: throughput / neurons * 1000");
    log(&mut logf, "  ─────────────────────────────────────────────────────────");
    log(&mut logf, &format!("  1x Full ALU (seq):    {:.3} / {:>5} * 1000 = {:.2}",
        seq_avg_tput, full_alu_neurons,
        seq_avg_tput / full_alu_neurons as f32 * 1000.0));
    log(&mut logf, &format!("  4x Full ALU (homo):   {:.3} / {:>5} * 1000 = {:.2}",
        homo_avg_tput, four_alu_neurons,
        homo_avg_tput / four_alu_neurons as f32 * 1000.0));
    log(&mut logf, &format!("  Hetero (Static):      {:.3} / {:>5} * 1000 = {:.2}",
        avg_static_tput, hetero_static_neurons,
        avg_static_tput / hetero_static_neurons as f32 * 1000.0));
    log(&mut logf, &format!("  Hetero (Priority-Q):  {:.3} / {:>5} * 1000 = {:.2}",
        avg_pq_tput, hetero_pq_neurons,
        avg_pq_tput / hetero_pq_neurons as f32 * 1000.0));
    log(&mut logf, &format!("  Hetero (EP Neural):   {:.3} / {:>5} * 1000 = {:.2}",
        avg_ep_tput, hetero_ep_neurons,
        avg_ep_tput / hetero_ep_neurons as f32 * 1000.0));
    log(&mut logf, &format!("  Hetero (EP Frozen):   {:.3} / {:>5} * 1000 = {:.2}",
        avg_frozen_tput, hetero_ep_neurons,
        avg_frozen_tput / hetero_ep_neurons as f32 * 1000.0));
    log(&mut logf, "");

    // Determine champion
    let configs = [
        ("1x Full ALU", seq_avg_tput / full_alu_neurons as f32 * 1000.0),
        ("4x Full ALU", homo_avg_tput / four_alu_neurons as f32 * 1000.0),
        ("Hetero Static", avg_static_tput / hetero_static_neurons as f32 * 1000.0),
        ("Hetero Priority-Q", avg_pq_tput / hetero_pq_neurons as f32 * 1000.0),
        ("Hetero EP Neural", avg_ep_tput / hetero_ep_neurons as f32 * 1000.0),
        ("Hetero EP Frozen", avg_frozen_tput / hetero_ep_neurons as f32 * 1000.0),
    ];
    let champion = configs.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    log(&mut logf, &format!("  CHAMPION: {} (efficiency = {:.2})", champion.0, champion.1));

    // Check if hetero+EP beats 4x full
    let hetero_ep_eff = avg_ep_tput / hetero_ep_neurons as f32 * 1000.0;
    let four_alu_eff = homo_avg_tput / four_alu_neurons as f32 * 1000.0;
    if hetero_ep_eff > four_alu_eff {
        log(&mut logf, &format!("  Hetero+EP ({:.2}) BEATS 4x Full ALU ({:.2}) by {:.1}%",
            hetero_ep_eff, four_alu_eff,
            (hetero_ep_eff - four_alu_eff) / four_alu_eff * 100.0));
        log(&mut logf, "  >>> SPECIALIZATION + LEARNED ROUTING WINS <<<");
    } else {
        log(&mut logf, &format!("  Hetero+EP ({:.2}) vs 4x Full ALU ({:.2}) -- diff = {:.1}%",
            hetero_ep_eff, four_alu_eff,
            (hetero_ep_eff - four_alu_eff) / four_alu_eff * 100.0));
    }
    log(&mut logf, "");

    // ================================================================
    // Phase 8: Key findings
    // ================================================================
    log(&mut logf, "================================================================");
    log(&mut logf, "  KEY FINDINGS");
    log(&mut logf, "================================================================");
    log(&mut logf, "");

    // Q1: Does EP learn to spread ADD/SUB across both ARITH units?
    let w2_tasks = workload_w2_arith_heavy(n_tasks, &mut Rng::new(777));
    let w2_ep = simulate(&w2_tasks, &RouterType::EpFloat(&ep_net), arrival_rate);
    let a_tasks = w2_ep.tasks_per_unit[0];
    let b_tasks = w2_ep.tasks_per_unit[1];
    let balance_ratio = if a_tasks.max(b_tasks) > 0 { a_tasks.min(b_tasks) as f32 / a_tasks.max(b_tasks) as f32 } else { 0.0 };
    log(&mut logf, &format!("  Q1: EP spreads ARITH across both units?"));
    log(&mut logf, &format!("      W2: ARITH_A={} tasks, ARITH_B={} tasks, balance={:.1}%",
        a_tasks, b_tasks, balance_ratio * 100.0));
    log(&mut logf, &format!("      {}", if balance_ratio > 0.3 { "YES -- load is distributed" } else { "NO -- one unit dominates" }));
    log(&mut logf, "");

    // Q2: Does EP reorder around MUL bottleneck?
    let w3_pq = &all_results[2].pq_m;
    let w3_ep = &all_results[2].ep_m;
    log(&mut logf, &format!("  Q2: Reordering around MUL bottleneck?"));
    log(&mut logf, &format!("      Priority-Q reorders: {}, EP reorders: {}", w3_pq.reorder_count, w3_ep.reorder_count));
    log(&mut logf, &format!("      PQ throughput: {:.3}, EP throughput: {:.3}", w3_pq.throughput(), w3_ep.throughput()));
    log(&mut logf, "");

    // Q3: Freeze quality
    log(&mut logf, &format!("  Q3: Freeze preserves routing quality?"));
    log(&mut logf, &format!("      Float vs Frozen agreement: {:.1}%", freeze_match));
    for wr in &all_results {
        let diff = ((wr.ep_m.throughput() - wr.frozen_m.throughput()) / wr.ep_m.throughput().max(0.001) * 100.0).abs();
        log(&mut logf, &format!("      {}: float={:.3} frozen={:.3} diff={:.1}%",
            wr.name, wr.ep_m.throughput(), wr.frozen_m.throughput(), diff));
    }
    log(&mut logf, "");

    // Q4: Neuron counts
    log(&mut logf, &format!("  Q4: Neuron budget summary"));
    log(&mut logf, &format!("      1x Full ALU:       {:>5} neurons", full_alu_neurons));
    log(&mut logf, &format!("      4x Full ALU:       {:>5} neurons", four_alu_neurons));
    log(&mut logf, &format!("      Hetero compute:    {:>5} neurons", TOTAL_COMPUTE_NEURONS));
    log(&mut logf, &format!("      EP router:         {:>5} neurons", ep_router_neurons));
    log(&mut logf, &format!("      Hetero+EP total:   {:>5} neurons", hetero_ep_neurons));
    log(&mut logf, &format!("      Savings vs 4x ALU: {:.1}x fewer neurons",
        four_alu_neurons as f32 / hetero_ep_neurons as f32));
    log(&mut logf, "");

    // Total time
    let total_time = t0.elapsed().as_secs_f32();
    log(&mut logf, &format!("  Total runtime: {:.1}s", total_time));
    log(&mut logf, "================================================================");
}

// ============================================================
// Homogeneous 4x ALU simulation (for comparison)
// Each unit accepts ALL operations
// ============================================================

fn simulate_homogeneous_4x(tasks: &[Task], arrival_rate: usize) -> SimMetrics {
    let n_units = 4;
    let mut units = vec![UnitState::new(); n_units];
    let mut metrics = SimMetrics::new(tasks.len());
    let mut queue: Vec<Task> = Vec::new();
    let mut next_task_idx = 0;
    let mut tick: usize = 0;
    let mut rr_last = 0usize;

    for i in 0..tasks.len() {
        metrics.task_arrival_tick[i] = i / arrival_rate;
    }

    loop {
        for _ in 0..arrival_rate {
            if next_task_idx < tasks.len() {
                queue.push(tasks[next_task_idx].clone());
                next_task_idx += 1;
            }
        }

        let mut dispatched_any = false;
        loop {
            if queue.is_empty() { break; }

            // Round-robin among all 4 units (each accepts any op)
            let mut found = false;
            for offset in 1..=n_units {
                let idx = (rr_last + offset) % n_units;
                if units[idx].is_free() {
                    rr_last = idx;
                    let task = queue.remove(0);
                    metrics.tasks_per_unit[idx] += 1;
                    units[idx].assign(task);
                    dispatched_any = true;
                    found = true;
                    break;
                }
            }
            if !found { break; }
        }

        if !dispatched_any && !queue.is_empty() {
            metrics.stall_ticks += 1;
        }

        for i in 0..n_units {
            if !units[i].is_free() {
                metrics.unit_busy_ticks[i] += 1;
            }
        }

        for i in 0..n_units {
            if let Some((task_id, _result)) = units[i].tick() {
                metrics.tasks_completed += 1;
                if task_id < metrics.task_arrival_tick.len() {
                    let lat = tick + 1 - metrics.task_arrival_tick[task_id];
                    metrics.task_latencies.push(lat);
                }
            }
        }

        tick += 1;
        metrics.total_ticks = tick;

        if next_task_idx >= tasks.len() && queue.is_empty()
            && units.iter().all(|u| u.is_free())
        {
            break;
        }
        if tick > tasks.len() * 20 + 200 { break; }
    }

    metrics
}
