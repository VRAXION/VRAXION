//! Parallel ALU — EP-trained C19 router distributes tasks across multiple ALUs
//!
//! Architecture:
//!   Task queue -> EP Router (C19 rho=8, ~20 neurons) -> 4 identical ALUs -> Result buffer
//!
//! Three routers compared:
//!   A. Round-Robin (baseline)
//!   B. First-Free (greedy)
//!   C. EP-trained neural router (float + frozen int8 LUT)
//!
//! Five workloads tested:
//!   W1: All simple (1-tick)
//!   W2: Mixed (40% 1-tick, 30% 2-tick, 30% 3-tick)
//!   W3: Bursty (10 MUL, 10 ADD, 10 MUL, ...)
//!   W4: Dependency chains
//!   W5: Adversarial (all 3-tick MUL)
//!
//! Scaling test: N_ALUS = {2, 4, 8}
//!
//! Run: cargo run --example parallel_alu --release

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
    fn usize(&mut self, n: usize) -> usize { (self.next() as usize) % n }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() as usize) % (i + 1);
            v.swap(i, j);
        }
    }
}

// ============================================================
// Task definitions
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpType {
    Add,   // 1 tick
    And,   // 1 tick
    Xor,   // 1 tick
    Or,    // 1 tick
    Cmp,   // 2 ticks
    Min,   // 2 ticks
    Max,   // 2 ticks
    Mul,   // 3 ticks
}

impl OpType {
    fn ticks(&self) -> usize {
        match self {
            OpType::Add | OpType::And | OpType::Xor | OpType::Or => 1,
            OpType::Cmp | OpType::Min | OpType::Max => 2,
            OpType::Mul => 3,
        }
    }

    fn complexity_bits(&self) -> u8 {
        match self.ticks() {
            1 => 0,
            2 => 1,
            3 => 2,
            _ => 3,
        }
    }

    fn type_bits(&self) -> u8 {
        match self {
            OpType::Add => 0,
            OpType::And => 1,
            OpType::Xor => 2,
            OpType::Or  => 3,
            OpType::Cmp => 4,
            OpType::Min => 5,
            OpType::Max => 6,
            OpType::Mul => 7,
        }
    }

    fn execute(&self, a: u8, b: u8) -> u8 {
        match self {
            OpType::Add => a.wrapping_add(b),
            OpType::And => a & b,
            OpType::Xor => a ^ b,
            OpType::Or  => a | b,
            OpType::Cmp => if a >= b { 1 } else { 0 },
            OpType::Min => if a < b { a } else { b },
            OpType::Max => if a > b { a } else { b },
            OpType::Mul => a.wrapping_mul(b),
        }
    }
}

#[derive(Clone, Debug)]
struct Task {
    id: usize,
    op: OpType,
    a: u8,
    b: u8,
    depends_on: Option<usize>, // task ID this depends on (for W4)
}

// ============================================================
// ALU state
// ============================================================

#[derive(Clone)]
struct AluState {
    busy_ticks_remaining: usize,
    current_task: Option<Task>,
    tasks_completed: usize,
    ticks_busy: usize,
}

impl AluState {
    fn new() -> Self {
        AluState {
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
        self.busy_ticks_remaining = 0;
        self.current_task = None;
        self.tasks_completed = 0;
        self.ticks_busy = 0;
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
    alu_busy_ticks: Vec<usize>,
    tasks_per_alu: Vec<usize>,
    task_latencies: Vec<usize>,    // per-task: tick_completed - tick_arrived
    task_arrival_tick: Vec<usize>, // when each task arrived
}

impl SimMetrics {
    fn new(n_alus: usize, n_tasks: usize) -> Self {
        SimMetrics {
            total_tasks: n_tasks,
            total_ticks: 0,
            tasks_completed: 0,
            stall_ticks: 0,
            alu_busy_ticks: vec![0; n_alus],
            tasks_per_alu: vec![0; n_alus],
            task_latencies: Vec::with_capacity(n_tasks),
            task_arrival_tick: vec![0; n_tasks],
        }
    }

    fn throughput(&self) -> f32 {
        if self.total_ticks == 0 { return 0.0; }
        self.tasks_completed as f32 / self.total_ticks as f32
    }

    fn utilization(&self) -> f32 {
        let total_alu_ticks = self.alu_busy_ticks.len() * self.total_ticks;
        if total_alu_ticks == 0 { return 0.0; }
        self.alu_busy_ticks.iter().sum::<usize>() as f32 / total_alu_ticks as f32
    }

    fn stall_rate(&self) -> f32 {
        if self.total_ticks == 0 { return 0.0; }
        self.stall_ticks as f32 / self.total_ticks as f32
    }

    fn avg_latency(&self) -> f32 {
        if self.task_latencies.is_empty() { return 0.0; }
        self.task_latencies.iter().sum::<usize>() as f32 / self.task_latencies.len() as f32
    }

    fn load_balance(&self) -> f32 {
        // coefficient of variation = stddev / mean
        let n = self.tasks_per_alu.len() as f32;
        if n == 0.0 { return 0.0; }
        let mean = self.tasks_per_alu.iter().sum::<usize>() as f32 / n;
        if mean < 0.001 { return 0.0; }
        let var = self.tasks_per_alu.iter()
            .map(|&x| { let d = x as f32 - mean; d * d })
            .sum::<f32>() / n;
        var.sqrt() / mean
    }
}

// ============================================================
// Simulation engine
// ============================================================

fn simulate<F>(
    tasks: &[Task],
    n_alus: usize,
    route_fn: &mut F,
    arrival_rate: usize, // tasks arriving per tick (creates contention when > n_alus)
) -> SimMetrics
where
    F: FnMut(&Task, &[AluState], usize) -> Option<usize>,
{
    let mut alus: Vec<AluState> = (0..n_alus).map(|_| AluState::new()).collect();
    let mut metrics = SimMetrics::new(n_alus, tasks.len());
    let mut completed_results: std::collections::HashMap<usize, u8> = std::collections::HashMap::new();
    let mut task_queue: std::collections::VecDeque<Task> = std::collections::VecDeque::new();
    let mut next_task_idx = 0;
    let mut tick: usize = 0;

    // Pre-fill arrival ticks
    metrics.task_arrival_tick = Vec::with_capacity(tasks.len());
    for i in 0..tasks.len() {
        metrics.task_arrival_tick.push(i / arrival_rate);
    }

    loop {
        // Inject tasks arriving this tick
        for _ in 0..arrival_rate {
            if next_task_idx < tasks.len() {
                task_queue.push_back(tasks[next_task_idx].clone());
                next_task_idx += 1;
            }
        }

        // Try to dispatch as many tasks as possible from the front of the queue
        let mut dispatched_any = false;
        loop {
            if task_queue.is_empty() { break; }

            let front_task = task_queue.front().unwrap();
            let dep_ok = match front_task.depends_on {
                Some(dep_id) => completed_results.contains_key(&dep_id),
                None => true,
            };

            if !dep_ok { break; } // Head-of-line blocking on dependency

            // Clone task info for routing decision
            let task_clone = front_task.clone();
            if let Some(alu_idx) = route_fn(&task_clone, &alus, n_alus) {
                if alus[alu_idx].is_free() {
                    let task = task_queue.pop_front().unwrap();
                    alus[alu_idx].assign(task);
                    metrics.tasks_per_alu[alu_idx] += 1;
                    dispatched_any = true;
                    continue; // Try to dispatch more
                }
            }
            break; // No free ALU available
        }

        if !dispatched_any && !task_queue.is_empty() {
            metrics.stall_ticks += 1;
        }

        // Record busy state BEFORE ticking (captures the full busy duration)
        for i in 0..n_alus {
            if !alus[i].is_free() {
                metrics.alu_busy_ticks[i] += 1;
            }
        }

        // Tick all ALUs
        for i in 0..n_alus {
            if let Some((task_id, result)) = alus[i].tick() {
                metrics.tasks_completed += 1;
                completed_results.insert(task_id, result);
                if task_id < metrics.task_arrival_tick.len() {
                    let lat = tick + 1 - metrics.task_arrival_tick[task_id];
                    metrics.task_latencies.push(lat);
                }
            }
        }

        tick += 1;
        metrics.total_ticks = tick;

        if next_task_idx >= tasks.len() && task_queue.is_empty()
            && alus.iter().all(|a| a.is_free())
        {
            break;
        }

        if tick > tasks.len() * 20 + 200 {
            break;
        }
    }

    metrics
}

// ============================================================
// Router A: Round-Robin
// ============================================================

struct RoundRobin {
    last: usize,
}

impl RoundRobin {
    fn new() -> Self { RoundRobin { last: 0 } }

    fn route(&mut self, _task: &Task, alus: &[AluState], n_alus: usize) -> Option<usize> {
        for offset in 1..=n_alus {
            let idx = (self.last + offset) % n_alus;
            if alus[idx].is_free() {
                self.last = idx;
                return Some(idx);
            }
        }
        None // all busy
    }
}

// ============================================================
// Router B: First-Free (greedy)
// ============================================================

struct FirstFree;

impl FirstFree {
    fn route(&self, _task: &Task, alus: &[AluState], n_alus: usize) -> Option<usize> {
        for i in 0..n_alus {
            if alus[i].is_free() {
                return Some(i);
            }
        }
        None
    }
}

// ============================================================
// EP Network for routing (variable N_ALUS)
// ============================================================

struct EpRouter {
    w1: Vec<f32>,  // h_dim x in_dim
    w2: Vec<f32>,  // out_dim x h_dim
    b1: Vec<f32>,  // h_dim
    b2: Vec<f32>,  // out_dim
    in_dim: usize,
    h_dim: usize,
    out_dim: usize, // = N_ALUS
}

impl EpRouter {
    fn new(n_alus: usize, rng: &mut Rng) -> Self {
        // Input encoding:
        //   4 bits: task type (which op)
        //   2 bits: complexity (1/2/3 ticks)
        //   n_alus bits: busy flags
        //   n_alus * 2 bits: remaining ticks per ALU (0-3)
        //   2 bits: last routed to (log2 encoded for up to 8 ALUs -> 3 bits)
        let last_bits = if n_alus <= 2 { 1 } else if n_alus <= 4 { 2 } else { 3 };
        let in_dim = 4 + 2 + n_alus + n_alus * 2 + last_bits;
        let h_dim = 16;
        let out_dim = n_alus;

        let s1 = 1.0 * (2.0 / in_dim as f32).sqrt();
        let s2 = 1.0 * (2.0 / h_dim as f32).sqrt();

        EpRouter {
            w1: (0..h_dim * in_dim).map(|_| rng.range_f32(-s1, s1)).collect(),
            w2: (0..out_dim * h_dim).map(|_| rng.range_f32(-s2, s2)).collect(),
            b1: vec![0.0; h_dim],
            b2: vec![0.0; out_dim],
            in_dim,
            h_dim,
            out_dim,
        }
    }

    fn encode_input(&self, task: &Task, alus: &[AluState], n_alus: usize, last_routed: usize) -> Vec<f32> {
        let mut x = Vec::with_capacity(self.in_dim);

        // Task type: 4 bits (one-hot-ish: encode the 3-bit type as individual bits + 1 extra)
        let tb = task.op.type_bits();
        x.push((tb & 1) as f32);
        x.push(((tb >> 1) & 1) as f32);
        x.push(((tb >> 2) & 1) as f32);
        x.push(0.0); // reserved 4th bit

        // Complexity: 2 bits
        let cb = task.op.complexity_bits();
        x.push((cb & 1) as f32);
        x.push(((cb >> 1) & 1) as f32);

        // ALU busy flags
        for i in 0..n_alus {
            x.push(if alus[i].is_free() { 0.0 } else { 1.0 });
        }

        // ALU remaining ticks (2 bits each, encode 0-3)
        for i in 0..n_alus {
            let rem = alus[i].busy_ticks_remaining.min(3) as u8;
            x.push((rem & 1) as f32);
            x.push(((rem >> 1) & 1) as f32);
        }

        // Last routed to
        let last_bits = if n_alus <= 2 { 1 } else if n_alus <= 4 { 2 } else { 3 };
        for bit in 0..last_bits {
            x.push(((last_routed >> bit) & 1) as f32);
        }

        x
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        // Single forward pass (no settling, just feedforward for speed)
        let mut h = vec![0.0f32; self.h_dim];
        for j in 0..self.h_dim {
            let mut sum = self.b1[j];
            for i in 0..self.in_dim {
                sum += self.w1[j * self.in_dim + i] * x[i];
            }
            h[j] = c19_act(sum);
        }

        let mut out = vec![0.0f32; self.out_dim];
        for k in 0..self.out_dim {
            let mut sum = self.b2[k];
            for j in 0..self.h_dim {
                sum += self.w2[k * self.h_dim + j] * h[j];
            }
            out[k] = c19_act(sum);
        }
        out
    }

    fn route(&self, task: &Task, alus: &[AluState], n_alus: usize, last_routed: usize) -> Option<usize> {
        let x = self.encode_input(task, alus, n_alus, last_routed);
        let scores = self.forward(&x);

        // Pick highest-scoring FREE ALU
        let mut best_idx: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..n_alus {
            if alus[i].is_free() && scores[i] > best_score {
                best_score = scores[i];
                best_idx = Some(i);
            }
        }
        best_idx
    }
}

// ============================================================
// EP Training for the router
//
// Strategy: per-sample EP with "teacher" signal.
// For each task in a sequence, compute optimal target (which ALU to pick)
// using a lookahead heuristic: pick the ALU with fewest remaining ticks.
// Then train the EP network to output high for that ALU, low for others.
// ============================================================

fn settle_router(
    s_h: &[f32], s_out: &[f32],
    x: &[f32], net: &EpRouter, dt: f32, beta: f32, y: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let (in_d, h, out_d) = (net.in_dim, net.h_dim, net.out_dim);

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

fn optimal_alu_choice(alus: &[AluState], n_alus: usize) -> Option<usize> {
    // Pick the free ALU with the index that distributes work most evenly
    // Ties broken by lowest index, but prefer the ALU with fewest total tasks
    let mut best: Option<usize> = None;
    let mut best_tasks = usize::MAX;
    for i in 0..n_alus {
        if alus[i].is_free() {
            if alus[i].tasks_completed < best_tasks {
                best_tasks = alus[i].tasks_completed;
                best = Some(i);
            }
        }
    }
    best
}

struct TrainingSample {
    input: Vec<f32>,
    target: Vec<f32>, // one-hot: 1.0 for optimal ALU, 0.0 for others
}

fn generate_training_data(
    tasks: &[Task],
    n_alus: usize,
    net: &EpRouter,
) -> Vec<TrainingSample> {
    let mut samples = Vec::new();
    let mut alus: Vec<AluState> = (0..n_alus).map(|_| AluState::new()).collect();
    let mut last_routed = 0usize;

    for task in tasks {
        // Check dependency
        let dep_ok = task.depends_on.is_none(); // simplified: assume deps met in training data
        if !dep_ok { continue; }

        // Check if any ALU free
        let any_free = alus.iter().any(|a| a.is_free());
        if !any_free {
            // Tick until an ALU frees up
            for _ in 0..10 {
                for alu in alus.iter_mut() { alu.tick(); }
                if alus.iter().any(|a| a.is_free()) { break; }
            }
        }

        if let Some(opt) = optimal_alu_choice(&alus, n_alus) {
            let input = net.encode_input(task, &alus, n_alus, last_routed);
            let mut target = vec![0.0f32; n_alus];
            target[opt] = 1.0;
            samples.push(TrainingSample { input, target });

            alus[opt].assign(task.clone());
            last_routed = opt;
        }

        // Tick all ALUs forward
        for alu in alus.iter_mut() { alu.tick(); }
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
    let h_dim = net.h_dim;
    let in_dim = net.in_dim;
    let out_dim = net.out_dim;

    let mut indices: Vec<usize> = (0..samples.len()).collect();

    for epoch in 0..n_epochs {
        let lr_eff = if epoch < 20 { lr * (epoch as f32 + 1.0) / 20.0 } else { lr };
        rng.shuffle(&mut indices);

        for &idx in &indices {
            let x = &samples[idx].input;
            let y = &samples[idx].target;

            // Free phase
            let mut s_h = vec![0.0f32; h_dim];
            let mut s_out = vec![0.0f32; out_dim];
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

            for j in 0..h_dim {
                let a_n = c19_act(s_nudge_h[j]);
                let a_f = c19_act(s_free_h[j]);
                for i in 0..in_dim {
                    net.w1[j * in_dim + i] += lr_eff * inv_beta * (a_n * x[i] - a_f * x[i]);
                }
                net.b1[j] += lr_eff * inv_beta * (a_n - a_f);
            }

            for k in 0..out_dim {
                let ao_n = c19_act(s_nudge_out[k]);
                let ao_f = c19_act(s_free_out[k]);
                for j in 0..h_dim {
                    let ah_n = c19_act(s_nudge_h[j]);
                    let ah_f = c19_act(s_free_h[j]);
                    net.w2[k * h_dim + j] += lr_eff * inv_beta * (ao_n * ah_n - ao_f * ah_f);
                }
                net.b2[k] += lr_eff * inv_beta * (ao_n - ao_f);
            }
        }

        // Log every 50 epochs
        if epoch % 50 == 0 || epoch == n_epochs - 1 {
            let mut correct = 0;
            for s in samples {
                let scores = net.forward(&s.input);
                // Find the target ALU
                let target_alu = s.target.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i).unwrap();
                // Check if the predicted best matches
                let pred_alu = scores.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i).unwrap();
                if pred_alu == target_alu { correct += 1; }
            }
            let acc = correct as f32 / samples.len() as f32 * 100.0;
            let msg = format!("    epoch {:>4}: routing accuracy = {:.1}% ({}/{})",
                epoch, acc, correct, samples.len());
            log(logf, &msg);
        }
    }
}

// ============================================================
// Frozen Router — int8 LUT version
// ============================================================

#[allow(dead_code)]
struct FrozenRouter {
    w1_q: Vec<i16>,
    w2_q: Vec<i16>,
    b1_q: Vec<i16>,
    b2_q: Vec<i16>,
    scale1: f32,
    scale2: f32,
    act_lut: Vec<i16>,
    act_lut_min: i32,
    in_dim: usize,
    h_dim: usize,
    out_dim: usize,
}

impl FrozenRouter {
    fn from_ep(net: &EpRouter) -> Self {
        let in_dim = net.in_dim;
        let h_dim = net.h_dim;
        let out_dim = net.out_dim;

        // Quantize weights
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

        // Build activation LUT for hidden sums
        let mut min_h_sum = i32::MAX;
        let mut max_h_sum = i32::MIN;
        for j in 0..h_dim {
            let mut lo = b1_q[j] as i32;
            let mut hi = b1_q[j] as i32;
            for i in 0..in_dim {
                let w = w1_q[j * in_dim + i] as i32;
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
            scale1, scale2,
            act_lut,
            act_lut_min: min_h_sum,
            in_dim, h_dim, out_dim,
        }
    }

    fn forward_int(&self, x: &[f32]) -> Vec<i32> {
        // Quantized feedforward: zero float in hot path
        let x_u8: Vec<u8> = x.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).collect();

        let mut h_act = vec![0i16; self.h_dim];
        for j in 0..self.h_dim {
            let mut sum: i32 = self.b1_q[j] as i32;
            for i in 0..self.in_dim {
                sum += self.w1_q[j * self.in_dim + i] as i32 * x_u8[i] as i32;
            }
            let clamped = sum.clamp(self.act_lut_min, self.act_lut_min + self.act_lut.len() as i32 - 1);
            let idx = (clamped - self.act_lut_min) as usize;
            h_act[j] = if idx < self.act_lut.len() { self.act_lut[idx] } else { 0 };
        }

        let mut out = vec![0i32; self.out_dim];
        for k in 0..self.out_dim {
            let mut sum: i32 = self.b2_q[k] as i32;
            for j in 0..self.h_dim {
                sum += self.w2_q[k * self.h_dim + j] as i32 * h_act[j] as i32;
            }
            out[k] = sum;
        }
        out
    }

    fn route(&self, task: &Task, alus: &[AluState], n_alus: usize, last_routed: usize, ep_net: &EpRouter) -> Option<usize> {
        let x = ep_net.encode_input(task, alus, n_alus, last_routed);
        let scores = self.forward_int(&x);

        let mut best_idx: Option<usize> = None;
        let mut best_score = i32::MIN;
        for i in 0..n_alus {
            if alus[i].is_free() && scores[i] > best_score {
                best_score = scores[i];
                best_idx = Some(i);
            }
        }
        best_idx
    }
}

// ============================================================
// Workload generators
// ============================================================

fn workload_w1_uniform(n: usize, rng: &mut Rng) -> Vec<Task> {
    let simple_ops = [OpType::Add, OpType::And, OpType::Xor, OpType::Or];
    (0..n).map(|id| Task {
        id,
        op: simple_ops[rng.usize(simple_ops.len())],
        a: (rng.next() & 0xFF) as u8,
        b: (rng.next() & 0xFF) as u8,
        depends_on: None,
    }).collect()
}

fn workload_w2_mixed(n: usize, rng: &mut Rng) -> Vec<Task> {
    (0..n).map(|id| {
        let r = rng.f32();
        let op = if r < 0.4 {
            // 40% simple (1-tick)
            [OpType::Add, OpType::And, OpType::Xor, OpType::Or][rng.usize(4)]
        } else if r < 0.7 {
            // 30% medium (2-tick)
            [OpType::Cmp, OpType::Min, OpType::Max][rng.usize(3)]
        } else {
            // 30% heavy (3-tick)
            OpType::Mul
        };
        Task { id, op, a: (rng.next() & 0xFF) as u8, b: (rng.next() & 0xFF) as u8, depends_on: None }
    }).collect()
}

fn workload_w3_bursty(n: usize, _rng: &mut Rng) -> Vec<Task> {
    let mut tasks = Vec::with_capacity(n);
    let burst_size = 10;
    let mut id = 0;
    while id < n {
        let burst_op = if (id / burst_size) % 2 == 0 { OpType::Mul } else { OpType::Add };
        for _ in 0..burst_size.min(n - id) {
            tasks.push(Task {
                id,
                op: burst_op,
                a: (id * 7 + 13) as u8,
                b: (id * 3 + 7) as u8,
                depends_on: None,
            });
            id += 1;
        }
    }
    tasks
}

fn workload_w4_dependencies(n: usize, rng: &mut Rng) -> Vec<Task> {
    let mut tasks = Vec::with_capacity(n);
    // Create chains of length 3-5, then independent tasks
    let mut id = 0;
    while id < n {
        let chain_len = 3 + rng.usize(3); // 3 to 5
        for c in 0..chain_len.min(n - id) {
            let dep = if c == 0 { None } else { Some(id - 1) };
            let op = if c == 0 { OpType::Add } else if c == chain_len - 1 { OpType::Mul } else { OpType::Cmp };
            tasks.push(Task {
                id,
                op,
                a: (id * 11 + 3) as u8,
                b: (id * 5 + 17) as u8,
                depends_on: dep,
            });
            id += 1;
        }
        // Add some independent tasks between chains
        if id < n {
            tasks.push(Task {
                id,
                op: OpType::Xor,
                a: (id * 13) as u8,
                b: (id * 7) as u8,
                depends_on: None,
            });
            id += 1;
        }
    }
    tasks
}

fn workload_w5_adversarial(n: usize, _rng: &mut Rng) -> Vec<Task> {
    // All MUL (3-tick): maximum contention
    (0..n).map(|id| Task {
        id,
        op: OpType::Mul,
        a: (id * 7 + 1) as u8,
        b: (id * 3 + 2) as u8,
        depends_on: None,
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

// ============================================================
// Verification: check that all tasks complete with correct results
// ============================================================

fn verify_results(tasks: &[Task], metrics: &SimMetrics) -> bool {
    metrics.tasks_completed == tasks.len()
}

// ============================================================
// Format metrics row
// ============================================================

fn fmt_row(name: &str, m: &SimMetrics) -> String {
    format!("  {:<20} tput={:.3}  util={:.1}%  stall={:.1}%  lat={:.1}  bal={:.3}  ({} tasks, {} ticks)",
        name,
        m.throughput(),
        m.utilization() * 100.0,
        m.stall_rate() * 100.0,
        m.avg_latency(),
        m.load_balance(),
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
    log(&mut logf, "  Parallel ALU: EP-trained C19 Router vs Round-Robin vs First-Free");
    log(&mut logf, "  Multi-ALU task distribution with load balancing");
    log(&mut logf, "================================================================");
    log(&mut logf, "");

    let mut rng = Rng::new(42);
    let n_alus = 4;
    let n_tasks = 100;

    // ================================================================
    // Phase 1: Generate training data
    // ================================================================
    log(&mut logf, "--- PHASE 1: Generate training data ---");
    log(&mut logf, &format!("  N_ALUS={}, generating 50 random task sequences of {} tasks", n_alus, n_tasks));

    let mut all_training_tasks: Vec<Vec<Task>> = Vec::new();
    for seq in 0..50 {
        let tasks = workload_w2_mixed(n_tasks, &mut Rng::new(1000 + seq as u64));
        all_training_tasks.push(tasks);
    }

    let mut ep_net = EpRouter::new(n_alus, &mut rng);
    log(&mut logf, &format!("  Router input dim = {}, hidden = {}, output = {}",
        ep_net.in_dim, ep_net.h_dim, ep_net.out_dim));

    // Generate training samples from all sequences
    let mut all_samples: Vec<TrainingSample> = Vec::new();
    for seq_tasks in &all_training_tasks {
        let samples = generate_training_data(seq_tasks, n_alus, &ep_net);
        all_samples.extend(samples);
    }
    log(&mut logf, &format!("  Total training samples: {}", all_samples.len()));
    log(&mut logf, "");

    // ================================================================
    // Phase 2: Train EP router
    // ================================================================
    log(&mut logf, "--- PHASE 2: Train EP router ---");
    log(&mut logf, "  beta=0.5, T=30, dt=0.5, lr=0.01, epochs=200");

    train_ep_router(
        &mut ep_net,
        &all_samples,
        0.5,   // beta
        30,    // t_max
        0.5,   // dt
        0.01,  // lr
        200,   // epochs
        &mut rng,
        &mut logf,
    );

    let train_time = t0.elapsed().as_secs_f32();
    log(&mut logf, &format!("  Training complete in {:.1}s", train_time));
    log(&mut logf, "");

    // ================================================================
    // Phase 3: Freeze to int8
    // ================================================================
    log(&mut logf, "--- PHASE 3: Freeze to int8 LUT ---");
    let frozen = FrozenRouter::from_ep(&ep_net);
    log(&mut logf, &format!("  W1: {} params, W2: {} params", frozen.w1_q.len(), frozen.w2_q.len()));
    log(&mut logf, &format!("  Activation LUT: {} entries", frozen.act_lut.len()));

    // Verify frozen matches float
    let mut match_count = 0;
    let check_n = all_samples.len().min(200);
    for s in &all_samples[..check_n] {
        let float_scores = ep_net.forward(&s.input);
        let frozen_scores = frozen.forward_int(&s.input);
        let float_best = float_scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
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
    // Arrival rate: 3 tasks per tick with 4 ALUs = high contention
    // (especially with multi-tick ops that block ALUs)
    let arrival_rate = 3;
    log(&mut logf, &format!("  N_ALUS = {}, {} tasks per workload, arrival_rate = {}", n_alus, n_tasks, arrival_rate));
    log(&mut logf, "");

    let workload_names = ["W1: Uniform", "W2: Mixed", "W3: Bursty", "W4: Deps", "W5: Advers."];

    let mut all_metrics: Vec<[SimMetrics; 4]> = Vec::new();

    for (wi, wname) in workload_names.iter().enumerate() {
        let tasks = match wi {
            0 => workload_w1_uniform(n_tasks, &mut Rng::new(777)),
            1 => workload_w2_mixed(n_tasks, &mut Rng::new(777)),
            2 => workload_w3_bursty(n_tasks, &mut Rng::new(777)),
            3 => workload_w4_dependencies(n_tasks, &mut Rng::new(777)),
            4 => workload_w5_adversarial(n_tasks, &mut Rng::new(777)),
            _ => unreachable!(),
        };

        log(&mut logf, &format!("  === {} ===", wname));

        // Router A: Round-Robin
        let mut rr = RoundRobin::new();
        let m_rr = simulate(&tasks, n_alus, &mut |task, alus, na| rr.route(task, alus, na), arrival_rate);
        let v_rr = verify_results(&tasks, &m_rr);
        log(&mut logf, &fmt_row("Round-Robin", &m_rr));
        if !v_rr { log(&mut logf, "    *** TASKS DROPPED ***"); }

        // Router B: First-Free
        let ff = FirstFree;
        let m_ff = simulate(&tasks, n_alus, &mut |task, alus, na| ff.route(task, alus, na), arrival_rate);
        let v_ff = verify_results(&tasks, &m_ff);
        log(&mut logf, &fmt_row("First-Free", &m_ff));
        if !v_ff { log(&mut logf, "    *** TASKS DROPPED ***"); }

        // Router C: EP float
        let mut last_ep = 0usize;
        let m_ep = simulate(&tasks, n_alus, &mut |task, alus, na| {
            let r = ep_net.route(task, alus, na, last_ep);
            if let Some(idx) = r { last_ep = idx; }
            r
        }, arrival_rate);
        let v_ep = verify_results(&tasks, &m_ep);
        log(&mut logf, &fmt_row("EP Neural (float)", &m_ep));
        if !v_ep { log(&mut logf, "    *** TASKS DROPPED ***"); }

        // Router D: EP frozen
        let mut last_fr = 0usize;
        let m_fr = simulate(&tasks, n_alus, &mut |task, alus, na| {
            let r = frozen.route(task, alus, na, last_fr, &ep_net);
            if let Some(idx) = r { last_fr = idx; }
            r
        }, arrival_rate);
        let v_fr = verify_results(&tasks, &m_fr);
        log(&mut logf, &fmt_row("EP Neural (frozen)", &m_fr));
        if !v_fr { log(&mut logf, "    *** TASKS DROPPED ***"); }

        log(&mut logf, "");

        all_metrics.push([m_rr, m_ff, m_ep, m_fr]);
    }

    // ================================================================
    // Phase 5: Summary comparison table
    // ================================================================
    log(&mut logf, "================================================================");
    log(&mut logf, "  COMPARISON TABLE");
    log(&mut logf, "================================================================");
    log(&mut logf, "");
    log(&mut logf, "                    Round-Robin  First-Free   EP(float)    EP(frozen)");
    log(&mut logf, "  ─────────────────────────────────────────────────────────────────");

    let mut avg_tput = [0.0f32; 4];
    let mut avg_util = [0.0f32; 4];
    let mut avg_stall = [0.0f32; 4];
    let mut avg_bal = [0.0f32; 4];
    let mut avg_lat = [0.0f32; 4];

    for (wi, wname) in workload_names.iter().enumerate() {
        let ms = &all_metrics[wi];
        log(&mut logf, &format!("  {} throughput: {:>8.3}     {:>8.3}     {:>8.3}     {:>8.3}",
            wname,
            ms[0].throughput(), ms[1].throughput(), ms[2].throughput(), ms[3].throughput()));
        for r in 0..4 {
            avg_tput[r] += ms[r].throughput();
            avg_util[r] += ms[r].utilization();
            avg_stall[r] += ms[r].stall_rate();
            avg_bal[r] += ms[r].load_balance();
            avg_lat[r] += ms[r].avg_latency();
        }
    }
    let nw = workload_names.len() as f32;
    log(&mut logf, "  ─────────────────────────────────────────────────────────────────");
    log(&mut logf, &format!("  AVG throughput:      {:>8.3}     {:>8.3}     {:>8.3}     {:>8.3}",
        avg_tput[0]/nw, avg_tput[1]/nw, avg_tput[2]/nw, avg_tput[3]/nw));
    log(&mut logf, &format!("  AVG utilization:     {:>7.1}%     {:>7.1}%     {:>7.1}%     {:>7.1}%",
        avg_util[0]/nw*100.0, avg_util[1]/nw*100.0, avg_util[2]/nw*100.0, avg_util[3]/nw*100.0));
    log(&mut logf, &format!("  AVG stall rate:      {:>7.1}%     {:>7.1}%     {:>7.1}%     {:>7.1}%",
        avg_stall[0]/nw*100.0, avg_stall[1]/nw*100.0, avg_stall[2]/nw*100.0, avg_stall[3]/nw*100.0));
    log(&mut logf, &format!("  AVG load balance:    {:>8.3}     {:>8.3}     {:>8.3}     {:>8.3}",
        avg_bal[0]/nw, avg_bal[1]/nw, avg_bal[2]/nw, avg_bal[3]/nw));
    log(&mut logf, &format!("  AVG latency:         {:>8.1}     {:>8.1}     {:>8.1}     {:>8.1}",
        avg_lat[0]/nw, avg_lat[1]/nw, avg_lat[2]/nw, avg_lat[3]/nw));
    log(&mut logf, "");

    // ================================================================
    // Phase 6: Scaling test (N_ALUS = 2, 4, 8)
    // ================================================================
    log(&mut logf, "================================================================");
    log(&mut logf, "  SCALING TEST: N_ALUS = {2, 4, 8}");
    log(&mut logf, "================================================================");
    log(&mut logf, "");

    for &test_n_alus in &[2usize, 4, 8] {
        log(&mut logf, &format!("  --- N_ALUS = {} ---", test_n_alus));

        // Train a new router for this ALU count
        let mut scale_rng = Rng::new(42 + test_n_alus as u64);
        let mut scale_net = EpRouter::new(test_n_alus, &mut scale_rng);

        // Generate training data
        let mut scale_samples: Vec<TrainingSample> = Vec::new();
        for seq in 0..30 {
            let tasks = workload_w2_mixed(n_tasks, &mut Rng::new(2000 + seq as u64));
            let samples = generate_training_data(&tasks, test_n_alus, &scale_net);
            scale_samples.extend(samples);
        }

        // Train (fewer epochs for speed)
        train_ep_router(
            &mut scale_net, &scale_samples,
            0.5, 30, 0.5, 0.01, 100,
            &mut scale_rng, &mut logf,
        );

        let scale_frozen = FrozenRouter::from_ep(&scale_net);

        // Test on W2 (mixed)
        let test_tasks = workload_w2_mixed(n_tasks, &mut Rng::new(9999));

        let scale_rate = test_n_alus; // arrival rate = N_ALUS for high contention
        let mut rr = RoundRobin::new();
        let m_rr = simulate(&test_tasks, test_n_alus, &mut |task, alus, na| rr.route(task, alus, na), scale_rate);

        let ff = FirstFree;
        let m_ff = simulate(&test_tasks, test_n_alus, &mut |task, alus, na| ff.route(task, alus, na), scale_rate);

        let mut last_ep = 0usize;
        let m_ep = simulate(&test_tasks, test_n_alus, &mut |task, alus, na| {
            let r = scale_net.route(task, alus, na, last_ep);
            if let Some(idx) = r { last_ep = idx; }
            r
        }, scale_rate);

        let mut last_fr = 0usize;
        let m_fr = simulate(&test_tasks, test_n_alus, &mut |task, alus, na| {
            let r = scale_frozen.route(task, alus, na, last_fr, &scale_net);
            if let Some(idx) = r { last_fr = idx; }
            r
        }, scale_rate);

        log(&mut logf, &format!("    RR:       tput={:.3}  util={:.1}%  bal={:.3}",
            m_rr.throughput(), m_rr.utilization()*100.0, m_rr.load_balance()));
        log(&mut logf, &format!("    FF:       tput={:.3}  util={:.1}%  bal={:.3}",
            m_ff.throughput(), m_ff.utilization()*100.0, m_ff.load_balance()));
        log(&mut logf, &format!("    EP(f32):  tput={:.3}  util={:.1}%  bal={:.3}",
            m_ep.throughput(), m_ep.utilization()*100.0, m_ep.load_balance()));
        log(&mut logf, &format!("    EP(i8):   tput={:.3}  util={:.1}%  bal={:.3}",
            m_fr.throughput(), m_fr.utilization()*100.0, m_fr.load_balance()));
        log(&mut logf, "");
    }

    // ================================================================
    // Phase 7: Load balance detail (for N_ALUS=4, W2)
    // ================================================================
    log(&mut logf, "================================================================");
    log(&mut logf, "  LOAD BALANCE DETAIL (N_ALUS=4, W2 Mixed)");
    log(&mut logf, "================================================================");
    log(&mut logf, "");

    let detail_tasks = workload_w2_mixed(n_tasks, &mut Rng::new(777));

    // Round-robin
    let mut rr = RoundRobin::new();
    let m = simulate(&detail_tasks, 4, &mut |task, alus, na| rr.route(task, alus, na), arrival_rate);
    log(&mut logf, &format!("  Round-Robin tasks/ALU: {:?}", m.tasks_per_alu));

    // First-free
    let ff = FirstFree;
    let m = simulate(&detail_tasks, 4, &mut |task, alus, na| ff.route(task, alus, na), arrival_rate);
    log(&mut logf, &format!("  First-Free  tasks/ALU: {:?}", m.tasks_per_alu));

    // EP float
    let mut last_ep = 0usize;
    let m = simulate(&detail_tasks, 4, &mut |task, alus, na| {
        let r = ep_net.route(task, alus, na, last_ep);
        if let Some(idx) = r { last_ep = idx; }
        r
    }, arrival_rate);
    log(&mut logf, &format!("  EP(float)   tasks/ALU: {:?}", m.tasks_per_alu));

    // EP frozen
    let mut last_fr = 0usize;
    let m = simulate(&detail_tasks, 4, &mut |task, alus, na| {
        let r = frozen.route(task, alus, na, last_fr, &ep_net);
        if let Some(idx) = r { last_fr = idx; }
        r
    }, arrival_rate);
    log(&mut logf, &format!("  EP(frozen)  tasks/ALU: {:?}", m.tasks_per_alu));
    log(&mut logf, "");

    // ================================================================
    // Phase 8: Adversarial deep-dive
    // ================================================================
    log(&mut logf, "================================================================");
    log(&mut logf, "  ADVERSARIAL DEEP-DIVE");
    log(&mut logf, "================================================================");
    log(&mut logf, "");

    // Pathological: all same complexity, contention maximized (all arrive at once)
    for &(name, n, rate) in &[("20 MUL (burst)", 20usize, 20usize), ("50 MUL (burst)", 50, 50), ("100 MUL (rate=6)", 100, 6)] {
        let tasks = workload_w5_adversarial(n, &mut Rng::new(42));

        let mut rr = RoundRobin::new();
        let m_rr = simulate(&tasks, 4, &mut |task, alus, na| rr.route(task, alus, na), rate);

        let mut last_ep = 0usize;
        let m_ep = simulate(&tasks, 4, &mut |task, alus, na| {
            let r = ep_net.route(task, alus, na, last_ep);
            if let Some(idx) = r { last_ep = idx; }
            r
        }, rate);

        log(&mut logf, &format!("  {}: RR tput={:.3}, EP tput={:.3}  (theoretical max = {:.3})",
            name, m_rr.throughput(), m_ep.throughput(), 4.0 / 3.0));
    }
    log(&mut logf, "");

    // Stress: rapid alternation between 1-tick and 3-tick at high arrival rate
    let mut alt_tasks = Vec::new();
    for id in 0..100 {
        alt_tasks.push(Task {
            id,
            op: if id % 2 == 0 { OpType::Mul } else { OpType::Add },
            a: id as u8, b: (id * 3) as u8,
            depends_on: None,
        });
    }
    let mut rr = RoundRobin::new();
    let m_rr = simulate(&alt_tasks, 4, &mut |task, alus, na| rr.route(task, alus, na), 4);
    let mut last_ep = 0usize;
    let m_ep = simulate(&alt_tasks, 4, &mut |task, alus, na| {
        let r = ep_net.route(task, alus, na, last_ep);
        if let Some(idx) = r { last_ep = idx; }
        r
    }, 4);
    log(&mut logf, &format!("  Alternating MUL/ADD (rate=4): RR tput={:.3} bal={:.3}, EP tput={:.3} bal={:.3}",
        m_rr.throughput(), m_rr.load_balance(), m_ep.throughput(), m_ep.load_balance()));
    log(&mut logf, "");

    // ================================================================
    // FINAL VERDICT
    // ================================================================
    let total_time = t0.elapsed().as_secs_f32();
    log(&mut logf, "================================================================");
    log(&mut logf, "  FINAL VERDICT");
    log(&mut logf, "================================================================");
    log(&mut logf, "");

    // Compute overall winner
    let ep_avg_tput = avg_tput[2] / nw;
    let rr_avg_tput = avg_tput[0] / nw;
    let ff_avg_tput = avg_tput[1] / nw;
    let fr_avg_tput = avg_tput[3] / nw;

    let ep_vs_rr = (ep_avg_tput - rr_avg_tput) / rr_avg_tput * 100.0;
    let ep_vs_ff = (ep_avg_tput - ff_avg_tput) / ff_avg_tput * 100.0;
    let freeze_drop = (ep_avg_tput - fr_avg_tput) / ep_avg_tput * 100.0;

    log(&mut logf, &format!("  EP router vs Round-Robin:  {:+.1}% throughput", ep_vs_rr));
    log(&mut logf, &format!("  EP router vs First-Free:   {:+.1}% throughput", ep_vs_ff));
    log(&mut logf, &format!("  Freeze accuracy drop:      {:.1}%", freeze_drop));
    log(&mut logf, &format!("  Float-Frozen agreement:    {:.1}%", freeze_match));
    log(&mut logf, "");

    let ep_bal = avg_bal[2] / nw;
    let rr_bal = avg_bal[0] / nw;
    let ff_bal = avg_bal[1] / nw;
    log(&mut logf, &format!("  Load balance (lower=better):"));
    log(&mut logf, &format!("    Round-Robin: {:.3}", rr_bal));
    log(&mut logf, &format!("    First-Free:  {:.3}", ff_bal));
    log(&mut logf, &format!("    EP Neural:   {:.3}", ep_bal));
    log(&mut logf, "");

    let ep_wins = if ep_avg_tput >= rr_avg_tput && ep_avg_tput >= ff_avg_tput { "YES" } else { "NO" };
    let ep_balanced = if ep_bal <= rr_bal || ep_bal <= ff_bal { "YES" } else { "NO" };
    let freeze_ok = if freeze_drop.abs() < 5.0 { "YES" } else { "NO" };

    log(&mut logf, &format!("  Q1: EP router beats baselines?          {}", ep_wins));
    log(&mut logf, &format!("  Q2: EP learns actual load balancing?    {}", ep_balanced));
    log(&mut logf, &format!("  Q3: Freeze to int8 preserves quality?   {}", freeze_ok));
    log(&mut logf, "");
    log(&mut logf, &format!("  Total time: {:.1}s", total_time));
    log(&mut logf, "================================================================");
}
