//! Microscope: tiny network, trace every tick.
//!
//! Topology: IN → A → B → C → A (loop) → OUT
//!           IN → OUT (direct)
//!
//! Watch charge, activation, spike at every neuron every tick.
//!
//! Run: cargo run --example micro_loop --release

const TICKS: usize = 24;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

#[derive(Clone)]
struct Neuron {
    name: &'static str,
    charge: i16,
    activation: i8,
    threshold: u8,    // effective = threshold + 1
    channel: u8,      // 1-8 phase gating
    polarity: i8,     // +1 or -1
}

impl Neuron {
    fn new(name: &'static str, threshold: u8, channel: u8, polarity: i8) -> Self {
        Neuron { name, charge: 0, activation: 0, threshold, channel, polarity }
    }

    fn spike_check(&mut self, tick: usize) -> bool {
        let pi = (tick as u8 + 9 - self.channel) & 7;
        let pm = PHASE_BASE[pi as usize];
        let threshold_eff = (self.threshold as i16 + 1) * pm;
        let charge_x10 = self.charge * 10;
        if charge_x10 >= threshold_eff {
            self.activation = self.polarity;
            self.charge = 0;
            true
        } else {
            self.activation = 0;
            false
        }
    }
}

struct Edge {
    from: usize,
    to: usize,
    mode: &'static str, // "add" or "mul"
}

fn main() {
    // 5 neurons: IN(0), A(1), B(2), C(3), OUT(4)
    let mut neurons = vec![
        Neuron::new("IN",  1, 1,  1),  // low threshold, fires easily
        Neuron::new("A",   2, 2,  1),  // medium threshold
        Neuron::new("B",   2, 3,  1),  // medium, different phase
        Neuron::new("C",   1, 4, -1),  // low threshold, INHIBITORY
        Neuron::new("OUT", 2, 1,  1),  // medium threshold
    ];

    // Edges: IN→A, A→B, B→C, C→A (feedback loop), A→OUT, IN→OUT (direct)
    let edges = vec![
        Edge { from: 0, to: 1, mode: "add" },  // IN → A
        Edge { from: 1, to: 2, mode: "add" },  // A → B
        Edge { from: 2, to: 3, mode: "add" },  // B → C
        Edge { from: 3, to: 1, mode: "add" },  // C → A (feedback, inhibitory!)
        Edge { from: 1, to: 4, mode: "add" },  // A → OUT
        Edge { from: 0, to: 4, mode: "add" },  // IN → OUT (direct path)
    ];

    println!("=== MICRO LOOP: 5 neurons, {} ticks ===\n", TICKS);
    println!("Topology:");
    println!("  IN(thr=1,ch=1,+) → A(thr=2,ch=2,+) → B(thr=2,ch=3,+) → C(thr=1,ch=4,-)");
    println!("                     A ← C (inhibitory feedback loop)");
    println!("                     A → OUT(thr=2,ch=1,+)");
    println!("                    IN → OUT (direct)\n");

    // === Test 1: Single input pulse ===
    println!("--- Test 1: Single pulse (input=1 on tick 0-1) ---\n");

    println!("{:>4} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | events",
        "tick", "IN(ch/act)", "A(ch/act)", "B(ch/act)", "C(ch/act)", "OUT(ch/act)");
    println!("{:-<4}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-------", "", "", "", "", "", "");

    for tick in 0..TICKS {
        // 1. Input injection (tick 0-1)
        if tick < 2 {
            neurons[0].charge += 3; // strong input
        }

        // 2. Propagate edges (scatter)
        let mut incoming = vec![0i16; 5];
        for edge in &edges {
            let act = neurons[edge.from].activation as i16;
            if act != 0 {
                incoming[edge.to] += act * neurons[edge.from].polarity as i16;
            }
        }

        // 3. Accumulate + spike
        let mut events = Vec::new();
        for i in 0..5 {
            neurons[i].charge += incoming[i];

            // Decay every 6 ticks
            if tick % 6 == 5 && neurons[i].charge > 0 {
                neurons[i].charge -= 1;
                events.push(format!("{} decay", neurons[i].name));
            }

            let fired = neurons[i].spike_check(tick);
            if fired {
                let pol = if neurons[i].polarity > 0 { "+" } else { "-" };
                events.push(format!("{} FIRE({})", neurons[i].name, pol));
            }
        }

        // Print state
        let event_str = if events.is_empty() { "—".to_string() } else { events.join(", ") };
        println!("{:>4} | {:>5}/{:>5} | {:>5}/{:>5} | {:>5}/{:>5} | {:>5}/{:>5} | {:>5}/{:>5} | {}",
            tick,
            neurons[0].charge, neurons[0].activation,
            neurons[1].charge, neurons[1].activation,
            neurons[2].charge, neurons[2].activation,
            neurons[3].charge, neurons[3].activation,
            neurons[4].charge, neurons[4].activation,
            event_str);
    }

    // === Test 2: Two different inputs ===
    println!("\n\n--- Test 2: Input A (strong=5) vs Input B (weak=1) ---\n");

    for (label, input_strength) in [("Strong(5)", 5i16), ("Weak(1)", 1i16)] {
        // Reset
        for n in &mut neurons { n.charge = 0; n.activation = 0; }

        println!("  {} — OUT activity over {} ticks:", label, TICKS);
        print!("    OUT charge: ");
        for tick in 0..TICKS {
            if tick < 2 { neurons[0].charge += input_strength; }
            let mut incoming = vec![0i16; 5];
            for edge in &edges {
                let act = neurons[edge.from].activation as i16;
                if act != 0 { incoming[edge.to] += act * neurons[edge.from].polarity as i16; }
            }
            for i in 0..5 {
                neurons[i].charge += incoming[i];
                if tick % 6 == 5 && neurons[i].charge > 0 { neurons[i].charge -= 1; }
                neurons[i].spike_check(tick);
            }
            print!("{:>3}", neurons[4].charge);
        }
        println!();
        print!("    OUT spikes: ");
        // Reset and trace spikes
        for n in &mut neurons { n.charge = 0; n.activation = 0; }
        for tick in 0..TICKS {
            if tick < 2 { neurons[0].charge += input_strength; }
            let mut incoming = vec![0i16; 5];
            for edge in &edges {
                let act = neurons[edge.from].activation as i16;
                if act != 0 { incoming[edge.to] += act * neurons[edge.from].polarity as i16; }
            }
            for i in 0..5 {
                neurons[i].charge += incoming[i];
                if tick % 6 == 5 && neurons[i].charge > 0 { neurons[i].charge -= 1; }
                neurons[i].spike_check(tick);
            }
            print!("{:>3}", if neurons[4].activation != 0 { "^" } else { "." });
        }
        println!("\n");
    }

    // === Test 3: CHARGE-BASED PROPAGATION (graded potential) ===
    println!("\n--- Test 3: Charge-based propagation (charge leaks through edges) ---\n");
    println!("  incoming[target] += activation[src] + charge[src] / 4");
    println!("  (charge propagates as graded potential, weaker than spike)\n");

    for n in &mut neurons { n.charge = 0; n.activation = 0; }

    println!("{:>4} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | events",
        "tick", "IN(ch/act)", "A(ch/act)", "B(ch/act)", "C(ch/act)", "OUT(ch/act)");
    println!("{:-<4}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-------", "", "", "", "", "", "");

    for tick in 0..TICKS {
        if tick < 2 { neurons[0].charge += 3; }

        // GRADED propagation: charge ITSELF leaks through edges
        let mut incoming = vec![0i16; 5];
        for edge in &edges {
            let spike_signal = neurons[edge.from].activation as i16 * neurons[edge.from].polarity as i16;
            let graded_signal = neurons[edge.from].charge / 2 * neurons[edge.from].polarity as i16;
            incoming[edge.to] += spike_signal + graded_signal;
        }

        let mut events = Vec::new();
        for i in 0..5 {
            neurons[i].charge += incoming[i];
            if tick % 6 == 5 && neurons[i].charge > 0 { neurons[i].charge -= 1; events.push(format!("{} decay", neurons[i].name)); }
            if neurons[i].spike_check(tick) {
                events.push(format!("{} FIRE({})", neurons[i].name, if neurons[i].polarity > 0 { "+" } else { "-" }));
            }
        }

        let event_str = if events.is_empty() { "—".to_string() } else { events.join(", ") };
        println!("{:>4} | {:>5}/{:>5} | {:>5}/{:>5} | {:>5}/{:>5} | {:>5}/{:>5} | {:>5}/{:>5} | {}",
            tick, neurons[0].charge, neurons[0].activation,
            neurons[1].charge, neurons[1].activation,
            neurons[2].charge, neurons[2].activation,
            neurons[3].charge, neurons[3].activation,
            neurons[4].charge, neurons[4].activation,
            event_str);
    }

    // === Test 4: Mul edge version ===
    println!("\n--- Test 4: Same topology but B→C is MUL instead of ADD ---");
    println!("  (C only fires if BOTH B active AND something else feeds C)\n");

    // Add a second input to C via mul
    println!("  Topology: IN→A→B→C(mul), IN→C(mul), C→A(inhib), A→OUT, IN→OUT");
    println!("  C gets: B*IN (mul) — coincidence detector\n");

    for (label, input_strength) in [("Strong(5)", 5i16), ("Weak(1)", 1i16)] {
        for n in &mut neurons { n.charge = 0; n.activation = 0; }

        let edges_mul = vec![
            Edge { from: 0, to: 1, mode: "add" },  // IN → A
            Edge { from: 1, to: 2, mode: "add" },  // A → B
            Edge { from: 2, to: 3, mode: "mul" },  // B → C (MUL)
            Edge { from: 0, to: 3, mode: "mul" },  // IN → C (MUL) — coincidence!
            Edge { from: 3, to: 1, mode: "add" },  // C → A (feedback)
            Edge { from: 1, to: 4, mode: "add" },  // A → OUT
            Edge { from: 0, to: 4, mode: "add" },  // IN → OUT
        ];

        print!("  {} — OUT charge: ", label);
        for tick in 0..TICKS {
            if tick < 2 { neurons[0].charge += input_strength; }

            let mut add_in = vec![0i16; 5];
            let mut mul_in = vec![1i16; 5];
            let mut has_mul = vec![false; 5];

            for edge in &edges_mul {
                let act = neurons[edge.from].activation as i16 * neurons[edge.from].polarity as i16;
                match edge.mode {
                    "add" => { add_in[edge.to] += act; }
                    "mul" => { has_mul[edge.to] = true; if act == 0 { mul_in[edge.to] = 0; } else { mul_in[edge.to] *= act; } }
                    _ => {}
                }
            }

            for i in 0..5 {
                let mul = if has_mul[i] { mul_in[i] } else { 0 };
                neurons[i].charge += add_in[i] + mul;
                if tick % 6 == 5 && neurons[i].charge > 0 { neurons[i].charge -= 1; }
                neurons[i].spike_check(tick);
            }
            print!("{:>3}", neurons[4].charge);
        }
        println!();
    }
}
