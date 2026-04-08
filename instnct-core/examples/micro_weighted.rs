//! Microscope: tiny weighted network. Watch every tick.
//!
//! IN → A(w=3) → B(w=2) → C(w=-2, inhib) → A(w=1, feedback)
//!                                           A → OUT(w=3)
//!                                          IN → OUT(w=1, direct)
//!
//! Run: cargo run --example micro_weighted --release

const TICKS: usize = 24;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

struct Neuron { name: &'static str, charge: i16, activation: i8, threshold: u8, channel: u8, polarity: i8 }
impl Neuron {
    fn new(name: &'static str, thr: u8, ch: u8, pol: i8) -> Self { Neuron { name, charge: 0, activation: 0, threshold: thr, channel: ch, polarity: pol } }
    fn spike_check(&mut self, tick: usize) -> bool {
        let pi = (tick as u8 + 9 - self.channel) & 7;
        let pm = PHASE_BASE[pi as usize];
        if self.charge * 10 >= (self.threshold as i16 + 1) * pm {
            self.activation = self.polarity; self.charge = 0; true
        } else { self.activation = 0; false }
    }
}

struct Edge { from: usize, to: usize, weight: i16 }

fn run_trace(label: &str, neurons: &mut Vec<Neuron>, edges: &[Edge], input_strength: i16) {
    for n in neurons.iter_mut() { n.charge = 0; n.activation = 0; }

    println!("  {} (input={})", label, input_strength);
    println!("  {:>4} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | events",
        "tick", "IN(ch/act)", "A(ch/act)", "B(ch/act)", "C(ch/act)", "OUT(ch/act)");
    println!("  {:-<4}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-------", "", "", "", "", "", "");

    for tick in 0..TICKS {
        if tick < 2 { neurons[0].charge += input_strength; }

        let mut incoming = vec![0i16; 5];
        for edge in edges {
            let act = neurons[edge.from].activation as i16;
            if act != 0 {
                incoming[edge.to] += act * neurons[edge.from].polarity as i16 * edge.weight;
            }
        }

        let mut events = Vec::new();
        for i in 0..5 {
            neurons[i].charge += incoming[i];
            if tick % 6 == 5 && neurons[i].charge > 0 { neurons[i].charge -= 1; events.push(format!("{} decay", neurons[i].name)); }
            if neurons[i].charge < 0 { neurons[i].charge = 0; } // no negative charge
            if neurons[i].spike_check(tick) {
                events.push(format!("{} FIRE({}{})", neurons[i].name, if neurons[i].polarity > 0 { "+" } else { "-" }, neurons[i].polarity.abs()));
            }
        }

        let ev = if events.is_empty() { "—".to_string() } else { events.join(", ") };
        println!("  {:>4} | {:>4}/{:>4} | {:>4}/{:>4} | {:>4}/{:>4} | {:>4}/{:>4} | {:>4}/{:>4} | {}",
            tick, neurons[0].charge, neurons[0].activation,
            neurons[1].charge, neurons[1].activation,
            neurons[2].charge, neurons[2].activation,
            neurons[3].charge, neurons[3].activation,
            neurons[4].charge, neurons[4].activation, ev);
    }
    println!();
}

fn main() {
    println!("=== MICRO WEIGHTED: spike × weight ===\n");

    // Test 1: Weighted edges, low thresholds
    println!("--- Test 1: Low threshold (thr=0 → eff=1), weighted edges ---");
    println!("  IN→A(w=3) A→B(w=2) B→C(w=2) C→A(w=-2,inhib) A→OUT(w=3) IN→OUT(w=1)\n");
    {
        let mut neurons = vec![
            Neuron::new("IN",  0, 1,  1),  // thr=0→eff=1, fires very easily
            Neuron::new("A",   0, 1,  1),  // same
            Neuron::new("B",   0, 1,  1),  // same
            Neuron::new("C",   0, 1, -1),  // inhibitory
            Neuron::new("OUT", 0, 1,  1),
        ];
        let edges = vec![
            Edge { from: 0, to: 1, weight: 3 },  // IN → A strong
            Edge { from: 1, to: 2, weight: 2 },  // A → B
            Edge { from: 2, to: 3, weight: 2 },  // B → C
            Edge { from: 3, to: 1, weight: 2 },  // C → A (inhib: polarity=-1, so net = -2)
            Edge { from: 1, to: 4, weight: 3 },  // A → OUT strong
            Edge { from: 0, to: 4, weight: 1 },  // IN → OUT weak direct
        ];
        run_trace("Strong input (5)", &mut neurons, &edges, 5);
        run_trace("Weak input (1)", &mut neurons, &edges, 1);
    }

    // Test 2: Different weights, same topology — does output differ?
    println!("--- Test 2: Can weights create different output for different inputs? ---");
    println!("  Two inputs: digit_0 = inject neuron 0, digit_1 = inject neuron 0 with different strength\n");
    {
        let mut neurons = vec![
            Neuron::new("IN",  0, 1,  1),
            Neuron::new("A",   1, 2,  1),  // thr=1→eff=2, needs stronger input
            Neuron::new("B",   2, 3,  1),  // thr=2→eff=3, high threshold
            Neuron::new("C",   0, 1, -1),
            Neuron::new("OUT", 1, 1,  1),
        ];
        let edges = vec![
            Edge { from: 0, to: 1, weight: 2 },
            Edge { from: 0, to: 2, weight: 1 },  // IN→B weak (only strong input activates B)
            Edge { from: 1, to: 4, weight: 3 },  // A→OUT strong
            Edge { from: 2, to: 4, weight: -2 },  // B→OUT inhibitory (B active → suppress OUT)
            Edge { from: 0, to: 4, weight: 1 },
        ];
        run_trace("digit=1 (weak)", &mut neurons, &edges, 1);
        run_trace("digit=3 (medium)", &mut neurons, &edges, 3);
        run_trace("digit=5 (strong)", &mut neurons, &edges, 5);
    }

    // Test 3: Two separate inputs that need to COMBINE
    println!("--- Test 3: Two inputs combining (addition-like) ---");
    println!("  IN_A(neuron0) + IN_B(neuron1) → hidden(neuron2) → OUT(neuron3)");
    println!("  Does OUT charge reflect A+B?\n");
    {
        let mut neurons = vec![
            Neuron::new("IN_A", 0, 1, 1),
            Neuron::new("IN_B", 0, 1, 1),
            Neuron::new("HID",  1, 2, 1),
            Neuron::new("OUT",  0, 1, 1),
            Neuron::new("_",    0, 1, 1), // unused
        ];
        let edges = vec![
            Edge { from: 0, to: 2, weight: 2 },  // IN_A → HID
            Edge { from: 1, to: 2, weight: 2 },  // IN_B → HID
            Edge { from: 2, to: 3, weight: 3 },  // HID → OUT
            Edge { from: 0, to: 3, weight: 1 },  // IN_A → OUT direct
            Edge { from: 1, to: 3, weight: 1 },  // IN_B → OUT direct
        ];

        // 0+0
        for n in neurons.iter_mut() { n.charge = 0; n.activation = 0; }
        for tick in 0..6 { // inject nothing
            let mut inc = vec![0i16; 5];
            for e in &edges { let a = neurons[e.from].activation as i16; if a != 0 { inc[e.to] += a * neurons[e.from].polarity as i16 * e.weight; } }
            for i in 0..5 { neurons[i].charge += inc[i]; if neurons[i].charge < 0 { neurons[i].charge = 0; } neurons[i].spike_check(tick); }
        }
        println!("  0+0: OUT charge = {}", neurons[3].charge);

        // 1+0
        for n in neurons.iter_mut() { n.charge = 0; n.activation = 0; }
        for tick in 0..6 {
            if tick < 2 { neurons[0].charge += 2; }
            let mut inc = vec![0i16; 5];
            for e in &edges { let a = neurons[e.from].activation as i16; if a != 0 { inc[e.to] += a * neurons[e.from].polarity as i16 * e.weight; } }
            for i in 0..5 { neurons[i].charge += inc[i]; if neurons[i].charge < 0 { neurons[i].charge = 0; } neurons[i].spike_check(tick); }
        }
        println!("  1+0: OUT charge = {}", neurons[3].charge);

        // 0+1
        for n in neurons.iter_mut() { n.charge = 0; n.activation = 0; }
        for tick in 0..6 {
            if tick < 2 { neurons[1].charge += 2; }
            let mut inc = vec![0i16; 5];
            for e in &edges { let a = neurons[e.from].activation as i16; if a != 0 { inc[e.to] += a * neurons[e.from].polarity as i16 * e.weight; } }
            for i in 0..5 { neurons[i].charge += inc[i]; if neurons[i].charge < 0 { neurons[i].charge = 0; } neurons[i].spike_check(tick); }
        }
        println!("  0+1: OUT charge = {}", neurons[3].charge);

        // 1+1
        for n in neurons.iter_mut() { n.charge = 0; n.activation = 0; }
        for tick in 0..6 {
            if tick < 2 { neurons[0].charge += 2; neurons[1].charge += 2; }
            let mut inc = vec![0i16; 5];
            for e in &edges { let a = neurons[e.from].activation as i16; if a != 0 { inc[e.to] += a * neurons[e.from].polarity as i16 * e.weight; } }
            for i in 0..5 { neurons[i].charge += inc[i]; if neurons[i].charge < 0 { neurons[i].charge = 0; } neurons[i].spike_check(tick); }
        }
        println!("  1+1: OUT charge = {}", neurons[3].charge);

        // 2+1
        for n in neurons.iter_mut() { n.charge = 0; n.activation = 0; }
        for tick in 0..6 {
            if tick < 2 { neurons[0].charge += 4; neurons[1].charge += 2; }
            let mut inc = vec![0i16; 5];
            for e in &edges { let a = neurons[e.from].activation as i16; if a != 0 { inc[e.to] += a * neurons[e.from].polarity as i16 * e.weight; } }
            for i in 0..5 { neurons[i].charge += inc[i]; if neurons[i].charge < 0 { neurons[i].charge = 0; } neurons[i].spike_check(tick); }
        }
        println!("  2+1: OUT charge = {}", neurons[3].charge);

        // 2+2
        for n in neurons.iter_mut() { n.charge = 0; n.activation = 0; }
        for tick in 0..6 {
            if tick < 2 { neurons[0].charge += 4; neurons[1].charge += 4; }
            let mut inc = vec![0i16; 5];
            for e in &edges { let a = neurons[e.from].activation as i16; if a != 0 { inc[e.to] += a * neurons[e.from].polarity as i16 * e.weight; } }
            for i in 0..5 { neurons[i].charge += inc[i]; if neurons[i].charge < 0 { neurons[i].charge = 0; } neurons[i].spike_check(tick); }
        }
        println!("  2+2: OUT charge = {}", neurons[3].charge);

        println!("\n  If OUT charge increases monotonically with A+B → the circuit COMPUTES addition.");
        println!("  If not → it's just threshold/spike artifacts.");
    }
}
