//! Microscope: rate coding with many ticks.
//! Does spike FREQUENCY encode input strength?
//!
//! Run: cargo run --example micro_rate --release

const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

struct Neuron {
    name: &'static str, charge: i16, activation: i8,
    threshold: u8, channel: u8, polarity: i8,
    refractory: u8, // 0=ready, 1=cooling
    spike_count: u32,
}
impl Neuron {
    fn new(name: &'static str, thr: u8, ch: u8, pol: i8) -> Self {
        Neuron { name, charge: 0, activation: 0, threshold: thr, channel: ch, polarity: pol, refractory: 0, spike_count: 0 }
    }
    fn tick(&mut self, incoming: i16, tick: usize) {
        if self.refractory > 0 { self.refractory -= 1; self.activation = 0; return; }
        self.charge = self.charge.saturating_add(incoming);
        // Decay every 6 ticks
        if tick % 6 == 5 && self.charge > 0 { self.charge -= 1; }
        // Clamp
        if self.charge < 0 { self.charge = 0; }
        let pi = (tick as u8 + 9 - self.channel) & 7;
        let pm = PHASE_BASE[pi as usize];
        if self.charge * 10 >= (self.threshold as i16 + 1) * pm {
            self.activation = self.polarity;
            self.charge = 0;
            self.refractory = 1; // 1 tick refractory
            self.spike_count += 1;
        } else {
            self.activation = 0;
        }
    }
    fn reset(&mut self) { self.charge = 0; self.activation = 0; self.refractory = 0; self.spike_count = 0; }
}

struct Edge { from: usize, to: usize, weight: i16 }

fn run(neurons: &mut [Neuron], edges: &[Edge], input_idx: &[usize], input_strength: &[i16], ticks: usize, input_ticks: usize) {
    for n in neurons.iter_mut() { n.reset(); }

    for tick in 0..ticks {
        // Input injection
        if tick < input_ticks {
            for (&idx, &strength) in input_idx.iter().zip(input_strength.iter()) {
                neurons[idx].charge += strength;
            }
        }

        // Scatter
        let mut incoming = vec![0i16; neurons.len()];
        for edge in edges {
            let act = neurons[edge.from].activation as i16;
            if act != 0 {
                incoming[edge.to] += act * neurons[edge.from].polarity as i16 * edge.weight;
            }
        }

        // Update
        for (i, inc) in incoming.iter().enumerate() {
            neurons[i].tick(*inc, tick);
        }
    }
}

fn main() {
    println!("=== RATE CODING MICROSCOPE ===\n");

    // Topology: IN_A(0), IN_B(1) → HID(2) → OUT(3), also IN_A→OUT, IN_B→OUT
    // Plus loop: OUT→HID feedback
    let edges = vec![
        Edge { from: 0, to: 2, weight: 2 },  // IN_A → HID
        Edge { from: 1, to: 2, weight: 2 },  // IN_B → HID
        Edge { from: 2, to: 3, weight: 3 },  // HID → OUT
        Edge { from: 0, to: 3, weight: 1 },  // IN_A → OUT direct
        Edge { from: 1, to: 3, weight: 1 },  // IN_B → OUT direct
    ];

    println!("Topology: IN_A→HID(w=2), IN_B→HID(w=2), HID→OUT(w=3), IN_A→OUT(w=1), IN_B→OUT(w=1)");
    println!("All neurons: thr=0 (fires easily), ch=1, refractory=1 tick\n");

    for &total_ticks in &[6, 24, 50, 100] {
        println!("--- {} ticks, input for first {} ticks ---", total_ticks, total_ticks.min(10));
        let input_ticks = total_ticks.min(10); // input active for first 10 ticks max

        println!("{:>6} {:>6} | {:>8} {:>8} {:>8} {:>8} | {:>10}",
            "A_in", "B_in", "IN_A_spk", "IN_B_spk", "HID_spk", "OUT_spk", "OUT_charge");
        println!("{:-<6} {:-<6}-+-{:-<8} {:-<8} {:-<8} {:-<8}-+-{:-<10}", "", "", "", "", "", "", "");

        for &(a_strength, b_strength) in &[
            (0i16, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2),
            (2, 1), (1, 2), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3),
            (4, 0), (4, 1), (4, 4), (5, 5),
        ] {
            let mut neurons = vec![
                Neuron::new("IN_A", 0, 1, 1),
                Neuron::new("IN_B", 0, 1, 1),
                Neuron::new("HID",  0, 1, 1),
                Neuron::new("OUT",  0, 1, 1),
            ];

            run(&mut neurons, &edges, &[0, 1], &[a_strength, b_strength], total_ticks, input_ticks);

            println!("{:>6} {:>6} | {:>8} {:>8} {:>8} {:>8} | {:>10}",
                a_strength, b_strength,
                neurons[0].spike_count, neurons[1].spike_count,
                neurons[2].spike_count, neurons[3].spike_count,
                neurons[3].charge);
        }

        // Check: does OUT_spk increase monotonically with A+B?
        let mut pairs: Vec<(i16, u32)> = Vec::new();
        for a in 0..=5i16 {
            for b in 0..=5i16 {
                let mut neurons = vec![
                    Neuron::new("IN_A", 0, 1, 1),
                    Neuron::new("IN_B", 0, 1, 1),
                    Neuron::new("HID",  0, 1, 1),
                    Neuron::new("OUT",  0, 1, 1),
                ];
                run(&mut neurons, &edges, &[0, 1], &[a, b], total_ticks, input_ticks);
                pairs.push((a + b, neurons[3].spike_count));
            }
        }
        // Check monotonicity
        let mut sum_to_spikes: std::collections::BTreeMap<i16, Vec<u32>> = std::collections::BTreeMap::new();
        for &(sum, spk) in &pairs { sum_to_spikes.entry(sum).or_default().push(spk); }

        print!("\n  SUM→OUT_spikes: ");
        let mut monotonic = true;
        let mut prev_mean = 0.0f64;
        for (&sum, spikes) in &sum_to_spikes {
            let mean = spikes.iter().sum::<u32>() as f64 / spikes.len() as f64;
            let min = *spikes.iter().min().unwrap();
            let max = *spikes.iter().max().unwrap();
            print!("{}→{:.0}({}-{}) ", sum, mean, min, max);
            if mean < prev_mean - 0.5 { monotonic = false; }
            prev_mean = mean;
        }
        if monotonic { println!(" ✓ MONOTONIC"); } else { println!(" ✗ NOT monotonic"); }
        println!();
    }
}
