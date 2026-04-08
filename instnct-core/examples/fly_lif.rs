//! Fly LIF: exact Drosophila parameters on mushroom body topology.
//! Two variables per neuron: g (synaptic current) + v (membrane voltage).
//! Exponential decay. Real weights from connectome.
//!
//! RUNNING: fly_lif
//!
//! Run: cargo run --example fly_lif --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

const TICKS: usize = 100;
const INPUT_TICKS: usize = 30;

#[derive(Clone)]
struct LIFNeuron {
    v: i32,           // membrane voltage (rest=0, threshold=7)
    g: i32,           // synaptic current (decays fast)
    refractory: u8,   // ticks remaining in refractory
    spike_count: u32,
    firing: bool,
}

impl LIFNeuron {
    fn new() -> Self { LIFNeuron { v: 0, g: 0, refractory: 0, spike_count: 0, firing: false } }

    fn step(&mut self) {
        if self.refractory > 0 {
            self.refractory -= 1;
            self.firing = false;
            return;
        }

        // Synaptic current decays: g -= g/5 (tau_syn = 5ms at 1ms/tick)
        self.g -= self.g / 5;

        // Membrane integration: v += (v_rest - v + g) / tau_m
        // v_rest = 0, tau_m = 20 ticks → v += (0 - v + g) / 20
        self.v += (-self.v + self.g) / 20;

        // Spike check
        if self.v >= 7 {
            self.firing = true;
            self.v = 0;     // reset
            self.g = 0;     // reset synaptic current too
            self.refractory = 2;
            self.spike_count += 1;
        } else {
            self.firing = false;
        }
    }
}

#[derive(Clone)]
struct FlyNet {
    neurons: Vec<LIFNeuron>,
    edges: Vec<(u16, u16, i16)>,  // (src, tgt, weight) — weight = synapse_count × sign
    input_neurons: Vec<usize>,
    output_neurons: Vec<usize>,
    hidden_neurons: Vec<usize>,
    neuron_class: Vec<String>,
    h: usize,
}

impl FlyNet {
    fn from_graphml(path: &str) -> Self {
        use std::io::Read;
        let mut xml = String::new();
        std::fs::File::open(path).unwrap().read_to_string(&mut xml).unwrap();

        let mut nodes: Vec<(String, String)> = Vec::new();
        let mut edges: Vec<(String, String, f64)> = Vec::new();

        // Parse nodes
        let mut i = 0;
        while let Some(pos) = xml[i..].find("<node ") {
            let start = i + pos;
            let end = xml[start..].find("</node>").map(|p| start + p + 7).unwrap_or(xml.len());
            let chunk = &xml[start..end];
            let id = extract_attr(chunk, "id");
            let class = extract_data(chunk, "d1");
            nodes.push((id, class));
            i = end;
        }

        i = 0;
        while let Some(pos) = xml[i..].find("<edge ") {
            let start = i + pos;
            let end = xml[start..].find("/>").or_else(|| xml[start..].find("</edge>"))
                .map(|p| start + p + 2).unwrap_or(xml.len());
            let chunk = &xml[start..end];
            let src = extract_attr(chunk, "source");
            let tgt = extract_attr(chunk, "target");
            let w: f64 = extract_data(chunk, "d3").parse().unwrap_or(1.0);
            if src != tgt { edges.push((src, tgt, w)); }
            i = end;
        }

        let mut id_map: HashMap<String, usize> = HashMap::new();
        for (idx, (id, _)) in nodes.iter().enumerate() { id_map.insert(id.clone(), idx); }

        let h = nodes.len();
        let mut input_neurons = Vec::new();
        let mut output_neurons = Vec::new();
        let mut hidden_neurons = Vec::new();
        let mut neuron_class = Vec::new();

        for (idx, (_, class)) in nodes.iter().enumerate() {
            neuron_class.push(class.clone());
            if class.contains("ORN") || class.contains("PN") || class.contains("Gust") {
                input_neurons.push(idx);
            } else if class.contains("MBON") {
                output_neurons.push(idx);
            } else if class.contains("KC") {
                hidden_neurons.push(idx);
            }
        }

        // Convert edges with REAL weights — sign from neurotransmitter
        // In mushroom body: most are excitatory (ACh), APL is inhibitory (GABA)
        let mut bio_edges = Vec::new();
        for (src_id, tgt_id, w) in &edges {
            if let (Some(&s), Some(&t)) = (id_map.get(src_id), id_map.get(tgt_id)) {
                // Sign: APL neurons are GABAergic (inhibitory)
                let sign: i16 = if neuron_class[s].contains("APL") { -1 } else { 1 };
                let weight = (*w as i16) * sign;
                bio_edges.push((s as u16, t as u16, weight));
            }
        }

        let neurons = vec![LIFNeuron::new(); h];
        FlyNet { neurons, edges: bio_edges, input_neurons, output_neurons, hidden_neurons, neuron_class, h }
    }

    fn reset(&mut self) {
        for n in &mut self.neurons { *n = LIFNeuron::new(); }
    }

    fn step(&mut self) {
        // Deliver spikes: if src fired, add weight to target's g
        let mut g_incoming = vec![0i32; self.h];
        for &(src, tgt, weight) in &self.edges {
            let s = src as usize; let t = tgt as usize;
            if s >= self.h || t >= self.h { continue; }
            if self.neurons[s].firing {
                g_incoming[t] += weight as i32;
            }
        }

        // Apply incoming to g, then step each neuron
        for i in 0..self.h {
            self.neurons[i].g += g_incoming[i];
            self.neurons[i].step();
        }
    }

    fn inject(&mut self, pattern: &[i32]) {
        for (idx, &neuron_idx) in self.input_neurons.iter().enumerate() {
            if idx < pattern.len() {
                self.neurons[neuron_idx].g += pattern[idx]; // inject into synaptic current
            }
        }
    }
}

fn extract_attr(chunk: &str, name: &str) -> String {
    let search = format!("{}=\"", name);
    if let Some(pos) = chunk.find(&search) {
        let start = pos + search.len();
        if let Some(end) = chunk[start..].find('"') { return chunk[start..start+end].to_string(); }
    }
    String::new()
}
fn extract_data(chunk: &str, key: &str) -> String {
    let search = format!("key=\"{}\"", key);
    if let Some(pos) = chunk.find(&search) {
        let after = &chunk[pos..];
        if let Some(gt) = after.find('>') { let rest = &after[gt+1..];
            if let Some(lt) = rest.find('<') { return rest[..lt].to_string(); } }
    }
    String::new()
}

fn main() {
    println!("=== FLY LIF: Exact Drosophila parameters ===");
    println!("RUNNING: fly_lif");
    println!("Two variables: g (synaptic current) + v (membrane voltage)");
    println!("Decay: g -= g/5 (tau=5ms), v += (-v + g)/20 (tau=20ms)");
    println!("Threshold: v >= 7, Reset: v=0 g=0, Refractory: 2 ticks\n");

    let mut net = FlyNet::from_graphml("/home/deck/work/flywire/mushroom_body.graphml");
    println!("Loaded: {} neurons, {} edges", net.h, net.edges.len());
    println!("  Input(ORN/PN): {}, Hidden(KC): {}, Output(MBON): {}\n",
        net.input_neurons.len(), net.hidden_neurons.len(), net.output_neurons.len());

    // Test 1: Different input strengths
    println!("--- Test 1: Input strength → spike rate ({} ticks) ---\n", TICKS);
    for &input_g in &[0i32, 1, 2, 5, 10, 20, 50] {
        net.reset();
        let pattern: Vec<i32> = vec![input_g; net.input_neurons.len()];

        for tick in 0..TICKS {
            if tick < INPUT_TICKS { net.inject(&pattern); }
            net.step();
        }

        let in_spk: u32 = net.input_neurons.iter().map(|&i| net.neurons[i].spike_count).sum();
        let kc_spk: u32 = net.hidden_neurons.iter().map(|&i| net.neurons[i].spike_count).sum();
        let out_spk: u32 = net.output_neurons.iter().map(|&i| net.neurons[i].spike_count).sum();
        let active_kc = net.hidden_neurons.iter().filter(|&&i| net.neurons[i].spike_count > 0).count();
        let active_out = net.output_neurons.iter().filter(|&&i| net.neurons[i].spike_count > 0).count();

        println!("  g_input={:>3}: IN={:>5} KC={:>5} ({:>3}/{} active) MBON={:>5} ({:>2}/{} active)",
            input_g, in_spk, kc_spk, active_kc, net.hidden_neurons.len(),
            out_spk, active_out, net.output_neurons.len());
    }

    // Test 2: Different patterns — do outputs differentiate?
    println!("\n--- Test 2: Different input patterns ---\n");
    let n_in = net.input_neurons.len();
    let patterns: Vec<(&str, Vec<i32>)> = vec![
        ("all_off",     vec![0; n_in]),
        ("all_10",      vec![10; n_in]),
        ("first_half",  (0..n_in).map(|i| if i < n_in/2 { 10 } else { 0 }).collect()),
        ("second_half", (0..n_in).map(|i| if i >= n_in/2 { 10 } else { 0 }).collect()),
        ("one_hot_0",   { let mut p = vec![0; n_in]; p[0] = 20; p }),
        ("one_hot_10",  { let mut p = vec![0; n_in]; if n_in > 10 { p[10] = 20; } p }),
        ("gradient",    (0..n_in).map(|i| (i * 20 / n_in) as i32).collect()),
    ];

    for (name, pattern) in &patterns {
        net.reset();
        for tick in 0..TICKS {
            if tick < INPUT_TICKS { net.inject(pattern); }
            net.step();
        }
        let kc_spk: u32 = net.hidden_neurons.iter().map(|&i| net.neurons[i].spike_count).sum();
        let out_spk: u32 = net.output_neurons.iter().map(|&i| net.neurons[i].spike_count).sum();
        let active_kc = net.hidden_neurons.iter().filter(|&&i| net.neurons[i].spike_count > 0).count();

        // Output pattern fingerprint: top 5 MBONs
        let mut mbon_spikes: Vec<(usize, u32)> = net.output_neurons.iter()
            .map(|&i| (i, net.neurons[i].spike_count)).collect();
        mbon_spikes.sort_by(|a, b| b.1.cmp(&a.1));
        let top5: String = mbon_spikes.iter().take(5)
            .map(|(i, s)| format!("{}:{}", i, s)).collect::<Vec<_>>().join(" ");

        println!("  {:>12}: KC={:>5}({:>3} active) MBON={:>5} | {}", name, kc_spk, active_kc, out_spk, top5);
    }

    // Test 3: Trace a few neurons tick by tick
    println!("\n--- Test 3: Neuron trace (first 30 ticks, input=10) ---\n");
    net.reset();
    let pattern = vec![10i32; n_in];

    // Pick: first input, first KC, first MBON
    let trace_neurons = vec![
        (net.input_neurons[0], "IN[0]"),
        (net.hidden_neurons[0], "KC[0]"),
        (net.output_neurons[0], "MBON[0]"),
    ];

    println!("{:>4} | {:>16} | {:>16} | {:>16}",
        "tick", "IN[0] v/g/spk", "KC[0] v/g/spk", "MBON[0] v/g/spk");
    println!("{:-<4}-+-{:-<16}-+-{:-<16}-+-{:-<16}", "", "", "", "");

    for tick in 0..30 {
        if tick < INPUT_TICKS { net.inject(&pattern); }
        net.step();

        let vals: Vec<String> = trace_neurons.iter().map(|&(idx, _)| {
            let n = &net.neurons[idx];
            let fire = if n.firing { "^" } else { "." };
            format!("{:>4}/{:>5}/{}", n.v, n.g, fire)
        }).collect();

        println!("{:>4} | {:>16} | {:>16} | {:>16}", tick, vals[0], vals[1], vals[2]);
    }
}
