//! Frankenstein: Drosophila larva mushroom body topology + INSTNCT spiking dynamics
//!
//! Use REAL biological connectivity (321 neurons, 16K edges) with our spike model.
//! Compare: bio topology vs random topology vs no edges.
//!
//! RUNNING: flybrain
//!
//! Run: cargo run --example flybrain --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::time::Instant;

const TICKS: usize = 50;
const INPUT_TICKS: usize = 20;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

#[derive(Clone)]
struct BioSpikeNet {
    edges: Vec<(u16, u16, i8)>,   // (src, tgt, weight)
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    charge: Vec<i16>,
    activation: Vec<i8>,
    refractory: Vec<u8>,
    spike_count: Vec<u32>,
    h: usize,
    input_neurons: Vec<usize>,     // ORN/PN indices
    output_neurons: Vec<usize>,    // MBON indices
    hidden_neurons: Vec<usize>,    // KC indices
    neuron_class: Vec<String>,
}

impl BioSpikeNet {
    fn new_from_graphml(path: &str, rng: &mut impl Rng) -> Self {
        use std::io::Read;
        let mut xml = String::new();
        std::fs::File::open(path).unwrap().read_to_string(&mut xml).unwrap();

        // Simple XML parsing (no dependency needed)
        let mut nodes: Vec<(String, String, String)> = Vec::new(); // (id, class, hemisphere)
        let mut edges: Vec<(String, String, f64)> = Vec::new(); // (src, tgt, weight)

        // Parse nodes
        let mut i = 0;
        while let Some(pos) = xml[i..].find("<node ") {
            let start = i + pos;
            let end = xml[start..].find("</node>").map(|p| start + p + 7).unwrap_or(xml.len());
            let chunk = &xml[start..end];

            let id = extract_attr(chunk, "id");
            let class = extract_data(chunk, "d1");
            let hemi = extract_data(chunk, "d2");
            nodes.push((id, class, hemi));
            i = end;
        }

        // Parse edges
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

        // Map string IDs to compact indices
        let mut id_map: HashMap<String, usize> = HashMap::new();
        for (idx, (id, _, _)) in nodes.iter().enumerate() {
            id_map.insert(id.clone(), idx);
        }

        let h = nodes.len();
        let mut input_neurons = Vec::new();
        let mut output_neurons = Vec::new();
        let mut hidden_neurons = Vec::new();
        let mut neuron_class = Vec::new();

        let mut threshold = vec![0u8; h];
        let mut channel = vec![0u8; h];
        let mut polarity = vec![1i8; h];

        for (idx, (_, class, _)) in nodes.iter().enumerate() {
            neuron_class.push(class.clone());
            if class.contains("ORN") || class.contains("PN") || class.contains("Gust") {
                input_neurons.push(idx);
            } else if class.contains("MBON") {
                output_neurons.push(idx);
            } else if class.contains("KC") {
                hidden_neurons.push(idx);
            }
            // Bio-inspired params
            threshold[idx] = rng.gen_range(8..=14); // HIGH threshold for real weights (up to 65)
            channel[idx] = rng.gen_range(1..=8);
            if class.contains("APL") { polarity[idx] = -1; } // APL is inhibitory in the fly
            else if rng.gen_ratio(2, 10) { polarity[idx] = -1; } // 20% random inhibitory
        }

        // Convert edges with weight quantization
        let mut bio_edges = Vec::new();
        for (src_id, tgt_id, w) in &edges {
            if let (Some(&s), Some(&t)) = (id_map.get(src_id), id_map.get(tgt_id)) {
                // Real synapse count — no clamping! i8 fits (max=65 < 127)
                bio_edges.push((s as u16, t as u16, (*w as i8).max(1)));
            }
        }

        BioSpikeNet {
            edges: bio_edges, threshold, channel, polarity,
            charge: vec![0; h], activation: vec![0; h],
            refractory: vec![0; h], spike_count: vec![0; h],
            h, input_neurons, output_neurons, hidden_neurons, neuron_class,
        }
    }

    fn reset(&mut self) {
        self.charge.iter_mut().for_each(|c| *c = 0);
        self.activation.iter_mut().for_each(|a| *a = 0);
        self.refractory.iter_mut().for_each(|r| *r = 0);
        self.spike_count.iter_mut().for_each(|s| *s = 0);
    }

    fn propagate(&mut self, tick: usize) {
        let h = self.h;
        let mut incoming = vec![0i16; h];
        for &(src, tgt, weight) in &self.edges {
            let s = src as usize; let t = tgt as usize;
            if s >= h || t >= h { continue; }
            let act = self.activation[s];
            if act != 0 {
                incoming[t] = incoming[t].saturating_add(act as i16 * self.polarity[s] as i16 * weight as i16);
            }
        }

        for i in 0..h {
            if self.refractory[i] > 0 { self.refractory[i] -= 1; self.activation[i] = 0; continue; }
            self.charge[i] = self.charge[i].saturating_add(incoming[i]);
            if tick % 6 == 5 && self.charge[i] > 0 { self.charge[i] -= 1; }
            if self.charge[i] < 0 { self.charge[i] = 0; }
            let pi = (tick as u8 + 9 - self.channel[i]) & 7;
            let pm = PHASE_BASE[pi as usize];
            if self.charge[i] * 10 >= (self.threshold[i] as i16 + 1) * pm {
                self.activation[i] = 1; self.charge[i] = 0; self.refractory[i] = 1;
                self.spike_count[i] += 1;
            } else { self.activation[i] = 0; }
        }
    }

    fn inject_input(&mut self, pattern: &[i8]) {
        for (idx, &neuron_idx) in self.input_neurons.iter().enumerate() {
            if idx < pattern.len() && pattern[idx] != 0 {
                self.charge[neuron_idx] += pattern[idx] as i16 * 2; // strong input
            }
        }
    }
}

fn extract_attr(chunk: &str, name: &str) -> String {
    let search = format!("{}=\"", name);
    if let Some(pos) = chunk.find(&search) {
        let start = pos + search.len();
        if let Some(end) = chunk[start..].find('"') {
            return chunk[start..start + end].to_string();
        }
    }
    String::new()
}

fn extract_data(chunk: &str, key: &str) -> String {
    let search = format!("key=\"{}\"", key);
    if let Some(pos) = chunk.find(&search) {
        let after = &chunk[pos..];
        if let Some(gt) = after.find('>') {
            let rest = &after[gt + 1..];
            if let Some(lt) = rest.find('<') {
                return rest[..lt].to_string();
            }
        }
    }
    String::new()
}

fn main() {
    println!("=== FRANKENSTEIN: Fly Mushroom Body + INSTNCT Spiking ===");
    println!("RUNNING: flybrain\n");

    let mut rng = StdRng::seed_from_u64(42);
    let mut net = BioSpikeNet::new_from_graphml("/home/deck/work/flywire/mushroom_body.graphml", &mut rng);

    println!("Mushroom Body loaded:");
    println!("  Neurons: {} (input={}, hidden/KC={}, output/MBON={})",
        net.h, net.input_neurons.len(), net.hidden_neurons.len(), net.output_neurons.len());
    println!("  Edges: {} (avg {:.1}/neuron)", net.edges.len(), net.edges.len() as f64 / net.h as f64);

    // Class distribution
    let mut class_counts: HashMap<String, usize> = HashMap::new();
    for c in &net.neuron_class { *class_counts.entry(c.clone()).or_default() += 1; }
    let mut sorted: Vec<_> = class_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    println!("  Classes: {:?}\n", sorted.iter().map(|(k,v)| format!("{}={}", k, v)).collect::<Vec<_>>().join(", "));

    // Test 1: inject different input strengths, observe output spike counts
    println!("--- Test 1: Input strength → output spike rate ---");
    println!("  Inject into {} input neurons for {} ticks, observe {} output neurons over {} ticks\n",
        net.input_neurons.len(), INPUT_TICKS, net.output_neurons.len(), TICKS);

    for &input_strength in &[0i8, 1, 2, 3, 5, 8] {
        net.reset();
        let pattern: Vec<i8> = (0..net.input_neurons.len()).map(|_| input_strength).collect();

        for tick in 0..TICKS {
            if tick < INPUT_TICKS { net.inject_input(&pattern); }
            net.propagate(tick);
        }

        let input_spikes: u32 = net.input_neurons.iter().map(|&i| net.spike_count[i]).sum();
        let hidden_spikes: u32 = net.hidden_neurons.iter().map(|&i| net.spike_count[i]).sum();
        let output_spikes: u32 = net.output_neurons.iter().map(|&i| net.spike_count[i]).sum();
        let active_output = net.output_neurons.iter().filter(|&&i| net.spike_count[i] > 0).count();

        println!("  input={}: IN_spk={:>5} KC_spk={:>5} MBON_spk={:>5} ({}/{} active)",
            input_strength, input_spikes, hidden_spikes, output_spikes, active_output, net.output_neurons.len());
    }

    // Test 2: different input PATTERNS (stimulate different subsets of input neurons)
    println!("\n--- Test 2: Different input patterns → different output patterns? ---\n");

    let n_inputs = net.input_neurons.len();
    let patterns: Vec<(&str, Vec<i8>)> = vec![
        ("all_off",  vec![0; n_inputs]),
        ("all_on",   vec![3; n_inputs]),
        ("first_half", (0..n_inputs).map(|i| if i < n_inputs/2 { 3 } else { 0 }).collect()),
        ("second_half", (0..n_inputs).map(|i| if i >= n_inputs/2 { 3 } else { 0 }).collect()),
        ("odds",     (0..n_inputs).map(|i| if i % 2 == 0 { 3 } else { 0 }).collect()),
        ("evens",    (0..n_inputs).map(|i| if i % 2 == 1 { 3 } else { 0 }).collect()),
        ("one_hot_0", { let mut p = vec![0; n_inputs]; p[0] = 5; p }),
        ("one_hot_5", { let mut p = vec![0; n_inputs]; if n_inputs > 5 { p[5] = 5; } p }),
    ];

    println!("{:>15} | {:>8} {:>8} {:>8} | output spike pattern (top 5 MBONs)",
        "pattern", "IN_spk", "KC_spk", "OUT_spk");
    println!("{:-<15}-+-{:-<8} {:-<8} {:-<8}-+------", "", "", "", "");

    for (name, pattern) in &patterns {
        net.reset();
        for tick in 0..TICKS {
            if tick < INPUT_TICKS { net.inject_input(pattern); }
            net.propagate(tick);
        }

        let in_spk: u32 = net.input_neurons.iter().map(|&i| net.spike_count[i]).sum();
        let kc_spk: u32 = net.hidden_neurons.iter().map(|&i| net.spike_count[i]).sum();
        let out_spk: u32 = net.output_neurons.iter().map(|&i| net.spike_count[i]).sum();

        // Top 5 output neurons by spike count
        let mut out_sorted: Vec<(usize, u32)> = net.output_neurons.iter().map(|&i| (i, net.spike_count[i])).collect();
        out_sorted.sort_by(|a, b| b.1.cmp(&a.1));
        let top5: String = out_sorted.iter().take(5).filter(|(_,s)| *s > 0)
            .map(|(i, s)| format!("{}:{}", i, s)).collect::<Vec<_>>().join(" ");

        println!("{:>15} | {:>8} {:>8} {:>8} | {}", name, in_spk, kc_spk, out_spk, top5);
    }

    // Test 3: Compare bio topology vs random topology
    println!("\n--- Test 3: Bio topology vs random topology (same neuron count + edge count) ---\n");

    let bio_edge_count = net.edges.len();
    let bio_h = net.h;

    // Random topology with same stats
    let mut random_net = net.clone();
    // Shuffle bio weights randomly across random edges (same weight distribution!)
    let bio_weights: Vec<i8> = net.edges.iter().map(|e| e.2).collect();
    random_net.edges.clear();
    let mut r2 = StdRng::seed_from_u64(9999);
    let mut wi = 0;
    while random_net.edges.len() < bio_edge_count {
        let s = r2.gen_range(0..bio_h) as u16;
        let t = r2.gen_range(0..bio_h) as u16;
        if s != t {
            let w = bio_weights[wi % bio_weights.len()];
            random_net.edges.push((s, t, w));
            wi += 1;
        }
    }

    for (label, network) in [("BIO", &mut net), ("RANDOM", &mut random_net)] {
        network.reset();
        let pattern = vec![3i8; n_inputs];
        for tick in 0..TICKS {
            if tick < INPUT_TICKS { network.inject_input(&pattern); }
            network.propagate(tick);
        }
        let in_spk: u32 = network.input_neurons.iter().map(|&i| network.spike_count[i]).sum();
        let kc_spk: u32 = network.hidden_neurons.iter().map(|&i| network.spike_count[i]).sum();
        let out_spk: u32 = network.output_neurons.iter().map(|&i| network.spike_count[i]).sum();
        let active_kc = network.hidden_neurons.iter().filter(|&&i| network.spike_count[i] > 0).count();
        let active_out = network.output_neurons.iter().filter(|&&i| network.spike_count[i] > 0).count();

        println!("  {}: IN={} KC={} (active {}/{}) MBON={} (active {}/{})",
            label, in_spk, kc_spk, active_kc, network.hidden_neurons.len(),
            out_spk, active_out, network.output_neurons.len());
    }
}
