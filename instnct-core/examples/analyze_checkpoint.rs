//! Deep topology analysis of a checkpoint — loops, hubs, signal paths, zone flow.
//!
//! Usage: cargo run --release --example analyze_checkpoint -- <checkpoint.bin>

use instnct_core::{load_checkpoint, InitConfig};
use std::collections::{HashMap, VecDeque};
use std::env;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: analyze_checkpoint <checkpoint.bin> [checkpoint2.bin ...]");
        std::process::exit(1);
    }

    for path in &args {
        println!("\n{}", "=".repeat(70));
        println!("  Analyzing: {path}");
        println!("{}\n", "=".repeat(70));
        analyze_one(path);
    }
}

fn analyze_one(path: &str) {
    let (net, _proj, meta) = load_checkpoint(path).expect("failed to load checkpoint");
    let h = net.neuron_count();

    // Reconstruct InitConfig to get zone boundaries
    let init = InitConfig::phi(h);
    let input_end = init.input_end();
    let output_start = init.output_start();
    let overlap_start = output_start;
    let overlap_end = input_end;

    println!("  Step: {}  Accuracy: {:.2}%  Label: {}", meta.step, meta.accuracy * 100.0, meta.label);
    println!("  H={h}, phi_dim={}, input=0..{input_end}, output={output_start}..{h}", init.phi_dim);
    println!("  Overlap: {overlap_start}..{overlap_end} ({} neurons)", overlap_end.saturating_sub(overlap_start));

    // ── Build adjacency ──
    let edges = net.edge_count();
    let density = edges as f64 / (h * h) as f64 * 100.0;
    let mut fwd: Vec<Vec<u16>> = vec![Vec::new(); h];
    let mut rev: Vec<Vec<u16>> = vec![Vec::new(); h];
    for edge in net.graph().iter_edges() {
        fwd[edge.source as usize].push(edge.target);
        rev[edge.target as usize].push(edge.source);
    }

    println!("\n  --- BASIC STATS ---");
    println!("  Edges: {edges}, density: {density:.2}%");

    // Degree distribution
    let mut in_deg = vec![0u32; h];
    let mut out_deg = vec![0u32; h];
    for i in 0..h {
        out_deg[i] = fwd[i].len() as u32;
        in_deg[i] = rev[i].len() as u32;
    }
    let avg_deg = edges as f64 / h as f64;
    let max_in = *in_deg.iter().max().unwrap_or(&0);
    let max_out = *out_deg.iter().max().unwrap_or(&0);
    let dead = (0..h).filter(|&i| in_deg[i] == 0 && out_deg[i] == 0).count();
    println!("  Avg degree: {avg_deg:.1}, max_in: {max_in}, max_out: {max_out}");
    println!("  Dead neurons: {dead}/{h}");

    // Top-5 hub neurons
    let mut by_total: Vec<(usize, u32)> = (0..h).map(|i| (i, in_deg[i] + out_deg[i])).collect();
    by_total.sort_by(|a, b| b.1.cmp(&a.1));
    println!("\n  --- TOP-10 HUB NEURONS ---");
    println!("  {:>4} {:>4} {:>4} {:>5} {:>6} {:>5} {:>4} {:>4}",
        "ID", "In", "Out", "Total", "Zone", "Thr", "Ch", "Pol");
    for &(id, total) in by_total.iter().take(10) {
        let zone = if id < overlap_start { "input" }
            else if id >= overlap_end && id < output_start { "hidden" }  // shouldn't exist if overlap_start == output_start
            else if id < overlap_end { "overlap" }
            else { "output" };
        let sd = &net.spike_data()[id];
        let pol = if net.polarity()[id] > 0 { "+" } else { "-" };
        println!("  {:>4} {:>4} {:>4} {:>5} {:>6} {:>5} {:>4} {:>4}",
            id, in_deg[id], out_deg[id], total, zone, sd.threshold, sd.channel, pol);
    }

    // ── Bidirectional pairs (2-cycles) ──
    let mut bidir = 0u32;
    for edge in net.graph().iter_edges() {
        if edge.source < edge.target && net.graph().has_edge(edge.target, edge.source) {
            bidir += 1;
        }
    }
    println!("\n  --- LOOPS & CYCLES ---");
    println!("  Bidirectional pairs (2-cycles): {bidir}");

    // Triangles (3-cycles)
    let mut triangles = 0u32;
    for a in 0..h {
        for &b in &fwd[a] {
            for &c in &fwd[b as usize] {
                if fwd[c as usize].contains(&(a as u16)) {
                    triangles += 1;
                }
            }
        }
    }
    triangles /= 3; // each counted 3×
    println!("  Triangles (3-cycles): {triangles}");

    // 4-cycles (approximate — sample)
    let mut four_cycles = 0u32;
    for a in 0..h.min(100) { // sample first 100 neurons
        for &b in &fwd[a] {
            for &c in &fwd[b as usize] {
                if c as usize == a { continue; }
                for &d in &fwd[c as usize] {
                    if d as usize == a || d as usize == b as usize { continue; }
                    if fwd[d as usize].contains(&(a as u16)) {
                        four_cycles += 1;
                    }
                }
            }
        }
    }
    println!("  4-cycles (sampled first 100 neurons): ~{four_cycles}");

    // Self-loops in recurrence (neurons that can reach themselves)
    let mut self_reachable = 0u32;
    for start in 0..h {
        let mut visited = vec![false; h];
        let mut queue = VecDeque::new();
        for &t in &fwd[start] {
            if t as usize == start { self_reachable += 1; break; } // direct self (shouldn't exist)
            if !visited[t as usize] { visited[t as usize] = true; queue.push_back((t as usize, 1u32)); }
        }
        let mut found = false;
        while let Some((node, depth)) = queue.pop_front() {
            if depth > 6 { continue; } // max 6 hops
            for &t in &fwd[node] {
                if t as usize == start { found = true; break; }
                if !visited[t as usize] { visited[t as usize] = true; queue.push_back((t as usize, depth + 1)); }
            }
            if found { break; }
        }
        if found { self_reachable += 1; }
    }
    println!("  Neurons in recurrent loops (≤6 hops): {self_reachable}/{h} ({:.0}%)",
        self_reachable as f64 / h as f64 * 100.0);

    // ── Zone flow analysis ──
    println!("\n  --- ZONE FLOW ---");
    let mut zone_flow: HashMap<(&str, &str), u32> = HashMap::new();
    let zone_name = |n: usize| -> &str {
        if n < overlap_start { "input-only" }
        else if n < overlap_end { "overlap" }
        else { "output-only" }
    };
    for edge in net.graph().iter_edges() {
        let src_zone = zone_name(edge.source as usize);
        let tgt_zone = zone_name(edge.target as usize);
        *zone_flow.entry((src_zone, tgt_zone)).or_insert(0) += 1;
    }
    let mut flows: Vec<_> = zone_flow.iter().collect();
    flows.sort_by(|a, b| b.1.cmp(a.1));
    for ((src, tgt), count) in &flows {
        println!("  {src:>12} → {tgt:<12}: {count}");
    }

    // ── Signal path analysis: BFS depth from input to output ──
    println!("\n  --- SIGNAL PATH DEPTH ---");
    // BFS from each input neuron, measure min hops to reach output zone
    let input_neurons: Vec<usize> = (0..input_end).collect();
    let mut min_depth_to_output = vec![u32::MAX; h];
    let mut queue = VecDeque::new();
    for &inp in &input_neurons {
        min_depth_to_output[inp] = 0;
        queue.push_back((inp, 0u32));
    }
    while let Some((node, depth)) = queue.pop_front() {
        if depth > 10 { continue; }
        for &tgt in &fwd[node] {
            if min_depth_to_output[tgt as usize] > depth + 1 {
                min_depth_to_output[tgt as usize] = depth + 1;
                queue.push_back((tgt as usize, depth + 1));
            }
        }
    }
    // How many output neurons reachable, at what depth?
    let mut depth_hist = vec![0u32; 11]; // 0..10
    let mut unreachable = 0u32;
    for n in output_start..h {
        let d = min_depth_to_output[n];
        if d == u32::MAX { unreachable += 1; }
        else if (d as usize) < depth_hist.len() { depth_hist[d as usize] += 1; }
    }
    let output_count = h - output_start;
    println!("  Output neurons reachable from input: {}/{output_count}",
        output_count as u32 - unreachable);
    println!("  Unreachable: {unreachable}");
    println!("  Depth distribution (hops from input to output neuron):");
    for (d, &count) in depth_hist.iter().enumerate() {
        if count > 0 {
            let bar: String = "#".repeat((count as usize).min(50));
            println!("    {d:>2} hops: {count:>4} {bar}");
        }
    }

    // ── Tick compatibility ──
    // With ticks_per_token=6, signal can travel at most 6 edges per token
    let reachable_in_ticks = depth_hist.iter().take(7).sum::<u32>(); // 0-6 hops
    println!("\n  Reachable within 6 ticks (1 token): {reachable_in_ticks}/{output_count} output neurons");

    // ── Parameter distributions ──
    println!("\n  --- NEURON PARAMETERS ---");
    let thresholds: Vec<u8> = net.spike_data().iter().map(|s| s.threshold).collect();
    let channels: Vec<u8> = net.spike_data().iter().map(|s| s.channel).collect();
    let polarities: Vec<i8> = net.polarity().to_vec();

    let n_inhib = polarities.iter().filter(|&&p| p < 0).count();
    let avg_th = thresholds.iter().map(|&t| t as f64).sum::<f64>() / h as f64;
    println!("  Inhibitory: {n_inhib}/{h} ({:.0}%)", n_inhib as f64 / h as f64 * 100.0);
    println!("  Threshold: avg={avg_th:.1}, distribution:");
    let mut th_hist = vec![0u32; 16];
    for &t in &thresholds { th_hist[t as usize] += 1; }
    for (t, &count) in th_hist.iter().enumerate() {
        if count > 0 {
            let bar: String = "#".repeat((count as usize).min(50));
            println!("    th={t:>2}: {count:>4} {bar}");
        }
    }

    println!("  Channel distribution:");
    let mut ch_hist = vec![0u32; 9];
    for &c in &channels { ch_hist[c as usize] += 1; }
    for (c, &count) in ch_hist.iter().enumerate() {
        if count > 0 {
            let bar: String = "#".repeat((count as usize).min(50));
            println!("    ch={c}: {count:>4} {bar}");
        }
    }
}
