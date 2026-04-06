//! Scatter-add reordering benchmark: original vs connectivity-reordered neuron layout.
//!
//! Tests whether reordering neuron IDs so that connected neurons have nearby IDs
//! improves cache performance of the scatter-add loop.
//!
//! Run: cargo run --example dense_bench --release

use instnct_core::{build_network, InitConfig, Network};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Reordering strategies
// ---------------------------------------------------------------------------

/// Identity permutation (no reorder = baseline)
fn perm_identity(h: usize) -> Vec<usize> {
    (0..h).collect()
}

/// Reverse Cuthill-McKee: BFS from highest-degree node, reverse the order.
/// Groups connected neurons together → nearby IDs → cache-friendly scatter.
fn perm_rcm(net: &Network) -> Vec<usize> {
    let h = net.neuron_count();

    // Build adjacency list (undirected) from edges
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); h];
    for edge in net.graph().iter_edges() {
        adj[edge.source as usize].push(edge.target as usize);
        adj[edge.target as usize].push(edge.source as usize);
    }

    // Find start node: highest degree
    let start = (0..h).max_by_key(|&n| adj[n].len()).unwrap_or(0);

    // BFS ordering
    let mut visited = vec![false; h];
    let mut order = Vec::with_capacity(h);
    let mut queue = VecDeque::new();
    queue.push_back(start);
    visited[start] = true;

    while let Some(node) = queue.pop_front() {
        order.push(node);
        // Sort neighbors by degree (ascending) for better RCM
        let mut neighbors: Vec<usize> = adj[node].iter()
            .copied()
            .filter(|&n| !visited[n])
            .collect();
        neighbors.sort_by_key(|&n| adj[n].len());
        neighbors.dedup();
        for n in neighbors {
            if !visited[n] {
                visited[n] = true;
                queue.push_back(n);
            }
        }
    }

    // Add any unvisited (disconnected) neurons
    for (n, &vis) in visited.iter().enumerate() {
        if !vis { order.push(n); }
    }

    // Reverse for Cuthill-McKee
    order.reverse();

    // Build permutation: perm[original_id] = new_id
    let mut perm = vec![0usize; h];
    for (new_id, &orig_id) in order.iter().enumerate() {
        perm[orig_id] = new_id;
    }
    perm
}

/// Activity-based reorder: sort neurons by firing frequency (most active first).
fn perm_activity(activation_counts: &[u32]) -> Vec<usize> {
    let h = activation_counts.len();
    let mut indices: Vec<usize> = (0..h).collect();
    indices.sort_by(|&a, &b| activation_counts[b].cmp(&activation_counts[a]));

    let mut perm = vec![0usize; h];
    for (new_id, &orig_id) in indices.iter().enumerate() {
        perm[orig_id] = new_id;
    }
    perm
}

// ---------------------------------------------------------------------------
// Reordered scatter-add
// ---------------------------------------------------------------------------

struct ReorderedCSR {
    offsets: Vec<u32>,  // [H+1] new-space
    targets: Vec<u16>,  // [E] targets in new-space
}

fn build_reordered_csr(net: &Network, perm: &[usize]) -> ReorderedCSR {
    let h = net.neuron_count();
    let mut edges_by_new_src: Vec<Vec<u16>> = vec![Vec::new(); h];

    for edge in net.graph().iter_edges() {
        let new_src = perm[edge.source as usize];
        let new_tgt = perm[edge.target as usize] as u16;
        edges_by_new_src[new_src].push(new_tgt);
    }

    // Sort targets within each source for sequential access
    for targets in &mut edges_by_new_src {
        targets.sort_unstable();
    }

    let mut offsets = Vec::with_capacity(h + 1);
    let mut targets = Vec::new();
    let mut offset = 0u32;
    for src_edges in &edges_by_new_src {
        offsets.push(offset);
        targets.extend_from_slice(src_edges);
        offset += src_edges.len() as u32;
    }
    offsets.push(offset);

    ReorderedCSR { offsets, targets }
}

fn scatter_add_reordered(
    csr: &ReorderedCSR,
    activation_reordered: &[i32],
    incoming: &mut [i32],
    h: usize,
) {
    incoming[..h].fill(0);
    for (neuron, &act) in activation_reordered[..h].iter().enumerate() {
        if act == 0 { continue; }
        let start = csr.offsets[neuron] as usize;
        let end = csr.offsets[neuron + 1] as usize;
        for &target in &csr.targets[start..end] {
            incoming[target as usize] += act;
        }
    }
}


// ---------------------------------------------------------------------------
// Measure average target "jump distance" (cache locality metric)
// ---------------------------------------------------------------------------

fn avg_jump_distance(csr: &ReorderedCSR, h: usize) -> f64 {
    let mut total_jump = 0u64;
    let mut count = 0u64;
    for src in 0..h {
        let start = csr.offsets[src] as usize;
        let end = csr.offsets[src + 1] as usize;
        for i in start + 1..end {
            let jump = (csr.targets[i] as i64 - csr.targets[i - 1] as i64).unsigned_abs();
            total_jump += jump;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { total_jump as f64 / count as f64 }
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

fn bench_one(h: usize) {
    let init = InitConfig::phi(h);
    let mut rng = StdRng::seed_from_u64(42);
    let net = build_network(&init, &mut rng);
    let edge_count = net.edge_count();

    // Random activation (~20% active)
    let activation: Vec<i32> = (0..h).map(|_| {
        if rng.gen_ratio(1, 5) { if rng.gen_bool(0.5) { 1 } else { -1 } } else { 0 }
    }).collect();

    // Simulate activity counts for activity-based reorder
    let activity_counts: Vec<u32> = activation.iter()
        .map(|&a| if a != 0 { 100 } else { 10 })
        .collect();

    // Build permutations
    let perm_id = perm_identity(h);
    let perm_r = perm_rcm(&net);
    let perm_a = perm_activity(&activity_counts);

    // Build reordered CSRs
    let csr_orig = build_reordered_csr(&net, &perm_id);
    let csr_rcm = build_reordered_csr(&net, &perm_r);
    let csr_act = build_reordered_csr(&net, &perm_a);

    // Reorder activation arrays
    let mut act_rcm = vec![0i32; h];
    let mut act_act = vec![0i32; h];
    for i in 0..h {
        act_rcm[perm_r[i]] = activation[i];
        act_act[perm_a[i]] = activation[i];
    }

    // Cache locality metric
    let jump_orig = avg_jump_distance(&csr_orig, h);
    let jump_rcm = avg_jump_distance(&csr_rcm, h);
    let jump_act = avg_jump_distance(&csr_act, h);

    let mut inc_orig = vec![0i32; h];
    let mut inc_rcm = vec![0i32; h];
    let mut inc_act = vec![0i32; h];

    let iters = if h <= 512 { 5000 } else if h <= 1024 { 1000 } else { 200 };

    // Bench original
    let t0 = Instant::now();
    for _ in 0..iters {
        scatter_add_reordered(&csr_orig, &activation, &mut inc_orig, h);
    }
    let ns_orig = t0.elapsed().as_nanos() / iters as u128;

    // Bench RCM reorder
    let t1 = Instant::now();
    for _ in 0..iters {
        scatter_add_reordered(&csr_rcm, &act_rcm, &mut inc_rcm, h);
    }
    let ns_rcm = t1.elapsed().as_nanos() / iters as u128;

    // Bench activity reorder
    let t2 = Instant::now();
    for _ in 0..iters {
        scatter_add_reordered(&csr_act, &act_act, &mut inc_act, h);
    }
    let ns_act = t2.elapsed().as_nanos() / iters as u128;

    let speedup_rcm = ns_orig as f64 / ns_rcm as f64;
    let speedup_act = ns_orig as f64 / ns_act as f64;

    println!("  H={:<5} edges={:<7} | orig: {:>7}ns (jump={:.0}) | RCM: {:>7}ns (jump={:.0}) {:.2}x | activity: {:>7}ns (jump={:.0}) {:.2}x",
        h, edge_count, ns_orig, jump_orig, ns_rcm, jump_rcm, speedup_rcm, ns_act, jump_act, speedup_act);
}

fn main() {
    println!("=== Scatter-Add Reordering Benchmark ===\n");
    println!("  Reorder strategies:");
    println!("    Original:  neuron ID = memory position (baseline)");
    println!("    RCM:       Reverse Cuthill-McKee (connected neurons adjacent)");
    println!("    Activity:  most active neurons first\n");

    for &h in &[256, 512, 1024, 2048] {
        bench_one(h);
    }
}
