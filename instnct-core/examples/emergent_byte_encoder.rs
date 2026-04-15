//! Emergent byte encoder — INSTNCT spiking network evolves its own encoding
//!
//! No sum, no weights, no activation function defined.
//! Just: input neurons (8 bits) → evolved sparse topology → output neurons
//! Evolution optimizes topology until output pattern = unique per byte.
//!
//! Run: cargo run --example emergent_byte_encoder --release

use instnct_core::{build_network, InitConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn main() {
    let t0 = Instant::now();

    println!("=== EMERGENT BYTE ENCODER — INSTNCT SPIKING ===\n");
    println!("  No predefined sum/weight/activation!");
    println!("  Just: 8 input bits → evolved topology → output pattern\n");

    // Build a small INSTNCT network
    // H neurons, 8 input, N output — evolution finds topology
    for &h in &[16, 32, 64, 128] {
        let tc = Instant::now();
        let mut cfg = InitConfig::phi(h);
        cfg.density_pct = 10;
        cfg.chain_count = 0;

        let mut rng = StdRng::seed_from_u64(42);
        let net = build_network(&cfg, &mut rng);

        // Check: feed each byte, read output pattern
        let input_end = cfg.phi_dim;
        let output_start = h - cfg.phi_dim;
        let output_size = h - output_start;

        println!("  H={}: {} neurons, input=0..{}, output={}..{} ({} output neurons), {} edges",
            h, h, input_end, output_start, h, output_size, net.edge_count());

        // Feed each of 27 bytes and collect output charge patterns
        let mut patterns: Vec<Vec<i8>> = Vec::new();
        for ch in 0..27u8 {
            // Create input spike pattern from byte bits
            let mut input = vec![0u8; h];
            for bit in 0..8 {
                if (ch >> bit) & 1 == 1 {
                    if bit < input_end { input[bit] = 1; }
                }
            }

            // Run propagation manually — charge accumulation
            // Simple version: just check which output neurons would get signal
            let mut charges = vec![0i32; h];
            // Set input charges
            for i in 0..8.min(input_end) {
                if (ch >> i) & 1 == 1 { charges[i] = 10; }
            }

            // Propagate through edges for a few ticks
            let edges = net.graph().edges();
            for _tick in 0..6 {
                let mut new_charges = charges.clone();
                for edge in &edges {
                    let s = edge.source as usize;
                    let t = edge.target as usize;
                    if charges[s] > 0 {
                        new_charges[t] += 1;
                    }
                }
                charges = new_charges;
            }

            // Read output: threshold at > 0
            let pattern: Vec<i8> = (output_start..h)
                .map(|i| if charges[i] > 0 { 1 } else { 0 })
                .collect();
            patterns.push(pattern);
        }

        // Count unique patterns
        let mut unique = patterns.clone();
        unique.sort();
        unique.dedup();
        let n_unique = unique.len();

        // Count correct round-trips (nearest hamming)
        let mut ok = 0;
        for i in 0..27 {
            let mut best = 0; let mut bd = u32::MAX;
            for j in 0..27 {
                let d: u32 = patterns[i].iter().zip(&patterns[j])
                    .map(|(&a,&b)| ((a-b).abs()) as u32).sum();
                if d < bd { bd = d; best = j; }
            }
            if best == i { ok += 1; }
        }

        let mark = if ok == 27 { " ★★★" } else if n_unique == 27 { " ★★" } else { "" };
        println!("    Unique patterns: {}/27, Round-trip: {}/27 ({:.2}s){}",
            n_unique, ok, tc.elapsed().as_secs_f64(), mark);

        if ok < 27 {
            // Show some collisions
            let mut collisions = 0;
            for i in 0..27 { for j in i+1..27 {
                if patterns[i] == patterns[j] {
                    let ci = if i==26{"space".to_string()} else {format!("'{}'", (i as u8+b'a') as char)};
                    let cj = if j==26{"space".to_string()} else {format!("'{}'", (j as u8+b'a') as char)};
                    if collisions < 3 { println!("    Collision: {} == {}", ci, cj); }
                    collisions += 1;
                }
            }}
            if collisions > 3 { println!("    ... and {} more collisions", collisions - 3); }
        }
        println!();
    }

    // Compare
    println!("━━━ COMPARISON ━━━\n");
    println!("  Exhaustive search (designed):  4N, 0.05s, 27/27 guaranteed");
    println!("  Random INSTNCT (emergent):     see above — random topology");
    println!("  Evolved INSTNCT (emergent):    NOT YET — needs evolution loop");
    println!("\n  Total: {:.2}s", t0.elapsed().as_secs_f64());
}
