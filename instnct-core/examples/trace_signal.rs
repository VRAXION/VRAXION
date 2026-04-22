//! Trace signal propagation tick-by-tick through a checkpoint.
//!
//! Feeds one byte-pair, then dumps per-tick state:
//! - Which neurons have charge > 0
//! - Which neurons fired
//! - Charge distribution per zone (input/overlap/output)
//! - Signal "alive" count per tick
//!
//! Usage: cargo run --release --example trace_signal -- <checkpoint.bin> <packed.bin>

use instnct_core::{load_checkpoint, InitConfig, Network, PropagationConfig, VcbpTable};
use std::env;
use std::path::Path;

const MAX_CHARGE: i32 = 7;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("Usage: trace_signal <checkpoint.bin> <packed.bin> [pair_hex]");
        eprintln!("  pair_hex: e.g. 7468 for 'th' (default: tries several common pairs)");
        std::process::exit(1);
    }

    let (mut net, _proj, meta) = load_checkpoint(&args[0]).expect("load checkpoint");
    let table = VcbpTable::from_packed(Path::new(&args[1])).unwrap();
    let h = net.neuron_count();
    let init = InitConfig::phi(h);
    let e = table.e;

    println!("Checkpoint: step={}, acc={:.2}%, H={h}", meta.step, meta.accuracy * 100.0);
    println!("Propagation: ticks={}, input_duration={}, decay={}",
        init.propagation.ticks_per_token,
        init.propagation.input_duration_ticks,
        init.propagation.decay_interval_ticks);

    // Test pairs
    let test_pairs: Vec<(u16, &str)> = if args.len() > 2 {
        let hex = u16::from_str_radix(&args[2], 16).expect("bad hex");
        vec![(hex, "custom")]
    } else {
        vec![
            (VcbpTable::pair_id(b't', b'h'), "th"),
            (VcbpTable::pair_id(b'e', b' '), "e_"),
            (VcbpTable::pair_id(b' ', b't'), "_t"),
            (VcbpTable::pair_id(b'a', b'l'), "al"),
            (VcbpTable::pair_id(b'\n', b' '), "\\n_"),
        ]
    };

    for (pair_id, label) in &test_pairs {
        println!("\n{}", "=".repeat(70));
        println!("  Input: '{}' (pair_id=0x{:04X}, hot={})", label, pair_id, table.is_hot(*pair_id));
        println!("{}", "=".repeat(70));
        trace_one_token(&mut net, &table, &init, *pair_id, h, e);
    }

    // Also trace what happens with NO input (reset + empty propagation)
    println!("\n{}", "=".repeat(70));
    println!("  Input: NONE (zero vector — tests spontaneous activity)");
    println!("{}", "=".repeat(70));
    trace_one_token(&mut net, &table, &init, u16::MAX, h, e); // MAX = signal for zero input
}

fn trace_one_token(net: &mut Network, table: &VcbpTable, init: &InitConfig, pair_id: u16, h: usize, e: usize) {
    let input_end = init.input_end();
    let output_start = init.output_start();

    // Build input vector
    let mut input = vec![0i32; h];
    if pair_id != u16::MAX {
        let emb = table.embed_id(pair_id);
        table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
    }

    // Show input charges
    let nonzero_input: Vec<(usize, i32)> = input.iter().enumerate()
        .filter(|(_, &v)| v > 0).map(|(i, &v)| (i, v)).collect();
    println!("\n  Input charges ({} non-zero of {e}):", nonzero_input.len());
    print!("    ");
    for (i, v) in &nonzero_input {
        print!("n{}={} ", i, v);
    }
    println!();

    // Reset network state
    net.reset();

    // We can't step tick-by-tick with the public API (propagate does all ticks).
    // Instead, propagate and then read the FINAL state.
    // BUT we CAN propagate multiple times to see charge accumulation.

    // Strategy: propagate the token, then read charges.
    // Then propagate an EMPTY token to see if signal persists.
    net.propagate(&input, &init.propagation).unwrap();

    // Read charge state after propagation
    let charges_after: Vec<u8> = net.spike_data().iter().map(|s| s.charge).collect();

    // Zone stats
    let input_zone_charges: Vec<u8> = charges_after[..input_end].to_vec();
    let overlap_charges: Vec<u8> = charges_after[output_start..input_end].to_vec();
    let output_zone_charges: Vec<u8> = charges_after[output_start..].to_vec();

    let alive = |charges: &[u8]| -> usize { charges.iter().filter(|&&c| c > 0).count() };
    let sum = |charges: &[u8]| -> u32 { charges.iter().map(|&c| c as u32).sum() };
    let max = |charges: &[u8]| -> u8 { *charges.iter().max().unwrap_or(&0) };

    println!("\n  After 1 token ({} ticks):", init.propagation.ticks_per_token);
    println!("    Zone          Alive/Total  Sum    Max  Avg");
    println!("    {:<14} {:>3}/{}     {:>4}   {:>3}  {:.1}",
        "Input-only", alive(&charges_after[..output_start]), output_start,
        sum(&charges_after[..output_start]), max(&charges_after[..output_start]),
        sum(&charges_after[..output_start]) as f64 / output_start as f64);
    println!("    {:<14} {:>3}/{}      {:>4}   {:>3}  {:.1}",
        "Overlap", alive(&overlap_charges), overlap_charges.len(),
        sum(&overlap_charges), max(&overlap_charges),
        sum(&overlap_charges) as f64 / overlap_charges.len().max(1) as f64);
    println!("    {:<14} {:>3}/{}     {:>4}   {:>3}  {:.1}",
        "Output-only", alive(&charges_after[input_end..]), h - input_end,
        sum(&charges_after[input_end..]), max(&charges_after[input_end..]),
        sum(&charges_after[input_end..]) as f64 / (h - input_end) as f64);
    println!("    {:<14} {:>3}/{}     {:>4}   {:>3}  {:.1}",
        "TOTAL", alive(&charges_after), h,
        sum(&charges_after), max(&charges_after),
        sum(&charges_after) as f64 / h as f64);

    // Top-10 highest charge neurons
    let mut by_charge: Vec<(usize, u8)> = charges_after.iter().enumerate().map(|(i, &c)| (i, c)).collect();
    by_charge.sort_by(|a, b| b.1.cmp(&a.1));
    println!("\n    Top-10 charged neurons:");
    println!("    {:>4} {:>6} {:>6} {:>4} {:>4} {:>4}",
        "ID", "Charge", "Zone", "Thr", "Ch", "Pol");
    for &(id, charge) in by_charge.iter().take(10) {
        if charge == 0 { break; }
        let zone = if id < output_start { "input" }
            else if id < input_end { "overlap" }
            else { "output" };
        let sd = &net.spike_data()[id];
        let pol = if net.polarity()[id] > 0 { "+" } else { "-" };
        println!("    {:>4} {:>6} {:>6} {:>4} {:>4} {:>4}",
            id, charge, zone, sd.threshold, sd.channel, pol);
    }

    // Now propagate 3 MORE empty tokens to see signal persistence
    println!("\n    Signal persistence (empty tokens after input):");
    for extra in 1..=5 {
        let empty = vec![0i32; h];
        net.propagate(&empty, &init.propagation).unwrap();
        let ch: Vec<u8> = net.spike_data().iter().map(|s| s.charge).collect();
        let total_alive = alive(&ch);
        let total_sum = sum(&ch);
        let out_alive = alive(&ch[output_start..]);
        let out_sum = sum(&ch[output_start..]);
        println!("    +{extra} empty tokens: alive={total_alive}/{h}, sum={total_sum}, output_alive={out_alive}/{}, out_sum={out_sum}",
            h - output_start);
    }
}
