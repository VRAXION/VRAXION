//! Quick: threshold × input strength → OUT spike count. Where's the sweet spot?
//!
//! Run: cargo run --example micro_thr_sweep --release

const TICKS: usize = 50;
const INPUT_TICKS: usize = 20;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

fn run(threshold: u8, channel: u8, input: i16, weight: i16) -> u32 {
    let mut charge: i16 = 0;
    let mut activation: i8 = 0;
    let mut refractory: u8 = 0;
    let mut spikes: u32 = 0;
    // Simple 2-neuron: IN → OUT(weighted)
    for tick in 0..TICKS {
        // IN always fires when input active
        let in_spike = if tick < INPUT_TICKS && input > 0 { 1i8 } else { 0 };
        // Incoming to OUT
        let incoming = in_spike as i16 * weight;
        if refractory > 0 { refractory -= 1; activation = 0; charge += incoming; continue; }
        charge += incoming;
        if tick % 6 == 5 && charge > 0 { charge -= 1; }
        if charge < 0 { charge = 0; }
        let pi = (tick as u8 + 9 - channel) & 7;
        let pm = PHASE_BASE[pi as usize];
        if charge * 10 >= (threshold as i16 + 1) * pm {
            activation = 1; charge = 0; refractory = 1; spikes += 1;
        } else { activation = 0; }
    }
    spikes
}

fn main() {
    println!("=== THRESHOLD × INPUT → SPIKE COUNT ===");
    println!("{} ticks, input for {} ticks, channel=1, weight=1\n", TICKS, INPUT_TICKS);

    print!("{:>4}", "thr");
    for inp in 1..=8i16 { print!(" {:>5}", format!("in={}", inp)); }
    println!("  | spread");
    println!("{:-<4} {:-<48}", "", "");

    for thr in 0..=8u8 {
        print!("{:>4}", thr);
        let mut counts = Vec::new();
        for inp in 1..=8i16 {
            let spk = run(thr, 1, inp, inp); // weight = input strength (stronger = more charge/tick)
            print!(" {:>5}", spk);
            counts.push(spk);
        }
        let min = *counts.iter().min().unwrap();
        let max = *counts.iter().max().unwrap();
        let distinct = {
            let mut c = counts.clone(); c.sort(); c.dedup(); c.len()
        };
        println!("  | {}-{} ({} distinct)", min, max, distinct);
    }

    println!("\n\nNow with weight=1 fixed (input ONLY varies charge injection rate):\n");
    print!("{:>4}", "thr");
    for inp in 1..=8i16 { print!(" {:>5}", format!("in={}", inp)); }
    println!("  | spread");
    println!("{:-<4} {:-<48}", "", "");

    for thr in 0..=8u8 {
        print!("{:>4}", thr);
        let mut counts = Vec::new();
        for inp in 1..=8i16 {
            let spk = run(thr, 1, inp, 1); // weight always 1, input varies charge
            print!(" {:>5}", spk);
            counts.push(spk);
        }
        let min = *counts.iter().min().unwrap();
        let max = *counts.iter().max().unwrap();
        let distinct = { let mut c = counts.clone(); c.sort(); c.dedup(); c.len() };
        println!("  | {}-{} ({} distinct)", min, max, distinct);
    }
}
