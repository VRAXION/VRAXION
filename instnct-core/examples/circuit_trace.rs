//! Circuit Trace: isolated component verification with hand-computed values.
//!
//! Each test constructs a minimal network, feeds known input, and compares
//! against values computed by hand (like tracing a circuit diagram).
//!
//! Run: cargo run --example circuit_trace --release

use instnct_core::{
    propagate_token, ConnectionGraph, Network, PropagationConfig, PropagationParameters,
    PropagationState, PropagationWorkspace, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---- Constants matching evolve_language.rs ----
const CHARS: usize = 27;
const NEURON_COUNT: usize = 256;
const PHI_DIM: usize = 158;
const INPUT_END: usize = PHI_DIM;
const SDR_ACTIVE_PCT: usize = 20;

fn run(name: &str, f: fn() -> bool, pass: &mut u32, fail: &mut u32) {
    if f() {
        println!("  [PASS] {name}");
        *pass += 1;
    } else {
        println!("  [FAIL] {name}");
        *fail += 1;
    }
}

fn main() {
    let mut pass = 0u32;
    let mut fail = 0u32;

    run("A1: SDR encoding", test_a1_sdr_encoding, &mut pass, &mut fail);
    run("A2: Single tick scatter", test_a2_single_tick_scatter, &mut pass, &mut fail);
    run("A3: Multi-tick chain delay", test_a3_multi_tick_chain, &mut pass, &mut fail);
    run("A4: Charge decay", test_a4_charge_decay, &mut pass, &mut fail);
    run("A5: Phase gating (8 channels)", test_a5_phase_gating, &mut pass, &mut fail);
    run("A6: Output prediction (predict_i8)", test_a6_output_prediction, &mut pass, &mut fail);
    run("A7: Snapshot/restore round-trip", test_a7_snapshot_restore, &mut pass, &mut fail);

    println!("\n=== CIRCUIT TRACE SUMMARY ===");
    println!("  PASS: {pass}  FAIL: {fail}  TOTAL: {}", pass + fail);
    if fail == 0 {
        println!("  All components verified.");
    } else {
        println!("  *** {fail} COMPONENT(S) FAILED ***");
    }
}

// ============================================================================
// A1: SDR Encoding
// ============================================================================
fn test_a1_sdr_encoding() -> bool {
    let mut rng = StdRng::seed_from_u64(42);
    let sdr = SdrTable::new(CHARS, NEURON_COUNT, INPUT_END, SDR_ACTIVE_PCT, &mut rng).unwrap();

    let expected_active = sdr.active_bits(); // 31

    for ci in 0..CHARS {
        let pattern = sdr.pattern(ci);
        // Check total length
        if pattern.len() != NEURON_COUNT {
            println!("    char {ci}: wrong length {} (expected {NEURON_COUNT})", pattern.len());
            return false;
        }
        // Count active bits in input zone [0, INPUT_END)
        let active_in_input: usize = pattern[..INPUT_END].iter().filter(|&&v| v == 1).count();
        if active_in_input != expected_active {
            println!("    char {ci}: {active_in_input} active in input zone (expected {expected_active})");
            return false;
        }
        // Check non-input zone [INPUT_END, NEURON_COUNT) is all zero
        let any_outside = pattern[INPUT_END..].iter().any(|&v| v != 0);
        if any_outside {
            println!("    char {ci}: non-zero values outside input zone [0,{INPUT_END})");
            return false;
        }
        // All values should be 0 or 1
        let invalid = pattern.iter().any(|&v| v != 0 && v != 1);
        if invalid {
            println!("    char {ci}: values other than 0/1 found");
            return false;
        }
    }
    true
}

// ============================================================================
// A2: Single Tick Scatter
// ============================================================================
fn test_a2_single_tick_scatter() -> bool {
    // 4 neurons, edges: 0->1, 0->2, 2->3
    // Input: [1,0,0,0] (injected on tick 0)
    // 1 tick, no decay, input_duration=1
    // All: threshold=0 (eff.1), channel=1, polarity=+1
    //
    // Tick 0 trace:
    //   Decay: skipped (interval=0)
    //   Input: activation += [1,0,0,0] -> [1,0,0,0]
    //   Scatter: inc[1]+=act[0]=1, inc[2]+=act[0]=1, inc[3]+=act[2]=0
    //            incoming = [0,1,1,0]
    //   Charge:  charge = [0+0, 0+1, 0+1, 0+0] = [0,1,1,0]
    //   Spike:   phase_base[(0+9-1)&7] = phase_base[0] = 7
    //            threshold_x10 = (0+1)*7 = 7
    //            neuron 0: charge=0, 0*10=0 >= 7? No -> act=0
    //            neuron 1: charge=1, 1*10=10 >= 7? Yes -> act=1, charge=0
    //            neuron 2: charge=1, 1*10=10 >= 7? Yes -> act=1, charge=0
    //            neuron 3: charge=0, 0*10=0 >= 7? No -> act=0
    //
    // Expected: activation=[0,1,1,0], charge=[0,0,0,0]

    let n = 4;
    let graph = ConnectionGraph::from_pairs(n, &[(0, 1), (0, 2), (2, 3)]);
    let threshold = vec![0u32; n];
    let channel = vec![1u8; n];
    let polarity = vec![1i32; n];
    let mut activation = vec![0i32; n];
    let mut charge = vec![0u32; n];
    let config = PropagationConfig {
        ticks_per_token: 1,
        input_duration_ticks: 1,
        decay_interval_ticks: 0,
        use_refractory: false,
    };
    let input = vec![1i32, 0, 0, 0];
    let params = PropagationParameters { threshold: &threshold, channel: &channel, polarity: &polarity };
    let mut refractory = vec![0u8; n];
    let mut state = PropagationState { activation: &mut activation, charge: &mut charge, refractory: &mut refractory };
    let mut ws = PropagationWorkspace::new(n);

    propagate_token(&input, &graph, &params, &mut state, &config, &mut ws).unwrap();

    let act_ok = activation == [0, 1, 1, 0];
    let chg_ok = charge == [0, 0, 0, 0];

    if !act_ok {
        println!("    activation: {activation:?} (expected [0,1,1,0])");
    }
    if !chg_ok {
        println!("    charge: {charge:?} (expected [0,0,0,0])");
    }
    act_ok && chg_ok
}

// ============================================================================
// A3: Multi-tick chain delay
// ============================================================================
fn test_a3_multi_tick_chain() -> bool {
    // 3 neurons, edges: 0->1, 1->2
    // threshold stored=1 (eff.2), channel=1, polarity=+1
    // 4 ticks, input_duration=2, decay_interval=0, input=[1,0,0]
    //
    // Tick 0:
    //   Input: act += [1,0,0] -> [1,0,0]
    //   Scatter: inc=[0, act[0]=1, act[1]=0] -> charge=[0,1,0]
    //   Spike: phase_base[(0+9-1)&7]=pb[0]=7. thr_x10=(1+1)*7=14
    //     n0: 0>=14? No  n1: 10>=14? No  n2: 0>=14? No
    //   -> act=[0,0,0], charge=[0,1,0]
    //
    // Tick 1:
    //   Input: act += [1,0,0] -> [1,0,0]
    //   Scatter: inc=[0,1,0] -> charge=[0,1+1=2,0]
    //   Spike: pb[(1+9-1)&7]=pb[1]=8. thr_x10=2*8=16
    //     n0: 0>=16? No  n1: 2*10=20>=16? YES -> fire  n2: 0>=16? No
    //   -> act=[0,1,0], charge=[0,0,0]
    //
    // Tick 2:
    //   Input: tick 2 >= input_duration 2 -> SKIP
    //   act still [0,1,0] from previous spike
    //   Scatter: inc=[0,0,act[1]=1] -> charge=[0,0,1]
    //   Spike: pb[(2+9-1)&7]=pb[2]=10. thr_x10=2*10=20
    //     n0: 0>=20? No  n1: 0>=20? No  n2: 1*10=10>=20? No
    //   -> act=[0,0,0], charge=[0,0,1]
    //
    // Tick 3:
    //   Input: SKIP
    //   act=[0,0,0]
    //   Scatter: inc=[0,0,0] -> charge=[0,0,1]
    //   Spike: pb[(3+9-1)&7]=pb[3]=12. thr_x10=2*12=24
    //     n2: 10>=24? No
    //   -> act=[0,0,0], charge=[0,0,1]

    let n = 3;
    let graph = ConnectionGraph::from_pairs(n, &[(0, 1), (1, 2)]);
    let threshold = vec![1u32; n]; // stored=1, effective=2
    let channel = vec![1u8; n];
    let polarity = vec![1i32; n];
    let mut activation = vec![0i32; n];
    let mut charge = vec![0u32; n];
    let config = PropagationConfig {
        ticks_per_token: 4,
        input_duration_ticks: 2,
        decay_interval_ticks: 0,
        use_refractory: false,
    };
    let input = vec![1i32, 0, 0];
    let params = PropagationParameters { threshold: &threshold, channel: &channel, polarity: &polarity };
    let mut refractory = vec![0u8; n];
    let mut state = PropagationState { activation: &mut activation, charge: &mut charge, refractory: &mut refractory };
    let mut ws = PropagationWorkspace::new(n);

    propagate_token(&input, &graph, &params, &mut state, &config, &mut ws).unwrap();

    let act_ok = activation == [0, 0, 0];
    let chg_ok = charge == [0, 0, 1];

    if !act_ok {
        println!("    activation: {activation:?} (expected [0,0,0])");
    }
    if !chg_ok {
        println!("    charge: {charge:?} (expected [0,0,1])");
    }
    act_ok && chg_ok
}

// ============================================================================
// A4: Charge decay
// ============================================================================
fn test_a4_charge_decay() -> bool {
    // 2 neurons, no edges
    // Pre-set charge: [5, 0], threshold=15 (eff.16, never fires)
    // 13 ticks, decay_interval=6 -> decay at tick 0, 6, 12
    // No input (duration=0), no edges
    //
    // Tick 0: charge=[5-1, 0] = [4,0]
    // Tick 6: charge=[4-1, 0] = [3,0]
    // Tick 12: charge=[3-1, 0] = [2,0]
    // (no spikes: highest charge=5, 5*10=50 < 16*7=112)
    //
    // Expected: charge=[2, 0]

    let n = 2;
    let graph = ConnectionGraph::new(n);
    let threshold = vec![15u32; n]; // stored=15, effective=16
    let channel = vec![1u8; n];
    let polarity = vec![1i32; n];
    let mut activation = vec![0i32; n];
    let mut charge = vec![5u32, 0]; // PRE-SET charge
    let config = PropagationConfig {
        ticks_per_token: 13,
        input_duration_ticks: 0,
        decay_interval_ticks: 6,
        use_refractory: false,
    };
    let input = vec![0i32; n];
    let params = PropagationParameters { threshold: &threshold, channel: &channel, polarity: &polarity };
    let mut refractory = vec![0u8; n];
    let mut state = PropagationState { activation: &mut activation, charge: &mut charge, refractory: &mut refractory };
    let mut ws = PropagationWorkspace::new(n);

    propagate_token(&input, &graph, &params, &mut state, &config, &mut ws).unwrap();

    let chg_ok = charge == [2, 0];
    if !chg_ok {
        println!("    charge: {charge:?} (expected [2, 0])");
    }
    chg_ok
}

// ============================================================================
// A5: Phase gating (8 channels)
// ============================================================================
fn test_a5_phase_gating() -> bool {
    // 8 neurons, no edges, charge=4, threshold stored=4 (eff.5)
    // Each neuron i has channel = i+1 (channels 1-8)
    // 1 tick (tick 0), no decay, no input
    //
    // Spike check: charge_x10=4*10=40, threshold_x10 = (4+1) * phase_base[(0+9-ch)&7]
    //   ch=1: pb[(0+9-1)&7]=pb[0]=7  -> thr=5*7=35.  40>=35? YES
    //   ch=2: pb[(0+9-2)&7]=pb[7]=8  -> thr=5*8=40.  40>=40? YES
    //   ch=3: pb[(0+9-3)&7]=pb[6]=10 -> thr=5*10=50. 40>=50? No
    //   ch=4: pb[(0+9-4)&7]=pb[5]=12 -> thr=5*12=60. No
    //   ch=5: pb[(0+9-5)&7]=pb[4]=13 -> thr=5*13=65. No
    //   ch=6: pb[(0+9-6)&7]=pb[3]=12 -> thr=5*12=60. No
    //   ch=7: pb[(0+9-7)&7]=pb[2]=10 -> thr=5*10=50. No
    //   ch=8: pb[(0+9-8)&7]=pb[1]=8  -> thr=5*8=40.  40>=40? YES
    //
    // Expected: activation=[1,1,0,0,0,0,0,1], charge=[0,0,4,4,4,4,4,0]

    let n = 8;
    let graph = ConnectionGraph::new(n);
    let threshold = vec![4u32; n];
    let channel: Vec<u8> = (1..=8).collect();
    let polarity = vec![1i32; n];
    let mut activation = vec![0i32; n];
    let mut charge = vec![4u32; n]; // PRE-SET
    let config = PropagationConfig {
        ticks_per_token: 1,
        input_duration_ticks: 0,
        decay_interval_ticks: 0,
        use_refractory: false,
    };
    let input = vec![0i32; n];
    let params = PropagationParameters { threshold: &threshold, channel: &channel, polarity: &polarity };
    let mut refractory = vec![0u8; n];
    let mut state = PropagationState { activation: &mut activation, charge: &mut charge, refractory: &mut refractory };
    let mut ws = PropagationWorkspace::new(n);

    propagate_token(&input, &graph, &params, &mut state, &config, &mut ws).unwrap();

    let expected_act = [1, 1, 0, 0, 0, 0, 0, 1];
    let expected_chg = [0u32, 0, 4, 4, 4, 4, 4, 0];

    let act_ok = activation == expected_act;
    let chg_ok = charge == expected_chg;
    if !act_ok {
        println!("    activation: {activation:?} (expected {expected_act:?})");
    }
    if !chg_ok {
        println!("    charge: {charge:?} (expected {expected_chg:?})");
    }
    act_ok && chg_ok
}

// ============================================================================
// A6: Output prediction (predict_i8)
// ============================================================================
fn predict_i8(charge_slice: &[u32], w: &[i8], num_classes: usize) -> u8 {
    let mut scores = vec![0i32; num_classes];
    for (i, &c) in charge_slice.iter().enumerate() {
        if c == 0 { continue; }
        let x = c as i32;
        let row = &w[i * num_classes..(i + 1) * num_classes];
        for (s, &wt) in scores.iter_mut().zip(row.iter()) {
            *s += x * wt as i32;
        }
    }
    scores.iter().enumerate().max_by_key(|&(_, &s)| s)
        .map(|(i, _)| i as u8).unwrap_or(0)
}

fn test_a6_output_prediction() -> bool {
    // Small example: 5 output neurons, 3 classes
    // charge = [0, 2, 0, 3, 0]  (neurons 1 and 3 active)
    // W (5 x 3):
    //   row 0: [ 10, -5,  3]  (charge=0, ignored)
    //   row 1: [  5,  2, -1]  (charge=2)
    //   row 2: [ -3,  7,  1]  (charge=0, ignored)
    //   row 3: [  1, -2,  4]  (charge=3)
    //   row 4: [  0,  0,  0]  (charge=0, ignored)
    //
    // scores[0] = 2*5 + 3*1 = 10 + 3 = 13
    // scores[1] = 2*2 + 3*(-2) = 4 - 6 = -2
    // scores[2] = 2*(-1) + 3*4 = -2 + 12 = 10
    //
    // argmax = class 0 (score 13)

    let charge: Vec<u32> = vec![0, 2, 0, 3, 0];
    let w: Vec<i8> = vec![
        10, -5,  3,  // row 0
         5,  2, -1,  // row 1
        -3,  7,  1,  // row 2
         1, -2,  4,  // row 3
         0,  0,  0,  // row 4
    ];

    let pred = predict_i8(&charge, &w, 3);
    let ok = pred == 0;
    if !ok {
        println!("    predicted class {pred} (expected 0)");
        // Print scores for debugging
        let mut scores = [0i32; 3];
        for (i, &c) in charge.iter().enumerate() {
            if c == 0 { continue; }
            let x = c as i32;
            for j in 0..3 {
                scores[j] += x * w[i * 3 + j] as i32;
            }
        }
        println!("    scores: {scores:?}");
    }
    ok
}

// ============================================================================
// A7: Snapshot/restore round-trip
// ============================================================================
fn test_a7_snapshot_restore() -> bool {
    let mut net = Network::new(8);
    let mut rng = StdRng::seed_from_u64(42);

    // Build some structure
    for _ in 0..10 {
        net.mutate_add_edge(&mut rng);
    }
    for i in 0..8 {
        net.threshold_mut()[i] = rng.gen_range(0..=7);
        net.channel_mut()[i] = rng.gen_range(1..=8);
        if rng.gen_ratio(1, 3) {
            net.polarity_mut()[i] = -1;
        }
    }

    // Propagate to create non-zero state
    let config = PropagationConfig {
        ticks_per_token: 6,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
        use_refractory: false,
    };
    let input: Vec<i32> = (0..8).map(|i| if i < 3 { 1 } else { 0 }).collect();
    net.propagate(&input, &config).unwrap();

    // Snapshot
    let snapshot = net.save_state();
    let saved_edges = net.edge_count();
    let saved_charge: Vec<u32> = net.charge().to_vec();
    let saved_activation: Vec<i32> = net.activation().to_vec();
    let saved_threshold: Vec<u32> = net.threshold().to_vec();

    // Mutate heavily
    for _ in 0..20 {
        net.mutate_add_edge(&mut rng);
    }
    net.threshold_mut()[0] = 15;
    net.channel_mut()[0] = 8;
    net.polarity_mut()[0] = -1;
    // Propagate again to change state
    let input2: Vec<i32> = (0..8).map(|i| if i >= 5 { 1 } else { 0 }).collect();
    net.propagate(&input2, &config).unwrap();

    // Verify state HAS changed
    let changed = net.edge_count() != saved_edges
        || net.charge() != saved_charge.as_slice()
        || net.threshold()[0] != saved_threshold[0];
    if !changed {
        println!("    state did not change after mutations (test is invalid)");
        return false;
    }

    // Restore
    net.restore_state(&snapshot);

    // Verify all state matches snapshot
    let edges_ok = net.edge_count() == saved_edges;
    let charge_ok = net.charge() == saved_charge.as_slice();
    let act_ok = net.activation() == saved_activation.as_slice();
    let thr_ok = net.threshold() == saved_threshold.as_slice();

    if !edges_ok {
        println!("    edges: {} (expected {})", net.edge_count(), saved_edges);
    }
    if !charge_ok {
        println!("    charge mismatch after restore");
    }
    if !act_ok {
        println!("    activation mismatch after restore");
    }
    if !thr_ok {
        println!("    threshold mismatch after restore");
    }

    edges_ok && charge_ok && act_ok && thr_ok
}
