//! A/B benchmark: individual bitmask/nibble arrays vs flat u8/i8 arrays.
//!
//! Instead of mixing fields, keep each field in its own compact form:
//!   - polarity: 1 bit per neuron in Vec<u64> bitmask (8x smaller, read-only)
//!   - threshold: nibble array, 2 neurons per byte (2x smaller, read-only)
//!   - charge: stays u8 (written every tick, nibble RMW too expensive)
//!   - channel: stays u8 (3 bits but awkward to pack)
//!   - activation: stays i8 (scatter-add hot path, don't touch)

mod common;

use common::{build_graph, print_harness_header, timed_run};
use std::hint::black_box;

const TICKS: usize = 12;
const MAX_CHARGE: u8 = 15;
const PHASE_BASE: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

// ---------------------------------------------------------------------------
// Bitmask helpers
// ---------------------------------------------------------------------------

struct PolarityBitmask {
    bits: Vec<u64>, // bit=1 means inhibitory (-1), bit=0 means excitatory (+1)
}

impl PolarityBitmask {
    fn new(n: usize) -> Self {
        Self {
            bits: vec![0u64; (n + 63) / 64],
        }
    }

    fn from_i8_slice(polarities: &[i8]) -> Self {
        let mut bm = Self::new(polarities.len());
        for (i, &p) in polarities.iter().enumerate() {
            if p < 0 {
                bm.bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        bm
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> i8 {
        if (self.bits[idx / 64] >> (idx % 64)) & 1 != 0 {
            -1
        } else {
            1
        }
    }

    fn memory_bytes(&self) -> usize {
        self.bits.len() * 8
    }
}

// ---------------------------------------------------------------------------
// Nibble array for threshold (4 bits per value, 2 values per byte)
// ---------------------------------------------------------------------------

struct NibbleArray {
    data: Vec<u8>,
}

impl NibbleArray {
    fn new(n: usize) -> Self {
        Self {
            data: vec![0u8; (n + 1) / 2],
        }
    }

    fn from_u8_slice(values: &[u8]) -> Self {
        let mut na = Self::new(values.len());
        for (i, &v) in values.iter().enumerate() {
            let byte_idx = i / 2;
            if i % 2 == 0 {
                na.data[byte_idx] = (na.data[byte_idx] & 0xF0) | (v & 0x0F);
            } else {
                na.data[byte_idx] = (na.data[byte_idx] & 0x0F) | (v << 4);
            }
        }
        na
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> u8 {
        let byte = self.data[idx / 2];
        if idx % 2 == 0 {
            byte & 0x0F
        } else {
            byte >> 4
        }
    }

    fn memory_bytes(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

struct BaselineFixture {
    sources: Vec<u16>,
    targets: Vec<u16>,
    activation: Vec<i8>,
    charge: Vec<u8>,
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    incoming: Vec<i16>,
    input: Vec<i32>,
    n: usize,
}

struct CompactFixture {
    sources: Vec<u16>,
    targets: Vec<u16>,
    activation: Vec<i8>,
    charge: Vec<u8>,
    threshold_nibbles: NibbleArray, // 4 bits each, read-only in hot path
    channel: Vec<u8>,              // stays u8
    polarity_bits: PolarityBitmask, // 1 bit each, read-only, sparse access
    incoming: Vec<i16>,
    input: Vec<i32>,
    n: usize,
}

fn build(neuron_count: usize, edge_prob_pct: u64) -> (BaselineFixture, CompactFixture) {
    let graph = build_graph(neuron_count, edge_prob_pct);
    let (src, tgt) = graph.edge_endpoints_pub();
    let mut input = vec![0i32; neuron_count];
    if neuron_count > 0 {
        input[0] = 1;
    }

    let threshold = vec![6u8; neuron_count];
    let polarity = vec![1i8; neuron_count];

    let baseline = BaselineFixture {
        sources: src.to_vec(),
        targets: tgt.to_vec(),
        activation: vec![0; neuron_count],
        charge: vec![0; neuron_count],
        threshold: threshold.clone(),
        channel: vec![1u8; neuron_count],
        polarity: polarity.clone(),
        incoming: vec![0; neuron_count],
        input: input.clone(),
        n: neuron_count,
    };

    let compact = CompactFixture {
        sources: src.to_vec(),
        targets: tgt.to_vec(),
        activation: vec![0; neuron_count],
        charge: vec![0; neuron_count],
        threshold_nibbles: NibbleArray::from_u8_slice(&threshold),
        channel: vec![1u8; neuron_count],
        polarity_bits: PolarityBitmask::from_i8_slice(&polarity),
        incoming: vec![0; neuron_count],
        input: input.clone(),
        n: neuron_count,
    };

    (baseline, compact)
}

// ---------------------------------------------------------------------------
// Propagation: baseline (current library)
// ---------------------------------------------------------------------------

fn propagate_baseline(f: &mut BaselineFixture) {
    let n = f.n;
    let edge_src = &f.sources;
    let edge_tgt = &f.targets;

    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for ch in f.charge.iter_mut() {
                *ch = ch.saturating_sub(1);
            }
        }
        if tick < 2 {
            for (act, &inp) in f.activation.iter_mut().zip(f.input.iter()) {
                *act = act.saturating_add(inp as i8);
            }
        }

        let incoming = &mut f.incoming[..n];
        incoming.fill(0);
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            incoming[tc[0] as usize] += f.activation[sc[0] as usize] as i16;
            incoming[tc[1] as usize] += f.activation[sc[1] as usize] as i16;
            incoming[tc[2] as usize] += f.activation[sc[2] as usize] as i16;
            incoming[tc[3] as usize] += f.activation[sc[3] as usize] as i16;
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            incoming[edge_tgt[i] as usize] += f.activation[edge_src[i] as usize] as i16;
        }

        for (ch, &sig) in f.charge[..n].iter_mut().zip(incoming.iter()) {
            let val = (*ch as i16) + sig;
            *ch = val.clamp(0, MAX_CHARGE as i16) as u8;
        }

        let phase_tick = tick % 8;
        for idx in 0..n {
            let ch_idx = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ch_idx) {
                PHASE_BASE[(phase_tick + 9 - ch_idx) & 7] as u16
            } else {
                10
            };
            let charge_x10 = f.charge[idx] as u16 * 10;
            let thresh_x10 = (f.threshold[idx] as u16 + 1) * pm;
            if charge_x10 >= thresh_x10 {
                f.activation[idx] = f.polarity[idx];
                f.charge[idx] = 0;
            } else {
                f.activation[idx] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Propagation: compact (bitmask polarity + nibble threshold)
// ---------------------------------------------------------------------------

fn propagate_compact(f: &mut CompactFixture) {
    let n = f.n;
    let edge_src = &f.sources;
    let edge_tgt = &f.targets;

    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for ch in f.charge.iter_mut() {
                *ch = ch.saturating_sub(1);
            }
        }
        if tick < 2 {
            for (act, &inp) in f.activation.iter_mut().zip(f.input.iter()) {
                *act = act.saturating_add(inp as i8);
            }
        }

        // Scatter-add: identical (activation is separate i8 array)
        let incoming = &mut f.incoming[..n];
        incoming.fill(0);
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            incoming[tc[0] as usize] += f.activation[sc[0] as usize] as i16;
            incoming[tc[1] as usize] += f.activation[sc[1] as usize] as i16;
            incoming[tc[2] as usize] += f.activation[sc[2] as usize] as i16;
            incoming[tc[3] as usize] += f.activation[sc[3] as usize] as i16;
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            incoming[edge_tgt[i] as usize] += f.activation[edge_src[i] as usize] as i16;
        }

        // Charge accumulation: same (charge stays u8)
        for (ch, &sig) in f.charge[..n].iter_mut().zip(incoming.iter()) {
            let val = (*ch as i16) + sig;
            *ch = val.clamp(0, MAX_CHARGE as i16) as u8;
        }

        // Spike: threshold from nibble array, polarity from bitmask
        let phase_tick = tick % 8;
        for idx in 0..n {
            let ch_idx = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ch_idx) {
                PHASE_BASE[(phase_tick + 9 - ch_idx) & 7] as u16
            } else {
                10
            };
            let charge_x10 = f.charge[idx] as u16 * 10;
            let threshold = f.threshold_nibbles.get(idx);
            let thresh_x10 = (threshold as u16 + 1) * pm;
            if charge_x10 >= thresh_x10 {
                f.activation[idx] = f.polarity_bits.get(idx);
                f.charge[idx] = 0;
            } else {
                f.activation[idx] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------

struct ABCase {
    name: &'static str,
    neuron_count: usize,
    edge_prob_pct: u64,
    iterations: usize,
}

fn run_ab(case: &ABCase) {
    println!(
        "\n=== {} | H={}, {}% density ===",
        case.name, case.neuron_count, case.edge_prob_pct
    );

    let (mut baseline, mut compact) = build(case.neuron_count, case.edge_prob_pct);

    let base_spike_bytes = case.neuron_count * 2; // threshold(u8) + polarity(i8)
    let compact_spike_bytes =
        compact.threshold_nibbles.memory_bytes() + compact.polarity_bits.memory_bytes();
    println!(
        "  edges: {} | threshold+polarity: baseline={}B compact={}B ({:.1}x smaller)",
        baseline.sources.len(),
        base_spike_bytes,
        compact_spike_bytes,
        base_spike_bytes as f64 / compact_spike_bytes.max(1) as f64
    );

    let ctrl_a = timed_run("CTRL-A (baseline)", case.iterations, || {
        baseline.activation.fill(0);
        baseline.charge.fill(0);
        propagate_baseline(black_box(&mut baseline));
    });
    let ctrl_b = timed_run("CTRL-B (baseline)", case.iterations, || {
        baseline.activation.fill(0);
        baseline.charge.fill(0);
        propagate_baseline(black_box(&mut baseline));
    });
    let noise_pct = ((ctrl_b.median_ns - ctrl_a.median_ns) / ctrl_a.median_ns * 100.0).abs();
    println!(
        "  NOISE: {noise_pct:.1}% ({})",
        if noise_pct <= 5.0 { "stable" } else { "noisy" }
    );

    let t_compact = timed_run("bitmask pol + nibble thresh", case.iterations, || {
        compact.activation.fill(0);
        compact.charge.fill(0);
        propagate_compact(black_box(&mut compact));
    });

    let base_ns = ctrl_a.median_ns;
    let delta = (t_compact.median_ns - base_ns) / base_ns * 100.0;

    println!("\n  RESULTS:");
    println!("    baseline (u8+i8):    {:>10.0} ns", base_ns);
    println!(
        "    bitmask+nibble:      {:>10.0} ns  ({:+.1}%)",
        t_compact.median_ns, delta
    );

    if delta < -noise_pct {
        println!("    VERDICT: compact wins by {:.1}%", delta.abs());
    } else if delta.abs() < noise_pct {
        println!("    VERDICT: within noise");
    } else {
        println!("    VERDICT: overhead > savings");
    }
}

fn main() {
    print_harness_header();

    let cases = [
        ABCase {
            name: "small",
            neuron_count: 256,
            edge_prob_pct: 5,
            iterations: 5_000,
        },
        ABCase {
            name: "medium",
            neuron_count: 1024,
            edge_prob_pct: 3,
            iterations: 2_000,
        },
        ABCase {
            name: "large",
            neuron_count: 4096,
            edge_prob_pct: 1,
            iterations: 500,
        },
        ABCase {
            name: "xlarge",
            neuron_count: 16384,
            edge_prob_pct: 0,
            iterations: 100,
        },
    ];

    for case in &cases {
        run_ab(case);
    }
}
