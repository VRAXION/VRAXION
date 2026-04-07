//! A/B benchmark: scatter-add with vs without software prefetch.
//!
//! The propagation hot loop does `incoming[target] += activation as i16; //[source]`
//! over a sparse edge list. When the network is large enough that neuron
//! arrays spill out of L1/L2, prefetching the next chunk's activation
//! values can hide DRAM latency.
//!
//! This bench inlines the full propagation loop twice: once baseline (identical
//! to the library), once with prefetch intrinsics on the scatter-add.

mod common;

use common::{build_graph, print_harness_header, timed_run};
use instnct_core::ConnectionGraph;
use std::hint::black_box;

// Mirror the library constants (propagation uses these internally).
const PHASE_BASE: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const MAX_CHARGE: u32 = 15;

struct BenchConfig {
    ticks_per_token: usize,
    input_duration_ticks: usize,
    decay_interval_ticks: usize,
}

struct Fixture {
    graph: ConnectionGraph,
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    input: Vec<i32>,
    neuron_count: usize,
}

struct Scratch {
    activation: Vec<i8>,
    charge: Vec<u8>,
    incoming: Vec<i16>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Self {
            activation: vec![0; n],
            charge: vec![0; n],
            incoming: vec![0; n],
        }
    }
    fn reset(&mut self) {
        self.activation.fill(0);
        self.charge.fill(0);
    }
}

fn build_fixture(neuron_count: usize, edge_prob_pct: u64) -> Fixture {
    let graph = build_graph(neuron_count, edge_prob_pct);
    let mut input = vec![0i32; neuron_count];
    if let Some(first) = input.first_mut() {
        *first = 1;
    }
    Fixture {
        graph,
        threshold: vec![6u8; neuron_count],
        channel: vec![1u8; neuron_count],
        polarity: vec![1i8; neuron_count],
        input,
        neuron_count,
    }
}

// ---------------------------------------------------------------------------
// Baseline: identical scatter-add to the library (chunks_exact(4), no prefetch)
// ---------------------------------------------------------------------------

fn propagate_baseline(f: &Fixture, s: &mut Scratch, cfg: &BenchConfig) {
    let n = f.neuron_count;
    let (edge_src, edge_tgt) = f.graph.edge_endpoints_pub();

    for tick in 0..cfg.ticks_per_token {
        if cfg.decay_interval_ticks > 0 && tick % cfg.decay_interval_ticks == 0 {
            for ch in s.charge.iter_mut() {
                *ch = ch.saturating_sub(1);
            }
        }
        if tick < cfg.input_duration_ticks {
            for (act, &inp) in s.activation.iter_mut().zip(f.input.iter()) {
                *act = act.saturating_add(inp as i8);
            }
        }

        let incoming = &mut s.incoming[..n];
        incoming.fill(0i16);
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            incoming[tc[0] as usize] += s.activation[sc[0] as usize] as i16;
            incoming[tc[1] as usize] += s.activation[sc[1] as usize] as i16;
            incoming[tc[2] as usize] += s.activation[sc[2] as usize] as i16;
            incoming[tc[3] as usize] += s.activation[sc[3] as usize] as i16;
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            incoming[edge_tgt[i] as usize] += s.activation[edge_src[i] as usize] as i16;
        }

        for (ch, &sig) in s.charge.iter_mut().zip(incoming.iter()) {
            *ch = { let val = (*ch as i16) + sig; val.clamp(0, MAX_CHARGE as i16) as u8 };
        }

        let phase_tick = tick % 8;
        for idx in 0..n {
            let ch_idx = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ch_idx) {
                PHASE_BASE[(phase_tick + 9 - ch_idx) & 7] as u16
            } else {
                10
            };
            let charge_x10 = s.charge[idx] as u16 * 10;
            let thresh_x10 = (f.threshold[idx] as u16 + 1) * pm;
            if charge_x10 >= thresh_x10 {
                s.activation[idx] = f.polarity[idx];
                s.charge[idx] = 0;
            } else {
                s.activation[idx] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Prefetch variant: prefetch activation[source] 1 chunk ahead
// ---------------------------------------------------------------------------

/// Software prefetch for read (T0 = all cache levels).
#[inline(always)]
unsafe fn prefetch_read(ptr: *const i8) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::_mm_prefetch;
        use std::arch::x86_64::_MM_HINT_T0;
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // ARM prefetch: PRFM PLDL1KEEP
        std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr; // no prefetch available
    }
}

fn propagate_prefetch(f: &Fixture, s: &mut Scratch, cfg: &BenchConfig) {
    let n = f.neuron_count;
    let (edge_src, edge_tgt) = f.graph.edge_endpoints_pub();

    for tick in 0..cfg.ticks_per_token {
        if cfg.decay_interval_ticks > 0 && tick % cfg.decay_interval_ticks == 0 {
            for ch in s.charge.iter_mut() {
                *ch = ch.saturating_sub(1);
            }
        }
        if tick < cfg.input_duration_ticks {
            for (act, &inp) in s.activation.iter_mut().zip(f.input.iter()) {
                *act = act.saturating_add(inp as i8);
            }
        }

        let incoming = &mut s.incoming[..n];
        incoming.fill(0i16);

        // Prefetch variant: use peekable chunks_exact to prefetch 1 chunk ahead
        // while keeping LLVM bounds-check elision from chunks_exact.
        let mut src_iter = edge_src.chunks_exact(4).peekable();
        let mut tgt_iter = edge_tgt.chunks_exact(4);
        let act_ptr = s.activation.as_ptr();

        while let (Some(sc), Some(tc)) = (src_iter.next(), tgt_iter.next()) {
            // Prefetch next chunk's activation reads
            if let Some(&next_sc) = src_iter.peek() {
                unsafe {
                    prefetch_read(act_ptr.add(next_sc[0] as usize));
                    prefetch_read(act_ptr.add(next_sc[1] as usize));
                    prefetch_read(act_ptr.add(next_sc[2] as usize));
                    prefetch_read(act_ptr.add(next_sc[3] as usize));
                }
            }

            incoming[tc[0] as usize] += s.activation[sc[0] as usize] as i16;
            incoming[tc[1] as usize] += s.activation[sc[1] as usize] as i16;
            incoming[tc[2] as usize] += s.activation[sc[2] as usize] as i16;
            incoming[tc[3] as usize] += s.activation[sc[3] as usize] as i16;
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            incoming[edge_tgt[i] as usize] += s.activation[edge_src[i] as usize] as i16;
        }

        for (ch, &sig) in s.charge.iter_mut().zip(incoming.iter()) {
            *ch = { let val = (*ch as i16) + sig; val.clamp(0, MAX_CHARGE as i16) as u8 };
        }

        let phase_tick = tick % 8;
        for idx in 0..n {
            let ch_idx = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ch_idx) {
                PHASE_BASE[(phase_tick + 9 - ch_idx) & 7] as u16
            } else {
                10
            };
            let charge_x10 = s.charge[idx] as u16 * 10;
            let thresh_x10 = (f.threshold[idx] as u16 + 1) * pm;
            if charge_x10 >= thresh_x10 {
                s.activation[idx] = f.polarity[idx];
                s.charge[idx] = 0;
            } else {
                s.activation[idx] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Prefetch variant 2: prefetch 2 chunks ahead (more pipeline depth)
// ---------------------------------------------------------------------------

fn propagate_prefetch_2ahead(f: &Fixture, s: &mut Scratch, cfg: &BenchConfig) {
    let n = f.neuron_count;
    let (edge_src, edge_tgt) = f.graph.edge_endpoints_pub();

    for tick in 0..cfg.ticks_per_token {
        if cfg.decay_interval_ticks > 0 && tick % cfg.decay_interval_ticks == 0 {
            for ch in s.charge.iter_mut() {
                *ch = ch.saturating_sub(1);
            }
        }
        if tick < cfg.input_duration_ticks {
            for (act, &inp) in s.activation.iter_mut().zip(f.input.iter()) {
                *act = act.saturating_add(inp as i8);
            }
        }

        let incoming = &mut s.incoming[..n];
        incoming.fill(0i16);

        // Prefetch variant 2: index-based with manual bounds elimination, 2 chunks ahead
        let n_full = edge_src.len() / 4 * 4;
        let act_ptr = s.activation.as_ptr();
        let mut ci = 0usize;
        while ci + 8 < n_full {
            // Prefetch 2 chunks (8 edges) ahead
            unsafe {
                prefetch_read(act_ptr.add(edge_src[ci + 8] as usize));
                prefetch_read(act_ptr.add(edge_src[ci + 9] as usize));
                prefetch_read(act_ptr.add(edge_src[ci + 10] as usize));
                prefetch_read(act_ptr.add(edge_src[ci + 11] as usize));
            }

            incoming[edge_tgt[ci] as usize]     += s.activation[edge_src[ci] as usize] as i16;
            incoming[edge_tgt[ci + 1] as usize] += s.activation[edge_src[ci + 1] as usize] as i16;
            incoming[edge_tgt[ci + 2] as usize] += s.activation[edge_src[ci + 2] as usize] as i16;
            incoming[edge_tgt[ci + 3] as usize] += s.activation[edge_src[ci + 3] as usize] as i16;
            ci += 4;
        }
        // Remaining edges (no prefetch)
        while ci < edge_src.len() {
            incoming[edge_tgt[ci] as usize] += s.activation[edge_src[ci] as usize] as i16;
            ci += 1;
        }

        for (ch, &sig) in s.charge.iter_mut().zip(incoming.iter()) {
            *ch = { let val = (*ch as i16) + sig; val.clamp(0, MAX_CHARGE as i16) as u8 };
        }

        let phase_tick = tick % 8;
        for idx in 0..n {
            let ch_idx = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ch_idx) {
                PHASE_BASE[(phase_tick + 9 - ch_idx) & 7] as u16
            } else {
                10
            };
            let charge_x10 = s.charge[idx] as u16 * 10;
            let thresh_x10 = (f.threshold[idx] as u16 + 1) * pm;
            if charge_x10 >= thresh_x10 {
                s.activation[idx] = f.polarity[idx];
                s.charge[idx] = 0;
            } else {
                s.activation[idx] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------

fn describe_noise_floor(noise_pct: f64) -> &'static str {
    if noise_pct <= 5.0 {
        "stable"
    } else if noise_pct <= 10.0 {
        "borderline"
    } else {
        "noisy"
    }
}

struct ABCase {
    name: &'static str,
    neuron_count: usize,
    edge_prob_pct: u64,
    iterations: usize,
    ticks: usize,
}

fn run_ab(case: &ABCase) {
    println!(
        "\n=== {} | H={}, {}% density, {} iters, {} ticks ===",
        case.name, case.neuron_count, case.edge_prob_pct, case.iterations, case.ticks
    );

    let fixture = build_fixture(case.neuron_count, case.edge_prob_pct);
    let cfg = BenchConfig {
        ticks_per_token: case.ticks,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
    };

    println!("  edges: {}", fixture.graph.edge_count());
    let working_set_kb = (fixture.graph.edge_count() * 16 + case.neuron_count * 30) / 1024;
    println!("  estimated working set: ~{}KB", working_set_kb);

    // CTRL: two baseline runs to measure noise floor
    let ctrl_a = {
        let mut s = Scratch::new(case.neuron_count);
        timed_run("CTRL-A (baseline #1)", case.iterations, || {
            s.reset();
            propagate_baseline(black_box(&fixture), black_box(&mut s), black_box(&cfg));
        })
    };
    let ctrl_b = {
        let mut s = Scratch::new(case.neuron_count);
        timed_run("CTRL-B (baseline #2)", case.iterations, || {
            s.reset();
            propagate_baseline(black_box(&fixture), black_box(&mut s), black_box(&cfg));
        })
    };
    let noise_pct = ((ctrl_b.median_ns - ctrl_a.median_ns) / ctrl_a.median_ns * 100.0).abs();
    println!(
        "  NOISE FLOOR: {noise_pct:.1}% ({})",
        describe_noise_floor(noise_pct)
    );

    // TEST A: prefetch 1 chunk ahead
    let pf1 = {
        let mut s = Scratch::new(case.neuron_count);
        timed_run("PREFETCH-1ahead", case.iterations, || {
            s.reset();
            propagate_prefetch(black_box(&fixture), black_box(&mut s), black_box(&cfg));
        })
    };

    // TEST B: prefetch 2 chunks ahead
    let pf2 = {
        let mut s = Scratch::new(case.neuron_count);
        timed_run("PREFETCH-2ahead", case.iterations, || {
            s.reset();
            propagate_prefetch_2ahead(black_box(&fixture), black_box(&mut s), black_box(&cfg));
        })
    };

    // Summary
    let baseline_ns = ctrl_a.median_ns;
    let pf1_delta = (pf1.median_ns - baseline_ns) / baseline_ns * 100.0;
    let pf2_delta = (pf2.median_ns - baseline_ns) / baseline_ns * 100.0;

    println!("\n  SUMMARY (vs baseline):");
    println!("    baseline:        {:.0} ns/iter", baseline_ns);
    println!(
        "    prefetch-1ahead: {:.0} ns/iter ({:+.1}%)",
        pf1.median_ns, pf1_delta
    );
    println!(
        "    prefetch-2ahead: {:.0} ns/iter ({:+.1}%)",
        pf2.median_ns, pf2_delta
    );

    if noise_pct > 10.0 {
        println!("    WARNING: noise floor too high for reliable comparison");
    } else if pf1_delta.abs() < noise_pct && pf2_delta.abs() < noise_pct {
        println!("    VERDICT: difference within noise — prefetch has no measurable effect at this size");
    } else if pf1_delta < -noise_pct || pf2_delta < -noise_pct {
        let best = if pf1_delta < pf2_delta {
            "prefetch-1ahead"
        } else {
            "prefetch-2ahead"
        };
        println!("    VERDICT: {best} wins — prefetch helps at this working set size");
    } else {
        println!("    VERDICT: prefetch adds overhead — not worth it at this size");
    }
}

fn main() {
    print_harness_header();

    let cases = [
        // Small: everything in L1 — prefetch should be neutral/overhead
        ABCase {
            name: "H=64 (L1-resident)",
            neuron_count: 64,
            edge_prob_pct: 5,
            iterations: 5_000,
            ticks: 12,
        },
        // Medium: L1/L2 boundary
        ABCase {
            name: "H=256 (L1/L2 boundary)",
            neuron_count: 256,
            edge_prob_pct: 5,
            iterations: 3_000,
            ticks: 12,
        },
        // Large: spills into L2/L3 — prefetch should start helping
        ABCase {
            name: "H=1024 (L2/L3 spill)",
            neuron_count: 1024,
            edge_prob_pct: 3,
            iterations: 1_000,
            ticks: 12,
        },
        // XL: solidly in L3 territory — prefetch should clearly help
        ABCase {
            name: "H=4096 (L3 territory)",
            neuron_count: 4096,
            edge_prob_pct: 1,
            iterations: 200,
            ticks: 12,
        },
    ];

    for case in &cases {
        run_ab(case);
    }
}
