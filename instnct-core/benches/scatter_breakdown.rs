//! Scatter-add micro-breakdown: what exactly costs time inside the edge loop?
//!
//! Isolates sub-operations of the scatter-add to find the true bottleneck:
//! - fill(0): zeroing the incoming buffer
//! - seq-read-only: iterate edges, read activation[src], accumulate into a single var (no random write)
//! - random-write-only: write constant 1 into incoming[target] (no random read)
//! - read+write: the actual scatter-add (random read activation + random RMW incoming)

mod common;

use common::{build_graph, print_harness_header, timed_run};
use instnct_core::ConnectionGraph;
use std::hint::black_box;

struct Fixture {
    graph: ConnectionGraph,
    activation: Vec<i8>,
    incoming: Vec<i16>,
    n: usize,
}

fn build_fixture(neuron_count: usize, edge_prob_pct: u64) -> Fixture {
    let graph = build_graph(neuron_count, edge_prob_pct);
    // Seed activation with some nonzero values so reads aren't trivially optimized out
    let mut activation = vec![0i8; neuron_count];
    for i in 0..neuron_count {
        activation[i] = ((i % 3) as i8) - 1; // -1, 0, 1 pattern
    }
    Fixture {
        graph,
        activation,
        incoming: vec![0i16; neuron_count],
        n: neuron_count,
    }
}

// --- Sub-operations, each run for TICKS iterations to match real workload ---

const TICKS: usize = 12;

/// Just zero the buffer (TICKS times)
fn bench_fill_only(f: &mut Fixture) {
    let n = f.n;
    for _tick in 0..TICKS {
        f.incoming[..n].fill(0);
    }
}

/// Sequential read of edge arrays + random read of activation[src].
/// Accumulate into a single variable (no random write to incoming[]).
/// This isolates the READ cost.
fn bench_read_only(f: &mut Fixture) -> i32 {
    let (edge_src, edge_tgt) = f.graph.edge_endpoints_pub();
    let mut sink: i32 = 0;
    for _tick in 0..TICKS {
        for (sc, _tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            sink = sink.wrapping_add(f.activation[sc[0] as usize] as i32);
            sink = sink.wrapping_add(f.activation[sc[1] as usize] as i32);
            sink = sink.wrapping_add(f.activation[sc[2] as usize] as i32);
            sink = sink.wrapping_add(f.activation[sc[3] as usize] as i32);
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            sink = sink.wrapping_add(f.activation[edge_src[i] as usize] as i32);
        }
    }
    sink
}

/// Random write to incoming[target], but with a constant value (no random read).
/// This isolates the WRITE cost.
fn bench_write_only(f: &mut Fixture) {
    let (edge_src, edge_tgt) = f.graph.edge_endpoints_pub();
    let n = f.n;
    for _tick in 0..TICKS {
        f.incoming[..n].fill(0);
        for (_sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            f.incoming[tc[0] as usize] += 1;
            f.incoming[tc[1] as usize] += 1;
            f.incoming[tc[2] as usize] += 1;
            f.incoming[tc[3] as usize] += 1;
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            f.incoming[edge_tgt[i] as usize] += 1;
        }
    }
}

/// Sequential iteration only: read edge_src and edge_tgt arrays, sum indices.
/// No activation or incoming access at all. Isolates pure iteration cost.
fn bench_seq_iterate(f: &mut Fixture) -> usize {
    let (edge_src, edge_tgt) = f.graph.edge_endpoints_pub();
    let mut sink: usize = 0;
    for _tick in 0..TICKS {
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            sink = sink.wrapping_add(sc[0] as usize).wrapping_add(tc[0] as usize);
            sink = sink.wrapping_add(sc[1] as usize).wrapping_add(tc[1] as usize);
            sink = sink.wrapping_add(sc[2] as usize).wrapping_add(tc[2] as usize);
            sink = sink.wrapping_add(sc[3] as usize).wrapping_add(tc[3] as usize);
        }
    }
    sink
}

/// Full scatter-add (the real thing)
fn bench_full_scatter(f: &mut Fixture) {
    let (edge_src, edge_tgt) = f.graph.edge_endpoints_pub();
    let n = f.n;
    for _tick in 0..TICKS {
        f.incoming[..n].fill(0);
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            f.incoming[tc[0] as usize] += f.activation[sc[0] as usize] as i16;
            f.incoming[tc[1] as usize] += f.activation[sc[1] as usize] as i16;
            f.incoming[tc[2] as usize] += f.activation[sc[2] as usize] as i16;
            f.incoming[tc[3] as usize] += f.activation[sc[3] as usize] as i16;
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            f.incoming[edge_tgt[i] as usize] += f.activation[edge_src[i] as usize] as i16;
        }
    }
}

struct SizeCase {
    name: &'static str,
    neuron_count: usize,
    edge_prob_pct: u64,
    iterations: usize,
}

fn run_breakdown(case: &SizeCase) {
    println!(
        "\n=== {} | H={}, {}% density ===",
        case.name, case.neuron_count, case.edge_prob_pct
    );

    let mut f = build_fixture(case.neuron_count, case.edge_prob_pct);
    println!(
        "  edges: {} | activation: {} bytes | incoming: {} bytes",
        f.graph.edge_count(),
        f.n * 4,
        f.n * 4
    );

    let t_full = {
        timed_run("full scatter-add", case.iterations, || {
            bench_full_scatter(black_box(&mut f));
        })
    };

    let t_fill = {
        timed_run("fill(0) only", case.iterations, || {
            bench_fill_only(black_box(&mut f));
        })
    };

    let t_seq = {
        timed_run("seq iterate edges (no array access)", case.iterations, || {
            black_box(bench_seq_iterate(black_box(&mut f)));
        })
    };

    let t_read = {
        timed_run("random read activation[src] only", case.iterations, || {
            black_box(bench_read_only(black_box(&mut f)));
        })
    };

    let t_write = {
        timed_run("random RMW incoming[tgt] only", case.iterations, || {
            bench_write_only(black_box(&mut f));
        })
    };

    println!("\n  SCATTER-ADD BREAKDOWN (% of full scatter):");
    println!("    full scatter-add:    {:>10.0} ns", t_full.median_ns);
    println!(
        "    fill(0):             {:>10.0} ns  ({:>5.1}%)",
        t_fill.median_ns,
        t_fill.median_ns / t_full.median_ns * 100.0
    );
    println!(
        "    seq iterate edges:   {:>10.0} ns  ({:>5.1}%)",
        t_seq.median_ns,
        t_seq.median_ns / t_full.median_ns * 100.0
    );
    println!(
        "    random read [src]:   {:>10.0} ns  ({:>5.1}%)",
        t_read.median_ns,
        t_read.median_ns / t_full.median_ns * 100.0
    );
    println!(
        "    random RMW [tgt]:    {:>10.0} ns  ({:>5.1}%)",
        t_write.median_ns,
        t_write.median_ns / t_full.median_ns * 100.0
    );

    // Estimate: is the full scatter = read + write, or is there synergy/conflict?
    let sum_rw = t_read.median_ns + t_write.median_ns;
    println!(
        "    read+write sum:      {:>10.0} ns  (full is {:+.1}% vs sum)",
        sum_rw,
        (t_full.median_ns - sum_rw) / sum_rw * 100.0
    );
}

fn main() {
    print_harness_header();

    let cases = [
        SizeCase {
            name: "small",
            neuron_count: 256,
            edge_prob_pct: 5,
            iterations: 5_000,
        },
        SizeCase {
            name: "medium",
            neuron_count: 1024,
            edge_prob_pct: 3,
            iterations: 2_000,
        },
        SizeCase {
            name: "large",
            neuron_count: 4096,
            edge_prob_pct: 1,
            iterations: 500,
        },
    ];

    for case in &cases {
        run_breakdown(case);
    }
}
