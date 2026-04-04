use instnct_core::ConnectionGraph;
use std::time::Instant;

const WARMUP: usize = 100;
const RUNS: usize = 7;
const GRAPH_SEED: u64 = 42;

#[derive(Clone, Copy, Debug)]
pub struct RunStats {
    pub median_ns: f64,
    pub stddev_ns: f64,
    pub cv_pct: f64,
    pub min_ns: f64,
    pub max_ns: f64,
}

impl RunStats {
    pub fn print(self, name: &str) {
        println!(
            "  {name:45} median={:>10.0} ns  sd={:>6.0}  cv={:>4.1}%  [{:.0}..{:.0}]",
            self.median_ns, self.stddev_ns, self.cv_pct, self.min_ns, self.max_ns
        );
    }
}

#[cfg(target_os = "windows")]
fn pin_and_boost() -> (bool, bool) {
    unsafe extern "system" {
        fn SetThreadAffinityMask(h_thread: isize, dw_thread_affinity_mask: usize) -> usize;
        fn GetCurrentThread() -> isize;
        fn SetPriorityClass(h_process: isize, dw_priority_class: u32) -> i32;
        fn GetCurrentProcess() -> isize;
    }

    unsafe {
        let pinned = SetThreadAffinityMask(GetCurrentThread(), 1) != 0;
        let boosted = SetPriorityClass(GetCurrentProcess(), 0x00000080) != 0;
        (pinned, boosted)
    }
}

pub fn print_harness_header() {
    #[cfg(target_os = "windows")]
    {
        let (pinned, boosted) = pin_and_boost();
        let pin_status = if pinned {
            "core 0 pinned"
        } else {
            "affinity FAILED"
        };
        let prio_status = if boosted {
            "HIGH priority"
        } else {
            "priority FAILED"
        };
        println!("Deterministic harness: {pin_status}, {prio_status}, {RUNS} runs/test\n");
    }

    #[cfg(not(target_os = "windows"))]
    {
        println!("Deterministic harness: best-effort (no affinity/priority), {RUNS} runs/test\n");
    }
}

pub fn build_graph(neuron_count: usize, edge_prob_pct: u64) -> ConnectionGraph {
    let mut graph = ConnectionGraph::new(neuron_count);
    let mut rng = GRAPH_SEED;
    for i in 0..neuron_count {
        for j in 0..neuron_count {
            if i == j {
                continue;
            }
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 100 < edge_prob_pct {
                graph.add_edge(i as u16, j as u16);
            }
        }
    }
    graph
}

pub fn timed_run(name: &str, iterations: usize, mut body: impl FnMut()) -> RunStats {
    for _ in 0..WARMUP {
        body();
    }

    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let start = Instant::now();
        for _ in 0..iterations {
            body();
        }
        times.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }

    times.sort_by(f64::total_cmp);
    let median_ns = times[RUNS / 2];
    let mean = times.iter().sum::<f64>() / RUNS as f64;
    let stddev_ns =
        (times.iter().map(|time| (time - mean).powi(2)).sum::<f64>() / RUNS as f64).sqrt();
    let cv_pct = stddev_ns / mean * 100.0;
    let stats = RunStats {
        median_ns,
        stddev_ns,
        cv_pct,
        min_ns: times[0],
        max_ns: times[RUNS - 1],
    };
    stats.print(name);
    stats
}
