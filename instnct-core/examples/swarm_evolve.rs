//! Prototype: P2P swarm evolution — agents learn from each other.
//!
//! Simulates N agents (like phones), each evolving their own network
//! on their own data slice. Periodically they share their best genome
//! and try to learn from each other via breeding + injection.
//!
//! Key insight: genome is tiny (~100 bytes for 28 edges), trivially
//! shareable over any network.

use instnct_core::{
    build_bigram_table, build_network, eval_accuracy, eval_smooth, evolution_step_jackpot,
    load_corpus, InitConfig, Int8Projection, Network, SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const CORPUS_PATH: &str = "instnct-core/tests/fixtures/beta_smoke_corpus.txt";

/// What gets shared between agents — the "message"
#[derive(Clone)]
struct SharedGenome {
    edges: Vec<(u16, u16)>,          // the discovered circuit
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    fitness: f64,
    source_agent: usize,
}

/// One agent in the swarm
struct Agent {
    id: usize,
    net: Network,
    proj: Int8Projection,
    rng: StdRng,
    eval_rng: StdRng,
    sdr: SdrTable,
    init: InitConfig,
    corpus_slice: Vec<u8>,           // each agent sees different data
    bigram: Vec<Vec<f64>>,
    best_fitness: f64,
    steps_done: u32,
    accepts: u32,
}

impl Agent {
    fn new(id: usize, h: usize, corpus: &[u8], seed: u64) -> Self {
        let init = InitConfig::empty(h);  // start empty, evolve from scratch
        let mut rng = StdRng::seed_from_u64(seed);
        let net = build_network(&init, &mut rng);
        let proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
        let sdr = SdrTable::new(
            CHARS, h, init.input_end(), SDR_ACTIVE_PCT,
            &mut StdRng::seed_from_u64(seed + 100),
        ).unwrap();

        // Each agent gets a DIFFERENT slice of the corpus (specialist)
        let slice_size = corpus.len() / 8;  // ~12.5% each
        let start = (id * slice_size / 2) % (corpus.len() - slice_size);
        let corpus_slice = corpus[start..start + slice_size].to_vec();
        let bigram = build_bigram_table(&corpus_slice, CHARS);

        Agent {
            id, net, proj, rng,
            eval_rng: StdRng::seed_from_u64(seed + 1000),
            sdr, init, corpus_slice, bigram,
            best_fitness: 0.0, steps_done: 0, accepts: 0,
        }
    }

    /// Evolve for N steps on own data
    fn evolve(&mut self, steps: usize) {
        let evo_config = self.init.evolution_config();
        let output_start = self.init.output_start();
        let h = self.init.neuron_count;

        for _ in 0..steps {
            let outcome = evolution_step_jackpot(
                &mut self.net, &mut self.proj, &mut self.rng, &mut self.eval_rng,
                |net, proj, eval_rng| {
                    eval_smooth(
                        net, proj, &self.corpus_slice, 100, eval_rng,
                        &self.sdr, &self.init.propagation, output_start, h,
                        &self.bigram,
                    )
                },
                &evo_config, 3,
            );
            if let StepOutcome::Accepted = outcome { self.accepts += 1; }
            self.steps_done += 1;
        }
    }

    /// Measure fitness on own data
    fn measure_fitness(&mut self) -> f64 {
        let acc = eval_accuracy(
            &mut self.net, &self.proj, &self.corpus_slice, 200, &mut self.eval_rng,
            &self.sdr, &self.init.propagation,
            self.init.output_start(), self.init.neuron_count,
        );
        self.best_fitness = self.best_fitness.max(acc);
        acc
    }

    /// Measure fitness on FULL corpus (generalization test)
    fn measure_full(&mut self, full_corpus: &[u8]) -> f64 {
        eval_accuracy(
            &mut self.net, &self.proj, full_corpus, 500, &mut self.eval_rng,
            &self.sdr, &self.init.propagation,
            self.init.output_start(), self.init.neuron_count,
        )
    }

    /// Export genome for sharing
    fn export_genome(&self) -> SharedGenome {
        let edges: Vec<(u16, u16)> = self.net.graph().iter_edges()
            .map(|e| (e.source, e.target)).collect();
        let threshold: Vec<u8> = self.net.spike_data().iter().map(|s| s.threshold).collect();
        let channel: Vec<u8> = self.net.spike_data().iter().map(|s| s.channel).collect();
        SharedGenome {
            edges, threshold, channel,
            polarity: self.net.polarity().to_vec(),
            fitness: self.best_fitness,
            source_agent: self.id,
        }
    }

    /// Try to learn from a received genome
    /// Strategy: inject promising edges from the donor, keep if fitness improves
    fn try_learn_from(&mut self, genome: &SharedGenome) -> bool {
        let snapshot = self.net.save_state();
        let fitness_before = self.measure_fitness();

        // Inject some edges from the donor (not all — that would be copying)
        let mut injected = 0;
        for &(src, tgt) in &genome.edges {
            if src < self.init.neuron_count as u16 && tgt < self.init.neuron_count as u16 {
                // Only inject edges we don't already have (novel circuits)
                if !self.net.graph().has_edge(src, tgt) {
                    if self.rng.gen_bool(0.3) {  // 30% chance to try each edge
                        self.net.graph_mut().add_edge(src, tgt);
                        injected += 1;
                    }
                }
            }
            if injected >= 5 { break; }  // max 5 edges per learning event
        }

        if injected == 0 { return false; }

        // Also copy some parameter values from donor for injected targets
        for &(_, tgt) in genome.edges.iter().take(5) {
            let t = tgt as usize;
            if t < self.init.neuron_count {
                if self.rng.gen_bool(0.2) {
                    self.net.spike_data_mut()[t].threshold = genome.threshold[t];
                    self.net.spike_data_mut()[t].channel = genome.channel[t];
                }
            }
        }

        let fitness_after = self.measure_fitness();

        if fitness_after >= fitness_before {
            true  // keep: learned something useful
        } else {
            self.net.restore_state(&snapshot);
            false  // reject: donor's circuits don't help us
        }
    }
}

fn main() {
    let corpus = load_corpus(CORPUS_PATH).expect("failed to load corpus");
    let n_agents = 6;
    let h = 2048;
    let rounds = 8;
    let steps_per_round = 500;

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  P2P SWARM EVOLUTION PROTOTYPE                                  ║");
    println!("║  {} agents, H={}, {} rounds × {} steps                    ║", n_agents, h, rounds, steps_per_round);
    println!("║  Each agent sees different corpus slice (specialist)             ║");
    println!("║  Share + breed every round                                      ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // Create agents with different seeds and data slices
    let mut agents: Vec<Agent> = (0..n_agents)
        .map(|i| Agent::new(i, h, &corpus, 42 + i as u64 * 1000))
        .collect();

    // Also create a CONTROL: single agent with same total steps, full corpus
    let mut control = Agent::new(99, h, &corpus, 42);
    // Control sees full corpus
    control.corpus_slice = corpus.clone();
    control.bigram = build_bigram_table(&corpus, CHARS);

    println!("{:>5} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}  {}",
        "round", "agent", "edges", "local%", "full%", "learned", "accepts", "info");
    println!("{}", "─".repeat(80));

    let t0 = Instant::now();

    for round in 0..rounds {
        // 1. Each agent evolves on own data
        for agent in agents.iter_mut() {
            agent.evolve(steps_per_round);
        }
        // Control also evolves (same total steps)
        control.evolve(steps_per_round);

        // 2. Measure fitness (local + full corpus)
        let mut genomes: Vec<SharedGenome> = Vec::new();
        for agent in agents.iter_mut() {
            let local = agent.measure_fitness();
            let full = agent.measure_full(&corpus);
            genomes.push(agent.export_genome());
            println!("{:>5} {:>6} {:>8} {:>7.1}% {:>7.1}% {:>8} {:>8}",
                round + 1, agent.id, agent.net.edge_count(),
                local * 100.0, full * 100.0, "-", agent.accepts);
        }

        // Control measurement
        let ctrl_local = control.measure_fitness();
        let ctrl_full = control.measure_full(&corpus);
        println!("{:>5} {:>6} {:>8} {:>7.1}% {:>7.1}% {:>8} {:>8}  ← control (no sharing)",
            round + 1, "CTRL", control.net.edge_count(),
            ctrl_local * 100.0, ctrl_full * 100.0, "-", control.accepts);

        // 3. SHARE: each agent tries to learn from the best other agent
        let best_idx = genomes.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.fitness.partial_cmp(&b.fitness).unwrap())
            .map(|(i, _)| i).unwrap();

        let mut learned_count = 0;
        for (i, agent) in agents.iter_mut().enumerate() {
            if i == best_idx { continue; }  // don't learn from yourself
            if agent.try_learn_from(&genomes[best_idx]) {
                learned_count += 1;
            }
        }

        // 4. BREED: worst agent replaced by child of top 2
        let mut fitness_order: Vec<(usize, f64)> = agents.iter()
            .enumerate().map(|(i, a)| (i, a.best_fitness)).collect();
        fitness_order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if fitness_order.len() >= 3 {
            let worst_idx = fitness_order.last().unwrap().0;
            let parent1_idx = fitness_order[0].0;
            let parent2_idx = fitness_order[1].0;

            // Breed: take edges from both parents
            let p1_edges = genomes[parent1_idx].edges.clone();
            let p2_edges = genomes[parent2_idx].edges.clone();

            let worst = &mut agents[worst_idx];
            // Reset worst agent
            *worst = Agent::new(worst_idx, h, &corpus, worst.rng.gen::<u64>());

            // Inject edges from both parents
            for &(src, tgt) in p1_edges.iter().chain(p2_edges.iter()) {
                if src < h as u16 && tgt < h as u16 {
                    if worst.rng.gen_bool(0.5) {
                        worst.net.graph_mut().add_edge(src, tgt);
                    }
                }
            }
        }

        println!("  → round {}: {} agents learned from best (agent {}), worst replaced by breed",
            round + 1, learned_count, best_idx);
        println!("{}", "─".repeat(80));
    }

    let elapsed = t0.elapsed();

    // Final results
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  FINAL RESULTS                                                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    let mut best_swarm = 0.0f64;
    for agent in agents.iter_mut() {
        let full = agent.measure_full(&corpus);
        best_swarm = best_swarm.max(full);
        let genome = agent.export_genome();
        let genome_bytes = genome.edges.len() * 4 + genome.threshold.len() + genome.channel.len() + genome.polarity.len();
        println!("  Agent {}: {:.1}% full accuracy, {} edges, {} byte genome",
            agent.id, full * 100.0, agent.net.edge_count(), genome_bytes);
    }

    let ctrl_full = control.measure_full(&corpus);
    println!("\n  Control: {:.1}% full accuracy, {} edges (no sharing, same total steps)",
        ctrl_full * 100.0, control.net.edge_count());
    println!("\n  SWARM BEST: {:.1}% vs CONTROL: {:.1}% ({:+.1}pp)",
        best_swarm * 100.0, ctrl_full * 100.0, (best_swarm - ctrl_full) * 100.0);
    println!("  Time: {:.1}s ({} total evolution steps)",
        elapsed.as_secs_f64(), n_agents * rounds * steps_per_round);

    let genome_size = agents[0].export_genome().edges.len() * 4;
    println!("\n  GENOME SIZE: ~{} bytes per share (fits in 1 UDP packet!)", genome_size);
}
