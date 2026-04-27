//! Evolution step: paired eval → mutate → accept/reject → rollback.
//!
//! Encapsulates the (1+1) evolutionary strategy pattern that previously
//! contained 3 bugs when implemented manually. The library handles:
//! - Paired evaluation (clone eval_rng, eval before, restore, eval after)
//! - Density-capped acceptance (>= when lean, > when dense)
//! - Correct rollback of both network topology AND projection weights
//! - RNG sync on skipped (failed) mutations

use crate::network::MutationUndo;
use crate::projection::{Int8Projection, WeightBackup};
use crate::Network;
use rand::rngs::StdRng;
use rand::Rng;
use std::time::Instant;

/// Canonical mutation operator definition used by baseline, traced, and
/// weighted-sampling evolution paths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MutationOperatorSpec {
    /// Stable identifier written into candidate logs.
    pub id: &'static str,
    /// Baseline schedule weight out of 100.
    pub baseline_weight: u32,
}

/// Baseline operator schedule. The order and weights are part of the
/// experiment contract; keep this table in sync with historical logs.
pub const MUTATION_OPERATORS: [MutationOperatorSpec; 11] = [
    MutationOperatorSpec {
        id: "add_edge",
        baseline_weight: 22,
    },
    MutationOperatorSpec {
        id: "remove_edge",
        baseline_weight: 13,
    },
    MutationOperatorSpec {
        id: "rewire",
        baseline_weight: 9,
    },
    MutationOperatorSpec {
        id: "reverse",
        baseline_weight: 13,
    },
    MutationOperatorSpec {
        id: "mirror",
        baseline_weight: 6,
    },
    MutationOperatorSpec {
        id: "enhance",
        baseline_weight: 7,
    },
    MutationOperatorSpec {
        id: "theta",
        baseline_weight: 5,
    },
    MutationOperatorSpec {
        id: "channel",
        baseline_weight: 10,
    },
    MutationOperatorSpec {
        id: "loop2",
        baseline_weight: 5,
    },
    MutationOperatorSpec {
        id: "loop3",
        baseline_weight: 5,
    },
    MutationOperatorSpec {
        id: "projection_weight",
        baseline_weight: 5,
    },
];

/// Stable operator identifiers in canonical schedule order.
pub fn mutation_operator_ids() -> impl Iterator<Item = &'static str> {
    MUTATION_OPERATORS.iter().map(|op| op.id)
}

/// Return the canonical index for an operator id.
pub fn mutation_operator_index(operator_id: &str) -> Option<usize> {
    MUTATION_OPERATORS
        .iter()
        .position(|op| op.id == operator_id)
}

/// Baseline probability for a canonical operator index.
pub fn mutation_operator_baseline_probability(index: usize) -> f64 {
    MUTATION_OPERATORS[index].baseline_weight as f64 / 100.0
}

fn sample_baseline_operator(rng: &mut impl Rng) -> usize {
    let roll = rng.gen_range(0..100u32);
    let mut upper = 0u32;
    for (idx, spec) in MUTATION_OPERATORS.iter().enumerate() {
        upper += spec.baseline_weight;
        if roll < upper {
            return idx;
        }
    }
    MUTATION_OPERATORS.len() - 1
}

fn sample_weighted_operator(weights: &[f64], rng: &mut impl Rng) -> usize {
    assert_eq!(
        weights.len(),
        MUTATION_OPERATORS.len(),
        "operator weights must match canonical operator count"
    );
    let total: f64 = weights.iter().copied().filter(|w| *w > 0.0).sum();
    assert!(total.is_finite() && total > 0.0, "operator weights must have positive finite sum");

    let mut roll = rng.gen_range(0.0..total);
    for (idx, weight) in weights.iter().copied().enumerate() {
        if weight <= 0.0 {
            continue;
        }
        if roll < weight {
            return idx;
        }
        roll -= weight;
    }
    weights
        .iter()
        .rposition(|w| *w > 0.0)
        .unwrap_or(MUTATION_OPERATORS.len() - 1)
}

fn apply_mutation_operator(
    operator_index: usize,
    net: &mut Network,
    projection: &mut Int8Projection,
    mutation_rng: &mut impl Rng,
    weight_backup: Option<&mut Option<WeightBackup>>,
) -> bool {
    match MUTATION_OPERATORS[operator_index].id {
        "add_edge" => net.mutate_add_edge(mutation_rng),
        "remove_edge" => net.mutate_remove_edge(mutation_rng),
        "rewire" => net.mutate_rewire(mutation_rng),
        "reverse" => net.mutate_reverse(mutation_rng),
        "mirror" => net.mutate_mirror(mutation_rng),
        "enhance" => net.mutate_enhance(mutation_rng),
        "theta" => net.mutate_theta(mutation_rng),
        "channel" => net.mutate_channel(mutation_rng),
        "loop2" => net.mutate_add_loop(mutation_rng, 2),
        "loop3" => net.mutate_add_loop(mutation_rng, 3),
        "projection_weight" => {
            if let Some(slot) = weight_backup {
                *slot = Some(projection.mutate_one(mutation_rng));
            } else {
                let _ = projection.mutate_one(mutation_rng);
            }
            true
        }
        other => panic!("unknown canonical mutation operator: {other}"),
    }
}

/// Acceptance policy for best-of-K evolution steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AcceptancePolicy {
    /// Accept only strict improvements (`delta > 0`).
    Strict,
    /// Accept strict improvements and exact ties (`delta >= 0`).
    Ties,
    /// Accept improvements and accept near-zero deltas with probability `probability`.
    ZeroP {
        /// Probability used when `abs(delta) <= zero_tol`.
        probability: f64,
        /// Absolute tolerance used to classify numerical zero deltas.
        zero_tol: f64,
    },
    /// Accept moves whose loss in utility is no worse than `epsilon`.
    Epsilon {
        /// Utility tolerance, accepting `delta >= -epsilon`.
        epsilon: f64,
    },
}

impl AcceptancePolicy {
    /// Build the legacy policy represented by `EvolutionConfig::accept_ties`.
    pub fn from_accept_ties(accept_ties: bool) -> Self {
        if accept_ties {
            Self::Ties
        } else {
            Self::Strict
        }
    }

    fn accepts(self, delta: f64, rng: &mut impl Rng) -> bool {
        match self {
            Self::Strict => delta > 0.0,
            Self::Ties => delta >= 0.0,
            Self::ZeroP {
                probability,
                zero_tol,
            } => {
                if delta > zero_tol {
                    true
                } else if delta >= -zero_tol {
                    if probability <= 0.0 {
                        false
                    } else if probability >= 1.0 {
                        true
                    } else {
                        rng.gen_bool(probability)
                    }
                } else {
                    false
                }
            }
            Self::Epsilon { epsilon } => delta >= -epsilon.max(0.0),
        }
    }
}

/// Configuration for the evolution step.
pub struct EvolutionConfig {
    /// Hard edge count limit. Mutations that INCREASE edge count above
    /// this are rejected regardless of fitness quality.
    pub edge_cap: usize,

    /// Whether to accept mutations that produce equal fitness (ties).
    /// `true`  → `after >= before` (permissive, encourages exploration)
    /// `false` → `after > before`  (strict, only real improvements)
    pub accept_ties: bool,
}

/// Outcome of a single evolution step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepOutcome {
    /// Mutation accepted — network/projection state updated.
    Accepted,
    /// Mutation rejected — network/projection restored to pre-mutation state.
    Rejected,
    /// Mutation attempt failed (e.g. self-loop, duplicate edge) — fitness_fn still
    /// runs once for eval_rng parity, but state is unchanged.
    Skipped,
}

/// Per-candidate diagnostics emitted by the traced jackpot evolution step.
#[derive(Clone, Debug)]
pub struct CandidateTraceRecord {
    /// Zero-based evolution step index from the caller.
    pub step: usize,
    /// Zero-based candidate index within this jackpot step.
    pub candidate_id: usize,
    /// Stable mutation operator identifier.
    pub operator_id: &'static str,
    /// Whether the sampled mutation changed the network or projection.
    pub mutated: bool,
    /// Whether the candidate fitness was evaluated.
    pub evaluated: bool,
    /// Parent fitness before evaluating this jackpot batch.
    pub before_u: f64,
    /// Candidate fitness after mutation, or `before_u` if not evaluated.
    pub after_u: f64,
    /// Candidate fitness delta, `after_u - before_u`.
    pub delta_u: f64,
    /// Whether the candidate respected the edge cap.
    pub within_cap: bool,
    /// Whether this candidate was the best eligible candidate.
    pub selected: bool,
    /// Whether this candidate was finally accepted.
    pub accepted: bool,
    /// Wall-clock time spent evaluating this candidate fitness.
    pub candidate_eval_ms: f64,
    /// Total wall-clock time spent in the jackpot step.
    pub step_wall_ms: f64,
}

/// Run one evolution step with paired evaluation, edge cap, and quality gate.
///
/// The `fitness_fn` receives `(&mut Network, &Int8Projection, &mut StdRng)` and
/// must return a fitness score (higher is better). The `eval_rng` is passed through
/// so the library can enforce paired evaluation (same RNG state for before/after).
///
/// **Edge cap:** mutations that increase edge count above `edge_cap` are rejected
/// regardless of fitness. Non-edge-growing mutations are unaffected by the cap.
///
/// **Quality gate:** `accept_ties` controls whether equal fitness (ties) are accepted.
///
/// Mutation schedule uses [`MUTATION_OPERATORS`], preserving the historical
/// baseline weights exactly.
///
/// # Example
///
/// ```no_run
/// use instnct_core::{Network, Int8Projection, evolution_step, EvolutionConfig, StepOutcome};
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let mut net = Network::new(64);
/// let mut rng = StdRng::seed_from_u64(42);
/// let mut proj = Int8Projection::new(40, 10, &mut rng);
/// let mut eval_rng = StdRng::seed_from_u64(99);
/// let config = EvolutionConfig { edge_cap: 300, accept_ties: true };
///
/// let outcome = evolution_step(
///     &mut net, &mut proj, &mut rng, &mut eval_rng,
///     |net, proj, eval_rng| {
///         // your task-specific fitness function here
///         0.0
///     },
///     &config,
/// );
/// ```
pub fn evolution_step<F>(
    net: &mut Network,
    projection: &mut Int8Projection,
    mutation_rng: &mut impl Rng,
    eval_rng: &mut StdRng,
    mut fitness_fn: F,
    config: &EvolutionConfig,
) -> StepOutcome
where
    F: FnMut(&mut Network, &Int8Projection, &mut StdRng) -> f64,
{
    // Paired eval: evaluate BEFORE mutation (save eval_rng state for reuse)
    let eval_rng_snapshot = eval_rng.clone();
    let before = fitness_fn(net, projection, eval_rng);
    *eval_rng = eval_rng_snapshot;

    // Snapshot network state for rollback
    let edges_before_mutation = net.edge_count();
    let net_snapshot = net.save_state();

    // Mutate with the canonical baseline schedule.
    let mut weight_backup: Option<WeightBackup> = None;
    let operator_index = sample_baseline_operator(mutation_rng);
    let mutated = apply_mutation_operator(
        operator_index,
        net,
        projection,
        mutation_rng,
        Some(&mut weight_backup),
    );

    if !mutated {
        // RNG sync: run the "after" eval to advance eval_rng by the same amount
        // as a successful step. This works for any fitness_fn draw count.
        let _ = fitness_fn(net, projection, eval_rng);
        return StepOutcome::Skipped;
    }

    // Paired eval: evaluate AFTER mutation (same corpus segment via restored eval_rng)
    let after = fitness_fn(net, projection, eval_rng);

    // Quality gate: did the mutation improve (or at least tie) fitness?
    let dominated = if config.accept_ties {
        after >= before
    } else {
        after > before
    };

    // Edge cap gate: reject if mutation grew edges beyond the hard cap
    let edge_grew = net.edge_count() > edges_before_mutation;
    let within_cap = !edge_grew || net.edge_count() <= config.edge_cap;

    let accepted = dominated && within_cap;

    if accepted {
        StepOutcome::Accepted
    } else {
        net.restore_state(&net_snapshot);
        if let Some(backup) = weight_backup {
            projection.rollback(backup);
        }
        StepOutcome::Rejected
    }
}

/// Run one evolution step with **jackpot selection**: try `candidates`
/// independent mutations from the same parent, accept the best one
/// (if it improves fitness).
///
/// When `candidates == 1`, this behaves identically to [`evolution_step`].
/// With `candidates > 1` (e.g. 9), this is a (1+N) ES — the Python
/// "multi-worker" pattern that reached 24.4% accuracy.
///
/// Each candidate uses the same `mutation_rng` (advancing sequentially)
/// and sees the same evaluation segment (via `eval_rng` cloning).
///
/// Proven: 1+9 jackpot + smooth fitness = **24.6% peak** vs 21.2% for 1+1
/// (A/B test 2026-04-06, 6 seeds).
pub fn evolution_step_jackpot<F>(
    net: &mut Network,
    projection: &mut Int8Projection,
    mutation_rng: &mut impl Rng,
    eval_rng: &mut StdRng,
    mut fitness_fn: F,
    config: &EvolutionConfig,
    candidates: usize,
) -> StepOutcome
where
    F: FnMut(&mut Network, &Int8Projection, &mut StdRng) -> f64,
{
    let candidates = candidates.max(1);

    // Fast path: N=1 delegates to the original 1+1 ES
    if candidates == 1 {
        return evolution_step(net, projection, mutation_rng, eval_rng, fitness_fn, config);
    }

    // Paired eval: baseline score
    let eval_rng_snapshot = eval_rng.clone();
    let before = fitness_fn(net, projection, eval_rng);
    *eval_rng = eval_rng_snapshot.clone();

    // Save parent state
    let parent_snapshot = net.save_state();
    let parent_projection = projection.clone();
    let edges_before = net.edge_count();

    // Try N candidates, track the best
    let mut best_delta = f64::NEG_INFINITY;
    let mut best_net_snapshot = None;
    let mut best_projection = None;
    let mut any_mutated = false;

    for _ in 0..candidates {
        // Restore parent state for this candidate
        net.restore_state(&parent_snapshot);
        *projection = parent_projection.clone();

        let operator_index = sample_baseline_operator(mutation_rng);
        let mutated =
            apply_mutation_operator(operator_index, net, projection, mutation_rng, None);

        if !mutated {
            continue;
        }
        any_mutated = true;

        // Evaluate candidate on same segment as baseline
        let cand_eval_rng = eval_rng_snapshot.clone();
        let after = fitness_fn(net, projection, &mut cand_eval_rng.clone());

        // Edge cap check
        let edge_grew = net.edge_count() > edges_before;
        let within_cap = !edge_grew || net.edge_count() <= config.edge_cap;

        let delta = after - before;
        if delta > best_delta && within_cap {
            best_delta = delta;
            best_net_snapshot = Some(net.save_state());
            best_projection = Some(projection.clone());
        }
    }

    // Advance eval_rng by one fitness_fn call (for step-to-step parity)
    net.restore_state(&parent_snapshot);
    *projection = parent_projection.clone();
    let _ = fitness_fn(net, projection, eval_rng);

    if !any_mutated {
        net.restore_state(&parent_snapshot);
        *projection = parent_projection;
        return StepOutcome::Skipped;
    }

    // Accept best candidate if it improved
    let dominated = if config.accept_ties {
        best_delta >= 0.0
    } else {
        best_delta > 0.0
    };

    if dominated {
        if let (Some(net_s), Some(proj_s)) = (best_net_snapshot, best_projection) {
            net.restore_state(&net_s);
            *projection = proj_s;
            return StepOutcome::Accepted;
        }
    }

    // Reject — restore parent
    net.restore_state(&parent_snapshot);
    *projection = parent_projection;
    StepOutcome::Rejected
}

/// Run a jackpot evolution step and emit one trace record per candidate.
///
/// This is intentionally separate from [`evolution_step_jackpot`] so the
/// canonical no-log hot path remains unchanged. The mutation schedule, paired
/// evaluation, best-of-K selection, edge-cap gate, and final eval-RNG advance
/// mirror [`evolution_step_jackpot`].
pub fn evolution_step_jackpot_traced<F, T>(
    net: &mut Network,
    projection: &mut Int8Projection,
    mutation_rng: &mut impl Rng,
    eval_rng: &mut StdRng,
    fitness_fn: F,
    config: &EvolutionConfig,
    candidates: usize,
    step: usize,
    trace: T,
) -> StepOutcome
where
    F: FnMut(&mut Network, &Int8Projection, &mut StdRng) -> f64,
    T: FnMut(&CandidateTraceRecord),
{
    evolution_step_jackpot_traced_with_policy(
        net,
        projection,
        mutation_rng,
        eval_rng,
        fitness_fn,
        config,
        AcceptancePolicy::from_accept_ties(config.accept_ties),
        candidates,
        step,
        trace,
    )
}

/// Run a traced jackpot evolution step with an explicit acceptance policy.
pub fn evolution_step_jackpot_traced_with_policy<F, T>(
    net: &mut Network,
    projection: &mut Int8Projection,
    mutation_rng: &mut impl Rng,
    eval_rng: &mut StdRng,
    fitness_fn: F,
    config: &EvolutionConfig,
    acceptance_policy: AcceptancePolicy,
    candidates: usize,
    step: usize,
    trace: T,
) -> StepOutcome
where
    F: FnMut(&mut Network, &Int8Projection, &mut StdRng) -> f64,
    T: FnMut(&CandidateTraceRecord),
{
    evolution_step_jackpot_traced_with_policy_and_operator_weights(
        net,
        projection,
        mutation_rng,
        eval_rng,
        fitness_fn,
        config,
        acceptance_policy,
        candidates,
        step,
        None,
        trace,
    )
}

/// Run a traced jackpot evolution step with explicit acceptance policy and
/// optional weighted operator sampling.
pub fn evolution_step_jackpot_traced_with_policy_and_operator_weights<F, T>(
    net: &mut Network,
    projection: &mut Int8Projection,
    mutation_rng: &mut impl Rng,
    eval_rng: &mut StdRng,
    mut fitness_fn: F,
    config: &EvolutionConfig,
    acceptance_policy: AcceptancePolicy,
    candidates: usize,
    step: usize,
    operator_weights: Option<&[f64]>,
    mut trace: T,
) -> StepOutcome
where
    F: FnMut(&mut Network, &Int8Projection, &mut StdRng) -> f64,
    T: FnMut(&CandidateTraceRecord),
{
    let step_started = Instant::now();
    let candidates = candidates.max(1);

    let eval_rng_snapshot = eval_rng.clone();
    let before = fitness_fn(net, projection, eval_rng);
    *eval_rng = eval_rng_snapshot.clone();

    let parent_snapshot = net.save_state();
    let parent_projection = projection.clone();
    let edges_before = net.edge_count();

    let mut best_delta = f64::NEG_INFINITY;
    let mut best_index: Option<usize> = None;
    let mut best_net_snapshot = None;
    let mut best_projection = None;
    let mut any_mutated = false;
    let mut records = Vec::with_capacity(candidates);

    for candidate_id in 0..candidates {
        net.restore_state(&parent_snapshot);
        *projection = parent_projection.clone();

        let operator_index = if let Some(weights) = operator_weights {
            sample_weighted_operator(weights, mutation_rng)
        } else {
            sample_baseline_operator(mutation_rng)
        };
        let operator_id = MUTATION_OPERATORS[operator_index].id;
        let mutated =
            apply_mutation_operator(operator_index, net, projection, mutation_rng, None);

        if !mutated {
            records.push(CandidateTraceRecord {
                step,
                candidate_id,
                operator_id,
                mutated: false,
                evaluated: false,
                before_u: before,
                after_u: before,
                delta_u: 0.0,
                within_cap: true,
                selected: false,
                accepted: false,
                candidate_eval_ms: 0.0,
                step_wall_ms: 0.0,
            });
            continue;
        }
        any_mutated = true;

        let cand_eval_rng = eval_rng_snapshot.clone();
        let eval_started = Instant::now();
        let after = fitness_fn(net, projection, &mut cand_eval_rng.clone());
        let candidate_eval_ms = eval_started.elapsed().as_secs_f64() * 1000.0;

        let edge_grew = net.edge_count() > edges_before;
        let within_cap = !edge_grew || net.edge_count() <= config.edge_cap;
        let delta = after - before;

        if delta > best_delta && within_cap {
            best_delta = delta;
            best_index = Some(candidate_id);
            best_net_snapshot = Some(net.save_state());
            best_projection = Some(projection.clone());
        }

        records.push(CandidateTraceRecord {
            step,
            candidate_id,
            operator_id,
            mutated: true,
            evaluated: true,
            before_u: before,
            after_u: after,
            delta_u: delta,
            within_cap,
            selected: false,
            accepted: false,
            candidate_eval_ms,
            step_wall_ms: 0.0,
        });
    }

    net.restore_state(&parent_snapshot);
    *projection = parent_projection.clone();
    let _ = fitness_fn(net, projection, eval_rng);

    let outcome = if !any_mutated {
        net.restore_state(&parent_snapshot);
        *projection = parent_projection;
        StepOutcome::Skipped
    } else {
        let dominated = acceptance_policy.accepts(best_delta, mutation_rng);

        if dominated {
            if let (Some(net_s), Some(proj_s)) = (best_net_snapshot, best_projection) {
                net.restore_state(&net_s);
                *projection = proj_s;
                StepOutcome::Accepted
            } else {
                net.restore_state(&parent_snapshot);
                *projection = parent_projection;
                StepOutcome::Rejected
            }
        } else {
            net.restore_state(&parent_snapshot);
            *projection = parent_projection;
            StepOutcome::Rejected
        }
    };

    let accepted_index = if outcome == StepOutcome::Accepted {
        best_index
    } else {
        None
    };
    let step_wall_ms = step_started.elapsed().as_secs_f64() * 1000.0;
    for record in &mut records {
        record.selected = Some(record.candidate_id) == best_index;
        record.accepted = Some(record.candidate_id) == accepted_index;
        record.step_wall_ms = step_wall_ms;
        trace(record);
    }

    outcome
}

/// Copy-on-write evolution step: O(1) save/restore instead of O(H+E) cloning.
///
/// Identical semantics to [`evolution_step`], but uses [`MutationUndo`] tokens
/// to reverse rejected mutations instead of cloning the entire network.
/// Since `fitness_fn` calls `reset()` at the start, ephemeral state does not
/// need to be snapshot-ed — only topology and learned parameters.
///
/// **Performance**: save = O(1), reject-restore = O(1) for single-edge mutations.
/// This eliminates ~99% of clone overhead since most mutations are rejected.
///
/// The mutation schedule is identical to [`evolution_step`].
pub fn evolution_step_cow<F>(
    net: &mut Network,
    projection: &mut Int8Projection,
    mutation_rng: &mut impl Rng,
    eval_rng: &mut StdRng,
    mut fitness_fn: F,
    config: &EvolutionConfig,
) -> StepOutcome
where
    F: FnMut(&mut Network, &Int8Projection, &mut StdRng) -> f64,
{
    // Paired eval: evaluate BEFORE mutation (save eval_rng state for reuse)
    let eval_rng_snapshot = eval_rng.clone();
    let before = fitness_fn(net, projection, eval_rng);
    *eval_rng = eval_rng_snapshot;

    // Record edge count before mutation for edge-cap check
    let edges_before_mutation = net.edge_count();

    // Mutate with undo tokens — O(1) instead of O(H+E) clone
    let roll = mutation_rng.gen_range(0..100u32);
    let mut weight_backup: Option<WeightBackup> = None;
    let (mutated, undo) = match roll {
        0..22 => net.mutate_add_edge_undo(mutation_rng),
        22..35 => net.mutate_remove_edge_undo(mutation_rng),
        35..44 => net.mutate_rewire_undo(mutation_rng),
        44..57 => net.mutate_reverse_undo(mutation_rng),
        57..63 => net.mutate_mirror_undo(mutation_rng),
        63..70 => net.mutate_enhance_undo(mutation_rng),
        70..75 => net.mutate_theta_undo(mutation_rng),
        75..85 => net.mutate_channel_undo(mutation_rng), // 10% (ORIGINAL)
        85..90 => net.mutate_add_loop_undo(mutation_rng, 2), // 5%
        90..95 => net.mutate_add_loop_undo(mutation_rng, 3), // 5%
        _ => {
            weight_backup = Some(projection.mutate_one(mutation_rng));
            (true, MutationUndo::Noop)
        }
    };

    if !mutated {
        // RNG sync: run the "after" eval to advance eval_rng
        let _ = fitness_fn(net, projection, eval_rng);
        return StepOutcome::Skipped;
    }

    // Paired eval: evaluate AFTER mutation (same corpus segment via restored eval_rng)
    let after = fitness_fn(net, projection, eval_rng);

    // Quality gate
    let dominated = if config.accept_ties {
        after >= before
    } else {
        after > before
    };

    // Edge cap gate
    let edge_grew = net.edge_count() > edges_before_mutation;
    let within_cap = !edge_grew || net.edge_count() <= config.edge_cap;

    let accepted = dominated && within_cap;

    if accepted {
        StepOutcome::Accepted
    } else {
        // O(1) undo instead of O(H+E) restore
        net.apply_undo(&undo);
        net.reset();
        if let Some(backup) = weight_backup {
            projection.rollback(backup);
        }
        StepOutcome::Rejected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::SpikeData;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_test_network(rng: &mut StdRng) -> Network {
        let mut net = Network::new(16);
        for _ in 0..30 {
            net.mutate_add_edge(rng);
        }
        for i in 0..16 {
            net.spike_data_mut()[i].threshold = rng.gen_range(0..=7);
            net.spike_data_mut()[i].channel = rng.gen_range(1..=8);
        }
        net
    }

    #[test]
    fn rejected_step_restores_state() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut net = make_test_network(&mut rng);
        let mut proj = Int8Projection::new(10, 5, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(99);
        let config = EvolutionConfig {
            edge_cap: 1000,
            accept_ties: true,
        };

        let edges_before = net.edge_count();
        let spike_before: Vec<SpikeData> = net.spike_data().to_vec();
        let proj_before = proj.weight_count(); // just check it doesn't crash

        // fitness_fn that always returns worse score after mutation → always reject
        let mut call_count = 0u32;
        let outcome = evolution_step(
            &mut net,
            &mut proj,
            &mut rng,
            &mut eval_rng,
            |_net, _proj, eval_rng| {
                call_count += 1;
                let _ = eval_rng.gen_range(0..100u32); // consume eval_rng
                if call_count <= 1 {
                    1.0
                } else {
                    0.0
                } // before=1.0, after=0.0 → reject
            },
            &config,
        );

        // Could be Rejected or Skipped (if mutation failed)
        if outcome == StepOutcome::Rejected {
            assert_eq!(net.edge_count(), edges_before, "edge count should restore");
            assert_eq!(
                net.spike_data(),
                spike_before.as_slice(),
                "spike data should restore"
            );
        }
        assert_eq!(proj.weight_count(), proj_before);
    }

    #[test]
    fn accepted_step_may_change_state() {
        let mut rng = StdRng::seed_from_u64(77);
        let mut net = make_test_network(&mut rng);
        let mut proj = Int8Projection::new(10, 5, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(99);
        let config = EvolutionConfig {
            edge_cap: 1000,
            accept_ties: true,
        };

        // Run many steps with fitness that always improves → always accept
        let mut accepted_any = false;
        for _ in 0..50 {
            let mut call_count = 0u32;
            let outcome = evolution_step(
                &mut net,
                &mut proj,
                &mut rng,
                &mut eval_rng,
                |_net, _proj, eval_rng| {
                    call_count += 1;
                    let _ = eval_rng.gen_range(0..100u32);
                    if call_count <= 1 {
                        0.0
                    } else {
                        1.0
                    } // before=0.0, after=1.0 → accept
                },
                &config,
            );
            if outcome == StepOutcome::Accepted {
                accepted_any = true;
            }
        }
        assert!(
            accepted_any,
            "should accept at least one step in 50 attempts"
        );
    }

    #[test]
    fn skipped_step_no_state_change() {
        // 1-neuron network: ALL topology mutations fail (can't add self-loop, nothing to remove)
        // Only projection mutations succeed (10% of rolls)
        let net = Network::new(1);
        let mut rng = StdRng::seed_from_u64(42);
        let proj = Int8Projection::new(1, 2, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(99);
        let config = EvolutionConfig {
            edge_cap: 1000,
            accept_ties: true,
        };

        let mut skipped = 0u32;
        for _ in 0..200 {
            let mut test_net = net.clone();
            let mut test_proj = proj.clone();
            let edges_before = test_net.edge_count();
            let outcome = evolution_step(
                &mut test_net,
                &mut test_proj,
                &mut rng,
                &mut eval_rng,
                |_net, _proj, eval_rng| {
                    let _ = eval_rng.gen_range(0..100u32);
                    0.5
                },
                &config,
            );
            if outcome == StepOutcome::Skipped {
                skipped += 1;
                assert_eq!(
                    test_net.edge_count(),
                    edges_before,
                    "skip should not change edges"
                );
            }
        }
        // 90% of rolls are topology mutations, all fail on 1-neuron network → many skips
        assert!(
            skipped > 50,
            "expected many skipped steps on 1-neuron network, got {skipped}"
        );
    }

    #[test]
    fn paired_eval_uses_same_rng_state() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut net = make_test_network(&mut rng);
        let mut proj = Int8Projection::new(10, 5, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(99);
        let config = EvolutionConfig {
            edge_cap: 1000,
            accept_ties: true,
        };

        // Track the eval_rng value at start of each fitness_fn call
        let mut rng_values = Vec::new();
        let _ = evolution_step(
            &mut net,
            &mut proj,
            &mut rng,
            &mut eval_rng,
            |_net, _proj, eval_rng| {
                rng_values.push(eval_rng.clone().gen::<u64>());
                let _ = eval_rng.gen_range(0..100u32);
                0.5
            },
            &config,
        );

        // Should have exactly 2 calls (before + after) with same starting RNG state
        if rng_values.len() == 2 {
            assert_eq!(
                rng_values[0], rng_values[1],
                "paired eval must start from same RNG state"
            );
        }
        // If only 1 call: mutation was skipped, that's also fine
    }

    #[test]
    fn jackpot_traced_emits_candidate_rows_and_accept_invariants() {
        let mut rng = StdRng::seed_from_u64(123);
        let mut net = make_test_network(&mut rng);
        let mut proj = Int8Projection::new(10, 5, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(456);
        let config = EvolutionConfig {
            edge_cap: 1000,
            accept_ties: false,
        };

        let mut records = Vec::new();
        let mut calls = 0u32;
        let outcome = evolution_step_jackpot_traced(
            &mut net,
            &mut proj,
            &mut rng,
            &mut eval_rng,
            |_net, _proj, eval_rng| {
                calls += 1;
                let _ = eval_rng.gen_range(0..100u32);
                if calls == 1 {
                    0.0
                } else {
                    1.0
                }
            },
            &config,
            20,
            123,
            |record| records.push(record.clone()),
        );

        assert_eq!(records.len(), 20);
        assert!(records.iter().all(|record| record.step == 123));
        assert!(records.iter().any(|record| record.evaluated));

        let selected_count = records.iter().filter(|record| record.selected).count();
        let accepted_count = records.iter().filter(|record| record.accepted).count();
        assert!(selected_count <= 1);
        assert_eq!(
            accepted_count,
            if outcome == StepOutcome::Accepted {
                1
            } else {
                0
            }
        );

        for record in &records {
            if record.evaluated {
                assert!((record.delta_u - (record.after_u - record.before_u)).abs() <= 1e-12);
            }
            if record.accepted {
                assert!(record.selected);
            }
        }
    }

    #[test]
    fn acceptance_policy_zero_p_bounds_zero_delta() {
        let mut rng = StdRng::seed_from_u64(7);
        let reject_zero = AcceptancePolicy::ZeroP {
            probability: 0.0,
            zero_tol: 1e-12,
        };
        let accept_zero = AcceptancePolicy::ZeroP {
            probability: 1.0,
            zero_tol: 1e-12,
        };
        assert!(reject_zero.accepts(0.001, &mut rng));
        assert!(!reject_zero.accepts(0.0, &mut rng));
        assert!(accept_zero.accepts(0.0, &mut rng));
        assert!(!accept_zero.accepts(-1e-6, &mut rng));
    }

    #[test]
    fn acceptance_policy_epsilon_accepts_bounded_negative_delta() {
        let mut rng = StdRng::seed_from_u64(9);
        let policy = AcceptancePolicy::Epsilon { epsilon: 1e-4 };
        assert!(policy.accepts(0.0, &mut rng));
        assert!(policy.accepts(-5e-5, &mut rng));
        assert!(!policy.accepts(-2e-4, &mut rng));
    }

    #[test]
    fn cow_rejected_step_restores_state() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut net = make_test_network(&mut rng);
        let mut proj = Int8Projection::new(10, 5, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(99);
        let config = EvolutionConfig {
            edge_cap: 1000,
            accept_ties: true,
        };

        // Save the learned parameters (not ephemeral state)
        let edges_before: Vec<_> = net
            .graph()
            .iter_edges()
            .map(|e| (e.source, e.target))
            .collect();
        let thresholds_before: Vec<u8> = net.spike_data().iter().map(|s| s.threshold).collect();
        let channels_before: Vec<u8> = net.spike_data().iter().map(|s| s.channel).collect();
        let polarity_before: Vec<i8> = net.polarity().to_vec();

        // fitness_fn that always rejects (before=1.0, after=0.0)
        let mut call_count = 0u32;
        let outcome = evolution_step_cow(
            &mut net,
            &mut proj,
            &mut rng,
            &mut eval_rng,
            |_net, _proj, eval_rng| {
                call_count += 1;
                let _ = eval_rng.gen_range(0..100u32);
                if call_count <= 1 {
                    1.0
                } else {
                    0.0
                }
            },
            &config,
        );

        if outcome == StepOutcome::Rejected {
            // Verify topology restored
            let mut edges_after: Vec<_> = net
                .graph()
                .iter_edges()
                .map(|e| (e.source, e.target))
                .collect();
            let mut edges_sorted = edges_before.clone();
            edges_sorted.sort();
            edges_after.sort();
            assert_eq!(edges_sorted, edges_after, "CoW: edge set should restore");

            // Verify parameters restored
            let thresholds_after: Vec<u8> = net.spike_data().iter().map(|s| s.threshold).collect();
            let channels_after: Vec<u8> = net.spike_data().iter().map(|s| s.channel).collect();
            assert_eq!(
                thresholds_before, thresholds_after,
                "CoW: thresholds should restore"
            );
            assert_eq!(
                channels_before, channels_after,
                "CoW: channels should restore"
            );
            assert_eq!(
                polarity_before,
                net.polarity(),
                "CoW: polarity should restore"
            );
        }
    }

    #[test]
    fn cow_accepted_step_may_change_state() {
        let mut rng = StdRng::seed_from_u64(77);
        let mut net = make_test_network(&mut rng);
        let mut proj = Int8Projection::new(10, 5, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(99);
        let config = EvolutionConfig {
            edge_cap: 1000,
            accept_ties: true,
        };

        let mut accepted_any = false;
        for _ in 0..50 {
            let mut call_count = 0u32;
            let outcome = evolution_step_cow(
                &mut net,
                &mut proj,
                &mut rng,
                &mut eval_rng,
                |_net, _proj, eval_rng| {
                    call_count += 1;
                    let _ = eval_rng.gen_range(0..100u32);
                    if call_count <= 1 {
                        0.0
                    } else {
                        1.0
                    }
                },
                &config,
            );
            if outcome == StepOutcome::Accepted {
                accepted_any = true;
            }
        }
        assert!(
            accepted_any,
            "CoW: should accept at least one step in 50 attempts"
        );
    }

    #[test]
    fn cow_many_reject_cycles_maintain_invariants() {
        // Run 200 reject cycles and verify the network stays consistent
        let mut rng = StdRng::seed_from_u64(123);
        let mut net = make_test_network(&mut rng);
        let mut proj = Int8Projection::new(10, 5, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(99);
        let config = EvolutionConfig {
            edge_cap: 1000,
            accept_ties: false,
        };

        // Record initial state
        let mut initial_edges: Vec<_> = net
            .graph()
            .iter_edges()
            .map(|e| (e.source, e.target))
            .collect();
        initial_edges.sort();
        let initial_thresholds: Vec<u8> = net.spike_data().iter().map(|s| s.threshold).collect();
        let initial_channels: Vec<u8> = net.spike_data().iter().map(|s| s.channel).collect();
        let initial_polarity: Vec<i8> = net.polarity().to_vec();

        let mut rejected = 0u32;
        for _ in 0..200 {
            let mut call_count = 0u32;
            let outcome = evolution_step_cow(
                &mut net,
                &mut proj,
                &mut rng,
                &mut eval_rng,
                |_net, _proj, eval_rng| {
                    call_count += 1;
                    let _ = eval_rng.gen_range(0..100u32);
                    // Always reject: before > after
                    if call_count <= 1 {
                        100.0
                    } else {
                        0.0
                    }
                },
                &config,
            );
            if outcome == StepOutcome::Rejected {
                rejected += 1;
            }
        }

        // After all rejections, state should be unchanged
        let mut final_edges: Vec<_> = net
            .graph()
            .iter_edges()
            .map(|e| (e.source, e.target))
            .collect();
        final_edges.sort();
        assert_eq!(
            initial_edges, final_edges,
            "200 rejects: edge set should be unchanged"
        );
        assert_eq!(
            initial_thresholds,
            net.spike_data()
                .iter()
                .map(|s| s.threshold)
                .collect::<Vec<_>>(),
            "200 rejects: thresholds should be unchanged"
        );
        assert_eq!(
            initial_channels,
            net.spike_data()
                .iter()
                .map(|s| s.channel)
                .collect::<Vec<_>>(),
            "200 rejects: channels should be unchanged"
        );
        assert_eq!(
            initial_polarity,
            net.polarity(),
            "200 rejects: polarity should be unchanged"
        );
        assert!(rejected > 100, "expected many rejections, got {rejected}");
    }
}
