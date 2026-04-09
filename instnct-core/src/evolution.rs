//! Evolution step: paired eval → mutate → accept/reject → rollback.
//!
//! Encapsulates the (1+1) evolutionary strategy pattern that previously
//! contained 3 bugs when implemented manually. The library handles:
//! - Paired evaluation (clone eval_rng, eval before, restore, eval after)
//! - Density-capped acceptance (>= when lean, > when dense)
//! - Correct rollback of both network topology AND projection weights
//! - RNG sync on skipped (failed) mutations

use crate::projection::{Int8Projection, WeightBackup};
use crate::Network;
use rand::rngs::StdRng;
use rand::Rng;

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

/// Apply a single random mutation using the canonical mutation schedule.
///
/// This is the same schedule used internally by [`evolution_step`] and
/// [`evolution_step_jackpot`].  Use this when you need manual control
/// over the evolution loop (e.g. custom fitness or acceptance logic)
/// while keeping the standard operator mix.
///
/// Returns `true` if the mutation was applied, `false` if it failed
/// (e.g. tried to add a duplicate edge).
///
/// ## Mutation schedule (v5.0)
///
/// | Weight | Operator |
/// |-------:|----------|
/// |   22 % | `add_edge` — topology growth |
/// |   13 % | `remove_edge` — topology pruning |
/// |    9 % | `rewire` — move one endpoint of an existing edge |
/// |   13 % | `reverse` — flip edge direction |
/// |    6 % | `mirror` — add reverse copy of an existing edge |
/// |    7 % | `enhance` — connect to high-degree neuron |
/// |    5 % | `theta` — threshold perturbation |
/// |   10 % | `channel` — phase channel perturbation |
/// |    5 % | `add_loop(2)` — bidirectional pair |
/// |    5 % | `add_loop(3)` — triangle circuit |
/// |    5 % | projection weight perturbation |
pub fn apply_mutation(
    net: &mut Network,
    projection: &mut Int8Projection,
    rng: &mut impl Rng,
) -> bool {
    let roll = rng.gen_range(0..100u32);
    match roll {
        0..22 => net.mutate_add_edge(rng),
        22..35 => net.mutate_remove_edge(rng),
        35..44 => net.mutate_rewire(rng),
        44..57 => net.mutate_reverse(rng),
        57..63 => net.mutate_mirror(rng),
        63..70 => net.mutate_enhance(rng),
        70..75 => net.mutate_theta(rng),
        75..85 => net.mutate_channel(rng),
        85..90 => net.mutate_add_loop(rng, 2),
        90..95 => net.mutate_add_loop(rng, 3),
        _ => {
            let _ = projection.mutate_one(rng);
            true
        }
    }
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
/// Mutation schedule (v5.0, matches [`apply_mutation`]):
/// - 22% add_edge, 13% remove_edge, 9% rewire, 13% reverse
/// - 6% mirror, 7% enhance, 5% theta, 10% channel
/// - 5% loop-2, 5% loop-3, 5% projection weight
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

    // Mutate: topology (90%) + loop (5%) + projection weight (5%)
    // Loop mutations create recurrent circuits atomically — critical for sparse networks.
    let roll = mutation_rng.gen_range(0..100u32);
    let mut weight_backup: Option<WeightBackup> = None;
    let mutated = match roll {
        0..22 => net.mutate_add_edge(mutation_rng),     // 22% topology growth
        22..35 => net.mutate_remove_edge(mutation_rng),  // 13% topology pruning
        35..44 => net.mutate_rewire(mutation_rng),       // 9% rewire existing edge
        44..57 => net.mutate_reverse(mutation_rng),      // 13% flip edge direction
        57..63 => net.mutate_mirror(mutation_rng),       // 6% add reverse of existing
        63..70 => net.mutate_enhance(mutation_rng),      // 7% connect to high-degree neuron
        70..75 => net.mutate_theta(mutation_rng),        // 5% threshold perturbation
        75..85 => net.mutate_channel(mutation_rng),      // 10% phase channel perturbation
        85..90 => net.mutate_add_loop(mutation_rng, 2),  // 5% loop-2 (bidirectional pair)
        90..95 => net.mutate_add_loop(mutation_rng, 3),  // 5% loop-3 (triangle circuit)
        _ => {                                           // 5% projection weight perturbation
            weight_backup = Some(projection.mutate_one(mutation_rng));
            true
        }
    };

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

        // Mutate (same 90/5/5 schedule as evolution_step)
        let roll = mutation_rng.gen_range(0..100u32);
        let mut _weight_backup: Option<WeightBackup> = None;
        let mutated = match roll {
            0..22 => net.mutate_add_edge(mutation_rng),
            22..35 => net.mutate_remove_edge(mutation_rng),
            35..44 => net.mutate_rewire(mutation_rng),
            44..57 => net.mutate_reverse(mutation_rng),
            57..63 => net.mutate_mirror(mutation_rng),
            63..70 => net.mutate_enhance(mutation_rng),
            70..75 => net.mutate_theta(mutation_rng),
            75..85 => net.mutate_channel(mutation_rng),
            85..90 => net.mutate_add_loop(mutation_rng, 2),
            90..95 => net.mutate_add_loop(mutation_rng, 3),
            _ => {
                _weight_backup = Some(projection.mutate_one(mutation_rng));
                true
            }
        };

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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_test_network(rng: &mut StdRng) -> Network {
        let mut net = Network::new(16);
        for _ in 0..30 {
            net.mutate_add_edge(rng);
        }
        for i in 0..16 {
            net.threshold_mut()[i] = rng.gen_range(0..=7);
            net.channel_mut()[i] = rng.gen_range(1..=8);
        }
        net
    }

    #[test]
    fn rejected_step_restores_state() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut net = make_test_network(&mut rng);
        let mut proj = Int8Projection::new(10, 5, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(99);
        let config = EvolutionConfig { edge_cap: 1000, accept_ties: true };

        let edges_before = net.edge_count();
        let threshold_before: Vec<u32> = net.threshold().to_vec();
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
                if call_count <= 1 { 1.0 } else { 0.0 } // before=1.0, after=0.0 → reject
            },
            &config,
        );

        // Could be Rejected or Skipped (if mutation failed)
        if outcome == StepOutcome::Rejected {
            assert_eq!(net.edge_count(), edges_before, "edge count should restore");
            assert_eq!(net.threshold(), threshold_before.as_slice(), "threshold should restore");
        }
        assert_eq!(proj.weight_count(), proj_before);
    }

    #[test]
    fn accepted_step_may_change_state() {
        let mut rng = StdRng::seed_from_u64(77);
        let mut net = make_test_network(&mut rng);
        let mut proj = Int8Projection::new(10, 5, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(99);
        let config = EvolutionConfig { edge_cap: 1000, accept_ties: true };

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
                    if call_count <= 1 { 0.0 } else { 1.0 } // before=0.0, after=1.0 → accept
                },
                &config,
            );
            if outcome == StepOutcome::Accepted {
                accepted_any = true;
            }
        }
        assert!(accepted_any, "should accept at least one step in 50 attempts");
    }

    #[test]
    fn skipped_step_no_state_change() {
        // 1-neuron network: ALL topology mutations fail (can't add self-loop, nothing to remove)
        // Only projection mutations succeed (10% of rolls)
        let net = Network::new(1);
        let mut rng = StdRng::seed_from_u64(42);
        let proj = Int8Projection::new(1, 2, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(99);
        let config = EvolutionConfig { edge_cap: 1000, accept_ties: true };

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
                assert_eq!(test_net.edge_count(), edges_before, "skip should not change edges");
            }
        }
        // 90% of rolls are topology mutations, all fail on 1-neuron network → many skips
        assert!(skipped > 50, "expected many skipped steps on 1-neuron network, got {skipped}");
    }

    #[test]
    fn paired_eval_uses_same_rng_state() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut net = make_test_network(&mut rng);
        let mut proj = Int8Projection::new(10, 5, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(99);
        let config = EvolutionConfig { edge_cap: 1000, accept_ties: true };

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
}
