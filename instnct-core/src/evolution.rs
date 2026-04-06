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
/// Mutation schedule (hardcoded, matching proven overnight config):
/// - 25% add_edge, 15% remove_edge, 10% rewire, 15% reverse
/// - 7% mirror, 8% enhance, 5% theta, 5% channel, 10% projection weight
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

    // Mutate: 8-op topology schedule (90%) + projection weight (10%)
    let roll = mutation_rng.gen_range(0..100u32);
    let mut weight_backup: Option<WeightBackup> = None;
    let mutated = match roll {
        0..25 => net.mutate_add_edge(mutation_rng),     // 25% topology growth
        25..40 => net.mutate_remove_edge(mutation_rng),  // 15% topology pruning
        40..50 => net.mutate_rewire(mutation_rng),       // 10% rewire existing edge
        50..65 => net.mutate_reverse(mutation_rng),      // 15% flip edge direction
        65..72 => net.mutate_mirror(mutation_rng),       // 7% add reverse of existing
        72..80 => net.mutate_enhance(mutation_rng),      // 8% connect to high-degree neuron
        80..85 => net.mutate_theta(mutation_rng),        // 5% threshold perturbation
        85..90 => net.mutate_channel(mutation_rng),      // 5% phase channel perturbation
        _ => {                                           // 10% projection weight perturbation
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
