# E7A12 Binary Mutation-Only Viability Audit Result

## Decision

```text
decision = e7a12_progressive_seed_mutation_bridge_viable
deterministic_replay_passed = true
checker_failure_count = 0
```

Artifact root:

```text
target/pilot_wave/e7a12_binary_mutation_only_viability_audit/
```

## Main Result

Binary mutation-only did not solve from scratch.

```text
best_from_scratch = sensitivity_guided_binary_from_scratch_mutation / baseline
eval = 0.7390625
gap_to_qat = 0.20625
solve_passed = false
```

But local mutation around a good binary seed did work.

```text
best_seeded_local = binary_mutation_bits_plus_scale / baseline
eval = 0.9500000
seed_gain = +0.003125
gap_to_qat = -0.0046875
solve_passed = true
```

The progressive-freeze seeded path was the decision driver:

```text
progressive_freeze_seeded_binary_local_mutation
mean_eval = 0.9453125
mean_seed_gain = +0.003125
solve_cases = 2 / 2
```

Random mutation control did not match guided mutation:

```text
random_mutation_control mean_eval = 0.54609375
sensitivity_guided_from_scratch mean_eval = 0.7234375
```

## Interpretation

The result does not support binary mutation-only as a from-scratch discovery engine. The search space is still too large for the tested mutation operators.

It does support a narrower bridge:

```text
QAT / freeze can discover a good binary region.
Mutation-only can locally repair or slightly improve that binary region.
```

This is VRAXION-relevant because the mutation/rollback substrate did real accept/reject work without backprop in the mutation arms:

```text
qat_seeded_binary_local_mutation accepted/rejected:
  baseline: 118 / 862
  interaction_stress: 123 / 857

progressive_freeze_seeded_binary_local_mutation accepted/rejected:
  baseline: 133 / 847
  interaction_stress: 132 / 848

binary_mutation_bits_plus_scale accepted/rejected:
  baseline: 88 / 892
  interaction_stress: 140 / 840
```

## Practical Read

```text
from-scratch binary mutation: no
QAT-seeded local mutation: yes, small repair signal
progressive-freeze seeded mutation: yes, bridge signal
scale-only mutation: useful in one case, not stable enough as main result
bits-plus-scale mutation: strongest single local result
```

Current path:

```text
int4 remains the practical baseline.
binary remains viable as a QAT/freeze-seeded mutation-repair branch.
```

## Next Step

The next experiment should isolate the bridge:

```text
E7A13_PROGRESSIVE_BINARY_SEED_WITHOUT_QAT_AUDIT
```

Question:

```text
Can we create the good binary seed without backprop/QAT,
then let mutation/rollback do the local repair?
```

If no, QAT remains the seed discovery tool. If yes, binary mutation becomes a more native VRAXION path.

## Boundary

This is a controlled binary matrix-core mutation audit. It does not make claims about natural-language reasoning, AGI, consciousness, or model-scale behavior.
