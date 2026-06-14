# E86 LocalGolden Seeded Curriculum Training Campaign

```text
decision = e86_localgolden_seeded_curriculum_training_confirmed
checker_failure_count = 0
seeds = 16
workers = 16
case_count = 170366
```

## Purpose

Start from CALC-SCRIBE v003 LocalGolden as a seed pocket. Run a governed
dataset-backed curriculum campaign. Track whether pocket count grows, plateaus,
or decreases during mutation. Promote only scoped components that improve
mixed-stream behavior without unsafe scope expansion.

## Result

```text
validation_action_min = 1.000000
adversarial_action_min = 1.000000
validation_false_call_max = 0.000000
adversarial_false_call_max = 0.000000
validation_false_commit_max = 0.000000
adversarial_false_commit_max = 0.000000
local_golden_candidate_count = 16 / 16
```

Mutation and lifecycle counters:

```text
accepted_mutations_total = 39
rejected_mutations_total = 409
rollback_count_total = 409
pruned_count_total = 7
quarantined_count_total = 9
```

## Evolution

Pocket count did not grow without bound. It grew early as useful adapters and
guards were found, then plateaued after clean mixed-stream behavior appeared.
Some seeds pruned the long-text scope guard when the discovered policy no longer
needed it.

```text
initial_pocket_count = 1
observed_count_min = 4
observed_count_max = 5
final_pocket_count_mean = 4.5625
final_pocket_count_min = 4
final_pocket_count_max = 5
final_counts = [5, 4, 5, 5, 5, 4, 5, 5, 5, 4, 5, 4, 5, 4, 4, 4]

growth_events = 32
decrease_events = 7
plateau_events = 409
plateau_detected = true
```

The converged component set was usually:

```text
calc_scribe_native_seed
square_trace_adapter
arrow_trace_adapter
invalid_trace_rejector
optional long_text_scope_guard
```

## Interpretation

E86 is the first larger dataset-backed pocket-curriculum campaign in this chain.
It shows that the pocket library can start from a governed LocalGolden seed,
mutate candidate adapter/guard components, reject unsafe overreach, and converge
to a small scoped set.

The useful count behavior was:

```text
start:
  1 seed pocket

early training:
  grows to 4-5 useful components

after clean behavior:
  plateaus
  occasional prune from 5 -> 4 when a guard becomes redundant
```

## Boundary

This is governed pocket-curriculum training.

Not claimed:

```text
open-domain model training
natural-language reasoning
GSM8K solving
Core memory
True Golden promotion
Gemma-level capability
production readiness
```
