# E7Z Low-Bit Canonical RAM Contract Probe Result

## Status

Evidence run complete.

```text
decision = e7z_full_low_bit_pocket_ram_preferred
best_system = int4_pocket_and_ram
deterministic_replay_passed = true
checker_failure_count = 0
```

Artifact root:

```text
target/pilot_wave/e7z_low_bit_canonical_ram_contract_probe/
```

## Evidence Configuration

```text
seeds = 100401..100416
train_rows_per_seed = 320
validation/heldout/OOD/counterfactual/adversarial_rows_per_seed = 144 each
pocket_pretrain_rows_per_seed = 360
pocket_validation_rows_per_seed = 144
pocket_epochs = 36
local_epochs = 24
full_epochs = 36
repair_generations = 14
repair_population = 12
cpu_workers = 16
device = cpu
```

The run wrote heartbeat/progress artifacts throughout the primary evidence run
and deterministic replay:

```text
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
mutation_repair_report.json
deterministic_replay.json
```

## Primary Scores

Mean evaluation metrics:

```text
continuous_direct_write_baseline                 useful=0.543772 acc=0.643772 valid=0.719754 next=0.081343 bits=2304
oracle_write_continuous_reference                useful=0.991765 acc=1.000000 valid=1.000000 next=0.000000 bits=2304
oracle_write_binary_projected                    useful=0.992881 acc=1.000000 valid=1.000000 next=0.000000 bits=72
oracle_write_ternary_projected                   useful=0.992845 acc=1.000000 valid=1.000000 next=0.000000 bits=144
oracle_write_int4_projected                      useful=0.992773 acc=1.000000 valid=1.000000 next=0.000000 bits=288
learned_binary_ram_boundary                      useful=0.527713 acc=0.627713 valid=0.425322 next=0.172276 bits=72
learned_ternary_ram_boundary                     useful=0.537804 acc=0.637804 valid=0.463213 next=0.157391 bits=144
learned_int4_ram_boundary                        useful=0.547352 acc=0.647352 valid=0.717031 next=0.082765 bits=288
learned_binary_ram_boundary_plus_mutation_repair useful=0.530859 acc=0.630859 valid=0.406674 next=0.179674 bits=72
learned_ternary_ram_boundary_plus_mutation_repair useful=0.541276 acc=0.641276 valid=0.460424 next=0.158859 bits=144
learned_int4_ram_boundary_plus_mutation_repair   useful=0.545399 acc=0.645399 valid=0.716103 next=0.083254 bits=288
pure_binary_pocket_and_ram                       useful=0.567379 acc=0.582140 valid=0.273951 next=0.220264 bits=72
pure_ternary_pocket_and_ram                      useful=0.581044 acc=0.603299 valid=0.364213 next=0.186396 bits=144
int4_pocket_and_ram                              useful=0.608701 acc=0.645942 valid=0.716199 next=0.083054 bits=288
mixed_precision_pocket_float_ram_lowbit          useful=0.543663 acc=0.643663 valid=0.615466 next=0.112100 bits=229.5
dense_graph_danger_control                       useful=0.518924 acc=0.618924 valid=0.622864 next=0.276382 bits=2304
```

## Interpretation

E7Z did not show that simply forcing the current learned producer -> RAM ->
consumer boundary into binary or ternary fixes the E7Y/E7X interface problem.
The best external-only learned low-bit RAM boundary was int4, but it only moved
from `0.543772` to `0.547352` usefulness.

The best non-oracle system was the full low-bit pocket plus RAM path:

```text
int4_pocket_and_ram useful = 0.608701
```

That is a real improvement over the continuous baseline, but it closes only
about `14.49%` of the baseline-to-oracle gap. Mutation repair did not help the
best learned boundary in this run:

```text
best_repair_system = learned_int4_ram_boundary_plus_mutation_repair
mutation_repair_gain = -0.001953
```

The key falsification result is the projected-oracle comparison:

```text
oracle_write_binary_projected  useful = 0.992881
oracle_write_ternary_projected useful = 0.992845
oracle_write_int4_projected    useful = 0.992773
```

So low-bit canonical RAM is expressive enough for this proxy if the code is
already the right code. The bottleneck is learning the canonical RAM code and
producer/consumer boundary, not the bit depth itself.

## Decision Rationale

The decision is:

```text
e7z_full_low_bit_pocket_ram_preferred
```

because the best learned low-bit result came from training the pocket and RAM
boundary in the low-bit regime (`int4_pocket_and_ram`), not from projecting only
the external RAM boundary after a continuous pocket.

Dense graph control did not win:

```text
dense_graph_danger_control useful = 0.518924
```

## Replay And Checker

Deterministic replay hash-matched the required artifacts, including:

```text
aggregate_metrics.json
decision.json
low_bit_boundary_report.json
mutation_repair_report.json
projected_oracle_report.json
system_results.json
task_generation_report.json
```

Checker result:

```text
failure_count = 0
```

## Boundary

E7Z is a controlled numeric pocket-router proxy result. It does not make
raw-language, AGI, consciousness, deployed-model, or model-scale claims.

## Recommended Next Step

Run a canonical-code learning probe rather than another pure bit-depth sweep.
The best target is:

```text
E8A_CANONICAL_RAM_CODE_LEARNING_AND_DISTILLATION_PROBE
```

Question:

```text
Can the system learn the oracle-like low-bit RAM code directly, with distillation
or contrastive producer/consumer alignment, without hardcoding the oracle write?
```
