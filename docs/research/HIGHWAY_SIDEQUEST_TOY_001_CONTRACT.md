# HIGHWAY_SIDEQUEST_TOY_001 Contract

## Purpose

Test whether a main-highway + sideprocessor + gated-writeback topology is useful before introducing Prismion-specific update rules.

This is topology-first:

```text
main_state h
  -> sideprocessors read h + current token
  -> sideprocessors propose deltas
  -> gates decide writeback
  -> main_state updates
```

Do not use English parser data. Do not provide intermediate labels during model training. Train from final outcome only.

## Tasks

Use abstract symbolic streams:

```text
A, B, C
anti_A, anti_B, anti_C
reset
mention_A, mention_B, mention_C
quote_anti_A, quote_anti_B, quote_anti_C
actually_A, actually_B, actually_C
instead_A, instead_B, instead_C
create_X, remove_X, restore_X, query_count
noise
```

Required task families:

- cancellation: `A -> A`, `A anti_A -> none`, `A anti_A B -> B`
- scope: `anti_A A -> none`, `anti_A reset A -> A`
- refocus: `A actually_B -> B`, `A instead_B -> B`
- mention/no-op: `mention_A -> none`, `mention_A B -> B`, `quote_anti_A A -> A`
- entity toy: `create_X remove_X restore_X query_count -> count`

Training labels are only final answers:

```text
NONE, A, B, C, COUNT0, COUNT1, COUNT2, COUNT3, COUNT4
```

## Arms

Run these arms:

```text
MLP_STATIC
SimpleRNN
GRU
LSTM
HIGHWAY_ONLY_SIDEPROCESSORS
HIGHWAY_SPARSE_SIDE_LINKS
HIGHWAY_DENSE_SIDE_LINKS
HIGHWAY_PRISMION_SIDEPROCESSORS
```

The Prismion arm is diagnostic in this run. Main topology verdicts compare standard RNNs against non-Prismion highway arms.

## Sweeps

Configurable variables:

```text
--widths 8,16,32
--side-counts 2,4,8
--depths 1,2,4
```

Keep parameter counts reported for every arm/config. Exact equality is not required for v1, but large differences must be visible in the result.

## Metrics

Report per arm/config/seed:

```text
final_answer_accuracy
heldout_composition_accuracy
length_generalization_accuracy
false_mutation_rate
false_cancellation_rate
cancellation_accuracy
scope_accuracy
refocus_accuracy
mention_noop_accuracy
entity_count_accuracy
seed_stability
parameter_count
epochs_to_threshold
```

After training, freeze the model and run linear probes:

```text
active_symbol_probe_accuracy
blocked_symbol_probe_accuracy
current_focus_probe_accuracy
entity_count_probe_accuracy
mutation_probe_accuracy
probe_mean_accuracy
```

For highway arms, run ablations:

```text
zero each sideprocessor
force all gates closed
force all gates open
randomize side links
```

Report:

```text
gate_mean
gate_std
side_ablation_max_drop
side_ablation_mean_drop
zero_gate_accuracy
open_gate_accuracy
randomized_link_accuracy
specialization_score
```

## Verdicts

```text
HIGHWAY_TOPOLOGY_POSITIVE
  highway-only or sparse highway beats GRU/LSTM by >= 0.10 on heldout composition and length generalization.

SPARSE_COORDINATION_POSITIVE
  sparse side links beat highway-only by >= 0.05 without dense winning by brute force.

DENSE_MONOLITH_WARNING
  dense wins but probes/ablations show weak specialization.

PRISMION_UPDATE_POSITIVE
  Prismion sideprocessors beat matched MLP sideprocessors, especially on cancellation/refocus/scope.

STANDARD_RNN_SUFFICIENT
  GRU/LSTM match or beat highway variants.

TASK_TOO_EASY
  static MLP solves heldout.

TASK_TOO_HARD
  all models fail.
```

## Run Hygiene

No black-box runs. Required outputs:

```text
queue.json
progress.jsonl
metrics.jsonl
summary.json
report.md
topology_curve.json
probe_curve.json
ablation_curve.json
examples_sample.jsonl
contract_snapshot.md
job_progress/*.jsonl
```

Use `--jobs auto85`; each worker sets one torch thread.

Do not commit raw `target/` outputs.
