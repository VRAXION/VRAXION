# E7A7 Low-Bit Repair Operator Audit Contract

## Purpose

E7A7 audits the E7A6 low-bit breakpoint. E7A6 showed that the plain matrix-core is stable through int4, degrades at int3, and breaks harder at ternary/binary while mutation repair gives partial recovery.

The E7A7 question is narrower:

```text
Which matrix-core block is damaged most by low-bit quantization,
and is partial recovery caused by a weak mutation repair operator
or by an information limit in ternary/binary representations?
```

## Scope

This probe reuses the E7A6 plain matrix-core and E7A3 task generation. It does not modify E7A6.

Audit levels:

```text
int3
ternary
binary
```

Blocks:

```text
input_projection = win
recurrent_state  = state
carry_gate       = carry_raw
state_bias       = bstate
output_head      = wout + bout
```

## Systems

```text
float32_matrix_core
low_bit_no_repair
block_only_low_bit
block_restored_to_int8
full_mutation_repair
targeted_block_mutation_repair
sensitive_pair_mutation_repair
quantization_aware_training
```

## Required Audits

1. Block-only low-bit damage:
   quantize one block to the target low-bit level while keeping all other blocks at int8.

2. Block restore gain:
   start from full low-bit and restore one block to int8.

3. Repair operator comparison:
   compare full repair, single-block targeted repair, and top-two sensitive block repair.

4. QAT control:
   train/fine-tune with straight-through fake quantization, then quantize and evaluate.

5. Deterministic replay:
   required artifact hashes must match between primary and replay.

## Metrics

Required metrics:

```text
eval accuracy
heldout/OOD/counterfactual/adversarial accuracy
drop vs float32
gain vs full low-bit
top restore block
full repair accuracy
best targeted repair accuracy
best pair repair accuracy
QAT accuracy
mutation attempts
accepted/rejected mutations
rollback count
parameter diff/hash
deterministic replay hash match
```

## Decisions

Allowed decisions:

```text
e7a7_sensitive_block_repair_sufficient
e7a7_output_or_state_bottleneck_identified
e7a7_qat_preferred_over_post_repair
e7a7_repair_operator_bottleneck_detected
e7a7_low_bit_information_limit_detected
e7a7_low_bit_breakpoint_audit_complete
e7a7_invalid_artifact_detected
```

Decision hints:

```text
targeted repair sufficient:
  targeted repair is within 0.005 of full repair
  and improves by at least 0.02 over full low-bit

state/output bottleneck:
  recurrent_state or output_head has restore gain >= 0.03

QAT preferred:
  QAT beats best post-quantization repair by >= 0.02

repair operator bottleneck:
  QAT reaches stable low-bit but repair does not

information limit:
  neither repair nor QAT restores stable ternary/binary performance
```

## Artifact Contract

Required artifacts:

```text
e7a7_backend_manifest.json
e7a7_task_generation_report.json
e7a7_block_damage_report.json
e7a7_block_restore_report.json
e7a7_repair_operator_report.json
e7a7_qat_report.json
e7a7_low_bit_bottleneck_report.json
e7a7_mutation_history.json
e7a7_no_synthetic_metric_audit.json
e7a7_deterministic_replay_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
progress.jsonl
e7a7_row_level_eval_sample_heldout.json
e7a7_row_level_eval_sample_ood.json
e7a7_row_level_eval_sample_counterfactual.json
e7a7_row_level_eval_sample_adversarial.json
```

All long-ish runs must write heartbeat artifacts and progress rows. No run may be treated as a black box.

## Checker Requirements

The checker fails on:

```text
missing required artifact
missing audit level or block
missing row-level samples
missing mutation attempts
rollback mismatch
missing replay hash match
mutation repair using optimizer/backprop
hardcoded improvement flags
forbidden broad claims in report
failure_count != 0
```

## Boundary

E7A7 is a controlled symbolic/numeric low-bit substrate audit. It is not a broad reasoning or model-scale result.
