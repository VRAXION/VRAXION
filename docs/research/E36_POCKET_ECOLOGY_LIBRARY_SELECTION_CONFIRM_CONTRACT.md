# E36 Pocket Ecology Library Selection Confirm Contract

## Summary

E36 tests whether a shared Pocket Library can act as a population-level
selection layer rather than a one-off archive.

Core question:

```text
Can many fresh worlds/runs score reusable Pocket Operators, promote the
mechanism pockets that repeatedly help, and quarantine stale/toxic/passenger
pockets?
```

This is the first controlled "Pocket Ecology" probe:

```text
fresh worlds
-> candidate pockets
-> paired ablation / random import / toxic control
-> pocket_value
-> lifecycle status
-> promoted stable library subset
```

## Scope

E36 reuses the E35 transfer setting:

```text
ProtocolFramingIngressPocket
TargetCodebookAdapter
stable target families
bit-slip target families
wrong-codebook toxic control
dirty scratch decoder
```

It does not try to solve bit-slip stream reassembly. It asks whether the
library can correctly keep the clean transferable mechanism pocket while not
mistaking a dirty monolithic decoder for a stable reusable library skill.

## Systems

```text
no_library_scratch
random_library_import
unfiltered_library_import
evaluated_library_import
evaluated_library_plus_adapter
wrong_toxic_pocket_control
oracle_invalid_control
```

## Pocket Candidates

```text
protocol_framing_ingress_v001
protocol_framing_no_adapter
dirty_start_only_decoder
wrong_rotated_codebook_pocket
dormant_unused_pocket
```

Lifecycle statuses:

```text
candidate
staging
stable
core
deprecated
banned
```

## Metrics

```text
target_world_success
stable_target_success
bitslip_target_success
wrong_feature_write_rate
false_frame_commit_rate
paired_utility_delta
ablation_drop
negative_delta_rate
useful_activation_rate
activation_count
activation_coverage
pocket_value
promoted_count
banned_count
deprecated_count
deterministic replay
checker failure count
sample-only checker
```

## Decision Labels

```text
e36_pocket_ecology_selection_confirmed
e36_pocket_ecology_selection_partial
e36_pocket_ecology_negative_transfer
e36_no_ecology_advantage_detected
e36_invalid_artifact_or_oracle_detected
```

## Positive Or Partial Signal

Positive or partial E36 requires:

```text
evaluated_library_plus_adapter beats random_library_import
ProtocolFramingIngressPocket is promoted stable/core
wrong_rotated_codebook_pocket is banned
dormant_unused_pocket is deprecated
dirty_start_only_decoder is not promoted stable/core
wrong-feature writes remain low for evaluated import
deterministic replay passes
target checker and sample-only checker pass
```

Full confirmation additionally requires that the evaluated library matches or
beats scratch on target success and no longer has the E35 bit-slip bottleneck.

## Hard Requirements

```text
real row-level eval
paired ablation against no-library scratch rows
random import control
unfiltered import control
toxic pocket control
AFK/dormant pocket control
progress.jsonl and hardware_heartbeat.jsonl
deterministic replay
sample-only checker
no gradient descent
no optimizer/backprop
no AGI/consciousness/model-scale claims
```

Boundary: E36 is a controlled Pocket Ecology selection probe over symbolic
binary-ingress worlds. It does not prove raw language reasoning, AGI,
consciousness, deployed-model behavior, or model-scale behavior.
