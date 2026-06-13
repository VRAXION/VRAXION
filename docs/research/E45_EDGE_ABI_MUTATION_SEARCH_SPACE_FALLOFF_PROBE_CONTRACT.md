# E45 Edge ABI Mutation Search Space Falloff Probe Contract

Milestone:

```text
E45_EDGE_ABI_MUTATION_SEARCH_SPACE_FALLOFF_PROBE
```

## Purpose

E44D found a stress-safe 12-bit active Proposal payload profile and suggested
a 16-bit physical fast lane. E45 asks a different question:

```text
How wide can an Edge ABI get before mutation learning becomes inefficient?
```

The probe freezes two nodes and mutates only the connection contract:

```text
Frozen producer node
-> mutable Edge ABI decoder / bit-position contract
-> Frozen consumer node
```

This is a controlled symbolic/numeric Edge ABI probe. It does not claim raw
language reasoning, AGI, consciousness, deployed behavior, or model scale.

## Systems

```text
structured_w16_i32_reference
structured_w64_i256_reference
structured_w128_i1024_reference
anonymous_w8_i32
anonymous_w12_i32
anonymous_w16_i32
anonymous_w24_i32
anonymous_w32_i32
anonymous_w64_i32
anonymous_w16_i256
anonymous_w32_i256
anonymous_w64_i256
anonymous_w96_i256
anonymous_w128_i256
anonymous_w64_i1024
anonymous_w128_i1024
random_w16_i32_control
```

Structured references use known layout and should pass. Anonymous systems must
learn ordered active bit positions via mutation, accept/reject, and rollback.

## Metrics

```text
heldout_success
OOD success
counterfactual success
adversarial success
wrong_commit_rate
false_ask_rate
bus_width
intent_count
data_bits
ordered_search_space_log10
attempts_to_95
attempts_to_99
accepted_rate
accepted/rejected/rollback mutation counts
last_improvement_generation
plateau_tail_generations
learning_curve_rows
deterministic replay
checker failure count
```

Final score alone is not sufficient. Learning falloff may be visible as
attempts-to-pass explosion, accepted-rate collapse, or long plateau tail even
when final accuracy eventually passes.

## Required Artifacts

```text
backend_manifest.json
search_space_report.json
learning_dynamics_report.json
final_candidates.json
system_results.json
row_level_results.jsonl
learning_curve.jsonl
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
progress.jsonl
hardware_heartbeat.jsonl
partial_aggregate_snapshot.json
results_table.md
report.md
```

The sample pack under
`docs/research/artifact_samples/e45_edge_abi_mutation_search_space_falloff_probe/`
must pass sample-only validation.

## Decisions

```text
e45_anonymous_wide_bus_learning_falloff_detected
e45_32bit_extended_lane_still_mutation_friendly
e45_64bit_anonymous_lane_mutation_friendly
e45_connection_needs_structured_layout
e45_invalid_artifact_detected
```

The expected useful boundary is likely:

```text
16 bit: safe fast lane
32 bit: possible extended lane if structured/masked
64 bit: should be framed/structured, not anonymous mutable fast-lane chaos
```
