# E46 Progressive Single Wire Edge ABI Growth Probe Contract

Milestone:

```text
E46_PROGRESSIVE_SINGLE_WIRE_EDGE_ABI_GROWTH_PROBE
```

## Purpose

E45 showed that anonymous bus width is not free: 64-bit / 256 intent remained
learnable, but wider or higher-intent anonymous buses fell off. E46 tests the
natural repair:

```text
start narrow
add one wire at a time only when capacity requires it
keep old learned wires stable
```

The probe freezes both endpoint nodes and mutates only the Edge ABI decoder and
growth policy.

## Systems

```text
fixed_w5_i256_too_narrow_control
fixed_w8_i256_direct
fixed_w16_i256_direct
progressive_plus1_freeze_old
progressive_plus1_no_freeze
progressive_block_plus4
structured_oracle_progressive_reference
random_growth_control
```

## Metrics

```text
heldout_success
OOD success
adversarial success
old_intent_success
old_intent_regression
growth_events
final_width
active_bits
attempts_to_95
accepted/rejected/rollback mutation counts
accepted_rate
wrong_commit_rate
false_ask_rate
deterministic replay
checker failure count
```

## Required Artifacts

```text
backend_manifest.json
growth_dynamics_report.json
growth_event_report.json
system_results.json
row_level_results.jsonl
growth_curve.jsonl
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
`docs/research/artifact_samples/e46_progressive_single_wire_edge_abi_growth_probe/`
must pass sample-only validation.

## Decisions

```text
e46_single_wire_growth_positive
e46_block_growth_preferred
e46_fixed_wide_bus_sufficient
e46_growth_causes_regression
e46_single_wire_growth_not_needed
e46_invalid_artifact_detected
```

Positive single-wire growth requires the progressive +1 system to reach the
final 256-intent target, keep old-intent regression <= 0.01, and pass replay
and checker.
