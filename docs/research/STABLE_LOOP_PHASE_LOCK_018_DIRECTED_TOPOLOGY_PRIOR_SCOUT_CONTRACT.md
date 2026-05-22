# STABLE_LOOP_PHASE_LOCK_018_DIRECTED_TOPOLOGY_PRIOR_SCOUT Contract

## Summary

018 tests the directed-topology hypothesis left open by 016/017:

```text
012-017 used spatial grid broadcast or edge packets without a learned/mutated
route graph.

In the real mutation graph, edges are directed. If mutation can create the right
A -> B -> C route without B -> A reverse edges, echo may disappear.
```

Core question:

```text
Is the blocker really node/grid bidirectional recurrence, and does an explicitly
directed topology prior expose a clean phase-transport route?
```

This is a runner-local scout. It does not claim production architecture, full
VRAXION, language grounding, consciousness, FlyWire validation, Prismion
uniqueness, or biological equivalence.

## Inputs

The probe uses the 011-017 completed local phase rule:

```text
phase_i + gate_g -> phase_(i+g)
```

It does not require a full FlyWire download. It uses prior repo evidence that
FlyWire/raw sampled topology was not the key winner, while hub/degree topology
had task-specific signal.

External FlyWire data may be used later, but this scout only tests the topology
mechanism:

```text
directed route graph vs bidirectional grid echo
```

## Required Files

```text
docs/research/STABLE_LOOP_PHASE_LOCK_018_DIRECTED_TOPOLOGY_PRIOR_SCOUT_CONTRACT.md
instnct-core/examples/phase_lane_directed_topology_prior_scout.rs
docs/research/STABLE_LOOP_PHASE_LOCK_018_DIRECTED_TOPOLOGY_PRIOR_SCOUT_RESULT.md
```

No public `instnct-core` API changes.

## Required Arms

```text
BIDIRECTIONAL_GRID_BASELINE
TRUE_PATH_DIRECTED_ROUTE_DIAGNOSTIC
TRUE_PATH_PLUS_REVERSE_ABLATION
PUBLIC_GRADIENT_DAG
PUBLIC_MONOTONE_XY_ROUTE
RANDOM_SAME_COUNT_DIRECTED
HUB_RICH_DIRECTED_PRIOR
DEGREE_PRESERVING_HUB_RANDOM
RECIPROCAL_EDGE_PRIOR
DIRECTION_SHUFFLE_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

Diagnostic-only:

```text
TRUE_PATH_DIRECTED_ROUTE_DIAGNOSTIC
TRUE_PATH_PLUS_REVERSE_ABLATION
```

These may use the private true path. They cannot support a public/deployable
claim, but they can answer whether a correct directed route graph would remove
the echo failure.

## Metrics

```text
phase_final_accuracy
long_path_accuracy
family_min_accuracy
correct_target_lane_probability_mean
best_tick_accuracy
target_arrival_rate
wrong_if_arrived_rate
gate_shuffle_collapse
same_target_counterfactual_accuracy

directed_edge_count
reciprocal_edge_fraction
backflow_edge_fraction
active_echo_power
target_wrong_power
target_correct_power
final_minus_best_gap

random_phase_rule_accuracy
direction_shuffle_accuracy
delta_vs_bidirectional_accuracy
delta_vs_bidirectional_wrong_if_arrived

forbidden_private_field_leak
nonlocal_edge_count
direct_output_leak_rate
```

## Decision Rules

If true-path directed routing passes but bidirectional grid fails:

```text
DIRECTED_ROUTE_ELIMINATES_ECHO_DIAGNOSTIC
GRID_BIDIRECTIONAL_ECHO_IS_BLOCKER
```

If adding reverse edges to true path causes collapse:

```text
REVERSE_EDGES_REINTRODUCE_ECHO
```

If public gradient or monotone route passes:

```text
PUBLIC_DIRECTED_TOPOLOGY_HAS_SIGNAL
```

If only true-path diagnostic passes:

```text
DIRECTED_TOPOLOGY_WORKS_ROUTING_POLICY_BLOCKED
```

If random same-count directed graph or random phase rule passes:

```text
CONTROL_CONTAMINATION
```

If hub/degree priors help but do not pass:

```text
HUB_DEGREE_PRIOR_PARTIAL_SIGNAL
```

If all directed priors fail:

```text
NO_DIRECTED_TOPOLOGY_SIGNAL
```

## Verdicts

```text
DIRECTED_ROUTE_ELIMINATES_ECHO_DIAGNOSTIC
GRID_BIDIRECTIONAL_ECHO_IS_BLOCKER
REVERSE_EDGES_REINTRODUCE_ECHO
DIRECTED_ROUTE_HAS_CLEAN_ARRIVAL_DIAGNOSTIC
FINAL_READOUT_TIMING_LIMIT
PUBLIC_DIRECTED_TOPOLOGY_HAS_SIGNAL
DIRECTED_TOPOLOGY_WORKS_ROUTING_POLICY_BLOCKED
HUB_DEGREE_PRIOR_PARTIAL_SIGNAL
FLYWIRE_EXACT_WIRING_NOT_REQUIRED
RANDOM_DIRECTED_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
CONTROL_CONTAMINATION
NO_DIRECTED_TOPOLOGY_SIGNAL
PRODUCTION_API_NOT_READY
```

## Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
topology_metrics.jsonl
echo_metrics.jsonl
family_metrics.jsonl
control_metrics.jsonl
locality_audit.jsonl
mechanism_ranking.json
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

No-black-box rule:

```text
append progress at heartbeat
append metrics after every arm/family/path/tick block
refresh summary.json and report.md on heartbeat
do not commit target/ outputs
```

## Quick Run

```powershell
cargo run -p instnct-core --example phase_lane_directed_topology_prior_scout --release -- ^
  --out target/pilot_wave/stable_loop_phase_lock_018_directed_topology_prior_scout/quick ^
  --seeds 2026 ^
  --eval-examples 512 ^
  --widths 8,12 ^
  --path-lengths 4,8,16,24 ^
  --ticks-list 8,16,24,32 ^
  --heartbeat-sec 15
```

## Claim Boundary

018 can support:

```text
directed route topology removes echo in diagnostic setting
reverse edges reintroduce echo
public routing/topology prior is or is not the blocker
```

018 cannot support:

```text
FlyWire proves VRAXION
exact biological wiring is needed
production architecture
full VRAXION
consciousness
language grounding
physical quantum behavior
```
