# STABLE_LOOP_PHASE_LOCK_016_DIRECTED_EDGE_PACKET_TRANSPORT Contract

## Summary

016 is a runner-local diagnostic probe for the carrier hypothesis:

```text
node/cell phase mass broadcast causes wrong-phase echo
directed edge packet state may prevent re-broadcast/reentry
```

The completed local phase rule is unchanged:

```text
phase_i + gate_g -> phase_(i+g)
```

The tested change is where the signal lives:

```text
old:
  cell B holds phase mass and broadcasts to neighbors

new:
  edge A->B carries packet
  B applies local gate
  B writes packet to edge B->C
  edge A->B is consumed/aged/cleared
```

No public `instnct-core` API changes. No production, full VRAXION,
language grounding, consciousness, Prismion uniqueness, or physical quantum
behavior claim.

## Files

```text
docs/research/STABLE_LOOP_PHASE_LOCK_016_DIRECTED_EDGE_PACKET_TRANSPORT_CONTRACT.md
instnct-core/examples/phase_lane_directed_edge_packet_transport.rs
docs/research/STABLE_LOOP_PHASE_LOCK_016_DIRECTED_EDGE_PACKET_TRANSPORT_RESULT.md
```

## Arms

```text
NODE_BROADCAST_BASELINE_014
BEST_PUBLIC_COMBO_014
MOMENTUM_LANES_015_BASELINE
EDGE_PACKET_FLOOD
EDGE_PACKET_PUBLIC_GRADIENT
EDGE_PACKET_ORACLE_ROUTE_CORRECT_PHASE_DIAGNOSTIC
EDGE_PACKET_ORACLE_ROUTE_RANDOM_PHASE_DIAGNOSTIC
EDGE_PACKET_CONSUME_PHASE_ONLY
EDGE_PACKET_CONSUME_FULL_EDGE
EDGE_PACKET_NO_REENTRY
EDGE_PACKET_TTL_PATH
EDGE_PACKET_TTL_PATH_PLUS_2
EDGE_PACKET_TTL_2X_PATH
EDGE_PACKET_PLUS_TARGET_SETTLED_READOUT
EDGE_PACKET_PLUS_CELL_LOCAL_NORMALIZATION
EDGE_PACKET_PLUS_PUBLIC_NO_BACKFLOW
RANDOM_RULE_EDGE_PACKET_CONTROL
RANDOM_ROUTE_EDGE_PACKET_CONTROL
```

Oracle routing may use `true_path` and is diagnostic-only. Public arms may use
only the wall mask, source, source phase, target, and per-cell gates.

## Metrics And Outputs

Core metrics:

```text
phase_final_accuracy
long_path_accuracy
family_min_accuracy
same_target_counterfactual_accuracy
gate_shuffle_collapse
target_arrival_rate
wrong_if_arrived_rate
wrong_phase_growth_rate
correct_phase_power
wrong_phase_power
correct_phase_margin
final_minus_best_gap
edge_packet_delivery_rate
edge_packet_drop_rate
edge_packet_reentry_rate
edge_packet_ttl_expiry_rate
edge_packet_consumed_rate
edge_packet_duplicate_rate
active_edge_fraction
packet_fanout_mean
packet_fanout_max
public_route_dead_end_rate
public_route_tie_rate
public_route_wrong_turn_rate
random_rule_edge_packet_accuracy
random_route_edge_packet_accuracy
wall_leak_rate
forbidden_private_field_leak
nonlocal_edge_count
direct_output_leak_rate
```

Required outputs:

```text
queue.json
progress.jsonl
metrics.jsonl
edge_packet_metrics.jsonl
routing_metrics.jsonl
ttl_metrics.jsonl
consume_metrics.jsonl
reentry_metrics.jsonl
family_metrics.jsonl
counterfactual_metrics.jsonl
random_control_metrics.jsonl
locality_audit.jsonl
mechanism_ranking.json
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

The runner appends progress and metrics continuously. There is no black-box
run.

## Decision Rules

Signal gate versus `NODE_BROADCAST_BASELINE_014`:

```text
long_path_accuracy +0.10
family_min_accuracy +0.20
wrong_if_arrived_rate -0.10
```

Positive gate for `EDGE_PACKET_RESCUES_LONG_CHAIN`:

```text
phase_final_accuracy >= 0.95
long_path_accuracy >= 0.95
family_min_accuracy >= 0.85
same_target_counterfactual_accuracy >= 0.85
gate_shuffle_collapse >= 0.50
wrong_if_arrived_rate <= 0.10
final_minus_best_gap <= 0.05
random rule/route controls remain weak
wall/private/nonlocal/direct leaks = 0
```

If oracle routing works while public routing fails, report routing-policy
blockage rather than public transport solved. If random rule or random route
controls pass, report control contamination rather than success.

## Claim Boundary

016 can support that directed edge-state packet transport has signal, or that
node-broadcast recurrence is likely the source of wrong-phase echo. It cannot
support production architecture, full VRAXION, language grounding,
consciousness, Prismion uniqueness, or physical quantum behavior.
