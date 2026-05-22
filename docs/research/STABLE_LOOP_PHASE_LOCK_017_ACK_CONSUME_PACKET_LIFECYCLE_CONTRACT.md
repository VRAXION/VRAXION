# STABLE_LOOP_PHASE_LOCK_017_ACK_CONSUME_PACKET_LIFECYCLE Contract

## Summary

017 is a runner-local diagnostic probe for the missing packet lifecycle:

```text
016:
  directed edge-state alone does not rescue long-chain transport

017:
  add explicit packet lifecycle:
    send -> receive_commit -> ack -> consume inbound -> forward once -> dedupe/reject replay
```

Core question:

```text
Does ACK/Consume turn edge packets from noisy carrier state into stable
long-chain phase transport, and does it create a cleaner future mutation/search
signal?
```

This is substrate-before-search. It does not run canonical mutation, change
public `instnct-core` APIs, or claim production architecture, full VRAXION,
language grounding, consciousness, Prismion uniqueness, or physical quantum
behavior.

## Files

```text
docs/research/STABLE_LOOP_PHASE_LOCK_017_ACK_CONSUME_PACKET_LIFECYCLE_CONTRACT.md
instnct-core/examples/phase_lane_ack_consume_packet_lifecycle.rs
docs/research/STABLE_LOOP_PHASE_LOCK_017_ACK_CONSUME_PACKET_LIFECYCLE_RESULT.md
```

## Required Stages

Before chain interpretation, run:

```text
ONE_EDGE_ACK_LIFECYCLE
  covers all 16 phase/gate pairs
  requires receive_commit, ack, consume_after_ack, forward_ready

STRAIGHT_CORRIDOR_ACK_PUBLIC
  no routing ambiguity
  if this works but serpentine fails, routing policy is blocker
```

Lifecycle tick order:

```text
1. read in_flight packet on edge A->B
2. B accepts only valid unseen generation
3. B applies local gate: phase_i + gate_B -> phase_(i+g)
4. B emits ACK
5. edge A->B is consumed after ACK
6. B forwards rotated packet once
7. duplicate/replay packets are dropped
8. target records receive-committed delivery in ledger
9. swap buffers
```

## Arms

```text
NODE_BROADCAST_BASELINE_014
BEST_PUBLIC_COMBO_014
EDGE_PACKET_016_BEST_PUBLIC
ONE_EDGE_ACK_LIFECYCLE
STRAIGHT_CORRIDOR_ACK_PUBLIC
ACK_WITH_TARGET_LEDGER
ACK_WITHOUT_TARGET_LEDGER
ACK_TARGET_LEDGER_ONLY_DIAGNOSTIC
ACK_PUBLIC_CORRIDOR_NO_REENTRY
ACK_PUBLIC_GRADIENT
ACK_ORACLE_ROUTE_DIAGNOSTIC
ACK_FLOOD_DENSE
ACK_DEDUPE_ON
ACK_DEDUPE_OFF_ABLATION
ACK_GENERATION_ID_ON
ACK_GENERATION_ID_OFF_ABLATION
ACK_CONSUME_PHASE_ONLY
ACK_CONSUME_FULL_EDGE
ACK_WITHOUT_CONSUME_ABLATION
CONSUME_WITHOUT_ACK_ABLATION
RANDOM_RULE_ACK_WITH_LEDGER
RANDOM_RULE_ACK_NO_LEDGER
RANDOM_ROUTE_ACK_CONTROL
```

Oracle routing may use `true_path` and is diagnostic-only. Public routing may
use only wall mask, source, source phase, target, previous edge identity, local
free neighbors, and gates.

## Metrics And Outputs

Core metrics:

```text
phase_final_accuracy
long_path_accuracy
family_min_accuracy
same_target_counterfactual_accuracy
gate_shuffle_collapse
target_delivery_rate
target_wrong_delivery_rate
wrong_if_delivered_rate
wrong_phase_growth_rate
final_minus_best_gap
one_edge_ack_success_rate
receive_commit_rate
ack_rate
ack_latency_mean
consume_after_ack_rate
unacked_packet_rate
duplicate_suppression_rate
replay_rejection_rate
generation_collision_rate
stale_generation_rejection_rate
valid_generation_accept_rate
in_flight_packet_count
target_ledger_power
dead_end_drop_rate
packet_fanout_mean
packet_fanout_max
candidate_delta_nonzero_fraction
positive_delta_fraction
accepted_delta_mean
candidate_delta_std
ack_lifecycle_delta_vs_node_broadcast
ack_lifecycle_delta_vs_016_edge_packet
random_rule_ack_accuracy
random_route_ack_accuracy
wall_leak_rate
forbidden_private_field_leak
nonlocal_edge_count
direct_output_leak_rate
public_routing_used_private_path
```

Required outputs:

```text
queue.json
progress.jsonl
metrics.jsonl
ack_lifecycle_metrics.jsonl
routing_metrics.jsonl
ledger_metrics.jsonl
dedupe_metrics.jsonl
generation_metrics.jsonl
credit_signal_metrics.jsonl
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

Positive gate:

```text
ACK_CONSUME_RESCUES_LONG_CHAIN only if a public non-diagnostic arm reaches:
  phase_final_accuracy >= 0.95
  long_path_accuracy >= 0.95
  family_min_accuracy >= 0.85
  same_target_counterfactual_accuracy >= 0.85
  gate_shuffle_collapse >= 0.50
  wrong_if_delivered_rate <= 0.10
  final_minus_best_gap <= 0.05
  ACK_WITHOUT_TARGET_LEDGER shows material improvement
  random rule/route controls remain weak
  wall/private/nonlocal/direct leaks = 0
```

If only target ledger works, report `TARGET_LEDGER_ONLY_NOT_TRANSPORT`. If
oracle works while public routing fails, report routing blockage. If random
controls pass, report control contamination rather than success.

## Claim Boundary

017 can support that ACK/Consume lifecycle has signal, rescues fixed toy
transport, or improves credit signal for future search. It cannot support
production architecture, full VRAXION, language grounding, consciousness,
Prismion uniqueness, or physical quantum behavior.
