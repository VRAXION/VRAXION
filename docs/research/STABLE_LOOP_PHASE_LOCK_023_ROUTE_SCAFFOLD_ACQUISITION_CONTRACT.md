# STABLE_LOOP_PHASE_LOCK_023_ROUTE_SCAFFOLD_ACQUISITION Contract

## Summary

022 showed that ordered successor breadcrumb fields are a useful route identity representation when a scaffold exists:

```text
cell -> next_route_cell
```

023 asks where the first useful scaffold can come from. This is a scaffold-acquisition probe, not a new phase-rule probe and not a production routing claim.

## Fixed Substrate

Keep the completed local phase rule:

```text
phase_i + gate_g -> phase_(i+g)
```

Keep directed route delivery with receive-commit ledger semantics from 019/021/022. No public `instnct-core` API changes.

## Required Arms

```text
TRUE_PATH_UPPER_BOUND
NOISY_BFS_SCAFFOLD
DISTANCE_FIELD_SCAFFOLD
FRONTIER_PARENT_SCAFFOLD
SOURCE_TARGET_ANCHOR_SCAFFOLD
BIDIRECTIONAL_CANDIDATE_FIELD_THEN_PRUNE
RANDOM_SPARSE_SCAFFOLD_PLUS_DELIVERY_REWARD
HUB_DEGREE_PRIOR_SCAFFOLD
SHORT_PATH_CURRICULUM_TO_LONG
DENSE_THEN_CRYSTALLIZE_PRUNE_ROUTE
RANDOM_FROM_SCRATCH_BASELINE
RANDOM_SCAFFOLD_CONTROL
RANDOM_PHASE_RULE_CONTROL
```

`TRUE_PATH_UPPER_BOUND` is diagnostic-only. Dense/prune and curriculum arms can support scaffold strategy claims, not production routing.

## Metrics

```text
phase_final_accuracy
sufficient_tick_final_accuracy
long_path_accuracy
family_min_accuracy
wrong_if_delivered_rate
initial_successor_link_accuracy
initial_route_order_accuracy
scaffold_coverage
scaffold_noise_rate
scaffold_reciprocal_rate
scaffold_branch_count
scaffold_cycle_count
repair_completion_success_rate
candidate_delta_nonzero_fraction
positive_delta_fraction
gate_shuffle_collapse
same_target_counterfactual_accuracy
random_control_accuracy
```

## Verdicts

```text
SCAFFOLD_ACQUISITION_POSITIVE
BFS_SCAFFOLD_USEFUL
DISTANCE_SCAFFOLD_USEFUL
FRONTIER_PARENT_SCAFFOLD_USEFUL
SOURCE_TARGET_ANCHOR_SCAFFOLD_USEFUL
DENSE_THEN_PRUNE_ROUTE_WORKS
CURRICULUM_REQUIRED
DELIVERY_REWARD_HAS_BUT_INSUFFICIENT_SIGNAL
RANDOM_FROM_SCRATCH_STILL_FAILS
TRUE_PATH_UPPER_BOUND_CONFIRMED
RANDOM_CONTROL_FAILS
RANDOM_PHASE_RULE_FAILS
PRODUCTION_API_NOT_READY
```

## Decision Rules

If scaffold plus 022-style repair/completion reaches:

```text
sufficient_tick_final_accuracy >= 0.95
long_path_accuracy >= 0.95
family_min_accuracy >= 0.85
wrong_if_delivered_rate <= 0.10
same_target_counterfactual_accuracy >= 0.85
gate_shuffle_collapse >= 0.50
```

then report `SCAFFOLD_ACQUISITION_POSITIVE`.

If only true-path or supplied/private scaffold works, public scaffold remains blocked.

If dense candidate field plus prune works, report `DENSE_THEN_PRUNE_ROUTE_WORKS`.

If short-path scaffold transfers to long paths, report `CURRICULUM_REQUIRED`.

If delivery-reward search has signal but misses the positive gate, report `DELIVERY_REWARD_HAS_BUT_INSUFFICIENT_SIGNAL`.

## Required Outputs

```text
queue.json
progress.jsonl
metrics.jsonl
scaffold_metrics.jsonl
delivery_metrics.jsonl
routing_metrics.jsonl
family_metrics.jsonl
counterfactual_metrics.jsonl
control_metrics.jsonl
locality_audit.jsonl
mechanism_ranking.json
summary.json
report.md
contract_snapshot.md
examples_sample.jsonl
job_progress/*.jsonl
```

No black-box runs: append heartbeat progress and refresh `summary.json` / `report.md` during long runs.

## Claim Boundary

023 can support scaffold acquisition strategies for ordered-successor route-token fields in toy phase-lane tasks. It cannot claim production architecture, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.
