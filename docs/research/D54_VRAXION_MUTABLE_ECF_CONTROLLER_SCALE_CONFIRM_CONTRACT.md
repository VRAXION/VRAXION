# D54 VRAXION Mutable ECF Controller Scale Confirm Contract

## Goal

Scale-confirm the D53 VRAXION-style mutable ECF controller integration on the
controlled symbolic joint formula discovery task.

Formula:

```text
operator(cell_a, cell_b) mod 9
```

Cells and operator remain hidden from fair mutable/controller arms.

## Boundary

D54 only scale-confirms VRAXION-style mutable ECF controller integration for
controlled symbolic joint formula discovery. It does not prove full VRAXION
sparse firing brain learning, raw visual Raven solving, AGI, consciousness,
DNA/genome success, architecture superiority, or production readiness.

## Required Arms

```text
D53_BEST_HYBRID_REPLAY
D52_RULE_TABLE_REPLAY
D50_HANDCODED_FULL_REFERENCE
RANDOM_POLICY_CONTROL
GREEDY_DECIDE_CONTROL
ALWAYS_COUNTER_CONTROL
COST_ONLY_MUTATION_CONTROL
VRAXION_MUTABLE_RULE_TABLE
VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER
VRAXION_MUTABLE_POCKET_STATE_CONTROLLER
VRAXION_MUTABLE_HYBRID_CONTROLLER
SPARSE_GATE_ABLATION
POCKET_STATE_ABLATION
MUTATION_DISABLED_CONTROL
```

## Required Reports

```text
d53_upstream_manifest.json
canonical_vraxion_audit_report.json
sparse_firing_usage_report.json
representation_report.json
mutation_acceptance_report.json
fitness_landscape_report.json
support_cost_frontier_report.json
false_confidence_report.json
action_distribution_report.json
component_ablation_report.json
controller_comparison_report.json
regime_breakdown_report.json
min_seed_gate_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
row_outputs_test.jsonl
row_outputs_ood.jsonl
```

## Positive Gates

Best VRAXION mutable arm:

```text
exact_joint >= 0.995
correlated_echo >= 0.995
adversarial_distractor >= 0.995
external_test_required >= 0.99
indistinguishable_abstain >= 0.99
false_confidence <= 0.01
support <= D50_HANDCODED_FULL_REFERENCE support
min_seed_exact >= 0.99
min_seed_correlated >= 0.99
min_seed_adversarial >= 0.99
beats random/greedy/cost-only controls
always-counter has higher support cost
failed_jobs = []
```

## Decisions

```text
vraxion_mutable_ecf_controller_scale_confirmed
  -> D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE

vraxion_mutable_controller_scale_confirmed_non_sparse
  -> D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE

vraxion_sparse_gate_controller_path_confirmed
  -> D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE

vraxion_mutable_ecf_controller_scale_not_confirmed
  -> D54_REPAIR
```

## No-Black-Box Rule

The runner must write `queue.json` immediately and append `progress.jsonl`
through row generation, train pack build, mutation search, evaluation, report
writing, and final decision. Long mutation runs must also write partial mutation
snapshots and partial evaluation metric snapshots.
