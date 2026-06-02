# D53 Mutable ECF Integration With VRAXION Mutation Architecture Contract

## Purpose

D53 tests whether the D52 mutable ECF controller policy can be represented,
mutated, selected, and validated through a VRAXION-style mutation architecture.

This does not learn the full formula solver. The fixed controlled symbolic task
remains:

```text
formula = operator(cell_a, cell_b) mod 9
```

Cells and operator remain hidden from fair controller inputs.

## Required Phase 0 Audit

D53 must identify and report the current canonical VRAXION implementation
surface before claiming any integration:

```text
canonical_vraxion_module_path
spiking_network_module_path
mutation_schedule_module_path
threshold/state limits
mutation operator list actually found
sparse_firing_used_in_d53
controller_genome_state_used
action output encoding smoke passed
```

The expected current architecture surface is:

```text
instnct-core/examples/neuron_grower.rs
instnct-core/src/network.rs
instnct-core/src/evolution.rs
```

## Arms

```text
D52_BEST_RULE_TABLE_REPLAY
HANDCODED_D50_FULL_REFERENCE
RANDOM_POLICY_CONTROL
GREEDY_DECIDE_CONTROL
ALWAYS_COUNTER_CONTROL
COST_ONLY_MUTATION_CONTROL
VRAXION_MUTABLE_RULE_TABLE
VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER
VRAXION_MUTABLE_POCKET_STATE_CONTROLLER
VRAXION_MUTABLE_HYBRID_CONTROLLER
```

## VRAXION-Style Scope

D53 uses mutable controller genomes:

```text
threshold/action-route genome
integer sparse feature gates
feature pockets with priority and action writeback
hybrid rule/gate overlay
```

This is not full sparse firing brain training. The runner must explicitly report
whether sparse firing was used.

## Reports

```text
d52_upstream_manifest.json
canonical_vraxion_smoke_report.json
representation_report.json
mutation_acceptance_report.json
fitness_landscape_report.json
policy_action_distribution_report.json
support_cost_frontier_report.json
false_confidence_report.json
regime_breakdown_report.json
controller_comparison_report.json
vraxion_integration_boundary_report.json
best_policy_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
```

## Positive Gate

Best VRAXION mutable controller must satisfy:

```text
exact_joint >= 0.995
correlated_echo >= 0.995
adversarial_distractor >= 0.995
external_test_required >= 0.99
false_confidence <= 0.01
indistinguishable_abstain >= 0.99
support_used <= HANDCODED_D50_FULL_REFERENCE support_used
beats random/greedy/cost-only
always-counter higher support cost
failed_jobs = []
```

## Decisions

```text
vraxion_mutable_ecf_controller_integration_positive -> D54_VRAXION_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRM
vraxion_mutable_ecf_controller_positive_high_cost -> D53C_SUPPORT_COST_OPTIMIZATION
vraxion_integration_partial_rule_table_only -> D53S_SPARSE_CONTROLLER_REPAIR
vraxion_mutation_controller_not_confirmed -> D53R_MUTATION_REPRESENTATION_REPAIR
```

## Boundary

D53 only tests VRAXION-style mutation integration for mutable ECF controller policy in controlled symbolic joint formula discovery. It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, full VRAXION brain learning, or architecture superiority.
