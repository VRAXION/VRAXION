# D52 Mutable ECF Controller Scale Confirm Contract

## Purpose

D52 scale-confirms the D51 mutable ECF controller policy on the controlled symbolic joint formula discovery task.

The controller learns only the policy over actions:

```text
DECIDE
REQUEST_SUPPORT
REQUEST_COUNTER_TOP1_TOP2
REQUEST_JOINT_COUNTER
REQUEST_EXTERNAL_TEST
ABSTAIN
```

The fixed symbolic task remains:

```text
formula = operator(cell_a, cell_b) mod 9
```

Cells and operator are hidden from fair controller inputs.

## Required Scope

Artifacts are written only under:

```text
target/pilot_wave/d52_mutable_ecf_controller_scale_confirm/
```

Tracked D52 files:

```text
docs/research/D52_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRM_CONTRACT.md
docs/research/D52_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRM_RESULT.md
scripts/probes/run_d52_mutable_ecf_controller_scale_confirm.py
scripts/probes/run_d52_mutable_ecf_controller_scale_confirm_check.py
```

## Arms

```text
D50_FULL_HANDCODED_REFERENCE
CAP_7_REFERENCE
CAP_9_REFERENCE
ALWAYS_COUNTER_CONTROL
RANDOM_POLICY_CONTROL
GREEDY_DECIDE_CONTROL
COST_ONLY_MUTABLE_CONTROL
MUTABLE_LINEAR_CONTROLLER
MUTABLE_RULE_TABLE_CONTROLLER
MUTABLE_SMALL_TREE_CONTROLLER
MUTABLE_HYBRID_CONTROLLER
BEST_D51_REPLAY
```

## Required Audits

```text
d51_upstream_manifest.json
fitness_audit_report.json
support_accounting_report.json
action_distribution_report.json
mutation_acceptance_report.json
support_cost_frontier_report.json
false_confidence_report.json
regime_breakdown_report.json
controller_generalization_report.json
min_seed_gate_report.json
```

## Positive Gate

Best mutable controller must satisfy:

```text
exact_joint >= 0.995
correlated_echo >= 0.995
adversarial_distractor >= 0.995
external_test_required >= 0.99
indistinguishable_abstain >= 0.99
false_confidence <= 0.01
min_seed_exact >= 0.99
support_used <= D50_FULL_HANDCODED_REFERENCE support_used
ALWAYS_COUNTER_CONTROL has higher support cost
RANDOM_POLICY_CONTROL and GREEDY_DECIDE_CONTROL are worse
COST_ONLY_MUTABLE_CONTROL fails accuracy/safety
failed_jobs = []
```

## Decisions

```text
mutable_ecf_controller_scale_confirmed -> D53_MUTABLE_ECF_INTEGRATION_WITH_VRAXION_MUTATION_ARCHITECTURE
mutable_ecf_controller_scale_confirmed_high_cost -> D52C_SUPPORT_COST_OPTIMIZATION
mutable_ecf_controller_scale_not_confirmed -> D52_REPAIR
mutable_controller_fitness_exploit_detected -> D52F_FITNESS_REPAIR
```

## Boundary

D52 only scale-confirms mutable control policy for controlled symbolic joint formula discovery. It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority.
