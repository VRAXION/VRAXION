# D60S Rust Sparse Mutation Safety Fitness Repair Contract

## Purpose

D60S repairs the D60 no-forgetting failure.

D60 proved that Rust sparse mutation can learn a hard support-capped controller, but the best hard controller regressed on the saturated D58/D59 replay distribution:

```text
D60 hard gain over D58 replay = +0.390650
D60 saturated stability = failed
next = D60S_SAFETY_FITNESS_REPAIR
```

D60S asks whether a Rust sparse controller can keep the hard-task gain while preserving the saturated replay behavior.

## Boundary

D60S only tests safety/no-forgetting fitness repair for mutation of a canonical Rust sparse ECF action controller in controlled symbolic joint formula discovery.

It does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, production readiness, or that the Rust sparse network is the formula solver.

## Fixed Task

Controlled symbolic joint formula discovery:

```text
formula = operator(cell_a, cell_b) mod 9
```

The Rust sparse network chooses only the ECF action:

```text
DECIDE
REQUEST_SUPPORT
REQUEST_COUNTER_TOP1_TOP2
REQUEST_JOINT_COUNTER
REQUEST_EXTERNAL_TEST
ABSTAIN
```

The symbolic formula solver remains fixed. Truth labels and true cells/operators are hidden from controller inputs.

## Evaluation Tracks

```text
SATURATED_STABILITY:
  D58/D59-style full-support replay. Mutation must not forget this path.

HARD_CAP8_LEARNING:
  D60 hard support-budget-cap-8 setting. Repair must keep the hard gain over D58 replay.

MIXED_EVAL:
  Alternates saturated and hard rows to test whether one controller/policy can behave correctly in both contexts.
```

The D58 hard replay baseline is recovered from D60:

```text
d58_hard_replay_exact = D60 hard best exact - D60 exact gain
```

## Arms

```text
D59_REFERENCE
D60_HARD_BEST_REPLAY
SINGLE_POLICY_MULTI_ENV_FITNESS
LEXICOGRAPHIC_SAFETY_FIRST_FITNESS
PARETO_MULTI_ENV_MUTATION
STABILITY_REGULARIZED_MUTATION
COST_ONLY_MUTATION_CONTROL
ACCURACY_ONLY_MUTATION_CONTROL
RANDOM_MUTATION_CONTROL
MUTATION_DISABLED_CONTROL
THRESHOLD_ABLATION
REWIRE_ABLATION
DUAL_POLICY_GATED_CONTROLLER
CONTEXT_GATED_POLICY_ENSEMBLE
RANDOM_POLICY_CONTROL
GREEDY_DECIDE_CONTROL
ALWAYS_COUNTER_CONTROL
SPIKE_SHUFFLE_CONTROL
```

The gated arms are not allowed to see truth labels. They may route only from task context such as saturated replay vs support-budget-cap-8 evaluation context.

## Required Reports

```text
d60_upstream_manifest.json
fitness_definition_report.json
multi_environment_eval_report.json
saturated_stability_report.json
hard_learning_report.json
mixed_eval_report.json
policy_gate_report.json
no_forgetting_report.json
pareto_frontier_report.json
mutation_causality_report.json
support_cost_frontier_report.json
safety_constraint_report.json
rust_invocation_report.json
ablation_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
row_outputs_test.jsonl
row_outputs_ood.jsonl
trained_policy_manifest.json
```

Long runs must write `queue.json`, `progress.jsonl`, partial mutation reports, and partial evaluation snapshots during the run.

## Positive Gates

For a repair candidate:

```text
saturated exact >= D59 reference exact - 0.002
hard exact >= 0.990
mixed exact >= 0.990
hard exact gain over D58 hard replay >= 0.300
false confidence <= 0.010
indistinguishable abstain >= 0.990
Rust path invoked
fallback_rows = 0
random/greedy/shuffle controls worse
failed_jobs = []
```

## Decisions

```text
rust_sparse_mutation_safety_fitness_repaired
  next = D61_RUST_SPARSE_MUTATION_SCALE_CONFIRM

gated_policy_required_for_no_forgetting
  next = D61_GATED_RUST_SPARSE_MUTATION_SCALE_CONFIRM

safety_fitness_repair_not_confirmed
  next = D60R_NO_FORGETTING_REPAIR

learning_signal_lost_under_safety_fitness
  next = D60L_LEARNING_SIGNAL_REPAIR
```

## Hard Gates

```text
no broad claims
no label echo fair oracle
no Python hash()
no fixed synthetic accuracies
no hit=random.random()<p fake sampling
truth hidden from controller inputs
Rust path invoked for Rust/gated arms
fallback_rows = 0 for Rust/gated arms
mutation representation explicit
gated policy basis reported
false confidence and abstain measured
controls required
failed jobs visible
```
