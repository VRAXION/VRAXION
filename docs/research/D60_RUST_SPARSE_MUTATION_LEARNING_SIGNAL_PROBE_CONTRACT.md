# D60 Rust Sparse Mutation Learning Signal Probe Contract

## Purpose

D60 tests whether the canonical Rust sparse ECF action controller has a real mutation learning signal.

D59 confirmed that the Rust sparse mutation path can be exercised, but the replay task was already saturated near exact joint accuracy 0.9994. D60 therefore separates two tracks:

```text
SATURATED_STABILITY:
  replay the D58/D59 task and check that mutation does not break the Rust path or safety behavior

HARD_NON_SATURATED_LEARNING:
  use a harder but oracle-solvable support-budget setting where replay is below ceiling
```

## Boundary

D60 only tests learning signal for mutation and selection of a canonical Rust sparse ECF action controller on controlled symbolic joint formula discovery.

It does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.

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

## Hard Variants

D60 must report difficulty and oracle upper bounds for:

```text
support_budget_cap_5
support_budget_cap_6
support_budget_cap_8
support_budget_cap_9
higher_counter_support_cost_cap_9
```

The selected hard variant must be oracle-solvable and non-saturated for replay when possible.

## Arms

```text
D58_REPLAY_REFERENCE
D59_BEST_REPLAY
MUTATION_DISABLED_CONTROL
RANDOM_MUTATION_CONTROL
COST_ONLY_MUTATION_CONTROL
ACCURACY_ONLY_MUTATION
SUPPORT_COST_TARGETED_MUTATION
HARD_STRESS_MUTATION
MULTI_OBJECTIVE_PARETO_MUTATION
LARGE_STEP_MUTATION
STRUCTURED_GATE_MUTATION
NOVELTY_DIVERSITY_MUTATION
RANDOM_POLICY_CONTROL
GREEDY_DECIDE_CONTROL
ALWAYS_COUNTER_CONTROL
SPIKE_SHUFFLE_CONTROL
THRESHOLD_ABLATION
REWIRE_ABLATION
```

## Required Reports

```text
d59_upstream_manifest.json
task_difficulty_report.json
oracle_upper_bound_report.json
saturated_track_report.json
hard_learning_track_report.json
mutation_causality_report.json
accepted_mutation_delta_report.json
pareto_frontier_report.json
support_cost_frontier_report.json
safety_constraint_report.json
fitness_definition_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
```

## Success Logic

Learning success on hard track requires the best mutated Rust controller to improve over D58 replay by at least one:

```text
exact +0.03
cost_adjusted +0.02
support -0.25 at same accuracy
```

Safety must hold:

```text
false_confidence <= 0.01
saturated track does not regress
Rust path invoked
fallback_rows = 0
failed_jobs = []
```

## Decisions

```text
rust_sparse_mutation_learning_signal_confirmed
  next = D61_RUST_SPARSE_MUTATION_SCALE_CONFIRM

rust_sparse_mutation_path_confirmed_no_learning_signal
  next = D60C_MUTATION_SEARCH_SPACE_REPAIR

d60_hard_task_invalid
  next = D60H_HARD_TASK_REDESIGN

rust_sparse_mutation_safety_failure
  next = D60S_SAFETY_FITNESS_REPAIR
```

## Hard Gates

```text
no broad claims
no label echo fair oracle
no Python hash()
no fixed synthetic accuracies
truth hidden from controller inputs
Rust path invoked for Rust arms
fallback_rows = 0 for Rust arms
oracle upper bounds reported
false confidence and abstain measured
controls required
failed jobs visible
```
