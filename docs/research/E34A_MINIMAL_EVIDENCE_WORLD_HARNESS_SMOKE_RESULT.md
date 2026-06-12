# E34A Minimal Evidence World Harness Smoke Result

Status: completed and checker validated.

## Decision

```text
decision = e34a_active_evidence_world_confirmed
primary_run_id = bfd5c9a7ecd660c0
cpu_confirm_run_id = ed66b45d67d1a952
target_checker_failure_count = 0
sample_only_checker_passed = true
gradient_descent_used = false
optimizer_used = false
backprop_used = false
```

## Primary Result

E34A tested a minimal active-evidence world:

```text
one verified initial observation
one untrusted rumor
INSPECT(feature) actions
ANSWER(cause) only after the candidate state is resolved
```

Primary evidence run:

```text
learned_mutation_policy:
  closed_loop_success = 1.000000
  answer_correct      = 1.000000
  trace_exact         = 1.000000
  wrong_confident     = 0.000000
  false_ask           = 0.000000
  redundant_actions   = 0.000000
  avg_steps           = 3.000000

ask_all_until_unique:
  closed_loop_success = 1.000000
  false_ask           = 0.339286
  avg_steps           = 3.366071

forced_initial_answer:
  closed_loop_success = 0.000000
  wrong_confident     = 1.000000

random_action_control:
  closed_loop_success = 0.473810
  wrong_confident     = 0.526190

oracle_info_gain_reference:
  closed_loop_success = 1.000000
  avg_steps           = 3.000000
```

The learned policy matched the oracle step count on this minimal world while
avoiding the ask-all baseline's redundant inspections.

## Splits

```text
learned_mutation_policy split closed_loop_success:
  heldout        = 1.000000
  OOD            = 1.000000
  counterfactual = 1.000000
  adversarial    = 1.000000
```

## Mutation / Rollback

```text
accepted_mutations = 2
rejected_mutations = 2158
rollback_count     = 2158
parameter_changed  = true
deterministic_replay_match_rate = 1.000000
```

The mutation loop found the useful evidence-seeking policy early, then rejected
later proposals that failed to improve it. This is expected for the minimal
world: the optimum is a small rule-like policy, not a long gradual training
curve.

## CPU Confirm

The independent CPU confirm reproduced the decision:

```text
decision = e34a_active_evidence_world_confirmed
closed_loop_success = 1.000000
trace_exact = 1.000000
wrong_confident = 0.000000
avg_steps = 3.000000
target_checker_failure_count = 0
sample_only_checker_passed = true
```

## Interpretation

E34A confirms the harness shape:

```text
limited initial evidence
-> active evidence acquisition
-> Flow Field candidate-state update
-> answer only after evidence resolves the state
```

The result does not prove raw text reasoning. It shows that, in a deterministic
minimal evidence world, the gradientless mutation/rollback path can learn an
active evidence-seeking policy that is both correct and more efficient than
blindly asking for every feature.

## Boundary

E34A is a deterministic symbolic/numeric active-evidence smoke probe. It does
not claim chatbot behavior, raw language understanding, AGI, consciousness,
deployed-model behavior, or model-scale behavior.

## Recommended Next Step

Scale the same closed-loop evidence acquisition idea toward a messier bridge:

```text
E34B_ACTIVE_EVIDENCE_WORLD_WITH_NOISY_TEXT_OBSERVATIONS
```

The key next question is whether the same evidence-seeking policy survives when
features are no longer clean binary nodes, but are surfaced through short,
noisy, naturalized text observations with visible evidence spans.
