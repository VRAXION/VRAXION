# E34B Active Evidence World With Noisy Text Observations Result

Status: completed and checker validated.

## Decision

```text
decision = e34b_noisy_text_active_evidence_confirmed
primary_run_id = 85e359193e571108
cpu_confirm_run_id = 9bf34b9f5ef2b066
target_checker_failure_count = 0
sample_only_checker_passed = true
gradient_descent_used = false
optimizer_used = false
backprop_used = false
```

## Primary Result

E34B moved E34A from clean binary feature observations to short noisy text
observations:

```text
verified initial text observation
untrusted rumor text observation
INSPECT_TEXT(feature)
text extraction into Flow Field evidence
ANSWER(cause) only after the candidate state is resolved
```

Primary evidence run:

```text
learned_mutation_text_policy:
  closed_loop_success     = 0.992262
  answer_correct          = 0.992262
  trace_exact             = 0.992262
  text_extraction_accuracy = 0.997421
  wrong_confident         = 0.000000
  false_ask               = 0.000000
  redundant_actions       = 0.000000
  avg_steps               = 4.000000

ask_all_text_until_unique:
  closed_loop_success     = 1.000000
  false_ask               = 0.305952
  avg_steps               = 4.335119

forced_initial_text_answer:
  closed_loop_success     = 0.000000
  wrong_confident         = 1.000000

random_text_action_control:
  closed_loop_success     = 0.439286
  wrong_confident         = 0.560714

keyword_shortcut_text_control:
  closed_loop_success     = 0.750000
  adversarial_success     = 0.000000
  adversarial_text_extract = 0.293452

oracle_text_policy_reference:
  closed_loop_success     = 1.000000
  avg_steps               = 4.000000
```

## Split Result

```text
learned_mutation_text_policy closed_loop_success:
  heldout        = 1.000000
  OOD            = 1.000000
  counterfactual = 1.000000
  adversarial    = 0.969048

learned_mutation_text_policy text_extraction_accuracy:
  heldout        = 1.000000
  OOD            = 1.000000
  counterfactual = 1.000000
  adversarial    = 0.989683
```

The remaining primary errors are concentrated in adversarial contrast-clause
text. This is a useful caveat, not a hidden failure: the checker keeps the
adversarial split visible and the decision threshold requires the shortcut
control to fail there.

## Mutation / Rollback

```text
accepted_mutations = 3
rejected_mutations = 2157
rollback_count     = 2157
parameter_changed  = true
deterministic_replay_match_rate = 1.000000
```

The mutation loop found a policy that combines targeted evidence acquisition
with source-aware text extraction. Later proposals were rejected/rolled back.

## Important Smoke Finding

The first E34B smoke exposed a real parser bug: substring matching treated the
`no` inside `note` as a negative value token. The runner was corrected to use
token-level matching before the evidence run.

This matters because it shows the noisy text layer is doing real work. A brittle
surface parser can break the active evidence loop even when the underlying
evidence strategy is sound.

## CPU Confirm

The independent CPU confirm reproduced the decision:

```text
decision = e34b_noisy_text_active_evidence_confirmed
closed_loop_success = 1.000000
trace_exact = 1.000000
text_extraction_accuracy = 1.000000
wrong_confident = 0.000000
target_checker_failure_count = 0
sample_only_checker_passed = true
```

## Interpretation

E34B confirms that the E34A closed-loop pattern survives a controlled noisy text
observation layer:

```text
limited initial text evidence
-> choose useful text observation to inspect
-> extract mechanical feature/value evidence from visible text
-> update Flow Field candidate state
-> answer after the state resolves
```

The result is not a raw language reasoning claim. It is a controlled bridge:

```text
clean feature evidence
-> text-mediated evidence acquisition
```

## Boundary

E34B is a deterministic symbolic/noisy-text active-evidence probe. It does not
claim chatbot behavior, general natural language understanding, AGI,
consciousness, deployed-model behavior, or model-scale behavior.

## Recommended Next Step

The next useful step is to stress the text interface rather than add a new
architecture:

```text
E34C_ACTIVE_EVIDENCE_TEXT_PARAPHRASE_AND_CONTRADICTION_STRESS
```

Key question:

```text
Does the active evidence loop survive broader paraphrase, multi-sentence
observations, multiple competing claims, and delayed contradiction repair?
```
