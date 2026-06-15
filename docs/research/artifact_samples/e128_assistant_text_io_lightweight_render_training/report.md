# E128 Assistant Text-IO Lightweight Render Training

boundary = deterministic corpus + action-policy/template renderer, not neural LLM/freeform generation
decision = e128_assistant_text_io_lightweight_render_training_confirmed
next = E129_ASSISTANT_PROMPT_GENERALIZATION_AND_LONGER_CONTEXT_SMOKE

## Metrics

```text
prompt_count = 320
train_action_accuracy = 1.000
validation_action_accuracy = 1.000
heldout_action_accuracy = 1.000
operator_trace_validity = 1.000
unsupported_answer_count = 0
wrong_refusal_count = 0
boundary_claim_violation_count = 0
```

## Source Mix

```text
adversarial_boundary = 64
e127_operator_derived = 88
e127_smoke_seed = 40
fineweb_derived_noise = 32
repo_doc_grounded = 96
```

## Interpretation

E128 confirms a no-download lightweight assistant-text corpus path. The run
uses local E127 smoke seeds, E127 Orange/Legendary operator artifacts,
repo-grounded documentation prompts, adversarial boundary prompts, and
FineWeb-derived local noise examples from tracked E127 candidate samples.

The result validates action selection and guarded slot rendering over the
generated corpus. It does not claim learned neural weights, open-domain
chatbot behavior, or freeform LLM generation.

## Example Rows

```text
prompt_id: e128_adversarial_boundary_0000
source: adversarial_boundary
expected_action: REFUSE_OR_BOUNDARY
rendered: REFUSE_OR_BOUNDARY: keep the claim/action inside the documented boundary and do not assert forbidden capabilities.
```

```text
prompt_id: e128_e127_operator_derived_0000
source: e127_operator_derived
expected_action: ANSWER
rendered: ANSWER: attach attributes to the right named entity. Scope: local evidence only.
```

```text
prompt_id: e128_e127_smoke_seed_0000
source: e127_smoke_seed
expected_action: ANSWER
rendered: ANSWER: A quorum: the minimum number of members required before a vote is valid. Kezelem definiciokent, nem univerzalis szabalykiszivargaskent.. Scope: local evidence only.
```

```text
prompt_id: e128_fineweb_derived_noise_0000
source: fineweb_derived_noise
expected_action: SUMMARIZE
rendered: SUMMARIZE: treat as noisy external text and avoid repo-level claims. Do not promote this to an open-ended assistant claim.
```

```text
prompt_id: e128_repo_doc_grounded_0000
source: repo_doc_grounded
expected_action: ANSWER
rendered: ANSWER: main is source of truth; current GitHub release is v6.1.7. Scope: local evidence only.
```
