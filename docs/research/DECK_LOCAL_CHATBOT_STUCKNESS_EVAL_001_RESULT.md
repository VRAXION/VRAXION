# DECK_LOCAL_CHATBOT_STUCKNESS_EVAL_001 Result

`DECK_LOCAL_CHATBOT_STUCKNESS_EVAL_001` evaluates the checkpoints currently present on the Deck against simple assistant-style prompts and stuckness/collapse checks.

This is eval-only. It does not train, does not use an LLM judge, and does not replace the missing `099/100/101` mainline artifact chain.

## Scope

The evaluated local checkpoints are:

```text
target/pilot_wave/deck_local_text_lm_smoke_001/extended_2500/checkpoints/deck_local_text_lm/model.pt
target/pilot_wave/deck_local_text_lm_smoke_001/smoke/checkpoints/deck_local_text_lm/model.pt
```

These are byte-level AG News next-byte language models, not chat instruction models.

The eval prompt families cover:

```text
FRESH_ASSISTANT_INSTRUCTION
FRESH_SHORT_EXPLANATION
FRESH_OPEN_DOMAIN_SIMPLE_QA
FRESH_MULTI_TURN_CONTEXT_CARRY
FRESH_HUNGARIAN_BASIC_CHAT
FRESH_ENGLISH_BASIC_CHAT
FRESH_UNSUPPORTED_REFUSAL
FRESH_BOUNDARY_INJECTION_REFUSAL
FRESH_ANTI_REPETITION
FRESH_CONTEXT_CONFLICT
```

## Runner

```text
scripts/probes/run_deck_local_chatbot_stuckness_eval_001.py
```

Default output:

```text
target/pilot_wave/deck_local_chatbot_stuckness_eval_001/smoke
```

Required local artifacts include:

```text
eval_config.json
upstream_manifest.json
model_manifest.json
eval_prompts.jsonl
generation_results.jsonl
family_metrics.json
stuckness_metrics.json
failure_map.json
human_readable_samples.jsonl
failure_case_samples.jsonl
summary.json
report.md
```

## Result

Status: `recorded`

Primary checkpoint/mode:

```text
deck_local_text_lm_extended_2500/sampled
```

Primary metrics:

```text
overall_generated_accuracy:  0.000
nonempty_generation_rate:   1.000
utf8_valid_generation_rate: 1.000
empty_output_rate:          0.000
static_output_rate:         0.000
repetition_rate:            0.000
copy_prompt_rate:           0.000
permanent_stuck_rate:       0.000
distinct_response_rate:     1.000
```

All candidate stuckness metrics:

```text
deck_local_text_lm_extended_2500/greedy:
  accuracy: 0.000
  stuck:    0.950
  utf8:     1.000

deck_local_text_lm_extended_2500/sampled:
  accuracy: 0.000
  stuck:    0.000
  utf8:     1.000

deck_local_text_lm_smoke/greedy:
  accuracy: 0.000
  stuck:    1.000
  utf8:     1.000

deck_local_text_lm_smoke/sampled:
  accuracy: 0.000
  stuck:    0.000
  utf8:     1.000
```

Main verdicts:

```text
CHATBOT_STUCKNESS_EVAL_RECORDED
OFFICIAL_100_CHAT_CHECKPOINT_MISSING
OFFICIAL_101_EVAL_NOT_AVAILABLE_OR_NOT_POSITIVE
REGULAR_CHATBOT_ACCURACY_WEAK
MULTI_TURN_CONTEXT_FAILS
HUNGARIAN_BASIC_FAILS
REFUSAL_FAILS
BOUNDARY_REFUSAL_FAILS
TEXT_LM_NOT_CHATBOT_READY
GPT_LIKE_READINESS_NOT_CLAIMED
PRODUCTION_CHAT_NOT_CLAIMED
```

## Interpretation

The local Deck model does generate nonempty UTF-8 continuations. With sampled decoding it does not permanently collapse into a repeated/static response on this prompt set.

However, it does not behave as a regular chatbot. It fails the assistant-family tasks under strict heuristic scoring:

```text
instruction following: 0.000
short explanation:     0.000
simple QA:             0.000
context carry:         0.000
Hungarian basic chat:  0.000
English basic chat:    0.000
unsupported refusal:   0.000
boundary refusal:      0.000
context conflict:      0.000
```

Greedy decoding is a separate failure mode: it frequently collapses into repeated news-like byte-text fragments. The extended checkpoint sampled path avoids that stuckness, but it still produces AG News-style gibberish rather than assistant answers.

## Current Answer

For the question:

```text
How good is it as a regular chatbot, and does it get permanently stuck?
```

The measured answer is:

```text
regular chatbot: no
sampled permanent stuckness: not on this small eval
greedy permanent stuckness: yes, high repetition/collapse
```

This confirms the earlier distinction:

```text
learns next-byte text modeling: yes
assistant/chat capability: not shown
```

## Next Useful Gate

The next meaningful step is not more stuckness eval on the same AG News byte LM. The next gate should train or load a genuinely assistant-shaped small model/checkpoint and then rerun this eval:

```text
instruction/chat supervised data
assistant prompt formatting
context carry examples
refusal/boundary examples
anti-repetition decoding checks
```

Until that exists, this Deck checkpoint remains a text-continuation smoke model, not a chatbot baseline.

## Claim Boundary

This result is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not hosted SaaS, not deployment readiness, and not safety alignment.
