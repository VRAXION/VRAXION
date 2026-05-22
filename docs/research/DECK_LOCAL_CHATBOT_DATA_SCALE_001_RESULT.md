# DECK_LOCAL_CHATBOT_DATA_SCALE_001 Result

`DECK_LOCAL_CHATBOT_DATA_SCALE_001` tests whether the current chatbot bottleneck is mainly data coverage or the tiny byte-level architecture/context.

The probe keeps the architecture fixed and compares:

```text
base_extended_text_lm
manual_small_zero_overlap
scaled_paraphrase_zero_overlap
```

All train/eval exact prompt overlap is blocked:

```text
manual_train_eval_exact_prompt_overlap_count: 0
scaled_train_eval_exact_prompt_overlap_count: 0
```

This is a bounded local data-scale probe. It is not GPT-like readiness, not open-domain assistant readiness, not production chat, and not safety alignment.

## Runner

```text
scripts/probes/run_deck_local_chatbot_data_scale_001.py
```

Default output:

```text
target/pilot_wave/deck_local_chatbot_data_scale_001/smoke
```

Lower-step scaled diagnostic:

```text
target/pilot_wave/deck_local_chatbot_data_scale_001/scaled_1000
```

## Default Data-Scale Run

Training setup:

```text
manual_small_zero_overlap:
  train examples: 20
  train steps:    1400

scaled_paraphrase_zero_overlap:
  train examples: 352
  train steps:    2200
```

Main metrics:

```text
base_extended_text_lm/sampled:
  heldout_total_accuracy: 0.038
  stuck:                  0.000
  score:                  0.038

manual_small_zero_overlap/sampled:
  bounded_accuracy:       0.450
  heldout_total_accuracy: 0.115
  stuck:                  0.065
  static:                 0.065
  score:                  0.164

scaled_paraphrase_zero_overlap/greedy:
  bounded_accuracy:       0.300
  heldout_total_accuracy: 0.346
  stuck:                  0.326
  static:                 0.326
  score:                  0.103

scaled_paraphrase_zero_overlap/sampled:
  bounded_accuracy:       0.200
  heldout_total_accuracy: 0.385
  stuck:                  0.457
  static:                 0.457
  score:                 -0.065
```

Verdicts:

```text
SCALED_PARAPHRASE_DATA_IMPROVES_HELDOUT
SCALED_PARAPHRASE_ARM_DOES_NOT_WIN_SCORE
SCALED_ARM_STUCKNESS_RISK
HELDOUT_TRANSFER_STILL_WEAK
OPEN_DOMAIN_CHATBOT_NOT_CLAIMED
```

## Lower-Step Scaled Diagnostic

Training setup:

```text
manual steps: 800
scaled steps: 1000
```

Main metrics:

```text
manual_small_zero_overlap/greedy:
  bounded_accuracy:       0.300
  heldout_total_accuracy: 0.115
  stuck:                  0.022
  static:                 0.000
  score:                  0.181

scaled_paraphrase_zero_overlap/greedy:
  bounded_accuracy:       0.250
  heldout_total_accuracy: 0.269
  stuck:                  0.370
  static:                 0.370
  score:                 -0.059

scaled_paraphrase_zero_overlap/sampled:
  bounded_accuracy:       0.150
  heldout_total_accuracy: 0.269
  stuck:                  0.370
  static:                 0.370
  score:                 -0.102
```

The lower-step scaled run still improves heldout accuracy relative to the small manual arm, but static/stuck behavior remains high.

## Interpretation

The larger paraphrase dataset does help heldout accuracy:

```text
manual heldout_total: 0.115
scaled heldout_total: 0.269 - 0.385
```

So data coverage is part of the bottleneck.

However, the scaled arm does not win the conservative score because it raises static/canned-response behavior:

```text
scaled stuck/static: 0.326 - 0.457
```

So the result is not:

```text
just add more data and the current model becomes a good chatbot
```

The measured result is:

```text
more paraphrase data improves heldout transfer,
but the tiny byte-level feed-forward model turns that data into canned/static response attractors.
```

## Current Bottleneck

The bottleneck is mixed:

```text
data coverage: yes, clearly matters
architecture/objective/context: also clearly limiting
```

Evidence:

```text
heldout improves with more paraphrase data
overall score does not improve because stuck/static risk rises
heldout remains below 0.40
```

Likely causes:

```text
96-byte context
feed-forward next-byte LM, not a sequence model
byte-level learning with tiny capacity
supervised answers create canned response attractors
no anti-static or contrastive objective
```

## Next Useful Gate

The next useful test should not simply add more of the same data. It should change one of:

```text
1. architecture:
   small recurrent or transformer byte/char model with longer context

2. objective:
   anti-static/canned-response penalty
   contrastive prompt-answer negatives

3. data:
   larger phenomenon-balanced paraphrase set
   but evaluated with strict no-overlap heldout and static-risk penalty
```

The cleanest next decision test is:

```text
same scaled data
current feed-forward byte-LM vs small recurrent/transformer sequence model
```

If the sequence model improves heldout without static collapse, the bottleneck is architecture/context.

## Claim Boundary

This result is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not hosted SaaS, not deployment readiness, and not safety alignment.
