# DECK_LOCAL_TEXT_LM_SMOKE_001 Result

`DECK_LOCAL_TEXT_LM_SMOKE_001` tests a harder question than the bounded slot learner:

```text
Can a fresh byte-level next-byte LM learn from non-synthetic web/news-like text
on the Deck, improve heldout eval loss, and beat simple controls?
```

This is independent from the missing `099/100/101` target artifact chain. It is not the official FineWeb gate and not the official 100/101 gate.

## Scope

The runner downloads AG News CSV from GitHub by default:

```text
https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv
```

This is web/news-like text, not FineWeb certification.

The model is a byte-level next-byte LM:

```text
previous seq_len bytes -> next byte
```

It records:

```text
train_loss_initial/final
eval_loss_before/after
next_byte_accuracy_before/after
unigram and bigram controls
checkpoint hash change
short continuation samples
```

## Runner

```text
scripts/probes/run_deck_local_text_lm_smoke_001.py
```

Default output:

```text
target/pilot_wave/deck_local_text_lm_smoke_001/smoke
```

Confirmation output:

```text
target/pilot_wave/deck_local_text_lm_smoke_001/extended_2500
```

## Positive Criteria

```text
checkpoint changes
train loss decreases
eval loss decreases
next-byte accuracy improves
trained eval loss beats unigram control
train/eval exact document overlap is zero
generation smoke is nonempty
```

## Result

Status: `positive`

The default Deck smoke passed:

```text
train_loss_initial:              5.568345
train_loss_final:                2.641981
eval_loss_before:                5.555097
eval_loss_after:                 2.669009
eval_loss_delta:                 2.886088
eval_perplexity_after:           14.425673
next_byte_accuracy_before:       0.002930
next_byte_accuracy_after:        0.267090
unigram_eval_loss:               3.307831
bigram_eval_loss:                2.643611
checkpoint_changed:              true
train_eval_exact_doc_overlap:    0
nonempty_generation_rate:        1.000
utf8_replacement_rate:           0.000
repetition_rate:                 0.500
wall_clock_sec:                  79.698
```

The default model beat the unigram control and strongly improved heldout loss, but it was still slightly worse than the bigram control. A longer Deck-compatible confirmation run with `2500` steps, `hidden=256`, and `embed_dim=48` passed more strongly:

```text
train_loss_initial:              5.533380
train_loss_final:                2.193929
eval_loss_before:                5.530833
eval_loss_after:                 2.475269
eval_loss_delta:                 3.055565
eval_perplexity_after:           11.884898
next_byte_accuracy_before:       0.004517
next_byte_accuracy_after:        0.306763
unigram_eval_loss:               3.307831
bigram_eval_loss:                2.643611
checkpoint_changed:              true
train_eval_exact_doc_overlap:    0
nonempty_generation_rate:        1.000
utf8_replacement_rate:           0.000
repetition_rate:                 0.200
wall_clock_sec:                  78.740
```

## Interpretation

This confirms that the local training and inference path can learn a non-synthetic byte-level language modeling task from a real downloaded text corpus on the Deck. The trained model improves heldout loss, improves next-byte accuracy, changes checkpoint weights, avoids train/eval exact-document overlap, and in the confirmation run beats both unigram and bigram controls.

The generated samples remain small-LM byte-text, with repeated common fragments and no assistant-level capability. The result should be read as a training/inference smoke pass, not as open-domain language understanding or assistant readiness.

## Claim Boundary

This is not FineWeb certification, not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not hosted SaaS, not deployment readiness, and not safety alignment.
