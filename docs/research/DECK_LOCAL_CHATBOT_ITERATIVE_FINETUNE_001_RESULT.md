# DECK_LOCAL_CHATBOT_ITERATIVE_FINETUNE_001 Result

`DECK_LOCAL_CHATBOT_ITERATIVE_FINETUNE_001` continues training from a saved Deck-local assistant-finetuned checkpoint and evaluates after every cycle.

This answers:

```text
If we keep a persistent checkpoint and iteratively train it, does chatbot behavior keep improving?
```

This is a bounded local probe. It is not GPT-like readiness, not open-domain assistant readiness, not production chat, and not safety alignment.

## Runner

```text
scripts/probes/run_deck_local_chatbot_iterative_finetune_001.py
```

Default output:

```text
target/pilot_wave/deck_local_chatbot_iterative_finetune_001/smoke
```

Source checkpoint:

```text
target/pilot_wave/deck_local_chatbot_finetune_smoke_001/no_exact_overlap/checkpoints/deck_local_chatbot_finetune_smoke/model.pt
```

If that checkpoint is missing, the runner falls back to:

```text
target/pilot_wave/deck_local_text_lm_smoke_001/extended_2500/checkpoints/deck_local_text_lm/model.pt
```

## Setup

The run uses the zero-exact-overlap assistant training examples from the finetune smoke. It intentionally avoids exact eval-prompt overlap:

```text
train_eval_exact_prompt_overlap_count: 0
```

Default run:

```text
cycles:          8
steps_per_cycle: 500
total_steps:     4000
```

Every cycle writes:

```text
checkpoints/cycle_000/model.pt
checkpoints/cycle_001/model.pt
...
checkpoints/cycle_008/model.pt
checkpoints/permanent_candidate/model.pt
```

The permanent candidate is selected by a score that rewards heldout/bounded accuracy and penalizes stuck/static/repetition behavior.

## Result

Status: `recorded`

Verdicts:

```text
ITERATIVE_CHATBOT_FINETUNE_RECORDED
PERMANENT_CANDIDATE_CHECKPOINT_SELECTED
HELDOUT_TRANSFER_DOES_NOT_IMPROVE_MATERIALLY
BOUNDED_CHAT_SCORE_PLATEAUS
BEST_CHECKPOINT_STUCKNESS_RISK
OPEN_DOMAIN_CHATBOT_NOT_CLAIMED
GPT_LIKE_READINESS_NOT_CLAIMED
PRODUCTION_CHAT_NOT_CLAIMED
```

Best selected checkpoint:

```text
cycle:                 0
decode mode:           sampled
bounded_accuracy:      0.550
heldout_accuracy:      0.250
overall_accuracy:      0.438
permanent_stuck_rate:  0.125
static_output_rate:    0.125
selection_score:       0.28625
```

Permanent candidate path:

```text
target/pilot_wave/deck_local_chatbot_iterative_finetune_001/smoke/checkpoints/permanent_candidate/model.pt
```

Important: the selected best checkpoint is `cycle_000`, meaning the starting persistent checkpoint was better than later continuation cycles under the conservative selection score.

## Trajectory

```text
cycle  mode     overall  bounded  heldout  stuck   static  score
0      greedy   0.344    0.500    0.083    0.156   0.156   0.063
0      sampled  0.438    0.550    0.250    0.125   0.125   0.286
1      greedy   0.406    0.500    0.250    0.125   0.125   0.269
1      sampled  0.344    0.400    0.250    0.219   0.219   0.117
2      greedy   0.375    0.550    0.083    0.094   0.094   0.159
2      sampled  0.406    0.600    0.083    0.156   0.156   0.098
3      greedy   0.375    0.500    0.167    0.188   0.156   0.107
3      sampled  0.312    0.450    0.083    0.156   0.156   0.046
4      greedy   0.344    0.500    0.083    0.188   0.188   0.024
4      sampled  0.375    0.550    0.083    0.188   0.188   0.041
5      greedy   0.344    0.500    0.083    0.156   0.156   0.063
5      sampled  0.344    0.500    0.083    0.062   0.062   0.180
6      greedy   0.406    0.550    0.167    0.156   0.156   0.164
6      sampled  0.375    0.550    0.083    0.219   0.219   0.002
7      greedy   0.344    0.500    0.083    0.156   0.156   0.063
7      sampled  0.375    0.500    0.167    0.219   0.219   0.068
8      greedy   0.375    0.500    0.167    0.062   0.062   0.264
8      sampled  0.312    0.400    0.167    0.062   0.062   0.229
```

## Interpretation

Iterative continuation did not materially improve heldout transfer.

The model already reached the useful part of this tiny dataset in the previous no-exact-overlap finetune. Additional cycles mostly produce small fluctuations:

```text
heldout accuracy never exceeds the starting sampled checkpoint's 0.250
bounded accuracy moves between 0.400 and 0.600
static/stuck risk remains nonzero
```

The main qualitative failure is canned-response mixing. On new prompts, the model often emits fragments such as:

```text
ready
Hello.
I cannot claim ...
Assistant: ...
```

instead of reliably mapping the instruction to the correct answer.

## Current Answer

For the question:

```text
If we keep a permanent checkpoint and iteratively train it, does it keep getting better as chat AI?
```

The measured answer is:

```text
not with this architecture/data setup
```

It can learn bounded assistant-shaped responses, but repeated training on the same small assistant dataset does not unlock robust heldout chatbot behavior.

## Bottleneck

The bottleneck is now clear:

```text
not basic learning
not checkpoint persistence
not nonempty generation
```

The bottleneck is:

```text
generalizing instruction-response behavior beyond the tiny prompt template set
```

Likely causes:

```text
small byte-level feed-forward LM
96-byte context
tiny synthetic assistant dataset
no semantic pretraining beyond AG News byte continuation
no sequence model memory beyond fixed context window
```

## Next Useful Gate

The next useful test should change data or architecture, not simply keep iterating on the same small dataset:

```text
1. larger zero-overlap assistant dataset
2. phenomenon-tagged heldout paraphrase suite
3. longer context / recurrent or transformer sequence model
4. anti-static/canned-response objective
```

## Claim Boundary

This is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not hosted SaaS, not deployment readiness, and not safety alignment.
