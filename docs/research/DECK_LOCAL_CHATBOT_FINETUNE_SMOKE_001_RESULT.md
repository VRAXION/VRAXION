# DECK_LOCAL_CHATBOT_FINETUNE_SMOKE_001 Result

`DECK_LOCAL_CHATBOT_FINETUNE_SMOKE_001` tests whether the Deck-local byte-level AG News text model becomes more chatbot-like after a small supervised assistant-format finetune.

This is a bounded local smoke. It is not GPT-like readiness, not open-domain assistant readiness, not production chat, and not safety alignment.

## Runner

```text
scripts/probes/run_deck_local_chatbot_finetune_smoke_001.py
```

Primary output:

```text
target/pilot_wave/deck_local_chatbot_finetune_smoke_001/smoke
```

No-exact-overlap control:

```text
target/pilot_wave/deck_local_chatbot_finetune_smoke_001/no_exact_overlap
```

## Setup

Base checkpoint:

```text
target/pilot_wave/deck_local_text_lm_smoke_001/extended_2500/checkpoints/deck_local_text_lm/model.pt
```

Finetune task:

```text
User: ...
Assistant: answer<END>
```

The eval compares:

```text
base_extended_text_lm/sampled
assistant_finetuned/greedy
assistant_finetuned/sampled
```

Prompt splits:

```text
bounded_eval
heldout_eval
```

## Main Bounded Run

This run includes the bounded eval prompts in the supervised finetune set. It is therefore a capacity / bounded-behavior smoke, not a generalization result.

```text
before accuracy:              0.0625
after accuracy:               0.71875
accuracy delta:               0.65625
bounded_eval accuracy after:  1.000
heldout_eval accuracy after:  0.250
train/eval exact overlap:     20
after permanent stuck rate:   0.53125
after static output rate:     0.500
after repetition rate:        0.03125
```

Verdict:

```text
ASSISTANT_FORMAT_FINETUNE_IMPROVES_CHAT_SCORE
BOUNDED_PROMPT_CHAT_SMOKE_PASSES
HELDOUT_PROMPT_TRANSFER_WEAK
TRAIN_EVAL_PROMPT_OVERLAP_PRESENT_BOUND_RESULT
FINETUNED_STUCKNESS_RISK
OPEN_DOMAIN_CHATBOT_NOT_CLAIMED
```

Interpretation:

```text
The tiny byte model can memorize / reproduce bounded assistant responses after supervised finetune.
This does not show robust chatbot generalization.
```

## No-Exact-Overlap Control

This run removes exact eval-prompt overlap from the hand-written train examples.

```text
before accuracy:              0.0625
after accuracy:               0.34375
accuracy delta:               0.28125
bounded_eval accuracy after:  0.500
heldout_eval accuracy after:  0.08333
train/eval exact overlap:     0
after permanent stuck rate:   0.15625
after static output rate:     0.15625
after repetition rate:        0.000
```

Verdict:

```text
ASSISTANT_FORMAT_FINETUNE_IMPROVEMENT_WEAK
BOUNDED_PROMPT_CHAT_SMOKE_WEAK
HELDOUT_PROMPT_TRANSFER_WEAK
FINETUNED_STUCKNESS_RISK
OPEN_DOMAIN_CHATBOT_NOT_CLAIMED
```

Interpretation:

```text
Without exact prompt overlap, supervised assistant-format finetune still improves over the base text model,
but the transfer is weak and not yet a regular chatbot result.
```

## What Improved

Compared with the base AG News byte LM, the finetuned model can produce bounded assistant-shaped answers, especially for seen prompt formats:

```text
Hello.
Szia.
ready
Rain is water that falls from clouds.
Nora.
I cannot provide a private API key.
```

This confirms that the model and training path can move from generic text continuation toward prompt-response behavior.

## What Failed

The main failure is transfer:

```text
heldout_eval remains weak
some prompts collapse to canned answers
static response rate rises after finetune
context/conflict/simple-QA transfer is poor
```

The finetune makes the model more assistant-shaped, but also exposes the limits of the current architecture/data:

```text
small byte-level feed-forward LM
96-byte context
tiny synthetic assistant dataset
no pretrained semantics
no robust instruction generalization
```

## Current Answer

For the question:

```text
Does chatbot behavior improve if we adjust / finetune it?
```

The measured answer is:

```text
yes, for bounded and especially seen assistant prompts
weakly, for no-exact-overlap prompts
no, for robust regular-chatbot readiness
```

The bottleneck is no longer simply "can the model learn?". It can. The bottleneck is:

```text
generalizing instruction/chat behavior beyond memorized prompt templates
```

## Next Useful Gate

The next useful test should increase one variable at a time:

```text
1. larger assistant dataset with zero exact eval overlap
2. stronger context architecture or longer context
3. heldout paraphrase families by phenomenon
4. anti-static/canned-answer penalty or decoding guard
```

Do not claim chatbot readiness until heldout transfer and stuckness both pass.

## Claim Boundary

This is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not hosted SaaS, not deployment readiness, and not safety alignment.
