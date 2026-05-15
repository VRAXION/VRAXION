# TOKEN_TO_STATE_UPDATE_VS_LATENT_001 Contract

## Goal

Test whether short story state updates require an explicit entity ledger, or whether a latent recurrent state can learn the same lifecycle/count behavior from raw story tokens.

Narrow question:

```text
raw story tokens
-> entity lifecycle state
-> queried count / answer
```

This is a toy state-update probe. It is not a symbol-grounding proof, not open natural-language understanding, and not a full PrismionCell test.

## Dataset

Entities:

```text
dog
cat
coin
robot
candle
key
```

Events:

```text
see/add
add_more
remove
restore
distractor_arrives
query_count
```

References:

```text
first
second
previous
other
it
```

Locked semantics:

```text
restore removed entity -> present again
restore entity that was not removed -> no-op + invalid_restore=true
missing/ambiguous references are tagged separately
query target type varies
answer is the count for the queried type only, not total entity count
```

Mandatory adversarial coverage:

```text
same-token-set order contrasts
event-order-shuffled contrasts
invalid restore cases
distractor entity queries
heldout noun/verb surfaces
coreference flips
```

## Arms

```text
EXPLICIT_LEDGER_ORACLE
  deterministic generator ledger labels; upper-bound sanity.

BAG_OF_TOKENS_MLP
  no order; should fail same-token-set contrasts.

STATIC_POSITION_MLP
  position-aware static model; detects position/template shortcut risk.

LATENT_GRU_ANSWER_ONLY
  raw story tokens -> answer only.

LATENT_GRU_FROZEN_LINEAR_PROBES
  train answer-only GRU, freeze encoder, then train linear probes on hidden states.

HYBRID_STATE_TEACHER
  raw story tokens -> answer + state auxiliary targets during training.
  Eval receives raw story only.

SHUFFLED_STATE_TEACHER
  same as hybrid, but shuffled/wrong state labels.
```

State targets:

```text
current_count_by_type
entity_presence_bits
removed_bits
restored_bits
last_reference_target
query_target_type
valid_restore_flag
ambiguous_reference_flag
```

## Metrics

Report:

```text
answer_accuracy
heldout_composition_accuracy
same_token_set_accuracy
event_order_shuffle_accuracy
coreference_accuracy
distractor_resistance
invalid_restore_accuracy
ambiguous_reference_accuracy
count_mae
linear_probe_count_accuracy
entity_presence_probe_accuracy
lifecycle_probe_accuracy
```

Validity gates:

```text
EXPLICIT_LEDGER_ORACLE accuracy >= 0.98
BAG_OF_TOKENS_MLP same-token-set accuracy <= 0.65
feature_leak_audit == pass
invalid/ambiguous cases reported separately
```

Positive verdicts:

```text
LATENT_STATE_POSITIVE
  LATENT_GRU answer >= 0.90 on heldout
  frozen linear count probe >= 0.90
  lifecycle probe >= 0.85
  same-token-set and order-shuffle controls pass

HYBRID_STATE_SUPERVISION_POSITIVE
  HYBRID answer >= LATENT_GRU + 0.10
  HYBRID state probes improve meaningfully
  HYBRID counterfactual pair accuracy improves meaningfully
  HYBRID beats SHUFFLED_STATE_TEACHER by >= 0.15

STATIC_SHORTCUT_WARNING
  STATIC_POSITION_MLP matches GRU while probes fail or controls are weak

SHORTCUT_OR_OPAQUE_SUCCESS
  final answers look good, but frozen linear probes fail

EXPLICIT_LEDGER_REQUIRED_FOR_NOW
  oracle passes, learned arms fail the latent/hybrid gates
```

## Claim Boundary

This probe tests a small controlled story grammar. A positive result says that the chosen carrier learned or exposed useful state-update structure on this toy suite. It does not prove open-ended language grounding, consciousness, or natural-language entity tracking at scale.
