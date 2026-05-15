# STATE_BOTTLENECK_001 Contract

## Goal

Test whether forcing an intermediate state representation improves object-lifecycle story tracking.

Locked route:

```text
surface story text
-> encoder
-> predicted state slots
-> answer only from predicted state
```

This is a controlled toy state-update probe. It is not open natural-language grounding and not a PrismionCell test.

## Dataset Boundary

Use the same surface story domain as `TOKEN_TO_STATE_UPDATE_VS_LATENT_001`:

```text
I see a dog.
The previous dog gets stolen.
They bring it back.
How many dogs?
```

The model-visible input must remain surface text. Do not feed event IDs such as:

```text
ADD_ENTITY DOG
REMOVE PREVIOUS
RESTORE IT
```

The probe reconstructs full state slots by replaying the surface story. Replayed surface semantics are the state source of truth for this probe. Mismatch against the previous generator answer is reported as `row_answer_match_rate`, because ambiguous forms such as `other X` can expose differences between hidden generator state and readable surface semantics.

## State Bottleneck

State slots:

```text
counts_by_type[6 x 5]
query_target_type[6]
invalid_restore_flag
impossible_reference_flag
ambiguous_reference_flag
```

Do not expose:

```text
answer
target_count
gold label
encoder hidden state to answer head
```

Allowed answer evidence:

```text
counts_by_type + query_target_type + flags
```

The deterministic answer decoder is:

```text
answer = counts_by_type[argmax(query_target_type)]
```

## Arms

```text
EXPLICIT_LEDGER_ORACLE
  replayed deterministic ledger labels; upper bound.

ORACLE_STATE_VISIBLE
  true state slots visible to deterministic decoder; verifies state schema sufficiency.

BAG_OF_TOKENS_MLP
  no order baseline.

STATIC_POSITION_MLP
  position-aware static shortcut baseline.

GRU_DIRECT_ANSWER
  raw tokens -> GRU hidden -> answer.

GRU_STATE_BOTTLENECK
  raw tokens -> GRU hidden -> predicted state slots -> answer.

NEURAL_SLOT_BOTTLENECK
  raw tokens -> GRU hidden -> compact slot vector -> predicted state slots -> answer.

SHUFFLED_STATE_BOTTLENECK
  same route as GRU_STATE_BOTTLENECK but shuffled state labels.
```

For bottleneck arms, evaluate:

```text
soft_state_learned_head_accuracy
hard_state_learned_head_accuracy
deterministic_state_decoder_accuracy
```

Hard state:

```text
counts_by_type = argmax per entity type
query_target_type = argmax
flags = sigmoid >= 0.5
```

No-hidden-bypass rule:

```text
answer_head may read only predicted state slots
answer_head must not read encoder_hidden, token embeddings, raw tokens, or gold answer
```

## Supervision Modes

```text
final_state_only
  state targets only at the final query point.

per_event_state_supervision
  count/flag state snapshots supervised after each event sentence.
  answer still comes only from final predicted state.
```

## Metrics

Report:

```text
answer_accuracy
same_token_set_accuracy
event_order_shuffle_accuracy
coreference_accuracy
invalid_restore_accuracy
distractor_resistance
heldout_composition_accuracy
count_slot_accuracy
query_slot_accuracy
flag_accuracy
state_slot_accuracy
count_mae
soft_vs_hard_gap
deterministic_decoder_gap
shuffled_control_gap
state_replay_audit
row_answer_match_rate
feature_leak_audit
```

## Run Hygiene

No black-box runs are allowed.

Every invocation must write append-only partial artifacts from the start:

```text
queue.json
progress.jsonl
metrics.jsonl
summary.json
report.md
job_progress/<arm>__<mode>__seed_<seed>.jsonl
```

Required cadence:

```text
parent progress heartbeat: <= --heartbeat-seconds
worker training heartbeat: <= --heartbeat-seconds during long jobs
epoch progress: every epoch
metrics row: immediately after each completed job
summary/report refresh: immediately after each completed job and heartbeat
```

If a run crashes, the partial files must still show:

```text
which jobs started
which jobs completed
latest worker epoch/batch progress
last partial aggregate available
```

Default heartbeat is 30 seconds.

Validity gates:

```text
EXPLICIT_LEDGER_ORACLE answer_accuracy >= 0.98
ORACLE_STATE_VISIBLE deterministic_state_decoder_accuracy >= 0.98
BAG_OF_TOKENS_MLP same-token/order controls remain weak
feature_leak_audit == pass
state_replay_audit == pass
```

## Verdicts

```text
STATE_BOTTLENECK_POSITIVE
  deterministic state decoder improves over direct GRU by >= 0.10,
  hard answer stays close to soft answer,
  state slots are accurate,
  same-token/order controls improve,
  shuffled state control fails clearly.

SOFT_BOTTLENECK_COVERT_CHANNEL
  soft answer is good but hard/deterministic answer is weak.

STATE_TARGET_WEAK
  oracle state visible fails.

SHUFFLED_CONTROL_FAIL
  shuffled bottleneck also works.

BOTTLENECK_PARTIAL
  state slots improve but answer/control gates do not fully pass.

EXPLICIT_LEDGER_REQUIRED_FOR_NOW
  oracle passes, learned bottleneck fails.
```

## Claim Boundary

A positive result would support forced state bottlenecks on this toy story domain. It would not prove symbol grounding, natural-language entity tracking, or consciousness.
