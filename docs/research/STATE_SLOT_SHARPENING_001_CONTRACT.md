# STATE_SLOT_SHARPENING_001 Contract

## Goal

`STATE_BOTTLENECK_001` showed a real but partial bottleneck signal:

```text
oracle state visible: 1.000
direct GRU: ~0.718
GRU bottleneck deterministic: ~0.700-0.703
state slot: ~0.872-0.923
count slot: ~0.727-0.879
shuffled deterministic: ~0.20
```

This probe asks what is needed to make the predicted state slots sharp enough that the answer can be decoded from them.

The locked route remains:

```text
surface story text
-> encoder
-> predicted state slots
-> answer only from predicted state
```

No PrismionCell is tested here. This is a targeted state-slot sharpening probe.

## Data Boundary

Reuse the surface-text story generator from `TOKEN_TO_STATE_UPDATE_VS_LATENT_001` / `STATE_BOTTLENECK_001`.

Model-visible input is naturalized toy text, for example:

```text
I see a dog.
The previous dog gets stolen.
They bring it back.
How many dogs?
```

Do not expose model-visible event IDs such as:

```text
ADD_ENTITY DOG
REMOVE PREVIOUS
RESTORE IT
```

Replayed surface semantics are the state source of truth. Mismatch with older hidden generator labels is reported as `row_answer_match_rate`.

## State Slots

Primary answer state:

```text
counts_by_type[6 x 5]
query_target_type[6]
invalid_restore_flag
impossible_reference_flag
ambiguous_reference_flag
```

No target count slot is allowed.

The deterministic answer decoder is:

```text
answer = counts_by_type[argmax(query_target_type)]
```

## Arms

```text
ORACLE_STATE_VISIBLE
  true state slots -> deterministic answer; sanity upper bound.

GRU_DIRECT_ANSWER
  raw text -> GRU hidden -> answer; baseline only.

STATE_BOTTLENECK_BASE
  previous bottleneck setup.

COUNT_WEIGHT_0P5 / 1P0 / 2P0 / 4P0 / 8P0
  sweep count auxiliary weight.

PER_EVENT_STATE_SUPERVISION
  supervise state snapshots after each event sentence.

DISCRETE_SLOT_PRESSURE
  entropy penalty to sharpen count/query slot distributions.

COUNT_BY_TYPE_ONLY
  train only counts_by_type + query_target; diagnoses count learning.

LIFECYCLE_ONLY
  train only query + lifecycle/flag slots; diagnostic only.

FULL_STATE_STRONG
  count + query + flags + per-event + entropy sharpening.

SHUFFLED_STATE_STRONG
  same as FULL_STATE_STRONG with shuffled state labels; negative control.
```

## Hard Audit

For every state arm report:

```text
soft_state_answer_accuracy
hard_state_answer_accuracy
deterministic_state_decoder_accuracy
```

Strong positive requires the hard/deterministic path to work. A soft-only win is treated as a covert channel risk.

## Required Controls

Report separately:

```text
same_token_set_accuracy
event_order_shuffle_accuracy
coreference_accuracy
invalid_restore_accuracy
distractor_resistance
heldout_composition_accuracy
```

Do not average invalid restore, ambiguous reference, or impossible reference away.

Feature leak audit must pass:

```text
answer head cannot read encoder hidden
answer head cannot read raw tokens
answer head cannot read token embeddings
answer head cannot read target_count
answer head cannot read answer label
```

Allowed answer inputs:

```text
predicted counts_by_type
predicted query_target_type
predicted lifecycle/reference flags
```

## Metrics

```text
answer_accuracy
hard_state_answer_accuracy
deterministic_state_decoder_accuracy
same_token_set_accuracy
event_order_shuffle_accuracy
coreference_accuracy
invalid_restore_accuracy
distractor_resistance
heldout_composition_accuracy
state_slot_accuracy
count_slot_accuracy
query_slot_accuracy
lifecycle_slot_accuracy
count_mae
shuffled_control_gap
soft_vs_hard_gap
entropy_of_count_slots
entropy_of_query_slots
per_event_state_accuracy
```

## Verdicts

```text
STATE_SLOT_SHARPENING_POSITIVE
COUNT_WEIGHT_WAS_BOTTLENECK
PER_EVENT_SUPERVISION_REQUIRED
DISCRETE_SLOT_PRESSURE_REQUIRED
SOFT_BOTTLENECK_COVERT_CHANNEL
SHUFFLED_CONTROL_FAIL
STATE_TARGET_WEAK
STILL_EXPLICIT_LEDGER_REQUIRED
READY_FOR_MIN_PRISMION_CELL
```

Positive requires:

```text
best bottleneck deterministic/hard answer >= GRU_DIRECT_ANSWER + 0.10
count_slot_accuracy >= 0.92
state_slot_accuracy >= 0.93
same_token_set_accuracy >= GRU_DIRECT_ANSWER + 0.10
shuffled control at least 0.15 worse
hard_state_answer does not collapse relative to soft_state_answer
deterministic_state_decoder is close to learned answer head
```

If positive, the next probe is `MIN_PRISMION_CELL_001`.

If negative, report whether the blocker is count slot, query target, lifecycle flags, coreference, invalid restore, or soft covert channel.

## Run Hygiene

No black-box runs are allowed.

Every run writes append-only partial artifacts from the start:

```text
queue.json
progress.jsonl
metrics.jsonl
summary.json
report.md
slot_curve.json
examples_sample.jsonl
contract_snapshot.md
job_progress/<arm>__seed_<seed>.jsonl
```

Required cadence:

```text
parent heartbeat: <= --heartbeat-sec
worker heartbeat: <= --heartbeat-sec during long jobs
epoch progress: every epoch
metrics row: immediately after each completed job
summary/report refresh: immediately after each completed job and heartbeat
```

Default heartbeat is 30 seconds.

`--jobs auto80` means about 80% of available logical CPUs. Each worker uses one Torch thread.

