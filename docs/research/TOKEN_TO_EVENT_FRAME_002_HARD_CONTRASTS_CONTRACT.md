# TOKEN_TO_EVENT_FRAME_002_HARD_CONTRASTS Contract

## Goal

V1 proved that:

```text
raw clause -> event frame -> identity ledger -> answer
```

works mechanically, but it was too easy: bag-of-tokens and static-position frame baselines also saturated.

V2 builds harder role/order/reference/no-op contrasts where lexical shortcuts should fail before any PrismionCell or learned state updater is attempted.

## Invariants

Keep the v1 frame schema:

```text
event_type: CREATE, REMOVE, RESTORE, QUERY_COUNT, NOOP_OR_INVALID
entity_type: dog, cat, coin, robot, candle, key, NONE
ref_type: NEW, FIRST, SECOND, PREVIOUS, OTHER, IT, NONE
quantity: 0..4
```

`DISTRACTOR_CREATE` remains forbidden. Distractor status is computed only relative to the later query target.

Learned frame arms pass only hard predicted frames to the deterministic identity ledger. There is no soft-hidden or direct-answer bypass in frame arms.

## Hard Contrast Classes

Add targeted stories for:

```text
same-token role swap
invisible/nested affected entity
passive/active affected entity
negation / almost / tried / planned / failed no-op
quote / mention trap
null action verbs
ghost references
restore validity and identity
query target distractors
event order contrasts
first / second / previous / other / it reference contrasts
```

Examples:

```text
The dog watches the robot get stolen. -> REMOVE robot
The robot watches the dog get stolen. -> REMOVE dog
The robot carries the key away. -> REMOVE key
The key is carried away by the robot. -> REMOVE key
The dog was not stolen. -> NOOP_OR_INVALID
The sign says the dog was stolen. -> NOOP_OR_INVALID
```

## Arms

```text
EVENT_FRAME_ORACLE
DIRECT_STORY_GRU_ANSWER
SEGMENTED_EVENT_FRAME_CLASSIFIER
STORY_TO_EVENT_FRAMES_GRU
BAG_OF_TOKENS_EVENT_FRAME
STATIC_POSITION_EVENT_FRAME
SHUFFLED_EVENT_FRAME_TEACHER
```

Use multi-head loss:

```text
1.0 * CE(event_type)
1.0 * CE(entity_type)
1.5 * CE(ref_type)
0.5 * CE(quantity)
```

## Metrics

Report:

```text
event_frame_exact_accuracy
event_type_accuracy
entity_type_accuracy
ref_type_accuracy
quantity_accuracy
ledger_answer_accuracy
illegal_frame_rate
same_token_role_swap_pair_accuracy
affected_entity_accuracy
invisible_entity_target_accuracy
passive_active_contrast_accuracy
negation_noop_accuracy
near_miss_noop_accuracy
modal_noop_accuracy
mention_trap_accuracy
null_action_accuracy
no_mutation_on_noop_accuracy
ghost_reference_accuracy
invalid_restore_accuracy
identity_restore_accuracy
previous_vs_other_accuracy
event_order_contrast_accuracy
same_token_story_pair_accuracy
query_target_accuracy
distractor_resistance
coreference_accuracy
```

Report confusion matrices for event type, entity type, ref type, and quantity.

## Verdicts

```text
EVENT_FRAME_HARD_POSITIVE
  segmented exact >= 0.90
  segmented ledger >= 0.90
  role swap pair >= 0.85
  affected entity >= 0.85
  no-op/negation/mention traps >= 0.85
  ref_type >= 0.85
  beats BAG/STATIC on hard contrasts
  shuffled collapses

EVENT_FRAME_LEXICAL_SHORTCUT
  BAG/STATIC match segmented on hard contrast metrics

ROLE_BINDING_BOTTLENECK
  event/entity good but affected entity or role swap poor

REFERENCE_BINDING_BOTTLENECK
  ref_type/coreference poor

NEGATION_MODAL_BOTTLENECK
  negation/almost/tried/mention traps fail

EVENT_FRAME_SEGMENTATION_BOTTLENECK
  segmented passes, story-to-frames fails

EVENT_FRAME_WEAK
  segmented learned frame accuracy poor

LEDGER_UPDATE_BOTTLENECK
  frames good but ledger answer poor
```

If `EVENT_FRAME_HARD_POSITIVE`, the next run is `STATE_UPDATER_FROM_EVENT_FRAMES_001`.

If not, do not build PrismionCell yet.

## Run Hygiene

No black-box runs.

Generated outputs:

```text
queue.json
progress.jsonl
metrics.jsonl
summary.json
report.md
frame_curve.json
hard_contrast_cases.jsonl
examples_sample.jsonl
contract_snapshot.md
job_progress/*.jsonl
```

Use `--jobs auto85`, one Torch thread per worker, and continuous heartbeat/progress logging.

