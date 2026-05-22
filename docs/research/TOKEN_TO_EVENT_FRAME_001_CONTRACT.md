# TOKEN_TO_EVENT_FRAME_001 Contract

## Goal

Test the smaller Black Box A step:

```text
raw event clause -> hard event frame -> deterministic identity ledger -> answer
```

This follows `STATE_SLOT_SHARPENING_001`, where direct final-state prediction from full raw stories stayed too noisy.

This probe is not a PrismionCell test, not full segmentation, and not symbol grounding.

## Critical Rule

`DISTRACTOR_CREATE` is not an event type.

A clause such as:

```text
A cat arrives.
```

is simply:

```text
event_type = CREATE
entity_type = cat
```

It becomes a distractor only later, relative to the query target.

## Event Frame

```python
@dataclass(frozen=True)
class EventFrame:
    event_type: str   # CREATE, REMOVE, RESTORE, QUERY_COUNT, NOOP_OR_INVALID
    entity_type: str  # dog, cat, coin, robot, candle, key, NONE
    ref_type: str     # NEW, FIRST, SECOND, PREVIOUS, OTHER, IT, NONE
    quantity: int     # 0..4
```

## Identity Ledger

```python
@dataclass
class EntityRecord:
    eid: int
    entity_type: str
    present: bool = True
    removed: bool = False
    created_step: int = 0
    last_mentioned_step: int = 0
    removed_step: int | None = None
```

Ledger state:

```text
entities
next_eid
step
last_mentioned_eid
last_created_eid
previous_created_eid_by_type
last_removed_eid_by_type
query_target_type
flags: impossible_reference, ambiguous_reference, invalid_restore, illegal_frame
```

Do not use a simple removed stack. Restore and reference resolution must preserve entity identity.

## Arms

```text
EVENT_FRAME_ORACLE
  gold frames -> deterministic ledger -> answer.

DIRECT_STORY_GRU_ANSWER
  raw full story -> answer baseline.

SEGMENTED_EVENT_FRAME_CLASSIFIER
  gold clause boundaries; each raw clause -> hard frame -> ledger.

STORY_TO_EVENT_FRAMES_GRU
  full story with clause separators -> per-clause hard frames -> ledger.
  No direct answer head.

BAG_OF_TOKENS_EVENT_FRAME
  raw clause bag -> event frame baseline.

STATIC_POSITION_EVENT_FRAME
  raw clause position features -> event frame baseline.

SHUFFLED_EVENT_FRAME_TEACHER
  segmented classifier with shuffled frame labels.
```

## Model Loss

Use separate heads:

```text
loss = 1.0 * CE(event_type)
     + 1.0 * CE(entity_type)
     + 1.5 * CE(ref_type)
     + 0.5 * CE(quantity)
```

`ref_type` is expected to be the hard channel and must be visible in diagnostics.

## Ledger Rules

Frame validation before update:

```text
CREATE requires entity_type != NONE
QUERY_COUNT requires entity_type != NONE
REMOVE/RESTORE cannot use ref_type NEW
CREATE should use ref_type NEW/NONE
quantity > 0 is meaningful only for CREATE
illegal frame -> NOOP_OR_INVALID + illegal_frame flag
```

Reference resolution:

```text
FIRST    -> first created entity of type
SECOND   -> second created entity of type
PREVIOUS -> compatible previous/last-mentioned entity of type
OTHER    -> unique other entity if resolvable, otherwise ambiguous
IT       -> compatible last-mentioned or last-removed entity
NEW/NONE -> no existing target unless NONE can resolve a unique typed entity
```

If resolution is ambiguous:

```text
ambiguous_reference += 1
no mutation
```

If resolution is impossible:

```text
impossible_reference += 1
no mutation
```

Invalid restore:

```text
restore entity that was not removed = no-op + invalid_restore
```

## Adversarial Stressors

Add report-only stressors:

```text
null/no-op actions:
  The dog sits.
  The robot shines.
  The cat sleeps.

ghost references:
  I see one dog. The second dog is stolen.

nested target diagnostic:
  The dog watches the robot get stolen.
```

`nested_target_accuracy` is diagnostic only in v1.

## Metrics

```text
event_frame_exact_accuracy
event_type_accuracy
entity_type_accuracy
ref_type_accuracy
quantity_accuracy
ledger_answer_accuracy
same_token_set_accuracy
event_order_contrast_accuracy
coreference_accuracy
distractor_resistance
heldout_verb_accuracy
heldout_reference_accuracy
heldout_noun_accuracy_diagnostic
invalid_restore_accuracy
impossible_reference_accuracy
ambiguous_reference_accuracy
illegal_frame_rate
null_action_accuracy
ghost_reference_accuracy
nested_target_accuracy
no_mutation_on_noop_accuracy
frame_to_answer_gap
direct_answer_vs_frame_ledger_gap
```

Report confusion matrices:

```text
event_type_confusion
entity_type_confusion
ref_type_confusion
quantity_confusion
```

## Verdicts

```text
EVENT_FRAME_POSITIVE
  segmented exact frame >= 0.90
  segmented ledger answer >= 0.90
  hard predicted frames work
  shuffled collapses

EVENT_FRAME_SEGMENTATION_BOTTLENECK
  segmented passes, story-to-frames fails

EVENT_FRAME_LEXICAL_SHORTCUT
  BAG/STATIC also solve story-level same-token/order/coreference/invalid cases

EVENT_FRAME_WEAK
  learned frame accuracy poor

LEDGER_UPDATE_BOTTLENECK
  frames good but ledger answer poor

EXPLICIT_EVENT_PARSER_REQUIRED_FOR_NOW
  oracle passes, learned frame predictors fail
```

## Run Hygiene

No black-box runs.

Generated outputs:

```text
target/pilot_wave/token_to_event_frame_001/
  queue.json
  progress.jsonl
  metrics.jsonl
  summary.json
  report.md
  frame_curve.json
  examples_sample.jsonl
  contract_snapshot.md
  job_progress/*.jsonl
```

Progress:

```text
progress.jsonl from start
metrics.jsonl after each completed job
summary.json/report.md refreshed on heartbeat and job completion
job_progress/*.jsonl with epoch/batch status
```

CPU:

```text
--jobs auto85
jobs = max(1, floor(os.cpu_count() * 0.85))
torch threads per worker = 1
```

## Claim Boundary

V1 assumes gold clause segmentation for the segmented arm. It tests raw clause to event frame, not full natural-language segmentation. It does not prove symbol grounding.

