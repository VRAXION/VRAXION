# TOKEN_TO_EVENT_FRAME_003_ANTI_STATIC Contract

## Goal

Test whether raw clause -> hard event-frame parsing survives when static position and template shortcuts are broken.

The target is not another high parser score. A positive result requires:

```text
SEGMENTED parser stays strong
STATIC_POSITION drops on hard contrasts
BAG drops on hard contrasts
POSITION_ONLY stays weak
SHUFFLED collapses
ORACLE validates ledger
```

Do not build PrismionCell. Do not replace the deterministic identity ledger. Do not add `DISTRACTOR_CREATE`.

## Schema

Keep the v2 frame schema:

```text
event_type: CREATE, REMOVE, RESTORE, QUERY_COUNT, NOOP_OR_INVALID
entity_type: dog, cat, coin, robot, candle, key, NONE
ref_type: NEW, FIRST, SECOND, PREVIOUS, OTHER, IT, NONE
quantity: 0..4
```

Learned frame arms must pass only hard predicted frames to the deterministic identity ledger. There is no soft-hidden or direct-answer bypass in frame arms.

## Dataset

Use deterministic templates only. Every generated clause carries `TemplateSpec` metadata:

```text
template_family
target_position_tag
train_allowed
heldout_eval
contrast_group_id
hardening_round_added
```

Use two eval slices and report them separately:

```text
in_distribution_hard
heldout_template
```

Required hard families:

```text
same-token role swap
target position invariance
active/passive contrast
cleft/focus
relative clause
fronted subordinate
target late
distractor before target
target before distractor
negation / almost / tried / planned / failed no-op
quote / mention traps
null action verbs
ghost references
restore identity
query target distractors
event order contrasts
```

## Arms

```text
EVENT_FRAME_ORACLE
DIRECT_STORY_GRU_ANSWER
SEGMENTED_EVENT_FRAME_CLASSIFIER
STORY_TO_EVENT_FRAMES_GRU
BAG_OF_TOKENS_EVENT_FRAME
STATIC_POSITION_EVENT_FRAME
POSITION_ONLY_EVENT_FRAME
SHUFFLED_EVENT_FRAME_TEACHER
```

## Adaptive Hardening

`--adaptive-hardening` runs bounded hardening rounds.

Stop conditions:

```text
STATIC hard_contrast_score <= 0.70 and SEGMENTED hard_contrast_score >= 0.85
  -> allow full run

SEGMENTED hard_contrast_score < 0.70
  -> PARSER_WEAK_UNDER_SHUFFLE

max hardening rounds reached
  -> STILL_STATIC_SOLVABLE
```

Write `hardening_rounds.jsonl`, `per_round_summary.json`, `template_pool_snapshot_round_N.json`, and `position_leak_audit.json`.

## Metrics

Report frame and ledger metrics:

```text
event_frame_exact_accuracy
event_type_accuracy
entity_type_accuracy
ref_type_accuracy
quantity_accuracy
ledger_answer_accuracy
```

Report anti-shortcut metrics:

```text
in_distribution_hard_accuracy
heldout_template_accuracy
target_position_invariance_accuracy
same_token_role_swap_pair_accuracy
pair_accuracy
per_member_accuracy
position_only_hard_contrast_score
bag_hard_contrast_score
static_hard_contrast_score
static_simple_clause_score
noop_frame_accuracy
noop_no_mutation_accuracy
negation_noop_accuracy
near_miss_noop_accuracy
modal_noop_accuracy
mention_trap_accuracy
ghost_reference_accuracy
invalid_restore_accuracy
identity_restore_accuracy
previous_vs_other_accuracy
event_order_contrast_accuracy
query_target_accuracy
distractor_resistance
```

Report position leak tables:

```text
event_type -> target_position_bucket
entity_type -> target_position_bucket
ref_type -> target_position_bucket
template_family -> target_position_bucket
```

## Verdicts

```text
EVENT_FRAME_ANTI_STATIC_POSITIVE
STILL_STATIC_SOLVABLE
POSITION_LEAK_WARNING
TEMPLATE_GENERALIZATION_WEAK
PARSER_WEAK_UNDER_SHUFFLE
ROLE_BINDING_BOTTLENECK
REFERENCE_BINDING_BOTTLENECK
NEGATION_MODAL_BOTTLENECK
EVENT_FRAME_SEGMENTATION_BOTTLENECK
LEDGER_UPDATE_BOTTLENECK
```

## Run Hygiene

No black-box runs.

Required outputs:

```text
queue.json
progress.jsonl
metrics.jsonl
summary.json
report.md
frame_curve.json
hard_contrast_cases.jsonl
hardening_rounds.jsonl
per_round_summary.json
template_pool_snapshot_round_N.json
position_leak_audit.json
examples_sample.jsonl
contract_snapshot.md
job_progress/*.jsonl
```

Use `--jobs auto85`, one Torch thread per worker, and continuous heartbeat/progress logging.
