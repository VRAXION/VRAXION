# TOKEN_TO_EVENT_FRAME_004_CURRICULUM_FACTORS Contract

## Purpose

V3 showed that anti-static syntactic shuffle finally weakened the shortcut baselines, but the learned segmented parser also collapsed. V4 isolates that failure by suite instead of mixing every hard factor at once.

Core question:

```text
raw surface clause -> hard event frame
```

Which factor breaks first?

- role binding / affected entity
- active/passive target detection
- negation, modal, near-miss, quote, and mention no-op handling
- heldout template transfer
- interaction between otherwise separable skills

This is not a PrismionCell test. The deterministic identity ledger remains the state/update oracle.

## Frame Route

```text
surface clause/story
  -> hard predicted event frame(s)
  -> deterministic identity-aware ledger
  -> answer
```

Learned event-frame arms must pass only hard frames into the ledger. There is no hidden-state answer bypass in frame arms.

Frame schema:

```text
event_type: CREATE, REMOVE, RESTORE, QUERY_COUNT, NOOP_OR_INVALID
entity_type: dog, cat, coin, robot, candle, key, NONE
ref_type: NEW, FIRST, SECOND, PREVIOUS, OTHER, IT, NONE
quantity: 0, 1, 2, 3, 4
```

`DISTRACTOR_CREATE` is forbidden. A clause does not know whether it is a distractor until the query target is known.

## Suites

### role_binding

Tests whether the parser can identify the affected entity rather than the observer.

Examples:

```text
The dog watches the robot get stolen. -> REMOVE robot
The robot watches the dog get stolen. -> REMOVE dog
The robot carries the key away. -> REMOVE key
The key is carried away by the robot. -> REMOVE key
```

### negation_modal

Tests lifecycle no-op inhibition.

Examples:

```text
The dog was stolen. -> REMOVE dog
The dog was not stolen. -> NOOP_OR_INVALID
The dog was almost stolen. -> NOOP_OR_INVALID
They tried to steal the dog. -> NOOP_OR_INVALID
They planned to steal the dog. -> NOOP_OR_INVALID
They failed to steal the dog. -> NOOP_OR_INVALID
The sign says the dog was stolen. -> NOOP_OR_INVALID
The word stolen appears next to the dog. -> NOOP_OR_INVALID
```

### template_curriculum

Adds syntactic families one layer at a time.

```text
round 0: active_simple + basic_passive
round 1: fronted_subordinate
round 2: relative_clause
round 3: cleft_focus
round 4: target_late
round 5: distractor-before/after-target
round 6: quote/mention + modal near-miss
```

### combined_recheck

Recombines the isolated factors into a mixed suite. If isolated suites pass but this fails, the blocker is factor interaction.

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

Controls remain required because a suite is only positive if the segmented parser wins while `STATIC`, `BAG`, and `POSITION_ONLY` do not match it on that suite's hard metrics.

## Metrics

Report per suite, arm, seed, and curriculum round:

```text
event_frame_exact_accuracy
ledger_answer_accuracy
role_binding_accuracy
affected_entity_accuracy
pair_accuracy
passive_active_contrast_accuracy
noop_frame_accuracy
noop_no_mutation_accuracy
negation_noop_accuracy
near_miss_noop_accuracy
modal_noop_accuracy
mention_trap_accuracy
heldout_template_accuracy
bag_hard_contrast_score
static_hard_contrast_score
position_only_hard_contrast_score
first_collapse_round
```

Also preserve V3 head-level frame metrics and confusion matrices where available.

## Verdicts

```text
ROLE_BINDING_POSITIVE
ROLE_BINDING_BOTTLENECK
NEGATION_MODAL_POSITIVE
NEGATION_MODAL_BOTTLENECK
TEMPLATE_CURRICULUM_POSITIVE
TEMPLATE_COLLAPSE_AT_ROUND_N
COMBINED_RECHECK_POSITIVE
STATIC_SHORTCUT_RECOVERED
BAG_SHORTCUT_RECOVERED
PARSER_WEAK_UNDER_CURRICULUM
LEDGER_UPDATE_BOTTLENECK
```

## Validity

Required:

```text
EVENT_FRAME_ORACLE ledger_answer_accuracy >= 0.98
feature_leak_audit == pass
no event IDs in model input
no answer label / target_count exposure
frame arms use hard predicted frames only
```

## Run Hygiene

No black-box runs. The runner must write:

```text
queue.json
progress.jsonl
metrics.jsonl
summary.json
report.md
curriculum_curve.json
suite_curve.json
hard_contrast_cases.jsonl
examples_sample.jsonl
contract_snapshot.md
job_progress/*.jsonl
```

Raw `target/` outputs are not committed.
