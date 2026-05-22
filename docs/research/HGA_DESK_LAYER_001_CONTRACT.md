# HGA-DESK-LAYER-001 / DeskCache Layer Ablation Contract

Status: `preregistered_target_only_layer_probe`

This probe tests which `AnchorCellCore` layer, if any, moves local model
preference from object-category / storage search toward active next-step search
on the existing `HGA-DESK-FAMILY-001` six-cell substrate.

This is not training, not an `AnchorWeave-v1.0` export, and not a validated
standard-cell collection.

## Claim Boundary

- Positive evidence means a local operational micro-signal for one or more
  model-facing AnchorCell layers.
- It does not prove grounding at scale, consciousness, or general desk-search
  ability.
- Exact target selection is diagnostic. The primary movement signal is the
  preregistered directional family: exact active target or active-use near miss.
- Free-response remains a manual gate. Automated scoring cannot finalize a
  strong positive without annotation.
- Generated outputs must stay under `target/`.
- Do not write under `data/anchorweave/cells/`.

## Substrate

The only allowed substrate is `HGA-DESK-FAMILY-001`.

The runner must fail early unless it can audit:

```text
family_manifest_path
cell_ids
candidate_family_count
canonical/paraphrase availability
candidate semantic families
```

Candidate text must use neutral site labels, for example:

```text
Check site N1 first; if it is not there, continue with the remaining desk search.
```

Candidate text must not contain:

```text
active-use
work-ready
ready-to-use
diagnostic
storage
trap
```

## Prompt Arms

```text
BASE
STYLE_CONTROL
STYLE_CONTROL_LEN_MATCHED
L1_IMPLICIT_JOB
L2_SALIENCE_MAP
L3_ACTION_OUTCOME_MAP
L4_DECISION_RULE
L5_FULL_DECISION_MAP
CORRUPTED_DECISION_MAP
OLD_INNER_VOICE_DIAGNOSTIC
```

Ablation semantics:

```text
L1-L4 = isolated single-layer arms
L5 = cumulative L1 + L2 + L3 + L4
OLD_INNER_VOICE_DIAGNOSTIC = previous negation-heavy monologue, diagnostic only
```

The correct layer arms should use positive framing:

```text
next work step -> low-handling place -> item can help the task begin
```

They should avoid long trap-noun exclusion lists.

## Semantic Families

Every candidate must have preregistered metadata:

```text
active_use_gold
active_use_near_miss
storage_surface
storage_container
small_object_clutter
dirty_area
personal_boundary
```

Scoring families:

```text
primary_success = active_use_gold
directional_success = active_use_gold OR active_use_near_miss
bad_policy = storage_surface OR storage_container OR small_object_clutter OR dirty_area OR personal_boundary
```

## Metrics

Token counts per arm:

```text
prompt_tokens
layer_tokens
total_tokens
```

Primary automated metric:

```text
cell_level_first_action_directional_rate
```

Secondary metrics:

```text
active_use_success_rate
bad_policy_first_action_rate
mean_cascade_policy_utility
found_within_2_steps_rate
exact_target_rate
```

Choices-only is cell-level, not seed-level:

```text
For each cell:
  average first-action directional success over order seeds and candidate families.
```

Choices-only guard:

```text
warning if cell-level directional rate > 0.35
invalid if cell-level directional rate > 0.40
```

Report order-seed volatility separately.

## Valid Cells

A cell is valid for automated layer comparison if:

```text
choices-only does not invalidate it
canonical + paraphrase candidate families exist
required arms complete forced-choice and cascade scoring
forced-choice or cascade metrics are interpretable
```

A full probe must include free-response. If free-response is skipped, final
status cannot be `LAYER001_STRONG_SIGNAL`.

## Useful Layer Gate

A layer is `USEFUL` only if all automated gates pass:

```text
valid_cells >= 4 / 6
cell-level first_action_directional_rate >= BASE + 0.20
cell-level first_action_directional_rate >= STYLE_CONTROL + 0.15
bad_policy_first_action_rate <= BASE - 0.15
mean_cascade_policy_utility > max(BASE, STYLE_CONTROL)
positive direction in canonical and paraphrase families
```

Manual gate still required:

```text
no task_frame_drift under that layer in free-response
```

## Strong Signal Gate

`LAYER001_STRONG_SIGNAL` requires:

```text
at least one non-corrupted layer is USEFUL
best useful layer beats CORRUPTED on directional rate and utility
CORRUPTED shifts toward bad policy families
choices-only is not invalid
free-response confirms the same directional family
OLD_INNER_VOICE_DIAGNOSTIC is not the best useful layer
```

If old inner voice is best:

```text
LAYER001_OLD_INNER_VOICE_CONFLICT
```

Other statuses:

```text
LAYER001_WEAK_LAYER_SIGNAL
LAYER001_NEGATIVE
LAYER001_INVALID_CHOICES_ONLY
LAYER001_NEEDS_MANUAL_FREE_RESPONSE
LAYER001_RESOURCE_BLOCKED
```

