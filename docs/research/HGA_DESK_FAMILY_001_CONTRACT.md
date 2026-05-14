# HGA-DESK-FAMILY-001 / Active-Use Search Family Contract

Status: `preregistered_target_only_family_probe`

This family tests whether a leak-safe human anchor shifts local model preference
from storage/category search toward active-use / next-work-step search across
multiple sibling desk-search cells.

This is not training, not an `AnchorWeave-v1.0` JSON export, and not a validated
standard-cell collection.

## Claim Boundary

- Positive evidence means an operational micro-signal: content-sensitive
  movement toward active-use search across a small sibling family.
- It does not prove grounding at scale, consciousness, or general desk-search
  ability.
- Exact target selection is diagnostic only. The primary target is the
  active-use success family: exact target, next-step work area, or active-use
  near miss.
- Choices-only and free-response guards remain mandatory.
- Generated outputs must stay under `target/`.
- Do not write under `data/anchorweave/cells/`.

## Mechanism Under Test

```text
CORRECT_INNER_VOICE:
  assistant intent
  not storage/category first
  avoid sorting/cleaning/private boundaries
  choose the clean active-use next-step area

CORRUPTED_INNER_VOICE:
  object category first
  usual holder/storage/container first
  broaden through storage/clutter before active-use reasoning
```

## Cells

| cell_id | object | active-use target | surface trap |
|---|---|---|---|
| `HGA-DESK-003-S03R` | security pass | flat black plate / central work area | pass holder |
| `HGA-DESK-004-S04` | signing pen | top page on clipboard / document area | pen cup |
| `HGA-DESK-005-S05` | approval stamp | small pad beside page / document packet | desk drawer |
| `HGA-DESK-006-S06` | presenter clicker | front edge of open laptop / laptop area | electronics pouch |
| `HGA-DESK-007-S07` | charging adapter | clear spot beside test device / charging area | cable box |
| `HGA-DESK-008-S08` | label sticker | flat spot beside package / label output area | stationery tray |

Each cell has the same four prompt arms:

```text
BASE
STYLE_CONTROL
CORRECT_INNER_VOICE
CORRUPTED_INNER_VOICE
```

## Candidate Categories

Primary success:

```text
gold
active_use_near_miss
```

Bad policy:

```text
surface_association
storage
small_object_clutter
dirty_storage_like
dirty_violation
personal_boundary
```

Exact target remains diagnostic and is recorded separately. Runtime candidate
wording uses neutral site labels such as `site N1`, while the Situation prompt
maps each site label to a physical desk place. The candidate list alone must
not reveal which site is active-use.

## Evaluation

Use deterministic candidate-token NLL scoring:

```text
Best first search plan: <candidate_text>
Best next search action: <candidate_text>
```

Evaluation views:

```text
forced_choice_nll: diagnostic
sequential_search_cascade: primary automated view
choices_only_baseline
free_response_audit
canonical + paraphrase candidate families
```

Order seeds:

```text
2026
2027
2028
2029
2030
```

Manual free-response categories:

```text
goldish_active_use_exact
active_use_near_miss
surface_storage
storage_container
small_object_clutter
dirty_or_liquid_area
personal_boundary
task_frame_drift
unavailable_shortcut
other
```

## Family Pass Gates

The family is positive only if all automated gates pass and manual
free-response review does not find task-frame drift under `CORRECT_INNER_VOICE`.

Automated gates:

```text
valid_cells >= 5 / 6
choices-only first-action active-use rate <= 0.40
CORRECT active_use_success_rate >= STYLE + 0.25
CORRECT active_use_success_rate >= BASE + 0.20
CORRECT active_use_success_rate >= CORRUPTED + 0.40
CORRUPTED bad_policy_rate >= CORRECT + 0.30
CORRECT cascade_policy_utility > max(BASE, STYLE)
CORRECT found_within_2_steps_rate >= 0.70
effect survives canonical and paraphrase candidate families directionally
```

Choices-only first-action active-use rates above `0.35` are reported as a
candidate-prior warning. Rates above `0.40` invalidate the family probe.

Final statuses:

```text
FAMILY001_PROBE_POSITIVE
FAMILY001_PROBE_NEGATIVE
FAMILY001_PROBE_INVALID_CHOICES_ONLY
FAMILY001_PROBE_NEEDS_MANUAL_FREE_RESPONSE
FAMILY001_PROBE_RESOURCE_BLOCKED
```
