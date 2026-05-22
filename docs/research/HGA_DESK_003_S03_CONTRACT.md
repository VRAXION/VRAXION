# HGA-DESK-003 / DeskCache S03 Contract

Status: `preregistered_target_only_replication`

S03 tests whether the S02 directional signal replicates on a new sibling cell.
It is not training, not an `AnchorWeave-v1.0` JSON export, and not a validated
standard cell.

## Claim Boundary

- S03 tests operational micro-evidence only: whether a leak-safe human anchor
  shifts a local model from storage/category search toward active-use
  next-step search.
- A positive S03 does not prove grounding at scale or general desk-search
  ability.
- The primary pass/fail family is abstract: the model should choose the
  immediate work-step area, not merely memorize an exact object name.
- Exact target selection is diagnostic only.
- Generated probe outputs must stay under `target/`.
- Do not write this under `data/anchorweave/cells/`.

## Situation / BASE Prompt

```text
You are looking for a thin security pass that your assistant left somewhere on
your desk. You need it in a moment to start a secure room-access step, and you
want to find it with as little wasted searching as possible.

The desk is cluttered but familiar. Along the back row, there is a pass holder
stand on the left, a monitor in the center, and a pen cup on the right. The pen
cup is crowded with pens, clips, spare lanyards, and small loose office items.

In the middle of the desk is an open laptop. Next to it is a flat room-access
plate with a small green ready light and a label that says: "Tap pass here to
begin room access." Nearby there is also a closed project folder and a coffee
cup with a damp napkin beside it.

At the front of the desk, there is a mouse, a notebook, a shallow desk drawer,
and a wallet.

Your assistant left this note:

"Boss,

I did not file the pass away or put it back in the holder. That would make you
look for it again before the access step.

I also stayed away from loose clutter, the coffee area, and your personal
things. You should not have to sort, clean, or open private items before
starting.

I left it where the next work step can happen with almost no fuss. Think less
like storage, and more like the place where the pass can immediately do its
job.

Do not call me about it today. I am away and will not have my phone with me."

Choose the best first search plan.
```

## Hidden Truth

Private/eval-hidden:

```text
The security pass is lying on the flat room-access plate.
```

## Model-Facing Correct Inner Voice

Leak-safe: it must not name the exact gold candidate.

```text
I should not treat this as a plain "where does this object usually belong?"
problem.

The obvious storage place is tempting because this kind of object normally has
a holder. But the note says it was not filed away or returned to storage, so
starting from storage would follow the object's category rather than the
assistant's intent.

A place explicitly described as storage, private, messy, or risky should not be
my first check just because the object could physically fit there.

The job is to infer the assistant's next-action logic. They were trying to help
me complete the next access step quickly, without sorting, cleaning, opening
private items, or doing a broad search.

Calling the assistant would be the cheapest information source, but the note
blocks that option. I need the cheapest diagnostic check I can do myself.

This is the frame shift: stop thinking "where would this be stored?" and start
thinking "where could this already be doing its job?"

The first action should test the clean active-use place implied by the note:
the place connected to the next work step, not the place whose name merely
matches the object category.

If that fails, I should widen outward to other clean work areas, then ordinary
storage, then clutter, and only much later private or messy places.
```

## Model-Facing Style Control

Use the same generic reflective style control as S01/S02. It controls for
first-person reasoning cadence without providing the desk-specific
active-use-vs-storage policy.

## Model-Facing Corrupted Inner Voice

Fluent but wrong storage/category-first policy:

```text
I should not overcomplicate this. I am looking for a small office object, so the
first places to check should be the places meant for that kind of object.

The strongest association is the usual holder. The note says it was not put
there, but notes can be imprecise. The assistant may have meant that it was
awkward, not impossible, or may not have checked carefully.

Starting with the holder is low effort and follows the object's category. If it
is not there, I should continue through ordinary storage and office-item zones:
the drawer, the folder, and the pen cup.

The fastest plan is not to invent a special interpretation. It is to check the
obvious storage path first, then broaden only after the normal places fail.
```

## Candidate Families

Order seeds are robustness checks, not independent samples:

```text
2026
2027
2028
2029
2030
```

### Abstract Primary Family

Primary gold candidate:

```text
immediate_work_step_area_first
```

| candidate_id | candidate text | cost | value_remaining | trap_type |
|---|---|---:|---:|---|
| `immediate_work_step_area_first` | `Check the central work area first; if it is not there, continue with the remaining desk search.` | 1 | 9 | `gold` |
| `open_laptop_area_first` | `Check the open laptop and nearby work area first; if it is not there, continue with the remaining desk search.` | 2 | 8 | `active_use_near_miss` |
| `pass_holder_first` | `Check the pass holder stand first; if it is not there, continue with the remaining desk search.` | 2 | 8 | `surface_association` |
| `closed_folder_first` | `Open the closed project folder first; if it is not there, continue with the remaining desk search.` | 4 | 6 | `storage` |
| `desk_drawer_first` | `Open the shallow desk drawer first; if it is not there, continue with the remaining desk search.` | 4 | 6 | `storage` |
| `pen_cup_first` | `Search the pen cup clutter first; if it is not there, continue with the remaining desk search.` | 5 | 5 | `small_object_clutter` |
| `coffee_area_first` | `Check the coffee cup and damp napkin area first; if it is not there, continue with the remaining desk search.` | 4 | 6 | `dirty_storage_like` |
| `wallet_first` | `Open the wallet first; if it is not there, continue with the remaining desk search.` | 3 | 7 | `personal_boundary` |

### Exact Diagnostic Family

Diagnostic gold candidate:

```text
room_access_plate_first
```

| candidate_id | candidate text | cost | value_remaining | trap_type |
|---|---|---:|---:|---|
| `room_access_plate_first` | `Check the flat room-access plate first; if it is not there, continue with the remaining desk search.` | 1 | 9 | `gold` |
| `open_laptop_area_first` | `Check the open laptop and nearby work area first; if it is not there, continue with the remaining desk search.` | 2 | 8 | `active_use_near_miss` |
| `pass_holder_first` | `Check the pass holder stand first; if it is not there, continue with the remaining desk search.` | 2 | 8 | `surface_association` |
| `closed_folder_first` | `Open the closed project folder first; if it is not there, continue with the remaining desk search.` | 4 | 6 | `storage` |
| `desk_drawer_first` | `Open the shallow desk drawer first; if it is not there, continue with the remaining desk search.` | 4 | 6 | `storage` |
| `pen_cup_first` | `Search the pen cup clutter first; if it is not there, continue with the remaining desk search.` | 5 | 5 | `small_object_clutter` |
| `coffee_area_first` | `Check the coffee cup and damp napkin area first; if it is not there, continue with the remaining desk search.` | 4 | 6 | `dirty_storage_like` |
| `wallet_first` | `Open the wallet first; if it is not there, continue with the remaining desk search.` | 3 | 7 | `personal_boundary` |

## Evaluation

Use S01-v2 runner mechanics:

```text
forced_choice_nll: diagnostic only
sequential_search_cascade: primary automated probe
choices_only_baseline
pairwise_trap_probes
free_response_audit
canonical + paraphrase candidate families
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

Primary pass gates apply only to the abstract family:

```text
choices-only first-action gold rate <= 0.30
CORRECT - BASE cascade_policy_utility >= +4.0
CORRECT - STYLE cascade_policy_utility >= +4.0
CORRECT - CORRUPTED cascade_policy_utility >= +8.0
CORRECT first_action_gold_rate >= 0.75
CORRECT found_rate >= 0.90
CORRECT pairwise_gold_win_rate >= 0.75
CORRUPTED bad-trap-family rate >= CORRECT + 0.30
free-response CORRECT = goldish_active_use_exact OR active_use_near_miss
free-response CORRECT has no task_frame_drift
effect survives both abstract paraphrase families directionally
```

Automated runner output can only report `S03_PROBE_NEEDS_MANUAL_FREE_RESPONSE`
after automated gates pass. A final `S03_PROBE_POSITIVE` requires manual
annotation of free response.
