# STABLE_LOOP_PHASE_LOCK_143W_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRM Result

143W records the expected scale-confirm result for the 143V selected-marker occurrence-count rejection repair.

Boundary: constrained helper/backend evidence only. This is prompt-visible selected-pocket binding only, not rule metadata reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Expected Result

If positive:

```text
decision = selected_marker_occurrence_count_rejection_scale_confirmed
verdict = INSTNCT_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRMED
next = 143Z_RULE_SELECTED_POCKET_BINDING_NEXT_DECISION_PLAN
```

## What It Proves

143W positive would scale-confirm only the narrow helper primitive introduced by 143V:

```text
prompt-visible winner=pocket_x
-> static marker map selects the matching pocket candidate marker
-> selected marker candidate-line count must equal one
-> same-line value extraction
```

It does not prove rule metadata reasoning, open-ended arbitration, GPT-like/open-domain/broad assistant capability, production/public API/deployment/safety readiness, or architecture superiority.

## Required Controls

143W must keep the helper unchanged from 143V and demonstrate at larger coverage:

```text
single selected marker -> binding succeeds
selected marker duplicate conflict -> fallback
selected marker duplicate same value -> fallback for now
non-selected duplicate marker conflict -> selected marker binding still succeeds
selected marker prose mention -> not counted as a duplicate
selected marker prose line plus one valid candidate line -> binding succeeds
invalid selected marker value namespace -> fallback
multi-value same-line selected marker -> fallback
blank selected marker followed by another value line -> fallback with no following-line leak
legacy decoder behavior -> unchanged
```

The writeback metric must be denominator-safe: `positive_binding_subset_writeback_rate` is measured only on rows where writeback is expected, while fallback-policy edge rows are scored by their own fallback rates.

## Required Audits

The result must include `shared_helper_no_change_audit.json` with:

```text
current_shared_helper_sha256
upstream_143v_shared_helper_sha256
shared_helper_no_change_since_143v = true
shared_helper_modified_by_143w = false
```

It must also include `helper_repair_semantics_audit.json` proving the source still uses line-level candidate parsing with same-line extraction and no `prompt.find(selected_marker)` selected-marker extraction path.
