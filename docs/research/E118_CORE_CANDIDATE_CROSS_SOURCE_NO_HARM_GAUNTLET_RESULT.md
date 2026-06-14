# E118 CoreCandidate Cross-Source No-Harm Gauntlet Result

## Verdict

```text
decision = e118_core_candidate_cross_source_no_harm_confirmed
checker_failure_count = 0
```

E118 ran the full 136 `CoreMemoryCandidate` set through source-diverse
falsification pressure. This was a no-harm / synthetic-imprint test, not a
PermaCore or TrueGolden promotion.

## Key Metrics

```text
candidate_count = 136
cross_source_no_harm_pass_count = 136
cross_source_no_harm_remaining_count = 0

case_count = 17408
source_family_count = 8
cases_per_source = 16

hard_negative_total = 0
false_commit_total = 0
unsupported_answer_total = 0
wrong_scope_call_total = 0
negative_transfer_total = 0
synthetic_imprint_total = 0
```

## 300k Status Split

```text
actual_300k_count = 77
e114_projected_300k_count = 59
```

Interpretation:

```text
77 Operators have actual E117 gauntlet activation >= 300k.
59 Operators are already CoreMemoryCandidate and reached the 300k line through
the E114 full-FineWeb projection path, not through E117 actual targeted gauntlet
activation.
```

The dashboard marks actual 300k Operators with the visual `Orange300K` status.
This is a probation marker, not PermaCore.

## Source Families

```text
e117_replay_pack
regenerated_alpha_weave_new_seed
fineweb_real_snippet_projection
human_dnd_public_evidence_cell
adversarial_negative_scope
stale_conflict_missing_evidence
active_set_selection_stress
ablation_without_operator
```

## Boundary

E118 confirms cross-source no-harm for the 136 CoreMemoryCandidates. It does
not promote any Operator to PermaCore, TrueGolden, or global Core memory.
