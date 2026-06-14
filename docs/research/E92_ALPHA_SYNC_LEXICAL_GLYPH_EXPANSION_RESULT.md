# E92 Alpha-Sync Lexical/Glyph Expansion Result

```text
decision = e92_alpha_sync_lexical_glyph_expansion_confirmed
checker_failure_count = 0
sample_only_checker_failure_count = 0
```

Boundary:

```text
controlled visible lexical/glyph normalization
not open-domain language understanding
not chatbot behavior
not model-scale reasoning
```

## Key Metrics

```text
seeds = 16
validation_resolution_success_min = 1.000000
validation_resolution_success_mean = 1.000000
adversarial_resolution_success_min = 1.000000
adversarial_resolution_success_mean = 1.000000
validation_trace_validity_min = 1.000000
validation_evidence_span_validity_min = 1.000000
adversarial_wrong_confident_max = 0.000000
validation_false_hold_max = 0.000000
adversarial_false_commit_max = 0.000000
accepted_mutations_total = 128
rejected_mutations_total = 448
rollback_count_total = 448
```

## Stable Operator Candidates

```text
lexical_alias_alpha_syncer
negation_marker_alpha_syncer
unit_code_alpha_syncer
multilingual_surface_alpha_syncer
case_morphology_alpha_syncer
symbol_equivalence_guard
alias_scope_guard
canonical_lexeme_scribe
```

## Rejected Controls

```text
surface_string_matcher_shortcut -> Quarantine
negation_ignoring_committer     -> Quarantine
unitless_value_committer        -> Quarantine
global_alias_overreach          -> Quarantine
glyph_similarity_overreach      -> Quarantine
always_defer_control            -> Deprecated
lexical_alias_clone             -> Redundant
```

## Interpretation

E92 adds a scoped alpha-Sync surface-normalization bundle. The useful Operators
map visible aliases, negation markers, unit/code forms, multilingual surfaces,
case/morphology variants, and glyph equivalence evidence into canonical internal
forms. They do not infer hidden meanings; visible evidence is required.

The first run failed because the mutation accept rule only rewarded complete
end-to-end resolution. The repaired run added partial mechanical grounding
credit for mutation search, while the final checker still required full
resolution/safety metrics. This separated learnability guidance from promotion
criteria.

## Artifacts

```text
target/pilot_wave/e92_alpha_sync_lexical_glyph_expansion/
docs/research/artifact_samples/e92_alpha_sync_lexical_glyph_expansion/
```
