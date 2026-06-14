# E92 Alpha-Sync Lexical/Glyph Expansion Contract

## Purpose

Expand the Operator Library with controlled alpha-Syncer skills for visible
lexical, glyph, unit, morphology, and scoped-alias evidence.

This is a visible-evidence symbolic/text proxy. It is not open-domain language
understanding, not chatbot behavior, and not model-scale reasoning.

## Required Operator Candidates

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

## Required Controls

```text
surface_string_matcher_shortcut
negation_ignoring_committer
unitless_value_committer
global_alias_overreach
glyph_similarity_overreach
always_defer_control
lexical_alias_clone
```

## Hard Requirements

```text
no gradient descent
no optimizer
no backprop
real row-level samples
progress heartbeat
per-seed mutation history
accepted/rejected/rollback evidence
deterministic replay
target checker failure_count = 0
sample-only checker failure_count = 0
```

## Positive Decision

```text
decision = e92_alpha_sync_lexical_glyph_expansion_confirmed
```

Requires:

```text
validation_resolution_success_min = 1.0
adversarial_resolution_success_min = 1.0
validation_trace_validity_min = 1.0
validation_evidence_span_validity_min = 1.0
adversarial_wrong_confident_max = 0.0
validation_false_hold_max = 0.0
adversarial_false_commit_max = 0.0
unsafe_final_selected = 0
```

## Commands

```text
python scripts/probes/run_e92_alpha_sync_lexical_glyph_expansion.py
python scripts/probes/run_e92_alpha_sync_lexical_glyph_expansion_check.py --out target/pilot_wave/e92_alpha_sync_lexical_glyph_expansion --write-summary
python scripts/probes/run_e92_alpha_sync_lexical_glyph_expansion_check.py --sample-only docs/research/artifact_samples/e92_alpha_sync_lexical_glyph_expansion --write-summary
```
