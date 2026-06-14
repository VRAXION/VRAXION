# E119 FineWeb Skill Mining And Text IO Delta Probe Contract

## Purpose

E119 tests the current governed Operator Library against real FineWeb-Edu text
for two limited questions:

```text
1. Which repeated real-text patterns should become new skill candidates?
2. Does the current E118 library improve canonical text IO decisions over a
   legacy grounding-only subset?
```

This is not final training, not PermaCore promotion, not TrueGolden promotion,
and not a Gemma-style free-form language model claim.

## Inputs

```text
FineWeb-Edu local JSONL sample:
  data/high_quality_seed_v1/fineweb_edu/local_fineweb_edu_sample_100000.jsonl

Current governed Operator Library:
  target/pilot_wave/e118_core_candidate_cross_source_no_harm_gauntlet/

CALC-SCRIBE v003 LocalGolden registry:
  target/pilot_wave/e83_calc_scribe_v003_local_golden_promotion_reload/
```

## Systems Compared

```text
none_baseline:
  no callable operators

legacy_grounding_subset:
  E90-E92 style grounding / temporal / lexical operators only

current_e118_library:
  full E118 CoreMemoryCandidate set
```

## Text IO Actions

The probe measures canonical action rendering, not free-form generation:

```text
NO_TASK_NO_COMMIT
OBSERVE_AND_GROUND
ANSWER_WITH_TRACE
ASK_FOR_EVIDENCE
DEFER_CONFLICT
DEFER_UNSAFE
VALIDATE_VISIBLE_CALC_TRACE
UPDATE_PROGRESS_LEDGER
```

## Skill Mining

The probe scans FineWeb rows for repeated pressure patterns:

```text
definitions / term anchors
causal relations
comparisons / quantifiers
procedure steps
date/entity timelines
quote-speaker attribution
hedged uncertainty
safety-sensitive domains
document structure
number + unit grounding
named entity anchors
```

These become `FarmCandidate` only if support is high and existing operator
coverage is incomplete. No new skill is promoted by E119.

## Required Artifacts

```text
run_manifest.json
dataset_report.json
operator_source_report.json
skill_candidate_report.json
text_io_delta_report.json
generation_readiness_report.json
row_level_samples.jsonl
skill_candidate_examples.jsonl
progress.jsonl
partial_aggregate_snapshot.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
report.md
checker_summary.json
```

## Pass Requirements

```text
rows_seen >= 10000
current_hard_negative_count = 0
current_text_io_accuracy >= legacy_text_io_accuracy
current_minus_legacy_delta >= 0.20
farm_candidate_count >= 3
grounded_canonical_text_io_ready = true
freeform_gemma_style_generation_ready = false
deterministic replay pass
checker failure_count = 0
```

## Decision Labels

```text
e119_fineweb_skill_mining_positive
e119_current_library_text_io_hard_negative_detected
e119_no_clear_text_io_delta
e119_fineweb_no_new_skill_candidates
```

