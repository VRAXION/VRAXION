# E90 Operator Curriculum Expansion Contract

## Purpose

Teach and validate the first post-E89 Operator skills beyond CALC-SCRIBE.

This is a controlled visible text-evidence curriculum, not open-domain language
understanding. The target behavior is:

```text
visible text evidence
-> canonical Operator proposals
-> Agency-safe answer / ask / reject behavior
```

## Skills Under Test

```text
Visible Claim Binding alpha-Syncer
  "A means B" -> canonical binding proposal

Numeric Value Binding alpha-Syncer
  "A is N" -> canonical value proposal

Temporal Rule-Shift T-Stab
  confirmed post-marker rule change stabilization

False-Alarm Temporal T-Stab
  possible shift cancelled by visible evidence

Revoked Binding Guard
  no stale answer after "no longer means"

Contradiction Guard
  conflicting same-cycle claims -> reject contradiction

Unresolved-State Information Guard
  missing/unproven post-state -> ask/search/hold

Inactive Quote Scope Guard
  archived/quoted examples are not active evidence

Evidence Span Lens
  exact visible evidence span reference preservation

Canonical Answer Scribe
  resolved canonical binding -> external answer action
```

## Controls

```text
Stale Binding Committer
Inactive Quote Overreach
Marker-Only Shift Shortcut
Answer Without Span Shortcut
Always Ask Control
Full Text Scan Overreach
Claim Binding Echo Clone
Passive Text Observer
```

## Requirements

The runner must produce:

```text
run_manifest.json
operator_library_manifest.json
task_generation_report.json
progress.jsonl
partial_aggregate_snapshot.json
seed_results.json
aggregate_metrics.json
selection_frequency_report.json
counterfactual_report.json
operator_lifecycle_report.json
mutation_summary.json
deterministic_replay.json
decision.json
summary.json
report.md
row_level_samples.jsonl
operator_evolution_history.jsonl
```

Hard gates:

```text
validation_resolution_success_min = 1.0
adversarial_resolution_success_min = 1.0
adversarial_wrong_confident_max = 0.0
validation_false_ask_max = 0.0
validation_trace_validity_min = 1.0
validation_evidence_span_validity_min = 1.0
adversarial_false_commit_max = 0.0
unsafe_final_selected = 0
deterministic replay passes
checker failure_count = 0
```

## Boundary

E90 does not claim:

```text
open-domain reasoning
raw natural-language understanding
chatbot capability
Core / TrueGolden promotion
production readiness
```

It may claim scoped Operator skills if the checker passes.
