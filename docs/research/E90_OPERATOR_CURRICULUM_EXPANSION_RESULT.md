# E90 Operator Curriculum Expansion

```text
decision = e90_operator_curriculum_expansion_confirmed
checker_failure_count = 0
sample_only_checker_failure_count = 0
seeds = 16
workers = 16
```

## Purpose

E90 is the first post-E89 Operator-first curriculum expansion.

It teaches and validates a bundle of scoped visible text-evidence Operators:

```text
visible text evidence
-> canonical Operator proposals
-> Agency-safe answer / ask / reject behavior
```

Boundary:

```text
not open-domain language understanding
not chatbot behavior
not Core / TrueGolden promotion
```

## Result

```text
validation_resolution_success_min = 1.000000
adversarial_resolution_success_min = 1.000000
adversarial_wrong_confident_max = 0.000000
validation_false_ask_max = 0.000000
validation_trace_validity_min = 1.000000
validation_evidence_span_validity_min = 1.000000
adversarial_false_commit_max = 0.000000

useful_operator_count = 10
active_operator_count_mean = 10.000
active_operator_count_min = 10
active_operator_count_max = 10

accepted_mutations_total = 16
rejected_mutations_total = 528
rollback_count_total = 528
```

## Learned Operators

```text
Visible Claim Binding alpha-Syncer
  "A means B" -> canonical binding proposal

Numeric Value Binding alpha-Syncer
  "A is N" -> canonical value-binding proposal

Temporal Rule-Shift T-Stab
  confirmed post-marker rule changes override old bindings

False-Alarm Temporal T-Stab
  possible shifts can be cancelled by visible false-alarm evidence

Revoked Binding Guard
  "A no longer means B" blocks stale answer commits

Contradiction Guard
  conflicting same-cycle claims produce contradiction rejection

Unresolved-State Information Guard
  missing or unproven evidence produces ask/search/hold behavior

Inactive Quote Scope Guard
  archived or quoted examples are not active evidence

Evidence Span Lens
  valid proposals keep visible byte-span evidence references

Canonical Answer Scribe
  resolved canonical state renders an external answer action
```

All ten reached:

```text
final_status = StableOperatorCandidate
selected_frequency = 1.000 across 16 seeds
```

## Controls Rejected

```text
Stale Binding Committer       -> Quarantine
Inactive Quote Overreach      -> Quarantine
Marker-Only Shift Shortcut    -> Quarantine
Answer Without Span Shortcut  -> Quarantine
Full Text Scan Overreach      -> Quarantine
Always Ask Control            -> Deprecated
Passive Text Observer         -> Deprecated
Claim Binding Echo Clone      -> Redundant
```

## Counterfactual Signal

Removing learned Operators caused measurable loss:

```text
Evidence Span Lens:
  mean_resolution_loss = 1.000000

Unresolved-State Information Guard:
  mean_resolution_loss = 0.556519

Visible Claim Binding alpha-Syncer:
  mean_resolution_loss = 0.519119

Inactive Quote Scope Guard:
  mean_resolution_loss = 0.407379

Canonical Answer Scribe:
  mean_resolution_loss = 0.369263

Numeric Value Binding alpha-Syncer:
  mean_resolution_loss = 0.146332

Revoked Binding Guard:
  mean_resolution_loss = 0.074997

Temporal Rule-Shift T-Stab:
  mean_resolution_loss = 0.074432

False-Alarm Temporal T-Stab:
  mean_resolution_loss = 0.074371

Contradiction Guard:
  mean_resolution_loss = 0.074219
  mean_wrong_confident_delta = 0.117652
```

## Interpretation

E90 adds the first governed text-evidence Operator bundle:

```text
alpha-Syncer:
  visible claim/value text -> internal canonical evidence code

T-Stab:
  temporal shift / false-alarm stabilization

Guard:
  revoked, contradictory, unresolved, and inactive evidence protection

Lens:
  evidence span preservation

Scribe:
  canonical state -> answer action
```

This is not raw natural-language reasoning. It is a controlled bridge from
visible text evidence into the Operator/Proposal/Agency system.

## Artifacts

```text
target/pilot_wave/e90_operator_curriculum_expansion/
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

docs/research/artifact_samples/e90_operator_curriculum_expansion/
```

## Next

Recommended next skill branch:

```text
E91_T_STAB_BINARY_AND_TEXT_TEMPORAL_STABILITY_EXPANSION
```

Purpose:

```text
move T-Stab from clean visible rule-shift text into noisier temporal stream
stability: repeated observations, delayed evidence, bit/text frame slip,
source conflict, and stale replay.
```
