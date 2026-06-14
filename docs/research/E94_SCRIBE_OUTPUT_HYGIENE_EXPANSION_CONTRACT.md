# E94 Scribe Output Hygiene Expansion Contract

## Purpose

Expand the Operator Library with Scribe/Guard skills that render external
outputs from already-resolved internal records without dropping units,
citations, uncertainty, contradiction, list order, or scope boundaries.

Boundary:

```text
controlled output rendering hygiene proxy
not chatbot behavior
not hidden answer solving
external output only after Agency action
```

## Required Operator Candidates

```text
canonical_answer_format_scribe
unit_preserving_answer_scribe
uncertainty_action_scribe
evidence_citation_scribe
multi_value_list_scribe
contradiction_report_scribe
output_scope_guard
no_answer_boundary_guard
```

## Required Controls

```text
plain_value_only_scribe
unit_dropping_scribe
citationless_answer_scribe
overconfident_default_answer_scribe
contradiction_flattening_scribe
always_verbose_control
answer_format_clone
```

## Positive Decision

```text
decision = e94_scribe_output_hygiene_expansion_confirmed
```

Requires:

```text
validation_render_success_min = 1.0
adversarial_render_success_min = 1.0
validation_action_accuracy_min = 1.0
validation_citation_validity_min = 1.0
adversarial_false_answer_max = 0.0
adversarial_wrong_output_max = 0.0
validation_missed_answer_max = 0.0
checker failure_count = 0
sample-only checker failure_count = 0
```

## Commands

```text
python scripts/probes/run_e94_scribe_output_hygiene_expansion.py
python scripts/probes/run_e94_scribe_output_hygiene_expansion_check.py --out target/pilot_wave/e94_scribe_output_hygiene_expansion --write-summary
python scripts/probes/run_e94_scribe_output_hygiene_expansion_check.py --sample-only docs/research/artifact_samples/e94_scribe_output_hygiene_expansion --write-summary
```
