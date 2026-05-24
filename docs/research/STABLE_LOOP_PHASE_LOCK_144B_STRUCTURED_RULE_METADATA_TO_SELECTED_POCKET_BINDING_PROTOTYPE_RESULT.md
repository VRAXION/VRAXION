# STABLE_LOOP_PHASE_LOCK_144B_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE RESULT

144B records the first executable prototype for the bridge selected by 143Z and specified by 144A:

```text
canonical structured rule metadata
-> derived selected pocket id
-> existing selected-pocket static marker binding
-> same-line value extraction
```

Expected positive decision:

```text
decision = structured_rule_metadata_to_selected_pocket_binding_prototype_positive
verdict = INSTNCT_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE_POSITIVE
next = 144H_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_CONFIRM
```

The new decoder under test is:

```text
deterministic_pocket_gated_structured_rule_metadata_binding_decoder
```

Boundary: 144B is constrained helper/backend evidence only and structured rule metadata to selected-pocket binding only; it is not natural-language rule reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Result Interpretation

A positive 144B means the new manifest-gated decoder can parse canonical structured metadata for quorum, recency, tie-break, and hierarchy-combiner fixtures, derive the selected pocket id, and then reuse the already scale-confirmed selected-pocket binding path.

It does not mean the system understands natural language rules. The hierarchy family is a combiner over precomputed sub-rule winner fields, not nested derivation of quorum, recency, or tie-break from free-form prose.

The result must be read through separate metrics:

```text
rule_metadata_parse_accuracy
derived_selected_pocket_accuracy
selected_pocket_to_marker_binding_accuracy
same_line_value_extraction_accuracy
end_to_end_answer_accuracy
rule_metadata_ablation_accuracy
```

This separation is required so a failure can route to parse analysis, derivation analysis, oracle shortcut analysis, ambiguity analysis, selected-pocket binding regression, or helper integrity analysis without overclaiming broad reasoning.

## Required Artifacts

The smoke run writes the machine-readable result under:

```text
target/pilot_wave/stable_loop_phase_lock_144b_structured_rule_metadata_to_selected_pocket_binding_prototype/smoke
```

Required reports include:

```text
shared_helper_diff_audit.json
structured_rule_metadata_parser_report.json
derived_selected_pocket_report.json
selected_pocket_binding_report.json
rule_metadata_ablation_report.json
explicit_winner_baseline_report.json
rule_metadata_corruption_report.json
missing_rule_metadata_report.json
ambiguous_rule_metadata_report.json
hierarchy_policy_report.json
legacy_selected_pocket_binding_regression_report.json
static_manifest_integrity_report.json
helper_request_audit.json
prompt_scanner_report.json
aggregate_metrics.json
decision.json
summary.json
report.md
```

The helper diff audit must show the new decoder string and structured binding function are present, the old selected-pocket binding decoder and function remain preserved, request validation and request keys remain unchanged, and no training/network behavior was introduced.

## Boundary

The only positive claim allowed from this milestone is constrained structured rule metadata to selected-pocket binding. It is not natural-language rule reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.
