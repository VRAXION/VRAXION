# STABLE_LOOP_PHASE_LOCK_144Z_STRUCTURED_RULE_METADATA_BINDING_NEXT_DECISION_PLAN Contract

144Z is a planning-only and artifact-only next-decision milestone after the positive 144H structured rule metadata binding scale confirm.

Boundary: 144Z is constrained helper/backend evidence only for structured rule metadata binding next decisions; it is not natural-language rule reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Expected Route

```text
decision = structured_rule_composition_priority_binding_prototype_plan_recommended
selected_option = mixed_structured_rule_composition_priority_binding_prototype
next = 145A_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE
```

144Z must not call `raw_generate`, import `shared_raw_generation_helper.py`, modify helper/backend/runtime/request-key surfaces, train, mutate checkpoints, or claim broad capability.

## Required Artifacts

Artifacts are generated only under:

```text
target/pilot_wave/stable_loop_phase_lock_144z_structured_rule_metadata_binding_next_decision_plan/
```

Required files:

```text
queue.json
progress.jsonl
analysis_config.json
upstream_144h_manifest.json
evidence_chain_summary.json
structured_rule_binding_state_report.json
next_decision_matrix.json
mixed_rule_composition_gap_analysis.json
target_145a_milestone_plan.json
anti_oracle_requirements.json
risk_register.json
decision.json
summary.json
report.md
```

The run must write `queue.json` and append `progress.jsonl` immediately, then write final artifacts atomically.

## Upstream 144H Gate

144Z must verify exactly:

```text
decision = structured_rule_metadata_to_selected_pocket_binding_scale_confirmed
verdict = INSTNCT_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_CONFIRMED
next = 144Z_STRUCTURED_RULE_METADATA_BINDING_NEXT_DECISION_PLAN
main_eval_rows = 6528
rule_metadata_parse_accuracy = 1.0
derived_selected_pocket_accuracy = 1.0
selected_pocket_to_marker_binding_accuracy = 1.0
same_line_value_extraction_accuracy = 1.0
end_to_end_answer_accuracy = 1.0
rule_metadata_ablation_accuracy = 0.0
wrong_family_extra_key_rejection_rate = 1.0
quorum_clear_winner_ignores_tie_break_accuracy = 1.0
legacy_143w_binding_regression_passed = true
shared_helper_no_change_since_144b = true
deterministic_replay_passed = true
```

## Decision

144H scale-confirmed single structured rule metadata families:

```text
quorum
recency
tie_break
hierarchy combiner
```

The next missing bridge is:

```text
multiple structured rule candidates
-> explicit priority/conflict policy
-> selected pocket identity
-> existing static marker binding
-> same-line value extraction
```

The decision matrix must compare the mixed structured rule composition option against narrower or broader alternatives:

```text
mixed_structured_rule_composition_priority_binding_prototype
structured_rule_metadata_robustness_extension
integration_into_broader_multi_pocket_arbitration_suite
stop_at_single_rule_structured_metadata_binding
```

and select `mixed_structured_rule_composition_priority_binding_prototype`.

## Target 145A Plan

`target_145a_milestone_plan.json` must be implementation-ready for:

```text
145A_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE
```

145A may modify the helper only behind a new manifest-gated decoder:

```text
deterministic_pocket_gated_mixed_structured_rule_composition_binding_decoder
```

Existing decoders must remain unchanged:

```text
deterministic_pocket_gated_rule_selected_pocket_binding_decoder
deterministic_pocket_gated_structured_rule_metadata_binding_decoder
```

145A must parse multiple canonical structured rule blocks, derive one candidate pocket from each valid block, apply explicit priority over block types, then reuse existing static marker binding and same-line value extraction.

Canonical block format:

```text
rule_block=quorum
votes=pocket_a,pocket_b,pocket_a
block_end

rule_block=recency
recency_order=pocket_c>pocket_b>pocket_a
block_end

rule_block=tie_break
tied=pocket_a,pocket_c
tie_break_order=pocket_c>pocket_a>pocket_b
block_end

priority=recency>quorum>tie_break
```

Strict rejection policy:

```text
missing block_end -> fallback
nested rule_block before block_end -> fallback
metadata outside block except priority= -> fallback
duplicate block type -> fallback
unknown block type -> fallback
empty block -> fallback
missing priority -> fallback
duplicate priority entries -> fallback
unknown priority entries -> fallback
priority entries referencing missing block types -> fallback
malformed priority separators -> fallback
multiple priority lines -> fallback
priority pocket oracle entries -> fallback
```

Invalid high-priority blocks are ignored if a lower-priority valid block exists. If all priority-referenced blocks are invalid, the helper must fallback.

145A must include the required subsets, metrics, gates, trace fields, prompt scanner forbids, anti-oracle requirements, parser rejection policies, and clean negative routes specified by `target_145a_milestone_plan.json`.

## Claim Limit

A future positive 145A would prove constrained mixed structured-rule composition with explicit priority policy only. It would not prove natural-language reasoning, open-ended arbitration, GPT-like capability, production readiness, or architecture superiority.
