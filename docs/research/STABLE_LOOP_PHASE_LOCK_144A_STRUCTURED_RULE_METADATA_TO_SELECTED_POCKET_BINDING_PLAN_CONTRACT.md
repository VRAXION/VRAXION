# STABLE_LOOP_PHASE_LOCK_144A_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PLAN Contract

144A is a planning-only and artifact-only milestone after the positive 143Z next-decision plan.

Boundary: constrained helper/backend evidence only, structured rule metadata to selected-pocket binding only, not natural-language rule reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

144A does not run helper generation, train, mutate checkpoints, modify `shared_raw_generation_helper.py`, modify helper/backend/request-key/runtime/product surfaces, or claim broad capability.

## Expected Decision

```text
decision = structured_rule_metadata_to_selected_pocket_binding_prototype_plan_recommended
selected_option = canonical_structured_rule_metadata_parser_plus_existing_selected_pocket_binding
next = 144B_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE
```

144A is the last planning step for this bridge unless its checker fails. 144B is the first executable prototype.

## Required Artifacts

144A writes:

```text
queue.json
progress.jsonl
analysis_config.json
upstream_143z_manifest.json
prototype_design_requirements.json
structured_rule_metadata_grammar_spec.json
rule_derivation_policy_matrix.json
prototype_options_matrix.json
selected_prototype_recommendation.json
target_144b_milestone_plan.json
anti_oracle_requirements.json
risk_register.json
decision.json
summary.json
report.md
```

## 144B Decoder

144B uses a new manifest-gated decoder:

```text
deterministic_pocket_gated_structured_rule_metadata_binding_decoder
```

144B may change the helper only behind that decoder. Existing selected-pocket binding decoder behavior from 143K/143V/143W must remain unchanged:

```text
deterministic_pocket_gated_rule_selected_pocket_binding_decoder
```

144B intended primitive:

```text
parse canonical structured rule metadata
derive selected pocket id
reuse existing selected pocket -> static marker -> same-line value extraction
emit ANSWER=E<value>
```

Trace fields required per row:

```text
parsed_rule_type
parsed_rule_fields
parse_success
derived_selected_pocket_id
binding_marker
extracted_value
generated_answer
failure_reason
```

## Grammar Policy

`structured_rule_metadata_grammar_spec.json` must define exact grammar and fallback policies:

```text
canonical structured rule metadata only
no free-form natural-language rule parsing
exact keys only per rule family
duplicate keys -> fallback
unknown keys -> fallback
missing required keys -> fallback
multiple rule_type lines -> fallback
invalid pocket ids -> fallback
malformed separators -> fallback
rule-derived subsets must not contain winner=pocket_* or selected_pocket_id
```

It must include quorum, recency, tie_break, and hierarchy. Hierarchy is a combiner fixture over precomputed sub-rule winners; it does not claim nested derivation of recency/quorum/tie_break from raw rule fields.

## Target 144B

`target_144b_milestone_plan.json` must be implementation-ready and include all required subsets, metrics, positive gates, clean negative routes, anti-oracle restrictions, and the new decoder name. It must explicitly state that no additional planning milestone is expected before 144B unless the 144A checker fails.

Rule-derived subsets must forbid hidden winner/oracle shortcuts, including `winner=pocket_*`, `selected_pocket_id`, final/winner value, answer/gold/target/resolved/expected output markers, per-row selected pocket request metadata, per-row manifest switching, payload marker narrowing to the correct pocket, and post-generation repair.
