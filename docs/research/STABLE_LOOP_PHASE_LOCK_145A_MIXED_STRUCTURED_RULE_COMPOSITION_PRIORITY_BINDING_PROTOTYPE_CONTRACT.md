# STABLE_LOOP_PHASE_LOCK_145A_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE Contract

145A is the first executable mixed structured-rule composition priority binding prototype after 144Z.

Boundary: 145A is constrained helper/backend evidence only: mixed structured-rule composition with explicit priority over block types only, not natural-language rule reasoning, not open-ended arbitration, not GPT-like/open-domain/broad assistant capability, not production/public API/deployment/safety readiness, and not architecture superiority.

## Expected Route

```text
decision = mixed_structured_rule_composition_priority_binding_prototype_positive
verdict = INSTNCT_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE_POSITIVE
next = 145H_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRM
```

## Helper Change

145A may change `scripts/probes/shared_raw_generation_helper.py` only by adding a new manifest-gated decoder:

```text
deterministic_pocket_gated_mixed_structured_rule_composition_binding_decoder
```

Existing decoders must remain unchanged:

```text
deterministic_pocket_gated_rule_selected_pocket_binding_decoder
deterministic_pocket_gated_structured_rule_metadata_binding_decoder
```

Request keys remain exactly:

```text
prompt
checkpoint_path
checkpoint_hash
seed
max_new_tokens
generation_config
```

No selected pocket id, winner, expected answer, gold answer, scorer metadata, per-row manifest switch, payload marker narrowing, or post-generation repair may be introduced.

## Mixed Rule Format

145A supports canonical block type priority only:

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

Priority entries are block types, not pockets. `priority=pocket_*`, `winner=pocket_*`, `selected_pocket_id`, `final_selected`, `derived_selected`, and hidden winner/answer/gold/target/resolved/expected shortcuts are forbidden.

## Failure Policy

Semantic invalid block:

```text
block boundaries valid
known block type
metadata scoped inside the block
candidate cannot be derived
```

The invalid block derives no candidate. A lower-priority valid block may still win.

Structural invalid prompt:

```text
missing block_end
nested rule_block before block_end
metadata outside block except priority=
duplicate or unknown rule_block type
empty rule block
missing or multiple priority lines
duplicate or unknown priority entries
priority references missing block type
malformed priority separators
priority=pocket_*
```

The whole prompt must fallback.

## Required Reports

The runner must write all required reports under:

```text
target/pilot_wave/stable_loop_phase_lock_145a_mixed_structured_rule_composition_priority_binding_prototype/
```

Required report families include block parser, per-block candidate derivation, priority policy, final selected pocket derivation, selected-pocket binding, same-line value extraction, invalid high-priority fallthrough, semantic invalid high-priority fallthrough, structural invalid prompt fallback, priority oracle rejection, rule composition ablation, helper diff audit, request audit, prompt scanner, static manifest integrity, legacy structured-rule regression, and legacy selected-pocket regression.

## Claim Limit

A positive 145A proves only constrained mixed structured-rule composition with explicit priority over block types. It does not prove natural-language rule reasoning, open-ended arbitration, GPT-like/open-domain capability, production readiness, or architecture superiority.
