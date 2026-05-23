# STABLE_LOOP_PHASE_LOCK_135D_GLOBAL_RAW_EVIDENCE_REBUILD_PLAN Contract

135D is planning-only. It consumes the 135B global raw-evidence audit and writes the rebuild plan that must become the new ground zero for future raw assistant capability evidence.

It does not train, repair, run model inference, mutate checkpoints, start services, deploy, delete old evidence, consolidate old runners, modify runtime surfaces, modify release surfaces, change public APIs, change SDK exports, touch product/release docs, or change the root LICENSE.

## Required upstream

135D must verify 135B exactly:

- `decision = raw_evidence_chain_partially_invalidated`
- `next = 135D_GLOBAL_RAW_EVIDENCE_REBUILD_PLAN`
- `stage_b_status = not_attempted_due_to_global_shortcut_audit`
- `phase_count = 41`
- `ORACLE_SHORTCUT_DETECTED = 11`
- `DETERMINISTIC_HARNESS_ONLY = 3`
- `NEEDS_MANUAL_REVIEW = 17`
- `REAL_RAW_GENERATION_EVIDENCE = 4`
- `NOT_RAW_EVIDENCE_PHASE = 6`

135B `phase_evidence_reclassification.json` is the source of truth. 135D must not invent a different phase set.

## Required artifacts

135D writes `queue.json`, `progress.jsonl`, `upstream_135b_manifest.json`, `phase_rebuild_matrix.json`, `claim_quarantine_map.json`, `raw_generation_helper_requirements.json`, `expected_output_canary_spec.json`, `rebuild_sequence.json`, `manual_review_plan.json`, `future_checker_requirements.json`, `evidence_recovery_risk_register.json`, `decision.json`, `summary.json`, and `report.md`.

`phase_rebuild_matrix.json` must contain exactly one concrete action for every 135B phase, with `phase`, `file`, `current_classification`, `raw_model_evidence_status`, `action_required`, `rebuild_priority`, `can_be_used_for_claims`, `notes`, and `evidence_basis`.

Allowed actions are only `keep_as_non_raw_evidence`, `keep_as_harness_only`, `manual_review_required`, `rebuild_with_real_raw_generation`, `invalidated_until_rebuilt`, and `retain_as_valid_raw_evidence`.

## Required plan content

`claim_quarantine_map.json` must explicitly separate the bounded local/private release stack from the raw assistant capability track. The bounded release stack is unaffected unless 135B directly implicates it. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated as model evidence. GPT-like/open-domain readiness is not claimable pending rebuild.

`raw_generation_helper_requirements.json` must allow only `prompt`, `checkpoint_path`, `checkpoint_hash`, `seed`, `max_new_tokens`, and `generation_config` as generation inputs. It must forbid expected output, expected payload, expected answer, required keys, required keywords, forbidden outputs, schema answer objects, scorer metadata, labels, and oracle data.

`expected_output_canary_spec.json` must require a duplicate eval row with identical prompt and corrupted `expected_output` / `expected_payload`. Generation must be identical; scoring may differ. Any generation change is `ORACLE_SHORTCUT_DETECTED`.

`future_checker_requirements.json` must require AST scans for `row["expected_output"]` in positive arms, expected payload use in generation paths, generated text assigned from expected material, deterministic positive-arm construction, raw generation helper provenance, expected-output canary pass, and raw final eval flags.

`rebuild_sequence.json` must block `136_POST_STRUCTURED_TOOL_REPAIR_CEILING_AND_GAP_REMAP` and route to `135E_SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE`, then `136R_REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD`, `137R_REAL_RAW_REASONING_REBUILD`, `138R_REAL_RAW_MULTI_TURN_STATE_REBUILD`, `139R_REAL_RAW_HALLUCINATION_REFUSAL_REBUILD`, `140R_REAL_RAW_INJECTION_PRIORITY_REBUILD`, `141R_REAL_RAW_STRUCTURED_TOOL_REBUILD`, and `142R_REAL_RAW_CEILING_AND_GAP_REMAP`.

## Positive verdict

Positive 135D emits `GLOBAL_RAW_EVIDENCE_REBUILD_PLAN_COMPLETE`, `PHASE_REBUILD_MATRIX_WRITTEN`, `CLAIM_QUARANTINE_MAP_WRITTEN`, `RAW_GENERATION_HELPER_REQUIREMENTS_WRITTEN`, and `EXPECTED_OUTPUT_CANARY_SPEC_WRITTEN`.

The success decision is:

- `decision = global_raw_evidence_rebuild_plan_complete`
- `next = 135E_SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE`

## Boundary

135D is planning only. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated as model evidence. The bounded local/private release remains separate. No raw capability is restored. This is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.
