# STABLE_LOOP_PHASE_LOCK_135D_GLOBAL_RAW_EVIDENCE_REBUILD_PLAN Result

135D implements the planning-only rebuild plan after 135B. It treats 135B `phase_evidence_reclassification.json` as the source of truth and requires the exact 41-phase distribution:

- `ORACLE_SHORTCUT_DETECTED = 11`
- `DETERMINISTIC_HARNESS_ONLY = 3`
- `NEEDS_MANUAL_REVIEW = 17`
- `REAL_RAW_GENERATION_EVIDENCE = 4`
- `NOT_RAW_EVIDENCE_PHASE = 6`

The runner writes a phase rebuild matrix, claim quarantine map, raw generation helper requirements, expected-output canary spec, manual review plan, future checker requirements, rebuild sequence, and risk register.

## Decision

When all gates pass:

- `decision = global_raw_evidence_rebuild_plan_complete`
- `next = 135E_SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE`

The plan explicitly blocks `136_POST_STRUCTURED_TOOL_REPAIR_CEILING_AND_GAP_REMAP` until the raw evidence chain is rebuilt through the shared helper and canary path.

## Evidence handling

Oracle-shortcut phases are invalidated until rebuilt. Deterministic harness phases remain harness-only. Manual-review phases cannot be used for raw claims until inspected. Existing real raw generation evidence is retained, but future rebuild work still standardizes helper provenance and expected-output canary gates.

The future checker requirements make raw helper provenance, expected-output canary pass, AST shortcut scans, raw final eval flags, and deterministic positive-arm rejection mandatory for future raw capability milestones.

## Boundary

135D is planning only. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated as model evidence. Bounded local/private release remains separate. No raw capability is restored. It is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.
