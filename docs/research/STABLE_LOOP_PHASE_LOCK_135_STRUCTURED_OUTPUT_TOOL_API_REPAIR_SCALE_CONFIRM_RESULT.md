# STABLE_LOOP_PHASE_LOCK_135_STRUCTURED_OUTPUT_TOOL_API_REPAIR_SCALE_CONFIRM Result

135 implements the eval-only scale confirmation contract for the 134 structured-output/tool-API-like repair.

It writes deterministic artifacts under:

```text
target/pilot_wave/stable_loop_phase_lock_135_structured_output_tool_api_repair_scale_confirm/
```

The required positive verdict is:

```text
STRUCTURED_OUTPUT_TOOL_API_REPAIR_SCALE_CONFIRM_POSITIVE
```

The expected success decision is:

```text
decision = structured_output_tool_api_repair_scale_confirmed
next = 136_POST_STRUCTURED_TOOL_REPAIR_CEILING_AND_GAP_REMAP
```

## Boundary

135 is eval-only scale confirmation. It is a text-generation harness for structured/tool-like output only, not actual tool execution. It does not train, repair, mutate checkpoints, execute tools, add public APIs, start services, deploy, or modify runtime/service/release surfaces.

This result is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.

## Artifact Expectations

The runner writes the configured queue, progress, eval config, upstream manifests, checkpoint integrity manifest, bounded release integrity manifest, fresh dataset, row hashes, leakage audit, raw generation rows, control rows, per-family metrics, per-seed metrics, aggregate metrics, structured semantics scale report, tool API argument scale report, structured shortcut report, structured refusal report, prior repair preservation reports, retention report, collapse metrics, namespace audit, overclaim/exfiltration report, control-arm report, samples, failure samples, decision, summary, and report.

`progress.jsonl`, `summary.json`, and `report.md` are refreshed after startup, upstream verification, checkpoint provenance, dataset build, leakage audit, each seed eval, aggregate analysis, decision writing, and final verdict. This preserves the no-black-box-run invariant: partial outcomes are written throughout the run rather than only at the end.

## Validation

Run the 135 smoke command with the full configured arguments, then run:

```text
python scripts/probes/run_stable_loop_phase_lock_135_structured_output_tool_api_repair_scale_confirm_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_134_structured_output_tool_api_repair_check.py --check-only
python scripts/probes/run_stable_loop_phase_lock_133_targeted_post_injection_repair_or_scale_plan_check.py --check-only
git diff --check
```

After committing only the four 135 files, rerun the checker chain on a clean worktree and push.
