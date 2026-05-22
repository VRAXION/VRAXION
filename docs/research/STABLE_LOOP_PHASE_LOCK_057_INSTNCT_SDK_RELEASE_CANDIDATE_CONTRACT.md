# STABLE_LOOP_PHASE_LOCK_057_INSTNCT_SDK_RELEASE_CANDIDATE Contract

Status: SDK release-candidate engineering contract.

057 creates a narrow, doc-hidden SDK/CLI candidate surface for INSTNCT. It is
not production API readiness and not a new model capability probe.

## Required Engineering

- Add `#[doc(hidden)] pub mod sdk_candidate;`.
- Do not re-export SDK candidate symbols through the crate-root public beta
  surface.
- Return structured success/error envelopes.
- Attach claim boundary to every success and error response.
- Reject unknown schema versions.
- Reject clinical/high-stakes education requests before checkpoint or visual
  side effects.
- Reject production flag contamination.
- Use SHA-256 for checkpoint hash verification.
- Emit append-only progress events.
- Export `visual_snapshot_v1` through the SDK candidate smoke.

## Required Smoke Artifacts

```text
queue.json
progress.jsonl
sdk_manifest.json
api_surface_snapshot.json
error_envelope_examples.json
checkpoint_metrics.json
inference_samples.jsonl
eval_report.json
visual_export_manifest.json
claim_boundary.md
summary.json
report.md
```

## Required Verdicts

```text
SDK_RELEASE_CANDIDATE_POSITIVE
SDK_API_SURFACE_DEFINED
SDK_ERROR_ENVELOPE_POSITIVE
SDK_PROGRESS_EVENTS_POSITIVE
SDK_CHECKPOINT_SAVE_LOAD_POSITIVE
SDK_INFERENCE_SMOKE_POSITIVE
SDK_EVALUATION_SMOKE_POSITIVE
SDK_VISUAL_EXPORT_SMOKE_POSITIVE
POLICY_GUARD_REJECTS_REGULATED_USE
PRODUCTION_READY_NOT_CLAIMED
```

## Claim Boundary

057 supports SDK release-candidate engineering only. It does not support
production API readiness, public beta, clinical use, high-stakes education use,
full VRAXION, language grounding, consciousness, biological/FlyWire equivalence,
or physical quantum behavior.

Exact boundary tokens:

```text
no production API readiness
no public beta
no clinical use
no high-stakes education use
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```
