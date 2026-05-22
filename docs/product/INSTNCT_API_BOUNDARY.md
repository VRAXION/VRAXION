# INSTNCT API Boundary

Status: 056 productization planning artifact.

This document defines the API boundary that must exist before INSTNCT can move
from research artifact to SDK release candidate.

## Current API Posture

Current route-grammar and visual export surfaces are experimental research
interfaces. They are usable for bounded evaluation and visual replay, but they
are not production APIs and should not be presented as stable public contracts.

## Future SDK Surface

The 057 SDK release candidate should expose only explicit, auditable calls:

```text
train(config, data_ref) -> checkpoint_ref
infer(checkpoint_ref, input_batch) -> inference_result
evaluate(checkpoint_ref, eval_suite_ref) -> eval_report
save_checkpoint(checkpoint_ref, destination) -> checkpoint_hash
load_checkpoint(source, expected_hash) -> checkpoint_ref
rollback(run_ref, checkpoint_ref) -> rollback_report
export_visual(run_ref, export_config) -> visual_bundle_ref
```

Each call must emit:

- Input schema version.
- Output schema version.
- Deterministic seed when applicable.
- Checkpoint hash where applicable.
- Error envelope on failure.
- Audit events for long-running operations.

## Long-Run Writeout Rule

No black-box long run is allowed. Any training, evaluation, visual export, or
deployment operation that can run longer than a few minutes must write partial
outcomes continuously.

Minimum writeouts:

- `progress.jsonl` every 20-60 seconds.
- `summary.json` updates at safe checkpoints.
- Append-only failure context.
- Resume pointer when applicable.

## Error Boundary

The SDK must reject:

- Unknown required schema versions.
- Missing required fields.
- Invalid checkpoint hashes.
- Unsupported deployment modes.
- Requests that ask for clinical or high-stakes education output without an
  approved compliance mode.

## Versioning Boundary

Release candidates must define:

- Semantic API version.
- Schema version.
- Backward-compatibility period.
- Deprecation path.
- Migration notes.

## Claim Boundary

056 supports productization planning only. It does not support production
release, public beta promotion, production API readiness, clinical readiness,
school high-stakes readiness, full VRAXION, language grounding, consciousness,
biological/FlyWire equivalence, or physical quantum behavior.

