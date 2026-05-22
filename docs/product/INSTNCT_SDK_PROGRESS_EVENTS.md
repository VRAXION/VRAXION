# INSTNCT SDK Progress Events

Status: 057 SDK release-candidate engineering artifact.

The SDK candidate follows the no-black-box-run rule. Even bounded smoke calls
write append-only operation events.

## Event Shape

```json
{
  "schema_version": "instnct_sdk_candidate_v1",
  "operation": "train",
  "phase": "start",
  "message": "train candidate start",
  "timestamp_ms": 0
}
```

## Required Smoke Operations

`progress.jsonl` must include start/completed events for:

```text
train
save
load
infer
evaluate
export_visual
rollback
invalid_schema_rejection
regulated_use_rejection
```

Rejected regulated requests may write rejection audit/progress events, but must
not write checkpoints or visual bundles.

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
