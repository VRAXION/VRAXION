# INSTNCT SDK API Reference

Status: 057 SDK release-candidate engineering artifact.

The SDK candidate API is an experimental, doc-hidden integration surface. It is
intended to make the bounded research workflow callable through structured
requests and responses.

## Schema

```text
schema_version = instnct_sdk_candidate_v1
```

Unknown schema versions must return `UNKNOWN_SCHEMA_VERSION`.

## Shared Context

Each request carries:

- `schema_version`
- `intended_use`
- `production_flags`
- optional `progress_path`

Production flags must remain false:

```text
production_default_training_enabled = false
public_beta_promoted = false
production_api_ready = false
```

## Calls

### train_candidate

Creates a deterministic smoke checkpoint and returns a `CheckpointRef`.

### infer_candidate

Verifies checkpoint SHA-256 and returns deterministic inference samples.

### evaluate_candidate

Verifies checkpoint SHA-256 and returns a bounded eval report.

### save_checkpoint_candidate

Copies a checkpoint and returns a SHA-256 hash.

### load_checkpoint_candidate

Verifies SHA-256 before loading the checkpoint.

### rollback_candidate

Confirms a rollback checkpoint is available and hash-verified.

### export_visual_candidate

Calls the existing research visual exporter and writes `visual_snapshot_v1`.

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
