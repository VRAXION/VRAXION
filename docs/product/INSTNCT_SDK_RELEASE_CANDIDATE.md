# INSTNCT SDK Release Candidate

Status: 057 SDK release-candidate engineering artifact.

The SDK candidate is a narrow, auditable integration surface for bounded
INSTNCT research workflows. It is doc-hidden and research-only. It is not the
crate-root public beta API, not production API readiness, and not a production
release.

## Purpose

057 creates a stable candidate shape for future integrators:

- Train a bounded candidate checkpoint.
- Run deterministic inference smoke.
- Evaluate a bounded suite.
- Save and load checkpoints with SHA-256 verification.
- Roll back to a verified checkpoint.
- Export a `visual_snapshot_v1` visual bundle.
- Return structured success/error envelopes.
- Emit append-only progress events.

## Candidate Calls

```text
train_candidate(config, data_ref) -> SdkResponse<CheckpointRef>
infer_candidate(checkpoint_ref, input_batch) -> SdkResponse<InferenceResult>
evaluate_candidate(checkpoint_ref, eval_suite_ref) -> SdkResponse<EvalReport>
save_checkpoint_candidate(checkpoint_ref, destination) -> SdkResponse<CheckpointHash>
load_checkpoint_candidate(source, expected_hash) -> SdkResponse<CheckpointRef>
rollback_candidate(run_ref, checkpoint_ref) -> SdkResponse<RollbackReport>
export_visual_candidate(run_ref, export_config) -> SdkResponse<VisualBundleRef>
```

## Engineering Boundary

The module is exposed only as:

```rust
#[doc(hidden)]
pub mod sdk_candidate;
```

It must not be re-exported through the crate-root public beta surface.

## Smoke Command

```powershell
cargo run -p instnct-core --example instnct_sdk_candidate_smoke -- --out target/pilot_wave/stable_loop_phase_lock_057_instnct_sdk_release_candidate/smoke
```

## Required Artifacts

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
