# E20B_ARTIFACT_PERSISTENT_BINARY_GROUNDING_REPLAY_CLOSURE_CONFIRM Contract

## Purpose

E20B closes the E20/E20A artifact audit gap by requiring a binary-grounding run to produce both full target artifacts and a compact committed artifact sample pack that can be replayed independently of `target/`.

## Boundary

This is an artifact-persistence and replay-closure milestone for a controlled synthetic codec-agnostic binary-stream grounding benchmark. It proves that the run can be audited from target artifacts and from a compact committed artifact sample pack. It does not prove real audio understanding, real vision understanding, GPT-like generation, AGI, consciousness, or production readiness.

## Required artifact sample pack

The committed sample pack lives under:

```text
docs/research/artifact_samples/e20b_artifact_persistent_binary_grounding_replay_closure/
```

It must include sample episode rows, trace rows, replay rows, schema, target artifact hashes, sample metrics, oracle/codebook/collapse audits, deterministic replay report, and boundary-claim report.

## Full-confirm minimums

Full confirmation requires at least 80 generations, population 128, 1800 heldout episodes, 1800 stress episodes, 10000 candidate evaluations, 80 checkpoints, 800 cross-codec episodes, 600 missing/corrupt modality episodes, 500 adversarial false-alignment episodes, and 1000 cross-modal necessary episodes.

## Required checks

The checker must support target-artifact mode and `--sample-only` mode. Sample-only mode must not depend on `target/` and must recompute sample metrics, validate schema, validate run id, detect oracle/codebook leakage canaries, detect stale/static/missing sample failures, and validate non-tautological traces.

## Run command

```bash
python3 scripts/probes/run_e20b_artifact_persistent_binary_grounding_replay_closure_confirm.py \
  --out target/pilot_wave/e20b_artifact_persistent_binary_grounding_replay_closure_confirm \
  --artifact-sample-dir docs/research/artifact_samples/e20b_artifact_persistent_binary_grounding_replay_closure \
  --strict-budget \
  --no-downshift \
  --generations 120 \
  --population 160 \
  --train-episodes 7000 \
  --validation-episodes 1600 \
  --heldout-episodes 2200 \
  --stress-episodes 2200 \
  --min-stream-length 64 \
  --max-stream-length 256 \
  --min-modalities 4 \
  --max-modalities 7 \
  --checkpoint-every 1 \
  --max-runtime-minutes 360 \
  --resume
```
