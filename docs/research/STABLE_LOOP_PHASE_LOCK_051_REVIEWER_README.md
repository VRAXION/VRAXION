# STABLE_LOOP_PHASE_LOCK_051 Reviewer README

This reviewer bundle explains how to validate and optionally reproduce the
049/050 adversarial frozen-eval result.

## Quick Check

The quick check is static bundle validation only. It reads committed files,
checks required docs, exact commands, expected hashes, paper tables, failure
controls, and claim boundaries. It does not run cargo and does not write child
target artifacts.

```powershell
python scripts/probes/run_stable_loop_phase_lock_051_reviewer_bundle_check.py --check-only
```

Expected quick-check result:

```text
check_pass = true
PAPER_REPRODUCTION_BUNDLE_POSITIVE
REVIEWER_BUNDLE_CHECK_PASSES
PRODUCTION_API_NOT_READY
```

## Full Reproduction

The full reproduction delegates to the 050 audit runner. That runner reruns the
049 child cargo example, writes heartbeat progress while the child is active,
validates hashes, validates metric schemas, checks leakage, documents known
failure controls, and regenerates paper tables from machine-readable child
artifacts.

```powershell
python scripts/probes/run_stable_loop_phase_lock_050_repro_audit.py --out target/pilot_wave/stable_loop_phase_lock_051_reviewer_bundle/full_repro --seeds 2026,2027,2028 --train-examples 8192 --heldout-examples 4096 --ood-examples 4096 --heartbeat-sec 20
```

Expected full-reproduction output root:

```text
target/pilot_wave/stable_loop_phase_lock_051_reviewer_bundle/full_repro
```

Expected child output root:

```text
target/pilot_wave/stable_loop_phase_lock_051_reviewer_bundle/full_repro/child_049
```

## Evidence Sources

```text
docs/research/STABLE_LOOP_PHASE_LOCK_049_ADVERSARIAL_FROZEN_EVAL_CORPUS.jsonl
instnct-core/examples/phase_lane_adversarial_frozen_eval_scale.rs
scripts/probes/run_stable_loop_phase_lock_050_repro_audit.py
docs/research/STABLE_LOOP_PHASE_LOCK_050_EXPECTED_HASHES.json
docs/research/STABLE_LOOP_PHASE_LOCK_050_REPRODUCIBILITY_PACKAGE_AND_PAPER_AUDIT_RESULT.md
scripts/probes/run_stable_loop_phase_lock_051_reviewer_bundle_check.py
```

Expected source hashes:

```text
corpus_sha256_normalized_lf = 6b44848ab9483e8267103538ca58198b198a7651e9f20025168143fef4e5cd56
runner_sha256_normalized_lf = 4777b479294bc571751582dee53b05a121eae1465a45e24870432f4828b81046
```

## Claim Boundary

Supports:

```text
reviewer-facing reproduction package for bounded 049/050 adversarial frozen-eval result
```

Does not support:

```text
production default training
public beta promotion
production API readiness
full VRAXION
language grounding
consciousness
biological/FlyWire equivalence
physical quantum behavior
```
