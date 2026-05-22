# STABLE_LOOP_PHASE_LOCK_051_PAPER_REPRODUCTION_BUNDLE Contract

## Summary

051 turns the already-positive 049/050 adversarial frozen-eval result into a
reviewer-facing reproduction bundle. It is a static documentation and
validation package, not a new capability probe.

The bundle is designed so a reviewer can run a quick static check first, then
optionally run the full 050 reproduction command that reruns the 049 child cargo
example.

## Required Reviewer Commands

Quick static bundle validation:

```powershell
python scripts/probes/run_stable_loop_phase_lock_051_reviewer_bundle_check.py --check-only
```

Optional full reproduction through 050:

```powershell
python scripts/probes/run_stable_loop_phase_lock_050_repro_audit.py --out target/pilot_wave/stable_loop_phase_lock_051_reviewer_bundle/full_repro --seeds 2026,2027,2028 --train-examples 8192 --heldout-examples 4096 --ood-examples 4096 --heartbeat-sec 20
```

## Required Source Evidence

```text
docs/research/STABLE_LOOP_PHASE_LOCK_049_ADVERSARIAL_FROZEN_EVAL_CORPUS.jsonl
instnct-core/examples/phase_lane_adversarial_frozen_eval_scale.rs
scripts/probes/run_stable_loop_phase_lock_050_repro_audit.py
docs/research/STABLE_LOOP_PHASE_LOCK_050_EXPECTED_HASHES.json
docs/research/STABLE_LOOP_PHASE_LOCK_050_REPRODUCIBILITY_PACKAGE_AND_PAPER_AUDIT_RESULT.md
```

Expected normalized-LF hashes from 050:

```text
corpus_sha256_normalized_lf = 6b44848ab9483e8267103538ca58198b198a7651e9f20025168143fef4e5cd56
runner_sha256_normalized_lf = 4777b479294bc571751582dee53b05a121eae1465a45e24870432f4828b81046
```

## Required Reviewer Bundle Files

```text
docs/research/STABLE_LOOP_PHASE_LOCK_051_REVIEWER_README.md
docs/research/STABLE_LOOP_PHASE_LOCK_051_ARTIFACT_CHECKLIST.md
docs/research/STABLE_LOOP_PHASE_LOCK_051_CLAIM_BOUNDARY.md
docs/research/STABLE_LOOP_PHASE_LOCK_051_LIMITATIONS.md
docs/research/STABLE_LOOP_PHASE_LOCK_051_ABLATION_NARRATIVE.md
docs/research/STABLE_LOOP_PHASE_LOCK_051_TABLES.md
scripts/probes/run_stable_loop_phase_lock_051_reviewer_bundle_check.py
```

## Verdicts

```text
PAPER_REPRODUCTION_BUNDLE_POSITIVE
REVIEWER_README_WRITTEN
ARTIFACT_CHECKLIST_WRITTEN
CLAIM_BOUNDARY_DOCUMENTED
LIMITATIONS_DOCUMENTED
ABLATION_NARRATIVE_WRITTEN
PAPER_TABLES_INCLUDED
REPRO_COMMANDS_INCLUDED
EXPECTED_HASHES_REFERENCED
REVIEWER_BUNDLE_CHECK_PASSES
PRODUCTION_API_NOT_READY
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
