# E20A_BINARY_GROUNDING_AUDIT_AND_HARDENING_CONFIRM Contract

## Purpose

E20A audits whether E20's clean binary-stream grounding result was genuine multi-pocket grounding or an artifact of oracle leakage, codebook leakage, static metrics, weak ablations, easy baselines, or single-codec collapse. It then runs a hardened binary grounding stress variant.

## Boundary

This is an audit and hardening milestone for a controlled synthetic codec-agnostic binary-stream grounding benchmark. It tests whether E20 remains valid under source audit, artifact audit, harder codebooks, stronger false alignment, partial observability, increased noise, and multi-pocket cross-modal necessity. It does not prove real audio understanding, real vision understanding, GPT-like generation, AGI, consciousness, or production readiness.

## Required phases

1. Static source audit of E20 runner/checker for oracle leakage, codebook separation, static metrics, ablation validity, baseline validity, collapse risk, and trace validity.
2. Artifact audit of E20 target artifacts when available, with at least 50 heldout and 50 stress sampled primary episodes.
3. Hardened E20A run with 6-16 entities, 64-256 stream lengths, 80-180 events, 4-7 modalities, stronger noise, stronger false alignment, partial observability, and cross-modal necessity.

## Full-confirm minimums

Full confirmation requires at least 80 generations, population 128, 1800 heldout episodes, 1800 stress episodes, 10000 candidate evaluations, 80 checkpoints, 800 cross-codec stress episodes, 600 missing/corrupt modality stress episodes, 800 heldout-codebook episodes, 500 adversarial false-alignment episodes, 1000 cross-modal-necessary episodes, and 100 sampled artifact-audit episodes if artifacts are available.

## Run command

```bash
python3 scripts/probes/run_e20a_binary_grounding_audit_and_hardening_confirm.py \
  --out target/pilot_wave/e20a_binary_grounding_audit_and_hardening_confirm \
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

python3 scripts/probes/run_e20a_binary_grounding_audit_and_hardening_confirm_check.py \
  --out target/pilot_wave/e20a_binary_grounding_audit_and_hardening_confirm \
  --write-summary
```
