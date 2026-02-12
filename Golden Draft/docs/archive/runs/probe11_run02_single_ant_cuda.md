# Probe 11 — Run 02: Single Ant Baseline (CUDA)

**Date:** 2026-02-09
**Version:** 2.10.581
**Branch:** nightly
**Device:** CUDA
**Steps completed:** 237 / 2500 (killed early — baseline data collected)

## Config

| Parameter | Value |
|-----------|-------|
| Task | Harmonic XOR: x[t] = sin(F=1) + sin(F=7), label = XOR(sign_slow, sign_fast) |
| Active ants | 1 / 7 (ant[0] only — ring_len=1071, weight=0.505) |
| Batch size | 16 |
| Seq len | 64 |
| LR | 1e-3 (Adam) |
| Checkpoint every | 10 steps |
| AGC | Enabled (defaults: grad_low=1.0, grad_high=5.0, scale_max=1.0) |
| Seed | 42 |

## Trajectory

| Step | MA100 | MA50 | MA10 | Loss | Gnorm |
|------|-------|------|------|------|-------|
| 1 | 62.5% | 62.5% | 62.5% | 2.35 | 2714 |
| 50 | 56.6% | 56.6% | 62.5% | 3.46 | 164 |
| 100 | 62.4% | 68.1% | 70.0% | 1.25 | 137 |
| 150 | 71.4% | 74.6% | 79.4% | 2.13 | 100 |
| 200 | 74.5% | 74.4% | 73.1% | 0.34 | 307 |
| 237 | 75.6% | 76.0% | 70.6% | 0.34 | 21 |

**Status at kill:** Still climbing. Not plateaued yet.

## Checkpoints saved

23 step checkpoints (every 10 steps, steps 10–230) + `best.pt` + `latest.pt`.
Each checkpoint: 2.5 MB (model + optimizer state + step + metrics).
Directory: `Golden Draft/logs/probe/checkpoints/`

## Purpose

Baseline run to establish single-ant performance on harmonic XOR with volume weighting
before activating additional ants. This is the same task and config as Run 01 (which
reached 87% MA100 at step 500+ but had NO checkpointing — weights were lost).

## Key observations

1. Trajectory matches Run 01 — same initial chaos, same climb rate through 200 steps
2. Checkpointing confirmed working — 23 step files + best + latest all present
3. Gnorms are spiky (21–2714 range) — AGC is active but gradient variance is high
4. At step 237, still well below the ~87% plateau seen in Run 01 at step 500+

## Next steps

- Run multi-ant experiment (--active-ants 2 or -1) to test volume weighting hypothesis
- Key question: does volume weighting prevent gradient starvation seen in probe10?
- Can resume from any checkpoint with --resume flag
