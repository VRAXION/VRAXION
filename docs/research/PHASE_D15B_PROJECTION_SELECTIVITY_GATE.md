# Phase D15B: Projection Selectivity Gate

Date: 2026-05-02

## Purpose

D15A showed that H=16384 / 400k-edge sparse high-H runs are feasible on the RTX 4070 Ti SUPER, but the best raw high-H signals were still vulnerable to semantic controls. D15B adds a stricter post-hoc gate:

```text
For each candidate, compare the real row against that same candidate's
random-label, random-bigram, unigram-decoy, and projection-shuffle controls.
```

This prevents arm-level safe-rate summaries from hiding candidate-level projection/readout artifacts.

## Inputs

- `output/phase_d15a_high_h_max_structured_scout_20260502/main_h16384_400k`
- `output/phase_d15a_high_h_max_structured_scout_20260502/confirm_beta8_lifted_h16384_400k`
- `output/phase_d15a_high_h_max_structured_scout_20260502/confirm_block_local_h16384_400k`

## Method

A candidate is only selective if:

- the real row is safe,
- no adversarial control row for the same candidate is safe,
- and the real `mo_score` beats the best control `mo_score`.

Classes:

- `SELECTIVE_STRONG`: margin >= `0.001`
- `SELECTIVE_WEAK`: margin >= `0.00025`
- `SELECTIVE_UNDER_MARGIN`: positive but below weak margin
- `CONTROL_SAFE_REJECT`: at least one control is also safe
- `CONTROL_MO_REJECT`: control MO is at least as high as real MO

## Results

The short D15A main run had small candidate-level selective signals:

| Run | Arm | Candidate-Level Result |
|---|---|---|
| D15A main | `beta8_lifted_v2` | 1 strong, 1 weak |
| D15A main | `block_local_projection` | 1 strong |
| D15A main | `threshold_mid` | 1 weak |

Both longer confirms blocked the apparent signal:

| Confirm | Arm | D10p Verdict | D15B Verdict | Reason |
|---|---|---|---|---|
| beta8 confirm | `beta8_lifted_v2` | `SEMANTIC_FAIL` | `D15B_PROJECTION_SELECTIVITY_BLOCKED` | no strong/weak selective candidate survived |
| block-local confirm | `block_local_projection` | `SEMANTIC_FAIL` | `D15B_PROJECTION_SELECTIVITY_BLOCKED` | all 31 real-safe rows were control-safe rejects |

Aggregate D15B verdict:

```text
D15B_PROJECTION_SELECTIVITY_BLOCKED
```

## Interpretation

The high-H space is runnable and reactive, but the current projection/readout path still produces target-independent or control-compatible wins at H=16384 / 400k edges. The main-run candidate-level sparks did not survive longer evaluation.

This means H=16384 brute-force or "more exhaustive" search should remain blocked for release-candidate purposes until the projection/readout objective is redesigned.

## Release-Ready Impact

```text
GLOBAL RELEASE-READY AI MAP

[1] H384 top_01 research checkpoint
    DONE: packaged and artifact-checked

[2] H384 atlas expansion
    DONE: wide/control-dominated signal

[3] H16384 / 400k high-H scout
    DONE: GPU feasible, raw signal exists

[4] Projection selectivity
    DONE: confirm blocked

[5] Next release-relevant work
    projection/readout redesign or H384 top_01 demo/capability packaging
```

Current best release-track asset remains the H384 `top_01` research checkpoint. High-H remains an infrastructure/science finding, not a release candidate.
