# Phase D16d Context Margin Polish

Date: 2026-05-02

## Purpose

D16c produced an artifact-clean weak context lead around `top_07`, but did not reach the full context-margin gate. D16d tested whether low-risk threshold-only polish around `top_07` can raise the real-vs-fake context margin without damaging the existing safety metrics.

## Setup

- Start checkpoint: `output/phase_d16b_context_climb_main_20260502/candidates/top_07.ckpt`
- Safety reference: `output/releases/v5.0.0-beta.10/seed2042_top01_h384_research.ckpt`
- Mode added: `context-margin-climb`
- Mutation scope: `threshold` only
- Smoke: `eval_len=512`, 20 proposals
- Main: `eval_len=1000`, 8 eval seeds, 240 proposals

The original 12x80 main was rescoped to 6x40 after smoke timing projected a 3+ hour run. This preserved the decision goal while keeping the run bounded.

## Result

Threshold-only D16d did not find a clean margin-improving candidate.

```text
total proposals: 240
accepted: 0
full pass: 0
best exported candidates: artifact-fail
```

The best safe exported candidate improved raw margin but failed because fake controls remained too competitive:

```text
best margin_mean: +0.000750
best margin_lower95: +0.000136
fake_beat_rate: 0.500
verdict: D16C_CONTEXT_ARTIFACT_FAIL
```

Observed pattern:

```text
positive margin + safety pass  -> usually fake/artifact dominated
stronger real margin           -> safety tradeoff, mostly unigram/smooth damage
accepted clean proposal        -> none
```

## Decision

Release promotion remains blocked.

D16d falsified the simplest low-risk path:

```text
top_07 + threshold-only polish is not enough to reach full context margin.
```

The next useful branch is either:

1. `edge,threshold` margin-aware fallback with a small bounded budget, or
2. context objective/readout redesign if the edge+threshold fallback also turns into artifact/safety tradeoff.

Do not run high-H brute force from this result.

## Progress Map

```text
Release-ready AI:
[========__] ~80%

[1] artifact-safe H384 top_01
    DONE

[2] partial context found
    DONE: D16b

[3] real-vs-fake context checked
    DONE: D16c weak-pass top_07

[4] threshold-only polish
    DONE: D16d
    verdict: no clean local threshold gain
        |
        v
[5] edge+threshold margin fallback or objective/readout redesign
    NEXT

[6] release-candidate package
    BLOCKED until full context margin pass + long confirm
```

## Artifacts

- `output/phase_d16d_context_margin_polish_20260502/smoke/`
- `output/phase_d16d_context_margin_polish_20260502/threshold_main/`

