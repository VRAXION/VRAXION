# Phase D16c Context Margin Confirm

Date: 2026-05-02

## Purpose

D16b found partial context-dependent behavior around the H384 `top_01` checkpoint, but the raw context gain was close to time-shuffled/fake context gain. D16c adds a stricter confirmation gate:

```text
context_margin =
  real sequential context gain
  - max(time_shuffle_gain, state_shuffle_gain, random_context_gain, no_network_gain)
```

This separates real order-carrying state from generic reverberation or projection/readout pulse artifacts.

## Inputs

- Reference checkpoint: `output/releases/v5.0.0-beta.10/seed2042_top01_h384_research.ckpt`
- D16b candidates: `output/phase_d16b_context_climb_main_20260502/candidates/top_01.ckpt` through `top_08.ckpt`
- Main eval: `eval_len=4000`, 16 seeds, 2 context-control repeats
- Artifact gate survivor check: D10r-v8 controls on `top_07`

## Result

Main D16c verdict: `D16C_CONTEXT_MARGIN_WEAK_PASS`

No candidate reached full `D16C_CONTEXT_MARGIN_PASS` because the required `margin_lower95 >= +0.0010` was not met.

| candidate | verdict | margin mean | margin lower95 | strongest fake | safety |
|---|---|---:|---:|---|---|
| `top_07` | `D16C_CONTEXT_MARGIN_WEAK_PASS` | `+0.000844` | `+0.000556` | `random_context` | pass |
| `top_03` | `D16C_CONTEXT_MARGIN_WEAK_PASS` | `+0.000578` | `+0.000307` | `random_context` | pass |
| `top_06` | `D16C_CONTEXT_ARTIFACT_FAIL` | `+0.000727` | `+0.000453` | `random_context` | pass |
| `top_02` | `D16C_CONTEXT_ARTIFACT_FAIL` | `+0.000398` | `+0.000172` | `random_context` | pass |

Reload sanity:

- `top_07`: `D16_PARTIAL_CONTEXT_SIGNAL`, `1/4` context-dependent predictions
- `top_03`: `D16_PARTIAL_CONTEXT_SIGNAL`, `2/4` context-dependent predictions

Artifact/state gate:

- `top_07`: `D10R_V8_STATE_IDENTITY_PASS`
- Blocking artifact controls: none

## Interpretation

D16c did not prove a release-grade context-carry candidate. It did prove that D16b was not pure noise: `top_07` and `top_03` have a small positive real-vs-fake context margin, keep the existing safety metrics, and survive reload sanity.

The best candidate, `top_07`, also survives the D10r-v8 artifact/state gate. That makes it a valid research lead, not a release candidate.

## Decision

Release promotion remains blocked.

Next step is not high-H brute force. The next useful step is D16d: margin-aware threshold polish or context-climb, where the search objective directly optimizes positive context margin instead of raw context gain.

## Progress Map

```text
Release-ready AI:
[========__] ~80%

[1] artifact-safe H384 top_01
    DONE

[2] partial context appears
    DONE: D16b

[3] real-vs-fake context margin
    DONE: D16c weak pass
        |
        |-- full pass?
        |     no
        |
        '-- weak pass + artifact clean?
              yes: top_07

[4] D16d margin-aware polish/search
    NEXT

[5] long confirm / release-candidate package
    BLOCKED until D16d reaches full context margin pass
```

## Artifacts

- `output/phase_d16c_context_margin_confirm_20260502/main/context_margin_summary.csv`
- `output/phase_d16c_context_margin_confirm_20260502/main/D16C_CONTEXT_MARGIN_CONFIRM_REPORT.md`
- `output/phase_d16c_context_margin_confirm_20260502/main/top07_d10r_v8_gate/D10R_HARDENED_EVAL_REPORT.md`

