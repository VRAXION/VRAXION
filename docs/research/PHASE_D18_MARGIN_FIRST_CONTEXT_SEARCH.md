# Phase D18 Margin-First Context Search

Date: 2026-05-02

## Summary

D18 changed the local H384 context search from "find any context-looking pulse" to the actual release-relevant objective:

```text
real sequential context gain
- max(fake context gains)
+ safety retention
+ EQ-bar/smooth retention
```

The search started from:

- Golden reference: `output/releases/v5.0.0-beta.10/seed2042_top01_h384_research.ckpt`
- Weak context leads: D16b `top_03.ckpt` and `top_07.ckpt`

## Verdict

```text
D18_MARGIN_CONTEXT_WEAK_RELEASE_PATH
```

D18 found artifact-clean context-margin candidates, but the best survivor remains just below the full confidence gate after the longer confirm.

## Evidence

### Main Search

Output root:

```text
output/phase_d18_margin_first_context_search_20260502/main
```

Main result:

```text
verdict:             D18_MARGIN_CONTEXT_PASS
total candidates:    1440
exported candidates: 12
accepted count:      14
pass count:          3
weak count:          27
output traps:        169
best start:          start_02_top_07
best lower95:        +0.001016
best fake beat rate: 0.25
```

This means the margin-first objective did find local candidates where real sequential context beat fake context controls while safety stayed intact.

### Top-3 Confirm

Output root:

```text
output/phase_d18_margin_first_context_search_20260502/confirm_top3
```

Confirm shape:

```text
eval_len: 4000
eval seeds: 974321..974336
control repeats: 2
```

Top confirm result:

```text
verdict:         D16C_CONTEXT_MARGIN_WEAK_PASS
best candidate:  D18 start_02_top_07/top_02.ckpt
margin mean:     +0.001125
margin lower95:  +0.000922
real gain:       +0.007985
strongest fake:  random_context +0.006258
safety pass:     true
```

The full pass threshold is `margin_lower95 >= +0.001000`, so the best candidate missed full pass by about `0.000078`.

### Reload Context Gate

Best D18 survivor:

```text
output/phase_d18_margin_first_context_search_20260502/main/candidates/start_02_top_07/top_02.ckpt
```

Reload gate:

```text
verdict:                       D16_PARTIAL_CONTEXT_SIGNAL
context-dependent predictions: 2/4
unique predictions:            3
```

This confirms that the context behavior survives checkpoint reload.

### D10r Artifact / State Gate

Output root:

```text
output/phase_d18_margin_first_context_search_20260502/d10r_gate_top02
```

Artifact result:

```text
verdict:           D10R_V8_STATE_IDENTITY_PASS
trusted MO:        +0.066555
trusted MO CI low: +0.001518
blocking controls: none
```

This means the D18 survivor did not fail the known random projection, no-network, or state-shuffle artifact controls.

## Interpretation

D18 moved the project forward because it separated two cases that D17 could not resolve:

```text
D17:
  raw EQ-bar bump exists, but fake context moves with it.

D18:
  when the search objective directly optimizes real-vs-fake margin,
  it can find artifact-clean partial context candidates.
```

However, D18 did not yet produce a promotion-ready context checkpoint. The confirmed margin is positive and artifact-clean, but still too close to the full confidence threshold.

## Release Impact

```text
Release-ready AI:
[========__] ~81%
```

Progress:

- H384 top_01 remains the golden artifact-safe checkpoint.
- D18 adds the first artifact-clean context-margin lead.
- Promotion remains blocked until the margin clears the full gate and survives long confirm.

## Next Step

The next best step is a small targeted D18b/D19 polish around `start_02_top_07/top_02.ckpt`, not H512/H8192 brute force.

Recommended run:

```text
threshold-first or edge+threshold local polish
objective: increase margin_lower95 above +0.001000
keep safety vs golden top_01
keep fake beat rate <= 0.25
then rerun eval_len=4000 confirm
```

Stop rule:

```text
If margin_lower95 cannot clear +0.001000 after one bounded polish,
stop local context mutation and redesign the context objective/readout.
```

## Progress Map

```text
[1] artifact-safe H384 top_01
    DONE

[2] weak context lead
    DONE: D16b/D16c top_07

[3] local landscape scan
    DONE: D17 output-only trap

[4] margin-first search
    DONE: D18 artifact-clean weak context lead
        |
        |-- full pass: not yet
        |
        '-- weak pass:
              NEXT: targeted margin polish around D18 top_02

[5] release candidate package
    BLOCKED until:
      full context margin pass
      D10r artifact pass
      16k / 30-seed long confirm
```
