# Phase D16: Top-01 Context Gate

Date: 2026-05-02

## Purpose

D13 established a trusted H384 `top_01` research checkpoint. D15B established
that current high-H projection/readout runs are still control-blocked. The next
release-relevant question is therefore:

```text
Does the trusted H384 checkpoint carry sequential context yet?
```

This is the bridge from "verified research checkpoint" toward a usable
language-like demo. A next-token system must be able to make the same current
token behave differently depending on previous tokens.

## Input

```text
checkpoint = output/releases/v5.0.0-beta.10/seed2042_top01_h384_research.ckpt
packed     = output/block_c_bytepair_champion/packed.bin
```

## Result

```text
D16_CONTEXT_BLOCKED
```

Measured facts from `chain_diagnosis`:

| Metric | Value |
|---|---:|
| H | 384 |
| checkpoint step | 40000 |
| checkpoint accuracy | 4.10% |
| input pairwise differentiation | 24.8 / 32 dims |
| input neurons reaching output | 32 / 32 |
| output pairwise differentiation | 56.0 dims |
| unique predictions on probe pairs | 2 / 8 |
| context-dependent predictions | 0 / 4 |

## Interpretation

The network is not dead:

- input vectors differ,
- all 32 input neurons in the primary embedding zone reach output,
- output charge patterns differ across byte-pairs.

But the recurrent state is not yet carrying useful sequence information in this
probe. Sequential and isolated token passes produce the same predictions.

## Training Decision

The next useful work is training/search, but not large-H brute force.

The correct training target is:

```text
H384 top_01 + edge/threshold search with an explicit context-carrying objective
```

Not:

```text
H16384/H8192 brute-force scaling with the current projection/readout objective
```

High-H remains blocked until projection/readout controls are clean. The trusted
near-term path is to improve the H384 checkpoint's context behavior first.

## Progress Map

```text
Release-ready AI
[=======___] ~74-76%

[1] artifact-safe H384 checkpoint
    DONE: top_01

[2] high-H brute force
    BLOCKED: projection/selectivity controls

[3] context-carrying capability
    CURRENT: D16_CONTEXT_BLOCKED

[4] next unlock
    D16b context objective + edge/threshold training around top_01

[5] release-candidate demo
    only after context-dependent predictions become stable
```
