# Phase D11e: H512 Embedding-Anchored Highway Bootstrap

Date: 2026-05-01

## Summary

D11a-D11d showed that the naive H512 D7-style baseline was not a fair scaling test:
the network was effectively dead at the readout. The VCBP input vectors differed
normally, but the real embedding dimensions `0..31` did not preserve
input-dependent signal into the H512 output/readout zone.

D11e adds an opt-in H512 bootstrap path to `evolve_mutual_inhibition`:

- `--embedding-anchored-highways N`
- `--diversity-guard-lambda X`

The fix does not resize the H384 checkpoint. It keeps H512 as a fresh network,
but anchors the real VCBP embedding dimensions directly into isolated readout
neurons so H512 starts with input-dependent output states.

## What Was Broken

`InitConfig::phi(512)` has a larger input/output overlap than H384, so the issue
was not missing overlap.

The issue was zone alignment:

- VCBP semantic input lives in dimensions `0..31`.
- H512 overlap starts much later (`196..316`).
- Random/rooted highways did not preserve enough input-dependent analog signal
  from `0..31` into the output zone.

Result before D11e:

- H512 strict baseline: `0%` accept, constant output.
- H512 scatter/ties variants: output reached, but stayed near-dead or constant.
- H512 chain diagnosis: input differentiation existed, but output diversity was
  `0.0%`.

## Code Change

D11e adds embedding-anchored highways after rooted pathways and before mutual
inhibition:

- each real embedding dimension gets `per_dim` direct output anchors;
- direct anchor targets have spurious incoming edges removed;
- anchor readout neurons use `threshold=15`, `channel=8`, `polarity=+1`;
- channel 8 prevents readout anchors from spike-resetting during a 6-tick token;
- a guarded optional score term rewards input-dependent output diversity.

The diversity guard is off by default and only changes behavior when
`--diversity-guard-lambda` is nonzero.

## Results

### Init-only D11e Anchor Smoke

Command shape:

```text
--H 512 --steps 0 --embedding-anchored-highways 2
```

Result:

- input `0..31` reaches output: `32/32`
- output impact: `3043` output-dim changes
- output charge diversity: `12.7%`
- unique predictions: `4/8`
- context effect: present

Verdict: `D11E_H512_IGNITION_PASS`

### Strict 1k Without Diversity Guard

Command shape:

```text
--H 512 --steps 1000 --embedding-anchored-highways 2
```

Result:

- accept rate: `16.1%`
- final accuracy: `0.40%`
- but output collapsed back to one dominant prediction

Interpretation: H512 was no longer dead, but the smooth/accuracy objective still
favored a single-attractor shortcut.

### Diversity-Guarded 2k Pilot

Command shape:

```text
--H 512 --steps 2000 --embedding-anchored-highways 2 --diversity-guard-lambda 0.1
```

Result:

- final accuracy: `0.50%`
- accept rate: `11.8%`
- quick adversarial diversity: `4/4` unique predictions
- chain diagnosis unique predictions: `4/8`
- output charge diversity: `8.4%`
- context effect: present

Verdict: `D11E_H512_BOOTSTRAP_AND_SEARCH_SIGNAL`

## Interpretation

D11e fixes the H512 activation/readout bottleneck. The previous H512 failure was
not evidence that scaling cannot work; it was an invalid H512 baseline because
the real embedding signal did not survive into the readout.

The new H512 path now has three required properties:

- input-dependent readout states;
- nonzero accepted mutations under strict search;
- a guard against immediate collapse into a constant-prediction attractor.

This is not yet a release-ready H512 checkpoint. It is a scaling pilot unlock.

## Next Gate

Run a longer D11f H512 guarded search:

- `--H 512`
- `--embedding-anchored-highways 2`
- `--diversity-guard-lambda 0.1`
- steps: `10k` bounded first, then `40k` only if the 10k run improves without
  diversity collapse.

Promotion remains blocked until H512 reaches a real multi-metric confirm.

## Progress Map

```text
GLOBAL RELEASE-READY AI MAP

[1] H384 top_01 release-candidate research checkpoint
    DONE

[2] D10 artifact/state-shuffle hardening
    DONE for H384 top_01

[3] H512 naive baseline
    FAILED: dead/constant readout

[4] H512 activation diagnosis
    DONE: real embedding dims did not preserve output signal

[5] H512 embedding-anchored highway bootstrap
    CURRENT RESULT: PASS
    D11e: input-dependent H512 readout + guarded search signal

[6] H512 guarded long search
    NEXT: D11f 10k/40k

[7] H512 multi-metric confirm
    BLOCKED until D11f produces a strong candidate

[8] release candidate package
    BLOCKED until H512 or independent H384 reproduction passes confirm
```
