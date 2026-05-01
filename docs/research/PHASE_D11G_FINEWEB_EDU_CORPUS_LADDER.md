# Phase D11g: FineWeb-Edu Tiny Corpus Ladder

Date: 2026-05-01

## Summary

D11g moved the H512 scaling check off the tiny Alice fixture and onto a local
FineWeb-Edu slice. This was a corpus-ladder smoke, not a release checkpoint run.

Goal:

```text
Does the D11e H512 embedding-anchored bootstrap still work on a more realistic
English web corpus, or was the signal Alice-specific?
```

## Corpus Artifact

Source:

```text
S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/*.parquet
```

Extraction command shape:

```text
extract_fineweb_txt --max-bytes 1000000 --max-files 1
  --min-language-score 0.95 --min-int-score 3
```

Output:

```text
output/phase_d11g_fineweb_edu_tiny_20260501/fineweb_edu_1m.txt
```

Extractor sanity:

- raw bytes written: `1,000,000`
- docs emitted / filtered / empty: `203 / 261 / 0`
- pass rate: `43.75%`
- post 27-class filter: `958,555` chars
- letters / whitespace: `82.7% / 17.3%`

## Corpus Load Sanity

The existing `diag_5way_fineweb` integration smoke loaded and trained on the
extracted corpus successfully.

Result:

- corpus len: `958,555` classes
- best heldout test: `45.60%`
- exit: OK

This proves the FineWeb-Edu text artifact is loadable and trainable through the
existing Rust corpus path.

## H512 VRAXION Smoke

Shared H512 configuration:

```text
--H 512
--embedding-anchored-highways 2
--diversity-guard-lambda 0.1
--jackpot 9
--ticks 6
--accept-policy strict
```

### Init-only

Result:

- final accuracy: `0.20%`
- quick diversity: `2/4` unique predictions
- average charge diff: `26.1%`

Verdict: H512 is not dead on FineWeb-Edu.

### 500-step guarded smoke

Result:

- final accuracy: `0.70%`
- peak accuracy: `0.70%`
- accept rate: `18.0%`
- quick diversity: `4/4`
- average charge diff: `17.5%`

Chain diagnosis:

- input `0..31` reaches output: `32/32`
- output charge diversity: `8.0%`
- unique predictions: `4/8`
- context effect: present

Verdict: `D11G_FINEWEB_H512_SHORT_SIGNAL`

### 2k guarded smoke

Result:

- final accuracy: `0.30%`
- peak accuracy: `0.40%`
- accept rate: `17.45%`
- quick diversity: `2/4`
- average charge diff: `13.6%`

Chain diagnosis:

- input `0..31` reaches output: `32/32`
- output charge diversity: `5.6%`
- unique predictions: `3/8`
- context effect: present

Verdict: `D11G_FINEWEB_H512_SEARCHABLE_BUT_WEAK`

## Interpretation

D11g confirms that the D11e H512 bootstrap is not Alice-only. The H512 network
still receives input-dependent signal and still accepts mutations on a real
FineWeb-Edu slice.

However, the 2k search did not improve past the 500-step peak. This is a weak
corpus-generalization signal, not a release-candidate result.

The main finding is:

```text
H512 scaling is now technically viable on FineWeb-Edu tiny,
but the current objective/search recipe is not yet strong enough for release.
```

## Next Gate

Do not jump to H1024/H8192 yet.

Recommended next step:

```text
D11h H512 FineWeb-Edu variant sweep
```

Arms:

- `diversity_guard_lambda = 0.05`
- `diversity_guard_lambda = 0.2`
- `embedding_anchored_highways = 4`
- `embedding_anchored_highways = 2 + lambda 0.2`

Run shape:

- 500-step timing/signal smoke per arm
- promote only arms with `>=0.70%` short accuracy and no diversity collapse
- then 10k on the best arm

## Progress Map

```text
GLOBAL RELEASE-READY AI MAP

[1] H384 top_01 research checkpoint
    DONE

[2] D10 artifact/state hardening
    DONE

[3] H512 dead baseline diagnosis
    DONE

[4] H512 embedding-anchored bootstrap
    DONE: D11e

[5] H512 Alice guarded search
    DONE: D11f peak 1.40%

[6] FineWeb-Edu tiny corpus ladder
    CURRENT RESULT: PASS-WEAK
    D11g confirms H512 is searchable on realistic English, but not strong yet

[7] H512 FineWeb variant sweep
    NEXT: D11h

[8] H512 long confirm
    BLOCKED until D11h finds a stronger arm

[9] release-ready AI candidate package
    BLOCKED until corpus-general H512 or independent H384 reproduction passes
```
