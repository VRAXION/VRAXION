# Phase D10o High-H Projection/Start Gate

Date: 2026-04-30

Verdict: `D10O_SEMANTIC_CONTROL_BLOCKED`

## Summary

D10o tested whether better high-H starts and projection variants can turn the
D10n H8192/H16384 scout signal into a controlled repeatable signal.

Result:

```text
Raw high-H signal improved.
H8192/100k produced many safe-positive proposals.
H16384/100k also produced weak but repeatable safe-positive proposals.

However, random-label controls were too positive.
Therefore the current D10o projection/start setup is not semantically trusted.
```

This is not a failure of GPU scaling. It is a semantic-control/projection
problem: high-H projection variants can create metric improvements that also
appear under shuffled target labels.

## H8192 / 100k

Short scout, `eval_len=128`, 2 eval seeds:

| arm | safe | echo trap | best MO |
|---|---:|---:|---:|
| beta8_lifted_v2 | 6 | 1 | 0.013138 |
| motif_no_echo | 9 | 3 | 0.009599 |
| projection_tiled | 10 | 1 | 0.006938 |
| projection_signed | 0 | 12 | 0.007678 |
| threshold_mid | 0 | 12 | 0.010189 |
| threshold_high | 5 | 5 | 0.006180 |

Random-label control: `3/16` safe-positive.

Confirm, `eval_len=1000`, 4 eval seeds, structured arms only:

| arm | safe | echo trap | best MO |
|---|---:|---:|---:|
| beta8_lifted_v2 | 7 | 0 | 0.005345 |
| motif_no_echo | 7 | 0 | 0.003073 |

Random-label control using `beta8_lifted_v2`: `13/16` safe-positive.

Interpretation:

```text
H8192/100k raw signal is strong.
But it fails semantic adversarial control.
Do not treat it as a real basin yet.
```

## H16384 / 100k

Short scout, `eval_len=128`, 2 eval seeds:

| arm | safe | echo trap | best MO |
|---|---:|---:|---:|
| beta8_lifted_v2 | 5 | 0 | 0.002172 |
| motif_no_echo | 0 | 0 | 0.000000 |
| projection_tiled | 0 | 5 | 0.004721 |
| threshold_mid | 6 | 0 | 0.005897 |
| threshold_high | 3 | 4 | 0.003096 |

Random-label control: `1/8` safe-positive.

Confirm, `eval_len=1000`, 4 eval seeds:

| arm | safe | echo trap | best MO |
|---|---:|---:|---:|
| beta8_lifted_v2 | 9 | 0 | 0.001416 |
| threshold_mid | 3 | 0 | 0.000894 |

Random-label control using `threshold_mid`: `2/16` safe-positive, plus
`11/16` positive-unsafe.

Interpretation:

```text
H16384/100k is weaker than H8192 but less catastrophically control-failing.
The effect size is small and the control is noisy.
This is a weak structure signal, not a pass.
```

## What This Means

D10o narrows the long-horizon bottleneck:

```text
Not blocked by GPU.
Not blocked by H size.
Not blocked by raw search signal.

Blocked by semantic trust:
  projection/eval can produce apparent gains under shuffled targets.
```

The high-H path is still alive, but the next step must harden the semantic
controls and projection design before claiming any high-H basin.

## Adversarial Lessons

```text
projection_tiled:
  increases raw signal but also creates target-shuffle false positives.

beta8_lifted_v2:
  strong H8192 raw signal but huge random-label leakage in confirm.

threshold_mid:
  useful at H16384, but control-positive-unsafe rate remains high.

motif_no_echo:
  reduced echo traps compared with earlier motif_guided, but does not solve
  semantic leakage by itself.
```

## Next Step

D10p should be a semantic/projection hardening gate, not another H-scaling run:

```text
D10p Semantic Projection Hardening

1. Add stricter controls:
   - random-label
   - random-bigram
   - unigram-only decoy
   - projection-shuffle control

2. Separate metric components:
   - smooth-only gain
   - accuracy gain
   - unigram lure
   - echo leakage

3. Test projection variants:
   - non-tiled projection
   - block-local projection
   - frozen beta8 rows only
   - learned/rescaled high-H projection if available

4. Only accept a high-H start if:
   - real labels show safe-positive rate
   - all shuffled/decoy controls stay near zero
   - eval_len=1000 confirm survives
```

No high-H promotion claim follows from D10o.

## Progress Map

```text
GLOBAL AI PLAN MAP

[1] H384 beta.8 generalist
    DONE

[2] causal mechanism
    DONE: edge + threshold co-adaptation

[3] D10b H384 seed replication
    RUNNING

[4] high-H GPU feasibility
    DONE: GPU reaches huge H

[5] high-H structure
    DONE:
      D10n -> H8192 can be revived
      D10o -> raw H8192/H16384 signal exists, but semantic controls fail/noisy

[6] next
    D10p semantic projection hardening
    goal: eliminate target-shuffle false positives before more scaling
```
