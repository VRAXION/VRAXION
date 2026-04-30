# Phase D10n High-H Structure Gate

Date: 2026-04-30

Verdict: `D10N_STRUCTURE_DEPENDENT_SCALING`

## Summary

D10n tested whether the D10m high-H flatness was caused by size itself, or by
bad density / structure / projection. The run reused the D10k GPU sparse scout
with adversarial controls.

Result:

```text
H4096 is a stable high-H positive reference.
H8192 is not dead: H8192/100k revives under motif-guided structure and survives
an eval_len=1000 confirm.
H16384 is not currently trustworthy: the apparent positives are either
control-failing or echo-trap dominated.
```

This means scaling is not monotonic, but it is also not purely local to H384.
The next scientific question is how to make the high-H structure fair and
stable, not whether the GPU can evaluate it.

## Scout Matrix

Short scout settings:

```text
eval_len: 128
eval_seeds: 986001,986002
proposals_per_arm: 16
arms: random_sparse, beta8_lifted, motif_guided
control: random_label_control
```

| run | verdict | control safe | key readout |
|---|---|---:|---|
| H4096 / 25k | `D10K_BETA8_PATTERN_SCALES` | 0/16 | beta8_lifted 4/16 safe, best MO 0.008885 |
| H4096 / 100k | `D10K_HIGH_H_SCOUT_SIGNAL` | 1/16 | all arms positive, but control mildly nonzero |
| H4096 / 400k | `D10K_HIGH_H_SCOUT_SIGNAL` | 0/16 | random/beta positive; motif flat |
| H8192 / 25k | `D10K_BETA8_PATTERN_SCALES` | 0/16 | very weak beta8_lifted 1/16 safe |
| H8192 / 100k | `D10K_BETA8_PATTERN_SCALES` | 0/16 | motif_guided 7/16 safe, best MO 0.011084 |
| H8192 / 400k | `D10K_BETA8_PATTERN_SCALES` | 0/16 | beta8_lifted echo-cliffy; motif 2/16 safe |
| H16384 / 25k | `D10K_NO_HIGH_H_SIGNAL` | 0/16 | flat |
| H16384 / 100k | `D10K_CONTROL_FAIL` | 5/16 | apparent signal rejected |
| H16384 / 400k | `D10K_HIGH_H_SCOUT_SIGNAL` | 3/16 | weak and control-contaminated |

## Confirm Runs

Two bands were confirmed with longer evaluation:

```text
eval_len: 1000
eval_seeds: 986001,986002,986003,986004
proposals_per_arm: 16
```

| run | verdict | control safe | random_sparse | beta8_lifted | motif_guided |
|---|---|---:|---|---|---|
| H4096 / 25k | `D10K_BETA8_PATTERN_SCALES` | 0/16 | 2/16 safe, MO 0.003241 | 3/16 safe, MO 0.004333 | 0/16 safe |
| H8192 / 100k | `D10K_BETA8_PATTERN_SCALES` | 1/16 | 1/16 safe, MO 0.003103 | 0/16 safe | 4/16 safe, MO 0.007031 |

Interpretation:

```text
H4096/25k is a reliable high-H reference point.
H8192/100k is the first revived higher-H structure point.
The H8192 signal is weaker and control-noisier than H4096, but it survives
longer eval and is not flat.
```

## Adversarial Notes

The adversarial checks changed the interpretation materially:

```text
H16384/100k looked strong on raw candidate counts, but random-label control
produced 5/16 safe-positive candidates. This run is rejected.

H16384/400k had a lower but nonzero control rate, and the real arms were mostly
echo-trap dominated. This is not a promotion signal.

H8192/100k is the best higher-H result because it survives eval_len=1000 and
the control safe rate is low, not because its raw scout count was largest.
```

## Long-Horizon Meaning

D10n upgrades the long-horizon state from "H8192+ looks flat" to:

```text
H8192 can be partially revived by structure and density.
H16384 still needs a better projection/start policy.
Scaling is structure-dependent, not automatic.
```

This supports a narrower version of the user's hypothesis:

```text
large sparse potential space can help,
but only if the active circuit is organized enough.
```

It does not support:

```text
just increase H and random sparse search will improve monotonically.
```

## Next Step

Run D10o as a fair high-H projection/start gate:

```text
D10o High-H Projection/Start Gate

1. Treat H4096/25k as positive reference.
2. Treat H8192/100k as the first revived target.
3. Build projection/start variants:
   - beta8_lifted_v2
   - motif_guided_without_echo_closure
   - projection-rescaled high-H variant
   - threshold-band variants
4. Confirm only candidates that pass random-label and echo controls.
```

No high-H release or "intelligence" claim follows from D10n alone.

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
    DONE:
      D10j/D10m mapped throughput frontier

[5] high-H sparse search
    DONE:
      D10k H1024/25k positive
      D10n H4096 stable, H8192 revived, H16384 not yet trustworthy

[6] next
    D10o projection/start gate
    goal: turn H8192/H16384 from scout signal into controlled repeatable signal
```
