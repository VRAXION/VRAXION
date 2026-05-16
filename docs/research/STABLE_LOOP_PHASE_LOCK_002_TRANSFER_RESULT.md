# STABLE_LOOP_PHASE_LOCK_002_TRANSFER Result

## Status

Implemented and ran the transfer probe after `STABLE_LOOP_PHASE_INTERFERENCE_003_PHASE_LOCK` showed a clean fixed-Prismion phase-lock win.

This run answers the adversarial question:

```text
Was the win Prismion-specific, or did the task reward a fixed/local complex multiplication primitive?
```

## Run

```text
out=target/pilot_wave/stable_loop_phase_lock_002_transfer/cpu_3seed_capped
phase_classes=4
width=32
seeds=2026,2027,2028
train_examples=1024
eval_examples=1024
epochs=3
jobs=6
device=cpu
completed_jobs=30
```

CPU was capped to `jobs=6` after `auto80` resolved to 19 workers on this machine and was too aggressive for interactive use.

## Verdict

```text
PHASE_LOCK_TRANSFER_TASK_VALID
FIXED_PRISMION_TRANSFER_POSITIVE
COMPLEX_MULTIPLY_SUFFICIENT
FIXED_PRIMITIVE_ONLY
```

Not supported:

```text
LEARNED_PRISMION_TRANSFER_POSITIVE
PRISMION_UNIQUELY_USEFUL
CANONICAL_MESSAGE_PASSING_RECOVERS
SUMMARY_OR_TARGET_SHORTCUT_RETURNS
```

## Summary Table

| Arm | Accuracy | Pair | Heldout Gate | Noisy Gate | Label Shuffle | Gate Shuffle | Pre-Wall | Wall Leak | Params | Trainable | Operator |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ORACLE_PHASE_LOCK_TRANSFER` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `23.1%` | `100.0%` | `0.0000` | `0.0%` | `0` | `0` | `ORACLE` |
| `FIXED_PRISMION_PHASE_LOCK_LOOP` | `99.7%` | `99.7%` | `99.4%` | `100.0%` | `23.0%` | `24.2%` | `0.0014` | `0.0%` | `0` | `0` | `FIXED_OPERATOR` |
| `COMPLEX_MULTIPLY_GNN` | `99.7%` | `99.7%` | `99.4%` | `100.0%` | `23.0%` | `24.2%` | `0.0014` | `0.0%` | `0` | `0` | `GENERIC_COMPLEX_MULTIPLY` |
| `LOCAL_MESSAGE_PASSING_GNN_PHASE_LOCK` | `39.4%` | `21.1%` | `38.5%` | `36.6%` | `22.7%` | `29.2%` | `0.0016` | `0.0%` | `0` | `0` | `CANONICAL_MESSAGE_PASSING` |
| `LOCAL_BILINEAR_PHASE_LOOP` | `19.2%` | `15.8%` | `15.3%` | `16.8%` | `7.0%` | `4.8%` | `0.2617` | `0.0%` | `18` | `18` | `GENERIC_BILINEAR` |
| `HARD_WALL_ABC_PHASE_LOCK_LOOP` | `19.1%` | `14.6%` | `14.9%` | `16.3%` | `7.5%` | `5.6%` | `0.4298` | `0.0%` | `2402` | `2402` | `LEARNED_LOCAL_CONV` |
| `LEARNED_PRISMION_PHASE_LOCK_LOOP` | `18.8%` | `16.4%` | `13.9%` | `17.0%` | `4.8%` | `4.8%` | `0.0598` | `0.0%` | `6` | `6` | `LEARNED_PRISMION_FACTOR` |
| `SUMMARY_DIRECT_HEAD` | `19.3%` | `12.9%` | `18.1%` | `17.8%` | `10.8%` | `11.5%` | `0.0000` | `0.0%` | `13253` | `13253` | `GLOBAL_SUMMARY_CONTROL` |
| `TARGET_MARKER_ONLY` | `18.8%` | `13.2%` | `15.8%` | `17.9%` | `11.0%` | `11.0%` | `0.0000` | `0.0%` | `5` | `5` | `TARGET_PRIOR_CONTROL` |
| `UNTIED_LOCAL_CNN_TARGET_READOUT_PHASE_LOCK` | `15.3%` | `10.2%` | `12.5%` | `13.3%` | `11.3%` | `11.3%` | `0.0000` | `0.0%` | `528674` | `528674` | `UNTIED_LOCAL_COMPUTE` |

Chance context:

```text
random_baseline    ~= 20.0%
majority_baseline  ~= 24.7%
chance_threshold   ~= 29.7%
```

The summary and target controls stayed below the chance threshold, so the result is not explained by target-marker or global-summary shortcut.

## Matched Seed Deltas

```text
Learned Prismion - ABC:
  mean_delta   = -0.0029
  lower95      = -0.0045
  positive     = 0/3 seeds

Learned Prismion - GNN:
  mean_delta   = -0.2062
  lower95      = -0.2159
  positive     = 0/3 seeds

Learned Prismion - Local Bilinear:
  mean_delta   = -0.0044
  lower95      = -0.0113
  positive     = 0/3 seeds

Learned Prismion - Complex Multiply GNN:
  mean_delta   = -0.8089
  lower95      = -0.8117
  positive     = 0/3 seeds

Fixed Prismion - Learned Prismion:
  mean_delta   = +0.8089
  lower95      = +0.8060
  positive     = 3/3 seeds
```

## Interpretation

The previous phase-lock win was real, but this transfer probe narrows what it means.

```text
Fixed Prismion phase-lock:
  99.7%

Generic complex multiply GNN:
  99.7%

Learned Prismion factor:
  18.8%
```

So the correct claim is:

```text
local complex multiplication / phase transport is sufficient for this task.
```

The current learned Prismion parameterization did not recover the fixed primitive from final labels in this short transfer run.

The current result is therefore:

```text
COMPLEX_MULTIPLY_SUFFICIENT
FIXED_PRIMITIVE_ONLY
```

not:

```text
PRISMION_UNIQUELY_USEFUL
LEARNED_PRISMION_TRANSFER_POSITIVE
```

## What This Proves

This does prove that the phase-lock result was not a summary/target shortcut and that the fixed operator survives transfer variants in the tested regime.

It also proves that the fixed Prismion arm is functionally equivalent to a generic explicit complex-multiply local message passing baseline on this task.

## What This Does Not Prove

```text
does not prove consciousness
does not prove full VRAXION
does not prove language grounding
does not prove general reasoning
does not prove Prismion is uniquely required
does not prove learned PrismionCell training is solved
```

## Next Step

The next useful experiment should not repeat fixed phase-lock. It should target one of two questions:

```text
1. Learning problem:
   Can a richer learned phase cell recover complex multiplication from final labels?

2. Primitive problem:
   Given complex multiplication is sufficient, does embedding that primitive into a broader stable-loop architecture improve tasks that need composition, cancellation, and memory?
```

Recommended next run:

```text
LEARNED_PHASE_CELL_001
```

with a trainable cell that has enough degrees of freedom to represent complex multiplication, and explicit supervision-free evaluation against the fixed complex-multiply operator.
