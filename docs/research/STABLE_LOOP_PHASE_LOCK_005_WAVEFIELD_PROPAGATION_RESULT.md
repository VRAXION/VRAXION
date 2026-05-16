# STABLE_LOOP_PHASE_LOCK_005_WAVEFIELD_PROPAGATION Result

Status: implemented, sanity run complete, bounded 3-seed smoke complete.

## Verdict

```text
WAVEFIELD_CREDIT_SMOOTHER_THAN_PARTICLE
INTERFERENCE_HELPFUL
TARGET_PROBABILITY_GRADIENT_PRESENT
```

This is a directional positive for the next modeling idea:

```text
complex amplitude-field propagation gives a much stronger target probability signal
than particle/frontier propagation on the same spatial phase-lock cases.
```

It is not a learned-cell result yet.

## Runs

Sanity:

```powershell
python scripts\probes\run_stable_loop_phase_lock_wavefield_probe.py `
  --out target\pilot_wave\stable_loop_phase_lock_005_wavefield_propagation\sanity `
  --seeds 2026 `
  --eval-examples 256 `
  --width 20 `
  --steps 40 `
  --jobs 6 `
  --device cpu `
  --heartbeat-sec 15
```

Result: 6/6 jobs, 36 probability-curve rows, 6.7 seconds.

Bounded smoke:

```powershell
python scripts\probes\run_stable_loop_phase_lock_wavefield_probe.py `
  --out target\pilot_wave\stable_loop_phase_lock_005_wavefield_propagation\smoke `
  --seeds 2026,2027,2028 `
  --eval-examples 512 `
  --width 24 `
  --steps 48 `
  --jobs 6 `
  --device cpu `
  --heartbeat-sec 15
```

Result: 18/18 jobs, 108 probability-curve rows, 38.0 seconds.

## Bounded Smoke Summary

| Arm | Argmax accuracy | Correct phase probability | NLL | Smooth gain | Monotonicity |
|---|---:|---:|---:|---:|---:|
| PARTICLE_FRONTIER_004_BASELINE | 24.3% | 25.1% | 3.333 | 0.006 | 0.533 |
| COMPLEX_WAVEFIELD_PROPAGATION | 100.0% | 93.0% | 0.084 | 0.684 | 0.800 |
| WAVEFIELD_WITH_INTERFERENCE | 100.0% | 93.0% | 0.084 | 0.684 | 0.800 |
| WAVEFIELD_NO_INTERFERENCE_ABLATION | 26.0% | 26.1% | 9.062 | 0.035 | 0.800 |
| WAVEFIELD_WITH_LOCAL_PHASE_LOSS | 100.0% | 93.0% | 0.084 | 0.684 | 0.800 |
| WAVEFIELD_TARGET_ONLY_LOSS | 100.0% | 93.0% | 0.084 | 0.684 | 0.800 |

The important comparison is not just argmax:

```text
particle/frontier correct target probability: 25.1%
wavefield/interference correct target probability: 93.0%
wavefield advantage: +67.9pp
```

The no-interference ablation stays near chance:

```text
WAVEFIELD_NO_INTERFERENCE_ABLATION argmax: 26.0%
WAVEFIELD_WITH_INTERFERENCE argmax:       100.0%
```

So the useful mechanism in this probe is not merely "spread signal everywhere"; it is coherent complex accumulation/interference.

## Interpretation

The 004 result said:

```text
raw local mutation did not discover spatial phase transport from final labels.
```

This 005 result suggests a plausible reason and next direction:

```text
particle/frontier credit is too sparse,
while complex wavefield propagation exposes a smoother target probability signal.
```

The alpha curve is still not a full optimization proof. It shows that when the local gate-rotation rule approaches the correct complex operation, the target probability becomes highly informative. It does not yet show that mutation-selection or gradient descent can reliably learn that operation from scratch.

## Claim Boundary

This probe tests whether complex amplitude-field propagation gives smoother credit assignment than particle/frontier propagation. It does not prove consciousness, full VRAXION, language grounding, or Prismion uniqueness.

It also does not prove:

```text
a learned cell can acquire this wavefield rule
mutation-selection can find the rule unaided
the phase-lock 004 blocker is fully solved
```

The next useful probe is a learned or mutation-selected wavefield cell, using `correct_phase_probability_at_target` / NLL as the fitness signal instead of only final argmax success.
