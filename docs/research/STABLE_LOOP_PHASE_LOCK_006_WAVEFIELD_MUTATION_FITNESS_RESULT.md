# STABLE_LOOP_PHASE_LOCK_006_WAVEFIELD_MUTATION_FITNESS Result

Status: implemented, sanity run complete, bounded 3-seed smoke complete.

## Verdict

```text
WAVEFIELD_MUTATION_STILL_NOT_SOLVED
```

Rejected claims:

```text
SOFT_WAVEFIELD_FITNESS_BEATS_HARD_ARGMAX
MUTATION_APPROACHES_ORACLE_COMPLEX_CELL
```

The 005 wavefield reference made the target probability signal much smoother, but this 006 microprobe shows that a naive local weight-mutation hillclimb still does not acquire the complex phase-transport cell.

## Runs

Sanity:

```powershell
python scripts\probes\run_stable_loop_phase_lock_wavefield_mutation_fitness_probe.py `
  --out target\pilot_wave\stable_loop_phase_lock_006_wavefield_mutation_fitness\sanity `
  --seeds 2026 `
  --train-examples 48 `
  --eval-examples 128 `
  --width 18 `
  --steps 36 `
  --search-steps 60 `
  --checkpoint-interval 10 `
  --jobs 5 `
  --device cpu `
  --heartbeat-sec 15
```

Result: 5/5 jobs, 46 accepted-candidate rows, 16.9 seconds.

Bounded smoke:

```powershell
python scripts\probes\run_stable_loop_phase_lock_wavefield_mutation_fitness_probe.py `
  --out target\pilot_wave\stable_loop_phase_lock_006_wavefield_mutation_fitness\smoke `
  --seeds 2026,2027,2028 `
  --train-examples 64 `
  --eval-examples 192 `
  --width 20 `
  --steps 40 `
  --search-steps 120 `
  --checkpoint-interval 20 `
  --jobs 5 `
  --device cpu `
  --heartbeat-sec 20
```

Result: 15/15 jobs, 204 accepted-candidate rows, 92.5 seconds.

## Bounded Smoke Summary

| Arm | Argmax accuracy | Correct probability | NLL | Distance to oracle weights | Acceptance |
|---|---:|---:|---:|---:|---:|
| IDENTITY_BASELINE | 27.1% | 27.1% | 10.576 | 2.449 | 0.0% |
| ORACLE_COMPLEX_WEIGHTS | 100.0% | 92.1% | 0.098 | 0.000 | 0.0% |
| HARD_ARGMAX_MUTATION | 26.9% | 26.9% | 10.040 | 2.498 | 25.0% |
| SOFT_PROB_MUTATION | 24.1% | 24.4% | 10.321 | 2.585 | 16.9% |
| SOFT_NLL_MUTATION | 26.2% | 26.0% | 10.118 | 2.674 | 14.7% |

The cell representation can express the correct primitive:

```text
ORACLE_COMPLEX_WEIGHTS:
  argmax = 100.0%
  correct target probability = 92.1%
```

But none of the mutation-fitness variants moved toward it:

```text
hard argmax mutation: 26.9%
soft probability mutation: 24.1%
soft NLL mutation: 26.2%
identity baseline: 27.1%
```

The accepted mutations mostly do not generalize to heldout cases and do not reduce distance to the oracle complex-multiply weights.

## Interpretation

This result narrows the blocker:

```text
005: wavefield/interference reference gives a strong target probability signal.
006: naive scalar weight mutation still does not find the representable local complex cell.
```

So the next blocker is not whether the wavefield formulation can solve the task. It can. The blocker is the search/training mechanism used to acquire the local rule.

Likely next directions:

```text
1. gradient-trained local wavefield cell
2. coordinate-wise / CMA-ES style mutation instead of one-weight hillclimb
3. primitive pretraining before spatial transfer
4. curriculum from single-step complex multiply to spatial wavefield propagation
```

## Claim Boundary

This probe only tests whether wavefield target-probability / NLL fitness improves a tiny local mutation search for a representable complex phase-transport cell.

It does not prove consciousness, full VRAXION, language grounding, or Prismion uniqueness. It also does not disprove wavefield propagation, because the oracle cell remains strong.
