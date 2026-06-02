# D60 Rust Sparse Mutation Learning Signal Probe Result

## Verdict

```text
decision = rust_sparse_mutation_safety_failure
verdict = D60_RUST_SPARSE_MUTATION_SAFETY_FAILURE
next = D60S_SAFETY_FITNESS_REPAIR
best_arm = COST_ONLY_MUTATION_CONTROL
scale_mode = scale-lite
```

## Run

```powershell
python scripts/probes/run_d60_rust_sparse_mutation_learning_signal_probe.py --out target/pilot_wave/d60_rust_sparse_mutation_learning_signal_probe/smoke --seeds 11401,11402,11403,11404,11405 --train-rows-per-seed 800 --test-rows-per-seed 800 --ood-rows-per-seed 800 --generations 120 --population 64 --workers auto --cpu-target 50-75 --heartbeat-sec 20 --heartbeat-generations 8 --mutation-train-packs 384 --mutation-validation-packs 384 --scale-mode scale-lite
python scripts/probes/run_d60_rust_sparse_mutation_learning_signal_probe_check.py --check-only --out target/pilot_wave/d60_rust_sparse_mutation_learning_signal_probe/smoke
```

## Key Finding

D60 confirmed a hard-track learning signal, but the learned hard-track controller regressed on the saturated replay track.

```text
Hard track, D58 replay:
  exact_joint = 0.605050
  correlated_echo = 0.011500
  adversarial_distractor = 0.013750
  support = 6.479

Hard track, best mutated:
  arm = COST_ONLY_MUTATION_CONTROL
  exact_joint = 0.995700
  correlated_echo = 0.990250
  adversarial_distractor = 0.988250
  support = 6.479
  false_confidence = 0.000000

Learning gain:
  exact_gain = +0.390650
  cost_adjusted_gain = +0.390650
  support_delta = 0.000000
```

The stability failure is real:

```text
D59 saturated reference exact = 0.999400
saturated stability gate = 0.997400
best mutated saturated exact = 0.995700
saturated false_confidence = 0.000000
fallback_rows = 0
```

So the mutation path learned a cheaper/harder policy for the support-capped task, but that policy is not safely reusable on the saturated D58/D59 distribution.

## Interpretation

```text
Rust mutation path: exercised
Hard-track learning signal: present
Safety/stability across saturated replay: failed
Best next step: repair fitness to preserve saturated safety while learning hard-track policy
```

This points to D60S rather than D61. The next repair should make the mutation fitness multi-environment: hard-task gain must be accepted only if saturated replay stays above the stability floor.

Boundary:

```text
D60 only tests learning signal for mutation and selection of a canonical Rust sparse ECF action controller on controlled symbolic joint formula discovery.
It does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
```
