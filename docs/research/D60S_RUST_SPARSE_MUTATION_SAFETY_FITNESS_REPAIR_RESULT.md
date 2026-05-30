# D60S Rust Sparse Mutation Safety Fitness Repair Result

## Verdict

```text
decision = gated_policy_required_for_no_forgetting
verdict = D60S_GATED_POLICY_REQUIRED_FOR_NO_FORGETTING
next = D61_GATED_RUST_SPARSE_MUTATION_SCALE_CONFIRM
best_arm = DUAL_POLICY_GATED_CONTROLLER
scale_mode = scale-lite
```

## Run

```powershell
python scripts/probes/run_d60s_rust_sparse_mutation_safety_fitness_repair.py --out target/pilot_wave/d60s_rust_sparse_mutation_safety_fitness_repair/smoke --seeds 11501,11502,11503,11504,11505 --train-rows-per-seed 800 --test-rows-per-seed 800 --ood-rows-per-seed 800 --generations 120 --population 64 --workers auto --cpu-target 50-75 --heartbeat-sec 20 --heartbeat-generations 8 --mutation-train-packs 384 --mutation-validation-packs 384 --scale-mode scale-lite
python scripts/probes/run_d60s_rust_sparse_mutation_safety_fitness_repair_check.py --check-only --out target/pilot_wave/d60s_rust_sparse_mutation_safety_fitness_repair/smoke
```

## Key Result

D60S repaired the D60 no-forgetting failure only through a non-truth context gate. The best single-policy mutation kept the hard-task gain but still regressed below the saturated replay safety floor.

```text
D59 saturated reference exact = 0.999400
saturated stability floor = 0.997400
D58 hard replay exact = 0.605050
```

Best D60S arm:

```text
arm = DUAL_POLICY_GATED_CONTROLLER
saturated exact = 0.999300
hard exact = 0.994750
mixed exact = 0.997000
hard gain vs D58 = +0.389700
saturated regression vs D59 = -0.000100
false_confidence = 0.000000
indistinguishable abstain = 1.000000
fallback_rows = 0
```

## Arm Table

| arm | saturated exact | hard exact | mixed exact | hard gain vs D58 | saturated regression |
| --- | ---: | ---: | ---: | ---: | ---: |
| DUAL_POLICY_GATED_CONTROLLER | 0.999300 | 0.994750 | 0.997000 | 0.389700 | -0.000100 |
| CONTEXT_GATED_POLICY_ENSEMBLE | 0.999300 | 0.994750 | 0.997000 | 0.389700 | -0.000100 |
| SINGLE_POLICY_MULTI_ENV_FITNESS | 0.994750 | 0.994750 | 0.994750 | 0.389700 | -0.004650 |
| LEXICOGRAPHIC_SAFETY_FIRST_FITNESS | 0.983700 | 0.983700 | 0.983700 | 0.378650 | -0.015700 |
| PARETO_MULTI_ENV_MUTATION | 0.983700 | 0.983700 | 0.983700 | 0.378650 | -0.015700 |
| STABILITY_REGULARIZED_MUTATION | 0.983700 | 0.983700 | 0.983700 | 0.378650 | -0.015700 |

## Interpretation

The D60 hard controller is useful under support-budget-cap-8, but applying that same policy everywhere forgets the saturated D58/D59 path. A single mutated controller did not satisfy the no-forgetting gate:

```text
SINGLE_POLICY_MULTI_ENV_FITNESS:
  saturated exact = 0.994750
  floor = 0.997400
  result = fails saturated no-forgetting
```

The successful repair is a gated controller:

```text
saturated/full-support context -> D59 reference controller
support-budget-cap-8 context   -> D60 hard controller
```

The gate is not allowed to use truth labels. It uses evaluation context only, so this result points to D61 gated scale confirm rather than a single-policy Rust mutation scale confirm.

## Validation

```powershell
python -m py_compile scripts/probes/run_d60s_rust_sparse_mutation_safety_fitness_repair.py
python -m py_compile scripts/probes/run_d60s_rust_sparse_mutation_safety_fitness_repair_check.py
python scripts/probes/run_d60s_rust_sparse_mutation_safety_fitness_repair_check.py --check-only --out target/pilot_wave/d60s_rust_sparse_mutation_safety_fitness_repair/smoke
git diff --check
```

Checker output:

```text
status = ok
decision = gated_policy_required_for_no_forgetting
best_arm = DUAL_POLICY_GATED_CONTROLLER
best_saturated_exact = 0.999300
best_hard_exact = 0.994750
best_mixed_exact = 0.997000
hard_gain_vs_D58 = 0.389700
false_confidence = 0.000000
abstain_floor = 1.000000
```

## Boundary

```text
D60S only tests safety/no-forgetting fitness repair for mutation of a canonical Rust sparse ECF action controller in controlled symbolic joint formula discovery.
It does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
```
