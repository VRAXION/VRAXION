# E7A Matrix Medium Primitive Scan Result

## Decision

```text
decision = e7a_observational_primitive_scan_complete
best_heldout_macro_family_system = c19_c_rho_mut_matrix_depth3
final_e7_verdict = intentionally deferred
checker = failure_count 0
deterministic_replay = passed
```

Run root:

```text
target/pilot_wave/e7a_matrix_medium_primitive_scan_v2
```

E7A is a primitive scan only. It does not confirm a full matrix-medium routing architecture.

## Main Primitive Findings

Linear depth collapse was confirmed:

```text
linear_matrix_depth1 collapse mismatch = 0
linear_matrix_depth3 collapse mismatch = 0
linear_matrix_depth6 collapse mismatch = 0
```

So a pure identity/linear deep stack does not buy inference expressivity here; it collapses to one effective matrix.

Heldout macro-family accuracy:

| system | heldout | OOD | counterfactual | adversarial | steps |
|---|---:|---:|---:|---:|---:|
| `linear_matrix_depth1` | 0.343750 | 0.375000 | 0.369792 | 0.401042 | 1.000 |
| `linear_matrix_depth3` | 0.317708 | 0.317708 | 0.300781 | 0.274740 | 3.000 |
| `linear_matrix_depth6` | 0.337240 | 0.342448 | 0.324219 | 0.388021 | 6.000 |
| `tanh_matrix_depth3` | 0.316406 | 0.316406 | 0.303385 | 0.354167 | 3.000 |
| `relu_matrix_depth3` | 0.325521 | 0.311198 | 0.307292 | 0.329427 | 3.000 |
| `c19_fixed_matrix_depth3` | 0.348958 | 0.350260 | 0.350260 | 0.388021 | 3.000 |
| `c19_rho0_matrix_depth3` | 0.342448 | 0.299479 | 0.304688 | 0.338542 | 3.000 |
| `c19_c_mut_matrix_depth3` | 0.299479 | 0.292969 | 0.326823 | 0.343750 | 3.000 |
| `c19_rho_mut_matrix_depth3` | 0.335938 | 0.309896 | 0.312500 | 0.345052 | 3.000 |
| `c19_c_rho_mut_matrix_depth3` | 0.348958 | 0.338542 | 0.324219 | 0.375000 | 3.000 |

## C19 Parameter Modes

Best C19 matrix-mode system:

```text
c19_c_rho_mut_matrix_depth3
```

But the margin over fixed C19 was effectively a tie on heldout:

```text
c19_fixed_matrix_depth3      heldout = 0.348958333333
c19_c_rho_mut_matrix_depth3  heldout = 0.348958333334
```

The toggles did not cleanly beat fixed C19 by the audit threshold:

```text
rho_off_beats_fixed_on_heldout = false
c_mut_beats_fixed_on_heldout = false
rho_mut_beats_fixed_on_heldout = false
c_rho_mut_beats_fixed_on_heldout = false
```

So E7A does not prove that mutable `c/rho` is useful. It only proves that those modes are now explicitly tested and deterministic.

## Recurrent / Halting

Recurrent and halting variants did not add clean value in this primitive setup:

| system | heldout | mean steps | halt efficiency |
|---|---:|---:|---:|
| `c19_fixed_recurrent_fixed6` | 0.320313 | 6.000000 | 0.000000 |
| `c19_c_rho_mut_recurrent_fixed6` | 0.270833 | 6.000000 | 0.000000 |
| `c19_fixed_recurrent_halting6` | 0.287760 | 3.980469 | 0.136458 |
| `c19_c_rho_mut_recurrent_halting6` | 0.268229 | 4.488281 | 0.070052 |
| `c19_c_rho_mut_recurrent_halting_restart6` | 0.276042 | 2.869792 | 0.168490 |

Halting/restart reduced step count, but not without accuracy loss:

```text
halting_reduced_steps_without_accuracy_loss = false
restart_reduced_steps_without_accuracy_loss = false
```

## Interpretation

Primitive-layer read:

```text
1. Linear stack collapse behaves exactly as expected.
2. Nonlinear matrix stacks show small signal, but not a strong solve.
3. Fixed C19 and c+rho-mutable C19 are close; no clean mutable-C19 win yet.
4. Recurrent halting/restart is active, but currently trades accuracy for fewer steps.
5. This is not enough to justify a full matrix-medium claim.
```

Recommended next step:

```text
E7B_MATRIX_MEDIUM_ROUTING_PROXY_IF_PRIMITIVES_ARE_INTERESTING
```

But E7B should not scale up blindly. It should first improve the primitive search operator or task shaping, because E7A accuracies are low and halting is not yet clean.
