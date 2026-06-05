# E6 Branch-Order Invariance Neural Retest Result

## Decision

```text
decision = e6_branch_order_invariance_training_succeeds
winner = e4_top_down_reference
next = E7_NEURAL_INVARIANCE_STRESS_AND_MUTATION_OPERATOR
```

E6 confirms that the E5 branch-order shortcut is fixable for the tiny neural baselines when branch order is randomized during training. The fixed-order neural controls still fail branch-order shuffle, so E5's artifact finding remains valid for the original fixed-order setup.

## Run

- Run root: `target/pilot_wave/e6_branch_order_invariance_neural_retest`
- Seeds: `77001,77002,77003,77004,77005`
- Splits per seed: train `800`, validation `300`, heldout `300`, OOD `300`, counterfactual `300`, adversarial `300`
- Gradient epochs: `120`
- Top-down mutation search: population `24`, generations `80`
- Execution mode: parallel
- Checker: `failure_count = 0`
- Deterministic replay: passed, hash match on required decision artifacts

## Core Metrics

| system | normal heldout | branch heldout | branch OOD | branch counterfactual | branch adversarial | clean invariant |
|---|---:|---:|---:|---:|---:|---|
| `e4_top_down_reference` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | yes |
| `mlp_fixed_order_gradient` | 1.000000 | 0.212367 | 0.212480 | 0.216300 | 0.199720 | no |
| `mlp_random_order_gradient` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.999693 | yes |
| `recurrent_fixed_order_gradient` | 1.000000 | 0.463273 | 0.479873 | 0.478560 | 0.455947 | no |
| `recurrent_random_order_gradient` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | yes |
| `choicewise_shared_random_order_gradient` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | yes |
| `random_classifier` | 0.052140 | 0.055820 | 0.050120 | 0.048967 | 0.056800 | no |

## Interpretation

Random-order training succeeded for both ordinary tiny neural systems:

- `mlp_random_order_gradient`
- `recurrent_random_order_gradient`

The order-equivariant choicewise scorer also passed cleanly:

- `choicewise_shared_random_order_gradient`

The fixed-order negative controls reproduced the E5 failure mode:

- `mlp_fixed_order_gradient`
- `recurrent_fixed_order_gradient`

The non-neural top-down reference also remained clean and parameter-efficient:

- `e4_top_down_reference`: `271` parameters
- `mlp_random_order_gradient`: `56935` parameters
- `recurrent_random_order_gradient`: `25095` parameters
- `choicewise_shared_random_order_gradient`: `36295` parameters

Top-down mutation search wrote `1920` attempts, `695` accepted, `1225` rejected, and `1225` rollbacks.

## Scientific Takeaway

E6 changes the E5 conclusion from "tiny neural models are shortcuting" to a narrower statement:

```text
Fixed-order neural training shortcuts on this proxy.
Branch-order randomized neural training can learn clean abstraction-routing invariance.
The non-neural top-down router remains a clean, much smaller reference.
```

This does not prove that neural nets are necessary. It shows they are viable under the right invariance training/control setup.

## Next

Run `E7_NEURAL_INVARIANCE_STRESS_AND_MUTATION_OPERATOR`: keep the successful random-order neural setup, stress it with harder symbolic routing variants, and test whether a mutation-only neural operator can approach the gradient-trained invariant behavior.
