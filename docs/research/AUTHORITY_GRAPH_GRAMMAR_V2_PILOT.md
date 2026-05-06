# Authority Graph Grammar v2 Pilot

## Goal

Test whether the wiring rules distilled from the hand-seeded and damaged-success graphs improve the failed Grammar v1 prior.

This is a quick pilot, not a long developmental search. It uses the same toy tasks and does not add new semantic concepts, neural layers, or backprop.

## Readout Policy

The pilot makes the previous readout caveat explicit: static outputs use `route_state` readout. The route state is the formal authority readout port for static refraction tasks. `readout_positive` and `readout_negative` nodes remain present for future explicit-edge readout work, but this pilot does not silently rely on them.

Temporal order tasks likewise read from `temporal_route` state.

## Grammar v2 Scaffold

- one route group per frame with recurrence
- guaranteed group-level token-to-route candidate coverage
- shared token->hub and hub->route bridge
- frame gates applied early to routes
- full route-level suppressor matrix
- subject/verb/object temporal role channel
- route-state authority readout policy
- redundant weak candidate paths for mutation

Grammar v2 does not wire exact task solutions like `dog+bite->danger` by name. It gives broad coverage paths and leaves signs/gains to mutation.

## Run Configuration

```json
{
  "seeds": 3,
  "steps": 5,
  "search_train_samples": 96,
  "validation_samples": 96,
  "final_test_samples": 256,
  "generations": 200,
  "population_size": 16,
  "mutation_scale": 0.18,
  "checkpoint_every": 25,
  "max_runtime_hours": 3.0,
  "decay": 0.35,
  "fitness_mode": "authority_shaped",
  "readout_policy": "route_state",
  "arms_requested": [
    "route_gate_hub_grammar",
    "grammar_v2_graph",
    "damaged_hand_seeded_50",
    "hand_seeded"
  ],
  "arms_completed": [
    "damaged_hand_seeded_50",
    "grammar_v2_graph",
    "hand_seeded",
    "route_gate_hub_grammar"
  ],
  "smoke": false,
  "completed": true,
  "started_unix": 1778096895.1914492
}
```

## Final-Test Results

| Arm | Success | Strong Success | Accuracy | Latent | Multi | Temporal | Authority | Wrong Frame | Recurrence Drop | Route Spec | Inactive | Edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `damaged_hand_seeded_50` | `0.000000` | `0.000000` | `0.828559` | `0.869792` | `0.865885` | `0.750000` | `0.226331` | `0.260417` | `0.250000` | `-0.030925` | `0.035815` | `45.666667` |
| `grammar_v2_graph` | `0.000000` | `0.000000` | `0.647569` | `0.692708` | `0.666667` | `0.583333` | `0.029724` | `0.101562` | `0.083333` | `-0.221100` | `0.124332` | `115.333333` |
| `hand_seeded` | `1.000000` | `1.000000` | `0.988715` | `0.979167` | `0.986979` | `1.000000` | `0.360225` | `0.354167` | `0.500000` | `0.039404` | `0.040185` | `91.000000` |
| `route_gate_hub_grammar` | `0.000000` | `0.000000` | `0.639323` | `0.671875` | `0.662760` | `0.583333` | `-0.043059` | `0.101562` | `0.083333` | `-0.136114` | `0.096309` | `55.666667` |

## Fitness Generalization

| Arm | Train Fitness | Validation Fitness | Final-Test Fitness | Generations |
|---|---:|---:|---:|---:|
| `damaged_hand_seeded_50` | `1.322762` | `1.298345` | `1.287791` | `200.000000` |
| `grammar_v2_graph` | `0.801935` | `0.814637` | `0.801894` | `200.000000` |
| `hand_seeded` | `1.711550` | `1.637505` | `1.687920` | `0.000000` |
| `route_gate_hub_grammar` | `0.803896` | `0.789611` | `0.793017` | `200.000000` |

## Leakage Audit

| Arm | Grammar Audit Pass | Direct Token->Readout Edges |
|---|---:|---:|
| `damaged_hand_seeded_50` | `True` | `0.000000` |
| `grammar_v2_graph` | `True` | `0.000000` |
| `hand_seeded` | `True` | `0.000000` |
| `route_gate_hub_grammar` | `True` | `0.000000` |

## Verdict

```json
{
  "grammar_v2_beats_v1": true,
  "grammar_v2_improves_authority": true,
  "grammar_v2_improves_temporal_order": false,
  "grammar_v2_reaches_success": false,
  "grammar_v2_closes_gap_to_damaged_hand": false,
  "hand_seeded_still_upper_bound": true,
  "readout_policy_explicitly_documented": true,
  "final_verdict_uses_final_test": true
}
```

## Interpretation

Grammar v2 is a weak positive over Grammar v1, but not a solved developmental prior.

Compared with the previous strongest weak grammar baseline (`route_gate_hub_grammar`), Grammar v2 improves the authority/refraction signal:

- `route_gate_hub_grammar` authority: `-0.043059`
- `grammar_v2_graph` authority: `0.029724`

It also slightly improves overall accuracy:

- `route_gate_hub_grammar` accuracy: `0.639323`
- `grammar_v2_graph` accuracy: `0.647569`

But the improvement is small and does not reach success:

- temporal order remains weak: `0.583333`
- wrong-frame sensitivity remains modest: `0.101562`
- route specialization is still negative: `-0.221100`
- inactive influence is high: `0.124332`
- edge count is bloated: `115.333333`, higher than the `91`-edge hand-seeded upper bound

The damaged-hand control is still much stronger under the same quick budget:

- damaged-hand accuracy: `0.828559`
- damaged-hand authority: `0.226331`
- damaged-hand temporal: `0.750000`

So the distilled scaffold helps, but it is not enough. The likely remaining issue is not just missing coverage; Grammar v2 now has coverage, but the random signs/gains create poor route specialization and excess inactive-route leakage. The next improvement should focus on sign/gain priors and edge budget discipline, not more brute force.

## Runtime Notes

- runtime seconds: `1786.912284`
- interrupted by wall clock: `False`
- completed records: `12`

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, or production validation.
