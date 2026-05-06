# Authority Gradient Search Quick Test

## Background

The hand-seeded frame-gated authority graph can solve the small latent-refraction, multi-aspect, and temporal-order tasks without neural layers or backprop. The previous minimality run showed that random weak priors do not rediscover the mechanism, while damaged hand graphs can partially recover.

## Why This Run Matters

This run asks whether a compact developmental grammar can generate and evolve the route/gate/recurrent/readout structure, or whether the mechanism still depends on manual wiring.

## Run Configuration

```json
{
  "seeds": 2,
  "steps": 5,
  "search_train_samples": 96,
  "validation_samples": 96,
  "final_test_samples": 256,
  "generations": 150,
  "population_size": 16,
  "mutation_scale": 0.18,
  "checkpoint_every": 25,
  "max_runtime_hours": 11.5,
  "decay": 0.35,
  "fitness_mode": "ab_compare",
  "fitness_modes_run": [
    "coarse",
    "authority_shaped"
  ],
  "arms_requested": [
    "random_graph",
    "route_grammar_graph",
    "route_gate_grammar_graph",
    "route_gate_recurrence_grammar",
    "route_gate_hub_grammar",
    "damaged_hand_seeded_50",
    "hand_seeded"
  ],
  "arms_completed": [
    "random_graph",
    "route_grammar_graph",
    "route_gate_grammar_graph",
    "route_gate_recurrence_grammar",
    "route_gate_hub_grammar",
    "damaged_hand_seeded_50",
    "hand_seeded"
  ],
  "smoke": false,
  "completed": true,
  "started_unix": 1778090686.411868
}
```

## Graph Grammar Arms

- `random_graph`: pure random signed graph baseline.
- `route_grammar_graph`: route groups and readout ports, but no frame gates or guaranteed recurrence.
- `route_gate_grammar_graph`: route grammar plus frame gates.
- `route_gate_recurrence_grammar`: route/gate grammar plus recurrent route and temporal memory edges.
- `route_gate_hub_grammar`: recurrence grammar plus shared hubs.
- `damaged_hand_seeded_50`: hand graph with 50% edges removed.
- `hand_seeded`: upper bound, evaluated but not evolved.

Grammar arms use random token-route wiring without task-label rule lookup. Exact task-solution wiring is allowed only in the hand-seeded upper bound and damaged-hand recovery baseline.

## Fitness Definition

```text
coarse fitness =
  1.0 * overall_accuracy
  + 0.5 * authority_refraction_score
  + 0.3 * temporal_order_accuracy
  + 0.2 * max(wrong_frame_drop, 0)
  - 0.05 * inactive_influence
  - 0.02 * edge_count_penalty

authority_shaped fitness =
  1.0 * overall_accuracy
  + 0.5 * authority_refraction_score
  + 0.3 * wrong_frame_drop
  + 0.25 * recurrence_drop
  + 0.25 * route_specialization
  + 0.3 * temporal_order_accuracy
  - 0.05 * inactive_influence
  - 0.02 * edge_count_penalty
```

Evolution uses search-train fitness, but best graph selection uses validation fitness. Final verdicts use final-test metrics.

## Main Final-Test Results

| Fitness | Arm | Success | Strong Success | Accuracy | Latent | Multi | Temporal | Authority | Wrong Frame | Recurrence Drop | Route Spec | Inactive | Edges |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `coarse` | `random_graph` | `0.000000` | `0.000000` | `0.531901` | `0.509766` | `0.523438` | `0.562500` | `-0.009163` | `0.058594` | `0.000000` | `-0.176676` | `0.067490` | `92.000000` |
| `coarse` | `route_grammar_graph` | `0.000000` | `0.000000` | `0.592448` | `0.644531` | `0.632812` | `0.500000` | `0.007786` | `0.000000` | `0.000000` | `-0.115438` | `0.019759` | `22.500000` |
| `coarse` | `route_gate_grammar_graph` | `0.000000` | `0.000000` | `0.592448` | `0.644531` | `0.632812` | `0.500000` | `-0.006026` | `0.000000` | `0.000000` | `-0.069671` | `0.012467` | `23.500000` |
| `coarse` | `route_gate_recurrence_grammar` | `0.000000` | `0.000000` | `0.613281` | `0.644531` | `0.632812` | `0.562500` | `-0.005282` | `0.000000` | `0.062500` | `-0.064666` | `0.012474` | `33.500000` |
| `coarse` | `route_gate_hub_grammar` | `0.000000` | `0.000000` | `0.621745` | `0.658203` | `0.644531` | `0.562500` | `-0.020220` | `0.062500` | `0.062500` | `-0.099506` | `0.049246` | `57.000000` |
| `coarse` | `damaged_hand_seeded_50` | `0.000000` | `0.000000` | `0.790365` | `0.845703` | `0.837891` | `0.687500` | `0.198225` | `0.238281` | `0.187500` | `-0.052551` | `0.038672` | `46.000000` |
| `coarse` | `hand_seeded` | `1.000000` | `1.000000` | `0.991536` | `0.982422` | `0.992188` | `1.000000` | `0.365708` | `0.351562` | `0.500000` | `0.040586` | `0.039594` | `91.000000` |
| `authority_shaped` | `random_graph` | `0.000000` | `0.000000` | `0.574219` | `0.583984` | `0.576172` | `0.562500` | `-0.016384` | `0.148438` | `0.000000` | `-0.115286` | `0.077283` | `96.000000` |
| `authority_shaped` | `route_grammar_graph` | `0.000000` | `0.000000` | `0.592448` | `0.644531` | `0.632812` | `0.500000` | `-0.003033` | `0.000000` | `0.000000` | `-0.031355` | `0.005987` | `21.500000` |
| `authority_shaped` | `route_gate_grammar_graph` | `0.000000` | `0.000000` | `0.592448` | `0.644531` | `0.632812` | `0.500000` | `-0.011188` | `0.000000` | `0.000000` | `-0.071973` | `0.015228` | `24.000000` |
| `authority_shaped` | `route_gate_recurrence_grammar` | `0.000000` | `0.000000` | `0.644531` | `0.660156` | `0.648438` | `0.625000` | `-0.012713` | `0.031250` | `0.125000` | `-0.058352` | `0.025705` | `35.000000` |
| `authority_shaped` | `route_gate_hub_grammar` | `0.000000` | `0.000000` | `0.606771` | `0.626953` | `0.630859` | `0.562500` | `-0.039040` | `0.089844` | `0.062500` | `-0.132827` | `0.091830` | `60.500000` |
| `authority_shaped` | `damaged_hand_seeded_50` | `0.000000` | `0.000000` | `0.790365` | `0.845703` | `0.837891` | `0.687500` | `0.198961` | `0.238281` | `0.187500` | `-0.045666` | `0.038223` | `44.500000` |
| `authority_shaped` | `hand_seeded` | `1.000000` | `1.000000` | `0.991536` | `0.982422` | `0.992188` | `1.000000` | `0.365708` | `0.351562` | `0.500000` | `0.040586` | `0.039594` | `91.000000` |

## Train / Validation / Final-Test Fitness

| Fitness | Arm | Train Fitness | Validation Fitness | Final-Test Fitness | Generations Completed | Generations To Threshold |
|---|---|---:|---:|---:|---:|---:|
| `coarse` | `random_graph` | `0.689049` | `0.688086` | `0.684194` | `150.000000` | `null` |
| `coarse` | `route_grammar_graph` | `0.740927` | `0.751826` | `0.740408` | `150.000000` | `null` |
| `coarse` | `route_gate_grammar_graph` | `0.734361` | `0.746259` | `0.733647` | `150.000000` | `null` |
| `coarse` | `route_gate_recurrence_grammar` | `0.771298` | `0.782915` | `0.771404` | `150.000000` | `null` |
| `coarse` | `route_gate_hub_grammar` | `0.781923` | `0.788239` | `0.777895` | `150.000000` | `null` |
| `coarse` | `damaged_hand_seeded_50` | `1.142296` | `1.141216` | `1.131340` | `150.000000` | `null` |
| `coarse` | `hand_seeded` | `1.530199` | `1.483795` | `1.522723` | `0.000000` | `0.000000` |
| `authority_shaped` | `random_graph` | `0.726829` | `0.718720` | `0.725524` | `150.000000` | `null` |
| `authority_shaped` | `route_grammar_graph` | `0.728032` | `0.738419` | `0.728068` | `150.000000` | `null` |
| `authority_shaped` | `route_gate_grammar_graph` | `0.713139` | `0.724937` | `0.712824` | `150.000000` | `null` |
| `authority_shaped` | `route_gate_recurrence_grammar` | `0.839175` | `0.844517` | `0.842734` | `150.000000` | `null` |
| `authority_shaped` | `route_gate_hub_grammar` | `0.761413` | `0.736449` | `0.747484` | `150.000000` | `null` |
| `authority_shaped` | `damaged_hand_seeded_50` | `1.207120` | `1.188021` | `1.191346` | `150.000000` | `null` |
| `authority_shaped` | `hand_seeded` | `1.705718` | `1.643354` | `1.693026` | `0.000000` | `0.000000` |

## Leakage Audit

| Fitness | Arm | Grammar Runs Pass | Direct Token->Readout Edges |
|---|---|---:|---:|
| `coarse` | `random_graph` | `True` | `0.000000` |
| `coarse` | `route_grammar_graph` | `True` | `0.000000` |
| `coarse` | `route_gate_grammar_graph` | `True` | `0.000000` |
| `coarse` | `route_gate_recurrence_grammar` | `True` | `0.000000` |
| `coarse` | `route_gate_hub_grammar` | `True` | `0.000000` |
| `coarse` | `damaged_hand_seeded_50` | `True` | `0.000000` |
| `coarse` | `hand_seeded` | `True` | `0.000000` |
| `authority_shaped` | `random_graph` | `True` | `0.000000` |
| `authority_shaped` | `route_grammar_graph` | `True` | `0.000000` |
| `authority_shaped` | `route_gate_grammar_graph` | `True` | `0.000000` |
| `authority_shaped` | `route_gate_recurrence_grammar` | `True` | `0.000000` |
| `authority_shaped` | `route_gate_hub_grammar` | `True` | `0.000000` |
| `authority_shaped` | `damaged_hand_seeded_50` | `True` | `0.000000` |
| `authority_shaped` | `hand_seeded` | `True` | `0.000000` |

## Best Graph Examples

- `coarse` / `random_graph` seed `0`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/coarse/random_graph_seed0.json`
- `coarse` / `route_grammar_graph` seed `0`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/coarse/route_grammar_graph_seed0.json`
- `coarse` / `route_gate_grammar_graph` seed `0`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/coarse/route_gate_grammar_graph_seed0.json`
- `coarse` / `route_gate_recurrence_grammar` seed `0`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/coarse/route_gate_recurrence_grammar_seed0.json`
- `coarse` / `route_gate_hub_grammar` seed `0`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/coarse/route_gate_hub_grammar_seed0.json`
- `coarse` / `damaged_hand_seeded_50` seed `0`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/coarse/damaged_hand_seeded_50_seed0.json`
- `coarse` / `hand_seeded` seed `0`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/coarse/hand_seeded_seed0.json`
- `authority_shaped` / `random_graph` seed `0`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/authority_shaped/random_graph_seed0.json`
- `authority_shaped` / `route_grammar_graph` seed `0`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/authority_shaped/route_grammar_graph_seed0.json`
- `authority_shaped` / `route_gate_grammar_graph` seed `0`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/authority_shaped/route_gate_grammar_graph_seed0.json`
- `authority_shaped` / `route_gate_recurrence_grammar` seed `0`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/authority_shaped/route_gate_recurrence_grammar_seed0.json`
- `authority_shaped` / `route_gate_hub_grammar` seed `0`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/authority_shaped/route_gate_hub_grammar_seed0.json`
- `authority_shaped` / `damaged_hand_seeded_50` seed `0`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/authority_shaped/damaged_hand_seeded_50_seed0.json`
- `authority_shaped` / `hand_seeded` seed `0`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/authority_shaped/hand_seeded_seed0.json`
- `coarse` / `random_graph` seed `1`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/coarse/random_graph_seed1.json`
- `coarse` / `route_grammar_graph` seed `1`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/coarse/route_grammar_graph_seed1.json`
- `coarse` / `route_gate_grammar_graph` seed `1`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/coarse/route_gate_grammar_graph_seed1.json`
- `coarse` / `route_gate_recurrence_grammar` seed `1`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/coarse/route_gate_recurrence_grammar_seed1.json`
- `coarse` / `route_gate_hub_grammar` seed `1`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/coarse/route_gate_hub_grammar_seed1.json`
- `coarse` / `damaged_hand_seeded_50` seed `1`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/coarse/damaged_hand_seeded_50_seed1.json`
- `coarse` / `hand_seeded` seed `1`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/coarse/hand_seeded_seed1.json`
- `authority_shaped` / `random_graph` seed `1`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/authority_shaped/random_graph_seed1.json`
- `authority_shaped` / `route_grammar_graph` seed `1`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/authority_shaped/route_grammar_graph_seed1.json`
- `authority_shaped` / `route_gate_grammar_graph` seed `1`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/authority_shaped/route_gate_grammar_graph_seed1.json`
- `authority_shaped` / `route_gate_recurrence_grammar` seed `1`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/authority_shaped/route_gate_recurrence_grammar_seed1.json`
- `authority_shaped` / `route_gate_hub_grammar` seed `1`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/authority_shaped/route_gate_hub_grammar_seed1.json`
- `authority_shaped` / `damaged_hand_seeded_50` seed `1`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/authority_shaped/damaged_hand_seeded_50_seed1.json`
- `authority_shaped` / `hand_seeded` seed `1`: `target/context-cancellation-probe/authority-gradient-search-quick/best_graphs/authority_shaped/hand_seeded_seed1.json`

## Failure Cases

- If grammar arms remain near random on final-test, the current mechanism still requires too much manual structure.
- If train fitness improves but validation/final-test does not, the search is overfitting the small train split.
- If accuracy improves without authority/refraction or wrong-frame drop, the graph is solving labels without the target authority mechanism.

## Minimal Surviving Prior

Read from final-test comparisons only: random vs route grammar, route vs route+gate, route+gate vs route+gate+recurrence, and recurrence vs recurrence+hub.

## Verdict

```json
{
  "shaped_fitness_improves_search": false,
  "search_space_problem_supported": false,
  "grammar_prior_still_too_weak": true,
  "damaged_hand_repair_improves": false,
  "random_search_still_insufficient": true,
  "final_verdict_uses_final_test": true
}
```

## Runtime Notes

- runtime seconds: `3888.215335`
- interrupted by wall clock: `False`
- completed records: `28`

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, production validation, or natural-language understanding.
