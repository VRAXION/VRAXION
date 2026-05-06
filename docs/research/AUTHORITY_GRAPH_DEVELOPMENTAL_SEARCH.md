# Authority Graph Developmental Search

## Background

The hand-seeded frame-gated authority graph can solve the small latent-refraction, multi-aspect, and temporal-order tasks without neural layers or backprop. The previous minimality run showed that random weak priors do not rediscover the mechanism, while damaged hand graphs can partially recover.

## Why This Run Matters

This run asks whether a compact developmental grammar can generate and evolve the route/gate/recurrent/readout structure, or whether the mechanism still depends on manual wiring.

## Run Configuration

```json
{
  "seeds": 5,
  "steps": 5,
  "search_train_samples": 128,
  "validation_samples": 128,
  "final_test_samples": 512,
  "generations": 2000,
  "population_size": 32,
  "mutation_scale": 0.18,
  "checkpoint_every": 25,
  "max_runtime_hours": 11.5,
  "decay": 0.35,
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
  "completed": false,
  "started_unix": 1778042707.01575
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
fitness =
  1.0 * overall_accuracy
  + 0.5 * authority_refraction_score
  + 0.3 * temporal_order_accuracy
  + 0.2 * max(wrong_frame_drop, 0)
  - 0.05 * inactive_influence
  - 0.02 * edge_count_penalty
```

Evolution uses search-train fitness, but best graph selection uses validation fitness. Final verdicts use final-test metrics.

## Main Final-Test Results

| Arm | Success | Strong Success | Accuracy | Latent | Multi | Temporal | Authority | Wrong Frame Drop | Inactive Influence | Edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `random_graph` | `0.000000` | `0.000000` | `0.743880` | `0.745313` | `0.761328` | `0.725000` | `0.014102` | `0.243750` | `0.103658` | `100.000000` |
| `route_grammar_graph` | `0.000000` | `0.000000` | `0.684115` | `0.767578` | `0.759766` | `0.525000` | `0.078072` | `0.170313` | `0.036847` | `28.200000` |
| `route_gate_grammar_graph` | `0.000000` | `0.000000` | `0.698047` | `0.780078` | `0.789062` | `0.525000` | `0.089639` | `0.168750` | `0.039715` | `28.400000` |
| `route_gate_recurrence_grammar` | `0.000000` | `0.000000` | `0.725749` | `0.770996` | `0.750000` | `0.656250` | `0.055595` | `0.145508` | `0.040418` | `37.250000` |
| `route_gate_hub_grammar` | `0.000000` | `0.000000` | `0.742350` | `0.785645` | `0.785156` | `0.656250` | `0.048496` | `0.192383` | `0.064220` | `58.500000` |
| `damaged_hand_seeded_50` | `0.250000` | `0.000000` | `0.910319` | `0.927734` | `0.928223` | `0.875000` | `0.242032` | `0.344727` | `0.038009` | `48.250000` |
| `hand_seeded` | `1.000000` | `1.000000` | `0.989583` | `0.981445` | `0.987305` | `1.000000` | `0.355418` | `0.358398` | `0.042279` | `91.000000` |

## Train / Validation / Final-Test Fitness

| Arm | Train Fitness | Validation Fitness | Final-Test Fitness | Generations Completed | Generations To Threshold |
|---|---:|---:|---:|---:|---:|
| `random_graph` | `0.996813` | `1.001312` | `0.990020` | `2000.000000` | `null` |
| `route_grammar_graph` | `0.910876` | `0.917994` | `0.906673` | `2000.000000` | `null` |
| `route_gate_grammar_graph` | `0.929932` | `0.930624` | `0.925889` | `1649.600000` | `null` |
| `route_gate_recurrence_grammar` | `0.973666` | `0.975396` | `0.969315` | `2000.000000` | `null` |
| `route_gate_hub_grammar` | `0.994478` | `1.004676` | `0.985882` | `2000.000000` | `null` |
| `damaged_hand_seeded_50` | `1.354659` | `1.359813` | `1.350275` | `2000.000000` | `1023.000000` |
| `hand_seeded` | `1.521701` | `1.489741` | `1.516858` | `0.000000` | `0.000000` |

## Leakage Audit

| Arm | Grammar Runs Pass | Direct Token->Readout Edges |
|---|---:|---:|
| `random_graph` | `True` | `0.000000` |
| `route_grammar_graph` | `True` | `0.000000` |
| `route_gate_grammar_graph` | `True` | `0.000000` |
| `route_gate_recurrence_grammar` | `True` | `0.000000` |
| `route_gate_hub_grammar` | `True` | `0.000000` |
| `damaged_hand_seeded_50` | `True` | `0.000000` |
| `hand_seeded` | `True` | `0.000000` |

## Best Graph Examples

- `random_graph` seed `0`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/random_graph_seed0.json`
- `route_grammar_graph` seed `0`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_grammar_graph_seed0.json`
- `route_gate_grammar_graph` seed `0`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_gate_grammar_graph_seed0.json`
- `route_gate_recurrence_grammar` seed `0`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_gate_recurrence_grammar_seed0.json`
- `route_gate_hub_grammar` seed `0`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_gate_hub_grammar_seed0.json`
- `damaged_hand_seeded_50` seed `0`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/damaged_hand_seeded_50_seed0.json`
- `hand_seeded` seed `0`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/hand_seeded_seed0.json`
- `random_graph` seed `1`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/random_graph_seed1.json`
- `route_grammar_graph` seed `1`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_grammar_graph_seed1.json`
- `route_gate_grammar_graph` seed `1`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_gate_grammar_graph_seed1.json`
- `route_gate_recurrence_grammar` seed `1`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_gate_recurrence_grammar_seed1.json`
- `route_gate_hub_grammar` seed `1`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_gate_hub_grammar_seed1.json`
- `damaged_hand_seeded_50` seed `1`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/damaged_hand_seeded_50_seed1.json`
- `hand_seeded` seed `1`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/hand_seeded_seed1.json`
- `random_graph` seed `2`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/random_graph_seed2.json`
- `route_grammar_graph` seed `2`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_grammar_graph_seed2.json`
- `route_gate_grammar_graph` seed `2`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_gate_grammar_graph_seed2.json`
- `route_gate_recurrence_grammar` seed `2`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_gate_recurrence_grammar_seed2.json`
- `route_gate_hub_grammar` seed `2`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_gate_hub_grammar_seed2.json`
- `damaged_hand_seeded_50` seed `2`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/damaged_hand_seeded_50_seed2.json`
- `hand_seeded` seed `2`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/hand_seeded_seed2.json`
- `random_graph` seed `3`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/random_graph_seed3.json`
- `route_grammar_graph` seed `3`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_grammar_graph_seed3.json`
- `route_gate_grammar_graph` seed `3`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_gate_grammar_graph_seed3.json`
- `route_gate_recurrence_grammar` seed `3`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_gate_recurrence_grammar_seed3.json`
- `route_gate_hub_grammar` seed `3`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_gate_hub_grammar_seed3.json`
- `damaged_hand_seeded_50` seed `3`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/damaged_hand_seeded_50_seed3.json`
- `hand_seeded` seed `3`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/hand_seeded_seed3.json`
- `random_graph` seed `4`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/random_graph_seed4.json`
- `route_grammar_graph` seed `4`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_grammar_graph_seed4.json`
- `route_gate_grammar_graph` seed `4`: `target/context-cancellation-probe/authority-graph-developmental-search/best_graphs/route_gate_grammar_graph_seed4.json`

## Failure Cases

- If grammar arms remain near random on final-test, the current mechanism still requires too much manual structure.
- If train fitness improves but validation/final-test does not, the search is overfitting the small train split.
- If accuracy improves without authority/refraction or wrong-frame drop, the graph is solving labels without the target authority mechanism.

## Minimal Surviving Prior

Read from final-test comparisons only: random vs route grammar, route vs route+gate, route+gate vs route+gate+recurrence, and recurrence vs recurrence+hub.

## Verdict

```json
{
  "supports_developmental_prior_search": false,
  "route_structure_required_for_evolution": false,
  "frame_gates_required_for_evolution": false,
  "recurrence_required_for_evolution": false,
  "hubs_help_evolution": false,
  "random_search_sufficient": false,
  "damaged_hand_recovery_supported": false,
  "manual_structure_still_dominates": true,
  "final_verdict_uses_final_test": true
}
```

## Runtime Notes

- runtime seconds: `41403.271284`
- interrupted by wall clock: `True`
- completed records: `31`

## Claim Boundary

Toy evidence only. Do not claim consciousness, biology, full VRAXION behavior, production validation, or natural-language understanding.
