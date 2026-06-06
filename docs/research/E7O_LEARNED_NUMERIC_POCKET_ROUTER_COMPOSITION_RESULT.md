# E7O Learned Numeric Pocket Router Composition Result

Run root:

```text
target/pilot_wave/e7o_learned_numeric_pocket_router_composition
```

## Decision

```text
decision = e7o_float_only_numeric_pocket_composition
best_non_reference_system = ternary_binary_numeric_pocket_router
deterministic_replay_passed = True
checker_failure_count = 0
```

Interpretation: real learned numeric pockets were trained successfully, but the composed numeric pocket-router stack did not reach a strong clean composition-confirmation threshold. The best non-reference system was the low-bit `ternary_binary_numeric_pocket_router`, but the overall result still points to a numeric pocket/flow-contract bottleneck rather than a fully confirmed learned numeric composition bridge.

## Mean Scores

```text
symbolic_proxy_pocket_router_reference           useful=0.991500 acc=1.000000 route=1.000000 bits=0.0
float_numeric_pocket_library_router              useful=0.576065 acc=0.696065 route=0.444444 bits=3288576.0
int8_numeric_pocket_library_router               useful=0.575602 acc=0.695602 route=0.444444 bits=823296.0
int4_pruned_numeric_pocket_library_router        useful=0.592372 acc=0.669907 route=0.583333 bits=351845.3
ternary_binary_numeric_pocket_router             useful=0.603481 acc=0.640162 route=0.777778 bits=146740.0
mixed_precision_numeric_pocket_router            useful=0.587683 acc=0.670486 route=0.583333 bits=378182.0
monolithic_backprop_model                        useful=0.592134 acc=0.669560 route=0.000000 bits=430144.0
monolithic_mutation_model                        useful=0.600116 acc=0.600116 route=0.000000 bits=2624.0
dense_graph_danger_control                       useful=0.565787 acc=0.705787 route=0.000000 bits=2635840.0
oracle_router_over_numeric_pockets               useful=0.514606 acc=0.634606 route=1.000000 bits=3288576.0
```

## Pocket Quality

```text
compare                float=0.997222 int8=0.997338 int4p=0.995718 binary=0.915162
mod_add                float=0.995949 int8=0.995949 int4p=0.992477 binary=0.716898
parity                 float=0.997801 int8=0.997917 int4p=0.996644 binary=0.675116
threshold              float=0.998380 int8=0.998264 int4p=0.992361 binary=0.847106
counterfactual_flip    float=0.994907 int8=0.995139 int4p=0.989815 binary=0.382292
verify                 float=0.999190 int8=0.999190 int4p=0.998843 binary=0.736806
```

## Findings

```text
learned float pockets alone: strong
numeric pocket composition: weak-to-moderate, not cleanly confirmed
best numeric router: ternary_binary_numeric_pocket_router
best precision signal: low-bit routing was more route-efficient than float/int8
main bottleneck: composition/interface/contract, not standalone pocket training
dense graph control: did not win
monolithic mutation: competitive usefulness, but no route structure
```

## Next

Run a contract-focused follow-up before adding new architecture:

```text
E7P_NUMERIC_POCKET_INTERFACE_CONTRACT_AUDIT
```

Primary question: why do near-perfect standalone numeric pockets degrade under composition? Test adapter normalization, result-slot hygiene, flow-state preservation, route-step loss, and pocket output calibration before scaling the router.

## Boundary

E7O is a controlled numeric pocket-router composition probe. It does not make raw-language, deployed-model, AGI, consciousness, or model-scale claims.
