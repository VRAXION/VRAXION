# E7Y Natural Output Bundle Width Audit Result

## Decision

```text
decision = e7y_output_bundle_width_not_sufficient
best_non_reference_system = output_bundle_N6
plateau_width = 6
deterministic_replay_passed = true
checker_failure_count = 0
```

## Mean Scores

```text
single_value_write_baseline      useful=0.499935 acc=0.599935 oracle_sim=0.763251 next=0.005413
output_bundle_N2                 useful=0.488650 acc=0.588650 oracle_sim=0.725981 next=0.022578
output_bundle_N3                 useful=0.494835 acc=0.594835 oracle_sim=0.735629 next=0.029388
output_bundle_N4                 useful=0.495812 acc=0.595812 oracle_sim=0.735881 next=0.038450
output_bundle_N5                 useful=0.496354 acc=0.596354 oracle_sim=0.723169 next=0.045479
output_bundle_N6                 useful=0.502322 acc=0.602322 oracle_sim=0.724387 next=0.052600
output_bundle_N8                 useful=0.501997 acc=0.601997 oracle_sim=0.721745 next=0.066597
output_bundle_N12                useful=0.462934 acc=0.562934 oracle_sim=0.701916 next=0.088724
oracle_write_reference           useful=0.986317 acc=1.000000 oracle_sim=1.000000 next=0.000000
dense_graph_danger_control       useful=0.503299 acc=0.603299 oracle_sim=0.622534 next=0.275783
```

## Interpretation

E7Y did not find a meaningful natural output bundle-width plateau. The best
non-reference width was `N=6`, but it improved usefulness by only about
`0.00239` over the single-value baseline and closed only about `0.49%` of the
oracle gap. This means the E7X failure is not explained by "one output cell is
too narrow."

The large oracle gap remains:

```text
single_value_write_baseline = 0.499935
best_non_reference_N6      = 0.502322
oracle_write_reference     = 0.986317
```

So E7X's bottleneck was not explained by "one output cell is too narrow." More
channels alone add cost and next-pocket compatibility error. The likely missing
piece is not output width, but a stronger state-transition/write contract:
how the written bundle is integrated into Flow/RAM and how the next pocket is
trained to consume that canonical transition.

## Boundary

E7Y is a controlled numeric Flow/RAM output-width diagnostic. It does not prove
raw-language learning, AGI, consciousness, or model-scale behavior.
