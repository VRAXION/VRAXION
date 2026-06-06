# E7M Anchor Working Copy Crystallization Probe Result

Run root:

```text
target/pilot_wave/e7m_anchor_working_copy_crystallization_probe
```

## Decision

```text
decision = e7m_direct_mutation_sufficient_anchor_unneeded
best_non_oracle_system = no_anchor_direct_mutation
best_system_including_oracle = no_anchor_direct_mutation
deterministic_replay_passed = true
checker_failure_count = 0
```

## Mean Scores

```text
no_anchor_direct_mutation                                net=0.551944 raw=0.663795 ood=0.546667 anchor=0.000 promP=0.000 prune=0.062
frozen_anchor_only                                       net=0.432428 raw=0.690894 ood=0.427030 anchor=1.000 promP=0.000 prune=0.500
frozen_anchor_plus_mutable_copy                          net=0.464752 raw=0.678054 ood=0.457732 anchor=1.000 promP=0.000 prune=0.250
frozen_anchor_plus_mutable_copy_plus_pruning             net=0.456814 raw=0.690805 ood=0.451828 anchor=1.000 promP=0.000 prune=0.500
frozen_anchor_plus_mutable_copy_plus_prune_and_promote   net=0.456814 raw=0.690805 ood=0.451828 anchor=1.000 promP=0.000 prune=0.500
multi_copy_competition                                   net=0.455761 raw=0.689752 ood=0.450775 anchor=1.000 promP=0.000 prune=0.500
random_copy_control                                      net=0.281451 raw=0.486412 ood=0.283649 anchor=1.000 promP=0.000 prune=0.000
oracle_anchor_reference                                  net=0.483255 raw=0.699390 ood=0.483548 anchor=1.000 promP=0.000 prune=0.500
```

## Interpretation

The anchor plus working-copy lifecycle did protect frozen anchors, preserved lineage, and pruning was stable. It did not improve net utility under the E7M cost model. The raw usefulness of anchor/prune variants stayed competitive, but spawn/copy/maintenance/prune overhead dominated enough that direct mutation remained better on every phase.

Promotion did not produce guarded wins in this run:

```text
promote_precision = 0.0
bad_promotion_rate = 0.0
prune_compression_ratio = 0.5
post_prune_utility_delta = 0.25
```

This means the promotion guard was conservative rather than unsafe: it avoided bad promotions, but did not find copy improvements worth promoting.

## Phase Winners

```text
phase_1_existing_library_sufficient          no_anchor_direct_mutation
phase_2_missing_reusable_transform           no_anchor_direct_mutation
phase_3_reuse_multiple_contexts              no_anchor_direct_mutation
phase_4_ood_counterfactual_generalization    no_anchor_direct_mutation
phase_5_damage_drift_repair                  no_anchor_direct_mutation
```

## Boundary

E7M is a controlled symbolic/numeric pocket-library lifecycle probe. It does not make raw-language, deployed-model, AGI, consciousness, or model-scale claims.
