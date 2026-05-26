# D34 Raven Pocket Corridor Baseline Suite Result

## Run status
- Smoke run: completed.
- Full run: completed.
- Decision is bounded and does not claim solve.

## Artifact roots
- `target/pilot_wave/d34_raven_pocket_corridor_baseline_suite/smoke`
- `target/pilot_wave/d34_raven_pocket_corridor_baseline_suite/full`

## Methods run
- random_baseline
- simple_neural_net_baseline
- direct_vraxion_mutation
- separate_population_evolution
- dna_u64_genome_encoding

## Unavailable methods
- shadow_clone_mutation (reconstruction unavailable in current repo)

## Smoke aggregate (8 seeds)
- random_baseline: test 0.1119, ood 0.0713
- simple_neural_net_baseline: test 0.2200, ood 0.1994
- direct_vraxion_mutation: test 0.4100, ood 0.3731
- separate_population_evolution: test 0.3363, ood 0.2988
- dna_u64_genome_encoding: test 0.2775, ood 0.2694

## Full aggregate (8 seeds)
- random_baseline: test 0.1076, ood 0.0807
- simple_neural_net_baseline: test 0.2130, ood 0.1933
- direct_vraxion_mutation: test 0.4098, ood 0.3761
- separate_population_evolution: test 0.3436, ood 0.3118
- dna_u64_genome_encoding: test 0.2946, ood 0.2599

## Decision
- `direct_mutation_beats_tested_dna_genome_encoding`
- Next: `D35_DIRECT_MUTATION_ROUTING_HARDENING_PLAN`

## Notes
- Random baseline remained near 1/9.
- No method reached solved thresholds (test>=0.90 and ood>=0.85).
- This run records baseline comparison and supports direct mutation as current baseline-to-beat.
