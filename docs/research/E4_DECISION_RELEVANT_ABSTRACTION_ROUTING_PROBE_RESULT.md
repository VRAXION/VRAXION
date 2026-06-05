# E4_DECISION_RELEVANT_ABSTRACTION_ROUTING_PROBE Result

Status: complete.

## Decision

```text
decision = e4_decision_relevant_abstraction_routing_confirmed
winner = top_down_hierarchical_router
next = E5_REAL_BACKEND_HIERARCHICAL_ROUTING_STRESS_SCALE
```

E4 supports the controlled symbolic claim that explicit verdict -> cause -> mechanism -> evidence routing can choose the useful answer level before expanding irrelevant details.

## Evidence Run

Canonical artifact root:

```text
target/pilot_wave/e4_decision_relevant_abstraction_routing_probe
```

Configuration:

```text
seeds = 75001,75002,75003,75004,75005
train_rows_per_seed = 800
validation_rows_per_seed = 300
heldout_rows_per_seed = 300
ood_rows_per_seed = 300
counterfactual_rows_per_seed = 300
population_size = 24
generations = 80
elite_count = 4
mutation_sigma = 0.10
```

Checker:

```text
failure_count = 0
```

## Key Metrics

```text
flat_detail_scanner:
  usefulness = 0.0
  verdict = 0.0
  level = 0.0
  causal_path = 0.0
  over_detail = 0.63
  irrelevant_branch = 1.0

bottom_up_evidence_scanner:
  usefulness = 0.899826666667
  verdict = 1.0
  level = 0.833333333333
  causal_path = 1.0
  over_detail = 0.131333333333
  irrelevant_branch = 0.044

top_down_hierarchical_router:
  usefulness = 1.0
  verdict = 1.0
  level = 1.0
  causal_path = 1.0
  over_detail = 0.0
  irrelevant_branch = 0.0

dynamic_state_medium_router:
  usefulness = 1.0
  verdict = 1.0
  level = 1.0
  causal_path = 1.0
  over_detail = 0.0
  irrelevant_branch = 0.0
```

## Backend Audit

```text
accepted_mutation_count_total = 2499
rejected_mutation_count_total = 5181
rollback_count_total = 5181
rollback_test_passed = true
deterministic_replay_passed = true
route_index_leak_detected = false
candidate_name_leak_detected = false
static_metric_dictionary_used = false
hardcoded_improvement_used = false
synthetic_harness_only = false
```

## Boundary

E4 is a controlled symbolic probe for decision-relevant answer-level routing. It is not raw natural-language reasoning, model-scale general reasoning, or evidence that the same behavior transfers to open-ended language without further probes.
