# E19_HARD_REPO_TEXT_OPEN_RETRIEVAL_REASONING_CONFIRM Result

## Status

This result document records the E19 hard real-repository-text open-retrieval and reasoning stress run. The machine-readable authoritative artifacts are written under:

```text
target/pilot_wave/e19_hard_repo_text_open_retrieval_reasoning_confirm/
```

## Boundary

This is a hard real-repository-text open-retrieval and reasoning stress audit for a controlled Flow text policy. It uses local project documents and adversarial deterministic task wrappers. It does not prove general natural-language AI, internet-scale LLM behavior, or production readiness.

## Result snapshot

- `decision = e19_hard_repo_text_open_retrieval_reasoning_confirmed`
- `next = E19_CHECKER_REVIEW_OR_NEXT_HARDENING_REPAIR`
- `primary_system = MUTATION_TRAINED_PRUNED_OPEN_RETRIEVAL_POLICY_PRIMARY`
- `positive_gate_passed = true`
- `checker_failure_count = 0`
- `run_budget_class = full_budget`
- `full_budget_met = true`
- `runtime_minutes = 3.054071`
- `generations_completed = 100`
- `population_size = 128`
- `candidate_count_evaluated = 12800`
- `checkpoint_count = 100`
- `heldout_episode_count = 1500`
- `stress_episode_count = 1500`
- `candidate_pool_mean = 500.000000`
- `hard_negative_count_mean = 50.000000`
- `open_retrieval_accuracy = 0.950000`
- `multi_hop_chain_accuracy = 1.000000`
- `missing_evidence_accuracy = 1.000000`
- `ambiguity_handling_accuracy = 1.000000`
- `hallucinated_answer_rate = 0.000000`
- `wrong_evidence_rate = 0.016000`
- `trace_validity = 0.984000`
- `renderer_faithfulness = 1.000000`

The checker recomputed aggregate metrics from per-episode logs and reported zero checker failures. Full confirmation is allowed for this run because the requested strict/no-downshift budget and all full-confirm minimums were met.
