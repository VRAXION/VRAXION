# E18 Repo Text Policy Stress And Latency Confirm Contract

## Goal

Implement and run `E18_REPO_TEXT_POLICY_STRESS_AND_LATENCY_CONFIRM`.

E18 directly stresses the E17 real-repository-text policy result under less
hinted same-domain conditions: no direct source path, paraphrased field names,
same-key conflicts, same-milestone distractors, target-not-first long contexts,
table/numeric stress, ambiguity/missing evidence, and measured per-query latency.

## Scope

Corpus inputs are local repository documents only:

```text
docs/research/*.md
docs/wiki/*.md
README*
CHANGELOG.md
```

Excluded inputs:

```text
target/
.git/
binary files
external datasets
neural libraries
self-leaking E18 generated documents
```

## Task Families

```text
NO_SOURCE_PATH_FIELD_EXTRACTION
PARAPHRASED_FIELD_EXTRACTION
SAME_KEY_CONFLICT_RETRIEVAL
SAME_MILESTONE_DISTRACTOR
TARGET_NOT_FIRST_LONG_CONTEXT
TABLE_NUMERIC_STRESS
METRIC_DELTA_STRESS
CROSS_DOC_CHAIN_STRESS
CAVEAT_BOUNDARY_PARAPHRASE
AMBIGUOUS_OR_MISSING_EVIDENCE
ADVERSARIAL_NOISY_CONTEXT
LATENCY_COST_STRESS
SOURCE_PATH_HINT_ABLATION
FIELD_NAME_HINT_ABLATION
HELDOUT_FUTURE_DOCS
```

## Systems

```text
E17_POLICY_REFERENCE
STATIC_KEYWORD_BASELINE
BM25_LIKE_BASELINE
HEADING_PATH_WEIGHTED_BASELINE
SOURCE_PATH_ORACLE_CONTROL
FIELD_NAME_ORACLE_CONTROL
HAND_AUTHORED_EXTRACTOR_CONTROL
RANDOM_POLICY_BASELINE
MUTATION_TRAINED_STRESS_POLICY
MUTATION_TRAINED_PRUNED_STRESS_POLICY_PRIMARY
NO_SOURCE_PATH_FEATURE_ABLATION
NO_HEADING_PATH_FEATURE_ABLATION
NO_TABLE_PARSER_ABLATION
NO_NUMERIC_PARSER_ABLATION
NO_ABSTAIN_POLICY_ABLATION
NO_DISTRACTOR_REJECTION_ABLATION
NO_LONG_CONTEXT_MEMORY_ABLATION
NO_PARAPHRASE_ALIAS_ABLATION
NO_CANONICAL_DECODER_STRICTNESS_ABLATION
```

Oracle/control systems are invalid as primary.

## Budget Rule

Full confirmation requires a real full-ish stress run:

```text
generations_completed >= 40
population_size >= 64
heldout_episode_count >= 800
stress_episode_count >= 800
run_budget_class != partial_downshifted
```

If the interactive runner downshifts below those minimums, the result must be
`e18_repo_text_policy_stress_and_latency_partial_downshifted` even if stress
metrics are otherwise good.

## Verification

```text
python3 -m py_compile scripts/probes/run_e18_repo_text_policy_stress_and_latency_confirm.py scripts/probes/run_e18_repo_text_policy_stress_and_latency_confirm_check.py
python3 scripts/probes/run_e18_repo_text_policy_stress_and_latency_confirm.py --out target/pilot_wave/e18_repo_text_policy_stress_and_latency_confirm --generations 80 --population 96 --train-episodes 2500 --validation-episodes 700 --heldout-episodes 1000 --stress-episodes 1000 --checkpoint-every 1 --max-runtime-minutes 360 --resume
python3 scripts/probes/run_e18_repo_text_policy_stress_and_latency_confirm_check.py --out target/pilot_wave/e18_repo_text_policy_stress_and_latency_confirm --write-summary
```

## Boundary

This is a real-repository-text stress and latency audit for a controlled Flow text policy. It uses local project documents and adversarial deterministic task wrappers. It does not prove general natural-language AI, internet-scale LLM behavior, or production readiness.
