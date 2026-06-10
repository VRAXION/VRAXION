# E17 Repo Text Mutation Training Overnight Audit Contract

## Goal

Implement and run `E17_REPO_TEXT_MUTATION_TRAINING_OVERNIGHT_AUDIT`.

This milestone moves the text-facing Flow policy audit from controlled synthetic
nonce-token streams into real local repository documents. The audit uses only
local markdown files from the repository, creates file-level train/validation/
heldout splits, generates deterministic task wrappers and labels from parsed
source text, trains Flow-style candidate policies by mutation/search, and
evaluates the final policy on heldout repository files.

## Scope

Included corpus roots:

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
generated output artifacts
external datasets
internet downloads
neural libraries
```

Task families:

```text
FIELD_EXTRACTION
METRIC_COMPARISON
RESULT_SUMMARY_CANONICAL
DOCUMENT_RETRIEVAL
CROSS_DOC_NEXT_CHAIN
CAVEAT_BOUNDARY_DETECTION
NOISY_CONTEXT_REPAIR
LONG_CONTEXT_MEMORY
TABLE_ROW_EXTRACTION
AMBIGUOUS_OR_MISSING_EVIDENCE
```

The policy space is non-neural and Flow-style: chunk scoring, heading/path
weighting, key-value extraction, table parsing, numeric parsing, canonical
decoding, evidence memory, distractor rejection, abstain thresholds, trace gates,
and cost penalties.

## Required Systems

```text
STATIC_KEYWORD_BASELINE
BM25_LIKE_BASELINE
HEADING_PATH_WEIGHTED_BASELINE
HAND_AUTHORED_EXTRACTOR_CONTROL
RANDOM_POLICY_BASELINE
MUTATION_TRAINED_REPO_TEXT_POLICY
MUTATION_TRAINED_PRUNED_REPO_TEXT_POLICY_PRIMARY
NO_HEADING_PATH_ABLATION
NO_TABLE_PARSER_ABLATION
NO_NUMERIC_PARSER_ABLATION
NO_ABSTAIN_POLICY_ABLATION
NO_DISTRACTOR_REJECTION_ABLATION
NO_LONG_CONTEXT_MEMORY_ABLATION
NO_CANONICAL_DECODER_STRICTNESS_ABLATION
```

The hand-authored extractor control is a privileged reference only and is invalid
as a primary proof system.

## Gate

Full confirmation requires:

```text
exact_answer_accuracy >= 0.70
canonical_object_accuracy >= 0.70
evidence_chunk_accuracy >= 0.75
retrieval_top1_accuracy >= 0.75
field_extraction_accuracy >= 0.80
table_row_extraction_accuracy >= 0.70
noisy_context_repair_accuracy >= 0.65
long_context_memory_accuracy >= 0.65
ambiguity_handling_accuracy >= 0.70
hallucinated_answer_rate <= 0.05
wrong_evidence_rate <= 0.10
trace_validity >= 0.90
renderer_faithfulness >= 0.98
exact_answer_accuracy beats BM25_LIKE_BASELINE by >= 0.08
canonical_object_accuracy beats STATIC_KEYWORD_BASELINE by >= 0.08
aggregate_recomputed_from_episode_logs == true
source_fixture_audit_passed == true
deterministic_replay_passed == true
checker_failure_count == 0
```

## Verification

```text
python3 -m py_compile scripts/probes/run_e17_repo_text_mutation_training_overnight_audit.py scripts/probes/run_e17_repo_text_mutation_training_overnight_audit_check.py
python3 scripts/probes/run_e17_repo_text_mutation_training_overnight_audit.py --out target/pilot_wave/e17_repo_text_mutation_training_overnight_audit --generations 80 --population 96 --train-episodes 2000 --validation-episodes 500 --heldout-episodes 800 --checkpoint-every 1 --max-runtime-minutes 360 --resume
python3 scripts/probes/run_e17_repo_text_mutation_training_overnight_audit_check.py --out target/pilot_wave/e17_repo_text_mutation_training_overnight_audit --write-summary
```

## Boundary

This is an overnight real-repository-text mutation-training audit for a controlled Flow text policy. It uses real local project documents, but task wrappers and labels are deterministically generated from those documents. It does not prove general natural-language AI, internet-scale LLM behavior, or production readiness.
