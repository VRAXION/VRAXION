# E132 External Math Text Skill Farm Mutation/Prune Orange Cycle Contract

## Purpose

E132 farms scoped math-text Operators from the external E132 seed pack and runs
them through an Orange/LegendaryCandidate gate:

```text
external support -> scoped candidate -> mutation/prune/challenger/reload
-> negative scope -> Orange/LegendaryCandidate or hold
```

## Required Inputs

- Normalized external dataset:
  `target/datasets/e132_external_math_text_seed_pack/normalized/e132_external_math_text_skill_seed.jsonl`
- Dataset manifests:
  - `target/datasets/e132_external_math_text_seed_pack/download_manifest.json`
  - `target/datasets/e132_external_math_text_seed_pack/normalized_manifest.json`

The committed artifact sample pack records the manifest and aggregate evidence;
the full raw/normalized dataset remains under ignored `target/`.

## Positive Gate

For each candidate Operator:

- external support must be at least the configured support minimum;
- scoped Orange campaign count must be at least 8;
- pressure family coverage must be at least 12;
- reload shadow pass must be true;
- negative-scope pass must be true;
- sibling challenger pass must be true;
- prune pass must be true;
- selected variant must be a scoped pruned form;
- qualified activation must reach the Orange/LegendaryCandidate threshold.

## Safety Gate

The run must keep all of these at zero:

```text
hard_negative
wrong_scope_call
false_commit
unsupported_answer
boundary_claim_violation
direct_flow_write
```

It must also include a rejected overbroad solver control that demonstrates
wrong-scope risk on hidden/prose math surfaces.

## Boundary

E132 is scoped math-text lens/guard farming only. It does not claim:

- GSM8K or MATH benchmark solving;
- natural-language word-problem solving;
- open-domain mathematical reasoning;
- neural training or learned weights;
- production assistant readiness;
- Core, PermaCore, or TrueGolden promotion.

## Artifacts

The runner writes:

```text
run_manifest.json
download_report.json
dataset_report.json
skill_candidate_report.json
operator_cards.json
operator_orange_results.json
variant_report.json
promotion_report.json
negative_scope_report.json
mutation_summary.json
mutation_events.jsonl
row_level_samples.jsonl
progress.jsonl
partial_aggregate_snapshot.json
aggregate_metrics.json
deterministic_replay.json
decision.json
summary.json
report.md
checker_summary.json
operator_registry/*.json
```

## Reproduce

```powershell
python private_probe_runner_removed
```

Smoke mode without the external dataset:

```powershell
python private_probe_runner_removed --dataset target/datasets/missing_e132_smoke.jsonl --allow-builtin-dataset --out target/ci/e132_smoke --sample-out "" --min-dataset-rows 1 --min-external-support 1 --min-orange 16 --dataset-row-limit 80
```
