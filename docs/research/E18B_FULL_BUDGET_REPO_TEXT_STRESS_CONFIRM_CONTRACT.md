# E18B_FULL_BUDGET_REPO_TEXT_STRESS_CONFIRM Contract

## Purpose

E18B is a full-budget-capable runner/checker milestone for real repository text stress evaluation of a controlled Flow text policy. It exists to follow the partial-downshifted E18 result without falsely claiming a full confirmation from a small online run.

## Boundary

This is a real-repository-text stress and latency audit for a controlled Flow text policy. It uses local project documents and adversarial deterministic task wrappers. It does not prove general natural-language AI, internet-scale LLM behavior, or production readiness.

## Budget modes

The runner exposes explicit budget mode flags:

- `--strict-budget`
- `--no-downshift`
- `--smoke`

If `--strict-budget` and `--no-downshift` are set, the runner must execute the requested budget or exit with an incomplete/insufficient-budget decision. It must not silently reduce generations, population, or episodes. If `--smoke` is set, the runner may use a tiny controlled budget but the decision must be preflight/smoke rather than full confirmed. If a non-smoke budget is below the full-confirm minimums, the decision must be partial, partial-downshifted, failed, or invalid, never full confirmed.

## Full-confirm minimums

A full E18B confirmation requires all of the following actual counts:

- `generations_completed >= 40`
- `population_size >= 64`
- `heldout_episode_count >= 800`
- `stress_episode_count >= 800`
- `candidate_count_evaluated >= 2560`
- `checkpoint_count >= 40`

Below these minimums, full confirmation is forbidden.

## Corpus and leakage policy

The runner uses only local repository files from these fixtures:

- `docs/research/*.md`
- `docs/wiki/*.md` if present
- `README*`
- `CHANGELOG.md`

It excludes `.git/`, `target/`, binary files, and generated artifact output. Train, validation, heldout, and stress splits are made by whole file and must not overlap.

## Task families

E18B focuses on the E18 bottleneck families:

1. `NO_SOURCE_PATH_FIELD_EXTRACTION`
2. `PARAPHRASED_FIELD_EXTRACTION`
3. `SAME_KEY_CONFLICT_RETRIEVAL`
4. `SAME_MILESTONE_DISTRACTOR`
5. `TARGET_NOT_FIRST_LONG_CONTEXT`
6. `ADVERSARIAL_NOISY_CONTEXT`
7. `LONG_CONTEXT_MEMORY`
8. `TABLE_NUMERIC_STRESS`
9. `METRIC_DELTA_STRESS`
10. `AMBIGUOUS_OR_MISSING_EVIDENCE`
11. `CAVEAT_BOUNDARY_PARAPHRASE`
12. `SOURCE_PATH_HINT_ABLATION`
13. `FIELD_NAME_HINT_ABLATION`

## Systems and controls

The comparison set includes keyword, BM25-like, heading/path weighted, random, mutation-trained, pruned primary, oracle controls, hand-authored control, and feature ablations. The source-path oracle, field-name oracle, and hand-authored extractor controls are invalid as primary systems.

## Checker requirements

The checker recomputes heldout/stress metrics, hint ablations, task-family metrics, latency metrics, training curve values from generation scores, decision gates, source fixture audit, split leakage audit, and budget class validity from the emitted artifacts. It fails on false full confirmation, aggregate metrics not backed by per-episode logs, source fixture failure, split overlap, oracle/control primary selection, static metric-table use, neural dependency claims, or broad AGI/general NLP/production readiness claims.

## Local full-budget command

```bash
python3 scripts/probes/run_e18b_full_budget_repo_text_stress_confirm.py \
  --out target/pilot_wave/e18b_full_budget_repo_text_stress_confirm \
  --strict-budget \
  --no-downshift \
  --generations 80 \
  --population 96 \
  --train-episodes 2500 \
  --validation-episodes 700 \
  --heldout-episodes 1000 \
  --stress-episodes 1000 \
  --checkpoint-every 1 \
  --max-runtime-minutes 360 \
  --resume

python3 scripts/probes/run_e18b_full_budget_repo_text_stress_confirm_check.py \
  --out target/pilot_wave/e18b_full_budget_repo_text_stress_confirm \
  --write-summary
```
