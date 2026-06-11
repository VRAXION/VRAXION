# E19_HARD_REPO_TEXT_OPEN_RETRIEVAL_REASONING_CONFIRM Contract

## Purpose

E19 hardens the E18B repository-text stress milestone by testing open retrieval and multi-hop reasoning over local repository documents with minimal scaffolding. It is not a new toy candidate-set task: hard episodes use no source-path hint, no exact field-key hint, and large candidate pools or full-corpus style retrieval.

## Boundary

This is a hard real-repository-text open-retrieval and reasoning stress audit for a controlled Flow text policy. It uses local project documents and adversarial deterministic task wrappers. It does not prove general natural-language AI, internet-scale LLM behavior, or production readiness.

## Corpus

Allowed sources are local repository documents only:

- `docs/research/*.md`
- `docs/wiki/*.md` if present
- `README*`
- `CHANGELOG.md`

Generated `target/`, `.git/`, binary files, and generated artifacts are excluded. Train, validation, heldout, and stress splits are made by whole file and must not overlap.

## Hardness requirements

Hard stress questions must not expose source paths, direct chunk ids, exact field keys, direct expected answers, task family labels, or tiny target-guaranteed candidate lists. Candidate pools must average at least 500 chunks for hard stress episodes and include hard negatives when available.

## Task families

E19 includes open no-path retrieval, indirect milestone identification, paraphrase field reasoning, multi-hop result chains, contradictory evidence resolution, missing/ambiguous calibration, hard-negative retrieval, target-not-first long context, two-chunk synthesis, numeric reasoning, table paraphrase, caveat synthesis, adversarial abstain, and heldout transfer composition.

## Full-confirm minimums

A full E19 confirmation requires at least 60 generations, population 96, 1200 heldout episodes, 1200 stress episodes, 6000 candidate evaluations, 60 checkpoints, average hard candidate pool at least 500, 200 no-source-path hard episodes, 200 ambiguous/missing episodes, and 150 multi-hop episodes.

## Checker

The checker recomputes metrics from per-episode logs, validates budget/candidate-pool distributions, audits source fixtures and split leakage, rejects oracle/control primaries, verifies that hard questions do not leak source paths/field keys/chunk ids, and forbids broad AGI/general NLP/production-readiness claims.

## Run command

```bash
python3 scripts/probes/run_e19_hard_repo_text_open_retrieval_reasoning_confirm.py \
  --out target/pilot_wave/e19_hard_repo_text_open_retrieval_reasoning_confirm \
  --strict-budget \
  --no-downshift \
  --generations 100 \
  --population 128 \
  --train-episodes 4000 \
  --validation-episodes 1000 \
  --heldout-episodes 1500 \
  --stress-episodes 1500 \
  --candidate-pool-size 500 \
  --checkpoint-every 1 \
  --max-runtime-minutes 360 \
  --resume

python3 scripts/probes/run_e19_hard_repo_text_open_retrieval_reasoning_confirm_check.py \
  --out target/pilot_wave/e19_hard_repo_text_open_retrieval_reasoning_confirm \
  --write-summary
```
