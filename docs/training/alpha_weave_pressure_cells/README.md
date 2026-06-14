# Alpha-Weave Pressure Cells

This directory stores curated, high-quality Alpha-Weave training cells for VRAXION Operator pressure training.

These files are **training data artifacts**, not runtime claims and not promotion proofs.

## Purpose

An Alpha-Weave Pressure Cell is a targeted teaching situation that activates one or more scoped Operators through visible evidence, while checking that the system does **not** commit unsafe or unsupported state.

The goal is to train and evaluate:

- when to answer
- when to ask for evidence
- when to defer
- when not to call a route at all
- which evidence span supports the action
- which evidence must be ignored as stale, weak, irrelevant, quoted, or out of scope
- whether the trace/citation is mechanically valid

## Required separation

Every cell must keep these layers separate:

```text
public_input
  What the candidate/runtime is allowed to see.
  This must contain no oracle answer, no target operator label, and no hidden trainer hint.

hidden_oracle
  Expected action, expected answer, required citation, required trace, and forbidden behavior.
  This is for checkers/trainers only.

training_metadata
  Target skill, target Operators, authoring metadata, rank seed, intended pressure, and routing budget.
  This must stay hidden from the candidate/runtime.

cell_pack_variants
  Counterfactual and adversarial variants of the same base cell.
  A useful cell should test both when to act and when not to act.
```

## Core invariants

A valid cell must preserve these rules:

```text
1. No hidden oracle leak in public_input.
2. No target_operator leak in public_input.
3. Evidence must be visible and citeable.
4. Answers require sufficient evidence coverage.
5. Missing evidence must route to ASK_FOR_EVIDENCE or DEFER.
6. Negative-scope examples must not trigger the active skill route.
7. A correct answer without valid trace/citation is still a failure.
8. Wrong-scope calls, false commits, unsupported answers, and over-budget full scans are hard negatives.
```

## Recommended cell quality ranks

```text
BronzeCell
  One clean answerable case and a valid schema.

SilverCell
  Base case plus missing-evidence, weak-source, and negative-scope variants.

GoldCell
  Includes stale replay, source-trust inversion, order swap, citation trap, and shortcut traps.

DiamondCell
  Survives multiple task families, hidden-oracle leak checks, replay checks, and non-template paraphrase stress.

CorePressureCell
  Approved for recurring regression replay and high-value Operator probation.
```

## File naming

Use one cell per file:

```text
cells/<cell_id>.yaml
```

Example:

```text
cells/aw_coldroom_status_000001.yaml
```

## Current first seed

```text
aw_coldroom_status_000001
family = evidence_conflict
skill = latest_verified_evidence_overrides_stale_status
rank_seed = SilverCell
```

This seed trains a scoped status-resolution situation where a latest verified repair note overrides an earlier verified failure report and a weak rumor. It also includes missing-evidence, weak-source, order-swap, source-inversion, and negative-scope variants.
