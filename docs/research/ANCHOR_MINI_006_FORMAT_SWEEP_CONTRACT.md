# ANCHOR-MINI-006 Format Sweep Contract

## Purpose

ANCHOR-MINI-006 tests which AnchorCell process representation format best
supports internalized decision learning under the MINI-005 shortcut-flip stress.

The question is:

```text
Which process representation is easiest to learn, hardest to shortcut, and most
robust when process labels are hidden at eval?
```

This is not a natural-language AnchorCell proof. MINI-006A is a symbolic
format-control sweep. MINI-006B, a later text serialization sweep, is required
before claiming that prose, JSON, triples, or tables are best as literal LLM
training text.

## Research Boundary

This work overlaps:

- Learning Using Privileged Information.
- Generalized distillation.
- Process supervision.
- Concept bottleneck / process bottleneck models.
- Chain-of-thought and rationale supervision.

The AnchorCell-specific claim is the combined pipeline:

```text
process decomposition + eval masking + shortcut-flip stress + shuffled control
```

Do not treat prose or inner-monologue wins as faithful reasoning unless they
beat compact structured controls and survive masking.

## Dataset

Reuse MINI-005:

```text
candidate_count: 4
category_count: 4
train shortcut: highest surface_prior points to gold with probability 0.90
eval shortcut: highest surface_prior points to wrong candidate with probability 0.90
seeds: 2026-2125
parallel jobs: 16
```

The true answer is the candidate whose effect category matches the goal
category. Non-oracle process formats must not receive precomputed `match_bits`
as eval inputs.

## Format Arms

```text
ANSWER_ONLY
  No process supervision. Direct shortcut baseline.

ORACLE_MATCH_BITS
  Visible match bits. Upper bound only.

RAW_SYMBOLIC_PROCESS
  MINI-005-style symbolic conjunction baseline.

PROSE_PROCESS
  Loose natural-language-like process signal.

INNER_MONOLOGUE_PROCESS
  Noisier, association-heavy process signal. Diagnostic only.

STRICT_JSON_PROCESS
  Nested schema-like process signal.

FLAT_KEY_VALUE_PROCESS
  Flat field signal without explicit relation binding.

RELATIONAL_TRIPLES_PROCESS
  Relation-style goal/effect/match structure.

ACTION_OUTCOME_TABLE_PROCESS
  Action/outcome process structure with surface/action information.

RELATION_PLUS_ACTION_PROCESS
  Relational structure plus action/outcome structure.

COMPACT_HYBRID_PROCESS
  Minimal AnchorCell-style ImplicitJob + Salience + Relations +
  ActionOutcome + DecisionRule.

SHUFFLED_COMPACT_HYBRID
  Semantically wrong compact hybrid. Negative control.
```

Expected hypothesis before running:

```text
Top expected practical formats:
1. COMPACT_HYBRID_PROCESS
2. RELATION_PLUS_ACTION_PROCESS
3. RELATIONAL_TRIPLES_PROCESS

High-risk formats:
- PROSE_PROCESS may activate irrelevant associations.
- INNER_MONOLOGUE_PROCESS may be noisy.
- STRICT_JSON_PROCESS may spend capacity on schema overhead.
- FLAT_KEY_VALUE_PROCESS may lack explicit relation binding.
```

## MINI-006A Symbolic Mapping

Each format arm maps to an allowed process feature layout in the audit sparse
carrier. This isolates process-decomposition quality from tokenizer effects.

Non-oracle eval features are limited to ordinary goal/effect/surface inputs and
derived raw category gates. Oracle `match_bits` are only allowed for
`ORACLE_MATCH_BITS`.

The process fitness must use the MINI-005 balanced positive/negative bit score
so sparse positive bits cannot be ignored while the answer still happens to be
right.

## MINI-006B Text Serialization Follow-Up

MINI-006B should later test literal serialized strings:

```text
prose vs strict JSON vs flat key-value vs triples vs table vs compact hybrid
```

It must track token counts and must use the same eval masking and shortcut-flip
controls. MINI-006A does not settle literal text format performance.

## Metrics

Report per arm:

```text
answer_eval_ood_accuracy
shortcut_trap_rate
process_bit_accuracy
process_exact_row_accuracy
true_process_bit_accuracy
train_surface_alignment
eval_surface_flip_rate
format_token_count_mean
format_token_count_p95
format_efficiency
shuffled_control_gap
oracle_gap
```

Valid stress requires:

```text
train_surface_alignment >= 0.85
eval_surface_flip_rate >= 0.85
valid_seeds >= 80 / 100
```

A format is `USEFUL` if:

```text
OOD accuracy >= ANSWER_ONLY + 0.25
OOD accuracy >= SHUFFLED_COMPACT_HYBRID + 0.25
shortcut_trap_rate <= 0.25
true_process_bit_accuracy >= 0.90
no oracle eval inputs
```

Best practical format excludes:

```text
ANSWER_ONLY
ORACLE_MATCH_BITS
RAW_SYMBOLIC_PROCESS
SHUFFLED_COMPACT_HYBRID
```

Tie-breakers:

```text
1. higher OOD accuracy
2. lower shortcut_trap_rate
3. higher format_efficiency
4. lower format_token_count_p95
```

## Status Rules

```text
ANCHOR_MINI_006_FORMAT_STRONG_SIGNAL
ANCHOR_MINI_006_FORMAT_WEAK_SIGNAL
ANCHOR_MINI_006_FORMAT_NO_CLEAR_WINNER
ANCHOR_MINI_006_FORMAT_INVALID_STRESS
ANCHOR_MINI_006_FORMAT_RESOURCE_BLOCKED
```

Strong signal requires:

```text
at least one practical non-oracle format is USEFUL
best practical format beats ANSWER_ONLY by >= 0.25
best practical format beats SHUFFLED_COMPACT_HYBRID by >= 0.25
ORACLE_MATCH_BITS remains an upper bound or is matched by learned process
prose/inner-monologue cannot be declared best unless it beats compact structured formats
```

## Hygiene

Generated outputs stay under:

```text
target/anchorweave/anchor_mini006_format_sweep/
```

Do not commit raw target outputs. Do not write under:

```text
data/anchorweave/cells/
```
