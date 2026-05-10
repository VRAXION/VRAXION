# ANCHOR-MINI-007 Text Carrier Contract

## Purpose

ANCHOR-MINI-007 tests whether AnchorCell-style process supervision still helps
when the process is represented as literal serialized text instead of symbolic
feature layouts.

Primary question:

```text
Can a small deterministic text carrier learn shortcut-resistant decisions from
serialized AnchorCell process text when that process text is hidden at eval?
```

This is a bridge test between:

```text
MINI-006 symbolic format win
-> model-facing serialized training text
```

It is not a Qwen, VRAXION, or natural-language AnchorCell proof.

## Research Boundary

This test is a small Learning Using Privileged Information / generalized
distillation probe. The process section is privileged training information.
Eval inputs must not contain the process section.

Strong results support only this narrow claim:

```text
Literal structured process text can shape a small text carrier toward
shortcut-resistant task-only decisions.
```

Do not treat a positive MINI-007 result as proof of symbol grounding, scale
generalization, or LLM fine-tuning performance.

## Dataset

Reuse the MINI-006 shortcut-flip task:

```text
candidate_count: 4
category_count: 4
train shortcut: highest surface_prior points to gold with probability 0.90
eval shortcut: highest surface_prior points to wrong candidate with probability 0.90
seeds: 2026-2125
```

The true answer is the candidate whose effect category matches the goal
category. The surface shortcut is deliberately flipped out-of-distribution.

## Eval Masking Rule

The answer label is never part of the input text.

For `ANSWER_ONLY`:

```text
train input: TASK only
eval input: TASK only
```

For process arms:

```text
train teacher view: TASK + PROCESS
train student view: TASK only
eval input: TASK only
```

Training uses answer loss on the task-only view plus answer/consistency loss
from the teacher process view. This is the mechanism that lets process text act
as privileged supervision while remaining hidden at eval.

If a run evaluates with visible `PROCESS`, it is an oracle diagnostic and must
not be reported as MINI-007 internalization evidence.

## Format Arms

```text
ANSWER_ONLY
  No process text. Shortcut baseline.

PROSE_PROCESS
  Short natural-language process statement.

INNER_MONOLOGUE_PROCESS
  Messier human-style trace. Diagnostic only.

STRICT_JSON_PROCESS
  Schema-like nested JSON process object.

FLAT_KEY_VALUE_PROCESS
  Compact key-value lines.

RELATIONAL_TRIPLES_PROCESS
  Goal/effect/match relation triples.

ACTION_OUTCOME_TABLE_PROCESS
  Compact action/outcome table with candidate policy.

RELATION_PLUS_ACTION_PROCESS
  Relation triples plus action/outcome table.

COMPACT_HYBRID_PROCESS
  Minimal AnchorCell-style ImplicitJob + Salience + Relations +
  ActionOutcome + DecisionRule.

SHUFFLED_ACTION_OUTCOME_TABLE
  Same surface format as the action/outcome table, but semantically wrong.

SHUFFLED_COMPACT_HYBRID
  Same surface format as compact hybrid, but semantically wrong.
```

Expected hypothesis before running:

```text
ACTION_OUTCOME_TABLE_PROCESS should beat answer-only, prose, inner monologue,
and shuffled action/outcome control.
```

## Text Carriers

Use CPU-safe deterministic PyTorch models:

```text
CHAR_BOW_MLP
  character counts -> MLP -> 4 answer logits

CHAR_CNN
  character embedding -> small 1D convolution -> pooled -> 4 answer logits

CHAR_GRU
  character embedding -> GRU final state -> 4 answer logits
```

Default full run:

```text
CHAR_GRU, CHAR_BOW_MLP
```

If `CHAR_GRU` is resource-prohibitive on the local CPU, `CHAR_CNN` may be used
as the fast order-aware fallback carrier. That fallback can produce an honest
negative or weak signal, but a full strong text-carrier claim still requires a
follow-up GRU or small LM run.

## Training Defaults

```text
train_examples: 1024
eval_examples: 1200
epochs: 220 full, 40 smoke
learning_rate: 0.003
hidden: 96
embedding_dim: 32
max_chars: 768
batch_size: 256
teacher_loss_weight: 1.0
consistency_loss_weight: 0.5
```

Determinism requirements:

```text
torch.set_num_threads(1)
fixed ASCII vocabulary
seed Python, NumPy, and Torch per job
CPU default
no generated target outputs committed
```

## Metrics

Report per model, format arm, and seed:

```text
answer_eval_ood_accuracy
shortcut_trap_rate
train_accuracy
format_char_count_mean
format_char_count_p95
format_token_count_mean
format_efficiency
same_format_shuffled_gap
answer_only_gap
train_surface_alignment
eval_surface_flip_rate
```

Valid stress requires:

```text
train_surface_alignment >= 0.85
eval_surface_flip_rate >= 0.85
valid_seeds >= 80 / 100 for a full result
```

A text format is `USEFUL` if:

```text
OOD accuracy >= ANSWER_ONLY + 0.20
OOD accuracy >= matching shuffled control + 0.20
shortcut_trap_rate <= 0.30
effect survives at least CHAR_GRU
eval input is TASK only
```

For a fallback CNN-only run, report the same gates against `CHAR_CNN` but label
the result as a fast text-carrier fallback, not the final GRU gate.

## Status Rules

```text
ANCHOR_MINI_007_TEXT_FORMAT_STRONG_SIGNAL
ANCHOR_MINI_007_TEXT_FORMAT_WEAK_SIGNAL
ANCHOR_MINI_007_TEXT_FORMAT_NEGATIVE
ANCHOR_MINI_007_TEXT_FORMAT_INVALID_STRESS
ANCHOR_MINI_007_TEXT_FORMAT_RESOURCE_BLOCKED
```

Strong signal requires:

```text
ACTION_OUTCOME_TABLE_PROCESS is useful
ACTION_OUTCOME_TABLE_PROCESS beats PROSE_PROCESS
ACTION_OUTCOME_TABLE_PROCESS beats INNER_MONOLOGUE_PROCESS
ACTION_OUTCOME_TABLE_PROCESS beats SHUFFLED_ACTION_OUTCOME_TABLE
COMPACT_HYBRID_PROCESS or RELATION_PLUS_ACTION_PROCESS is directionally positive
ANSWER_ONLY is pulled by the flipped shortcut and fails OOD
```

## Outputs

Generated outputs stay under:

```text
target/anchorweave/anchor_mini007_text_carrier/
```

Required generated files:

```text
queue.json
progress.jsonl
metrics.jsonl
summary.json
report.md
format_curve.json
contract_snapshot.md
```

If positive, commit only:

```text
docs/research/ANCHOR_MINI_007_TEXT_CARRIER_CONTRACT.md
tools/anchorweave/run_anchor_mini007_text_carrier.py
docs/research/ANCHOR_MINI_007_RESULT.md
```

Do not commit raw `target/` outputs.
