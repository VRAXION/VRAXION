# ANCHOR-MINI-011 Learned Raw-Byte PLAN Parser Contract

## Purpose

ANCHOR-MINI-011 removes the MINI-010 schema-aware parser from the
non-oracle runtime path.

The narrow claim is:

```text
Raw task bytes can train a PLAN-first sparse carrier to build internal
process/policy state and resist shortcut-flip eval better than direct answer,
aux-only PLAN, shuffled teacher, and shortcut teacher controls.
```

This is not a natural-language AnchorCell test. It is the byte parser gate:

```text
training: raw bytes + PLAN supervision + answer target
eval: raw bytes only -> internal PLAN -> answer
```

## Dataset

Reuse the MINI-010 shortcut-flip setup:

```text
candidate_count: 4
category_count: 4
train shortcut: highest surface_prior points to gold with probability 0.90
eval shortcut: highest surface_prior points to wrong candidate with probability 0.90
seeds: 2026-2125
```

Stages:

```text
same_template_raw
  train: canonical_fixed
  eval: canonical_fixed

template_transfer_raw
  train: canonical_fixed + field_order_swap + noise_fields
  eval: slot_order_perm + alias_long
```

Rules:

```text
No answer label in task_bytes
No match bit in task_bytes
No policy bit in task_bytes
No oracle PLAN bit in task_bytes for non-oracle carriers
No schema-aware decoded goal/effect/surface fields as non-oracle runtime input
```

Allowed non-oracle runtime features:

```text
raw byte/digit at absolute position
candidate index one-hot
raw absolute-position goal/effect equality gates
raw absolute-position shifted-goal/effect equality gates
bias
```

Teacher labels may be generated from dataset metadata for training and metrics,
but must not be exposed as input features.

## Carrier Arms

```text
RAW_DIRECT_ANSWER
  Direct answer from raw bytes. Expected to shortcut.

RAW_AUX_PLAN_DIRECT_ANSWER
  PLAN fields are trained as auxiliary outputs, but final answer remains direct.

RAW_PLAN_FIRST
  Primary candidate. Final answer routes through predicted PLAN/policy state.

RAW_PLAN_FIRST_HYBRID
  PLAN route plus direct bypass.

RAW_SHUFFLED_TEACHER
  PLAN-first route with wrong shifted PLAN labels.

RAW_SHORTCUT_TEACHER
  Teacher praises the surface shortcut.

RAW_ORACLE_DECODED_PLAN_VISIBLE
  Upper-bound diagnostic only; decoded PLAN is visible.
```

PLAN fields:

```text
goal_category
candidate_effect_category
surface_shortcut
shortcut_valid
invalid_shortcut_rejection
candidate_policy_bits
answer_from_plan
```

## Metrics

Report per carrier, seed, and stage:

```text
answer_eval_accuracy
shortcut_trap_rate
goal_category_accuracy
effect_category_accuracy
surface_shortcut_accuracy
shortcut_validity_accuracy
invalid_shortcut_rejection_accuracy
policy_bit_accuracy
plan_exact_row_accuracy
answer_from_plan_consistency
train_surface_alignment
eval_surface_flip_rate
raw_input_integrity
feature_leak_audit
edge_count
```

Valid stress:

```text
train_surface_alignment >= 0.85
eval_surface_flip_rate >= 0.85
raw_input_integrity == true
feature_leak_audit == pass
valid_seeds >= 80 / 100
```

Strong pass per stage:

```text
RAW_DIRECT_ANSWER shortcut_trap_rate >= 0.45
RAW_PLAN_FIRST answer_eval_accuracy >= RAW_DIRECT_ANSWER + 0.25
RAW_PLAN_FIRST answer_eval_accuracy >= RAW_SHUFFLED_TEACHER + 0.25
RAW_PLAN_FIRST answer_eval_accuracy >= RAW_SHORTCUT_TEACHER + 0.25
RAW_PLAN_FIRST shortcut_trap_rate <= 0.25
RAW_PLAN_FIRST goal_category_accuracy >= 0.90
RAW_PLAN_FIRST effect_category_accuracy >= 0.90
RAW_PLAN_FIRST policy_bit_accuracy >= 0.90
RAW_PLAN_FIRST plan_exact_row_accuracy >= 0.85
RAW_PLAN_FIRST answer_from_plan_consistency >= 0.95
RAW_AUX_PLAN_DIRECT_ANSWER does not match RAW_PLAN_FIRST unless answer_from_plan_consistency proves routing
RAW_PLAN_FIRST_HYBRID remains directionally positive
RAW_ORACLE_DECODED_PLAN_VISIBLE is an upper bound or matched
```

Statuses:

```text
ANCHOR_MINI_011_RAW_BYTE_STRONG_POSITIVE
  same_template_raw and template_transfer_raw both pass.

ANCHOR_MINI_011_RAW_BYTE_FIXED_ONLY
  same_template_raw passes, template_transfer_raw fails.

ANCHOR_MINI_011_RAW_BYTE_WEAK_POSITIVE
  main effect is positive but a secondary PLAN metric misses gate.

ANCHOR_MINI_011_RAW_BYTE_NEGATIVE
  RAW_PLAN_FIRST does not beat direct/shuffled/shortcut controls.

ANCHOR_MINI_011_INVALID_STRESS
  shortcut-flip stress is invalid.

ANCHOR_MINI_011_RESOURCE_BLOCKED
  run could not complete.
```

## Commands

Smoke:

```bash
cargo build --release -p instnct-core --example evolve_anchor_mini011

python tools/anchorweave/run_anchor_mini011_learned_byte_plan_parser.py ^
  --out target/anchorweave/anchor_mini011_learned_byte_plan_parser/smoke ^
  --seeds 2026,2027 ^
  --jobs 4 ^
  --max-steps 400 ^
  --stages same_template_raw ^
  --carriers RAW_DIRECT_ANSWER,RAW_PLAN_FIRST,RAW_SHUFFLED_TEACHER,RAW_ORACLE_DECODED_PLAN_VISIBLE
```

Full:

```bash
python tools/anchorweave/run_anchor_mini011_learned_byte_plan_parser.py ^
  --out target/anchorweave/anchor_mini011_learned_byte_plan_parser/full_2026_05_10 ^
  --seeds 2026-2125 ^
  --jobs 24 ^
  --budget-hours 8
```

Required generated outputs:

```text
queue.json
progress.jsonl
metrics.jsonl
summary.json
report.md
stage_curve.json
contract_snapshot.md
examples_sample.jsonl
jobs/<stage_carrier_seed>/report.json
```

## Claim Boundary

MINI-011 tests parserless raw-byte PLAN learning in a toy sparse carrier. It
does not prove natural-language AnchorCells, Qwen behavior, full INSTNCT
recurrent behavior, or symbol grounding at scale.

Generated outputs stay under:

```text
target/anchorweave/anchor_mini011_learned_byte_plan_parser/
```

Do not write under:

```text
data/anchorweave/cells/
```
