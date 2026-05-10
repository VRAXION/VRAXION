# ANCHOR-MINI-012A: Neural Template-Scaling Parser Diagnostic

## Claim

MINI-012A tests whether MINI-011 template-transfer failure came from too little
template diversity or from the sparse/raw carrier being too position-bound.

Narrow claim:

```text
If a small neural character parser can learn PLAN-first behavior across held-out
serialization templates, then the next bottleneck is the sparse parser carrier,
not the PLAN-first training signal.
```

This is not a final AnchorCell proof and not natural-language grounding.

## Task

Reuse the MINI-011 shortcut-flip task:

```text
4 candidates
4 categories
train surface shortcut points to gold with p=0.90
eval surface shortcut points to a wrong candidate with p=0.90
```

Training sees:

```text
raw task text + PLAN supervision + answer target
```

Eval sees:

```text
raw task text only
```

No answer labels, match bits, policy bits, or PLAN fields may appear in eval
input text.

## Models

```text
CHAR_BOW_MLP
CHAR_CNN
CHAR_GRU
```

`CHAR_BOW_MLP` is a lexical shortcut control. `CHAR_CNN` is a fast local-pattern
parser. `CHAR_GRU` is the primary sequence parser baseline.

## Arms

```text
ANSWER_ONLY_DIRECT
  raw text -> answer only

AUX_PLAN_DIRECT
  raw text -> answer + PLAN auxiliary heads, final answer remains direct

PLAN_FIRST
  raw text -> predicted PLAN/policy -> answer

SHUFFLED_PLAN_FIRST
  PLAN-first route trained with shifted/wrong PLAN labels
```

## Template Scaling

```text
train_template_count: 1, 3, 8, 16, 32, 64
held_out_eval_template_count: 32
train_examples: 4096
eval_examples: 2048
```

Training size stays fixed so the main variable is template diversity.

## Metrics

Report per model, arm, template count, and seed:

```text
answer_eval_accuracy
shortcut_trap_rate
goal_category_accuracy
effect_category_accuracy
policy_bit_accuracy
plan_exact_row_accuracy
answer_from_plan_consistency
train_surface_alignment
eval_surface_flip_rate
heldout_template_accuracy
```

Valid stress requires:

```text
train_surface_alignment >= 0.85
eval_surface_flip_rate >= 0.85
feature_leak_audit == pass
```

## Diagnostic Outcomes

```text
TEMPLATE_DIVERSITY_HELPS
  PLAN_FIRST improves as train_template_count rises.

SPARSE_CARRIER_BOTTLENECK
  CHAR_GRU or CHAR_CNN PLAN_FIRST succeeds on held-out templates while MINI-011
  sparse transfer failed.

TASK_OR_FORMAT_PROBLEM
  PLAN_FIRST fails even with 64 templates on CHAR_GRU.

LEXICAL_SHORTCUT_WARNING
  CHAR_BOW_MLP succeeds strongly too, weakening parser-specific claims.
```

Sequence-model success threshold:

```text
PLAN_FIRST answer_eval_accuracy >= 0.80
PLAN_FIRST shortcut_trap_rate <= 0.25
PLAN_FIRST beats ANSWER_ONLY_DIRECT by >= 0.25
PLAN_FIRST beats SHUFFLED_PLAN_FIRST by >= 0.25
```

## Run

Smoke:

```bash
python tools/anchorweave/run_anchor_mini012_neural_template_scaling.py ^
  --out target/anchorweave/anchor_mini012_neural_template_scaling/smoke ^
  --seeds 2026 ^
  --models CHAR_BOW_MLP,CHAR_CNN ^
  --arms ANSWER_ONLY_DIRECT,PLAN_FIRST,SHUFFLED_PLAN_FIRST ^
  --template-counts 1,8 ^
  --train-examples 512 ^
  --eval-examples 512 ^
  --epochs 10 ^
  --jobs 2
```

Fast diagnostic:

```bash
python tools/anchorweave/run_anchor_mini012_neural_template_scaling.py ^
  --out target/anchorweave/anchor_mini012_neural_template_scaling/fast_2026_05_10 ^
  --seeds 2026-2030 ^
  --models CHAR_BOW_MLP,CHAR_CNN,CHAR_GRU ^
  --arms ANSWER_ONLY_DIRECT,AUX_PLAN_DIRECT,PLAN_FIRST,SHUFFLED_PLAN_FIRST ^
  --template-counts 1,3,8,16,32,64 ^
  --train-examples 4096 ^
  --eval-examples 2048 ^
  --eval-template-count 32 ^
  --epochs 60 ^
  --jobs 8 ^
  --budget-minutes 45
```

Generated outputs stay under `target/`. Do not write under
`data/anchorweave/cells/`.
