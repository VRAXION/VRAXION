# ANCHOR-MINI-003 Result

Status: `ANCHOR_MINI_003_STRONG_POSITIVE`

Run:

```bash
python tools/anchorweave/run_anchor_mini003.py \
  --out target/anchorweave/anchor_mini003/final \
  --seeds 2026,2027,2028,2029,2030
```

## Summary

`ANCHOR-MINI-003` tested whether decomposed AnchorCell-style auxiliary
supervision improves OOD trap resistance when a surface shortcut flips between
train and eval.

All arms received identical input features. Only the training targets differed:

```text
ANSWER_ONLY
ANCHOR_MULTI_TASK
SHUFFLED_ANCHOR_MULTI_TASK
```

The stress was valid: the train surface shortcut usually pointed to gold, while
the eval surface shortcut usually pointed to a wrong candidate. The answer-only
arm learned that shortcut and failed OOD. The anchor arm learned the latent
goal/effect category structure and resisted the flipped shortcut. The shuffled
auxiliary control did not reproduce the improvement.

## Aggregate Metrics

| arm | answer_eval_ood | goal_eval | effect_eval | shortcut_trap |
|---|---:|---:|---:|---:|
| `ANSWER_ONLY` | 0.132 | 0.252 | 0.245 | 0.878 |
| `ANCHOR_MULTI_TASK` | 1.000 | 1.000 | 1.000 | 0.000 |
| `SHUFFLED_ANCHOR_MULTI_TASK` | 0.179 | 0.271 | 0.235 | 0.644 |

## Per-Seed Summary

| seed | answer_only | anchor | shuffled | base_trap | anchor_trap | train_align | eval_flip |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 0.138 | 1.000 | 0.176 | 0.884 | 0.000 | 0.923 | 0.887 |
| 2027 | 0.130 | 1.000 | 0.168 | 0.889 | 0.000 | 0.908 | 0.892 |
| 2028 | 0.119 | 1.000 | 0.168 | 0.885 | 0.000 | 0.918 | 0.909 |
| 2029 | 0.143 | 1.000 | 0.183 | 0.857 | 0.000 | 0.889 | 0.907 |
| 2030 | 0.131 | 1.000 | 0.201 | 0.875 | 0.000 | 0.908 | 0.896 |

## Interpretation

This is deterministic toy-level evidence that decomposed anchor supervision can
improve shortcut-resistant OOD generalization over answer-only training while
keeping input features identical.

It does not prove Qwen LoRA behavior, VRAXION architecture advantage,
natural-language AnchorCells, or grounding at scale.

Next recommended gate: run an architecture/carrier A/B using the same MINI-003
task shape across a tiny MLP, a small causal LM, and a VRAXION Nano-style
carrier.
