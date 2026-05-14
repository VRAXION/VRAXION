# ANCHOR-MINI-011 Result

## Verdict

```text
ANCHOR_MINI_011_RAW_BYTE_WEAK_POSITIVE
```

ANCHOR-MINI-011 tested whether PLAN-first sparse routing survives when
schema-aware decoded runtime fields are removed and the carrier sees raw
byte records only.

## Run

```bash
C:\Users\kenes\AppData\Local\Programs\Python\Python311\python.exe S:\Git\VRAXION_anchorwiki\tools\anchorweave\run_anchor_mini011_learned_byte_plan_parser.py --out target/anchorweave/anchor_mini011_learned_byte_plan_parser/full_2026_05_10 --seeds 2026-2125 --jobs 24 --budget-hours 8 --skip-build
```

Runtime:

```text
4934.63 seconds
```

## Stage: same_template_raw

| carrier | answer_eval | trap_rate | goal | effect | policy_bit | plan_exact | consistency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `RAW_DIRECT_ANSWER` | 0.100 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `RAW_AUX_PLAN_DIRECT_ANSWER` | 0.100 | 1.000 | 0.250 | 0.793 | 0.899 | 0.047 | 0.144 |
| `RAW_PLAN_FIRST` | 0.963 | 0.015 | 1.000 | 1.000 | 0.944 | 0.786 | 1.000 |
| `RAW_PLAN_FIRST_HYBRID` | 0.908 | 0.052 | 0.250 | 0.811 | 0.909 | 0.057 | 0.890 |
| `RAW_SHUFFLED_TEACHER` | 0.003 | 0.332 | 1.000 | 1.000 | 0.513 | 0.000 | 1.000 |
| `RAW_SHORTCUT_TEACHER` | 0.100 | 1.000 | 1.000 | 1.000 | 0.549 | 0.099 | 1.000 |
| `RAW_ORACLE_DECODED_PLAN_VISIBLE` | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Stage: template_transfer_raw

| carrier | answer_eval | trap_rate | goal | effect | policy_bit | plan_exact | consistency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `RAW_DIRECT_ANSWER` | 0.279 | 0.090 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `RAW_AUX_PLAN_DIRECT_ANSWER` | 0.277 | 0.097 | 0.610 | 0.177 | 0.723 | 0.000 | 0.242 |
| `RAW_PLAN_FIRST` | 0.224 | 0.261 | 0.625 | 0.142 | 0.715 | 0.000 | 1.000 |
| `RAW_PLAN_FIRST_HYBRID` | 0.243 | 0.162 | 0.621 | 0.176 | 0.707 | 0.000 | 0.404 |
| `RAW_SHUFFLED_TEACHER` | 0.265 | 0.254 | 0.625 | 0.140 | 0.723 | 0.000 | 1.000 |
| `RAW_SHORTCUT_TEACHER` | 0.278 | 0.097 | 0.625 | 0.153 | 0.581 | 0.000 | 1.000 |
| `RAW_ORACLE_DECODED_PLAN_VISIBLE` | 1.000 | 0.000 | 0.625 | 0.142 | 1.000 | 0.000 | 1.000 |

## Interpretation

MINI-011 gives a useful but bounded positive result:

```text
same_template_raw:
  RAW_PLAN_FIRST learned a raw-byte fixed-layout PLAN route.
  Direct and aux-direct remained shortcut-bound.

template_transfer_raw:
  The raw absolute-position carrier did not learn a format-invariant parser.
```

The result is `WEAK_POSITIVE` rather than `STRONG_POSITIVE` because
`same_template_raw` passed the main decision behavior but missed the strict
`plan_exact_row >= 0.85` gate, and `template_transfer_raw` failed.

## Claim Boundary

This is a toy raw-byte parser result. It does not prove natural-language
AnchorCells, Qwen behavior, or symbol grounding at scale.
