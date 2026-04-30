# Phase D10p Semantic Projection Hardening

Date: 2026-04-30

Verdict: `D10P_AUTOPILOT_SMOKE_READY`

## Summary

D10p adds the semantic control gate needed after D10o. The issue was not GPU
capacity or raw high-H signal; it was target-independent false positives under
shuffled labels/projections.

Implemented:

```text
tools/_scratch/d10p_semantic_projection_hardening.py
tools/_scratch/d10_agentic_autopilot.py
```

The autopilot runs resumable D10 phases, writes status/events/progress artifacts,
and monitors D10b without blocking on it.

## Smoke Result

Smoke command used reduced settings:

```text
autopilot max_phases: 1
phase: D10p_smoke
H: 8192
edges: 100000
eval_len: 64
eval_seeds: 988001,988002
proposals_per_arm: 2
arms: beta8_lifted_v2,motif_no_echo
```

Artifacts written:

```text
output/phase_d10_autopilot_20260430_smoke/status.json
output/phase_d10_autopilot_20260430_smoke/events.jsonl
output/phase_d10_autopilot_20260430_smoke/progress_map.md
output/phase_d10_autopilot_20260430_smoke/D10p_smoke/run_summary.json
```

Smoke verdict:

```text
AUTOPILOT_CYCLE_COMPLETE
D10p_smoke: D10P_SEMANTIC_FAIL
```

This is acceptable for smoke because the smoke is not rankable evidence. It
proves the control machinery works. In the smoke, `motif_no_echo` showed real
safe-positive signal, but failed because controls were stronger:

```text
real_safe_rate: 0.50
max_control_safe_rate: 1.00
control_adjusted_safe_rate: -0.50
```

No-op deltas were zero.

## Semantic Controls

D10p evaluates every candidate family under:

```text
real labels
random_label
random_bigram
unigram_decoy
projection_shuffle
```

Acceptance for an arm:

```text
SEMANTIC_PASS:
  real safe rate >= 25%
  max control safe rate <= 10%
  real - max_control >= 20%

SEMANTIC_FAIL:
  any control safe rate > 25%
```

GPU-only D10p output is not promotion evidence. Any pass only unlocks D10q
controlled confirm.

## Autopilot Behavior

The runner:

```text
resumes from status.json
writes append-only events.jsonl
writes progress_map.md
checks D10b run_summary.json each phase
does not commit generated output
does not release/tag/promote checkpoints
```

Default queue:

```text
D10p_smoke
D10p_scout_H4096_E25000
D10p_scout_H8192_E100000
D10p_scout_H16384_E100000
optional D10p_confirm_* only if a scout produces SEMANTIC_PASS
```

## Next Step

Run a real autopilot cycle:

```text
python tools/_scratch/d10_agentic_autopilot.py \
  --device cuda \
  --max-phases 4 \
  --out output/phase_d10_autopilot_20260430_main
```

If D10p produces `D10P_SEMANTIC_PASS`, the next phase is D10q controlled
high-H confirm. If D10p stays blocked, stop high-H scaling and redesign
projection/eval semantics.

## Progress Map

```text
GLOBAL AI PLAN

[1] H384 beta.8 generalist
    DONE

[2] mechanism
    DONE: edge + threshold co-adaptation

[3] H384 seed replication
    RUNNING: D10b CPU

[4] high-H raw signal
    DONE: D10k-D10o

[5] semantic trust
    CURRENT: D10p autopilot smoke ready

[6] controlled proof
    NEXT IF D10p PASSES: D10q

[7] final verdict
    UNIVERSAL / STRUCTURE_DEPENDENT / LOCAL_ONLY / SEMANTIC_BLOCKED
```
