# STABLE_LOOP_PHASE_LOCK_055 Visual Demo README

## What To Open

Start the isolated visual lab:

```powershell
cd tools/visual_lab
npm run dev
```

Open:

```text
http://127.0.0.1:5173/
```

Select bundle:

```text
055_real_run_replay_closure
```

This is the default bundle after 055. The same selector still exposes:

```text
052_smoke_minimal
053_real_run_ingest
054_larger_playback_smoke
055_real_run_replay_closure
```

## What Each Page Shows

Topology shows the visual projection graph for the selected checkpoint. In the
055 bundle it highlights the source-to-target route, the diagnostic label
pocket, failure-control pockets, pruned shortcut controls, and output entropy
node.

Playback shows checkpoint and tick replay. For 055, checkpoint 100 has tick
snapshots that show the passing route projection and rollback-gated closure
view.

Diff compares checkpoint states. Use first-to-selected to see the transition
from checkpoint 000 collapse controls to checkpoint 100 passing replay. Use
previous-to-selected to inspect the 050 to 100 closure change.

Metrics shows the 049/050-derived values rendered through the visual lab:
heldout, OOD, family-min, hard distractor, long-OOD, unique output count,
top-output concentration, majority-output concentration, output entropy, and
collapse state.

## Why The Checkpoints Exist

checkpoint 000 represents:

```text
NO_ROUTE_GRAMMAR_ADVERSARIAL_FROZEN_BASELINE
```

It is the collapsed no-route baseline: low heldout/OOD, family-min zero, one
unique output, top output rate 1.000, collapse true.

checkpoint 050 represents:

```text
FROZEN_EVAL_048_REFERENCE
```

It is the partial/collapsed reference: route-answer behavior is visible, but
family-min remains zero and output concentration is still high.

checkpoint 100 represents:

```text
ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER
```

It is the passing 049 arm: heldout/OOD/family-min/hard/long-OOD all pass,
unique output coverage is 75/75, output entropy is high, and collapse is false.

## Source Metrics

The values come from bounded 049/050 adversarial frozen-eval evidence. The
visual bundle preserves exact source float values, including:

```text
checkpoint 100 top_output_rate = 0.0732421875
checkpoint 100 majority_output_rate = 0.0546875
checkpoint 100 output_entropy = 5.40437231483324
```

Docs may round `top_output_rate` to `0.073` for readability; the closure checker
uses the exact value with epsilon <= 1e-9.

## Why This Is Not A New Model Result

055 does not run new training and does not introduce a new model capability. It
packages existing 049/050 evidence as a real-metric visual projection so the
visual lab can replay, inspect, and audit the result.

In short: 055 is not a new model result.

The 049 run did not emit raw internal topology snapshots. Therefore 055 is not
raw internal topology and not a raw model graph capture.

## Boundary

055 supports Visual V1 closure, schema-first visual lab replay, tiny sample,
larger playback, real-result replay, metric alignment against bounded 049/050
evidence, demo docs, and static closure checking.

055 is not production dashboard readiness, not production API readiness, not
public beta promotion, not a new model capability, not raw internal model graph
capture, not full VRAXION, not language grounding, not biological/FlyWire
equivalence, not physical quantum behavior, and not consciousness.
