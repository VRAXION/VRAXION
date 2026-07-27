# Ingest Prompt For Future Agent

Use this prompt when opening the repo later from GitHub or a Steam Deck.

```text
You are continuing the VRAXION 2026-05-26 night marathon.

First read:
docs/research/VRAXION_NIGHT_MARATHON_2026_05_26_INFODUMP/README.md
docs/research/VRAXION_NIGHT_MARATHON_2026_05_26_INFODUMP/CODE_AND_COMMANDS.md
docs/research/VRAXION_NIGHT_MARATHON_2026_05_26_INFODUMP/ARTIFACT_INDEX.json

Then inspect:
instnct-core/examples/raven_pocket_smoke.rs

Do not assume the full Raven task is solved.
The known strong result is only:
clean WANTED_POCKET=P# -> selected pocket P# reached 100% train/test with direct VRAXION mutation.

The known weak points are:
1. current DNA/genome GA search plateaued around 52-56% on the same clean selector task,
2. symbol equality is weak or not yet fully tested under direct mutation,
3. extra grid/pocket/noise lamps break routing.

Continue by running the direct equality test to completion, with queue.json/progress.jsonl/summary.json writeout.
Do not start a black-box run.
Do not mix unrelated AnchorWeave dirty files into the VRAXION experiment commit.
```

