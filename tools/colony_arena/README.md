# VRAXION Colony Arena

Small visual arena for testing evolved behavior policies with the D51 mutable
rule-table controller as an evidence gate.

This is intentionally not a Screeps integration yet. It is a local, deterministic
Screeps-like tick/intent lab: the policy reads a compact world state, the D51
gate decides whether to act directly or spend extra evidence, the world applies
the movement intent, and the trainer scores survival, goal collection, route
quality, wall hits, threat avoidance, and information cost.

## Run

```powershell
cd S:\Git\VRAXION\tools\colony_arena
npm run smoke
npm run smoke:fast
python -m http.server 5173
```

Then open:

```text
http://localhost:5173
```

## Artifacts

The smoke trainer writes append-only progress under:

```text
S:\Git\VRAXION\target\colony_arena_smoke\<run_id>\
```

Core files:

- `queue.json`
- `progress.jsonl`
- `summary.json`
- `best_policy.json`
- `sample_replay.json`
- `adversarial_replay.json`
- `report.md`

Current smoke gates include normal/mixed test success, adversarial success,
caught-rate ceilings, and a minimum D51 non-DECIDE evidence-action rate.
