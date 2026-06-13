# E63 Pocket Observatory Visual Debug Dashboard

Status: completed.

## Decision

```text
decision = e63_pocket_observatory_dashboard_ready
pocket_count = 8
cycle_count = 12
false_commits = 0
```

## What It Provides

- Self-contained `index.html` dashboard.
- Pocket list with lifecycle, calls, accepted/rejected proposals, scores, and last activity.
- Cycle-by-pocket activity heatmap.
- Flow Field commit grid.
- Proposal/Agency timeline views.
- Relative artifact auto-refresh when served over HTTP.
- Embedded sample + file-picker fallback when opened directly.

## How To Open

```powershell
cd target/pilot_wave/e63_pocket_observatory_visual_debug_dashboard
python -m http.server 8763
# then open http://127.0.0.1:8763/index.html
```

## Boundary

E63 is a local visual/debug dashboard for Pocket ecology artifacts. It is not a new training run, model capability claim, production API, AGI claim, consciousness claim, or model-scale behavior claim.
