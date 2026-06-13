# E63 Pocket Observatory Visual Debug Dashboard Result

Status: completed.

## Decision

```text
decision = e63_pocket_observatory_dashboard_ready
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
browser_render = pass
```

## What Was Added

```text
scripts/probes/run_e63_pocket_observatory_visual_debug_dashboard.py
scripts/probes/run_e63_pocket_observatory_visual_debug_dashboard_check.py
docs/research/E63_POCKET_OBSERVATORY_VISUAL_DEBUG_DASHBOARD_CONTRACT.md
docs/research/E63_POCKET_OBSERVATORY_VISUAL_DEBUG_DASHBOARD_RESULT.md
docs/research/artifact_samples/e63_pocket_observatory_visual_debug_dashboard/
```

The dashboard writes a self-contained `index.html` plus JSON/JSONL artifacts:

```text
Pocket list
Pocket lifecycle/status metrics
cycle-by-pocket activity heatmap
Flow Field commit grid
Proposal/Agency timeline
selected Pocket detail pane
manual refresh
auto-refresh
embedded sample fallback
file-picker fallback
relative artifact fetch when served over HTTP
```

## Target Run

```text
out = target/pilot_wave/e63_pocket_observatory_visual_debug_dashboard
pocket_count = 8
cycle_count = 12
false_commits = 0
```

## Validation

```powershell
python -m py_compile scripts/probes/run_e63_pocket_observatory_visual_debug_dashboard.py scripts/probes/run_e63_pocket_observatory_visual_debug_dashboard_check.py
python scripts/probes/run_e63_pocket_observatory_visual_debug_dashboard.py --out target/pilot_wave/e63_pocket_observatory_visual_debug_dashboard --artifact-sample-dir docs/research/artifact_samples/e63_pocket_observatory_visual_debug_dashboard
python scripts/probes/run_e63_pocket_observatory_visual_debug_dashboard_check.py --out target/pilot_wave/e63_pocket_observatory_visual_debug_dashboard --write-summary
python scripts/probes/run_e63_pocket_observatory_visual_debug_dashboard_check.py --sample-only docs/research/artifact_samples/e63_pocket_observatory_visual_debug_dashboard --write-summary
```

Result:

```text
py_compile = pass
target checker = pass, failure_count 0
sample-only checker = pass, failure_count 0
deterministic replay hash check = pass
```

Browser/render validation:

```powershell
python -m http.server 8763 --bind 127.0.0.1 --directory target/pilot_wave/e63_pocket_observatory_visual_debug_dashboard
npx playwright screenshot --wait-for-timeout=1500 http://127.0.0.1:8763/index.html target/pilot_wave/e63_pocket_observatory_visual_debug_dashboard/browser_render.png
```

Result:

```text
HTTP status = 200
Playwright screenshot = pass
visible UI = Pocket list + Activity Heatmap + Flow Field Commit Grid + Selected Pocket panel
source mode shown by UI = live relative artifacts
```

## How To Open

```powershell
cd target/pilot_wave/e63_pocket_observatory_visual_debug_dashboard
python -m http.server 8763
```

Then open:

```text
http://127.0.0.1:8763/index.html
```

If opened directly as a local file, the dashboard still has an embedded sample
fallback and a file-picker fallback, but live auto-refresh of relative artifacts
works best through a local HTTP server.

## Boundary

E63 is a local visualization/debug tool. It does not train or mutate Pockets,
does not claim new model capability, and does not claim production dashboard/API
readiness.
