# Nightly Results Tables

This folder contains the repo-local, curated summary tables for current `origin/nightly` experiments.

Tracked outputs:

- `runs_master.csv`
  - every detected `training_output/<run_root>/train_log.csv` + `ckpt_latest.pt` pair
  - deduped by `run_root`
  - includes build-spec fields and final/best metrics
- `derived/runs_golden.csv`
  - curated, user-facing subset
  - intended for source-of-truth comparisons and remote review
- `derived/runs_quarantined.csv`
  - scratch, tests, and uncurated runs kept for forensics but not promoted

Rebuild from the current local `training_output/` tree:

```powershell
Set-Location -LiteralPath 'S:\AI\_tmp\nightly_worktree\v4'
python 'tools/results_ingest.py'
```

Current golden categories:

- current-corpus CPU A/B
- long-run production checkpoint summary
- needle-task memory/pointer/slot ablations

Raw telemetry JSONs under `dev_notes/telemetry/` remain intentionally untracked scratch artifacts.
