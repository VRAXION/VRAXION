<!-- Canonical source for the mirrored GitHub wiki page. Sync with tools/sync_wiki_from_repo.py. -->

# Engineering Protocol

This page defines the run contract behind VRAXION results: how experiments are described, checked, and accepted as public evidence.

Audience: contributors, evaluators, and readers who need reproducible proof rather than qualitative claims. Docs-source and sync rules live in [`CONTRIBUTING.md`](https://github.com/VRAXION/VRAXION/blob/main/CONTRIBUTING.md).

## At a Glance

- Every meaningful run needs an objective metric, one budget mode, hard fail gates, and a required artifact bundle.
- The minimum evidence bundle is `run_cmd.txt`, `env.json`, `metrics.json`, and `summary.md`.
- A **Current mainline** claim must match code on `main`; everything else belongs under **Validated finding** or **Experimental branch**.

## Use This Page When

- you need to know what counts as evidence
- you need the run bundle and fail gates before trusting a result
- you need the promotion boundary between mainline, validated, and experimental work

## Run Contract

Every meaningful run must declare:

- an objective metric
- one budget mode
- hard fail gates
- a required artifact bundle

Results outside that contract can still be useful probes, but they should not be promoted as canonical evidence.

## Required Evidence

| Artifact | Purpose |
|---|---|
| `run_cmd.txt` | Exact command and flags used for the run |
| `env.json` | Environment snapshot: OS, GPU/runtime, Python, package versions |
| `metrics.json` | Time series and summary metrics for the run |
| `summary.md` | Human verdict, including PASS/FAIL and the reason |

Optional extras such as checkpoints, plots, CSV exports, or live logs are useful, but they do not replace the core evidence bundle.

## Fail Gates

| Gate | Trigger |
|---|---|
| OOM / runtime failure | Any out-of-memory or driver/runtime failure invalidates the run |
| NaN / Inf | Any NaN or Inf in tracked metrics invalidates the run |
| Step-time explosion | `p95(step_time) > 2.5 × median(step_time)` |
| Heartbeat stall | No progress after warmup for `max(60s, 10 × median step time)` |
| VRAM guard breach | Peak reserved VRAM exceeds `0.92 × total VRAM` |

These gates apply to probes, sweeps, and training runs alike.

## Sweep Discipline

- Choose exactly one budget mode per sweep: `iso-VRAM`, `iso-params`, or `iso-FLOPs/step`.
- Run the systems curve first: throughput, stability, step-time tails, and resource limits.
- Only run the quality curve after the systems curve is stable.
- Start with a coarse sweep, then rerun the best cells with multiple seeds under the same contract.
- If a result does not reproduce under the same contract, treat it as unconfirmed.

## How This Connects to Public Truth

- **Current mainline** means the setting is actually shipped in code on `main`.
- **Validated finding** means the result is reproducible, but not yet promoted into the canonical code path.
- **Experimental branch** means the direction is active, but should not be described as a default.

Use those labels consistently across the repo front door, the architecture page, and the findings page. If code and docs disagree about **Current mainline**, the code wins.

## Read Next

- [VRAXION Home](Home)
- [INSTNCT Architecture](INSTNCT-Architecture)
- [Validated Findings](Validated-Findings)
- [Project Timeline](Release-Notes)
