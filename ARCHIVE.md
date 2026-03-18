# Archive Policy

This repository keeps the default branch intentionally small.

## What stays on `main`

Only the active self-wiring graph line belongs on `main`:

- `v4.2/model/graph.py`
- `v4.2/model/graph_v3.c`
- `v4.2/lib/`
- `v4.2/tests/` benchmark and stress harnesses that still support the current line
- public documentation for the current self-wiring direction

## What does not stay on `main`

The following are archived instead of kept in the active branch list:

- superseded repo eras
- half-ready research detours
- ephemeral review branches
- one-off experimental integration branches

## Active archive branches

These remain as branches because they represent older repository eras that are still worth browsing directly:

- `archive/pre-self-wiring-main-20260317`
- `archive/pre-self-wiring-claude-20260317`

## Snapshot tags

Short-lived or now-removed branch tips are preserved as tags instead of keeping more long-lived remote branches around.

Current archive tags:

- `archive/nightly-20260318`
- `archive/v4.1-20260318`
- `archive/v4.2-20260318`
- `archive/review-main-branch-47XVJ-20260318`
- `archive/test-capacitor-neuron-config-EacK8-20260318`

## Why this policy exists

A clean branch surface matters.

Benefits:

- fewer misleading "compare / pull request" prompts
- clearer public signal about what the repo is actually about
- easier onboarding for readers who only need the active self-wiring path
- safer preservation of older work without mixing it into the active default branch

## Practical rule

If a line is not part of the current self-wiring mainline doctrine, archive it.
If it still matters historically, keep a tag or an archive branch.
If it neither matters historically nor supports the current line, delete it.
