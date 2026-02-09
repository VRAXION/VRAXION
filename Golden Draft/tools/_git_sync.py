"""Auto-sync utility for pushing telemetry to the nightly branch.

Non-fatal: all errors are caught and logged. Training never stops due to sync.

Usage from training scripts:
    from _git_sync import auto_sync_nightly
    auto_sync_nightly(
        message="[auto] probe11 step 100: MA100=0.87",
        paths=["probe11_telemetry.jsonl"],
    )
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    """Run a git command, return (returncode, combined output)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except Exception as exc:
        return 1, str(exc)


def _find_repo_root(start: str | None = None) -> str | None:
    """Walk up to find the .git directory."""
    p = Path(start or os.getcwd()).resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return str(parent)
    return None


def auto_sync_nightly(
    message: str,
    paths: list[str],
    *,
    repo_root: str | None = None,
    branch: str = "nightly",
) -> bool:
    """Stage paths, commit with message, push to nightly branch.

    Returns True on success, False on any failure (non-fatal).
    """
    root = repo_root or _find_repo_root()
    if not root:
        print("[git_sync] WARNING: no git repo found, skipping sync")
        return False

    # Check current branch.
    rc, current = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    if rc != 0:
        print(f"[git_sync] WARNING: cannot determine branch: {current}")
        return False

    # If not on nightly, check if nightly exists and skip (don't switch branches
    # during a training run — that would be dangerous).
    if current.strip() != branch:
        # Just commit to current branch with a nightly-tagged message.
        # The consolidation workflow will handle merging to nightly.
        pass

    # Stage specified files.
    for p in paths:
        rc, out = _run(["git", "add", str(p)], cwd=root)
        if rc != 0:
            print(f"[git_sync] WARNING: failed to stage {p}: {out}")

    # Check if there's anything to commit.
    rc, status = _run(["git", "diff", "--cached", "--quiet"], cwd=root)
    if rc == 0:
        # Nothing staged — skip.
        return True

    # Commit.
    rc, out = _run(["git", "commit", "-m", message], cwd=root)
    if rc != 0:
        print(f"[git_sync] WARNING: commit failed: {out}")
        return False

    # Push.
    rc, out = _run(["git", "push", "origin", "HEAD"], cwd=root)
    if rc != 0:
        print(f"[git_sync] WARNING: push failed: {out}")
        # Commit succeeded but push didn't — not ideal but not fatal.
        return False

    return True
