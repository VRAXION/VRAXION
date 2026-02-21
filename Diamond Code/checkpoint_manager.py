"""
Checkpoint management for VRAXION training pipeline.

Manages a draft ring (rotating autosaves) + golden saves (permanent milestones):
  - Drafts: saved every checkpoint_every steps, oldest pruned beyond max_drafts
  - Golden: manually promoted or auto-promoted on mastery, never auto-deleted
  - Auto-resume: training resumes from latest draft by default
  - Manifest: JSON database tracking step, loss, bit_acc, dataset, tags

Directory structure:
    checkpoints/<run>/
        drafts/            <-- rotating ring
        golden/            <-- permanent saves
        checkpoint_latest.pt  <-- copy of newest draft (backwards compat)
        manifest.json      <-- metadata for all checkpoints

Usage:
    import checkpoint_manager

    # Save a draft (every N steps)
    checkpoint_manager.save_draft(model, optimizer, step, loss, bit_acc, ckpt_dir)

    # Auto-resume on launch
    start_step, _ = checkpoint_manager.load_latest(ckpt_dir, model, optimizer)

    # Promote to golden on mastery
    checkpoint_manager.promote_to_golden(ckpt_dir, step, tags=["mastered_echo256"])
"""

import json
import shutil
import time
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 2


# ── Manifest I/O ─────────────────────────────────────────────────────────────

def _empty_manifest() -> dict:
    return {"version": MANIFEST_VERSION, "checkpoints": {}, "latest_draft": None}


def _rebuild_manifest(checkpoint_dir: str) -> dict:
    """Rebuild manifest by scanning drafts/ and golden/ directories."""
    root = Path(checkpoint_dir)
    manifest = _empty_manifest()

    for subdir, ckpt_type in [("drafts", "draft"), ("golden", "golden")]:
        d = root / subdir
        if not d.exists():
            continue
        for f in sorted(d.glob("*.pt")):
            rel = f"{subdir}/{f.name}"
            step = 0
            try:
                ck = torch.load(f, map_location="cpu", weights_only=False)
                step = ck.get("step", 0)
            except Exception:
                pass
            manifest["checkpoints"][rel] = {
                "type": ckpt_type,
                "step": step,
                "loss": 0.0,
                "bit_acc": 0.0,
                "dataset": "",
                "timestamp": "unknown",
                "tags": [],
            }

    # Set latest_draft to highest-step draft
    drafts = [(k, v) for k, v in manifest["checkpoints"].items() if v["type"] == "draft"]
    if drafts:
        drafts.sort(key=lambda kv: kv[1]["step"], reverse=True)
        manifest["latest_draft"] = drafts[0][0]

    n = len(manifest["checkpoints"])
    if n > 0:
        print(f"  [CKPT] Manifest rebuilt from directory: {n} checkpoints")
    return manifest


def _clean_stale_entries(checkpoint_dir: str, manifest: dict) -> dict:
    """Remove manifest entries for files that no longer exist on disk."""
    root = Path(checkpoint_dir)
    stale = [k for k in manifest["checkpoints"] if not (root / k).exists()]
    for k in stale:
        print(f"  [CKPT] WARNING: removing stale manifest entry: {k}")
        del manifest["checkpoints"][k]
    if stale and manifest.get("latest_draft") in stale:
        # Recalculate latest_draft
        drafts = [(k, v) for k, v in manifest["checkpoints"].items() if v["type"] == "draft"]
        if drafts:
            drafts.sort(key=lambda kv: kv[1]["step"], reverse=True)
            manifest["latest_draft"] = drafts[0][0]
        else:
            manifest["latest_draft"] = None
    return manifest


def load_manifest(checkpoint_dir: str) -> dict:
    """Load manifest.json. Rebuild from directory listing if corrupt or missing."""
    manifest_path = Path(checkpoint_dir) / MANIFEST_NAME

    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            if not isinstance(manifest.get("checkpoints"), dict):
                raise ValueError("malformed manifest")
        except (json.JSONDecodeError, ValueError, KeyError):
            manifest = _rebuild_manifest(checkpoint_dir)
    else:
        root = Path(checkpoint_dir)
        has_files = (
            (root / "drafts").exists() and any((root / "drafts").glob("*.pt"))
        ) or (
            (root / "golden").exists() and any((root / "golden").glob("*.pt"))
        )
        if has_files:
            manifest = _rebuild_manifest(checkpoint_dir)
        else:
            manifest = _empty_manifest()

    manifest = _clean_stale_entries(checkpoint_dir, manifest)
    return manifest


def save_manifest(checkpoint_dir: str, manifest: dict):
    """Atomically write manifest.json (write to .tmp then rename)."""
    manifest_path = Path(checkpoint_dir) / MANIFEST_NAME
    tmp_path = manifest_path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(manifest, f, indent=2)
        if manifest_path.exists():
            manifest_path.unlink()
        tmp_path.rename(manifest_path)
    except IOError as e:
        print(f"  [CKPT] ERROR: failed to save manifest: {e}")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except IOError:
                pass


# ── Checkpoint dict builder ──────────────────────────────────────────────────

def _build_checkpoint_dict(model, optimizer, step: int, config: dict = None) -> dict:
    """Build the checkpoint dict (same fields as existing save_checkpoint)."""
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_accuracy": 0.0,
        "num_bits": getattr(model, "num_bits", 8),
        "bits_per_being": getattr(model, "bits_per_being", 0),
        "min_coverage": getattr(model, "min_coverage", 0),
    }
    if config is not None:
        checkpoint["config"] = config
    if hasattr(model, "being_states"):
        checkpoint["being_states"] = model.being_states.copy()
    return checkpoint


# ── Save draft ───────────────────────────────────────────────────────────────

def save_draft(
    model,
    optimizer,
    step: int,
    loss: float,
    bit_acc: float,
    checkpoint_dir: str,
    dataset: str = None,
    config: dict = None,
    max_drafts: int = 15,
) -> Optional[str]:
    """Save a draft checkpoint to drafts/ subdir.

    Updates manifest, prunes oldest drafts, updates checkpoint_latest.pt.
    Returns path to saved file, or None on error.
    """
    root = Path(checkpoint_dir)
    drafts_dir = root / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)

    # Disk space guard (soft — warn and skip, don't crash)
    try:
        free = shutil.disk_usage(str(root)).free
        if free < 900 * 1024 * 1024:
            print(f"  [CKPT] WARNING: low disk ({free // (1024**2)} MB free). Skipping draft save.")
            return None
    except OSError:
        pass

    fname = f"draft_step_{step:07d}.pt"
    draft_path = drafts_dir / fname
    tmp_path = draft_path.with_suffix(".pt.tmp")

    checkpoint = _build_checkpoint_dict(model, optimizer, step, config)

    try:
        torch.save(checkpoint, tmp_path)
        if draft_path.exists():
            draft_path.unlink()
        tmp_path.rename(draft_path)
    except (IOError, OSError) as e:
        print(f"  [CKPT] ERROR: draft save failed: {e}")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except IOError:
                pass
        return None

    # Update manifest
    rel_path = f"drafts/{fname}"
    manifest = load_manifest(checkpoint_dir)
    manifest["checkpoints"][rel_path] = {
        "type": "draft",
        "step": step,
        "loss": round(float(loss), 6),
        "bit_acc": round(float(bit_acc), 4),
        "dataset": dataset or "",
        "timestamp": datetime.datetime.now().isoformat()[:19],
        "tags": [],
    }
    manifest["latest_draft"] = rel_path
    save_manifest(checkpoint_dir, manifest)

    # Backwards-compat: copy to checkpoint_latest.pt
    _update_latest_copy(root, draft_path)

    # Prune old drafts
    prune_drafts(checkpoint_dir, max_drafts)

    print(f"  [CKPT] Draft saved: {fname} (step {step}, loss={loss:.4f}, bit_acc={bit_acc:.4f})")
    return str(draft_path)


def _update_latest_copy(checkpoint_dir: Path, draft_path: Path):
    """Keep checkpoint_latest.pt as a copy of the newest draft (backwards compat)."""
    latest = checkpoint_dir / "checkpoint_latest.pt"
    tmp = checkpoint_dir / "checkpoint_latest.pt.tmp"
    try:
        shutil.copy2(str(draft_path), str(tmp))
        if latest.exists():
            latest.unlink()
        tmp.rename(latest)
    except (IOError, OSError):
        pass  # non-fatal


# ── Prune drafts ─────────────────────────────────────────────────────────────

def prune_drafts(checkpoint_dir: str, max_drafts: int = 15):
    """Delete oldest drafts beyond the limit."""
    root = Path(checkpoint_dir)
    manifest = load_manifest(checkpoint_dir)

    drafts = [(k, v) for k, v in manifest["checkpoints"].items() if v["type"] == "draft"]
    drafts.sort(key=lambda kv: kv[1]["step"])

    to_delete = drafts[: max(0, len(drafts) - max_drafts)]
    if not to_delete:
        return

    for rel_path, _ in to_delete:
        abs_path = root / rel_path
        if abs_path.exists():
            try:
                abs_path.unlink()
                print(f"  [CKPT] Pruned old draft: {Path(rel_path).name}")
            except IOError:
                continue
        del manifest["checkpoints"][rel_path]

    save_manifest(checkpoint_dir, manifest)


# ── Promote to golden ────────────────────────────────────────────────────────

def promote_to_golden(
    checkpoint_dir: str,
    step: int,
    tags: List[str] = None,
    source_path: str = None,
) -> Optional[str]:
    """Copy a checkpoint to golden/ subdir (permanent save).

    Copies (not moves) so the draft ring is unaffected.
    Returns path to golden file, or None on error.
    """
    root = Path(checkpoint_dir)
    golden_dir = root / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(checkpoint_dir)

    # Find source
    if source_path is None:
        latest_rel = manifest.get("latest_draft")
        if not latest_rel:
            print("  [CKPT] ERROR: no latest_draft to promote")
            return None
        source = root / latest_rel
    else:
        source = Path(source_path)

    if not source.exists():
        print(f"  [CKPT] ERROR: source not found: {source}")
        return None

    # Build golden filename
    tag_slug = tags[0] if tags else f"step{step:07d}"
    # Sanitize for filesystem
    tag_slug = "".join(c if c.isalnum() or c in "_-" else "_" for c in tag_slug)
    golden_name = f"golden_{tag_slug}_step{step:07d}.pt"
    golden_path = golden_dir / golden_name

    try:
        shutil.copy2(str(source), str(golden_path))
    except IOError as e:
        print(f"  [CKPT] ERROR: golden copy failed: {e}")
        return None

    # Get source metadata
    source_rel = str(source.relative_to(root)).replace("\\", "/")
    src_meta = manifest["checkpoints"].get(source_rel, {})

    golden_rel = f"golden/{golden_name}"
    manifest["checkpoints"][golden_rel] = {
        "type": "golden",
        "step": step,
        "loss": src_meta.get("loss", 0.0),
        "bit_acc": src_meta.get("bit_acc", 0.0),
        "dataset": src_meta.get("dataset", ""),
        "timestamp": datetime.datetime.now().isoformat()[:19],
        "tags": tags or [],
    }
    save_manifest(checkpoint_dir, manifest)

    print(f"  [CKPT] GOLDEN saved: {golden_name} (step {step}, tags={tags})")
    return str(golden_path)


# ── Load latest (auto-resume) ────────────────────────────────────────────────

def load_latest(
    checkpoint_dir: str,
    model,
    optimizer=None,
    load_fn=None,
) -> Tuple[int, float]:
    """Auto-resume from the highest-step checkpoint (golden or draft).

    Golden saves are canon — if the latest golden is newer than any draft,
    resume from golden. Otherwise resume from the latest draft (continuing
    work beyond the golden baseline).

    Args:
        load_fn: callable(path, model, optimizer) -> (step, acc).
                 Pass test_swarm_config.load_checkpoint from the caller
                 to avoid circular imports.

    Fallback chain:
      1. Highest-step checkpoint (golden or draft) from manifest
      2. Next-highest if corrupt
      3. checkpoint_latest.pt (backwards compat)
      4. Fresh start (0, 0.0)

    Returns (step, best_accuracy).
    """
    if load_fn is None:
        # Minimal fallback: just torch.load + model.load_state_dict
        def load_fn(path, mdl, opt):
            ck = torch.load(path, weights_only=False)
            mdl.load_state_dict(ck["model_state_dict"], strict=False)
            if opt and "optimizer_state_dict" in ck:
                try:
                    opt.load_state_dict(ck["optimizer_state_dict"])
                except Exception:
                    pass
            s = ck.get("step", 0)
            print(f"  [LOAD] Resumed from checkpoint: {path}")
            print(f"        Step: {s}")
            return s, ck.get("best_accuracy", 0.0)

    root = Path(checkpoint_dir)

    if not root.exists():
        print(f"  [CKPT] No checkpoint dir found. Starting fresh.")
        return 0, 0.0

    manifest = load_manifest(checkpoint_dir)

    # Collect ALL checkpoints (golden + drafts), sort by step descending
    all_ckpts = sorted(
        manifest["checkpoints"].items(),
        key=lambda kv: kv[1]["step"],
        reverse=True,
    )

    # Try each in order (highest step first)
    for rel, meta in all_ckpts:
        abs_path = root / rel
        if not abs_path.exists():
            continue
        try:
            step, acc = load_fn(str(abs_path), model, optimizer)
            _type = meta.get("type", "unknown")
            _tags = meta.get("tags", [])
            if _type == "golden":
                print(f"  [CKPT] Resumed from GOLDEN: {Path(rel).name} (tags={_tags})")
            return step, acc
        except Exception as e:
            print(f"  [CKPT] WARNING: {Path(rel).name} corrupt ({e}). Trying next...")
            del manifest["checkpoints"][rel]
            save_manifest(checkpoint_dir, manifest)
            continue

    # Fallback: checkpoint_latest.pt (backwards compat)
    legacy = root / "checkpoint_latest.pt"
    if legacy.exists():
        try:
            print(f"  [CKPT] Falling back to checkpoint_latest.pt")
            step, acc = load_fn(str(legacy), model, optimizer)
            return step, acc
        except Exception:
            pass

    print(f"  [CKPT] No checkpoint found in {root}. Starting fresh.")
    return 0, 0.0


# ── List golden saves ────────────────────────────────────────────────────────

def list_golden(checkpoint_dir: str) -> List[dict]:
    """Return golden checkpoint entries sorted by step (ascending)."""
    manifest = load_manifest(checkpoint_dir)
    golden = [
        {"path": k, **v}
        for k, v in manifest["checkpoints"].items()
        if v["type"] == "golden"
    ]
    golden.sort(key=lambda e: e["step"])
    return golden
