"""
Dataset retirement and staleness management for VRAXION v2.0.

Manages the data/golden/ directory and manifest.json:
  - Retire mastered datasets (move from traindat/ to golden/)
  - Track staleness counters for dream rehearsal rotation
  - Atomic manifest writes, corruption recovery, stale entry cleanup

Usage:
    from retirement import retire_dataset, pick_stalest_dataset, update_staleness_after_rehearsal

    # Retire a mastered dataset
    retire_dataset("data/", "math_addition.traindat", step=1000)

    # During dream: pick stalest for rehearsal
    name = pick_stalest_dataset("data/golden/")

    # After rehearsal: update counters
    update_staleness_after_rehearsal("data/golden/", name)
"""

import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional


GOLDEN_DIR_NAME = "golden"
MANIFEST_NAME = "manifest.json"


def _ensure_golden_dir(data_root: str) -> Path:
    """Create data/golden/ if it doesn't exist. Returns Path to golden dir."""
    golden = Path(data_root) / GOLDEN_DIR_NAME
    golden.mkdir(parents=True, exist_ok=True)
    return golden


def _rebuild_manifest(golden_dir: str) -> dict:
    """Rebuild manifest from directory listing when manifest.json is corrupt."""
    golden_path = Path(golden_dir)
    manifest = {"datasets": {}}
    for f in sorted(golden_path.glob("*.traindat")):
        manifest["datasets"][f.name] = {
            "staleness": 0,
            "retired_at_step": -1,
            "retired_date": "unknown",
        }
    if manifest["datasets"]:
        print(f"  [RETIRE] manifest rebuilt from directory: {len(manifest['datasets'])} datasets")
    return manifest


def _clean_stale_entries(golden_dir: str, manifest: dict) -> dict:
    """Remove manifest entries for files that no longer exist on disk."""
    golden_path = Path(golden_dir)
    existing = {f.name for f in golden_path.glob("*.traindat")}
    stale = [k for k in manifest["datasets"] if k not in existing]
    for k in stale:
        print(f"  [RETIRE] WARNING: removing stale manifest entry: {k}")
        del manifest["datasets"][k]
    return manifest


def load_manifest(golden_dir: str) -> dict:
    """Load manifest.json from golden dir. Rebuild from dir listing if corrupt.

    Returns:
        {"datasets": {"name.traindat": {"staleness": int, "retired_at_step": int,
                                          "retired_date": str}}}
    """
    manifest_path = Path(golden_dir) / MANIFEST_NAME
    manifest = {"datasets": {}}

    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            if not isinstance(manifest.get("datasets"), dict):
                raise ValueError("malformed manifest")
        except (json.JSONDecodeError, ValueError, KeyError):
            manifest = _rebuild_manifest(golden_dir)
    else:
        # No manifest file — check if there are .traindat files to rebuild from
        golden_path = Path(golden_dir)
        if golden_path.exists() and any(golden_path.glob("*.traindat")):
            manifest = _rebuild_manifest(golden_dir)

    manifest = _clean_stale_entries(golden_dir, manifest)
    return manifest


def save_manifest(golden_dir: str, manifest: dict):
    """Atomically write manifest.json (write to .tmp then rename).

    Catches IOError for disk-full scenarios.
    """
    manifest_path = Path(golden_dir) / MANIFEST_NAME
    tmp_path = manifest_path.with_suffix('.json.tmp')
    try:
        with open(tmp_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        # Atomic rename (on Windows, remove target first)
        if manifest_path.exists():
            manifest_path.unlink()
        tmp_path.rename(manifest_path)
    except IOError as e:
        print(f"  [RETIRE] ERROR: failed to save manifest: {e}")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except IOError:
                pass


def retire_dataset(data_root: str, filename: str, step: int) -> bool:
    """Move a .traindat from data/traindat/ to data/golden/.

    Also moves .meta.json sidecar if present.
    Updates manifest.json with staleness=0 and retirement metadata.
    Idempotent: if file already in golden, updates manifest timestamp only.

    Returns True if retired successfully, False on error.
    """
    traindat_dir = Path(data_root) / "traindat"
    golden_dir = _ensure_golden_dir(data_root)

    src = traindat_dir / filename
    dst = golden_dir / filename

    manifest = load_manifest(str(golden_dir))

    # Idempotent: already retired
    if dst.exists() and filename in manifest["datasets"]:
        manifest["datasets"][filename]["retired_at_step"] = step
        manifest["datasets"][filename]["retired_date"] = time.strftime('%Y-%m-%d')
        save_manifest(str(golden_dir), manifest)
        return True

    # Move .traindat
    if src.exists():
        try:
            shutil.move(str(src), str(dst))
        except IOError as e:
            print(f"  [RETIRE] ERROR: failed to move {filename}: {e}")
            return False
    elif dst.exists():
        pass  # already moved but not in manifest
    else:
        print(f"  [RETIRE] WARNING: {filename} not found in traindat/ or golden/")
        return False

    # Move .meta.json sidecar
    meta_name = filename.replace('.traindat', '.meta.json')
    meta_src = traindat_dir / meta_name
    meta_dst = golden_dir / meta_name
    if meta_src.exists() and not meta_dst.exists():
        try:
            shutil.move(str(meta_src), str(meta_dst))
        except IOError:
            pass  # non-fatal

    # Update manifest
    manifest["datasets"][filename] = {
        "staleness": 0,
        "retired_at_step": step,
        "retired_date": time.strftime('%Y-%m-%d'),
    }
    save_manifest(str(golden_dir), manifest)
    print(f"  [RETIRE] {filename} -> golden/ at step {step}")
    return True


def pick_stalest_dataset(golden_dir: str) -> Optional[str]:
    """Return filename of golden dataset with highest staleness.

    Returns None if no golden datasets exist (cold start).
    Tiebreak: alphabetical (deterministic).
    """
    manifest = load_manifest(golden_dir)
    if not manifest["datasets"]:
        return None

    # Sort by (-staleness, name) for deterministic tiebreak
    candidates = sorted(
        manifest["datasets"].items(),
        key=lambda kv: (-kv[1].get("staleness", 0), kv[0])
    )
    return candidates[0][0]


def update_staleness_after_rehearsal(golden_dir: str, rehearsed_filename: str):
    """After rehearsing a golden dataset: reset its staleness to 0,
    increment all others by 1."""
    manifest = load_manifest(golden_dir)
    if rehearsed_filename not in manifest["datasets"]:
        print(f"  [STALE] WARNING: {rehearsed_filename} not in manifest, skipping update")
        return

    for name, entry in manifest["datasets"].items():
        if name == rehearsed_filename:
            entry["staleness"] = 0
        else:
            entry["staleness"] = entry.get("staleness", 0) + 1

    save_manifest(golden_dir, manifest)


def list_golden_datasets(golden_dir: str) -> List[str]:
    """Return sorted list of .traindat filenames in golden dir."""
    golden_path = Path(golden_dir)
    if not golden_path.exists():
        return []
    return sorted(f.name for f in golden_path.glob("*.traindat"))
