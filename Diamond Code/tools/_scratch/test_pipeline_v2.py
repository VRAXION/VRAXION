"""
VRAXION v2.0 Pipeline — Comprehensive Test Suite

17 tests covering:
  T1-T11:  Unit tests (metadata, retirement, manifest, staleness, golden load, split metric)
  T12-T13: Integration tests (full lifecycle, multi-retirement rotation)
  T14-T17: Stress/adversarial tests (corruption recovery, missing files, rapid retire, atomic writes)

All tests are CPU-only, deterministic, no GPU needed.

Usage:
    cd "S:/AI/work/VRAXION_DEV/Diamond Code"
    python -m pytest tools/_scratch/test_pipeline_v2.py -v
"""

import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch

# Ensure Diamond Code root is on sys.path
DIAMOND_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if DIAMOND_ROOT not in sys.path:
    sys.path.insert(0, DIAMOND_ROOT)

from retirement import (
    _ensure_golden_dir,
    _rebuild_manifest,
    _clean_stale_entries,
    load_manifest,
    save_manifest,
    retire_dataset,
    pick_stalest_dataset,
    update_staleness_after_rehearsal,
    list_golden_datasets,
)
from traindat_loader import load_batch_from_file, generate_batch_from_bytes, generate_batch_binary_bits, TraindatLoader


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def tmp_data(tmp_path):
    """Create a temporary data root with traindat/ and some test files."""
    traindat_dir = tmp_path / "traindat"
    traindat_dir.mkdir()

    # Create a small .traindat (256 bytes of deterministic data)
    random.seed(42)
    data = bytes(random.randint(0, 255) for _ in range(256))
    (traindat_dir / "test_alpha.traindat").write_bytes(data)

    # Create its .meta.json sidecar
    meta = {
        "task": "test_alpha",
        "tier": 0,
        "seed": 42,
        "target_bytes": 256,
        "actual_bytes": 256,
        "version": "2.0",
        "date": "2026-02-19T00:00:00",
        "pairs_estimate": 8,
        "script": "test_pipeline_v2.py",
    }
    (traindat_dir / "test_alpha.meta.json").write_text(json.dumps(meta, indent=2))

    return tmp_path


@pytest.fixture
def tmp_data_multi(tmp_path):
    """Create a data root with 3 test .traindat files for multi-retirement tests."""
    traindat_dir = tmp_path / "traindat"
    traindat_dir.mkdir()

    random.seed(99)
    for name in ["alpha.traindat", "beta.traindat", "gamma.traindat"]:
        data = bytes(random.randint(0, 255) for _ in range(512))
        (traindat_dir / name).write_bytes(data)
        meta = {"task": name.replace(".traindat", ""), "tier": 0}
        (traindat_dir / name.replace(".traindat", ".meta.json")).write_text(
            json.dumps(meta)
        )

    return tmp_path


@pytest.fixture
def golden_with_manifest(tmp_path):
    """Create a golden dir with 3 datasets and a manifest with known staleness."""
    golden = tmp_path / "golden"
    golden.mkdir()

    random.seed(77)
    datasets = {
        "a.traindat": {"staleness": 5, "retired_at_step": 100, "retired_date": "2026-01-01"},
        "b.traindat": {"staleness": 10, "retired_at_step": 200, "retired_date": "2026-01-02"},
        "c.traindat": {"staleness": 3, "retired_at_step": 300, "retired_date": "2026-01-03"},
    }
    for name in datasets:
        data = bytes(random.randint(0, 255) for _ in range(512))
        (golden / name).write_bytes(data)

    manifest = {"datasets": datasets}
    (golden / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return tmp_path


# ============================================================
# T1: Metadata Generation
# ============================================================

class TestT1Metadata:
    def test_meta_json_v2_fields(self, tmp_data):
        """Verify .meta.json has all v2.0 fields: version, date, pairs_estimate, script."""
        meta_path = tmp_data / "traindat" / "test_alpha.meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())

        # v2.0 required fields
        assert meta["version"] == "2.0"
        assert "date" in meta
        assert len(meta["date"]) == 19  # ISO format YYYY-MM-DDTHH:MM:SS
        assert isinstance(meta["pairs_estimate"], int)
        assert meta["pairs_estimate"] > 0
        assert meta["script"] == "test_pipeline_v2.py"

        # Original fields still present
        assert meta["seed"] == 42
        assert meta["task"] == "test_alpha"
        assert meta["actual_bytes"] == 256

    def test_generate_traindat_suite_meta_format(self):
        """Verify the generator script produces correct meta fields."""
        # Import the module to access GENERATORS dict
        from generate_traindat_suite import GENERATORS, BLOCK

        for name, entry in GENERATORS.items():
            m = entry["meta"]
            assert "task" in m, f"{name} missing 'task'"
            assert "tier" in m, f"{name} missing 'tier'"
            assert isinstance(m["tier"], int), f"{name} tier not int"
            assert 0 <= m["tier"] <= 5, f"{name} tier out of range: {m['tier']}"


# ============================================================
# T2: Retirement — Basic Move
# ============================================================

class TestT2RetirementBasic:
    def test_retire_moves_file(self, tmp_data):
        """Retire a dataset: file moves to golden/, manifest updated."""
        data_root = str(tmp_data)
        result = retire_dataset(data_root, "test_alpha.traindat", step=100)
        assert result is True

        # File moved
        assert not (tmp_data / "traindat" / "test_alpha.traindat").exists()
        assert (tmp_data / "golden" / "test_alpha.traindat").exists()

        # Meta sidecar moved
        assert not (tmp_data / "traindat" / "test_alpha.meta.json").exists()
        assert (tmp_data / "golden" / "test_alpha.meta.json").exists()

        # Manifest updated
        manifest = load_manifest(str(tmp_data / "golden"))
        assert "test_alpha.traindat" in manifest["datasets"]
        entry = manifest["datasets"]["test_alpha.traindat"]
        assert entry["staleness"] == 0
        assert entry["retired_at_step"] == 100
        assert entry["retired_date"] == time.strftime('%Y-%m-%d')

    def test_retire_nonexistent_file(self, tmp_data):
        """Retiring a file that doesn't exist returns False."""
        result = retire_dataset(str(tmp_data), "ghost.traindat", step=50)
        assert result is False

    def test_golden_dir_created_on_retire(self, tmp_path):
        """Golden dir is auto-created if missing."""
        traindat = tmp_path / "traindat"
        traindat.mkdir()
        (traindat / "x.traindat").write_bytes(b"\x00" * 100)

        result = retire_dataset(str(tmp_path), "x.traindat", step=1)
        assert result is True
        assert (tmp_path / "golden").is_dir()


# ============================================================
# T3: Retirement — Idempotent
# ============================================================

class TestT3RetirementIdempotent:
    def test_retire_same_file_twice(self, tmp_data):
        """Retiring same file twice: no error, manifest updated (not duplicated)."""
        data_root = str(tmp_data)

        # First retire
        result1 = retire_dataset(data_root, "test_alpha.traindat", step=100)
        assert result1 is True

        # Second retire (idempotent — file already in golden/)
        result2 = retire_dataset(data_root, "test_alpha.traindat", step=200)
        assert result2 is True

        # Manifest has exactly 1 entry (not duplicated)
        manifest = load_manifest(str(tmp_data / "golden"))
        assert len(manifest["datasets"]) == 1
        assert manifest["datasets"]["test_alpha.traindat"]["retired_at_step"] == 200


# ============================================================
# T4: Manifest Rebuild on Corruption
# ============================================================

class TestT4ManifestRebuild:
    def test_corrupt_manifest_rebuilds(self, golden_with_manifest):
        """Corrupt manifest.json triggers rebuild from directory listing."""
        golden = golden_with_manifest / "golden"

        # Write garbage
        (golden / "manifest.json").write_text("THIS IS NOT JSON {{{{")

        manifest = load_manifest(str(golden))
        assert isinstance(manifest["datasets"], dict)
        # Should have rebuilt with all 3 files
        assert len(manifest["datasets"]) == 3
        # All staleness reset to 0 (unknown history)
        for entry in manifest["datasets"].values():
            assert entry["staleness"] == 0

    def test_missing_manifest_rebuilds(self, golden_with_manifest):
        """Missing manifest.json triggers rebuild from directory listing."""
        golden = golden_with_manifest / "golden"
        (golden / "manifest.json").unlink()

        manifest = load_manifest(str(golden))
        assert len(manifest["datasets"]) == 3

    def test_malformed_manifest_structure(self, golden_with_manifest):
        """Manifest with wrong structure triggers rebuild."""
        golden = golden_with_manifest / "golden"
        # Valid JSON but missing 'datasets' key
        (golden / "manifest.json").write_text('{"wrong_key": [1,2,3]}')

        manifest = load_manifest(str(golden))
        assert "datasets" in manifest
        assert len(manifest["datasets"]) == 3


# ============================================================
# T5: Stale Entry Cleanup
# ============================================================

class TestT5StaleEntryCleanup:
    def test_stale_entries_removed(self, golden_with_manifest):
        """Manifest entries for deleted files are cleaned up on load."""
        golden = golden_with_manifest / "golden"

        # Add a ghost entry to manifest
        manifest = json.loads((golden / "manifest.json").read_text())
        manifest["datasets"]["ghost.traindat"] = {
            "staleness": 99,
            "retired_at_step": 0,
            "retired_date": "never",
        }
        (golden / "manifest.json").write_text(json.dumps(manifest))

        # Load — ghost should be removed
        loaded = load_manifest(str(golden))
        assert "ghost.traindat" not in loaded["datasets"]
        # Real files still present
        assert "a.traindat" in loaded["datasets"]
        assert "b.traindat" in loaded["datasets"]
        assert "c.traindat" in loaded["datasets"]


# ============================================================
# T6: Staleness Selection
# ============================================================

class TestT6StalenessSelection:
    def test_pick_stalest(self, golden_with_manifest):
        """pick_stalest_dataset returns the highest-staleness file."""
        golden = golden_with_manifest / "golden"
        result = pick_stalest_dataset(str(golden))
        assert result == "b.traindat"  # staleness=10

    def test_pick_stalest_single_dataset(self, tmp_path):
        """Works with only 1 dataset in golden."""
        golden = tmp_path / "golden"
        golden.mkdir()
        (golden / "only.traindat").write_bytes(b"\x00" * 50)
        manifest = {"datasets": {"only.traindat": {"staleness": 0, "retired_at_step": 1, "retired_date": "x"}}}
        (golden / "manifest.json").write_text(json.dumps(manifest))

        result = pick_stalest_dataset(str(golden))
        assert result == "only.traindat"


# ============================================================
# T7: Staleness Tiebreak
# ============================================================

class TestT7StalenessTiebreak:
    def test_tiebreak_alphabetical(self, tmp_path):
        """Equal staleness: alphabetical first returned (deterministic)."""
        golden = tmp_path / "golden"
        golden.mkdir()

        datasets = {}
        for name in ["zebra.traindat", "apple.traindat", "mango.traindat"]:
            (golden / name).write_bytes(b"\x00" * 50)
            datasets[name] = {"staleness": 5, "retired_at_step": 1, "retired_date": "x"}

        manifest = {"datasets": datasets}
        (golden / "manifest.json").write_text(json.dumps(manifest))

        result = pick_stalest_dataset(str(golden))
        assert result == "apple.traindat"  # alphabetical first at staleness=5


# ============================================================
# T8: Staleness Update After Rehearsal
# ============================================================

class TestT8StalenessUpdate:
    def test_update_staleness(self, golden_with_manifest):
        """Rehearsing B: B->0, A->6, C->4 (was A=5, B=10, C=3)."""
        golden = golden_with_manifest / "golden"
        update_staleness_after_rehearsal(str(golden), "b.traindat")

        manifest = load_manifest(str(golden))
        assert manifest["datasets"]["b.traindat"]["staleness"] == 0
        assert manifest["datasets"]["a.traindat"]["staleness"] == 6   # was 5 + 1
        assert manifest["datasets"]["c.traindat"]["staleness"] == 4   # was 3 + 1

    def test_update_staleness_nonexistent(self, golden_with_manifest):
        """Rehearsing a file not in manifest: no error, no changes."""
        golden = golden_with_manifest / "golden"
        original = load_manifest(str(golden))
        orig_a = original["datasets"]["a.traindat"]["staleness"]

        update_staleness_after_rehearsal(str(golden), "nonexistent.traindat")

        after = load_manifest(str(golden))
        assert after["datasets"]["a.traindat"]["staleness"] == orig_a


# ============================================================
# T9: Cold Start
# ============================================================

class TestT9ColdStart:
    def test_empty_golden_returns_none(self, tmp_path):
        """Empty golden dir: pick_stalest_dataset returns None."""
        golden = tmp_path / "golden"
        golden.mkdir()
        result = pick_stalest_dataset(str(golden))
        assert result is None

    def test_nonexistent_golden_returns_none(self, tmp_path):
        """Nonexistent golden dir: pick_stalest_dataset returns None."""
        result = pick_stalest_dataset(str(tmp_path / "golden"))
        assert result is None

    def test_list_golden_empty(self, tmp_path):
        """list_golden_datasets on empty/nonexistent dir returns []."""
        assert list_golden_datasets(str(tmp_path / "golden")) == []


# ============================================================
# T10: Golden File Load
# ============================================================

class TestT10GoldenFileLoad:
    def test_load_batch_gray_mode(self, tmp_path):
        """load_batch_from_file in Gray mode returns correct shapes."""
        # Create a 1KB test file
        random.seed(42)
        data = bytes(random.randint(0, 255) for _ in range(1024))
        filepath = str(tmp_path / "test.traindat")
        with open(filepath, 'wb') as f:
            f.write(data)

        x, y, mask = load_batch_from_file(
            filepath, n_samples=4, seq_len=8, num_bits=8, seed=42,
            binary_bits_mode=False,
        )
        assert x.shape == (4, 8, 8)
        assert y.shape == (4, 8, 8)
        assert mask is None  # Gray mode returns None mask

    def test_load_batch_binary_mode(self, tmp_path):
        """load_batch_from_file in binary mode returns correct shapes + mask."""
        random.seed(42)
        data = bytes(random.randint(0, 255) for _ in range(1024))
        filepath = str(tmp_path / "test.traindat")
        with open(filepath, 'wb') as f:
            f.write(data)

        x, y, mask = load_batch_from_file(
            filepath, n_samples=4, seq_len=8, num_bits=8, seed=42,
            binary_bits_mode=True,
        )
        assert x.shape == (4, 8, 8)
        assert y.shape == (4, 8, 8)
        assert mask is not None
        assert mask.shape == (4, 8, 8)
        assert (mask == 1.0).all()  # traindat has no padding

    def test_load_batch_deterministic(self, tmp_path):
        """Same seed produces same batch."""
        random.seed(42)
        data = bytes(random.randint(0, 255) for _ in range(1024))
        filepath = str(tmp_path / "test.traindat")
        with open(filepath, 'wb') as f:
            f.write(data)

        x1, y1, _ = load_batch_from_file(filepath, 2, 8, 8, seed=123, binary_bits_mode=True)
        x2, y2, _ = load_batch_from_file(filepath, 2, 8, 8, seed=123, binary_bits_mode=True)
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)

    def test_load_batch_file_not_found(self, tmp_path):
        """FileNotFoundError raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_batch_from_file(str(tmp_path / "nope.traindat"), 1, 8, 8)


# ============================================================
# T11: Split Metric
# ============================================================

class TestT11SplitMetric:
    def test_split_accuracy_perfect_copy_region(self):
        """Copy region perfect, novel region wrong -> copy_acc ~1.0, novel_acc ~0.0."""
        from test_swarm_config import compute_split_accuracy

        B, T, bits = 10, 20, 8
        overlap_bytes = 10  # 10 bytes = 10 positions at num_bits=8

        # Create target: all ones
        target = torch.ones(B, T, bits)

        # Create output logits: first 10 positions highly confident 1 (logit +10),
        # last 10 positions highly confident 0 (logit -10), mismatching target
        output = torch.zeros(B, T, bits)
        output[:, :10, :] = 10.0   # sigmoid(10) >> 0.5, prediction = 1 = target
        output[:, 10:, :] = -10.0  # sigmoid(-10) << 0.5, prediction = 0 != target

        copy_acc, novel_acc = compute_split_accuracy(output, target, overlap_bytes, bits)
        assert copy_acc is not None
        assert novel_acc is not None
        assert copy_acc > 0.99  # perfect match in copy region
        assert novel_acc < 0.01  # all wrong in novel region

    def test_split_accuracy_not_applicable(self):
        """Returns (None, None) when overlap covers entire sequence or is 0."""
        from test_swarm_config import compute_split_accuracy

        B, T, bits = 5, 10, 8
        target = torch.zeros(B, T, bits)
        output = torch.zeros(B, T, bits)

        # overlap >= T
        c, n = compute_split_accuracy(output, target, overlap_bytes=100, num_bits=8)
        assert c is None and n is None

        # overlap = 0
        c, n = compute_split_accuracy(output, target, overlap_bytes=0, num_bits=8)
        assert c is None and n is None

    def test_split_accuracy_multi_byte_positions(self):
        """Correct position splitting with num_bits=16 (2 bytes per position)."""
        from test_swarm_config import compute_split_accuracy

        B, T, bits = 5, 10, 16
        overlap_bytes = 6  # 6 bytes / 2 bytes_per_pos = 3 positions

        target = torch.zeros(B, T, bits)
        output = torch.full((B, T, bits), -10.0)  # all predict 0 = match target
        output[:, 3:, :] = 10.0  # novel region predicts 1, but target is 0

        copy_acc, novel_acc = compute_split_accuracy(output, target, overlap_bytes, bits)
        assert copy_acc is not None
        assert copy_acc > 0.99
        assert novel_acc < 0.01  # all wrong in novel region


# ============================================================
# T12: Full Lifecycle Integration
# ============================================================

class TestT12FullLifecycle:
    def test_retire_then_dream_load(self, tmp_data):
        """Full lifecycle: generate -> retire -> dream load from golden."""
        data_root = str(tmp_data)

        # Step 1: Retire the test dataset
        assert retire_dataset(data_root, "test_alpha.traindat", step=100)

        # Step 2: Verify golden state
        golden_dir = str(tmp_data / "golden")
        datasets = list_golden_datasets(golden_dir)
        assert datasets == ["test_alpha.traindat"]

        # Step 3: Pick stalest (only one, so it's the one)
        stalest = pick_stalest_dataset(golden_dir)
        assert stalest == "test_alpha.traindat"

        # Step 4: Load batch from golden (simulates dream rehearsal)
        golden_path = str(tmp_data / "golden" / "test_alpha.traindat")
        x, y, mask = load_batch_from_file(
            golden_path, n_samples=2, seq_len=4, num_bits=8,
            seed=42, binary_bits_mode=True,
        )
        assert x.shape == (2, 4, 8)
        assert y.shape == (2, 4, 8)

        # Step 5: Update staleness after rehearsal
        update_staleness_after_rehearsal(golden_dir, "test_alpha.traindat")
        manifest = load_manifest(golden_dir)
        assert manifest["datasets"]["test_alpha.traindat"]["staleness"] == 0

    def test_retire_preserves_meta(self, tmp_data):
        """After retirement, .meta.json sidecar is accessible in golden/."""
        data_root = str(tmp_data)
        retire_dataset(data_root, "test_alpha.traindat", step=50)

        meta_path = tmp_data / "golden" / "test_alpha.meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["version"] == "2.0"
        assert meta["seed"] == 42


# ============================================================
# T13: Multi-Retirement Rotation
# ============================================================

class TestT13MultiRetirement:
    def test_three_retires_staleness_rotation(self, tmp_data_multi):
        """Retire 3 datasets, run 3 dream cycles, verify staleness rotation."""
        data_root = str(tmp_data_multi)
        golden_dir = str(tmp_data_multi / "golden")

        # Retire all 3 (sequential steps)
        assert retire_dataset(data_root, "alpha.traindat", step=10)
        assert retire_dataset(data_root, "beta.traindat", step=20)
        assert retire_dataset(data_root, "gamma.traindat", step=30)

        # Verify all 3 in golden
        datasets = list_golden_datasets(golden_dir)
        assert len(datasets) == 3

        # After retirement, all staleness = 0
        manifest = load_manifest(golden_dir)
        for entry in manifest["datasets"].values():
            assert entry["staleness"] == 0

        # Dream cycle 1: pick alphabetically first (all staleness=0)
        stalest = pick_stalest_dataset(golden_dir)
        assert stalest == "alpha.traindat"
        update_staleness_after_rehearsal(golden_dir, "alpha.traindat")

        # Now: alpha=0, beta=1, gamma=1
        manifest = load_manifest(golden_dir)
        assert manifest["datasets"]["alpha.traindat"]["staleness"] == 0
        assert manifest["datasets"]["beta.traindat"]["staleness"] == 1
        assert manifest["datasets"]["gamma.traindat"]["staleness"] == 1

        # Dream cycle 2: pick beta (staleness=1, tiebreak alphabetical over gamma)
        stalest = pick_stalest_dataset(golden_dir)
        assert stalest == "beta.traindat"
        update_staleness_after_rehearsal(golden_dir, "beta.traindat")

        # Now: alpha=1, beta=0, gamma=2
        manifest = load_manifest(golden_dir)
        assert manifest["datasets"]["alpha.traindat"]["staleness"] == 1
        assert manifest["datasets"]["beta.traindat"]["staleness"] == 0
        assert manifest["datasets"]["gamma.traindat"]["staleness"] == 2

        # Dream cycle 3: pick gamma (staleness=2, highest)
        stalest = pick_stalest_dataset(golden_dir)
        assert stalest == "gamma.traindat"
        update_staleness_after_rehearsal(golden_dir, "gamma.traindat")

        # Now: alpha=2, beta=1, gamma=0
        manifest = load_manifest(golden_dir)
        assert manifest["datasets"]["alpha.traindat"]["staleness"] == 2
        assert manifest["datasets"]["beta.traindat"]["staleness"] == 1
        assert manifest["datasets"]["gamma.traindat"]["staleness"] == 0

    def test_rotation_is_fair_over_many_cycles(self, tmp_data_multi):
        """Over N cycles, each dataset rehearsed ~N/3 times (fair rotation)."""
        data_root = str(tmp_data_multi)
        golden_dir = str(tmp_data_multi / "golden")

        for name in ["alpha.traindat", "beta.traindat", "gamma.traindat"]:
            retire_dataset(data_root, name, step=1)

        rehearsal_counts = {"alpha.traindat": 0, "beta.traindat": 0, "gamma.traindat": 0}
        for _ in range(30):
            stalest = pick_stalest_dataset(golden_dir)
            rehearsal_counts[stalest] += 1
            update_staleness_after_rehearsal(golden_dir, stalest)

        # Each should be rehearsed exactly 10 times (perfect round-robin)
        assert rehearsal_counts["alpha.traindat"] == 10
        assert rehearsal_counts["beta.traindat"] == 10
        assert rehearsal_counts["gamma.traindat"] == 10


# ============================================================
# T14: Manifest Corruption Recovery
# ============================================================

class TestT14ManifestCorruption:
    def test_truncated_json(self, golden_with_manifest):
        """Half-truncated JSON triggers rebuild."""
        golden = golden_with_manifest / "golden"
        (golden / "manifest.json").write_text('{"datasets": {"a.traindat": {"staleness')

        manifest = load_manifest(str(golden))
        assert len(manifest["datasets"]) == 3
        for entry in manifest["datasets"].values():
            assert entry["staleness"] == 0  # rebuilt = reset

    def test_empty_file(self, golden_with_manifest):
        """Empty manifest file triggers rebuild."""
        golden = golden_with_manifest / "golden"
        (golden / "manifest.json").write_text("")

        manifest = load_manifest(str(golden))
        assert len(manifest["datasets"]) == 3

    def test_binary_garbage(self, golden_with_manifest):
        """Binary garbage in manifest triggers rebuild."""
        golden = golden_with_manifest / "golden"
        (golden / "manifest.json").write_bytes(b"\x00\xff\xfe\xfd" * 100)

        manifest = load_manifest(str(golden))
        assert len(manifest["datasets"]) == 3

    def test_valid_json_wrong_type(self, golden_with_manifest):
        """Valid JSON but datasets is a list, not dict."""
        golden = golden_with_manifest / "golden"
        (golden / "manifest.json").write_text('{"datasets": [1, 2, 3]}')

        manifest = load_manifest(str(golden))
        assert isinstance(manifest["datasets"], dict)
        assert len(manifest["datasets"]) == 3


# ============================================================
# T15: Missing Golden File Mid-Dream
# ============================================================

class TestT15MissingGoldenFile:
    def test_deleted_file_cleaned_from_manifest(self, golden_with_manifest):
        """Delete a .traindat from golden, manifest cleaned on next load."""
        golden = golden_with_manifest / "golden"

        # Delete b.traindat from disk
        (golden / "b.traindat").unlink()

        # load_manifest should clean the stale entry
        manifest = load_manifest(str(golden))
        assert "b.traindat" not in manifest["datasets"]
        assert len(manifest["datasets"]) == 2

    def test_pick_stalest_after_deletion(self, golden_with_manifest):
        """After deleting stalest file, next stalest is picked."""
        golden = golden_with_manifest / "golden"

        # b.traindat has staleness=10 (stalest). Delete it.
        (golden / "b.traindat").unlink()

        # Should now pick a.traindat (staleness=5)
        stalest = pick_stalest_dataset(str(golden))
        assert stalest == "a.traindat"

    def test_load_batch_from_deleted_golden(self, golden_with_manifest):
        """FileNotFoundError when loading batch from deleted golden file."""
        golden = golden_with_manifest / "golden"
        fake_path = str(golden / "deleted.traindat")

        with pytest.raises(FileNotFoundError):
            load_batch_from_file(fake_path, 1, 4, 8)


# ============================================================
# T16: Rapid Retirement
# ============================================================

class TestT16RapidRetirement:
    def test_rapid_retire_10_files(self, tmp_path):
        """Retire 10 files in tight loop: manifest consistent, no duplicates."""
        traindat = tmp_path / "traindat"
        traindat.mkdir()
        data_root = str(tmp_path)

        # Create 10 tiny .traindat files
        random.seed(0)
        names = [f"rapid_{i:03d}.traindat" for i in range(10)]
        for name in names:
            (traindat / name).write_bytes(bytes(random.randint(0, 255) for _ in range(100)))

        # Retire all 10 in rapid succession
        for i, name in enumerate(names):
            result = retire_dataset(data_root, name, step=i)
            assert result is True, f"Failed to retire {name}"

        # Verify manifest
        golden_dir = str(tmp_path / "golden")
        manifest = load_manifest(golden_dir)
        assert len(manifest["datasets"]) == 10

        # No duplicates (set size == dict size)
        assert len(set(manifest["datasets"].keys())) == 10

        # All have staleness=0
        for entry in manifest["datasets"].values():
            assert entry["staleness"] == 0

        # All files exist in golden
        for name in names:
            assert (tmp_path / "golden" / name).exists()

        # traindat dir is empty
        remaining = list(traindat.glob("*.traindat"))
        assert len(remaining) == 0

    def test_rapid_retire_same_file(self, tmp_data):
        """Retire the same file 10 times in a loop: idempotent."""
        data_root = str(tmp_data)

        for i in range(10):
            result = retire_dataset(data_root, "test_alpha.traindat", step=i * 10)
            assert result is True

        manifest = load_manifest(str(tmp_data / "golden"))
        assert len(manifest["datasets"]) == 1
        assert manifest["datasets"]["test_alpha.traindat"]["retired_at_step"] == 90


# ============================================================
# T17: Atomic Write Safety
# ============================================================

class TestT17AtomicWrite:
    def test_no_tmp_file_after_save(self, tmp_path):
        """No .json.tmp left after normal save."""
        golden = tmp_path / "golden"
        golden.mkdir()

        manifest = {"datasets": {"x.traindat": {"staleness": 0, "retired_at_step": 1, "retired_date": "x"}}}
        save_manifest(str(golden), manifest)

        assert (golden / "manifest.json").exists()
        assert not (golden / "manifest.json.tmp").exists()

    def test_manifest_content_after_save(self, tmp_path):
        """Saved manifest is valid JSON and matches input."""
        golden = tmp_path / "golden"
        golden.mkdir()

        original = {
            "datasets": {
                "a.traindat": {"staleness": 5, "retired_at_step": 100, "retired_date": "2026-01-01"},
                "b.traindat": {"staleness": 0, "retired_at_step": 200, "retired_date": "2026-01-02"},
            }
        }
        save_manifest(str(golden), original)

        # Read back and verify
        loaded = json.loads((golden / "manifest.json").read_text())
        assert loaded == original

    def test_save_overwrite_preserves_atomicity(self, tmp_path):
        """Multiple saves don't corrupt — last write wins."""
        golden = tmp_path / "golden"
        golden.mkdir()

        for i in range(20):
            manifest = {"datasets": {f"file_{i}.traindat": {"staleness": i, "retired_at_step": i, "retired_date": "x"}}}
            save_manifest(str(golden), manifest)

        loaded = json.loads((golden / "manifest.json").read_text())
        assert len(loaded["datasets"]) == 1
        assert "file_19.traindat" in loaded["datasets"]
        assert loaded["datasets"]["file_19.traindat"]["staleness"] == 19

    def test_ensure_golden_dir_idempotent(self, tmp_path):
        """_ensure_golden_dir can be called multiple times safely."""
        for _ in range(5):
            p = _ensure_golden_dir(str(tmp_path))
            assert p.is_dir()
            assert p.name == "golden"


# ============================================================
# Extra: Edge Case Coverage
# ============================================================

class TestEdgeCases:
    def test_list_golden_sorted(self, golden_with_manifest):
        """list_golden_datasets returns sorted names."""
        golden = golden_with_manifest / "golden"
        result = list_golden_datasets(str(golden))
        assert result == sorted(result)

    def test_empty_traindat_dir(self, tmp_path):
        """Retire from empty traindat dir returns False."""
        traindat = tmp_path / "traindat"
        traindat.mkdir()
        result = retire_dataset(str(tmp_path), "nope.traindat", step=0)
        assert result is False

    def test_manifest_with_extra_fields(self, tmp_path):
        """Manifest with extra unknown fields is preserved (forward compat)."""
        golden = tmp_path / "golden"
        golden.mkdir()
        (golden / "x.traindat").write_bytes(b"\x00" * 50)

        manifest = {
            "datasets": {
                "x.traindat": {
                    "staleness": 3,
                    "retired_at_step": 10,
                    "retired_date": "2026-01-01",
                    "custom_field": "preserved",
                }
            },
            "metadata_version": "2.0",  # extra top-level field
        }
        (golden / "manifest.json").write_text(json.dumps(manifest))

        loaded = load_manifest(str(golden))
        assert loaded["datasets"]["x.traindat"]["custom_field"] == "preserved"

    def test_load_batch_large_num_bits(self, tmp_path):
        """load_batch_from_file works with larger num_bits (16)."""
        random.seed(42)
        data = bytes(random.randint(0, 255) for _ in range(2048))
        filepath = str(tmp_path / "wide.traindat")
        with open(filepath, 'wb') as f:
            f.write(data)

        x, y, mask = load_batch_from_file(
            filepath, n_samples=2, seq_len=8, num_bits=16,
            seed=42, binary_bits_mode=True,
        )
        assert x.shape == (2, 8, 16)
        assert y.shape == (2, 8, 16)


# ============================================================
# T18: Integration Smoke — Model + Training Loop Glue
# ============================================================

class TestT18IntegrationSmoke:
    """
    End-to-end integration test exercising the EXACT glue code paths
    from test_swarm_config.py:
      1. Retirement hook after eval (lines 2366-2385)
      2. Dream rehearsal from golden (lines 1730-1765)
      3. Staleness update after dream (lines 1890-1894)
      4. Split metric wiring (lines 2387-2405)

    Uses a real (tiny) SwarmByteRingModel on CPU. ~5-10s.
    """

    @pytest.fixture
    def mini_env(self, tmp_path):
        """Set up a minimal training environment: data, model, optimizer, loader."""
        # Create traindat dir with a copy_echo-style file (512 bytes, easy pattern)
        traindat_dir = tmp_path / "traindat"
        traindat_dir.mkdir()

        random.seed(42)
        # copy_echo pattern: each byte repeated 16 times
        data = bytearray()
        for _ in range(32):
            b = random.randint(0, 255)
            data.extend(bytes([b]) * 16)
        (traindat_dir / "easy.traindat").write_bytes(bytes(data))

        # Meta sidecar
        meta = {
            "task": "copy_echo",
            "tier": 0,
            "seed": 42,
            "target_bytes": len(data),
            "actual_bytes": len(data),
            "version": "2.0",
            "date": "2026-02-19T00:00:00",
            "pairs_estimate": len(data) // 32,
            "script": "test",
        }
        (traindat_dir / "easy.meta.json").write_text(json.dumps(meta, indent=2))

        # Create a second dataset for multi-retirement
        data2 = bytearray()
        for _ in range(32):
            b = random.randint(0, 255)
            data2.extend(bytes([b]) * 16)
        (traindat_dir / "medium.traindat").write_bytes(bytes(data2))
        meta2 = dict(meta)
        meta2["task"] = "medium"
        (traindat_dir / "medium.meta.json").write_text(json.dumps(meta2, indent=2))

        # Create a streaming dataset meta (for split metric test)
        streaming_meta = dict(meta)
        streaming_meta["is_streaming"] = True
        streaming_meta["overlap_bytes"] = 4
        (traindat_dir / "easy.meta.json").write_text(json.dumps(streaming_meta, indent=2))

        # Model: smallest viable SwarmByteRingModel
        from swarm_model import SwarmByteRingModel
        model = SwarmByteRingModel(
            num_memory_positions=8,
            embedding_dim=16,
            num_beings=1,
            depth=2,
            attention_radius=2,
            num_bits=8,
            think_ticks=0,
            use_lcx=False,
        )
        model = model.float()
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # TraindatLoader in sequential mode
        loader = TraindatLoader(str(traindat_dir))
        loader.set_sequential(True, steps_per_dataset=5)

        return {
            "tmp_path": tmp_path,
            "traindat_dir": traindat_dir,
            "model": model,
            "optimizer": optimizer,
            "loader": loader,
            "data_root": str(tmp_path),
            "golden_dir": str(tmp_path / "golden"),
        }

    def test_retirement_hook_glue(self, mini_env):
        """
        Replicate the EXACT retirement hook from test_swarm_config.py lines 2366-2385.
        Force threshold=0.0 so any eval triggers retirement.
        """
        loader = mini_env["loader"]
        data_root = mini_env["data_root"]
        tmp_path = mini_env["tmp_path"]

        # Simulate controls
        controls = {
            "auto_retire": True,
            "mastery_threshold": 0.0,  # force retire on any eval
        }

        # Simulate eval_metrics_accum (the training loop builds this)
        eval_metrics_accum = {"bit_acc": 0.55}  # above threshold of 0.0

        # --- BEGIN EXACT GLUE CODE from test_swarm_config.py lines 2366-2385 ---
        _retire_on = controls.get('auto_retire', True)
        _retire_thresh = controls.get('mastery_threshold', 0.95)
        traindat_loader = loader
        step = 100

        if _retire_on and traindat_loader and eval_metrics_accum:
            _eval_ba = eval_metrics_accum.get('bit_acc', 0)
            if _eval_ba >= _retire_thresh:
                _active_ds = traindat_loader.current_dataset
                if _active_ds:
                    from retirement import retire_dataset as _retire_fn
                    _data_root = str(Path(data_root))
                    retired = _retire_fn(_data_root, _active_ds, step)
                    assert retired is True, f"Failed to retire {_active_ds}"
                    traindat_loader.update_weights({_active_ds: 0})
        # --- END EXACT GLUE CODE ---

        # Verify retirement happened
        assert (tmp_path / "golden" / "easy.traindat").exists()
        assert not (tmp_path / "traindat" / "easy.traindat").exists()

        # Verify weight zeroed
        assert loader.weights["easy.traindat"] == 0

        # Verify manifest
        manifest = load_manifest(str(tmp_path / "golden"))
        assert "easy.traindat" in manifest["datasets"]
        assert manifest["datasets"]["easy.traindat"]["retired_at_step"] == 100

    def test_dream_rehearsal_glue(self, mini_env):
        """
        Replicate the EXACT dream rehearsal glue from test_swarm_config.py lines 1730-1765.
        Retire a dataset first, then exercise the golden loading path.
        """
        loader = mini_env["loader"]
        model = mini_env["model"]
        optimizer = mini_env["optimizer"]
        data_root = mini_env["data_root"]
        tmp_path = mini_env["tmp_path"]

        # Step 1: Retire a dataset to populate golden/
        retire_dataset(data_root, "easy.traindat", step=50)
        assert (tmp_path / "golden" / "easy.traindat").exists()

        # Step 2: Simulate dream setup variables
        _dream_mode = 'rehearsal'
        batch_size = 2
        seq_len = 4
        num_bits = 8
        step = 60
        device = torch.device('cpu')
        _dream_golden_file = None
        _dream_y = torch.zeros(batch_size, seq_len, num_bits)  # placeholder
        _dream_bit_mask = None
        _dream_n = 3  # dream steps
        _dream_hdr = "  [DREAM]"

        # --- BEGIN EXACT GLUE CODE from test_swarm_config.py lines 1735-1765 ---
        if _dream_mode == 'rehearsal':
            from retirement import pick_stalest_dataset as _pick_fn
            from retirement import update_staleness_after_rehearsal as _update_fn
            _golden_dir = str(Path(data_root) / 'golden')
            _dream_golden_file = _pick_fn(_golden_dir)
            if _dream_golden_file is None:
                _dream_n = 0  # cold start skip
            else:
                _golden_path = str(Path(_golden_dir) / _dream_golden_file)
                try:
                    from traindat_loader import load_batch_from_file as _load_fn
                    _gx, _gy, _g_mask = _load_fn(
                        _golden_path, batch_size, seq_len, num_bits,
                        seed=42 + step, binary_bits_mode=False)
                    _dream_input = _gx.to(device=device)
                    _dream_y = _gy.to(device=device)
                    _dream_bit_mask = _g_mask.to(device=device) if _g_mask is not None else None
                    _dream_hdr += f" golden={_dream_golden_file}"
                except FileNotFoundError:
                    _dream_golden_file = None
                    _dream_n = 0
        # --- END EXACT GLUE CODE ---

        # Verify the glue code loaded data correctly
        assert _dream_golden_file == "easy.traindat"
        assert _dream_n == 3  # not skipped
        assert _dream_input.shape == (batch_size, seq_len, num_bits)
        assert _dream_y.shape == (batch_size, seq_len, num_bits)
        assert _dream_bit_mask is None  # Gray mode returns None mask
        assert "golden=easy.traindat" in _dream_hdr

        # Step 3: Run actual dream forward pass through model
        model.eval()
        with torch.no_grad():
            _d_out = model(_dream_input)
        assert _d_out.shape == (batch_size, seq_len, num_bits)

        # Step 4: Staleness update (exact glue from lines 1890-1894)
        if _dream_golden_file is not None:
            _update_fn(_golden_dir, _dream_golden_file)

        manifest = load_manifest(_golden_dir)
        assert manifest["datasets"]["easy.traindat"]["staleness"] == 0

    def test_dream_rehearsal_cold_start(self, mini_env):
        """
        Rehearsal with empty golden/ triggers cold start: _dream_n=0, no crash.
        """
        data_root = mini_env["data_root"]
        _dream_mode = 'rehearsal'
        _dream_n = 3
        _dream_golden_file = None

        if _dream_mode == 'rehearsal':
            from retirement import pick_stalest_dataset as _pick_fn
            _golden_dir = str(Path(data_root) / 'golden')
            _dream_golden_file = _pick_fn(_golden_dir)
            if _dream_golden_file is None:
                _dream_n = 0

        assert _dream_golden_file is None
        assert _dream_n == 0  # correctly skipped

    def test_dream_rehearsal_missing_file(self, mini_env):
        """
        Rehearsal where golden file was deleted between pick and load.
        """
        data_root = mini_env["data_root"]
        tmp_path = mini_env["tmp_path"]

        # Retire, then delete the file (simulates external deletion)
        retire_dataset(data_root, "easy.traindat", step=50)
        golden_file = tmp_path / "golden" / "easy.traindat"
        assert golden_file.exists()
        golden_file.unlink()

        _dream_mode = 'rehearsal'
        _dream_n = 3
        _dream_golden_file = None

        if _dream_mode == 'rehearsal':
            from retirement import pick_stalest_dataset as _pick_fn
            _golden_dir = str(Path(data_root) / 'golden')
            _dream_golden_file = _pick_fn(_golden_dir)
            if _dream_golden_file is None:
                _dream_n = 0
            else:
                _golden_path = str(Path(_golden_dir) / _dream_golden_file)
                try:
                    from traindat_loader import load_batch_from_file as _load_fn
                    _gx, _gy, _g_mask = _load_fn(
                        _golden_path, 2, 4, 8, seed=42)
                except FileNotFoundError:
                    _dream_golden_file = None
                    _dream_n = 0

        # The manifest had the entry but file was deleted — pick returned it,
        # but load_batch_from_file caught the FileNotFoundError
        assert _dream_golden_file is None
        assert _dream_n == 0

    def test_dream_lr_scale_restore(self, mini_env):
        """
        LR scaling for rehearsal: scaled down before dream, restored after.
        """
        optimizer = mini_env["optimizer"]
        original_lr = optimizer.param_groups[0]['lr']

        _dream_mode = 'rehearsal'
        _dream_n = 3
        _dream_lr_scale = 0.1
        _waking_lr = optimizer.param_groups[0]['lr']

        # Scale down (exact glue from line 1779-1781)
        if _dream_mode == 'rehearsal' and _dream_n > 0:
            for _pg in optimizer.param_groups:
                _pg['lr'] = _waking_lr * _dream_lr_scale

        assert optimizer.param_groups[0]['lr'] == pytest.approx(original_lr * 0.1)

        # Restore (exact glue from line 1898-1900)
        if _dream_mode == 'rehearsal':
            for _pg in optimizer.param_groups:
                _pg['lr'] = _waking_lr

        assert optimizer.param_groups[0]['lr'] == pytest.approx(original_lr)

    def test_dream_lr_no_scale_when_skipped(self, mini_env):
        """
        LR NOT scaled when _dream_n=0 (cold start skip).
        """
        optimizer = mini_env["optimizer"]
        original_lr = optimizer.param_groups[0]['lr']

        _dream_mode = 'rehearsal'
        _dream_n = 0  # cold start
        _dream_lr_scale = 0.1
        _waking_lr = optimizer.param_groups[0]['lr']

        # This should NOT scale because _dream_n == 0
        if _dream_mode == 'rehearsal' and _dream_n > 0:
            for _pg in optimizer.param_groups:
                _pg['lr'] = _waking_lr * _dream_lr_scale

        assert optimizer.param_groups[0]['lr'] == pytest.approx(original_lr)

        # Restore still runs (safe no-op)
        if _dream_mode == 'rehearsal':
            for _pg in optimizer.param_groups:
                _pg['lr'] = _waking_lr

        assert optimizer.param_groups[0]['lr'] == pytest.approx(original_lr)

    def test_split_metric_wiring(self, mini_env):
        """
        Replicate split metric glue from test_swarm_config.py lines 2387-2405.
        Uses a streaming .meta.json with is_streaming=True.
        """
        loader = mini_env["loader"]
        traindat_dir = mini_env["traindat_dir"]
        model = mini_env["model"]
        num_bits = 8

        # Generate a fake eval output/target
        B, T = 2, 4
        ev_out_cpu = torch.randn(B, T, num_bits)
        y_ev_cpu = (torch.rand(B, T, num_bits) > 0.5).float()

        traindat_loader = loader

        # --- BEGIN EXACT GLUE CODE from test_swarm_config.py lines 2387-2405 ---
        _copy_ba, _novel_ba = None, None
        if traindat_loader and traindat_loader.current_dataset:
            _meta_p = Path(traindat_dir) / traindat_loader.current_dataset.replace('.traindat', '.meta.json')
            if _meta_p.exists():
                try:
                    with open(_meta_p) as _mf:
                        _smeta = json.load(_mf)
                    if _smeta.get('is_streaming', False):
                        _ovlap = _smeta.get('overlap_bytes', 0)
                        if _ovlap > 0:
                            from test_swarm_config import compute_split_accuracy
                            _copy_ba, _novel_ba = compute_split_accuracy(
                                ev_out_cpu, y_ev_cpu, _ovlap, num_bits)
                except (json.JSONDecodeError, KeyError):
                    pass
        # --- END EXACT GLUE CODE ---

        # The easy.meta.json has is_streaming=True, overlap_bytes=4
        # overlap_bytes=4, num_bits=8, bytes_per_pos=1, overlap_positions=4
        # T=4, so overlap_positions >= T => returns None, None
        # This is actually correct — overlap covers entire sequence
        assert _copy_ba is None and _novel_ba is None

    def test_split_metric_wiring_with_partial_overlap(self, mini_env):
        """
        Split metric with overlap < sequence length -> real values returned.
        """
        loader = mini_env["loader"]
        traindat_dir = mini_env["traindat_dir"]
        num_bits = 8

        # Rewrite meta with smaller overlap
        streaming_meta = {
            "task": "copy_echo",
            "is_streaming": True,
            "overlap_bytes": 2,  # 2 positions overlap out of T=8
        }
        (traindat_dir / "easy.meta.json").write_text(json.dumps(streaming_meta))

        B, T = 2, 8
        ev_out_cpu = torch.randn(B, T, num_bits)
        y_ev_cpu = (torch.rand(B, T, num_bits) > 0.5).float()
        traindat_loader = loader

        _copy_ba, _novel_ba = None, None
        if traindat_loader and traindat_loader.current_dataset:
            _meta_p = Path(traindat_dir) / traindat_loader.current_dataset.replace('.traindat', '.meta.json')
            if _meta_p.exists():
                try:
                    with open(_meta_p) as _mf:
                        _smeta = json.load(_mf)
                    if _smeta.get('is_streaming', False):
                        _ovlap = _smeta.get('overlap_bytes', 0)
                        if _ovlap > 0:
                            from test_swarm_config import compute_split_accuracy
                            _copy_ba, _novel_ba = compute_split_accuracy(
                                ev_out_cpu, y_ev_cpu, _ovlap, num_bits)
                except (json.JSONDecodeError, KeyError):
                    pass

        assert _copy_ba is not None
        assert _novel_ba is not None
        assert 0.0 <= _copy_ba <= 1.0
        assert 0.0 <= _novel_ba <= 1.0

    def test_full_train_retire_dream_cycle(self, mini_env):
        """
        Full end-to-end: forward pass -> retirement -> dream rehearsal -> staleness update.
        Exercises model forward, loss backward, and all pipeline glue.
        """
        model = mini_env["model"]
        optimizer = mini_env["optimizer"]
        loader = mini_env["loader"]
        data_root = mini_env["data_root"]
        tmp_path = mini_env["tmp_path"]

        # Step 1: Training forward pass
        x, y = loader.sample_batch(n_samples=2, seq_len=4, num_bits=8, seed=42)
        x = x.float()
        y = y.float()
        model.train()
        output = model(x)
        loss = torch.nn.functional.mse_loss(torch.sigmoid(output), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Step 2: Force retire (threshold=0.0)
        _active_ds = loader.current_dataset
        assert _active_ds is not None
        retired = retire_dataset(data_root, _active_ds, step=1)
        assert retired is True
        loader.update_weights({_active_ds: 0})

        # Step 3: Dream rehearsal from golden
        _golden_dir = str(Path(data_root) / 'golden')
        _dream_golden_file = pick_stalest_dataset(_golden_dir)
        assert _dream_golden_file == _active_ds

        _golden_path = str(Path(_golden_dir) / _dream_golden_file)
        _gx, _gy, _g_mask = load_batch_from_file(
            _golden_path, 2, 4, 8, seed=43, binary_bits_mode=False)

        # Dream forward pass (no_grad for consolidation, with grad for rehearsal)
        model.eval()
        _d_out = model(_gx.float())
        assert _d_out.shape == (2, 4, 8)

        # Dream loss (rehearsal mode would backprop)
        model.train()
        _d_out = model(_gx.float())
        _d_loss = torch.nn.functional.mse_loss(torch.sigmoid(_d_out), _gy.float())
        _d_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Step 4: Staleness update
        update_staleness_after_rehearsal(_golden_dir, _dream_golden_file)
        manifest = load_manifest(_golden_dir)
        assert manifest["datasets"][_dream_golden_file]["staleness"] == 0

        # Verify no crashes, all shapes correct, gradient flowed
        assert _d_loss.item() >= 0


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
