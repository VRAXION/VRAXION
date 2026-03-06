from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
if str(ROOT / "model") not in sys.path:
    sys.path.insert(0, str(ROOT / "model"))

from nightly_research_runner import (  # type: ignore[import-not-found]
    SURFACES,
    VARIANTS,
    _build_meta,
    _effective_global_flags,
    _ring_trace_guard,
    _surface_guards,
)
from instnct import func_linear_pointer_window_tns, func_shortest_arc_delta_tns  # type: ignore[import-not-found]


def _fake_trace(batch=2, seq=4, steps=3, M=16, read_width=3, write_width=3):
    ptr_steps = steps * seq
    center_sum = steps * seq * batch
    return {
        "ring_trace_summary": {
            "ptr_unique_frac": seq / M,
            "read_unique_frac": 0.25,
            "write_unique_frac": 0.25,
            "read_center_dist_mean": 0.67,
            "write_center_dist_mean": 0.67,
            "read_write_overlap_mean": 1.0,
        },
        "ring_trace": {
            "ptr_trace": [i % seq for i in range(ptr_steps)],
            "read_idx_trace": [[0, 1, 2] for _ in range(ptr_steps)],
            "write_idx_trace": [[0, 1, 2] for _ in range(ptr_steps)],
            "tap_idx_trace": [[15, 14] for _ in range(ptr_steps)],
            "read_weight_trace": [[0.2, 0.6, 0.2] for _ in range(ptr_steps)],
            "write_weight_trace": [[0.2, 0.6, 0.2] for _ in range(ptr_steps)],
            "read_write_overlap_trace": [1.0 for _ in range(ptr_steps)],
            "center_hist": [center_sum // seq for _ in range(seq)] + [0 for _ in range(M - seq)],
            "read_hist": [center_sum * read_width // seq for _ in range(seq)] + [0 for _ in range(M - seq)],
            "tap_hist": [center_sum * 2 // seq for _ in range(seq)] + [0 for _ in range(M - seq)],
            "write_hist": [center_sum * write_width // seq for _ in range(seq)] + [0 for _ in range(M - seq)],
        },
    }


def test_surface_and_variant_presets_exist():
    assert set(SURFACES) == {"small_wikitext_fresh", "fast_memory_carry", "wikitext_sequential_carry"}
    assert set(VARIANTS) == {"LL", "LLT", "GL", "GG"}


def test_small_fresh_pointer_guard_passes_on_capped_trace():
    meta = _build_meta("small_wikitext_fresh", "LL", SURFACES["small_wikitext_fresh"])
    result = _fake_trace(
        batch=meta["seq"],  # dummy, not used for guard itself
        seq=meta["seq"],
        steps=2,
        M=meta["ring_slots"],
    )
    guards = _surface_guards("small_wikitext_fresh", result, meta)
    assert guards["fresh_pointer_coverage_capped"] is True


def test_sequential_carry_pointer_guard_requires_broader_coverage():
    meta = _build_meta("wikitext_sequential_carry", "LL", SURFACES["wikitext_sequential_carry"])
    result = _fake_trace(batch=2, seq=meta["seq"], steps=16, M=meta["ring_slots"])
    result["ring_trace_summary"]["ptr_unique_frac"] = 1.0
    guards = _surface_guards("wikitext_sequential_carry", result, meta)
    assert guards["carry_pointer_coverage_exceeds_fresh_bound"] is True


def test_trace_guard_detects_expected_histogram_shapes():
    result = _fake_trace(batch=2, seq=4, steps=3, M=16, read_width=3, write_width=3)
    guards = _ring_trace_guard(result, batch=2, seq=4, steps=3)
    assert guards["ptr_steps_ok"] is True
    assert guards["center_hist_ok"] is True
    assert guards["read_hist_ok"] is True
    assert guards["write_hist_ok"] is True


def test_effective_global_flags_require_nonlocal_telemetry():
    result = {
        "topk_outside_local_frac": 0.75,
        "write_topk_outside_local_frac": 0.72,
        "ring_trace_summary": {
            "read_center_dist_mean": 5.0,
            "write_center_dist_mean": 5.0,
        },
    }
    flags = _effective_global_flags(result, "GG")
    assert flags["effective_global_read"] is True
    assert flags["effective_global_write"] is True


def test_linear_pointer_window_preserves_integer_case():
    import torch

    ptr = torch.tensor([2.0, 5.0])
    offsets = torch.tensor([-1, 0, 1], dtype=torch.long)
    weights = torch.tensor([[0.2, 0.6, 0.2], [0.2, 0.6, 0.2]])
    center, alpha, idx, merged_w = func_linear_pointer_window_tns(ptr, offsets, weights, 8)

    assert torch.equal(center, torch.tensor([2, 5]))
    assert torch.allclose(alpha, torch.zeros_like(alpha))
    assert torch.equal(idx[0], torch.tensor([1, 2, 3, 4]))
    assert torch.equal(idx[1], torch.tensor([4, 5, 6, 7]))
    assert torch.allclose(merged_w[:, :3], weights)
    assert torch.allclose(merged_w[:, 3], torch.zeros(2))


def test_shortest_arc_delta_wraps_across_seam():
    import torch

    current = torch.tensor([63.75, 0.25], dtype=torch.float32)
    target = torch.tensor([0.25, 63.75], dtype=torch.float32)
    delta = func_shortest_arc_delta_tns(current, target, 64)

    assert torch.allclose(delta, torch.tensor([0.5, -0.5]))
