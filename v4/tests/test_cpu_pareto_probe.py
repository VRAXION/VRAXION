from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from cpu_pareto_probe import _rank_summaries  # type: ignore[import-not-found]


def test_rank_summaries_prefers_bpc_then_acc_then_speed():
    summaries = [
        {"id": "slow_best_bpc", "final_bpc": 4.0, "final_acc": 0.30, "s_per_step": 0.20},
        {"id": "worse_bpc", "final_bpc": 4.1, "final_acc": 0.99, "s_per_step": 0.01},
        {"id": "same_bpc_better_acc", "final_bpc": 4.0, "final_acc": 0.31, "s_per_step": 0.30},
        {"id": "same_bpc_acc_faster", "final_bpc": 4.0, "final_acc": 0.31, "s_per_step": 0.10},
        {"id": "invalid", "final_bpc": None, "final_acc": 0.50, "s_per_step": 0.05},
    ]

    ranked, winner = _rank_summaries(summaries)

    assert winner is not None
    assert winner["id"] == "same_bpc_acc_faster"
    assert [item["id"] for item in ranked] == [
        "same_bpc_acc_faster",
        "same_bpc_better_acc",
        "slow_best_bpc",
        "worse_bpc",
        "invalid",
    ]
    assert ranked[-1]["valid_for_ranking"] is False
