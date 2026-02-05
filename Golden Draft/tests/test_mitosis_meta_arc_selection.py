"""Behavior locks for arc-safe selection helpers (static-space v1.2)."""

from __future__ import annotations

import unittest

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from tools.mitosis_meta_from_eval import (
    best_circular_window_start,
    expert_counts_from_router_map,
    main as meta_main,
    parent_only_addresses_in_arc,
)


class MitosisMetaArcSelectionTests(unittest.TestCase):
    def test_topk_loss_policy_is_deferred(self) -> None:
        rc = meta_main(["--checkpoint", "nonexistent.pt", "--output", "out.json", "--address-policy", "topk_loss"])
        self.assertEqual(rc, 2)

    def test_best_circular_window_tie_breaks_lowest_start(self) -> None:
        # Two equal maxima at starts 0 and 2; choose 0.
        start, total = best_circular_window_start([5, 0, 5, 0], window=1)
        self.assertEqual(start, 0)
        self.assertEqual(total, 5)

    def test_best_circular_window_wrap_is_considered(self) -> None:
        # Best window is the wrap-around window starting at index 3: 9 + 9 = 18.
        start, total = best_circular_window_start([9, 0, 0, 9], window=2)
        self.assertEqual(start, 3)
        self.assertEqual(total, 18)

    def test_parent_only_addresses_filters_non_parent(self) -> None:
        router_map = [0, 0, 1, 0, 1]
        addrs = parent_only_addresses_in_arc(router_map, parent=0, start=0, length=5)
        self.assertEqual(addrs, [0, 1, 3])

    def test_parent_only_addresses_preserves_arc_order_with_wrap(self) -> None:
        router_map = [0, 1, 0, 0]
        addrs = parent_only_addresses_in_arc(router_map, parent=0, start=3, length=3)
        # Arc indices: 3,0,1 -> parent-only = 3,0
        self.assertEqual(addrs, [3, 0])

    def test_expert_counts_aggregates_visit_counts(self) -> None:
        router_map = [0, 0, 1, 1]
        counts = [10, 1, 2, 3]
        out = expert_counts_from_router_map(router_map, counts, num_experts=2)
        self.assertEqual(out, [11, 5])


if __name__ == "__main__":
    unittest.main()
