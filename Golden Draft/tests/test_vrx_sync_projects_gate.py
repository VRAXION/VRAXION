"""Tests for vrx_sync_linear_projects project gating."""

from __future__ import annotations

import unittest

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from tools import vrx_sync_linear_projects as sync


class TestVrxSyncProjectsGate(unittest.TestCase):
    def test_public_roadmap_is_optional(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_project_by_title(owner: str, title: str) -> dict:
            calls.append((owner, title))
            if title == sync.PRIVATE_ARCHIVE_TITLE:
                return {"number": 3, "id": "PVT_FAKE", "visibility": "PRIVATE"}
            if title == sync.PUBLIC_ROADMAP_TITLE:
                raise AssertionError("public roadmap lookup should not be required")
            raise AssertionError(f"unexpected project title: {title}")

        orig = sync._project_by_title
        sync._project_by_title = fake_project_by_title  # type: ignore[assignment]
        try:
            out = sync._ensure_projects("Kenessy")
        finally:
            sync._project_by_title = orig  # type: ignore[assignment]

        self.assertEqual(out.get("number"), 3)
        self.assertIn(("Kenessy", sync.PRIVATE_ARCHIVE_TITLE), calls)

