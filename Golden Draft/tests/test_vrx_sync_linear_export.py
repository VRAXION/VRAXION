"""Tests for vrx_sync_linear_projects Linear export parsing."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from tools import vrx_sync_linear_projects as sync


class TestVrxSyncLinearExport(unittest.TestCase):
    def _write_export(self, obj: dict) -> Path:
        fd, name = tempfile.mkstemp(prefix="vrx_linear_export_", suffix=".json")
        os.close(fd)
        p = Path(name)
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        self.addCleanup(lambda: p.unlink(missing_ok=True))
        return p

    def test_parses_attachments_shapes(self) -> None:
        export = {
            "issues": [
                {
                    "identifier": "VRA-1",
                    "title": "one",
                    "url": "https://linear.app/x/issue/VRA-1",
                    "description": "desc",
                    "project": "VRAXION_WALL",
                    "state": "Backlog",
                    "labels": ["L1", "L2"],
                    "attachments": [{"title": "PR", "url": "https://github.com/VRAXION/VRAXION/pull/1"}],
                    "blocked_by": [{"identifier": "VRA-9", "title": "block"}],
                    "blocks": [{"identifier": "VRA-10", "title": "blocks"}],
                },
                {
                    "identifier": "VRA-2",
                    "title": "two",
                    "url": "https://linear.app/x/issue/VRA-2",
                    "description": "",
                    "project": "VRAXION_IDEAS",
                    "state": "Todo",
                    "attachments": [["A", "http://a"], ["B", "http://b"]],
                },
                {
                    # PowerShell ConvertTo-Json can collapse a 1-item list-of-pairs into a
                    # single ["title","url"] pair.
                    "identifier": "VRA-3",
                    "title": "three",
                    "url": "https://linear.app/x/issue/VRA-3",
                    "description": "",
                    "project": "VRAXION_IDEAS",
                    "state": "Done",
                    "attachments": ["Single", "http://single"],
                },
            ]
        }

        path = self._write_export(export)
        issues = sync._load_linear_export(path)
        self.assertEqual([i.identifier for i in issues], ["VRA-1", "VRA-2", "VRA-3"])

        self.assertEqual(issues[0].labels, ("L1", "L2"))
        self.assertEqual(issues[0].attachments, (("PR", "https://github.com/VRAXION/VRAXION/pull/1"),))
        self.assertEqual(issues[0].blocked_by[0].identifier, "VRA-9")
        self.assertEqual(issues[0].blocks[0].identifier, "VRA-10")

        self.assertEqual(issues[1].attachments, (("A", "http://a"), ("B", "http://b")))
        self.assertEqual(issues[2].attachments, (("Single", "http://single"),))
