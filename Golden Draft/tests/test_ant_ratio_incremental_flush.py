import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)


class TestAntRatioIncrementalFlush(unittest.TestCase):
    def test_flush_writes_after_each_config(self) -> None:
        from tools import ant_ratio_sweep_v0 as sweep
        from tools import ant_ratio_packet_v0 as pktmod
        from tools import ant_ratio_plot_v0 as pltmod

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            out_root = td_path / "out"
            batch_targets = td_path / "batch_targets.json"
            batch_targets.write_text(
                json.dumps(
                    {
                        "rows": [
                            {"ant_tier": "small", "expert_heads": 1, "chosen_batch": 2, "unusable": False},
                            {"ant_tier": "real", "expert_heads": 4, "chosen_batch": 3, "unusable": False},
                        ]
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            def _fake_probe(**kwargs):
                out_dir = Path(kwargs["out_dir"])
                if "real_E4" in str(out_dir):
                    raise sweep.SweepError("forced probe failure")
                out_dir.mkdir(parents=True, exist_ok=True)
                return out_dir, {"stability_pass": True, "had_oom": False, "had_nan": False, "had_inf": False}

            def _fake_train(**kwargs):
                run_root = Path(kwargs["run_root"])
                run_root.mkdir(parents=True, exist_ok=True)
                ckpt = run_root / "checkpoint_last_good.pt"
                ckpt.write_bytes(b"fake")
                return ckpt, 0

            def _fake_eval(**kwargs):
                run_root = Path(kwargs["run_root"])
                rep = run_root / "report.json"
                rep.write_text("{}", encoding="utf-8")
                return rep, True

            def _fake_packet(**kwargs):
                return {
                    "schema_version": "ant_ratio_packet_v0",
                    "git_commit": "fake",
                    "generated_utc": "2026-02-07T00:00:00Z",
                    "ant_tier": kwargs.get("ant_tier_override", "small"),
                    "expert_heads": 1,
                    "batch_size": 2,
                    "precision": "fp16",
                    "amp": 1,
                    "vram_ratio_reserved": 0.84,
                    "throughput_tokens_per_s": 100.0,
                    "assoc_byte_disjoint_accuracy": 0.12,
                    "assoc_eval_n": 512,
                    "token_budget_steps": int(kwargs.get("capability_steps_override", 30)),
                    "probe_run_root": str(kwargs.get("probe_run_root")),
                    "assoc_run_root": str(kwargs.get("assoc_run_root")),
                    "stability_pass": True,
                    "fail_reasons": [],
                }

            real_write_csv = sweep._write_csv
            with (
                mock.patch.dict(sys.modules, {"ant_ratio_packet_v0": pktmod, "ant_ratio_plot_v0": pltmod}),
                mock.patch("tools.ant_ratio_sweep_v0._repo_root", return_value=td_path),
                mock.patch("tools.ant_ratio_sweep_v0._run_probe", side_effect=_fake_probe),
                mock.patch("tools.ant_ratio_sweep_v0._run_capability_train", side_effect=_fake_train),
                mock.patch("tools.ant_ratio_sweep_v0._run_capability_eval", side_effect=_fake_eval),
                mock.patch.object(pktmod, "build_packet", side_effect=_fake_packet),
                mock.patch.object(pltmod, "build_html", return_value="<html></html>"),
                mock.patch("tools.ant_ratio_sweep_v0._write_csv", wraps=real_write_csv) as mwrite_csv,
            ):
                rc = sweep.main(
                    [
                        "--batch-targets",
                        str(batch_targets),
                        "--out-root",
                        str(out_root),
                        "--flush-every-config",
                        "1",
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertGreaterEqual(mwrite_csv.call_count, 3)

            summary = out_root / "ant_ratio_summary.csv"
            packets = out_root / "ant_ratio_packets.jsonl"
            failures = out_root / "sweep_failures.json"
            self.assertTrue(summary.exists())
            self.assertTrue(packets.exists())
            self.assertTrue(failures.exists())

            with summary.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(sum(1 for r in rows if r.get("status") == "ok"), 1)
            self.assertEqual(sum(1 for r in rows if r.get("status") == "error"), 1)

            packet_lines = [ln for ln in packets.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(packet_lines), 1)
            fail_obj = json.loads(failures.read_text(encoding="utf-8"))
            self.assertEqual(len(fail_obj.get("failures", [])), 1)


if __name__ == "__main__":
    unittest.main()
