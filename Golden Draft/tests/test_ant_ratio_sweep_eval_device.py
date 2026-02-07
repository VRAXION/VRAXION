import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)


class TestAntRatioSweepEvalDevice(unittest.TestCase):
    def test_eval_device_is_passed_and_report_is_artifact_truth(self) -> None:
        from tools.ant_ratio_sweep_v0 import _run_capability_eval

        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)

            # Stub out the eval tool path existence check.
            tool = repo_root / "Golden Draft" / "tools" / "eval_ckpt_assoc_byte.py"
            tool.parent.mkdir(parents=True, exist_ok=True)
            tool.write_text("# stub", encoding="utf-8")

            # run_root must contain report.json for artifact-truth behavior.
            run_root = repo_root / "runs" / "assoc"
            run_root.mkdir(parents=True, exist_ok=True)
            rep = run_root / "report.json"
            rep.write_text('{"eval_acc": 0.0, "eval_n": 512}', encoding="utf-8")
            (run_root / "vraxion_eval.log").write_text(
                "[eval_ckpt][heartbeat] stage=eval elapsed_s=1 interval_s=60 pulse=1\n",
                encoding="utf-8",
            )

            ckpt = run_root / "checkpoint_last_good.pt"
            ckpt.write_bytes(b"fake")

            with mock.patch(
                "tools.ant_ratio_sweep_v0.subprocess.run", return_value=SimpleNamespace(returncode=123)
            ) as mrun:
                got, hb_seen = _run_capability_eval(
                    repo_root=repo_root,
                    run_root=run_root,
                    checkpoint=ckpt,
                    eval_samples=512,
                    batch_size=3,
                    device="cpu",
                    eval_seed_offset=1000003,
                    force_disjoint=True,
                    timeout_s=1,
                    heartbeat_s=60,
                )

            self.assertEqual(got, rep)
            self.assertTrue(hb_seen)
            args, kwargs = mrun.call_args
            cmd = args[0]
            # cmd is a list of strings. Ensure '--device cpu' is present.
            self.assertIn("--device", cmd)
            didx = cmd.index("--device")
            self.assertEqual(cmd[didx + 1], "cpu")
            hidx = cmd.index("--heartbeat-s")
            self.assertEqual(cmd[hidx + 1], "60")


if __name__ == "__main__":
    unittest.main()
