import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)


class TestAntRatioEvalTimeoutZero(unittest.TestCase):
    def test_timeout_zero_disables_subprocess_timeout(self) -> None:
        from tools.ant_ratio_sweep_v0 import _run_capability_eval

        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            tool = repo_root / "Golden Draft" / "tools" / "eval_ckpt_assoc_byte.py"
            tool.parent.mkdir(parents=True, exist_ok=True)
            tool.write_text("# stub", encoding="utf-8")

            run_root = repo_root / "runs" / "assoc"
            run_root.mkdir(parents=True, exist_ok=True)
            report = run_root / "report.json"
            report.write_text("{}", encoding="utf-8")
            (run_root / "vraxion_eval.log").write_text("", encoding="utf-8")
            checkpoint = run_root / "checkpoint_last_good.pt"
            checkpoint.write_bytes(b"fake")

            with mock.patch(
                "tools.ant_ratio_sweep_v0.subprocess.run", return_value=SimpleNamespace(returncode=0)
            ) as mrun:
                got_report, hb_seen = _run_capability_eval(
                    repo_root=repo_root,
                    run_root=run_root,
                    checkpoint=checkpoint,
                    eval_samples=512,
                    batch_size=2,
                    device="cpu",
                    eval_seed_offset=1000003,
                    force_disjoint=True,
                    timeout_s=0,
                    heartbeat_s=60,
                )

            self.assertEqual(got_report, report)
            self.assertFalse(hb_seen)
            _, kwargs = mrun.call_args
            self.assertIsNone(kwargs.get("timeout"))


if __name__ == "__main__":
    unittest.main()
