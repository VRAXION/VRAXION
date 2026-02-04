import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tools.gpu_env_dump as gpu_env_dump


class TestGpuEnvDump(unittest.TestCase):
    def test_schema_key_set_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            with patch.object(gpu_env_dump.importlib, "import_module", side_effect=ImportError("no torch")):
                with patch.object(gpu_env_dump.shutil, "which", return_value=None):
                    env_path = gpu_env_dump.write_env_json(out_dir=out_dir, precision="fp16", amp=1)

            data = json.loads(env_path.read_text(encoding="utf-8"))
            self.assertEqual(set(data.keys()), set(gpu_env_dump.ENV_KEYS_V1))
            self.assertEqual(data["env_schema_version"], "v1")
            self.assertIsInstance(data["errors"], list)
            self.assertIn(data["amp"], (None, 0, 1))
            self.assertIn(data["git_dirty"], (None, 0, 1))

    def test_ascii_safe_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            with patch.object(gpu_env_dump.shutil, "which", return_value=None):
                env_path = gpu_env_dump.write_env_json(out_dir=out_dir, precision="unknown", amp=None)

            raw = env_path.read_bytes()
            self.assertTrue(all(b < 128 for b in raw), "env.json must be ASCII-safe (ensure_ascii=True)")

    def test_nvidia_smi_missing_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)

            def which_side_effect(name: str) -> None:
                if name == "nvidia-smi":
                    return None
                return None

            with patch.object(gpu_env_dump.shutil, "which", side_effect=which_side_effect):
                env_path = gpu_env_dump.write_env_json(out_dir=out_dir, precision="fp32", amp=0)

            data = json.loads(env_path.read_text(encoding="utf-8"))
            self.assertIn("nvidia_smi_not_found", data["errors"])
            self.assertIsNone(data["nvidia_smi_query"])

    def test_torch_import_failure_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            with patch.object(gpu_env_dump.importlib, "import_module", side_effect=ImportError("no torch")):
                env_path = gpu_env_dump.write_env_json(out_dir=out_dir, precision="fp16", amp=1)

            data = json.loads(env_path.read_text(encoding="utf-8"))
            self.assertIsNone(data["torch_version"])
            self.assertTrue(any(e.startswith("torch_import_failed:") for e in data["errors"]))

    def test_git_not_repo_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            fake_root = Path(td) / "not_a_repo"
            fake_root.mkdir(parents=True, exist_ok=True)

            with patch.object(gpu_env_dump, "_repo_root", return_value=fake_root):
                env_path = gpu_env_dump.write_env_json(out_dir=out_dir, precision="unknown", amp=None)

            data = json.loads(env_path.read_text(encoding="utf-8"))
            # Any non-empty git_error is acceptable; we just require no crash and stable schema.
            self.assertIsInstance(data["git_error"], (str, type(None)))

    def test_atomic_write_no_tmp_left_behind(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            env_path = gpu_env_dump.write_env_json(out_dir=out_dir, precision="unknown", amp=None)
            self.assertTrue(env_path.exists())
            self.assertFalse((out_dir / "env.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()

