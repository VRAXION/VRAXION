"""Behavior locks for VRX_MITOSIS environment gating."""

from __future__ import annotations

import os
import unittest

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from tools import env_utils


class MitosisEnvFlagTests(unittest.TestCase):
    def test_env_is_one_parses_strictly(self) -> None:
        with conftest.temporary_env(VRX_MITOSIS=None):
            self.assertFalse(env_utils.env_is_one(os.environ, "VRX_MITOSIS"))

        with conftest.temporary_env(VRX_MITOSIS="0"):
            self.assertFalse(env_utils.env_is_one(os.environ, "VRX_MITOSIS"))

        with conftest.temporary_env(VRX_MITOSIS="1"):
            self.assertTrue(env_utils.env_is_one(os.environ, "VRX_MITOSIS"))

        with conftest.temporary_env(VRX_MITOSIS="true"):
            self.assertFalse(env_utils.env_is_one(os.environ, "VRX_MITOSIS"))


if __name__ == "__main__":
    unittest.main()

