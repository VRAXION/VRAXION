import time
import unittest

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)


class TestEvalCkptHeartbeat(unittest.TestCase):
    def test_heartbeat_emits_pulses(self) -> None:
        from tools.eval_ckpt_assoc_byte import _start_eval_heartbeat

        logs = []

        def _log(msg: str) -> None:
            logs.append(str(msg))

        stop, thread = _start_eval_heartbeat(_log, 1)
        try:
            time.sleep(2.2)
        finally:
            stop.set()
            thread.join(timeout=1.0)

        hb = [line for line in logs if "[eval_ckpt][heartbeat]" in line]
        self.assertGreaterEqual(len(hb), 2)


if __name__ == "__main__":
    unittest.main()
