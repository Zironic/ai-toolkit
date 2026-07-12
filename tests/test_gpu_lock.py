"""The GPU lock's only real logic is deciding whether a holder is still alive."""

import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import smoke_runtime as _gpu_lock
from smoke_runtime import GpuBusy, gpu_lock


class GpuLockTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(os.environ["TEMP"]) / f"test_gpu_lock_{os.getpid()}.lock"
        os.environ["AI_TOOLKIT_GPU_LOCK_PATH"] = str(self.path)
        os.environ.pop("AI_TOOLKIT_GPU_LOCK", None)
        os.environ.pop("AI_TOOLKIT_GPU_LOCK_WAIT", None)
        self.addCleanup(os.environ.pop, "AI_TOOLKIT_GPU_LOCK_PATH", None)
        self.addCleanup(lambda: self.path.unlink(missing_ok=True))

    def _write_lock(self, **fields):
        record = {"name": "other", "pid": 1, "started": 0.0, "detail": ""}
        record.update(fields)
        self.path.write_text(json.dumps(record), encoding="utf-8")

    def test_acquires_and_releases(self):
        with gpu_lock("smoke_a"):
            self.assertTrue(self.path.exists())
            held = json.loads(self.path.read_text(encoding="utf-8"))
            self.assertEqual(held["pid"], os.getpid())
            self.assertEqual(held["name"], "smoke_a")
        self.assertFalse(self.path.exists())

    def test_refuses_while_a_live_holder_runs(self):
        # A real, live process that is not us: a sleeping child.
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        import psutil

        self._write_lock(
            pid=child.pid, started=psutil.Process(child.pid).create_time()
        )
        with self.assertRaises(GpuBusy) as caught:
            with gpu_lock("smoke_b"):
                pass
        self.assertIn(str(child.pid), str(caught.exception))
        # The live holder's lock must survive our refusal.
        self.assertTrue(self.path.exists())

    def test_reclaims_a_dead_holders_lock(self):
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        self._write_lock(pid=child.pid, started=time.time())
        with gpu_lock("smoke_c"):
            held = json.loads(self.path.read_text(encoding="utf-8"))
            self.assertEqual(held["pid"], os.getpid())

    def test_recycled_pid_is_not_mistaken_for_the_holder(self):
        # Our own pid, but a start time that is not ours: a different process
        # that happens to have inherited the pid. Must count as dead, not alive.
        self._write_lock(pid=os.getpid(), started=1.0)
        self.assertFalse(_gpu_lock._holder_is_alive({"pid": os.getpid(), "started": 1.0}))
        with gpu_lock("smoke_d"):
            held = json.loads(self.path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(
                held["started"], _gpu_lock.psutil.Process().create_time(), places=3
            )

    def test_corrupt_lock_file_is_reclaimed_not_wedged(self):
        self.path.write_text("{not json", encoding="utf-8")
        with gpu_lock("smoke_e"):
            self.assertEqual(
                json.loads(self.path.read_text(encoding="utf-8"))["pid"], os.getpid()
            )

    def test_release_does_not_delete_someone_elses_lock(self):
        with gpu_lock("smoke_f"):
            # Simulate being wrongly reclaimed mid-run: the file is now theirs.
            self._write_lock(pid=999999, started=123.0)
        self.assertTrue(self.path.exists())
        self.assertEqual(
            json.loads(self.path.read_text(encoding="utf-8"))["pid"], 999999
        )

    def test_disabled_by_env_is_a_no_op(self):
        os.environ["AI_TOOLKIT_GPU_LOCK"] = "0"
        self.addCleanup(os.environ.pop, "AI_TOOLKIT_GPU_LOCK", None)
        with gpu_lock("smoke_g"):
            self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
