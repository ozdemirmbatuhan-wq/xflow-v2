from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from aeropt.flow5 import Flow5CancelledError
from server import OptimizationJobs


class OptimizationJobTests(unittest.TestCase):
    def test_cancelled_job_keeps_latest_foil_and_wing_best_snapshots(self):
        foil = {"airfoil": {"name": "foil-best"}, "objective": 0.02}
        wing = {"wing": {"geometry": {"span": 1.8}}, "objective": 0.01}

        def cancelled_run(payload, *, progress_callback, cancel_event):
            progress_callback(
                {
                    "stage": "foil_search",
                    "current": 1,
                    "total": 8,
                    "message": "foil",
                    "best_so_far": {"foil": foil},
                }
            )
            progress_callback(
                {
                    "stage": "wing_search",
                    "current": 1,
                    "total": 8,
                    "message": "wing",
                    "best_so_far": {"wing": wing},
                }
            )
            progress_callback(
                {
                    "stage": "wing_search",
                    "current": 2,
                    "total": 8,
                    "message": "no replacement",
                }
            )
            raise Flow5CancelledError("stopped")

        jobs = OptimizationJobs()
        with patch("server.run_design", side_effect=cancelled_run):
            created = jobs.create({})
            job_id = str(created["id"])
            deadline = time.monotonic() + 2.0
            snapshot = jobs.snapshot(job_id)
            while snapshot and snapshot["status"] not in {"cancelled", "failed"}:
                if time.monotonic() >= deadline:
                    self.fail("background optimization job did not stop")
                time.sleep(0.01)
                snapshot = jobs.snapshot(job_id)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["status"], "cancelled")
        self.assertEqual(snapshot["best_so_far"], {"foil": foil, "wing": wing})
        self.assertNotIn("best_so_far", snapshot["progress"])
        self.assertIn("ara sonuç hazır", snapshot["progress"]["message"])


if __name__ == "__main__":
    unittest.main()
