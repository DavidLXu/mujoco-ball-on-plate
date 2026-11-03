"""End-to-end smoke test: the ball must actually track the circle."""

from __future__ import annotations

import tempfile
import unittest

from simulate import run


class BallOnPlateTest(unittest.TestCase):
    def test_circle_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run(
                duration=12.0,
                render=False,
                outdir=tmp,
                make_preview_gif=False,
                verbose=False,
            )
        metrics = result["metrics"]

        self.assertLess(metrics["steady_rms_mm"], 2.0)
        self.assertLess(metrics["steady_max_mm"], 3.0)
        self.assertLess(metrics["slip_rms_mm_s"], 10.0)
        self.assertLess(metrics["max_torque_nm"], metrics["torque_limit"])
        self.assertFalse(metrics["saturation_fraction"] > 0.0)


if __name__ == "__main__":
    unittest.main()
