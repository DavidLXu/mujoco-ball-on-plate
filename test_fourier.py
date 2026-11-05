"""Fast unit checks for the Fourier drawing reference."""

from __future__ import annotations

import unittest

import numpy as np

from drawing import FOURIER_PROFILE_POINTS, FourierDrawing, make_fourier_path


class FourierDrawingTest(unittest.TestCase):
    def test_path_is_inside_plate_and_starts_at_rest(self) -> None:
        path = make_fourier_path(
            FOURIER_PROFILE_POINTS, harmonics=40, target_span=0.46, smooth_sigma=40.0
        )
        self.assertLess(np.max(np.abs(path.points[:, 1])), 0.235)
        self.assertLess(np.max(np.abs(path.points[:, 0])), 0.20)

        drawing = FourierDrawing(
            path, a_lat=0.08, a_tan=0.015, v_max=0.06, v_min=0.003
        )
        state = drawing.sample(0.0)
        np.testing.assert_allclose(state.p, drawing.p0, atol=1e-9)
        np.testing.assert_allclose(state.v, 0.0, atol=1e-9)
        self.assertLess(drawing.max_accel, 0.3)


if __name__ == "__main__":
    unittest.main()
