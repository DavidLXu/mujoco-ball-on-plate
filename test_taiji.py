"""Geometry checks for the taiji reference."""

from __future__ import annotations

import unittest

import numpy as np

from taiji_draw import TaijiReference


class TaijiReferenceTest(unittest.TestCase):
    def test_geometry_and_ends(self) -> None:
        ref = TaijiReference(speed=0.045, smooth_sigma=18.0)
        self.assertGreater(ref.L, 3.0)
        self.assertGreater(ref.T, 60.0)
        self.assertLess(ref.max_accel, 0.6)
        self.assertEqual(len(ref.guide), 5)
        np.testing.assert_allclose(ref.sample(0.0).p, ref.p0, atol=1e-9)
        np.testing.assert_allclose(ref.sample(ref.T).p, ref.p0, atol=2e-4)
        # there must be both pen-down and pen-up sections
        self.assertTrue(ref.ink_grid.any())
        self.assertTrue((~ref.ink_grid).any())


if __name__ == "__main__":
    unittest.main()
