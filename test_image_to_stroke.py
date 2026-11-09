"""Unit checks for the image-to-contour front end."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from image_to_stroke import extract

ROOT = Path(__file__).resolve().parent


class ImageToStrokeTest(unittest.TestCase):
    def test_alpha_star(self) -> None:
        ext = extract(ROOT / "assets" / "demo_star.png")
        self.assertEqual(ext.info["mode"], "alpha")
        self.assertGreater(ext.mask.mean(), 0.1)
        self.assertGreaterEqual(len(ext.points), 8)
        self.assertLess(np.max(np.abs(ext.points)), 1.01)

    def test_auto_engraving(self) -> None:
        ext = extract(ROOT / "assets" / "fourier_wiki.jpg")
        self.assertEqual(ext.info["mode"], "auto")
        self.assertGreater(ext.mask.mean(), 0.25)
        self.assertGreater(len(ext.points), 80)
        self.assertLess(np.max(np.abs(ext.points)), 1.01)


if __name__ == "__main__":
    unittest.main()
