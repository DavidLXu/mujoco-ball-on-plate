#!/usr/bin/env python3
"""Extract a closed silhouette from the public-domain Fourier engraving.

Source: Wikimedia Commons, File:Joseph Fourier.jpg
        engraving by A. F. B. Geille after J. Boilly (public domain).

The extracted outer contour is saved as a closed 2-D point list.  It is the
input to the Fourier-descriptor drawing pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure, morphology

ROOT = Path(__file__).resolve().parent
IMAGE = ROOT / "assets" / "fourier_wiki.jpg"
OUT = ROOT / "assets" / "fourier_portrait_points.npy"
PREVIEW = ROOT / "outputs" / "fourier" / "source_contour.png"


def extract(thresh: int = 220, rdp_tolerance: float = 1.4):
    gray = np.asarray(Image.open(IMAGE).convert("L"))
    mask = gray < thresh
    mask = morphology.remove_small_objects(mask, max_size=50)
    mask = morphology.closing(mask, morphology.disk(3))
    mask = ndi.binary_fill_holes(mask)

    labels, n = ndi.label(mask)
    sizes = ndi.sum(mask, labels, range(1, n + 1))
    mask = labels == (int(np.argmax(sizes)) + 1)
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_holes(mask, max_size=5000)

    contours = sorted(measure.find_contours(mask.astype(float), 0.5), key=len, reverse=True)
    contour = contours[0]
    contour = measure.approximate_polygon(contour, tolerance=rdp_tolerance)
    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]

    # image (row, col) -> mathematical (x, y), y up
    xy = np.c_[contour[:, 1], -contour[:, 0]]

    # Rotate to start at the bottom centre so the closed seam is smooth.
    i = int(np.argmin((xy[:, 0] - 0.5 * (xy[:, 0].min() + xy[:, 0].max())) ** 2
                     + (xy[:, 1] - xy[:, 1].min()) ** 2))
    xy = np.roll(xy, -i, axis=0)

    # Normalise: centre bbox, longest side = 2, so target_span maps it later.
    xy -= np.array([0.5 * (xy[:, 0].min() + xy[:, 0].max()),
                    0.5 * (xy[:, 1].min() + xy[:, 1].max())])
    xy *= 2.0 / max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    xy = np.vstack([xy, xy[0]])
    return xy, mask, gray, contour


def main() -> None:
    points, mask, gray, contour = extract()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, points)
    print(f"saved {len(points) - 1} points -> {OUT}")

    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 8.8))
    ax.imshow(gray, cmap="gray")
    ax.plot(contour[:, 1], contour[:, 0], color="#ef4444", lw=1.2)
    ax.set_axis_off()
    ax.set_title("Source engraving and extracted one-stroke silhouette")
    fig.savefig(PREVIEW, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
