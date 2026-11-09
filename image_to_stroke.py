"""Turn an arbitrary bitmap into a closed one-stroke contour.

Pipeline
--------
    image -> foreground mask -> largest external contour -> normalised points

The foreground can come from an alpha channel, a supplied mask, a manual
threshold, or an automatic border-background segmentation.  The automatic mode
works best for logos, line art, silhouettes, and objects photographed against a
reasonably clean background.  For a busy photo, pass a mask with --mask (or use
--threshold).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage import filters, measure, morphology


@dataclass
class Extraction:
    image: np.ndarray          # RGB uint8, resized working image
    alpha: np.ndarray | None   # optional alpha
    mask: np.ndarray           # foreground mask
    contour: np.ndarray        # (M, 2) row, col in working image
    points: np.ndarray         # (M+1, 2) normalised, y-up, closed
    info: dict


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text or "image"


def load_image(path: str | Path, max_side: int = 1400) -> tuple[np.ndarray, np.ndarray | None]:
    im = ImageOps.exif_transpose(Image.open(path))
    alpha = None
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        alpha = np.asarray(im.getchannel("A"), dtype=np.uint8)
        # For previews, put transparent images on a light neutral background.
        bg = Image.new("RGBA", im.size, (248, 248, 248, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        im = im.convert("RGB")

    w, h = im.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        new = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        im = im.resize(new, Image.Resampling.LANCZOS)
        if alpha is not None:
            alpha = np.asarray(
                Image.fromarray(alpha).resize(new, Image.Resampling.LANCZOS)
            )
    rgb = np.asarray(im, dtype=np.uint8)
    if alpha is not None and alpha.max() == 0:
        alpha = None
    return rgb, alpha


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    area = mask.size
    # A relative closing radius bridges the gaps between engraving/line-art
    # strokes and turns them into a solid silhouette, while leaving the large
    # shape intact.
    radius = max(2, int(round(min(mask.shape) * 0.010)))
    mask = morphology.closing(mask, morphology.disk(radius))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, max_size=max(16, int(area * 0.0002)))
    labels, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, labels, range(1, n + 1))
    mask = labels == (int(np.argmax(sizes)) + 1)
    mask = morphology.remove_small_holes(mask, max_size=max(32, int(area * 0.001)))
    return ndi.binary_fill_holes(mask)


def _border_mask(shape: tuple[int, int], frac: float = 0.025) -> np.ndarray:
    h, w = shape
    bw = max(2, int(round(min(h, w) * frac)))
    m = np.zeros((h, w), dtype=bool)
    m[:bw, :] = True
    m[-bw:, :] = True
    m[:, :bw] = True
    m[:, -bw:] = True
    return m


def extract_mask(
    rgb: np.ndarray,
    alpha: np.ndarray | None = None,
    mode: str = "auto",
    threshold: float | None = None,
    invert: bool = False,
    mask_path: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Return a cleaned foreground mask and an info dictionary."""
    if mask_path is not None:
        m = np.asarray(Image.open(mask_path).convert("L"))
        if m.shape != rgb.shape[:2]:
            m = np.asarray(
                Image.fromarray(m).resize(
                    (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST
                )
            )
        mask = m > 127
        info = {"mode": "mask", "threshold": 127}
    elif mode == "alpha":
        if alpha is None:
            raise ValueError("image has no alpha channel; use --mode auto or threshold")
        mask = alpha > 8
        info = {"mode": "alpha", "threshold": 8}
    elif mode == "threshold":
        if threshold is None:
            threshold = 200
        gray = np.asarray(Image.fromarray(rgb).convert("L"))
        mask = gray < float(threshold)
        info = {"mode": "threshold", "threshold": float(threshold)}
    elif mode == "auto":
        if alpha is not None and 0 < float(np.mean(alpha > 8)) < 0.999:
            mask = alpha > 8
            info = {"mode": "alpha", "threshold": 8}
        else:
            border = _border_mask(rgb.shape[:2])
            border_pixels = rgb[border].astype(float)
            bg = np.median(border_pixels, axis=0)
            dist = np.linalg.norm(rgb.astype(float) - bg, axis=2)
            try:
                th = float(filters.threshold_otsu(dist))
            except ValueError:
                th = float(np.percentile(dist, 90))

            cand_fg = dist > th
            cand_bg = ~cand_fg
            # Pick the polarity whose foreground occupies less of the image
            # border.  This handles both dark objects on white paper and light
            # objects on a dark background.
            def score(c: np.ndarray) -> float:
                area = float(np.mean(c))
                border_frac = float(np.mean(c[border]))
                penalty = 0.0
                if area < 0.005 or area > 0.97:
                    penalty += 2.0
                if area < 0.02:
                    penalty += 0.3
                return border_frac + penalty

            if score(cand_fg) <= score(cand_bg):
                mask = cand_fg
                polarity = "dist>th"
            else:
                mask = cand_bg
                polarity = "dist<=th"
            info = {
                "mode": "auto",
                "threshold": th,
                "background_rgb": bg.tolist(),
                "polarity": polarity,
            }
    else:
        raise ValueError(f"unknown mask mode: {mode}")

    if invert:
        mask = ~mask
        info["invert"] = True

    mask = _clean_mask(mask)
    area_frac = float(np.mean(mask))
    info["foreground_fraction"] = area_frac
    if area_frac < 0.002:
        raise ValueError(
            "foreground mask is almost empty; try --mode threshold/--threshold, "
            "--invert, or --mask"
        )
    if area_frac > 0.985:
        raise ValueError(
            "foreground fills nearly the whole image; try --invert or --mask"
        )
    return mask, info


def contour_from_mask(
    mask: np.ndarray, tolerance: float | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return the largest outer contour (row,col) and normalised closed points."""
    contours = sorted(
        measure.find_contours(mask.astype(float), 0.5), key=len, reverse=True
    )
    if not contours or len(contours[0]) < 16:
        raise ValueError("no usable contour found")
    contour = contours[0]
    tol = tolerance if tolerance is not None else max(1.0, 0.0018 * max(mask.shape))
    contour = measure.approximate_polygon(contour, tolerance=tol)
    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    if len(contour) < 8:
        raise ValueError("contour is too simple; lower --tolerance")

    # image (row, col) -> mathematical (x, y), y up
    xy = np.c_[contour[:, 1], -contour[:, 0]].astype(float)
    xy -= np.array(
        [0.5 * (xy[:, 0].min() + xy[:, 0].max()),
         0.5 * (xy[:, 1].min() + xy[:, 1].max())]
    )
    span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
    xy *= 2.0 / span

    # Start at the bottom-centre so the closed seam is on a benign section.
    bottom = int(np.argmin((xy[:, 0]) ** 2 + (xy[:, 1] - xy[:, 1].min()) ** 2))
    xy = np.roll(xy, -bottom, axis=0)
    xy = np.vstack([xy, xy[0]])
    info = {
        "contour_vertices": int(len(contour)),
        "rdp_tolerance_px": float(tol),
        "normalised_bbox": [xy.min(0).tolist(), xy.max(0).tolist()],
    }
    return contour, xy, info


def extract(
    path: str | Path,
    mode: str = "auto",
    threshold: float | None = None,
    invert: bool = False,
    mask_path: str | Path | None = None,
    max_side: int = 1400,
    tolerance: float | None = None,
) -> Extraction:
    rgb, alpha = load_image(path, max_side=max_side)
    mask, info = extract_mask(
        rgb, alpha, mode=mode, threshold=threshold, invert=invert, mask_path=mask_path
    )
    contour, points, cinfo = contour_from_mask(mask, tolerance=tolerance)
    info.update(cinfo)
    return Extraction(
        image=rgb, alpha=alpha, mask=mask, contour=contour, points=points, info=info
    )


def save_source_contour(ext: Extraction, path: str | Path, title: str) -> Path:
    """Save a publication-style source image with the extracted contour."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    fig, ax = plt.subplots(figsize=(7.2, 8.8))
    ax.imshow(ext.image)
    ax.plot(ext.contour[:, 1], ext.contour[:, 0], color="#ef4444", lw=1.25)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
