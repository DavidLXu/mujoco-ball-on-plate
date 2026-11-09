#!/usr/bin/env python3
"""Image-to-MuJoCo drawing pipeline.

One command takes an arbitrary bitmap to a closed one-stroke drawing:

    image
      -> foreground mask            (alpha / auto background / threshold / mask)
      -> largest external contour
      -> normalised Fourier descriptors
      -> curvature-limited reference
      -> MuJoCo ball-on-plate tracking
      -> 4x dual-view video + plots + metrics

Examples
--------
    python auto_draw.py assets/fourier_wiki.jpg
    python auto_draw.py assets/demo_star.png --harmonics 20
    python auto_draw.py photo.jpg --mask photo_mask.png
    python auto_draw.py logo.png --mode alpha --no-video
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from draw_fourier import run as run_drawing
from image_to_stroke import extract, save_source_contour, slugify


ROOT = Path(__file__).resolve().parent


def _save_comparison(ext, path, ball_path, out_path, title, metrics):
    """Source + extracted contour beside the Fourier/ball reconstruction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plotting import BG, CYAN, MUTED, ORANGE, PANEL, TEXT, _style

    _style()
    fig = plt.figure(figsize=(15, 8.6), facecolor=BG)
    gs = fig.add_gridspec(
        1, 2, left=0.035, right=0.965, top=0.84, bottom=0.12, wspace=0.08
    )

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(ext.image)
    ax.plot(ext.contour[:, 1], ext.contour[:, 0], color="#ef4444", lw=1.4)
    ax.set_axis_off()
    ax.set_title("Source image + extracted silhouette", color=TEXT, fontsize=13, pad=12)
    ax.text(
        0.5, -0.035, f"mask mode: {ext.info.get('mode', '?')}",
        transform=ax.transAxes, ha="center", va="top", color=MUTED, fontsize=9,
    )

    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(PANEL)
    ax.plot(path.points[:, 0], path.points[:, 1], color=CYAN, lw=1.3,
            ls=(0, (4, 4)), alpha=0.9, label=f"{metrics['harmonics']}-harmonic guide")
    ax.plot(ball_path[:, 0], ball_path[:, 1], color=ORANGE, lw=2.0,
            label="ball-drawn path")
    ax.scatter([ball_path[0, 0]], [ball_path[0, 1]], marker="x", s=60,
               color=ORANGE, lw=2, zorder=5)
    ax.set_aspect("equal")
    lim = max(0.235, 1.15 * float(np.max(np.abs(path.points))))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")
    ax.set_title("One-stroke Fourier reconstruction + physical ball track",
                 color=TEXT, fontsize=13, pad=12)
    ax.legend(loc="lower left", frameon=True, facecolor=BG, edgecolor="#334155",
              fontsize=8.5)
    ax.text(
        0.5, -0.035,
        f"K={metrics['harmonics']} · {metrics['path_length_m']:.3f} m · "
        f"{metrics['rms_mm']:.2f} mm RMS · {metrics['drawing_duration_s']:.1f} s",
        transform=ax.transAxes, ha="center", va="top", color=MUTED, fontsize=9,
    )

    fig.suptitle(title, color=TEXT, fontsize=17, fontweight="bold", y=0.965)
    fig.text(
        0.5, 0.915,
        "The ball draws the main external contour; interior image detail is not "
        "part of a single closed stroke.",
        ha="center", color=MUTED, fontsize=10,
    )
    fig.savefig(out_path, dpi=190, facecolor=BG)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="input image (PNG/JPG/WebP)")
    parser.add_argument("--name", default=None, help="display name; defaults to filename")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--mode", choices=["auto", "alpha", "threshold"], default="auto")
    parser.add_argument("--threshold", type=float, default=None,
                        help="threshold for --mode threshold")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--mask", default=None, help="optional foreground mask image")
    parser.add_argument("--max-side", type=int, default=1400)
    parser.add_argument("--tolerance", type=float, default=None,
                        help="RDP contour tolerance in pixels")

    parser.add_argument("--harmonics", type=int, default=40)
    parser.add_argument("--span", type=float, default=0.46)
    parser.add_argument("--smooth", type=float, default=40.0)
    parser.add_argument("--a-lat", type=float, default=0.08)
    parser.add_argument("--a-tan", type=float, default=0.015)
    parser.add_argument("--v-max", type=float, default=0.06)

    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--speed", type=float, default=4.0)
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.exists():
        parser.error(f"input image not found: {image_path}")

    name = args.name or image_path.stem.replace("_", " ").replace("-", " ").strip()
    slug = slugify(name)
    outdir = Path(args.outdir) if args.outdir else ROOT / "outputs" / "auto" / slug
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] extracting contour from {image_path}")
    ext = extract(
        image_path, mode=args.mode, threshold=args.threshold, invert=args.invert,
        mask_path=args.mask, max_side=args.max_side, tolerance=args.tolerance,
    )
    print(f"      mask={ext.info.get('mode')}  area={ext.mask.mean():.3f}  "
          f"contour points={len(ext.points) - 1}")

    # Save the reproducible intermediate artefacts.
    Image.fromarray(ext.image).save(outdir / "source.png")
    Image.fromarray((ext.mask * 255).astype(np.uint8)).save(outdir / "mask.png")
    np.save(outdir / "contour_points.npy", ext.points)
    save_source_contour(ext, outdir / "source_contour.png",
                        f"{name}: extracted one-stroke contour")

    print(f"[2/3] Fourier fitting + MuJoCo control (K={args.harmonics})")
    result = run_drawing(
        points=ext.points,
        source_image=image_path,
        name=name,
        title=f"{name.upper()} · ONE-STROKE",
        prefix="stroke",
        harmonics=args.harmonics,
        span=args.span,
        smooth_sigma=args.smooth,
        speed_cfg={"a_lat": args.a_lat, "a_tan": args.a_tan, "v_max": args.v_max},
        render=not args.no_video,
        width=args.width,
        height=args.height,
        fps=args.fps,
        speed_factor=args.speed,
        outdir=outdir,
        make_preview_gif=not args.no_gif,
        verbose=True,
    )

    metrics = result["metrics"]
    metrics["input_image"] = str(image_path)
    metrics["extraction"] = ext.info
    with open(outdir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    with open(outdir / "metrics.txt", "w", encoding="utf-8") as fh:
        for k, v in metrics.items():
            fh.write(f"{k}: {v}\n")

    # Comparison figure uses the actual ball trajectory from the run.
    _save_comparison(
        ext, result["path"], result["log"]["p"],
        outdir / "comparison.png", f"{name}: image → Fourier → ball",
        metrics,
    )

    print(f"[3/3] done: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
