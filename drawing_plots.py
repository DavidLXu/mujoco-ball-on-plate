"""Plots for the arbitrary-shape / Fourier one-stroke drawing demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotting import BG, CYAN, GREEN, GRID, MUTED, ORANGE, PANEL, PINK, TEXT, _style


def make_drawing_plots(log: dict, metrics: dict, path: str | Path, title: str | None = None) -> Path:
    _style()
    path = Path(path)
    t = log["t"]
    p = log["p"]
    p_ref = log["p_ref"]

    fig = plt.figure(figsize=(14.5, 10.5))
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.97, top=0.90, bottom=0.075,
                          hspace=0.30, wspace=0.22)

    # 1. drawn portrait
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(p_ref[:, 0], p_ref[:, 1], color=CYAN, lw=1.0, ls=(0, (4, 5)),
            alpha=0.55, label="Fourier guide")
    ax.plot(p[:, 0], p[:, 1], color=ORANGE, lw=2.1, label="ball path", zorder=3)
    ax.scatter(p_ref[0, 0], p_ref[0, 1], marker="x", s=50, color=MUTED, lw=2,
               zorder=4, label="start / finish")
    ax.set_aspect("equal", adjustable="box")
    lim = 0.225
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")
    ax.set_title("One-stroke Fourier portrait  (plate frame)")
    ax.legend(loc="lower right", frameon=True, facecolor=BG, edgecolor="#334155")
    ax.text(0.02, 0.02,
            f"RMS {metrics['rms_mm']:.2f} mm   max {metrics['max_mm']:.2f} mm",
            transform=ax.transAxes, color=TEXT, fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.42", fc="#020617", ec="#334155", alpha=0.85))

    # 2. tracking error
    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0, color="#475569", lw=0.8)
    ax.plot(t, log["err"] * 1000, color=GREEN, lw=1.3)
    ax.fill_between(t, 0, log["err"] * 1000, color=GREEN, alpha=0.10)
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("position error  [mm]")
    ax.set_title("Tracking error")
    ax.grid(True, alpha=0.55)
    ax.set_xlim(0, t[-1])
    ax.text(0.02, 0.06,
            f"RMS {metrics['rms_mm']:.2f} mm    p95 {metrics['p95_mm']:.2f} mm    "
            f"max {metrics['max_mm']:.2f} mm",
            transform=ax.transAxes, color=MUTED, fontsize=9)

    # 3. plate attitude
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, np.degrees(log["pitch"]), color=ORANGE, lw=1.5, label="pitch (actual)")
    ax.plot(t, np.degrees(log["pitch_ref"]), color=ORANGE, lw=1.0, ls="--",
            alpha=0.55, label="pitch (command)")
    ax.plot(t, np.degrees(log["roll"]), color=CYAN, lw=1.5, label="roll (actual)")
    ax.plot(t, np.degrees(log["roll_ref"]), color=CYAN, lw=1.0, ls="--",
            alpha=0.55, label="roll (command)")
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("plate angle  [deg]")
    ax.set_title("Two-axis plate actuation")
    ax.grid(True, alpha=0.55)
    ax.legend(loc="upper right", ncol=2)
    ax.set_xlim(0, t[-1])

    # 4. torques
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, log["tau_pitch"], color=ORANGE, lw=1.4, label="pitch motor")
    ax.plot(t, log["tau_roll"], color=CYAN, lw=1.4, label="roll motor")
    y_lim = max(0.25, 1.35 * float(metrics["max_torque_nm"]))
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("joint torque  [N m]")
    ax.set_title("Actuator effort")
    ax.grid(True, alpha=0.55)
    ax.legend(loc="upper right", ncol=2)
    ax.set_xlim(0, t[-1])
    ax.text(0.02, 0.06,
            f"peak |tau|  {metrics['max_torque_nm']:.3f} N m    "
            f"limit ±{metrics['torque_limit']:.1f} N m",
            transform=ax.transAxes, color=MUTED, fontsize=9)

    fig.suptitle(
        title or "Arbitrary-shape control  ·  ball draws a one-stroke image",
        color=TEXT, fontsize=16, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.925,
        f"{metrics['harmonics']} Fourier harmonics · curvature-limited speed · "
        "torque-controlled 2-axis plate · passive rolling ball",
        color=MUTED, fontsize=9.5, ha="center",
    )

    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def make_fourier_analysis(
    raw_points: np.ndarray,
    fourier_path,
    drawing,
    metrics: dict,
    path: str | Path,
    title: str | None = None,
) -> Path:
    _style()
    path = Path(path)
    k, mag = fourier_path.spectrum()
    energy = fourier_path.cumulative_energy() * 100.0

    fig = plt.figure(figsize=(14.5, 10.0))
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.97, top=0.89, bottom=0.08,
                          hspace=0.33, wspace=0.24)

    # 1. control polygon vs reconstruction
    ax = fig.add_subplot(gs[0, 0])
    raw = np.asarray(raw_points, dtype=float)
    raw = (raw - fourier_path.input_center) * fourier_path.input_scale
    raw = np.vstack([raw, raw[0]])
    ax.plot(raw[:, 0], raw[:, 1], color="#64748b", lw=1.0, alpha=0.55,
            label="one-stroke control polygon")
    ax.plot(fourier_path.points[:, 0], fourier_path.points[:, 1],
            color=ORANGE, lw=1.7, label="Fourier reconstruction")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("Closed stroke  →  Fourier descriptors")
    ax.legend(loc="lower right", frameon=True, facecolor=BG, edgecolor="#334155")
    ax.text(0.02, 0.02,
            f"K = {fourier_path.harmonics}    {len(raw)-1} control points",
            transform=ax.transAxes, color=TEXT, fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.42", fc="#020617", ec="#334155", alpha=0.85))

    # 2. spectrum
    ax = fig.add_subplot(gs[0, 1])
    ax.stem(k[1:], mag[1:], linefmt="#334155", markerfmt="o", basefmt=" ")
    for line in ax.get_lines():
        line.set_markersize(3)
    ax.set_yscale("log")
    ax.set_xlabel("harmonic  k")
    ax.set_ylabel("|c_k|  [m]")
    ax.set_title("Fourier coefficient spectrum")
    ax.grid(True, which="both", alpha=0.40)
    ax.set_xlim(-1, k[-1] + 1)

    # 3. cumulative energy
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(k, energy, color=CYAN, lw=2.0)
    ax.axhline(99.0, color="#64748b", lw=0.9, ls=":")
    ax.axvline(fourier_path.harmonics, color=ORANGE, lw=1.0, ls="--")
    ax.set_xlabel("number of harmonics  K")
    ax.set_ylabel("captured AC energy  [%]")
    ax.set_title("Reconstruction convergence")
    ax.grid(True, alpha=0.55)
    ax.set_xlim(0, k[-1])
    ax.set_ylim(max(0.0, energy.min() - 1.0), 100.5)
    ax.text(0.02, 0.10,
            f"{energy[fourier_path.harmonics]:.3f}% of AC energy at K={fourier_path.harmonics}",
            transform=ax.transAxes, color=MUTED, fontsize=9)

    # 4. speed profile
    ax = fig.add_subplot(gs[1, 1])
    tt = np.linspace(0, drawing.T, len(drawing.v_nodes))
    ax.plot(tt, drawing.v_nodes, color=ORANGE, lw=1.7, label="path speed")
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("speed  [m/s]")
    ax.set_title("Curvature-limited drawing speed")
    ax.grid(True, alpha=0.55)
    ax.set_xlim(0, drawing.T)
    ax2 = ax.twinx()
    ax2.semilogy(tt, drawing.curvature, color=CYAN, lw=0.7, alpha=0.45)
    ax2.set_ylabel("curvature  [1/m]", color=CYAN)
    ax2.tick_params(axis="y", colors=CYAN)
    ax2.spines["right"].set_color("#334155")
    ax.text(0.02, 0.06,
            f"stroke {drawing.path_length:.2f} m    duration {drawing.T:.1f} s    "
            f"peak speed {drawing.max_speed:.2f} m/s",
            transform=ax.transAxes, color=MUTED, fontsize=9)

    fig.suptitle(
        title or "Fourier descriptor construction of the one-stroke image",
        color=TEXT, fontsize=16, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.925,
        "the ball follows the band-limited curve; speed is reduced where curvature is high",
        color=MUTED, fontsize=9.5, ha="center",
    )
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path
