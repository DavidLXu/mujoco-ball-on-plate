"""Result plots for the MuJoCo ball-on-plate experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


BG = "#0b1220"
PANEL = "#0f172a"
GRID = "#1e293b"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
CYAN = "#22d3ee"
ORANGE = "#fb923c"
GREEN = "#34d399"
PINK = "#f472b6"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#cbd5e1",
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def make_plots(log: dict, metrics: dict, path: str | Path) -> Path:
    _style()
    path = Path(path)
    t = log["t"]
    p = log["p"]
    p_ref = log["p_ref"]
    phase = log["phase"]
    radius = metrics["radius"]

    fig = plt.figure(figsize=(14.5, 10.5), constrained_layout=False)
    gs = fig.add_gridspec(
        2, 2, left=0.065, right=0.97, top=0.90, bottom=0.075, hspace=0.30, wspace=0.22
    )

    # ------------------------------------------------------------------ #
    # 1. planar path
    # ------------------------------------------------------------------ #
    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(Circle((0, 0), radius, fill=False, color=CYAN, ls=(0, (4, 5)), lw=1.2))
    ax.plot(p_ref[:, 0], p_ref[:, 1], color=CYAN, lw=1.0, alpha=0.35, label="reference")
    ax.plot(p[:, 0], p[:, 1], color=ORANGE, lw=2.2, label="ball (MuJoCo contact)", zorder=3)
    ax.scatter([0], [0], s=55, marker="x", color=MUTED, lw=2, zorder=4, label="start")
    ax.annotate(
        "smooth spin-up\n0 - 4 s",
        xy=(0.016, -0.010),
        xytext=(0.055, -0.095),
        color="#fed7aa",
        fontsize=8.5,
        arrowprops=dict(arrowstyle="-", color="#fdba74", lw=0.9, alpha=0.8),
    )
    ax.annotate(
        "CCW",
        xy=(0.0, radius * 0.98),
        xytext=(-0.012, radius * 1.13),
        color=CYAN,
        fontsize=9,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.0, alpha=0.8),
    )
    ax.set_aspect("equal", adjustable="box")
    lim = radius * 1.38
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("plate x  [m]")
    ax.set_ylabel("plate y  [m]")
    ax.set_title("Planar trajectory  (plate frame)")
    ax.grid(True, alpha=0.55)
    ax.legend(loc="upper right")
    ax.text(
        0.04,
        0.055,
        f"steady RMS  {metrics['steady_rms_mm']:.2f} mm\n"
        f"steady max  {metrics['steady_max_mm']:.2f} mm",
        transform=ax.transAxes,
        color=TEXT,
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.45", fc="#020617", ec="#334155", alpha=0.85),
    )

    # ------------------------------------------------------------------ #
    # 2. error decomposition
    # ------------------------------------------------------------------ #
    ax = fig.add_subplot(gs[0, 1])
    ax.axvspan(metrics["steady_start"], t[-1], color=CYAN, alpha=0.055, lw=0)
    ax.axhline(0, color="#475569", lw=0.8)
    ax.plot(t, log["err_rad"] * 1000, color=GREEN, lw=1.5, label="radial error")
    ax.plot(t, log["err_tan"] * 1000, color=PINK, lw=1.5, label="tangential error")
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("position error  [mm]")
    ax.set_title("Tracking error")
    ax.grid(True, alpha=0.55)
    ax.legend(loc="upper right", ncol=2)
    ax.text(
        0.02,
        0.06,
        f"radial RMS  {metrics['steady_radial_rms_mm']:.2f} mm    "
        f"tangential RMS  {metrics['steady_tangential_rms_mm']:.2f} mm",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9,
    )
    ax.set_xlim(0, t[-1])

    # ------------------------------------------------------------------ #
    # 3. plate attitude
    # ------------------------------------------------------------------ #
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, np.degrees(log["pitch"]), color=ORANGE, lw=1.6, label="pitch (actual)")
    ax.plot(
        t,
        np.degrees(log["pitch_ref"]),
        color=ORANGE,
        lw=1.0,
        ls="--",
        alpha=0.55,
        label="pitch (command)",
    )
    ax.plot(t, np.degrees(log["roll"]), color=CYAN, lw=1.6, label="roll (actual)")
    ax.plot(
        t,
        np.degrees(log["roll_ref"]),
        color=CYAN,
        lw=1.0,
        ls="--",
        alpha=0.55,
        label="roll (command)",
    )
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("plate angle  [deg]")
    ax.set_title("Two-axis plate actuation")
    ax.grid(True, alpha=0.55)
    ax.legend(loc="upper right", ncol=2)
    ax.set_xlim(0, t[-1])

    # ------------------------------------------------------------------ #
    # 4. torques
    # ------------------------------------------------------------------ #
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, log["tau_pitch"], color=ORANGE, lw=1.5, label="pitch motor")
    ax.plot(t, log["tau_roll"], color=CYAN, lw=1.5, label="roll motor")
    tau_lim = metrics["torque_limit"]
    y_lim = max(0.15, 1.35 * float(metrics["max_torque_nm"]))
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("joint torque  [N m]")
    ax.set_title("Actuator effort")
    ax.grid(True, alpha=0.55)
    ax.legend(loc="upper right", ncol=2)
    ax.set_xlim(0, t[-1])
    ax.text(
        0.02,
        0.06,
        f"peak |tau|  {metrics['max_torque_nm']:.3f} N m    "
        f"actuator limit ±{tau_lim:.1f} N m (not reached)",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9,
    )

    fig.suptitle(
        "Ball-on-Plate Circular Tracking  ·  MuJoCo rigid-body contact + rolling dynamics",
        color=TEXT,
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.925,
        "free-flying ball · no-slip contact · torque-controlled 2-axis gimbal · 2 kHz physics / 1 kHz control",
        color=MUTED,
        fontsize=9.5,
        ha="center",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path
