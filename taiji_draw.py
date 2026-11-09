#!/usr/bin/env python3
"""Draw a taiji (yin-yang) with one continuous ball motion.

A full taiji glyph contains four ink strokes (outer circle, S-curve, two dots).
A single closed inked curve cannot contain all four without retracing or lifting
the pen.  This demo keeps the ball moving continuously and marks the connecting
moves as pen-up: the ball still travels, but only the four taiji strokes are
rendered as ink.  The resulting video shows a complete taiji.

    python taiji_draw.py
    python taiji_draw.py --no-video
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

if platform.system() == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from controller import BallPlateController, ControllerConfig, ReferenceState, read_plate_ball_state
from draw_fourier import (
    Projector,
    _add_trail_3d,
    _compose_views,
    _make_gif,
    _panel_tag,
)
from plotting import BG, CYAN, GREEN, MUTED, ORANGE, PANEL, PINK, TEXT, _style


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ball_on_plate.xml"
FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _circle(cx, cy, r, a0, a1, n):
    a = np.linspace(a0, a1, n)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def taiji_segments(R: float = 0.21):
    """Return (points, pen_down) polylines in plate coordinates."""
    rdot = 0.12 * R
    outer = _circle(0, 0, R, np.pi / 2, np.pi / 2 + 2 * np.pi, 900)
    s_upper = _circle(0, R / 2, R / 2, np.pi / 2, -np.pi / 2, 360)
    s_lower = _circle(0, -R / 2, R / 2, np.pi / 2, 3 * np.pi / 2, 360)
    dot_u = _circle(0, R / 2, rdot, -np.pi / 2, 3 * np.pi / 2, 120)
    dot_l = _circle(0, -R / 2, rdot, -np.pi / 2, 3 * np.pi / 2, 120)

    def line(a, b, n=80):
        return np.linspace(a, b, n)

    segs = [
        (outer, True),
        (s_upper, True),
        (s_lower, True),
        (line(np.array([0, -R]), np.array([0, R / 2 - rdot])), False),
        (dot_u, True),
        (line(np.array([0, R / 2 - rdot]), np.array([0, -R / 2 - rdot])), False),
        (dot_l, True),
        (line(np.array([0, -R / 2 - rdot]), np.array([0, R])), False),
    ]
    return segs


class TaijiReference:
    """Continuous ball reference with a pen-down mask."""

    def __init__(self, R=0.21, n=4096, fs=2000.0, speed=0.045, smooth_sigma=18.0):
        segs = taiji_segments(R)
        pts = []
        ink = []
        self.guide = []
        for p, down in segs:
            if down:
                self.guide.append(p.copy())
            p = np.asarray(p, float)
            for i, q in enumerate(p):
                if i == 0 and pts:
                    continue
                pts.append(q)
                ink.append(bool(down))
        pts = np.asarray(pts, float)
        ink = np.asarray(ink, bool)

        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        keep = np.r_[True, np.diff(s) > 1e-12]
        pts, ink, s = pts[keep], ink[keep], s[keep]
        L = float(s[-1])
        su = np.linspace(0.0, L, n, endpoint=False)
        x = np.interp(su, s, pts[:, 0])
        y = np.interp(su, s, pts[:, 1])
        ink_d = ink[np.clip(np.searchsorted(s, su, side="right") - 1, 0, len(ink) - 1)]
        x = gaussian_filter1d(x, smooth_sigma, mode="wrap")
        y = gaussian_filter1d(y, smooth_sigma, mode="wrap")
        # close the periodic curve
        x[-1], y[-1] = x[0], y[0]
        ink_d = gaussian_filter1d(ink_d.astype(float), 3.0, mode="wrap") > 0.5

        self.L = L
        self.n = n
        self.raw_points = pts
        self.ink = ink_d
        self.curve = CubicSpline(np.linspace(0, L, n + 1),
                                 np.r_[x, x[0]], bc_type="periodic")
        self.curve_y = CubicSpline(np.linspace(0, L, n + 1),
                                   np.r_[y, y[0]], bc_type="periodic")

        # Choose a smooth time law; the travel segments are included so the
        # ball never stops between the four ink strokes.
        self.T = L / speed
        u = np.linspace(0, 1, int(self.T * fs) + 1)
        S = 6 * u**5 - 15 * u**4 + 10 * u**3
        Sp = 30 * u**4 - 60 * u**3 + 30 * u**2
        Spp = 120 * u**3 - 180 * u**2 + 60 * u
        ss = L * S
        sd = L * Sp / self.T
        sdd = L * Spp / self.T**2

        px = self.curve(ss)
        py = self.curve_y(ss)
        vx = self.curve(ss, 1) * sd
        vy = self.curve_y(ss, 1) * sd
        ax = self.curve(ss, 2) * sd**2 + self.curve(ss, 1) * sdd
        ay = self.curve_y(ss, 2) * sd**2 + self.curve_y(ss, 1) * sdd

        self.t_grid = u * self.T
        self.p_grid = np.c_[px, py]
        self.v_grid = np.c_[vx, vy]
        self.a_grid = np.c_[ax, ay]
        self.ink_grid = self._ink_at_s(ss)
        self.max_accel = float(np.max(np.linalg.norm(self.a_grid, axis=1)))
        self.max_speed = float(np.max(np.linalg.norm(self.v_grid, axis=1)))
        self.p0 = self.p_grid[0].copy()

    def _ink_at_s(self, s):
        u = (np.asarray(s) % self.L) / self.L
        i = np.clip((u * self.n).astype(int), 0, self.n - 1)
        return self.ink[i]

    def sample(self, t):
        tc = float(np.clip(t, 0.0, self.T))
        p = np.array([np.interp(tc, self.t_grid, self.p_grid[:, 0]),
                      np.interp(tc, self.t_grid, self.p_grid[:, 1])])
        v = np.array([np.interp(tc, self.t_grid, self.v_grid[:, 0]),
                      np.interp(tc, self.t_grid, self.v_grid[:, 1])])
        a = np.array([np.interp(tc, self.t_grid, self.a_grid[:, 0]),
                      np.interp(tc, self.t_grid, self.a_grid[:, 1])])
        return ReferenceState(
            t=tc, p=p, v=v, a=a, radius=float(np.linalg.norm(p)),
            phase=float(2.0 * np.pi * tc / self.T),
        )

    def ink_at(self, t):
        i = int(np.clip(round(float(t) * 2000.0), 0, len(self.ink_grid) - 1))
        return bool(self.ink_grid[i])


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def _hud(frame, t, T, err_mm, progress, speed):
    image = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w, h = image.size
    try:
        ft = ImageFont.truetype(str(FONT), int(w * 0.0250))
        fb = ImageFont.truetype(str(FONT), int(w * 0.0195))
        fs = ImageFont.truetype(str(FONT), int(w * 0.0155))
    except OSError:
        ft = fb = fs = ImageFont.load_default()
    x0, y0 = int(w * 0.032), int(h * 0.030)
    bw, bh = int(w * 0.700), int(h * 0.165)
    d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=int(w * 0.012),
                        fill=(8, 15, 30, 200), outline=(56, 189, 248, 85), width=1)
    d.rectangle((x0, y0 + int(bh * 0.12), x0 + int(w * 0.0045), y0 + int(bh * 0.88)),
                fill=(251, 146, 60, 255))
    tx = x0 + int(w * 0.020)
    d.text((tx, y0 + int(bh * 0.08)), "TAIJI · ONE CONTINUOUS MOTION",
           font=ft, fill=(241, 245, 249, 255))
    d.text((tx, y0 + int(bh * 0.39)),
           f"t = {t:5.1f} / {T:4.1f} s      error = {err_mm:4.2f} mm",
           font=fb, fill=(226, 232, 240, 255))
    d.text((tx, y0 + int(bh * 0.63)),
           f"{speed:g}× speed   ·   3 pen-down strokes   ·   travel not inked",
           font=fs, fill=(148, 163, 184, 255))
    bx, by = tx, y0 + int(bh * 0.84)
    bw2, bh2 = bw - int(w * 0.040), max(3, int(h * 0.0045))
    d.rounded_rectangle((bx, by, bx + bw2, by + bh2), radius=bh2 // 2,
                        fill=(30, 41, 59, 230))
    if progress > 0:
        d.rounded_rectangle((bx, by, bx + max(bh2, int(bw2 * progress)), by + bh2),
                            radius=bh2 // 2, fill=(251, 146, 60, 255))
    return np.asarray(Image.alpha_composite(image, layer).convert("RGB"))


def _overlay_taiji(base, projector, guide_world, trail_segments, ball_world,
                   ball_radius):
    image = Image.fromarray(base).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    for g in guide_world:
        px, z = projector.project(g)
        if np.all(z > 1e-6):
            d.line([tuple(p) for p in px], fill=(34, 211, 238, 55), width=2,
                   joint="curve")
    for seg in trail_segments:
        if len(seg) < 2:
            continue
        px, z = projector.project(np.asarray(seg))
        if np.all(z > 1e-6):
            pts = [tuple(p) for p in px]
            d.line(pts, fill=(251, 146, 60, 38), width=9, joint="curve")
            d.line(pts, fill=(255, 176, 74, 240), width=3, joint="curve")

    bp, _ = projector.project(ball_world)
    ep, _ = projector.project(ball_world + projector.right * (ball_radius + 0.004))
    rad = float(np.linalg.norm(ep[0] - bp[0])) + 3.0
    u, v = bp[0]
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).ellipse((u - rad, v - rad, u + rad, v + rad), fill=255)
    alpha = layer.getchannel("A")
    alpha.paste(0, (0, 0, image.width, image.height), mask)
    layer.putalpha(alpha)
    return np.asarray(Image.alpha_composite(image, layer).convert("RGB"))


def _plot(log, path, out):
    _style()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = log["t"]
    fig = plt.figure(figsize=(14.5, 10.0))
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.97, top=0.90, bottom=0.08,
                          hspace=0.32, wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    for seg in path.guide:
        ax.plot(seg[:, 0], seg[:, 1], color=CYAN, lw=1.2, alpha=0.7)
    ink = np.asarray(log.get("ink", np.ones(len(log["p"]), dtype=bool)), dtype=bool)
    starts = np.where(np.diff(np.r_[False, ink].astype(int)) == 1)[0]
    ends = np.where(np.diff(np.r_[ink, False].astype(int)) == -1)[0] + 1
    for a, b in zip(starts, ends):
        if b - a >= 2:
            ax.plot(log["p"][a:b, 0], log["p"][a:b, 1], color=ORANGE, lw=1.8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.235, 0.235)
    ax.set_ylim(-0.235, 0.235)
    ax.axis("off")
    ax.set_title("Taiji ink + ball path")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, log["err"] * 1000, color=GREEN, lw=1.2)
    ax.fill_between(t, 0, log["err"] * 1000, color=GREEN, alpha=0.1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [mm]")
    ax.set_title("Tracking error")
    ax.grid(True, alpha=0.5)
    ax.set_xlim(0, t[-1])

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, np.degrees(log["roll"]), color=CYAN, lw=1.4, label="roll")
    ax.plot(t, np.degrees(log["pitch"]), color=ORANGE, lw=1.4, label="pitch")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("angle [deg]")
    ax.set_title("Plate attitude")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_xlim(0, t[-1])

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, log["tau_pitch"], color=ORANGE, lw=1.2, label="pitch motor")
    ax.plot(t, log["tau_roll"], color=CYAN, lw=1.2, label="roll motor")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("torque [N m]")
    ax.set_title("Actuator effort")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_xlim(0, t[-1])

    fig.suptitle("Taiji on a ball-on-plate system  ·  one continuous motion",
                 color=TEXT, fontsize=16, fontweight="bold", y=0.965)
    fig.savefig(out, dpi=190)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(render=True, width=900, height=900, fps=60, speed_factor=4.0,
        outdir="outputs/taiji", gif=True, verbose=True):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ref = TaijiReference()
    print(f"path length {ref.L:.3f} m, duration {ref.T:.2f} s, "
          f"max accel {ref.max_accel:.2f} m/s²")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tmat = model.material("target").id
    for gid in range(model.ngeom):
        if int(model.geom_matid[gid]) == tmat:
            model.geom_rgba[gid] = [0, 0, 0, 0]
            model.geom_group[gid] = 3
    for gname in ("hub", "ball_mark_a", "ball_mark_b", "ball_mark_c"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid >= 0:
            model.geom_rgba[gid] = [0, 0, 0, 0]
    for i in range(model.nlight):
        model.light_castshadow[i] = 0

    radius = float(model.geom_size[model.body("ball").geomadr[0], 0])
    ctrl = BallPlateController(
        model, ref,
        ControllerConfig(kp=60, kd=18, kp_joint=140, kd_joint=18,
                         max_accel=2.0, max_tilt=0.30),
    )
    R = np.asarray(data.body("plate").xmat).reshape(3, 3)
    o = np.asarray(data.body("plate").xpos)
    p0 = o + R @ np.array([ref.p0[0], ref.p0[1], 0.0]) + R[:, 2] * (radius + 1e-4)
    bj = model.joint("ball_free").id
    qa = model.jnt_qposadr[bj]
    data.qpos[qa:qa + 3] = p0
    data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

    duration = ref.T + 2.0
    dt = float(model.opt.timestep)
    n_steps = int(round(duration / dt))
    renderer = projector = writer = None
    if render:
        try:
            renderer = mujoco.Renderer(model, height=height, width=width)
            projector = Projector(model, "drawing", width, height)
            import imageio.v2 as imageio
            writer = imageio.get_writer(
                outdir / "taiji.mp4", fps=fps, codec="libx264", quality=8,
                macro_block_size=None, pixelformat="yuv420p",
                ffmpeg_params=["-crf", "18", "-preset", "medium"],
            )
        except Exception as exc:
            print(f"[warn] render unavailable: {exc}")
            render = False

    guide_local = [np.asarray(g, float) for g in ref.guide]
    log = {k: [] for k in ("t", "p", "err", "roll", "pitch", "roll_ref",
                           "pitch_ref", "tau_pitch", "tau_roll", "ink")}
    trail_segments = []
    current = []
    was_ink = False
    ink_points = []
    next_frame = 0.0
    frame_period = speed_factor / fps
    preview_saved = False
    wall = time.perf_counter()
    last = None

    for step in range(n_steps + 1):
        if step % 2 == 0:
            last = ctrl.update(data)

        st = read_plate_ball_state(model, data)
        rs = ref.sample(float(data.time))
        p = rs.p
        err = p - st.p[:2]
        contact = st.ball_world - radius * st.R[:, 2] + 0.0009 * st.R[:, 2]
        ink = ref.ink_at(float(data.time))

        if step % 10 == 0:
            log["t"].append(float(data.time))
            log["p"].append(st.p[:2].copy())
            log["err"].append(float(np.linalg.norm(err)))
            log["roll"].append(st.roll)
            log["pitch"].append(st.pitch)
            log["roll_ref"].append(last.roll_des if last else 0)
            log["pitch_ref"].append(last.pitch_des if last else 0)
            log["tau_pitch"].append(float(last.tau[0]) if last else 0)
            log["tau_roll"].append(float(last.tau[1]) if last else 0)
            log["ink"].append(bool(ink))

        if ink:
            if not was_ink:
                current = []
            current.append(contact)
            ink_points.append(contact)
        elif was_ink:
            if len(current) >= 2:
                trail_segments.append(current)
            current = []
        was_ink = ink

        if render and data.time + 1e-9 >= next_frame and data.time < duration - 1e-9:
            for i in range(model.nlight):
                model.light_castshadow[i] = 0
            renderer.update_scene(data, camera="drawing")
            projector.update(renderer.scene)
            top = renderer.render()
            # use current plate pose, not the initial one
            Rp = np.asarray(data.body("plate").xmat).reshape(3, 3)
            op = np.asarray(data.body("plate").xpos)
            gw = []
            for g in guide_local:
                local = np.c_[g, np.full(len(g), 0.0009)]
                gw.append(op + local @ Rp.T)
            segs = trail_segments + ([current] if len(current) >= 2 else [])
            top = _overlay_taiji(top, projector, gw, segs, st.ball_world, radius)
            top = _hud(top, min(float(data.time), ref.T), ref.T,
                       float(np.linalg.norm(err) * 1000), min(float(data.time) / ref.T, 1.0),
                       speed_factor)
            top = _panel_tag(top, "TOP VIEW", "taiji ink, travel hidden")

            for i in range(model.nlight):
                model.light_castshadow[i] = 1
            renderer.update_scene(data, camera="track")
            _add_trail_3d(renderer.scene, ink_points)
            view3d = renderer.render()
            view3d = _panel_tag(view3d, "3D VIEW", "table, plate and free ball")
            frame = _compose_views(top, view3d, speed_factor=speed_factor)
            writer.append_data(np.ascontiguousarray(frame))
            if not preview_saved and data.time >= ref.T + 0.3:
                Image.fromarray(frame).save(outdir / "preview.png")
                preview_saved = True
            next_frame += frame_period

        if step < n_steps:
            mujoco.mj_step(model, data)

    if len(current) >= 2:
        trail_segments.append(current)
    if writer is not None:
        writer.close()
        if verbose:
            print(f"[video] {outdir/'taiji.mp4'} "
                  f"({duration/speed_factor:.1f} s at {speed_factor:g}x, "
                  f"{2*width}x{height}, wall {time.perf_counter()-wall:.1f} s)")
        if gif:
            _make_gif(outdir / "taiji.mp4", outdir / "preview.gif",
                      duration=min(ref.T + 1, duration))

    arrays = {k: np.asarray(v) for k, v in log.items()}
    metrics = {
        "drawing": "Taiji (yin-yang)",
        "path_length_m": ref.L,
        "duration_s": ref.T,
        "video_duration_s": duration / speed_factor,
        "speed_factor": speed_factor,
        "rms_mm": float(np.sqrt(np.mean(arrays["err"] ** 2)) * 1000),
        "max_mm": float(np.max(arrays["err"]) * 1000),
        "max_tilt_deg": float(np.degrees(max(np.max(np.abs(arrays["pitch"])),
                                            np.max(np.abs(arrays["roll"]))))),
        "max_torque_nm": float(max(np.max(np.abs(arrays["tau_pitch"])),
                                   np.max(np.abs(arrays["tau_roll"])))),
        "max_ref_accel_m_s2": ref.max_accel,
        "pen_down_strokes": int(np.sum(np.diff(np.r_[False, ref.ink_grid].astype(int)) == 1)),
    }
    np.savez_compressed(outdir / "trajectory.npz", **arrays)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot(arrays, ref, outdir / "tracking.png")
    if verbose:
        print(f"RMS {metrics['rms_mm']:.3f} mm, max {metrics['max_mm']:.3f} mm")
        print(f"outputs: {outdir.resolve()}")
    if renderer is not None:
        renderer.close()
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--width", type=int, default=900)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--speed", type=float, default=4.0)
    p.add_argument("--outdir", default="outputs/taiji")
    p.add_argument("--no-gif", action="store_true")
    a = p.parse_args(argv)
    run(render=not a.no_video, width=a.width, height=a.height, fps=a.fps,
        speed_factor=a.speed, outdir=a.outdir, gif=not a.no_gif)
    return 0


if __name__ == "__main__":
    sys.exit(main())
