#!/usr/bin/env python3
"""Let the ball draw a Fourier portrait as one continuous stroke.

This is the arbitrary-shape counterpart of simulate.py:

  * a closed one-stroke line drawing is turned into complex Fourier
    coefficients (Fourier descriptors);
  * a curvature-aware timing law slows the stroke down at sharp details;
  * the existing cascaded ball-on-plate controller tracks the resulting
    reference while MuJoCo resolves all contact/rolling physics;
  * the actual ball track is rendered as a persistent line on the plate.

Run:
    python draw_fourier.py
    python draw_fourier.py --no-video
    python draw_fourier.py --harmonics 60 --span 0.42
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

if platform.system() == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from controller import BallPlateController, ControllerConfig, read_plate_ball_state
from drawing import FOURIER_PROFILE_POINTS, FourierDrawing, make_fourier_path
from drawing_plots import make_drawing_plots, make_fourier_analysis


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ball_on_plate.xml"
FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")


# --------------------------------------------------------------------------- #
# Camera projection / 2-D stroke overlay
# --------------------------------------------------------------------------- #
class Projector:
    """Project world points to pixels for the fixed top camera."""

    def __init__(self, model, camera_name: str, width: int, height: int):
        self.model = model
        self.width = width
        self.height = height
        self.fovy = np.deg2rad(float(model.cam_fovy[model.camera(camera_name).id]))
        self.aspect = width / height
        self.pos = None
        self.fwd = None
        self.up = None
        self.right = None

    def update(self, scene) -> None:
        # MuJoCo's renderer exposes two (stereo) GL cameras even for a single
        # camera; the midpoint is the actual optical centre.
        c0 = scene.camera[0]
        c1 = scene.camera[1] if len(scene.camera) > 1 else scene.camera[0]
        self.pos = 0.5 * (np.asarray(c0.pos) + np.asarray(c1.pos))
        self.fwd = np.asarray(c0.forward, dtype=float)
        self.up = np.asarray(c0.up, dtype=float)
        self.right = np.cross(self.fwd, self.up)
        self.right /= np.linalg.norm(self.right) + 1e-12

    def project(self, points) -> tuple[np.ndarray, np.ndarray]:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        d = pts - self.pos
        z = d @ self.fwd
        x = (d @ self.right) / np.maximum(z * np.tan(self.fovy / 2) * self.aspect, 1e-8)
        y = (d @ self.up) / np.maximum(z * np.tan(self.fovy / 2), 1e-8)
        u = (x + 1.0) * 0.5 * self.width
        v = (1.0 - y) * 0.5 * self.height
        return np.c_[u, v], z


def _path_world(data, local_xy: np.ndarray) -> np.ndarray:
    """Lift plate-frame 2-D points onto the current plate surface."""
    R = np.asarray(data.body("plate").xmat, dtype=float).reshape(3, 3)
    origin = np.asarray(data.body("plate").xpos, dtype=float)
    local = np.c_[local_xy, np.full(len(local_xy), 0.0009)]
    return origin + local @ R.T


def _hud(
    frame: np.ndarray,
    t: float,
    duration: float,
    err_mm: float,
    progress: float,
    harmonics: int,
    path_length: float,
) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = image.size
    try:
        f_title = ImageFont.truetype(str(FONT), int(w * 0.0255))
        f_body = ImageFont.truetype(str(FONT), int(w * 0.0195))
        f_small = ImageFont.truetype(str(FONT), int(w * 0.0155))
    except OSError:
        f_title = f_body = f_small = ImageFont.load_default()

    x0, y0 = int(w * 0.032), int(h * 0.030)
    box_w, box_h = int(w * 0.585), int(h * 0.165)
    draw.rounded_rectangle(
        (x0, y0, x0 + box_w, y0 + box_h),
        radius=int(w * 0.012),
        fill=(8, 15, 30, 200),
        outline=(56, 189, 248, 85),
        width=1,
    )
    draw.rectangle(
        (x0, y0 + int(box_h * 0.12), x0 + int(w * 0.0045), y0 + int(box_h * 0.88)),
        fill=(251, 146, 60, 255),
    )
    tx = x0 + int(w * 0.020)
    draw.text((tx, y0 + int(box_h * 0.08)),
              "J. FOURIER · ONE-STROKE PORTRAIT", font=f_title,
              fill=(241, 245, 249, 255))
    draw.text((tx, y0 + int(box_h * 0.39)),
              f"t = {t:5.1f} / {duration:4.1f} s      error = {err_mm:4.2f} mm",
              font=f_body, fill=(226, 232, 240, 255))
    draw.text((tx, y0 + int(box_h * 0.63)),
              f"K = {harmonics} Fourier harmonics   ·   stroke {path_length:.2f} m"
              "   ·   2 kHz / 1 kHz",
              font=f_small, fill=(148, 163, 184, 255))

    # progress bar
    bar_x0 = tx
    bar_y0 = y0 + int(box_h * 0.84)
    bar_w = box_w - int(w * 0.040)
    bar_h = max(3, int(h * 0.0045))
    draw.rounded_rectangle((bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h),
                           radius=bar_h // 2, fill=(30, 41, 59, 230))
    if progress > 0:
        draw.rounded_rectangle(
            (bar_x0, bar_y0, bar_x0 + max(bar_h, int(bar_w * progress)), bar_y0 + bar_h),
            radius=bar_h // 2, fill=(251, 146, 60, 255),
        )

    image = Image.alpha_composite(image, overlay).convert("RGB")
    return np.asarray(image)


def _overlay_stroke(
    base: np.ndarray,
    projector: Projector,
    reference_world: np.ndarray,
    trail_world: list[np.ndarray],
    ball_world: np.ndarray,
    ball_radius: float,
    progress: float,
) -> np.ndarray:
    """Draw the reference guide and the accumulated ball trail with PIL."""
    image = Image.fromarray(base).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    ref_px, ref_z = projector.project(reference_world)
    if np.all(ref_z > 1e-6):
        draw.line([tuple(p) for p in ref_px], fill=(34, 211, 238, 64), width=2,
                  joint="curve")

    if len(trail_world) >= 2:
        trail_px, trail_z = projector.project(np.asarray(trail_world))
        if np.all(trail_z > 1e-6):
            pts = [tuple(p) for p in trail_px]
            # soft glow + crisp core
            draw.line(pts, fill=(251, 146, 60, 34), width=9, joint="curve")
            draw.line(pts, fill=(255, 176, 74, 238), width=3, joint="curve")

    # Keep the 3-D ball clean: erase the overlay in a disk around it.
    ball_px, _ = projector.project(ball_world)
    edge_px, _ = projector.project(ball_world + projector.right * (ball_radius + 0.004))
    rad = float(np.linalg.norm(edge_px[0] - ball_px[0])) + 3.0
    u, v = ball_px[0]
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).ellipse((u - rad, v - rad, u + rad, v + rad), fill=255)
    alpha = layer.getchannel("A")
    alpha.paste(0, (0, 0, image.width, image.height), mask)
    layer.putalpha(alpha)

    image = Image.alpha_composite(image, layer)
    return np.asarray(image.convert("RGB"))


def _make_gif(video_path: Path, gif_path: Path, duration: float) -> bool:
    ffmpeg = "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).exists():
        ffmpeg = "ffmpeg"
    # 4x timelapse so the whole 34 s drawing is easy to preview.
    vf = (
        "setpts=0.25*PTS,fps=12,scale=540:-1:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5"
    )
    try:
        subprocess.run(
            [ffmpeg, "-y", "-t", str(duration), "-i", str(video_path), "-vf", vf,
             "-loop", "0", str(gif_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


# --------------------------------------------------------------------------- #
# Main simulation
# --------------------------------------------------------------------------- #
def run(
    harmonics: int = 45,
    span: float = 0.42,
    render: bool = True,
    width: int = 900,
    height: int = 900,
    fps: int = 60,
    outdir: str | Path = "outputs/fourier",
    make_preview_gif: bool = True,
    verbose: bool = True,
) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Rotate the closed portrait so it starts/ends at the smooth bottom centre,
    # not at a sharp shoulder corner.
    raw = FOURIER_PROFILE_POINTS[:-1].copy()
    start_idx = int(np.argmin((raw[:, 0] - 0.05) ** 2 + (raw[:, 1] + 1.08) ** 2))
    raw = np.roll(raw, -start_idx, axis=0)

    fourier_path = make_fourier_path(
        raw, harmonics=harmonics, target_span=span, smooth_sigma=8.0
    )
    # Slow enough that the plate never runs out of authority, including at the
    # sharp nose/shoulder details.
    drawing = FourierDrawing(
        fourier_path, a_lat=0.10, a_tan=0.02, v_max=0.07, v_min=0.003
    )

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Hide the circular target dots of the original demo; the Fourier guide is
    # drawn as a 2-D overlay instead.
    target_mat = model.material("target").id
    for gid in range(model.ngeom):
        if int(model.geom_matid[gid]) == target_mat:
            model.geom_rgba[gid] = [0.0, 0.0, 0.0, 0.0]
            model.geom_group[gid] = 3
    # The decorative gimbal hub is just below the plate and can show through
    # the thin plate at some angles; hide it (and the ball orientation dots)
    # for a clean top-down drawing view.
    for name in ("hub", "ball_mark_a", "ball_mark_b", "ball_mark_c"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            model.geom_rgba[gid] = [0.0, 0.0, 0.0, 0.0]
    # In the top-down view the directional-light shadow map produces a bright
    # clipped disk on the plate; disable shadows for a clean drawing surface.
    for i in range(model.nlight):
        model.light_castshadow[i] = 0

    ball_radius = float(model.geom_size[model.body("ball").geomadr[0], 0])
    controller = BallPlateController(
        model,
        drawing,
        ControllerConfig(kp=60.0, kd=18.0, kp_joint=140.0, kd_joint=18.0,
                          max_accel=2.0, max_tilt=0.30),
    )

    # Seed the free ball exactly on the plate at the start of the stroke.
    R = np.asarray(data.body("plate").xmat, dtype=float).reshape(3, 3)
    origin = np.asarray(data.body("plate").xpos, dtype=float)
    p0_world = origin + R @ np.array([drawing.p0[0], drawing.p0[1], 0.0]) \
        + R[:, 2] * (ball_radius + 1.0e-4)
    bj = model.joint("ball_free").id
    qa = model.jnt_qposadr[bj]
    data.qpos[qa : qa + 3] = p0_world
    data.qpos[qa + 3 : qa + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    duration = drawing.T + 2.2
    dt = float(model.opt.timestep)
    n_steps = int(round(duration / dt))
    control_every = 2
    log_every = 10

    log: dict[str, list] = {
        key: []
        for key in (
            "t", "p", "p_ref", "err", "roll", "pitch", "roll_ref", "pitch_ref",
            "tau_pitch", "tau_roll", "slip", "normal_force", "progress",
        )
    }

    # Rendering setup.
    renderer = None
    projector = None
    writer = None
    trail: list[np.ndarray] = []
    preview_saved = False
    if render:
        try:
            renderer = mujoco.Renderer(model, height=height, width=width)
            projector = Projector(model, "drawing", width, height)
            import imageio.v2 as imageio

            video_path = outdir / "ball_on_plate_fourier.mp4"
            writer = imageio.get_writer(
                video_path, fps=fps, codec="libx264", quality=8,
                macro_block_size=None, pixelformat="yuv420p",
                ffmpeg_params=["-crf", "18", "-preset", "medium"],
            )
        except Exception as exc:
            print(f"[warn] off-screen rendering unavailable ({exc}); continuing headless.")
            render = False
            renderer = None
            writer = None

    # Reference guide is drawn at a reduced density.
    guide_xy = drawing.path.points[::4].copy()
    last_diag = None
    sat_count = 0
    ctrl_count = 0
    next_frame = 0.0
    frame_period = 1.0 / fps
    wall_start = time.perf_counter()

    for step in range(n_steps + 1):
        if step % control_every == 0:
            last_diag = controller.update(data)
            ctrl_count += 1
            sat_count += int(last_diag.saturated)

        state = read_plate_ball_state(model, data)
        ref = drawing.sample(float(data.time))
        err = ref.p - state.p[:2]
        err_norm = float(np.linalg.norm(err))
        normal_force, slip = _contact_metrics(model, data, state, ball_radius)
        contact_world = state.ball_world - ball_radius * state.R[:, 2]
        contact_world = contact_world + 0.0009 * state.R[:, 2]

        if step % log_every == 0:
            log["t"].append(float(data.time))
            log["p"].append(state.p[:2].copy())
            log["p_ref"].append(ref.p.copy())
            log["err"].append(err_norm)
            log["roll"].append(state.roll)
            log["pitch"].append(state.pitch)
            log["roll_ref"].append(last_diag.roll_des if last_diag else 0.0)
            log["pitch_ref"].append(last_diag.pitch_des if last_diag else 0.0)
            log["tau_pitch"].append(float(last_diag.tau[0]) if last_diag else 0.0)
            log["tau_roll"].append(float(last_diag.tau[1]) if last_diag else 0.0)
            log["slip"].append(slip)
            log["normal_force"].append(normal_force)
            log["progress"].append(drawing.progress(data.time))

        if render and data.time + 1e-9 >= next_frame and data.time < duration - 1e-9:
            trail.append(contact_world)
            renderer.update_scene(data, camera="drawing")
            projector.update(renderer.scene)
            frame = renderer.render()
            ref_world = _path_world(data, guide_xy)
            frame = _overlay_stroke(
                frame, projector, ref_world, trail, state.ball_world,
                ball_radius, drawing.progress(data.time),
            )
            frame = _hud(
                frame, min(float(data.time), drawing.T), drawing.T, err_norm * 1000.0,
                drawing.progress(data.time), harmonics, drawing.path_length,
            )
            writer.append_data(np.ascontiguousarray(frame))
            if not preview_saved and data.time >= drawing.T + 0.35:
                Image.fromarray(frame).save(outdir / "preview.png")
                preview_saved = True
            next_frame += frame_period

        if step < n_steps:
            mujoco.mj_step(model, data)

    if writer is not None:
        writer.close()
        wall = time.perf_counter() - wall_start
        if verbose:
            print(f"[video] {outdir / 'ball_on_plate_fourier.mp4'} "
                  f"({duration:.1f} s, {width}x{height}, {fps} fps, wall {wall:.1f} s)")
        if make_preview_gif:
            gif = outdir / "preview.gif"
            if _make_gif(outdir / "ball_on_plate_fourier.mp4", gif,
                         duration=min(drawing.T + 1.0, duration)):
                if verbose:
                    print(f"[gif]   {gif}")

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    arrays = {k: np.asarray(v) for k, v in log.items()}
    t = arrays["t"]
    e = arrays["err"]
    draw_window = (t >= 0.4) & (t <= drawing.T + 0.05)
    saturation = sat_count / max(ctrl_count, 1)
    cumulative = fourier_path.cumulative_energy()

    metrics = {
        "drawing": "Joseph Fourier one-stroke profile silhouette",
        "harmonics": harmonics,
        "target_span_m": span,
        "path_length_m": drawing.path_length,
        "drawing_duration_s": drawing.T,
        "video_duration_s": duration,
        "control_rate_hz": 1.0 / (dt * control_every),
        "rms_mm": float(np.sqrt(np.mean(e[draw_window] ** 2)) * 1000.0),
        "max_mm": float(np.max(e[draw_window]) * 1000.0),
        "p95_mm": float(np.percentile(e[draw_window], 95) * 1000.0),
        "final_error_mm": float(e[-1] * 1000.0),
        "max_tilt_deg": float(np.degrees(max(
            np.max(np.abs(arrays["pitch"])), np.max(np.abs(arrays["roll"]))))),
        "max_torque_nm": float(max(
            np.max(np.abs(arrays["tau_pitch"])), np.max(np.abs(arrays["tau_roll"])))),
        "torque_limit": float(model.actuator_ctrlrange[0, 1]),
        "saturation_fraction": float(saturation),
        "slip_rms_mm_s": float(np.sqrt(np.mean(arrays["slip"] ** 2)) * 1000.0),
        "max_ref_speed_m_s": drawing.max_speed,
        "max_ref_accel_m_s2": drawing.max_accel,
        "fourier_energy_percent": float(cumulative[harmonics] * 100.0),
        "start_point_m": drawing.p0.tolist(),
    }

    np.savez_compressed(
        outdir / "trajectory.npz", **arrays,
        reference_path=drawing.path.points,
        control_points=raw,
        fourier_coeff=drawing.path.coeff,
        fourier_ks=drawing.path.ks,
    )
    with open(outdir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    with open(outdir / "metrics.txt", "w", encoding="utf-8") as fh:
        fh.write("Fourier one-stroke drawing on a ball-on-plate system\n")
        fh.write("=================================================\n")
        for k, v in metrics.items():
            fh.write(f"{k}: {v}\n")

    make_drawing_plots(arrays, metrics, outdir / "tracking.png")
    make_fourier_analysis(raw, fourier_path, drawing, metrics, outdir / "fourier_analysis.png")

    if verbose:
        print("\n=== Fourier drawing result ===")
        print(f"harmonics / stroke   : {harmonics} / {drawing.path_length:.3f} m")
        print(f"drawing duration     : {drawing.T:.2f} s")
        print(f"tracking RMS / max   : {metrics['rms_mm']:.3f} / {metrics['max_mm']:.3f} mm")
        print(f"p95 error            : {metrics['p95_mm']:.3f} mm")
        print(f"Fourier energy       : {metrics['fourier_energy_percent']:.2f}%")
        print(f"max plate tilt       : {metrics['max_tilt_deg']:.2f} deg")
        print(f"max joint torque     : {metrics['max_torque_nm']:.3f} N m")
        print(f"slip RMS             : {metrics['slip_rms_mm_s']:.3f} mm/s")
        print(f"outputs              : {outdir.resolve()}")

    if renderer is not None:
        renderer.close()
    return {"metrics": metrics, "log": arrays, "drawing": drawing, "path": fourier_path}


def _contact_metrics(model, data, state, radius: float) -> tuple[float, float]:
    ball_body = model.body("ball").id
    plate_body = model.body("plate").id
    n = state.R[:, 2]
    contact_pt = state.ball_world - radius * n
    v_ball_contact = state.ball_vel_world + np.cross(state.ball_spin_world, -radius * n)
    v_plate_contact = np.cross(state.omega_world, contact_pt - state.origin)
    slip = float(np.linalg.norm(v_ball_contact - v_plate_contact))
    normal = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        if (b1 == ball_body and b2 == plate_body) or (
            b1 == plate_body and b2 == ball_body
        ):
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f)
            normal += float(f[0])
    return normal, slip


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmonics", type=int, default=45)
    parser.add_argument("--span", type=float, default=0.42, help="longest drawing span [m]")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--outdir", type=str, default="outputs/fourier")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args(argv)

    run(
        harmonics=args.harmonics,
        span=args.span,
        render=not args.no_video,
        width=args.width,
        height=args.height,
        fps=args.fps,
        outdir=args.outdir,
        make_preview_gif=not args.no_gif,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
