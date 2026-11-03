#!/usr/bin/env python3
"""Run the MuJoCo ball-on-plate circular-tracking experiment.

Examples
--------
    python simulate.py --no-video
    python simulate.py --render --width 1080 --fps 60
    python simulate.py --duration 12 --render --outdir outputs_test
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

# MuJoCo on macOS must use the CGL backend for off-screen rendering.
if platform.system() == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")

import mujoco
import numpy as np

from controller import (
    BallPlateController,
    CircleReference,
    CircleReferenceConfig,
    ControllerConfig,
    read_plate_ball_state,
)
from plotting import make_plots


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ball_on_plate.xml"
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial.ttf")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _seed_ball_at_centre(model, data) -> None:
    """Place the free ball exactly above the plate centre with zero velocity."""
    mujoco.mj_forward(model, data)
    ball_j = model.joint("ball_free").id
    qadr = model.jnt_qposadr[ball_j]
    plate_z = float(data.body("plate").xpos[2])
    radius = float(model.geom_size[model.body("ball").geomadr[0], 0])
    data.qpos[qadr : qadr + 3] = [0.0, 0.0, plate_z + radius + 2.0e-4]
    data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def _contact_metrics(model, data, state, radius: float) -> tuple[float, float]:
    """Return (normal force, slip speed) for the ball/plate contact."""
    ball_body = model.body("ball").id
    plate_body = model.body("plate").id
    n = state.R[:, 2]
    contact_pt = state.ball_world - radius * n
    v_ball_contact = state.ball_vel_world + np.cross(
        state.ball_spin_world, -radius * n
    )
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


def _hud(frame: np.ndarray, t: float, err_mm: float, cfg: CircleReferenceConfig) -> np.ndarray:
    """Draw a compact heads-up display onto an RGB frame."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = image.size
    try:
        f_title = ImageFont.truetype(str(FONT_PATH), int(w * 0.027))
        f_body = ImageFont.truetype(str(FONT_PATH), int(w * 0.020))
        f_small = ImageFont.truetype(str(FONT_PATH), int(w * 0.0165))
    except OSError:
        f_title = f_body = f_small = ImageFont.load_default()

    x0, y0 = int(w * 0.032), int(h * 0.032)
    box_w, box_h = int(w * 0.50), int(h * 0.145)
    draw.rounded_rectangle(
        (x0, y0, x0 + box_w, y0 + box_h),
        radius=int(w * 0.012),
        fill=(8, 15, 30, 195),
        outline=(56, 189, 248, 90),
        width=1,
    )
    draw.rectangle(
        (x0, y0 + int(box_h * 0.12), x0 + int(w * 0.0045), y0 + int(box_h * 0.88)),
        fill=(251, 146, 60, 255),
    )
    tx = x0 + int(w * 0.020)
    draw.text((tx, y0 + int(box_h * 0.10)), "CIRCULAR BALL-ON-PLATE", font=f_title, fill=(241, 245, 249, 255))
    draw.text(
        (tx, y0 + int(box_h * 0.43)),
        f"t = {t:5.2f} s      tracking error = {err_mm:4.2f} mm",
        font=f_body,
        fill=(226, 232, 240, 255),
    )
    draw.text(
        (tx, y0 + int(box_h * 0.70)),
        f"circle r = {cfg.radius:.2f} m, T = {cfg.period:.1f} s"
        "    |    2 kHz physics / 1 kHz control",
        font=f_small,
        fill=(148, 163, 184, 255),
    )

    image = Image.alpha_composite(image, overlay).convert("RGB")
    return np.asarray(image)


def _add_trail(scene, points: deque, r: float) -> None:
    """Append fading visual-only spheres showing the recent ball path."""
    n = len(points)
    if n < 2:
        return
    for i, pt in enumerate(points):
        if scene.ngeom >= scene.maxgeom:
            return
        age = (i + 1) / n
        rgba = np.array(
            [
                1.0 - 0.40 * (1.0 - age),
                0.48 - 0.30 * (1.0 - age),
                0.07,
                0.18 + 0.72 * age,
            ]
        )
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.0019, 0.0, 0.0]),
            pt,
            np.eye(3).ravel(),
            rgba,
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.emission = 0.55 * age
        scene.ngeom += 1


def _make_gif(video_path: Path, gif_path: Path, start: float, duration: float) -> bool:
    ffmpeg = "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).exists():
        ffmpeg = "ffmpeg"
    vf = (
        "fps=15,scale=540:-1:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-loop",
        "0",
        str(gif_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def run(
    duration: float = 20.0,
    render: bool = True,
    width: int = 900,
    height: int = 900,
    fps: int = 60,
    outdir: str | Path = "outputs",
    make_preview_gif: bool = True,
    verbose: bool = True,
) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ref_cfg = CircleReferenceConfig(radius=0.16, period=4.0, ramp_time=4.0)
    reference = CircleReference(ref_cfg)
    ctrl_cfg = ControllerConfig(kp=60.0, kd=18.0, ki=0.0, kp_joint=140.0, kd_joint=18.0)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    controller = BallPlateController(model, reference, ctrl_cfg)
    _seed_ball_at_centre(model, data)

    # Logging at 200 Hz (controls are at 1 kHz, physics at 2 kHz).
    dt = float(model.opt.timestep)
    log_every = 10
    control_every = 2
    n_steps = int(round(duration / dt))

    log: dict[str, list] = {
        key: []
        for key in (
            "t",
            "p",
            "v",
            "p_ref",
            "v_ref",
            "err",
            "err_rad",
            "err_tan",
            "phase",
            "radius_ref",
            "roll",
            "pitch",
            "roll_ref",
            "pitch_ref",
            "tau_pitch",
            "tau_roll",
            "ball_world",
            "contact_world",
            "slip",
            "normal_force",
        )
    }

    # Rendering setup.
    renderer = None
    cam = None
    writer = None
    trail = deque(maxlen=180)
    preview_saved = False
    if render:
        try:
            renderer = mujoco.Renderer(model, height=height, width=width)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0.0, 0.0, 0.625]
            cam.distance = 1.12
            cam.azimuth = 135.0
            cam.elevation = -27.0
            import imageio.v2 as imageio

            video_path = outdir / "ball_on_plate_circle.mp4"
            writer = imageio.get_writer(
                video_path,
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
                pixelformat="yuv420p",
                ffmpeg_params=["-crf", "18", "-preset", "medium"],
            )
        except Exception as exc:  # rendering is optional
            print(f"[warn] off-screen rendering unavailable ({exc}); continuing headless.")
            render = False
            renderer = None
            writer = None

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
        ref = reference.sample(float(data.time))
        err = ref.p - state.p[:2]
        e_rad = float(err @ np.array([np.cos(ref.phase), np.sin(ref.phase)]))
        e_tan = float(err @ np.array([-np.sin(ref.phase), np.cos(ref.phase)]))
        normal_force, slip = _contact_metrics(model, data, state, ctrl_cfg.ball_radius)
        contact_pt = state.ball_world - ctrl_cfg.ball_radius * state.R[:, 2]
        contact_pt = contact_pt + 0.0008 * state.R[:, 2]

        if step % log_every == 0:
            log["t"].append(float(data.time))
            log["p"].append(state.p[:2].copy())
            log["v"].append(state.v[:2].copy())
            log["p_ref"].append(ref.p.copy())
            log["v_ref"].append(ref.v.copy())
            log["err"].append(float(np.linalg.norm(err)))
            log["err_rad"].append(e_rad)
            log["err_tan"].append(e_tan)
            log["phase"].append(float(ref.phase))
            log["radius_ref"].append(float(ref.radius))
            log["roll"].append(state.roll)
            log["pitch"].append(state.pitch)
            log["roll_ref"].append(last_diag.roll_des if last_diag else 0.0)
            log["pitch_ref"].append(last_diag.pitch_des if last_diag else 0.0)
            log["tau_pitch"].append(float(last_diag.tau[0]) if last_diag else 0.0)
            log["tau_roll"].append(float(last_diag.tau[1]) if last_diag else 0.0)
            log["ball_world"].append(state.ball_world.copy())
            log["contact_world"].append(contact_pt.copy())
            log["slip"].append(slip)
            log["normal_force"].append(normal_force)

        # Render frames on a wall-clock-like schedule (frame times, not steps).
        if render and data.time + 1e-9 >= next_frame and data.time < duration - 1e-9:
            trail.append(contact_pt)
            renderer.update_scene(data, camera=cam)
            _add_trail(renderer.scene, trail, ctrl_cfg.ball_radius)
            frame = renderer.render()

            err_mm = float(np.linalg.norm(err) * 1000.0)
            frame = _hud(frame, float(data.time), err_mm, ref_cfg)
            writer.append_data(np.ascontiguousarray(frame))

            if not preview_saved and data.time >= 10.0:
                from PIL import Image

                Image.fromarray(frame).save(outdir / "preview.png")
                preview_saved = True
            next_frame += frame_period

        if step < n_steps:
            mujoco.mj_step(model, data)

    if writer is not None:
        writer.close()
        wall = time.perf_counter() - wall_start
        if verbose:
            print(
                f"[video] {outdir / 'ball_on_plate_circle.mp4'} "
                f"({duration:.1f} s, {width}x{height}, {fps} fps, wall {wall:.1f} s)"
            )
        if make_preview_gif:
            gif_path = outdir / "preview.gif"
            if _make_gif(
                outdir / "ball_on_plate_circle.mp4",
                gif_path,
                start=min(8.0, max(0.0, duration - 8.0)),
                duration=min(8.0, duration),
            ):
                if verbose:
                    print(f"[gif]   {gif_path}")

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    arrays = {k: np.asarray(v) for k, v in log.items()}
    t = arrays["t"]
    e = arrays["err"]
    steady_start = max(ref_cfg.ramp_time + 4.0, 0.62 * duration)
    steady_start = min(steady_start, max(0.0, duration - 2.0))
    steady = t >= steady_start

    err_rad = arrays["err_rad"]
    err_tan = arrays["err_tan"]
    p_actual = arrays["p"]
    mean_radius = float(np.mean(np.linalg.norm(p_actual[steady], axis=1)))
    phase_lag = float(
        np.degrees(np.arctan2(np.mean(err_tan[steady]), max(mean_radius, 1e-9)))
    )

    metrics = {
        "duration_s": float(duration),
        "physics_dt_s": dt,
        "control_rate_hz": 1.0 / (dt * control_every),
        "reference_radius_m": ref_cfg.radius,
        "reference_period_s": ref_cfg.period,
        "ramp_time_s": ref_cfg.ramp_time,
        "steady_window_s": [float(t[steady][0]), float(t[-1])],
        "steady_rms_mm": float(np.sqrt(np.mean(e[steady] ** 2)) * 1000.0),
        "steady_max_mm": float(np.max(e[steady]) * 1000.0),
        "steady_radial_rms_mm": float(np.sqrt(np.mean(err_rad[steady] ** 2)) * 1000.0),
        "steady_tangential_rms_mm": float(np.sqrt(np.mean(err_tan[steady] ** 2)) * 1000.0),
        "steady_mean_radius_m": mean_radius,
        "steady_phase_lag_deg": phase_lag,
        "overall_rms_mm": float(np.sqrt(np.mean(e**2)) * 1000.0),
        "max_tilt_deg": float(
            np.degrees(max(np.max(np.abs(arrays["pitch"])), np.max(np.abs(arrays["roll"]))))
        ),
        "max_torque_nm": float(
            max(np.max(np.abs(arrays["tau_pitch"])), np.max(np.abs(arrays["tau_roll"])))
        ),
        "torque_limit": float(model.actuator_ctrlrange[0, 1]),
        "saturation_fraction": float(sat_count / max(ctrl_count, 1)),
        "slip_rms_mm_s": float(np.sqrt(np.mean(arrays["slip"] ** 2)) * 1000.0),
        "slip_max_mm_s": float(np.max(arrays["slip"]) * 1000.0),
        "normal_force_mean_n": float(np.mean(arrays["normal_force"])),
        "normal_force_min_n": float(np.min(arrays["normal_force"])),
        "normal_force_max_n": float(np.max(arrays["normal_force"])),
        "final_error_mm": float(e[-1] * 1000.0),
        "radius": ref_cfg.radius,
        "steady_start": float(steady_start),
    }

    np.savez_compressed(outdir / "trajectory.npz", **arrays)
    with open(outdir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    make_plots(arrays, metrics, outdir / "tracking.png")

    with open(outdir / "metrics.txt", "w", encoding="utf-8") as fh:
        fh.write("Ball-on-plate circular tracking\n")
        fh.write("===============================\n")
        for k, v in metrics.items():
            fh.write(f"{k}: {v}\n")

    if verbose:
        print("\n=== tracking result ===")
        print(f"steady RMS error     : {metrics['steady_rms_mm']:.3f} mm")
        print(f"steady max error     : {metrics['steady_max_mm']:.3f} mm")
        print(f"radial / tangential  : {metrics['steady_radial_rms_mm']:.3f} / "
              f"{metrics['steady_tangential_rms_mm']:.3f} mm RMS")
        print(f"phase lag            : {metrics['steady_phase_lag_deg']:.3f} deg")
        print(f"max plate tilt       : {metrics['max_tilt_deg']:.2f} deg")
        print(f"max joint torque     : {metrics['max_torque_nm']:.4f} N m")
        print(f"slip speed RMS       : {metrics['slip_rms_mm_s']:.3f} mm/s")
        print(f"outputs              : {outdir.resolve()}")

    if renderer is not None:
        renderer.close()
    return {"metrics": metrics, "log": arrays, "model": model, "data": data}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0, help="simulated seconds")
    parser.add_argument("--no-video", action="store_true", help="skip rendering")
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args(argv)

    run(
        duration=args.duration,
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
