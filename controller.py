"""Dynamics-aware control for the MuJoCo ball-on-plate plant.

The controller is deliberately split into the two time scales that a real
ball-on-plate machine has:

1. An outer Cartesian loop treats the ball as a rolling sphere on a (slowly)
   tilting plate.  For a uniform solid sphere rolling without slipping,

       a_ball = (5/7) * g_tangent,

   where g_tangent is the component of gravity in the plate plane.  Given a
   desired ball acceleration, the required plate roll/pitch is obtained by
   inverting that relation analytically.  This is feedback linearisation of
   the rolling constraint, not a scripted ball motion.

2. An inner joint-space PD loop tracks the commanded plate angles with the
   two hinge torques.  The ball is never actuated; it only feels contact and
   friction from the plate.

All positions/velocities used by the outer loop are expressed in the plate
frame, because that is the frame in which the rolling constraint is defined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Reference trajectory
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CircleReferenceConfig:
    radius: float = 0.16          # [m]
    period: float = 4.0           # [s]
    ramp_time: float = 4.0        # [s] smooth spin-up from the plate centre
    clockwise: bool = False


@dataclass(frozen=True)
class ReferenceState:
    t: float
    p: np.ndarray        # desired ball position in plate frame [m]
    v: np.ndarray        # desired ball velocity in plate frame [m/s]
    a: np.ndarray        # desired ball acceleration in plate frame [m/s^2]
    radius: float        # instantaneous reference radius [m]
    phase: float         # reference phase [rad]


class CircleReference:
    """Smoothly started circular reference with analytic derivatives.

    The radius is multiplied by a quintic smootherstep.  Both its first and
    second derivatives vanish at the ends, so at t=0 the ball starts at rest
    at the plate centre and the reference acceleration is exactly zero.
    """

    def __init__(self, cfg: CircleReferenceConfig = CircleReferenceConfig()):
        self.cfg = cfg
        self.omega = (1.0 if cfg.clockwise else -1.0) * 2.0 * np.pi / cfg.period

    @staticmethod
    def _smootherstep(t: float, T: float) -> tuple[float, float, float]:
        if t <= 0.0:
            return 0.0, 0.0, 0.0
        if t >= T:
            return 1.0, 0.0, 0.0
        u = t / T
        s = 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
        sd = (30.0 * u**4 - 60.0 * u**3 + 30.0 * u**2) / T
        sdd = (120.0 * u**3 - 180.0 * u**2 + 60.0 * u) / T**2
        return s, sd, sdd

    def sample(self, t: float) -> ReferenceState:
        cfg = self.cfg
        s, sd, sdd = self._smootherstep(t, cfg.ramp_time)
        r = cfg.radius * s
        rd = cfg.radius * sd
        rdd = cfg.radius * sdd

        phase = self.omega * t
        c, s_phi = np.cos(phase), np.sin(phase)
        e_r = np.array([c, s_phi])
        e_t = np.array([-s_phi, c])

        p = r * e_r
        v = rd * e_r + r * self.omega * e_t
        a = (rdd - r * self.omega**2) * e_r + (2.0 * rd * self.omega) * e_t
        return ReferenceState(t=t, p=p, v=v, a=a, radius=r, phase=phase)


# --------------------------------------------------------------------------- #
# Plant state
# --------------------------------------------------------------------------- #
@dataclass
class PlateBallState:
    roll: float
    pitch: float
    roll_rate: float
    pitch_rate: float
    R: np.ndarray              # plate -> world rotation
    origin: np.ndarray         # plate hinge origin in world
    omega_world: np.ndarray    # plate angular velocity in world
    ball_world: np.ndarray
    ball_vel_world: np.ndarray
    ball_spin_world: np.ndarray
    p: np.ndarray              # ball position in plate frame [m]
    v: np.ndarray              # ball velocity relative to plate, plate frame [m/s]


def read_plate_ball_state(model, data) -> PlateBallState:
    """Read the physical state and express the ball state in the plate frame."""
    plate_id = model.body("plate").id
    ball_id = model.body("ball").id
    pitch_j = model.joint("pitch").id
    roll_j = model.joint("roll").id
    ball_j = model.joint("ball_free").id

    roll = float(data.qpos[model.jnt_qposadr[roll_j]])
    pitch = float(data.qpos[model.jnt_qposadr[pitch_j]])
    roll_rate = float(data.qvel[model.jnt_dofadr[roll_j]])
    pitch_rate = float(data.qvel[model.jnt_dofadr[pitch_j]])

    R = np.asarray(data.body(plate_id).xmat, dtype=float).reshape(3, 3).copy()
    origin = np.asarray(data.body(plate_id).xpos, dtype=float).copy()

    # d/dt(R) = [omega_world]x R  and  R = Ry(pitch) Rx(roll), hence
    # omega_plate = R * (roll_rate, pitch_rate*cos(roll), -pitch_rate*sin(roll)).
    omega_world = R @ np.array(
        [roll_rate, pitch_rate * np.cos(roll), -pitch_rate * np.sin(roll)]
    )

    ball_world = np.asarray(data.body(ball_id).xpos, dtype=float).copy()
    R_ball = np.asarray(data.body(ball_id).xmat, dtype=float).reshape(3, 3).copy()
    dof = model.jnt_dofadr[ball_j]
    ball_vel_world = np.asarray(data.qvel[dof : dof + 3], dtype=float).copy()
    # A free joint stores angular velocity in the body's local frame.
    ball_spin_world = R_ball @ np.asarray(
        data.qvel[dof + 3 : dof + 6], dtype=float
    )

    # Velocity of the plate material point coincident with the ball centre.
    v_plate_at_ball = np.cross(omega_world, ball_world - origin)
    v_rel_world = ball_vel_world - v_plate_at_ball

    p = R.T @ (ball_world - origin)
    v = R.T @ v_rel_world
    return PlateBallState(
        roll=roll,
        pitch=pitch,
        roll_rate=roll_rate,
        pitch_rate=pitch_rate,
        R=R,
        origin=origin,
        omega_world=omega_world,
        ball_world=ball_world,
        ball_vel_world=ball_vel_world,
        ball_spin_world=ball_spin_world,
        p=p,
        v=v,
    )


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
@dataclass
class ControllerConfig:
    # outer Cartesian loop [1/s^2], [1/s]
    kp: float = 60.0
    kd: float = 18.0
    ki: float = 0.0
    # optional first-order filters; 0 disables them
    vel_filter_hz: float = 0.0
    joint_rate_filter_hz: float = 0.0
    # inner plate-angle loop [N*m/rad], [N*m*s/rad]
    kp_joint: float = 140.0
    kd_joint: float = 18.0

    max_tilt: float = 0.30              # [rad]
    max_accel: float = 2.0              # [m/s^2]
    integral_limit: float = 0.02        # [m*s]
    ball_radius: float = 0.025          # [m]


@dataclass
class ControlDiagnostics:
    t: float
    reference: ReferenceState
    state: PlateBallState
    error: np.ndarray
    error_rate: np.ndarray
    accel_cmd: np.ndarray
    roll_des: float
    pitch_des: float
    tau: np.ndarray
    saturated: bool = False


class BallPlateController:
    """Cascaded feedback-linearising controller.

    Outer loop:
        a_cmd = a_ref + Kp (p_ref - p) + Kd (v_ref - v) [+ Ki integral]
        a_cmd = (5/7) g [ sin(pitch), -sin(roll) cos(pitch) ]
        => pitch_des, roll_des

    Inner loop:
        tau = Kp_j (theta_des - theta) - Kd_j theta_dot
    """

    def __init__(
        self,
        model,
        reference: CircleReference,
        cfg: ControllerConfig = ControllerConfig(),
    ):
        self.model = model
        self.reference = reference
        self.cfg = cfg
        self.integral = np.zeros(2)
        self._last_t: float | None = None
        self._vel_filtered: np.ndarray | None = None
        self._pitch_rate_f = 0.0
        self._roll_rate_f = 0.0
        self.last: ControlDiagnostics | None = None

        self.pitch_joint = model.joint("pitch").id
        self.roll_joint = model.joint("roll").id
        self.pitch_act = model.actuator("pitch_motor").id
        self.roll_act = model.actuator("roll_motor").id

    def reset(self) -> None:
        self.integral[:] = 0.0
        self._last_t = None
        self._vel_filtered = None
        self._pitch_rate_f = 0.0
        self._roll_rate_f = 0.0
        self.last = None

    def observe(self, data) -> PlateBallState:
        return read_plate_ball_state(self.model, data)

    def update(self, data) -> ControlDiagnostics:
        cfg = self.cfg
        state = self.observe(data)
        now = float(data.time)
        ref = self.reference.sample(now)

        e = ref.p - state.p[:2]
        dt = now - self._last_t if self._last_t is not None else float(self.model.opt.timestep)
        dt = max(dt, 1e-6)

        v_meas = state.v[:2]
        if cfg.vel_filter_hz > 0.0:
            alpha = 1.0 - np.exp(-2.0 * np.pi * cfg.vel_filter_hz * dt)
            if self._vel_filtered is None:
                self._vel_filtered = v_meas.copy()
            else:
                self._vel_filtered += alpha * (v_meas - self._vel_filtered)
            v_meas = self._vel_filtered
        ed = ref.v - v_meas

        if cfg.ki > 0.0:
            self.integral += e * dt
            norm = np.linalg.norm(self.integral)
            if norm > cfg.integral_limit:
                self.integral *= cfg.integral_limit / norm

        a_cmd = ref.a + cfg.kp * e + cfg.kd * ed + cfg.ki * self.integral

        # Keep the commanded acceleration inside the achievable tilt envelope.
        an = float(np.linalg.norm(a_cmd))
        if an > cfg.max_accel:
            a_cmd = a_cmd * (cfg.max_accel / an)

        # Invert a = (5/7) g [sin(beta), -sin(alpha) cos(beta)].
        g = 9.81
        s_max = np.sin(cfg.max_tilt)
        pitch_des = float(np.arcsin(np.clip((7.0 / 5.0) * a_cmd[0] / g, -s_max, s_max)))
        cy = np.cos(pitch_des)
        roll_des = float(
            -np.arcsin(
                np.clip((7.0 / 5.0) * a_cmd[1] / (g * cy), -s_max, s_max)
            )
        )

        pitch_rate, roll_rate = state.pitch_rate, state.roll_rate
        if cfg.joint_rate_filter_hz > 0.0:
            alpha = 1.0 - np.exp(-2.0 * np.pi * cfg.joint_rate_filter_hz * dt)
            self._pitch_rate_f += alpha * (pitch_rate - self._pitch_rate_f)
            self._roll_rate_f += alpha * (roll_rate - self._roll_rate_f)
            pitch_rate, roll_rate = self._pitch_rate_f, self._roll_rate_f

        tau = np.array(
            [
                cfg.kp_joint * (pitch_des - state.pitch)
                - cfg.kd_joint * pitch_rate,
                cfg.kp_joint * (roll_des - state.roll)
                - cfg.kd_joint * roll_rate,
            ]
        )
        lo = self.model.actuator_ctrlrange[self.pitch_act, 0]
        hi = self.model.actuator_ctrlrange[self.pitch_act, 1]
        tau_clipped = np.clip(tau, lo, hi)
        saturated = bool(np.any(np.abs(tau_clipped - tau) > 1e-12))

        data.ctrl[self.pitch_act] = tau_clipped[0]
        data.ctrl[self.roll_act] = tau_clipped[1]

        self._last_t = now
        diag = ControlDiagnostics(
            t=now,
            reference=ref,
            state=state,
            error=e,
            error_rate=ed,
            accel_cmd=a_cmd,
            roll_des=roll_des,
            pitch_des=pitch_des,
            tau=tau_clipped,
            saturated=saturated,
        )
        self.last = diag
        return diag
