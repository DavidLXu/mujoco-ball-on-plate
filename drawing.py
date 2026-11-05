"""Fourier-descriptor drawing references for the ball-on-plate plant.

The default portrait is the outer silhouette of the public-domain engraving
"Joseph Fourier" by A. F. B. Geille after J. Boilly (Wikimedia Commons).  The
closed contour is extracted by extract_portrait.py; only the silhouette is used
because the ball can draw one continuous closed stroke (not the engraving's
interior hatching).

A closed one-stroke drawing is represented as a complex Fourier series

    z(u) = sum_k c_k exp(2*pi*i*k*u),      u in [0, 1).

The coefficients are obtained with an FFT of a densely sampled closed polyline.
Keeping K harmonics gives a smooth, band-limited drawing that is compatible
with the finite acceleration/tilt authority of the plate.

The reference is then time-parameterised with a curvature-aware speed limit:
sharp details (nose, cravat) are traversed slowly, long smooth strokes quickly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d

from controller import ReferenceState


# --------------------------------------------------------------------------- #
# Default portrait: closed silhouette extracted from the real engraving.
# The extraction script and source image are in extract_portrait.py / assets/.
# --------------------------------------------------------------------------- #
_PORTRAIT_POINTS = Path(__file__).resolve().parent / "assets" / "fourier_portrait_points.npy"
if not _PORTRAIT_POINTS.exists():
    raise FileNotFoundError(
        f"missing {_PORTRAIT_POINTS}; run `python extract_portrait.py` first"
    )
FOURIER_PROFILE_POINTS = np.load(_PORTRAIT_POINTS)


@dataclass
class FourierPath:
    """A closed 2-D curve represented by a truncated complex Fourier series."""

    coeff: np.ndarray          # complex coefficients, ordered like `ks`
    ks: np.ndarray             # integer harmonics, [0..K, -K..-1]
    n_samples: int             # samples used for the FFT
    all_coeff: np.ndarray      # full FFT, for the spectrum plot
    points: np.ndarray         # dense reconstructed path, (N, 2)
    target_span: float
    smooth_sigma: float
    input_center: np.ndarray   # centre removed before scaling
    input_scale: float         # scale applied to normalised input

    @property
    def harmonics(self) -> int:
        return int(np.max(np.abs(self.ks)))

    def evaluate(self, u):
        u = np.asarray(u, dtype=float)
        E = np.exp(2j * np.pi * np.outer(u, self.ks))
        z = E @ self.coeff
        dz = E @ (self.coeff * (2j * np.pi * self.ks))
        ddz = E @ (self.coeff * (-((2 * np.pi * self.ks) ** 2)))
        return z, dz, ddz

    def spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Return positive harmonic numbers and their magnitudes."""
        k = np.arange(0, self.harmonics + 1)
        mag = np.abs(self.all_coeff[: self.harmonics + 1])
        return k, mag

    def cumulative_energy(self) -> np.ndarray:
        """Fraction of AC energy captured by the first K harmonics."""
        p = np.abs(self.all_coeff) ** 2
        p[0] = 0.0
        total = p.sum()
        out = np.zeros(self.harmonics + 1)
        for k in range(1, self.harmonics + 1):
            out[k] = (p[1 : k + 1].sum() + p[-k:].sum()) / total
        return out


def make_fourier_path(
    points: np.ndarray,
    harmonics: int = 50,
    n_samples: int = 8192,
    smooth_sigma: float = 8.0,
    target_span: float = 0.36,
) -> FourierPath:
    """Fit a closed Fourier descriptor curve to a polyline.

    Parameters
    ----------
    points:
        Closed/open polyline in arbitrary 2-D coordinates.  It is resampled by
        arc length and periodically smoothed before the FFT.
    harmonics:
        Number of positive harmonics kept.  The negative-frequency conjugate
        coefficients are kept as well, so 2K+1 complex terms are used.
    target_span:
        Extent of the longest bounding-box side after fitting, in metres.
    """
    pts = np.asarray(points, dtype=float).copy()
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    # Remove duplicate points for the interpolation (zero-length segments).
    keep = np.r_[True, np.diff(s) > 1e-12]
    s, pts = s[keep], pts[keep]

    su = np.linspace(0.0, s[-1], n_samples, endpoint=False)
    x = np.interp(su, s, pts[:, 0])
    y = np.interp(su, s, pts[:, 1])
    if smooth_sigma > 0:
        x = gaussian_filter1d(x, smooth_sigma, mode="wrap")
        y = gaussian_filter1d(y, smooth_sigma, mode="wrap")

    z = x + 1j * y
    cx = 0.5 * (x.min() + x.max())
    cy = 0.5 * (y.min() + y.max())
    z -= cx + 1j * cy
    span = max(np.ptp(x), np.ptp(y))
    input_scale = target_span / span
    z *= input_scale

    all_coeff = np.fft.fft(z) / n_samples
    ks = np.r_[np.arange(0, harmonics + 1), np.arange(-harmonics, 0)]
    idx = np.r_[np.arange(0, harmonics + 1), np.arange(n_samples - harmonics, n_samples)]
    coeff = all_coeff[idx]

    path = FourierPath(
        coeff=coeff,
        ks=ks,
        n_samples=n_samples,
        all_coeff=all_coeff,
        points=np.zeros((n_samples, 2)),
        target_span=target_span,
        smooth_sigma=smooth_sigma,
        input_center=np.array([cx, cy]),
        input_scale=float(input_scale),
    )
    u = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    zz, _, _ = path.evaluate(u)
    path.points = np.c_[zz.real, zz.imag]
    return path


class FourierDrawing:
    """Time-parameterised one-stroke drawing, sampled at 2 kHz.

    The path speed is limited by curvature (centripetal acceleration) and by a
    tangential acceleration limit.  The resulting flatness profile is a good
    feed-forward reference for the cascaded ball-on-plate controller.
    """

    def __init__(
        self,
        path: FourierPath,
        a_lat: float = 0.60,
        a_tan: float = 0.12,
        v_max: float = 0.16,
        v_min: float = 0.004,
        fs: float = 2000.0,
    ):
        self.path = path
        self.a_lat = a_lat
        self.a_tan = a_tan
        self.v_max = v_max
        self.v_min = v_min
        self.fs = fs

        n = path.n_samples
        u = np.linspace(0.0, 1.0, n + 1)
        z, dz, ddz = path.evaluate(u)
        ds = np.abs(np.diff(z))
        self.path_length = float(ds.sum())

        kappa = np.abs(np.imag(np.conj(dz) * ddz)) / np.maximum(np.abs(dz) ** 3, 1e-12)
        kappa = gaussian_filter1d(kappa, sigma=4.0, mode="nearest")
        v_lim = np.minimum(v_max, np.sqrt(a_lat / np.maximum(kappa, 1e-3)))
        # A moving minimum makes the profile conservative around sharp details;
        # the subsequent Gaussian removes the staircase and lets the tangential
        # acceleration limit produce a smooth spin-up / slow-down.
        v_lim = minimum_filter1d(v_lim, size=81, mode="nearest")
        v_lim = gaussian_filter1d(v_lim, sigma=60.0, mode="nearest")
        v_lim = np.maximum(v_lim, v_min)
        self.curvature = kappa
        self.speed_limit = v_lim

        v = v_lim.copy()
        v[0] = 0.0
        v[-1] = 0.0
        for i in range(1, len(v)):
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_tan * ds[i - 1]))
        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_tan * ds[i]))

        v_eff = np.maximum(v, v_min)
        dt = ds / (0.5 * (v_eff[:-1] + v_eff[1:]))
        t_nodes = np.r_[0.0, np.cumsum(dt)]
        self.T = float(t_nodes[-1])
        self.t_nodes = t_nodes
        self.u_nodes = u
        self.v_nodes = v
        self.z_nodes = z
        self.ds_nodes = ds

        # Analytic derivatives with respect to time.
        udot = v / np.maximum(np.abs(dz), 1e-12)
        zdot = dz * udot
        vdot = np.gradient(v, t_nodes)
        zz_dot_dz = np.real(np.conj(dz) * ddz)
        uddot = (vdot - zz_dot_dz / np.maximum(np.abs(dz), 1e-12) * udot**2) / np.maximum(
            np.abs(dz), 1e-12
        )
        zddot = ddz * udot**2 + dz * uddot

        self.p_nodes = np.c_[z.real, z.imag]
        self.v_nodes_xy = np.c_[zdot.real, zdot.imag]
        self.a_nodes = np.c_[zddot.real, zddot.imag]
        self.max_speed = float(np.max(np.linalg.norm(self.v_nodes_xy, axis=1)))
        self.max_accel = float(np.max(np.linalg.norm(self.a_nodes, axis=1)))
        self.p0 = self.p_nodes[0].copy()

        # Uniform-time table for cheap interpolation in the control loop.
        self.t_grid = np.arange(0.0, self.T + 0.5 / fs, 1.0 / fs)
        self.p_grid = np.c_[
            np.interp(self.t_grid, t_nodes, self.p_nodes[:, 0]),
            np.interp(self.t_grid, t_nodes, self.p_nodes[:, 1]),
        ]
        self.v_grid = np.c_[
            np.interp(self.t_grid, t_nodes, self.v_nodes_xy[:, 0]),
            np.interp(self.t_grid, t_nodes, self.v_nodes_xy[:, 1]),
        ]
        self.a_grid = np.c_[
            np.interp(self.t_grid, t_nodes, self.a_nodes[:, 0]),
            np.interp(self.t_grid, t_nodes, self.a_nodes[:, 1]),
        ]

    def sample(self, t: float) -> ReferenceState:
        tc = float(np.clip(t, 0.0, self.T))
        p = np.array([np.interp(tc, self.t_grid, self.p_grid[:, 0]),
                      np.interp(tc, self.t_grid, self.p_grid[:, 1])])
        v = np.array([np.interp(tc, self.t_grid, self.v_grid[:, 0]),
                      np.interp(tc, self.t_grid, self.v_grid[:, 1])])
        a = np.array([np.interp(tc, self.t_grid, self.a_grid[:, 0]),
                      np.interp(tc, self.t_grid, self.a_grid[:, 1])])
        return ReferenceState(
            t=tc,
            p=p,
            v=v,
            a=a,
            radius=float(np.linalg.norm(p)),
            phase=float(2.0 * np.pi * tc / self.T),
        )

    def progress(self, t: float) -> float:
        return float(np.clip(t / self.T, 0.0, 1.0))
