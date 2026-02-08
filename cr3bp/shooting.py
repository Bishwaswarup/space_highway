"""Shooting methods for periodic orbits."""

from __future__ import annotations

import numpy as np

from .integrators import rk45_integrate
from .variational import variational_eom


def _integrate_to_y_crossing(
    state0: np.ndarray,
    mu: float,
    *,
    h_max: float = 0.05,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> tuple[float, np.ndarray, np.ndarray]:
    phi0 = np.eye(6).reshape(-1)
    y0 = np.hstack([state0, phi0])

    def deriv(t: float, y: np.ndarray) -> np.ndarray:
        return variational_eom(t, y, mu)

    ts, ys = None, None
    for tf in (20.0, 40.0, 80.0):
        ts, ys = rk45_integrate(
            deriv,
            y0,
            0.0,
            tf,
            rtol=rtol,
            atol=atol,
            h_max=h_max,
        )
        y_vals = ys[:, 1]
        cross_idx = None
        for i in range(1, len(ts)):
            if ts[i] <= 0.0:
                continue
            if y_vals[i - 1] == 0.0:
                continue
            if y_vals[i - 1] * y_vals[i] < 0:
                cross_idx = i
                break
        if cross_idx is not None:
            break

    if cross_idx is None:
        raise RuntimeError("No y=0 crossing found in integration window.")

    t0, t1 = ts[cross_idx - 1], ts[cross_idx]
    y0_aug, y1_aug = ys[cross_idx - 1], ys[cross_idx]
    alpha = abs(y0_aug[1]) / (abs(y0_aug[1]) + abs(y1_aug[1]))
    t_cross = t0 + alpha * (t1 - t0)
    y_cross = y0_aug + alpha * (y1_aug - y0_aug)

    state_cross = y_cross[:6]
    phi_cross = y_cross[6:].reshape((6, 6))
    return t_cross, state_cross, phi_cross


def find_planar_lyapunov(
    mu: float,
    *,
    x0: float,
    vy0_guess: float,
    tol: float = 1e-10,
    max_iter: int = 20,
) -> dict[str, np.ndarray | float]:
    """Differential correction for a planar Lyapunov orbit about L1 or L2.

    Uses symmetry: start at y=0, z=0, vx=0; correct vy0 to enforce vx=0 at y=0 crossing.
    """
    state0 = np.array([x0, 0.0, 0.0, 0.0, vy0_guess, 0.0], dtype=float)

    for _ in range(max_iter):
        t_half, state_half, phi_half = _integrate_to_y_crossing(state0, mu)
        vx_half = state_half[3]
        if abs(vx_half) < tol:
            period = 2 * t_half
            return {
                "state0": state0.copy(),
                "half_period": t_half,
                "period": period,
                "state_half": state_half.copy(),
                "phi_half": phi_half.copy(),
            }

        dvx_dvy0 = phi_half[3, 4]
        if dvx_dvy0 == 0:
            raise RuntimeError("Singular correction matrix (dvx/dvy0 = 0).")

        delta_vy0 = -vx_half / dvx_dvy0
        state0[4] += delta_vy0

    raise RuntimeError("Differential correction did not converge.")


def find_halo_orbit(
    mu: float,
    *,
    x0: float,
    z0_guess: float,
    vy0_guess: float,
    tol: float = 1e-10,
    max_iter: int = 30,
) -> dict[str, np.ndarray | float]:
    """Differential correction for a 3D halo orbit about L1 or L2.

    Uses symmetry: start at y=0 with vx=0 and vz=0, correct (vy0, z0)
    to enforce vx=0 and vz=0 at the next y=0 crossing.
    """
    state0 = np.array([x0, 0.0, z0_guess, 0.0, vy0_guess, 0.0], dtype=float)

    for _ in range(max_iter):
        t_half, state_half, phi_half = _integrate_to_y_crossing(state0, mu)
        vx_half = state_half[3]
        vz_half = state_half[5]

        if max(abs(vx_half), abs(vz_half)) < tol:
            period = 2 * t_half
            return {
                "state0": state0.copy(),
                "half_period": t_half,
                "period": period,
                "state_half": state_half.copy(),
                "phi_half": phi_half.copy(),
            }

        jac = np.array(
            [
                [phi_half[3, 4], phi_half[3, 2]],
                [phi_half[5, 4], phi_half[5, 2]],
            ],
            dtype=float,
        )
        rhs = -np.array([vx_half, vz_half], dtype=float)

        if abs(np.linalg.det(jac)) < 1e-14:
            raise RuntimeError("Singular correction matrix for halo orbit.")

        delta = np.linalg.solve(jac, rhs)
        norm_delta = np.linalg.norm(delta)
        if norm_delta > 0.1:
            delta = delta * (0.1 / norm_delta)
        state0[4] += delta[0]
        state0[2] += delta[1]

    raise RuntimeError("Halo differential correction did not converge.")


def monodromy_matrix(
    mu: float,
    state0: np.ndarray,
    period: float,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> np.ndarray:
    """Integrate STM over one period to obtain the monodromy matrix."""
    phi0 = np.eye(6).reshape(-1)
    y0 = np.hstack([state0, phi0])

    def deriv(t: float, y: np.ndarray) -> np.ndarray:
        return variational_eom(t, y, mu)

    ts, ys = rk45_integrate(deriv, y0, 0.0, period, rtol=rtol, atol=atol)
    y_final = ys[-1]
    return y_final[6:].reshape((6, 6))
