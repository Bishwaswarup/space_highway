"""Continuation tools for families of halo orbits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .shooting import _integrate_to_y_crossing, find_halo_orbit, find_planar_lyapunov


@dataclass
class HaloOrbit:
    state0: np.ndarray
    period: float
    z0: float
    vy0: float


def halo_continuation(
    mu: float,
    *,
    x0: float,
    z0_start: float,
    vy0_start: float,
    z0_step: float,
    n_orbits: int,
) -> List[HaloOrbit]:
    """Generate a family of halo orbits via simple continuation in z0."""
    orbits: List[HaloOrbit] = []
    z0 = z0_start
    vy0 = vy0_start

    for _ in range(n_orbits):
        result = find_halo_orbit(mu, x0=x0, z0_guess=z0, vy0_guess=vy0)
        state0 = result["state0"]
        period = float(result["period"])
        z0 = float(state0[2])
        vy0 = float(state0[4])
        orbits.append(HaloOrbit(state0=state0.copy(), period=period, z0=z0, vy0=vy0))
        z0 += z0_step

    return orbits


def _halo_correction_with_arc(
    mu: float,
    u_guess: np.ndarray,
    u_pred: np.ndarray,
    tangent: np.ndarray,
    *,
    tol: float = 1e-10,
    max_iter: int = 40,
) -> Tuple[np.ndarray, float]:
    u = u_guess.copy()
    for _ in range(max_iter):
        x0, z0, vy0 = u
        state0 = np.array([x0, 0.0, z0, 0.0, vy0, 0.0], dtype=float)
        t_half, state_half, phi_half = _integrate_to_y_crossing(state0, mu)
        vx_half = state_half[3]
        vz_half = state_half[5]

        f1 = vx_half
        f2 = vz_half
        f3 = float(np.dot(u - u_pred, tangent))

        if max(abs(f1), abs(f2), abs(f3)) < tol:
            return u, 2.0 * t_half

        jac = np.array(
            [
                [phi_half[3, 0], phi_half[3, 2], phi_half[3, 4]],
                [phi_half[5, 0], phi_half[5, 2], phi_half[5, 4]],
                [tangent[0], tangent[1], tangent[2]],
            ],
            dtype=float,
        )
        rhs = -np.array([f1, f2, f3], dtype=float)

        if abs(np.linalg.det(jac)) < 1e-14:
            raise RuntimeError("Singular Jacobian in pseudo-arclength correction.")

        delta = np.linalg.solve(jac, rhs)
        norm_delta = np.linalg.norm(delta)
        if norm_delta > 0.05:
            delta = delta * (0.05 / norm_delta)
        u += delta

    raise RuntimeError("Pseudo-arclength correction did not converge.")


def halo_pseudo_arclength(
    mu: float,
    *,
    x0_a: float,
    z0_a: float,
    vy0_a: float,
    x0_b: float,
    z0_b: float,
    vy0_b: float,
    ds: float,
    n_steps: int,
) -> List[HaloOrbit]:
    """Generate a family of halo orbits using pseudo-arclength continuation."""
    orbits: List[HaloOrbit] = []

    sol_a = find_halo_orbit(mu, x0=x0_a, z0_guess=z0_a, vy0_guess=vy0_a)
    sol_b = find_halo_orbit(mu, x0=x0_b, z0_guess=z0_b, vy0_guess=vy0_b)

    u_a = np.array([sol_a["state0"][0], sol_a["state0"][2], sol_a["state0"][4]], dtype=float)
    u_b = np.array([sol_b["state0"][0], sol_b["state0"][2], sol_b["state0"][4]], dtype=float)

    orbits.append(
        HaloOrbit(
            state0=sol_a["state0"].copy(),
            period=float(sol_a["period"]),
            z0=float(sol_a["state0"][2]),
            vy0=float(sol_a["state0"][4]),
        )
    )
    orbits.append(
        HaloOrbit(
            state0=sol_b["state0"].copy(),
            period=float(sol_b["period"]),
            z0=float(sol_b["state0"][2]),
            vy0=float(sol_b["state0"][4]),
        )
    )

    for _ in range(n_steps):
        tangent = u_b - u_a
        tangent /= np.linalg.norm(tangent)
        u_pred = u_b + ds * tangent
        u_guess = u_pred.copy()

        u_new, period = _halo_correction_with_arc(mu, u_guess, u_pred, tangent)
        state0 = np.array([u_new[0], 0.0, u_new[1], 0.0, u_new[2], 0.0], dtype=float)

        orbits.append(HaloOrbit(state0=state0.copy(), period=period, z0=float(u_new[1]), vy0=float(u_new[2])))
        u_a = u_b
        u_b = u_new

    return orbits


def halo_from_planar_seed(
    mu: float,
    *,
    x0_planar: float,
    vy0_planar: float,
    z0_step: float = 0.01,
    tol: float = 1e-9,
) -> HaloOrbit:
    """Generate a first 3D halo orbit using a planar Lyapunov seed."""
    u_a = np.array([x0_planar, 0.0, vy0_planar], dtype=float)
    u_b = np.array([x0_planar, z0_step, vy0_planar], dtype=float)
    tangent = u_b - u_a
    tangent /= np.linalg.norm(tangent)
    u_pred = u_b.copy()
    u_new, period = _halo_correction_with_arc(mu, u_b.copy(), u_pred, tangent, tol=tol)
    state0 = np.array([u_new[0], 0.0, u_new[1], 0.0, u_new[2], 0.0], dtype=float)
    return HaloOrbit(state0=state0, period=period, z0=float(u_new[1]), vy0=float(u_new[2]))


def halo_from_planar(
    mu: float,
    *,
    x0: float,
    vy0_guess: float,
    z0_step: float = 0.01,
    tol: float = 1e-9,
) -> HaloOrbit:
    """Find a planar Lyapunov orbit, then lift to a 3D halo orbit."""
    planar = find_planar_lyapunov(mu, x0=x0, vy0_guess=vy0_guess)
    state0 = planar["state0"]
    return halo_from_planar_seed(
        mu,
        x0_planar=float(state0[0]),
        vy0_planar=float(state0[4]),
        z0_step=z0_step,
        tol=tol,
    )
