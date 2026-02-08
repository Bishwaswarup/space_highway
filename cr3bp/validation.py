"""Validation utilities for Phase 1."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .eom import cr3bp_eom
from .integrators import rk45_integrate


def lagrange_points(mu: float) -> dict[str, np.ndarray]:
    """Compute L1-L5 points in the rotating frame."""

    def f(x: float) -> float:
        r1 = abs(x + mu)
        r2 = abs(x - 1 + mu)
        return x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3

    def df(x: float) -> float:
        r1 = abs(x + mu)
        r2 = abs(x - 1 + mu)
        term1 = (1 - mu) * (1 / r1**3 - 3 * (x + mu) ** 2 / r1**5)
        term2 = mu * (1 / r2**3 - 3 * (x - 1 + mu) ** 2 / r2**5)
        return 1 - term1 - term2

    def solve(x0: float) -> float:
        x = x0
        for _ in range(100):
            fx = f(x)
            dfx = df(x)
            if dfx == 0:
                break
            dx = -fx / dfx
            x += dx
            if abs(dx) < 1e-12:
                break
        return x

    l1 = solve(0.7 - mu)
    l2 = solve(1.2 - mu)
    l3 = solve(-1.0 - mu)
    l4 = np.array([0.5 - mu, math.sqrt(3) / 2, 0.0])
    l5 = np.array([0.5 - mu, -math.sqrt(3) / 2, 0.0])

    return {
        "L1": np.array([l1, 0.0, 0.0]),
        "L2": np.array([l2, 0.0, 0.0]),
        "L3": np.array([l3, 0.0, 0.0]),
        "L4": l4,
        "L5": l5,
    }


def lagrangian_check(mu: float) -> dict[str, float]:
    """Return acceleration norms at L1-L5 (should be ~0)."""
    points = lagrange_points(mu)
    acc_norms = {}
    for name, pos in points.items():
        state = np.hstack([pos, np.zeros(3)])
        acc = cr3bp_eom(0.0, state, mu)[3:]
        acc_norms[name] = float(np.linalg.norm(acc))
    return acc_norms


def zero_mass_energy_check(
    *,
    r0: float = 1.2,
    tf: float = 20.0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> dict[str, float]:
    """Check inertial energy conservation for mu=0."""
    mu = 0.0
    r0_vec = np.array([r0, 0.0, 0.0])
    v_circ = math.sqrt(1.0 / r0)
    v0_inertial = np.array([0.0, v_circ, 0.0])
    omega = np.array([0.0, 0.0, 1.0])
    v0_rot = v0_inertial - np.cross(omega, r0_vec)

    state0 = np.hstack([r0_vec, v0_rot])

    def deriv(t: float, state: np.ndarray) -> np.ndarray:
        return cr3bp_eom(t, state, mu)

    ts, ys = rk45_integrate(deriv, state0, 0.0, tf, rtol=rtol, atol=atol)

    energies = []
    for t, s in zip(ts, ys):
        x, y, z, vx, vy, vz = s
        rot = np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])
        r_inertial = rot @ np.array([x, y, z])
        v_rot = np.array([vx, vy, vz])
        v_inertial = rot @ (v_rot + np.cross(omega, np.array([x, y, z])))
        r = np.linalg.norm(r_inertial)
        v2 = np.dot(v_inertial, v_inertial)
        energy = 0.5 * v2 - 1.0 / r
        energies.append(energy)

    energies = np.array(energies)
    drift = float(np.max(np.abs(energies - energies[0])))
    return {
        "energy_initial": float(energies[0]),
        "energy_max_drift": drift,
        "num_steps": int(len(ts)),
    }
