"""Equations of motion and invariants for the CR3BP."""

from __future__ import annotations

import numpy as np


def _r1_r2(state: np.ndarray, mu: float) -> tuple[float, float]:
    x, y, z = state[0], state[1], state[2]
    r1 = np.sqrt((x + mu) ** 2 + y * y + z * z)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y * y + z * z)
    return r1, r2


def cr3bp_eom(t: float, state: np.ndarray, mu: float) -> np.ndarray:
    """CR3BP equations of motion in the rotating frame.

    State is [x, y, z, vx, vy, vz] in nondimensional units.
    """
    x, y, z, vx, vy, vz = state
    r1, r2 = _r1_r2(state, mu)

    dU_dx = x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    dU_dy = y - (1 - mu) * y / r1**3 - mu * y / r2**3
    dU_dz = -(1 - mu) * z / r1**3 - mu * z / r2**3

    ax = 2 * vy + dU_dx
    ay = -2 * vx + dU_dy
    az = dU_dz

    return np.array([vx, vy, vz, ax, ay, az], dtype=float)


def jacobi_constant(state: np.ndarray, mu: float) -> float:
    """Compute the Jacobi constant for a CR3BP state."""
    x, y, z, vx, vy, vz = state
    r1, r2 = _r1_r2(state, mu)
    omega = 0.5 * (x * x + y * y) + (1 - mu) / r1 + mu / r2
    v2 = vx * vx + vy * vy + vz * vz
    return 2 * omega - v2
