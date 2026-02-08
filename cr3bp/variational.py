"""Variational equations and STM integration for CR3BP."""

from __future__ import annotations

import numpy as np

from .eom import cr3bp_eom, _r1_r2


def variational_eom(t: float, y: np.ndarray, mu: float) -> np.ndarray:
    """Augmented CR3BP dynamics with variational equations.

    y = [state(6), Phi(36)] with Phi flattened row-major.
    """
    state = y[:6]
    x, yv, z = state[0], state[1], state[2]
    r1, r2 = _r1_r2(state, mu)

    r1_2 = r1 * r1
    r2_2 = r2 * r2
    r1_5 = r1_2 * r1_2 * r1
    r2_5 = r2_2 * r2_2 * r2

    mu1 = 1 - mu
    mu2 = mu

    Uxx = (
        1
        - mu1 * (1 / r1**3 - 3 * (x + mu) ** 2 / r1_5)
        - mu2 * (1 / r2**3 - 3 * (x - 1 + mu) ** 2 / r2_5)
    )
    Uyy = (
        1
        - mu1 * (1 / r1**3 - 3 * yv * yv / r1_5)
        - mu2 * (1 / r2**3 - 3 * yv * yv / r2_5)
    )
    Uzz = (
        -mu1 * (1 / r1**3 - 3 * z * z / r1_5)
        - mu2 * (1 / r2**3 - 3 * z * z / r2_5)
    )
    Uxy = 3 * mu1 * (x + mu) * yv / r1_5 + 3 * mu2 * (x - 1 + mu) * yv / r2_5
    Uxz = 3 * mu1 * (x + mu) * z / r1_5 + 3 * mu2 * (x - 1 + mu) * z / r2_5
    Uyz = 3 * mu1 * yv * z / r1_5 + 3 * mu2 * yv * z / r2_5

    A = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [Uxx, Uxy, Uxz, 0.0, 2.0, 0.0],
            [Uxy, Uyy, Uyz, -2.0, 0.0, 0.0],
            [Uxz, Uyz, Uzz, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    phi = y[6:].reshape((6, 6))
    phi_dot = A @ phi

    dy = np.zeros_like(y)
    dy[:6] = cr3bp_eom(t, state, mu)
    dy[6:] = phi_dot.reshape(-1)
    return dy
