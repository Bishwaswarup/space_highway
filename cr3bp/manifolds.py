"""Manifold generation tools for CR3BP periodic orbits."""

from __future__ import annotations

import numpy as np

from .integrators import rk45_integrate
from .variational import variational_eom


def _interp_state(t_query: float, ts: np.ndarray, ys: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(ts, t_query)
    if idx <= 0:
        return ys[0].copy()
    if idx >= len(ts):
        return ys[-1].copy()
    t0, t1 = ts[idx - 1], ts[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    alpha = (t_query - t0) / (t1 - t0)
    return y0 + alpha * (y1 - y0)


def integrate_orbit(
    state0: np.ndarray,
    mu: float,
    period: float,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a periodic orbit for one period."""
    def deriv(t: float, y: np.ndarray) -> np.ndarray:
        from .eom import cr3bp_eom

        return cr3bp_eom(t, y, mu)

    return rk45_integrate(deriv, state0, 0.0, period, rtol=rtol, atol=atol)


def integrate_orbit_with_stm(
    state0: np.ndarray,
    mu: float,
    period: float,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate orbit and STM for one period."""
    phi0 = np.eye(6).reshape(-1)
    y0 = np.hstack([state0, phi0])

    def deriv(t: float, y: np.ndarray) -> np.ndarray:
        return variational_eom(t, y, mu)

    return rk45_integrate(deriv, y0, 0.0, period, rtol=rtol, atol=atol)


def unstable_eigenpair(monodromy: np.ndarray) -> tuple[np.ndarray, complex]:
    eigvals, eigvecs = np.linalg.eig(monodromy)
    idx = np.argmax(np.abs(eigvals))
    return eigvecs[:, idx], eigvals[idx]


def stable_eigenpair(monodromy: np.ndarray) -> tuple[np.ndarray, complex]:
    eigvals, eigvecs = np.linalg.eig(monodromy)
    idx = np.argmin(np.abs(eigvals))
    return eigvecs[:, idx], eigvals[idx]


def sample_manifold(
    state0: np.ndarray,
    mu: float,
    period: float,
    *,
    n_samples: int = 50,
    eps: float = 1e-6,
    manifold_time: float = 5.0,
    direction: str = "unstable",
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> dict[str, np.ndarray]:
    """Generate unstable manifold trajectories from a periodic orbit.

    Returns a dict with orbit samples and manifold trajectories.
    """
    ts_aug, ys_aug = integrate_orbit_with_stm(state0, mu, period, rtol=rtol, atol=atol)

    monodromy = ys_aug[-1][6:].reshape((6, 6))
    if direction == "unstable":
        v0, lam = unstable_eigenpair(monodromy)
        time_sign = 1.0
    elif direction == "stable":
        v0, lam = stable_eigenpair(monodromy)
        time_sign = -1.0
    else:
        raise ValueError("direction must be 'unstable' or 'stable'")
    v0_real = np.real(v0)

    sample_ts = np.linspace(0.0, period, n_samples, endpoint=False)
    orbit_states = []
    manifold_trajs = []
    manifold_times = []

    for t in sample_ts:
        y_aug_t = _interp_state(t, ts_aug, ys_aug)
        state_t = y_aug_t[:6]
        phi_t = y_aug_t[6:].reshape((6, 6))

        v_t = phi_t @ v0_real
        v_t = v_t / np.linalg.norm(v_t)
        perturbed_state = state_t + eps * v_t

        def deriv(tt: float, y: np.ndarray) -> np.ndarray:
            from .eom import cr3bp_eom

            return cr3bp_eom(tt, y, mu)

        t_start = t
        t_end = t + time_sign * manifold_time
        t_man, y_man = rk45_integrate(
            deriv,
            perturbed_state,
            t_start,
            t_end,
            rtol=rtol,
            atol=atol,
        )

        orbit_states.append(state_t)
        manifold_trajs.append(y_man)
        manifold_times.append(t_man)

    return {
        "orbit_times": sample_ts,
        "orbit_states": np.vstack(orbit_states),
        "manifold_trajectories": np.array(manifold_trajs, dtype=object),
        "manifold_times": np.array(manifold_times, dtype=object),
        "eigenvalue": np.array([lam]),
        "direction": np.array([direction]),
    }
