"""Compute a 3D halo orbit using differential correction."""

from __future__ import annotations

import numpy as np

from cr3bp.constants import EARTH_MOON_MU
from cr3bp.continuation import halo_from_planar
from cr3bp.validation import lagrange_points
from cr3bp.shooting import find_halo_orbit, monodromy_matrix


def main() -> None:
    mu = EARTH_MOON_MU
    l1 = lagrange_points(mu)["L1"][0]
    x0 = float(l1 - 0.01)
    z0_guesses = [0.01, 0.02, 0.03, 0.05]
    vy0_guesses = [0.15, 0.2, 0.23, 0.25]

    result = None
    for z0_guess in z0_guesses:
        for vy0_guess in vy0_guesses:
            try:
                result = find_halo_orbit(
                    mu,
                    x0=x0,
                    z0_guess=z0_guess,
                    vy0_guess=vy0_guess,
                    tol=1e-9,
                    max_iter=40,
                )
                break
            except RuntimeError:
                continue
        if result is not None:
            break

    if result is None:
        for z0_step in (0.01, 0.005, 0.002, 0.001):
            try:
                halo = halo_from_planar(mu, x0=x0, vy0_guess=0.2, z0_step=z0_step, tol=1e-9)
                result = {
                    "state0": halo.state0,
                    "period": halo.period,
                }
                break
            except RuntimeError:
                continue

    if result is None:
        raise RuntimeError("Halo solver failed for all guesses and planar lift attempts.")
    state0 = result["state0"]
    period = float(result["period"])

    mono = monodromy_matrix(mu, state0, period)
    eigvals = list(np.linalg.eigvals(mono))

    print("Phase 2: 3D Halo Orbit (L1) Summary")
    print("-" * 60)
    print(f"Initial state: {state0}")
    print(f"Period: {period:.6f}")
    print("Monodromy eigenvalues:")
    for ev in eigvals:
        print(f"  {ev:.6e}")


if __name__ == "__main__":
    main()
