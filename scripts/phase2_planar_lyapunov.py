"""Compute a planar Lyapunov orbit using differential correction."""

from __future__ import annotations

import numpy as np

from cr3bp.constants import EARTH_MOON_MU
from cr3bp.shooting import find_planar_lyapunov, monodromy_matrix


def main() -> None:
    mu = EARTH_MOON_MU
    x0 = 0.836914  # near L1 on the Earth side
    vy0_guess = 0.25

    result = find_planar_lyapunov(mu, x0=x0, vy0_guess=vy0_guess)
    state0 = result["state0"]
    period = float(result["period"])

    mono = monodromy_matrix(mu, state0, period)
    eigvals = np.linalg.eigvals(mono)

    print("Phase 2: Planar Lyapunov (L1) Summary")
    print("-" * 60)
    print(f"Initial state: {state0}")
    print(f"Period: {period:.6f}")
    print("Monodromy eigenvalues:")
    for ev in eigvals:
        print(f"  {ev:.6e}")


if __name__ == "__main__":
    main()
