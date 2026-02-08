"""Generate unstable manifolds from a halo orbit."""

from __future__ import annotations

import os

import numpy as np

from cr3bp.constants import EARTH_MOON_MU
from cr3bp.manifolds import sample_manifold
from cr3bp.shooting import find_halo_orbit


def main() -> None:
    mu = EARTH_MOON_MU
    x0 = 0.836914
    z0_guess = 0.05
    vy0_guess = 0.25

    halo = find_halo_orbit(mu, x0=x0, z0_guess=z0_guess, vy0_guess=vy0_guess)
    state0 = halo["state0"]
    period = float(halo["period"])

    data = sample_manifold(
        state0,
        mu,
        period,
        n_samples=60,
        eps=1e-6,
        manifold_time=5.0,
        direction="unstable",
    )

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "phase3_manifolds.npz")
    np.savez(
        out_path,
        orbit_times=data["orbit_times"],
        orbit_states=data["orbit_states"],
        eigenvalue=data["eigenvalue"],
        direction=data["direction"],
        manifold_times=data["manifold_times"],
        manifold_trajectories=data["manifold_trajectories"],
    )

    print("Phase 3: Manifold Generation")
    print("-" * 60)
    print(f"Halo state0: {state0}")
    print(f"Period: {period:.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
