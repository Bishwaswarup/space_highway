"""Compute a Poincare section for Earth re-entry mapping."""

from __future__ import annotations

import os

import numpy as np

from cr3bp.constants import EARTH_MOON_DISTANCE_KM, EARTH_MOON_MU, EARTH_RADIUS_KM
from cr3bp.manifolds import sample_manifold
from cr3bp.poincare import poincare_section
from cr3bp.shooting import find_halo_orbit


def main() -> None:
    mu = EARTH_MOON_MU
    altitude_km = 120.0
    radius_nd = (EARTH_RADIUS_KM + altitude_km) / EARTH_MOON_DISTANCE_KM
    utc_epoch = os.environ.get("UTC_EPOCH")

    manifold_path = os.path.join("outputs", "phase3_manifolds.npz")
    if os.path.exists(manifold_path):
        data = np.load(manifold_path, allow_pickle=True)
        manifold_trajs = data["manifold_trajectories"]
        manifold_times = data["manifold_times"] if "manifold_times" in data else None
    else:
        halo = find_halo_orbit(mu, x0=0.836914, z0_guess=0.05, vy0_guess=0.25)
        state0 = halo["state0"]
        period = float(halo["period"])
        data = sample_manifold(state0, mu, period, n_samples=60, eps=1e-6, manifold_time=5.0)
        manifold_trajs = data["manifold_trajectories"]
        manifold_times = data["manifold_times"] if "manifold_times" in data else None

    section = poincare_section(
        manifold_trajs,
        mu,
        radius_nd,
        manifold_times=manifold_times,
        utc_epoch=utc_epoch,
    )

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "phase4_poincare.npz")
    np.savez(out_path, hits=section["hits"], feasible=section["feasible"])

    print("Phase 4: Poincare Section")
    print("-" * 60)
    print(f"Hits: {len(section['hits'])}")
    print(f"Feasible: {len(section['feasible'])}")
    if utc_epoch:
        print(f"UTC epoch: {utc_epoch}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
