"""Generate a family of halo orbits using continuation."""

from __future__ import annotations

import os

import numpy as np

from cr3bp.constants import EARTH_MOON_MU
from cr3bp.continuation import halo_continuation


def main() -> None:
    mu = EARTH_MOON_MU
    x0 = 0.836914
    z0_start = 0.02
    vy0_start = 0.23
    z0_step = 0.005
    n_orbits = 8

    orbits = halo_continuation(
        mu,
        x0=x0,
        z0_start=z0_start,
        vy0_start=vy0_start,
        z0_step=z0_step,
        n_orbits=n_orbits,
    )

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "phase3_halo_family.npz")
    np.savez(
        out_path,
        state0s=np.vstack([o.state0 for o in orbits]),
        periods=np.array([o.period for o in orbits]),
        z0s=np.array([o.z0 for o in orbits]),
        vy0s=np.array([o.vy0 for o in orbits]),
    )

    print("Phase 3: Halo Continuation")
    print("-" * 60)
    print(f"Generated: {len(orbits)} orbits")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
