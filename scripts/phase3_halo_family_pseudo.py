"""Generate a halo family using pseudo-arclength continuation."""

from __future__ import annotations

import os

import numpy as np

from cr3bp.constants import EARTH_MOON_MU
from cr3bp.continuation import halo_pseudo_arclength


def main() -> None:
    mu = EARTH_MOON_MU
    x0_a, z0_a, vy0_a = 0.836914, 0.02, 0.23
    x0_b, z0_b, vy0_b = 0.836914, 0.03, 0.235
    ds = 0.01
    n_steps = 6

    orbits = halo_pseudo_arclength(
        mu,
        x0_a=x0_a,
        z0_a=z0_a,
        vy0_a=vy0_a,
        x0_b=x0_b,
        z0_b=z0_b,
        vy0_b=vy0_b,
        ds=ds,
        n_steps=n_steps,
    )

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "phase3_halo_family_pseudo.npz")
    np.savez(
        out_path,
        state0s=np.vstack([o.state0 for o in orbits]),
        periods=np.array([o.period for o in orbits]),
        z0s=np.array([o.z0 for o in orbits]),
        vy0s=np.array([o.vy0 for o in orbits]),
    )

    print("Phase 3: Halo Continuation (Pseudo-Arclength)")
    print("-" * 60)
    print(f"Generated: {len(orbits)} orbits")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
