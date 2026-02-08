"""Plot a family of halo orbits (multiple orbits in one figure)."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from cr3bp.constants import EARTH_MOON_MU
from cr3bp.continuation import halo_pseudo_arclength
from cr3bp.manifolds import integrate_orbit
from cr3bp.plotting import plot_orbits_matplotlib


def main() -> None:
    mu = EARTH_MOON_MU
    data_path = os.path.join("outputs", "phase3_halo_family_pseudo.npz")
    if os.path.exists(data_path):
        data = np.load(data_path)
        state0s = data["state0s"]
        periods = data["periods"]
    else:
        orbits = halo_pseudo_arclength(
            mu,
            x0_a=0.836914,
            z0_a=0.02,
            vy0_a=0.23,
            x0_b=0.836914,
            z0_b=0.03,
            vy0_b=0.235,
            ds=0.01,
            n_steps=4,
        )
        state0s = np.vstack([o.state0 for o in orbits])
        periods = np.array([o.period for o in orbits])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for state0, period in zip(state0s, periods):
        ts, ys = integrate_orbit(state0, mu, float(period))
        ax.plot(ys[:, 0], ys[:, 1], ys[:, 2], lw=1)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Halo Orbit Family")
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "halo_family.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
