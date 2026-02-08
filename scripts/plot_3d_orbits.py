"""Plot 3D halo orbits and manifold tubes from saved outputs."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from cr3bp.plotting import export_plotly, plot_orbits_matplotlib, plot_orbits_plotly


def main() -> None:
    data_path = os.path.join("outputs", "phase3_manifolds.npz")
    if not os.path.exists(data_path):
        raise SystemExit("Missing outputs/phase3_manifolds.npz. Run Phase 3 first.")

    data = np.load(data_path, allow_pickle=True)
    orbit_states = data["orbit_states"]
    manifold_trajs = data["manifold_trajectories"]

    fig = plot_orbits_matplotlib(orbit_states, manifold_trajs)
    os.makedirs("outputs", exist_ok=True)
    fig_path = os.path.join("outputs", "phase3_manifolds.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    fig_plotly = plot_orbits_plotly(orbit_states, manifold_trajs)
    html_path = os.path.join("outputs", "phase3_manifolds.html")
    export_plotly(fig_plotly, html_path)

    print("Saved plots:")
    print(f"  {fig_path}")
    print(f"  {html_path}")


if __name__ == "__main__":
    main()
