"""Plotting utilities for 3D orbits and manifold tubes."""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_orbits_matplotlib(
    orbit_states: np.ndarray,
    manifold_trajectories: Iterable[np.ndarray] | None = None,
    *,
    title: str = "Orbit and Manifold Tubes",
) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(orbit_states[:, 0], orbit_states[:, 1], orbit_states[:, 2], color="black", lw=2, label="Orbit")
    if manifold_trajectories is not None:
        for traj in manifold_trajectories:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="tab:blue", alpha=0.4, lw=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_orbits_plotly(
    orbit_states: np.ndarray,
    manifold_trajectories: Iterable[np.ndarray] | None = None,
    *,
    title: str = "Orbit and Manifold Tubes",
) -> "plotly.graph_objects.Figure":
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=orbit_states[:, 0],
            y=orbit_states[:, 1],
            z=orbit_states[:, 2],
            mode="lines",
            line=dict(color="black", width=4),
            name="Orbit",
        )
    )
    if manifold_trajectories is not None:
        for traj in manifold_trajectories:
            fig.add_trace(
                go.Scatter3d(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    z=traj[:, 2],
                    mode="lines",
                    line=dict(color="royalblue", width=2),
                    opacity=0.35,
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
        ),
    )
    return fig


def export_plotly(fig: "plotly.graph_objects.Figure", out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)
