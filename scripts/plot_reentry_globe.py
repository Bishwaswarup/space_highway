"""Plot feasible re-entry points on a Plotly globe."""

from __future__ import annotations

import os

import numpy as np


def main() -> None:
    data_path = os.path.join("outputs", "phase4_poincare.npz")
    if not os.path.exists(data_path):
        raise SystemExit("Missing outputs/phase4_poincare.npz. Run Phase 4 first.")

    data = np.load(data_path)
    feasible = data["feasible"]
    if feasible.size == 0:
        raise SystemExit("No feasible points in phase4_poincare.npz.")

    lat_rad = feasible[:, 8]
    lon_rad = feasible[:, 9]
    lat_ecef = feasible[:, 10]
    lon_ecef = feasible[:, 11]
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lat=lat_deg,
            lon=lon_deg,
            mode="markers",
            marker=dict(size=6, color="crimson"),
            name="Inertial lat/lon",
        )
    )
    if np.isfinite(lat_ecef).all():
        fig.add_trace(
            go.Scattergeo(
                lat=np.degrees(lat_ecef),
                lon=np.degrees(lon_ecef),
                mode="markers",
                marker=dict(size=6, color="royalblue"),
                name="Earth-fixed lat/lon",
            )
        )
    fig.update_layout(
        title="Feasible Re-entry Points",
        geo=dict(showland=True, landcolor="rgb(230, 230, 230)", projection_type="orthographic"),
    )

    out_path = os.path.join("outputs", "phase4_reentry_globe.html")
    os.makedirs("outputs", exist_ok=True)
    fig.write_html(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
