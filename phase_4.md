Phase 4 Summary
===============

What we implemented
- Stable manifold generation by reversing time along the stable eigenvector.
- Poincaré section mapping at an Earth-centered spherical interface.
- Filtering of trajectories by flight path angle for re-entry feasibility.
- Inertial-frame conversion at the interface to compute latitude/longitude.
- Optional Earth-fixed conversion using a UTC epoch (GMST-based rotation).

Key files
- `cr3bp/manifolds.py`
- `cr3bp/poincare.py`
- `scripts/phase4_poincare.py`

How to run
`PYTHONPATH=. python3 scripts/phase4_poincare.py`

Outputs
- `outputs/phase4_poincare.npz`
  - `hits` (all sphere crossings)
  - `feasible` (filtered by flight path angle)
  - Columns now include inertial coordinates and lat/lon.
  - Earth-fixed lat/lon available when `UTC_EPOCH` is set.
