# Mapping Unstable Invariant Manifolds in the Earth-Moon System

This repository implements a thesis-grade numerical pipeline for CR3BP halo orbits, invariant manifolds, and re-entry mapping.

## Quick Start
1. Create a virtual environment and install dependencies:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install numpy matplotlib plotly`

2. Run validation (Phase 1):
   - `PYTHONPATH=. python3 scripts/phase1_validate.py`

3. Solve a 3D halo orbit (Phase 2):
   - `PYTHONPATH=. python3 scripts/phase2_halo_orbit.py`

4. Generate manifolds (Phase 3):
   - `PYTHONPATH=. python3 scripts/phase3_manifolds.py`

5. Compute Poincaré section (Phase 4):
   - `UTC_EPOCH="2026-02-07T00:00:00Z" PYTHONPATH=. python3 scripts/phase4_poincare.py`

6. Plot 3D orbits + manifolds:
   - `PYTHONPATH=. python3 scripts/plot_3d_orbits.py`

7. Plot re-entry globe:
   - `PYTHONPATH=. python3 scripts/plot_reentry_globe.py`

8. Plot halo family:
   - `PYTHONPATH=. python3 scripts/plot_halo_family.py`

## Reproducible Pipeline
Run these in order:
1. `PYTHONPATH=. python3 scripts/phase1_validate.py`
2. `PYTHONPATH=. python3 scripts/phase2_halo_orbit.py`
3. `PYTHONPATH=. python3 scripts/phase3_manifolds.py`
4. `UTC_EPOCH="2026-02-07T00:00:00Z" PYTHONPATH=. python3 scripts/phase4_poincare.py`
5. `PYTHONPATH=. python3 scripts/plot_3d_orbits.py`
6. `PYTHONPATH=. python3 scripts/plot_reentry_globe.py`
7. `PYTHONPATH=. python3 scripts/plot_halo_family.py`

Outputs are written to `outputs/`.

## Structure
- `cr3bp/`: core library (EOMs, integrator, shooting, manifolds, continuation)
- `scripts/`: reproducible runs for each phase
- `notebooks/`: exploratory and results notebooks
- `phase_*.md`: phase summaries
- `report/`: LaTeX thesis scaffold
- `notebooks/results.ipynb`: polished results notebook

## Report
- `report/main.tex` is a LaTeX scaffold including the Error Budget section.
  - Build with `pdflatex main.tex` from `report/` if LaTeX is installed.
