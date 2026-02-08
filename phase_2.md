Phase 2 Summary
===============

What we implemented
- Variational equations for the CR3BP to propagate the State Transition Matrix (STM).
- Differential-correction shooting methods for:
  - Planar Lyapunov orbits (baseline).
  - Full 3D halo orbits using symmetry and a 2-parameter correction.
- Monodromy matrix computation over one full period.
- Eigenvalue extraction for stability characterization.

Key files
- `cr3bp/variational.py`
- `cr3bp/shooting.py`
- `scripts/phase2_planar_lyapunov.py`
- `scripts/phase2_halo_orbit.py`

How to run
`PYTHONPATH=. python3 scripts/phase2_planar_lyapunov.py`
`PYTHONPATH=. python3 scripts/phase2_halo_orbit.py`

Notes
- The halo solver corrects `vy0` and `z0` to enforce `vx=0` and `vz=0` at the y=0 crossing.
- If convergence is slow, tighten/loosen tolerances or adjust the initial guesses.
