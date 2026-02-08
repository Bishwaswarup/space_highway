Phase 1 Summary
===============

What we implemented
- Core CR3BP equations of motion in the rotating frame (nondimensionalized).
- Jacobi constant computation for energy-like invariant tracking.
- Custom adaptive RK45 (Dormand-Prince) integrator with step-size control.
- Validation suite:
  - Lagrange point acceleration norms (L1–L5).
  - Zero-mass (mu=0) inertial energy conservation test.

Key files
- `cr3bp/eom.py`
- `cr3bp/integrators.py`
- `cr3bp/validation.py`
- `scripts/phase1_validate.py`

Validation results (example run)
- L1–L5 acceleration norms ~1e-16 (numerical zero).
- Zero-mass test max inertial energy drift ~4e-12.

Notes
- The integrator adapts step size to handle varying dynamics near primaries.
- Tolerances are adjustable for tighter error control.
