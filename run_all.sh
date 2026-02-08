#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

python3 scripts/phase1_validate.py
python3 scripts/phase2_halo_orbit.py
python3 scripts/phase3_manifolds.py
UTC_EPOCH="${UTC_EPOCH:-2026-02-07T00:00:00Z}" python3 scripts/phase4_poincare.py
python3 scripts/plot_3d_orbits.py
python3 scripts/plot_reentry_globe.py
python3 scripts/plot_halo_family.py

echo "All phases complete. Outputs in outputs/."
