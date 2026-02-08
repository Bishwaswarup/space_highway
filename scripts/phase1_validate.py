"""Run Phase 1 validation checks."""

from __future__ import annotations

from cr3bp.constants import EARTH_MOON_MU
from cr3bp.validation import lagrangian_check, zero_mass_energy_check


def main() -> None:
    print("Phase 1 Validation")
    print("-" * 60)

    lagrange_acc = lagrangian_check(EARTH_MOON_MU)
    print("Lagrangian Check (accel norms):")
    for name, acc in lagrange_acc.items():
        print(f"  {name}: {acc:.3e}")

    print()
    zero_mass = zero_mass_energy_check()
    print("Zero-Mass Energy Check (mu=0):")
    print(f"  Initial Energy: {zero_mass['energy_initial']:.6e}")
    print(f"  Max Energy Drift: {zero_mass['energy_max_drift']:.3e}")
    print(f"  Steps: {zero_mass['num_steps']}")


if __name__ == "__main__":
    main()
