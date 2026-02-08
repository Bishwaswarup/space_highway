"""Core CR3BP tools for Phase 1."""

from .constants import EARTH_MOON_DISTANCE_KM, EARTH_MOON_MU, EARTH_RADIUS_KM
from .continuation import halo_continuation, halo_from_planar, halo_from_planar_seed, halo_pseudo_arclength
from .plotting import export_plotly, plot_orbits_matplotlib, plot_orbits_plotly
from .poincare import poincare_section
from .eom import cr3bp_eom, jacobi_constant
from .integrators import rk45_integrate
from .manifolds import integrate_orbit, sample_manifold, stable_eigenpair, unstable_eigenpair
from .shooting import find_halo_orbit, find_planar_lyapunov, monodromy_matrix
from .variational import variational_eom
from .validation import (
    lagrange_points,
    lagrangian_check,
    zero_mass_energy_check,
)

__all__ = [
    "EARTH_MOON_MU",
    "EARTH_MOON_DISTANCE_KM",
    "EARTH_RADIUS_KM",
    "halo_continuation",
    "halo_from_planar",
    "halo_from_planar_seed",
    "halo_pseudo_arclength",
    "poincare_section",
    "plot_orbits_matplotlib",
    "plot_orbits_plotly",
    "export_plotly",
    "cr3bp_eom",
    "jacobi_constant",
    "rk45_integrate",
    "find_planar_lyapunov",
    "find_halo_orbit",
    "monodromy_matrix",
    "variational_eom",
    "integrate_orbit",
    "sample_manifold",
    "unstable_eigenpair",
    "stable_eigenpair",
    "lagrange_points",
    "lagrangian_check",
    "zero_mass_energy_check",
]
