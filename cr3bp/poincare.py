"""Poincaré section utilities for Earth re-entry mapping."""

from __future__ import annotations

import numpy as np


def _earth_center(mu: float) -> np.ndarray:
    return np.array([-mu, 0.0, 0.0], dtype=float)


def _flight_path_angle(state: np.ndarray, mu: float) -> float:
    r_vec = state[:3] - _earth_center(mu)
    v_vec = state[3:]
    r_hat = r_vec / np.linalg.norm(r_vec)
    vr = np.dot(v_vec, r_hat)
    v = np.linalg.norm(v_vec)
    return float(np.arcsin(vr / v))


def _rot_to_inertial(state: np.ndarray, t: float) -> np.ndarray:
    c = np.cos(t)
    s = np.sin(t)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    r_rot = state[:3]
    v_rot = state[3:]
    omega = np.array([0.0, 0.0, 1.0])
    r_inertial = rot @ r_rot
    v_inertial = rot @ (v_rot + np.cross(omega, r_rot))
    return np.hstack([r_inertial, v_inertial])


def _lat_lon_from_r(r_vec: np.ndarray) -> tuple[float, float]:
    r = np.linalg.norm(r_vec)
    lat = np.arcsin(r_vec[2] / r)
    lon = np.arctan2(r_vec[1], r_vec[0])
    return float(lat), float(lon)


def _utc_to_julian_date(utc_iso: str) -> float:
    date_str, time_str = utc_iso.split("T")
    year, month, day = [int(x) for x in date_str.split("-")]
    if time_str.endswith("Z"):
        time_str = time_str[:-1]
    hh, mm, ss = time_str.split(":")
    hour = int(hh)
    minute = int(mm)
    second = float(ss)

    if month <= 2:
        year -= 1
        month += 12

    a = year // 100
    b = 2 - a + a // 4
    jd_day = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
    frac_day = (hour + minute / 60.0 + second / 3600.0) / 24.0
    return jd_day + frac_day


def _gmst_from_jd(jd: float) -> float:
    t = (jd - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - (t * t * t) / 38710000.0
    )
    gmst_rad = np.deg2rad(gmst_deg % 360.0)
    return float(gmst_rad)


def _eci_to_ecef(r_eci: np.ndarray, gmst_rad: float) -> np.ndarray:
    c = np.cos(gmst_rad)
    s = np.sin(gmst_rad)
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot @ r_eci


def first_sphere_crossing(
    trajectory: np.ndarray,
    mu: float,
    radius_nd: float,
    *,
    times: np.ndarray | None = None,
    utc_epoch: str | None = None,
) -> dict[str, float] | None:
    """Find first crossing of a spherical surface about Earth."""
    c = _earth_center(mu)
    r_vals = np.linalg.norm(trajectory[:, :3] - c, axis=1)
    for i in range(1, len(r_vals)):
        if r_vals[i - 1] > radius_nd and r_vals[i] <= radius_nd:
            r0, r1 = r_vals[i - 1], r_vals[i]
            alpha = (r0 - radius_nd) / (r0 - r1) if r0 != r1 else 0.0
            state = trajectory[i - 1] + alpha * (trajectory[i] - trajectory[i - 1])
            if times is not None:
                t0, t1 = times[i - 1], times[i]
                t_cross = t0 + alpha * (t1 - t0)
                state_inertial = _rot_to_inertial(state, t_cross)
                lat, lon = _lat_lon_from_r(state_inertial[:3] - _earth_center(mu))
                if utc_epoch is not None:
                    jd0 = _utc_to_julian_date(utc_epoch)
                    gmst = _gmst_from_jd(jd0 + t_cross / (2 * np.pi))
                    r_ecef = _eci_to_ecef(state_inertial[:3], gmst)
                    lat_ecef, lon_ecef = _lat_lon_from_r(r_ecef - _earth_center(mu))
                else:
                    lat_ecef, lon_ecef = float("nan"), float("nan")
            else:
                t_cross = float("nan")
                state_inertial = state
                lat, lon = _lat_lon_from_r(state[:3] - _earth_center(mu))
                lat_ecef, lon_ecef = float("nan"), float("nan")

            fpa = _flight_path_angle(state, mu)
            return {
                "x": float(state[0]),
                "y": float(state[1]),
                "z": float(state[2]),
                "vx": float(state[3]),
                "vy": float(state[4]),
                "vz": float(state[5]),
                "fpa_rad": fpa,
                "t": float(t_cross),
                "lat_rad": lat,
                "lon_rad": lon,
                "lat_ecef_rad": lat_ecef,
                "lon_ecef_rad": lon_ecef,
                "x_i": float(state_inertial[0]),
                "y_i": float(state_inertial[1]),
                "z_i": float(state_inertial[2]),
            }
    return None


def poincare_section(
    manifold_trajectories: np.ndarray,
    mu: float,
    radius_nd: float,
    *,
    manifold_times: np.ndarray | None = None,
    utc_epoch: str | None = None,
    fpa_min_deg: float = -7.0,
    fpa_max_deg: float = -5.0,
) -> dict[str, np.ndarray]:
    """Collect Earth sphere crossings and filter by flight path angle."""
    hits = []
    feasible = []
    for idx, traj in enumerate(manifold_trajectories):
        times = None
        if manifold_times is not None and idx < len(manifold_times):
            times = manifold_times[idx]
        hit = first_sphere_crossing(traj, mu, radius_nd, times=times, utc_epoch=utc_epoch)
        if hit is None:
            continue
        hits.append(hit)
        fpa_deg = np.degrees(hit["fpa_rad"])
        if fpa_min_deg <= fpa_deg <= fpa_max_deg:
            feasible.append(hit)

    def to_array(entries: list[dict[str, float]]) -> np.ndarray:
        if not entries:
            return np.zeros((0, 15))
        return np.array(
            [
                [
                    e["x"],
                    e["y"],
                    e["z"],
                    e["vx"],
                    e["vy"],
                    e["vz"],
                    e["fpa_rad"],
                    e["t"],
                    e["lat_rad"],
                    e["lon_rad"],
                    e["lat_ecef_rad"],
                    e["lon_ecef_rad"],
                    e["x_i"],
                    e["y_i"],
                    e["z_i"],
                ]
                for e in entries
            ],
            dtype=float,
        )

    return {
        "hits": to_array(hits),
        "feasible": to_array(feasible),
    }
