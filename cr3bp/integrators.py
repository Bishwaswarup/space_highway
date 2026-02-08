"""Adaptive integrators for CR3BP integration."""

from __future__ import annotations

import numpy as np


def rk45_integrate(
    deriv,
    y0: np.ndarray,
    t0: float,
    tf: float,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    h0: float | None = None,
    h_min: float = 1e-12,
    h_max: float = 0.1,
    max_steps: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ODEs with Dormand-Prince RK45 adaptive step size.

    Returns times and states arrays.
    """
    y0 = np.array(y0, dtype=float)
    t = float(t0)
    tf = float(tf)
    direction = 1.0 if tf >= t else -1.0

    if h0 is None:
        h0 = 1e-3 * direction
    else:
        h0 = float(h0) * direction

    h = np.clip(h0, -abs(h_max), abs(h_max))
    ts = [t]
    ys = [y0.copy()]

    # Dormand-Prince coefficients
    c2 = 1 / 5
    c3 = 3 / 10
    c4 = 4 / 5
    c5 = 8 / 9
    c6 = 1.0
    c7 = 1.0

    a21 = 1 / 5
    a31 = 3 / 40
    a32 = 9 / 40
    a41 = 44 / 45
    a42 = -56 / 15
    a43 = 32 / 9
    a51 = 19372 / 6561
    a52 = -25360 / 2187
    a53 = 64448 / 6561
    a54 = -212 / 729
    a61 = 9017 / 3168
    a62 = -355 / 33
    a63 = 46732 / 5247
    a64 = 49 / 176
    a65 = -5103 / 18656
    a71 = 35 / 384
    a72 = 0.0
    a73 = 500 / 1113
    a74 = 125 / 192
    a75 = -2187 / 6784
    a76 = 11 / 84

    # 5th-order solution weights (b) and 4th-order (b_hat)
    b1 = 35 / 384
    b2 = 0.0
    b3 = 500 / 1113
    b4 = 125 / 192
    b5 = -2187 / 6784
    b6 = 11 / 84
    b7 = 0.0

    b1_hat = 5179 / 57600
    b2_hat = 0.0
    b3_hat = 7571 / 16695
    b4_hat = 393 / 640
    b5_hat = -92097 / 339200
    b6_hat = 187 / 2100
    b7_hat = 1 / 40

    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0

    for _ in range(max_steps):
        if direction * (t - tf) >= 0:
            break

        if direction * (t + h - tf) > 0:
            h = tf - t

        k1 = deriv(t, y0)
        k2 = deriv(t + c2 * h, y0 + h * (a21 * k1))
        k3 = deriv(t + c3 * h, y0 + h * (a31 * k1 + a32 * k2))
        k4 = deriv(t + c4 * h, y0 + h * (a41 * k1 + a42 * k2 + a43 * k3))
        k5 = deriv(
            t + c5 * h,
            y0 + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4),
        )
        k6 = deriv(
            t + c6 * h,
            y0 + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5),
        )
        y5 = y0 + h * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        k7 = deriv(t + c7 * h, y5)

        y4 = y0 + h * (
            b1_hat * k1
            + b2_hat * k2
            + b3_hat * k3
            + b4_hat * k4
            + b5_hat * k5
            + b6_hat * k6
            + b7_hat * k7
        )

        err = np.linalg.norm(y5 - y4, ord=np.inf)
        scale = atol + rtol * max(np.linalg.norm(y0, ord=np.inf), np.linalg.norm(y5, ord=np.inf))
        err_norm = err / scale if scale > 0 else err

        if err_norm <= 1.0:
            t = t + h
            y0 = y5
            ts.append(t)
            ys.append(y0.copy())

        if err_norm == 0:
            factor = max_factor
        else:
            factor = safety * err_norm ** (-0.2)
            factor = np.clip(factor, min_factor, max_factor)

        h *= factor
        if abs(h) < h_min:
            h = h_min * direction
        if abs(h) > h_max:
            h = h_max * direction

    return np.array(ts), np.vstack(ys)
