"""Minimal executable SIR/SEIR models with a finite exponential comparison.

The finite exponential fit is intentionally not presented as a standalone
Dirichlet-series solver. It is a least-squares approximation of the RK4
trajectory, useful for numerical comparison and model sanity checks.
"""

from __future__ import annotations

import numbers

import numpy as np


def _validate_times(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1D array")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must contain only finite values")
    if np.any(np.diff(times) < 0):
        raise ValueError("times must be non-decreasing")
    return times


def _validate_positive_scalar(value: float, name: str, *, allow_zero: bool = False) -> float:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    elif value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def sir_rhs(t: float, y: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """Right-hand side of the SIR model. Here y is a population vector in proportions."""
    s, i, r = y
    return np.array(
        [-beta * s * i, beta * s * i - gamma * i, gamma * i],
        dtype=float,
    )


def seir_rhs(t: float, y: np.ndarray, beta: float, sigma: float, gamma: float) -> np.ndarray:
    """Right-hand side of the SEIR model. Here y is a population vector in proportions."""
    s, e, i, r = y
    return np.array(
        [-beta * s * i, beta * s * i - sigma * e, sigma * e - gamma * i, gamma * i],
        dtype=float,
    )


def rk4_step(rhs, t: float, y: np.ndarray, dt: float, **params) -> np.ndarray:
    """One RK4 step for a vector state.

    RK4 is a standard reference integrator, but it does not guarantee positivity for
    arbitrary step sizes; a large dt may produce a slightly negative component.
    """
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive step")
    y = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("state vector contains non-finite values")
    k1 = rhs(t, y, **params)
    k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1, **params)
    k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2, **params)
    k4 = rhs(t + dt, y + dt * k3, **params)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_sir(
    beta: float,
    gamma: float,
    s0: float,
    i0: float,
    r0: float,
    times: np.ndarray,
) -> np.ndarray:
    """Solve the SIR model on a given time grid using RK4.

    The variables represent proportions of the total population (not absolute
    counts unless explicitly rescaled later).
    """
    beta = _validate_positive_scalar(beta, "beta")
    gamma = _validate_positive_scalar(gamma, "gamma")
    s0 = _validate_positive_scalar(s0, "s0", allow_zero=True)
    i0 = _validate_positive_scalar(i0, "i0", allow_zero=True)
    r0 = _validate_positive_scalar(r0, "r0", allow_zero=True)
    times = _validate_times(times)

    y = np.empty((times.size, 3), dtype=float)
    y[0] = np.array([s0, i0, r0], dtype=float)
    for n in range(times.size - 1):
        h = times[n + 1] - times[n]
        y[n + 1] = rk4_step(sir_rhs, times[n], y[n], h, beta=beta, gamma=gamma)
    if np.any(~np.isfinite(y)):
        raise FloatingPointError("Non-finite values encountered during RK4 integration")
    if np.any(y < -1e-12):
        raise FloatingPointError("RK4 integration produced negative values; reduce the time step")
    return y


def solve_seir(
    beta: float,
    sigma: float,
    gamma: float,
    s0: float,
    e0: float,
    i0: float,
    r0: float,
    times: np.ndarray,
) -> np.ndarray:
    """Solve the SEIR model on a given time grid using RK4."""
    beta = _validate_positive_scalar(beta, "beta")
    sigma = _validate_positive_scalar(sigma, "sigma")
    gamma = _validate_positive_scalar(gamma, "gamma")
    s0 = _validate_positive_scalar(s0, "s0", allow_zero=True)
    e0 = _validate_positive_scalar(e0, "e0", allow_zero=True)
    i0 = _validate_positive_scalar(i0, "i0", allow_zero=True)
    r0 = _validate_positive_scalar(r0, "r0", allow_zero=True)
    times = _validate_times(times)

    y = np.empty((times.size, 4), dtype=float)
    y[0] = np.array([s0, e0, i0, r0], dtype=float)
    for n in range(times.size - 1):
        h = times[n + 1] - times[n]
        y[n + 1] = rk4_step(seir_rhs, times[n], y[n], h, beta=beta, sigma=sigma, gamma=gamma)
    if np.any(~np.isfinite(y)):
        raise FloatingPointError("Non-finite values encountered during RK4 integration")
    if np.any(y < -1e-12):
        raise FloatingPointError("RK4 integration produced negative values; reduce the time step")
    return y


def fit_exponential_series(
    times: np.ndarray,
    target: np.ndarray,
    n_terms: int = 8,
    decay: float = 0.25,
) -> np.ndarray:
    """Fit a finite exponential basis to a time-dependent target.

    This is a numerical approximation tool used to compare with an RK4 trajectory.
    It is not an independent Dirichlet-series solver and it should not be mistaken
    for a full analytical derivation of the report's method.
    """
    times = _validate_times(times)
    target = np.asarray(target, dtype=float)
    if target.ndim == 1:
        target = target[:, np.newaxis]
    if target.ndim != 2:
        raise ValueError("target must be a 1D or 2D array")
    if target.shape[0] != times.size:
        raise ValueError("target rows must match the number of time points")
    if not np.all(np.isfinite(target)):
        raise ValueError("target contains non-finite values")
    if not isinstance(n_terms, numbers.Integral) or n_terms <= 0:
        raise ValueError("n_terms must be a positive integer")
    decay = _validate_positive_scalar(decay, "decay")

    basis = np.exp(-np.outer(times, np.arange(n_terms, dtype=float)) * decay)
    coefficients, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)
    return basis @ coefficients


def dirichlet_approximation(
    times: np.ndarray,
    target: np.ndarray,
    n_terms: int = 8,
    decay: float = 0.25,
) -> np.ndarray:
    """Backward-compatible alias for the finite exponential least-squares fit."""
    return fit_exponential_series(times, target, n_terms=n_terms, decay=decay)


def rk4_step_refinement_error(rhs, y0: np.ndarray, dt: float, n_steps: int, **params) -> float:
    """Relative difference between a coarse RK4 solve and its refined counterpart."""
    if not isinstance(n_steps, numbers.Integral) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    dt = _validate_positive_scalar(dt, "dt")
    y0 = np.asarray(y0, dtype=float)
    if y0.ndim == 0:
        y0 = y0.reshape(1)
    if not np.all(np.isfinite(y0)):
        raise ValueError("y0 contains non-finite values")

    coarse_times = np.linspace(0.0, dt * n_steps, n_steps + 1)
    coarse = np.empty((coarse_times.size, y0.size), dtype=float)
    coarse[0] = y0
    for idx in range(n_steps):
        coarse[idx + 1] = rk4_step(rhs, coarse_times[idx], coarse[idx], coarse_times[idx + 1] - coarse_times[idx], **params)

    refined_times = np.linspace(0.0, dt * n_steps, 2 * n_steps + 1)
    refined = np.empty((refined_times.size, y0.size), dtype=float)
    refined[0] = y0
    for idx in range(2 * n_steps):
        refined[idx + 1] = rk4_step(rhs, refined_times[idx], refined[idx], refined_times[idx + 1] - refined_times[idx], **params)

    coarse_final = coarse[-1]
    refined_final = refined[-1]
    normalization = max(np.linalg.norm(refined_final, ord=np.inf), 1e-12)
    return np.linalg.norm(coarse_final - refined_final, ord=np.inf) / normalization


def _demo() -> None:
    times = np.linspace(0.0, 20.0, 250)
    sir = solve_sir(beta=0.6, gamma=0.2, s0=0.9, i0=0.1, r0=0.0, times=times)
    approx = fit_exponential_series(times, sir, n_terms=8, decay=0.22)
    err = np.linalg.norm(sir - approx, ord=np.inf) / np.linalg.norm(sir, ord=np.inf)
    print("SIR max relative error (finite exponential fit):", err)
    print("Final SIR state:", np.round(sir[-1], 6))

    seir = solve_seir(beta=0.7, sigma=0.3, gamma=0.2, s0=0.9, e0=0.05, i0=0.05, r0=0.0, times=times)
    print("Final SEIR state:", np.round(seir[-1], 6))
    print(
        "RK4 refinement error (example):",
        rk4_step_refinement_error(sir_rhs, np.array([0.9, 0.1, 0.0]), dt=0.5, n_steps=20, beta=0.6, gamma=0.2),
    )


if __name__ == "__main__":
    _demo()


__all__ = [
    "sir_rhs",
    "seir_rhs",
    "rk4_step",
    "solve_sir",
    "solve_seir",
    "fit_exponential_series",
    "dirichlet_approximation",
    "rk4_step_refinement_error",
]
