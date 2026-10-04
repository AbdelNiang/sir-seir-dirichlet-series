"""Minimal executable SIR/SEIR models with a Dirichlet-style approximation.

This module stays intentionally small and faithful to the report's core idea:
SIR and SEIR are solved with a deterministic RK4 integrator, and each state is
approximated by a truncated exponential series of the form
    X(t) ≈ sum_{k=0}^{N-1} a_k exp(-lambda * k * t).
The implementation is intentionally conservative: it compares the fitted series to
an RK4 reference, but it does not claim a mathematically exact closed form beyond
that numerical approximation.
"""

from __future__ import annotations

import numpy as np


def sir_rhs(t: float, y: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """Right-hand side of the SIR model."""
    s, i, r = y
    return np.array(
        [-beta * s * i, beta * s * i - gamma * i, gamma * i],
        dtype=float,
    )


def seir_rhs(t: float, y: np.ndarray, beta: float, sigma: float, gamma: float) -> np.ndarray:
    """Right-hand side of the SEIR model."""
    s, e, i, r = y
    return np.array(
        [-beta * s * i, beta * s * i - sigma * e, sigma * e - gamma * i, gamma * i],
        dtype=float,
    )


def rk4_step(rhs, t: float, y: np.ndarray, dt: float, **params) -> np.ndarray:
    """One RK4 step for a vector state."""
    y = np.asarray(y, dtype=float)
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
    """Solve the SIR model on a given time grid using RK4."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1D array")
    if np.any(np.diff(times) < 0):
        raise ValueError("times must be non-decreasing")

    y = np.empty((times.size, 3), dtype=float)
    y[0] = np.array([s0, i0, r0], dtype=float)
    for n in range(times.size - 1):
        h = times[n + 1] - times[n]
        y[n + 1] = rk4_step(sir_rhs, times[n], y[n], h, beta=beta, gamma=gamma)
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
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1D array")
    if np.any(np.diff(times) < 0):
        raise ValueError("times must be non-decreasing")

    y = np.empty((times.size, 4), dtype=float)
    y[0] = np.array([s0, e0, i0, r0], dtype=float)
    for n in range(times.size - 1):
        h = times[n + 1] - times[n]
        y[n + 1] = rk4_step(seir_rhs, times[n], y[n], h, beta=beta, sigma=sigma, gamma=gamma)
    return y


def dirichlet_approximation(
    times: np.ndarray,
    target: np.ndarray,
    n_terms: int = 8,
    decay: float = 0.25,
) -> np.ndarray:
    """Fit a truncated exponential series to a time-dependent target.

    The basis is exp(-(k * decay) * t), which matches the report's Dirichlet-style
    idea while staying numerically robust and reproducible.
    """
    times = np.asarray(times, dtype=float)
    target = np.asarray(target, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be a 1D array")
    if target.ndim == 1:
        target = target[:, np.newaxis]
    if target.shape[0] != times.size:
        raise ValueError("target rows must match the number of time points")
    if n_terms <= 0:
        raise ValueError("n_terms must be positive")
    if decay <= 0:
        raise ValueError("decay must be positive")

    basis = np.exp(-np.outer(times, np.arange(n_terms, dtype=float)) * decay)
    coefficients, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)
    return basis @ coefficients


def _demo() -> None:
    times = np.linspace(0.0, 20.0, 250)
    sir = solve_sir(beta=0.6, gamma=0.2, s0=0.9, i0=0.1, r0=0.0, times=times)
    approx = dirichlet_approximation(times, sir, n_terms=8, decay=0.22)
    err = np.linalg.norm(sir - approx, ord=np.inf) / np.linalg.norm(sir, ord=np.inf)
    print("SIR max relative error (Dirichlet fit):", err)
    print("Final SIR state:", np.round(sir[-1], 6))

    seir = solve_seir(beta=0.7, sigma=0.3, gamma=0.2, s0=0.9, e0=0.05, i0=0.05, r0=0.0, times=times)
    print("Final SEIR state:", np.round(seir[-1], 6))


if __name__ == "__main__":
    _demo()
