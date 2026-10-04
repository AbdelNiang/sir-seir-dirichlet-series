import numpy as np

from epidemic_models import (
    dirichlet_approximation,
    rk4_step,
    seir_rhs,
    sir_rhs,
    solve_seir,
    solve_sir,
)


def test_sir_rhs_conserves_population():
    y0 = np.array([0.9, 0.1, 0.0])
    dy = sir_rhs(0.0, y0, beta=0.6, gamma=0.2)
    assert np.isclose(dy.sum(), 0.0, atol=1e-12)
    assert np.isfinite(dy).all()


def test_seir_rhs_conserves_population():
    y0 = np.array([0.8, 0.1, 0.1, 0.0])
    dy = seir_rhs(0.0, y0, beta=0.8, sigma=0.3, gamma=0.2)
    assert np.isclose(dy.sum(), 0.0, atol=1e-12)
    assert np.isfinite(dy).all()


def test_rk4_step_is_stable_for_sir():
    y0 = np.array([0.9, 0.1, 0.0])
    y1 = rk4_step(sir_rhs, 0.0, y0, 0.1, beta=0.6, gamma=0.2)
    assert np.all(y1 >= -1e-12)
    assert np.isclose(y1.sum(), 1.0, atol=1e-8)


def test_dirichlet_approximation_matches_reference_on_short_interval():
    times = np.linspace(0.0, 15.0, 200)
    y_rk4 = solve_sir(beta=0.6, gamma=0.2, s0=0.9, i0=0.1, r0=0.0, times=times)
    y_dir = dirichlet_approximation(
        times,
        target=y_rk4,
        n_terms=8,
        decay=0.25,
    )
    rel_error = np.linalg.norm(y_rk4 - y_dir, ord=np.inf) / np.linalg.norm(y_rk4, ord=np.inf)
    assert rel_error < 0.25


def test_seir_solution_stays_positive_and_conservative():
    times = np.linspace(0.0, 20.0, 200)
    y = solve_seir(beta=0.7, sigma=0.3, gamma=0.2, s0=0.9, e0=0.05, i0=0.05, r0=0.0, times=times)
    assert np.all(y >= -1e-12)
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-8)
