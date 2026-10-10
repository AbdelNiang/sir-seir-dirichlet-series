import numpy as np
import pytest

from epidemic_models import (
    fit_exponential_series,
    rk4_step,
    rk4_step_refinement_error,
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


def test_fit_exponential_series_matches_reference_on_short_interval():
    times = np.linspace(0.0, 15.0, 200)
    y_rk4 = solve_sir(beta=0.6, gamma=0.2, s0=0.9, i0=0.1, r0=0.0, times=times)
    y_fit = fit_exponential_series(
        times,
        target=y_rk4,
        n_terms=8,
        decay=0.25,
    )
    rel_error = np.max(np.abs(y_rk4 - y_fit)) / np.max(np.abs(y_rk4))
    assert rel_error < 0.25


def test_rk4_refinement_converges_against_analytic_sir_case():
    errors = []
    for step_count in (20, 40):
        times = np.linspace(0.0, 4.0, step_count + 1)
        solution = solve_sir(
            beta=0.4,
            gamma=0.7,
            s0=0.0,
            i0=0.3,
            r0=0.7,
            times=times,
        )
        exact_infected = 0.3 * np.exp(-0.7 * times)
        exact = np.column_stack((np.zeros_like(times), exact_infected, 1.0 - exact_infected))
        errors.append(np.max(np.abs(solution - exact)))

    refinement_error = rk4_step_refinement_error(
        sir_rhs,
        np.array([0.0, 0.3, 0.7]),
        dt=0.2,
        n_steps=20,
        beta=0.4,
        gamma=0.7,
    )
    assert errors[1] < errors[0] / 10.0
    assert 0.0 < refinement_error < errors[0]


def test_fit_exponential_series_rejects_invalid_inputs():
    times = np.linspace(0.0, 1.0, 5)
    target = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    with pytest.raises(ValueError, match="times"):
        fit_exponential_series(np.array([0.0, -1.0, 1.0]), target)
    with pytest.raises(ValueError, match="n_terms|decay"):
        fit_exponential_series(times, target, n_terms=0)


def test_sir_rejects_invalid_state_and_time_grid():
    with pytest.raises(ValueError, match="sum to 1"):
        solve_sir(0.6, 0.2, 0.8, 0.1, 0.0, np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        solve_sir(0.6, 0.2, 0.9, 0.1, 0.0, np.array([0.0, 1.0, 1.0]))


def test_seir_solution_stays_positive_and_conservative():
    times = np.linspace(0.0, 20.0, 200)
    y = solve_seir(beta=0.7, sigma=0.3, gamma=0.2, s0=0.9, e0=0.05, i0=0.05, r0=0.0, times=times)
    assert np.all(y >= -1e-12)
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-8)
