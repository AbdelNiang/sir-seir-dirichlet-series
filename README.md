# SIR/SEIR with a finite exponential comparison

This repository contains a minimal executable core for the report's SIR and SEIR models.
It is intentionally narrow and reproducible: the models are solved with a reference RK4
integrator, and a finite exponential basis is used as a practical comparison tool.

## What is implemented

- SIR model with RK4 integration
- SEIR model with RK4 integration
- a least-squares fit on a finite exponential basis, used as a numerical comparison
- deterministic validation tests for positivity and conservation
- explicit validation of time grids, model parameters, and non-finite outputs

## State variables and assumptions

The state variables are proportions of a closed population, not absolute counts:

- S + I + R = 1 for SIR,
- S + E + I + R = 1 for SEIR.

This means a trajectory is interpreted as a normalized population vector. If you want
absolute counts, rescale the initial conditions and solver outputs externally.

## Important limitation

This repository does not yet implement a standalone Dirichlet-series solver derived
independently from the report. The current `fit_exponential_series()` routine is a
finite-dimensional least-squares fit against the RK4 trajectory. It is useful for
numerical comparison, but it should not be interpreted as a complete derivation of the
report's analytical method without the missing coefficients and derivation from the
source document.

## Installation and run

```bash
python3 -m pip install -e .[test]
python3 epidemic_models.py
python3 -m pytest -q
```

## Numerical caution

- RK4 is a valid reference scheme, but it does not guarantee positivity for an
  arbitrary time step.
- A sign change in the state vector indicates a time step that is too large for the
  chosen regime or an unstable configuration.
- The displayed fit error is a relative sup-norm error between the RK4 trajectory and
  the finite exponential approximation; it is not a proof of a closed-form analytical
  solution.
- The RK4 integration error and the exponential-series fitting error are distinct
  quantities and should not be conflated.

## Scope and honesty

This is a minimal executable version consistent with the report, not an exhaustive
implementation of the full theoretical derivation. The goal is to provide a faithful,
reproducible baseline that can be audited and extended rationally.

## Report

The PDF report is kept in `rapport/` and `references/`.

