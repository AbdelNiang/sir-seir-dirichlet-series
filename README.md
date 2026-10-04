# SIR/SEIR with a finite exponential comparison

This repository contains a minimal executable core for the report's SIR and SEIR models.
It is intentionally narrow and reproducible: the models are solved with a reference RK4
integrator, and a finite exponential basis is used as a practical comparison tool.

## What is implemented

- SIR model with RK4 integration
- SEIR model with RK4 integration
- a least-squares fit on a finite exponential basis, used as a numerical comparison
- deterministic validation tests for positivity and conservation

## Important limitation

This repository does not yet implement a standalone Dirichlet-series solver derived
independently from the report. The current `dirichlet_approximation()` routine is a
finite-dimensional least-squares fit against the RK4 trajectory. It is useful for
numerical comparison, but it should not be interpreted as a complete derivation of the
report's analytical method without the missing coefficients and derivation from the
source document.

## Run

```bash
python3 epidemic_models.py
python3 -m pytest -q
```

## Scope and honesty

This is a minimal executable version consistent with the report, not an exhaustive
implementation of the full theoretical derivation. The goal is to provide a faithful,
reproducible baseline that can be audited and extended rationally.

## Report

The PDF report is kept in `rapport/` and `references/`.

