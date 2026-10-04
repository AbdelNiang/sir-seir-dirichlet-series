# SIR/SEIR with a Dirichlet-style approximation

This repository contains a minimal executable core for the report's SIR and SEIR models.
It is intentionally narrow and reproducible: the models are solved with a reference RK4
integrator, and a finite exponential series is used as a lightweight Dirichlet-style
approximation for comparison.

## Included

- SIR model with RK4 integration
- SEIR model with RK4 integration
- truncated exponential approximation in the spirit of the report
- deterministic validation tests for positivity and conservation

## Run

```bash
python epidemic_models.py
python -m pytest -q
```

## Scope and honesty

This is a minimal executable version consistent with the report, not an exhaustive
implementation of the full theoretical derivation. The goal is to provide a faithful,
reproducible baseline that can be audited and extended rationally.

## Report

The PDF report is kept in `rapport/` and `references/`.

