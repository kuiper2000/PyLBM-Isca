---
title: PyLBM-Isca Users' Guide
---

# PyLBM-Isca Users' Guide

:::{note} What this book is
This is a **Users' Guide** for [PyLBM-Isca](https://github.com/kuiper2000/PyLBM-Isca),
written in the spirit of Masahiro Watanabe's *Linear Baroclinic Model (LBM)
Package Users' Guide* (v2.2, 2005). It walks through the same tasks — building
the model, prescribing a basic state and a forcing, and reading out the steady
and transient response — and then the same diagnostic machinery: solving the
steady response by **matrix inversion** and by an **iterative/accelerated**
method, the **Storm Track Model**, **modal decomposition** (eigenvalues and
singular vectors), the **nonlinear** residual-forcing run, and the
**barotropic** reduction.
:::

PyLBM-Isca is a **Linear Baroclinic Model** whose dynamical core is a
linearisation of the Isca / GFDL `atmos_spectral` primitive equations about a
prescribed basic state. It is a spectral, pure-sigma, vorticity–divergence,
semi-implicit model at **T42 L20**, with a NumPy (CPU) or CuPy (GPU) backend
chosen at run time. It follows the *structure* of the Watanabe–Kimoto LBM — same
grid, same file conventions, same workflow — while deriving every dynamical term
from the GPLv3 Isca sources, so the whole model can be released under the GPL.

## How this guide maps onto the Watanabe–Kimoto guide

| WK guide chapter | Here | PyLBM-Isca realisation |
|---|---|---|
| 1. About this package | [](01-about.md) | purpose, equations, numerics, file formats |
| 2. Getting started | [](02-getting-started.md) | install, the TOML *namelist*, basic states |
| 3. Steady response | [](03-steady-response.ipynb) | time integration **and** matrix inversion (direct + iterative/AIM) |
| 4. Storm Track Model | [](04-storm-track.ipynb) | variance maintained by stochastic transient forcing |
| 5. Modal decomposition | [](05-modal-decomposition.ipynb) | eigenmodes and singular (optimal-growth) vectors |
| 6. Nonlinear model | [](06-nonlinear.ipynb) | the retained kinetic-energy nonlinearity; residual forcing |
| 7. Moist LBM | [](07-moist.md) | documented; moist *coupling* is a roadmap item |
| 8. Barotropic model | [](08-barotropic.ipynb) | single-level nondivergent reduction, full spectrum |
| A. Mathematical principles | [](a-math-principles.md) | the linear dynamical system and its operator |
| B. Troubleshooting | [](b-troubleshooting.md) | common questions, where to change things |
| C. The `LBM` driver | [](c-lbm-driver.md) | complete reference for the main `model.py` driver — `Config`, `run()`, the time loop |

:::{tip} One operator, many chapters
Everything past a plain forward integration is a wrapper around a single object:
the **linearised tendency** $\mathbf{L}(\bar{x})\,x'$. Chapters 3–6 and 8 all
build their diagnostics on the small helper module
[`lbmtools.py`](https://github.com/kuiper2000/PyLBM-Isca/blob/main/docs/book/lbmtools.py),
whose `TangentLinear` class packs the five spectral perturbation fields into one
complex vector and exposes `A x` as a matrix-free `scipy.sparse.linalg`
operator. That is what makes `eigs`, `svds` and `gmres` one-liners.
:::

## Running the examples

The computational chapters build their own basic states analytically, so they
need **no external data** and run offline in a few seconds each at the reduced
truncation used here (T21 L20). To reproduce the published figures at the full
T42 L20 resolution with the realistic ERA5 basic state, see
[`REPRODUCE.md`](https://github.com/kuiper2000/PyLBM-Isca/blob/main/REPRODUCE.md)
and the notebooks under `examples/`.
