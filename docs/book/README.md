# PyLBM-Isca Users' Guide (Jupyter Book)

A multi-chapter **Users' Guide** for PyLBM-Isca, modelled on Masahiro Watanabe's
*Linear Baroclinic Model (LBM) Package Users' Guide* (v2.2, 2005). It walks
through the model and then builds every diagnostic of the WK guide **on the one
object the package exposes — the linearised tendency** $\mathbf{L}(\bar x)\,x'$:
the steady response by time integration and by matrix inversion, the Storm Track
Model, the modal (eigenvalue) decomposition, the nonlinear tangent-linear test,
and the barotropic reduction (with optimal-growth singular vectors).

## Contents

| file | chapter |
|---|---|
| `intro.md` | landing page + chapter map |
| `01-about.md` | purpose, equations, numerics, file formats |
| `02-getting-started.md` | install, the TOML namelist, basic states |
| `03-steady-response.ipynb` | time integration + matrix inversion (direct & iterative) |
| `04-storm-track.ipynb` | stochastically maintained variance (Lyapunov) |
| `05-modal-decomposition.ipynb` | eigenvalues, baroclinic instability, preferred modes |
| `06-nonlinear.ipynb` | the retained kinetic-energy nonlinearity; tangent-linear test |
| `07-moist.md` | moist LBM — documented (coupling is a roadmap item) |
| `08-barotropic.ipynb` | steady response, instability, optimal growth |
| `a-math-principles.md` | the linear dynamical system and its operator |
| `b-troubleshooting.md` | common questions |
| `lbmtools.py` | the shared operator helper every notebook imports |

## `lbmtools.py`

All computational chapters import this one module. It provides:

- **analytic basic states** (`resting_state`, `solid_body`, `baroclinic_jet`,
  `zonal_mean`) so the book runs offline;
- **`TangentLinear`** — a matrix-free complex linear operator $\mathbf{A}$ built
  from the model's own semi-implicit step, so its steady-state fixed point is
  *exactly* the model's; it packs the five spectral fields into one vector, plugs
  into `scipy.sparse.linalg`, and (restricted to one zonal wavenumber on a
  zonal-mean state) builds a small dense block for `eig`/direct solves;
- **`Barotropic`** — the gravity-wave-free single-level reduction used for the
  storm-track and optimal-growth analyses, where the energy norm is unambiguous.

The computational chapters run at **T21 L20** so the whole book builds offline in
a couple of minutes; pass the defaults (`Spectral()`, `VerticalGrid()`) for the
full T42 L20 configuration.

## Build

[Jupyter Book 2](https://next.jupyterbook.org) (MyST):

```bash
pip install jupyter-book            # v2.x (mystmd-based)
cd docs/book
jupyter-book start                  # live preview at http://localhost:3000
jupyter-book build --html           # static site -> _build/html
```

The notebooks are committed **with their outputs embedded**, so the HTML build
only renders (no execution). To re-execute them (e.g. after editing
`lbmtools.py`):

```bash
python _run.py 03-steady-response.ipynb     # one notebook, in place
# or regenerate the source from the _gen*.py scripts, then _run.py each
```

`_gen*.py` regenerate the notebook *source* (code + markdown, no outputs) via
`_mknb.py`; `_run.py` executes a notebook in place with `nbclient`.
