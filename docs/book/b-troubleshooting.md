# Appendix B — Troubleshooting

Short answers to the questions the Watanabe–Kimoto guide's Appendix B covered,
translated to PyLBM-Isca.

## B.1 How do I run this on a different system?

PyLBM-Isca is pure Python; there is no machine-specific build. Any platform with
NumPy + SciPy runs the CPU backend. For the GPU backend, install the CuPy wheel
matching your CUDA toolkit (`cupy-cuda12x` / `cupy-cuda11x`); the model then
selects the GPU automatically with `device="auto"` and falls back to CPU if no
device is present. The arrays, the transforms and the whole time loop follow the
backend of the `Spectral` object you pass in.

## B.2 How do I make a basic state or forcing on the T42 Gaussian grid?

Build an analytic state directly on the model grid (see
[](02-getting-started.md) and `lbmtools.baroclinic_jet` / `resting_state` /
`solid_body`), or interpolate gridded data to the Gaussian latitudes
`Spectral().lat_deg` and the sigma levels `VerticalGrid().sig`. To re-use the
Watanabe–Kimoto `bs.*.bin` / `frc.*.bin` binaries, write a Fortran-sequential
file on the 17 standard pressure levels (`constants.PLEV17`) and load it with
`pylbm_isca.basicstate.load_basic_state`; see
[`data/README.md`](https://github.com/kuiper2000/PyLBM-Isca/blob/main/data/README.md).

## B.3 The model blew up / "model blew up at step N"

The integrator raises `FloatingPointError` when a field goes non-finite. Usual
causes and fixes:

- **A growing basic state.** If $\bar{x}$ supports a baroclinic instability, the
  linear model *will* grow without bound — that is physics, not a bug. Diagnose
  it with [](05-modal-decomposition.ipynb) (a positive-$\operatorname{Re}$
  eigenvalue), and either add dissipation (raise `ray0`, shorten the `ddrag*`
  timescales, enable `sponge`) or ask a transient (singular-vector) question
  instead of a steady one.
- **Time step too long.** The semi-implicit scheme is stable for the gravity
  waves but not for strong advection by a fast jet; reduce `delt` (e.g. 1200 →
  600 s) for `umax ≳ 60` m s⁻¹.
- **Forcing too strong for a linear run.** The response scales linearly with $f$;
  if you only want a pattern, scale $f$ down.

## B.4 I want to change the model — where do I look?

| to change… | edit |
|---|---|
| a physical constant / the resolution | `pylbm_isca/constants.py` |
| a dynamical (tendency) term | `pylbm_isca/dynamics.py` |
| the Simmons–Burridge vertical operators | `pylbm_isca/vertical.py` |
| the semi-implicit solve / damping | `pylbm_isca/integrator.py` |
| the time loop, history, `Config` | `pylbm_isca/model.py` |
| the namelist tables | `pylbm_isca/namelist.py` |
| basic-state / forcing readers | `pylbm_isca/basicstate.py`, `forcing.py` |
| output formats | `pylbm_isca/output.py` |
| **a diagnostic in this guide** | `docs/book/lbmtools.py` |

The equation-to-code table in
[`docs/equation_code_map.md`](https://github.com/kuiper2000/PyLBM-Isca/blob/main/docs/equation_code_map.md)
points every governing-equation term at its line in the source.

## B.5 The diagnostic notebooks are slow

The computational chapters are sized to run offline in seconds at **T21 L20**.
If you raise the truncation or the number of retained zonal wavenumbers, cost
grows fast: the dense per-$m$ operator (`TangentLinear.dense`) costs one tendency
evaluation per column, and the full-operator GMRES costs one per iteration. Keep
`nmax` modest for interactive work; move to the GPU backend and the full T42 grid
for production. Cap BLAS oversubscription with `OMP_NUM_THREADS` if a dense build
pins every core.

## B.6 Does this reproduce the Watanabe–Kimoto LBM?

It is *comparable*, not identical: same grid, same file conventions, same
workflow and the same diagnostics, but a different dynamical core (Simmons &
Burridge / Isca vertical scheme, GFDL constants, a genuine Robert–Asselin
filter). See the README's "How it differs" section and
[`REPRODUCE.md`](https://github.com/kuiper2000/PyLBM-Isca/blob/main/REPRODUCE.md)
for the validation experiments.
