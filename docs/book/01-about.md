# 1. About this package

## 1.1 Purpose and application

Fully nonlinear climate models reproduce the atmosphere well enough that
interpreting *why* they do something is often as hard as understanding the
atmosphere itself. A **linear** model removes that difficulty by construction:
it linearises the primitive equations about a fixed basic state $\bar{x}$, so the
response to a small forcing is governed by a single linear operator and can be
decomposed, inverted and diagnosed exactly.

PyLBM-Isca is such a model. Its applications — the same set the Watanabe–Kimoto
LBM package was built for — are:

- **steady linear response** to a prescribed forcing ([](03-steady-response.ipynb)),
  obtained either by
  - time integration of the linear model to a stationary state, or
  - direct/iterative inversion of the linear operator (the stationary-wave and
    accelerated-iterative methods);
- the **Storm Track Model** — the variance a stochastic transient forcing
  maintains against dissipation ([](04-storm-track.ipynb));
- **modal decomposition** — eigenmodes and singular (optimal-growth) vectors of
  the linear operator ([](05-modal-decomposition.ipynb));
- time integration of the model with the single retained **nonlinearity** and a
  residual forcing ([](06-nonlinear.ipynb));
- a **barotropic** single-level reduction ([](08-barotropic.ipynb)).

The *moist* linear model (WK guide Ch. 7) — a prognostic humidity coupled to a
convective heating closure — is **documented but not yet developed** here
([](07-moist.md)): PyLBM-Isca already carries a humidity perturbation $q'$ and
uses virtual temperature in the pressure-gradient and hydrostatic terms, but the
heating–moisture *feedback* that closes a moist mode is left as a roadmap item.

## 1.2 Mathematical basis of the linear model

The model integrates the spectral primitive equations in a **pure-sigma**
vertical coordinate, written in vorticity–divergence form and linearised about a
time-independent basic state. Writing the perturbation state as

$$
x' = (\zeta',\, D',\, T',\, q',\, \pi'), \qquad
\pi' = \ln(\bar{p}_s + p_s') - \ln \bar{p}_s,
$$

(vorticity, divergence, temperature, specific humidity and the exact
log-surface-pressure perturbation), the model advances

$$
\frac{\partial x'}{\partial t} = \mathbf{L}(\bar{x})\, x' \;-\; \mathbf{D}\, x' \;+\; f,
$$

where $\mathbf{L}(\bar{x})$ is the linearised dynamical tendency, $\mathbf{D}$
collects the (linear) dissipation and $f$ is the external forcing. The dynamical
tendency assembles, term by term:

- the absolute-vorticity flux $A' = (\zeta+f)\,\mathbf{v}$ linearised to
  $\bar{\zeta+f}\,\mathbf{v}' + \zeta'\,\bar{\mathbf{v}}$;
- the pressure-gradient force $-R\,\Tv\,c_{pg}\,\nabla\ln p_s$, with **virtual**
  temperature $\Tv = T(1+0.608\,q)$ (in pure sigma $c_{pg}=1$);
- the Simmons–Burridge energy-conserving vertical operators (the $\alpha$,
  log-pressure and hydrostatic weights);
- centred vertical advection $-\dot\sigma\,\partial_\sigma(\cdot)$ of momentum,
  temperature and humidity;
- the adiabatic (compression) term $-\kappa\,\Tv\,X$ with the Simmons–Burridge
  $\omega/\sigma$-dot factor $X$;
- the continuity equation for $\pi'$.

Every term is a (basic-state $\times$ perturbation) product — i.e. genuinely
linear — with the **single** documented exception of the kinetic-energy term
$\tfrac12(u'^2+v'^2)$ that Isca retains in the divergence equation; set
`strictly_linear=True` to drop it and recover the exact tangent-linear operator
(this is exactly the quantity [](06-nonlinear.ipynb) isolates). The complete
term-to-code correspondence is tabulated in
[`docs/equation_code_map.md`](https://github.com/kuiper2000/PyLBM-Isca/blob/main/docs/equation_code_map.md)
and summarised in [](a-math-principles.md).

The **dissipation** $\mathbf{D}$ comprises $\nabla^4$ hyperdiffusion, a top
sponge layer, per-level Rayleigh drag (on $\zeta',D'$) and Newtonian cooling
(on $T',q'$), and an optional implicit vertical diffusion. All are linear and
diagonal in the spherical-harmonic index, so they enter the operator as simple
per-$(n,k)$ coefficients.

## 1.3 About the numerics

| aspect | choice |
|---|---|
| horizontal | spherical-harmonic transform, triangular truncation (T42 default) |
| vertical | Simmons & Burridge (1981) energy-conserving, pure sigma, 20 levels |
| time scheme | semi-implicit leapfrog; gravity waves solved implicitly per total wavenumber $n$ |
| time filter | genuine Robert–Asselin ($\gamma=0.04$; Robert 1966, Asselin 1972) |
| start-up | a short sequence of halved forward sub-steps (`inistp`) |
| backend | NumPy (CPU) or CuPy (GPU), inherited from the `Spectral` object |

The gravity-wave matrix depends only on the total wavenumber $n$, so there is one
$K_{\max}\times K_{\max}$ matrix per $n$ (43 of them at T42), built and inverted
once whenever the time step changes. The reference profile the matrix is built
about is the **basic-state mean virtual temperature** — the proper LBM choice —
rather than Isca's isothermal 300 K implicit reference.

## 1.4 File format

PyLBM-Isca reads and writes the **same formats** as the Watanabe–Kimoto LBM, so
the two models are drop-in comparable:

- **Basic state / forcing** — Fortran-sequential binary on a 17-level pressure
  grid (`bs.*.bin`, `frc.*.bin`); see [](02-getting-started.md).
- **History output** — GrADS `.grd` + `.ctl` (record order
  `psi, chi, u, v, w, t, z, ps`), with optional netCDF (`--netcdf`). The sample
  `.ctl` carries `OPTIONS YREV` because the model grid is ordered north→south.
- **Run configuration** — a TOML (or YAML) *namelist* whose tables map
  one-to-one onto the Fortran `&nmXXX` groups ([](02-getting-started.md)).

## 1.5 Feedback to the developer

PyLBM-Isca is developed in the open. Please file issues, questions and pull
requests at <https://github.com/kuiper2000/PyLBM-Isca>, and cite the archived
release (see `CITATION.cff`) if you use it. The model is developed with the
support of the original Watanabe–Kimoto LBM authors, with open distribution and
collaboration being arranged.
