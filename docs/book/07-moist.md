# 7. Moist LBM

:::{warning} Status: documented, not yet developed
This chapter is the one the present guide deliberately **does not implement** as
a runnable experiment. PyLBM-Isca already carries the *dry-dynamical* ingredients
a moist LBM needs — a prognostic humidity perturbation and virtual temperature in
the buoyancy terms — but the **moisture–heating feedback** that closes a *moist
mode* (the part that makes a "moist LBM" different from a dry LBM with a humidity
tracer) is left as a roadmap item. This chapter records the design so the gap is
precise and the implementation is a small, well-scoped step.
:::

## 7.1 What is already here

The Watanabe–Kimoto moist LBM (their Ch. 7) adds a prognostic moisture equation
and couples humidity to the temperature tendency through a convective heating.
PyLBM-Isca already has the first half:

- a **prognostic humidity perturbation** $q'$, advected (horizontally and
  vertically) by the basic state and perturbation winds — the fourth component
  of the state vector $x'=(\zeta',D',T',q',\pi')$;
- **virtual temperature** $\Tv = T(1+0.608\,q)$ in the pressure-gradient force
  and the hydrostatic geopotential, so humidity already feeds back on the
  dynamics through buoyancy (`dynamics.py`: `tv`, `phiv`);
- per-level **Newtonian damping / vertical diffusion** of $q'$ (`ddragq`,
  `vdifq`), so a humidity anomaly relaxes like temperature.

What this gives you today is a **passive-moisture** linear model: $q'$ evolves
and loads the buoyancy, but there is no feedback *from* $q'$ *to* a diabatic
heating.

## 7.2 What a moist LBM adds

A moist LBM closes the loop with a **heating that depends on the moisture (and
vertical motion) perturbation**. Schematically the temperature equation gains a
term

$$
\left.\frac{\partial T'}{\partial t}\right|_{\text{moist}}
 = \frac{L_c}{c_p}\, \dot{q}'_{\text{cond}}, \qquad
\dot{q}'_{\text{cond}} = \mathcal{M}\big(q',\,\omega',\,\bar{x}\big),
$$

with $\mathcal{M}$ a **linearised convective closure** — in the simplest
Betts–Miller / moisture-convergence form, a relaxation of $q'$ toward a
reference profile gated by the basic-state convective mask, or a
$-\langle\omega' \partial_p \bar{q}\rangle$ moisture-convergence heating. Because
every ingredient ($q'$, $\omega'$, the basic-state moisture $\bar{q}$ and the
convective mask) is *already diagnosed in* `dynamics.py`, $\mathcal{M}$ enters as
one more **basic-state $\times$ perturbation** block in the linear operator
$\mathbf{L}$ — it does not change the solver, the time scheme or the file
formats.

The physically interesting consequence is a **moist mode**: the closure reduces
the effective static stability and can turn an otherwise-damped equatorial mode
into a slowly growing, eastward-moving one — the linear skeleton of the MJO and
of moisture-mode theory. In the diagnostic language of this guide it would appear
as a least-damped (or growing) eigenvalue of $\mathbf{L}$ in
[](05-modal-decomposition.ipynb), and as an amplified, phase-shifted steady
response in [](03-steady-response.ipynb).

## 7.3 Where it would plug in

Because of the operator design used throughout this guide, adding moisture
coupling is local:

| piece | file | change |
|---|---|---|
| convective closure $\mathcal{M}$ | `pylbm_isca/dynamics.py` | add a heating term to `gt_t` built from `qp`, `omega`, `bs.q` |
| closure parameters | `pylbm_isca/model.py` (`Config`) + `namelist.py` | a `[moist]` table ($L_c/c_p$, relaxation time, mask threshold) |
| basic-state SST / $\bar{q}$ | `pylbm_isca/basicstate.py` | already loads $\bar{q}$; add an SST / anomalous-SST reader (WK §7.1.2/7.2.2) |
| diagnostics | — | the modal and steady-response chapters work **unchanged** |

The WK guide's §7.1 (moist time integration) and §7.2 (moist steady response,
including the "associated heating response") then follow exactly the dry recipes
in [](03-steady-response.ipynb), with $\mathbf{L}$ replaced by its moist
counterpart. That is the sense in which "the rest" of this guide is complete: the
machinery that would run the moist experiments is already in place; only the
closure $\mathcal{M}$ remains to be written.
