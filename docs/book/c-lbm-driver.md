# Appendix C — The `LBM` driver (complete reference)

Every chapter of this guide ends up calling one object: `LBM`, defined in
[`pylbm_isca/model.py`](https://github.com/kuiper2000/PyLBM-Isca/blob/main/pylbm_isca/model.py).
It is the Python analogue of the Watanabe–Kimoto *time-advance linear model*
(their §3.1, the `tintgr` project and the `linear-run.*.csh` script): it holds
the basic state, the dynamical core and the dissipation, owns the three time
levels, and runs the semi-implicit leapfrog loop.

The getting-started chapter ([](02-getting-started.md)) showed the three-line
happy path. This appendix is the reference behind it: what the constructor
builds, what **every** `Config` knob does, what **every** `run()` argument does,
and exactly what happens on each step of the integration. Read it once; after
that the other chapters are variations on a theme.

:::{tip} The whole driver in one sentence
`LBM(bs, sp, vg, cfg).run(...)` assembles the linear operator
$\mathbf{L}(\bar{x})$ and the dissipation $\mathbf{D}$ once, then advances
$\partial_t x' = \mathbf{L}x' - \mathbf{D}x' + f$ with a semi-implicit,
Robert–Asselin-filtered leapfrog, sampling an 8-variable daily history as it
goes.
:::

---

## C.1 Constructing the model

```python
LBM(basic_state, spectral, vertical, config=None, basic_spectral=None)
```

| argument | type | what it is |
|---|---|---|
| `basic_state` | object with `u, v, t, q` `(nlat, nlon, kmax)` and `ps` `(nlat, nlon)` | the background $\bar{x}$ the equations are linearised about. Any object exposing those five grid-space fields qualifies — the analytic helpers in `lbmtools.py`, or `basicstate.load_basic_state(...)` for a `bs.*.bin` file. |
| `spectral` | `Spectral` | the Gaussian grid **and** the spherical-harmonic transforms. Its array backend (`sp.xp`, NumPy or CuPy) becomes the backend for the whole model — see [](b-troubleshooting.md). |
| `vertical` | `VerticalGrid` | the Simmons–Burridge sigma-coordinate operators (hydrostatic, $\alpha$, vertical-advection weights). |
| `config` | `Config` or `None` | every namelist knob (next section). `None` ⇒ all defaults (T42 L20, the `linearneutral.t42l20` profile). |
| `basic_spectral` | `Spectral` or `None` | a **non-batched** transform used only to build the basic-state terms. Pass it when `spectral` is an ensemble (batched) transform, because the basic state has no member axis. Otherwise leave it `None`. |

What the constructor does, in order:

1. **Basic-state terms** — `BasicStateTerms(...)` builds `TBAR` (the reference
   virtual-temperature profile) and every `TBAR`-dependent vertical matrix.
2. **Dynamics** — `Dynamics(...)`, the linearised tendency
   $\mathbf{L}(\bar{x})\,x'$, honouring `strictly_linear`.
3. **Dissipation** — `Damping(...)` ($\nabla^4$ hyperdiffusion + sponge +
   per-level Rayleigh drag / Newtonian cooling), `SemiImplicit(...)` (the
   gravity-wave solve), `VerticalDiffusion(...)`, wired together in
   `Integrator(...)`.
4. **Three time levels** — `ga` (current, $t$), `gb` (previous, $t-\Delta t$)
   and `gc` (scratch for $t+\Delta t$), each a zeroed `State`. If the transform
   is batched **or** `nmem > 1`, these carry a leading member axis.

After construction the model is at `time = 0.0`, `istep = 0`, with zero
perturbation. Read-only `model.device` reports `"cpu"` or `"gpu"`.

---

## C.2 `Config` — every knob

`Config` is a dataclass whose fields mirror the Fortran `&nmXXX` namelist groups
one-to-one; a TOML/YAML namelist loads straight into it with
`Config.from_namelist(path)` (see [](02-getting-started.md)).
Any field left unset keeps the default shown here.

### Time stepping (`&nmdelt`, `&nmtime`)

| field | default | unit | meaning |
|---|---|---|---|
| `delt` | `1200.0` | s | time step (20 min). Shorten it for a fast jet ([](b-troubleshooting.md)). |
| `ndays` | `66.0` | day | length of the integration. `nsteps = round(ndays · steps_per_day)`. |
| `inistp` | `2` | — | number of **halved forward sub-steps** used to start the leapfrog (§C.4). |

`Config.steps_per_day` is a derived property, `round(86400 / delt)`.

### Horizontal diffusion (`&nmhdif`)

| field | default | meaning |
|---|---|---|
| `hdif_order` | `4` | order of the $\nabla^{2p}$ hyperdiffusion ($\nabla^4$). |
| `hdif_tefold` | `2.0` | e-folding time of the smallest retained wave… |
| `hdif_tunit` | `"HOUR"` | …in these units (`"HOUR"`/`"DAY"`/…). |
| `sponge` | `True` | enable the top-of-model sponge layer. |
| `ray0` | `30.0` | sponge strength scale. |

### Rayleigh drag / Newtonian cooling (`&nmdamp`)

Per-level damping timescales, **`k=0` at the bottom**. The defaults are the
`linearneutral.t42l20` profile: a 0.5-day boundary-layer drag in the lowest
three layers, a 0.5-day sponge in the top two, and 20-day Newtonian damping in
between.

| field | applies to | default |
|---|---|---|
| `ddragv` | vorticity $\zeta'$ | the 20-element `DDRAG_DEFAULT` list |
| `ddragd` | divergence $D'$ | `DDRAG_DEFAULT` |
| `ddragt` | temperature $T'$ | `DDRAG_DEFAULT` |
| `ddragq` | humidity $q'$ | `None` (off) |
| `damp_tunit` | — | `"DAY"` |

### Vertical diffusion (`&nmvdif`)

| field | default | meaning |
|---|---|---|
| `vdifv`, `vdifd`, `vdift` | `1.0e3` | implicit vertical diffusion of $\zeta', D', T'$. |
| `vdifq` | `None` | humidity vertical diffusion (off by default). |

### Integration / fidelity

| field | default | meaning |
|---|---|---|
| `robert` | `0.04` | Robert–Asselin time-filter coefficient $\gamma$ (Isca `leapfrog.F90`). |
| `strictly_linear` | `False` | `True` drops the retained $\tfrac12(u'^2+v'^2)$ kinetic-energy term, recovering the exact tangent-linear operator (§C.8, [](06-nonlinear.ipynb)). |
| `nmem` | `1` | ensemble size — run this many perturbation experiments in one batched integration (§C.7). |

### History

| field | default | meaning |
|---|---|---|
| `out_interval_days` | `1.0` | write cadence; `out_every = round(out_interval_days · steps_per_day)`. |
| `time_average` | `True` | average each output interval (vs. instantaneous samples). |

### Moist / backend

| field | default | meaning |
|---|---|---|
| `moist` | `False` | advisory flag used by the namelist/`run_lbm` wiring; the *actual* moist path is switched on by passing a `MoistPhysics` to `run(moist=...)` ([](07-moist.md)). |
| `device` | `"auto"` | `"cpu"`/`"gpu"`/`"auto"`, used only when a driver builds the `Spectral`/`VerticalGrid` for you. |

---

## C.3 `run()` — every argument

```python
model.run(mjo_forcing=None, steady_forcing=None, writer=None,
          moist=None, sst_anom=None, wg_anom=None,
          verbose=True, progress_every=240)
```

| argument | type | role |
|---|---|---|
| `steady_forcing` | `SteadyForcing` or `None` | **time-constant** sources on vorticity, divergence, temperature and $\ln p_s$. Coerced onto the device once. This is the workhorse for the steady-response chapter. |
| `mjo_forcing` | `MJOForcing` or `None` | a **time-varying temperature** forcing, stepped once per step via `.step()`. When present it **replaces** the steady *temperature* forcing (`steady_forcing`'s `vor/div/ps` still apply). |
| `writer` | object with `.add(day, fields)` and `.close()`, or `None` | the history sink. `None` ⇒ no history is sampled at all (and `diagnostics()` is not called inside the loop, saving work). |
| `moist` | `MoistPhysics` or `None` | enables the interactive-moisture path (§C.4): convective + large-scale-condensation heating/moistening and the SST-anomaly surface flux are recomputed from the state every step and added to the $T'$/$q'$ tendencies, and the implicit moist damping `IMPTQ` is applied after the leapfrog. |
| `sst_anom`, `wg_anom` | arrays or `None` | prescribed SST anomaly (K) and soil-wetness anomaly (fraction) on the grid, driving the external forcing $F_e$ in a moist run. Coerced to the device once. |
| `verbose` | `bool` | print the backend banner and the periodic progress line. |
| `progress_every` | `int` | print a `|u|max / |T|max / |ps|max` line every this-many advanced steps (`0`/`None` ⇒ silent). |

`run()` mutates the model in place (`time`, `istep`, `ga/gb/gc`) and returns
`self`, so it chains: `psi = model.run(...).diagnostics()["psi"]`.

:::{important} Initial perturbation vs. steady forcing
To study the response to a *forcing*, pass `steady_forcing`/`mjo_forcing` and
start from rest. To study the evolution of an initial *perturbation* (the storm
track / singular-vector questions), seed it with `set_initial_perturbation(...)`
(§C.9) and run with no forcing. This is the same `FRC`-vs-`SFRC` distinction the
Watanabe–Kimoto `linear-run` script makes.
:::

---

## C.4 Inside the loop — one step at a time

The loop runs until `advanced == nsteps`. `istep` counts *attempts*; `advanced`
counts the full-`delt` steps that actually move `time` forward — the startup
sub-steps do **not** count toward `ndays`.

1. **Startup ramp (`inistp`).** While `istep ≤ inistp`, the step is a *forward*
   (Euler-like) sub-step with a **halved** time step,
   $\Delta t = \texttt{delt}/2^{\,\texttt{inistp}-\texttt{istep}+1}$ — so with
   the default `inistp=2` the first two sub-steps use `delt/4` then `delt/2`.
   On these steps `gb` is copied from `ga` (a forward start) instead of being
   Robert-filtered, `time` is not advanced and no history is written. From
   `istep > inistp` onward every step is a full-`delt` leapfrog.

2. **Assemble the forcing for this step.** The temperature forcing is
   `mjo_forcing.step()` if given, else the steady `t`; vorticity/divergence/$\ln
   p_s$ come from the steady forcing. In a moist run, `moist.tendencies(ga, …)`
   is evaluated from the **current** state and its heat/moisture sources are
   **added** to the temperature forcing and supplied as the humidity forcing.

3. **Tendency.** `dyn.tendencies(ga, …)` evaluates $\mathbf{L}(\bar{x})\,x' + f$
   in spectral space.

4. **Top-of-step diagnostics.** If a `writer` is attached, `diagnostics()` is
   sampled **now** (all fields at the pre-update time $t$) so every history
   variable is consistent; in a moist run the physics diagnostics
   (`q1c, q2c, q1l, q2l, pr`) are appended.

5. **Semi-implicit leapfrog + Robert–Asselin.** `integ.step(dt, prev, tend)`
   advances from the $t-\Delta t$ level (gravity waves solved implicitly per
   total wavenumber $n$); the result lands in `gc`. Off the startup ramp, `gb`
   is replaced by the Robert–Asselin-filtered value
   `gb.robert_asselin(ga, gc, robert)`, then `ga ← gc`.

6. **Moist implicit damping.** In a moist run, `moist.imptq(ga.t, ga.q, dt)`
   applies the implicit $T'$/$q'$ damping to the freshly updated state.

7. **Blow-up guard.** If any field goes non-finite, `run()` raises
   `FloatingPointError("model blew up at step N …")` — see
   [](b-troubleshooting.md) for the usual causes (growing basic state, time
   step too long, forcing too strong).

---

## C.5 History sampling and averaging

History is sampled **at the top of the step** (all fields at time $t$, before the
update) and **trapezoidally** combined with the previous step's sample —
`0.5·(this + previous)` — so the Python daily history uses the same leapfrog
sampling point as the Fortran LBM and the two codes agree to round-off in the
early response.

- With `time_average=True` (default), those per-step means are accumulated and,
  every `out_every` steps, their average is written: `writer.add(day, fields)`.
- With `time_average=False`, the per-step mean itself is written every
  `out_every` steps (an instantaneous-style snapshot).

Each `fields` dict is moved to the host (`asnumpy`) at write time; everything
upstream stays on the device. The **writer protocol** is minimal — any object
with `add(day:int, fields:dict[str, np.ndarray])` and `close()` works. The
built-in GrADS/netCDF writers live in `pylbm_isca/output.py`; the record order is
`psi, chi, u, v, w, t, z, ps` (see [](01-about.md)).

---

## C.6 `diagnostics()` — the eight history variables

```python
fields = model.diagnostics()   # dict of device arrays, grid space
```

The same eight quantities the Watanabe–Kimoto output carries, all in grid space,
all diagnosed from the current time level $t$:

| key | field | from | unit |
|---|---|---|---|
| `psi` | streamfunction $\psi'$ | $\nabla^{-2}\zeta'$ | m² s⁻¹ |
| `chi` | velocity potential $\chi'$ | $\nabla^{-2}D'$ | m² s⁻¹ |
| `u` | zonal wind $u'$ | state | m s⁻¹ |
| `v` | meridional wind $v'$ | state | m s⁻¹ |
| `w` | $\omega'$ (pressure vertical velocity) | tendency diag | Pa s⁻¹ |
| `t` | temperature $T'$ | state | K |
| `z` | geopotential height $z'$ | hydrostatic$(T')/g$ | m |
| `p` | surface pressure $p_s'$ | state | Pa |

A moist run additionally exposes `q1c, q2c` (convective heat/moisture sources),
`q1l, q2l` (large-scale condensation) and `pr` (precipitation) in the history
dict. `diagnostics()` is cheap but not free (two inverse Laplacians and a
hydrostatic integral); the loop only calls it when a `writer` is present.

---

## C.7 Ensemble (batched) runs

Set `nmem > 1` (or pass an already-batched `Spectral`) to integrate many
perturbation experiments at once. The three time levels then carry a leading
member axis — `State.zeros_ensemble` makes the 3-D fields `(nmem, nlat, nlon,
kmax)` and `ps` `(nmem, nlat, nlon)`, while `ps_bar` stays 2-D and broadcasts.
Every per-step operation runs on all members simultaneously, which is where the
GPU backend pays off. Because the basic state has no member axis, pass a plain
single-member transform as `basic_spectral` so `BasicStateTerms` is built
without one:

```python
sp_batched = Spectral(..., nmem=32)          # batched transform
sp_single  = Spectral(...)                   # plain transform for the basic state
model = LBM(bs, sp_batched, vg, Config(nmem=32), basic_spectral=sp_single)
```

---

## C.8 `strictly_linear` and the one nonlinearity

PyLBM-Isca is linear with a **single** documented exception: the
$\tfrac12(u'^2+v'^2)$ kinetic-energy term Isca keeps in the divergence equation.
Leave `strictly_linear=False` (default) to match Isca; set it `True` to drop
that term and obtain the exact tangent-linear operator. The nonlinear chapter
([](06-nonlinear.ipynb)) isolates precisely this term, and the modal/steady
chapters that treat $\mathbf{L}$ as a matrix implicitly assume the strict form.

---

## C.9 Seeding an initial perturbation

```python
model.set_initial_perturbation(u=None, v=None, t=None, q=None, ps=None)
```

Writes whichever grid-space fields you supply into **both** `ga` (t) and `gb`
(t−Δt), so the leapfrog starts from a consistent, non-zero state. Use it for the
transient-growth / storm-track questions (seed a perturbation, run with no
forcing); omit it for the forced-response questions (start from rest, pass a
forcing to `run()`).

---

## C.10 Minimal recipes

```python
from pylbm_isca import Spectral, VerticalGrid, LBM, Config
import lbmtools as lt

sp, vg = Spectral(), VerticalGrid()
bs = lt.baroclinic_jet(sp, vg, umax=40.0)

# (a) steady response to a Gaussian heat source
from pylbm_isca.forcing import SteadyForcing
heat = lt.gaussian_heat(sp, vg, amp_k_per_day=1.0, lon0=180, lat0=0)  # K/s array
frc = SteadyForcing.zeros(sp.nlat, sp.nlon, vg.kmax); frc.t = heat
model = LBM(bs, sp, vg, Config(ndays=20.0))
psi = model.run(steady_forcing=frc).diagnostics()["psi"]

# (b) transient growth of an initial perturbation (no forcing)
model = LBM(bs, sp, vg, Config(ndays=10.0))
model.set_initial_perturbation(t=initial_T_anomaly)
model.run()

# (c) time-varying (MJO) heating, with a history writer
from pylbm_isca.forcing import MJOForcing
from pylbm_isca.output import GradsWriter          # see output.py
model = LBM(bs, sp, vg, Config(ndays=35.0))
model.run(mjo_forcing=MJOForcing("frc.phase1.bin"),
          writer=GradsWriter("out/run1", sp, vg))

# (d) interactive-moisture run (see Chapter 7 for status)
from pylbm_isca import MoistPhysics, MoistParams
moist = MoistPhysics(vg, bs, params=MoistParams(), delt=1200.0)
model = LBM(bs, sp, vg, Config(ndays=30.0, moist=True))
model.run(moist=moist, sst_anom=sst_K)
```

For the full governing equations behind $\mathbf{L}$ and $\mathbf{D}$ see
[](a-math-principles.md) and the term-by-term map in
[`docs/equation_code_map.md`](https://github.com/kuiper2000/PyLBM-Isca/blob/main/docs/equation_code_map.md);
for where to change the driver itself see [](b-troubleshooting.md).
