# 2. Getting started

The Watanabe–Kimoto guide opens by unpacking a tarball, editing `Lmake.inc`, and
building a Fortran library. PyLBM-Isca is a pure-Python package, so "getting
started" is an import — but the *workflow* is identical: pick a backend, build
the transform and vertical grid, prescribe a basic state and a forcing, then
integrate.

## 2.1 Install

```bash
pip install numpy scipy          # required
pip install xarray netCDF4       # optional, netCDF history output
pip install cupy-cuda12x         # optional, the GPU backend (CUDA 12.x)
```

There is no compile step and no `Lmake.inc`: the CPU (NumPy) backend needs
nothing beyond NumPy and SciPy. The GPU backend is selected automatically when a
CuPy build is importable and a device is present (`device="auto"`), and
everything downstream — the transforms, the time loop, the operator in this
guide — inherits that backend from the `Spectral` object.

```python
from pylbm_isca import Spectral, VerticalGrid, LBM, Config, gpu_available

device = "gpu" if gpu_available() else "cpu"
sp = Spectral(device=device)        # T42 Gaussian grid + transforms
vg = VerticalGrid(device=device)    # Simmons-Burridge L20 sigma operators
print(sp.nmax, sp.nlon, sp.nlat, vg.kmax)   # 42 128 64 20
```

:::{tip} Resolution
`Spectral(nlon=64, nlat=32, nmax=21)` gives a T21 grid. The computational
chapters of this guide use **T21 L20** so they run offline in seconds; pass the
defaults for the full **T42 L20** configuration that matches the
Watanabe–Kimoto grid term for term.
:::

## 2.2 Configure a run from a file — the *namelist*

The Fortran LBM drives every run from a `SYSIN` file: a stack of `&nmXXX … &end`
groups. PyLBM-Isca keeps that tradition as a single **TOML** (or YAML) file whose
tables map one-to-one onto the original groups. Point the driver at it with
`--namelist`, or build a `Config` from it in a notebook:

```python
from pylbm_isca import Config
cfg = Config.from_namelist("examples/mjo_phase1.toml")
```

```toml
# examples/mjo_phase1.toml  (abridged)
[time]                      # &nmtime / &nmdelt
ndays  = 35
delt   = 1200.0            # time step [s]  (= 20 min)

[damp]                      # &nmdamp  -- per-level Rayleigh drag, k=0 at bottom
ray0   = 30.0
sponge = true
ddragv = [0.5, 0.5, 0.5, 5, 15, 20, 20, 20, 20, 20,
          20, 20, 20, 20, 20, 20, 20, 20, 0.5, 0.5]

[input]                     # &nminit
basic_state = "data/bs.1979.bin"
[forcing]                   # &nmfrc / &nmsfrc
mjo    = "data/frc.phase1.bin"
spinup = 435
[output]                    # &nmhist / &nmhisd
out    = "out/isca.1979.phase1"
```

| table | Fortran group | knobs |
|---|---|---|
| `[time]`        | `&nmtime` / `&nmdelt` | `ndays`, `delt`, `inistp` |
| `[hdif]`        | `&nmhdif` | `order`, `tefold`, `tunit` ($\nabla^4$ diffusion) |
| `[damp]`        | `&nmdamp` | per-level `ddragv/d/t`, `tunit`, `sponge`, `ray0` |
| `[vdif]`        | `&nmvdif` | per-level `vdifv/d/t/q` |
| `[integration]` | — | `robert`, `strictly_linear`, `nmem` |
| `[input]`       | `&nminit` | `basic_state`, `kmo`, `navg`, `zonal_mean` |
| `[forcing]`     | `&nmfrc` / `&nmsfrc` | `mjo`, `steady`, `case`, `spinup`, `scale` |
| `[output]`      | `&nmhist` / `&nmhisd` | `out`, `netcdf`, `interval_days`, `instantaneous` |
| `[backend]`     | — | `device` |

Any omitted value keeps the model default, and **any command-line flag overrides
the file**, so the namelist and the equivalent command line produce identical
runs. Unknown sections or keys raise immediately, as the Fortran reader did.

## 2.3 Preparing basic states

The linear response is only as meaningful as the basic state it linearises
about. PyLBM-Isca accepts three kinds:

**(a) Analytic states** — built in a few lines, no files. Any object with
grid-space `u, v, t, q` (shape `(nlat, nlon, kmax)`) and `ps` (`(nlat, nlon)`)
is a valid basic state. The guide's helper module provides the three used
throughout:

```python
import lbmtools as lt
bs = lt.resting_state(sp, vg)                 # resting isothermal (Gill)
bs = lt.solid_body(sp, vg, u0=25.0)           # super-rotation
bs = lt.baroclinic_jet(sp, vg, umax=40.0)     # mid-latitude baroclinic jet
```

**(b) Realistic states from ERA5** — the Watanabe–Kimoto `bs.*.bin` format: a
Fortran-sequential file on 17 standard pressure levels, interpolated to the 20
sigma levels on load. PyLBM-Isca reads it unchanged (the WK guide's §2.4.1/2.4.2
NCEP/ERA40 recipes apply verbatim, with ERA5 substituted):

```python
from pylbm_isca.basicstate import load_basic_state
bs = load_basic_state("data/bs.1979.bin", sp, vg, kmo=1, navg=3)
```

The `bs.1979.bin` (DJF 1979) state is archived on
[Zenodo](https://doi.org/10.5281/zenodo.24086134); the loaders accept any
Fortran-sequential file of the same 17-level layout, so you can substitute your
own. `kmo`/`navg` select the month and the averaging window; `zonal_mean=True`
gives a zonally symmetric state (which decouples the operator by zonal
wavenumber — useful for the modal and barotropic chapters).

**(c) Zonal-mean states** — pass `zonal_mean=True` to the loader, or average any
state over longitude, to obtain a $\bar{x}(\phi,\sigma)$ background.

## 2.4 Forcing

Two forcing kinds mirror the WK package:

```python
from pylbm_isca.forcing import SteadyForcing, MJOForcing

frc = SteadyForcing.zeros(sp.nlat, sp.nlon, vg.kmax)
frc.t = heating_in_K_per_s                    # a steady heat source
# ... or a time-varying heating read from the WK frc.*.bin format:
mjo = MJOForcing("data/frc.phase1.bin")
```

`SteadyForcing` carries constant `vor, div, t, ps` sources; `MJOForcing` steps a
time-varying temperature forcing. The next chapter uses a steady Gaussian heat
source, built with the helper `lt.gaussian_heat`.

## 2.5 A first integration

```python
model = LBM(bs, sp, vg, Config(ndays=20.0))
model.run(steady_forcing=frc, verbose=True)
psi = model.diagnostics()["psi"]              # streamfunction, grid space
```

That is the whole workflow. The remaining chapters keep this skeleton — build
`sp, vg`; choose `bs`; choose `f` — and change only the *question* asked of the
operator.

:::{seealso} The `LBM` driver in full
This chapter shows the happy path. For a complete reference to the main
`model.py` driver — **every** `Config` knob, **every** `run()` argument, the
startup ramp, the semi-implicit/Robert–Asselin step, history averaging, the
moist path and ensemble runs — see [](c-lbm-driver.md).
:::
