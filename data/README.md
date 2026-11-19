# Input data

## NetCDF inputs (committed)

Self-contained NetCDF inputs for the **realistic dry** experiment
(`examples/example_3_realistic.ipynb`) and the **moist** experiment
(`examples/example_4_moist.ipynb`). These ship in the repository.

| file | size | description |
|---|---|---|
| `ncep.clim.t42.nc`   | 25 MB | **NCEP** pressure-level **basic state** climatology (12 months, 17 levels) on the 128×64 Gaussian grid; dims `(month, plev, lat, lon)`, vars z/rhum/q/t/u/v/omega, north-first. Read by `load_basic_state` (`.nc` path); JFM is `kmo=1, navg=3`. |
| `sst.clim.t42.nc`    | 0.2 MB | **Mean SST** climatology `(month, lat, lon)`, var `sst` [K]. Surface-flux basic state for the moist model (`load_moist_surface`). |
| `wg.clim.t42.nc`     | 0.1 MB | **Soil wetness** climatology `(month, lat, lon)`, var `wg` (0–1). Evaporation efficiency for the moist model (`load_moist_surface`). |
| `frcsst.nino.t21.nc` | 21 KB | **El Niño SST anomaly** on a coarse T21 (64×32) grid, var `s` [K]; spectrally regridded to T42 by `load_sst_anomaly`. Prescribed forcing for the moist model. |

Run the moist model from these with:

```bash
python run_lbm.py --basic-state data/ncep.clim.t42.nc --moist \
    --days 10 --out out/isca.moist.elnino --netcdf
```

## Binary inputs (Zenodo)

Binary inputs for the MJO forcing and the seasonality comparison. The idealised
Gill and solid-body examples build their basic states analytically and need
nothing from this folder.

These files are **not stored in git** — they are archived on Zenodo and
downloaded here on first run:

> **DOI:** [10.5281/zenodo.24086134](https://doi.org/10.5281/zenodo.24086134)
> · record <https://zenodo.org/records/24086134>

| file | size | description |
|---|---|---|
| `bs.1979.bin`    | 11 MB | **ERA5** reanalysis **basic state**, DJF 1979, interpolated onto the model's 17 standard pressure levels. Fortran-sequential big-endian `real*4`, ordered month → variable → level (z, rhum, q, t, u, v, omega) on the 128×64 Gaussian grid. Read by `pylbm_isca.basicstate.load_basic_state`. |
| `bs.jja.bin`     | 11 MB | **ERA5** reanalysis **summer (JJA) basic state**, same layout as `bs.1979.bin`, used for the seasonality comparison (`analysis/summer_validation.py`). The fields were downloaded only up to 100 hPa, so the levels above are extrapolated; the analysis holds them isothermally above 100 hPa (those levels lie in the top sponge). Supply a full-depth file to avoid this. |
| `frc.phase1.bin` | 46 MB | **MJO phase-1** time-varying temperature forcing, 73 days × 20 levels on the same grid. Read by `pylbm_isca.forcing.MJOForcing`. |

Both files use a Fortran-sequential, big-endian `real*4` layout and are
interchangeable with any file of the same structure — substitute your own
basic state / forcing by pointing `--basic-state` / `--mjo-forcing` (or the
notebook's `DATA` path) elsewhere.

## Provenance

`bs.1979.bin` and `frc.phase1.bin` are the author's own derived products: the
basic state is computed from **ERA5** reanalysis (DJF 1979) by the preprocessing
in `pylbm_isca.basicstate`, and the MJO phase-1 heating is constructed as
described in the manuscript (Sect. 4.3). Neither file is taken from the
permission-restricted Watanabe–Kimoto LBM distribution.
