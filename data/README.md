# Input data

All example inputs are **archived on Zenodo** and downloaded into `data/` on
first run — none are stored in git (`*.nc` and `*.bin` are gitignored). The
idealised Gill and solid-body examples build their basic states analytically and
need nothing from this folder.

> **DOI (latest version):**
> [10.5281/zenodo.24086133](https://doi.org/10.5281/zenodo.24086133) — the
> *concept* DOI, which always resolves to the newest version
> · latest record <https://zenodo.org/records/24086134>

## NetCDF inputs

NetCDF inputs for the **realistic dry** experiment
(`examples/example_3_realistic.ipynb`) and the **moist** experiment
(`examples/example_4_moist.ipynb`).

| file | size | description |
|---|---|---|
| `ncep.clim.t42.nc`   | 25 MB | **NCEP** pressure-level **basic state** climatology (12 months, 17 levels) on the 128×64 Gaussian grid; dims `(month, plev, lat, lon)`, vars z/rhum/q/t/u/v/omega, north-first. Read by `load_basic_state` (`.nc` path); JFM is `kmo=1, navg=3`. |
| `sst.clim.t42.nc`    | 0.2 MB | **Mean SST** climatology `(month, lat, lon)`, var `sst` [K]. Surface-flux basic state for the moist model (`load_moist_surface`). |
| `wg.clim.t42.nc`     | 0.1 MB | **Soil wetness** climatology `(month, lat, lon)`, var `wg` (0–1). Evaporation efficiency for the moist model (`load_moist_surface`). |
| `frcsst.nino.t21.nc` | 21 KB | **El Niño SST anomaly** on a coarse T21 (64×32) grid, var `s` [K]; spectrally regridded to T42 by `load_sst_anomaly`. Prescribed forcing for the moist model. |
| `bs.1979.nc`     | 8 MB  | **ERA5** reanalysis **basic state**, DJF 1979 (3 months × 17 levels), on the 128×64 Gaussian grid; dims `(month, plev, lat, lon)`, vars z/rhum/q/t/u/v/omega, north-first. Read by `load_basic_state` (`.nc` path). Converted from `bs.1979.bin`. |
| `bs.jja.nc`      | 7 MB  | **ERA5** reanalysis **summer (JJA) basic state**, same structure as `bs.1979.nc`, used for the seasonality comparison. The fields were ingested only up to 100 hPa, so the levels above are extrapolated (they lie in the top sponge). Converted from `bs.jja.bin`. |
| `frc.phase1.nc`  | 8 MB  | **MJO phase-1** time-varying temperature forcing, 73 days × 20 levels on the 128×64 grid; var `t`, dims `(day, lev, lat, lon)`, north-first. Read by `pylbm_isca.forcing.MJOForcing` (`.nc` path). Converted from `frc.phase1.bin`. |

Run the realistic MJO experiment and the moist model from these with:

```bash
python run_lbm.py --basic-state data/bs.1979.nc \
    --mjo-forcing data/frc.phase1.nc \
    --days 35 --out out/isca.1979.phase1 --netcdf

python run_lbm.py --basic-state data/ncep.clim.t42.nc --moist \
    --days 10 --out out/isca.moist.elnino --netcdf
```

The basic-state / forcing files are interchangeable with any file of the same
structure — substitute your own by pointing `--basic-state` / `--mjo-forcing`
(or the notebook's `DATA` path) elsewhere.

## Legacy binary inputs (optional)

The original Watanabe–Kimoto Fortran-sequential, big-endian `real*4` binaries
(`bs.1979.bin`, `bs.jja.bin`, `frc.phase1.bin`) are still readable — the loaders
auto-select the binary path for a non-`.nc` extension — but the NetCDF files
above are the canonical inputs. Regenerate the NetCDF from the binaries at any
time with:

```bash
python data/convert_bin_to_nc.py
```

The script is lossless: it carries a round-trip check asserting each `.nc` reads
back bit-for-bit identically to the binary reader before the file is kept.

## Provenance

The DJF 1979 basic state and the MJO phase-1 forcing are the author's own
derived products: the basic state is computed from **ERA5** reanalysis (DJF
1979) by the preprocessing in `pylbm_isca.basicstate`, and the MJO phase-1
heating is constructed as described in the manuscript (Sect. 4.3). Neither file
is taken from the permission-restricted Watanabe–Kimoto LBM distribution. The
`.nc` files are byte-faithful conversions of the original binaries.
