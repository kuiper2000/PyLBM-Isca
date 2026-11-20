"""Convert the legacy Fortran-sequential ``*.bin`` LBM inputs to CF NetCDF.

This makes the whole input set NetCDF so nothing depends on the big-endian
``real*4`` Watanabe-Kimoto layout any more.  Three files are converted:

===================  ============  =========================================
binary               -> NetCDF     content
===================  ============  =========================================
``bs.1979.bin``      bs.1979.nc    ERA5 DJF 1979 basic state (3 mo x 17 lev)
``bs.jja.bin``       bs.jja.nc     ERA5 JJA basic state (3 mo x 17 lev)
``frc.phase1.bin``   frc.phase1.nc MJO phase-1 heating (73 days x 20 lev)
===================  ============  =========================================

The basic-state files are written with the SAME structure/coordinates as the
hand-made ``ncep.clim.t42.nc`` (dims ``(month, plev, lat, lon)``, vars
z/rhum/q/t/u/v/omega, **north-first**), so they are read by the existing ``.nc``
branch of :func:`pylbm_isca.basicstate.load_basic_state`.  The raw binary is
south-first, so the latitude axis is flipped here on write (exactly the
``yflip=True`` the binary reader applies).

The MJO file is written as var ``t`` with dims ``(day, lev, lat, lon)``,
north-first (the binary is already north-first), read by the ``.nc`` branch of
:class:`pylbm_isca.forcing.MJOForcing`.

Run from the repo root::

    python data/convert_bin_to_nc.py

Idempotent: re-running overwrites the ``.nc`` outputs.  A built-in round-trip
check asserts the NetCDF reads back bit-for-bit identically to the binary
readers before the files are kept.
"""

import os

import numpy as np
from netCDF4 import Dataset

from pylbm_isca.constants import IMAX, JMAX, KMAX, PLEV17, SIGM
from pylbm_isca.basicstate import read_pressure_levels
from pylbm_isca.forcing import read_fortran_records

HERE = os.path.dirname(os.path.abspath(__file__))

# Variable order inside the pressure-level basic-state file.
BS_VARS = ["z", "rhum", "q", "t", "u", "v", "omega"]
NLEV_BS = 17                       # 17 standard ERA5 pressure levels
UNDEF_IN = -9.999e20               # input undefined sentinel

# Coordinate arrays come from the reference NetCDF so every basic state shares
# the identical lat/lon/plev axes.
REF_NC = os.path.join(HERE, "ncep.clim.t42.nc")


def _ref_coords():
    with Dataset(REF_NC) as ds:
        return (np.asarray(ds["lat"][:]),
                np.asarray(ds["lon"][:]),
                np.asarray(ds["plev"][:]))


def _read_basic_state_raw(path, nvar=len(BS_VARS), nlev=NLEV_BS):
    """Read every variable/level of a bs.*.bin as (month, var, lev, lat, lon).

    South-first, exactly as stored (no flip here); the flip to north-first is
    applied once, on write.  Mirrors the record layout of
    :func:`pylbm_isca.basicstate.read_pressure_levels`.
    """
    npts = IMAX * JMAX
    raw = np.fromfile(path, dtype=">f4")
    if raw.size % (npts + 2):
        raise ValueError(f"{path}: not a whole number of Fortran records")
    rec = raw.reshape(-1, npts + 2)
    markers = rec[:, 0].view(">i4")
    if not np.all(markers == npts * 4):
        raise ValueError(f"{path}: bad record markers (got {markers[0]}, "
                         f"want {npts * 4})")
    data = rec[:, 1:-1].reshape(-1, JMAX, IMAX).astype(np.float64)
    per_month = nvar * nlev
    if data.shape[0] % per_month:
        raise ValueError(f"{path}: {data.shape[0]} records not a multiple of "
                         f"{per_month}")
    nmonth = data.shape[0] // per_month
    return data.reshape(nmonth, nvar, nlev, JMAX, IMAX), nmonth


def convert_basic_state(name, src, dst):
    lat, lon, plev = _ref_coords()
    if not np.allclose(plev, PLEV17):
        raise ValueError("reference plev disagrees with PLEV17")
    arr, nmonth = _read_basic_state_raw(src)         # (mo, var, lev, lat, lon)
    arr = arr[:, :, :, ::-1, :]                      # south-first -> north-first
    n_undef = int(np.count_nonzero(arr == UNDEF_IN))

    with Dataset(dst, "w", format="NETCDF4") as ds:
        ds.createDimension("month", nmonth)
        ds.createDimension("plev", NLEV_BS)
        ds.createDimension("lat", JMAX)
        ds.createDimension("lon", IMAX)

        v = ds.createVariable("month", "i4", ("month",))
        v[:] = np.arange(1, nmonth + 1)
        v.long_name = "calendar month index within the averaging window"

        v = ds.createVariable("plev", "f8", ("plev",))
        v[:] = plev
        v.units = "hPa"; v.long_name = "pressure"
        v.standard_name = "air_pressure"; v.positive = "down"; v.axis = "Z"

        v = ds.createVariable("lat", "f8", ("lat",))
        v[:] = lat
        v.units = "degrees_north"; v.long_name = "latitude"
        v.standard_name = "latitude"; v.axis = "Y"
        v.note = ("Gaussian latitudes, north-first (descending). Orientation is "
                  "fixed by this coordinate; do not re-flip by convention.")

        v = ds.createVariable("lon", "f8", ("lon",))
        v[:] = lon
        v.units = "degrees_east"; v.long_name = "longitude"
        v.standard_name = "longitude"; v.axis = "X"

        for iv, vname in enumerate(BS_VARS):
            var = ds.createVariable(vname, "f4", ("month", "plev", "lat", "lon"),
                                    zlib=True, complevel=4)
            var[:] = arr[:, iv].astype(np.float32)

        ds.title = f"LBM basic-state input ({name})"
        ds.Conventions = "CF-1.8"
        ds.source_binary = os.path.basename(src)
        ds.history = ("Converted from Fortran-sequential big-endian real*4 "
                      "(month->variable->level) by data/convert_bin_to_nc.py; "
                      "latitude flipped south-first -> north-first.")
        ds.note = ("Read with load_basic_state(..., .nc path), which selects "
                   "yflip=False because this file is already north-first.")
    return nmonth, n_undef


def convert_mjo_forcing(src, dst, nlev=KMAX, ndays=73):
    rec = read_fortran_records(src)                  # (nrec, lat, lon) north-first
    expect = ndays * nlev
    if rec.shape[0] != expect:
        raise ValueError(f"{src}: {rec.shape[0]} records != {ndays}x{nlev}")
    t = rec.reshape(ndays, nlev, JMAX, IMAX)         # (day, lev, lat, lon)
    lat, lon, _ = _ref_coords()
    sig_full = 0.5 * (SIGM[:-1] + SIGM[1:])          # 20 full sigma levels

    with Dataset(dst, "w", format="NETCDF4") as ds:
        ds.createDimension("day", ndays)
        ds.createDimension("lev", nlev)
        ds.createDimension("lat", JMAX)
        ds.createDimension("lon", IMAX)

        v = ds.createVariable("day", "i4", ("day",))
        v[:] = np.arange(1, ndays + 1)
        v.long_name = "forcing day index"

        v = ds.createVariable("lev", "f8", ("lev",))
        v[:] = sig_full
        v.units = "1"; v.long_name = "sigma (full level)"
        v.positive = "down"; v.axis = "Z"

        v = ds.createVariable("lat", "f8", ("lat",))
        v[:] = lat
        v.units = "degrees_north"; v.long_name = "latitude"
        v.standard_name = "latitude"; v.axis = "Y"
        v.note = "Gaussian latitudes, north-first (descending)."

        v = ds.createVariable("lon", "f8", ("lon",))
        v[:] = lon
        v.units = "degrees_east"; v.long_name = "longitude"
        v.standard_name = "longitude"; v.axis = "X"

        var = ds.createVariable("t", "f4", ("day", "lev", "lat", "lon"),
                                zlib=True, complevel=4)
        var[:] = t.astype(np.float32)
        var.long_name = "MJO temperature forcing"
        var.units = "K s-1"

        ds.title = "LBM MJO phase-1 temperature forcing"
        ds.Conventions = "CF-1.8"
        ds.source_binary = os.path.basename(src)
        ds.history = ("Converted from Fortran-sequential big-endian real*4 "
                      "(day->level) by data/convert_bin_to_nc.py; north-first, "
                      "no latitude flip.")
        ds.note = "Read by pylbm_isca.forcing.MJOForcing (.nc path, yflip=False)."
    return ndays, nlev


def _check_basic_state(src, dst):
    """The .nc must read back identically to the binary basic-state reader."""
    from pylbm_isca.basicstate import read_ncep_nc
    for kmo in (1,):
        a = read_pressure_levels(src, kmo=kmo, navg=1)      # yflip=True inside
        b = read_ncep_nc(dst, kmo=kmo, navg=1)
        for k in a:
            if not np.array_equal(a[k].astype(np.float32),
                                  b[k].astype(np.float32)):
                raise AssertionError(f"{dst}: {k} round-trip mismatch")


def _check_mjo(src, dst):
    from pylbm_isca.forcing import MJOForcing
    a = MJOForcing(src)
    b = MJOForcing(dst)
    if not np.array_equal(a.gtf.astype(np.float32), b.gtf.astype(np.float32)):
        raise AssertionError(f"{dst}: MJO forcing round-trip mismatch")


def main():
    jobs_bs = [
        ("ERA5 reanalysis, DJF 1979", "bs.1979.bin", "bs.1979.nc"),
        ("ERA5 reanalysis, JJA",      "bs.jja.bin",  "bs.jja.nc"),
    ]
    for name, b, n in jobs_bs:
        src, dst = os.path.join(HERE, b), os.path.join(HERE, n)
        if not os.path.exists(src):
            print(f"skip {b}: not found")
            continue
        nmonth, n_undef = convert_basic_state(name, src, dst)
        _check_basic_state(src, dst)
        undef = f", {n_undef} undef points" if n_undef else ""
        print(f"wrote {n}  ({nmonth} months x {NLEV_BS} lev{undef})  "
              f"[round-trip OK]")

    src = os.path.join(HERE, "frc.phase1.bin")
    dst = os.path.join(HERE, "frc.phase1.nc")
    if os.path.exists(src):
        ndays, nlev = convert_mjo_forcing(src, dst)
        _check_mjo(src, dst)
        print(f"wrote frc.phase1.nc  ({ndays} days x {nlev} lev)  "
              f"[round-trip OK]")
    else:
        print("skip frc.phase1.bin: not found")


if __name__ == "__main__":
    main()
