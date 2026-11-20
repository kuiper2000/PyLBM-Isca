#!/usr/bin/env python3
# Copyright (C) 2026  PyLBM-Isca authors.  GPLv3 (or later).
"""Command-line driver for PyLBM-Isca (Isca/GFDL-core Linear Baroclinic Model).

Example, reproducing an MJO phase-1 teleconnection experiment on the ERA5
DJF 1979 basic state (inputs archived on Zenodo; see data/README.md)::

    python run_lbm.py \\
        --basic-state data/bs.1979.nc \\
        --mjo-forcing data/frc.phase1.nc \\
        --days 35 --out out/isca.1979.phase1 --device auto

writes ``out/isca.1979.phase1.grd`` + ``.ctl`` (GrADS record order) and,
with ``--netcdf``, ``.nc`` as well.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pylbm_isca import Config, IMAX, JMAX, NMAX, Spectral, LBM, VerticalGrid  # noqa: E402
from pylbm_isca.basicstate import (load_basic_state,               # noqa: E402
                                   load_moist_surface, load_sst_anomaly)
from pylbm_isca.forcing import MJOForcing, SteadyForcing, autodetect  # noqa: E402
from pylbm_isca.moist import MoistParams, MoistPhysics             # noqa: E402
from pylbm_isca.output import GradsWriter, MultiWriter, NetCDFWriter  # noqa: E402
from pylbm_isca.regrid import (detect_grid, regrid_basic_state,    # noqa: E402
                             regrid_mjo_forcing, regrid_steady_forcing)


def build_parser():
    p = argparse.ArgumentParser(
        description="Linear Baroclinic Model on the Isca/GFDL spectral core, "
                    "T42 L20 moist, dynamics-only (GPLv3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("namelist")
    g.add_argument("--namelist", default=None,
                   help="TOML/YAML run-configuration file (the PyLBM-Isca "
                        "namelist); any CLI flag given still overrides it")

    g = p.add_argument_group("input")
    g.add_argument("--basic-state", default=None,
                   help="ERA5 pressure-level basic state, e.g. bs.1979.nc "
                        "(.nc or legacy .bin)")
    g.add_argument("--kmo", type=int, default=1, help="first month (1-based)")
    g.add_argument("--navg", type=int, default=3, help="months to average")
    g.add_argument("--zonal-mean-bs", action="store_true",
                   help="use a zonally uniform basic state")

    g = p.add_argument_group("forcing")
    g.add_argument("--mjo-forcing", default=None,
                   help="time-varying temperature forcing, e.g. "
                        "frc.phase1.nc (.nc or legacy .bin)")
    g.add_argument("--steady-forcing", default=None,
                   help="steady vor/div/T/lnPs forcing, 61 records per case")
    g.add_argument("--case", type=int, default=1, help="steady-forcing case")
    g.add_argument("--forcing-south-first", dest="yflip", action="store_true",
                   default=False, help="flip forcing rows (default: north-first)")
    g.add_argument("--spinup", type=int, default=435,
                   help="calls holding day 1 before the forcing marches")
    g.add_argument("--forcing-scale", type=float, default=1.0)

    g = p.add_argument_group("moist")
    g.add_argument("--moist", action="store_true",
                   help="run the interactive-moisture model (convective "
                        "heating + large-scale condensation + SST-anomaly flux)")
    g.add_argument("--sst-mean", default="data/sst.clim.t42.nc",
                   help="mean-SST climatology NetCDF for the moist basic state")
    g.add_argument("--wg-mean", default="data/wg.clim.t42.nc",
                   help="soil-wetness climatology NetCDF for the moist basic state")
    g.add_argument("--sst-anom", default="data/frcsst.nino.t21.nc",
                   help="prescribed SST-anomaly forcing NetCDF (moist runs)")
    g.add_argument("--mca-ttau0", type=float, default=6.0,
                   help="convective T relaxation time [hour]")
    g.add_argument("--mca-qtau0", type=float, default=6.0,
                   help="convective q relaxation time [hour]")
    g.add_argument("--mca-ttauc", type=float, default=12.0,
                   help="convective correction-term time [hour]")
    g.add_argument("--mca-sigkb", type=float, default=0.9,
                   help="PBL-top sigma (cloud base)")
    g.add_argument("--owes", action="store_true", default=False,
                   help="enable the wind-evaporation feedback")

    g = p.add_argument_group("integration")
    g.add_argument("--days", type=float, default=35.0)
    g.add_argument("--delt", type=float, default=1200.0, help="time step [s]")
    g.add_argument("--inistp", type=int, default=2)
    g.add_argument("--robert", type=float, default=0.04,
                   help="Robert-Asselin filter coefficient")
    g.add_argument("--no-sponge", action="store_true",
                   help="disable the implicit top sponge")
    g.add_argument("--strictly-linear", action="store_true",
                   help="drop the 0.5*(u'^2+v'^2) term Isca keeps")

    g = p.add_argument_group("output")
    g.add_argument("--out", default="out/isca", help="output path stem")
    g.add_argument("--netcdf", action="store_true", help="also write netCDF")
    g.add_argument("--out-interval", type=float, default=1.0, help="days")
    g.add_argument("--instantaneous", action="store_true")
    g.add_argument("--quiet", action="store_true")

    g = p.add_argument_group("backend")
    g.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # The namelist (TOML/YAML) supplies defaults; re-parse so any flag given on
    # the command line still overrides the file -- "command line beats SYSIN".
    spec = None
    if args.namelist:
        from pylbm_isca.namelist import load_namelist
        spec = load_namelist(args.namelist)
        parser.set_defaults(**spec.arg_defaults)
        args = parser.parse_args(argv)

    if not args.basic_state:
        parser.error("a basic state is required "
                     "(--basic-state FILE or [input] basic_state in --namelist)")

    sp = Spectral(device=args.device)
    vg = VerticalGrid(device=args.device)

    _src_cache = {}

    def source_spectral(s_nlon, s_nlat):
        key = (s_nlon, s_nlat)
        if key not in _src_cache:
            _src_cache[key] = Spectral(nlon=s_nlon, nlat=s_nlat,
                                       nmax=(s_nlon - 1) // 3, device=args.device)
        return _src_cache[key]

    if not args.quiet:
        from pylbm_isca import device_name
        print(f"PyLBM-Isca  T{sp.nmax} {sp.nlon}x{sp.nlat} L{vg.kmax}  "
              f"backend={device_name(sp.xp)}")

    # A NetCDF basic state is already on the model grid (no Fortran record
    # markers for detect_grid); binaries may be on a coarser grid and regridded.
    if args.basic_state.endswith(".nc"):
        bs = load_basic_state(args.basic_state, sp, vg, kmo=args.kmo,
                              navg=args.navg, ozm=args.zonal_mean_bs)
    else:
        bs_nlon, bs_nlat = detect_grid(args.basic_state)
        bs_regrid = (bs_nlon, bs_nlat) != (sp.nlon, sp.nlat)
        bs_sp = source_spectral(bs_nlon, bs_nlat) if bs_regrid else sp
        bs = load_basic_state(args.basic_state, bs_sp, vg, kmo=args.kmo,
                              navg=args.navg, ozm=args.zonal_mean_bs)
        if bs_regrid:
            bs = regrid_basic_state(bs, bs_sp, sp)
    if not args.quiet:
        bs.summary()

    # --- moist physics (optional) -------------------------------------
    moist = None
    sst_anom = None
    if args.moist:
        sst_mean, wg_mean = load_moist_surface(args.sst_mean, args.wg_mean, sp)
        moist = MoistPhysics(
            vg, bs, sst=sst_mean, wg=wg_mean, delt=args.delt, device=args.device,
            params=MoistParams(ttau0=args.mca_ttau0, qtau0=args.mca_qtau0,
                               ttauc=args.mca_ttauc, sigkb=args.mca_sigkb,
                               owes=args.owes))
        if args.sst_anom:
            sst_anom = load_sst_anomaly(args.sst_anom, sp)
        if not args.quiet:
            import numpy as _np
            print(f"  moist: SST anomaly |max| "
                  f"{float(_np.abs(sst_anom).max()):.2f} K"
                  if sst_anom is not None else "  moist: no SST anomaly")

    mjo = None
    if args.mjo_forcing:
        if args.mjo_forcing.endswith(".nc"):
            # A NetCDF forcing carries its own grid/length; it is assumed to be
            # on the model grid (no Fortran record markers for detect_grid).
            mjo = MJOForcing(args.mjo_forcing, nlev=vg.kmax,
                             nt=int(round(86400.0 / args.delt)),
                             spinup=args.spinup, yflip=args.yflip,
                             scale=args.forcing_scale)
            if mjo.gtf.shape[1:3] != (sp.nlat, sp.nlon):
                raise SystemExit(
                    f"{args.mjo_forcing}: grid {mjo.gtf.shape[2]}x"
                    f"{mjo.gtf.shape[1]} != model {sp.nlon}x{sp.nlat}; "
                    "regrid the NetCDF forcing to the model grid first")
        else:
            f_nlon, f_nlat = detect_grid(args.mjo_forcing)
            info = autodetect(args.mjo_forcing, nlon=f_nlon, nlat=f_nlat)
            ndays = info.get("mjo_days")
            if ndays is None:
                raise SystemExit(
                    f"{args.mjo_forcing}: not a multiple of {vg.kmax}")
            mjo = MJOForcing(args.mjo_forcing, ndays=ndays, nlon=f_nlon,
                             nlat=f_nlat, nt=int(round(86400.0 / args.delt)),
                             spinup=args.spinup, yflip=args.yflip,
                             scale=args.forcing_scale)
            if (f_nlon, f_nlat) != (sp.nlon, sp.nlat):
                mjo = regrid_mjo_forcing(mjo, source_spectral(f_nlon, f_nlat), sp)

    steady = None
    if args.steady_forcing:
        f_nlon, f_nlat = detect_grid(args.steady_forcing)
        steady = SteadyForcing.from_file(args.steady_forcing, nlev=vg.kmax,
                                         ncase=args.case, yflip=args.yflip,
                                         nlon=f_nlon, nlat=f_nlat)
        if (f_nlon, f_nlat) != (sp.nlon, sp.nlat):
            steady = regrid_steady_forcing(
                steady, source_spectral(f_nlon, f_nlat), sp)

    cfg_kw = dict(spec.config_extra) if spec else {}
    cfg_kw.update(delt=args.delt, inistp=args.inistp, ndays=args.days,
                  robert=args.robert, sponge=not args.no_sponge,
                  strictly_linear=args.strictly_linear,
                  out_interval_days=args.out_interval,
                  time_average=not args.instantaneous, moist=args.moist)
    cfg = Config(**cfg_kw)
    model = LBM(bs, sp, vg, cfg)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    writers = [GradsWriter(args.out + ".grd", sp, vg, moist_vars=args.moist,
                           title="PyLBM-Isca T42L20 (Isca core)")]
    if args.netcdf:
        writers.append(NetCDFWriter(args.out + ".nc", sp, vg,
                                    moist_vars=args.moist))
    writer = MultiWriter(*writers)

    model.run(mjo_forcing=mjo, steady_forcing=steady, writer=writer,
              moist=moist, sst_anom=sst_anom, verbose=not args.quiet)
    if not args.quiet:
        print(f"wrote {args.out}.grd and {args.out}.ctl"
              + (f" and {args.out}.nc" if args.netcdf else ""))
    return model


if __name__ == "__main__":
    main()
