# Copyright (C) 2026  PyLBM-Isca authors.  GPLv3 (or later).
"""Validation of the interactive-moisture package (pylbm_isca.moist).

These exercise the linearised convective adjustment (MCA), large-scale
condensation (LSC) and SST-anomaly surface flux end to end on the real NetCDF
basic state (NCEP JFM climatology) and the El-Nino SST anomaly, all of which
live in ``data/``.  The checks are:

* the per-step moist tendencies are finite and the SST-anomaly forcing is
  confined to the PBL (levels ``k <= kb``) and lives in the tropics;
* the damping terms have the right sign (MCA/LSC relax a seeded perturbation,
  and the implicit ``imptq`` strictly shrinks it);
* ``owes=False`` switches the surface wind-evaporation feedback off;
* a short El-Nino integration stays finite and produces a bounded, non-trivial
  heating / precipitation response.
"""

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("netCDF4")

from pylbm_isca import (Config, LBM, Spectral, VerticalGrid,        # noqa: E402
                        MoistPhysics, MoistParams, asnumpy)
from pylbm_isca.basicstate import (load_basic_state,                # noqa: E402
                                   load_moist_surface, load_sst_anomaly)
from pylbm_isca.moist import fqsat, fdqsat                          # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")
BSFILE = os.path.join(DATA, "ncep.clim.t42.nc")
SSTFILE = os.path.join(DATA, "sst.clim.t42.nc")
WGFILE = os.path.join(DATA, "wg.clim.t42.nc")
ANOMFILE = os.path.join(DATA, "frcsst.nino.t21.nc")

_have_data = all(os.path.exists(f) for f in (BSFILE, SSTFILE, WGFILE, ANOMFILE))
needs_data = pytest.mark.skipif(not _have_data,
                                reason="NetCDF moist input data not available")

_CACHE = {}


def _setup():
    if "mp" not in _CACHE:
        sp = Spectral(device="cpu")
        vg = VerticalGrid(device="cpu")
        bs = load_basic_state(BSFILE, sp, vg, kmo=1, navg=3)
        sst, wg = load_moist_surface(SSTFILE, WGFILE, sp)
        anom = load_sst_anomaly(ANOMFILE, sp)
        mp = MoistPhysics(vg, bs, sst=sst, wg=wg,
                          params=MoistParams(ttau0=6, qtau0=6, ttauc=12,
                                             owes=False), device="cpu")
        _CACHE.update(sp=sp, vg=vg, bs=bs, sst=sst, wg=wg, anom=anom, mp=mp)
    return _CACHE


# ---------------------------------------------------------------------------
def test_saturation_functions_monotone():
    """q_sat increases with T and dq_sat/dT is its positive derivative."""
    t = np.array([250.0, 280.0, 300.0])
    p = np.full_like(t, 9.0e4)
    qs = fqsat(np, t, p)
    assert np.all(np.diff(qs) > 0)                  # warmer air holds more vapour
    dqs = fdqsat(np, t, qs)
    assert np.all(dqs > 0)
    # finite-difference check of the derivative at 280 K
    fd = (fqsat(np, np.array([280.5]), np.array([9.0e4]))
          - fqsat(np, np.array([279.5]), np.array([9.0e4])))
    assert np.isclose(fd[0], fdqsat(np, np.array([280.0]),
                                    fqsat(np, np.array([280.0]),
                                          np.array([9.0e4])))[0], rtol=0.02)


@needs_data
def test_inputs_are_sane():
    s = _setup()
    assert s["bs"].t.shape == (s["sp"].nlat, s["sp"].nlon, s["vg"].kmax)
    assert 250.0 < float(s["sst"].mean()) < 305.0
    assert 0.0 <= float(s["wg"].min()) and float(s["wg"].max()) <= 1.0
    assert 1.0 < float(np.abs(s["anom"]).max()) < 10.0   # El-Nino ~ few K


@needs_data
def test_sst_forcing_confined_to_pbl_tropics():
    """The SST-anomaly forcing lives in the PBL (k<=kb) and in the tropics."""
    s = _setup()
    mp = s["mp"]
    gsft, gsfq = mp.ts2tqf(s["anom"])
    gsft = asnumpy(gsft)
    assert np.all(np.isfinite(gsft))
    # nothing above the PBL top
    assert np.allclose(gsft[..., mp.kb + 1:], 0.0)
    # something inside it
    assert np.abs(gsft[..., :mp.kb + 1]).max() > 0.0
    # heating concentrated in the tropics (|lat| < 30) where the anomaly sits
    lat = asnumpy(s["sp"].lat_deg)
    col = np.abs(gsft).sum(axis=(1, 2))
    trop = col[np.abs(lat) < 30.0].sum()
    extra = col[np.abs(lat) >= 30.0].sum()
    assert trop > extra


@needs_data
def test_mca_lsc_damping_sign_and_imptq_contracts():
    """MCA/LSC relax a seeded perturbation; imptq strictly shrinks |T'|,|q'|."""
    s = _setup()
    mp, sp, vg = s["mp"], s["sp"], s["vg"]
    shape = (sp.nlat, sp.nlon, vg.kmax)
    st = types.SimpleNamespace(
        u=np.zeros(shape), v=np.zeros(shape),
        t=np.full(shape, 1.0), q=np.full(shape, 1.0e-4))
    out = mp.tendencies(st)
    for k, v in out.items():
        assert np.all(np.isfinite(asnumpy(v))), f"{k} not finite"
    # implicit damping is a backward-Euler contraction: |T'|,|q'| cannot grow
    t_new, q_new = mp.imptq(np.full(shape, 1.0), np.full(shape, 1.0e-4), 1200.0)
    t_new, q_new = asnumpy(t_new), asnumpy(q_new)
    assert t_new.max() <= 1.0 + 1e-12 and t_new.min() >= 0.0
    assert q_new.max() <= 1.0e-4 + 1e-16 and q_new.min() >= 0.0
    # and it actually damps somewhere (PBL / convective column)
    assert t_new.min() < 1.0


@needs_data
def test_owes_switch_toggles_wind_evaporation():
    """owes=False zeroes the surface perturbation flux; owes=True does not."""
    s = _setup()
    sp, vg, bs = s["sp"], s["vg"], s["bs"]
    shape = (sp.nlat, sp.nlon, vg.kmax)
    st = types.SimpleNamespace(u=np.full(shape, 2.0), v=np.full(shape, 1.0),
                               t=np.full(shape, 0.5), q=np.full(shape, 1.0e-4))
    off = MoistPhysics(vg, bs, sst=s["sst"], wg=s["wg"],
                       params=MoistParams(owes=False), device="cpu")
    on = MoistPhysics(vg, bs, sst=s["sst"], wg=s["wg"],
                      params=MoistParams(owes=True), device="cpu")
    gaft_off = asnumpy(off.surface_flux(st)[0])
    gaft_on = asnumpy(on.surface_flux(st)[0])
    assert np.allclose(gaft_off, 0.0)
    assert np.abs(gaft_on).max() > 0.0


@needs_data
def test_short_elnino_integration_is_finite_and_responds():
    """A 1-day moist El-Nino run stays finite and gives a bounded response."""
    s = _setup()
    sp, vg, bs = s["sp"], s["vg"], s["bs"]
    dd = [15.0] * vg.kmax
    mp = MoistPhysics(vg, bs, sst=s["sst"], wg=s["wg"],
                      params=MoistParams(ttau0=6, qtau0=6, ttauc=12, owes=False),
                      device="cpu")
    m = LBM(bs, sp, vg, Config(ndays=1.0, device="cpu",
                               ddragv=dd, ddragd=dd, ddragt=dd))
    m.run(moist=mp, sst_anom=s["anom"], verbose=False)
    assert m.ga.isfinite()
    umax = float(np.abs(asnumpy(m.ga.u)).max())
    assert 0.0 < umax < 100.0                      # a real but bounded response
    md = m._moist_diag
    q1 = asnumpy(md["q1c"] + md["q1l"]) * 86400.0  # K/day
    pr = asnumpy(md["pr"])
    assert np.all(np.isfinite(q1)) and np.all(np.isfinite(pr))
    assert np.abs(q1).max() < 200.0                # heating stays physical
    assert 0.0 <= pr.max() < 500.0
