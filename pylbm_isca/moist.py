# Copyright (C) 2026  PyLBM-Isca authors.  GPLv3 (or later).
"""Moist linear physics for PyLBM-Isca — convective heating, large-scale
condensation and SST-anomaly surface fluxes.

This adds an interactive moisture package to the (otherwise dry) Isca-core
linear baroclinic model.  The scheme is the linearised moist physics of the
Watanabe-Kimoto moist LBM (User's Guide §7, "Moist LBM"; Appendix A.4, "linear
convective interaction with dynamics"), i.e. the forced perturbation system

    d_t X + (L − F_i) X = F_e ,

* ``F_i(X)``  — *internal* feedback: the heat source Q1 and moisture sink Q2
  produced by a **linearized convective adjustment** (MCA, after Neelin & Yu
  1994) and a **linear large-scale condensation** (LSC) scheme, both functions
  of the perturbation state X=(…,T',q').
* ``F_e(T_s')`` — *external* forcing: linearized surface bulk fluxes driven by
  the prescribed SST anomaly T_s' (and ground-wetness anomaly), a function of
  the SST anomaly and the basic state only.

The module is deliberately *dynamical-core agnostic*: it only needs a vertical
grid exposing ``sig`` / ``sigm`` / ``dsig`` and the hydrostatic matrix ``ww``,
a basic state with grid fields ``u,v,t,q,ps`` (``ps`` in hPa), and the physical
constants in :mod:`pylbm_isca.constants`.  The geopotential height used by the
convection scheme is therefore built with the *Isca* ``ww`` matrix, so the
moist physics is consistent with the host dynamics.

Scheme map (Betts-Miller linearised convection + Louis surface flux)
--------------------------------------------------------------------
=====================  ==========================================  =================
role                                                               here
=====================  ==========================================  =================
surface bulk-flux setup (Louis, Exner, air-sea ΔT/Δq)              ``MoistPhysics.__init__``
Louis (1979/82) bulk coefficients                                  ``_blkcof``
convective-adjustment reference profiles (SETMCA)                  ``MoistPhysics.__init__``
cloud-top level at zero buoyancy                                   ``_kctop``
LSC weight + dQsat/dT (SETLSC)                                     ``MoistPhysics.__init__``
SST anomaly → steady Fh,Fq (TS2TQF + WG2TQF)                       ``ts2tqf``
Ta,Qa,Va → surface Fh,Fq (+ wind-evap, UV2TQF)                     ``surface_flux``
Ta,Qa → MCA Fh,Fq (TQ2TQF)                                         ``mca``
Ta,Qa → LSC Fh,Fq (TQ2TQF_L)                                       ``lsc``
implicit damping of the above (IMPTQ)                              ``imptq``
column heat source → precipitation (Q2P)                           ``precip``
=====================  ==========================================  =================

Grid / index conventions follow the rest of the package: grid arrays are
``(nlat, nlon, kmax)`` north-first, ``k=0`` is the bottom sigma layer.  The
PBL-top level ``KB`` and the per-column cloud-top ``KTOP`` are stored here as
0-based indices (``kb``, ``ktop``).  Everything is vectorised over the
horizontal with level masks, so the whole scheme runs backend-agnostically on
numpy or cupy.
"""

from dataclasses import dataclass

import numpy as np

from .backend import asnumpy, resolve, to_device
from .constants import (CP, EL, EMELT, EPSV, EPSVT, ES0, FKARM, GRAV, RAIR,
                        RVAP, S2D, TMELT, TQICE)

P00 = 1.0e5   # reference pressure [Pa]


# ---------------------------------------------------------------------------
# Clausius–Clapeyron saturation functions
# ---------------------------------------------------------------------------
def fqsat(xp, t, p):
    """Saturation water-vapour mixing ratio.

    ``FQSAT(T,P) = EPSV*ES0/P * exp( L/Rv (1/Tmelt − 1/T) )`` with the latent
    heat switching to ``L+Lf`` below ``TQICE`` via ``SIGN``.  ``p`` in Pa.
    """
    latent = EL + 0.5 * EMELT * (1.0 - xp.sign(t - TQICE))
    return EPSV * ES0 / p * xp.exp(latent / RVAP * (1.0 / TMELT - 1.0 / t))


def fdqsat(xp, t, qs):
    """d(QSAT)/dT; latent heat switches at ``TMELT``."""
    latent = EL + 0.5 * EMELT * (1.0 - xp.sign(t - TMELT))
    return latent * qs / (RVAP * t * t)


# ---------------------------------------------------------------------------
# Namelist parameters (surface flux / Louis coefs / convection / LSC)
# ---------------------------------------------------------------------------
@dataclass
class MoistParams:
    """Every moist tuning knob, with the moist-LBM reference defaults.

    Groups mirror the reference namelist: ``nmsfcm`` surface flux, ``nmsfcl``
    Louis bulk coefficients, ``nmmca`` convective adjustment, ``nmlsc``
    large-scale condensation.
    """

    # --- surface flux (nmsfcm) -------------------------------------------
    ths: float = 300.0          # basic potential temperature [K]
    urmin: float = 0.5          # min wind for Ri number [m/s]
    urmax: float = 1000.0
    usmin: float = 2.5          # min wind speed [m/s]
    usmax: float = 1000.0
    z0m: float = 1.0e-3         # roughness length, momentum [m]
    z0h: float = 1.0e-3         # roughness length, heat [m]
    z0e: float = 1.0e-3         # roughness length, moisture [m]
    gfdtmx: float = 4.0         # bounds on mean air–sea ΔT [K]
    gfdtmn: float = 2.0
    gfdqmx: float = 1.2e-3      # bounds on mean air–sea Δq [kg/kg]
    gfdqmn: float = 0.0
    wgcrt: float = 0.5          # critical soil wetness
    etlv: float = 2.0           # reference height for T [m]
    eqlv: float = 2.0           # reference height for q [m]
    evlv: float = 10.0          # reference height for wind [m]
    expw: float = 1.0           # exponent for PBL flux redistribution
    owes: bool = True           # wind-evaporation feedback on?
    # --- Louis bulk coefficients (nmsfcl) --------------------------------
    chcons: float = -1.0        # >0 forces constant bulk coef
    newtrl: bool = False        # neutral stratification?
    cmmin: float = 1.0e-5
    chmin: float = 1.0e-5
    cemin: float = 1.0e-5
    cmmax: float = 1.0
    chmax: float = 1.0
    cemax: float = 1.0
    fb: float = 9.4             # Louis b
    fbs: float = 4.7            # Louis b'
    fdm: float = 7.4            # Louis d (momentum)
    fdh: float = 5.3            # Louis d (heat/moisture)
    # --- convective adjustment (nmmca) -----------------------------------
    ocum: bool = True
    ttau0: float = 2.0          # T relaxation time [hour]
    qtau0: float = 2.0          # q relaxation time [hour]
    ttauc: float = 12.0         # correction-term time [hour]
    mca_tunit: str = "HOUR"
    sigkb: float = 0.9          # PBL-top sigma (cloud base)
    thcape: float = 3.0e3       # CAPE threshold for convective efficiency
    ocef: bool = False          # use CAPE-based efficiency?
    cefmx: float = 0.9
    mca_flnd: float = 1.0       # land multiplier on relaxation time
    # --- large-scale condensation (nmlsc) --------------------------------
    olsc: bool = True
    ttaul0: float = 2.0         # T relaxation time [hour]
    qtaul0: float = 2.0         # q relaxation time [hour]
    lsc_tunit: str = "HOUR"
    dqrat: float = 1.0e-3       # humidity scale for the LSC tanh weight
    lsc_flnd: float = 1.0


_TUNIT_SEC = {"SEC": 1.0, "MIN": 60.0, "HOUR": 3600.0, "DAY": 86400.0}


def _to_seconds(value, unit):
    return value * _TUNIT_SEC[unit.strip().upper()]


# ---------------------------------------------------------------------------
# Louis bulk coefficients, host numpy
# ---------------------------------------------------------------------------
def _blkcof(rib, vabs, za, p):
    """CM, CH, CE from the bulk Richardson number (Louis scheme).

    All inputs/outputs are numpy arrays on the horizontal grid.
    """
    if p.chcons > 0.0:
        cm = np.full_like(rib, p.chcons)
        return cm.copy(), cm.copy(), cm.copy()

    if p.newtrl:
        cmx = FKARM ** 2 / np.log(za / p.z0m)
        cm = cmx / np.log(za / p.z0m)
        ch = cmx / np.log(za / p.z0h)
        ce = cmx / np.log(za / p.z0e)
    else:
        cm = FKARM ** 2 / np.log(za / p.z0m) ** 2
        ch = cm.copy()
        ce = cm.copy()
        fcx = p.fb * np.sqrt(za / p.z0m)
        fcm = p.fdm * cm * fcx
        fch = p.fdh * cm * fcx
        fzh = np.log(p.z0m / p.z0h) / np.log(za / p.z0m)
        fze = np.log(p.z0m / p.z0e) / np.log(za / p.z0m)

        def _fmfh(rib0):
            pos = rib >= 0.0
            fm_pos = 1.0 / (1.0 + p.fbs * rib0) ** 2
            rix = np.sqrt(np.abs(rib0))
            fm_neg = 1.0 - p.fb * rib0 / (1.0 + fcm * rix)
            fh_neg = 1.0 - p.fb * rib0 / (1.0 + fch * rix)
            fm = np.where(pos, fm_pos, fm_neg)
            fh = np.where(pos, fm_pos, fh_neg)
            return fm, fh

        t0th = 1.0 / (1.0 + fzh)
        rib0 = rib * t0th
        fm, fh = _fmfh(rib0)
        t0th = 1.0 / (1.0 + fzh / np.sqrt(fm) * fh)
        q0qe = 1.0 / (1.0 + fze / np.sqrt(fm) * fh)
        rib0 = rib * t0th
        fm, fh = _fmfh(rib0)
        cm = cm * fm
        ch = ch * fh * t0th
        ce = ce * fh * q0qe

    cm = np.clip(cm, p.cmmin, p.cmmax)
    ch = np.clip(ch, p.chmin, p.chmax)
    ce = np.clip(ce, p.cemin, p.cemax)
    return cm, ch, ce


def _kctop(gfhs, gfhb, dsig):
    """Cloud-top level (0-based) and CAPE.

    Cloud top is the highest level ``K`` in 1..KMAX-1 whose saturated moist
    static energy ``H*`` first falls to/below the PBL value ``Hb``, raised by
    one level; ``KTOP=1`` (``ktop=0``) if there is no such level.
    """
    kmax = gfhs.shape[-1]
    mask = gfhs[..., :kmax - 1] <= gfhb[..., None]
    kidx = np.arange(kmax - 1)
    ksat = np.where(mask, kidx, -1).max(axis=-1)              # largest K-1
    ktop = np.where(ksat >= 0, ksat + 1, 0)                   # KTOP-1 (0-based)
    kf = np.arange(kmax)
    inrange = kf[None, None, :] <= ktop[..., None]
    contrib = np.maximum(0.0, (gfhb[..., None] - gfhs) * dsig)
    cape = np.where(inrange, contrib, 0.0).sum(axis=-1)
    return ktop, cape


class MoistPhysics:
    """Linearized moist physics (surface flux + MCA + LSC) about a basic state.

    All stratification-dependent coefficients are built once here (the ``SET*``
    setup blocks) and cached as device arrays; the per-step methods
    (:meth:`surface_flux`, :meth:`mca`, :meth:`lsc`, :meth:`ts2tqf`,
    :meth:`imptq`, :meth:`precip`) are whole-array ops over the grid.

    Parameters
    ----------
    vg : VerticalGrid               provides sig / sigm / dsig and ww
    bs : object with u,v,t,q,ps     basic state (grid, ps in hPa)
    sst, wg, zs, idx : (nlat,nlon)  mean SST [K], soil wetness, topography [m],
                                    land/sea index (0=ocean, >0=land)
    params : MoistParams
    delt : float                    model time step [s] (only for reference)
    device : "cpu"|"gpu"|"auto"
    """

    def __init__(self, vg, bs, sst=None, wg=None, zs=None, idx=None,
                 params=None, delt=1200.0, device="auto"):
        self.xp = resolve(device)
        self.vg = vg
        self.p = params or MoistParams()
        p = self.p
        self.delt = float(delt)

        # --- host copies of everything (setup is one-off) -----------------
        gfu = asnumpy(bs.u).astype(np.float64)
        gfv = asnumpy(bs.v).astype(np.float64)
        gft = asnumpy(bs.t).astype(np.float64)
        gfq = asnumpy(bs.q).astype(np.float64)
        gfps = asnumpy(bs.ps).astype(np.float64)                 # hPa
        nlat, nlon, kmax = gft.shape
        self.shape = (nlat, nlon, kmax)
        self.kmax = kmax
        self.gfps = gfps                                          # BS ps [hPa]

        sig = asnumpy(vg.sig).astype(np.float64)
        sigm = asnumpy(vg.sigm).astype(np.float64)
        dsig = asnumpy(vg.dsig).astype(np.float64)
        ww = asnumpy(vg.ww).astype(np.float64)

        z2 = (nlat, nlon)
        sst = np.full(z2, 300.0) if sst is None else asnumpy(sst).astype(np.float64)
        wg = np.ones(z2) if wg is None else asnumpy(wg).astype(np.float64)
        zs = np.zeros(z2) if zs is None else asnumpy(zs).astype(np.float64)
        idx = np.zeros(z2) if idx is None else asnumpy(idx).astype(np.float64)
        self.gfsst, self.gfwg, self.gdzs, self.gidx = sst, wg, zs, idx

        gftv = gft * (1.0 + EPSVT * gfq)                          # virtual temp
        self.gftv = gftv

        # PBL top level KB: highest full level with sig > sigkb
        kb = int(np.where(sig > p.sigkb)[0].max())
        self.kb = kb
        kf = np.arange(kmax)

        # =================================================================
        # SETMCA  (convective-adjustment reference profiles)
        # =================================================================
        akappa = RAIR / CP
        # basic-state geopotential height Z: Z_k = Zs + (Tv @ WW^T)/g, using the
        # host dynamical core's hydrostatic matrix WW.
        gfz = zs[..., None] + (gftv @ ww.T) / GRAV
        gdpm = gfps[..., None] * sigm[:kmax] * 100.0              # half-lev p [Pa]
        qsat = fqsat(np, gft, gdpm)
        gfhs = CP * gft + GRAV * gfz + EL * qsat                  # H*
        dqsat_mca = fdqsat(np, gft, qsat)
        alp = np.minimum(1.0, gfq / qsat)                        # RH
        gfhb = CP * gft[..., 0] + GRAV * gfz[..., 0] + EL * gfq[..., 0]
        ktop, cape = _kctop(gfhs, gfhb, dsig)

        k1 = np.minimum(kmax - 1, ktop + 1)                      # K1-1 (0-based)
        denom = np.maximum(1.0e-2, sigm[kb] - sigm[k1])
        dsigt = np.minimum(1.0, dsig / denom[..., None])

        if p.ocef:
            gceff = np.minimum(p.cefmx, cape / p.thcape)
        else:
            gceff = np.ones(z2)

        # AREFC: correction near cloud top
        sigm_k1 = sigm[ktop + 1]            # SIGM(KTOP+1)
        sigm_k2 = sigm[np.maximum(0, ktop - 1)]   # SIGM(max(1,KTOP-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            dsiga = (sigm[kf][None, None, :] - sigm_k1[..., None]) / (
                (sigm_k2 - sigm_k1)[..., None])
        arefc_range = kf[None, None, :] <= np.minimum(ktop + 1, kmax - 1)[..., None]
        arefc = np.where(arefc_range & (dsiga >= 0.0),
                         np.minimum(1.0, np.sqrt(np.abs(dsiga))), 0.0)

        # reference weighting profiles AREF, BREF
        gam0 = dqsat_mca * EL / CP
        gam = 1.0 / (1.0 + gam0)
        term = gam * dsig / sig                                  # per level
        term_kb = np.where(kf >= kb, term, 0.0)
        csum = np.cumsum(term_kb, axis=-1)                      # sum_{KB..k}
        conv = (kf[None, None, :] >= kb) & (kf[None, None, :] <= ktop[..., None])
        aref = np.where(conv, arefc * gceff[..., None] * gam
                        * np.exp(-akappa * csum), 0.0)
        bref = gam0 * aref
        arefv = (aref * dsigt).sum(axis=-1)
        brefv = (bref * dsigt).sum(axis=-1)

        k1cnt = (np.minimum(kmax - 1, ktop + 1) - kb)           # (K1-KB)
        with np.errstate(divide="ignore", invalid="ignore"):
            dtcw = np.where(conv & (arefv[..., None] != 0.0),
                            aref * dsigt * k1cnt[..., None] / arefv[..., None],
                            1.0)
        tflgc = np.where(conv, 1.0, 0.0)

        # MCA relaxation rates [1/s]; FLND over land
        ttau1 = 1.0 / _to_seconds(p.ttau0, p.mca_tunit)
        qtau1 = 1.0 / _to_seconds(p.qtau0, p.mca_tunit)
        ttac1 = 1.0 / _to_seconds(p.ttauc, p.mca_tunit)
        land = idx > 0.0
        fl = np.where(land, p.mca_flnd, 1.0)
        on = 1.0 if p.ocum else 0.0
        self.ttau = on * ttau1 * fl
        self.qtau = on * qtau1 * fl
        self.ttac = on * ttac1 * fl
        self.qtac = on * ttac1 * fl
        self.aref, self.bref, self.dtcw, self.dsigt = aref, bref, dtcw, dsigt
        self.arefv, self.brefv = arefv, brefv
        self.gceff, self.alp, self.dqsat_mca = gceff, alp, dqsat_mca
        self.tflgc = tflgc
        self.ktop = ktop
        # GETMCAT returns TTAU·GCEFF for the implicit damping
        self.ttauc_imp = self.ttau * gceff
        self.qtauc_imp = self.qtau * gceff

        # =================================================================
        # SETLSC  (large-scale condensation weight + dQsat/dT)
        # =================================================================
        # WTLSC and dQsat/dT use the HALF-level pressure GFPS*SIGM(K), not the
        # full-level pressure.
        gdpm_l = gfps[..., None] * sigm[:kmax] * 100.0
        qsat_l = fqsat(np, gft, gdpm_l)
        dqf = (gfq - qsat_l) / p.dqrat
        self.dqsata = fdqsat(np, gft, qsat_l)
        self.wtlsc = (np.tanh(dqf) + 1.0) / 2.0
        ttaul1 = 1.0 / _to_seconds(p.ttaul0, p.lsc_tunit)
        qtaul1 = 1.0 / _to_seconds(p.qtaul0, p.lsc_tunit)
        fll = np.where(land, p.lsc_flnd, 1.0)
        onl = 1.0 if p.olsc else 0.0
        self.ttaul = onl * ttaul1 * fll
        self.qtaul = onl * qtaul1 * fll
        self.tflgl = np.ones((nlat, nlon, kmax))
        self.qflgl = np.ones((nlat, nlon, kmax))

        # =================================================================
        # SETBULK  (surface bulk-flux coefficients)
        # =================================================================
        gdp_s = gfps * sig[0] * 100.0                            # full-lev p(1)
        gdpm_s = gfps * sigm[0] * 100.0                          # half-lev p(1)
        za = RAIR * gftv[..., 0] / GRAV * (1.0 - sig[0])         # Z1 - Zs
        gdrho = gdpm_s / (RAIR * gftv[..., 0])
        fctv = np.minimum(1.0, p.evlv / za) ** 0.3
        fctq = np.minimum(1.0, p.eqlv / za) ** 0.6
        vfabs = np.sqrt(gfu[..., 0] ** 2 + gfv[..., 0] ** 2) * fctv
        vfabs = np.clip(vfabs, p.usmin, p.usmax)
        self.duabs = gfu[..., 0] * fctv ** 2 / vfabs
        self.dvabs = gfv[..., 0] * fctv ** 2 / vfabs
        exner = (gdp_s / P00) ** akappa
        exnerm = (gdpm_s / P00) ** akappa
        beta = np.minimum(1.0, np.sqrt(wg / p.wgcrt))
        beta = np.where(idx == 0.0, 1.0, beta)
        dbeta = np.where(
            (beta >= 1.0) | (beta < 0.4), 0.0,
            1.0 / np.sqrt(np.maximum(4.0 * wg * p.wgcrt, 1e-30)))
        qsat_sfc = fqsat(np, sst, gdpm_s)
        dqsat_sfc = fdqsat(np, sst, qsat_sfc)
        gfdt = np.clip(sst / exnerm - gft[..., 0] / exner, p.gfdtmn, p.gfdtmx)
        gfdq = np.clip((qsat_sfc - gfq[..., 0]) * fctq, p.gfdqmn, p.gfdqmx)
        # no air–sea T difference over land / partially wet ocean
        gfdt = np.where((idx != 0.0) | (beta < 1.0), 0.0, gfdt)
        rib = (-GRAV / p.ths * gfdt
               / np.clip(vfabs ** 2, p.urmin, p.urmax) * za)

        # PBL flux redistribution weight + layer thickness DELZ
        sigw = (1.0 / (1.0 - sig)) ** p.expw
        sigwt = sigw[:kb + 1].sum()
        sigw = sigw / sigwt
        with np.errstate(divide="ignore", invalid="ignore"):
            delz = gfps[..., None] * 100.0 * dsig / GRAV / sigw
            delz_inv = np.where(kf[None, None, :] <= kb, 1.0 / delz, 0.0)
        self.delz_inv = delz_inv

        cmv, chv, cev = _blkcof(rib, vfabs, za, p)
        self.cmvf = cmv * vfabs * fctv
        self.chvf = chv * vfabs
        self.cevf = cev * vfabs
        self.chv, self.cev = chv, cev
        self.gadt = self.chvf * exnerm / exner
        self.gadq = self.cevf
        self.gfwq = self.cevf * gfdq * dbeta
        self.gfdt, self.gfdq, self.beta = gfdt, gfdq, beta
        self.dqsat_sfc = dqsat_sfc
        self.gdrho = gdrho

        # convenience level-coordinate masks (0-based)
        self._conv_mask = conv
        self._kb_mask = (kf[None, None, :] <= kb)

        # move all per-step arrays onto the device
        self._to_device()

    # ------------------------------------------------------------------
    def _to_device(self):
        names = ("delz_inv", "chvf", "cevf", "chv", "cev", "cmvf",
                 "duabs", "dvabs", "gadt", "gadq", "gfwq",
                 "gfdt", "gfdq", "beta", "dqsat_sfc", "gdrho",
                 "ttau", "qtau", "ttac", "qtac", "aref", "bref", "dtcw",
                 "dsigt", "arefv", "brefv", "gceff", "alp", "dqsat_mca",
                 "tflgc", "ttauc_imp", "qtauc_imp",
                 "dqsata", "wtlsc", "ttaul", "qtaul", "tflgl", "qflgl",
                 "gfps", "_conv_mask", "_kb_mask")
        for nm in names:
            setattr(self, nm, to_device(np.asarray(getattr(self, nm)), self.xp))

    # ================================================================
    # External forcing F_e : SST / wetness anomaly  (TS2TQF + WG2TQF)
    # ================================================================
    def ts2tqf(self, sst_anom, wg_anom=None):
        """Steady heat/moisture forcing from the SST (and wetness) anomaly.

        ``GSFT = CHVF·T_s'/DELZ`` and ``GSFQ = CEVF·dQsat·T_s'·beta/DELZ``,
        plus the wetness term ``GSFQ += GFWQ·W'/DELZ``.  Spread over the PBL
        (levels ``k<=KB``) by ``DELZ``.  Returns ``(gsft, gsfq)``.
        """
        xp = self.xp
        s = xp.asarray(sst_anom)[..., None]
        gsft = self.chvf[..., None] * s * self.delz_inv
        gsfq = (self.cevf[..., None] * self.dqsat_sfc[..., None] * s
                * self.beta[..., None] * self.delz_inv)
        if wg_anom is not None:
            w = xp.asarray(wg_anom)[..., None]
            gsfq = gsfq + self.gfwq[..., None] * w * self.delz_inv
        return gsft, gsfq

    # ================================================================
    # Surface flux of the perturbation (UV2TQF)
    # ================================================================
    def surface_flux(self, st):
        """Surface-flux tendencies of (T', q') and the wind-evaporation feedback.

        Returns ``(gaft, gafq, gawt, gawq)`` where ``gaw*`` is the wind-driven
        (evaporation-feedback) production and ``gaf*`` the linear damping of the
        lowest-level perturbation.  Both are redistributed over the PBL.
        """
        xp = self.xp
        u0, v0 = st.u[..., 0:1], st.v[..., 0:1]
        t0, q0 = st.t[..., 0:1], st.q[..., 0:1]
        wspd = self.duabs[..., None] * u0 + self.dvabs[..., None] * v0
        gawt = self.chv[..., None] * wspd * self.gfdt[..., None] * self.delz_inv
        gawq = (self.cev[..., None] * wspd * self.gfdq[..., None]
                * self.beta[..., None] * self.delz_inv)
        gaft = -self.gadt[..., None] * t0 * self.delz_inv
        gafq = -self.gadq[..., None] * q0 * self.beta[..., None] * self.delz_inv
        if not self.p.owes:
            gawt = xp.zeros_like(gawt)
            gawq = xp.zeros_like(gawq)
            gaft = xp.zeros_like(gaft)
            gafq = xp.zeros_like(gafq)
        return gaft, gafq, gawt, gawq

    # ================================================================
    # Convective adjustment (TQ2TQF) — adds to gaw*/gaf*
    # ================================================================
    def mca(self, st, gaft, gafq, gawt, gawq):
        """Add the linearized convective-adjustment Q1/Q2 (Neelin & Yu 1994).

        Returns the updated ``(gaft, gafq, gawt, gawq)``.  The production terms
        ``DQ1 = ttau·AREF·Hb/Cp`` (heat) and ``DQ2 = qtau·BREF·Hb/L`` (moisture)
        are driven by the PBL moist-static-energy anomaly ``Hb = Cp·T'_1 +
        L·q'_1``; ``Q1C`` is the energy-conserving correction and ``DTD/DQD``
        the local relaxation.
        """
        m = self._conv_mask
        # vertical averages over the convective column KB..KTOP; DSIGT is already
        # zero-weighted outside [KB,KTOP] via the mask.
        wcol = m * self.dsigt
        tvavg = (st.t * wcol).sum(axis=-1)
        qvavg = (st.q * wcol).sum(axis=-1)
        hb = CP * st.t[..., 0] + EL * st.q[..., 0]
        delh = (self.arefv + self.brefv) * hb - CP * tvavg - EL * qvavg
        dtc = delh / CP
        hbL = hb[..., None]
        dq1 = self.ttau[..., None] * self.aref * hbL / CP
        dq2 = self.qtau[..., None] * self.bref * hbL / EL
        q1c = (-self.ttac[..., None] * dtc[..., None] * self.dtcw
               * self.gceff[..., None])
        dtd = -self.ttau[..., None] * self.gceff[..., None] * st.t
        dqd = -self.qtau[..., None] * self.gceff[..., None] * st.q
        gawt = gawt + m * (dq1 + q1c)
        gawq = gawq + m * dq2
        gaft = gaft + m * dtd
        gafq = gafq + m * dqd
        return gaft, gafq, gawt, gawq

    # ================================================================
    # Large-scale condensation (TQ2TQF_L)
    # ================================================================
    def lsc(self, st):
        """Linear large-scale condensation Q1L/Q2L.

        ``DQ1 = ttau·W·q'·L/Cp`` (heating from moisture excess), ``DQ2 =
        qtau·W·T'·dQsat`` (moistening), with ``W = (tanh((q_bar−q_sat)/dqrat)+1)/2``
        the basic-state saturation weight; ``DTD/DQD`` are the relaxation terms.
        Returns ``(gaftl, gafql, gawtl, gawql)``.
        """
        w = self.wtlsc
        dq1 = self.ttaul[..., None] * w * st.q * EL / CP
        dq2 = self.qtaul[..., None] * w * st.t * self.dqsata
        dtd = -self.ttaul[..., None] * w * st.t * self.dqsata * EL / CP
        dqd = -self.qtaul[..., None] * w * st.q
        return dtd, dqd, dq1, dq2

    # ================================================================
    # Implicit damping of the surface / MCA / LSC terms (IMPTQ)
    # ================================================================
    def imptq(self, gat, gaq, delt):
        """Backward-Euler damping of T'/q' by the moist relaxation terms.

        Applied in grid space to the *updated* state after the leapfrog.
        Modifies and returns ``(gat, gaq)``.
        """
        dt2 = 2.0 * delt
        # surface bulk damping, k<=KB
        gat = gat / (1.0 + dt2 * self.gadt[..., None] * self.delz_inv)
        gaq = gaq / (1.0 + dt2 * self.gadq[..., None] * self.beta[..., None]
                     * self.delz_inv)
        # MCA damping, KB<=k<=KTOP via TFLGC
        gat = gat / (1.0 + dt2 * self.ttauc_imp[..., None] * self.tflgc)
        gaq = gaq / (1.0 + dt2 * self.qtauc_imp[..., None] * self.tflgc)
        # LSC damping, all levels
        gat = gat / (1.0 + dt2 * self.ttaul[..., None] * self.wtlsc
                     * self.dqsata * self.tflgl * EL / CP)
        gaq = gaq / (1.0 + dt2 * self.qtaul[..., None] * self.wtlsc * self.qflgl)
        return gat, gaq

    # ================================================================
    # Precipitation diagnostic (Q2P)
    # ================================================================
    def precip(self, gqc, gql):
        """Column-integrated heat source → precipitation [mm/day].

        ``PR = Cp·Ps·∫(Q1c+Q1l) dσ · 86400·100 / (L·g)``; ``gqc`` is the total
        convective heat source (GAFT+GAWT) and ``gql`` the LSC heat source
        (GAFTL+GAWTL), both excluding the SST forcing.  ``Ps`` is the
        basic-state surface pressure [hPa].
        """
        dsig = self.vg.dsig
        qint = ((gqc + gql) * dsig).sum(axis=-1)
        return CP * self.gfps * qint * S2D * 1.0e2 / (EL * GRAV)

    # ================================================================
    # Convenience: the full per-step moist tendency bundle
    # ================================================================
    def tendencies(self, st, sst_anom=None, wg_anom=None):
        """Assemble all moist grid-space tendencies for one step.

        Returns a dict with the explicit forcings that feed the dynamics
        (``frc_t = gsft+gawt+gawtl``, ``frc_q = gsfq+gawq+gawql``) and the
        diagnostic heat/moisture sources Q1C/Q2C/Q1L/Q2L and precipitation.
        """
        xp = self.xp
        if sst_anom is None:
            gsft = xp.zeros(self.shape)
            gsfq = xp.zeros(self.shape)
        else:
            gsft, gsfq = self.ts2tqf(sst_anom, wg_anom)
        gaft, gafq, gawt, gawq = self.surface_flux(st)
        gaft, gafq, gawt, gawq = self.mca(st, gaft, gafq, gawt, gawq)
        gaftl, gafql, gawtl, gawql = self.lsc(st)

        frc_t = gsft + gawt + gawtl
        frc_q = gsfq + gawq + gawql
        # precipitation uses conv+LSC heat sources, excluding SST
        pr = self.precip(gaft + gawt, gaftl + gawtl)
        return {
            "frc_t": frc_t, "frc_q": frc_q,
            "q1c": gaft + gawt + gsft, "q2c": gafq + gawq + gsfq,
            "q1l": gaftl + gawtl, "q2l": gafql + gawql,
            "pr": pr, "gsft": gsft, "gsfq": gsfq,
            "gaft": gaft, "gafq": gafq, "gawt": gawt, "gawq": gawq,
            "gaftl": gaftl, "gafql": gafql, "gawtl": gawtl, "gawql": gawql,
        }
