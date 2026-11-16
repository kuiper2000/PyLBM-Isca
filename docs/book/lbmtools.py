# Copyright (C) 2026  PyLBM-Isca authors.  GPLv3 (or later).
"""Shared helpers for the PyLBM-Isca Users' Guide.

The published package ships one thing the Watanabe--Kimoto guide builds whole
chapters on: the **linearised tendency** ``L(x_bar) x'``.  Everything the guide
does beyond a plain forward integration -- solving the steady response by matrix
inversion, the iterative (accelerated) solver, the storm-track variance, the
eigen/singular-mode decomposition and the barotropic reduction -- is a thin
wrapper around that one operator.  This module provides that wrapper so the
notebooks stay short and every chapter runs on *the same* operator.

What is here
------------
``baroclinic_jet`` / ``resting_state``
    Self-contained analytic basic states (no Zenodo download), at any
    truncation, so the book builds offline.
``TangentLinear``
    A matrix-free complex linear operator ``A`` with
    ``dx/dt = A x (+ forcing)``.  ``A x`` is the strictly-linear Isca tendency
    (``Dynamics(strictly_linear=True)``) plus the linear dissipation ``D`` (the
    hyperdiffusion / sponge / Rayleigh-Newtonian coefficients of
    :class:`pylbm_isca.integrator.Damping`).  It packs the five spectral
    perturbation fields ``(zeta', D', T', q', pi')`` -- over the triangular
    coefficients only -- into one complex vector, so it plugs straight into
    ``scipy.sparse.linalg`` (``eigs``, ``svds``, ``gmres``).
``forcing_vector``
    The spectral source vector ``f`` for a grid-space heating / vorticity /
    mass source, obtained by evaluating the tendency of the *zero* state with
    the forcing switched on.
``propagator``
    The finite-time tangent-linear propagator ``M(tau) = x(tau) <- x(0)``,
    integrated with the model's own semi-implicit leapfrog (used for the
    singular-vector / optimal-growth analysis).
"""
from types import SimpleNamespace

import numpy as np

from pylbm_isca import Config, LBM, Spectral, VerticalGrid
from pylbm_isca.backend import asnumpy
from pylbm_isca.constants import OMEGA as OMEGA_
from pylbm_isca.dynamics import BasicStateTerms, Dynamics
from pylbm_isca.integrator import (Damping, Integrator, SemiImplicit,
                                   VerticalDiffusion)
from pylbm_isca.state import State

try:                                    # SciPy is a hard dependency of the guide
    from scipy.sparse.linalg import LinearOperator
except Exception:                       # pragma: no cover
    LinearOperator = None


# ---------------------------------------------------------------------------
# Analytic basic states (self-contained; no external files)
# ---------------------------------------------------------------------------
def resting_state(sp, vg, t0=250.0, ps=1.0e5):
    """A resting, isothermal basic state (the Gill / Matsuno background)."""
    shape = (sp.nlat, sp.nlon, vg.kmax)
    return SimpleNamespace(
        u=np.zeros(shape), v=np.zeros(shape),
        t=np.full(shape, t0), q=np.zeros(shape),
        ps=np.full((sp.nlat, sp.nlon), ps))


def baroclinic_jet(sp, vg, umax=40.0, lat0=45.0, width=15.0,
                   t0=255.0, dtdy=40.0, dtdsig=8.0, q0=0.0, ps=1.0e5):
    """A mid-latitude baroclinic jet (vertical + horizontal shear).

    A Gaussian zonal jet centred at ``lat0`` whose amplitude grows upward
    (``sqrt(sigma_full)``-like), over a pole-to-equator temperature contrast and
    a weak static-stability gradient.  It is *not* in exact thermal-wind
    balance -- it does not need to be: it is only the anchor the tendency is
    linearised about -- but it exercises every baroclinic term (vertical
    advection, the PGF and the energy conversion), so its operator has genuine
    growing modes.
    """
    nlat, nlon, k = sp.nlat, sp.nlon, vg.kmax
    lat = np.arcsin(asnumpy(sp.mu))[:, None, None]       # radians
    sig = asnumpy(vg.sig)[None, None, :]
    jet = umax * np.exp(-((lat - np.deg2rad(lat0)) / np.deg2rad(width)) ** 2)
    u = (jet * np.sqrt(np.clip(sig, 0, None))) * np.ones((nlat, nlon, k))
    v = np.zeros((nlat, nlon, k))
    t = (t0 - dtdy * np.sin(lat) ** 2 + dtdsig * sig) * np.ones((nlat, nlon, k))
    q = q0 * np.exp(-((lat) / np.deg2rad(25)) ** 2) * np.ones((nlat, nlon, k))
    return SimpleNamespace(u=u, v=v, t=t, q=q,
                           ps=np.full((nlat, nlon), ps))


def solid_body(sp, vg, u0=25.0, t0=250.0, ps=1.0e5):
    """Solid-body super-rotation ``u = u0 cos(phi)`` (Sect. 4.2 of the guide)."""
    shape = (sp.nlat, sp.nlon, vg.kmax)
    coslat = np.sqrt(1.0 - asnumpy(sp.mu) ** 2)
    u = np.broadcast_to(u0 * coslat[:, None, None], shape).copy()
    return SimpleNamespace(u=u, v=np.zeros(shape),
                           t=np.full(shape, t0), q=np.zeros(shape),
                           ps=np.full((sp.nlat, sp.nlon), ps))


def zonal_mean(bs):
    """Symmetrise a basic state in longitude: ``bar{x}(phi, sigma)``.

    A zonally-symmetric basic state makes the linear operator **block-diagonal in
    zonal wavenumber** ``m``: a perturbation of a single ``m`` stays at that
    ``m``.  That is what lets the modal and barotropic chapters work with small
    dense per-``m`` matrices instead of the full operator.
    """
    def zm(a):
        a = np.asarray(a, dtype=np.float64)
        return np.broadcast_to(a.mean(axis=1, keepdims=True), a.shape).copy()
    return SimpleNamespace(u=zm(bs.u), v=zm(bs.v), t=zm(bs.t),
                           q=zm(bs.q), ps=zm(bs.ps))


# ---------------------------------------------------------------------------
# The matrix-free tangent-linear operator
# ---------------------------------------------------------------------------
def no_damping_config(base=None):
    """A Config with the dissipation switched (almost) off -- for exposing the
    bare dynamical instability in the modal chapter."""
    c = Config(**(base.__dict__ if base else {}))
    c.ddragv = c.ddragd = c.ddragt = c.ddragq = None     # no Rayleigh/Newtonian
    c.sponge = False                                     # no top sponge
    c.ray0 = 0.0
    c.hdif_tefold = 1.0e9                                 # ~ no hyperdiffusion
    c.vdifv = c.vdifd = c.vdift = c.vdifq = None          # no vertical diffusion
    return c


class TangentLinear:
    """Matrix-free linear operator ``A`` with ``dx/dt = A x`` about ``bs``.

    ``A`` is the model's **own** complete linear tendency: one explicit
    (strictly-linear) :class:`~pylbm_isca.dynamics.Dynamics` evaluation followed
    by one semi-implicit :meth:`~pylbm_isca.integrator.Integrator.step`, read out
    as ``A x = (step(dt, x, dyn(x)) - x) / (2 dt)``.  Doing it through the step
    is what makes ``A`` *complete*: the Isca core deliberately hands the
    gravity-wave coupling (geopotential<->divergence<->compression) and the
    dissipation to the integrator, so the explicit tendency alone is **not** the
    full operator.  Because the step uses the model's time step ``dt = cfg.delt``,
    the fixed point of ``A`` is *exactly* the steady state the model's time
    integration converges to.

    Parameters
    ----------
    sp, vg : Spectral, VerticalGrid
    bs : basic state (any object with grid-space ``u, v, t, q, ps``)
    cfg : Config or None
        The integration knobs -- the time step ``delt`` and every dissipation
        profile (``ddrag*``, ``ray0``, ``hdif_*``, ``sponge``, ``vdif*``).
        Pass :func:`no_damping_config` for the (nearly) inviscid operator.
    mvals : int, sequence or None
        Restrict the packed state to these zonal wavenumbers (exact only for a
        zonal-mean basic state; see :func:`zonal_mean`).
    fields : tuple of str
        Which prognostic fields are *active* in the state vector.  The default
        carries all five; the barotropic chapter passes ``("vor",)``.
    """

    FIELDS = ("vor", "div", "t", "q", "pi")

    def __init__(self, sp, vg, bs, cfg=None, damping=True,
                 fields=("vor", "div", "t", "q", "pi"), mvals=None, dt=None):
        self.sp, self.vg = sp, vg
        self.xp = sp.xp
        self.cfg = cfg or (Config() if damping else no_damping_config())
        self.fields = tuple(fields)
        self.ps_bar = self.xp.asarray(np.asarray(bs.ps, dtype=np.float64))
        # dt controls the read-out.  dt = cfg.delt (default) makes the operator's
        # fixed point *exactly* the model's steady state -- use it for the steady
        # response.  A small dt makes (step - I)/2dt -> the continuous tendency L
        # -- use it for eigen/singular-mode analysis, where the leapfrog
        # distortion of the fast gravity waves would otherwise dominate.
        self.dt = float(self.cfg.delt if dt is None else dt)

        self.bst = BasicStateTerms(sp, vg, bs)
        self.dyn = Dynamics(sp, vg, self.bst, strictly_linear=True)

        # The model's own integrator: dissipation + semi-implicit gravity-wave
        # solve + vertical diffusion, built from the Config exactly as LBM does.
        c = self.cfg
        self.damp = Damping(
            sp, vg, order=c.hdif_order, tefold=c.hdif_tefold,
            tunit_hdif=c.hdif_tunit, ddragv=c.ddragv, ddragd=c.ddragd,
            ddragt=c.ddragt, ddragq=c.ddragq, tunit_damp=c.damp_tunit,
            ray0=c.ray0, sponge=c.sponge)
        self.semi = SemiImplicit(sp, vg, self.damp)
        self.vdiff = VerticalDiffusion(vg, c.vdifv, c.vdifd, c.vdift, c.vdifq)
        self.integ = Integrator(sp, vg, self.damp, self.semi, self.vdiff)

        # Triangular-coefficient index (n >= m).  ``mvals`` optionally restricts
        # to a subset of zonal wavenumbers -- exact only for a zonal-mean basic
        # state, where the operator is block-diagonal in m (see ``zonal_mean``).
        nmax, kmax = sp.nmax, vg.kmax
        self.kmax = kmax
        mm, nn = np.meshgrid(np.arange(nmax + 1), np.arange(nmax + 1),
                             indexing="ij")
        self.mask = nn >= mm                              # (m, n) bool
        if mvals is not None:
            self.mvals = tuple(int(m) for m in np.atleast_1d(mvals))
            keep = np.zeros_like(self.mask)
            keep[list(self.mvals), :] = True
            self.mask = self.mask & keep
        else:
            self.mvals = tuple(range(nmax + 1))
        self.ncoef = int(self.mask.sum())
        self._mi, self._ni = np.nonzero(self.mask)

        # Per-field block layout in the packed complex vector.
        self._blocks = []                                 # (name, start, size, is3d)
        off = 0
        for name in self.fields:
            is3d = name != "pi"
            size = self.ncoef * (kmax if is3d else 1)
            self._blocks.append((name, off, size, is3d))
            off += size
        self.size = off                                   # complex length

    # -- packing -------------------------------------------------------------
    def pack(self, spec):
        """Dict of spectral fields -> flat complex vector (host numpy)."""
        out = np.empty(self.size, dtype=np.complex128)
        for name, off, size, is3d in self._blocks:
            arr = asnumpy(spec[name])
            if is3d:
                out[off:off + size] = arr[self._mi, self._ni, :].ravel()
            else:
                out[off:off + size] = arr[self._mi, self._ni]
        return out

    def unpack(self, vec):
        """Flat complex vector -> dict of full (m, n[, k]) device spectral fields."""
        xp, nmax, kmax = self.xp, self.sp.nmax, self.kmax
        spec = {}
        for name in self.FIELDS:
            spec[name] = xp.zeros((nmax + 1, nmax + 1) + ((kmax,) if name != "pi"
                                  else ()), dtype=xp.complex128)
        vec = np.asarray(vec, dtype=np.complex128)
        for name, off, size, is3d in self._blocks:
            block = vec[off:off + size]
            if is3d:
                tmp = np.zeros((nmax + 1, nmax + 1, kmax), dtype=np.complex128)
                tmp[self._mi, self._ni, :] = block.reshape(self.ncoef, kmax)
            else:
                tmp = np.zeros((nmax + 1, nmax + 1), dtype=np.complex128)
                tmp[self._mi, self._ni] = block
            spec[name] = xp.asarray(tmp)
        return spec

    # -- the operator --------------------------------------------------------
    def _step_tendency(self, spec, frc=None):
        """Complete linear tendency ``(step(dt, spec, dyn(spec)+frc) - spec)/2dt``.

        ``spec`` is used both as the state the explicit tendency is evaluated at
        *and* as the leapfrog ``prev`` level, so for a time-independent state the
        read-out is the model's complete discrete tendency (dynamics + implicit
        gravity waves + dissipation).
        """
        st = State.zeros(self.ps_bar, self.kmax, xp=self.xp)
        st.load_spectral(self.sp, spec)
        wv, wd, wt, wp, wq = self.dyn.tendencies(st, **(frc or {}))
        new = self.integ.step(self.dt, spec, (wv, wd, wt, wp, wq))
        inv = 1.0 / (2.0 * self.dt)
        return {k: (new[k] - spec[k]) * inv for k in self.FIELDS}

    def apply_spec(self, spec):
        """``A`` applied in spectral space: returns a dict of tendencies."""
        return self._step_tendency(spec)

    def matvec(self, vec):
        return self.pack(self.apply_spec(self.unpack(vec)))

    def aslinearoperator(self):
        if LinearOperator is None:                        # pragma: no cover
            raise RuntimeError("scipy is required for aslinearoperator()")
        return LinearOperator((self.size, self.size), matvec=self.matvec,
                              dtype=np.complex128)

    def energy_metric(self, tref=250.0):
        """Diagonal energy metric ``W`` (real, length ``size``).

        ``x* W x`` approximates the total perturbation energy: rotational +
        divergent kinetic energy ``1/2 * a^2/(n(n+1)) * (|zeta|^2+|D|^2)`` and
        available potential energy ``(c_p/2 T_ref) |T|^2``, each mass-weighted by
        the layer thickness ``dsig_k`` and doubled for ``m>0`` (the conjugate
        negative-``m`` coefficient).  Use it to turn the raw packed 2-norm into a
        physical norm for the singular-vector / optimal-growth analysis, where the
        bare coefficient norm is meaningless (it mixes ``zeta ~ 1e-5`` with
        ``T ~ 1``).
        """
        from pylbm_isca.constants import CP, RAIR
        a2 = self.sp.a ** 2
        nn1 = np.where(self._ni > 0, self._ni * (self._ni + 1.0), np.inf)
        wm = np.where(self._mi > 0, 2.0, 1.0)             # conjugate doubling
        dsig = asnumpy(self.vg.dsig)
        W = np.empty(self.size)
        for name, off, size, is3d in self._blocks:
            if is3d:
                base = np.repeat((wm)[:, None], self.kmax, axis=1)   # (ncoef,k)
                dk = dsig[None, :]
                if name in ("vor", "div"):
                    w = 0.5 * a2 / nn1[:, None] * dk * base
                elif name == "t":
                    w = 0.5 * CP / tref * dk * base
                else:                                     # q: negligible energy
                    w = 1e-12 * np.ones((self.ncoef, self.kmax)) * base
                W[off:off + size] = w.ravel()
            else:                                         # pi (surface pressure)
                W[off:off + size] = 0.5 * RAIR * tref * 1e-4 * wm
        return np.maximum(W, 1e-30)

    def dense(self):
        """Build the explicit dense matrix ``A`` (size x size, complex).

        One matvec per column.  Only tractable when the state vector is small --
        i.e. when restricted to one or a few zonal wavenumbers via ``mvals`` on a
        zonal-mean basic state.  This is the matrix the direct steady-response
        solve inverts and the modal decomposition diagonalises.
        """
        A = np.empty((self.size, self.size), dtype=np.complex128)
        e = np.zeros(self.size, dtype=np.complex128)
        for j in range(self.size):
            e[j] = 1.0
            A[:, j] = self.matvec(e)
            e[j] = 0.0
        return A

    # -- forcing -------------------------------------------------------------
    def _zero_spec(self):
        xp, nmax, kmax = self.xp, self.sp.nmax, self.kmax
        return {n: xp.zeros((nmax + 1, nmax + 1) + ((kmax,) if n != "pi" else ()),
                            dtype=xp.complex128) for n in self.FIELDS}

    def forcing_vector(self, frc_t=None, frc_vor=None, frc_div=None,
                       frc_ps=None, frc_q=None):
        """Spectral source vector ``f`` for grid-space forcing fields.

        Defined so the steady state solves ``A x = -f``: it is the complete
        tendency the forcing alone drives through the semi-implicit step, so the
        matrix-inversion / GMRES solution lands on the same fixed point the time
        integration converges to.
        """
        kw = {}
        for key, val in (("frc_t", frc_t), ("frc_vor", frc_vor),
                         ("frc_div", frc_div), ("frc_ps", frc_ps),
                         ("frc_q", frc_q)):
            if val is not None:
                kw[key] = self.xp.asarray(np.asarray(val, dtype=np.float64))
        return self.pack(self._step_tendency(self._zero_spec(), frc=kw))

    # -- diagnostics ---------------------------------------------------------
    def to_state(self, vec):
        """Unpack a (complex) mode/vector into a grid-space State for plotting."""
        st = State.zeros(self.ps_bar, self.kmax, xp=self.xp)
        st.load_spectral(self.sp, self.unpack(vec))
        return st


# ---------------------------------------------------------------------------
# Barotropic (single-level, non-divergent) vorticity operator
# ---------------------------------------------------------------------------
class Barotropic:
    """Linear barotropic vorticity operator about a zonal flow ``ubar(phi)``.

    One layer, non-divergent, no gravity waves -- so, unlike the full LBM
    operator, it has a clean enstrophy/energy norm.  That makes it the tractable
    home for the optimal-growth (singular-vector, {doc}`08-barotropic.ipynb`) and
    stochastic storm-track (Lyapunov, {doc}`04-storm-track.ipynb`) analyses that
    the primitive-equation operator's fast gravity waves make delicate.

        d zeta'/dt = -div[ v'(zeta_bar + f) + ubar zeta' ] - r zeta' - nu4 grad^4 zeta'
    """

    def __init__(self, sp, ubar_g, drag_days=15.0, nu4=2.0e16):
        self.sp = sp
        self.ubar = np.asarray(ubar_g, dtype=np.float64)
        zbar_s = sp.uv_to_vordiv(self.ubar, np.zeros_like(self.ubar))[0]
        self.absz = (asnumpy(sp.wave_to_grid(zbar_s))
                     + (2.0 * OMEGA_ * asnumpy(sp.mu))[:, None])
        self.r = 1.0 / (drag_days * 86400.0)
        self.nu4 = nu4

    def tend(self, z_s):
        """Spectral vorticity tendency for a spectral vorticity field ``z_s``."""
        sp = self.sp
        psi_s = sp.inverse_laplacian(z_s)
        u, v = sp.psichi_to_uv(psi_s, sp.xp.zeros_like(psi_s))
        zg = asnumpy(sp.wave_to_grid(z_s))
        Fx = asnumpy(u) * self.absz + self.ubar * zg
        Fy = asnumpy(v) * self.absz
        t = sp.neg_divergence(Fx, Fy)
        t = t - self.r * z_s - self.nu4 * sp.laplacian(sp.laplacian(z_s))
        return t

    def block(self, m):
        """Dense operator for a single zonal wavenumber ``m`` (vorticity coeffs).

        Returns ``(A, ns)`` with ``ns = m..N`` the total wavenumbers and ``A`` the
        ``len(ns) x len(ns)`` complex matrix.
        """
        N = self.sp.nmax
        ns = np.arange(m, N + 1)
        A = np.empty((len(ns), len(ns)), dtype=np.complex128)
        z = np.zeros((N + 1, N + 1), dtype=np.complex128)
        for i, n in enumerate(ns):
            z[m, n] = 1.0
            A[:, i] = asnumpy(self.tend(self.sp.xp.asarray(z)))[m, ns]
            z[m, n] = 0.0
        return A, ns

    def psi_grid(self, z_s):
        """Grid streamfunction of a spectral vorticity field."""
        return asnumpy(self.sp.wave_to_grid(self.sp.inverse_laplacian(z_s))).real


# ---------------------------------------------------------------------------
# Finite-time propagator (singular-vector / optimal-growth analysis)
# ---------------------------------------------------------------------------
def propagator(tl, days, cfg=None, device=None):
    """Return a matrix-free ``LinearOperator`` for ``M(tau) = x(tau) <- x(0)``.

    ``tau = days`` is integrated with the model's own semi-implicit leapfrog
    (full dissipation), so the propagator is the genuine tangent-linear
    evolution, not a one-step Euler approximation.  One matvec = one short
    integration; keep ``days`` and the mode count small in a notebook.
    """
    if LinearOperator is None:                            # pragma: no cover
        raise RuntimeError("scipy is required for propagator()")
    base_cfg = cfg or tl.cfg

    def _matvec(vec):
        spec = tl.unpack(vec)
        st = State.zeros(tl.ps_bar, tl.kmax, xp=tl.xp)
        st.load_spectral(tl.sp, spec)
        c = Config(**{**base_cfg.__dict__, "ndays": float(days)})
        # rebuild a model on the same basic state and run with no forcing
        bs = SimpleNamespace(u=asnumpy(tl.bst.u), v=asnumpy(tl.bst.v),
                             t=asnumpy(tl.bst.t), q=asnumpy(tl.bst.q),
                             ps=asnumpy(tl.ps_bar))
        model = LBM(bs, tl.sp, tl.vg, c)
        model.set_initial_perturbation(u=st.u, v=st.v, t=st.t, q=st.q,
                                       ps=st.ps)
        model.run(verbose=False)
        return tl.pack(model.ga.to_spectral(tl.sp))

    return LinearOperator((tl.size, tl.size), matvec=_matvec,
                          dtype=np.complex128)


# ---------------------------------------------------------------------------
# Small conveniences used across chapters
# ---------------------------------------------------------------------------
def gaussian_heat(sp, vg, lat0=0.0, lon0=180.0, dlon=15.0, dlat=8.0,
                  amp_k_per_day=1.0, sig_bot=0.1):
    """A steady Gaussian mid-tropospheric heating (K s^-1), grid space."""
    lon = sp.lon_deg
    lat = sp.lat_deg
    sig = asnumpy(vg.sig)
    dl = (lon[None, :] - lon0 + 180.0) % 360.0 - 180.0
    hxy = np.exp(-(dl / dlon) ** 2 - ((lat[:, None] - lat0) / dlat) ** 2)
    vz = np.sin(np.pi * (1.0 - sig)) * (sig > sig_bot)
    return hxy[:, :, None] * vz[None, None, :] * (amp_k_per_day / 86400.0)


def growth_rate(eigenvalue):
    """Growth rate [day^-1] and period [day] from a tendency eigenvalue (s^-1)."""
    gr = float(np.real(eigenvalue)) * 86400.0
    om = float(np.imag(eigenvalue))
    period = (2.0 * np.pi / abs(om) / 86400.0) if abs(om) > 0 else np.inf
    return gr, period
