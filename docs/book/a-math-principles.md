# Appendix A — Mathematical principles of linear dynamics

This appendix collects the linear-algebra that the diagnostic chapters rest on.
It mirrors Appendix A of the Watanabe–Kimoto guide, specialised to the
PyLBM-Isca operator.

## A.1 The linear dynamical system

Write the perturbation state as a single vector $x'$ (the packed
$(\zeta',D',T',q',\pi')$ of [`lbmtools.TangentLinear`](https://github.com/kuiper2000/PyLBM-Isca/blob/main/docs/book/lbmtools.py)). The model is the linear, autonomous, forced system

$$
\frac{d x'}{dt} = \mathbf{A}\,x' + f, \qquad
\mathbf{A} \equiv \mathbf{L}(\bar{x}) - \mathbf{D},
$$

where $\mathbf{L}(\bar{x})$ is the linearised dynamical tendency about the basic
state $\bar{x}$ and $\mathbf{D}\succeq 0$ the dissipation. Three questions span
the guide, and each is a standard operation on $\mathbf{A}$:

| question | operation | chapter |
|---|---|---|
| steady response to steady $f$ | solve $\mathbf{A}\,x'_\infty = -f$ | [](03-steady-response.ipynb) |
| variance kept up by stochastic $f$ | Lyapunov / time average | [](04-storm-track.ipynb) |
| free evolution $\;x'(t)=e^{\mathbf{A}t}x'(0)$ | eigen / singular modes of $\mathbf{A}$, $e^{\mathbf{A}\tau}$ | [](05-modal-decomposition.ipynb) |

The **steady state** exists and is unique whenever $\mathbf{A}$ is nonsingular
(every eigenvalue has nonzero real part — guaranteed when the dissipation damps
every mode), and then $x'_\infty = -\mathbf{A}^{-1}f$. If a basic state supports
a *growing* mode ($\operatorname{Re}\lambda>0$), no bounded steady response
exists and the time integration diverges — which is itself the diagnostic of
[](05-modal-decomposition.ipynb).

## A.2 The operator and its matrix

$\mathbf{A}$ is **linear but not symmetric** (advection is skew-ish; the basic
state breaks self-adjointness), i.e. **non-normal**: $\mathbf{A}\mathbf{A}^{\!*}
\neq \mathbf{A}^{\!*}\mathbf{A}$. Non-normality is the whole physical interest of
the model — it is why eigenmodes are not orthogonal, why a stable operator
($\operatorname{Re}\lambda<0$ for all $\lambda$) can still grow transiently, and
why eigenvectors and singular vectors give *different* "preferred patterns".

PyLBM-Isca never forms $\mathbf{A}$ as a matrix in a normal run — it only ever
applies it (one tendency evaluation). For diagnostics we can either

- **apply** it matrix-free (`TangentLinear.matvec`, wrapped as a
  `scipy.sparse.linalg.LinearOperator`) and use iterative solvers / Arnoldi, or
- **build** it densely (`TangentLinear.dense`, one matvec per column) when the
  state vector is small enough — which, for a zonally-symmetric basic state,
  means restricting to one zonal wavenumber $m$.

### Decoupling by zonal wavenumber

For a **zonally-symmetric** basic state $\bar{x}(\phi,\sigma)$, $\mathbf{A}$ is
**block-diagonal in $m$**: a perturbation of zonal wavenumber $m$ generates a
tendency of the same $m$ only. The full operator splits into independent blocks
$\mathbf{A}_m$ of dimension $\sim (N\!-\!m\!+\!1)\times K \times 4$ — a few
hundred to a thousand at T21 L20 — each small enough for dense `numpy.linalg.eig`
and `svd`. This is the classical LBM route to eigenvalues and singular vectors,
and the reason [](05-modal-decomposition.ipynb) and [](08-barotropic.ipynb)
start by calling `lbmtools.zonal_mean`.

## A.3 Detection of preferred modes

Two "preferred patterns" of a non-normal operator, both used in the guide:

**Eigenmodes** solve $\mathbf{A}\,\phi_j = \lambda_j\,\phi_j$. The real part of
$\lambda_j$ is a growth rate (day$^{-1}$ after multiplying by 86 400), the
imaginary part an angular frequency; $2\pi/|\operatorname{Im}\lambda_j|$ is the
period. The **least-damped** eigenmode (largest $\operatorname{Re}\lambda$) is
the pattern the free system relaxes onto last, and a candidate for a *preferred
mode of variability* — the response to broadband forcing projects most strongly
onto it (Appendix A.3 of the WK guide; Navarra 1993; Simmons et al. 1983).

**Singular vectors** (optimal vectors) diagonalise the finite-time **propagator**
$\mathbf{M}(\tau)=e^{\mathbf{A}\tau}$ under a chosen norm:
$\mathbf{M}=\mathbf{U}\,\Sigma\,\mathbf{V}^{\!*}$. The leading right singular
vector $v_1$ is the initial perturbation that **grows most** over $\tau$
(amplification $\sigma_1$), and $u_1$ is the pattern it becomes. For a
non-normal $\mathbf{A}$, $\sigma_1$ can exceed $e^{\tau\max\operatorname{Re}\lambda}$
— transient growth with no growing eigenmode (Farrell 1988; optimal perturbations
/ storm-track seeding). [](05-modal-decomposition.ipynb) computes both.

## A.4 Linear convective interaction with dynamics

The moist extension (WK Appendix A.4; [](07-moist.md)) is, in this language,
the statement that a convective closure $\mathcal{M}$ adds a block to
$\mathbf{L}$ coupling $q'$ (and $\omega'$) back into the $T'$ tendency. Because it
is still linear and still a (basic-state $\times$ perturbation) product, every
operation in A.1–A.3 applies **unchanged** to the moist $\mathbf{A}$ — the
eigenvalue that the closure pushes toward (or past) $\operatorname{Re}\lambda=0$
is the *moist mode*. The implementation hook is tabulated in
[](07-moist.md).
