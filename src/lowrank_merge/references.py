"""Independent small-matrix references used only for verification.

These deliberately avoid the whitening / eigendecomposition path of wrrr.py
and the dual machinery of minimax.py:

  * rrr_izenman_raw: classical reduced-rank regression computed from raw
    stacked samples (OLS via Gaussian elimination, then projection of the
    fitted values onto their top-k left singular vectors).
  * grid_rank1_weighted: d_out = 2, k = 1. Exhaustive grid over the column
    direction u(theta) plus golden-section refinement; for fixed u the optimal
    row vector solves a least-squares problem on raw samples.
  * grid_rank1_minimax: d_out = 2, k = 1, worst-task objective. Grid over
    theta; for fixed u the inner problem min_z max_t e_t(u z^T) is convex and
    is solved with the ellipsoid method (subgradient-based, no duality).
  * als_weighted: multi-start alternating least squares for the fixed-weight
    problem (local method; used to check that the closed form is never beaten).

A numerical agreement with these references is implementation evidence on the
tested instances, not a proof.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from . import linalg as la
from .linalg import Matrix


@dataclass
class RawTask:
    delta: Matrix  # (d_out, d_in)
    x: Matrix      # (d_in, n)
    norm: float = 1.0  # error divisor s_t

    @property
    def n(self) -> int:
        return len(self.x[0])


def raw_error(m: Matrix, t: RawTask) -> float:
    r = la.matmul(la.sub(m, t.delta), t.x)
    return la.frob2(r) / t.n / t.norm


def raw_weighted(m: Matrix, tasks: Sequence[RawTask], weights: Sequence[float]) -> float:
    return sum(w * raw_error(m, t) for w, t in zip(weights, tasks))


def _stack(tasks: Sequence[RawTask], weights: Sequence[float]) -> Tuple[Matrix, Matrix]:
    xs, ys = [], []
    for w, t in zip(weights, tasks):
        c = math.sqrt(w / (t.n * t.norm))
        xs.append(la.scale(c, t.x))
        ys.append(la.scale(c, la.matmul(t.delta, t.x)))
    return la.hstack(xs), la.hstack(ys)


def rrr_izenman_raw(tasks: Sequence[RawTask], weights: Sequence[float], k: int) -> Matrix:
    """Reduced-rank regression of stacked targets on stacked inputs.

    Requires the stacked input Gram to be non-singular.
    """
    xt, yt = _stack(tasks, weights)
    sxx = la.gram_rows(xt)                      # X X^T
    syx = la.matmul(yt, la.transpose(xt))       # Y X^T
    # M_ols = syx sxx^{-1}  <=>  sxx M_ols^T = syx^T
    m_ols = la.transpose(la.solve(sxx, la.transpose(syx)))
    fitted = la.matmul(m_ols, xt)
    u, _, _ = la.svd(fitted)
    uk = la.columns(u, range(k))
    return la.matmul(la.matmul(uk, la.transpose(uk)), m_ols)


def _best_row_given_col(u: Sequence[float], tasks: Sequence[RawTask],
                        weights: Sequence[float]) -> Matrix:
    """argmin_v sum_t w_t e_t(u v^T) for a fixed column vector u (raw samples)."""
    d_in = len(tasks[0].x)
    a = la.zeros(d_in, d_in)
    b = la.zeros(d_in, 1)
    uu = sum(x * x for x in u)
    for w, t in zip(weights, tasks):
        c = w / (t.n * t.norm)
        la.axpy_inplace(a, c * uu, la.gram_rows(t.x))
        # sum_j x_j x_j^T D^T u
        dtu = la.matvec(la.transpose(t.delta), u)
        proj = [sum(t.x[i][j] * dtu[i] for i in range(d_in)) for j in range(t.n)]  # x_j^T D^T u
        for i in range(d_in):
            b[i][0] += c * sum(t.x[i][j] * proj[j] for j in range(t.n))
    v = la.solve(a, b)
    return [[ui * v[j][0] for j in range(d_in)] for ui in u]


def _golden(f, a: float, b: float, tol: float = 1e-12, max_iter: int = 200) -> Tuple[float, float]:
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    x = 0.5 * (a + b)
    return x, f(x)


def grid_rank1_weighted(tasks: Sequence[RawTask], weights: Sequence[float],
                        n_grid: int = 720) -> Tuple[float, Matrix]:
    if len(tasks[0].delta) != 2:
        raise ValueError("grid reference implemented for d_out = 2 only")

    def obj(theta: float) -> float:
        u = [math.cos(theta), math.sin(theta)]
        return raw_weighted(_best_row_given_col(u, tasks, weights), tasks, weights)

    thetas = [math.pi * i / n_grid for i in range(n_grid)]
    vals = [obj(th) for th in thetas]
    i0 = min(range(n_grid), key=lambda i: vals[i])
    h = math.pi / n_grid
    th, val = _golden(obj, thetas[i0] - h, thetas[i0] + h)
    if vals[i0] < val:
        th, val = thetas[i0], vals[i0]
    u = [math.cos(th), math.sin(th)]
    return val, _best_row_given_col(u, tasks, weights)


# ----------------------------------------------------------------------------
# convex inner problem via the ellipsoid method
# ----------------------------------------------------------------------------

def _row_quads(u: Sequence[float], tasks: Sequence[RawTask]):
    """For M = u z^T (||u|| = 1): e_t = z^T A_t z - 2 b_t^T z + c_t (per task)."""
    out = []
    for t in tasks:
        n, s = t.n, t.norm
        a = la.scale(1.0 / (n * s), la.gram_rows(t.x))
        dtu = la.matvec(la.transpose(t.delta), u)
        b = la.matvec(a, dtu)          # Sigma D^T u / s
        c = raw_error(la.zeros(len(t.delta), len(t.x)), t)
        out.append((a, b, c))
    return out


def _quad_val_grad(z, quads):
    best, best_g = -math.inf, None
    for a, b, c in quads:
        az = la.matvec(a, z)
        v = sum(p * q for p, q in zip(z, az)) - 2.0 * sum(p * q for p, q in zip(b, z)) + c
        if v > best:
            best = v
            best_g = [2.0 * (p - q) for p, q in zip(az, b)]
    return best, best_g


def ellipsoid_min_max_quads(quads, radius: float, iters: int = 600) -> Tuple[float, List[float]]:
    """Central-cut ellipsoid method in factored form E = {x + L u : ||u|| <= 1}.

    The factored update keeps L L^T positive semidefinite in floating point
    (the plain rank-one update of P = L L^T can lose definiteness and stall).
    """
    n = len(quads[0][1])
    if n < 2:
        raise ValueError("ellipsoid reference needs dimension >= 2")
    x = [0.0] * n
    lmat = la.scale(radius, la.eye(n))
    best_v, best_x = _quad_val_grad(x, quads)[0], x[:]
    c1 = n / math.sqrt(n * n - 1.0)
    c2 = 1.0 - math.sqrt((n - 1.0) / (n + 1.0))
    for _ in range(iters):
        v, g = _quad_val_grad(x, quads)
        if v < best_v:
            best_v, best_x = v, x[:]
        ltg = la.matvec(la.transpose(lmat), g)
        nrm = math.sqrt(sum(a * a for a in ltg))
        if nrm <= 1e-300:
            break
        gt = [a / nrm for a in ltg]
        lgt = la.matvec(lmat, gt)
        x = [a - b / (n + 1.0) for a, b in zip(x, lgt)]
        # L <- c1 * L (I - c2 gt gt^T)
        lmat = [[c1 * (lmat[i][j] - c2 * lgt[i] * gt[j]) for j in range(n)] for i in range(n)]
    v, _ = _quad_val_grad(x, quads)
    if v < best_v:
        best_v, best_x = v, x[:]
    if math.sqrt(sum(a * a for a in best_x)) > 0.9 * radius:
        raise RuntimeError("ellipsoid optimum near the initial boundary; increase radius")
    return best_v, best_x


def grid_rank1_minimax(tasks: Sequence[RawTask], n_grid: int = 360,
                       iters: int = 600, radius: float | None = None) -> Tuple[float, Matrix]:
    if len(tasks[0].delta) != 2:
        raise ValueError("grid reference implemented for d_out = 2 only")
    if radius is None:
        scale_d = max(la.frob(t.delta) for t in tasks)
        radius = 50.0 * (1.0 + scale_d)

    def inner(theta: float) -> Tuple[float, List[float]]:
        u = [math.cos(theta), math.sin(theta)]
        return ellipsoid_min_max_quads(_row_quads(u, tasks), radius, iters)

    thetas = [math.pi * i / n_grid for i in range(n_grid)]
    vals = [inner(th)[0] for th in thetas]
    i0 = min(range(n_grid), key=lambda i: vals[i])
    h = math.pi / n_grid
    th, val = _golden(lambda s: inner(s)[0], thetas[i0] - h, thetas[i0] + h, tol=1e-10)
    if vals[i0] < val:
        th, val = thetas[i0], vals[i0]
    v, z = inner(th)
    u = [math.cos(th), math.sin(th)]
    return v, [[ui * zj for zj in z] for ui in u]


def als_weighted(tasks: Sequence[RawTask], weights: Sequence[float], k: int,
                 restarts: int = 20, iters: int = 500, seed: int = 0) -> Tuple[float, Matrix]:
    """Best objective over random-start ALS runs (M = P Q^T), raw-sample Grams."""
    rng = random.Random(seed)
    d_out, d_in = len(tasks[0].delta), len(tasks[0].x)
    s = la.zeros(d_in, d_in)
    c = la.zeros(d_out, d_in)
    for w, t in zip(weights, tasks):
        cw = w / (t.n * t.norm)
        g = la.gram_rows(t.x)
        la.axpy_inplace(s, cw, g)
        la.axpy_inplace(c, cw, la.matmul(t.delta, g))
    best_v, best_m = math.inf, None
    for _ in range(restarts):
        p = la.random_normal(d_out, k, rng)
        m = None
        for _ in range(iters):
            # Q^T = (P^T P)^{-1} P^T C S^{-1}
            ptp = la.matmul(la.transpose(p), p)
            rhs = la.matmul(la.transpose(p), c)                      # k x d_in
            y = la.solve(ptp, rhs)                                   # (P^T P)^{-1} P^T C
            qt = la.transpose(la.solve(s, la.transpose(y)))          # y S^{-1}
            q = la.transpose(qt)                                     # d_in x k
            # P = C Q (Q^T S Q)^{-1}
            qsq = la.matmul(la.matmul(qt, s), q)
            p = la.transpose(la.solve(qsq, la.transpose(la.matmul(c, q))))
            m = la.matmul(p, qt)
        v = raw_weighted(m, tasks, weights)
        if v < best_v:
            best_v, best_m = v, m
    return best_v, best_m
