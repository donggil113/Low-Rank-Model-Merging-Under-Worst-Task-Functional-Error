"""Minimal dense linear algebra in pure Python (stdlib only).

NumPy is not available in the execution environment and installing packages
was not authorised, so this module provides the small set of operations the
CPU reference needs. Matrices are lists of row lists of floats. Everything is
O(n^3) per call and intended for matrices up to a few dozen rows/columns.

Algorithms:
  * symmetric eigendecomposition: cyclic Jacobi rotations
  * SVD: one-sided (Hestenes) Jacobi on the tall orientation
  * linear solves: Gaussian elimination with partial pivoting
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

Matrix = List[List[float]]
Vector = List[float]

EPS = 2.220446049250313e-16


class LinAlgError(ValueError):
    """Raised for singular systems or non-convergence."""


# ----------------------------------------------------------------------------
# construction / shape helpers
# ----------------------------------------------------------------------------

def zeros(m: int, n: int) -> Matrix:
    return [[0.0] * n for _ in range(m)]


def eye(n: int) -> Matrix:
    out = zeros(n, n)
    for i in range(n):
        out[i][i] = 1.0
    return out


def shape(a: Matrix) -> Tuple[int, int]:
    return (len(a), len(a[0]) if a else 0)


def copy(a: Matrix) -> Matrix:
    return [row[:] for row in a]


def transpose(a: Matrix) -> Matrix:
    if not a:
        return []
    return [list(col) for col in zip(*a)]


def diag(v: Sequence[float]) -> Matrix:
    out = zeros(len(v), len(v))
    for i, x in enumerate(v):
        out[i][i] = float(x)
    return out


def random_normal(m: int, n: int, rng: random.Random, std: float = 1.0) -> Matrix:
    return [[rng.gauss(0.0, std) for _ in range(n)] for _ in range(m)]


def columns(a: Matrix, idx: Sequence[int]) -> Matrix:
    return [[row[j] for j in idx] for row in a]


def hstack(blocks: Sequence[Matrix]) -> Matrix:
    m = len(blocks[0])
    return [sum((b[i] for b in blocks), []) for i in range(m)]


def vstack(blocks: Sequence[Matrix]) -> Matrix:
    out: Matrix = []
    for b in blocks:
        out.extend(row[:] for row in b)
    return out


# ----------------------------------------------------------------------------
# arithmetic
# ----------------------------------------------------------------------------

def matmul(a: Matrix, b: Matrix) -> Matrix:
    if len(a[0]) != len(b):
        raise ValueError(f"matmul shape mismatch {shape(a)} @ {shape(b)}")
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def matvec(a: Matrix, v: Sequence[float]) -> Vector:
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def add(a: Matrix, b: Matrix) -> Matrix:
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def sub(a: Matrix, b: Matrix) -> Matrix:
    return [[x - y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def scale(c: float, a: Matrix) -> Matrix:
    return [[c * x for x in row] for row in a]


def axpy_inplace(acc: Matrix, c: float, a: Matrix) -> None:
    """acc += c * a (in place)."""
    for ra, rb in zip(acc, a):
        for j, x in enumerate(rb):
            ra[j] += c * x


def lincomb(coeffs: Sequence[float], mats: Sequence[Matrix]) -> Matrix:
    m, n = shape(mats[0])
    out = zeros(m, n)
    for c, a in zip(coeffs, mats):
        if c != 0.0:
            axpy_inplace(out, c, a)
    return out


def trace(a: Matrix) -> float:
    return sum(a[i][i] for i in range(min(shape(a))))


def frob2(a: Matrix) -> float:
    return sum(x * x for row in a for x in row)


def frob(a: Matrix) -> float:
    return math.sqrt(frob2(a))


def inner(a: Matrix, b: Matrix) -> float:
    """Frobenius inner product <a, b> = tr(a^T b)."""
    return sum(x * y for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def max_abs(a: Matrix) -> float:
    return max((abs(x) for row in a for x in row), default=0.0)


def symmetrize(a: Matrix) -> Matrix:
    n = len(a)
    return [[0.5 * (a[i][j] + a[j][i]) for j in range(n)] for i in range(n)]


def gram_rows(x: Matrix) -> Matrix:
    """x @ x^T for x of shape (d, n) (columns are samples)."""
    d = len(x)
    out = zeros(d, d)
    for i in range(d):
        xi = x[i]
        for j in range(i, d):
            v = sum(p * q for p, q in zip(xi, x[j]))
            out[i][j] = v
            out[j][i] = v
    return out


# ----------------------------------------------------------------------------
# symmetric eigendecomposition (cyclic Jacobi)
# ----------------------------------------------------------------------------

def sym_eig(a: Matrix, tol: float = 1e-15, max_sweeps: int = 100) -> Tuple[Vector, Matrix]:
    """Eigendecomposition of a symmetric matrix.

    Returns (w, V) with eigenvalues w sorted in descending order and V holding
    the corresponding orthonormal eigenvectors as columns, so a = V diag(w) V^T.
    """
    n = len(a)
    s = symmetrize(a)
    v = eye(n)
    for _ in range(max_sweeps):
        off = sum(s[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        diag_norm = sum(s[i][i] ** 2 for i in range(n))
        if off <= (tol ** 2) * max(diag_norm, 1e-300):
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = s[p][q]
                if apq == 0.0:
                    continue
                app, aqq = s[p][p], s[q][q]
                theta = (aqq - app) / (2.0 * apq)
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                sn = t * c
                for k in range(n):
                    skp, skq = s[k][p], s[k][q]
                    s[k][p] = c * skp - sn * skq
                    s[k][q] = sn * skp + c * skq
                for k in range(n):
                    spk, sqk = s[p][k], s[q][k]
                    s[p][k] = c * spk - sn * sqk
                    s[q][k] = sn * spk + c * sqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - sn * vkq
                    v[k][q] = sn * vkp + c * vkq
    else:
        raise LinAlgError("Jacobi eigendecomposition did not converge")
    w = [s[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: -w[i])
    return [w[i] for i in order], columns(v, order)


def sym_fn(a: Matrix, fn, w_v: Tuple[Vector, Matrix] | None = None) -> Matrix:
    """Apply a scalar function to the spectrum of a symmetric matrix."""
    w, v = w_v if w_v is not None else sym_eig(a)
    fw = [fn(x) for x in w]
    n = len(w)
    return [[sum(v[i][k] * fw[k] * v[j][k] for k in range(n)) for j in range(n)] for i in range(n)]


# ----------------------------------------------------------------------------
# SVD (one-sided Jacobi)
# ----------------------------------------------------------------------------

def svd(a: Matrix, tol: float = 1e-15, max_sweeps: int = 100) -> Tuple[Matrix, Vector, Matrix]:
    """Thin SVD a = U diag(s) V^T with s sorted descending.

    U has shape (m, p), V has shape (n, p), p = min(m, n). Columns of U that
    belong to zero singular values are completed to an orthonormal set.
    """
    m, n = shape(a)
    if m < n:
        u, s, v = svd(transpose(a), tol=tol, max_sweeps=max_sweeps)
        return v, s, u
    # work on columns of a (m >= n)
    cols = [list(c) for c in zip(*a)]  # n columns of length m
    vcols = [[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)]
    for _ in range(max_sweeps):
        rotated = False
        for p in range(n - 1):
            cp = cols[p]
            for q in range(p + 1, n):
                cq = cols[q]
                alpha = sum(x * x for x in cp)
                beta = sum(x * x for x in cq)
                gamma = sum(x * y for x, y in zip(cp, cq))
                if gamma == 0.0 or abs(gamma) <= tol * math.sqrt(alpha * beta):
                    continue
                rotated = True
                zeta = (beta - alpha) / (2.0 * gamma)
                t = math.copysign(1.0, zeta) / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
                c = 1.0 / math.sqrt(1.0 + t * t)
                sn = c * t
                for k in range(m):
                    x, y = cp[k], cq[k]
                    cp[k] = c * x - sn * y
                    cq[k] = sn * x + c * y
                vp, vq = vcols[p], vcols[q]
                for k in range(n):
                    x, y = vp[k], vq[k]
                    vp[k] = c * x - sn * y
                    vq[k] = sn * x + c * y
        if not rotated:
            break
    else:
        raise LinAlgError("one-sided Jacobi SVD did not converge")
    norms = [math.sqrt(sum(x * x for x in c)) for c in cols]
    order = sorted(range(n), key=lambda j: -norms[j])
    s = [norms[j] for j in order]
    smax = s[0] if s else 0.0
    ucols: List[Vector] = []
    for j in order:
        nj = norms[j]
        if nj > max(smax, 1e-300) * m * EPS * 10 and nj > 0.0:
            ucols.append([x / nj for x in cols[j]])
        else:
            ucols.append(None)  # type: ignore[arg-type]
    ucols = _complete_orthonormal(ucols, m)
    u = [[ucols[j][i] for j in range(n)] for i in range(m)]
    v = [[vcols[j][i] for j in order] for i in range(n)]
    return u, s, v


def _complete_orthonormal(cols: List[Vector | None], m: int) -> List[Vector]:
    """Replace None entries by unit vectors orthogonal to all others."""
    fixed = [c for c in cols if c is not None]
    out: List[Vector] = []
    basis_iter = iter(range(m))
    for c in cols:
        if c is not None:
            out.append(c)
            continue
        while True:
            e = [0.0] * m
            e[next(basis_iter)] = 1.0
            for b in fixed:
                proj = sum(x * y for x, y in zip(e, b))
                e = [x - proj * y for x, y in zip(e, b)]
            nrm = math.sqrt(sum(x * x for x in e))
            if nrm > 1e-8:
                e = [x / nrm for x in e]
                fixed.append(e)
                out.append(e)
                break
    return out


def truncated_svd(a: Matrix, k: int) -> Tuple[Matrix, Vector, Matrix]:
    u, s, v = svd(a)
    k = min(k, len(s))
    return columns(u, range(k)), s[:k], columns(v, range(k))


def svd_truncate(a: Matrix, k: int) -> Matrix:
    """Best rank-k approximation in Frobenius norm (Eckart-Young)."""
    m, n = shape(a)
    if k <= 0:
        return zeros(m, n)
    u, s, v = truncated_svd(a, k)
    us = [[u[i][j] * s[j] for j in range(len(s))] for i in range(m)]
    return matmul(us, transpose(v))


def singular_values(a: Matrix) -> Vector:
    return svd(a)[1]


def numerical_rank(a: Matrix, rtol: float | None = None) -> int:
    s = singular_values(a)
    if not s or s[0] == 0.0:
        return 0
    if rtol is None:
        rtol = max(shape(a)) * EPS * 10
    return sum(1 for x in s if x > rtol * s[0])


def orthonormal_basis(a: Matrix, k: int) -> Matrix:
    """k orthonormal columns spanning the leading column space of a.

    If rank(a) < k the basis is completed with arbitrary orthonormal columns.
    """
    u, _, _ = svd(a)
    return columns(u, range(min(k, len(u[0]))))


# ----------------------------------------------------------------------------
# linear systems
# ----------------------------------------------------------------------------

def solve(a: Matrix, b: Matrix) -> Matrix:
    """Solve a x = b by Gaussian elimination with partial pivoting."""
    n = len(a)
    aug = [a[i][:] + b[i][:] for i in range(n)]
    ncols = len(b[0])
    scale_ref = max_abs(a)
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) <= n * EPS * max(scale_ref, 1e-300):
            raise LinAlgError("singular matrix in solve")
        aug[col], aug[piv] = aug[piv], aug[col]
        pr = aug[col]
        inv_p = 1.0 / pr[col]
        for r in range(col + 1, n):
            f = aug[r][col] * inv_p
            if f != 0.0:
                row = aug[r]
                for j in range(col, n + ncols):
                    row[j] -= f * pr[j]
    x = zeros(n, ncols)
    for i in range(n - 1, -1, -1):
        row = aug[i]
        for j in range(ncols):
            acc = row[n + j] - sum(row[k] * x[k][j] for k in range(i + 1, n))
            x[i][j] = acc / row[i]
    return x


def inv(a: Matrix) -> Matrix:
    return solve(a, eye(len(a)))


def cond_sym_psd(w: Sequence[float]) -> float:
    """Condition number from eigenvalues of a PSD matrix (inf if singular)."""
    wmax = max(w)
    wmin = min(w)
    if wmin <= 0.0:
        return math.inf
    return wmax / wmin


def random_orthogonal(n: int, rng: random.Random) -> Matrix:
    q, _, _ = svd(random_normal(n, n, rng))
    return q
