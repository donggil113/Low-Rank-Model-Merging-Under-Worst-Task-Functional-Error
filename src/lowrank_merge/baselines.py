"""Layer-level baselines, all operating on effective deltas unless noted.

Unofficial re-implementations are labelled as such. They follow the method
descriptions read in the papers (see PRIOR_ART.md) and have NOT been checked
against the official code; they are CPU-fixture baselines, not faithful
reproductions for reporting.

Information access per method is returned in `MethodOutput.access` so that
comparisons can state it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import linalg as la
from .adapters import LoRAAdapter
from .linalg import Matrix
from .objectives import QuadTask, gram
from .wrrr import wrrr


@dataclass
class MethodOutput:
    name: str
    M: Matrix
    target_rank: Optional[int]
    access: Dict[str, object]
    hparams: Dict[str, object] = field(default_factory=dict)
    info: Dict[str, object] = field(default_factory=dict)


DATA_FREE = {"calibration_inputs": False, "task_identity_for_calibration": False,
             "dev_inputs_for_tuning": False, "labels": False, "uses_factors": False}


def _access(**kw) -> Dict[str, object]:
    out = dict(DATA_FREE)
    out.update(kw)
    return out


# ----------------------------------------------------------------------------
# data-free
# ----------------------------------------------------------------------------

def task_arithmetic(deltas: Sequence[Matrix], coef: float) -> Matrix:
    return la.scale(coef, la.lincomb([1.0] * len(deltas), deltas))


def ta_svd(deltas: Sequence[Matrix], k: int, coef: float) -> MethodOutput:
    """coef * sum_t D_t, then Eckart-Young truncation (merge-then-compress)."""
    m = la.svd_truncate(task_arithmetic(deltas, coef), k)
    return MethodOutput("ta_svd", m, k, _access(), {"coef": coef})


def knots_decompose(deltas: Sequence[Matrix]):
    """KnOTS alignment: SVD of the concatenation [D_1, ..., D_T] (along inputs).

    D_t = U diag(s) V_t^T with a shared U (d_out x p) and task blocks V_t.
    Concatenation axis as described in Stoica et al. (ICLR 2025); UNOFFICIAL.
    """
    d_in = len(deltas[0][0])
    cat = la.hstack(deltas)
    u, s, v = la.svd(cat)  # v: (T*d_in) x p
    keep = [i for i, x in enumerate(s) if x > s[0] * 1e-12] if s and s[0] > 0 else []
    u = la.columns(u, keep)
    s = [s[i] for i in keep]
    v = la.columns(v, keep)
    v_blocks = [v[t * d_in:(t + 1) * d_in] for t in range(len(deltas))]
    return u, s, v_blocks


def _ties(vectors: Sequence[List[float]], density: float) -> List[float]:
    """TIES-merging (trim, elect sign, disjoint mean) on flat vectors."""
    n = len(vectors[0])
    trimmed = []
    for vec in vectors:
        k_keep = max(1, int(round(density * n)))
        thr = sorted((abs(x) for x in vec), reverse=True)[k_keep - 1]
        trimmed.append([x if abs(x) >= thr else 0.0 for x in vec])
    out = [0.0] * n
    for j in range(n):
        tot = sum(tv[j] for tv in trimmed)
        sgn = 1.0 if tot >= 0 else -1.0
        agree = [tv[j] for tv in trimmed if tv[j] != 0.0 and (tv[j] > 0) == (sgn > 0)]
        out[j] = sum(agree) / len(agree) if agree else 0.0
    return out


def knots(deltas: Sequence[Matrix], k: Optional[int], merge: str, coef: float,
          density: float = 0.2) -> MethodOutput:
    """KnOTS-TA or KnOTS-TIES (unofficial), optionally truncated to rank k.

    KnOTS itself sets no final-rank budget; truncating to k is OUR addition so
    that all methods are compared at the same final rank.
    """
    u, s, v_blocks = knots_decompose(deltas)
    p = len(s)
    d_in = len(deltas[0][0])
    if merge == "ta":
        v_m = la.scale(coef, la.lincomb([1.0] * len(v_blocks), v_blocks))
    elif merge == "ties":
        flat = [[x for row in vb for x in row] for vb in v_blocks]
        merged = _ties(flat, density)
        v_m = [[coef * merged[i * p + j] for j in range(p)] for i in range(d_in)]
    else:
        raise ValueError(merge)
    us = [[u[i][j] * s[j] for j in range(p)] for i in range(len(u))]
    m_full = la.matmul(us, la.transpose(v_m))
    rank_full = la.numerical_rank(m_full)
    m = la.svd_truncate(m_full, k) if k is not None else m_full
    return MethodOutput(f"knots_{merge}", m, k, _access(),
                        {"coef": coef, "density": density if merge == "ties" else None},
                        {"rank_before_truncation": rank_full, "unofficial": True})


def ctm_like_ta(deltas: Sequence[Matrix], k: int, coef: float,
                hooi_iters: int = 50) -> MethodOutput:
    """Compress-then-Merge-like baseline (unofficial, simplified).

    Shared orthonormal U (d_out x k), V (d_in x k) from a Tucker-2 fit of the
    deltas (HOSVD init + HOOI), cores O_t = U^T D_t V merged by averaging,
    M = U (coef * sum_t O_t) V^T; rank <= k by construction. The per-task norm
    blending (beta) and TIES/Iso-C core merging of He et al. (2026) are omitted.
    """
    u = la.orthonormal_basis(la.hstack(deltas), k)
    v = la.orthonormal_basis(la.transpose(la.vstack(deltas)), k)
    for _ in range(hooi_iters):
        a = la.zeros(len(u), len(u))
        for d in deltas:
            dv = la.matmul(d, v)
            la.axpy_inplace(a, 1.0, la.matmul(dv, la.transpose(dv)))
        u = la.columns(la.sym_eig(a)[1], range(k))
        b = la.zeros(len(v), len(v))
        for d in deltas:
            dtu = la.matmul(la.transpose(d), u)
            la.axpy_inplace(b, 1.0, la.matmul(dtu, la.transpose(dtu)))
        v = la.columns(la.sym_eig(b)[1], range(k))
    core = la.zeros(k, k)
    for d in deltas:
        la.axpy_inplace(core, coef, la.matmul(la.matmul(la.transpose(u), d), v))
    m = la.matmul(la.matmul(u, core), la.transpose(v))
    return MethodOutput("ctm_like_ta", m, k, _access(), {"coef": coef}, {"unofficial": True})


def factor_average(adapters: Sequence[LoRAAdapter], k: int) -> MethodOutput:
    """M = mean_t(s_t B_t) @ mean_t(A_t) (NOT gauge invariant; diagnostic only).

    Requires equal adapter ranks. Identical adapters are reproduced exactly.
    """
    t_n = len(adapters)
    b = la.scale(1.0 / t_n, la.lincomb([ad.scaling for ad in adapters], [ad.B for ad in adapters]))
    a = la.scale(1.0 / t_n, la.lincomb([1.0] * t_n, [ad.A for ad in adapters]))
    m = la.matmul(b, a)
    if k < len(a):
        m = la.svd_truncate(m, k)
    return MethodOutput("factor_average", m, k, _access(uses_factors=True))


# ----------------------------------------------------------------------------
# calibration-based
# ----------------------------------------------------------------------------

def regmean_gram(sigma: Matrix, alpha: float) -> Matrix:
    """RegMean off-diagonal shrinkage: alpha*G + (1-alpha)*diag(G).

    This changes the objective (Jin et al. 2023, Sec. 3.3); alpha = 1 is the
    plain empirical functional error.
    """
    n = len(sigma)
    return [[sigma[i][j] * (alpha if i != j else 1.0) for j in range(n)] for i in range(n)]


def regmean_tasks(deltas: Sequence[Matrix], xs: Sequence[Matrix], alpha: float) -> List[QuadTask]:
    out = []
    for i, (d, x) in enumerate(zip(deltas, xs)):
        g = regmean_gram(gram(x), alpha)
        h = la.matmul(d, g)
        out.append(QuadTask(f"t{i}", g, h, la.inner(h, d), 1.0))
    return out


def regmean(deltas: Sequence[Matrix], xs: Sequence[Matrix], alpha: float,
            k: Optional[int], truncation: str) -> MethodOutput:
    """RegMean (uniform task weights, per-task Gram normalised by n_t).

    truncation='none'   : full-rank RegMean (rank budget NOT respected)
    truncation='euclid' : Eckart-Young truncation of the RegMean solution
    truncation='whitened': rank-k truncation in the S^{1/2} metric, i.e. the
                          exact minimiser of the (alpha-modified) summed error
                          under rank <= k (reduced-rank regression)
    """
    tasks = regmean_tasks(deltas, xs, alpha)
    w = [1.0 / len(tasks)] * len(tasks)
    mode = "pinv"
    full = wrrr(tasks, w, None, mode=mode).M
    if truncation == "none":
        m, tr = full, None
    elif truncation == "euclid":
        m, tr = la.svd_truncate(full, k), k
    elif truncation == "whitened":
        m, tr = wrrr(tasks, w, k, mode=mode).M, k
    else:
        raise ValueError(truncation)
    return MethodOutput(f"regmean_{truncation}", m, tr,
                        _access(calibration_inputs=True, task_identity_for_calibration=True),
                        {"alpha": alpha})
