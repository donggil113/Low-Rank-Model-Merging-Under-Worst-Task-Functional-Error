"""Fixed-task-weight weighted reduced-rank regression (WRRR).

For fixed weights lambda on the simplex and quadratic task errors
e_t(M) = [tr(M G_t M^T) - 2 tr(M H_t^T) + c_t] / s_t, the weighted objective

    L_lambda(M) = sum_t lambda_t e_t(M)
                = tr(M S M^T) - 2 tr(M C^T) + const,
    S = sum_t w_t G_t,  C = sum_t w_t H_t,  w_t = lambda_t / s_t,

has identity weighting on the output side, so min_{rank(M) <= k} L_lambda is
classical reduced-rank regression (Izenman 1975):

  * S positive definite ("pd"):  M* = SVD_k(C S^{-1/2}) S^{-1/2}.
    This is a global minimiser (Eckart-Young after the invertible change of
    variables N = M S^{1/2}, which preserves rank). It is unique iff
    sigma_k(C S^{-1/2}) > sigma_{k+1}(C S^{-1/2}). With k >= full rank it
    reduces to the (lambda-weighted) RegMean solution C S^{-1}.

  * S singular, original objective ("pinv"): L_lambda only depends on M
    restricted to range(S). If C has no component on null(S) (always true for
    data-derived tasks with lambda_t > 0, since null(S) is contained in every
    null(Sigma_t)), M* = SVD_k(C S^{+1/2}) S^{+1/2} is a global minimiser; it is
    the one that acts as zero on null(S). Any M* + Y P_null(S) attains the same
    calibration objective, so the calibration data do NOT identify M on null(S)
    and test behaviour there is a modelling choice (here: revert to the base).
    If C has a component on null(S) the objective is unbounded below; we raise.

  * ridge (explicitly a DIFFERENT objective): L_lambda(M) + eps * ||M||_F^2,
    solved exactly by replacing S with S + eps I. Its minimiser is not a
    minimiser of L_lambda; the value of L_lambda at it is only an upper
    estimate of min L_lambda, so it must not be used as a dual lower bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import linalg as la
from .linalg import LinAlgError, Matrix
from .objectives import QuadTask

MODES = ("pd", "pinv", "ridge")


class SingularGramError(LinAlgError):
    """S is not numerically positive definite but mode='pd' was requested."""


class UnboundedObjectiveError(LinAlgError):
    """C has a component on null(S): the fixed-weight objective is unbounded."""


@dataclass
class WRRRResult:
    M: Matrix
    weights: List[float]
    k: Optional[int]
    mode: str
    ridge: float
    objective: float  # sum_t lambda_t e_t(M) of the ORIGINAL (un-penalised) objective
    penalized_objective: Optional[float]
    exact_for_original_objective: bool
    whitened_singular_values: List[float]
    unique: Optional[bool]
    spectral_gap_rel: Optional[float]
    gram_eig_min: float
    gram_eig_max: float
    gram_null_dim: int
    info: Dict[str, float] = field(default_factory=dict)

    def per_task(self, tasks: Sequence[QuadTask]) -> List[float]:
        return [t.value(self.M) for t in tasks]


def default_rtol(n: int) -> float:
    return max(n, 1) * la.EPS * 100.0


def combine(tasks: Sequence[QuadTask], weights: Sequence[float]):
    if len(tasks) != len(weights):
        raise ValueError("one weight per task required")
    if any(w < 0 for w in weights):
        raise ValueError("task weights must be non-negative")
    eff = [w / t.norm for w, t in zip(weights, tasks)]
    s = la.lincomb(eff, [t.G for t in tasks])
    c = la.lincomb(eff, [t.H for t in tasks])
    return la.symmetrize(s), c


def weighted_objective(m: Matrix, tasks: Sequence[QuadTask], weights: Sequence[float]) -> float:
    return sum(w * t.value(m) for w, t in zip(weights, tasks))


def wrrr(tasks: Sequence[QuadTask], weights: Sequence[float], k: Optional[int],
         mode: str = "pd", ridge: float = 0.0, rtol: Optional[float] = None) -> WRRRResult:
    """Solve min_{rank(M) <= k} sum_t lambda_t e_t(M) (see module docstring).

    k=None means no rank constraint.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "ridge" and ridge <= 0.0:
        raise ValueError("mode='ridge' needs ridge > 0")
    if mode != "ridge" and ridge != 0.0:
        raise ValueError("ridge > 0 is only allowed with mode='ridge' (it changes the objective)")
    s, c = combine(tasks, weights)
    p = len(s)
    d_out = len(c)
    if rtol is None:
        rtol = default_rtol(p)
    w_raw, v = la.sym_eig(s)
    eig_max = max(w_raw[0], 0.0)
    thr = rtol * eig_max if eig_max > 0 else 0.0
    null_idx = [i for i, x in enumerate(w_raw) if x <= thr]
    info: Dict[str, float] = {}

    if mode == "pd":
        if null_idx or eig_max <= 0.0:
            raise SingularGramError(
                f"combined Gram S is not positive definite: eig_min={w_raw[-1]:.3e}, "
                f"eig_max={w_raw[0]:.3e}, rtol={rtol:.1e}; use mode='pinv' (same objective) "
                f"or mode='ridge' (different objective)")
        w_used = w_raw
    elif mode == "pinv":
        if null_idx:
            # C must vanish on null(S), otherwise the objective is unbounded below
            v0 = la.columns(v, null_idx)
            leak = la.frob(la.matmul(c, v0))
            cn = la.frob(c)
            info["null_leak_rel"] = leak / cn if cn > 0 else 0.0
            if cn > 0 and leak > math.sqrt(rtol) * cn:
                raise UnboundedObjectiveError(
                    f"C has relative component {leak / cn:.3e} on null(S)")
        w_used = [x if x > thr else 0.0 for x in w_raw]
    else:  # ridge
        w_used = [x + ridge for x in w_raw]

    inv_sqrt = [1.0 / math.sqrt(x) if x > 0.0 else 0.0 for x in w_used]
    # S^{-1/2} (or its pseudo-inverse / ridge variant) from the eigenpairs
    s_isqrt = [[sum(v[i][q] * inv_sqrt[q] * v[j][q] for q in range(p)) for j in range(p)]
               for i in range(p)]
    cw = la.matmul(c, s_isqrt)  # C S^{-1/2}
    uu, sv, vv = la.svd(cw)
    full = min(d_out, p)
    if k is None or k >= full:
        n_mat = cw
        k_eff = full
        unique = None
        gap = None
    else:
        k_eff = k
        # Eckart-Young truncation of C S^{-1/2}
        n_mat = [[sum(uu[i][q] * sv[q] * vv[j][q] for q in range(k)) for j in range(p)]
                 for i in range(d_out)]
        if k > 0:
            sk, sk1 = sv[k - 1], sv[k]
            gap = (sk - sk1) / sv[0] if sv[0] > 0 else 0.0
            unique = sk - sk1 > 1e-9 * max(sv[0], 1e-300)
        else:
            gap, unique = None, True
    m = la.matmul(n_mat, s_isqrt)
    obj = weighted_objective(m, tasks, weights)
    pen = obj + ridge * la.frob2(m) if mode == "ridge" else None
    info["k_effective"] = float(k_eff)
    return WRRRResult(
        M=m,
        weights=list(weights),
        k=k,
        mode=mode,
        ridge=ridge,
        objective=obj,
        penalized_objective=pen,
        exact_for_original_objective=(mode != "ridge"),
        whitened_singular_values=list(sv),
        unique=unique,
        spectral_gap_rel=gap,
        gram_eig_min=w_raw[-1],
        gram_eig_max=w_raw[0],
        gram_null_dim=len(null_idx),
        info=info,
    )
