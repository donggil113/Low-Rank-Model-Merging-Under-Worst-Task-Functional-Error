"""Worst-task (minimax) low-rank merging with a Lagrangian dual lower bound.

Problem:   OPT = min_{rank(M) <= k}  F(M),   F(M) = max_t e_t(M).

Facts used (standard, not claimed as new):
  * max_t e_t(M) = max_{lambda in simplex} sum_t lambda_t e_t(M).
  * Weak duality: for every lambda,
        g(lambda) = min_{rank(M) <= k} sum_t lambda_t e_t(M) <= OPT.
    g is concave (a pointwise min of linear functions of lambda) and the
    vector (e_t(M_lambda))_t is a supergradient at lambda.
  * g(lambda) is computed EXACTLY by weighted reduced-rank regression (wrrr.py)
    whenever S(lambda) is PD ('pd') or the 'pinv' conditions hold. The ridge
    variant changes the objective and is therefore refused here.
  * Every feasible M gives an upper bound F(M) >= OPT.
  * Because of the rank constraint the problem is non-convex and the duality
    gap  min F - max g  can be strictly positive (see tests for an explicit
    two-task example). With no rank constraint the problem is convex and, by
    Sion's minimax theorem, the gap is zero.

The solver therefore reports (upper_bound, lower_bound, gap). It never claims
global optimality unless the gap closes to tolerance, and then only up to the
numerical accuracy of the inner solves.

Primal heuristics (all accept-if-improved, so F never increases):
  1. candidates M_lambda from the dual iterates;
  2. block-convex alternation: fixing an orthonormal column basis U makes
     M = U Z and each e_t convex in Z; fixing a row basis V makes M = W V^T
     and each e_t convex in W. Each block is a convex minimax solved by the
     same dual ascent without rank constraint (zero duality gap);
  3. seeded perturbation + gradient descent on a log-sum-exp smoothing of the
     max over a balanced factorisation M = P Q^T.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import linalg as la
from .linalg import Matrix
from .objectives import QuadTask
from .wrrr import SingularGramError, wrrr


@dataclass
class DualConfig:
    iters: int = 300          # diminishing-step mirror ascent (robust, O(1/sqrt(N)))
    eta0: float = 1.0
    polish_iters: int = 200   # backtracking mirror ascent from the best lambda so far
    weight_floor: float = 1e-12
    mode: str = "pd"


@dataclass
class RefineConfig:
    block_rounds: int = 10
    block_dual: DualConfig = field(default_factory=lambda: DualConfig(iters=300, mode="pinv"))
    local_restarts: int = 3
    local_iters: int = 150
    local_taus: Tuple[float, ...] = (3e-2, 1e-2, 3e-3, 1e-3)
    perturb_scale: float = 0.1
    seed: int = 0
    rel_improve_tol: float = 1e-12   # accept a candidate only if it lowers F by more
    block_min_rel_gain: float = 1e-9 # stop block rounds when a round gains less
    skip_if_gap_rel: float = 1e-9    # no refinement when the dual gap is already closed


@dataclass
class DualResult:
    lower_bound: float
    lb_weights: List[float]
    best_M: Matrix
    upper_bound: float
    ub_weights: List[float]
    avg_M: Optional[Matrix]
    history: List[Dict[str, object]]
    n_solves: int
    nonunique_steps: int


@dataclass
class MinimaxResult:
    M: Matrix
    upper_bound: float
    lower_bound: float
    gap: float
    gap_rel: float
    per_task: List[float]
    lb_weights: List[float]
    dual: DualResult
    refine_log: List[Dict[str, object]]
    n_wrrr_solves: int
    n_objective_evals: int

    def certified(self, tol: float = 1e-6) -> bool:
        return self.gap_rel <= tol


def max_error(tasks: Sequence[QuadTask], m: Matrix) -> Tuple[float, List[float]]:
    vals = [t.value(m) for t in tasks]
    return max(vals), vals


def _normalise(weights: List[float], floor: float) -> List[float]:
    w = [max(x, floor) for x in weights]
    s = sum(w)
    return [x / s for x in w]


def dual_ascent(tasks: Sequence[QuadTask], k: Optional[int], cfg: DualConfig,
                init_weights: Optional[Sequence[float]] = None) -> DualResult:
    """Mirror (exponentiated-gradient) ascent on the concave dual g(lambda)."""
    if cfg.mode == "ridge":
        raise ValueError("ridge changes the objective; its values are not valid lower bounds")
    t_n = len(tasks)
    lam = _normalise(list(init_weights) if init_weights else [1.0 / t_n] * t_n, cfg.weight_floor)
    best_lb, lb_w = -math.inf, lam[:]
    best_ub, ub_w, best_m = math.inf, lam[:], None
    avg_m: Optional[Matrix] = None
    avg_count = 0
    convex = k is None
    history: List[Dict[str, object]] = []
    nonunique = 0
    for i in range(cfg.iters):
        res = wrrr(tasks, lam, k, mode=cfg.mode)
        e = res.per_task(tasks)
        g = sum(l * x for l, x in zip(lam, e))
        f = max(e)
        if res.unique is False:
            nonunique += 1
        if g > best_lb:
            best_lb, lb_w = g, lam[:]
        if f < best_ub:
            best_ub, ub_w, best_m = f, lam[:], res.M
        if convex:
            # ergodic primal average (valid only without a rank constraint)
            avg_count += 1
            if avg_m is None:
                avg_m = la.copy(res.M)
            else:
                avg_m = la.add(la.scale(1.0 - 1.0 / avg_count, avg_m),
                               la.scale(1.0 / avg_count, res.M))
        history.append({"iter": i, "g": g, "F": f, "weights": lam[:],
                        "unique": res.unique})
        emax = max(e)
        if emax <= 0.0:
            break
        eta = cfg.eta0 / math.sqrt(i + 1.0) / emax
        lam = _normalise([l * math.exp(eta * (x - emax)) for l, x in zip(lam, e)],
                         cfg.weight_floor)
    # Polish: monotone mirror ascent with backtracking from the best lambda.
    # Every evaluated g(lambda) is a valid lower bound, so this can only tighten
    # the bound; it may stall at a kink of g, which is why it runs second.
    lam = lb_w[:]
    eta = cfg.eta0
    g_cur = best_lb
    e_cur = None
    for i in range(cfg.polish_iters):
        if e_cur is None:
            res = wrrr(tasks, lam, k, mode=cfg.mode)
            e_cur = res.per_task(tasks)
            history.append({"iter": len(history), "g": g_cur, "F": max(e_cur),
                            "weights": lam[:], "unique": res.unique, "polish": True})
        emax = max(e_cur)
        if emax <= 0.0 or eta < 1e-12:
            break
        cand = _normalise([l * math.exp(eta / emax * (x - emax)) for l, x in zip(lam, e_cur)],
                          cfg.weight_floor)
        res = wrrr(tasks, cand, k, mode=cfg.mode)
        e_new = res.per_task(tasks)
        g_new = sum(l * x for l, x in zip(cand, e_new))
        f_new = max(e_new)
        history.append({"iter": len(history), "g": g_new, "F": f_new, "weights": cand[:],
                        "unique": res.unique, "polish": True})
        if res.unique is False:
            nonunique += 1
        if f_new < best_ub:
            best_ub, ub_w, best_m = f_new, cand[:], res.M
        if g_new > g_cur:
            lam, g_cur, e_cur = cand, g_new, e_new
            eta *= 1.5
            if g_new > best_lb:
                best_lb, lb_w = g_new, cand[:]
        else:
            eta *= 0.5
    if avg_m is not None:
        f_avg, _ = max_error(tasks, avg_m)
        if f_avg < best_ub:
            best_ub, best_m = f_avg, avg_m
    assert best_m is not None
    return DualResult(lower_bound=best_lb, lb_weights=lb_w, best_M=best_m,
                      upper_bound=best_ub, ub_weights=ub_w, avg_M=avg_m,
                      history=history, n_solves=len(history),
                      nonunique_steps=nonunique)


# ----------------------------------------------------------------------------
# block-convex sub-problems
# ----------------------------------------------------------------------------

def column_block_tasks(tasks: Sequence[QuadTask], u: Matrix) -> List[QuadTask]:
    """Tasks for M = U Z with U^T U = I:  e_t(UZ) in terms of Z (k x p)."""
    ut = la.transpose(u)
    return [QuadTask(name=t.name, G=t.G, H=la.matmul(ut, t.H), c=t.c, norm=t.norm)
            for t in tasks]


def row_block_tasks(tasks: Sequence[QuadTask], v: Matrix) -> List[QuadTask]:
    """Tasks for M = W V^T with V^T V = I:  e_t(W V^T) in terms of W (d_out x k)."""
    vt = la.transpose(v)
    return [QuadTask(name=t.name, G=la.symmetrize(la.matmul(la.matmul(vt, t.G), v)),
                     H=la.matmul(t.H, v), c=t.c, norm=t.norm)
            for t in tasks]


def _block_step(tasks, m, k, which, cfg: RefineConfig, counters):
    if which == "col":
        basis = la.orthonormal_basis(m, k)
        sub = column_block_tasks(tasks, basis)
    else:
        basis = la.orthonormal_basis(la.transpose(m), k)
        sub = row_block_tasks(tasks, basis)
    try:
        dres = dual_ascent(sub, None, cfg.block_dual)
    except SingularGramError:
        return None, None
    counters["solves"] += dres.n_solves
    z = dres.best_M
    cand = la.matmul(basis, z) if which == "col" else la.matmul(z, la.transpose(basis))
    return cand, dres


# ----------------------------------------------------------------------------
# smoothed local descent on factors
# ----------------------------------------------------------------------------

def _smooth_max(vals: Sequence[float], tau: float) -> Tuple[float, List[float]]:
    mx = max(vals)
    ex = [math.exp((v - mx) / tau) for v in vals]
    z = sum(ex)
    return mx + tau * math.log(z), [x / z for x in ex]


def _factors(m: Matrix, k: int) -> Tuple[Matrix, Matrix]:
    u, s, v = la.truncated_svd(m, k)
    rs = [math.sqrt(x) for x in s]
    p = [[u[i][j] * rs[j] for j in range(len(s))] for i in range(len(u))]
    q = [[v[i][j] * rs[j] for j in range(len(s))] for i in range(len(v))]
    return p, q


def _local_descent(tasks, p, q, cfg: RefineConfig, counters, scale_f: float):
    def f_and_grad(pp, qq, tau):
        m = la.matmul(pp, la.transpose(qq))
        vals = [t.value(m) for t in tasks]
        counters["evals"] += 1
        fs, w = _smooth_max(vals, tau)
        gm = la.lincomb(w, [t.grad(m) for t in tasks])
        return fs, la.matmul(gm, qq), la.matmul(la.transpose(gm), pp), max(vals)

    step = 1.0
    for tau_rel in cfg.local_taus:
        tau = tau_rel * max(scale_f, 1e-300)
        fs, gp, gq, _ = f_and_grad(p, q, tau)
        for _ in range(cfg.local_iters):
            gn2 = la.frob2(gp) + la.frob2(gq)
            if gn2 <= 1e-30:
                break
            accepted = False
            while step > 1e-14:
                p2 = la.sub(p, la.scale(step, gp))
                q2 = la.sub(q, la.scale(step, gq))
                fs2, gp2, gq2, _ = f_and_grad(p2, q2, tau)
                if fs2 <= fs - 1e-4 * step * gn2:
                    p, q, fs, gp, gq = p2, q2, fs2, gp2, gq2
                    accepted = True
                    step *= 1.5
                    break
                step *= 0.5
            if not accepted:
                break
    return p, q


def refine(tasks: Sequence[QuadTask], m0: Matrix, k: int, cfg: RefineConfig,
           counters: Dict[str, int]) -> Tuple[Matrix, List[Dict[str, object]]]:
    """Monotone primal improvement of F starting from a rank-<=k matrix m0."""
    log: List[Dict[str, object]] = []
    m = m0
    f, _ = max_error(tasks, m)
    log.append({"stage": "start", "F": f})

    def accept(cand, stage):
        nonlocal m, f
        if cand is None:
            log.append({"stage": stage, "F": None, "accepted": False})
            return False
        fc, _ = max_error(tasks, cand)
        ok = fc < f - cfg.rel_improve_tol * max(1.0, abs(f))
        log.append({"stage": stage, "F": fc, "accepted": ok})
        if ok:
            m, f = cand, fc
        return ok

    def block_phase(tag):
        for r in range(cfg.block_rounds):
            f_round = f
            for which in ("col", "row"):
                cand, _ = _block_step(tasks, m, k, which, cfg, counters)
                accept(cand, f"{tag}_block_{which}_{r}")
            if f > f_round - cfg.block_min_rel_gain * max(1.0, abs(f_round)):
                break

    block_phase("b1")
    rng = random.Random(cfg.seed)
    scale_f = max(f, 1e-300)
    for r in range(cfg.local_restarts + 1):
        p, q = _factors(m, k)
        if r > 0:
            nrm = max(la.frob(p), la.frob(q), 1e-12)
            sd = cfg.perturb_scale * nrm / math.sqrt(len(p) * max(k, 1))
            p = la.add(p, la.random_normal(len(p), len(p[0]), rng, sd))
            q = la.add(q, la.random_normal(len(q), len(q[0]), rng, sd))
        p, q = _local_descent(tasks, p, q, cfg, counters, scale_f)
        accept(la.matmul(p, la.transpose(q)), f"local_{r}")
    block_phase("b2")
    return m, log


def solve_minimax(tasks: Sequence[QuadTask], k: int, dual_cfg: Optional[DualConfig] = None,
                  refine_cfg: Optional[RefineConfig] = None,
                  do_refine: bool = True) -> MinimaxResult:
    dual_cfg = dual_cfg or DualConfig()
    refine_cfg = refine_cfg or RefineConfig()
    d_out, p = len(tasks[0].H), len(tasks[0].G)
    full = min(d_out, p)
    k_arg: Optional[int] = None if k >= full else k
    dres = dual_ascent(tasks, k_arg, dual_cfg)
    counters = {"solves": dres.n_solves, "evals": 0}
    m = dres.best_M
    log: List[Dict[str, object]] = []
    dual_gap_rel = (dres.upper_bound - dres.lower_bound) / max(abs(dres.upper_bound), 1e-300)
    if do_refine and k_arg is not None and k > 0:
        if dual_gap_rel <= refine_cfg.skip_if_gap_rel:
            log = [{"stage": "skipped", "reason": "dual gap already closed",
                    "dual_gap_rel": dual_gap_rel}]
        else:
            m, log = refine(tasks, m, k, refine_cfg, counters)
    ub, vals = max_error(tasks, m)
    lb = dres.lower_bound
    gap = ub - lb
    return MinimaxResult(M=m, upper_bound=ub, lower_bound=lb, gap=gap,
                         gap_rel=gap / max(abs(ub), 1e-300), per_task=vals,
                         lb_weights=dres.lb_weights, dual=dres, refine_log=log,
                         n_wrrr_solves=counters["solves"],
                         n_objective_evals=counters["evals"])
