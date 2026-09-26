"""Run all layer-level methods on one problem with explicit split discipline.

Splits:
  cal  - used to FIT calibration-based methods (Grams / functional errors)
  dev  - used ONLY to pick scalar hyper-parameters of baselines that have them
         (selection metric fixed in the config); the proposed method has none
  test - reported; never used for fitting or selection
"""

from __future__ import annotations

import time
import traceback
import tracemalloc
from typing import Callable, Dict, List, Optional, Sequence

from . import baselines as bl
from . import linalg as la
from .adapters import LoRAAdapter, param_account
from .linalg import Matrix
from .minimax import DualConfig, RefineConfig, solve_minimax
from .objectives import TaskSplit, evaluate, quad_tasks
from .wrrr import wrrr


def _select(candidates: Sequence[bl.MethodOutput], tasks: Sequence[TaskSplit],
            split: str, metric: str) -> tuple:
    evals = []
    for c in candidates:
        ev = evaluate(c.M, tasks, splits=(split,))
        evals.append((ev["summary"][split][metric], c, ev))
    evals.sort(key=lambda e: e[0])
    grid_log = [{"hparams": c.hparams, f"{split}_{metric}": v} for v, c, _ in evals]
    best = evals[0][1]
    if len(candidates) > 1:
        best.access = {**best.access, "dev_inputs_for_tuning": True}
    return best, grid_log


def build_methods(cfg: Dict[str, object]) -> Dict[str, Callable]:
    grids = cfg["tuning_grids"]
    k = int(cfg["k"])
    sel_split = str(cfg["selection_split"])
    sel_metric = str(cfg["selection_metric"])
    dual_cfg = DualConfig(**cfg.get("dual", {}))
    refine_raw = dict(cfg.get("refine", {}))
    if "block_dual" in refine_raw:
        refine_raw["block_dual"] = DualConfig(**refine_raw["block_dual"])
    if "local_taus" in refine_raw:
        refine_raw["local_taus"] = tuple(refine_raw["local_taus"])
    refine_cfg = RefineConfig(**refine_raw)
    wrrr_mode = str(cfg.get("wrrr_mode", "pd"))

    def zero(adapters, tasks):
        d_out, d_in = len(tasks[0].delta), len(tasks[0].delta[0])
        return bl.MethodOutput("zero_base_model", la.zeros(d_out, d_in), 0, bl._access()), []

    def ta_svd(adapters, tasks):
        deltas = [ad.effective_delta() for ad in adapters]
        cands = [bl.ta_svd(deltas, k, c) for c in grids["coef"]]
        return _select(cands, tasks, sel_split, sel_metric)

    def knots_ta(adapters, tasks):
        deltas = [ad.effective_delta() for ad in adapters]
        cands = [bl.knots(deltas, k, "ta", c) for c in grids["coef"]]
        return _select(cands, tasks, sel_split, sel_metric)

    def knots_ties(adapters, tasks):
        deltas = [ad.effective_delta() for ad in adapters]
        cands = [bl.knots(deltas, k, "ties", c, dens)
                 for c in grids["ties_coef"] for dens in grids["ties_density"]]
        return _select(cands, tasks, sel_split, sel_metric)

    def ctm_like(adapters, tasks):
        deltas = [ad.effective_delta() for ad in adapters]
        cands = [bl.ctm_like_ta(deltas, k, c) for c in grids["coef"]]
        return _select(cands, tasks, sel_split, sel_metric)

    def regmean_factory(trunc):
        def f(adapters, tasks):
            deltas = [ad.effective_delta() for ad in adapters]
            xs = [t.x_cal for t in tasks]
            cands = [bl.regmean(deltas, xs, a, k, trunc) for a in grids["regmean_alpha"]]
            return _select(cands, tasks, sel_split, sel_metric)
        return f

    def wrrr_uniform_rel(adapters, tasks):
        qt = quad_tasks(_with_deltas(tasks, adapters), "cal", relative=True)
        res = wrrr(qt, [1.0 / len(qt)] * len(qt), k, mode=wrrr_mode)
        return bl.MethodOutput(
            "wrrr_uniform_rel", res.M, k,
            bl._access(calibration_inputs=True, task_identity_for_calibration=True),
            {}, {"unique": res.unique, "gram_eig_min": res.gram_eig_min,
                 "spectral_gap_rel": res.spectral_gap_rel}), []

    def minimax_rel(adapters, tasks):
        qt = quad_tasks(_with_deltas(tasks, adapters), "cal", relative=True)
        dc = DualConfig(**{**dual_cfg.__dict__, "mode": wrrr_mode})
        res = solve_minimax(qt, k, dc, refine_cfg)
        return bl.MethodOutput(
            "minimax_rel", res.M, k,
            bl._access(calibration_inputs=True, task_identity_for_calibration=True),
            {}, {"cal_upper_bound": res.upper_bound, "cal_lower_bound": res.lower_bound,
                 "cal_gap_rel": res.gap_rel, "lb_weights": res.lb_weights,
                 "n_wrrr_solves": res.n_wrrr_solves,
                 "n_objective_evals": res.n_objective_evals,
                 "dual_nonunique_steps": res.dual.nonunique_steps,
                 "refine_log": res.refine_log}), []

    def factor_avg(adapters, tasks):
        return bl.factor_average(adapters, k), []

    table = {
        "zero_base_model": zero,
        "ta_svd": ta_svd,
        "knots_ta_trunc": knots_ta,
        "knots_ties_trunc": knots_ties,
        "ctm_like_ta": ctm_like,
        "regmean_full_rank": regmean_factory("none"),
        "regmean_euclid_trunc": regmean_factory("euclid"),
        "regmean_whitened_trunc": regmean_factory("whitened"),
        "wrrr_uniform_rel": wrrr_uniform_rel,
        "minimax_rel": minimax_rel,
        "factor_average_diag": factor_avg,
    }
    return {name: table[name] for name in cfg["methods"]}


def _with_deltas(tasks: Sequence[TaskSplit], adapters: Sequence[LoRAAdapter]) -> List[TaskSplit]:
    """Tasks whose deltas are recomputed from the (possibly re-gauged) adapters."""
    return [TaskSplit(t.name, ad.effective_delta(), t.x_cal, t.x_test, t.x_dev)
            for t, ad in zip(tasks, adapters)]


def run_all(adapters: Sequence[LoRAAdapter], tasks: Sequence[TaskSplit],
            cfg: Dict[str, object]) -> List[Dict[str, object]]:
    methods = build_methods(cfg)
    k = int(cfg["k"])
    tasks_eval = _with_deltas(tasks, adapters)
    rows = []
    for name, fn in methods.items():
        tracing = tracemalloc.is_tracing()
        if tracing:
            tracemalloc.reset_peak()
            base_mem = tracemalloc.get_traced_memory()[0]
        t0 = time.perf_counter()
        try:
            out, grid_log = fn(adapters, tasks_eval)
        except Exception as exc:  # preserve failures instead of dropping the method
            rows.append({"method": name, "status": "FAILED",
                         "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc(),
                         "wall_time_s": time.perf_counter() - t0})
            continue
        wall = time.perf_counter() - t0
        peak = (tracemalloc.get_traced_memory()[1] - base_mem) if tracing else None
        ev = evaluate(out.M, tasks_eval, splits=("cal", "dev", "test"))
        rank = la.numerical_rank(out.M) if la.frob(out.M) > 0 else 0
        acct = param_account(adapters, merged_rank=rank).as_dict()
        rows.append({
            "method": name,
            "status": "OK",
            "final_rank": rank,
            "rank_budget": k,
            "within_rank_budget": rank <= k,
            "access": out.access,
            "selected_hparams": out.hparams,
            "tuning_grid_evaluations": len(grid_log) if grid_log else 0,
            "tuning_log": grid_log,
            "wall_time_s": wall,
            "peak_tracemalloc_bytes": peak,
            "param_accounting": acct,
            "eval": ev,
            "info": out.info,
            "M": out.M,
        })
    return rows
