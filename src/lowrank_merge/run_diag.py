"""P2 stage 2: moment / normaliser 2x2 diagnostic on the synthetic fixture.

    PYTHONPATH=src python3 -m lowrank_merge.run_diag \
        --config configs/p2_stage2_moment_normalizer_diag.json

Development diagnostic, not a confirmatory experiment (see the config). For
each teacher (the recorded fixture's problem seeds) and each calibration draw
(the original, already-seen draw plus pre-registered resamples) it fits
wrrr_uniform_rel and minimax_rel under four arms {empirical, population}
moment x {empirical, population} normaliser, audits the minimax bounds with
the independent raw-data Izenman reference at lambda*, and evaluates every
fit on the same population worst-task relative error.

Caps are enforced in-process: RLIMIT_CPU / RLIMIT_AS before any work, a
SIGXCPU handler that converts the hard stop into a recorded failure, and a
budget gate before each fit that records NOT_RUN_CAP instead of starting it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import resource
import signal
import statistics
import time
import traceback
from typing import Dict, List, Optional

from . import linalg as la
from .manifest import (ROOT, environment, git_info, peak_rss_bytes, sha256_file,
                       sha256_json, source_hashes, write_json)
from .minimax import DualConfig, RefineConfig, solve_minimax
from .objectives import functional_error, adapter_energy, gram
from .population import (input_map, mc_second_moment, mc_tolerance_ok, population_moment,
                         population_relative_errors, quad_task, resample_calibration)
from .references import RawTask, raw_error, raw_weighted, rrr_izenman_raw
from .synthetic import SyntheticSpec, make_problem
from .wrrr import wrrr


class CapExceeded(RuntimeError):
    pass


def _on_sigxcpu(signum, frame):  # pragma: no cover - only on cap violation
    raise CapExceeded("RLIMIT_CPU soft limit reached (SIGXCPU)")


def cpu_used() -> float:
    ch = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + ch.ru_utime + ch.ru_stime


def set_caps(cpu_s: int, as_bytes: int) -> Dict[str, object]:
    signal.signal(signal.SIGXCPU, _on_sigxcpu)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 30))
    resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    return {"RLIMIT_CPU": resource.getrlimit(resource.RLIMIT_CPU),
            "RLIMIT_AS": resource.getrlimit(resource.RLIMIT_AS),
            "threads_env": "1 (pure-Python code is single-threaded)", "workers": 1}


def _hash_matrix(m) -> str:
    h = hashlib.sha256()
    for row in m:
        h.update(",".join(float(x).hex() for x in row).encode())
    return h.hexdigest()


def _sv_ratio(m, k: int) -> Optional[float]:
    s = la.singular_values(m)
    if not s or s[0] == 0.0 or len(s) <= k:
        return None
    return s[k] / s[0]


def audit_minimax(res, qts, raws, k: int, mode: str) -> Dict[str, object]:
    """Bound audit at lambda* with the independent raw-data reference."""
    lam = res.lb_weights
    at = wrrr(qts, lam, k, mode=mode)
    ref_m = rrr_izenman_raw(raws, lam, k)
    g_ref = raw_weighted(ref_m, raws, lam)
    g_raw_solver = raw_weighted(at.M, raws, lam)
    ub_raw = max(raw_error(res.M, r) for r in raws)
    sv = at.whitened_singular_values
    return {
        "lambda_star": lam,
        "lambda_min": min(lam),
        "lambda_sum_minus_1": sum(lam) - 1.0,
        "lower_bound_quadform": res.lower_bound,
        "g_raw_eval_of_solver_M": g_raw_solver,
        "g_independent_izenman": g_ref,
        "g_rel_discrepancy": abs(res.lower_bound - g_ref) / max(abs(g_ref), 1e-300),
        "M_rel_diff_vs_izenman": la.frob(la.sub(at.M, ref_m)) / max(la.frob(ref_m), 1e-300),
        "upper_bound_quadform": res.upper_bound,
        "upper_bound_raw_eval": ub_raw,
        "ub_rel_discrepancy": abs(res.upper_bound - ub_raw) / max(abs(ub_raw), 1e-300),
        "gap_rel": res.gap_rel,
        "mode": mode,
        "S_eig_min": at.gram_eig_min,
        "S_eig_max": at.gram_eig_max,
        "S_cond": at.gram_eig_max / at.gram_eig_min if at.gram_eig_min > 0 else None,
        "S_null_dim": at.gram_null_dim,
        "whitened_sigma_k": sv[k - 1],
        "whitened_sigma_k_plus_1": sv[k] if len(sv) > k else None,
        "unique_at_lambda_star": at.unique,
        "M_sigma_k1_over_sigma_1": _sv_ratio(res.M, k),
        "rank_M": la.numerical_rank(res.M),
        "n_wrrr_solves": res.n_wrrr_solves,
        "refine": res.refine_log[0]["stage"] if res.refine_log else None,
        "solver_failure_refuted": (res.gap_rel <= 1e-6
                                   and abs(res.lower_bound - g_ref) <= 1e-9 * max(abs(g_ref), 1e-300)
                                   and la.numerical_rank(res.M) <= k),
    }


def fit_cell(teacher, draw, arm, tasks, xcal, pop_moments, lmaps, spec, fixture_cfg,
             k: int) -> List[Dict[str, object]]:
    """Fit both methods for one (teacher, draw, arm) and evaluate them."""
    deltas = [t.delta for t in tasks]
    emp = [gram(x) for x in xcal]
    fit_mom = emp if arm["moment"] == "empirical" else pop_moments
    norm_mom = emp if arm["normalizer"] == "empirical" else pop_moments
    qts = [quad_task(t.name, d, fm, nm) for t, d, fm, nm in zip(tasks, deltas, fit_mom, norm_mom)]
    # raw-data twin of the fit objective for the independent reference
    d_in = spec.d_in
    raws = []
    for d, x, lmap, q in zip(deltas, xcal, lmaps, qts):
        xr = x if arm["moment"] == "empirical" else la.scale(d_in ** 0.5, lmap)
        raws.append(RawTask(d, xr, q.norm))
    mode = str(fixture_cfg.get("wrrr_mode", "pd"))
    dual_cfg = DualConfig(**{**fixture_cfg["dual"], "mode": mode})
    refine_raw = dict(fixture_cfg["refine"])
    refine_raw["block_dual"] = DualConfig(**refine_raw["block_dual"])
    refine_raw["local_taus"] = tuple(refine_raw["local_taus"])
    refine_cfg = RefineConfig(**refine_raw)
    out = []
    for method in ("wrrr_uniform_rel", "minimax_rel"):
        c0, w0 = cpu_used(), time.perf_counter()
        audit = None
        if method == "wrrr_uniform_rel":
            r = wrrr(qts, [1.0 / len(qts)] * len(qts), k, mode=mode)
            m = r.M
            extra = {"unique": r.unique, "S_eig_min": r.gram_eig_min, "S_eig_max": r.gram_eig_max,
                     "whitened_sigma_k": r.whitened_singular_values[k - 1],
                     "whitened_sigma_k_plus_1": r.whitened_singular_values[k]}
        else:
            r = solve_minimax(qts, k, dual_cfg, refine_cfg)
            m = r.M
            extra = {}
            audit = audit_minimax(r, qts, raws, k, mode)
        cpu_fit, wall_fit = cpu_used() - c0, time.perf_counter() - w0
        pop_rel = population_relative_errors(m, deltas, lmaps)
        fit_vals = [q.value(m) for q in qts]
        emp_rel = [functional_error(m, d, x) / adapter_energy(d, x) for d, x in zip(deltas, xcal)]
        test_rel = [functional_error(m, t.delta, t.x_test) / adapter_energy(t.delta, t.x_test)
                    for t in tasks]
        out.append({
            "teacher": teacher, "draw": draw, "arm": arm["name"], "method": method,
            "status": "OK",
            "F_pop": max(pop_rel), "mean_pop": sum(pop_rel) / len(pop_rel),
            "pop_rel_per_task": pop_rel,
            "fit_objective_per_task": fit_vals, "fit_objective_max": max(fit_vals),
            "emp_cal_rel_per_task": emp_rel,
            "test_rel_per_task_PREVIOUSLY_SEEN": test_rel,
            "normalizer_used": [q.norm for q in qts],
            "rank_M": la.numerical_rank(m),
            "M_sha256": _hash_matrix(m),
            "cpu_s": cpu_fit, "wall_s": wall_fit,
            "audit": audit, **extra,
        })
    return out


def classify(deltas_by_teacher: Dict[int, List[float]], rules) -> Dict[str, object]:
    allv = [d for v in deltas_by_teacher.values() for d in v]
    if not allv:
        return {"verdict": "NOT_EVALUATED"}
    mean = statistics.fmean(allv)
    tmeans = {t: statistics.fmean(v) for t, v in deltas_by_teacher.items() if v}
    n_pos = sum(1 for v in tmeans.values() if v > 0)
    complete = all(len(v) == 3 for v in deltas_by_teacher.values()) and len(deltas_by_teacher) == 3
    if not complete:
        verdict = "INCOMPLETE_CELLS"
    elif mean > 0.02 and n_pos >= 2:
        verdict = "PROBLEM_PRESENT"
    elif mean <= 0.0:
        verdict = "PROBLEM_ABSENT"
    else:
        verdict = "AMBIGUOUS"
    return {"verdict": verdict, "mean_delta": mean, "teacher_mean_delta": tmeans,
            "n_teachers_delta_pos": n_pos, "n_cells": len(allv), "cells": allv}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg_path = os.path.abspath(args.config)
    with open(cfg_path) as f:
        cfg = json.load(f)
    fixture_path = os.path.join(ROOT, cfg["inherits_fixture_config"])
    with open(fixture_path) as f:
        fixture = json.load(f)
    caps = cfg["resource_caps"]
    cap_info = set_caps(int(caps["cpu_seconds_total"]), int(caps["peak_rss_bytes"]))
    cpu_cap = float(caps["cpu_seconds_total"])
    run_dir = args.out or os.path.join(
        ROOT, "runs", f"{cfg['experiment']}_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}")
    os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "raw_log.jsonl"), "a")

    def log(ev):
        logf.write(json.dumps({"ts": dt.datetime.now(dt.timezone.utc).isoformat(), **ev},
                              default=str) + "\n")
        logf.flush()

    t_wall0 = time.perf_counter()
    manifest = {
        "run_dir": os.path.relpath(run_dir, ROOT),
        "command": "PYTHONPATH=src python3 -m lowrank_merge.run_diag --config "
                   + os.path.relpath(cfg_path, ROOT),
        "nature": cfg["nature"],
        "config_path": os.path.relpath(cfg_path, ROOT),
        "config_file_sha256": sha256_file(cfg_path),
        "fixture_config_sha256": sha256_file(fixture_path),
        "git": git_info(), "source_sha256": source_hashes(), "environment": environment(),
        "caps": cap_info, "install_download_cost": "none (no installs, no downloads)",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "RUNNING",
    }
    write_json(os.path.join(run_dir, "manifest.json"), manifest)
    spec = SyntheticSpec(**fixture["synthetic"])
    k = int(cfg["k"])
    status = "COMPLETED"
    rows: List[Dict[str, object]] = []
    mc_rows: List[Dict[str, object]] = []
    data_hashes: Dict[str, str] = {}
    max_fit_cpu = 10.0
    try:
        # 1) closed-form population moment vs the generator (Monte Carlo)
        mc = cfg["monte_carlo_check"]
        for teacher in mc["teachers"]:
            for t in range(spec.n_tasks):
                pop = population_moment(spec, teacher, t)
                est = mc_second_moment(spec, teacher, t, int(mc["n_samples"]), int(mc["seed"]))
                ok, worst_z = mc_tolerance_ok(est, pop, int(mc["n_samples"]))
                rec = {"teacher": teacher, "task": t, "ok": ok, "worst_abs_z": worst_z}
                mc_rows.append(rec)
                log({"event": "mc_moment_check", **rec})
        if not all(r["ok"] for r in mc_rows):
            raise RuntimeError("population moment closed form failed the Monte Carlo check")
        # 2) fits
        recorded = cfg.get("expected_teacher_data_sha256", {})
        for teacher in cfg["fixed_from_fixture"]["teacher_seeds"]:
            prob = make_problem(spec, teacher)
            data_hashes[f"teacher{teacher}"] = prob.meta["data_sha256"]
            if str(teacher) in recorded and recorded[str(teacher)] != prob.meta["data_sha256"]:
                raise RuntimeError(f"teacher {teacher} differs from the recorded fixture")
            lmaps = [input_map(spec, teacher, t) for t in range(spec.n_tasks)]
            pops = [population_moment(spec, teacher, t) for t in range(spec.n_tasks)]
            draws = []
            if cfg["include_original_draw"]["enabled"]:
                draws.append(("orig_seen", [t.x_cal for t in prob.tasks]))
            for rs in cfg["calibration_resample_seeds"]:
                xs = [resample_calibration(spec, teacher, t, rs) for t in range(spec.n_tasks)]
                draws.append((f"r{rs}", xs))
            for draw, xs in draws:
                data_hashes[f"teacher{teacher}_{draw}"] = hashlib.sha256(
                    "".join(_hash_matrix(x) for x in xs).encode()).hexdigest()
            done_pop = False
            for draw, xs in draws:
                for arm in cfg["arms"]:
                    is_pp = arm["moment"] == "population" and arm["normalizer"] == "population"
                    if is_pp and done_pop:
                        continue  # identical for every draw: computed once per teacher
                    draw_label = "all_draws" if is_pp else draw
                    if cpu_used() + 2.5 * max_fit_cpu > cpu_cap - 30:
                        for method in ("wrrr_uniform_rel", "minimax_rel"):
                            rec = {"teacher": teacher, "draw": draw_label, "arm": arm["name"],
                                   "method": method, "status": "NOT_RUN_CAP"}
                            rows.append(rec)
                            log({"event": "fit", **rec})
                        status = "STOPPED_AT_CPU_CAP"
                        continue
                    try:
                        recs = fit_cell(teacher, draw_label, arm, prob.tasks, xs, pops, lmaps,
                                        spec, fixture, k)
                    except CapExceeded:
                        raise
                    except Exception as exc:
                        recs = [{"teacher": teacher, "draw": draw_label, "arm": arm["name"],
                                 "method": m_, "status": "FAILED", "error": repr(exc),
                                 "traceback": traceback.format_exc()}
                                for m_ in ("wrrr_uniform_rel", "minimax_rel")]
                    for rec in recs:
                        rows.append(rec)
                        log({"event": "fit", **rec})
                        if rec.get("status") == "OK":
                            max_fit_cpu = max(max_fit_cpu, rec["cpu_s"])
                    if is_pp:
                        done_pop = True
    except CapExceeded as exc:
        status = "FAILED_CPU_CAP"
        log({"event": "cap_exceeded", "error": str(exc)})
    except MemoryError as exc:
        status = "FAILED_OOM"
        log({"event": "oom", "error": repr(exc)})
    except Exception as exc:
        status = "FAILED"
        log({"event": "run_failed", "error": repr(exc), "traceback": traceback.format_exc()})

    # ---- aggregation per pre-registered rules --------------------------------
    ok = [r for r in rows if r.get("status") == "OK"]
    by = {(r["teacher"], r["draw"], r["arm"], r["method"]): r for r in ok}
    new_draws = [f"r{s}" for s in cfg["calibration_resample_seeds"]]
    arms_summary = {}
    for arm in cfg["arms"]:
        name = arm["name"]
        is_pp = arm["moment"] == "population" and arm["normalizer"] == "population"
        per_teacher: Dict[int, List[float]] = {}
        f_means = {"wrrr_uniform_rel": [], "minimax_rel": []}
        for teacher in cfg["fixed_from_fixture"]["teacher_seeds"]:
            draws = ["all_draws"] * 3 if is_pp else new_draws
            for dr in draws:
                a = by.get((teacher, dr, name, "minimax_rel"))
                b = by.get((teacher, dr, name, "wrrr_uniform_rel"))
                if a and b:
                    per_teacher.setdefault(teacher, []).append(a["F_pop"] - b["F_pop"])
                    f_means["minimax_rel"].append(a["F_pop"])
                    f_means["wrrr_uniform_rel"].append(b["F_pop"])
        s = classify(per_teacher, cfg["decision_rules"])
        s["note"] = ("population/population uses no calibration samples: one value per teacher "
                     "repeated 3x in the cell list (not independent)" if is_pp else "")
        s["mean_F_pop"] = {m_: (statistics.fmean(v) if v else None) for m_, v in f_means.items()}
        orig = {}
        for teacher in cfg["fixed_from_fixture"]["teacher_seeds"]:
            dr = "all_draws" if is_pp else "orig_seen"
            a = by.get((teacher, dr, name, "minimax_rel"))
            b = by.get((teacher, dr, name, "wrrr_uniform_rel"))
            if a and b:
                orig[teacher] = {"F_pop_minimax": a["F_pop"], "F_pop_uniform": b["F_pop"],
                                 "delta": a["F_pop"] - b["F_pop"]}
        s["orig_seen_draw"] = orig
        arms_summary[name] = s
    pp_checks = []
    for teacher in cfg["fixed_from_fixture"]["teacher_seeds"]:
        a = by.get((teacher, "all_draws", "pop_moment__pop_norm", "minimax_rel"))
        b = by.get((teacher, "all_draws", "pop_moment__pop_norm", "wrrr_uniform_rel"))
        if a and b:
            slack = a["audit"]["upper_bound_quadform"] - a["audit"]["lower_bound_quadform"]
            pp_checks.append({"teacher": teacher, "F_pop_minimax": a["F_pop"],
                              "F_pop_uniform": b["F_pop"], "minimax_gap_abs": slack,
                              "consistent": a["F_pop"] <= b["F_pop"] + slack + 1e-12,
                              "fit_objective_equals_eval": abs(a["fit_objective_max"] - a["F_pop"])})
    audits = [r["audit"] for r in ok if r["method"] == "minimax_rel"]
    audit_summary = {
        "n_minimax_fits": len(audits),
        "all_solver_failure_refuted": all(a["solver_failure_refuted"] for a in audits) if audits else None,
        "max_gap_rel": max((a["gap_rel"] for a in audits), default=None),
        "max_g_rel_discrepancy_vs_izenman": max((a["g_rel_discrepancy"] for a in audits), default=None),
        "max_ub_rel_discrepancy_raw": max((a["ub_rel_discrepancy"] for a in audits), default=None),
        "max_M_sigma_k1_over_sigma_1": max((a["M_sigma_k1_over_sigma_1"] or 0.0 for a in audits), default=None),
        "min_lambda": min((a["lambda_min"] for a in audits), default=None),
        "all_unique_at_lambda_star": all(a["unique_at_lambda_star"] for a in audits) if audits else None,
        "max_S_cond": max((a["S_cond"] or 0.0 for a in audits), default=None),
        "any_null_space": any(a["S_null_dim"] for a in audits) if audits else None,
    }
    status_counts: Dict[str, int] = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    results = {"status": status, "arms": arms_summary, "pop_pop_consistency": pp_checks,
               "bound_audit": audit_summary, "mc_moment_check": mc_rows,
               "cell_status_counts": status_counts, "rows": rows}
    write_json(os.path.join(run_dir, "results.json"), results)
    manifest.update({"status": status, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                     "wall_time_s": time.perf_counter() - t_wall0, "cpu_time_s": cpu_used(),
                     "peak_rss_bytes": peak_rss_bytes(), "data_sha256": data_hashes,
                     "cell_status_counts": status_counts})
    write_json(os.path.join(run_dir, "manifest.json"), manifest)
    logf.close()
    print(f"{status}: {os.path.relpath(run_dir, ROOT)} cpu={manifest['cpu_time_s']:.1f}s "
          f"wall={manifest['wall_time_s']:.1f}s cells={status_counts}")
    return 0 if status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
