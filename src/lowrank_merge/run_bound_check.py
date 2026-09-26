"""Stage 3: numerical check of the draft's deterministic bounds on stage-2 instances.

    PYTHONPATH=src python3 -m lowrank_merge.run_bound_check --config configs/p2_stage3_bound_check.json

Re-analysis only. Stage 2 stored fitted matrices as SHA-256; they are
regenerated with the same code path (quad_task + solve_minimax with the fixture
solver settings) and accepted only if the hash matches the recorded value.
Checks, per (teacher, draw, arm):
  * Lemma 1 / Lemma 3 pointwise at W_hat and W* for every task;
  * Corollary 1 (Emp/Pop arm) or Lemma 3 + Lemma 2 (Emp/Emp arm):
        F(W_hat) - F(W*) <= 2*delta + eps_opt,
    with delta the uniform-deviation bound on {rank <= r, ||W||_F <= B}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import time
import traceback
from typing import Dict, List

from . import linalg as la
from .manifest import ROOT, environment, git_info, peak_rss_bytes, sha256_file, source_hashes, write_json
from .minimax import DualConfig, RefineConfig, solve_minimax
from .objectives import gram
from .population import input_map, population_moment, population_relative_errors, quad_task, resample_calibration
from .run_diag import CapExceeded, _hash_matrix, cpu_used, set_caps
from .synthetic import SyntheticSpec, make_problem

ARM_LABEL = {"emp_moment__pop_norm": "Emp/Pop (Cor.~1)", "emp_moment__emp_norm": "Emp/Emp (Lem.~3)",
             "pop_moment__pop_norm": "Pop/Pop"}


def op_norm_sym(a) -> float:
    w, _ = la.sym_eig(a)
    return max(abs(w[0]), abs(w[-1]))


def e_val(w, d, s, denom) -> float:
    y = la.sub(w, d)
    return la.inner(la.matmul(y, s), y) / denom


def solver_cfgs(fixture):
    mode = str(fixture.get("wrrr_mode", "pd"))
    dual = DualConfig(**{**fixture["dual"], "mode": mode})
    raw = dict(fixture["refine"])
    raw["block_dual"] = DualConfig(**raw["block_dual"])
    raw["local_taus"] = tuple(raw["local_taus"])
    return dual, RefineConfig(**raw)


def fit(tasks_d, fit_mom, norm_mom, k, dual, refine):
    qts = [quad_task(f"task{i}", d, fm, nm) for i, (d, fm, nm) in enumerate(zip(tasks_d, fit_mom, norm_mom))]
    res = solve_minimax(qts, k, dual, refine)
    return res, qts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg_path = os.path.abspath(args.config)
    cfg = json.load(open(cfg_path))
    caps = cfg["resource_caps"]
    cap_info = set_caps(int(caps["cpu_seconds_total"]), int(caps["peak_rss_bytes"]))
    cpu_cap = float(caps["cpu_seconds_total"])
    fixture = json.load(open(os.path.join(ROOT, cfg["fixture_config"])))
    s2 = json.load(open(os.path.join(ROOT, cfg["source_run"], "results.json")))
    rec_hash = {(r["teacher"], r["draw"], r["arm"], r["method"]): r["M_sha256"]
                for r in s2["rows"] if r["status"] == "OK"}
    rec_row = {(r["teacher"], r["draw"], r["arm"], r["method"]): r for r in s2["rows"] if r["status"] == "OK"}
    run_dir = args.out or os.path.join(ROOT, "runs", f"{cfg['experiment']}_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}")
    os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "raw_log.jsonl"), "a")

    def log(ev):
        logf.write(json.dumps({"ts": dt.datetime.now(dt.timezone.utc).isoformat(), **ev}, default=str) + "\n")
        logf.flush()

    t0 = time.perf_counter()
    manifest = {"run_dir": os.path.relpath(run_dir, ROOT),
                "command": "PYTHONPATH=src python3 -m lowrank_merge.run_bound_check --config " + os.path.relpath(cfg_path, ROOT),
                "nature": cfg["nature"], "config_file_sha256": sha256_file(cfg_path),
                "source_results_sha256": sha256_file(os.path.join(ROOT, cfg["source_run"], "results.json")),
                "git": git_info(), "source_sha256": source_hashes(), "environment": environment(),
                "caps": cap_info, "install_download_cost": "none", "status": "RUNNING",
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(os.path.join(run_dir, "manifest.json"), manifest)
    spec = SyntheticSpec(**fixture["synthetic"])
    k = int(fixture["k"])
    dual, refine = solver_cfgs(fixture)
    rows: List[Dict[str, object]] = []
    n_refits = n_match = 0
    status = "COMPLETED"
    try:
        for teacher in cfg["teachers"]:
            prob = make_problem(spec, teacher)
            deltas = [t.delta for t in prob.tasks]
            pops = [population_moment(spec, teacher, t) for t in range(spec.n_tasks)]
            lmaps = [input_map(spec, teacher, t) for t in range(spec.n_tasks)]
            d_pop = [la.inner(la.matmul(d, s), d) for d, s in zip(deltas, pops)]
            s_op = [op_norm_sym(s) for s in pops]
            # reference solution W* (pop/pop), regenerated and hash-checked
            res_star, _ = fit(deltas, pops, pops, k, dual, refine)
            n_refits += 1
            ok_star = _hash_matrix(res_star.M) == rec_hash[(teacher, "all_draws", "pop_moment__pop_norm", "minimax_rel")]
            n_match += ok_star
            w_star = res_star.M
            f_star = max(population_relative_errors(w_star, deltas, lmaps))
            log({"event": "reference_refit", "teacher": teacher, "hash_match": ok_star, "F_star": f_star,
                 "lb_pop": res_star.lower_bound})
            for draw in cfg["draws"]:
                xs = ([t.x_cal for t in prob.tasks] if draw == "orig_seen" else
                      [resample_calibration(spec, teacher, t, int(draw[1:])) for t in range(spec.n_tasks)])
                emp = [gram(x) for x in xs]
                d_emp = [la.inner(la.matmul(d, s), d) for d, s in zip(deltas, emp)]
                dS = [la.sub(s, sh) for s, sh in zip(pops, emp)]
                dS_op = [op_norm_sym(a) for a in dS]
                for arm in cfg["arms_checked"]:
                    if cpu_used() > cpu_cap - 60:
                        rows.append({"teacher": teacher, "draw": draw, "arm": arm, "status": "NOT_RUN_CAP"})
                        log({"event": "cell", **rows[-1]})
                        status = "STOPPED_AT_CPU_CAP"
                        continue
                    c0 = cpu_used()
                    norm = d_pop if arm == "emp_moment__pop_norm" else d_emp
                    norm_mom = pops if arm == "emp_moment__pop_norm" else emp
                    res, qts = fit(deltas, emp, norm_mom, k, dual, refine)
                    n_refits += 1
                    ok = _hash_matrix(res.M) == rec_hash[(teacher, draw, arm, "minimax_rel")]
                    n_match += ok
                    w_hat = res.M
                    f_hat_pop = max(population_relative_errors(w_hat, deltas, lmaps))
                    eps = res.upper_bound - res.lower_bound
                    bnorm = max(la.frob(w_hat), la.frob(w_star))
                    rad = [(bnorm + la.frob(d)) ** 2 for d in deltas]
                    if arm == "emp_moment__pop_norm":
                        per_task = [dS_op[i] * rad[i] / d_pop[i] for i in range(len(deltas))]
                    else:
                        per_task = [dS_op[i] / d_emp[i] * rad[i] * (1.0 + s_op[i] * la.frob2(deltas[i]) / d_pop[i])
                                    for i in range(len(deltas))]
                    delta = max(per_task)
                    bound = 2.0 * delta + eps
                    excess = f_hat_pop - f_star
                    # pointwise checks at W_hat and W*
                    pw = []
                    for wname, w in (("W_hat", w_hat), ("W_star", w_star)):
                        for i, d in enumerate(deltas):
                            true_e = e_val(w, d, pops[i], d_pop[i])
                            if arm == "emp_moment__pop_norm":
                                emp_e = e_val(w, d, emp[i], d_pop[i])
                                rhs = dS_op[i] * la.frob2(la.sub(w, d)) / d_pop[i]
                            else:
                                emp_e = e_val(w, d, emp[i], d_emp[i])
                                rhs = dS_op[i] / d_emp[i] * (la.frob2(la.sub(w, d)) + true_e * la.frob2(d))
                            lhs = abs(true_e - emp_e)
                            pw.append({"at": wname, "task": i, "lhs": lhs, "rhs": rhs,
                                       "holds": lhs <= rhs * (1 + 1e-12) + 1e-15})
                    row = {"teacher": teacher, "draw": draw, "arm": arm, "status": "OK", "hash_match": ok,
                           "hash_match_W_star": ok_star,
                           "F_pop_W_hat": f_hat_pop, "recorded_F_pop": rec_row[(teacher, draw, arm, "minimax_rel")]["F_pop"],
                           "F_pop_W_star": f_star, "eps_opt": eps, "B": bnorm,
                           "rel_op_err": [dS_op[i] / s_op[i] for i in range(len(deltas))],
                           "delta": delta, "bound": bound, "excess": excess,
                           "bound_holds": excess <= bound, "bound_over_excess": (bound / excess) if excess > 0 else None,
                           "pointwise": pw, "pointwise_all_hold": all(p["holds"] for p in pw),
                           "cpu_s": cpu_used() - c0}
                    rows.append(row)
                    log({"event": "cell", **row})
    except CapExceeded as exc:
        status = "FAILED_CPU_CAP"
        log({"event": "cap_exceeded", "error": str(exc)})
    except MemoryError as exc:
        status = "FAILED_OOM"
        log({"event": "oom", "error": repr(exc)})
    except Exception as exc:
        status = "FAILED"
        log({"event": "run_failed", "error": repr(exc), "traceback": traceback.format_exc()})

    ok_rows = [r for r in rows if r["status"] == "OK"]
    table = []
    for arm in cfg["arms_checked"]:
        rr = [r for r in ok_rows if r["arm"] == arm]
        if not rr:
            continue
        rel = [max(r["rel_op_err"]) for r in rr]
        table.append({"arm": arm, "arm_label": ARM_LABEL[arm], "n": len(rr),
                      "rel_op_err_median": statistics.median(rel), "rel_op_err_min": min(rel), "rel_op_err_max": max(rel),
                      "bound_median": statistics.median(r["bound"] for r in rr),
                      "bound_min": min(r["bound"] for r in rr), "bound_max": max(r["bound"] for r in rr),
                      "excess_median": statistics.median(r["excess"] for r in rr),
                      "excess_min": min(r["excess"] for r in rr), "excess_max": max(r["excess"] for r in rr),
                      "n_holds": sum(1 for r in rr if r["bound_holds"])})
    pw_all = [p for r in ok_rows for p in r["pointwise"]]
    rel_all = [max(r["rel_op_err"]) for r in ok_rows]
    summary = {
        "n_cells": len(ok_rows),
        "n_bounds_hold": sum(1 for r in ok_rows if r["bound_holds"]),
        "n_pointwise_checks": len(pw_all),
        "n_pointwise_hold": sum(1 for p in pw_all if p["holds"]),
        "median_max_rel_op_err": statistics.median(rel_all) if rel_all else None,
        "min_bound": min((r["bound"] for r in ok_rows), default=None),
        "max_excess": max((r["excess"] for r in ok_rows), default=None),
        "median_B": statistics.median(r["B"] for r in ok_rows) if ok_rows else None,
        "max_abs_F_pop_diff_vs_recorded": max((abs(r["F_pop_W_hat"] - r["recorded_F_pop"]) for r in ok_rows), default=None),
    }
    results = {"status": status, "n_refits": n_refits, "n_hash_match": n_match, "summary": summary,
               "table": table, "rows": rows}
    write_json(os.path.join(run_dir, "results.json"), results)
    manifest.update({"status": status, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                     "wall_time_s": time.perf_counter() - t0, "cpu_time_s": cpu_used(),
                     "peak_rss_bytes": peak_rss_bytes(), "n_refits": n_refits, "n_hash_match": n_match})
    write_json(os.path.join(run_dir, "manifest.json"), manifest)
    logf.close()
    print(f"{status}: {os.path.relpath(run_dir, ROOT)} refits={n_refits} hash_match={n_match} "
          f"cpu={manifest['cpu_time_s']:.1f}s")
    return 0 if status == "COMPLETED" and n_match == n_refits else 1


if __name__ == "__main__":
    raise SystemExit(main())
