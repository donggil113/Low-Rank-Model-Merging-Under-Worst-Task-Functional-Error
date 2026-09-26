"""CPU entry point for the synthetic layer-level fixture.

    PYTHONPATH=src python -m lowrank_merge.run_cpu --config configs/cpu_synthetic.json

Writes to runs/<run_id>/:
  raw_log.jsonl   one JSON event per method x seed (incl. merged matrices,
                  tuning grids, failures), gauge checks, singular-Gram checks,
                  reference checks
  results.json    aggregated per-method results
  summary.md      human-readable table
  manifest.json   git/config/data/source hashes, environment, wall time, memory

The synthetic fixture exercises the implementation. It is not evidence about
real adapters or end-to-end task performance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import statistics
import time
import traceback
import tracemalloc
from typing import Dict, List

from . import linalg as la
from .adapters import random_gauge
from .manifest import (ROOT, environment, git_info, peak_rss_bytes, sha256_file,
                       sha256_json, source_hashes, write_json)
from .minimax import solve_minimax
from .objectives import QuadTask, quad_tasks
from .pipeline import run_all
from .references import RawTask, grid_rank1_minimax
from .synthetic import SyntheticSpec, make_problem
from .wrrr import SingularGramError, wrrr


class RawLog:
    def __init__(self, path: str) -> None:
        self.f = open(path, "a")

    def write(self, event: Dict[str, object]) -> None:
        event = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), **event}
        self.f.write(json.dumps(event, default=str) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def _strip(row: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in row.items() if k != "M"}


def gauge_check(prob, cfg, rows_ref, log: RawLog, seed: int) -> List[Dict[str, object]]:
    gc = cfg["gauge_check"]
    rng = random.Random(int(gc["seed"]) + seed)
    gauged = [ad.gauge(random_gauge(ad.rank_budget, float(gc["cond"]), rng)) for ad in prob.adapters]
    rows_g = run_all(gauged, prob.tasks, cfg)
    out = []
    by_name = {r["method"]: r for r in rows_ref}
    for rg in rows_g:
        r0 = by_name.get(rg["method"])
        if r0 is None or r0.get("status") != "OK" or rg.get("status") != "OK":
            rec = {"method": rg["method"], "status": "NOT_COMPARABLE"}
        else:
            diff = la.frob(la.sub(rg["M"], r0["M"])) / max(la.frob(r0["M"]), 1e-300)
            if la.frob(r0["M"]) == 0.0 and la.frob(rg["M"]) == 0.0:
                diff = 0.0
            tw = abs(rg["eval"]["summary"]["test"]["worst_rel"]
                     - r0["eval"]["summary"]["test"]["worst_rel"])
            rec = {"method": rg["method"], "status": "OK", "rel_diff_M": diff,
                   "abs_diff_test_worst_rel": tw,
                   "rank_before": r0["final_rank"], "rank_after": rg["final_rank"]}
        out.append(rec)
        log.write({"event": "gauge_check", "seed": seed, "cond": gc["cond"], **rec})
    return out


def singular_case(cfg, log: RawLog) -> List[Dict[str, object]]:
    sc = cfg["singular_case"]
    spec = SyntheticSpec(**{**cfg["synthetic"], "input_subspace_dim": sc["input_subspace_dim"],
                            "n_cal": sc["n_cal"]})
    prob = make_problem(spec, int(sc["seed"]))
    qt = quad_tasks(prob.tasks, "cal", relative=True)
    k = int(cfg["k"])
    lam = [1.0 / len(qt)] * len(qt)
    out = []
    for mode, ridge in [("pd", 0.0), ("pinv", 0.0)] + [("ridge", r) for r in sc["ridges"]]:
        rec: Dict[str, object] = {"mode": mode, "ridge": ridge}
        try:
            res = wrrr(qt, lam, k, mode=mode, ridge=ridge)
            rec.update({"status": "OK", "cal_weighted_objective": res.objective,
                        "penalized_objective": res.penalized_objective,
                        "exact_for_original_objective": res.exact_for_original_objective,
                        "gram_null_dim": res.gram_null_dim,
                        "gram_eig_min": res.gram_eig_min, "gram_eig_max": res.gram_eig_max,
                        "final_rank": la.numerical_rank(res.M) if la.frob(res.M) else 0})
        except SingularGramError as exc:
            rec.update({"status": "REFUSED_AS_DESIGNED", "error": str(exc)})
        out.append(rec)
        log.write({"event": "singular_gram_case", **rec})
    return out


def reference_checks(cfg, log: RawLog) -> List[Dict[str, object]]:
    rc = cfg["reference_checks"]
    rng = random.Random(int(rc["seed"]))
    out = []
    # published instance: Tantipongpipat et al. 2019, Lemma 6.2
    bs = [[[2.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 2.0]], [[2.0, -1.0], [-1.0, 2.0]]]
    res = solve_minimax([QuadTask(f"g{i}", b, b, 4.0) for i, b in enumerate(bs)], 1)
    rec = {"instance": "tantipongpipat2019_lemma6.2", "lower_bound": res.lower_bound,
           "expected_relaxation": 4 - 7 / 4, "upper_bound": res.upper_bound,
           "expected_exact": 4 - 26 / 17}
    out.append(rec)
    log.write({"event": "reference_check", **rec})
    for i in range(int(rc["n_instances"])):
        d_in = rng.choice([2, 3])
        n_t = rng.choice([2, 3, 4])
        raws, qts = [], []
        for t in range(n_t):
            delta = la.random_normal(2, d_in, rng)
            x = la.matmul(la.random_normal(d_in, d_in, rng), la.random_normal(d_in, 6, rng))
            e0 = la.frob2(la.matmul(delta, x)) / 6
            raws.append(RawTask(delta, x, e0))
        from .objectives import quad_task_from_data
        qts = [quad_task_from_data(f"t{t}", r.delta, r.x, relative=True) for t, r in enumerate(raws)]
        t0 = time.perf_counter()
        res = solve_minimax(qts, 1)
        t1 = time.perf_counter()
        ref, _ = grid_rank1_minimax(raws, n_grid=int(rc["n_grid"]))
        t2 = time.perf_counter()
        rec = {"instance": f"random_{i}", "d_out": 2, "d_in": d_in, "n_tasks": n_t,
               "lower_bound": res.lower_bound, "reference_grid_ellipsoid": ref,
               "upper_bound": res.upper_bound,
               "lb_le_ref": res.lower_bound <= ref * (1 + 1e-9),
               "ref_le_ub": ref <= res.upper_bound * (1 + 1e-9),
               "ub_minus_ref_rel": (res.upper_bound - ref) / ref,
               "ref_minus_lb_rel": (ref - res.lower_bound) / ref,
               "solver_s": t1 - t0, "reference_s": t2 - t1}
        out.append(rec)
        log.write({"event": "reference_check", **rec})
    return out


def aggregate(all_rows: Dict[int, List[Dict[str, object]]]) -> Dict[str, object]:
    methods: Dict[str, Dict[str, list]] = {}
    for seed, rows in all_rows.items():
        for r in rows:
            m = methods.setdefault(r["method"], {"seeds": [], "status": []})
            m["status"].append(r.get("status"))
            m["seeds"].append(seed)
            if r.get("status") != "OK":
                continue
            for sp in ("cal", "dev", "test"):
                for key in ("worst_rel", "mean_rel"):
                    m.setdefault(f"{sp}_{key}", []).append(r["eval"]["summary"][sp][key])
            m.setdefault("final_rank", []).append(r["final_rank"])
            m.setdefault("wall_time_s", []).append(r["wall_time_s"])
            m.setdefault("access", r["access"])
            m.setdefault("within_rank_budget", []).append(r["within_rank_budget"])
    summ = {}
    for name, m in methods.items():
        s: Dict[str, object] = {"status": m["status"], "access": m.get("access")}
        for key, vals in m.items():
            if key in ("seeds", "status", "access"):
                continue
            if vals and isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool):
                s[key] = {"mean": statistics.fmean(vals),
                          "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                          "values": vals}
            else:
                s[key] = vals
        summ[name] = s
    return summ


def paired(all_rows, target: str, metric=("test", "worst_rel")) -> Dict[str, object]:
    out = {}
    for seed, rows in all_rows.items():
        by = {r["method"]: r for r in rows if r.get("status") == "OK"}
        if target not in by:
            continue
        tv = by[target]["eval"]["summary"][metric[0]][metric[1]]
        for name, r in by.items():
            if name == target:
                continue
            d = tv - r["eval"]["summary"][metric[0]][metric[1]]
            out.setdefault(name, []).append(d)
    return {n: {"diffs_target_minus_method": v, "mean": statistics.fmean(v),
                "n_seeds_target_better": sum(1 for x in v if x < 0)} for n, v in out.items()}


def summary_md(cfg, agg, pair, gauge, singular, refs, manifest) -> str:
    lines = [f"# CPU synthetic fixture: {cfg['experiment']}", "",
             "Implementation check on synthetic data; NOT evidence about real adapters.", "",
             f"k = {cfg['k']}, seeds = {cfg['seeds']}, primary metric = {cfg['primary_metric']}", "",
             "| method | final rank | test worst_rel (mean±sd) | test mean_rel | cal worst_rel | "
             "access (cal / dev-tuning / factors) |",
             "|---|---|---|---|---|---|"]
    for name, s in agg.items():
        if "test_worst_rel" not in s:
            lines.append(f"| {name} | FAILED {s['status']} | | | | |")
            continue
        a = s["access"]
        lines.append(
            f"| {name} | {s['final_rank']} | {s['test_worst_rel']['mean']:.4f} ± "
            f"{s['test_worst_rel']['sd']:.4f} | {s['test_mean_rel']['mean']:.4f} | "
            f"{s['cal_worst_rel']['mean']:.4f} | {a['calibration_inputs']} / "
            f"{a['dev_inputs_for_tuning']} / {a['uses_factors']} |")
    lines += ["", "## Paired test worst_rel difference (minimax_rel minus method; negative = minimax better)", ""]
    for n, p in pair.items():
        lines.append(f"- {n}: mean {p['mean']:+.4f}, minimax better in "
                     f"{p['n_seeds_target_better']}/{len(p['diffs_target_minus_method'])} seeds")
    lines += ["", "## Gauge (B R, R^-1 A) invariance: max relative change of merged M over seeds", ""]
    worst: Dict[str, float] = {}
    for recs in gauge.values():
        for r in recs:
            if r["status"] == "OK":
                worst[r["method"]] = max(worst.get(r["method"], 0.0), r["rel_diff_M"])
    for n, v in worst.items():
        lines.append(f"- {n}: {v:.2e}")
    lines += ["", "## Singular combined Gram (inputs in a subspace)", ""]
    for r in singular:
        lines.append(f"- mode={r['mode']} ridge={r['ridge']}: {r['status']}"
                     + (f", cal objective {r['cal_weighted_objective']:.6f}, exact_for_original="
                        f"{r['exact_for_original_objective']}" if r["status"] == "OK" else ""))
    lines += ["", "## Reference checks (d_out=2, k=1)", ""]
    for r in refs:
        lines.append("- " + ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                                      for k, v in r.items()))
    lines += ["", f"Wall time {manifest['wall_time_s']:.1f}s, peak RSS "
              f"{manifest['peak_rss_bytes'] / 2**20:.1f} MiB, git {manifest['git']['commit']} "
              f"(dirty={manifest['git']['dirty']})", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", default=None, help="comma list overriding config seeds")
    ap.add_argument("--tracemalloc", action="store_true",
                    help="per-method Python allocation peaks (slows pure-Python code ~3x)")
    args = ap.parse_args(argv)
    cfg_path = os.path.abspath(args.config)
    with open(cfg_path) as f:
        cfg = json.load(f)
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    run_id = args.out or os.path.join(
        ROOT, "runs", f"{cfg['experiment']}_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}")
    os.makedirs(run_id, exist_ok=True)
    log = RawLog(os.path.join(run_id, "raw_log.jsonl"))
    if args.tracemalloc:
        tracemalloc.start()
    t_start = time.perf_counter()
    manifest: Dict[str, object] = {
        "run_dir": os.path.relpath(run_id, ROOT),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": "PYTHONPATH=src python -m lowrank_merge.run_cpu --config "
                   + os.path.relpath(cfg_path, ROOT) + (f" --seeds {args.seeds}" if args.seeds else "")
                   + (" --tracemalloc" if args.tracemalloc else ""),
        "config_path": os.path.relpath(cfg_path, ROOT),
        "config_file_sha256": sha256_file(cfg_path),
        "config_effective_sha256": sha256_json(cfg),
        "git": git_info(),
        "source_sha256": source_hashes(),
        "environment": environment(),
        "status": "RUNNING",
    }
    write_json(os.path.join(run_id, "manifest.json"), manifest)
    all_rows: Dict[int, List[Dict[str, object]]] = {}
    gauge: Dict[int, List[Dict[str, object]]] = {}
    data_hashes: Dict[int, str] = {}
    status = "COMPLETED"
    singular: List[Dict[str, object]] = []
    refs: List[Dict[str, object]] = []
    try:
        spec = SyntheticSpec(**cfg["synthetic"])
        for seed in cfg["seeds"]:
            prob = make_problem(spec, seed)
            data_hashes[seed] = prob.meta["data_sha256"]
            rows = run_all(prob.adapters, prob.tasks, cfg)
            for r in rows:
                log.write({"event": "method_result", "seed": seed, **r})
            all_rows[seed] = rows
            if cfg["gauge_check"]["enabled"]:
                gauge[seed] = gauge_check(prob, cfg, rows, log, seed)
            if time.perf_counter() - t_start > float(cfg["resource_caps"]["max_wall_s"]):
                status = "STOPPED_AT_WALL_CAP"
                break
        if cfg["singular_case"]["enabled"]:
            singular = singular_case(cfg, log)
        if cfg["reference_checks"]["enabled"]:
            refs = reference_checks(cfg, log)
    except Exception as exc:
        status = "FAILED"
        log.write({"event": "run_failed", "error": repr(exc), "traceback": traceback.format_exc()})
    agg = aggregate(all_rows)
    pair = paired(all_rows, "minimax_rel")
    results = {"per_method": agg, "paired_minimax_vs": pair,
               "gauge_checks": gauge, "singular_gram_case": singular,
               "reference_checks": refs, "status": status,
               "per_seed_rows": {s: [_strip(r) for r in rows] for s, rows in all_rows.items()}}
    write_json(os.path.join(run_id, "results.json"), results)
    manifest.update({
        "status": status,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_time_s": time.perf_counter() - t_start,
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_tracemalloc_bytes": (tracemalloc.get_traced_memory()[1]
                                   if tracemalloc.is_tracing() else "NOT_MEASURED"),
        "data_sha256_per_seed": data_hashes,
        "seeds_completed": list(all_rows.keys()),
    })
    write_json(os.path.join(run_id, "manifest.json"), manifest)
    with open(os.path.join(run_id, "summary.md"), "w") as f:
        f.write(summary_md(cfg, agg, pair, gauge, singular, refs, manifest))
    log.close()
    print(f"{status}: {os.path.relpath(run_id, ROOT)}  wall={manifest['wall_time_s']:.1f}s")
    return 0 if status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
