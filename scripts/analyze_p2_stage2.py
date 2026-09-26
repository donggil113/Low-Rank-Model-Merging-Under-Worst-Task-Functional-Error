"""Derived tables for the P2 stage-2 diagnostic (reads results.json only).

    python3 scripts/analyze_p2_stage2.py runs/p2_stage2_moment_normalizer_diag_20260926T155649Z
"""

import json
import statistics
import sys

d = sys.argv[1]
r = json.load(open(f"{d}/results.json"))
print("status", r["status"], r["cell_status_counts"])
print("bound audit", json.dumps(r["bound_audit"], indent=1, sort_keys=True))
for c in r["pop_pop_consistency"]:
    print("pop/pop", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
for arm, s in r["arms"].items():
    print(f"\n== {arm}: {s['verdict']} mean_delta={s['mean_delta']:+.4f} "
          f"sd={statistics.stdev(s['cells']):.3f} n_teachers_pos={s['n_teachers_delta_pos']}")
    print("   teacher mean delta", {k: round(v, 4) for k, v in s["teacher_mean_delta"].items()})
    print("   cells", [round(x, 3) for x in s["cells"]])
    print("   mean F_pop", {k: round(v, 4) for k, v in s["mean_F_pop"].items()})
    print("   orig_seen", {t: round(x["delta"], 4) for t, x in s["orig_seen_draw"].items()})
rows = [x for x in r["rows"] if x["status"] == "OK"]
by = {(x["teacher"], x["draw"], x["arm"], x["method"]): x for x in rows}
print("\norig_seen emp/emp: previously-seen test-sample worst vs population worst vs fit max")
for t in (0, 1, 2):
    for m in ("wrrr_uniform_rel", "minimax_rel"):
        x = by[(t, "orig_seen", "emp_moment__emp_norm", m)]
        print(" ", t, m, "test", round(max(x["test_rel_per_task_PREVIOUSLY_SEEN"]), 3),
              "pop", round(x["F_pop"], 3), "fit_max", round(x["fit_objective_max"], 3))
new = [(t, dr) for t in (0, 1, 2) for dr in ("r101", "r102", "r103")]
for arm in ("emp_moment__emp_norm", "emp_moment__pop_norm", "pop_moment__emp_norm"):
    for m in ("wrrr_uniform_rel", "minimax_rel"):
        xs = [by[(t, dr, arm, m)] for t, dr in new]
        print(f"optimism {arm} {m}: mean fit max {statistics.fmean(x['fit_objective_max'] for x in xs):.3f}"
              f" -> mean F_pop {statistics.fmean(x['F_pop'] for x in xs):.3f}")
ratios = []
for t, dr in new + [(t, "orig_seen") for t in (0, 1, 2)]:
    emp = by[(t, dr, "emp_moment__emp_norm", "minimax_rel")]["normalizer_used"]
    pop = by[(t, "all_draws", "pop_moment__pop_norm", "minimax_rel")]["normalizer_used"]
    ratios += [a / b for a, b in zip(emp, pop)]
print(f"\nempirical/population normaliser ratio: min {min(ratios):.3f} max {max(ratios):.3f}")
lam = [(x["teacher"], x["draw"], x["arm"], [round(v, 3) for v in x["audit"]["lambda_star"]])
       for x in rows if x["method"] == "minimax_rel"]
for item in sorted(lam):
    print("lambda*", *item)
cpu = [x["cpu_s"] for x in rows]
print(f"\ncpu per fit min {min(cpu):.3f} max {max(cpu):.2f} sum {sum(cpu):.1f}")
