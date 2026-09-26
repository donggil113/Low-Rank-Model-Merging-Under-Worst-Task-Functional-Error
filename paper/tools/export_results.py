"""Export every number used in the paper from raw result files.

    python3 paper/tools/export_results.py

Reads only committed run artifacts and writes
  paper/generated/numbers.tex        \\newcommand macros used by abstract and body
  paper/generated/provenance.tsv     macro -> source file -> field path -> raw value
  paper/tables/*.tex                 tables
  paper/figures/fig_paired_2x2.tex   TikZ paired plot (teacher-level means)
No number in the paper should be typed by hand.
"""

import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPER = os.path.join(ROOT, "paper")
S1 = "runs/cpu_synthetic_layer_fixture_20260926T152312Z"
S2 = "runs/p2_stage2_moment_normalizer_diag_20260926T155649Z"
S3_GLOB = "runs/p2_stage3_bound_check_"
ADM = "runs/p2_stage2_adapter_admission_20260926T155226Z/admission.json"

ARMS = [("emp_moment__emp_norm", "Emp/Emp", "EE"), ("emp_moment__pop_norm", "Emp/Pop", "EP"),
        ("pop_moment__emp_norm", "Pop/Emp", "PE"), ("pop_moment__pop_norm", "Pop/Pop", "PP")]
LET = {0: "A", 1: "B", 2: "C"}
NEW = ["r101", "r102", "r103"]

macros = []   # (name, text)
prov = []     # (name, text, source, field, raw)


def load(p):
    with open(os.path.join(ROOT, p)) as f:
        return json.load(f)


def m(name, value, fmt, source, field):
    if isinstance(value, str):
        txt = value
    else:
        txt = format(value, fmt)
    macros.append((name, txt))
    prov.append((name, txt, source, field, repr(value)))


def sci(x, digits=1):
    """LaTeX scientific notation, e.g. 2.0\\times10^{-9}."""
    s = f"{x:.{digits}e}"
    mant, exp = s.split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


def signed(x, nd=3):
    return f"{x:+.{nd}f}".replace("-", "$-$") if x < 0 else f"{x:+.{nd}f}"


def main():
    s1 = load(f"{S1}/results.json")
    s1m = load(f"{S1}/manifest.json")
    s2 = load(f"{S2}/results.json")
    s2m = load(f"{S2}/manifest.json")
    adm = load(ADM)
    src1, src2 = f"{S1}/results.json", f"{S2}/results.json"

    # ---------------- stage 1 -------------------------------------------------
    pm = s1["per_method"]
    for meth, tag in (("minimax_rel", "Minimax"), ("wrrr_uniform_rel", "Uniform")):
        m(f"SOne{tag}TestWorstMean", pm[meth]["test_worst_rel"]["mean"], ".3f", src1,
          f"per_method.{meth}.test_worst_rel.mean")
        m(f"SOne{tag}CalWorstMean", pm[meth]["cal_worst_rel"]["mean"], ".3f", src1,
          f"per_method.{meth}.cal_worst_rel.mean")
        for seed, rows in s1["per_seed_rows"].items():
            r = [x for x in rows if x["method"] == meth][0]
            m(f"SOne{tag}TestWorstSeed{LET[int(seed)]}", r["eval"]["summary"]["test"]["worst_rel"],
              ".3f", src1, f"per_seed_rows.{seed}[method={meth}].eval.summary.test.worst_rel")
    m("SOneWall", s1m["wall_time_s"], ".1f", f"{S1}/manifest.json", "wall_time_s")
    m("SOneK", 4, "d", "configs/cpu_synthetic.json", "k")

    # ---------------- configuration values -----------------------------------
    c1 = load("configs/cpu_synthetic.json")
    c2 = load("configs/p2_stage2_moment_normalizer_diag.json")
    syn = c1["synthetic"]
    for key, name in (("n_tasks", "CfgK"), ("d_in", "CfgDin"), ("d_out", "CfgDout"),
                      ("lora_rank", "CfgLoraRank"), ("n_cal", "CfgNcal"), ("n_dev", "CfgNdev"),
                      ("n_test", "CfgNtest")):
        m(name, syn[key], "d", "configs/cpu_synthetic.json", f"synthetic.{key}")
    m("CfgR", c1["k"], "d", "configs/cpu_synthetic.json", "k")
    m("CfgGains", ", ".join(f"{g:g}" for g in syn["adapter_gains"]), "s", "configs/cpu_synthetic.json",
      "synthetic.adapter_gains")
    m("CfgDecay", syn["cov_decay"], "g", "configs/cpu_synthetic.json", "synthetic.cov_decay")
    m("CfgRho", syn["shared_rho"], "g", "configs/cpu_synthetic.json", "synthetic.shared_rho")
    m("CfgNTeachers", len(c2["fixed_from_fixture"]["teacher_seeds"]), "d",
      "configs/p2_stage2_moment_normalizer_diag.json", "len(fixed_from_fixture.teacher_seeds)")
    m("CfgNResamples", len(c2["calibration_resample_seeds"]), "d",
      "configs/p2_stage2_moment_normalizer_diag.json", "len(calibration_resample_seeds)")
    m("CfgNewCells", len(c2["fixed_from_fixture"]["teacher_seeds"]) * len(c2["calibration_resample_seeds"]), "d",
      "configs/p2_stage2_moment_normalizer_diag.json", "teachers x resamples")
    m("CfgMCSamples", c2["monte_carlo_check"]["n_samples"], "d", "configs/p2_stage2_moment_normalizer_diag.json",
      "monte_carlo_check.n_samples")
    m("CfgPresentThr", 0.02, ".2f", "configs/p2_stage2_moment_normalizer_diag.json",
      "decision_rules.problem_present (threshold +0.02, parsed from text)")
    assert "+0.02" in c2["decision_rules"]["problem_present"]
    m("CfgDualIters", c1["dual"]["iters"], "d", "configs/cpu_synthetic.json", "dual.iters")
    m("CfgPolishIters", c1["dual"]["polish_iters"], "d", "configs/cpu_synthetic.json", "dual.polish_iters")

    # ---------------- stage 2: bookkeeping -----------------------------------
    rows = [r for r in s2["rows"] if r["status"] == "OK"]
    by = {(r["teacher"], r["draw"], r["arm"], r["method"]): r for r in rows}
    m("STwoCellsOK", s2["cell_status_counts"].get("OK", 0), "d", src2, "cell_status_counts.OK")
    m("STwoCellsTotal", sum(s2["cell_status_counts"].values()), "d", src2, "sum(cell_status_counts)")
    m("STwoCPU", s2m["cpu_time_s"], ".1f", f"{S2}/manifest.json", "cpu_time_s")
    m("STwoWall", s2m["wall_time_s"], ".1f", f"{S2}/manifest.json", "wall_time_s")
    m("STwoRSS", s2m["peak_rss_bytes"] / 2**20, ".1f", f"{S2}/manifest.json", "peak_rss_bytes/2^20")
    m("STwoCPUCap", s2m["caps"]["RLIMIT_CPU"][0], "d", f"{S2}/manifest.json", "caps.RLIMIT_CPU[0]")
    m("STwoCommit", s2m["git"]["commit"][:7], "s", f"{S2}/manifest.json", "git.commit")

    # Monte Carlo moment check
    mc = s2["mc_moment_check"]
    m("STwoMCN", len(mc), "d", src2, "len(mc_moment_check)")
    m("STwoMCWorstZ", max(r["worst_abs_z"] for r in mc), ".2f", src2, "max(mc_moment_check[].worst_abs_z)")

    # ---------------- stage 2: numerical audit --------------------------------
    audits = [r["audit"] for r in rows if r["method"] == "minimax_rel"]
    ba = s2["bound_audit"]
    m("STwoNFits", ba["n_minimax_fits"], "d", src2, "bound_audit.n_minimax_fits")
    m("STwoMaxGapRel", sci(ba["max_gap_rel"]), "s", src2, "bound_audit.max_gap_rel")
    m("STwoMaxGDisc", sci(ba["max_g_rel_discrepancy_vs_izenman"]), "s", src2,
      "bound_audit.max_g_rel_discrepancy_vs_izenman")
    m("STwoMaxUBDisc", sci(ba["max_ub_rel_discrepancy_raw"]), "s", src2, "bound_audit.max_ub_rel_discrepancy_raw")
    m("STwoMaxRankRatio", sci(ba["max_M_sigma_k1_over_sigma_1"]), "s", src2,
      "bound_audit.max_M_sigma_k1_over_sigma_1")
    m("STwoMinLambda", ba["min_lambda"], ".3f", src2, "bound_audit.min_lambda")
    m("STwoMaxCond", ba["max_S_cond"], ".1f", src2, "bound_audit.max_S_cond")
    svgap = [(a["whitened_sigma_k"] - a["whitened_sigma_k_plus_1"]) / a["whitened_sigma_k"] for a in audits]
    m("STwoMinSvGap", min(svgap), ".3f", src2,
      "min over minimax rows of (audit.whitened_sigma_k - audit.whitened_sigma_k_plus_1)/audit.whitened_sigma_k")
    n_refined = sum(1 for a in audits if a["refine"] != "skipped")
    m("STwoNRefined", n_refined, "d", src2, "count(audit.refine != 'skipped')")

    # ---------------- stage 2: 2x2 --------------------------------------------
    arms = s2["arms"]
    for key, label, tag in ARMS:
        a = arms[key]
        m(f"STwoDelta{tag}", signed(a["mean_delta"]), "s", src2, f"arms.{key}.mean_delta")
        m(f"STwoVerdict{tag}", a["verdict"].replace("_", "\\_"), "s", src2, f"arms.{key}.verdict")
        if key != "pop_moment__pop_norm":
            m(f"STwoDeltaSD{tag}", statistics.stdev(a["cells"]), ".3f", src2, f"stdev(arms.{key}.cells)")
        for t in (0, 1, 2):
            m(f"STwoDelta{tag}T{LET[t]}", signed(a["teacher_mean_delta"][str(t)]), "s", src2,
              f"arms.{key}.teacher_mean_delta.{t}")
            if key != "pop_moment__pop_norm":
                m(f"STwoOrigDelta{tag}T{LET[t]}", signed(a["orig_seen_draw"][str(t)]["delta"]), "s", src2,
                  f"arms.{key}.orig_seen_draw.{t}.delta")
    for t in (0, 1, 2):
        for meth, tag in (("minimax_rel", "Minimax"), ("wrrr_uniform_rel", "Uniform")):
            r = by[(t, "orig_seen", "emp_moment__emp_norm", meth)]
            m(f"STwoOrig{tag}SampleT{LET[t]}", max(r["test_rel_per_task_PREVIOUSLY_SEEN"]), ".3f", src2,
              f"rows[teacher={t},draw=orig_seen,arm=emp_moment__emp_norm,method={meth}]"
              ".max(test_rel_per_task_PREVIOUSLY_SEEN)")
            m(f"STwoOrig{tag}PopT{LET[t]}", r["F_pop"], ".3f", src2,
              f"rows[teacher={t},draw=orig_seen,arm=emp_moment__emp_norm,method={meth}].F_pop")
            rp = by[(t, "all_draws", "pop_moment__pop_norm", meth)]
            m(f"STwoPP{tag}T{LET[t]}", rp["F_pop"], ".3f", src2,
              f"rows[teacher={t},draw=all_draws,arm=pop_moment__pop_norm,method={meth}].F_pop")

    # optimism (new draws only)
    opt = {}
    for key, label, tag in ARMS[:3]:
        for meth, mtag in (("minimax_rel", "Minimax"), ("wrrr_uniform_rel", "Uniform")):
            xs = [by[(t, d, key, meth)] for t in (0, 1, 2) for d in NEW]
            fit = statistics.fmean(x["fit_objective_max"] for x in xs)
            pop = statistics.fmean(x["F_pop"] for x in xs)
            opt[(key, meth)] = (fit, pop)
            m(f"STwoOptFit{tag}{mtag}", fit, ".3f", src2, f"mean over new draws of rows[arm={key},method={meth}].fit_objective_max")
            m(f"STwoOptPop{tag}{mtag}", pop, ".3f", src2, f"mean over new draws of rows[arm={key},method={meth}].F_pop")
            m(f"STwoOptGap{tag}{mtag}", signed(pop - fit), "s", src2, "difference of the two above")
    # normaliser ratios
    ratios = []
    for t in (0, 1, 2):
        pop = by[(t, "all_draws", "pop_moment__pop_norm", "minimax_rel")]["normalizer_used"]
        for d in NEW + ["orig_seen"]:
            emp = by[(t, d, "emp_moment__emp_norm", "minimax_rel")]["normalizer_used"]
            ratios += [a / b for a, b in zip(emp, pop)]
    m("STwoNormRatioMin", min(ratios), ".2f", src2, "min emp/pop normalizer_used ratio (emp_moment__emp_norm vs pop_moment__pop_norm)")
    m("STwoNormRatioMax", max(ratios), ".2f", src2, "max of the same ratio")
    # lambda for task index 2 in pop/pop
    for t in (0, 1, 2):
        lam = by[(t, "all_draws", "pop_moment__pop_norm", "minimax_rel")]["audit"]["lambda_star"]
        m(f"STwoPPLambdaTwoT{LET[t]}", lam[2], ".3f", src2,
          f"rows[teacher={t},arm=pop_moment__pop_norm,method=minimax_rel].audit.lambda_star[2]")

    # ---------------- admission ---------------------------------------------
    m("AdmNAdapters", len(adm["adapters"]), "d", ADM, "len(adapters)")
    m("AdmBaseSha", adm["base_model"]["current_sha"][:7], "s", ADM, "base_model.current_sha")
    ranks = {v["r"] for v in adm["adapters"].values()}
    assert len(ranks) == 1
    m("AdmLoraR", ranks.pop(), "d", ADM, "adapters.*.r (identical for all adapters)")
    m("AdmCoLAScore", adm["publisher_reported_lora16_scores_on_own_task"]["cola"]["lora"], ".1f", ADM,
      "publisher_reported_lora16_scores_on_own_task.cola.lora (publisher-reported, not reproduced)")
    cpu_mm = sorted(r["cpu_s"] for r in rows if r["method"] == "minimax_rel")
    cpu_un = sorted(r["cpu_s"] for r in rows if r["method"] == "wrrr_uniform_rel")
    m("STwoMinimaxCPUMedian", statistics.median(cpu_mm), ".1f", src2, "median(rows[method=minimax_rel].cpu_s) (includes bound audit)")
    m("STwoMinimaxCPUMax", cpu_mm[-1], ".1f", src2, "max(rows[method=minimax_rel].cpu_s)")
    m("STwoUniformCPUMedianMs", 1000 * statistics.median(cpu_un), ".0f", src2, "1000*median(rows[method=wrrr_uniform_rel].cpu_s)")
    m("SOneMinimaxRawMean", s1["per_method"]["minimax_rel"]["test_worst_rel"]["mean"], ".6f", src1,
      "per_method.minimax_rel.test_worst_rel.mean")

    # ---------------- stage 3 (optional, formula check) ------------------------
    s3dirs = sorted(d for d in os.listdir(os.path.join(ROOT, "runs")) if d.startswith(os.path.basename(S3_GLOB)))
    s3 = None
    if s3dirs:
        s3p = f"runs/{s3dirs[-1]}"
        s3 = load(f"{s3p}/results.json")
        s3m = load(f"{s3p}/manifest.json")
        src3 = f"{s3p}/results.json"
        m("SThreeStatus", s3["status"].replace("_", "\\_"), "s", src3, "status")
        m("SThreeCPU", s3m["cpu_time_s"], ".1f", f"{s3p}/manifest.json", "cpu_time_s")
        m("SThreeNRefits", s3["n_refits"], "d", src3, "n_refits")
        m("SThreeNHashMatch", s3["n_hash_match"], "d", src3, "n_hash_match")
        for k_, v in s3["summary"].items():
            name = "SThree" + "".join(p.capitalize() for p in k_.split("_"))
            if isinstance(v, float):
                m(name, v if abs(v) >= 1e-3 or v == 0 else sci(v), ".3f" if not isinstance(v, str) else "s",
                  src3, f"summary.{k_}")
            elif isinstance(v, int):
                m(name, v, "d", src3, f"summary.{k_}")

    # ---------------- write macros --------------------------------------------
    os.makedirs(os.path.join(PAPER, "generated"), exist_ok=True)
    with open(os.path.join(PAPER, "generated", "numbers.tex"), "w") as f:
        f.write("% AUTO-GENERATED by paper/tools/export_results.py -- do not edit\n")
        for name, txt in macros:
            f.write(f"\\newcommand{{\\{name}}}{{{txt}}}\n")
    with open(os.path.join(PAPER, "generated", "provenance.tsv"), "w") as f:
        f.write("macro\trendered\tsource_file\tfield\traw_value\n")
        for row in prov:
            f.write("\t".join(str(x) for x in row) + "\n")

    write_tables(s1, s2, by, opt, s3)
    write_figure(by)
    print(f"{len(macros)} macros, tables and figure written")


def write_tables(s1, s2, by, opt, s3):
    tdir = os.path.join(PAPER, "tables")
    os.makedirs(tdir, exist_ok=True)
    # stage 1
    names = {"zero_base_model": "Base model ($W=0$)", "ta_svd": "TA + SVD", "knots_ta_trunc": "KnOTS-TA + trunc.$^\\dagger$",
             "knots_ties_trunc": "KnOTS-TIES + trunc.$^\\dagger$", "ctm_like_ta": "CtM-like$^\\dagger$",
             "regmean_full_rank": "RegMean (full rank)$^\\ddagger$", "regmean_euclid_trunc": "RegMean + Eucl.\\ trunc.",
             "regmean_whitened_trunc": "RegMean + whitened trunc.", "wrrr_uniform_rel": "Uniform WRRR (rel.)",
             "minimax_rel": "Minimax (rel.)", "factor_average_diag": "Factor averaging$^\\S$"}
    lines = ["% AUTO-GENERATED by paper/tools/export_results.py", "\\begin{tabular}{lcccc}", "\\toprule",
             "Method & Final rank & Test worst (mean $\\pm$ sd) & Test mean & Cal.\\ worst \\\\", "\\midrule"]
    for key, label in names.items():
        s = s1["per_method"][key]
        rank = "/".join(str(v) for v in s["final_rank"]["values"])
        rank = rank if len(set(s["final_rank"]["values"])) > 1 else str(s["final_rank"]["values"][0])
        lines.append(f"{label} & {rank} & {s['test_worst_rel']['mean']:.3f} $\\pm$ {s['test_worst_rel']['sd']:.3f} & "
                     f"{s['test_mean_rel']['mean']:.3f} & {s['cal_worst_rel']['mean']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tdir, "tab_stage1.tex"), "w").write("\n".join(lines) + "\n")

    # provenance of the disputed numbers
    lines = ["% AUTO-GENERATED", "\\begin{tabular}{llccc}", "\\toprule",
             "Solution & Evaluation & Teacher 0 & Teacher 1 & Teacher 2 \\\\", "\\midrule"]
    for meth, lab in (("minimax_rel", "Minimax"), ("wrrr_uniform_rel", "Uniform WRRR")):
        rs = [by[(t, "orig_seen", "emp_moment__emp_norm", meth)] for t in (0, 1, 2)]
        lines.append(f"{lab} & test sample ($n{{=}}512$, seen) & " +
                     " & ".join(f"{max(r['test_rel_per_task_PREVIOUSLY_SEEN']):.3f}" for r in rs) + " \\\\")
        lines.append(f"{lab} & population $F$ & " + " & ".join(f"{r['F_pop']:.3f}" for r in rs) + " \\\\")
        lines.append(f"{lab} & calibration $\\hat F$ & " + " & ".join(f"{r['fit_objective_max']:.3f}" for r in rs) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tdir, "tab_orig_draw.tex"), "w").write("\n".join(lines) + "\n")

    # 2x2 teacher-level paired table
    lines = ["% AUTO-GENERATED", "\\begin{tabular}{lcccc}", "\\toprule",
             "Moment / normaliser & Teacher 0 & Teacher 1 & Teacher 2 & Mean over 9 cells (sd) \\\\", "\\midrule"]
    for key, label, tag in ARMS:
        a = s2["arms"][key]
        tm = [a["teacher_mean_delta"][str(t)] for t in (0, 1, 2)]
        sd = f" ({statistics.stdev(a['cells']):.3f})" if key != "pop_moment__pop_norm" else " (--)"
        lines.append(f"{label} & " + " & ".join(f"{v:+.3f}" for v in tm) + f" & {a['mean_delta']:+.3f}{sd} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tdir, "tab_2x2.tex"), "w").write("\n".join(lines).replace("-0.", "$-$0.") + "\n")

    # per-draw table (appendix)
    lines = ["% AUTO-GENERATED", "\\begin{tabular}{llcccccc}", "\\toprule",
             "Teacher & Draw & \\multicolumn{2}{c}{Emp/Emp} & \\multicolumn{2}{c}{Emp/Pop} & \\multicolumn{2}{c}{Pop/Emp} \\\\",
             " & & Unif. & Minimax & Unif. & Minimax & Unif. & Minimax \\\\", "\\midrule"]
    for t in (0, 1, 2):
        for d in ["orig_seen"] + NEW:
            cells = []
            for key, _, _ in ARMS[:3]:
                cells.append(f"{by[(t, d, key, 'wrrr_uniform_rel')]['F_pop']:.3f}")
                cells.append(f"{by[(t, d, key, 'minimax_rel')]['F_pop']:.3f}")
            dl = "orig.\\ (seen)" if d == "orig_seen" else d
            lines.append(f"{t} & {dl} & " + " & ".join(cells) + " \\\\")
        pu = by[(t, 'all_draws', 'pop_moment__pop_norm', 'wrrr_uniform_rel')]['F_pop']
        pmx = by[(t, 'all_draws', 'pop_moment__pop_norm', 'minimax_rel')]['F_pop']
        lines.append(f"{t} & Pop/Pop (oracle) & \\multicolumn{{6}}{{c}}{{Unif.\\ {pu:.3f}, Minimax {pmx:.3f}}} \\\\")
        if t < 2:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tdir, "tab_per_draw.tex"), "w").write("\n".join(lines) + "\n")

    # optimism table
    lines = ["% AUTO-GENERATED", "\\begin{tabular}{llccc}", "\\toprule",
             "Moment / normaliser & Solution & Fitted objective $\\hat F(\\hat W)$ & Population $F(\\hat W)$ & Difference \\\\",
             "\\midrule"]
    for key, label, tag in ARMS[:3]:
        for meth, lab in (("wrrr_uniform_rel", "Uniform WRRR"), ("minimax_rel", "Minimax")):
            fit, pop = opt[(key, meth)]
            lines.append(f"{label} & {lab} & {fit:.3f} & {pop:.3f} & {pop - fit:+.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tdir, "tab_optimism.tex"), "w").write("\n".join(lines) + "\n")

    # numerical audit per arm
    rows = [r for r in s2["rows"] if r["status"] == "OK" and r["method"] == "minimax_rel"]
    lines = ["% AUTO-GENERATED", "\\begin{tabular}{lcccccc}", "\\toprule",
             "Moment / normaliser & Fits & max rel.\\ gap & max $|g-g_{\\mathrm{ref}}|/g_{\\mathrm{ref}}$ & "
             "min $\\lambda_k$ & max $\\kappa(S_\\lambda)$ & min $(\\sigma_r-\\sigma_{r+1})/\\sigma_r$ \\\\", "\\midrule"]
    for key, label, _ in ARMS:
        au = [r["audit"] for r in rows if r["arm"] == key]
        gaps = [(a["whitened_sigma_k"] - a["whitened_sigma_k_plus_1"]) / a["whitened_sigma_k"] for a in au]
        lines.append(f"{label} & {len(au)} & ${sci(max(a['gap_rel'] for a in au))}$ & "
                     f"${sci(max(a['g_rel_discrepancy'] for a in au))}$ & {min(a['lambda_min'] for a in au):.3f} & "
                     f"{max(a['S_cond'] for a in au):.1f} & {min(gaps):.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tdir, "tab_audit.tex"), "w").write("\n".join(lines) + "\n")

    if s3 is not None:
        lines = ["% AUTO-GENERATED", "\\begin{tabular}{lccccc}", "\\toprule",
                 "Arm & Cells & $\\max_k\\|S_k-\\hat S_k\\|_{\\mathrm{op}}/\\|S_k\\|_{\\mathrm{op}}$ & "
                 "bound $2\\delta+\\epsilon$ & realised $F(\\hat W)-F(W^\\star)$ & bound holds \\\\", "\\midrule"]
        for row in s3["table"]:
            lines.append(f"{row['arm_label']} & {row['n']} & {row['rel_op_err_median']:.2f} "
                         f"[{row['rel_op_err_min']:.2f}, {row['rel_op_err_max']:.2f}] & "
                         f"{row['bound_median']:.2f} [{row['bound_min']:.2f}, {row['bound_max']:.2f}] & "
                         f"{row['excess_median']:+.3f} [{row['excess_min']:+.3f}, {row['excess_max']:+.3f}] & "
                         f"{row['n_holds']}/{row['n']} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(tdir, "tab_bound_check.tex"), "w").write(
            "\n".join(lines).replace("[-", "[$-$").replace(", -", ", $-$") + "\n")


def write_figure(by):
    """Paired teacher-level plot: uniform (blue circle) -> minimax (orange square)."""
    blue, orange = "2A78D6", "EB6834"
    ymin, ymax = 0.45, 0.85
    H = 4.0     # cm plot height
    def y(v):
        return (v - ymin) / (ymax - ymin) * H
    out = ["% AUTO-GENERATED by paper/tools/export_results.py (TikZ; teacher means over the 3 new draws;",
           "% Pop/Pop uses no calibration draw). Colours: validated categorical slots 1-2.",
           f"\\definecolor{{unifc}}{{HTML}}{{{blue}}}", f"\\definecolor{{minic}}{{HTML}}{{{orange}}}",
           "\\begin{tikzpicture}[x=1cm,y=1cm,font=\\footnotesize]"]
    # axis
    out.append(f"\\draw[black!60] (0,0) -- (0,{H:.3f});")
    for v in [0.45, 0.55, 0.65, 0.75, 0.85]:
        out.append(f"\\draw[black!15] (0,{y(v):.3f}) -- (12.6,{y(v):.3f});")
        out.append(f"\\node[anchor=east,text=black!70] at (0,{y(v):.3f}) {{{v:.2f}}};")
    out.append(f"\\node[rotate=90,anchor=south,text=black!80] at (-0.9,{H/2:.3f}) {{population worst-task $F$}};")
    for i, (key, label, _) in enumerate(ARMS):
        x0 = 0.6 + i * 3.1
        xu, xm = x0, x0 + 1.6
        out.append(f"\\node[anchor=north,text=black!80] at ({(xu + xm) / 2:.2f},-0.35) {{{label}}};")
        out.append(f"\\node[anchor=north,text=black!55] at ({xu:.2f},-0.02) {{U}};")
        out.append(f"\\node[anchor=north,text=black!55] at ({xm:.2f},-0.02) {{M}};")
        for t in (0, 1, 2):
            if key == "pop_moment__pop_norm":
                u = by[(t, "all_draws", key, "wrrr_uniform_rel")]["F_pop"]
                mm = by[(t, "all_draws", key, "minimax_rel")]["F_pop"]
            else:
                u = statistics.fmean(by[(t, d, key, "wrrr_uniform_rel")]["F_pop"] for d in NEW)
                mm = statistics.fmean(by[(t, d, key, "minimax_rel")]["F_pop"] for d in NEW)
            out.append(f"\\draw[black!45,line width=0.6pt] ({xu:.2f},{y(u):.3f}) -- ({xm:.2f},{y(mm):.3f});")
            out.append(f"\\fill[unifc,draw=white,line width=0.4pt] ({xu:.2f},{y(u):.3f}) circle (2.2pt);")
            out.append(f"\\fill[minic,draw=white,line width=0.4pt] ({xm - 0.075:.3f},{y(mm) - 0.075:.3f}) "
                       f"rectangle ({xm + 0.075:.3f},{y(mm) + 0.075:.3f});")
            out.append(f"\\node[anchor=west,text=black!70,font=\\scriptsize] at ({xm + 0.12:.2f},{y(mm):.3f}) {{{t}}};")
    # legend
    out.append(f"\\fill[unifc] (0.4,{H + 0.45:.3f}) circle (2.2pt);")
    out.append(f"\\node[anchor=west] at (0.55,{H + 0.45:.3f}) {{U: uniform WRRR}};")
    out.append(f"\\fill[minic] (3.425,{H + 0.375:.3f}) rectangle (3.575,{H + 0.525:.3f});")
    out.append(f"\\node[anchor=west] at (3.65,{H + 0.45:.3f}) {{M: minimax}};")
    out.append(f"\\node[anchor=west,text=black!70] at (6.2,{H + 0.45:.3f}) {{digits: teacher index}};")
    out.append("\\end{tikzpicture}")
    os.makedirs(os.path.join(PAPER, "figures"), exist_ok=True)
    open(os.path.join(PAPER, "figures", "fig_paired_2x2.tex"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
