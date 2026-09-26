"""Build paper/main.pdf with pdflatex + bibtex under a cumulative CPU budget.

    python3 paper/tools/build.py

Each step runs as a child process with RLIMIT_CPU set to the remaining budget
(600 CPU-s in total for builds and static checks, tracked in
paper/build_log.jsonl). After the build it records the engine version, page
count, the page on which the main text ends (label sec:end-of-main-text), and
counts of undefined references/citations and overfull boxes.
"""

import json
import os
import re
import resource
import subprocess
import time

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(PAPER, "build_log.jsonl")
BUDGET = 600.0


def spent() -> float:
    if not os.path.exists(LEDGER):
        return 0.0
    tot = 0.0
    for line in open(LEDGER):
        tot += json.loads(line).get("cpu_s", 0.0)
    return tot


def run(step, cmd):
    remaining = BUDGET - spent()
    if remaining <= 5:
        raise SystemExit(f"build CPU budget exhausted ({spent():.1f}/{BUDGET} s); step {step} NOT_RUN")
    lim = int(remaining)

    def pre():
        resource.setrlimit(resource.RLIMIT_CPU, (lim, lim))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True, preexec_fn=pre)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "step": step, "cmd": " ".join(cmd),
           "exit": p.returncode, "cpu_s": cpu, "wall_s": time.time() - t0}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return p, rec


def main():
    steps = [("pdflatex-1", ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]),
             ("bibtex", ["bibtex", "main"]),
             ("pdflatex-2", ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]),
             ("pdflatex-3", ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])]
    status = "BUILT"
    for step, cmd in steps:
        p, rec = run(step, cmd)
        print(f"{step}: exit {rec['exit']} cpu {rec['cpu_s']:.1f}s")
        if p.returncode != 0:
            status = f"FAILED_AT_{step}"
            tail = (p.stdout[-3000:] if p.stdout else "") + (p.stderr[-1000:] if p.stderr else "")
            print(tail)
            break
    report = {"status": status, "build_cpu_spent_total_s": spent(), "budget_s": BUDGET}
    log = open(os.path.join(PAPER, "main.log"), errors="ignore").read() if os.path.exists(os.path.join(PAPER, "main.log")) else ""
    report["engine"] = log.splitlines()[0] if log else None
    report["undefined_references"] = len(re.findall(r"Reference `[^']+' on page \d+ undefined", log))
    report["undefined_citations"] = len(re.findall(r"Citation `[^']+' on page \d+ undefined", log))
    report["overfull_hbox"] = len(re.findall(r"Overfull \\hbox", log))
    report["underfull_hbox"] = len(re.findall(r"Underfull \\hbox", log))
    report["latex_warnings"] = len(re.findall(r"LaTeX Warning", log))
    blg = os.path.join(PAPER, "main.blg")
    if os.path.exists(blg):
        b = open(blg, errors="ignore").read()
        report["bibtex_warnings"] = len(re.findall(r"^Warning--", b, flags=re.M))
        report["bibtex_errors"] = len(re.findall(r"error message", b))
    pdf = os.path.join(PAPER, "main.pdf")
    if status == "BUILT" and os.path.exists(pdf):
        info = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
        m = re.search(r"Pages:\s+(\d+)", info)
        report["pages_total"] = int(m.group(1)) if m else None
        aux = open(os.path.join(PAPER, "main.aux"), errors="ignore").read()
        m = re.search(r"\\newlabel\{sec:end-of-main-text\}\{\{[^}]*\}\{(\d+)\}", aux)
        report["main_text_ends_on_page"] = int(m.group(1)) if m else None
        m = re.search(r"\\newlabel\{sec:intro\}\{\{[^}]*\}\{(\d+)\}", aux)
        report["main_text_starts_on_page"] = int(m.group(1)) if m else None
    json.dump(report, open(os.path.join(PAPER, "build_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
