"""Pilot preflight: validate the adapter-pilot config and list every blocker.

    PYTHONPATH=src python -m lowrank_merge.pilot --config configs/pilot_4task_flan_t5_base_glue.json

It never downloads, installs or trains anything. It checks that the config
fixes everything that must be fixed before execution (splits, metric, MEI,
seeds, tuning grids, caps, stopping rules, information access per method,
task-identity disclosure), and reports blockers:

  BLOCKED_APPROVAL      a required approval is false
  BLOCKED_DEPENDENCY    a required Python module is not importable
  BLOCKED_LICENSE       a license/usage status is not VERIFIED
  BLOCKED_UNPINNED      a model/adapter revision or hash is not pinned
  CONFIG_INVALID        a required field is missing or inconsistent

Exit code 0 only if there are no blockers (READY_FOR_PILOT, which is a
statement about the checklist, not about novelty or expected results).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
from typing import Dict, List

from .manifest import ROOT, environment, git_info, sha256_file, write_json

REQUIRED_TOP = [
    "experiment", "approvals", "backend", "base_model", "adapters", "splits",
    "task_identity_disclosure", "merge_scope", "rank_budget", "methods", "tuning_grids",
    "primary_metric", "minimum_effect_of_interest", "seeds", "resource_caps", "stopping_rules",
    "datasets", "code",
]
REQUIRED_METHOD_KEYS = ["calibration_inputs", "dev_tuning", "labels", "final_rank"]
REQUIRED_SPLITS = ["calibration", "development", "test"]
REQUIRED_APPROVALS = ["download_weights", "download_datasets", "install_dependencies"]


def validate(cfg: Dict[str, object]) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []

    def bad(code: str, msg: str) -> None:
        issues.append({"code": code, "detail": msg})

    for key in REQUIRED_TOP:
        if key not in cfg:
            bad("CONFIG_INVALID", f"missing top-level field '{key}'")
    if issues:
        return issues
    adapters = cfg["adapters"]
    if len(adapters) != 4:
        bad("CONFIG_INVALID", f"pilot-0 requires exactly 4 adapters, got {len(adapters)}")
    tasks = [a.get("task") for a in adapters]
    if len(set(tasks)) != len(tasks):
        bad("CONFIG_INVALID", "duplicate task names")
    base = cfg["base_model"]["id"]
    for a in adapters:
        if a.get("declared_base") != base:
            bad("CONFIG_INVALID", f"adapter {a.get('task')} declares base {a.get('declared_base')} != {base}")
        for f in ("lora_r", "lora_alpha", "target_modules"):
            if a.get(f) in (None, "", []):
                bad("CONFIG_INVALID", f"adapter {a.get('task')} missing {f}")
    for sp in REQUIRED_SPLITS:
        if sp not in cfg["splits"]:
            bad("CONFIG_INVALID", f"missing split '{sp}'")
    if cfg["splits"].get("calibration", {}).get("labels_used") not in (False,):
        bad("CONFIG_INVALID", "calibration must declare labels_used=false")
    if cfg["splits"].get("calibration", {}).get("source") in (
            cfg["splits"].get("test", {}).get("source"),):
        if "disjointness" not in cfg["splits"]:
            bad("CONFIG_INVALID", "calibration and test share a source without a disjointness statement")
    for name, m in cfg["methods"].items():
        for f in REQUIRED_METHOD_KEYS:
            if f not in m:
                bad("CONFIG_INVALID", f"method {name} missing access field '{f}'")
    for f in ("merging_uses_task_identity", "inference_uses_task_identity", "task_specific_heads"):
        if f not in cfg["task_identity_disclosure"]:
            bad("CONFIG_INVALID", f"task_identity_disclosure missing '{f}'")
    mei = cfg["minimum_effect_of_interest"]
    if not isinstance(mei, dict) or "comparator" not in mei:
        bad("CONFIG_INVALID", "minimum_effect_of_interest must name its comparator")
    if not cfg["stopping_rules"]:
        bad("CONFIG_INVALID", "stopping_rules empty")
    if not cfg["rank_budget"].get("final_rank_per_layer"):
        bad("CONFIG_INVALID", "rank_budget.final_rank_per_layer empty")
    return issues


def blockers(cfg: Dict[str, object]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    # pilot-0 is CPU inference only: gpu_compute / paid_api are not required
    for k in REQUIRED_APPROVALS:
        v = cfg["approvals"].get(k)
        if v is not True:
            out.append({"code": "BLOCKED_APPROVAL", "detail": f"approval '{k}' is {v}"})
    for mod in cfg["backend"]["required_python_modules"]:
        if importlib.util.find_spec(mod) is None:
            out.append({"code": "BLOCKED_DEPENDENCY", "detail": f"python module '{mod}' not installed"})
    items = [("base_model", cfg["base_model"])] + [
        (f"adapter:{a['task']}", a) for a in cfg["adapters"]] + [
        ("datasets", cfg["datasets"]), ("code", cfg["code"])]
    for name, it in items:
        if it.get("license_status") != "VERIFIED":
            out.append({"code": "BLOCKED_LICENSE",
                        "detail": f"{name}: license_status={it.get('license_status')}"})
    for name, it in [("base_model", cfg["base_model"])] + [
            (f"adapter:{a['task']}", a) for a in cfg["adapters"]]:
        if not it.get("revision") or not it.get("weights_sha256"):
            out.append({"code": "BLOCKED_UNPINNED", "detail": f"{name}: revision/hash not pinned"})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    path = os.path.abspath(args.config)
    with open(path) as f:
        cfg = json.load(f)
    issues = validate(cfg)
    blk = blockers(cfg) if not issues else []
    status = "CONFIG_INVALID" if issues else ("BLOCKED" if blk else "READY_FOR_PILOT")
    out_dir = args.out or os.path.join(
        ROOT, "runs", f"pilot_preflight_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}")
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "status": status,
        "science_status": "SCIENCE_NOT_EVALUATED",
        "config_path": os.path.relpath(path, ROOT),
        "config_sha256": sha256_file(path),
        "config_issues": issues,
        "blockers": blk,
        "blocker_counts": {c: sum(1 for b in blk if b["code"] == c)
                           for c in sorted({b["code"] for b in blk})},
        "git": git_info(),
        "environment": environment(),
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "note": "Preflight only. Nothing was downloaded, installed or trained.",
    }
    write_json(os.path.join(out_dir, "preflight.json"), report)
    print(f"{status}: {len(issues)} config issues, {len(blk)} blockers -> "
          f"{os.path.relpath(out_dir, ROOT)}/preflight.json")
    for b in issues + blk:
        print(f"  [{b['code']}] {b['detail']}")
    return 0 if status == "READY_FOR_PILOT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
