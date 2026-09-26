"""Run manifests: code/config/data hashes, environment, time and memory."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def git_info() -> Dict[str, object]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception as exc:  # pragma: no cover - environment dependent
            return f"ERROR: {exc}"
    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": head or "NO_COMMIT",
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "dirty_files": status.splitlines()[:50],
    }


def source_hashes() -> Dict[str, str]:
    out = {}
    src = os.path.join(ROOT, "src", "lowrank_merge")
    for name in sorted(os.listdir(src)):
        if name.endswith(".py"):
            out[name] = sha256_file(os.path.join(src, name))
    return out


def environment() -> Dict[str, object]:
    mods: Dict[str, str] = {}
    for mod in ("numpy", "scipy", "torch", "transformers", "peft", "datasets"):
        try:
            m = __import__(mod)
            mods[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            mods[mod] = "NOT_INSTALLED"
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "optional_modules": mods,
    }


def peak_rss_bytes() -> int:
    # ru_maxrss is KiB on Linux
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def write_json(path: str, obj: object) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_default)


def _default(o: object):
    if isinstance(o, float) and (o != o):
        return "NaN"
    return str(o)


def list_files(paths: List[str]) -> Dict[str, str]:
    return {os.path.relpath(p, ROOT): sha256_file(p) for p in paths if os.path.exists(p)}
