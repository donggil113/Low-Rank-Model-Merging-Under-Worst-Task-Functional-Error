import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from lowrank_merge import linalg as la  # noqa: E402


def rel_diff(a, b):
    """||a - b||_F / max(||a||_F, ||b||_F, tiny) for matrices."""
    den = max(la.frob(a), la.frob(b), 1e-300)
    return la.frob(la.sub(a, b)) / den


def rel_scalar(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def make_raw_tasks(rng, d_out, d_in, n_tasks, n, delta_rank=None, cov_scales=None,
                   relative=False):
    """Random tasks: returns (TaskSplit list, RawTask list) sharing the same data."""
    from lowrank_merge.objectives import TaskSplit, adapter_energy
    from lowrank_merge.references import RawTask

    splits, raws = [], []
    for t in range(n_tasks):
        r = delta_rank or min(d_out, d_in)
        delta = la.matmul(la.random_normal(d_out, r, rng), la.random_normal(r, d_in, rng))
        mix = la.random_normal(d_in, d_in, rng)
        sc = (cov_scales[t] if cov_scales else 1.0)
        x = la.scale(sc, la.matmul(mix, la.random_normal(d_in, n, rng)))
        xt = la.scale(sc, la.matmul(mix, la.random_normal(d_in, n, rng)))
        splits.append(TaskSplit(name=f"t{t}", delta=delta, x_cal=x, x_test=xt))
        norm = adapter_energy(delta, x) if relative else 1.0
        raws.append(RawTask(delta=delta, x=x, norm=norm))
    return splits, raws
