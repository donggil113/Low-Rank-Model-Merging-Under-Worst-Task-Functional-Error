"""Per-task empirical functional error and the worst-task (minimax) objective.

For one linear layer with base weight W0, task adapter delta D_t and inputs
X_t (d_in x n_t, columns are samples), replacing D_t by a merged delta M
changes the layer output on task t by (M - D_t) X_t. The empirical functional
error is

    e_t(M) = (1 / n_t) * || (M - D_t) X_t ||_F^2
           = tr((M - D_t) Sigma_t (M - D_t)^T),   Sigma_t = X_t X_t^T / n_t.

It is layer-local: it measures the change of this layer's pre-activation
output under the *task's own* input distribution. It is not the end-to-end
task loss, and improvements in it need not transfer.

The relative error divides by the adapter's own functional energy
e_t(0) = tr(D_t Sigma_t D_t^T), i.e. the error of dropping the adapter, so
tasks with different output scales are comparable inside a max.

The worst-task objective is  F(M) = max_t  e_t(M) / s_t  with s_t = 1
(absolute) or s_t = e_t(0) on the calibration split (relative).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from . import linalg as la
from .linalg import Matrix


@dataclass
class QuadTask:
    """Quadratic task error  e(M) = [tr(M G M^T) - 2 tr(M H^T) + c] / norm.

    The data-derived form has G = Sigma_t, H = D_t Sigma_t,
    c = tr(D_t Sigma_t D_t^T). Reduced sub-problems (fixed column or row
    space) keep the same algebraic form with smaller G/H, so the solvers work
    on this representation.
    """

    name: str
    G: Matrix
    H: Matrix
    c: float
    norm: float = 1.0

    def value(self, m: Matrix) -> float:
        mg = la.matmul(m, self.G)
        quad = la.inner(mg, m)
        lin = la.inner(m, self.H)
        return (quad - 2.0 * lin + self.c) / self.norm

    def grad(self, m: Matrix) -> Matrix:
        return la.scale(2.0 / self.norm, la.sub(la.matmul(m, self.G), self.H))


def gram(x: Matrix) -> Matrix:
    """Sigma = X X^T / n for X of shape (d_in, n)."""
    n = len(x[0])
    return la.scale(1.0 / n, la.gram_rows(x))


def quad_task_from_data(name: str, delta: Matrix, x: Matrix,
                        relative: bool) -> QuadTask:
    sigma = gram(x)
    h = la.matmul(delta, sigma)
    c = la.inner(h, delta)  # tr(D Sigma D^T)
    if relative and c <= 0.0:
        raise ValueError(f"task {name}: adapter has zero functional energy on these inputs")
    return QuadTask(name=name, G=sigma, H=h, c=c, norm=c if relative else 1.0)


def functional_error(m: Matrix, delta: Matrix, x: Matrix) -> float:
    """(1/n) ||(M - D) X||_F^2 evaluated directly on the samples."""
    n = len(x[0])
    return la.frob2(la.matmul(la.sub(m, delta), x)) / n


def adapter_energy(delta: Matrix, x: Matrix) -> float:
    n = len(x[0])
    return la.frob2(la.matmul(delta, x)) / n


def relative_functional_error(m: Matrix, delta: Matrix, x: Matrix) -> float:
    e0 = adapter_energy(delta, x)
    if e0 <= 0.0:
        raise ValueError("adapter has zero functional energy on these inputs")
    return functional_error(m, delta, x) / e0


def worst_task(values: Sequence[float]) -> float:
    return max(values)


@dataclass
class TaskSplit:
    """One task at one layer: effective delta plus calibration/test inputs."""

    name: str
    delta: Matrix
    x_cal: Matrix
    x_test: Matrix
    x_dev: Matrix | None = None

    def split(self, which: str) -> Matrix:
        if which == "cal":
            return self.x_cal
        if which == "test":
            return self.x_test
        if which == "dev":
            if self.x_dev is None:
                raise ValueError(f"task {self.name}: no dev split")
            return self.x_dev
        raise ValueError(f"unknown split {which!r}")


def quad_tasks(tasks: Sequence[TaskSplit], split: str, relative: bool) -> List[QuadTask]:
    return [quad_task_from_data(t.name, t.delta, t.split(split), relative) for t in tasks]


def evaluate(m: Matrix, tasks: Sequence[TaskSplit],
             splits: Sequence[str] = ("cal", "test")) -> Dict[str, object]:
    """Per-task absolute and relative functional errors, from raw samples."""
    out: Dict[str, object] = {"per_task": {}, "summary": {}}
    per_task: Dict[str, Dict[str, float]] = out["per_task"]  # type: ignore[assignment]
    summary: Dict[str, Dict[str, object]] = out["summary"]  # type: ignore[assignment]
    for t in tasks:
        per_task[t.name] = {}
    for sp in splits:
        rel_vals: List[float] = []
        abs_vals: List[float] = []
        for t in tasks:
            x = t.split(sp)
            e_abs = functional_error(m, t.delta, x)
            e0 = adapter_energy(t.delta, x)
            e_rel = e_abs / e0
            per_task[t.name][f"{sp}_abs"] = e_abs
            per_task[t.name][f"{sp}_rel"] = e_rel
            abs_vals.append(e_abs)
            rel_vals.append(e_rel)
        worst_idx = max(range(len(tasks)), key=lambda i: rel_vals[i])
        summary[sp] = {
            "worst_rel": rel_vals[worst_idx],
            "worst_task": tasks[worst_idx].name,
            "mean_rel": sum(rel_vals) / len(rel_vals),
            "worst_abs": max(abs_vals),
            "mean_abs": sum(abs_vals) / len(abs_vals),
        }
    return out
