"""Synthetic LoRA tasks at one linear layer, with separate cal/dev/test samples.

This is a controlled fixture for exercising the code path. It carries no
evidence about real adapters.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import linalg as la
from .adapters import LoRAAdapter
from .linalg import Matrix
from .objectives import TaskSplit


@dataclass
class SyntheticSpec:
    d_out: int = 12
    d_in: int = 16
    n_tasks: int = 4
    lora_rank: int = 4
    lora_alpha: float = 8.0
    shared_rho: float = 0.5       # correlation of factors with a shared component
    adapter_gains: Sequence[float] = (1.0, 1.0, 0.6, 1.6)
    cov_decay: float = 0.8        # eigenvalue decay of each task's input covariance
    n_cal: int = 32
    n_dev: int = 64
    n_test: int = 512
    input_subspace_dim: Optional[int] = None  # if set, inputs live in a subspace (singular Grams)


@dataclass
class SyntheticProblem:
    spec: SyntheticSpec
    seed: int
    adapters: List[LoRAAdapter]
    tasks: List[TaskSplit]
    meta: Dict[str, object] = field(default_factory=dict)


def _sub_rng(seed: int, *tags: object) -> random.Random:
    h = hashlib.sha256(repr((seed,) + tags).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _task_input_map(spec: SyntheticSpec, seed: int, t: int) -> Matrix:
    """L_t with x = L_t z, z ~ N(0, I): covariance L_t L_t^T."""
    rng = _sub_rng(seed, "cov", t)
    q = la.random_orthogonal(spec.d_in, rng)
    d = spec.input_subspace_dim or spec.d_in
    sd = [math.sqrt(spec.cov_decay ** i) if i < d else 0.0 for i in range(spec.d_in)]
    if spec.input_subspace_dim is not None:
        # the same subspace for all tasks, so the combined Gram is singular
        base_q = la.random_orthogonal(spec.d_in, _sub_rng(seed, "shared_subspace"))
        mix = la.random_orthogonal(d, rng)
        qq = la.matmul(la.columns(base_q, range(d)), mix)
        return [[qq[i][j] * sd[j] for j in range(d)] for i in range(spec.d_in)]
    return [[q[i][j] * sd[j] for j in range(spec.d_in)] for i in range(spec.d_in)]


def _samples(l_map: Matrix, n: int, rng: random.Random) -> Matrix:
    z = la.random_normal(len(l_map[0]), n, rng)
    return la.matmul(l_map, z)


def make_problem(spec: SyntheticSpec, seed: int) -> SyntheticProblem:
    if len(spec.adapter_gains) != spec.n_tasks:
        raise ValueError("adapter_gains must have one entry per task")
    r = spec.lora_rank
    rng_s = _sub_rng(seed, "shared")
    b_shared = la.random_normal(spec.d_out, r, rng_s, 1.0 / math.sqrt(r))
    a_shared = la.random_normal(r, spec.d_in, rng_s, 1.0 / math.sqrt(spec.d_in))
    rho = spec.shared_rho
    adapters: List[LoRAAdapter] = []
    tasks: List[TaskSplit] = []
    for t in range(spec.n_tasks):
        rng = _sub_rng(seed, "adapter", t)
        b_own = la.random_normal(spec.d_out, r, rng, 1.0 / math.sqrt(r))
        a_own = la.random_normal(r, spec.d_in, rng, 1.0 / math.sqrt(spec.d_in))
        g = spec.adapter_gains[t]
        b = la.scale(g, la.add(la.scale(math.sqrt(rho), b_shared),
                               la.scale(math.sqrt(1 - rho), b_own)))
        a = la.add(la.scale(math.sqrt(rho), a_shared), la.scale(math.sqrt(1 - rho), a_own))
        ad = LoRAAdapter(name=f"task{t}", B=b, A=a, scaling=spec.lora_alpha / r)
        adapters.append(ad)
        l_map = _task_input_map(spec, seed, t)
        x_cal = _samples(l_map, spec.n_cal, _sub_rng(seed, "x", t, "cal"))
        x_dev = _samples(l_map, spec.n_dev, _sub_rng(seed, "x", t, "dev"))
        x_test = _samples(l_map, spec.n_test, _sub_rng(seed, "x", t, "test"))
        tasks.append(TaskSplit(name=ad.name, delta=ad.effective_delta(),
                               x_cal=x_cal, x_dev=x_dev, x_test=x_test))
    prob = SyntheticProblem(spec=spec, seed=seed, adapters=adapters, tasks=tasks)
    prob.meta["data_sha256"] = problem_hash(prob)
    return prob


def _hash_matrix(h, m: Matrix) -> None:
    for row in m:
        h.update(",".join(float(x).hex() for x in row).encode())
        h.update(b";")


def problem_hash(prob: SyntheticProblem) -> str:
    h = hashlib.sha256()
    for ad, t in zip(prob.adapters, prob.tasks):
        for m in (ad.B, ad.A, t.x_cal, t.x_dev or [[0.0]], t.x_test):
            _hash_matrix(h, m)
        h.update(float(ad.scaling).hex().encode())
    return h.hexdigest()


def with_tasks(tasks: Sequence[TaskSplit], deltas: Sequence[Matrix]) -> List[TaskSplit]:
    return [TaskSplit(name=t.name, delta=d, x_cal=t.x_cal, x_test=t.x_test, x_dev=t.x_dev)
            for t, d in zip(tasks, deltas)]


def split_sizes(prob: SyntheticProblem) -> Dict[str, Tuple[int, int, int]]:
    return {t.name: (len(t.x_cal[0]), len(t.x_dev[0]) if t.x_dev else 0, len(t.x_test[0]))
            for t in prob.tasks}
