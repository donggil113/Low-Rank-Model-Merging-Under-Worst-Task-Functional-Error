"""Population moments of the synthetic fixture (oracle diagnostics only).

In synthetic.py every task input is x = L_t z with z ~ N(0, I) drawn by
random.gauss(0, 1) (see _task_input_map / _samples). Hence
    E[x] = L_t E[z] = 0   and   E[x x^T] = L_t L_t^T,
so the raw (uncentered) second moment, which is what the empirical
Sigma_hat = X X^T / n estimates, equals the covariance. The population
functional error and the relative-error normaliser are
    e_t(M) = tr((M - D_t) Sigma_t (M - D_t)^T) = ||(M - D_t) L_t||_F^2,
    E_t    = tr(D_t Sigma_t D_t^T)             = ||D_t L_t||_F^2.
The L_t form is evaluated directly (no Gram cancellation) and is used as the
common evaluation objective; mc_second_moment checks the closed form against
the generator itself.

Population information is not available to a deployable method.
"""

from __future__ import annotations

import math
from typing import List, Sequence

from . import linalg as la
from .linalg import Matrix
from .objectives import QuadTask
from .synthetic import SyntheticSpec, _samples, _sub_rng, _task_input_map


def input_map(spec: SyntheticSpec, teacher_seed: int, t: int) -> Matrix:
    return _task_input_map(spec, teacher_seed, t)


def population_moment(spec: SyntheticSpec, teacher_seed: int, t: int) -> Matrix:
    lmap = input_map(spec, teacher_seed, t)
    return la.matmul(lmap, la.transpose(lmap))


def mc_second_moment(spec: SyntheticSpec, teacher_seed: int, t: int, n: int,
                     seed: int) -> Matrix:
    """Monte Carlo E[x x^T] from the generator's own sampling routine."""
    x = _samples(input_map(spec, teacher_seed, t), n, _sub_rng(seed, "mc", teacher_seed, t))
    return la.scale(1.0 / n, la.gram_rows(x))


def mc_tolerance_ok(mc: Matrix, pop: Matrix, n: int, n_sigma: float = 6.0) -> tuple:
    """Entrywise check with Var(x_i x_j) = S_ii S_jj + S_ij^2 (zero-mean Gaussian)."""
    worst = 0.0
    d = len(pop)
    for i in range(d):
        for j in range(d):
            se = math.sqrt((pop[i][i] * pop[j][j] + pop[i][j] ** 2) / n)
            z = abs(mc[i][j] - pop[i][j]) / max(se, 1e-300)
            worst = max(worst, z)
    return worst <= n_sigma, worst


def resample_calibration(spec: SyntheticSpec, teacher_seed: int, t: int,
                         resample_seed: int) -> Matrix:
    """A fresh n_cal draw from the same task distribution (new stream)."""
    return _samples(input_map(spec, teacher_seed, t), spec.n_cal,
                    _sub_rng(teacher_seed, "x", t, "cal_resample", resample_seed))


def quad_task(name: str, delta: Matrix, moment: Matrix, norm_moment: Matrix) -> QuadTask:
    """Relative-error task whose fit moment and normaliser moment may differ."""
    h = la.matmul(delta, moment)
    c = la.inner(h, delta)
    e0 = la.inner(la.matmul(delta, norm_moment), delta)
    if e0 <= 0.0:
        raise ValueError(f"task {name}: zero normaliser")
    return QuadTask(name=name, G=moment, H=h, c=c, norm=e0)


def population_relative_errors(m: Matrix, deltas: Sequence[Matrix],
                               lmaps: Sequence[Matrix]) -> List[float]:
    """||(M - D_t) L_t||^2 / ||D_t L_t||^2 per task (common evaluation objective)."""
    out = []
    for d, lmap in zip(deltas, lmaps):
        num = la.frob2(la.matmul(la.sub(m, d), lmap))
        den = la.frob2(la.matmul(d, lmap))
        out.append(num / den)
    return out
