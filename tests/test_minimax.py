"""Worst-task solver: bound validity, known special solutions, references.

The solver makes no global-optimality claim. Hard assertions are
  * lower_bound <= independent reference optimum <= upper_bound,
  * known closed-form / published optima on special instances.
Solver quality (upper bound minus reference) is a regression guard only.
"""

import math
import random
import unittest

from _util import la, make_raw_tasks, rel_diff

from lowrank_merge.minimax import (DualConfig, dual_ascent, max_error, solve_minimax)
from lowrank_merge.objectives import QuadTask, quad_tasks
from lowrank_merge.references import grid_rank1_minimax
from lowrank_merge.wrrr import wrrr


def fair_pca_tasks(bs, const=None):
    """D_t = I: e_t(M) = tr(M B M^T) - 2 tr(M B) + c_t.

    For a projection P this is c_t - <B_t, P>; any rank-k M can be replaced by
    the projector onto its column space without increasing any e_t, so
    min_{rank<=k} max_t e_t is min-max (reconstruction-type) fair PCA.
    """
    return [QuadTask(f"g{i}", b, b, la.trace(b) if const is None else const)
            for i, b in enumerate(bs)]


class TestKnownSolutions(unittest.TestCase):
    def test_published_nonzero_gap_instance(self):
        """Tantipongpipat et al. (NeurIPS 2019), Lemma 6.2, k=3 groups, d=1.

        max_P min_i <B_i, P>: SDP relaxation value 7/4, exact rank-1 value
        26/17 at P = [[16,4],[4,1]]/17. With a common constant c = 4 the
        worst-task error is 4 - min_i <B_i, P>, so OPT = 4 - 26/17 while the
        Lagrangian dual (= Fantope relaxation, by Sion) is 4 - 7/4.
        """
        bs = [[[2.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 2.0]], [[2.0, -1.0], [-1.0, 2.0]]]
        res = solve_minimax(fair_pca_tasks(bs, const=4.0), 1)
        self.assertAlmostEqual(res.lower_bound, 4.0 - 7.0 / 4.0, delta=1e-6)
        self.assertAlmostEqual(res.upper_bound, 4.0 - 26.0 / 17.0, delta=1e-6)
        self.assertGreater(res.gap, 0.2)
        self.assertFalse(res.certified(1e-3))
        x_hat = [[16 / 17, 4 / 17], [4 / 17, 1 / 17]]
        self.assertLess(rel_diff(res.M, x_hat), 1e-4)

    def test_two_group_projection_case_has_zero_gap(self):
        """D_t = I, T = 2: the SDP relaxation is exact (Tantipongpipat et al.
        2019, Thm 1.2 / Sec. 2 with affine f_i), so the dual bound is tight."""
        rng = random.Random(5)
        for trial in range(3):
            bs = []
            for _ in range(2):
                a = la.random_normal(4, 4, rng)
                bs.append(la.matmul(a, la.transpose(a)))
            for k in (1, 2):
                res = solve_minimax(fair_pca_tasks(bs), k)
                self.assertLess(res.gap_rel, 1e-6, (trial, k))

    def test_symmetric_two_task_rank1(self):
        """D1 = e1 e1^T, D2 = e2 e2^T, Sigma = I, k = 1: OPT = 3/4 at
        M = (1/4) [[1,1],[1,1]]; lambda = (1/2,1/2) is a tie (non-unique
        truncation) and g(1/2,1/2) = 3/4, so the bound is tight."""
        d1 = [[1.0, 0.0], [0.0, 0.0]]
        d2 = [[0.0, 0.0], [0.0, 1.0]]
        tasks = [QuadTask("a", la.eye(2), d1, 1.0), QuadTask("b", la.eye(2), d2, 1.0)]
        res = solve_minimax(tasks, 1)
        self.assertAlmostEqual(res.lower_bound, 0.75, places=9)
        self.assertAlmostEqual(res.upper_bound, 0.75, places=7)
        self.assertLess(rel_diff(res.M, [[0.25, 0.25], [0.25, 0.25]]), 1e-4)

    def test_single_task_equals_eckart_young(self):
        rng = random.Random(6)
        splits, _ = make_raw_tasks(rng, 4, 5, 1, 12)
        tasks = quad_tasks(splits, "cal", relative=True)
        for k in (1, 2, 3):
            ref = wrrr(tasks, [1.0], k).objective
            res = solve_minimax(tasks, k)
            self.assertAlmostEqual(res.lower_bound, ref, delta=1e-10)
            self.assertAlmostEqual(res.upper_bound, ref, delta=1e-10)

    def test_full_rank_convex_case_closes_gap(self):
        rng = random.Random(7)
        for _ in range(3):
            splits, _ = make_raw_tasks(rng, 3, 4, 3, 8)
            tasks = quad_tasks(splits, "cal", relative=True)
            res = solve_minimax(tasks, 3)
            self.assertLess(res.gap_rel, 1e-6)


class TestBoundsAgainstReference(unittest.TestCase):
    def test_bounds_bracket_grid_ellipsoid_reference(self):
        rng = random.Random(11)
        quality = []
        for trial in range(3):
            n_tasks = 2 + trial % 2
            splits, raws = make_raw_tasks(rng, 2, 2, n_tasks, 6, relative=True)
            tasks = quad_tasks(splits, "cal", relative=True)
            res = solve_minimax(tasks, 1)
            ref, _ = grid_rank1_minimax(raws, n_grid=240)
            self.assertLessEqual(res.lower_bound, ref * (1 + 1e-9), trial)
            self.assertLessEqual(ref, res.upper_bound * (1 + 1e-9), trial)
            quality.append((res.upper_bound - ref) / ref)
        # regression guard, set after observing <= 3e-4 relative on these instances
        self.assertLess(max(quality), 1e-2)

    def test_weak_duality_against_random_feasible_points(self):
        rng = random.Random(12)
        splits, _ = make_raw_tasks(rng, 4, 5, 3, 10)
        tasks = quad_tasks(splits, "cal", relative=True)
        dres = dual_ascent(tasks, 2, DualConfig(iters=100, polish_iters=50))
        for _ in range(200):
            m = la.matmul(la.random_normal(4, 2, rng), la.random_normal(2, 5, rng))
            f, _ = max_error(tasks, m)
            self.assertLessEqual(dres.lower_bound, f)

    def test_refinement_is_monotone_and_rank_feasible(self):
        rng = random.Random(13)
        splits, _ = make_raw_tasks(rng, 5, 6, 4, 10, relative=True)
        tasks = quad_tasks(splits, "cal", relative=True)
        res = solve_minimax(tasks, 2)
        self.assertLessEqual(res.upper_bound, res.dual.upper_bound + 1e-15)
        accepted = [s["F"] for s in res.refine_log
                    if s["stage"] == "start" or s.get("accepted")]
        for a, b in zip(accepted, accepted[1:]):
            self.assertLess(b, a)
        self.assertLessEqual(la.numerical_rank(res.M), 2)
        self.assertLessEqual(res.lower_bound, res.upper_bound)

    def test_ridge_refused_for_dual_bound(self):
        rng = random.Random(14)
        splits, _ = make_raw_tasks(rng, 3, 3, 2, 6)
        with self.assertRaises(ValueError):
            dual_ascent(quad_tasks(splits, "cal", True), 1, DualConfig(mode="ridge"))


if __name__ == "__main__":
    unittest.main()
