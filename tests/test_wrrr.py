"""Fixed-weight weighted reduced-rank regression vs independent references.

Numerical agreement on these instances is implementation evidence only.
"""

import random
import unittest

from _util import la, make_raw_tasks, rel_diff, rel_scalar

from lowrank_merge.objectives import (QuadTask, TaskSplit, functional_error, gram,
                                      quad_task_from_data, quad_tasks)
from lowrank_merge.references import (RawTask, als_weighted, grid_rank1_weighted,
                                      raw_weighted, rrr_izenman_raw)
from lowrank_merge.wrrr import (SingularGramError, UnboundedObjectiveError, wrrr)


def rand_simplex(rng, t):
    w = [rng.random() + 0.05 for _ in range(t)]
    s = sum(w)
    return [x / s for x in w]


class TestObjective(unittest.TestCase):
    def test_gram_form_equals_raw_samples(self):
        rng = random.Random(0)
        splits, _ = make_raw_tasks(rng, 5, 4, 2, 9)
        m = la.random_normal(5, 4, rng)
        for t in splits:
            q = quad_task_from_data(t.name, t.delta, t.x_cal, relative=False)
            self.assertLess(rel_scalar(q.value(m), functional_error(m, t.delta, t.x_cal)), 1e-11)
            qr = quad_task_from_data(t.name, t.delta, t.x_cal, relative=True)
            self.assertAlmostEqual(qr.value(la.zeros(5, 4)), 1.0, places=12)


class TestWRRRAgainstReferences(unittest.TestCase):
    def test_matches_izenman_raw_data_rrr(self):
        rng = random.Random(1)
        for trial in range(6):
            d_out, d_in = rng.choice([(5, 7), (7, 5), (6, 6)])
            t_n = rng.choice([1, 2, 3, 4])
            relative = bool(trial % 2)
            splits, raws = make_raw_tasks(rng, d_out, d_in, t_n, 12, relative=relative,
                                          cov_scales=[0.5 + rng.random() for _ in range(t_n)])
            tasks = quad_tasks(splits, "cal", relative=relative)
            lam = rand_simplex(rng, t_n)
            for k in range(0, min(d_out, d_in) + 1):
                res = wrrr(tasks, lam, k)
                ref = rrr_izenman_raw(raws, lam, k) if k > 0 else la.zeros(d_out, d_in)
                v_res = raw_weighted(res.M, raws, lam)
                v_ref = raw_weighted(ref, raws, lam)
                # absolute tolerance on the scale of the zero-merge objective
                # (exact recovery makes both values ~0, where relative error is meaningless)
                scale0 = raw_weighted(la.zeros(d_out, d_in), raws, lam)
                self.assertLess(abs(v_res - v_ref), 1e-10 * scale0, (trial, k))
                self.assertLess(abs(res.objective - v_ref), 1e-10 * scale0)
                if res.unique:
                    self.assertLess(rel_diff(res.M, ref), 1e-7, (trial, k))
                self.assertLessEqual(la.numerical_rank(res.M), k)

    def test_matches_grid_search_rank1_2d(self):
        rng = random.Random(2)
        for trial in range(3):
            d_in = 3
            splits, raws = make_raw_tasks(rng, 2, d_in, 3, 8, relative=True)
            tasks = quad_tasks(splits, "cal", relative=True)
            lam = rand_simplex(rng, 3)
            res = wrrr(tasks, lam, 1)
            v_grid, m_grid = grid_rank1_weighted(raws, lam, n_grid=720)
            # closed form must never be worse than the grid, and the grid must reach it
            self.assertLessEqual(res.objective, v_grid * (1 + 1e-10))
            self.assertLess(rel_scalar(res.objective, v_grid), 1e-8, trial)

    def test_never_beaten_by_multistart_als(self):
        rng = random.Random(3)
        splits, raws = make_raw_tasks(rng, 5, 4, 3, 10, relative=True)
        tasks = quad_tasks(splits, "cal", relative=True)
        lam = rand_simplex(rng, 3)
        res = wrrr(tasks, lam, 2)
        v_als, _ = als_weighted(raws, lam, 2, restarts=10, iters=300, seed=0)
        self.assertLessEqual(res.objective, v_als * (1 + 1e-9))
        self.assertLess(rel_scalar(res.objective, v_als), 1e-6)


class TestWRRRSpecialCases(unittest.TestCase):
    def test_single_task_exact_recovery_when_k_ge_rank(self):
        rng = random.Random(4)
        splits, _ = make_raw_tasks(rng, 6, 5, 1, 20, delta_rank=2)
        tasks = quad_tasks(splits, "cal", relative=False)
        res = wrrr(tasks, [1.0], 2)
        self.assertLess(rel_diff(res.M, splits[0].delta), 1e-9)
        self.assertLess(functional_error(res.M, splits[0].delta, splits[0].x_cal),
                        1e-18 * la.frob2(splits[0].delta) + 1e-20)

    def test_single_task_identity_cov_is_eckart_young(self):
        rng = random.Random(5)
        d = la.random_normal(4, 6, rng)
        t = QuadTask(name="t", G=la.eye(6), H=d, c=la.frob2(d))
        s = la.singular_values(d)
        for k in range(0, 5):
            res = wrrr([t], [1.0], k)
            self.assertLess(rel_diff(res.M, la.svd_truncate(d, k)) if k else la.frob(res.M), 1e-9)
            self.assertAlmostEqual(res.objective, sum(x * x for x in s[k:]), places=9)

    def test_identity_covariances_uniform_weights_is_ta_svd(self):
        rng = random.Random(6)
        ds = [la.random_normal(5, 5, rng) for _ in range(3)]
        tasks = [QuadTask(name=str(i), G=la.eye(5), H=d, c=la.frob2(d)) for i, d in enumerate(ds)]
        mean = la.lincomb([1 / 3] * 3, ds)
        res = wrrr(tasks, [1 / 3] * 3, 2)
        self.assertLess(rel_diff(res.M, la.svd_truncate(mean, 2)), 1e-9)

    def test_full_rank_is_weighted_regmean(self):
        rng = random.Random(7)
        splits, _ = make_raw_tasks(rng, 4, 6, 3, 10)
        tasks = quad_tasks(splits, "cal", relative=False)
        lam = [0.2, 0.3, 0.5]
        s = la.lincomb(lam, [gram(t.x_cal) for t in splits])
        c = la.lincomb(lam, [la.matmul(t.delta, gram(t.x_cal)) for t in splits])
        regmean = la.transpose(la.solve(s, la.transpose(c)))  # C S^{-1}
        for k in (None, 4, 10):
            res = wrrr(tasks, lam, k)
            self.assertLess(rel_diff(res.M, regmean), 1e-9)

    def test_k_zero_is_zero(self):
        rng = random.Random(8)
        splits, _ = make_raw_tasks(rng, 3, 3, 2, 5)
        res = wrrr(quad_tasks(splits, "cal", False), [0.5, 0.5], 0)
        self.assertEqual(la.frob(res.M), 0.0)

    def test_nonunique_truncation_is_flagged(self):
        d1 = [[1.0, 0.0], [0.0, 0.0]]
        d2 = [[0.0, 0.0], [0.0, 1.0]]
        tasks = [QuadTask("a", la.eye(2), d1, 1.0), QuadTask("b", la.eye(2), d2, 1.0)]
        res = wrrr(tasks, [0.5, 0.5], 1)
        self.assertFalse(res.unique)
        self.assertAlmostEqual(res.objective, 0.75, places=12)


class TestSingularGram(unittest.TestCase):
    """Inputs confined to a subspace: S singular. pd / pinv / ridge are distinct."""

    def setUp(self):
        rng = random.Random(9)
        d_out, d_in, sub = 4, 6, 3
        basis = la.columns(la.random_orthogonal(d_in, rng), range(sub))
        self.basis = basis
        self.splits, self.raws = [], []
        for t in range(2):
            delta = la.random_normal(d_out, d_in, rng)
            x = la.matmul(basis, la.random_normal(sub, 7, rng))
            xt = la.random_normal(d_in, 7, rng)  # test inputs leave the subspace
            self.splits.append(TaskSplit(f"t{t}", delta, x, xt))
            self.raws.append(RawTask(delta, x))
        self.tasks = quad_tasks(self.splits, "cal", relative=False)
        self.lam = [0.4, 0.6]

    def test_pd_mode_refuses(self):
        with self.assertRaises(SingularGramError):
            wrrr(self.tasks, self.lam, 2, mode="pd")

    def test_pinv_is_exact_minimiser_on_range_and_zero_on_null(self):
        res = wrrr(self.tasks, self.lam, 2, mode="pinv")
        self.assertEqual(res.gram_null_dim, 3)
        self.assertTrue(res.exact_for_original_objective)
        # reference: reduce to coordinates in the input subspace
        b = self.basis
        reduced = [RawTask(la.matmul(r.delta, b), la.matmul(la.transpose(b), r.x))
                   for r in self.raws]
        ref_small = rrr_izenman_raw(reduced, self.lam, 2)
        v_ref = raw_weighted(ref_small, reduced, self.lam)
        self.assertLess(rel_scalar(res.objective, v_ref), 1e-9)
        # acts as zero on null(S): M (I - B B^T) = 0
        p_null = la.sub(la.eye(6), la.matmul(b, la.transpose(b)))
        self.assertLess(la.frob(la.matmul(res.M, p_null)), 1e-9 * la.frob(res.M))
        # non-identifiability: adding any Y P_null leaves the calibration objective unchanged
        y = la.random_normal(4, 6, random.Random(10))
        m2 = la.add(res.M, la.matmul(y, p_null))
        self.assertLess(rel_scalar(raw_weighted(m2, self.raws, self.lam), res.objective), 1e-9)

    def test_ridge_changes_objective_and_converges_to_pinv(self):
        pinv = wrrr(self.tasks, self.lam, 2, mode="pinv")
        prev = None
        for eps in (1e-1, 1e-3, 1e-5, 1e-7):
            rid = wrrr(self.tasks, self.lam, 2, mode="ridge", ridge=eps)
            self.assertFalse(rid.exact_for_original_objective)
            # pinv minimises the original objective, ridge minimises the penalised one
            self.assertLessEqual(pinv.objective, rid.objective * (1 + 1e-12))
            pen_pinv = pinv.objective + eps * la.frob2(pinv.M)
            self.assertLessEqual(rid.penalized_objective, pen_pinv * (1 + 1e-12))
            d = rel_diff(rid.M, pinv.M)
            if prev is not None:
                self.assertLess(d, prev)
            prev = d
        self.assertLess(prev, 1e-4)

    def test_ridge_requires_explicit_mode(self):
        with self.assertRaises(ValueError):
            wrrr(self.tasks, self.lam, 2, mode="pinv", ridge=1e-3)

    def test_unbounded_objective_detected(self):
        # hand-built task whose linear term leaks into null(G)
        g = [[1.0, 0.0], [0.0, 0.0]]
        h = [[1.0, 1.0]]
        with self.assertRaises(UnboundedObjectiveError):
            wrrr([QuadTask("bad", g, h, 1.0)], [1.0], 1, mode="pinv")


if __name__ == "__main__":
    unittest.main()
