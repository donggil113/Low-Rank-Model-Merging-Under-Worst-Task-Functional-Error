"""(B R, R^{-1} A) invariance, final rank, parameter accounting, split hygiene."""

import random
import unittest

from _util import la, rel_diff

from lowrank_merge import baselines as bl
from lowrank_merge.adapters import LoRAAdapter, factorize_rank_k, param_account, random_gauge
from lowrank_merge.minimax import DualConfig, RefineConfig, solve_minimax
from lowrank_merge.objectives import TaskSplit, evaluate, quad_tasks
from lowrank_merge.synthetic import SyntheticSpec, make_problem, problem_hash
from lowrank_merge.wrrr import wrrr

SMALL = SyntheticSpec(d_out=5, d_in=6, n_tasks=3, lora_rank=2, adapter_gains=(1.0, 0.7, 1.3),
                      n_cal=12, n_dev=10, n_test=40)
FAST_DUAL = DualConfig(iters=60, polish_iters=40)
FAST_REFINE = RefineConfig(block_rounds=2, block_dual=DualConfig(iters=60, polish_iters=40,
                                                                 mode="pinv"),
                           local_restarts=1, local_iters=40, local_taus=(1e-2, 1e-3))


def all_methods(adapters, tasks, k):
    deltas = [ad.effective_delta() for ad in adapters]
    xs = [t.x_cal for t in tasks]
    ts = [TaskSplit(t.name, d, t.x_cal, t.x_test, t.x_dev) for t, d in zip(tasks, deltas)]
    qt = quad_tasks(ts, "cal", relative=True)
    out = {
        "ta_svd": bl.ta_svd(deltas, k, 0.5).M,
        "knots_ta": bl.knots(deltas, k, "ta", 0.5).M,
        "knots_ties": bl.knots(deltas, k, "ties", 1.0, 0.5).M,
        "ctm_like": bl.ctm_like_ta(deltas, k, 0.5).M,
        "regmean_euclid": bl.regmean(deltas, xs, 0.9, k, "euclid").M,
        "regmean_whitened": bl.regmean(deltas, xs, 1.0, k, "whitened").M,
        "wrrr_uniform": wrrr(qt, [1 / len(qt)] * len(qt), k).M,
        "minimax": solve_minimax(qt, k, FAST_DUAL, FAST_REFINE).M,
        "factor_average": bl.factor_average(adapters, k).M,
    }
    return out


class TestGaugeInvariance(unittest.TestCase):
    def setUp(self):
        self.prob = make_problem(SMALL, 3)
        self.k = 2

    def _gauged(self, cond, seed):
        rng = random.Random(seed)
        return [ad.gauge(random_gauge(ad.rank_budget, cond, rng)) for ad in self.prob.adapters]

    def test_effective_delta_invariant(self):
        for ad, ag in zip(self.prob.adapters, self._gauged(10.0, 1)):
            self.assertLess(rel_diff(ad.effective_delta(), ag.effective_delta()), 1e-13)

    def test_all_delta_based_methods_invariant_factor_average_not(self):
        ref = all_methods(self.prob.adapters, self.prob.tasks, self.k)
        got = all_methods(self._gauged(10.0, 2), self.prob.tasks, self.k)
        for name in ref:
            d = rel_diff(ref[name], got[name])
            if name == "factor_average":
                self.assertGreater(d, 1e-2, "factor averaging should NOT be invariant")
            else:
                self.assertLess(d, 1e-8, name)

    def test_ill_conditioned_gauge_degrades_only_by_rounding(self):
        ref = all_methods(self.prob.adapters, self.prob.tasks, self.k)
        got = all_methods(self._gauged(1e6, 4), self.prob.tasks, self.k)
        for name in ("ta_svd", "wrrr_uniform", "regmean_whitened"):
            # rounding in B R and R^{-1} A scales with cond(R) * eps
            self.assertLess(rel_diff(ref[name], got[name]), 1e-6, name)


class TestRankAndAccounting(unittest.TestCase):
    def test_final_rank_within_budget(self):
        prob = make_problem(SMALL, 5)
        for k in (1, 2, 3):
            for name, m in all_methods(prob.adapters, prob.tasks, k).items():
                self.assertLessEqual(la.numerical_rank(m), k, (name, k))

    def test_knots_ta_equals_task_arithmetic(self):
        """Stoica et al. (2025) Sec. 5.1: KnOTS with linear averaging is TA."""
        prob = make_problem(SMALL, 6)
        deltas = [ad.effective_delta() for ad in prob.adapters]
        kn = bl.knots(deltas, None, "ta", 0.5)
        self.assertLess(rel_diff(kn.M, bl.task_arithmetic(deltas, 0.5)), 1e-10)
        self.assertGreater(kn.info["rank_before_truncation"], SMALL.lora_rank)

    def test_ctm_like_full_rank_is_ta_and_hooi_improves_fit(self):
        square = SyntheticSpec(**{**SMALL.__dict__, "d_out": 6, "d_in": 6})
        sq = [ad.effective_delta() for ad in make_problem(square, 10).adapters]
        # k = d: U and V are complete bases, so the projection is the identity
        full = bl.ctm_like_ta(sq, 6, 0.5)
        self.assertLess(rel_diff(full.M, bl.task_arithmetic(sq, 0.5)), 1e-9)
        prob = make_problem(SMALL, 10)
        deltas = [ad.effective_delta() for ad in prob.adapters]

        def captured(u_iters):
            out = bl.ctm_like_ta(deltas, 2, 1.0, hooi_iters=u_iters)
            u = la.orthonormal_basis(out.M, 2)
            v = la.orthonormal_basis(la.transpose(out.M), 2)
            return sum(la.frob2(la.matmul(la.matmul(la.transpose(u), d), v)) for d in deltas)
        self.assertGreaterEqual(captured(30), captured(0) * (1 - 1e-12))

    def test_regmean_full_rank_exceeds_budget(self):
        prob = make_problem(SMALL, 7)
        deltas = [ad.effective_delta() for ad in prob.adapters]
        out = bl.regmean(deltas, [t.x_cal for t in prob.tasks], 1.0, 2, "none")
        self.assertGreater(la.numerical_rank(out.M), 2)
        self.assertIsNone(out.target_rank)

    def test_param_accounting(self):
        prob = make_problem(SMALL, 8)
        acct = param_account(prob.adapters, merged_rank=2).as_dict()
        self.assertEqual(acct["input_adapter_params_total"], 3 * 2 * (5 + 6))
        self.assertEqual(acct["merged_params_if_stored_as_factors"], 2 * (5 + 6))
        self.assertEqual(acct["dense_delta_params"], 30)
        self.assertAlmostEqual(acct["merged_over_input_ratio"], 1 / 3)

    def test_factorize_roundtrip(self):
        rng = random.Random(9)
        m = la.matmul(la.random_normal(5, 2, rng), la.random_normal(2, 6, rng))
        ad = factorize_rank_k(m, 2)
        self.assertLess(rel_diff(ad.effective_delta(), m), 1e-12)
        self.assertEqual(ad.n_params(), 2 * 11)


class TestSplitsAndEvaluation(unittest.TestCase):
    def test_splits_are_disjoint_streams_and_stable(self):
        p1 = make_problem(SMALL, 11)
        bigger = SyntheticSpec(**{**SMALL.__dict__, "n_test": 80})
        p2 = make_problem(bigger, 11)
        for t1, t2 in zip(p1.tasks, p2.tasks):
            # changing the test size must not change calibration or dev samples
            self.assertEqual(t1.x_cal, t2.x_cal)
            self.assertEqual(t1.x_dev, t2.x_dev)
            cal_cols = {tuple(c) for c in zip(*t1.x_cal)}
            test_cols = {tuple(c) for c in zip(*t1.x_test)}
            dev_cols = {tuple(c) for c in zip(*t1.x_dev)}
            self.assertFalse(cal_cols & test_cols)
            self.assertFalse(cal_cols & dev_cols)
        self.assertEqual(problem_hash(p1), make_problem(SMALL, 11).meta["data_sha256"])
        self.assertNotEqual(problem_hash(p1), problem_hash(p2))

    def test_evaluate_per_task_and_summary(self):
        prob = make_problem(SMALL, 12)
        m = la.zeros(5, 6)
        ev = evaluate(m, prob.tasks, splits=("cal", "test"))
        for name, vals in ev["per_task"].items():
            self.assertAlmostEqual(vals["cal_rel"], 1.0, places=12)
            self.assertAlmostEqual(vals["test_rel"], 1.0, places=12)
        self.assertAlmostEqual(ev["summary"]["test"]["worst_rel"], 1.0, places=12)
        self.assertIn(ev["summary"]["test"]["worst_task"], ev["per_task"])


class TestAdapterValidation(unittest.TestCase):
    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            LoRAAdapter("bad", B=[[1.0, 2.0]], A=[[1.0, 2.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
