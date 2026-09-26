"""Regression tests for the stage-2 population / normaliser diagnostic."""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

from _util import ROOT, SRC, la, rel_scalar

from lowrank_merge import population as pp
from lowrank_merge.objectives import quad_task_from_data
from lowrank_merge.run_diag import classify, fit_cell
from lowrank_merge.synthetic import SyntheticSpec, make_problem

SMALL = SyntheticSpec(d_out=5, d_in=6, n_tasks=3, lora_rank=2, adapter_gains=(1.0, 0.7, 1.3),
                      n_cal=12, n_dev=10, n_test=40)
FIXTURE_FAST = {
    "wrrr_mode": "pd",
    "dual": {"iters": 60, "polish_iters": 40},
    "refine": {"block_rounds": 1, "block_dual": {"iters": 30, "polish_iters": 10, "mode": "pinv"},
               "local_restarts": 0, "local_iters": 10, "local_taus": [0.01]},
}


class TestPopulationMoments(unittest.TestCase):
    def test_closed_form_matches_generator_monte_carlo(self):
        pop = pp.population_moment(SMALL, 4, 1)
        est = pp.mc_second_moment(SMALL, 4, 1, 4000, 9)
        ok, worst = pp.mc_tolerance_ok(est, pop, 4000)
        self.assertTrue(ok, worst)

    def test_population_evaluator_matches_quadform(self):
        prob = make_problem(SMALL, 4)
        m = la.random_normal(5, 6, random.Random(0))
        for t_idx, t in enumerate(prob.tasks):
            lmap = pp.input_map(SMALL, 4, t_idx)
            pop = pp.population_moment(SMALL, 4, t_idx)
            q = pp.quad_task(t.name, t.delta, pop, pop)
            direct = pp.population_relative_errors(m, [t.delta], [lmap])[0]
            self.assertLess(rel_scalar(q.value(m), direct), 1e-10)

    def test_empirical_arm_equals_existing_relative_task(self):
        prob = make_problem(SMALL, 5)
        m = la.random_normal(5, 6, random.Random(1))
        for t in prob.tasks:
            g = la.scale(1.0 / len(t.x_cal[0]), la.gram_rows(t.x_cal))
            a = pp.quad_task(t.name, t.delta, g, g).value(m)
            b = quad_task_from_data(t.name, t.delta, t.x_cal, relative=True).value(m)
            self.assertLess(rel_scalar(a, b), 1e-12)

    def test_resample_is_new_and_deterministic(self):
        prob = make_problem(SMALL, 6)
        a = pp.resample_calibration(SMALL, 6, 0, 101)
        self.assertEqual(a, pp.resample_calibration(SMALL, 6, 0, 101))
        self.assertNotEqual(a, prob.tasks[0].x_cal)
        self.assertNotEqual(a, pp.resample_calibration(SMALL, 6, 0, 102))
        self.assertEqual(la.shape(a), (6, SMALL.n_cal))


class TestDiagnosticLogic(unittest.TestCase):
    def test_classify_rules(self):
        pos = {0: [0.1] * 3, 1: [0.05] * 3, 2: [-0.01] * 3}
        self.assertEqual(classify(pos, None)["verdict"], "PROBLEM_PRESENT")
        neg = {0: [-0.1] * 3, 1: [0.01] * 3, 2: [-0.01] * 3}
        self.assertEqual(classify(neg, None)["verdict"], "PROBLEM_ABSENT")
        amb = {0: [0.01] * 3, 1: [0.01] * 3, 2: [-0.005] * 3}
        self.assertEqual(classify(amb, None)["verdict"], "AMBIGUOUS")
        self.assertEqual(classify({0: [0.1] * 3}, None)["verdict"], "INCOMPLETE_CELLS")

    def test_pop_pop_fit_is_consistent_and_audited(self):
        prob = make_problem(SMALL, 7)
        lmaps = [pp.input_map(SMALL, 7, t) for t in range(3)]
        pops = [pp.population_moment(SMALL, 7, t) for t in range(3)]
        arm = {"name": "pop_moment__pop_norm", "moment": "population", "normalizer": "population"}
        recs = fit_cell(7, "all_draws", arm, prob.tasks, [t.x_cal for t in prob.tasks], pops,
                        lmaps, SMALL, FIXTURE_FAST, 2)
        by = {r["method"]: r for r in recs}
        mm, un = by["minimax_rel"], by["wrrr_uniform_rel"]
        slack = mm["audit"]["upper_bound_quadform"] - mm["audit"]["lower_bound_quadform"]
        # fit objective == evaluation objective in this arm
        self.assertLess(abs(mm["fit_objective_max"] - mm["F_pop"]), 1e-10)
        self.assertLessEqual(mm["F_pop"], un["F_pop"] + slack + 1e-12)
        self.assertLess(mm["audit"]["g_rel_discrepancy"], 1e-8)
        self.assertLessEqual(mm["rank_M"], 2)


class TestRunDiagCaps(unittest.TestCase):
    """Runs the entry point in a child process so the rlimits never touch this process."""

    def _run(self, cpu_cap):
        with tempfile.TemporaryDirectory() as d:
            fixture = {**FIXTURE_FAST, "synthetic": {
                "d_out": 5, "d_in": 6, "n_tasks": 3, "lora_rank": 2,
                "adapter_gains": [1.0, 0.7, 1.3], "n_cal": 12, "n_dev": 10, "n_test": 40}}
            fpath = os.path.join(d, "fixture.json")
            with open(fpath, "w") as f:
                json.dump(fixture, f)
            with open(os.path.join(ROOT, "configs", "p2_stage2_moment_normalizer_diag.json")) as f:
                cfg = json.load(f)
            cfg.update({"inherits_fixture_config": fpath, "expected_teacher_data_sha256": {},
                        "k": 2, "calibration_resample_seeds": [101, 102, 103],
                        "monte_carlo_check": {"n_samples": 2000, "seed": 1, "teachers": [0]}})
            cfg["resource_caps"] = {**cfg["resource_caps"], "cpu_seconds_total": cpu_cap}
            cpath = os.path.join(d, "cfg.json")
            with open(cpath, "w") as f:
                json.dump(cfg, f)
            out = os.path.join(d, "run")
            env = {**os.environ, "PYTHONPATH": SRC}
            proc = subprocess.run([sys.executable, "-m", "lowrank_merge.run_diag", "--config",
                                   cpath, "--out", out], env=env, capture_output=True,
                                  text=True, timeout=600)
            with open(os.path.join(out, "results.json")) as f:
                res = json.load(f)
            with open(os.path.join(out, "manifest.json")) as f:
                man = json.load(f)
            return proc.returncode, res, man

    def test_tiny_cap_records_not_run_instead_of_starting(self):
        code, res, man = self._run(40)
        self.assertEqual(code, 1)
        self.assertEqual(res["status"], "STOPPED_AT_CPU_CAP")
        self.assertNotIn("OK", res["cell_status_counts"])
        self.assertGreater(res["cell_status_counts"]["NOT_RUN_CAP"], 0)
        self.assertEqual(man["caps"]["RLIMIT_CPU"][0], 40)

    def test_small_full_run_completes(self):
        code, res, man = self._run(900)
        self.assertEqual(code, 0, res.get("status"))
        # 3 teachers x (4 draws x 3 arms + 1 pop/pop) x 2 methods
        self.assertEqual(res["cell_status_counts"], {"OK": 78})
        self.assertTrue(res["bound_audit"]["all_solver_failure_refuted"] in (True, False))
        self.assertTrue(all(c["consistent"] for c in res["pop_pop_consistency"]))


if __name__ == "__main__":
    unittest.main()
