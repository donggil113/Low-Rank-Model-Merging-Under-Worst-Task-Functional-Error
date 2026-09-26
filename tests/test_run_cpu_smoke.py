"""End-to-end smoke test of the CPU entry point on a tiny config."""

import contextlib
import io
import json
import os
import tempfile
import unittest

from _util import ROOT

from lowrank_merge import run_cpu


class TestRunCpuSmoke(unittest.TestCase):
    def test_tiny_run_writes_all_artifacts(self):
        with open(os.path.join(ROOT, "configs", "cpu_synthetic.json")) as f:
            cfg = json.load(f)
        cfg.update({
            "experiment": "smoke",
            "seeds": [0],
            "k": 2,
            "synthetic": {"d_out": 5, "d_in": 6, "n_tasks": 3, "lora_rank": 2,
                          "adapter_gains": [1.0, 0.7, 1.3], "n_cal": 12, "n_dev": 10,
                          "n_test": 30},
            "dual": {"iters": 40, "polish_iters": 20},
            "refine": {"block_rounds": 1,
                       "block_dual": {"iters": 30, "polish_iters": 10, "mode": "pinv"},
                       "local_restarts": 0, "local_iters": 10, "local_taus": [0.01]},
            "tuning_grids": {"coef": [0.5], "ties_coef": [1.0], "ties_density": [0.5],
                             "regmean_alpha": [1.0]},
            "singular_case": {"enabled": True, "input_subspace_dim": 4, "n_cal": 4,
                              "seed": 1, "ridges": [0.01]},
            "reference_checks": {"enabled": False},
        })
        with tempfile.TemporaryDirectory() as d:
            cfg_path = os.path.join(d, "cfg.json")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            out = os.path.join(d, "run")
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_cpu.main(["--config", cfg_path, "--out", out])
            self.assertEqual(code, 0)
            for name in ("raw_log.jsonl", "results.json", "summary.md", "manifest.json"):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)
            with open(os.path.join(out, "manifest.json")) as f:
                man = json.load(f)
            self.assertEqual(man["status"], "COMPLETED")
            self.assertIn("0", {str(k) for k in man["data_sha256_per_seed"]})
            with open(os.path.join(out, "results.json")) as f:
                res = json.load(f)
            statuses = {m: s["status"] for m, s in res["per_method"].items()}
            self.assertTrue(all(v == ["OK"] for v in statuses.values()), statuses)
            modes = {r["mode"]: r["status"] for r in res["singular_gram_case"]}
            self.assertEqual(modes["pd"], "REFUSED_AS_DESIGNED")
            self.assertEqual(modes["pinv"], "OK")


if __name__ == "__main__":
    unittest.main()
