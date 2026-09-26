"""Pilot preflight: the shipped config is complete and correctly BLOCKED."""

import contextlib
import copy
import io
import json
import os
import tempfile
import unittest

from _util import ROOT

from lowrank_merge import pilot

CFG = os.path.join(ROOT, "configs", "pilot_4task_flan_t5_base_glue.json")


def load():
    with open(CFG) as f:
        return json.load(f)


class TestPilotConfig(unittest.TestCase):
    def test_config_is_complete(self):
        self.assertEqual(pilot.validate(load()), [])

    def test_blocked_on_approval_license_and_pins(self):
        codes = {b["code"] for b in pilot.blockers(load())}
        self.assertIn("BLOCKED_APPROVAL", codes)
        self.assertIn("BLOCKED_LICENSE", codes)
        self.assertIn("BLOCKED_UNPINNED", codes)

    def test_every_method_discloses_access(self):
        cfg = load()
        for name, m in cfg["methods"].items():
            for f in pilot.REQUIRED_METHOD_KEYS:
                self.assertIn(f, m, name)
        self.assertTrue(cfg["task_identity_disclosure"]["merging_uses_task_identity"])

    def test_validation_catches_mistakes(self):
        cfg = load()
        bad = copy.deepcopy(cfg)
        bad["adapters"] = bad["adapters"][:3]
        self.assertTrue(any("exactly 4" in i["detail"] for i in pilot.validate(bad)))
        bad = copy.deepcopy(cfg)
        bad["adapters"][1]["declared_base"] = "t5-base"
        self.assertTrue(any("declares base" in i["detail"] for i in pilot.validate(bad)))
        bad = copy.deepcopy(cfg)
        del bad["methods"]["minimax_rel"]["calibration_inputs"]
        self.assertTrue(any("access field" in i["detail"] for i in pilot.validate(bad)))
        bad = copy.deepcopy(cfg)
        bad["splits"]["calibration"]["labels_used"] = True
        self.assertTrue(pilot.validate(bad))
        bad = copy.deepcopy(cfg)
        del bad["minimum_effect_of_interest"]
        self.assertTrue(pilot.validate(bad))

    def test_cli_writes_report_and_exits_nonzero_when_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(io.StringIO()):
                code = pilot.main(["--config", CFG, "--out", d])
            self.assertEqual(code, 2)
            with open(os.path.join(d, "preflight.json")) as f:
                rep = json.load(f)
            self.assertEqual(rep["status"], "BLOCKED")
            self.assertEqual(rep["science_status"], "SCIENCE_NOT_EVALUATED")


if __name__ == "__main__":
    unittest.main()
