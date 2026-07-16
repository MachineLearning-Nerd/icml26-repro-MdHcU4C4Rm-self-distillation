from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "full"


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((OUT / "summary.json").read_text())
        cls.structural = pd.read_csv(OUT / "structural_trials.csv")
        cls.stationary = pd.read_csv(OUT / "stationary_counterexamples.csv")
        cls.oneshot = pd.read_csv(OUT / "oneshot_trials.csv")
        cls.limit = pd.read_csv(OUT / "isotropic_limit.csv")
        cls.brute = pd.read_csv(OUT / "brute_force_checks.csv")

    def test_theorem_22_identities(self):
        np.testing.assert_allclose(self.structural.xi_oracle, self.structural.xi_derivative,
                                   rtol=1e-10, atol=2e-12)
        np.testing.assert_allclose(self.structural.gain, self.structural.gain_derivative,
                                   rtol=1e-10, atol=2e-12)
        self.assertTrue((self.structural.D > 1e-12).all())

    def test_claim_one_is_falsified_as_written(self):
        self.assertEqual(len(self.stationary), 64)
        self.assertTrue((self.stationary.D > 1e-12).all())
        self.assertLess(self.stationary.risk_derivative.abs().max(), 3e-13)
        self.assertLess(self.stationary.xi_oracle.abs().max(), 1e-12)
        self.assertLess(self.stationary.gain.abs().max(), 1e-24)

    def test_corrected_nonstationary_theorem(self):
        self.assertEqual(len(self.structural), 1408)
        self.assertTrue((self.structural.gain > 1e-10).all())
        self.assertTrue((np.sign(self.structural.xi_oracle) ==
                         -np.sign(self.structural.risk_derivative)).all())

    def test_negative_weights_and_exact_transition(self):
        self.assertGreater((self.structural.xi_oracle < -1e-10).sum(), 500)
        below = self.limit[self.limit["lambda"] < 0.5]
        above = self.limit[self.limit["lambda"] > 0.5]
        at = self.limit[np.isclose(self.limit["lambda"], 0.5)]
        self.assertTrue((below.xi_optimal > 0).all())
        self.assertTrue((above.xi_optimal < 0).all())
        self.assertEqual(float(at.xi_optimal.iloc[0]), 0.0)
        self.assertEqual(float(at.gain.iloc[0]), 0.0)

    def test_one_shot_consistency(self):
        self.assertEqual(len(self.oneshot), 288)
        self.assertTrue((self.oneshot.D_gcv > 0).all())
        aggregate = self.oneshot.groupby(["p", "lambda"]).agg(
            weight=("abs_weight_error", "median"), risk=("excess_risk", "median"))
        for lam in (0.1, 0.5, 2.0):
            self.assertLess(aggregate.loc[(400, lam), "weight"], aggregate.loc[(50, lam), "weight"])
            self.assertLess(aggregate.loc[(400, lam), "risk"], aggregate.loc[(50, lam), "risk"])

    def test_independent_brute_force_argmin(self):
        self.assertEqual(len(self.brute), 12)
        self.assertLessEqual(self.brute.abs_difference.max(), self.brute.grid_step.max() / 2 + 1e-12)

    def test_complete_outputs_and_cpu_metadata(self):
        expected = {"claim_evidence.csv", "structural_trials.csv", "stationary_counterexamples.csv",
                    "oneshot_trials.csv", "oneshot_summary.csv", "isotropic_limit.csv",
                    "brute_force_checks.csv", "self_distillation_evidence.png", "summary.json",
                    "source_manifest.json"}
        self.assertTrue(expected.issubset({p.name for p in OUT.iterdir()}))
        self.assertTrue(self.summary["compute"]["cpu_only"])
        self.assertFalse(self.summary["compute"]["gpu_used"])


if __name__ == "__main__":
    unittest.main()
