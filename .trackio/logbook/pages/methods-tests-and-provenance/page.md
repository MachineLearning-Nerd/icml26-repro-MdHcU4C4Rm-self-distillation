# Methods, tests, and provenance


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_6b5de72aedd7", "created_at": "2026-07-16T16:06:43+00:00", "title": "Independent exact-risk protocol"}
-->
# Methods and provenance

For `y=X beta+epsilon`, ridge uses
`beta_hat=(X'X/n+lambda I)^(-1)X'y/n`. The pure-distilled coefficient applies
the same smoother a second time. Population squared risk is evaluated exactly
as `(beta_hat-beta)' Sigma (beta_hat-beta)+sigma^2`; no held-out Monte Carlo
approximation can hide or create a small gain.

Seven tests fail closed on: both Theorem 2.2 identities; 64 nondegenerate
stationary equality counterexamples; 1,408 corrected strict gains; negative
weights and the exact sign transition; one-shot error/regret contraction;
dense brute-force argmins; and complete CPU-only outputs.

Paper PDF SHA-256:
`5ee256f603d43dec7ffc8e67971e738ef1085f99f54528cb544baf4dcbd8c2c2`.
Official repository pinned at `7215dda72fc63149fca730248bebc34aa4d3cc8b`.
The independent script imports no official module. The artifact retains the
paper, exact claim snapshot, selected official-code snapshot, source audit,
scripts, tests, all raw CSV rows, figure, summary, and SHA-256 manifest.


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_cfc9dff0b595", "created_at": "2026-07-16T16:06:44+00:00", "title": "Complete CPU reproduction workspace", "artifact": "unconstrained-self-distillation-repro/unconstrained-self-distillation-cpu-reproduction:v0", "artifact_type": "dataset"}
-->
**📦 Artifact** `unconstrained-self-distillation-repro/unconstrained-self-distillation-cpu-reproduction:v0` · dataset

https://huggingface.co/buckets/DineshAI/MdHcU4C4Rm-artifacts#unconstrained-self-distillation-repro/unconstrained-self-distillation-cpu-reproduction:v0


---
<!-- trackio-cell
{"type": "code", "id": "cell_36363a00a7ca", "created_at": "2026-07-16T16:07:23+00:00", "title": "Full exact-risk reproduction", "command": [".venv/bin/python", "reproduction/reproduce.py", "--output-dir", "outputs/full"], "exit_code": 0, "duration_s": 5.245}
-->
````bash
$ .venv/bin/python reproduction/reproduce.py --output-dir outputs/full
````

exit 0 · 5.2s


````python title=reproduce.py
#!/usr/bin/env python3
"""Independent CPU reproduction of unconstrained ridge self-distillation.

The implementation uses only the displayed ridge and GCV equations in the
paper.  It never imports the authors' code.  Conditional population risks are
calculated analytically, eliminating test-set Monte Carlo noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RidgeProblem:
    y: np.ndarray
    beta: np.ndarray
    sigma: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    rhs_eig: np.ndarray
    uy: np.ndarray

    @classmethod
    def make(cls, n: int, p: int, seed: int, kind: str) -> "RidgeProblem":
        rng = np.random.default_rng(seed)
        if kind == "isotropic":
            sigma = np.eye(p)
            transform = np.eye(p)
            beta = rng.normal(size=p)
        elif kind == "anisotropic":
            rho = 0.5
            idx = np.arange(p)
            sigma = rho ** np.abs(idx[:, None] - idx[None, :])
            transform = np.linalg.cholesky(sigma)
            # Align the signal with a high-variance population direction.
            _, population_vectors = np.linalg.eigh(sigma)
            beta = population_vectors[:, -1] + 0.25 * rng.normal(size=p)
        else:
            raise ValueError(kind)
        beta /= np.sqrt(beta @ sigma @ beta)
        x = rng.normal(size=(n, p)) @ transform.T
        y = x @ beta + rng.normal(size=n)
        gram = x.T @ x / n
        eigvals, eigvecs = np.linalg.eigh(gram)
        rhs_eig = eigvecs.T @ (x.T @ y / n)
        # U'y without materializing the n x n smoother; X V = U diag(sqrt(n d)).
        xv = x @ eigvecs
        uy = np.zeros(p)
        positive = eigvals > 1e-14
        uy[positive] = (xv[:, positive].T @ y) / np.sqrt(n * eigvals[positive])
        return cls(y, beta, sigma, eigvals, eigvecs, rhs_eig, uy)

    def quantities(self, lam: float) -> dict[str, float]:
        denom = self.eigvals + lam
        teacher_eig = self.rhs_eig / denom
        pure_eig = (self.eigvals / denom) * teacher_eig
        teacher = self.eigvecs @ teacher_eig
        pure = self.eigvecs @ pure_eig
        err_t = teacher - self.beta
        err_p = pure - self.beta
        noise_var = 1.0
        risk = float(err_t @ self.sigma @ err_t + noise_var)
        risk_pd = float(err_p @ self.sigma @ err_p + noise_var)
        cross = float(err_t @ self.sigma @ err_p + noise_var)
        discrepancy = float((teacher - pure) @ self.sigma @ (teacher - pure))
        derivative_vec = self.eigvecs @ (-self.rhs_eig / denom**2)
        derivative = float(2.0 * err_t @ self.sigma @ derivative_vec)
        xi_direct = (risk - cross) / discrepancy
        xi_derivative = -lam * derivative / (2.0 * discrepancy)
        gain_direct = (risk - cross) ** 2 / discrepancy
        gain_derivative = lam**2 * derivative**2 / (4.0 * discrepancy)

        # Equations (17)-(19): corrected residuals from H and H^2.
        shrink = self.eigvals / denom
        y_norm2 = float(self.y @ self.y)
        projected_norm2 = float(self.uy @ self.uy)
        orthogonal_norm2 = max(0.0, y_norm2 - projected_norm2)
        residual_t_sq = orthogonal_norm2 + float(np.sum(((1.0 - shrink) * self.uy) ** 2))
        residual_p_sq = orthogonal_norm2 + float(np.sum(((1.0 - shrink**2) * self.uy) ** 2))
        residual_cross = orthogonal_norm2 + float(np.sum((1.0 - shrink) * (1.0 - shrink**2) * self.uy**2))
        n = self.y.size
        corr_t = 1.0 - float(np.sum(shrink)) / n
        corr_p = 1.0 - float(np.sum(shrink**2)) / n
        rhat = residual_t_sq / (n * corr_t**2)
        rhat_pd = residual_p_sq / (n * corr_p**2)
        chat = residual_cross / (n * corr_t * corr_p)
        dhat = rhat + rhat_pd - 2.0 * chat
        xi_gcv = (rhat - chat) / dhat
        return {
            "lambda": lam, "risk_teacher": risk, "risk_pure": risk_pd,
            "cross_risk": cross, "D": discrepancy, "risk_derivative": derivative,
            "xi_oracle": xi_direct, "xi_derivative": xi_derivative,
            "risk_optimal": risk - gain_direct, "gain": gain_direct,
            "gain_derivative": gain_derivative, "gcv_teacher": rhat,
            "gcv_pure": rhat_pd, "gcv_cross": chat, "D_gcv": dhat,
            "xi_gcv": xi_gcv,
        }


def structural_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    lambdas = np.geomspace(0.02, 20.0, 22)
    rows, stationary = [], []
    for kind_index, kind in enumerate(("isotropic", "anisotropic")):
        for seed in range(32):
            problem = RidgeProblem.make(400, 200, 10_000 * kind_index + seed, kind)
            for lam in lambdas:
                q = problem.quantities(float(lam))
                q.update(kind=kind, seed=seed, n=400, p=200)
                rows.append(q)
            grid = np.geomspace(0.002, 100.0, 100)
            derivs = [problem.quantities(float(x))["risk_derivative"] for x in grid]
            brackets = [(a, b) for a, b, da, db in zip(grid[:-1], grid[1:], derivs[:-1], derivs[1:]) if da * db < 0]
            if not brackets:
                raise AssertionError(f"No stationary root for {kind=} {seed=}")
            a, b = brackets[0]
            root = brentq(lambda x: problem.quantities(float(x))["risk_derivative"], a, b, xtol=1e-13)
            q = problem.quantities(float(root))
            q.update(kind=kind, seed=seed, n=400, p=200, stationary_lambda=root)
            stationary.append(q)
    return pd.DataFrame(rows), pd.DataFrame(stationary)


def oneshot_experiment() -> pd.DataFrame:
    rows = []
    for p in (50, 100, 200, 400):
        n = 2 * p
        for seed in range(24):
            problem = RidgeProblem.make(n, p, 50_000 + 1000 * p + seed, "isotropic")
            for lam in (0.1, 0.5, 2.0):
                q = problem.quantities(lam)
                rows.append({
                    "n": n, "p": p, "gamma": p / n, "seed": seed, "lambda": lam,
                    "xi_oracle": q["xi_oracle"], "xi_gcv": q["xi_gcv"],
                    "abs_weight_error": abs(q["xi_gcv"] - q["xi_oracle"]),
                    "sign_match": int(np.sign(q["xi_gcv"]) == np.sign(q["xi_oracle"])),
                    "excess_risk": q["D"] * (q["xi_gcv"] - q["xi_oracle"]) ** 2,
                    "D_gcv": q["D_gcv"],
                })
    return pd.DataFrame(rows)


def isotropic_limit() -> pd.DataFrame:
    gamma = 0.5
    sigma2 = 1.0
    r2 = 1.0
    lambdas = np.unique(np.r_[np.geomspace(0.02, 20.0, 80), gamma * sigma2 / r2])
    rows = []
    for lam in np.sort(lambdas):
        risk = 1.0 + (lam**2 * r2 + gamma * sigma2) / (1.0 + lam) ** 2
        derivative = 2.0 * (lam * r2 - gamma * sigma2) / (1.0 + lam) ** 3
        discrepancy = lam**2 * (r2 + gamma * sigma2) / (1.0 + lam) ** 4
        xi = -lam * derivative / (2.0 * discrepancy)
        gain = lam**2 * derivative**2 / (4.0 * discrepancy)
        rows.append({"lambda": lam, "risk": risk, "risk_derivative": derivative,
                     "D": discrepancy, "xi_optimal": xi, "gain": gain})
    return pd.DataFrame(rows)


def brute_force_checks(structural: pd.DataFrame) -> pd.DataFrame:
    """Numerically minimize the risk quadratic without using its optimizer."""
    selected = structural.iloc[np.linspace(0, len(structural) - 1, 12, dtype=int)]
    rows = []
    for _, row in selected.iterrows():
        oracle = float(row.xi_oracle)
        grid = np.linspace(oracle - 2.0, oracle + 2.0, 400_001)
        risk = row.risk_teacher - 2.0 * grid * (row.risk_teacher - row.cross_risk) + grid**2 * row.D
        brute = float(grid[int(np.argmin(risk))])
        rows.append({"kind": row.kind, "seed": int(row.seed), "lambda": row["lambda"],
                     "xi_closed_form": oracle, "xi_brute_force": brute,
                     "abs_difference": abs(oracle - brute), "grid_step": float(grid[1] - grid[0])})
    return pd.DataFrame(rows)


def build_summary(structural: pd.DataFrame, stationary: pd.DataFrame,
                  oneshot: pd.DataFrame, limit: pd.DataFrame, wall: float) -> dict:
    nonstationary = structural[np.abs(structural.risk_derivative) > 1e-10]
    one = oneshot.groupby(["p", "lambda"], as_index=False).agg(
        median_weight_error=("abs_weight_error", "median"),
        median_excess_risk=("excess_risk", "median"),
        sign_accuracy=("sign_match", "mean"), min_D_gcv=("D_gcv", "min"))
    endpoints = {}
    for lam in (0.1, 0.5, 2.0):
        small = one[(one.p == 50) & (one["lambda"] == lam)].iloc[0]
        large = one[(one.p == 400) & (one["lambda"] == lam)].iloc[0]
        endpoints[str(lam)] = {
            "weight_error_reduction_factor": float(small.median_weight_error / large.median_weight_error),
            "excess_risk_reduction_factor": float(small.median_excess_risk / large.median_excess_risk),
            "p400_sign_accuracy": float(large.sign_accuracy),
        }
    boundary = limit.iloc[(limit["lambda"] - 0.5).abs().argsort()[:1]].iloc[0]
    return {
        "paper": {"openreview": "MdHcU4C4Rm", "arxiv": "2602.17565"},
        "claim_1": {
            "verdict": "falsified_as_written",
            "reason": "The claim omits Theorem 2.2's nonstationary condition R'(lambda) != 0.",
            "stationary_counterexamples": int(len(stationary)),
            "stationary_all_nondegenerate": bool((stationary.D > 1e-12).all()),
            "max_abs_stationary_derivative": float(stationary.risk_derivative.abs().max()),
            "max_abs_stationary_weight": float(stationary.xi_oracle.abs().max()),
            "max_abs_stationary_gain": float(stationary.gain.abs().max()),
            "corrected_nonstationary_checks": int(len(nonstationary)),
            "corrected_strict_gains": int((nonstationary.gain > 0).sum()),
            "min_corrected_gain": float(nonstationary.gain.min()),
        },
        "claim_2": {
            "verdict": "verified",
            "negative_weights": int((structural.xi_oracle < -1e-10).sum()),
            "positive_weights": int((structural.xi_oracle > 1e-10).sum()),
            "sign_rule_checks": int(len(nonstationary)),
            "sign_rule_passes": int((np.sign(nonstationary.xi_oracle) == -np.sign(nonstationary.risk_derivative)).sum()),
            "max_weight_identity_error": float((structural.xi_oracle - structural.xi_derivative).abs().max()),
            "isotropic_boundary_lambda": 0.5,
            "boundary_weight": float(boundary.xi_optimal),
            "boundary_gain": float(boundary.gain),
        },
        "claim_3": {"verdict": "verified", "fits": int(len(oneshot)), "endpoints": endpoints,
                    "all_gcv_discrepancies_positive": bool((oneshot.D_gcv > 0).all())},
        "cross_checks": {
            "max_gain_identity_error": float((structural.gain - structural.gain_derivative).abs().max()),
            "structural_rows": int(len(structural)), "stationary_rows": int(len(stationary)),
        },
        "compute": {"wall_seconds": wall, "cpu_only": True, "gpu_used": False,
                    "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "processor": platform.processor(), "platform": platform.platform()},
    }


def plot_results(structural: pd.DataFrame, stationary: pd.DataFrame,
                 oneshot: pd.DataFrame, limit: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    sample = structural[(structural.kind == "isotropic") & (structural.seed == 0)]
    axes[0].semilogx(sample["lambda"], sample.xi_oracle, "o-", label="finite-sample oracle")
    axes[0].semilogx(limit["lambda"], limit.xi_optimal, "-", label="isotropic limit")
    axes[0].axhline(0, color="black", lw=0.8); axes[0].axvline(0.5, color="gray", ls="--")
    axes[0].set(title="Optimal weight and sign transition", xlabel="ridge penalty", ylabel="optimal mixing")
    axes[0].legend(fontsize=8)
    axes[1].scatter(stationary.stationary_lambda, stationary.gain, s=13, alpha=0.7)
    axes[1].set(title="Stationary counterexamples", xlabel="root-solved penalty", ylabel="optimal risk gain")
    grouped = oneshot.groupby(["p", "lambda"]).abs_weight_error.median().reset_index()
    for lam, part in grouped.groupby("lambda"):
        axes[2].loglog(part.p, part.abs_weight_error, "o-", label=f"lambda={lam:g}")
    axes[2].set(title="One-shot GCV convergence", xlabel="dimension p", ylabel="median |weight error|")
    axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "full")
    args = parser.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    structural, stationary = structural_experiment()
    oneshot = oneshot_experiment()
    limit = isotropic_limit()
    brute = brute_force_checks(structural)
    structural.to_csv(out / "structural_trials.csv", index=False)
    stationary.to_csv(out / "stationary_counterexamples.csv", index=False)
    oneshot.to_csv(out / "oneshot_trials.csv", index=False)
    oneshot.groupby(["p", "lambda"], as_index=False).agg(
        median_weight_error=("abs_weight_error", "median"), median_excess_risk=("excess_risk", "median"),
        sign_accuracy=("sign_match", "mean"), min_D_gcv=("D_gcv", "min")).to_csv(out / "oneshot_summary.csv", index=False)
    limit.to_csv(out / "isotropic_limit.csv", index=False)
    brute.to_csv(out / "brute_force_checks.csv", index=False)
    plot_results(structural, stationary, oneshot, limit, out / "self_distillation_evidence.png")
    wall = time.perf_counter() - start
    summary = build_summary(structural, stationary, oneshot, limit, wall)
    summary["cross_checks"]["brute_force_checks"] = int(len(brute))
    summary["cross_checks"]["max_brute_force_weight_error"] = float(brute.abs_difference.max())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([
        {"claim": 1, "verdict": "falsified_as_written", "decisive_evidence":
         f"{len(stationary)} nondegenerate root-solved stationary penalties have zero optimal weight and zero gain; corrected theorem passes {len(structural)}/{len(structural)} nonstationary checks."},
        {"claim": 2, "verdict": "verified", "decisive_evidence":
         f"{int((structural.xi_oracle < -1e-10).sum())} negative optima; sign rule passes {len(structural)}/{len(structural)} checks; exact isotropic transition at lambda=0.5."},
        {"claim": 3, "verdict": "verified", "decisive_evidence":
         f"{len(oneshot)} one-shot fits through p=400; median weight and excess-risk errors shrink at all three penalties without refitting or grid search."},
    ]).to_csv(out / "claim_evidence.csv", index=False)
    manifest = {}
    audited_paths = [ROOT / "paper.pdf", ROOT / "claims.json", ROOT / "README.md",
                     ROOT / "SOURCE_AUDIT.md", *sorted((ROOT / "reproduction").glob("*")),
                     *sorted((ROOT / "source" / "official-snapshot").glob("*")), *out.glob("*")]
    for path in sorted(set(audited_paths)):
        if path.is_file() and path.name != "source_manifest.json":
            manifest[str(path.relative_to(ROOT))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest["official_code"] = {"repository": "https://github.com/hhd357/optimal_self_distillation_ridge",
                                 "commit": "7215dda72fc63149fca730248bebc34aa4d3cc8b",
                                 "imported_by_reproduction": False}
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

````


````output
{
  "paper": {
    "openreview": "MdHcU4C4Rm",
    "arxiv": "2602.17565"
  },
  "claim_1": {
    "verdict": "falsified_as_written",
    "reason": "The claim omits Theorem 2.2's nonstationary condition R'(lambda) != 0.",
    "stationary_counterexamples": 64,
    "stationary_all_nondegenerate": true,
    "max_abs_stationary_derivative": 1.9576354426398268e-14,
    "max_abs_stationary_weight": 5.207888399099298e-14,
    "max_abs_stationary_gain": 2.544043748557646e-28,
    "corrected_nonstationary_checks": 1408,
    "corrected_strict_gains": 1408,
    "min_corrected_gain": 1.949232328837568e-06
  },
  "claim_2": {
    "verdict": "verified",
    "negative_weights": 737,
    "positive_weights": 671,
    "sign_rule_checks": 1408,
    "sign_rule_passes": 1408,
    "max_weight_identity_error": 1.438849039914203e-13,
    "isotropic_boundary_lambda": 0.5,
    "boundary_weight": -0.0,
    "boundary_gain": 0.0
  },
  "claim_3": {
    "verdict": "verified",
    "fits": 288,
    "endpoints": {
      "0.1": {
        "weight_error_reduction_factor": 1.7498447649552147,
        "excess_risk_reduction_factor": 2.4310103792119295,
        "p400_sign_accuracy": 1.0
      },
      "0.5": {
        "weight_error_reduction_factor": 1.9647123009929757,
        "excess_risk_reduction_factor": 3.4291142276773323,
        "p400_sign_accuracy": 0.20833333333333334
      },
      "2.0": {
        "weight_error_reduction_factor": 1.650744413040252,
        "excess_risk_reduction_factor": 2.331211500870086,
        "p400_sign_accuracy": 1.0
      }
    },
    "all_gcv_discrepancies_positive": true
  },
  "cross_checks": {
    "max_gain_identity_error": 1.6653345369377348e-14,
    "structural_rows": 1408,
    "stationary_rows": 64,
    "brute_force_checks": 12,
    "max_brute_force_weight_error": 4.440892098500626e-16
  },
  "compute": {
    "wall_seconds": 4.1648332451004535,
    "cpu_only": true,
    "gpu_used": false,
    "python": "3.12.13",
    "numpy": "2.3.5",
    "scipy": "1.17.1",
    "processor": "",
    "platform": "Linux-7.0.9-arch2-1-x86_64-with-glibc2.43"
  }
}

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_0a2f3b3d5424", "created_at": "2026-07-16T16:07:24+00:00", "title": "Seven fail-closed tests", "command": [".venv/bin/python", "-m", "unittest", "-v", "reproduction/test_reproduction.py"], "exit_code": 0, "duration_s": 0.371}
-->
````bash
$ .venv/bin/python -m unittest -v reproduction/test_reproduction.py
````

exit 0 · 0.4s


````python title=test_reproduction.py
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

````


````output
test_claim_one_is_falsified_as_written (reproduction.test_reproduction.ReproductionTests.test_claim_one_is_falsified_as_written) ... ok
test_complete_outputs_and_cpu_metadata (reproduction.test_reproduction.ReproductionTests.test_complete_outputs_and_cpu_metadata) ... ok
test_corrected_nonstationary_theorem (reproduction.test_reproduction.ReproductionTests.test_corrected_nonstationary_theorem) ... ok
test_independent_brute_force_argmin (reproduction.test_reproduction.ReproductionTests.test_independent_brute_force_argmin) ... ok
test_negative_weights_and_exact_transition (reproduction.test_reproduction.ReproductionTests.test_negative_weights_and_exact_transition) ... ok
test_one_shot_consistency (reproduction.test_reproduction.ReproductionTests.test_one_shot_consistency) ... ok
test_theorem_22_identities (reproduction.test_reproduction.ReproductionTests.test_theorem_22_identities) ... ok

----------------------------------------------------------------------
Ran 7 tests in 0.050s

OK

````
