# Source and claim audit

This audit fixes the paper version, official-code boundary, claim anchors, and
the distinction between committed raw evidence and historical prose.

## Paper identity

- Title: Optimal Unconstrained Self-Distillation in Ridge Regression: Strict Improvements, Precise Asymptotics, and One-Shot Tuning
- Authors: Hien Dang; Pratik Patil; Alessandro Rinaldo
- arXiv: 2602.17565
- OpenReview: MdHcU4C4Rm
- Local PDF: paper.pdf
- PDF SHA-256: 5ee256f603d43dec7ffc8e67971e738ef1085f99f54528cb544baf4dcbd8c2c2

The local PDF is the source anchor for this repository. The latest source
version and any future paper revision should be recorded before extending a
verdict.

## Official-code boundary

The official repository is
https://github.com/hhd357/optimal_self_distillation_ridge at commit
7215dda72fc63149fca730248bebc34aa4d3cc8b. The files under
source/official-snapshot are provenance snapshots only. The independent
reproduction imports no official module.

The reproduction derives the conditional ridge risks, deterministic-equivalent
formulas, GCV quantities, and curvature protocol in reproduction/. It uses the
cached CIFAR feature file for the documented Claim 6 protocol; that input is
identified separately in SOURCE_MANIFEST.md.

## Claim mapping

### Claim 1 — Theorem 2.2

The summary sentence “strictly improves at every lambda” is too broad because
the theorem's strict inequality requires R'(lambda) != 0. The root-solved
stationary sweep finds 64 nondegenerate cases with zero optimal weight and zero
gain, while the corrected nonstationary sweep has 1,408/1,408 positive gains.
This is recorded as a boundary/falsification of the compressed statement, not
as a rejection of the correctly conditioned theorem.

### Claim 2 — negative weights and sign rule

The exact identity xi*=-lambda R'(lambda)/(2D) is evaluated from the same
conditional population-risk quantities. The sign rule and negative-weight
counts come from structural_trials.csv; the isotropic transition is checked
independently from isotropic_limit.csv.

### Claim 3 — Theorem 3.1

claim3_asymptotics.py reconstructs the kappa fixed point, trace functionals,
alignment functionals, and risk limits in Equations 12--19. It compares exact
finite-sample conditional quantities under anisotropic AR(1) covariance with
the deterministic equivalents as p,n grow at fixed gamma. The producer and
historical plot are committed, but the raw convergence CSV and summary
section are absent from the current checkout, so the claim is blocked.

### Claim 4 — Corollary 3.2

The isotropic closed form gives the transition lambda*=gamma sigma^2/r^2.
The committed output covers values below, at, and above the transition and the
test checks the sign on both sides and exact zero at the boundary.

### Claim 5 — Theorem 4.1

oneshot_experiment() evaluates the GCV residual corrections and compares GCV's
xi with the exact oracle xi on 288 fits. No grid, split, or student refit is
used. The committed CSVs and test cover this scoped numerical claim.

### Claim 6 — Proposition 2.3

claim6_curvature.py implements the curvature condition and the four-dataset
protocol, including the cached pretrained ResNet-18 CIFAR feature input. The
historical report says the held-out outcome agrees with Table 2, but the raw
summary expected by the current test is missing. The claim remains blocked
until a fresh run seals the artifact chain.

## Artifact completeness finding

The current main tree contains ten files in outputs/full. It does not contain
claim3_convergence_trials.csv, claim3_convergence.png,
claim6_curvature_summary.csv, or claim6_curvature.png. Because the test suite
loads those files in setUpClass, a clean checkout cannot currently reproduce
the report's six-claim conclusion. This finding is intentionally part of the
release status.
