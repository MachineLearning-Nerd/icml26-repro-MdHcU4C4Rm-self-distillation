# Optimal Unconstrained Self-Distillation in Ridge Regression

Independent ICML 2026 reproduction and evidence audit for:

> **Optimal Unconstrained Self-Distillation in Ridge Regression: Strict Improvements, Precise Asymptotics, and One-Shot Tuning**

Paper authors: **Hien Dang, Pratik Patil, and Alessandro Rinaldo**

Paper: [arXiv:2602.17565](https://arxiv.org/abs/2602.17565) · OpenReview
MdHcU4C4Rm

Final repository: [MachineLearning-Nerd/icml26-self-distillation-ridge](https://github.com/MachineLearning-Nerd/icml26-self-distillation-ridge)

Previous repository name: icml26-repro-MdHcU4C4Rm-self-distillation

**Collection status:** PARTIALLY_VERIFIED_WITH_LITERAL_CLAIM_BOUNDARY_AND_UNSEALED_CLAIMS

This repository is an independent reproduction, not the authors' official
implementation. It keeps the exact population-risk calculations, branch
lineage, committed artifacts, and historical evaluator record separate from
claims that cannot currently be sealed from a fresh checkout.

## What the paper does

The paper studies one round of self-distillation for ridge regression. A
teacher is fit with ridge penalty lambda, a pure-distilled student applies the
same smoother to the teacher's predictions, and an unconstrained mixture
parameter xi chooses the best affine combination of teacher labels and
teacher predictions. The paper asks:

1. when the optimal mixture strictly improves the teacher;
2. why the optimal mixture can be negative;
3. what the anisotropic proportional-asymptotic limits are;
4. where the isotropic sign transition occurs;
5. whether one-shot GCV consistently estimates xi without grid search or
   refitting; and
6. whether a curvature test predicts global improvement on real datasets.

The implementation evaluates conditional population risk analytically as
(beta_hat-beta)^T Sigma (beta_hat-beta)+sigma^2. It therefore avoids
confusing a finite held-out test fluctuation with a theorem result.

## Current claim ledger

| # | Paper claim | Current status | Evidence and production path |
| --- | --- | --- | --- |
| 1 | Theorem 2.2 strict improvement | FALSIFIED_AS_WRITTEN; VERIFIED_CORRECTED_NONSTATIONARY_THEOREM | The compressed claim fails at 64 root-solved stationary penalties: maximum absolute gain 2.54e-28. The corrected theorem requires R'(lambda) != 0 and passes 1,408/1,408 nonstationary cases with minimum gain 1.949e-6. |
| 2 | Negative optimal mixing and sign rule | VERIFIED_SCOPED | 737 negative and 671 positive optima; sign rule passes 1,408/1,408; closed-form identity error is 1.44e-13; isotropic transition is lambda*=0.5. |
| 3 | Anisotropic Theorem 3.1 deterministic equivalents | BLOCKED_REPRODUCTION_REQUIRED | The producer reproduction/claim3_asymptotics.py and historical report exist, but the current checkout lacks outputs/full/claim3_convergence_trials.csv and its raw summary; the fresh test suite cannot seal this result. |
| 4 | Corollary 3.2 isotropic sign transition | VERIFIED_SCOPED | outputs/full/isotropic_limit.csv and the sign-transition test show xi*>0 below 0.5, xi*<0 above 0.5, and exact zero at 0.5. |
| 5 | Theorem 4.1 one-shot GCV consistency | VERIFIED_SCOPED | 288 committed one-shot fits across p=50,100,200,400 and lambda=0.1,0.5,2.0; median weight and excess-risk errors decrease with p and all GCV discrepancies are positive. |
| 6 | Proposition 2.3 curvature test on four real datasets | BLOCKED_REPRODUCTION_REQUIRED | The producer, cached CIFAR features, report, and plot exist, but outputs/full/claim6_curvature_summary.csv and the raw Claim 6 output are absent from the current checkout. The historical report is recorded, not freshly sealed. |

The Claim 1 distinction matters: the paper's theorem includes the
nonstationary condition R'(lambda) != 0, while the older claim summary
compressed it into an unconditional sentence. The audit records the
unqualified sentence as false and the corrected theorem as supported.

Claims 3 and 6 are intentionally not called verified merely because plots or
historical prose exist. Their producer scripts describe exactly how to
recreate the missing outputs; until those outputs are committed and the
fail-closed tests pass from a fresh clone, their status remains blocked.

## Historical evaluator record

The repository's detailed publication report records a previous live result of
**8/12**, with Claims 3 and 6 inconclusive. No new judge score is claimed here.
The forecast range of 11--12/12 is a forecast only and must not be read as an
evaluation result.

The scored logbook is preserved at
[Hugging Face Space DineshAI/MdHcU4C4Rm](https://huggingface.co/spaces/DineshAI/MdHcU4C4Rm).

## How each claim is produced

| Claim | Producer | Committed raw evidence | Test or audit |
| --- | --- | --- | --- |
| 1 | reproduction/reproduce.py structural and stationary sweeps | outputs/full/structural_trials.csv, stationary_counterexamples.csv, claim_evidence.csv, summary.json | test_claim_one_strict_improvement_and_boundary and test_corrected_nonstationary_theorem |
| 2 | Exact ridge-risk identities plus isotropic_limit() | structural_trials.csv, isotropic_limit.csv, summary.json | test_negative_weights_and_exact_transition |
| 3 | reproduction/claim3_asymptotics.py reconstructs Eqs. 12--16 and compares finite conditional risks to deterministic equivalents | Historical report and reports/self-distillation/images/claim3_convergence.png; raw outputs/full/claim3_convergence_trials.csv is missing | test_claim3_anisotropic_asymptotic_convergence is fail-closed and currently cannot load its missing artifact |
| 4 | isotropic_limit() and the exact closed-form sign formula | outputs/full/isotropic_limit.csv | test_negative_weights_and_exact_transition |
| 5 | oneshot_experiment() computes GCV and oracle xi without refitting | outputs/full/oneshot_trials.csv and oneshot_summary.csv | test_one_shot_consistency |
| 6 | reproduction/claim6_curvature.py reproduces Proposition 2.3 and the four-dataset protocol | Cached data/cifar10_resnet18_features.npz and historical report/plot; raw outputs/full/claim6_curvature_summary.csv is missing | test_claim6_real_data_curvature is fail-closed and currently cannot load its missing artifact |

The complete source-to-verdict explanation is
[CLAIM_EVIDENCE.md](CLAIM_EVIDENCE.md). Source hashes and the clean-room
boundary are in [SOURCE_AUDIT.md](SOURCE_AUDIT.md) and
[SOURCE_MANIFEST.md](SOURCE_MANIFEST.md).

## Branch guide

The final branch names describe the scientific role of each route:

| Final branch | Purpose |
| --- | --- |
| [main](https://github.com/MachineLearning-Nerd/icml26-self-distillation-ridge/tree/main) | Publication surface, current README, status, claim ledger, and cumulative report |
| [baseline/judged-8-of-12](https://github.com/MachineLearning-Nerd/icml26-self-distillation-ridge/tree/baseline/judged-8-of-12) | Frozen baseline for Claims 1, 2, 4, and 5 and the earlier evaluator state |
| [proof/claim-3-theorem-3-1](https://github.com/MachineLearning-Nerd/icml26-self-distillation-ridge/tree/proof/claim-3-theorem-3-1) | Theorem 3.1 anisotropic deterministic-equivalent producer |
| [experiment/claim-6-curvature-four-datasets](https://github.com/MachineLearning-Nerd/icml26-self-distillation-ridge/tree/experiment/claim-6-curvature-four-datasets) | Proposition 2.3 four-dataset curvature experiment and cached CIFAR protocol |

The exact legacy mapping and pre-cleanup tips are in
[BRANCH_AUDIT.md](BRANCH_AUDIT.md). A branch name is a route to code, not
proof by itself.

## Reproduce or close the open artifact gap

The fixed historical command is:

~~~bash
bash reproduction/run.sh
~~~

It recreates the locked uv environment, runs reproduction/reproduce.py, and
executes the nine-test fail-closed suite. The current checkout should be
treated as incomplete until that command produces the missing Claim 3 and
Claim 6 files and the suite passes from a clean clone.

For a quick inspection of the current evidence surface:

~~~bash
uv sync --frozen
uv run python -m unittest -v reproduction.test_reproduction
uv run python verify_final.py
~~~

The historical branch logs record 7/7, 8/8, and 9/9 test milestones, but the
current committed output set does not contain all files those tests load. This
README deliberately reports that discrepancy.

## Citation

~~~bibtex
@misc{dang2026optimal,
  title         = {Optimal Unconstrained Self-Distillation in Ridge Regression: Strict Improvements, Precise Asymptotics, and One-Shot Tuning},
  author        = {Dang, Hien and Patil, Pratik and Rinaldo, Alessandro},
  year          = {2026},
  eprint        = {2602.17565},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  url           = {https://arxiv.org/abs/2602.17565}
}
~~~

Please also cite this audit using [CITATION.cff](CITATION.cff) when reusing
the scripts or evidence.

## Thank you

Thank you to **Hien Dang, Pratik Patil, and Alessandro Rinaldo** for making
the paper's identities, asymptotic formulas, and self-distillation perspective
available for independent study. This repository aims to be a respectful and
traceable reproduction: positive conclusions are scoped to committed
artifacts, and unresolved artifact gaps are named plainly so they can be
repaired without rewriting history.

## Limitations

- Claim 1's literal/unqualified summary is false at stationary penalties, while
  the corrected nonstationary theorem is supported by the recorded sweep.
- Claim 3 is finite-scale convergence evidence, not a proof of the asymptotic
  theorem; its raw current-checkout output is missing.
- Claim 6 depends on external dataset protocols and a cached CIFAR feature
  file; its raw current-checkout summary is missing.
- The historical report and forecast do not substitute for a new live judge
  result.
