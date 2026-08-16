# Optimal Unconstrained Self-Distillation: claim-by-claim audit

This report is the paper-facing explanation of the current repository state.
It separates committed raw evidence from results that appear only in the
historical publication narrative.

## Verdict summary

| Claim | Current verdict | Committed evidence state |
| --- | --- | --- |
| Theorem 2.2 strict improvement | FALSIFIED_AS_WRITTEN for the unqualified summary; corrected nonstationary theorem supported | 64 stationary equality cases and 1,408 corrected nonstationary cases are committed |
| Negative optimal mixing and sign rule | VERIFIED_SCOPED | Structural sweep and isotropic limit CSVs are committed |
| Theorem 3.1 anisotropic equivalents | BLOCKED_REPRODUCTION_REQUIRED | Producer and historical plot exist; raw convergence CSV is missing |
| Corollary 3.2 isotropic transition | VERIFIED_SCOPED | Isotropic limit CSV is committed |
| Theorem 4.1 one-shot GCV | VERIFIED_SCOPED | 288 one-shot fits are committed |
| Proposition 2.3 four-dataset curvature | BLOCKED_REPRODUCTION_REQUIRED | Producer, cached CIFAR input, and historical plot exist; raw curvature CSV is missing |

## What is reproduced exactly

For a ridge teacher and pure-distilled student, the implementation computes
conditional population risk analytically as

~~~text
(beta_hat-beta)^T Sigma (beta_hat-beta) + sigma^2
~~~

The structural sweep checks the closed-form optimal mixture, the derivative
identity, the gain identity, and the stationary boundary. The isotropic sweep
checks the exact transition lambda*=gamma sigma^2/r^2. The one-shot sweep
compares GCV's estimate with the oracle value without grid search, splitting,
or refitting.

Recorded values:

- 64 root-solved stationary penalties have zero gain within 2.544e-28.
- 1,408 corrected nonstationary cases have positive gain; the minimum is
  1.949232e-6.
- 737 optimal weights are negative and 671 are positive.
- The sign rule passes 1,408/1,408 checks.
- 288 one-shot GCV fits are present.
- The maximum closed-form weight identity error is 1.439e-13.

## The two unsealed producers

### Theorem 3.1

reproduction/claim3_asymptotics.py reconstructs the kappa fixed point,
trace/alignment functionals, and deterministic-equivalent risks for anisotropic
AR(1) covariance at fixed gamma=0.5. The historical report records MAD
reductions of about 3.7x--4.2x as p grows from 50 to 800, and the committed
plot illustrates that result.

The current outputs/full directory does not contain the producer's
claim3_convergence_trials.csv or claim3_convergence.png, and summary.json has
no claim_3_asymptotic section. The result is therefore not sealed.

### Proposition 2.3

reproduction/claim6_curvature.py reproduces the curvature condition and
four-dataset protocol for BlogFeedback, Communities and Crime, CIFAR10
ResNet-18 features, and Air Quality. The historical report records held-out
agreement with Table 2 and identifies Communities as a boundary case.

The current outputs/full directory does not contain
claim6_curvature_summary.csv or claim6_curvature.png, and summary.json has no
claim_6_curvature section. The result is therefore not sealed.

## Historical evaluator state

The detailed repository record reports a previous live score of 8/12, with
Claims 3 and 6 inconclusive. The forecast of 11--12/12 is not a judge result.
No new evaluator score is claimed by this audit.

## Reproduction command

~~~bash
bash reproduction/run.sh
~~~

The command is expected to regenerate all outputs, then run the nine-test
fail-closed suite. A current clean checkout should be considered incomplete
until the missing Claim 3 and Claim 6 files are produced and independently
verified.

## Lineage

- [baseline/judged-8-of-12](https://github.com/MachineLearning-Nerd/icml26-self-distillation-ridge/tree/baseline/judged-8-of-12)
- [proof/claim-3-theorem-3-1](https://github.com/MachineLearning-Nerd/icml26-self-distillation-ridge/tree/proof/claim-3-theorem-3-1)
- [experiment/claim-6-curvature-four-datasets](https://github.com/MachineLearning-Nerd/icml26-self-distillation-ridge/tree/experiment/claim-6-curvature-four-datasets)
