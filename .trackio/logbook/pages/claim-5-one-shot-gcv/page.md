# Claim 5 - One-shot GCV tuning (Theorem 4.1)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c5", "created_at": "2026-07-25T00:00:00+00:00", "title": "VERIFIED: consistent one-shot GCV, no grid/split/refit"}
-->
## Exact scored claim

> Theorem 4.1 proposes a one-shot generalized cross-validation (GCV) estimator of `ξ*` that is consistent in proportional asymptotics and avoids grid search, sample splitting, and student refitting (§4, Eqs. 17–19).

## Verdict: VERIFIED

Equations (17)–(19) are implemented directly from the eigenspectrum of the training smoother: degrees-of-freedom-corrected residuals of the teacher `H` and pure-distilled `H²` give one closed-form GCV ratio per dataset. **No weight grid, no sample split, no student refit** — a single eigendecomposition per problem.

| p | λ | median weight error | median excess risk | sign accuracy |
|---:|---:|---:|---:|---:|
| 50 | 0.1 | 0.4083 | 6.46e-03 | 100.0% |
| 50 | 0.5 | 0.3071 | 7.67e-03 | 54.2% |
| 50 | 2.0 | 0.3696 | 7.17e-03 | 100.0% |
| 400 | 0.1 | 0.2333 | 2.66e-03 | 100.0% |
| 400 | 0.5 | 0.1563 | 2.24e-03 | 20.8% |
| 400 | 2.0 | 0.2239 | 3.08e-03 | 100.0% |

From p=50 to p=400, median weight error decreases 1.65×–1.96× and excess risk 2.33×–3.43× at all three penalties. At λ=0.1 and 2.0 (where the oracle sign is separated from zero) p=400 sign accuracy is 100%. At λ=0.5 the asymptotic oracle weight is exactly zero, so finite-sample sign is unstable by construction there — magnitude and regret convergence are the meaningful checks. Raw data: `oneshot_trials.csv` (288 fits), `oneshot_summary.csv`.
