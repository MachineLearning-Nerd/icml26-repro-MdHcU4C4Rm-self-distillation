# Status — Optimal Unconstrained Self-Distillation

Updated: 2026-08-16

Paper: Optimal Unconstrained Self-Distillation in Ridge Regression: Strict
Improvements, Precise Asymptotics, and One-Shot Tuning

OpenReview: MdHcU4C4Rm

Collection status:
PARTIALLY_VERIFIED_WITH_LITERAL_CLAIM_BOUNDARY_AND_UNSEALED_CLAIMS

## Current conclusion

The current checkout contains committed evidence for the exact ridge-risk
identities, stationary boundary analysis, negative mixing, the isotropic sign
transition, and one-shot GCV convergence. It also contains the producer code
and historical report for the anisotropic Theorem 3.1 sweep and the four-data-
set curvature experiment.

However, the raw Claim 3 and Claim 6 output files expected by the nine-test
suite are not present in outputs/full. Those claims remain
BLOCKED_REPRODUCTION_REQUIRED until a fresh run regenerates and commits the
outputs and the suite passes from a clean clone.

## Claim states

- Claim 1: FALSIFIED_AS_WRITTEN for the unqualified summary; the corrected
  Theorem 2.2 nonstationary condition passes 1,408/1,408 cases.
- Claim 2: VERIFIED_SCOPED; 737 negative optima, 671 positive optima, and
  1,408/1,408 sign-rule checks.
- Claim 3: BLOCKED_REPRODUCTION_REQUIRED; producer present, raw convergence
  output absent.
- Claim 4: VERIFIED_SCOPED; exact isotropic transition at lambda*=0.5.
- Claim 5: VERIFIED_SCOPED; 288 one-shot GCV fits with shrinking errors.
- Claim 6: BLOCKED_REPRODUCTION_REQUIRED; producer and historical report
  present, raw curvature summary absent.

## Evaluator state

- Detailed historical report: 8/12 live
- Claims 3 and 6: historically inconclusive
- New live score: not claimed
- Forecast: 11--12/12 only

## Main entry points

- README.md — paper explanation, claim ledger, branch guide, citation, and thanks
- claims.json — machine-readable six-claim ledger
- CLAIM_EVIDENCE.md — source-to-producer-to-verdict paths
- SOURCE_AUDIT.md and SOURCE_MANIFEST.md — source and artifact boundary
- BRANCH_AUDIT.md — branch rename and history map
- EVIDENCE_MANIFEST.json — release manifest
- verify_final.py — lightweight fail-closed state check
