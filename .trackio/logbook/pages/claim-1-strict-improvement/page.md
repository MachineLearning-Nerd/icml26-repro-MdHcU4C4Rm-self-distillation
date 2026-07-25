# Claim 1 - Strict improvement (Theorem 2.2)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c1_v2", "created_at": "2026-07-25T00:00:00+00:00", "title": "VERIFIED under R'(lambda)!=0; 64 stationary penalties are the equality boundary"}
-->
## Exact scored claim

> For any nonstationary ridge penalty λ (`R'(λ)≠0`), the optimally mixed self-distilled student strictly improves on the teacher's squared prediction risk, with `ξ*(λ) = −(λ/2)·R'(λ)/D(λ)` obeying `sign(ξ*(λ)) = −sign(R'(λ))` (Theorem 2.2, Eq. 9).

## Verdict: VERIFIED

Conditional population risks are evaluated **exactly** as `(β̂−β)ᵀΣ(β̂−β)+σ²` (no held-out Monte Carlo). Across **1408 nonstationary** grid evaluations (n=400, p=200; 32 seeds × {isotropic, AR(1) anisotropic} × 22 penalties):

- every case strictly improves — minimum gain `1.949e-06` (no zero or negative gain);
- the identity `ξ* = −λR'/(2D)` holds within `1.44e-13`;
- the sign rule `sign(ξ*) = −sign(R')` passes **1408/1408**;
- 12 independent dense (400 001-point) brute-force argmins agree within `4.44e-16`.

**Boundary case (equality, not a counterexample).** For each of 64 problems a stationary penalty is bracketed and root-solved (not grid-picked). All remain nondegenerate (`D>0`) with `R'(λ*)≈0` (max `1.96e-14`), `ξ*≈0` (max `5.21e-14`), and gain `≈0` (max `2.54e-28`) — exactly the equality Theorem 2.2 predicts at a ridge-optimal stationary point. These are the equality boundary, preserved on the [historical "stationary counterexamples" page](#/claim-1-stationary-counterexamples).

This corrects the earlier "falsified as written" framing: the theorem's `R'(λ)≠0` condition is part of the claim, and under it the strict improvement is verified.

Raw data: `structural_trials.csv` (1408 rows), `stationary_counterexamples.csv` (64 rows), `brute_force_checks.csv`.
